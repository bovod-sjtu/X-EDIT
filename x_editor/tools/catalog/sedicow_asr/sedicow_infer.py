"""Adapter from X-Editor SE-DiCoW service requests to official DiCoW inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, default=_default_official_repo())
    parser.add_argument("--dicow-model", default=_default_model("BUT-FIT__SE-DiCoW", "BUT-FIT/SE-DiCoW"))
    parser.add_argument("--diarization-model", default=os.getenv("SEDICOW_DIARIZATION_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md"))
    parser.add_argument("--diarization-cache-dir", default=os.getenv("SEDICOW_DIARIZATION_CACHE_DIR"))
    parser.add_argument(
        "--device",
        default=os.getenv("SEDICOW_DEVICE", "auto"),
        help="Torch device for DiCoW inference: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.getenv("SEDICOW_EMBEDDING_BATCH_SIZE", "16")))
    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=int(os.getenv("SEDICOW_SEGMENTATION_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--return-all-segments",
        action="store_true",
        default=os.getenv("SEDICOW_RETURN_ALL_SEGMENTS", "0") == "1",
        help="Return every official speaker segment instead of only the requested speaker.",
    )
    args = parser.parse_args()

    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    audio = Path(request["audio_path"]).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio}")

    official_repo = args.official_repo.expanduser().resolve()
    _prepare_import_paths(official_repo)

    # Keep stdout clean for the wrapper JSON parser; third-party progress/warnings
    # are still visible in the wrapper stderr.
    with contextlib.redirect_stdout(sys.stderr):
        result = _run_official_pipeline(
            audio,
            dicow_model=args.dicow_model,
            diarization_model=args.diarization_model,
            diarization_cache_dir=args.diarization_cache_dir,
            device_arg=args.device,
            embedding_batch_size=args.embedding_batch_size,
            segmentation_batch_size=args.segmentation_batch_size,
        )

    parsed = _parse_official_text(str(result.get("text") or ""))
    selection = select_native_speaker(parsed, request, return_all=args.return_all_segments)
    return_segments = selection.pop("segments")

    response = {
        "backend": "sedicow_official_dicow",
        "model": str(args.dicow_model),
        "diarization_model": str(args.diarization_model),
        "official_repo": str(official_repo),
        "speaker_id": request.get("speaker_id"),
        "text": _join_text(return_segments),
        "segments": return_segments,
        **selection,
        "language": request.get("language"),
        "raw": {
            "official_text": result.get("text"),
            "per_spk_outputs": result.get("per_spk_outputs"),
            "all_segments": parsed,
            "requested_speaker": _normalize_requested_speaker(request.get("speaker_id")),
            "returned_all_segments": selection["selection_status"] == "all_speakers",
        },
    }
    print(json.dumps(response, ensure_ascii=False))


def select_native_speaker(segments, request, *, return_all=False):
    """Select only within this provider's namespace, never by a foreign ID.

    External diarization hints do not bind identities in the official pipeline.
    Missing or unbound targets stay empty instead of silently returning everyone.
    """
    requested = _normalize_requested_speaker(request.get("speaker_id"))
    namespace = "sedicow_asr"
    if requested is None or return_all:
        selected, status = segments, "all_speakers"
    elif request.get("speaker_namespace") != namespace:
        selected, status = [], "unresolved_namespace"
    else:
        selected = [s for s in segments if _same_speaker(s.get("speaker_id"), requested)]
        status = "matched" if selected else "unmatched_speaker"
    return {
        "segments": selected,
        "speaker_namespace": namespace,
        "selection_status": status,
        "available_speaker_ids": sorted({str(s["speaker_id"]) for s in segments}),
    }


def _default_official_repo() -> Path:
    return Path(os.getenv("SEDICOW_REPO") or Path(__file__).resolve().parent / "vendor" / "DiCoW")


def _default_model(local_name: str, fallback: str) -> str:
    env_key = "SEDICOW_DICOW_MODEL" if "SE-DiCoW" in local_name or "DiCoW" in local_name else "SEDICOW_DIARIZATION_MODEL"
    if os.getenv(env_key):
        return str(os.getenv(env_key))
    local = Path(__file__).resolve().parent / "checkpoints" / "hf" / local_name
    return str(local) if local.exists() else fallback


def _prepare_import_paths(official_repo: Path) -> None:
    if not official_repo.exists():
        raise FileNotFoundError(f"official DiCoW repo not found: {official_repo}")
    modules_cache = Path(os.getenv("SEDICOW_HF_MODULES_CACHE") or Path(__file__).resolve().parent / "checkpoints" / "hf" / "modules_cache")
    modules_cache.mkdir(parents=True, exist_ok=True)
    hf_home = Path(os.getenv("SEDICOW_HF_HOME") or Path(__file__).resolve().parent / "checkpoints" / "hf" / "hf_home")
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
    diarizen_override = os.getenv("SEDICOW_DIARIZEN_REPO")
    diarizen_repo = (Path(diarizen_override).expanduser().resolve(strict=True)
                     if diarizen_override else official_repo / "DiariZen")
    if diarizen_override and not (diarizen_repo / "diarizen").is_dir():
        raise FileNotFoundError(f"Configured DiariZen source has no diarizen package: {diarizen_repo}")
    # An explicitly pinned standalone DiariZen must not be shadowed by the
    # older DiCoW submodule. Model inference and request semantics are unchanged.
    for path in (official_repo, diarizen_repo):
        if path.exists():
            sys.path.insert(0, str(path))
    if "gradio" not in sys.modules:
        try:
            __import__("gradio")
        except ModuleNotFoundError:
            gradio_stub = types.ModuleType("gradio")
            gradio_stub.Info = lambda *args, **kwargs: None
            sys.modules["gradio"] = gradio_stub


def _run_official_pipeline(
    audio: Path,
    *,
    dicow_model: str,
    diarization_model: str,
    diarization_cache_dir: str | None,
    device_arg: str,
    embedding_batch_size: int,
    segmentation_batch_size: int,
) -> dict[str, Any]:
    _preseed_remote_code_cache(Path(dicow_model))

    import torch
    from diarizen.pipelines.inference import DiariZenPipeline
    from inference import create_lower_uppercase_mapping
    from pipeline import DiCoWPipeline
    from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer
    _patch_runtime_compat(DiCoWPipeline)

    if device_arg == "auto":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        if device_arg.startswith("cuda:") and not re.fullmatch(r"cuda:\d+", device_arg):
            raise ValueError(f"invalid SEDICOW_DEVICE value: {device_arg}")
        if device_arg not in {"cpu", "cuda"} and not device_arg.startswith("cuda:"):
            raise ValueError(f"invalid SEDICOW_DEVICE value: {device_arg}")
        device = torch.device(device_arg)
    if hasattr(torch.serialization, "add_safe_globals"):
        safe_globals = [torch.torch_version.TorchVersion]
        try:
            from pyannote.audio.core.task import Problem, Resolution, Specifications, Task

            safe_globals.extend([Specifications, Problem, Resolution, Task])
        except Exception:
            pass
        torch.serialization.add_safe_globals(safe_globals)

    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(dicow_model, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(dicow_model)
    tokenizer = AutoTokenizer.from_pretrained(dicow_model)
    create_lower_uppercase_mapping(tokenizer)
    dicow.set_tokenizer(tokenizer)
    dicow.config.model_type = "whisper"

    diar_pipeline = DiariZenPipeline.from_pretrained(diarization_model, cache_dir=diarization_cache_dir).to(device)
    diar_pipeline.embedding_batch_size = embedding_batch_size
    diar_pipeline.segmentation_batch_size = segmentation_batch_size

    pipeline = DiCoWPipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )
    return pipeline(str(audio), return_timestamps=True)


def _preseed_remote_code_cache(model_path: Path) -> None:
    if not model_path.exists() or not model_path.is_dir():
        return
    modules_cache = Path(os.environ.get("HF_MODULES_CACHE") or "")
    if not modules_cache:
        return
    target = modules_cache / "transformers_modules" / model_path.name.replace("-", "_hyphen_")
    target.mkdir(parents=True, exist_ok=True)
    (modules_cache / "__init__.py").touch()
    (modules_cache / "transformers_modules" / "__init__.py").touch()
    (target / "__init__.py").touch()
    for source in model_path.glob("*.py"):
        shutil.copy2(source, target / source.name)


def _patch_runtime_compat(dicow_pipeline_cls: type) -> None:
    import torch
    import transformers.pipelines.automatic_speech_recognition as asr_module
    from librosa import load as librosa_load
    from soundfile import write as soundfile_write
    from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

    asr_module.is_torchcodec_available = lambda: False

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if not isinstance(inputs, str):
            raise ValueError("For now input must be a string representing a path to an audio file")

        inp_aud, sr = librosa_load(inputs, sr=16_000, mono=True)
        with tempfile.TemporaryDirectory(prefix="xeditor_sedicow_resampled_") as tmp:
            resampled_path = str(Path(tmp) / "resampled.wav")
            soundfile_write(resampled_path, inp_aud, sr, format="wav")

            generator = AutomaticSpeechRecognitionPipeline.preprocess(
                self,
                {"array": inp_aud, "sampling_rate": sr},
                chunk_length_s=chunk_length_s,
                stride_length_s=stride_length_s,
            )
            samples = next(generator)

            diarization_output = self.diarization_pipeline(resampled_path)
            per_speaker_samples = []
            for speaker in diarization_output.labels():
                per_speaker_samples.append(diarization_output.label_timeline(speaker))
            diarization_mask = self.get_diarization_mask(per_speaker_samples, samples["input_features"].shape[-1] // 2)
            stno_masks = []
            for i, _speaker_samples in enumerate(per_speaker_samples):
                stno_mask = self.get_stno_mask(diarization_mask, i)
                stno_masks.append(stno_mask)
            samples["stno_mask"] = torch.stack(stno_masks, axis=0).to(
                samples["input_features"].device,
                dtype=samples["input_features"].dtype,
            )
            samples["input_features"] = samples["input_features"].repeat(len(per_speaker_samples), 1, 1)
            samples["attention_mask"] = torch.ones(
                samples["input_features"].shape[0],
                samples["input_features"].shape[2],
                dtype=torch.bool,
                device=samples["input_features"].device,
            )
            if "num_frames" in samples:
                del samples["num_frames"]

            if hasattr(self.model.config, "use_enrollments") and self.model.config.use_enrollments:
                all_input_features = []
                all_attention_masks = []
                all_stno_masks = []
                enroll_input_features = []
                enroll_attention_masks = []
                enroll_stno_masks = []
                original_stno_length = samples["stno_mask"].shape[-1]

                for idx, stno_mask in enumerate(samples["stno_mask"]):
                    all_input_features.append(samples["input_features"][idx])
                    all_attention_masks.append(samples["attention_mask"][idx])
                    all_stno_masks.append(stno_mask)
                    enrollment_features, enrollment_attention, enrollment_stno = self._process_enrollment_sample(
                        samples,
                        idx,
                        stno_mask,
                        original_stno_length,
                    )
                    enroll_input_features.append(enrollment_features)
                    enroll_attention_masks.append(enrollment_attention)
                    enroll_stno_masks.append(enrollment_stno)

                samples["input_features"] = torch.stack(all_input_features, dim=0)
                samples["attention_mask"] = torch.stack(all_attention_masks, dim=0)
                samples["stno_mask"] = torch.stack(all_stno_masks, dim=0)
                samples["enrollments"] = {
                    "input_features": torch.stack(enroll_input_features, dim=0),
                    "attention_mask": torch.stack(enroll_attention_masks, dim=0),
                    "stno_mask": torch.stack(enroll_stno_masks, dim=0),
                }

            yield samples

    dicow_pipeline_cls.preprocess = preprocess


def _parse_official_text(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    speaker_blocks = re.finditer(
        r"Speaker\s+(?P<speaker>\d+)\s*:\s*(?P<body>.*?)(?=\n\s*\n.*?Speaker\s+\d+\s*:|\Z)",
        text,
        flags=re.DOTALL,
    )
    for block in speaker_blocks:
        speaker_id = block.group("speaker")
        body = block.group("body")
        for match in re.finditer(r"<\|(?P<start>\d+(?:\.\d+)?)\|>(?P<text>.*?)<\|(?P<end>\d+(?:\.\d+)?)\|>", body, re.DOTALL):
            start = float(match.group("start"))
            end = float(match.group("end"))
            phrase = _clean_text(match.group("text"))
            if end <= start and not phrase:
                continue
            segments.append(
                {
                    "speaker_id": speaker_id,
                    "start_sec": start,
                    "end_sec": end,
                    "text": phrase,
                    "confidence": None,
                }
            )
    return segments


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def _normalize_requested_speaker(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"(\d+)$", text)
    return match.group(1) if match else text


def _same_speaker(left: Any, right: str) -> bool:
    return _normalize_requested_speaker(left) == right


def _join_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(item.get("text") or "").strip() for item in segments if str(item.get("text") or "").strip())


if __name__ == "__main__":
    main()
