"""Isolated speaker diarization backend service for X-Editor."""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from support_projection import SupportedDiarization, run_with_source_support

DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"


class DiarizationRequest(BaseModel):
    audio_path: str
    input_sha256: str | None = None
    model: str = DEFAULT_MODEL
    num_speakers: int | None = None


app = FastAPI(title="X-Editor Diarization Backend", version="0.1.0")
_PIPELINES: dict[str, Any] = {}
_PIPELINE_LOCK = threading.Lock()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Diarization Backend", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    diarizen_available = _module_available("diarizen")
    pyannote_audio_available = _module_available("pyannote.audio")
    default_model_available = diarizen_available if _is_diarizen_model(DEFAULT_MODEL) else pyannote_audio_available
    return {
        "status": "healthy" if default_model_available else "not_ready",
        "backend": "diarization",
        "default_model": DEFAULT_MODEL,
        "default_model_available": default_model_available,
        "loaded_models": sorted(_PIPELINES),
        "diarizen_available": diarizen_available,
        "pyannote_audio_available": pyannote_audio_available,
    }


@app.post("/diarize")
def diarize(request: DiarizationRequest) -> dict[str, Any]:
    audio = Path(request.audio_path).expanduser().resolve()
    if not audio.exists():
        raise HTTPException(status_code=400, detail=f"audio not found: {audio}")

    try:
        output = _run_pipeline(audio, request.model, request.num_speakers, input_sha256=request.input_sha256)
        projection = output if isinstance(output, SupportedDiarization) else None
        if projection is not None:
            output = projection.annotation
        annotation = (
            getattr(output, "exclusive_speaker_diarization", None)
            or getattr(output, "speaker_diarization", output)
        )
        if projection is not None:
            # Execute the same exact source coordinates that the receipt proves.
            # Decimal display rounding can collapse short, valid intersections.
            segments = [
                {key: row[key] for key in ("speaker_id", "start_sec", "end_sec")}
                for row in projection.source_support_projection["projected_annotation"]
            ]
        else:
            segments = _annotation_to_segments(annotation)
        result = {
            "backend": "diarization_service",
            "model": request.model,
            "segments": segments,
            "speakers": sorted({seg["speaker_id"] for seg in segments}),
        }
        if projection is not None:
            result.update(raw_annotation=projection.raw_annotation,
                          source_support_projection=projection.source_support_projection)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"diarization failed: {type(exc).__name__}: {exc}") from exc


def _run_pipeline(audio: Path, model: str, num_speakers: int | None, *, input_sha256: str | None = None) -> Any:
    pipeline = _load_pipeline(model)
    if _is_diarizen_model(model):
        return run_with_source_support(pipeline, audio, expected_sha256=input_sha256)

    kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = int(num_speakers)
    return pipeline(_pyannote_audio_input(audio), **kwargs)


def _load_pipeline(model: str) -> Any:
    with _PIPELINE_LOCK:
        if model in _PIPELINES:
            return _PIPELINES[model]

        if _is_diarizen_model(model):
            try:
                from diarizen.pipelines.inference import DiariZenPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "diarizen is unavailable in the diarization service environment. "
                    "Run x_editor/tools/catalog/diarization/setup.sh."
                ) from exc

            _allow_trusted_torch_checkpoints()
            pipeline = DiariZenPipeline.from_pretrained(model)
        else:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise RuntimeError("pyannote.audio is unavailable in the diarization service environment.") from exc

            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            try:
                pipeline = Pipeline.from_pretrained(model, token=hf_token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
            if pipeline is None:
                raise RuntimeError(f"could not load diarization model {model}; check HF model access/token")
            try:
                import torch

                if torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
            except Exception:
                pass

        _PIPELINES[model] = pipeline
        return pipeline


def _annotation_to_segments(annotation: Any) -> list[dict[str, Any]]:
    segments = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(_speaker_segment(turn, speaker))
    else:
        for item in annotation:
            if len(item) == 2:
                turn, speaker = item
            else:
                turn, speaker = item[0], item[-1]
            segments.append(_speaker_segment(turn, speaker))
    return segments


def _speaker_segment(turn: Any, speaker: Any) -> dict[str, Any]:
    return {
        "speaker_id": str(speaker),
        "start_sec": round(float(turn.start), 6),
        "end_sec": round(float(turn.end), 6),
    }


def _pyannote_audio_input(audio_path: Path) -> Any:
    try:
        import soundfile as sf
        import torch

        samples, sample_rate = sf.read(str(audio_path), always_2d=True)
        waveform = torch.from_numpy(samples.T).float()
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception:
        return str(audio_path)


def _is_diarizen_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "diarizen" in lowered or lowered.startswith("but-fit/")


def _allow_trusted_torch_checkpoints() -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torch

        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="X-Editor diarization backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
