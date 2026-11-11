"""audio-separator RoFormer backend service.

The backend owns the heavy `audio-separator` dependency and model cache. The
planner-visible X-Editor tool talks to this process over HTTP.
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
import os
import shutil
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

if __package__:
    from .time_axis import declare_source_aligned_audio
    from .stem_selection import select_stem
else:
    from time_axis import declare_source_aligned_audio
    from stem_selection import select_stem


DEFAULT_BS_ROFORMER_MODEL = os.getenv(
    "X_EDITOR_AUDIO_SEPARATOR_BS_ROFORMER_MODEL",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
)
DEFAULT_MELBAND_ROFORMER_MODEL = os.getenv(
    "X_EDITOR_AUDIO_SEPARATOR_MELBAND_ROFORMER_MODEL",
    "vocals_mel_band_roformer.ckpt",
)


class SeparationRequest(BaseModel):
    input_audio_path: str
    output_dir: str
    output_path: str | None = None
    model_family: str = "bs_roformer"
    model_filename: str | None = None
    output_format: str = "WAV"
    stem_policy: str = "vocals"


app = FastAPI(title="X-Editor Audio Separator RoFormer Backend", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Audio Separator RoFormer Backend", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    audio_separator_available = importlib.util.find_spec("audio_separator") is not None
    onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
    ready = audio_separator_available and onnxruntime_available
    return {
        "status": "healthy" if ready else "not_ready",
        "backend": "audio_separator_roformer",
        "audio_separator_available": audio_separator_available,
        "onnxruntime_available": onnxruntime_available,
        "inference_endpoint": "/separate",
        "default_bs_roformer_model": DEFAULT_BS_ROFORMER_MODEL,
        "default_melband_roformer_model": DEFAULT_MELBAND_ROFORMER_MODEL,
    }


@app.post("/separate")
def separate(request: SeparationRequest) -> dict[str, Any]:
    if importlib.util.find_spec("audio_separator") is None:
        raise HTTPException(
            status_code=503,
            detail="audio-separator is not installed in this backend environment; run setup.sh first.",
        )
    if importlib.util.find_spec("onnxruntime") is None:
        raise HTTPException(
            status_code=503,
            detail="onnxruntime is not installed in this backend environment; run setup.sh first.",
        )

    input_audio = Path(request.input_audio_path)
    if not input_audio.exists():
        raise HTTPException(status_code=400, detail=f"input audio not found: {input_audio}")

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = request.model_filename or _default_model_for_family(request.model_family)

    try:
        from audio_separator.separator import Separator

        separator_kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "output_format": request.output_format.upper(),
        }
        model_file_dir = os.getenv("X_EDITOR_AUDIO_SEPARATOR_MODEL_DIR")
        if model_file_dir:
            separator_kwargs["model_file_dir"] = model_file_dir
        separator = Separator(**separator_kwargs)
        separator.load_model(model_filename=model_filename)
        # Upstream 0.44.2 indexes the final RoFormer chunk relative to the
        # input end. Inputs shorter than that chunk require right padding.
        # Keep the original timeline, then trim every returned stem by samples.
        with tempfile.TemporaryDirectory(prefix="xedit-roformer-") as temporary:
            prepared, padding = _prepare_short_roformer_input(input_audio, separator, Path(temporary))
            produced = separator.separate(str(prepared))
            stem_paths = _resolve_produced_paths(output_dir, produced)
            if not stem_paths:
                raise RuntimeError("audio-separator finished without producing any stem files")
            if padding is not None:
                _trim_padded_stems(stem_paths, padding["source_duration_sec"])

        selected_path = _select_stem(stem_paths, request.stem_policy)
        output_path = None
        if request.output_path:
            output = Path(request.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_path, output)
            output_path = str(output)
        else:
            output_path = str(selected_path)

        result = {
            "output_audio_path": output_path,
            "selected_stem_path": str(selected_path),
            "stem_paths": [str(path) for path in stem_paths],
            "model_family": request.model_family,
            "model_filename": model_filename,
            "backend": "audio_separator_roformer",
            "input_padding": padding,
        }
        # The audited RoFormer implementation accumulates each predicted chunk
        # back at its original sample index. Pitch-shift processing is excluded.
        model = getattr(separator, "model_instance", None)
        if getattr(model, "is_roformer", False) and getattr(model, "pitch_shift", None) == 0:
            from importlib.metadata import version
            if version("audio-separator") == "0.44.2":
                receipt = declare_source_aligned_audio(
                    input_audio, [*stem_paths, Path(output_path)],
                    basis=("audio-separator 0.44.2 RoFormer source-index overlap-add; pitch_shift=0"
                           + ("; right zero-padding followed by source-duration sample trim" if padding else "")),
                )
                if receipt is not None:
                    result["source_time_axis"] = receipt
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"audio-separator failed: {type(exc).__name__}: {exc}")


def _prepare_short_roformer_input(input_audio: Path, separator: Any, temporary: Path):
    import numpy as np
    import soundfile as sf

    model = getattr(separator, "model_instance", None)
    if not getattr(model, "is_roformer", False) or getattr(model, "pitch_shift", None) != 0:
        return input_audio, None
    info = sf.info(input_audio)
    duration = info.frames / info.samplerate
    config = model.model_data_cfgdict
    segment = (model.segment_size if model.override_model_segment_size or duration < 10.0
               else config.inference.dim_t)
    hop = getattr(config.model, "stft_hop_length", config.audio.hop_length)
    chunk = int(hop) * (int(segment) - 1)
    sample_rate = int(model.sample_rate)
    if info.frames <= 0 or chunk <= 0 or sample_rate <= 0:
        raise ValueError("Invalid source or RoFormer chunk geometry")
    if duration * sample_rate >= chunk:
        return input_audio, None
    # One target-rate sample of margin covers resampling rounding at the end.
    needed = math.ceil((chunk + 1) * info.samplerate / sample_rate)
    audio, source_rate = sf.read(input_audio, dtype="float64", always_2d=True)
    padded = np.pad(audio, ((0, needed - len(audio)), (0, 0)))
    path = temporary / "padded_input.wav"
    sf.write(path, padded, source_rate, subtype="DOUBLE")
    return path, {"side": "right", "mode": "zero", "source_frames": info.frames,
                  "source_sample_rate": source_rate, "source_duration_sec": duration,
                  "added_frames": needed - info.frames, "model_chunk_frames": chunk,
                  "model_sample_rate": sample_rate}


def _trim_padded_stems(paths: list[Path], source_duration_sec: float) -> None:
    import soundfile as sf

    for path in paths:
        info = sf.info(path)
        wanted = round(source_duration_sec * info.samplerate)
        audio, rate = sf.read(path, dtype="float64", always_2d=True)
        if wanted <= 0 or len(audio) < wanted:
            raise ValueError("Separated stem is shorter than the original source")
        sf.write(path, audio[:wanted], rate, format=info.format, subtype=info.subtype)


def _default_model_for_family(model_family: str) -> str:
    normalized = model_family.strip().lower().replace("-", "_")
    if normalized in {"melband_roformer", "mel_band_roformer", "melband"}:
        return DEFAULT_MELBAND_ROFORMER_MODEL
    if normalized in {"bs_roformer", "bsroformer", "roformer"}:
        return DEFAULT_BS_ROFORMER_MODEL
    if os.getenv("X_EDITOR_AUDIO_SEPARATOR_DEFAULT_MODEL"):
        return os.environ["X_EDITOR_AUDIO_SEPARATOR_DEFAULT_MODEL"]
    raise HTTPException(status_code=400, detail=f"unsupported model_family: {model_family}")


def _resolve_produced_paths(output_dir: Path, produced: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(produced, (str, Path)):
        produced_items = [produced]
    elif isinstance(produced, list):
        produced_items = produced
    else:
        produced_items = []

    for item in produced_items:
        path = Path(str(item))
        if not path.is_absolute():
            path = output_dir / path
        if path.exists() and path.is_file():
            paths.append(path)

    if not paths:
        paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3"})
    return sorted({path.resolve() for path in paths})


def _select_stem(stem_paths: list[Path], stem_policy: str) -> Path:
    try:
        return select_stem(stem_paths, stem_policy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="audio-separator RoFormer backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8042)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
