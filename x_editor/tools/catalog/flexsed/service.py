"""FastAPI service for JHU-LCAP FlexSED open-vocabulary sound event detection.

The service is a thin wrapper around the official FlexSED ``api.FlexSED``
class.  It starts without importing heavy model dependencies so X-Editor can
report precise availability and bootstrap instructions before real inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .timestamps import (
        FRAME_RATE_HZ,
        FRAME_STEP_SEC,
        extract_event_timestamps,
        propose_event_peak_windows,
        summarize_window_events,
    )
except ImportError:  # pragma: no cover - direct script execution
    from timestamps import (
        FRAME_RATE_HZ,
        FRAME_STEP_SEC,
        extract_event_timestamps,
        propose_event_peak_windows,
        summarize_window_events,
    )

DEFAULT_RUNTIME_ROOT = str(Path.home() / ".cache" / "x-editor" / "artifacts")
DEFAULT_SERVICE_OUTPUT_DIR = str(Path(os.getenv("X_EDITOR_RUNTIME_ROOT", DEFAULT_RUNTIME_ROOT)) / "flexsed")
DEFAULT_CKPT_URL = "https://huggingface.co/Higobeatz/FlexSED/resolve/main/ckpts/flexsed_as.pt"


class DetectRequest(BaseModel):
    audio_path: str | None = None
    input_audio_path: str | None = None
    events: list[str] = Field(default_factory=list)
    event_list: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_duration_sec: float = Field(default=0.08, ge=0.0)
    merge_gap_sec: float = Field(default=0.08, ge=0.0)
    norm_audio: bool = True
    output_dir: str | None = None
    save_heatmap_png: bool = True
    save_heatmap_npy: bool = True
    include_scores: bool = False
    max_scores_json_frames: int = Field(default=2000, ge=1)
    fname: str | None = None
    start_sec: float | None = Field(default=None, ge=0.0)
    end_sec: float | None = Field(default=None, ge=0.0)
    p: float = Field(default=3.0, gt=0.0)
    score_mode: str = Field(default="max")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RuntimeState:
    model: Any = None
    model_error: str | None = None


app = FastAPI(title="X-Editor FlexSED Backend", version="0.1.0")
_STATE = RuntimeState()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "flexsed", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    availability = _availability()
    return {
        "status": "healthy" if availability["available"] else "degraded",
        "backend": "JHU-LCAP/FlexSED",
        "availability": availability,
        "capabilities": {
            "open_vocabulary_sound_event_detection": True,
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_step_sec": FRAME_STEP_SEC,
            "timestamps": "thresholded contiguous frame spans",
            "heatmap": ["png", "npy_matrix"],
        },
    }


@app.get("/availability")
def availability() -> dict[str, Any]:
    return _availability()


@app.post("/detect")
def detect(request: DetectRequest) -> dict[str, Any]:
    raw_audio_path = request.audio_path or request.input_audio_path
    if not raw_audio_path:
        _raise(400, "FlexSED requires audio_path or input_audio_path.")
    audio_path = Path(raw_audio_path).expanduser()
    if not audio_path.exists():
        _raise(400, f"input audio not found: {audio_path}")
    events = _normalize_events(request.events, request.event_list)
    if not events:
        _raise(400, "FlexSED requires at least one event label.")

    availability = _availability()
    if not availability["available"]:
        _raise(
            503,
            "FlexSED runtime is not available. Configure the official checkout and dependencies before inference.",
            availability=availability,
        )

    model = _ensure_model()
    started = time.time()
    try:
        preds = model.run_inference(str(audio_path), events, norm_audio=request.norm_audio)
    except Exception as exc:  # pragma: no cover - depends on real backend
        _raise(503, f"FlexSED inference failed: {type(exc).__name__}: {exc}", availability=availability)

    scores = _preds_to_scores(preds)
    if scores.shape[0] != len(events):
        _raise(
            502,
            f"FlexSED returned {scores.shape[0]} event rows for {len(events)} requested events.",
            boundary={"events": events, "scores_shape": list(scores.shape)},
        )
    duration_sec = _duration_seconds(audio_path)
    output_dir = Path(request.output_dir or DEFAULT_SERVICE_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(request.fname or audio_path.stem)

    heatmap_png_path: str | None = None
    heatmap_npy_path: str | None = None
    if request.save_heatmap_png:
        heatmap_png_path = _save_heatmap_png(model, preds, events, output_dir, stem)
    if request.save_heatmap_npy:
        heatmap_npy_path = str(output_dir / f"{stem}.flexsed_scores.npy")
        np.save(heatmap_npy_path, scores)

    timestamps = extract_event_timestamps(
        scores,
        events,
        threshold=request.threshold,
        min_duration_sec=request.min_duration_sec,
        merge_gap_sec=request.merge_gap_sec,
        frame_rate_hz=FRAME_RATE_HZ,
        audio_duration_sec=duration_sec,
    )
    candidate_windows = propose_event_peak_windows(
        scores,
        events,
        frame_rate_hz=FRAME_RATE_HZ,
        audio_duration_sec=duration_sec,
    )
    heatmap = {
        "frame_rate_hz": FRAME_RATE_HZ,
        "frame_step_sec": FRAME_STEP_SEC,
        "time_axis": "seconds = frame_index / frame_rate_hz; frame i covers [i/frame_rate, (i+1)/frame_rate]",
        "event_axis": "rows follow the request events order",
        "shape": [int(scores.shape[0]), int(scores.shape[1])],
        "png_path": heatmap_png_path,
        "npy_path": heatmap_npy_path,
    }
    result: dict[str, Any] = {
        "backend": "flexsed",
        "input_audio_path": str(audio_path),
        "events": events,
        "threshold": request.threshold,
        "min_duration_sec": request.min_duration_sec,
        "merge_gap_sec": request.merge_gap_sec,
        "frame_rate_hz": FRAME_RATE_HZ,
        "frame_step_sec": FRAME_STEP_SEC,
        "audio_duration_sec": duration_sec,
        "num_frames": int(scores.shape[1]),
        "timestamps": timestamps,
        "candidate_windows": candidate_windows,
        "heatmap": heatmap,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if request.start_sec is not None or request.end_sec is not None:
        if request.start_sec is None or request.end_sec is None:
            _raise(400, "Window summary requires both start_sec and end_sec.")
        try:
            window_summary = summarize_window_events(
                scores,
                events,
                start_sec=request.start_sec,
                end_sec=request.end_sec,
                p=request.p,
                frame_rate_hz=FRAME_RATE_HZ,
                audio_duration_sec=duration_sec,
                score_mode=request.score_mode,
                min_score=request.min_score,
            )
        except ValueError as exc:
            _raise(400, str(exc))
        result["window_event_summary"] = window_summary
        result["window_event_line"] = window_summary["line"]
    if request.include_scores:
        if scores.shape[1] > request.max_scores_json_frames:
            result["scores_json_omitted"] = {
                "reason": "too_many_frames",
                "num_frames": int(scores.shape[1]),
                "max_scores_json_frames": request.max_scores_json_frames,
                "npy_path": heatmap_npy_path,
            }
        else:
            result["scores"] = scores.tolist()
    return result


def _availability() -> dict[str, Any]:
    repo_dir = _repo_dir()
    config_path = _config_path(repo_dir)
    ckpt_path = _ckpt_path(repo_dir)
    api_path = repo_dir / "api.py" if repo_dir else None
    missing: list[str] = []
    if not repo_dir or not repo_dir.exists():
        missing.append("FLEXSED_REPO_DIR official checkout")
    if api_path and not api_path.exists():
        missing.append("api.py")
    if config_path and not config_path.exists():
        missing.append(str(config_path))
    deps = _dependency_status()
    missing.extend(item for item in deps["missing"] if item not in missing)
    return {
        "available": not missing,
        "repo_dir": str(repo_dir) if repo_dir else None,
        "api_path": str(api_path) if api_path else None,
        "config_path": str(config_path) if config_path else None,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "ckpt_exists": bool(ckpt_path and ckpt_path.exists()),
        "ckpt_url": os.getenv("FLEXSED_CKPT_URL", DEFAULT_CKPT_URL),
        "device": os.getenv("FLEXSED_DEVICE", "cuda"),
        "dependencies": deps,
        "missing": missing,
        "model_loaded": _STATE.model is not None,
        "model_error": _STATE.model_error,
    }


def _ensure_model() -> Any:
    if _STATE.model is not None:
        return _STATE.model
    repo_dir = _repo_dir()
    if not repo_dir:
        raise RuntimeError("FLEXSED_REPO_DIR is not configured.")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    try:
        from api import FlexSED  # type: ignore

        _STATE.model = FlexSED(
            config_path=str(_config_path(repo_dir)),
            ckpt_path=str(_ckpt_path(repo_dir)),
            ckpt_url=os.getenv("FLEXSED_CKPT_URL", DEFAULT_CKPT_URL),
            device=os.getenv("FLEXSED_DEVICE", "cuda"),
        )
    except Exception as exc:  # pragma: no cover - depends on real backend
        _STATE.model_error = f"{type(exc).__name__}: {exc}"
        raise
    _STATE.model_error = None
    return _STATE.model


def _repo_dir() -> Path | None:
    raw = os.getenv("FLEXSED_REPO_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    for candidate in (Path(__file__).resolve().parent / "vendor" / "FlexSED",):
        if candidate.exists():
            return candidate.resolve()
    return None


def _config_path(repo_dir: Path | None) -> Path | None:
    raw = os.getenv("FLEXSED_CONFIG_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_dir / "src" / "configs" / "model.yml") if repo_dir else None


def _ckpt_path(repo_dir: Path | None) -> Path | None:
    raw = os.getenv("FLEXSED_CKPT_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_dir / "ckpts" / "flexsed_as.pt") if repo_dir else None


def _dependency_status() -> dict[str, Any]:
    required = {
        "torch": "torch",
        "librosa": "librosa",
        "matplotlib": "matplotlib",
        "transformers": "transformers",
        "einops": "einops",
    }
    present: list[str] = []
    missing: list[str] = []
    for label, module_name in required.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(label)
        else:
            present.append(label)
    return {"present": present, "missing": missing}


def _preds_to_scores(preds: Any) -> np.ndarray:
    if hasattr(preds, "detach"):
        preds = preds.detach()
    if hasattr(preds, "cpu"):
        preds = preds.cpu()
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    matrix = np.asarray(preds, dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[1] == 1:
        matrix = np.squeeze(matrix, axis=1)
    if matrix.ndim != 2:
        raise ValueError(f"FlexSED predictions must be [events, 1, frames] or [events, frames], got {matrix.shape}")
    return np.clip(matrix, 0.0, 1.0)


def _save_heatmap_png(model: Any, preds: Any, events: list[str], output_dir: Path, stem: str) -> str:
    if hasattr(model, "to_multi_plot"):
        try:
            return str(model.to_multi_plot(preds, events, out_dir=str(output_dir), fname=f"{stem}.flexsed_heatmap"))
        except Exception:
            pass
    return _save_heatmap_png_fallback(_preds_to_scores(preds), events, output_dir, stem)


def _save_heatmap_png_fallback(scores: np.ndarray, events: list[str], output_dir: Path, stem: str) -> str:
    import matplotlib.pyplot as plt

    frame_count = scores.shape[1]
    plt.figure(figsize=(12, len(events) * 0.6 + 2))
    plt.imshow(
        scores,
        aspect="auto",
        cmap="Blues",
        extent=[0, frame_count / FRAME_RATE_HZ, 0, len(events)],
        vmin=0,
        vmax=1,
        origin="lower",
    )
    plt.colorbar(label="Probability")
    plt.yticks(np.arange(len(events)) + 0.5, events)
    plt.xlabel("Time (s)")
    plt.ylabel("Events")
    plt.title("FlexSED Event Predictions")
    save_path = output_dir / f"{stem}.flexsed_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(save_path)


def _normalize_events(events: list[str], event_list: str | None) -> list[str]:
    values = [str(item).strip() for item in events if str(item).strip()]
    if event_list:
        values.extend(item.strip() for item in re.split(r"[,\n;]+", event_list) if item.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                return round(wav.getnframes() / float(wav.getframerate()), 3)
        except Exception:
            return None
    return None


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "flexsed"


def _raise(status_code: int, reason: str, **extra: Any) -> None:
    detail = {"reason": reason, **extra}
    raise HTTPException(status_code=status_code, detail=detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the X-Editor FlexSED adapter service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8057)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
