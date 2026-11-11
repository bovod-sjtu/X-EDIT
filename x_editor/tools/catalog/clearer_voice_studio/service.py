"""ClearerVoice speech-enhancement backend with explicit integration boundary.

This service is for single-speaker speech enhancement through ClearVoice.
The audio-separator RoFormer backend separates vocals from accompaniment, not
individual overlapping speakers. Target-speaker extraction needs a separately
available and qualified tool.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any

if __package__:
    from .backend_readiness import require_model_checkpoints, select_visible_cuda_device
else:
    from backend_readiness import require_model_checkpoints, select_visible_cuda_device

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class TSERequest(BaseModel):
    mixture_audio_path: str
    target_speaker_hint: str
    reference_audio_path: str | None = None
    output_dir: str
    return_residual: bool = True


class EnhancementRequest(BaseModel):
    input_audio_path: str
    output_path: str
    mode: str = "general"
    preserve_background: bool = False
    enhancement_goal: str | None = None


app = FastAPI(title="X-Editor ClearerVoice-Studio Backend", version="0.1.0")
_ENHANCEMENT_MODEL: Any = None
_ENHANCEMENT_MODEL_LOCK = threading.Lock()


def _get_enhancement_model() -> Any:
    global _ENHANCEMENT_MODEL
    with _ENHANCEMENT_MODEL_LOCK:
        if _ENHANCEMENT_MODEL is None:
            _install_torch_compile_compat()
            from clearvoice import ClearVoice
            from clearvoice.networks import SpeechModel
            import torch

            # ClearVoice 0.1.2 enumerates physical nvidia-smi indices, which are
            # invalid when this service sees a remapped CUDA device subset.
            def free_visible_gpu(_model):
                return select_visible_cuda_device(torch.cuda)

            SpeechModel.get_free_gpu = free_visible_gpu
            candidate = ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormer2_SE_48K"],
            )
            require_model_checkpoints(candidate)
            _ENHANCEMENT_MODEL = candidate
    return _ENHANCEMENT_MODEL


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "ClearerVoice Backend", "status": "running", "docs": "/docs"}


@app.get("/health")
def health(deep: bool = False) -> dict[str, Any]:
    clearvoice_available = importlib.util.find_spec("clearvoice") is not None
    if deep and clearvoice_available:
        try:
            _get_enhancement_model()
        except (ImportError, OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "ClearerVoice model is not ready: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
    return {
        "status": "healthy" if clearvoice_available else "not_ready",
        "backend": "clearer_voice_studio",
        "clearvoice_available": clearvoice_available,
        "deep_check": deep,
        "model_loaded": _ENHANCEMENT_MODEL is not None,
        "torch_compile_compat": _torch_compile_compat_available(),
        "speech_enhancement": "MossFormer2_SE_48K",
        "tse": "not_available_audio_only; installed ClearVoice exposes AV_MossFormer2_TSE_16K",
        "copy_fallback_enabled": _copy_fallback_enabled(),
        "inference_endpoint": "/enhance",
    }


@app.post("/tse/extract")
def extract_target_speaker(_: TSERequest) -> dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Audio-only TSE is not available in the installed ClearVoice package. The available TSE model is AV_MossFormer2_TSE_16K and requires audio-visual lip cue inputs.",
    )


@app.post("/enhance")
def enhance(request: EnhancementRequest) -> dict[str, Any]:
    input_path = Path(request.input_audio_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"input audio not found: {request.input_audio_path}")
    output_path = Path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = _get_enhancement_model()
        with tempfile.TemporaryDirectory(prefix="clearvoice_enhance_") as tmp:
            normalized_input = Path(tmp) / "input.wav"
            _normalize_wav(input_path, normalized_input)
            model(input_path=str(normalized_input), online_write=True, output_path=tmp)
            wavs = list(Path(tmp).rglob("*.wav"))
            wavs = [p for p in wavs if p.resolve() != normalized_input.resolve()]
            if not wavs:
                raise RuntimeError("ClearVoice did not produce a wav file")
            shutil.copy2(wavs[0], output_path)
        return {"output_audio_path": str(output_path)}
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "clearvoice":
            raise HTTPException(
                status_code=503,
                detail="clearvoice runtime is not installed in this backend environment; start service.py with a ClearVoice runtime Python or run X_EDITOR_CLEARERVOICE_INSTALL_FULL_DEPS=1 ./setup.sh.",
            )
        if _copy_fallback_enabled():
            try:
                shutil.copy2(input_path, output_path)
                return {
                    "output_audio_path": str(output_path),
                    "warning": f"ClearVoice enhancement fallback copied input: {type(exc).__name__}: {exc}",
                }
            except Exception as copy_exc:
                raise HTTPException(status_code=500, detail=f"ClearVoice enhancement failed: {type(exc).__name__}: {exc}; fallback failed: {copy_exc}")
        raise HTTPException(status_code=500, detail=f"ClearVoice enhancement failed: {type(exc).__name__}: {exc}")


def _normalize_wav(input_path: Path, output_path: Path) -> None:
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.99
    sf.write(str(output_path), audio, sr, subtype="PCM_16")


def _copy_fallback_enabled() -> bool:
    return os.getenv("X_EDITOR_CLEARVOICE_ALLOW_COPY_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}


def _install_torch_compile_compat() -> bool:
    """Provide a tiny ``torch._compile`` shim for old torch runtimes.

    Some ClearVoice dependency stacks import ``torch._compile`` even when they
    only need decorator-like compile/disable helpers. Older DSW images can ship
    a torch build without that private module, causing ClearVoice import to fail
    before inference starts. The shim is intentionally identity-only.
    """

    if "torch._compile" in sys.modules:
        return False
    try:
        import torch
    except Exception:
        return False
    try:
        if importlib.util.find_spec("torch._compile") is not None:
            return False
    except (ImportError, ValueError):
        pass

    module = types.ModuleType("torch._compile")

    def identity_decorator(fn: Any = None, *_args: Any, **_kwargs: Any) -> Any:
        if callable(fn):
            return fn

        def decorate(inner: Any) -> Any:
            return inner

        return decorate

    module.compile = getattr(torch, "compile", identity_decorator)
    module.disable = identity_decorator
    module._disable_dynamo = identity_decorator

    def missing_attr(name: str) -> Any:
        if name in {"compile", "disable", "_disable_dynamo"}:
            return identity_decorator
        raise AttributeError(name)

    module.__getattr__ = missing_attr  # type: ignore[attr-defined]
    sys.modules["torch._compile"] = module
    return True


def _torch_compile_compat_available() -> bool:
    try:
        return importlib.util.find_spec("torch._compile") is not None or "torch._compile" in sys.modules
    except (ImportError, ValueError):
        return "torch._compile" in sys.modules


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ClearerVoice-Studio backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
