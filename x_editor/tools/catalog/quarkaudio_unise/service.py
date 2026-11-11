"""QuarkAudio-UniSE backend service for audio-only target speaker extraction."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.getenv("QUARKAUDIO_UNISE_REPO_DIR") or BACKEND_DIR / "vendor" / "QuarkAudio-UniSE").expanduser()
CKPT_PATH = REPO_DIR / "checkpoints" / "epoch=20-step=109367.ckpt"
BICODEC_CONFIG = REPO_DIR / "checkpoints" / "BiCodec" / "config.yaml"
BICODEC_MODEL = REPO_DIR / "checkpoints" / "BiCodec" / "model.safetensors"
WAV2VEC_CONFIG = REPO_DIR / "checkpoints" / "wav2vec2-large-xlsr-53" / "config.json"
MODEL_RUNTIME_IMPORTS = (
    "torch",
    "torchaudio",
    "pytorch_lightning",
    "transformers",
    "soundfile",
    "librosa",
    "numpy",
    "scipy",
    "omegaconf",
    "safetensors",
    "einops",
    "einx",
    "soxr",
    "x_transformers",
)


class TSERequest(BaseModel):
    mixture_audio_path: str
    reference_audio_path: str
    output_path: str
    enroll_duration: float = 5.0


app = FastAPI(title="X-Editor QuarkAudio-UniSE Backend", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "QuarkAudio-UniSE Backend", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    dependency_status = _runtime_dependency_status()
    critical_import_errors = _critical_runtime_import_errors()
    return {
        "status": "healthy" if _runtime_ready() else "not_ready",
        "backend": "quarkaudio_unise",
        "repo_dir": str(REPO_DIR),
        "repo_dir_env": "QUARKAUDIO_UNISE_REPO_DIR",
        "ckpt_present": CKPT_PATH.exists(),
        "bicodec_present": BICODEC_CONFIG.exists() and BICODEC_MODEL.exists(),
        "wav2vec_present": WAV2VEC_CONFIG.exists(),
        "runtime_dependencies": dependency_status,
        "critical_import_errors": critical_import_errors,
        "mode": "tse",
        "inference_endpoint": "/tse/extract",
    }


@app.post("/tse/extract")
def extract_target_speaker(request: TSERequest) -> dict[str, Any]:
    if not _runtime_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "QuarkAudio-UniSE checkpoints are incomplete. Required: "
                "QuarkAudio/QuarkAudio-UniSE epoch ckpt and SparkAudio/Spark-TTS-0.5B BiCodec/wav2vec files under checkpoints/."
            ),
        )
    mixture = Path(request.mixture_audio_path)
    reference = Path(request.reference_audio_path)
    if not mixture.exists():
        raise HTTPException(status_code=400, detail=f"mixture audio not found: {mixture}")
    if not reference.exists():
        raise HTTPException(status_code=400, detail=f"reference audio not found: {reference}")
    output_path = Path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="unise_tse_") as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src"
            enroll_dir = tmp_path / "enroll"
            tgt_dir = tmp_path / "tgt"
            save_dir = tmp_path / "out"
            for d in [src_dir, enroll_dir, tgt_dir, save_dir]:
                d.mkdir(parents=True, exist_ok=True)
            name = "input.wav"
            shutil.copy2(mixture, src_dir / name)
            shutil.copy2(reference, enroll_dir / name)
            shutil.copy2(mixture, tgt_dir / name)
            config_path = tmp_path / "config.yaml"
            config = _build_config(src_dir, enroll_dir, tgt_dir, request.enroll_duration)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            subprocess.run(
                [sys.executable, "test.py", "--config", str(config_path), "--save_enhanced", str(save_dir)],
                cwd=str(REPO_DIR),
                env={**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")},
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            produced = sorted(save_dir.glob("*.wav"))
            if not produced:
                raise RuntimeError("UniSE finished without producing a wav file")
            shutil.copy2(produced[0], output_path)
        return {"target_audio_path": str(output_path), "output_audio_path": str(output_path)}
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"UniSE failed: {exc.stderr[-2000:]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"UniSE TSE failed: {type(exc).__name__}: {exc}")


def _runtime_ready() -> bool:
    return (
        REPO_DIR.exists()
        and CKPT_PATH.exists()
        and BICODEC_CONFIG.exists()
        and BICODEC_MODEL.exists()
        and WAV2VEC_CONFIG.exists()
        and all(_runtime_dependency_status().values())
        and not _critical_runtime_import_errors()
    )


def _runtime_dependency_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in MODEL_RUNTIME_IMPORTS}


def _critical_runtime_import_errors() -> dict[str, str]:
    errors: dict[str, str] = {}
    for name in ("torch", "torchaudio"):
        try:
            __import__(name)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return errors


def _build_config(src_dir: Path, enroll_dir: Path, tgt_dir: Path, enroll_duration: float) -> dict[str, Any]:
    with open(REPO_DIR / "conf" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["accelerator"] = "cpu"
    config["devices"] = 1
    config["ckpt_path"] = str(CKPT_PATH)
    config["codec_ckpt_dir"] = str(REPO_DIR / "checkpoints")
    config["dataset_config"]["test_kwargs"] = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch": 1,
        "mode": "tse",
        "data_enroll_dir": str(enroll_dir),
        "enroll_duration": enroll_duration,
        "data_src_dir": str(src_dir),
        "data_tgt_dir": str(tgt_dir),
    }
    return config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="QuarkAudio-UniSE backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
