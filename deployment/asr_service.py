"""X-Edit HTTP adapter for the separately installed, pinned Qwen3-ASR SDK."""
from __future__ import annotations

import argparse
import base64
import binascii
import inspect
import io
from pathlib import Path
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class AudioInput(BaseModel):
    data: str
    format: str = "wav"


class TranscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: AudioInput
    language: str | None = None
    backend: str | None = None
    context: str | None = Field(default=None, description="Unverified recognition hint, not a transcript")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[TranscribeRequest]
    backend: str | None = None


def normalize_language(value):
    language = (value or "").strip()
    alias = language.lower().replace("_", "-")
    if alias in {"", "auto"}:
        return None
    if alias in {"zh", "zho", "chi", "cmn", "cn", "chinese", "mandarin", "zh-cn", "zh-hans"}:
        return "Chinese"
    if alias in {"en", "eng", "english", "en-us", "en-gb"}:
        return "English"
    return language


def create_app(model, *, model_path: str, sdk_path: str = "") -> FastAPI:
    if "context" not in inspect.signature(model.transcribe).parameters:
        raise RuntimeError("The installed Qwen3-ASR SDK must explicitly support context")
    app = FastAPI(title="X-Edit Qwen3-ASR", version="1.0")
    inference_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "default_backend": "qwen3-asr"}

    @app.get("/info")
    def info():
        return {"default_backend": "qwen3-asr", "backends": {"qwen3-asr": {
            "inference": "transformers", "model": model_path, "sdk_path": sdk_path,
            "supports_context": True, "model_loaded": True,
            "buffer_batch_size": 1, "buffer_wait_ms": 0,
        }}}

    def transcribe_one(request, batch_backend=None):
        if (request.backend or batch_backend or "qwen3-asr") != "qwen3-asr":
            raise HTTPException(400, "Unsupported ASR backend; context is never silently discarded")
        try:
            waveform, sample_rate = sf.read(
                io.BytesIO(base64.b64decode(request.audio.data, validate=True)), dtype="float32")
            if not waveform.size or not np.isfinite(waveform).all():
                raise ValueError("Empty or non-finite audio")
        except (binascii.Error, ValueError, RuntimeError) as error:
            raise HTTPException(400, "Invalid audio input") from error
        language = normalize_language(request.language)
        if not np.any(waveform):
            return {"text": "", "language": language}
        arguments = {"audio": (waveform, sample_rate), "language": language}
        if request.context:
            arguments["context"] = request.context
        with inference_lock:
            results = model.transcribe(**arguments)
        if len(results) != 1:
            raise RuntimeError("ASR must return exactly one result for each input")
        return {"text": str(results[0].text), "language": results[0].language or language}

    @app.post("/transcribe")
    def transcribe(request: TranscribeRequest):
        return transcribe_one(request)

    @app.post("/transcribe_batch")
    def transcribe_batch(request: BatchRequest):
        return {"results": [transcribe_one(item, request.backend) for item in request.items]}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    model_path = args.model.resolve(strict=True)
    if not (model_path / "config.json").is_file():
        raise ValueError("Supply an explicitly downloaded local model snapshot")
    import torch
    import uvicorn
    import qwen_asr
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        str(model_path), dtype=getattr(torch, args.dtype), device_map=args.device,
        max_inference_batch_size=1, max_new_tokens=args.max_new_tokens)
    app = create_app(model, model_path=str(model_path), sdk_path=str(qwen_asr.__file__))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
