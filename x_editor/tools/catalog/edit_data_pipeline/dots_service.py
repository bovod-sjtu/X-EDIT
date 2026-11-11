"""Original JSON service around the separately installed public Edit runtime."""
from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from x_editor.core.text_equivalence import equivalent_transcript_copy

ALLOWED_TAGS = {
    "text_edit": {"sub", "ins", "del"},
    "pitch_rate_edit": {"pitch", "rate"},
    "emotion_style_edit": {"emo"},
    "pause_edit": {"pause"},
}


class AudioPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: str = Field(min_length=1, max_length=128 * 1024 * 1024)
    format: str = Field(default="wav", max_length=20)


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"]
    audio: AudioPayload
    source_text: str = Field(min_length=1)
    instruction_st: str = Field(min_length=1)
    target_text: str | None = None
    num_steps: int = Field(default=10, ge=1, le=128)
    guidance_scale: float = Field(default=1.2, ge=0, le=10, allow_inf_nan=False)
    use_xvector: bool | Literal["auto"] = "auto"
    speaker_scale: float = Field(default=1.5, ge=0, le=10, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def create_app(runtime, *, identity: dict) -> FastAPI:
    from dots_tts.data.edit_instruction import (
        instruction_operation_tags, render_source_text, render_target_text,
    )
    import torch

    app = FastAPI(title="X·Edit dots.tts.edit adapter")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "backend": "dots_tts_edit", **identity}

    @app.options("/execute")
    def options():
        return {"methods": ["POST"]}

    @app.post("/execute")
    def execute(request: EditRequest):
        try:
            tags = instruction_operation_tags(request.instruction_st)
            if not tags or not tags <= ALLOWED_TAGS[request.operation]:
                raise ValueError("instruction_st tags do not match the selected operation")
            source = render_source_text(request.instruction_st)
            target = render_target_text(request.instruction_st)
            if not source.strip() or not target.strip():
                raise ValueError("instruction_st must render nonempty source and target transcripts")
            if not equivalent_transcript_copy(source, request.source_text):
                raise ValueError("instruction_st source does not match source_text")
            if request.target_text is not None and not equivalent_transcript_copy(target, request.target_text):
                raise ValueError("instruction_st target does not match target_text")
            content = base64.b64decode(request.audio.data, validate=True)
            audio, sample_rate = sf.read(io.BytesIO(content), always_2d=True)
            if not audio.size or not np.isfinite(audio).all() or sample_rate <= 0:
                raise ValueError("Source audio is empty or nonfinite")
        except (ValueError, sf.LibsndfileError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Model busy; retry explicitly after the active request")
        try:
            with tempfile.TemporaryDirectory(prefix="xedit-dots-") as directory:
                source_path = Path(directory) / "source.wav"
                sf.write(source_path, audio, sample_rate, subtype="FLOAT")
                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.inference_mode():
                    torch.manual_seed(request.seed)
                    result = runtime.generate_edit(
                        source_audio_path=str(source_path), instruction=request.instruction_st,
                        source_text=source, target_text=target,
                        use_xvector=request.use_xvector, speaker_scale=request.speaker_scale,
                        num_steps=request.num_steps, guidance_scale=request.guidance_scale,
                    )
            waveform = result["audio"].float().cpu().reshape(-1).numpy()
            if not waveform.size or not np.isfinite(waveform).all():
                raise RuntimeError("Model generated empty or nonfinite audio")
            buffer = io.BytesIO()
            sf.write(buffer, waveform, result["sample_rate"], format="WAV", subtype="PCM_16")
            settings = request.model_dump(include={"num_steps", "guidance_scale", "use_xvector", "speaker_scale", "seed"})
            return {
                "status": "ok", "backend": "dots_tts_edit", "operation": request.operation,
                "audio": {"data": base64.b64encode(buffer.getvalue()).decode(), "format": "wav"},
                "source_text": result["source_text"], "target_text": result["target_text"],
                "metadata": {**identity, "sampling": settings, "time_used": result.get("time_used"),
                             "source_text_transport": {"authored": request.source_text, "rendered": source}},
            }
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            lock.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8049)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--max-generate-length", type=int, default=500)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        parser.error("请自行下载并安装 dots.tts.edit；--model-path must name a local model directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from dots_tts.edit_runtime import DotsTtsEditRuntime
    import uvicorn

    runtime = DotsTtsEditRuntime.from_pretrained(
        str(model_path), precision="bfloat16", optimize=args.optimize,
        max_generate_length=args.max_generate_length,
    )
    identity = {
        "model_id": "dots-studio/dots.tts.edit", "model_revision": args.model_revision,
        "source_revision": args.source_revision, "revision_evidence": "operator-supplied; verify against deployment receipt",
        "precision": "bfloat16", "optimize": args.optimize,
        "max_generate_length": args.max_generate_length,
    }
    uvicorn.run(create_app(runtime, identity=identity), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
