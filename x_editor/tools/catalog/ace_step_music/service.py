"""Optional X-Editor HTTP adapter for an official ACE-Step 1.5 API server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from client import AceStepMusicClient


class GenerateRequest(BaseModel):
    instruction: str
    output_path: str
    lyrics: str = ""
    music_prompt: str = ""
    source_audio_path: str | None = None
    reference_audio_path: str | None = None
    mode: str = "text2music"
    vocal_style: str = "singing"
    vocal_language: str = "unknown"
    duration_seconds: float = 10.0
    audio_format: str = "wav"
    repainting_start: float = 0.0
    repainting_end: float = -1.0
    audio_cover_strength: float = 0.8
    inference_steps: int = 8
    guidance_scale: float = 7.0
    seed: int = -1
    use_random_seed: bool = True
    thinking: bool = True
    use_format: bool = True
    bpm: int | None = None
    key_scale: str | None = None
    time_signature: str | None = None
    model: str | None = None
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = Field(default=1800.0, ge=1.0)


app = FastAPI(title="X-Editor ACE-Step Music Adapter", version="0.1.0")


def _client(timeout: float = 1800.0, poll_interval: float = 2.0) -> AceStepMusicClient:
    return AceStepMusicClient(
        os.getenv("ACE_STEP_API_URL") or os.getenv("ACE_STEP_UPSTREAM_URL") or os.getenv("ACE_STEP_SERVICE_URL"),
        timeout=timeout,
        poll_interval=poll_interval,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "ace_step_music", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    upstream = _client()
    try:
        upstream_health = upstream.health()
        upstream_compatibility = upstream.validate_api_compatibility()
        available = True
        error = None
    except Exception as exc:  # pragma: no cover - depends on external service
        upstream_health = None
        upstream_compatibility = None
        available = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "healthy" if available else "degraded",
        "backend": "ace_step_1_5_api",
        "upstream_url": upstream.base_url,
        "upstream_available": available,
        "upstream_error": error,
        "upstream_health": upstream_health,
        "upstream_compatibility": upstream_compatibility,
        "capabilities": {
            "music_vocal_generation": True,
            "singing": True,
            "rap": True,
            "task_types": ["text2music", "cover", "repaint"],
            "guarantees": ["output_file_if_upstream_succeeds"],
            "non_guarantees": ["speaker_identity", "exact_timing", "speech_pause_preservation"],
        },
    }


@app.post("/generate_or_edit")
def generate_or_edit(request: GenerateRequest) -> dict[str, Any]:
    try:
        return _client(request.timeout_seconds, request.poll_interval_seconds).generate_or_edit(**request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"reason": str(exc), "backend": "ace_step_1_5_api"}) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional X-Editor ACE-Step adapter service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8043)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
