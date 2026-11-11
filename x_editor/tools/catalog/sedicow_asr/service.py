"""SE-DiCoW speaker-attributed ASR service wrapper."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SegmentHint(BaseModel):
    speaker_id: str | None = None
    start_sec: float
    end_sec: float
    text: str | None = None


class SpeakerTranscribeRequest(BaseModel):
    audio_path: str
    speaker_id: str | None = None
    speaker_namespace: str | None = None
    diarization_segments: list[SegmentHint] = Field(default_factory=list)
    qwen_text: str | None = None
    language: str | None = None
    reason: str | None = None


app = FastAPI(title="X-Editor SE-DiCoW ASR Backend", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "sedicow_asr", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    command = _command_template()
    repo = _official_repo()
    repo_exists = bool(repo and Path(repo).expanduser().exists())
    ready = bool(command) and (repo is None or repo_exists)
    return {
        "status": "healthy" if ready else "not_ready",
        "backend": "sedicow_asr",
        "command_configured": bool(command),
        "command_template": command,
        "adapter": str((Path(__file__).resolve().parent / "sedicow_infer.py")),
        "inference_python": _inference_python(),
        "official_repo": repo,
        "official_repo_exists": repo_exists,
        "hf_home": os.getenv("SEDICOW_HF_HOME") or os.getenv("HF_HOME"),
        "hf_modules_cache": os.getenv("SEDICOW_HF_MODULES_CACHE") or os.getenv("HF_MODULES_CACHE"),
        "pythonpath_entries": [item for item in os.getenv("PYTHONPATH", "").split(os.pathsep) if item],
    }


@app.post("/transcribe_speaker")
def transcribe_speaker(request: SpeakerTranscribeRequest) -> dict[str, Any]:
    audio = Path(request.audio_path).expanduser().resolve()
    if not audio.exists():
        raise HTTPException(status_code=400, detail=f"audio not found: {audio}")
    command = _command_template()
    if not command:
        raise HTTPException(
            status_code=503,
            detail=(
                "SEDICOW_TRANSCRIBE_COMMAND is not configured and the default "
                "official DiCoW adapter assets were not found. Run download_official_assets.py."
            ),
        )

    payload = request.model_dump()
    payload["audio_path"] = str(audio)
    with tempfile.TemporaryDirectory(prefix="xeditor_sedicow_") as tmp:
        request_json = Path(tmp) / "request.json"
        request_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        resolved = _resolve_command(command, request_json=request_json, audio=audio, request=request)
        try:
            completed = subprocess.run(
                resolved,
                cwd=_official_repo() or None,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(os.getenv("SEDICOW_TRANSCRIBE_TIMEOUT", "600")),
                check=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SE-DiCoW command failed: {type(exc).__name__}: {exc}") from exc
        if completed.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"SE-DiCoW command exited {completed.returncode}: {completed.stderr[-4000:]}",
            )
        try:
            data = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"SE-DiCoW command did not emit JSON: {completed.stdout[:1000]}",
            ) from exc
    return _normalize_response(data, request)


def _command_template() -> str:
    explicit = str(os.getenv("SEDICOW_TRANSCRIBE_COMMAND") or "").strip()
    if explicit:
        return explicit
    catalog_dir = Path(__file__).resolve().parent
    adapter = catalog_dir / "sedicow_infer.py"
    if not adapter.exists() or not _official_repo():
        return ""
    python = _inference_python()
    return f"{shlex.quote(python)} {shlex.quote(str(adapter))} --request-json {{request_json}}"


def _official_repo() -> str | None:
    explicit = str(os.getenv("SEDICOW_REPO") or "").strip()
    if explicit:
        return explicit
    repo = Path(__file__).resolve().parent / "vendor" / "DiCoW"
    if repo.exists():
        return str(repo)
    return None


def _inference_python() -> str:
    explicit = str(os.getenv("SEDICOW_INFERENCE_PYTHON") or "").strip()
    if explicit:
        return explicit
    return sys.executable


def _resolve_command(
    command: str,
    *,
    request_json: Path,
    audio: Path,
    request: SpeakerTranscribeRequest,
) -> list[str]:
    formatted = command.format(
        request_json=str(request_json),
        audio_path=str(audio),
        speaker_id=request.speaker_id or "",
        language=request.language or "",
    )
    return shlex.split(formatted)


def _normalize_response(data: Any, request: SpeakerTranscribeRequest) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="SE-DiCoW response must be a JSON object")
    if data.get("selection_status") in {"unresolved_namespace", "unmatched_speaker"}:
        raise HTTPException(status_code=422, detail={
            "code": data["selection_status"], "requested_speaker": request.speaker_id,
            "requested_namespace": request.speaker_namespace,
            "available_speaker_ids": data.get("available_speaker_ids", []),
            "message": "Request a complete transcript or an ID from this audio's SE-DiCoW report; external speaker numbers are not identity mappings.",
        })
    raw_segments = data.get("segments") or data.get("speaker_segments") or []
    segments = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        if speaker_id is None:
            speaker_id = item.get("speaker", request.speaker_id)
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if speaker_id is None or start is None or end is None:
            continue
        segments.append(
            {
                "speaker_id": str(speaker_id),
                "start_sec": round(float(start), 6),
                "end_sec": round(float(end), 6),
                "text": str(item.get("text") or item.get("transcript") or "").strip(),
                "confidence": item.get("confidence"),
            }
        )
    return {
        "backend": "sedicow_asr",
        "text": str(data.get("text") or "").strip(),
        "language": data.get("language") or request.language,
        "speaker_id": request.speaker_id,
        "speaker_namespace": data.get("speaker_namespace"),
        "selection_status": data.get("selection_status"),
        "segments": segments,
        "raw": data,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="X-Editor SE-DiCoW ASR backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8046)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
