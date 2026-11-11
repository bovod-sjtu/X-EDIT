"""MOSS-Transcribe-Diarize service wrapper for X-Editor perception."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # Keep parser/helper imports usable from the main repo env.
    uvicorn = None
    _SERVICE_DEPENDENCY_ERROR: ImportError | None = exc

    class HTTPException(RuntimeError):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class BaseModel:
        def __init__(self, **data: Any):
            for key in getattr(self, "__annotations__", {}):
                setattr(self, key, data.get(key, getattr(self.__class__, key, None)))

        def dict(self) -> dict[str, Any]:
            return {key: getattr(self, key, None) for key in getattr(self, "__annotations__", {})}

        def model_dump(self) -> dict[str, Any]:
            return self.dict()

    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any):
            self.missing_dependency = _SERVICE_DEPENDENCY_ERROR

        def get(self, *args: Any, **kwargs: Any):
            return lambda func: func

        def post(self, *args: Any, **kwargs: Any):
            return lambda func: func
else:
    _SERVICE_DEPENDENCY_ERROR = None


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
_PERSISTENT_RUNNER: Any | None = None


class MossTranscribeRequest(BaseModel):
    audio_path: str
    qwen_text: str | None = None
    language: str | None = None
    reason: str | None = None


app = FastAPI(title="X-Editor MOSS-Transcribe-Diarize Backend", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "moss_transcribe_diarize", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    command = _command_template()
    openai_base_url = _openai_base_url()
    persistent_local = _persistent_local_enabled()
    ready = bool(command or openai_base_url or persistent_local)
    return {
        "status": "healthy" if ready else "not_ready",
        "backend": "moss_transcribe_diarize",
        "openai_base_url": openai_base_url,
        "model": _model_id(),
        "persistent_local": persistent_local,
        "persistent_local_loaded": _PERSISTENT_RUNNER is not None,
        "command_configured": bool(command),
        "command_template": command,
        "adapter": str(Path(__file__).resolve().parent / "moss_infer.py"),
        "inference_python": _inference_python(),
        "official_repo": _official_repo(),
        "capability_boundary": (
            "Chinese speaker-aware initialization evidence only; Qwen3-ASR remains "
            "the wording prior and StrongOmni/Gemini review remains downstream."
        ),
    }


@app.post("/transcribe_diarize")
def transcribe_diarize(request: MossTranscribeRequest) -> dict[str, Any]:
    audio = Path(request.audio_path).expanduser().resolve()
    if not audio.exists():
        raise HTTPException(status_code=400, detail=f"audio not found: {audio}")
    try:
        if _openai_base_url():
            data = _run_openai_compatible(audio, request)
        elif _persistent_local_enabled():
            data = _run_persistent_local(audio, request)
        elif _command_template():
            data = _run_command(audio, request)
        else:
            raise HTTPException(
                status_code=503,
                detail="Configure MOSS_OPENAI_BASE_URL for SGLang/vLLM or MOSS_TRANSCRIBE_COMMAND for local inference.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MOSS inference failed: {type(exc).__name__}: {exc}") from exc
    return _normalize_response(data, request)


def _run_openai_compatible(audio: Path, request: MossTranscribeRequest) -> dict[str, Any]:
    url = f"{_openai_base_url()}/v1/audio/transcriptions"
    data: dict[str, Any] = {
        "model": _model_id(),
        "response_format": os.getenv("MOSS_RESPONSE_FORMAT", "verbose_json"),
        "temperature": os.getenv("MOSS_TEMPERATURE", "0"),
        "max_new_tokens": os.getenv("MOSS_MAX_NEW_TOKENS", "65536"),
    }
    if request.language:
        data["language"] = request.language
    prompt = _prompt(request)
    if prompt:
        data["prompt"] = prompt
    with audio.open("rb") as handle:
        response = requests.post(
            url,
            data=data,
            files={"file": (audio.name, handle, _audio_mime_type(audio))},
            timeout=float(os.getenv("MOSS_TRANSCRIBE_TIMEOUT", "900")),
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"MOSS OpenAI-compatible HTTP {response.status_code}: {response.text[:4000]}",
        )
    if data["response_format"] == "text":
        return {"text": response.text}
    return response.json()


def _run_command(audio: Path, request: MossTranscribeRequest) -> dict[str, Any]:
    payload = _request_payload(request)
    payload["audio_path"] = str(audio)
    with tempfile.TemporaryDirectory(prefix="xeditor_moss_") as tmp:
        request_json = Path(tmp) / "request.json"
        request_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        command = _resolve_command(_command_template(), request_json=request_json, audio=audio, request=request)
        completed = subprocess.run(
            command,
            cwd=_official_repo() or None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(os.getenv("MOSS_TRANSCRIBE_TIMEOUT", "900")),
            check=False,
        )
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"MOSS command exited {completed.returncode}: {completed.stderr[-4000:]}",
        )
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {"text": completed.stdout.strip(), "stderr": completed.stderr[-4000:]}


def _request_payload(request: MossTranscribeRequest) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


def _run_persistent_local(audio: Path, request: MossTranscribeRequest) -> dict[str, Any]:
    runner = _persistent_runner()
    result = runner.transcribe(
        audio,
        prompt=_prompt(request) or _default_moss_prompt(),
        max_new_tokens=int(os.getenv("MOSS_MAX_NEW_TOKENS", "4096")),
        decoding="greedy",
    )
    return {
        "backend": "moss_transcribe_diarize_persistent_local",
        "model": _model_id(),
        "text": result.text,
        "segments": parse_moss_transcript(result.text),
        "raw": {key: value for key, value in result.to_dict().items() if key != "text"},
    }


def _persistent_runner() -> Any:
    global _PERSISTENT_RUNNER
    if _PERSISTENT_RUNNER is not None:
        return _PERSISTENT_RUNNER
    repo = _official_repo()
    if repo:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
    try:
        from moss_transcribe_diarize.app.model_runner import ModelRunner
    except Exception as exc:
        raise RuntimeError(
            "Persistent local MOSS runner requires the official repo on PYTHONPATH. "
            "Set MOSS_REPO to the pinned MOSS-Transcribe-Diarize checkout; see docs/DEPLOYMENT.md."
        ) from exc
    _PERSISTENT_RUNNER = ModelRunner(
        _model_id(),
        device=os.getenv("MOSS_DEVICE", "auto"),
        dtype=os.getenv("MOSS_DTYPE", "bf16"),
    )
    return _PERSISTENT_RUNNER


def _normalize_response(data: Any, request: MossTranscribeRequest) -> dict[str, Any]:
    if isinstance(data, str):
        data = {"text": data}
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="MOSS response must be a JSON object or transcript string")
    text = str(data.get("text") or data.get("transcript") or data.get("output") or "").strip()
    segments = parse_moss_transcript(text) if text else []
    if not segments:
        segments = _segments_from_payload(data)
    if not text:
        text = " ".join(item.get("text", "") for item in segments).strip()
    return {
        "backend": "moss_transcribe_diarize",
        "model": data.get("model") or _model_id(),
        "text": text,
        "language": data.get("language") or request.language,
        "segments": segments,
        "raw": data,
    }


def _segments_from_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = data.get("segments") or data.get("speaker_segments") or []
    segments = []
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        text = str(item.get("text") or item.get("transcript") or "").strip()
        speaker = item.get("speaker_id", item.get("speaker", item.get("label")))
        if speaker is None:
            parsed = parse_moss_transcript(text)
            if parsed:
                segments.extend(parsed)
                continue
            speaker = f"S{index:02d}"
        if start is None or end is None:
            continue
        segments.append(_segment(speaker, start, end, text, source="moss_payload"))
    return sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"]))


def parse_moss_transcript(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"\[(?P<start>\d+(?:\.\d+)?)\]\s*\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
        flags=re.IGNORECASE | re.DOTALL,
    )
    segments = []
    for match in pattern.finditer(str(text or "")):
        body = re.sub(r"\s+", " ", match.group("text")).strip()
        if not body:
            continue
        segments.append(_segment(match.group("speaker"), match.group("start"), match.group("end"), body, source="moss_compact_text"))
    return sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"]))


def _segment(speaker: Any, start: Any, end: Any, text: str, *, source: str) -> dict[str, Any]:
    return {
        "speaker_id": _normalize_speaker_id(speaker),
        "start_sec": round(float(start), 6),
        "end_sec": round(float(end), 6),
        "text": text,
        "source": source,
    }


def _normalize_speaker_id(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text or "0"


def _prompt(request: MossTranscribeRequest) -> str | None:
    base = os.getenv("MOSS_PROMPT")
    if base:
        return base
    if request.language and str(request.language).lower() in {"zh", "zho", "chi", "chinese", "mandarin", "cmn"}:
        return (
            "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
            "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
        )
    return None


def _openai_base_url() -> str:
    return str(os.getenv("MOSS_OPENAI_BASE_URL") or "").rstrip("/")


def _model_id() -> str:
    return os.getenv("MOSS_OPENAI_MODEL") or os.getenv("MOSS_MODEL_ID") or DEFAULT_MODEL


def _persistent_local_enabled() -> bool:
    return str(os.getenv("MOSS_PERSISTENT_LOCAL") or "").strip().lower() in {"1", "true", "yes", "on"}


def _default_moss_prompt() -> str:
    try:
        from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

        return str(DEFAULT_PROMPT)
    except Exception:
        return ""


def _audio_mime_type(audio: Path) -> str:
    suffix = audio.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".flac":
        return "audio/flac"
    return "audio/wav"


def _command_template() -> str:
    explicit = str(os.getenv("MOSS_TRANSCRIBE_COMMAND") or "").strip()
    if explicit:
        return explicit
    adapter = Path(__file__).resolve().parent / "moss_infer.py"
    repo = _official_repo()
    if adapter.exists() and repo:
        return f"{shlex.quote(_inference_python())} {shlex.quote(str(adapter))} --request-json {{request_json}}"
    return ""


def _official_repo() -> str | None:
    explicit = str(os.getenv("MOSS_REPO") or "").strip()
    if explicit:
        return explicit
    repo = Path(__file__).resolve().parent / "vendor" / "MOSS-Transcribe-Diarize"
    if repo.exists():
        return str(repo)
    return None


def _inference_python() -> str:
    return os.getenv("MOSS_TRANSCRIBE_INFERENCE_PYTHON") or os.getenv("MOSS_TRANSCRIBE_SERVICE_PYTHON") or sys.executable


def _resolve_command(command: str, *, request_json: Path, audio: Path, request: MossTranscribeRequest) -> list[str]:
    return shlex.split(
        command.format(
            request_json=str(request_json),
            audio_path=str(audio),
            language=request.language or "",
        )
    )


def main() -> None:
    import argparse

    if _SERVICE_DEPENDENCY_ERROR is not None or uvicorn is None:
        raise RuntimeError(
            "MOSS-Transcribe-Diarize service dependencies are missing. "
            "Run x_editor/tools/catalog/moss_transcribe_diarize/setup.sh or use MOSS_TRANSCRIBE_SERVICE_PYTHON."
        ) from _SERVICE_DEPENDENCY_ERROR

    parser = argparse.ArgumentParser(description="X-Editor MOSS-Transcribe-Diarize backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8048)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
