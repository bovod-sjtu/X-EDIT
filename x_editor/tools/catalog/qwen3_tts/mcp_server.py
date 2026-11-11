"""MCP wrapper for Qwen3-TTS synthesis."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from client import Qwen3TTSClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server

try:
    import requests
except ImportError:  # pragma: no cover - service environments should include requests
    requests = None


INPUT_SCHEMA = {
    "type": "object",
    "required": ["text", "output_path"],
    "properties": {
        "text": {"type": "string"},
        "output_path": {"type": "string"},
        "mode": {"type": "string", "enum": ["custom_voice", "voice_clone"], "default": "custom_voice"},
        "voice_design_backend": {
            "type": "string",
            "enum": ["qwen3_tts", "auto"],
            "default": "qwen3_tts",
        },
        "speaker": {"type": "string"},
        "instruct": {"type": "string"},
        "language": {"type": "string", "default": "Auto"},
        "ref_audio_path": {"type": "string"},
        "ref_text": {"type": "string"},
        "x_vector_only_mode": {"type": "boolean", "default": False},
        "timeout_seconds": {"type": "number"},
    },
}


def synthesize(arguments: dict) -> dict:
    client = Qwen3TTSClient(
        _service_url_for_mode(str(arguments.get("mode") or "custom_voice")),
        timeout=float(os.getenv("X_EDITOR_QWEN3_TTS_TIMEOUT", "300")),
    )
    return client.synthesize(**arguments)


def _service_url_for_mode(mode: str) -> str:
    if mode == "voice_clone":
        return _first_healthy_url(
            _candidate_urls(
                "QWEN3_TTS_CLONE_SERVICE_URL",
                "QWEN3_TTS_VOICE_CLONE_SERVICE_URL",
                "QWEN3_TTS_SERVICE_URL",
                defaults=("http://127.0.0.1:8019", "http://127.0.0.1:8017"),
            ),
            mode="voice_clone",
        )
    return _first_healthy_url(
        _candidate_urls(
            "QWEN3_TTS_CUSTOM_SERVICE_URL",
            "QWEN3_TTS_SERVICE_URL",
            defaults=("http://127.0.0.1:8017", "http://127.0.0.1:8019"),
        ),
        mode="custom_voice",
    )


def _candidate_urls(*env_names: str, defaults: tuple[str, ...]) -> list[str]:
    values = [os.getenv(name) for name in env_names]
    values.extend(defaults)
    deduped: list[str] = []
    for value in values:
        if not value:
            continue
        url = value.rstrip("/")
        if url not in deduped:
            deduped.append(url)
    return deduped


def _first_healthy_url(candidates: Iterable[str], *, mode: str) -> str:
    urls = list(candidates)
    if requests is None:
        return urls[0]
    for url in urls:
        try:
            response = requests.get(f"{url}/health", timeout=2.0)
            if response.status_code >= 500 or response.status_code == 404:
                continue
            data = response.json()
        except Exception:
            continue
        if mode == "voice_clone" and data.get("base_loaded") and data.get("tokenizer_loaded"):
            return url
        if mode != "voice_clone" and data.get("custom_voice_loaded"):
            return url
    return urls[0]


if __name__ == "__main__":
    run_stdio_server(
        "qwen3_tts",
        [
            LocalTool(
                name="qwen3_tts.synthesize",
                description="Synthesize speech with Qwen3-TTS custom voice or voice clone service.",
                input_schema=INPUT_SCHEMA,
                handler=synthesize,
            )
        ],
    )
