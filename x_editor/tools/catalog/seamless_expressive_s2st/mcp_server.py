"""MCP wrapper for SeamlessExpressive / SeamlessM4T v2 S2ST."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from client import SeamlessExpressiveS2STClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["input_audio_path", "target_language", "output_path"],
    "properties": {
        "input_audio_path": {"type": "string"},
        "target_language": {
            "type": "string",
            "description": "Target language code/name. Prefer Seamless ISO-639-3 codes such as fra, deu, spa, eng, cmn.",
        },
        "output_path": {"type": "string"},
        "source_language": {"type": "string"},
        "mode": {"type": "string", "enum": ["auto", "expressive", "m4t_v2"], "default": "auto"},
        "preserve_vocal_style": {"type": "boolean", "default": True},
        "preserve_speech_rate": {"type": "boolean", "default": True},
        "preserve_pauses": {"type": "boolean", "default": True},
        "allow_fallback_to_m4t": {"type": "boolean", "default": False},
        "duration_factor": {"type": "number", "default": 1.0},
        "timeout_seconds": {"type": "number", "default": 1800.0},
        "extra_args": {"type": "array", "items": {"type": "string"}},
    },
}


def translate(arguments: dict) -> dict:
    client = SeamlessExpressiveS2STClient(
        os.getenv("SEAMLESS_S2ST_SERVICE_URL", "http://127.0.0.1:8039")
    )
    return client.translate(**arguments)


if __name__ == "__main__":
    run_stdio_server(
        "seamless_expressive_s2st",
        [
            LocalTool(
                name="seamless_expressive_s2st.translate",
                description=(
                    "Translate speech to speech with Meta SeamlessExpressive when available, "
                    "or SeamlessM4T v2 when expressive preservation is not required."
                ),
                input_schema=INPUT_SCHEMA,
                handler=translate,
            )
        ],
    )
