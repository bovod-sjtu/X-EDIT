"""MCP wrapper for ClearerVoice speech enhancement."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CATALOG_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CATALOG_ROOT / "clearer_voice_studio"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from client import ClearerVoiceClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["input_audio_path", "output_path"],
    "properties": {
        "input_audio_path": {"type": "string"},
        "output_path": {"type": "string"},
        "mode": {"type": "string", "default": "general"},
        "preserve_background": {"type": "boolean", "default": False},
        "enhancement_goal": {"type": "string"},
    },
}


def enhance(arguments: dict) -> dict:
    client = ClearerVoiceClient(os.getenv("CLEARERVOICE_SERVICE_URL", "http://127.0.0.1:8027"))
    return client.enhance(
        input_audio_path=arguments["input_audio_path"],
        output_path=arguments["output_path"],
        mode=arguments.get("mode", "general"),
        preserve_background=bool(arguments.get("preserve_background", False)),
        enhancement_goal=arguments.get("enhancement_goal"),
    )


if __name__ == "__main__":
    run_stdio_server(
        "speech_enhancement",
        [
            LocalTool(
                name="speech_enhancement.enhance",
                description="Enhance speech clarity through the ClearerVoice-Studio backend.",
                input_schema=INPUT_SCHEMA,
                handler=enhance,
            )
        ],
    )
