"""MCP wrapper for QuarkAudio-UniSE target speaker extraction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CATALOG_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CATALOG_ROOT / "quarkaudio_unise"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from client import QuarkAudioUniSEClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["mixture_audio_path", "reference_audio_path", "output_path"],
    "properties": {
        "mixture_audio_path": {"type": "string"},
        "reference_audio_path": {"type": "string"},
        "output_path": {"type": "string"},
        "enroll_duration": {"type": "number", "default": 5.0},
    },
}


def extract_target_speaker(arguments: dict) -> dict:
    client = QuarkAudioUniSEClient(os.getenv("QUARKAUDIO_UNISE_SERVICE_URL", "http://127.0.0.1:8025"))
    return client.extract_target_speaker(
        mixture_audio_path=arguments["mixture_audio_path"],
        reference_audio_path=arguments["reference_audio_path"],
        output_path=arguments["output_path"],
        enroll_duration=float(arguments.get("enroll_duration", 5.0)),
    )


if __name__ == "__main__":
    run_stdio_server(
        "tse",
        [
            LocalTool(
                name="tse.extract_target_speaker",
                description="Extract a target speaker from mixed audio using QuarkAudio-UniSE with reference enrollment audio.",
                input_schema=INPUT_SCHEMA,
                handler=extract_target_speaker,
            )
        ],
    )
