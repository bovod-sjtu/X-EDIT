"""MCP wrapper for vocal/accompaniment separation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CATALOG_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, CATALOG_ROOT / "audio_separator_roformer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from client import AudioSeparatorRoFormerClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["input_audio_path", "output_dir"],
    "properties": {
        "input_audio_path": {"type": "string"},
        "output_dir": {"type": "string"},
        "output_path": {"type": "string"},
        "model_family": {"type": "string", "default": "bs_roformer"},
        "model_filename": {"type": "string"},
        "output_format": {"type": "string", "default": "WAV"},
        "stem_policy": {"type": "string", "default": "vocals"},
    },
}


def separate(arguments: dict) -> dict:
    client = AudioSeparatorRoFormerClient(
        os.getenv("AUDIO_SEPARATOR_ROFORMER_SERVICE_URL", "http://127.0.0.1:8042")
    )
    return client.separate(
        input_audio_path=arguments["input_audio_path"],
        output_dir=arguments["output_dir"],
        output_path=arguments.get("output_path"),
        model_family=arguments.get("model_family", "bs_roformer"),
        model_filename=arguments.get("model_filename"),
        output_format=arguments.get("output_format", "WAV"),
        stem_policy=arguments.get("stem_policy", "vocals"),
    )


if __name__ == "__main__":
    run_stdio_server(
        "speech_separation",
        [
            LocalTool(
                name="speech_separation.separate",
                description="Separate vocals from accompaniment; the configured RoFormer models do not isolate individual overlapping speakers. Select a labeled vocals or instrumental stem.",
                input_schema=INPUT_SCHEMA,
                handler=separate,
            )
        ],
    )
