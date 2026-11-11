"""MCP wrapper for ACE-Step 1.5 music vocal generation/editing."""

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

from client import AceStepMusicClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["instruction", "output_path"],
    "properties": {
        "instruction": {"type": "string"},
        "output_path": {"type": "string"},
        "lyrics": {"type": "string"},
        "music_prompt": {"type": "string"},
        "source_audio_path": {"type": "string"},
        "reference_audio_path": {"type": "string"},
        "mode": {"type": "string", "enum": ["text2music", "cover", "repaint"], "default": "text2music"},
        "vocal_style": {"type": "string", "enum": ["singing", "rap", "music_vocal"], "default": "singing"},
        "vocal_language": {"type": "string", "default": "unknown"},
        "duration_seconds": {"type": "number", "default": 10.0},
        "audio_format": {"type": "string", "enum": ["wav", "wav32", "mp3", "flac", "opus", "aac"], "default": "wav"},
        "repainting_start": {"type": "number", "default": 0.0},
        "repainting_end": {"type": "number", "default": -1.0},
        "audio_cover_strength": {"type": "number", "default": 0.8},
        "inference_steps": {"type": "integer", "default": 8},
        "guidance_scale": {"type": "number", "default": 7.0},
        "seed": {"type": "integer", "default": -1},
        "use_random_seed": {"type": "boolean", "default": True},
        "thinking": {"type": "boolean", "default": True},
        "use_format": {"type": "boolean", "default": True},
        "bpm": {"type": "integer"},
        "key_scale": {"type": "string"},
        "time_signature": {"type": "string"},
        "model": {"type": "string"},
        "poll_interval_seconds": {"type": "number", "default": 2.0},
        "timeout_seconds": {"type": "number", "default": 1800.0},
    },
}


def generate_or_edit(arguments: dict) -> dict:
    client = AceStepMusicClient(
        os.getenv("ACE_STEP_API_URL") or os.getenv("ACE_STEP_UPSTREAM_URL") or os.getenv("ACE_STEP_SERVICE_URL"),
        timeout=float(arguments.get("timeout_seconds") or 1800.0),
        poll_interval=float(arguments.get("poll_interval_seconds") or 2.0),
    )
    return client.generate_or_edit(**arguments)


if __name__ == "__main__":
    run_stdio_server(
        "ace_step_music",
        [
            LocalTool(
                name="ace_step_music.generate_or_edit",
                description=(
                    "Generate or edit singing/rap/music-vocal audio with an official "
                    "ACE-Step 1.5 API server. Fails clearly if the service or model is unavailable."
                ),
                input_schema=INPUT_SCHEMA,
                handler=generate_or_edit,
            )
        ],
    )
