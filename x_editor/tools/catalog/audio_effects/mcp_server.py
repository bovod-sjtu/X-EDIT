"""MCP wrapper for deterministic audio effects."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from x_editor.runtime.audio_effects import apply_audio_effect
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["input_audio_path", "output_path"],
    "properties": {
        "input_audio_path": {"type": "string"},
        "output_path": {"type": "string"},
        "instruction": {"type": "string"},
        "preset": {
            "type": "string",
            "enum": [
                "telephone",
                "old_radio",
                "underwater",
                "strong_wind",
                "crowd_noise",
                "background_music",
                "muffled_wall",
                "whisper_style",
                "shouting_style",
                "animal_vocal_style",
                "match_channel_reference",
            ],
        },
        "effect": {"type": "string"},
        "strength": {"type": "number", "default": 0.75},
        "reference_audio_path": {"type": "string"},
        "preserve_duration": {"type": "boolean", "default": True},
        "output_dir": {"type": "string"},
    },
}


def apply(arguments: dict) -> dict:
    return apply_audio_effect(
        input_audio_path=arguments["input_audio_path"],
        output_path=arguments["output_path"],
        instruction=arguments.get("instruction", ""),
        preset=arguments.get("preset"),
        effect=arguments.get("effect"),
        strength=float(arguments.get("strength", 0.75)),
        reference_audio_path=arguments.get("reference_audio_path"),
        preserve_duration=bool(arguments.get("preserve_duration", True)),
        output_dir=arguments.get("output_dir"),
    )


if __name__ == "__main__":
    run_stdio_server(
        "audio_effects",
        [
            LocalTool(
                name="audio_effects.apply",
                description="Apply deterministic speech-preserving acoustic/channel effects.",
                input_schema=INPUT_SCHEMA,
                handler=apply,
            )
        ],
    )
