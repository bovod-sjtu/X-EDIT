"""MCP wrapper for spatial room reverb."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from x_editor.runtime.spatial_reverb import apply_spatial_reverb
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["input_audio_path", "output_path"],
    "properties": {
        "input_audio_path": {"type": "string"},
        "output_path": {"type": "string"},
        "instruction": {"type": "string"},
        "room_preset": {"type": "string"},
        "room_size_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "t60": {"type": "number"},
        "dry_wet": {"type": "number"},
        "source_distance_m": {"type": "number"},
        "receiver_position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "source_position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "rir_length_sec": {"type": "number"},
        "preserve_duration": {"type": "boolean", "default": False},
        "backend": {"type": "string", "default": "auto"},
        "wet_gain_db": {"type": "number", "default": 0.0},
        "output_dir": {"type": "string"},
    },
}


def apply(arguments: dict) -> dict:
    return apply_spatial_reverb(
        input_audio_path=arguments["input_audio_path"],
        output_path=arguments["output_path"],
        instruction=arguments.get("instruction", ""),
        room_preset=arguments.get("room_preset"),
        room_size_m=arguments.get("room_size_m"),
        t60=arguments.get("t60"),
        dry_wet=arguments.get("dry_wet"),
        source_distance_m=arguments.get("source_distance_m"),
        receiver_position_m=arguments.get("receiver_position_m"),
        source_position_m=arguments.get("source_position_m"),
        rir_length_sec=arguments.get("rir_length_sec"),
        preserve_duration=bool(arguments.get("preserve_duration", False)),
        backend=arguments.get("backend", "auto"),
        wet_gain_db=float(arguments.get("wet_gain_db", 0.0)),
        output_dir=arguments.get("output_dir"),
    )


if __name__ == "__main__":
    run_stdio_server(
        "spatial_reverb",
        [
            LocalTool(
                name="spatial_reverb.apply",
                description="Apply room impulse response reverb using optional gpuRIR or synthetic fallback.",
                input_schema=INPUT_SCHEMA,
                handler=apply,
            )
        ],
    )
