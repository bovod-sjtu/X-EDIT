"""MCP wrapper for FlexSED open-vocabulary sound event detection."""

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

from client import FlexSEDClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["events"],
    "properties": {
        "audio_path": {"type": "string"},
        "input_audio_path": {"type": "string"},
        "events": {"type": "array", "items": {"type": "string"}},
        "event_list": {"type": "string"},
        "threshold": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
        "min_duration_sec": {"type": "number", "default": 0.08, "minimum": 0.0},
        "merge_gap_sec": {"type": "number", "default": 0.08, "minimum": 0.0},
        "norm_audio": {"type": "boolean", "default": True},
        "output_dir": {"type": "string"},
        "save_heatmap_png": {"type": "boolean", "default": True},
        "save_heatmap_npy": {"type": "boolean", "default": True},
        "include_scores": {"type": "boolean", "default": False},
        "max_scores_json_frames": {"type": "integer", "default": 2000},
        "fname": {"type": "string"},
        "start_sec": {"type": "number", "minimum": 0.0},
        "end_sec": {"type": "number", "minimum": 0.0},
        "p": {"type": "number", "default": 3.0, "exclusiveMinimum": 0.0},
        "score_mode": {"type": "string", "enum": ["max", "mean", "p95"], "default": "max"},
        "min_score": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
        "timeout_seconds": {"type": "number"},
    },
}


def detect_events(arguments: dict) -> dict:
    client = FlexSEDClient(
        os.getenv("FLEXSED_SERVICE_URL", "http://127.0.0.1:8057"),
        timeout=float(os.getenv("X_EDITOR_FLEXSED_TIMEOUT", "900")),
    )
    return client.detect(**arguments)


if __name__ == "__main__":
    run_stdio_server(
        "flexsed",
        [
            LocalTool(
                name="flexsed.detect_events",
                description=(
                    "Run FlexSED open-vocabulary sound event detection and return thresholded "
                    "event timestamps plus heatmap artifacts."
                ),
                input_schema=INPUT_SCHEMA,
                handler=detect_events,
            )
        ],
    )
