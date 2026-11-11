"""MCP wrapper for the local paralinguistic event asset bank."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from client import ParalinguisticEventBankClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["event", "output_path"],
    "properties": {
        "event": {"type": "string", "enum": ["breath", "laugh", "cough", "sigh"]},
        "output_path": {"type": "string"},
        "source_audio_path": {"type": "string"},
        "event_asset_path": {"type": "string"},
        "event_bank_dir": {"type": "string"},
    },
}


def materialize(arguments: dict) -> dict:
    return ParalinguisticEventBankClient().materialize(**arguments)


if __name__ == "__main__":
    run_stdio_server(
        "paralinguistic_event_bank",
        [
            LocalTool(
                name="paralinguistic_event_bank.materialize",
                description="Materialize a breath/laugh/cough/sigh event asset WAV for local insertion workflows.",
                input_schema=INPUT_SCHEMA,
                handler=materialize,
            )
        ],
    )
