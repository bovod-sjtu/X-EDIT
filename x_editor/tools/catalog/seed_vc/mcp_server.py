"""MCP wrapper for SeedVC reference voice conversion."""

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

from client import SeedVCClient
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


INPUT_SCHEMA = {
    "type": "object",
    "required": ["source_audio_path", "reference_audio_path", "output_path"],
    "properties": {
        "source_audio_path": {"type": "string"},
        "reference_audio_path": {"type": "string"},
        "output_path": {"type": "string"},
        "diffusion_steps": {"type": "integer", "default": 30},
        "length_adjust": {"type": "number", "default": 1.0},
        "inference_cfg_rate": {"type": "number", "default": 0.7},
        "seed": {"type": "integer", "default": 0},
    },
}


def convert(arguments: dict) -> dict:
    client = SeedVCClient(os.getenv("SEED_VC_SERVICE_URL", "http://127.0.0.1:8003"))
    return client.convert(
        source_audio_path=arguments["source_audio_path"],
        reference_audio_path=arguments["reference_audio_path"],
        output_path=arguments["output_path"],
        diffusion_steps=int(arguments.get("diffusion_steps", 30)),
        length_adjust=float(arguments.get("length_adjust", 1.0)),
        inference_cfg_rate=float(arguments.get("inference_cfg_rate", 0.7)),
        seed=int(arguments.get("seed", 0)),
    )


if __name__ == "__main__":
    run_stdio_server(
        "seed_vc",
        [
            LocalTool(
                name="seed_vc.convert",
                description="Convert source speech to match a reference speaker using a running SeedVC service.",
                input_schema=INPUT_SCHEMA,
                handler=convert,
            )
        ],
    )
