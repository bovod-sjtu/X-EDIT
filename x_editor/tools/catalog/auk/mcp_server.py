"""AuK MCP transport, launched with the lightweight project runtime."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from x_editor.tools.catalog.auk.client import edit_audio
from x_editor.tools.loader import load_manual
from x_editor.tools.mcp.server import LocalTool, run_stdio_server


def edit(arguments):
    return edit_audio(arguments, base_url=os.environ.get("AUK_SERVICE_URL", "http://127.0.0.1:8065"),
                      timeout=float(os.environ.get("X_EDITOR_AUK_TIMEOUT", "600")))


if __name__ == "__main__":
    run_stdio_server("auk", [LocalTool(name="auk.edit", description="Instruction-conditioned AuK audio editing.",
                                      input_schema=load_manual("auk").input_schema, handler=edit)])
