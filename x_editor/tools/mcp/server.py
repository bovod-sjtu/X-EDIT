"""Helpers for implementing local MCP-style stdio servers."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from x_editor.tools.external_services import ExternalToolSetupRequired


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class LocalTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


def run_stdio_server(server_name: str, tools: list[LocalTool]) -> None:
    """Run a minimal line-delimited JSON-RPC server on stdin/stdout."""

    tool_map = {tool.name: tool for tool in tools}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}
            if method == "initialize":
                result = {"server": server_name, "protocol": "xeditor-jsonrpc-stdio"}
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in tools
                    ]
                }
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if name not in tool_map:
                    raise ValueError(f"Unknown tool: {name}")
                output = tool_map[name].handler(arguments)
                result = {
                    "content": [{"type": "json", "json": output}],
                    "is_error": False,
                }
            elif method == "shutdown":
                result = {"ok": True}
                _write_response(request_id, result)
                break
            else:
                raise ValueError(f"Unknown method: {method}")
            _write_response(request_id, result)
        except Exception as exc:
            _write_error(request.get("id") if "request" in locals() else None, exc)


def _write_response(request_id: int | None, result: dict[str, Any]) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}, ensure_ascii=False), flush=True)


def _write_error(request_id: int | None, exc: Exception) -> None:
    print(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(exc),
                    **({"data": exc.to_dict()} if isinstance(exc, ExternalToolSetupRequired) else {}),
                    "traceback": traceback.format_exc(limit=5),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
