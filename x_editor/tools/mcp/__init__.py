"""Minimal MCP-style JSON-RPC transport for X-Editor tools.

Keep exports lazy so isolated stdio tool servers can import
``x_editor.tools.mcp.server`` without installing the client-side pydantic
schemas.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MCPClient", "MCPServerConfig", "MCPServerManager", "MCPToolInfo", "MCPToolResult"]


def __getattr__(name: str) -> Any:
    if name == "MCPClient":
        from x_editor.tools.mcp.client import MCPClient

        return MCPClient
    if name == "MCPServerManager":
        from x_editor.tools.mcp.server_manager import MCPServerManager

        return MCPServerManager
    if name in {"MCPServerConfig", "MCPToolInfo", "MCPToolResult"}:
        from x_editor.tools.mcp.schemas import MCPServerConfig, MCPToolInfo, MCPToolResult

        return {
            "MCPServerConfig": MCPServerConfig,
            "MCPToolInfo": MCPToolInfo,
            "MCPToolResult": MCPToolResult,
        }[name]
    raise AttributeError(name)
