"""Lifecycle manager for local MCP-style tool servers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from x_editor.tools.mcp.client import MCPClient, resolve_command
from x_editor.tools.mcp.schemas import MCPServerConfig


class MCPServerManager:
    def __init__(self) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()

    def register_config(self, name: str, config: MCPServerConfig) -> None:
        self._configs[name] = config

    async def get_client(self, name: str) -> MCPClient:
        if name not in self._configs:
            raise KeyError(f"Unknown MCP server: {name}")
        config = self._configs[name]
        if config.lifecycle == "per_call":
            client = self._create_client(config)
            await client.start()
            return client
        async with self._lock:
            if name not in self._clients or not self._clients[name].is_running:
                client = self._create_client(config)
                await client.start()
                self._clients[name] = client
            return self._clients[name]

    async def shutdown_all(self) -> None:
        async with self._lock:
            for client in list(self._clients.values()):
                await client.stop()
            self._clients.clear()

    async def stop_client(self, name: str) -> None:
        async with self._lock:
            client = self._clients.pop(name, None)
        if client is not None:
            await client.stop()

    def _create_client(self, config: MCPServerConfig) -> MCPClient:
        working_dir = str(Path(config.working_dir).resolve()) if config.working_dir else None
        command = resolve_command(config.command, working_dir)
        return MCPClient(
            command=command,
            working_dir=working_dir,
            env=config.env,
            startup_timeout_sec=config.startup_timeout_sec,
        )
