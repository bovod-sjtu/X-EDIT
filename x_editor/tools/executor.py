"""Tool registry and executor for X-Editor MCP tools."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from x_editor.tools.loader import get_catalog_dir, load_manual
from x_editor.tools.mcp.schemas import MCPServerConfig, ToolResult
from x_editor.tools.mcp.server_manager import MCPServerManager
from x_editor.tools.service_bootstrap import ServiceBootstrapAgent
from x_editor.tools.selection import current_tool_selection, tool_selection_context
from x_editor.tools.edit_backend import selected_edit_backend
from x_editor.tools.environments import configure_client_environment, PYTHON_ENVIRONMENT_KEYS

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


class ToolExecutor:
    """Execute selected tools through their local MCP wrapper."""

    def __init__(
        self,
        catalog_dir: Path | None = None,
        *,
        service_bootstrap_agent: ServiceBootstrapAgent | None = None,
        enable_service_bootstrap: bool | None = None,
        ) -> None:
        self.catalog_dir = catalog_dir or get_catalog_dir()
        self.selection = current_tool_selection()
        self.server_manager = MCPServerManager()
        self.service_bootstrap_agent = service_bootstrap_agent or ServiceBootstrapAgent(
            enabled=enable_service_bootstrap,
        )
        self._registered: set[str] = set()
        self._tool_configs: dict[str, dict[str, Any]] = {}
        self.bootstrap_reports: list[dict[str, Any]] = []

    async def execute(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        if self.selection is not None and not self.selection.allows_call(tool_id, arguments):
            return ToolResult(tool_name=tool_id, success=False,
                              output={"code": "tool_not_enabled"},
                              error_message=f"Tool operation is not enabled by tools configuration: {tool_id}")
        with tool_selection_context(self.selection):
            return await self._execute_enabled(tool_id, arguments)

    async def _execute_enabled(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        manual = load_manual(tool_id, self.catalog_dir)
        tool_config = await self._ensure_registered(tool_id)
        bootstrap_report = await self.service_bootstrap_agent.ensure_tool_services(
            tool_id=tool_id,
            tool_dir=self.catalog_dir / tool_id,
            tool_config=tool_config,
            arguments=arguments,
        )
        if bootstrap_report.services or bootstrap_report.skipped:
            self.bootstrap_reports.append(bootstrap_report.to_dict())
        if not bootstrap_report.ok:
            return ToolResult(
                tool_name=manual.mcp_tool_name,
                success=False,
                output={"service_bootstrap": bootstrap_report.to_dict()},
                error_message=_bootstrap_error_message(bootstrap_report.to_dict()),
            )
        client = await self.server_manager.get_client(tool_id)
        try:
            timeout = _tool_call_timeout(tool_id)
            if timeout is not None:
                mcp_result = await asyncio.wait_for(
                    client.call_tool(manual.mcp_tool_name, arguments),
                    timeout=timeout,
                )
            else:
                mcp_result = await client.call_tool(manual.mcp_tool_name, arguments)
        except asyncio.TimeoutError:
            await self.server_manager.stop_client(tool_id)
            return ToolResult(
                tool_name=manual.mcp_tool_name,
                success=False,
                output={"service_bootstrap": bootstrap_report.to_dict()} if bootstrap_report.services else {},
                error_message=f"MCP tool invocation timed out after {timeout:.1f}s",
            )
        except asyncio.CancelledError:
            await self.server_manager.stop_client(tool_id)
            raise
        except Exception as exc:
            return ToolResult(
                tool_name=manual.mcp_tool_name,
                success=False,
                output={"service_bootstrap": bootstrap_report.to_dict()} if bootstrap_report.services else {},
                error_message=f"MCP tool invocation failed for {tool_id}: {type(exc).__name__}: {exc}",
            )
        if mcp_result.is_error:
            return ToolResult(
                tool_name=manual.mcp_tool_name,
                success=False,
                output={"service_bootstrap": bootstrap_report.to_dict()} if bootstrap_report.services else {},
                error_message=mcp_result.error or "MCP tool returned error",
            )
        output: dict[str, Any] = {"content": mcp_result.content}
        if mcp_result.content:
            first = mcp_result.content[0]
            if isinstance(first, dict) and "json" in first and isinstance(first["json"], dict):
                output.update(first["json"])
        if bootstrap_report.services:
            output["service_bootstrap"] = bootstrap_report.to_dict()
        return ToolResult(tool_name=manual.mcp_tool_name, success=True, output=output)

    async def shutdown(self) -> None:
        await self.server_manager.shutdown_all()

    async def _ensure_registered(self, tool_id: str) -> dict[str, Any]:
        if tool_id in self._registered:
            return self._tool_configs.get(tool_id, {})
        if yaml is None:
            raise RuntimeError("PyYAML is required to load tool config") from YAML_IMPORT_ERROR
        tool_dir = self.catalog_dir / tool_id
        config_path = tool_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Tool config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        server_data = data.get("server")
        if not isinstance(server_data, dict):
            raise ValueError(f"Tool config missing server object: {config_path}")
        if isinstance(server_data.get("env"), dict):
            server_data["env"] = {
                key: os_value if key not in PYTHON_ENVIRONMENT_KEYS and (os_value := os.environ.get(key)) is not None else value
                for key, value in server_data["env"].items()
            }
        if tool_id == "edit_data_pipeline":
            server_data.setdefault("env", {})["X_EDITOR_EDIT_BACKEND"] = selected_edit_backend()
        configure_client_environment(server_data, tool_dir)
        if server_data.get("working_dir"):
            working = Path(server_data["working_dir"])
            if not working.is_absolute():
                server_data["working_dir"] = str((tool_dir / working).resolve())
        config = MCPServerConfig(**server_data)
        self.server_manager.register_config(tool_id, config)
        self._registered.add(tool_id)
        self._tool_configs[tool_id] = data
        return data


def _bootstrap_error_message(report: dict[str, Any]) -> str:
    services = report.get("services") or []
    parts = []
    for service in services:
        if not service.get("ok"):
            message = service.get("message") or service.get("status") or "service bootstrap failed"
            parts.append(f"{service.get('name')}: {message}")
    return "Service bootstrap failed before MCP tool invocation: " + "; ".join(parts)


def _tool_call_timeout(tool_id: str) -> float | None:
    env_name = "X_EDITOR_MCP_CALL_TIMEOUT_" + "".join(ch if ch.isalnum() else "_" for ch in tool_id.upper())
    raw = os.getenv(env_name) or os.getenv("X_EDITOR_MCP_CALL_TIMEOUT")
    if not raw:
        return None
    value = float(raw)
    return value if value > 0 else None
