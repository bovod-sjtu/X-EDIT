"""Subprocess JSON-RPC client for local MCP-style tools."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from x_editor.tools.mcp.schemas import MCPToolInfo, MCPToolResult
from x_editor.tools.environments import clean_python_environment


class MCPClient:
    """Line-delimited JSON-RPC client over stdio."""

    def __init__(
        self,
        command: list[str],
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        startup_timeout_sec: float = 60.0,
    ) -> None:
        self.command = command
        self.working_dir = working_dir
        self.env = env or {}
        self.startup_timeout_sec = startup_timeout_sec
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self.is_running:
            return
        env = clean_python_environment()
        env.update(self.env)
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_dir,
            env=env,
        )
        try:
            await asyncio.wait_for(self.initialize(), timeout=self.startup_timeout_sec)
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        if not self._process:
            return
        try:
            if self.is_running:
                await asyncio.wait_for(self._request("shutdown", {}), timeout=1.0)
        except Exception:
            pass
        if self.is_running:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    async def initialize(self) -> dict[str, Any]:
        return await self._request("initialize", {})

    async def list_tools(self) -> list[MCPToolInfo]:
        result = await self._request("tools/list", {})
        return [MCPToolInfo(**item) for item in result.get("tools", [])]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        return MCPToolResult(**result)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP process is not running")
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._process.stdin.drain()
        line = await self._process.stdout.readline()
        if not line:
            stderr = b""
            if self._process.stderr:
                try:
                    stderr = await asyncio.wait_for(self._process.stderr.read(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass
            raise RuntimeError(f"MCP server closed stdout. stderr={stderr.decode('utf-8', 'replace')[:1000]}")
        response = json.loads(line.decode("utf-8"))
        if response.get("id") != self._request_id:
            raise RuntimeError(f"MCP response id mismatch: {response}")
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "MCP request failed"))
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise RuntimeError(f"MCP result must be object, got {type(result).__name__}")
        return result


def resolve_command(command: list[str], working_dir: str | None) -> list[str]:
    """Resolve relative executable path against working_dir."""

    if not command:
        return command
    if command[0] == "uv":
        uv_bin = os.getenv("X_EDITOR_UV_BIN") or shutil.which("uv")
        return [uv_bin or command[0], *command[1:]]
    if not working_dir:
        return command
    exe = Path(command[0])
    if exe.is_absolute():
        return command
    candidate = Path(working_dir) / exe
    if candidate.exists():
        return [str(candidate), *command[1:]]
    if _is_relative_venv_python(exe):
        override = _tool_python_override(working_dir)
        if override is not None:
            return [str(override), *command[1:]]
        shared = _shared_checkout_executable(working_dir, exe)
        if shared is not None:
            return [str(shared), *command[1:]]
        if os.getenv("X_EDITOR_ALLOW_SYSTEM_PYTHON_TOOL_FALLBACK", "").lower() in {"1", "true", "yes"}:
            return [sys.executable, *command[1:]]
        return [str(candidate), *command[1:]]
    return command


def _is_relative_venv_python(exe: Path) -> bool:
    normalized = exe.as_posix()
    return normalized in {".venv/bin/python", ".venv/bin/python3"} or normalized.endswith("/.venv/bin/python")


def _tool_python_override(working_dir: str | None) -> Path | None:
    env_names = ["X_EDITOR_TOOL_PYTHON"]
    if working_dir:
        tool_key = _env_key(Path(working_dir).name)
        if tool_key:
            env_names.insert(0, f"X_EDITOR_{tool_key}_PYTHON")
    for env_name in env_names:
        value = os.getenv(env_name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.exists():
            return candidate
    return None


def _shared_checkout_executable(working_dir: str | None, exe: Path) -> Path | None:
    if not working_dir:
        return None
    suffix = _checkout_relative_working_dir(Path(working_dir))
    if suffix is None:
        return None
    for root in _shared_checkout_roots():
        candidate = root / suffix / exe
        if candidate.exists():
            return candidate
    return None


def _checkout_relative_working_dir(working_dir: Path) -> Path | None:
    resolved = working_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return resolved.relative_to(repo_root)
    except ValueError:
        pass

    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == "x_editor" and index < len(parts) - 1:
            return Path(*parts[index:])
    return None


def _shared_checkout_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []
    for env_name in (
        "X_EDITOR_SHARED_X_EDITOR_ROOT",
        "X_EDITOR_MAIN_CHECKOUT",
        "X_EDITOR_MAIN_X_EDITOR_ROOT",
    ):
        value = os.getenv(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    roots_value = os.getenv("X_EDITOR_SHARED_X_EDITOR_ROOTS")
    if roots_value:
        candidates.extend(Path(item).expanduser() for item in roots_value.split(os.pathsep) if item.strip())

    # Only explicitly configured shared roots are portable deployment inputs.

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or candidate == repo_root:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _workspace_username(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "user" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _env_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.upper()).strip("_")
