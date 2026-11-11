"""Schemas for X-Editor's minimal MCP-style transport."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    command: list[str]
    working_dir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    lifecycle: Literal["session", "per_call"] = "session"
    startup_timeout_sec: float = 60.0
    resources: dict[str, Any] = Field(default_factory=dict)


class MCPToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
    error: str | None = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
