"""Explicit Python environment boundaries for local tool clients."""
from __future__ import annotations

import os
from pathlib import Path


PYTHON_ENVIRONMENT_KEYS = (
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT",
)


def clean_python_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items()
            if key not in PYTHON_ENVIRONMENT_KEYS}


def configure_client_environment(server: dict, tool_dir: Path) -> None:
    command = server.get("command", [])
    if "mcp_server.py" not in command:
        return
    configured_root = os.environ.get("X_EDITOR_MCP_ENV_ROOT")
    environment = ((Path(configured_root).expanduser() / tool_dir.name)
                   if configured_root else tool_dir / ".venv").resolve()
    server.setdefault("env", {})["UV_PROJECT_ENVIRONMENT"] = str(environment)
    if not configured_root:
        return
    python = environment / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(
            f"MCP environment missing for {tool_dir.name}: {python}. "
            "Run deployment/client_envs.py --tools <your tools.yaml> "
            "--env-root <X_EDITOR_MCP_ENV_ROOT> first."
        )
    if command[0] == "uv":
        server["command"] = [*command[:2], "--no-sync", *command[2:]]
    elif command[0] == ".venv/bin/python":
        server["command"] = [str(python), *command[1:]]
    server["env"]["VIRTUAL_ENV"] = str(environment)
