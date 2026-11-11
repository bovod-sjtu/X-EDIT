"""Install selected lightweight MCP clients into separate uv environments."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x_editor.tools.environments import clean_python_environment


def sync_project(project: Path, environment: Path) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv==0.11.28 first")
    env = clean_python_environment()
    for key in ("UV_NO_SYNC", "UV_FROZEN", "UV_NO_DEPS"):
        env.pop(key, None)
    env["UV_PROJECT_ENVIRONMENT"] = str(environment.resolve())
    subprocess.run(
        [uv, "sync", "--locked", "--no-dev", "--python", "3.11.15",
         "--project", str(project)], env=env, cwd=ROOT, check=True,
    )


def install_clients(env_root: Path, catalog_ids: set[str]) -> None:
    catalog = ROOT / "x_editor/tools/catalog"
    for tool_id in sorted(catalog_ids):
        project = catalog / tool_id
        if not (project / "mcp_server.py").is_file():
            continue
        if not (project / "config.yaml").is_file():
            continue
        dependency_project = project if (project / "uv.lock").is_file() else ROOT
        print(f"Installing MCP client {tool_id}: {env_root / tool_id}", flush=True)
        sync_project(dependency_project, env_root / tool_id)


def main() -> None:
    import yaml
    from x_editor.tools.selection import ToolSelection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    args = parser.parse_args()
    selection = ToolSelection.model_validate(yaml.safe_load(args.tools.read_text()))
    install_clients(args.env_root.expanduser().resolve(), selection.catalog_ids())


if __name__ == "__main__":
    main()
