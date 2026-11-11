"""Initialize selected MCP clients and list tools; do not invoke model services."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from x_editor.tools.executor import ToolExecutor
from x_editor.tools.selection import ToolSelection, tool_selection_context


async def check(selection: ToolSelection) -> list[dict]:
    results = []
    with tool_selection_context(selection):
        executor = ToolExecutor(enable_service_bootstrap=False)
        try:
            for tool_id in sorted(selection.catalog_ids()):
                directory = executor.catalog_dir / tool_id
                if not (directory / "mcp_server.py").is_file():
                    continue
                if not (directory / "config.yaml").is_file():
                    continue
                try:
                    await executor._ensure_registered(tool_id)
                    client = await executor.server_manager.get_client(tool_id)
                    tools = await client.list_tools()
                    results.append({"tool": tool_id, "passed": bool(tools),
                                    "operations": [tool.name for tool in tools]})
                except Exception as error:
                    results.append({"tool": tool_id, "passed": False, "error": str(error)})
        finally:
            await executor.shutdown()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    args = parser.parse_args()
    selection = ToolSelection.model_validate(yaml.safe_load(args.tools.read_text()))
    results = asyncio.run(check(selection))
    passed = all(row["passed"] for row in results)
    print(json.dumps({"passed": passed, "inference_verified": False,
                      "clients": results}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
