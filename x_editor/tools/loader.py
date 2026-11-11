"""Load X-Editor tool capability cards and detailed manuals."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from x_editor.core.schemas import ToolCapability, ToolManual

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by environments only
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


def get_catalog_dir() -> Path:
    """Return the local X-Editor tool catalog directory."""

    return Path(__file__).parent / "catalog"


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load tool metadata") from YAML_IMPORT_ERROR


def _read_yaml(path: Path) -> dict[str, Any]:
    _require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return data


def iter_tool_dirs(catalog_dir: Path | None = None) -> list[Path]:
    """List catalog directories that define an X-Editor capability card."""

    root = catalog_dir or get_catalog_dir()
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "capability.yaml").exists()
    )


def load_capability(tool_id: str, catalog_dir: Path | None = None) -> ToolCapability:
    """Load one lightweight tool capability card."""

    root = catalog_dir or get_catalog_dir()
    path = root / tool_id / "capability.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Capability not found for tool '{tool_id}': {path}")
    return ToolCapability(**_read_yaml(path))


def load_all_capabilities(catalog_dir: Path | None = None) -> dict[str, ToolCapability]:
    """Load all planner-visible capability cards keyed by tool_id."""

    from x_editor.tools.selection import current_tool_selection

    selection = current_tool_selection()
    enabled = selection.catalog_ids() if selection is not None else None
    capabilities: dict[str, ToolCapability] = {}
    for tool_dir in iter_tool_dirs(catalog_dir):
        if enabled is not None and tool_dir.name not in enabled:
            continue
        capability = ToolCapability(**_read_yaml(tool_dir / "capability.yaml"))
        capabilities[capability.tool_id] = capability
    return capabilities


def load_manual(tool_id: str, catalog_dir: Path | None = None) -> ToolManual:
    """Load one detailed tool invocation manual."""

    root = catalog_dir or get_catalog_dir()
    path = root / tool_id / "manual.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Manual not found for tool '{tool_id}': {path}")
    return ToolManual(**_read_yaml(path))


def render_capability_summary(capabilities: dict[str, ToolCapability]) -> str:
    """Render a compact planner-facing capability summary."""

    lines: list[str] = []
    for tool_id, cap in sorted(capabilities.items()):
        lines.append(f"### {tool_id}: {cap.display_name}")
        lines.append(f"Category: {cap.category}")
        if cap.backend:
            lines.append(f"Backend: {cap.backend}")
        lines.append(f"Purpose: {cap.purpose}")
        if cap.when_to_use:
            lines.append("Use when:")
            lines.extend(f"- {item}" for item in cap.when_to_use)
        if cap.do_not_use_when:
            lines.append("Do not use when:")
            lines.extend(f"- {item}" for item in cap.do_not_use_when)
        lines.append("")
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect X-Editor tool metadata")
    parser.add_argument("--list-capabilities", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--manual", help="Tool id whose manual should be printed")
    parser.add_argument("--config", type=Path, help="Runtime configuration containing the tool selection")
    args = parser.parse_args()

    context = nullcontext()
    if args.config:
        from x_editor.api import RuntimeConfig
        from x_editor.tools.selection import tool_selection_context

        context = tool_selection_context(RuntimeConfig.from_file(args.config).tools)
    with context:
        capabilities = load_all_capabilities()
        if args.manual:
            if args.manual not in capabilities:
                parser.error("tool is not enabled in the selected configuration")
            print(json.dumps(load_manual(args.manual).model_dump(), indent=2, ensure_ascii=False))
            return
    if args.summary:
        print(render_capability_summary(capabilities))
        return

    if args.list_capabilities:
        print(json.dumps({k: v.model_dump() for k, v in capabilities.items()}, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
