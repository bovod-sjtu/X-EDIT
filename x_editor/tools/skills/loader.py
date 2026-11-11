"""Legacy skill loader.

This module is kept only for compatibility with early X-Editor prototypes.
New tools must use ``x_editor/tools/catalog/<tool_id>/capability.yaml`` and
``manual.yaml`` loaded by ``x_editor.tools.loader``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_skills_dir() -> Path:
    """Get the directory containing skill YAML files."""
    return Path(__file__).parent


def load_skill(skill_id: str) -> dict[str, Any] | None:
    """Load a single skill YAML by its skill_id."""
    if not HAS_YAML:
        return None
    path = get_skills_dir() / f"{skill_id}.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_skills() -> dict[str, dict[str, Any]]:
    """Load all available skill YAMLs, keyed by skill_id."""
    skills: dict[str, dict[str, Any]] = {}
    if not HAS_YAML:
        return skills
    for path in get_skills_dir().glob("*.yaml"):
        if path.name.startswith("_"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "skill_id" in data:
            skills[data["skill_id"]] = data
    return skills


def filter_skills_by_dimensions(
    skills: dict[str, dict[str, Any]],
    dimensions: list[str],
) -> dict[str, dict[str, Any]]:
    """Filter skills that affect at least one of the given dimensions."""
    return {
        k: v for k, v in skills.items()
        if any(d in v.get("dimensions", []) for d in dimensions)
    }


def build_formal_tool_script(
    skills: dict[str, dict[str, Any]],
    active_tool_names: set[str] | None = None,
) -> str:
    """
    Build a formal tool script markdown for injection into the planner prompt.

    Args:
        skills: All loaded skills, keyed by skill_id
        active_tool_names: Optional set of actually available tool names from registry.
                          If provided, only include skills whose tool_names overlap.
    """
    lines = [
        "## Formal Tool Registry Script",
        "",
        "The following tools are available for this editing workflow.",
        "Each tool includes: capability, parameters, execution constraints,",
        "and instruction mappings (when to use based on structured instruction).",
        "",
    ]

    for skill_id, skill in sorted(skills.items()):
        # Filter by active tools if provided
        if active_tool_names is not None:
            skill_tools = set(skill.get("tool_names", []))
            if not skill_tools.intersection(active_tool_names):
                continue

        lines.append(f"### [{skill_id}] {skill.get('display_name', skill_id)}")
        lines.append(f"**Source:** {skill.get('source_mcp', 'unknown')}")
        lines.append(f"**Dimensions:** {', '.join(skill.get('dimensions', []))}")
        lines.append("")
        lines.append(f"**Description:** {skill.get('description', '')}")
        lines.append("")

        # Capability
        cap = skill.get("detailed_capability", "")
        if cap:
            lines.append("**Capability:**")
            lines.append(cap)
            lines.append("")

        # Parameters
        params = skill.get("parameters", [])
        if params:
            lines.append("**Parameters:**")
            for p in params:
                req = "required" if p.get("required") else f"optional, default={p.get('default', 'none')}"
                constraints = f" ({p.get('constraints', '')})" if p.get("constraints") else ""
                lines.append(
                    f"- `{p['name']}` ({p.get('type', 'any')}, {req}){constraints}: "
                    f"{p.get('description', '')}"
                )
            lines.append("")

        # Output
        out = skill.get("output", {})
        if out:
            lines.append(f"**Output:** {out.get('type', 'unknown')} — {out.get('description', '')}")
            lines.append("")

        # Constraints
        constraints = skill.get("execution_constraints", {})
        if constraints:
            lines.append("**Execution Constraints:**")
            if constraints.get("should_run_after"):
                lines.append(f"- Must run AFTER: {', '.join(constraints['should_run_after'])}")
            if constraints.get("should_run_before"):
                lines.append(f"- Must run BEFORE: {', '.join(constraints['should_run_before'])}")
            if constraints.get("mutually_exclusive_with"):
                lines.append(f"- Mutually exclusive with: {', '.join(constraints['mutually_exclusive_with'])}")
            if constraints.get("can_repeat") is False:
                lines.append("- Can only be used ONCE per workflow")
            lines.append("")

        # Instruction Mapping
        mapping = skill.get("instruction_mapping", {})
        if mapping:
            lines.append("**When to Use (Instruction Mapping):**")
            for key, value in mapping.items():
                lines.append(f"- `{key}` → {value}")
            lines.append("")

        # Examples
        examples = skill.get("examples", [])
        if examples:
            lines.append("**Examples:**")
            for ex in examples:
                lines.append(f"- Scenario: {ex.get('scenario', '')}")
                if ex.get("parameters"):
                    lines.append(f"  Parameters: {ex['parameters']}")
                if ex.get("expected_outcome"):
                    lines.append(f"  Expected: {ex['expected_outcome']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
