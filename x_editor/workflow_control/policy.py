"""Explicit policy parsing for workflow-control enforcement."""

from __future__ import annotations

from typing import Any

from x_editor.workflow_control.schemas import (
    ArtifactStoreMode,
    FinalAssemblerMode,
    RepairAction,
)

SUPPORTED_ENFORCED_REPAIR_ACTIONS: frozenset[RepairAction] = frozenset(
    {"repair_step_arguments"}
)


def artifact_store_mode(context: dict[str, Any]) -> ArtifactStoreMode:
    """Return the explicit artifact authority mode."""

    raw = str(context.get("workflow_artifact_store_mode") or "disabled").strip().lower()
    if raw not in {"disabled", "enforced"}:
        raise ValueError(
            "workflow_artifact_store_mode must be one of ['disabled', 'enforced']"
        )
    return raw  # type: ignore[return-value]


def final_assembler_mode(context: dict[str, Any]) -> FinalAssemblerMode:
    """Return the explicit non-authoritative final assembly observation mode."""

    raw = str(context.get("workflow_final_assembler_mode") or "disabled").strip().lower()
    if raw not in {"disabled", "shadow"}:
        raise ValueError(
            "workflow_final_assembler_mode must be one of ['disabled', 'shadow']"
        )
    return raw  # type: ignore[return-value]


def enforced_repair_actions(context: dict[str, Any]) -> frozenset[RepairAction]:
    """Return the explicit action allowlist for enforced mode."""

    raw = context.get("workflow_enforced_repair_actions")
    if raw in (None, ""):
        return frozenset()
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = [str(item).strip() for item in raw]
    else:
        values = [str(raw).strip()]
    return frozenset(value for value in values if value)  # type: ignore[arg-type]


def repair_action_is_enforced(
    context: dict[str, Any], action: RepairAction
) -> bool:
    """Return true only for an explicitly allowlisted action in enforced mode."""

    mode = str(context.get("workflow_control_mode") or "disabled").strip().lower()
    return mode == "enforced" and action in enforced_repair_actions(context)


def argument_repair_max_attempts(context: dict[str, Any]) -> int:
    """Return the bounded per-step argument-repair attempt count."""

    return max(0, int(context.get("workflow_argument_repair_max_attempts", 1)))
