"""Conditional edge routers for X-Editor graph."""

from __future__ import annotations

from x_editor.core.state import AudioEditState
from x_editor.core.constants import PlanningStatus


def structuring_router(state: AudioEditState) -> str:
    """Route after intent_structuring_node based on planning status."""
    status = state.get("planning_status")
    if status == PlanningStatus.FAIL.value:
        return "fail"
    if status == PlanningStatus.NEED_CLARIFICATION.value:
        max_rounds = 2
        if state.get("clarification_round", 0) >= max_rounds:
            return "fail"
        return "clarify"
    return "finalize"
