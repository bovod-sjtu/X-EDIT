"""Routers for the tool-aware graph."""

from __future__ import annotations

from x_editor.core.state import AudioEditState


def capability_router(state: AudioEditState) -> str:
    status = state.get("planning_status")
    if status in {"UNSUPPORTED", "FAIL"}:
        return "fail"
    return "plan"
