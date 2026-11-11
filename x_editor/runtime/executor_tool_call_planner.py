"""Provider-neutral executor tool-call planning contracts."""

from __future__ import annotations

from typing import Any, Protocol

from x_editor.core.schemas import AudioEditStep, ExecutorToolCall, ToolManual


class ExecutorToolCallPlannerError(RuntimeError):
    """Raised when an executor repair provider cannot return a valid call."""


class ExecutorToolCallPlanner(Protocol):
    """Provider interface for one-step executor parameter repair."""

    provider_name: str

    def plan_tool_call(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        operation_spec: dict[str, Any],
        manual: ToolManual | None = None,
        verifier_error: dict[str, Any] | str | None = None,
    ) -> ExecutorToolCall:
        """Return a strict normalized call for a single failed step."""
