"""Decision resolvers for audit and callback-driven interactive modes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from x_editor.workflow_control.schemas import DecisionCheckpoint, DecisionResolution


class DecisionResolver(Protocol):
    """Resolve a checkpoint without coupling workflow control to a UI."""

    resolver_id: str

    def resolve(self, checkpoint: DecisionCheckpoint) -> DecisionResolution: ...


class AuditDecisionResolver:
    """Leave checkpoints unresolved while recording that they were observed."""

    resolver_id = "audit"

    def resolve(self, checkpoint: DecisionCheckpoint) -> DecisionResolution:
        return DecisionResolution(
            checkpoint_id=checkpoint.checkpoint_id,
            status="pending",
            resolved_by=self.resolver_id,
            rationale="Audit mode records the checkpoint without changing execution.",
        )


class CallbackDecisionResolver:
    """Resolve checkpoints through an injected synchronous user-interface callback."""

    def __init__(
        self,
        callback: Callable[[DecisionCheckpoint], str | DecisionResolution | None],
        *,
        resolver_id: str = "callback",
    ) -> None:
        self.callback = callback
        self.resolver_id = resolver_id

    def resolve(self, checkpoint: DecisionCheckpoint) -> DecisionResolution:
        selected = self.callback(checkpoint)
        if isinstance(selected, DecisionResolution):
            if selected.checkpoint_id != checkpoint.checkpoint_id:
                raise ValueError(
                    "callback resolution belongs to a different checkpoint"
                )
            self._validate_selected_option(checkpoint, selected.selected_option_id)
            return selected
        if selected is None:
            return DecisionResolution(
                checkpoint_id=checkpoint.checkpoint_id,
                status="pending",
                resolved_by=self.resolver_id,
                rationale="The callback did not resolve this checkpoint.",
            )
        self._validate_selected_option(checkpoint, selected)
        return DecisionResolution(
            checkpoint_id=checkpoint.checkpoint_id,
            status="accepted",
            selected_option_id=selected,
            resolved_by=self.resolver_id,
        )

    @staticmethod
    def _validate_selected_option(
        checkpoint: DecisionCheckpoint, selected_option_id: str | None
    ) -> None:
        if selected_option_id is None:
            return
        valid_options = {option.option_id for option in checkpoint.options}
        if selected_option_id not in valid_options:
            raise ValueError(f"unknown checkpoint option: {selected_option_id}")
