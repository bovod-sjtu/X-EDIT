"""One-way observation and fault-isolated recording for execution facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from x_editor.workflow_control.event_log import WorkflowEventLog
from x_editor.workflow_control.schemas import (
    WorkflowEvent,
    WorkflowEventStatus,
    WorkflowStage,
)


class ExecutionEventObserver(Protocol):
    """Receive immutable execution facts without returning control decisions."""

    def emit(
        self,
        *,
        stage: WorkflowStage,
        kind: str,
        status: WorkflowEventStatus = "observed",
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Observe one fact; the return value is intentionally ignored."""


class ExecutionEventRecorder:
    """Keep an ordered event view while isolating persistence and observer faults."""

    def __init__(
        self,
        *,
        run_id: str,
        event_log_path: str | None = None,
        observer: ExecutionEventObserver | None = None,
    ) -> None:
        self.run_id = run_id
        self.events: list[WorkflowEvent] = []
        self.observation_errors: list[dict[str, Any]] = []
        self._observer = observer if observer is not self else None
        self._log: WorkflowEventLog | None = None
        if not event_log_path:
            return
        try:
            candidate = WorkflowEventLog(Path(event_log_path), run_id=run_id)
            if candidate.event_count:
                self.record_observation_error(
                    component="workflow_event_log",
                    operation="initialize",
                    message=(
                        f"run_id {run_id!r} already has persisted events; "
                        "continuing with in-memory observation only"
                    ),
                    path=str(candidate.path),
                )
                return
            self._log = candidate
        except Exception as exc:
            self.record_observation_error(
                component="workflow_event_log",
                operation="initialize",
                error=exc,
                path=str(Path(event_log_path).expanduser()),
            )

    @property
    def event_log_path(self) -> str | None:
        return str(self._log.path) if self._log is not None else None

    def record_observation_error(
        self,
        *,
        component: str,
        operation: str,
        error: Exception | None = None,
        message: str | None = None,
        **details: Any,
    ) -> None:
        record: dict[str, Any] = {
            "component": component,
            "operation": operation,
            **details,
        }
        if error is not None:
            record["error_type"] = type(error).__name__
            record["error_message"] = str(error)
        elif message is not None:
            record["error_type"] = "ObservationError"
            record["error_message"] = message
        self.observation_errors.append(record)

    def emit(
        self,
        *,
        stage: WorkflowStage,
        kind: str,
        status: WorkflowEventStatus = "observed",
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        event: WorkflowEvent | None = None
        if self._log is not None:
            try:
                event = self._log.append(
                    stage=stage,
                    kind=kind,
                    status=status,
                    attempt_id=attempt_id,
                    subject_refs=subject_refs,
                    payload=payload,
                )
            except Exception as exc:
                failed_path = str(self._log.path)
                self._log = None
                self.record_observation_error(
                    component="workflow_event_log",
                    operation="append",
                    error=exc,
                    path=failed_path,
                    event_kind=kind,
                )
        if event is None:
            event = WorkflowEvent(
                run_id=self.run_id,
                sequence=len(self.events) + 1,
                stage=stage,
                kind=kind,
                status=status,
                attempt_id=attempt_id,
                subject_refs=list(subject_refs or []),
                payload=dict(payload or {}),
            )
        self.events.append(event)
        self._forward_to_observer(event)
        return event

    def _forward_to_observer(self, event: WorkflowEvent) -> None:
        if self._observer is None:
            return
        forwarded_payload = dict(event.payload)
        upstream_source = forwarded_payload.get("source_event_ref")
        if isinstance(upstream_source, dict):
            forwarded_payload["upstream_source_event_ref"] = dict(
                upstream_source
            )
        forwarded_payload["source_event_ref"] = {
            "event_id": event.event_id,
            "run_id": event.run_id,
            "sequence": event.sequence,
        }
        try:
            self._observer.emit(
                stage=event.stage,
                kind=event.kind,
                status=event.status,
                attempt_id=event.attempt_id,
                subject_refs=list(event.subject_refs),
                payload=forwarded_payload,
            )
        except Exception as exc:
            self.record_observation_error(
                component="execution_event_observer",
                operation="emit",
                error=exc,
                event_kind=event.kind,
            )


__all__ = ["ExecutionEventObserver", "ExecutionEventRecorder"]
