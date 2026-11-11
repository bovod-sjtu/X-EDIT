"""Append-only JSONL storage for workflow-control events."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from x_editor.workflow_control.schemas import (
    DecisionCheckpoint,
    DecisionResolution,
    RepairComparison,
    RepairDecision,
    WorkflowEvent,
    WorkflowEventStatus,
    WorkflowStage,
)


class WorkflowEventLogError(RuntimeError):
    """Raised when an audit log cannot be read or appended safely."""


class WorkflowEventLog:
    """Write ordered workflow events without rewriting prior records."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self.path = Path(path).expanduser()
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = self._load_last_sequence()

    @property
    def event_count(self) -> int:
        return self._sequence

    def append(
        self,
        *,
        stage: WorkflowStage,
        kind: str,
        status: WorkflowEventStatus = "observed",
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        with self._lock:
            event = WorkflowEvent(
                run_id=self.run_id,
                sequence=self._sequence + 1,
                stage=stage,
                kind=kind,
                status=status,
                attempt_id=attempt_id,
                subject_refs=list(subject_refs or []),
                payload=dict(payload or {}),
            )
            line = json.dumps(
                event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            )
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
                    handle.flush()
            except OSError as exc:
                raise WorkflowEventLogError(
                    f"cannot append workflow event log {self.path}: {exc}"
                ) from exc
            self._sequence = event.sequence
            return event

    def record_checkpoint(self, checkpoint: DecisionCheckpoint) -> WorkflowEvent:
        return self.append(
            stage=checkpoint.stage,
            kind="decision_checkpoint_created",
            status="pending",
            subject_refs=checkpoint.subject_refs,
            payload={"checkpoint": checkpoint.model_dump(mode="json")},
        )

    def record_resolution(
        self, resolution: DecisionResolution, *, stage: WorkflowStage
    ) -> WorkflowEvent:
        status: WorkflowEventStatus = {
            "pending": "pending",
            "accepted": "accepted",
            "auto_accepted": "accepted",
            "rejected": "rejected",
        }[resolution.status]
        return self.append(
            stage=stage,
            kind="decision_checkpoint_resolved",
            status=status,
            payload={"resolution": resolution.model_dump(mode="json")},
        )

    def record_repair_decision(
        self,
        decision: RepairDecision,
        *,
        attempt_id: str | None = None,
    ) -> WorkflowEvent:
        return self.append(
            stage="repair",
            kind="repair_decision_observed"
            if decision.shadow_only
            else "repair_decision_selected",
            status="observed",
            attempt_id=attempt_id,
            subject_refs=decision.subject_refs,
            payload={"decision": decision.model_dump(mode="json")},
        )

    def record_repair_comparison(
        self,
        comparison: RepairComparison,
        *,
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
    ) -> WorkflowEvent:
        status: WorkflowEventStatus = {
            "accepted": "accepted",
            "rejected": "rejected",
            "failed": "failed",
            "not_run": "completed",
            "unknown": "observed",
        }[comparison.outcome]
        return self.append(
            stage="repair",
            kind="repair_decision_compared",
            status=status,
            attempt_id=attempt_id,
            subject_refs=subject_refs,
            payload={"comparison": comparison.model_dump(mode="json")},
        )

    def read_events(self) -> list[WorkflowEvent]:
        """Return this run's events in authoritative sequence order."""

        if not self.path.exists():
            return []
        events: list[WorkflowEvent] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise WorkflowEventLogError(
                            "invalid JSON in workflow event log "
                            f"{self.path} at line {line_number}: {exc}"
                        ) from exc
                    if record.get("run_id") != self.run_id:
                        continue
                    try:
                        events.append(WorkflowEvent.model_validate(record))
                    except ValueError as exc:
                        raise WorkflowEventLogError(
                            f"invalid workflow event in {self.path} at line {line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise WorkflowEventLogError(
                f"cannot read workflow event log {self.path}: {exc}"
            ) from exc
        events.sort(key=lambda item: item.sequence)
        expected = list(range(1, len(events) + 1))
        if [item.sequence for item in events] != expected:
            raise WorkflowEventLogError(
                f"non-contiguous workflow sequence for run {self.run_id} in {self.path}"
            )
        return events

    def _load_last_sequence(self) -> int:
        if not self.path.exists():
            return 0
        last_sequence = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise WorkflowEventLogError(
                            f"invalid JSON in workflow event log {self.path} at line {line_number}: {exc}"
                        ) from exc
                    if record.get("run_id") != self.run_id:
                        continue
                    sequence = record.get("sequence")
                    if not isinstance(sequence, int) or sequence < 1:
                        raise WorkflowEventLogError(
                            f"invalid sequence in workflow event log {self.path} at line {line_number}"
                        )
                    last_sequence = max(last_sequence, sequence)
        except OSError as exc:
            raise WorkflowEventLogError(
                f"cannot read workflow event log {self.path}: {exc}"
            ) from exc
        return last_sequence
