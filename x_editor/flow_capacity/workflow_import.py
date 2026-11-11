"""Read-only projection from workflow events into capacity ledger facts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .identity import stable_id
from .schemas import AttemptFact, FailureUnknownFact


@dataclass(frozen=True)
class WorkflowLedgerImport:
    attempts: tuple[AttemptFact, ...]
    failures_unknowns: tuple[FailureUnknownFact, ...]


def workflow_events_to_ledger_facts(
    events: Sequence[Any],
    *,
    assessment_record_id: str,
) -> WorkflowLedgerImport:
    """Import explicit event facts without deriving objective utility or capacity."""

    rows = [_event_dict(item) for item in events]
    if not rows:
        return WorkflowLedgerImport(attempts=(), failures_unknowns=())
    rows.sort(key=lambda item: int(item["sequence"]))
    run_ids = {str(item.get("run_id") or "") for item in rows}
    if len(run_ids) != 1 or "" in run_ids:
        raise ValueError("workflow import requires exactly one non-empty run_id")
    sequences = [int(item["sequence"]) for item in rows]
    if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
        raise ValueError("workflow import requires contiguous event sequence")

    run_id = next(iter(run_ids))
    selected_attempt_id = _selected_attempt_id(rows)
    events_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[FailureUnknownFact] = []
    for row in rows:
        attempt_id = str(row.get("attempt_id") or "")
        if attempt_id:
            events_by_attempt[attempt_id].append(row)
        if _is_failure_or_blocked(row):
            payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
            reason_codes = _reason_codes(row, payload)
            for index, reason_code in enumerate(reason_codes):
                failures.append(
                    FailureUnknownFact(
                        fact_id=stable_id(
                            "workflow_fact",
                            {
                                "event_id": row.get("event_id"),
                                "reason_code": reason_code,
                                "index": index,
                            },
                        ),
                        run_id=run_id,
                        assessment_record_id=assessment_record_id,
                        attempt_id=attempt_id or None,
                        category=(
                            "blocked"
                            if row.get("kind") == "artifact_store_blocked"
                            else "failure"
                        ),
                        reason_code=reason_code,
                        failure_stage=str(row.get("stage") or "") or None,
                        failure_owner=str(payload.get("failure_owner") or row.get("stage") or "") or None,
                        evidence_refs=_string_list(payload.get("evidence_refs")),
                        source_event_id=str(row.get("event_id") or "") or None,
                    )
                )

    attempts = tuple(
        _attempt_fact(
            run_id=run_id,
            attempt_id=attempt_id,
            rows=attempt_rows,
            selected_attempt_id=selected_attempt_id,
        )
        for attempt_id, attempt_rows in sorted(events_by_attempt.items())
    )
    return WorkflowLedgerImport(attempts=attempts, failures_unknowns=tuple(failures))


def _attempt_fact(
    *,
    run_id: str,
    attempt_id: str,
    rows: Sequence[Mapping[str, Any]],
    selected_attempt_id: str | None,
) -> AttemptFact:
    statuses = {str(item.get("status") or "") for item in rows}
    kinds = {str(item.get("kind") or "") for item in rows}
    attempt_started = bool(rows)
    if "failed" in statuses or "artifact_store_blocked" in kinds:
        execution_state = "failed"
    elif statuses.intersection({"completed", "accepted"}):
        execution_state = "succeeded"
    else:
        execution_state = "not_attempted"
    costs: dict[str, float] = defaultdict(float)
    evidence_refs: set[str] = set()
    for item in rows:
        payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
        raw_cost = payload.get("cost")
        if isinstance(raw_cost, Mapping):
            for key, value in raw_cost.items():
                if isinstance(value, (int, float)):
                    costs[str(key)] += float(value)
        evidence_refs.update(_string_list(payload.get("evidence_refs")))
    return AttemptFact(
        run_id=run_id,
        attempt_id=attempt_id,
        attempt_kind="runtime",
        attempt_group_id=run_id,
        selected_for_forward_flow=attempt_id == selected_attempt_id,
        attempt_started=attempt_started,
        execution_state=execution_state,
        cost=dict(sorted(costs.items())),
        evidence_refs=sorted(evidence_refs),
        source_event_ids=[str(item.get("event_id") or "") for item in rows if item.get("event_id")],
    )


def _selected_attempt_id(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for row in reversed(rows):
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        selected = payload.get("selected_attempt_id")
        if selected:
            return str(selected)
        if row.get("kind") == "run_completed" and row.get("attempt_id"):
            return str(row["attempt_id"])
    return None


def _is_failure_or_blocked(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "failed" or row.get("kind") == "artifact_store_blocked"


def _reason_codes(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    for key in ("failure_codes", "reason_codes"):
        values = _string_list(payload.get(key))
        if values:
            return values
    return [str(row.get("kind") or row.get("status") or "workflow_failure")]


def _event_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported workflow event type: {type(value)!r}")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if str(item)]
