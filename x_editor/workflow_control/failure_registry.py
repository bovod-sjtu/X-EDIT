"""Project per-run failure and repair history from workflow events."""

from __future__ import annotations

from typing import Any, Iterable

from x_editor.workflow_control.schemas import (
    WorkflowEvent,
    WorkflowFailureOccurrence,
    WorkflowFailureRegistry,
    WorkflowRepairHistoryEntry,
)


def project_workflow_failure_registry(
    events: Iterable[WorkflowEvent | dict[str, Any]],
) -> WorkflowFailureRegistry:
    """Rebuild one run's terminal failure and repair history without side effects."""

    rows = sorted((_event(item) for item in events), key=lambda item: item.sequence)
    if not rows:
        raise ValueError("workflow failure projection requires at least one event")
    run_ids = {item.run_id for item in rows}
    if len(run_ids) != 1:
        raise ValueError("workflow failure projection accepts exactly one run_id")
    run_completed = next(
        (item for item in reversed(rows) if item.kind == "run_completed"), None
    )
    if run_completed is None or run_completed.status not in {"completed", "failed"}:
        raise ValueError("workflow failure projection requires a terminal run_completed event")

    artifact_steps = _artifact_step_index(rows)
    failures = [
        occurrence
        for event in rows
        if (occurrence := _failure_occurrence(event, artifact_steps)) is not None
    ]
    repairs = [
        repair
        for event in rows
        if (repair := _repair_history_entry(event)) is not None
    ]
    store_failure = next(
        (
            item
            for item in reversed(failures)
            if item.event_kind == "artifact_store_blocked"
        ),
        None,
    )
    terminal_failure = store_failure or (failures[-1] if failures else None)
    payload = run_completed.payload
    status = "unresolved" if run_completed.status == "failed" else "clear"
    terminal_steps = list(terminal_failure.failed_step_ids if terminal_failure else [])
    terminal_artifacts = list(
        terminal_failure.failed_artifact_refs if terminal_failure else []
    )
    terminal_codes = list(terminal_failure.failure_codes if terminal_failure else [])
    if status == "unresolved":
        terminal_steps = _dedupe(
            [
                *terminal_steps,
                *_string_list(
                    _mapping(payload.get("execution_summary")).get("failed_steps")
                ),
            ]
        )
        if not terminal_codes:
            terminal_codes = ["run_failed_without_structured_cause"]

    return WorkflowFailureRegistry(
        run_id=rows[0].run_id,
        status=status,
        run_status=run_completed.status,
        terminal_failure_stage=(
            terminal_failure.stage
            if status == "unresolved" and terminal_failure
            else None
        ),
        terminal_failed_step_ids=terminal_steps if status == "unresolved" else [],
        terminal_failed_artifact_refs=terminal_artifacts if status == "unresolved" else [],
        unresolved_failure_codes=terminal_codes if status == "unresolved" else [],
        legacy_final_output_path=_string_or_none(payload.get("legacy_final_output_path")),
        authoritative_final_output_path=_string_or_none(payload.get("final_output_path")),
        failure_history=failures,
        repair_history=repairs,
        source_event_ids=[item.event_id for item in rows],
    )


def _failure_occurrence(
    event: WorkflowEvent,
    artifact_steps: dict[str, str],
) -> WorkflowFailureOccurrence | None:
    payload = event.payload
    if event.kind.endswith("execution_completed"):
        failed_steps = _string_list(
            _mapping(payload.get("execution_summary")).get("failed_steps")
        )
        if failed_steps:
            return _occurrence(
                event,
                failed_step_ids=failed_steps,
                failure_codes=["execution_step_failed"],
                summary=f"Execution reported failed steps: {', '.join(failed_steps)}.",
            )

    if event.kind == "artifact_disposition_audited":
        disposition = str(payload.get("disposition") or "")
        if disposition not in {"candidate", "rejected"}:
            return None
        artifact_refs = _artifact_refs(event, payload)
        reasons = _string_list(payload.get("reasons"))
        return _occurrence(
            event,
            failed_step_ids=_steps_for_artifacts(artifact_refs, artifact_steps, event),
            failed_artifact_refs=artifact_refs,
            failure_codes=reasons or [f"artifact_{disposition}"],
            summary=(
                "Artifact remained a candidate without acceptance evidence."
                if disposition == "candidate"
                else "Artifact was rejected by finalization audit."
            ),
        )

    if event.kind == "artifact_finalization_audited":
        audit = _mapping(payload.get("audit"))
        if audit.get("status") == "would_finalize":
            return None
        artifact_refs = _dedupe(
            [
                *_string_list(audit.get("candidate_artifact_refs")),
                *_string_list(audit.get("rejected_artifact_refs")),
            ]
        )
        findings = [
            item for item in audit.get("findings") or [] if isinstance(item, dict)
        ]
        failed_steps = _dedupe(
            [
                *_string_list(audit.get("unresolved_required_steps")),
                *[
                    str(item.get("step_id"))
                    for item in findings
                    if item.get("step_id")
                ],
                *_steps_for_artifacts(artifact_refs, artifact_steps, event),
            ]
        )
        return _occurrence(
            event,
            failed_step_ids=failed_steps,
            failed_artifact_refs=artifact_refs,
            failure_codes=_dedupe(
                [str(item.get("code")) for item in findings if item.get("code")]
            )
            or [f"artifact_audit_{audit.get('status') or 'unresolved'}"],
            summary=f"Artifact finalization audit ended as {audit.get('status') or 'unresolved'}.",
        )

    if event.kind == "artifact_store_blocked":
        commit = _mapping(payload.get("commit"))
        artifact_refs = _dedupe(
            [
                *_string_list(commit.get("candidate_artifact_refs")),
                *_string_list(commit.get("rejected_artifact_refs")),
            ]
        )
        return _occurrence(
            event,
            failed_step_ids=_steps_for_artifacts(artifact_refs, artifact_steps, event),
            failed_artifact_refs=artifact_refs,
            failure_codes=_string_list(commit.get("failure_codes"))
            or ["artifact_store_blocked"],
            summary=str(
                commit.get("plain_summary")
                or "ArtifactStore blocked authoritative final delivery."
            ),
        )
    return None


def _repair_history_entry(event: WorkflowEvent) -> WorkflowRepairHistoryEntry | None:
    payload = event.payload
    if event.kind == "repair_decision_selected":
        decision = _mapping(payload.get("decision"))
        return WorkflowRepairHistoryEntry(
            event_id=event.event_id,
            sequence=event.sequence,
            attempt_id=event.attempt_id,
            event_kind=event.kind,
            repair_stage=_string_or_none(decision.get("stage")),
            action=str(decision.get("action") or "unknown"),
            scope=_string_or_none(decision.get("scope")),
            status=event.status,
            outcome="selected",
            subject_refs=_dedupe(
                [*event.subject_refs, *_string_list(decision.get("subject_refs"))]
            ),
            reason_codes=_string_list(decision.get("reason_codes")),
            details={"decision_id": decision.get("decision_id")},
        )
    if event.kind == "argument_repair_enforcement_completed":
        return WorkflowRepairHistoryEntry(
            event_id=event.event_id,
            sequence=event.sequence,
            attempt_id=event.attempt_id,
            event_kind=event.kind,
            repair_stage="tool_contract",
            action="repair_step_arguments",
            scope="argument",
            status=event.status,
            outcome=_string_or_none(payload.get("status")),
            accepted=event.status == "accepted",
            subject_refs=list(event.subject_refs),
            details=_selected_details(payload),
        )
    if event.kind == "legacy_repair_attempt_completed":
        accepted = payload.get("accepted")
        return WorkflowRepairHistoryEntry(
            event_id=event.event_id,
            sequence=event.sequence,
            attempt_id=event.attempt_id,
            event_kind=event.kind,
            repair_stage=_string_or_none(payload.get("repair_stage")),
            action="replan_global",
            scope="plan",
            status=event.status,
            outcome="accepted" if accepted is True else "rejected",
            accepted=accepted if isinstance(accepted, bool) else event.status == "accepted",
            subject_refs=list(event.subject_refs),
            details=_selected_details(payload),
        )
    return None


def _artifact_step_index(events: list[WorkflowEvent]) -> dict[str, str]:
    result: dict[str, str] = {}
    for event in events:
        if event.kind != "artifact_candidate_observed":
            continue
        artifact = _mapping(event.payload.get("artifact"))
        artifact_ref = _string_or_none(artifact.get("artifact_ref_id"))
        step_id = _string_or_none(artifact.get("producer_step_id"))
        if artifact_ref and step_id:
            result[artifact_ref] = step_id
    return result


def _occurrence(
    event: WorkflowEvent,
    *,
    failed_step_ids: list[str] | None = None,
    failed_artifact_refs: list[str] | None = None,
    failure_codes: list[str] | None = None,
    summary: str,
) -> WorkflowFailureOccurrence:
    return WorkflowFailureOccurrence(
        event_id=event.event_id,
        sequence=event.sequence,
        stage=event.stage,
        event_kind=event.kind,
        status=event.status,
        attempt_id=event.attempt_id,
        failed_step_ids=_dedupe(failed_step_ids or []),
        failed_artifact_refs=_dedupe(failed_artifact_refs or []),
        failure_codes=_dedupe(failure_codes or []),
        summary=summary,
    )


def _steps_for_artifacts(
    artifact_refs: list[str], artifact_steps: dict[str, str], event: WorkflowEvent
) -> list[str]:
    steps = [artifact_steps[item] for item in artifact_refs if item in artifact_steps]
    steps.extend(item for item in event.subject_refs if item not in artifact_refs)
    return _dedupe(steps)


def _artifact_refs(event: WorkflowEvent, payload: dict[str, Any]) -> list[str]:
    explicit = _string_or_none(payload.get("artifact_ref_id"))
    return _dedupe([*([explicit] if explicit else []), *event.subject_refs[:1]])


def _selected_details(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "attempt",
        "quality_before",
        "quality_after",
        "score_before",
        "score_after",
        "previous_violations",
        "previous_defects",
        "remaining_violations",
        "rejection_reason",
        "error",
        "legacy_fallback_action",
    )
    return {key: payload[key] for key in keys if payload.get(key) not in (None, [], {})}


def _event(value: WorkflowEvent | dict[str, Any]) -> WorkflowEvent:
    return value if isinstance(value, WorkflowEvent) else WorkflowEvent.model_validate(value)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _string_or_none(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))
