"""Non-authoritative artifact and verifier evidence observation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from x_editor.core.schemas import AudioEditPlan
from x_editor.workflow_control.artifact_acceptance import DELIVERY_CONTRACT_CHECK_IDS
from x_editor.workflow_control.schemas import (
    ArtifactAcceptanceTier,
    ArtifactAuditRecord,
    ArtifactDeliveryProvenance,
    ArtifactFinalizationAudit,
    ArtifactShadowAggregate,
    ArtifactShadowObservation,
    ArtifactShadowRecord,
    ArtifactVerificationOutcome,
    ArtifactVerificationScope,
    VerifierShadowRun,
)

_VERIFIER_OUTCOMES = {
    "passed",
    "failed",
    "inconclusive",
    "skipped",
    "unavailable",
}
_VERIFICATION_SCOPES = {
    "artifact_integrity",
    "operation_invariant",
    "step_diagnostic",
    "objective",
}
_DELIVERY_PROVENANCE = {"tool", "composite", "local"}


class ArtifactShadowLogError(RuntimeError):
    """Raised when a persisted shadow observation cannot be decoded."""


class ArtifactShadowObserver:
    """Project existing runtime evidence without changing artifact authority."""

    def observe(
        self,
        plan: AudioEditPlan,
        audit: ArtifactFinalizationAudit,
    ) -> ArtifactShadowObservation:
        verifier_runs = [
            _verifier_shadow_run(report)
            for report in _verification_reports(plan)
        ]
        artifacts = [_artifact_shadow_record(item) for item in audit.artifacts]
        shadow_final_ref = _shadow_preferred_final_ref(audit)
        return ArtifactShadowObservation(
            audit_id=audit.audit_id,
            attempt_id=audit.attempt_id,
            audit_status=audit.status,
            verifier_runs=verifier_runs,
            artifacts=artifacts,
            legacy_final_artifact_ref_id=audit.legacy_final_artifact_ref_id,
            shadow_preferred_final_artifact_ref_id=shadow_final_ref,
            final_relation=_final_relation(
                audit.legacy_final_artifact_ref_id,
                shadow_final_ref,
            ),
            verifier_outcome_counts=_counts(run.outcome for run in verifier_runs),
            artifact_disposition_counts=_counts(
                artifact.disposition for artifact in artifacts
            ),
            artifact_verification_status_counts=_counts(
                artifact.verification_status for artifact in artifacts
            ),
            acceptance_tier_counts=_counts(
                artifact.acceptance_tier for artifact in artifacts
            ),
            candidate_reason_counts=_counts(
                reason
                for artifact in artifacts
                if artifact.disposition == "candidate"
                for reason in artifact.candidate_reasons
            ),
            delivery_provenance_counts=_counts(
                run.delivery_provenance
                for run in verifier_runs
                if run.delivery_provenance is not None
            ),
        )


def summarize_artifact_shadow_observations(
    observations: Iterable[ArtifactShadowObservation],
) -> ArtifactShadowAggregate:
    """Aggregate persisted observations without requiring plans or audio files."""

    items = list(observations)
    verifier_outcomes: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    final_relations: Counter[str] = Counter()
    acceptance_tiers: Counter[str] = Counter()
    delivery_provenance: Counter[str] = Counter()
    operation_verifier_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    operation_dispositions: dict[str, Counter[str]] = defaultdict(Counter)
    operations: set[str] = set()

    for observation in items:
        final_relations[observation.final_relation] += 1
        for run in observation.verifier_runs:
            verifier_outcomes[run.outcome] += 1
            if run.delivery_provenance is not None:
                delivery_provenance[run.delivery_provenance] += 1
            if run.operation:
                operations.add(run.operation)
                operation_verifier_outcomes[run.operation][run.outcome] += 1
        for artifact in observation.artifacts:
            dispositions[artifact.disposition] += 1
            acceptance_tiers[artifact.acceptance_tier] += 1
            if artifact.operation:
                operations.add(artifact.operation)
                operation_dispositions[artifact.operation][artifact.disposition] += 1
            if artifact.disposition == "candidate":
                candidate_reasons.update(artifact.candidate_reasons)

    return ArtifactShadowAggregate(
        observation_count=len(items),
        attempt_count=len({item.audit_id for item in items}),
        operations=sorted(operations),
        verifier_outcome_counts=_sorted_counter(verifier_outcomes),
        operation_verifier_outcome_counts={
            operation: _sorted_counter(operation_verifier_outcomes[operation])
            for operation in sorted(operation_verifier_outcomes)
        },
        artifact_disposition_counts=_sorted_counter(dispositions),
        operation_artifact_disposition_counts={
            operation: _sorted_counter(operation_dispositions[operation])
            for operation in sorted(operation_dispositions)
        },
        candidate_reason_counts=_sorted_counter(candidate_reasons),
        final_relation_counts=_sorted_counter(final_relations),
        acceptance_tier_counts=_sorted_counter(acceptance_tiers),
        delivery_provenance_counts=_sorted_counter(delivery_provenance),
    )


def load_artifact_shadow_observations(
    event_log_path: str | Path,
) -> list[ArtifactShadowObservation]:
    """Load shadow observations from an append-only workflow event log."""

    path = Path(event_log_path).expanduser()
    observations: list[ArtifactShadowObservation] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ArtifactShadowLogError(
                        f"invalid workflow event JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if event.get("kind") != "artifact_shadow_observed":
                    continue
                payload = event.get("payload")
                raw_observation = (
                    payload.get("observation") if isinstance(payload, dict) else None
                )
                try:
                    observations.append(
                        ArtifactShadowObservation.model_validate(raw_observation)
                    )
                except ValidationError as exc:
                    raise ArtifactShadowLogError(
                        "invalid artifact shadow observation at "
                        f"{path}:{line_number}: {exc.error_count()} validation errors"
                    ) from exc
    except OSError as exc:
        raise ArtifactShadowLogError(
            f"cannot read workflow event log {path}: {exc}"
        ) from exc
    return observations


def _verification_reports(plan: AudioEditPlan) -> list[dict[str, Any]]:
    reports = plan.metadata.get("verification_reports") or []
    return [dict(report) for report in reports if isinstance(report, dict)]


def _verifier_shadow_run(report: dict[str, Any]) -> VerifierShadowRun:
    return VerifierShadowRun(
        component=str(report.get("component") or "unknown"),
        step_id=_string_or_none(report.get("step_id")),
        operation=_string_or_none(report.get("operation")),
        outcome=_report_outcome(report),
        scope=_report_scope(report),
        acceptance_grade=bool(report.get("acceptance_grade")),
        check_ids=_string_list(report.get("check_ids")),
        check_outcomes=_check_outcomes(report),
        subject_paths=_string_list(report.get("subject_paths")),
        delivery_provenance=_delivery_provenance_from_report(report),
    )


def _artifact_shadow_record(artifact: ArtifactAuditRecord) -> ArtifactShadowRecord:
    failed_required_check_ids = sorted(
        {
            check_id
            for evidence in artifact.verification_evidence
            if evidence.acceptance_grade and evidence.subject_path_matched
            for check_id in evidence.check_ids
            if check_id in artifact.required_check_ids
            and evidence.check_outcomes.get(check_id, evidence.outcome) == "failed"
        }
    )
    return ArtifactShadowRecord(
        artifact_ref_id=artifact.artifact_ref_id,
        step_id=artifact.producer_step_id,
        operation=artifact.producer_operation,
        acceptance_tier=_acceptance_tier(artifact),
        acceptance_policy_id=artifact.acceptance_policy_id,
        disposition=artifact.disposition,
        verification_status=artifact.verification_status,
        candidate_reasons=(
            list(artifact.disposition_reasons)
            if artifact.disposition == "candidate"
            else []
        ),
        missing_check_ids=list(artifact.missing_check_ids),
        failed_required_check_ids=failed_required_check_ids,
        legacy_selected=artifact.legacy_selected,
        primary_output=artifact.primary_output,
    )


def _acceptance_tier(artifact: ArtifactAuditRecord) -> ArtifactAcceptanceTier:
    if artifact.acceptance_policy_id is None:
        return "uncovered"
    if set(artifact.required_check_ids) == set(DELIVERY_CONTRACT_CHECK_IDS):
        return "delivery_contract"
    return "operation_principle"


def _shadow_preferred_final_ref(audit: ArtifactFinalizationAudit) -> str | None:
    legacy = next(
        (
            artifact
            for artifact in audit.artifacts
            if artifact.artifact_ref_id == audit.legacy_final_artifact_ref_id
        ),
        None,
    )
    if legacy is not None and legacy.disposition == "accepted":
        return legacy.artifact_ref_id
    accepted_primary = [
        artifact
        for artifact in audit.artifacts
        if artifact.disposition == "accepted" and artifact.primary_output
    ]
    return accepted_primary[-1].artifact_ref_id if accepted_primary else None


def _final_relation(legacy_ref: str | None, shadow_ref: str | None) -> str:
    if legacy_ref is None:
        return "legacy_missing"
    if shadow_ref is None:
        return "no_shadow_eligible"
    if legacy_ref == shadow_ref:
        return "match"
    return "differs"


def _delivery_provenance_from_report(
    report: dict[str, Any],
) -> ArtifactDeliveryProvenance | None:
    if not str(report.get("component") or "").startswith(
        "framework.delivery_contract."
    ):
        return None
    metrics = report.get("metrics")
    checks = metrics.get("checks") if isinstance(metrics, dict) else None
    execution = (
        checks.get("framework.delivery_contract.execution")
        if isinstance(checks, dict)
        else None
    )
    provenance = execution.get("provenance") if isinstance(execution, dict) else None
    return provenance if provenance in _DELIVERY_PROVENANCE else None


def _report_outcome(report: dict[str, Any]) -> ArtifactVerificationOutcome:
    outcome = str(report.get("outcome") or "")
    if outcome in _VERIFIER_OUTCOMES:
        return outcome  # type: ignore[return-value]
    return "passed" if report.get("passed") is not False else "failed"


def _report_scope(report: dict[str, Any]) -> ArtifactVerificationScope:
    scope = str(report.get("scope") or "step_diagnostic")
    if scope in _VERIFICATION_SCOPES:
        return scope  # type: ignore[return-value]
    return "step_diagnostic"


def _check_outcomes(
    report: dict[str, Any],
) -> dict[str, ArtifactVerificationOutcome]:
    raw = report.get("check_outcomes")
    if not isinstance(raw, dict):
        return {}
    return {
        str(check_id): outcome  # type: ignore[misc]
        for check_id, value in raw.items()
        if (outcome := str(value)) in _VERIFIER_OUTCOMES
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None and str(value) else None


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return _sorted_counter(Counter(str(value) for value in values))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}
