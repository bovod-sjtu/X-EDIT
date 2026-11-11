"""Audit-only artifact disposition and final-path projection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import soundfile as sf

from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.workflow_control.artifact_acceptance import (
    ArtifactAcceptancePolicyRegistry,
    build_default_artifact_acceptance_policy_registry,
)
from x_editor.workflow_control.schemas import (
    ArtifactAcceptancePolicy,
    ArtifactAuditAttemptOutcome,
    ArtifactAuditRecord,
    ArtifactFinalizationAudit,
    ArtifactFinalizationFinding,
    ArtifactIntegrityCheck,
    ArtifactVerificationEvidence,
    ArtifactVerificationOutcome,
    ArtifactVerificationStatus,
)

_PRIMARY_OUTPUT_KEYS = {
    "generated_audio_path",
    "output_audio_path",
    "output_path",
    "target_audio_path",
}


class ArtifactFinalizationAuditor:
    """Project artifact facts without changing plans, paths, or repair choices."""

    def __init__(
        self,
        acceptance_policies: ArtifactAcceptancePolicyRegistry | None = None,
    ) -> None:
        self.acceptance_policies = (
            acceptance_policies
            or build_default_artifact_acceptance_policy_registry()
        )

    def audit(
        self,
        plan: AudioEditPlan,
        *,
        attempt_id: str,
        attempt_outcome: ArtifactAuditAttemptOutcome = "selected",
    ) -> ArtifactFinalizationAudit:
        steps = {step.id: step for step in plan.steps}
        verification_by_step = _verification_reports_by_step(plan)
        parent_paths_by_step = _parent_paths_by_step(plan)
        legacy_final_path = _string_or_none(plan.metadata.get("final_output_path"))
        rows = _candidate_rows(plan)
        artifacts: list[ArtifactAuditRecord] = []

        for row in rows:
            step_id = str(row.get("step_id") or "")
            step = steps.get(step_id)
            path = str(row.get("path") or "")
            checks = _integrity_checks(path)
            integrity_passed = not any(check.status == "failed" for check in checks)
            reports = verification_by_step.get(step_id, [])
            verification_status = _verification_status(reports)
            verification_evidence = _verification_evidence(reports, path)
            policy = self.acceptance_policies.policy_for(
                step.operation if step is not None else None
            )
            required_check_ids = list(policy.required_check_ids) if policy else []
            satisfied_check_ids = _satisfied_check_ids(
                verification_evidence, required_check_ids
            )
            missing_check_ids = sorted(
                set(required_check_ids) - set(satisfied_check_ids)
            )
            disposition, reasons = _disposition(
                step=step,
                integrity_passed=integrity_passed,
                verification_status=verification_status,
                attempt_outcome=attempt_outcome,
                policy=policy,
                evidence=verification_evidence,
                missing_check_ids=missing_check_ids,
            )
            artifacts.append(
                ArtifactAuditRecord(
                    artifact_ref_id=_artifact_ref_id(
                        attempt_id=attempt_id,
                        step_id=step_id,
                        path=path,
                    ),
                    attempt_id=attempt_id,
                    path=path,
                    role=str(row.get("role") or "audio"),
                    kind=str(row.get("kind") or "step_output"),
                    producer_step_id=step_id,
                    producer_operation=step.operation if step is not None else None,
                    producer_step_status=step.status if step is not None else None,
                    primary_output=bool(row.get("primary_output")),
                    parent_paths=list(parent_paths_by_step.get(step_id, [])),
                    provenance=_string_or_none(row.get("source")),
                    integrity_checks=checks,
                    integrity_passed=integrity_passed,
                    verification_status=verification_status,
                    verification_refs=[
                        str(item.get("component"))
                        for item in reports
                        if item.get("component")
                    ],
                    verification_evidence=verification_evidence,
                    acceptance_policy_id=policy.policy_id if policy else None,
                    required_check_ids=required_check_ids,
                    satisfied_check_ids=satisfied_check_ids,
                    missing_check_ids=missing_check_ids,
                    disposition=disposition,
                    disposition_reasons=reasons,
                    legacy_selected=False,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "kind",
                            "path",
                            "primary_output",
                            "role",
                            "source",
                            "step_id",
                        }
                    },
                )
            )

        _attach_parent_refs(artifacts)
        legacy_matches = [
            item for item in artifacts if _same_path(item.path, legacy_final_path)
        ]
        legacy_final = legacy_matches[-1] if legacy_matches else None
        if legacy_final is not None:
            legacy_final.legacy_selected = True
        required_step_ids = sorted(step.id for step in plan.steps if step.required)
        required_artifact_refs, unresolved_required_steps = _required_artifacts(
            plan.steps,
            artifacts,
            self.acceptance_policies,
        )
        findings = _finalization_findings(
            plan=plan,
            artifacts=artifacts,
            legacy_final_path=legacy_final_path,
            legacy_final=legacy_final,
            unresolved_required_steps=unresolved_required_steps,
        )
        status = _finalization_status(
            legacy_final=legacy_final,
            findings=findings,
            unresolved_required_steps=unresolved_required_steps,
        )
        return ArtifactFinalizationAudit(
            attempt_id=attempt_id,
            attempt_outcome=attempt_outcome,
            status=status,
            legacy_final_path=legacy_final_path,
            legacy_final_artifact_ref_id=(
                legacy_final.artifact_ref_id if legacy_final is not None else None
            ),
            legacy_final_disposition=(
                legacy_final.disposition if legacy_final is not None else None
            ),
            artifacts=artifacts,
            accepted_artifact_refs=[
                item.artifact_ref_id
                for item in artifacts
                if item.disposition == "accepted"
            ],
            candidate_artifact_refs=[
                item.artifact_ref_id
                for item in artifacts
                if item.disposition == "candidate"
            ],
            rejected_artifact_refs=[
                item.artifact_ref_id
                for item in artifacts
                if item.disposition == "rejected"
            ],
            required_step_ids=required_step_ids,
            required_artifact_refs=required_artifact_refs,
            unresolved_required_steps=unresolved_required_steps,
            findings=findings,
        )


def _candidate_rows(plan: AudioEditPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    declared = {
        (str(item.get("from_step") or ""), _path_key(str(item.get("path") or ""))): item
        for item in plan.metadata.get("output_artifacts") or []
        if isinstance(item, dict) and item.get("from_step") and item.get("path")
    }
    for item in plan.metadata.get("audio_artifacts") or []:
        if not isinstance(item, dict) or item.get("kind") != "step_output":
            continue
        step_id = str(item.get("step_id") or "")
        path = _string_or_none(item.get("path"))
        if not step_id or not path:
            continue
        key = (step_id, _path_key(path))
        row = dict(item)
        declaration = declared.get(key)
        if declaration is not None:
            row["artifact_id"] = declaration["artifact_id"]
            row["role"] = declaration["role"]
            row["primary_output"] = bool(declaration["primary"])
            row["declared_output"] = True
        else:
            row["primary_output"] = _is_primary_result_key(item.get("result_key"))
        if key not in seen:
            rows.append(row)
            seen.add(key)

    for step in plan.steps:
        path, result_key = _primary_step_output(step)
        if not path:
            continue
        key = (step.id, _path_key(path))
        if key in seen:
            for row in rows:
                if (
                    str(row.get("step_id") or "") == step.id
                    and _same_path(row.get("path"), path)
                    and not row.get("declared_output")
                ):
                    row["primary_output"] = True
            continue
        rows.append(
            {
                "artifact_id": f"{step.id}:{result_key}",
                "step_id": step.id,
                "role": step.expected_output or result_key,
                "kind": "step_output",
                "path": path,
                "source": f"step_result.{step.id}.{result_key}",
                "result_key": result_key,
                "primary_output": True,
            }
        )
        seen.add(key)
    return rows


def _primary_step_output(step: AudioEditStep) -> tuple[str | None, str]:
    if not isinstance(step.result, dict):
        return None, "output_audio_path"
    for key in (
        "output_audio_path",
        "target_audio_path",
        "output_path",
        "generated_audio_path",
    ):
        value = _string_or_none(step.result.get(key))
        if value:
            return value, key
    return None, "output_audio_path"


def _integrity_checks(path_text: str) -> list[ArtifactIntegrityCheck]:
    path = Path(path_text).expanduser()
    exists = path.exists()
    checks = [
        ArtifactIntegrityCheck(
            check_id="path_exists",
            status="passed" if exists else "failed",
            plain_summary="Artifact path exists." if exists else "Artifact path is missing.",
        )
    ]
    if not exists:
        return checks

    regular_file = path.is_file()
    checks.append(
        ArtifactIntegrityCheck(
            check_id="regular_file",
            status="passed" if regular_file else "failed",
            plain_summary=(
                "Artifact path is a regular file."
                if regular_file
                else "Artifact path is not a regular file."
            ),
        )
    )
    if not regular_file:
        return checks

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        checks.append(
            ArtifactIntegrityCheck(
                check_id="file_stat",
                status="failed",
                plain_summary="Artifact metadata cannot be read.",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        return checks

    checks.append(
        ArtifactIntegrityCheck(
            check_id="non_empty_file",
            status="passed" if size_bytes > 0 else "failed",
            plain_summary=(
                "Artifact file is non-empty."
                if size_bytes > 0
                else "Artifact file is empty."
            ),
            details={"size_bytes": size_bytes},
        )
    )
    try:
        with path.open("rb") as handle:
            handle.read(1)
        checks.append(
            ArtifactIntegrityCheck(
                check_id="file_readable",
                status="passed",
                plain_summary="Artifact file is readable.",
            )
        )
    except OSError as exc:
        checks.append(
            ArtifactIntegrityCheck(
                check_id="file_readable",
                status="failed",
                plain_summary="Artifact file is not readable.",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        )

    if path.suffix.lower() != ".wav":
        checks.append(
            ArtifactIntegrityCheck(
                check_id="audio_container",
                status="not_checked",
                plain_summary="Container validation is not implemented for this format.",
                details={"suffix": path.suffix.lower()},
            )
        )
        return checks

    try:
        with sf.SoundFile(str(path), "r") as audio:
            channels = int(audio.channels)
            sample_rate = int(audio.samplerate)
            frame_count = int(audio.frames)
            subtype = str(audio.subtype or "")
            if frame_count > 0:
                audio.read(frames=1, dtype="float32", always_2d=True)
        valid = all(value > 0 for value in (channels, sample_rate, frame_count))
        checks.append(
            ArtifactIntegrityCheck(
                check_id="audio_container",
                status="passed" if valid else "failed",
                plain_summary=(
                    "WAV container has usable audio properties."
                    if valid
                    else "WAV container has invalid audio properties."
                ),
                details={
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "frame_count": frame_count,
                    "subtype": subtype,
                    "duration_sec": frame_count / sample_rate if sample_rate else None,
                },
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        checks.append(
            ArtifactIntegrityCheck(
                check_id="audio_container",
                status="failed",
                plain_summary="WAV container cannot be decoded.",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
    return checks


def _verification_reports_by_step(
    plan: AudioEditPlan,
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for item in plan.metadata.get("verification_reports") or []:
        if not isinstance(item, dict) or not item.get("step_id"):
            continue
        reports.setdefault(str(item["step_id"]), []).append(item)
    return reports


def _verification_status(
    reports: list[dict[str, Any]],
) -> ArtifactVerificationStatus:
    if not reports:
        return "not_run"
    if any(
        item.get("passed") is False and item.get("severity") == "blocking"
        for item in reports
    ):
        return "failed"
    outcomes = [_report_outcome(item) for item in reports]
    if "failed" in outcomes:
        return "passed_with_warnings"
    if "passed" in outcomes:
        return "passed"
    if "inconclusive" in outcomes:
        return "inconclusive"
    if "unavailable" in outcomes:
        return "unavailable"
    return "skipped"


def _verification_evidence(
    reports: list[dict[str, Any]],
    artifact_path: str,
) -> list[ArtifactVerificationEvidence]:
    evidence: list[ArtifactVerificationEvidence] = []
    for report in reports:
        subject_paths = _string_list(report.get("subject_paths"))
        scope = str(report.get("scope") or "step_diagnostic")
        if scope not in {
            "artifact_integrity",
            "operation_invariant",
            "step_diagnostic",
            "objective",
        }:
            scope = "step_diagnostic"
        evidence.append(
            ArtifactVerificationEvidence(
                component=str(report.get("component") or "unknown"),
                outcome=_report_outcome(report),
                scope=scope,
                acceptance_grade=bool(report.get("acceptance_grade")),
                check_ids=_string_list(report.get("check_ids")),
                check_outcomes=_check_outcomes(report),
                subject_paths=subject_paths,
                subject_path_matched=any(
                    _same_path(path, artifact_path) for path in subject_paths
                ),
                confidence=_string_or_none(report.get("confidence")),
                severity=_string_or_none(report.get("severity")),
            )
        )
    return evidence


def _report_outcome(report: dict[str, Any]) -> ArtifactVerificationOutcome:
    outcome = str(report.get("outcome") or "")
    if outcome in {"passed", "failed", "inconclusive", "skipped", "unavailable"}:
        return outcome
    return "passed" if report.get("passed") is not False else "failed"


def _check_outcomes(
    report: dict[str, Any],
) -> dict[str, ArtifactVerificationOutcome]:
    raw = report.get("check_outcomes")
    if not isinstance(raw, dict):
        return {}
    outcomes: dict[str, ArtifactVerificationOutcome] = {}
    for check_id, value in raw.items():
        outcome = str(value)
        if outcome in {
            "passed",
            "failed",
            "inconclusive",
            "skipped",
            "unavailable",
        }:
            outcomes[str(check_id)] = outcome
    return outcomes


def _satisfied_check_ids(
    evidence: list[ArtifactVerificationEvidence],
    required_check_ids: list[str],
) -> list[str]:
    required = set(required_check_ids)
    return sorted(
        {
            check_id
            for item in evidence
            if item.acceptance_grade
            and item.subject_path_matched
            and item.scope in {"artifact_integrity", "operation_invariant"}
            for check_id in item.check_ids
            if check_id in required
            and item.check_outcomes.get(check_id, item.outcome) == "passed"
        }
    )


def _disposition(
    *,
    step: AudioEditStep | None,
    integrity_passed: bool,
    verification_status: ArtifactVerificationStatus,
    attempt_outcome: ArtifactAuditAttemptOutcome,
    policy: ArtifactAcceptancePolicy | None,
    evidence: list[ArtifactVerificationEvidence],
    missing_check_ids: list[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if attempt_outcome == "rejected":
        reasons.append("runner_attempt_rejected")
    if step is None:
        reasons.append("producer_step_missing")
    elif step.status != "success":
        reasons.append(f"producer_step_{step.status}")
    if not integrity_passed:
        reasons.append("integrity_check_failed")
    has_path_bound_acceptance_failure = any(
        item.acceptance_grade
        and item.subject_path_matched
        and any(
            item.check_outcomes.get(check_id, item.outcome) == "failed"
            for check_id in item.check_ids
        )
        for item in evidence
    )
    if verification_status == "failed" and not has_path_bound_acceptance_failure:
        reasons.append("blocking_verification_failed")
    if reasons:
        return "rejected", reasons
    if attempt_outcome == "provisional":
        return "candidate", ["runner_attempt_not_selected"]
    if policy is None:
        reasons = ["artifact_acceptance_policy_missing"]
        intentionally_deferred = any(
            item.component == "framework.evaluator_deferred"
            and item.outcome == "skipped"
            and item.subject_path_matched
            for item in evidence
        )
        if intentionally_deferred:
            reasons.append("verification_deferred_to_evaluator")
        elif verification_status == "not_run":
            reasons.append("runtime_verification_not_run")
        else:
            reasons.append("acceptance_grade_evidence_missing")
        return "candidate", reasons
    failed_required_checks = sorted(
        {
            check_id
            for item in evidence
            if item.acceptance_grade
            and item.subject_path_matched
            for check_id in item.check_ids
            if check_id in policy.required_check_ids
            and item.check_outcomes.get(check_id, item.outcome) == "failed"
        }
    )
    if failed_required_checks:
        return "rejected", [
            f"required_artifact_check_failed:{check_id}"
            for check_id in failed_required_checks
        ]
    if missing_check_ids:
        return "candidate", [
            f"required_artifact_check_missing:{check_id}"
            for check_id in missing_check_ids
        ]
    return "accepted", ["required_artifact_checks_passed"]


def _parent_paths_by_step(plan: AudioEditPlan) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in plan.metadata.get("executor_step_records") or []:
        if not isinstance(item, dict) or not item.get("step_id"):
            continue
        step_id = str(item["step_id"])
        if item.get("consumed_audio_paths_observed") is True:
            result.setdefault(step_id, []).extend(
                str(path)
                for path in item.get("consumed_audio_paths") or []
                if path
            )
            continue
        path = _string_or_none(item.get("current_audio_before"))
        if path:
            result.setdefault(step_id, []).append(path)
    return {key: _dedupe(values) for key, values in result.items()}


def _attach_parent_refs(artifacts: list[ArtifactAuditRecord]) -> None:
    refs_by_path: dict[str, list[str]] = {}
    for artifact in artifacts:
        refs_by_path.setdefault(_path_key(artifact.path), []).append(
            artifact.artifact_ref_id
        )
    for artifact in artifacts:
        artifact.parent_artifact_ref_ids = _dedupe(
            ref
            for parent_path in artifact.parent_paths
            for ref in refs_by_path.get(_path_key(parent_path), [])
            if ref != artifact.artifact_ref_id
        )


def _required_artifacts(
    steps: list[AudioEditStep],
    artifacts: list[ArtifactAuditRecord],
    acceptance_policies: ArtifactAcceptancePolicyRegistry,
) -> tuple[list[str], list[str]]:
    required_refs: list[str] = []
    unresolved: list[str] = []
    by_step: dict[str, list[ArtifactAuditRecord]] = {}
    for artifact in artifacts:
        by_step.setdefault(artifact.producer_step_id, []).append(artifact)
    for step in steps:
        if not step.required:
            continue
        if step.status in {"failed", "skipped", "pending", "running"}:
            unresolved.append(step.id)
            continue
        step_artifacts = by_step.get(step.id, [])
        if not step_artifacts:
            if acceptance_policies.policy_for(step.operation) is not None:
                unresolved.append(step.id)
            continue
        primary = [item for item in step_artifacts if item.primary_output]
        considered = primary or step_artifacts
        accepted = [item for item in considered if item.disposition == "accepted"]
        if accepted:
            required_refs.extend(item.artifact_ref_id for item in accepted)
        else:
            unresolved.append(step.id)
    return _dedupe(required_refs), _dedupe(unresolved)


def _finalization_findings(
    *,
    plan: AudioEditPlan,
    artifacts: list[ArtifactAuditRecord],
    legacy_final_path: str | None,
    legacy_final: ArtifactAuditRecord | None,
    unresolved_required_steps: list[str],
) -> list[ArtifactFinalizationFinding]:
    findings: list[ArtifactFinalizationFinding] = []
    failed_required = [
        step.id
        for step in plan.steps
        if step.required and step.status in {"failed", "skipped", "pending", "running"}
    ]
    if failed_required:
        findings.append(
            ArtifactFinalizationFinding(
                code="required_steps_incomplete",
                severity="blocking",
                plain_summary="Required steps are incomplete and cannot support finalization.",
                details={"step_ids": failed_required},
            )
        )
    if not legacy_final_path:
        findings.append(
            ArtifactFinalizationFinding(
                code="legacy_final_path_missing",
                severity="blocking",
                plain_summary="Legacy execution did not select a final output path.",
            )
        )
    elif legacy_final is None:
        findings.append(
            ArtifactFinalizationFinding(
                code="legacy_final_not_observed",
                severity="warning",
                plain_summary="Legacy final path is not represented by an observed step artifact.",
                details={"legacy_final_path": legacy_final_path},
            )
        )
    elif legacy_final.disposition == "rejected":
        findings.append(
            ArtifactFinalizationFinding(
                code="legacy_final_rejected",
                severity="blocking",
                plain_summary="Legacy final path points to an artifact rejected by audit checks.",
                artifact_ref_id=legacy_final.artifact_ref_id,
                step_id=legacy_final.producer_step_id,
            )
        )
    elif legacy_final.disposition == "candidate":
        findings.append(
            ArtifactFinalizationFinding(
                code="legacy_final_unverified",
                severity="warning",
                plain_summary="Legacy final path exists but lacks runtime-verifier acceptance evidence.",
                artifact_ref_id=legacy_final.artifact_ref_id,
                step_id=legacy_final.producer_step_id,
            )
        )
    if unresolved_required_steps:
        findings.append(
            ArtifactFinalizationFinding(
                code="required_artifacts_unaccepted",
                severity="warning",
                plain_summary="Required output-producing steps lack an accepted artifact.",
                details={"step_ids": unresolved_required_steps},
            )
        )
    if not artifacts:
        findings.append(
            ArtifactFinalizationFinding(
                code="no_step_artifacts_observed",
                severity="warning",
                plain_summary="No step-produced audio artifacts were available for finalization audit.",
            )
        )
    return findings


def _finalization_status(
    *,
    legacy_final: ArtifactAuditRecord | None,
    findings: list[ArtifactFinalizationFinding],
    unresolved_required_steps: list[str],
) -> str:
    if any(finding.severity == "blocking" for finding in findings):
        return "blocked"
    if (
        legacy_final is not None
        and legacy_final.disposition == "accepted"
        and not unresolved_required_steps
    ):
        return "would_finalize"
    return "insufficient_evidence"


def _artifact_ref_id(*, attempt_id: str, step_id: str, path: str) -> str:
    digest = hashlib.sha256(
        f"{attempt_id}\0{step_id}\0{_path_key(path)}".encode()
    ).hexdigest()[:20]
    return f"artifact_audit_{digest}"


def _is_primary_result_key(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    leaf = value.rsplit(".", 1)[-1]
    return leaf in _PRIMARY_OUTPUT_KEYS


def _same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return _path_key(left) == _path_key(right)


def _path_key(value: Any) -> str:
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(value)


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if item not in (None, "")]


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
