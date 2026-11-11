"""Audit accepted upstream artifacts and conservative suffix-resume boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import soundfile as sf

from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.workflow_control.artifact_acceptance import (
    ArtifactAcceptancePolicyRegistry,
    build_default_artifact_acceptance_policy_registry,
)
from x_editor.workflow_control.schemas import (
    ArtifactFinalizationAudit,
    ReusableArtifactCheckpoint,
    SuffixResumeCandidateAssessment,
    SuffixResumeCheckpoint,
    SuffixResumeExecutionAdmission,
    SuffixResumeFinding,
    SuffixResumeReuseReceipt,
    SourceArtifactCheckpoint,
    WorkflowFailureRegistry,
)


_INVALIDATION_GUARDS = [
    "source_contract_changed",
    "reusable_step_changed",
    "dependency_graph_changed",
    "artifact_no_longer_accepted",
    "failed_step_unlocalized",
]


class SuffixResumeAuditor:
    """Find reusable accepted ancestors without changing execution behavior."""

    def audit(
        self,
        plan: AudioEditPlan,
        artifact_audit: ArtifactFinalizationAudit,
        failure_registry: WorkflowFailureRegistry,
    ) -> SuffixResumeCheckpoint:
        step_ids = [step.id for step in plan.steps]
        step_by_id = {step.id: step for step in plan.steps}
        step_fingerprints = {
            step.id: _fingerprint(_step_contract(step)) for step in plan.steps
        }
        base = {
            "attempt_id": artifact_audit.attempt_id,
            "plan_fingerprint": _plan_fingerprint(plan),
            "source_contract_fingerprint": _source_contract_fingerprint(plan),
            "step_fingerprints": step_fingerprints,
            "invalidation_guard_codes": list(_INVALIDATION_GUARDS),
            "legacy_action_observed": _last_repair_action(failure_registry),
        }
        if failure_registry.status == "clear":
            return SuffixResumeCheckpoint(
                status="not_applicable",
                findings=[
                    SuffixResumeFinding(
                        code="run_completed_without_failure",
                        severity="info",
                        plain_summary="The run completed, so no suffix resume is needed.",
                    )
                ],
                **base,
            )

        dependencies = _step_dependencies(plan, artifact_audit)
        base["step_dependencies"] = {
            step_id: _ordered(parents, step_ids)
            for step_id, parents in dependencies.items()
        }
        cycle_nodes = _cycle_nodes(dependencies)
        unknown_dependency_targets = _unknown_dependency_targets(plan)
        failure_candidates = _dedupe(
            [
                *failure_registry.terminal_failed_step_ids,
                *artifact_audit.unresolved_required_steps,
                *_steps_for_artifacts(
                    failure_registry.terminal_failed_artifact_refs,
                    artifact_audit,
                ),
            ]
        )
        unknown_failures = [item for item in failure_candidates if item not in step_by_id]
        failed_steps = [item for item in failure_candidates if item in step_by_id]
        findings: list[SuffixResumeFinding] = []
        if unknown_failures:
            findings.append(
                SuffixResumeFinding(
                    code="failed_step_unknown",
                    severity="blocking",
                    plain_summary="Failure references do not resolve to plan steps.",
                    details={"unknown_step_ids": unknown_failures},
                )
            )
        if cycle_nodes:
            findings.append(
                SuffixResumeFinding(
                    code="dependency_cycle_detected",
                    severity="blocking",
                    plain_summary="The step dependency graph contains a cycle.",
                    details={"step_ids": _ordered(cycle_nodes, step_ids)},
                )
            )
        if unknown_dependency_targets:
            findings.append(
                SuffixResumeFinding(
                    code="dependency_target_unknown",
                    severity="blocking",
                    plain_summary="A step dependency references a step outside the plan.",
                    details={"dependencies": unknown_dependency_targets},
                )
            )
        if (
            not failed_steps
            or unknown_failures
            or cycle_nodes
            or unknown_dependency_targets
        ):
            if not failed_steps:
                findings.append(
                    SuffixResumeFinding(
                        code="failed_step_unlocalized",
                        severity="blocking",
                        plain_summary="No concrete failed plan step could be localized.",
                    )
                )
            return SuffixResumeCheckpoint(
                status="full_replay_required",
                failed_step_ids=_ordered(failed_steps, step_ids),
                restart_step_ids=_ordered(failed_steps, step_ids),
                rerun_step_ids=step_ids,
                findings=findings,
                **base,
            )

        failed_set = set(failed_steps)
        restart_steps = [
            step_id
            for step_id in failed_steps
            if not (_ancestors(step_id, dependencies) & failed_set)
        ]
        rerun = set(restart_steps)
        for step_id in restart_steps:
            rerun.update(_descendants(step_id, dependencies))
        upstream = set()
        for step_id in rerun:
            upstream.update(_ancestors(step_id, dependencies))
        reusable = []
        for artifact in artifact_audit.artifacts:
            if not (
                artifact.disposition == "accepted"
                and artifact.producer_step_id in upstream
                and artifact.producer_step_id not in rerun
            ):
                continue
            size_bytes, content_sha256 = _file_identity(artifact.path)
            reusable.append(
                ReusableArtifactCheckpoint(
                    artifact_ref_id=artifact.artifact_ref_id,
                    producer_step_id=artifact.producer_step_id,
                    producer_result_sha256=_fingerprint(step_by_id[artifact.producer_step_id].result),
                    path=artifact.path,
                    role=artifact.role,
                    kind=artifact.kind,
                    primary_output=artifact.primary_output,
                    result_key=_canonical_resume_result_key(
                        artifact.metadata.get("result_key")
                    ),
                    verification_status=artifact.verification_status,
                    acceptance_policy_id=artifact.acceptance_policy_id,
                    required_check_ids=artifact.required_check_ids,
                    satisfied_check_ids=artifact.satisfied_check_ids,
                    verification_refs=artifact.verification_refs,
                    parent_paths=artifact.parent_paths,
                    parent_artifact_ref_ids=artifact.parent_artifact_ref_ids,
                    size_bytes=size_bytes,
                    content_sha256=content_sha256,
                    original_tool_invocation_count=_step_tool_invocation_count(
                        plan, artifact.producer_step_id
                    ),
                )
            )
        reusable.sort(
            key=lambda item: (
                step_ids.index(item.producer_step_id),
                item.artifact_ref_id,
            )
        )
        reusable_steps = _ordered(
            {item.producer_step_id for item in reusable}, step_ids
        )
        ordered_failed = _ordered(failed_set, step_ids)
        ordered_restart = _ordered(restart_steps, step_ids)
        ordered_rerun = _ordered(rerun, step_ids)
        findings.append(
            SuffixResumeFinding(
                code="failed_suffix_localized",
                severity="info",
                plain_summary="Failed steps and their dependent suffix were localized.",
                details={
                    "failed_step_ids": ordered_failed,
                    "restart_step_ids": ordered_restart,
                    "rerun_step_ids": ordered_rerun,
                },
            )
        )
        if reusable:
            findings.append(
                SuffixResumeFinding(
                    code="accepted_upstream_available",
                    severity="info",
                    plain_summary="Accepted upstream artifacts are available for shadow reuse.",
                    details={
                        "step_ids": reusable_steps,
                        "artifact_ref_ids": [item.artifact_ref_id for item in reusable],
                    },
                )
            )
            status = "eligible"
        else:
            findings.append(
                SuffixResumeFinding(
                    code="accepted_upstream_unavailable",
                    severity="warning",
                    plain_summary=(
                        "No accepted upstream artifact has proven lineage "
                        "to the failed suffix."
                    ),
                )
            )
            status = "full_replay_required"
        return SuffixResumeCheckpoint(
            status=status,
            failed_step_ids=ordered_failed,
            restart_step_ids=ordered_restart,
            rerun_step_ids=ordered_rerun,
            reusable_step_ids=reusable_steps,
            reusable_artifacts=reusable,
            source_artifacts=_source_artifact_checkpoints(
                plan, artifact_audit, reusable
            ),
            findings=findings,
            **base,
        )

    def assess_candidate(
        self,
        checkpoint: SuffixResumeCheckpoint,
        original_plan: AudioEditPlan,
        candidate_plan: AudioEditPlan,
    ) -> SuffixResumeCandidateAssessment:
        candidate_fingerprint = _plan_fingerprint(candidate_plan)
        if checkpoint.status != "eligible":
            return SuffixResumeCandidateAssessment(
                checkpoint_id=checkpoint.checkpoint_id,
                status="full_replay_required",
                candidate_plan_fingerprint=candidate_fingerprint,
                invalidated_step_ids=[step.id for step in candidate_plan.steps],
                rerun_step_ids=[step.id for step in candidate_plan.steps],
                reason_codes=["checkpoint_not_eligible"],
                findings=[
                    SuffixResumeFinding(
                        code="checkpoint_not_eligible",
                        severity="blocking",
                        plain_summary="The original failure has no reusable accepted prefix.",
                    )
                ],
            )

        original_ids = [step.id for step in original_plan.steps]
        candidate_ids = [step.id for step in candidate_plan.steps]
        original_steps = {step.id: step for step in original_plan.steps}
        candidate_steps = {step.id: step for step in candidate_plan.steps}
        findings: list[SuffixResumeFinding] = []
        reasons: list[str] = []
        fatal = False
        invalidated: set[str] = set()

        if _source_contract_fingerprint(original_plan) != _source_contract_fingerprint(
            candidate_plan
        ):
            fatal = True
            reasons.append("source_contract_changed")
            findings.append(
                SuffixResumeFinding(
                    code="source_contract_changed",
                    severity="blocking",
                    plain_summary="Source audio or global execution parameters changed.",
                )
            )
        if original_ids != candidate_ids:
            fatal = True
            reasons.append("plan_step_set_changed")
            findings.append(
                SuffixResumeFinding(
                    code="plan_step_set_changed",
                    severity="blocking",
                    plain_summary="Candidate plan step ids or ordering changed.",
                    details={
                        "original_step_ids": original_ids,
                        "candidate_step_ids": candidate_ids,
                    },
                )
            )

        if not fatal:
            for step_id in checkpoint.rerun_step_ids:
                original = original_steps[step_id]
                candidate = candidate_steps[step_id]
                if (
                    original.operation != candidate.operation
                    or original.inputs != candidate.inputs
                    or original.depends_on != candidate.depends_on
                ):
                    fatal = True
                    reasons.append("dependency_graph_changed")
                    findings.append(
                        SuffixResumeFinding(
                            code="dependency_graph_changed",
                            severity="blocking",
                            plain_summary=(
                                "A suffix step changed operation, inputs, "
                                "or dependencies."
                            ),
                            step_id=step_id,
                        )
                    )
            for step_id in checkpoint.reusable_step_ids:
                candidate_step = candidate_steps[step_id]
                if checkpoint.step_fingerprints[step_id] != _fingerprint(
                    _step_contract(candidate_step)
                ):
                    invalidated.add(step_id)
                    reasons.append("reusable_step_changed")
                    findings.append(
                        SuffixResumeFinding(
                            code="reusable_step_changed",
                            severity="warning",
                            plain_summary="An accepted upstream step changed and cannot be reused.",
                            step_id=step_id,
                        )
                    )

        if fatal:
            return SuffixResumeCandidateAssessment(
                checkpoint_id=checkpoint.checkpoint_id,
                status="full_replay_required",
                candidate_plan_fingerprint=candidate_fingerprint,
                invalidated_step_ids=candidate_ids,
                rerun_step_ids=candidate_ids,
                reason_codes=_dedupe(reasons),
                findings=findings,
            )

        dependencies = {
            step_id: set(parents)
            for step_id, parents in checkpoint.step_dependencies.items()
        }
        expanded_rerun = set(checkpoint.rerun_step_ids)
        for step_id in invalidated:
            expanded_rerun.add(step_id)
            expanded_rerun.update(_descendants(step_id, dependencies))
        retained = [
            item.artifact_ref_id
            for item in checkpoint.reusable_artifacts
            if item.producer_step_id not in expanded_rerun
        ]
        if not retained:
            reasons.append("accepted_prefix_fully_invalidated")
            findings.append(
                SuffixResumeFinding(
                    code="accepted_prefix_fully_invalidated",
                    severity="blocking",
                    plain_summary="Candidate changes invalidate every reusable upstream artifact.",
                )
            )
            return SuffixResumeCandidateAssessment(
                checkpoint_id=checkpoint.checkpoint_id,
                status="full_replay_required",
                candidate_plan_fingerprint=candidate_fingerprint,
                invalidated_step_ids=_ordered(invalidated, original_ids),
                rerun_step_ids=original_ids,
                reason_codes=_dedupe(reasons),
                findings=findings,
            )
        findings.append(
            SuffixResumeFinding(
                code="candidate_prefix_reuse_admissible",
                severity="info",
                plain_summary="Candidate changes preserve at least one accepted upstream artifact.",
                details={"retained_artifact_refs": retained},
            )
        )
        return SuffixResumeCandidateAssessment(
            checkpoint_id=checkpoint.checkpoint_id,
            status="admissible",
            candidate_plan_fingerprint=candidate_fingerprint,
            retained_reusable_artifact_refs=retained,
            invalidated_step_ids=_ordered(invalidated, original_ids),
            rerun_step_ids=_ordered(expanded_rerun, original_ids),
            reason_codes=_dedupe(reasons),
            findings=findings,
        )


class SuffixResumeAdmissionGate:
    """Revalidate a shadow checkpoint before it may skip actual execution."""

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
        checkpoint: SuffixResumeCheckpoint,
        original_plan: AudioEditPlan,
        candidate_plan: AudioEditPlan,
    ) -> SuffixResumeExecutionAdmission:
        candidate_fingerprint = _plan_fingerprint(candidate_plan)
        auditor = SuffixResumeAuditor()
        assessment = auditor.assess_candidate(
            checkpoint, original_plan, candidate_plan
        )
        if assessment.status != "admissible":
            return _full_replay_admission(
                candidate_plan,
                checkpoint=checkpoint,
                assessment_id=assessment.assessment_id,
                reason_codes=assessment.reason_codes,
                findings=assessment.findings,
            )

        findings: list[SuffixResumeFinding] = []
        reasons: list[str] = []

        def reject(
            code: str,
            summary: str,
            *,
            step_id: str | None = None,
            artifact_ref_id: str | None = None,
            details: dict[str, Any] | None = None,
        ) -> None:
            reasons.append(code)
            findings.append(
                SuffixResumeFinding(
                    code=code,
                    severity="blocking",
                    plain_summary=summary,
                    step_id=step_id,
                    artifact_ref_id=artifact_ref_id,
                    details=details or {},
                )
            )

        if checkpoint.schema_version != "x-editor.suffix-resume-checkpoint/v2":
            reject(
                "checkpoint_identity_evidence_missing",
                "The checkpoint predates content-identity enforcement.",
            )
        if checkpoint.plan_fingerprint != _plan_fingerprint(original_plan):
            reject(
                "checkpoint_original_plan_mismatch",
                "The supplied original plan does not match the saved checkpoint.",
            )
        if (
            checkpoint.source_contract_fingerprint
            != _source_contract_fingerprint(original_plan)
        ):
            reject(
                "checkpoint_source_contract_mismatch",
                "The original source contract does not match the saved checkpoint.",
            )

        retained_refs = set(assessment.retained_reusable_artifact_refs)
        retained = [
            item
            for item in checkpoint.reusable_artifacts
            if item.artifact_ref_id in retained_refs
        ]
        retained_steps = _ordered(
            {item.producer_step_id for item in retained},
            [step.id for step in candidate_plan.steps],
        )
        primary_by_step: dict[str, list[ReusableArtifactCheckpoint]] = {
            step_id: [] for step_id in retained_steps
        }
        for artifact in retained:
            if artifact.primary_output:
                primary_by_step.setdefault(artifact.producer_step_id, []).append(
                    artifact
                )
        for step_id in retained_steps:
            count = len(primary_by_step.get(step_id, []))
            if count != 1:
                reject(
                    "reusable_primary_artifact_ambiguous",
                    "A reused step must have exactly one accepted primary output.",
                    step_id=step_id,
                    details={"primary_artifact_count": count},
                )

        retained_sources = [
            source
            for source in checkpoint.source_artifacts
            if not source.consumer_artifact_ref_ids
            or retained_refs.intersection(source.consumer_artifact_ref_ids)
        ]
        for source in retained_sources:
            identity = _revalidated_audio_identity(source.path)
            if source.size_bytes is None or not source.content_sha256:
                reject(
                    "source_identity_evidence_missing",
                    "A source file lacks a frozen content identity.",
                    details={"source_ref": source.source_ref, "path": source.path},
                )
            elif identity != (source.size_bytes, source.content_sha256):
                reject(
                    "source_artifact_changed",
                    "A source file changed or is no longer readable.",
                    details={"source_ref": source.source_ref, "path": source.path},
                )

        candidate_steps = {step.id: step for step in candidate_plan.steps}
        receipts: list[SuffixResumeReuseReceipt] = []
        for step_id in retained_steps:
            artifacts = primary_by_step.get(step_id, [])
            if len(artifacts) != 1:
                continue
            artifact = artifacts[0]
            step = candidate_steps[step_id]
            saved_result = next(s.result for s in original_plan.steps if s.id == step_id)
            if artifact.producer_result_sha256 and _fingerprint(saved_result) != artifact.producer_result_sha256:
                reject("saved_step_result_changed", "Retained timeline or text state changed after checkpointing.", step_id=step_id)
                continue
            policy = self.acceptance_policies.policy_for(step.operation)
            if _canonical_resume_result_key(artifact.result_key) != "output_audio_path":
                reject(
                    "reusable_result_binding_unsupported",
                    "Only an unambiguous output_audio_path binding can be reused.",
                    step_id=step_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                    details={"result_key": artifact.result_key},
                )
                continue
            if artifact.size_bytes is None or not artifact.content_sha256:
                reject(
                    "reusable_artifact_identity_evidence_missing",
                    "A reusable artifact lacks a frozen content identity.",
                    step_id=step_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                )
                continue
            if _revalidated_audio_identity(artifact.path) != (
                artifact.size_bytes,
                artifact.content_sha256,
            ):
                reject(
                    "reusable_artifact_changed",
                    "A reusable artifact changed or is no longer readable.",
                    step_id=step_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                )
                continue
            if policy is None:
                reject(
                    "current_acceptance_policy_missing",
                    "The reused operation no longer has an acceptance policy.",
                    step_id=step_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                )
                continue
            if (
                artifact.acceptance_policy_id != policy.policy_id
                or artifact.required_check_ids != list(policy.required_check_ids)
            ):
                reject(
                    "acceptance_policy_changed",
                    "The current artifact acceptance policy differs from the saved policy.",
                    step_id=step_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                )
                continue
            if sorted(artifact.satisfied_check_ids) != sorted(
                artifact.required_check_ids
            ):
                reject(
                    "saved_acceptance_incomplete",
                    "The saved artifact did not satisfy every required framework check.",
                    step_id=step_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                )
                continue
            report = _reuse_verification_report(
                checkpoint=checkpoint,
                artifact=artifact,
                operation=step.operation,
            )
            receipts.append(
                SuffixResumeReuseReceipt(
                    checkpoint_id=checkpoint.checkpoint_id,
                    artifact_ref_id=artifact.artifact_ref_id,
                    producer_step_id=step_id,
                    operation=step.operation,
                    path=artifact.path,
                    size_bytes=artifact.size_bytes,
                    content_sha256=artifact.content_sha256,
                    acceptance_policy_id=policy.policy_id,
                    required_check_ids=list(policy.required_check_ids),
                    source_refs=[
                        item.source_ref
                        for item in retained_sources
                        if not item.consumer_artifact_ref_ids
                        or artifact.artifact_ref_id
                        in item.consumer_artifact_ref_ids
                    ],
                    parent_paths=artifact.parent_paths,
                    skipped_tool_call_count=artifact.original_tool_invocation_count,
                    retained_output=(saved_result if artifact.producer_result_sha256 and isinstance(saved_result, dict) else {}),
                    verification_report=report,
                )
            )

        if reasons or len(receipts) != len(retained_steps):
            return _full_replay_admission(
                candidate_plan,
                checkpoint=checkpoint,
                assessment_id=assessment.assessment_id,
                reason_codes=_dedupe(reasons or ["reuse_receipt_incomplete"]),
                findings=findings,
            )
        findings.append(
            SuffixResumeFinding(
                code="suffix_resume_execution_admitted",
                severity="info",
                plain_summary=(
                    "Every retained artifact, source, step contract, dependency, "
                    "and acceptance policy was revalidated."
                ),
                details={"reused_step_ids": retained_steps},
            )
        )
        return SuffixResumeExecutionAdmission(
            checkpoint_id=checkpoint.checkpoint_id,
            assessment_id=assessment.assessment_id,
            status="admitted",
            candidate_plan_fingerprint=candidate_fingerprint,
            reused_step_ids=retained_steps,
            rerun_step_ids=assessment.rerun_step_ids,
            receipts=receipts,
            findings=findings,
            skipped_tool_call_count=sum(
                item.skipped_tool_call_count for item in receipts
            ),
            behavior_changed=True,
        )


def _step_dependencies(
    plan: AudioEditPlan,
    artifact_audit: ArtifactFinalizationAudit | None,
) -> dict[str, set[str]]:
    step_ids = {step.id for step in plan.steps}
    dependencies = {
        step.id: {item for item in step.depends_on if item in step_ids}
        for step in plan.steps
    }
    if artifact_audit is None:
        return dependencies
    artifact_steps = {
        item.artifact_ref_id: item.producer_step_id for item in artifact_audit.artifacts
    }
    path_steps = {
        _path_key(item.path): item.producer_step_id
        for item in artifact_audit.artifacts
        if item.path
    }
    for artifact in artifact_audit.artifacts:
        parents = {
            artifact_steps[item]
            for item in artifact.parent_artifact_ref_ids
            if item in artifact_steps
        }
        parents.update(
            path_steps[_path_key(item)]
            for item in artifact.parent_paths
            if _path_key(item) in path_steps
        )
        dependencies.setdefault(artifact.producer_step_id, set()).update(
            item for item in parents if item != artifact.producer_step_id
        )
    return dependencies


def _full_replay_admission(
    candidate_plan: AudioEditPlan,
    *,
    checkpoint: SuffixResumeCheckpoint | None,
    checkpoint_id: str | None = None,
    assessment_id: str | None = None,
    reason_codes: list[str] | None = None,
    findings: list[SuffixResumeFinding] | None = None,
) -> SuffixResumeExecutionAdmission:
    return SuffixResumeExecutionAdmission(
        checkpoint_id=(
            checkpoint.checkpoint_id if checkpoint is not None else checkpoint_id
        ),
        assessment_id=assessment_id,
        status="full_replay_required",
        candidate_plan_fingerprint=_plan_fingerprint(candidate_plan),
        rerun_step_ids=[step.id for step in candidate_plan.steps],
        reason_codes=_dedupe(reason_codes or ["suffix_resume_not_admitted"]),
        findings=findings or [],
    )


def full_replay_admission(
    candidate_plan: AudioEditPlan,
    *,
    reason_code: str,
    plain_summary: str,
    checkpoint_id: str | None = None,
) -> SuffixResumeExecutionAdmission:
    """Create a typed fallback when an execution request cannot be parsed."""

    return _full_replay_admission(
        candidate_plan,
        checkpoint=None,
        checkpoint_id=checkpoint_id,
        reason_codes=[reason_code],
        findings=[
            SuffixResumeFinding(
                code=reason_code,
                severity="blocking",
                plain_summary=plain_summary,
            )
        ],
    )


def _source_artifact_checkpoints(
    plan: AudioEditPlan,
    artifact_audit: ArtifactFinalizationAudit,
    reusable: list[ReusableArtifactCheckpoint],
) -> list[SourceArtifactCheckpoint]:
    produced_paths = {
        _path_key(item.path) for item in artifact_audit.artifacts if item.path
    }
    for item in plan.metadata.get("audio_artifacts") or []:
        if isinstance(item, dict) and item.get("path"):
            produced_paths.add(_path_key(str(item["path"])))

    candidates: list[tuple[str, str, list[str]]] = []
    if plan.input_audio:
        candidates.append(
            (
                "plan.input_audio",
                plan.input_audio,
                [item.artifact_ref_id for item in reusable],
            )
        )
    for artifact in reusable:
        for index, path in enumerate(artifact.parent_paths):
            if _path_key(path) not in produced_paths:
                candidates.append(
                    (
                        f"artifact:{artifact.artifact_ref_id}:parent:{index}",
                        path,
                        [artifact.artifact_ref_id],
                    )
                )

    by_path: dict[str, SourceArtifactCheckpoint] = {}
    for source_ref, path, consumer_refs in candidates:
        key = _path_key(path)
        existing = by_path.get(key)
        if existing is not None:
            existing.consumer_artifact_ref_ids = _dedupe(
                [*existing.consumer_artifact_ref_ids, *consumer_refs]
            )
            continue
        size_bytes, content_sha256 = _file_identity(path)
        by_path[key] = SourceArtifactCheckpoint(
            source_ref=source_ref,
            path=path,
            consumer_artifact_ref_ids=consumer_refs,
            size_bytes=size_bytes,
            content_sha256=content_sha256,
        )
    return list(by_path.values())


def _step_tool_invocation_count(plan: AudioEditPlan, step_id: str) -> int:
    total = 0
    for item in plan.metadata.get("tool_invocations") or []:
        if not isinstance(item, dict) or str(item.get("step_id") or "") != step_id:
            continue
        attempts = item.get("attempts")
        if isinstance(attempts, list) and attempts:
            total += len(attempts)
        else:
            total += 1
    return total


def _file_identity(path_text: str) -> tuple[int | None, str | None]:
    path = Path(path_text).expanduser()
    try:
        if not path.is_file():
            return None, None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()
    except OSError:
        return None, None


def _revalidated_audio_identity(path_text: str) -> tuple[int, str] | None:
    size_bytes, content_sha256 = _file_identity(path_text)
    if size_bytes is None or content_sha256 is None:
        return None
    try:
        info = sf.info(str(Path(path_text).expanduser()))
    except (OSError, RuntimeError, ValueError):
        return None
    if int(info.frames) <= 0 or int(info.samplerate) <= 0:
        return None
    return size_bytes, content_sha256


def _reuse_verification_report(
    *,
    checkpoint: SuffixResumeCheckpoint,
    artifact: ReusableArtifactCheckpoint,
    operation: str,
) -> dict[str, Any]:
    return {
        "component": "framework.suffix_resume_reuse",
        "step_id": artifact.producer_step_id,
        "operation": operation,
        "passed": True,
        "score": 1.0,
        "confidence": "high",
        "severity": "info",
        "outcome": "passed",
        "scope": "artifact_integrity",
        "acceptance_grade": True,
        "check_ids": list(artifact.required_check_ids),
        "check_outcomes": {
            check_id: "passed" for check_id in artifact.required_check_ids
        },
        "subject_paths": [artifact.path],
        "metrics": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "artifact_ref_id": artifact.artifact_ref_id,
            "content_sha256": artifact.content_sha256,
            "size_bytes": artifact.size_bytes,
            "acceptance_policy_id": artifact.acceptance_policy_id,
            "revalidated_unchanged": True,
        },
        "findings": [
            "Artifact bytes, source identities, step contract, dependencies, "
            "and acceptance policy are unchanged from the accepted attempt."
        ],
        "repair_hints": [],
        "repair_actions": [],
        "evidence_paths": {},
    }


def _steps_for_artifacts(
    artifact_refs: list[str], artifact_audit: ArtifactFinalizationAudit
) -> list[str]:
    wanted = set(artifact_refs)
    return _dedupe(
        [
            artifact.producer_step_id
            for artifact in artifact_audit.artifacts
            if artifact.artifact_ref_id in wanted
        ]
    )


def _unknown_dependency_targets(plan: AudioEditPlan) -> list[dict[str, str]]:
    step_ids = {step.id for step in plan.steps}
    return [
        {"step_id": step.id, "depends_on": dependency}
        for step in plan.steps
        for dependency in step.depends_on
        if dependency not in step_ids
    ]


def _ancestors(step_id: str, dependencies: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    stack = list(dependencies.get(step_id, set()))
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(dependencies.get(current, set()))
    return result


def _descendants(step_id: str, dependencies: dict[str, set[str]]) -> set[str]:
    reverse: dict[str, set[str]] = {item: set() for item in dependencies}
    for child, parents in dependencies.items():
        for parent in parents:
            reverse.setdefault(parent, set()).add(child)
    result: set[str] = set()
    stack = list(reverse.get(step_id, set()))
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(reverse.get(current, set()))
    return result


def _cycle_nodes(dependencies: dict[str, set[str]]) -> set[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            cycles.add(step_id)
            return
        if step_id in visited:
            return
        visiting.add(step_id)
        for parent in dependencies.get(step_id, set()):
            if parent in visiting:
                cycles.update({step_id, parent})
            else:
                visit(parent)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in dependencies:
        visit(step_id)
    return cycles


def _plan_fingerprint(plan: AudioEditPlan) -> str:
    return _fingerprint(
        {
            "request": plan.request,
            "input_audio": plan.input_audio,
            "global_params": plan.global_params,
            "missing_inputs": plan.missing_inputs,
            "steps": [_step_contract(step) for step in plan.steps],
        }
    )


def _source_contract_fingerprint(plan: AudioEditPlan) -> str:
    return _fingerprint(
        {
            "input_audio": plan.input_audio,
            "global_params": plan.global_params,
            "missing_inputs": plan.missing_inputs,
        }
    )


def _step_contract(step: AudioEditStep) -> dict[str, Any]:
    return {
        "id": step.id,
        "operation": step.operation,
        "description": step.description,
        "inputs": step.inputs,
        "params": step.params,
        "expected_output": step.expected_output,
        "depends_on": step.depends_on,
        "required": step.required,
        "continue_on_error": step.continue_on_error,
    }


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_repair_action(registry: WorkflowFailureRegistry) -> str | None:
    return registry.repair_history[-1].action if registry.repair_history else None


def _ordered(values: Any, order: list[str]) -> list[str]:
    wanted = set(values)
    return [item for item in order if item in wanted]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_resume_result_key(value: Any) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    leaf = text.rsplit(".", 1)[-1]
    return "output_audio_path" if leaf == "output_audio_path" else text


def _path_key(value: str) -> str:
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        return str(Path(value).expanduser())
