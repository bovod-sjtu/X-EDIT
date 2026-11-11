"""Deterministic, non-authoritative scheduling for sparse checkpoint evaluation."""

from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import Field, model_validator

from x_editor.runtime.module_checkpoints import (
    CheckpointEvaluationRole,
    CheckpointModule,
    ModuleCheckpoint,
)

from .schemas import StrictModel, VersionedRef


class CheckpointRubricAssignment(StrictModel):
    """Pre-registered evaluator-side rubric assignment for one case boundary."""

    schema_version: Literal["x-editor.checkpoint-rubric-assignment/v1"] = (
        "x-editor.checkpoint-rubric-assignment/v1"
    )
    assignment_id: str
    runtime_case_fingerprint: str
    module: CheckpointModule
    boundary_kinds: list[str] = Field(min_length=1)
    evaluation_roles: list[CheckpointEvaluationRole] = Field(
        default_factory=lambda: ["candidate_output"]
    )
    objective_refs: list[str] = Field(min_length=1)
    rubric_set_ref: VersionedRef
    evaluator_policy_ref: VersionedRef
    registered_before_candidate_output: Literal[True] = True

    @model_validator(mode="after")
    def validate_unique_fields(self) -> CheckpointRubricAssignment:
        if len(self.boundary_kinds) != len(set(self.boundary_kinds)):
            raise ValueError("rubric boundary kinds must be unique")
        if len(self.evaluation_roles) != len(set(self.evaluation_roles)):
            raise ValueError("rubric evaluation roles must be unique")
        if len(self.objective_refs) != len(set(self.objective_refs)):
            raise ValueError("rubric objective refs must be unique")
        return self


class SparseEvaluationPolicy(StrictModel):
    """Frozen density and eligibility policy for one evaluation schedule."""

    schema_version: Literal["x-editor.sparse-evaluation-policy/v1"] = (
        "x-editor.sparse-evaluation-policy/v1"
    )
    policy_id: str
    sampling_seed: str
    sampling_fraction: float = Field(gt=0, le=1)
    eligible_modules: list[CheckpointModule] = Field(min_length=1)
    include_failure_diagnostics: bool = False
    max_jobs: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_modules(self) -> SparseEvaluationPolicy:
        if len(self.eligible_modules) != len(set(self.eligible_modules)):
            raise ValueError("eligible evaluation modules must be unique")
        return self


class CheckpointEvaluationJob(StrictModel):
    """One evaluator-side job that cannot mutate a runtime decision."""

    schema_version: Literal["x-editor.checkpoint-evaluation-job/v1"] = (
        "x-editor.checkpoint-evaluation-job/v1"
    )
    job_id: str
    policy_id: str
    assignment_id: str
    checkpoint_id: str
    checkpoint_output_fingerprint: str
    runtime_case_fingerprint: str
    assigned_arm: Literal["W0", "W1", "W2"]
    module: CheckpointModule
    boundary_kind: str
    evaluation_role: CheckpointEvaluationRole
    objective_refs: list[str]
    rubric_set_ref: VersionedRef
    evaluator_policy_ref: VersionedRef
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source_event_id: str
    runtime_control_authority: Literal[False] = False


class EvaluationScheduleSkip(StrictModel):
    checkpoint_id: str
    reason_code: Literal[
        "module_not_eligible",
        "failure_diagnostic_not_enabled",
        "rubric_assignment_missing",
        "deterministic_sample_excluded",
        "max_jobs_reached",
    ]


class SparseEvaluationSchedule(StrictModel):
    """Immutable schedule output; it contains no evaluator results or repairs."""

    schema_version: Literal["x-editor.sparse-evaluation-schedule/v1"] = (
        "x-editor.sparse-evaluation-schedule/v1"
    )
    policy: SparseEvaluationPolicy
    checkpoint_count: int = Field(ge=0)
    jobs: list[CheckpointEvaluationJob] = Field(default_factory=list)
    skipped: list[EvaluationScheduleSkip] = Field(default_factory=list)


def schedule_sparse_evaluations(
    checkpoints: Sequence[ModuleCheckpoint],
    assignments: Sequence[CheckpointRubricAssignment],
    *,
    policy: SparseEvaluationPolicy,
) -> SparseEvaluationSchedule:
    """Join frozen rubrics to checkpoints and sample without reading output quality."""

    assignment_index: dict[
        tuple[str, CheckpointModule, str, CheckpointEvaluationRole],
        CheckpointRubricAssignment,
    ] = {}
    for assignment in assignments:
        for boundary_kind in assignment.boundary_kinds:
            for evaluation_role in assignment.evaluation_roles:
                key = (
                    assignment.runtime_case_fingerprint,
                    assignment.module,
                    boundary_kind,
                    evaluation_role,
                )
                if key in assignment_index:
                    raise ValueError(
                        "multiple rubric assignments match the same checkpoint boundary"
                    )
                assignment_index[key] = assignment

    jobs: list[CheckpointEvaluationJob] = []
    skipped: list[EvaluationScheduleSkip] = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.checkpoint_id):
        if checkpoint.module not in policy.eligible_modules:
            skipped.append(
                EvaluationScheduleSkip(
                    checkpoint_id=checkpoint.checkpoint_id,
                    reason_code="module_not_eligible",
                )
            )
            continue
        if (
            checkpoint.evaluation_role == "failure_diagnostic"
            and not policy.include_failure_diagnostics
        ):
            skipped.append(
                EvaluationScheduleSkip(
                    checkpoint_id=checkpoint.checkpoint_id,
                    reason_code="failure_diagnostic_not_enabled",
                )
            )
            continue
        assignment = assignment_index.get(
            (
                checkpoint.context.runtime_case_fingerprint,
                checkpoint.module,
                checkpoint.boundary_kind,
                checkpoint.evaluation_role,
            )
        )
        if assignment is None:
            skipped.append(
                EvaluationScheduleSkip(
                    checkpoint_id=checkpoint.checkpoint_id,
                    reason_code="rubric_assignment_missing",
                )
            )
            continue
        if not _selected(checkpoint.checkpoint_id, policy=policy):
            skipped.append(
                EvaluationScheduleSkip(
                    checkpoint_id=checkpoint.checkpoint_id,
                    reason_code="deterministic_sample_excluded",
                )
            )
            continue
        if policy.max_jobs is not None and len(jobs) >= policy.max_jobs:
            skipped.append(
                EvaluationScheduleSkip(
                    checkpoint_id=checkpoint.checkpoint_id,
                    reason_code="max_jobs_reached",
                )
            )
            continue
        job_id = _digest(
            policy.policy_id,
            assignment.assignment_id,
            checkpoint.checkpoint_id,
            checkpoint.output_fingerprint,
        )
        jobs.append(
            CheckpointEvaluationJob(
                job_id=job_id,
                policy_id=policy.policy_id,
                assignment_id=assignment.assignment_id,
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_output_fingerprint=checkpoint.output_fingerprint,
                runtime_case_fingerprint=(
                    checkpoint.context.runtime_case_fingerprint
                ),
                assigned_arm=checkpoint.context.assigned_arm,
                module=checkpoint.module,
                boundary_kind=checkpoint.boundary_kind,
                evaluation_role=checkpoint.evaluation_role,
                objective_refs=list(assignment.objective_refs),
                rubric_set_ref=assignment.rubric_set_ref,
                evaluator_policy_ref=assignment.evaluator_policy_ref,
                artifact_refs=list(checkpoint.artifact_refs),
                evidence_refs=list(checkpoint.evidence_refs),
                source_event_id=checkpoint.source_event.event_id,
            )
        )
    return SparseEvaluationSchedule(
        policy=policy,
        checkpoint_count=len(checkpoints),
        jobs=jobs,
        skipped=skipped,
    )


def _selected(checkpoint_id: str, *, policy: SparseEvaluationPolicy) -> bool:
    digest = hashlib.sha256(
        f"{policy.sampling_seed}:{checkpoint_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < policy.sampling_fraction


def _digest(*values: str) -> str:
    payload = "\x1f".join(values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CheckpointEvaluationJob",
    "CheckpointRubricAssignment",
    "EvaluationScheduleSkip",
    "SparseEvaluationPolicy",
    "SparseEvaluationSchedule",
    "schedule_sparse_evaluations",
]
