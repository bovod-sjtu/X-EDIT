"""Provider-neutral contracts for the read-only Evaluation-Guidance path.

These models are intentionally outside the forward runtime. Evaluators observe
immutable checkpoints and artifacts, Guiders turn findings into questions for a
future attempt, and a separately configured controller may authorize that new
attempt. None of the contracts can mutate or gate the attempt being measured.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from x_editor.evaluation.delivery_review import (
    DeliveryCoverageReferenceBinding,
    DeliveryCoverageRequirement,
    DeliveryReviewContext,
    DeliveryReviewResult,
)

from x_editor.core.schemas import StrictModel
from x_editor.evaluation.fact_constraints import FactConstraint
from x_editor.evaluation.roles import EvaluatorRole

EvaluationFocus = Literal[
    "artifact_measurements",
    "perception_evidence",
    "reasoner_semantics",
    "planner_decision",
    "plan_graph",
    "tool_strategy",
    "intermediate_step",
    "final_delivery",
]
EvaluationMode = Literal[
    "off",
    "final_only",
    "sparse_steps",
    "all_steps",
    "failure_forensics",
]
EvaluationVisibility = Literal["blind_quality", "forensic"]
EvaluationUse = Literal[
    "calibration",
    "capacity_measurement",
    "realized_flow_diagnostic",
    "failure_forensics",
    "guided_attempt_diagnostic",
    "heldout_outcome",
]
EvidenceVisibility = Literal["judge", "forensic_only", "prompt_hidden"]
EvidenceState = Literal["available", "conflicting", "missing"]
EvidenceKind = Literal[
    "public_instruction",
    "session_state",
    "reviewed_perception",
    "reasoner_memo",
    "planner_decision",
    "plan_ir",
    "candidate_operation_card",
    "tool_parameter_schema",
    "effect_contract",
    "verifier_trace",
    "failure_trace",
    "other",
]
ArtifactRole = Literal["input", "output", "reference", "source", "intermediate"]
RubricApplicability = Literal["applicable", "not_applicable", "unknown"]
JudgeSlotStatus = Literal["completed", "provider_failure", "invalid_output"]
JudgeSamplingMode = Literal["deterministic", "stochastic", "unknown"]
JudgeIndependenceClaim = Literal["independent", "correlated", "unspecified"]
GuidanceTargetModule = Literal[
    "reasoner",
    "planner",
    "perception",
    "executor",
    "tool",
    "verifier",
    "workflow",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FORBIDDEN_JUDGE_VISIBLE_KEYS = frozenset(
    {
        "expected_answer",
        "oracle_answer",
        "right_choice",
        "scorer_output",
        "target_waveform",
    }
)


class FrozenEvaluationModel(StrictModel):
    """Immutable base for preregistered evaluation declarations."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_evaluation_fingerprint(value: Any) -> str:
    """Return a deterministic content hash for one JSON-compatible value."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvaluationSamplingPolicy(FrozenEvaluationModel):
    """Outcome-independent selection policy frozen before execution output."""

    schema_version: Literal["x-editor.evaluation-sampling-policy/v1"] = (
        "x-editor.evaluation-sampling-policy/v1"
    )
    policy_id: str = Field(min_length=1)
    mode: EvaluationMode = "off"
    sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: str = Field(default="x-editor-evaluation-v1", min_length=1)
    include_final_delivery: bool = True
    frozen_before_execution: Literal[True] = True
    outcome_conditioned_selection: Literal[False] = False

    @property
    def policy_fingerprint(self) -> str:
        return canonical_evaluation_fingerprint(self)


class FrozenEvaluationSelection(FrozenEvaluationModel):
    """One auditable selection decision for an evaluation unit."""

    schema_version: Literal["x-editor.frozen-evaluation-selection/v1"] = (
        "x-editor.frozen-evaluation-selection/v1"
    )
    selection_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    policy_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    unit_id: str = Field(min_length=1)
    focus: EvaluationFocus
    selected: bool
    selection_reason: str = Field(min_length=1)
    selected_before_output: bool
    outcome_independent: bool
    capacity_eligible: bool
    forensic_only: bool = False
    selection_hash: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_authority(self) -> FrozenEvaluationSelection:
        if self.capacity_eligible and not self.selected:
            raise ValueError("an unselected unit cannot be capacity eligible")
        if self.capacity_eligible and (
            not self.selected_before_output or not self.outcome_independent or self.forensic_only
        ):
            raise ValueError(
                "capacity selection must be pre-output, outcome-independent, and non-forensic"
            )
        if self.forensic_only and (
            self.selected_before_output or self.outcome_independent or self.capacity_eligible
        ):
            raise ValueError("forensic selection cannot be pre-output or capacity eligible")
        return self


class EvaluationEvidenceItem(FrozenEvaluationModel):
    """One immutable text or structured evidence item visible to an Evaluator."""

    evidence_id: str = Field(min_length=1)
    kind: EvidenceKind
    label: str = Field(min_length=1)
    payload: Any
    visibility: EvidenceVisibility = "judge"
    evidence_state: EvidenceState = "available"
    content_sha256: str = ""

    @model_validator(mode="after")
    def freeze_and_validate(self) -> EvaluationEvidenceItem:
        if self.visibility == "judge":
            forbidden = _find_forbidden_keys(self.payload)
            if forbidden:
                raise ValueError(
                    f"judge-visible evidence contains hidden outcome fields: {sorted(forbidden)}"
                )
        expected = canonical_evaluation_fingerprint(self.payload)
        if self.content_sha256 and self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match evidence payload")
        object.__setattr__(self, "content_sha256", expected)
        return self


class EvaluationInputSnapshot(FrozenEvaluationModel):
    """Content-addressed module input captured before its output exists."""

    schema_version: Literal["x-editor.evaluation-input-snapshot/v1"] = (
        "x-editor.evaluation-input-snapshot/v1"
    )
    snapshot_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    module: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    candidate_operation_card_refs: tuple[str, ...] = ()
    captured_before_module_output: Literal[True] = True
    contains_evaluator_feedback: Literal[False] = False
    contains_expected_answer: Literal[False] = False
    snapshot_fingerprint: str = ""

    @model_validator(mode="after")
    def freeze_snapshot(self) -> EvaluationInputSnapshot:
        for label, values in (
            ("evidence", self.evidence_refs),
            ("artifact", self.artifact_refs),
            ("candidate operation card", self.candidate_operation_card_refs),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} refs must be unique")
        expected = canonical_evaluation_fingerprint(
            self.model_dump(mode="json", exclude={"snapshot_fingerprint"})
        )
        if self.snapshot_fingerprint and self.snapshot_fingerprint != expected:
            raise ValueError("snapshot_fingerprint does not match snapshot content")
        object.__setattr__(self, "snapshot_fingerprint", expected)
        return self


class EvaluationArtifactRef(FrozenEvaluationModel):
    """Content-addressed audio attachment available only to the evaluation path."""

    artifact_ref_id: str = Field(min_length=1)
    role: ArtifactRole
    path: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    duration_sec: float | None = Field(default=None, gt=0.0)
    timeline_revision: int | None = Field(default=None, ge=0)
    source_node_id: str | None = None

    @field_validator("duration_sec")
    @classmethod
    def validate_duration(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("artifact duration must be finite")
        return value


class EvaluationAudioRegion(FrozenEvaluationModel):
    """Exact region named by one atomic audio rubric."""

    region_id: str = Field(min_length=1)
    artifact_ref_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    label: str = Field(min_length=1)
    contains_unedited_context: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> EvaluationAudioRegion:
        if not math.isfinite(self.start_sec) or not math.isfinite(self.end_sec):
            raise ValueError("audio region bounds must be finite")
        if self.end_sec <= self.start_sec:
            raise ValueError("audio region end_sec must exceed start_sec")
        return self


class MaterializedEvaluationAudioClip(FrozenEvaluationModel):
    """A bounded evaluator-only clip with exact target coordinates in the clip."""

    clip_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    source_artifact_ref_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    original_start_sec: float = Field(ge=0.0)
    original_end_sec: float = Field(gt=0.0)
    clip_start_sec: float = Field(ge=0.0)
    clip_end_sec: float = Field(gt=0.0)
    target_start_in_clip_sec: float = Field(ge=0.0)
    target_end_in_clip_sec: float = Field(gt=0.0)
    context_padding_sec: float = Field(ge=0.0, le=0.3)

    @model_validator(mode="after")
    def validate_clip_geometry(self) -> MaterializedEvaluationAudioClip:
        values = (
            self.original_start_sec,
            self.original_end_sec,
            self.clip_start_sec,
            self.clip_end_sec,
            self.target_start_in_clip_sec,
            self.target_end_in_clip_sec,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("clip coordinates must be finite")
        if self.original_end_sec <= self.original_start_sec:
            raise ValueError("original region has invalid geometry")
        if self.clip_end_sec <= self.clip_start_sec:
            raise ValueError("materialized clip has invalid geometry")
        if self.target_end_in_clip_sec <= self.target_start_in_clip_sec:
            raise ValueError("target region has invalid in-clip geometry")
        clip_duration = self.clip_end_sec - self.clip_start_sec
        if self.target_end_in_clip_sec > clip_duration + 1e-6:
            raise ValueError("target region exceeds materialized clip duration")
        return self


class EvaluationEffectAtom(FrozenEvaluationModel):
    """One requested observable change, frozen before output is inspected."""

    effect_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    effect_kind: str = Field(min_length=1)
    target_refs: tuple[str, ...] = ()
    region_ids: tuple[str, ...] = ()


RubricEvidenceSource = Literal[
    "measurement_evidence",
    "public_instruction",
    "subject_output",
    "context_evidence",
    "candidate_operation_cards",
    "audio_regions",
]


class RubricEvidenceRequirement(FrozenEvaluationModel):
    """Minimum judge-visible evidence needed to grade one rubric.

    ``acceptable_kinds`` applies to context evidence and candidate cards.  A
    requirement is satisfied when at least ``minimum_count`` available items
    match any declared source.  Conflicting items never satisfy it.
    """

    requirement_id: str = Field(min_length=1)
    acceptable_sources: tuple[RubricEvidenceSource, ...] = Field(min_length=1)
    acceptable_kinds: tuple[EvidenceKind, ...] = ()
    minimum_count: int = Field(default=1, ge=1, le=64)

    @model_validator(mode="after")
    def validate_requirement(self) -> RubricEvidenceRequirement:
        if len(self.acceptable_sources) != len(set(self.acceptable_sources)):
            raise ValueError("minimum-evidence sources must be unique")
        if len(self.acceptable_kinds) != len(set(self.acceptable_kinds)):
            raise ValueError("minimum-evidence kinds must be unique")
        if self.acceptable_kinds and not set(self.acceptable_sources).intersection(
            {"context_evidence", "candidate_operation_cards"}
        ):
            raise ValueError("minimum-evidence kinds require an evidence collection source")
        return self


class AtomicEvaluationRubric(FrozenEvaluationModel):
    """One question graded on the shared 0-3 ordinal scale."""

    rubric_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    question: str = Field(min_length=1)
    observable: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0.0)
    evidence_region_ids: tuple[str, ...] = ()
    target_module: GuidanceTargetModule | None = None
    minimum_evidence: tuple[RubricEvidenceRequirement, ...] = (
        RubricEvidenceRequirement(
            requirement_id="module_output",
            acceptable_sources=("subject_output",),
        ),
    )
    capacity_pass_grades: tuple[int, ...] = (2, 3)
    guidance_actionable_grades: tuple[int, ...] = (0, 1)
    authored_before_output: Literal[True] = True
    fact_constraints: tuple[FactConstraint, ...] = ()

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rubric weight must be finite")
        return value

    @model_validator(mode="after")
    def validate_grade_policies(self) -> AtomicEvaluationRubric:
        for label, values in (
            ("capacity pass", self.capacity_pass_grades),
            ("guidance actionable", self.guidance_actionable_grades),
        ):
            if len(values) != len(set(values)) or any(value < 0 or value > 3 for value in values):
                raise ValueError(f"{label} grades must be unique values in [0, 3]")
        return self


class EvaluationJudgePolicy(FrozenEvaluationModel):
    """Fixed judge slots; invalid transport may retry only within the same slot."""

    schema_version: Literal["x-editor.evaluation-judge-policy/v1"] = (
        "x-editor.evaluation-judge-policy/v1"
    )
    slot_ids: tuple[str, ...] = ("judge_1", "judge_2", "judge_3")
    minimum_valid_slots: int = Field(default=2, ge=1)
    retry_invalid_transport_within_slot: bool = True
    maximum_transport_attempts_per_slot: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def validate_slots(self) -> EvaluationJudgePolicy:
        if not self.slot_ids or any(not value.strip() for value in self.slot_ids):
            raise ValueError("judge slots must be non-empty")
        if len(self.slot_ids) != len(set(self.slot_ids)):
            raise ValueError("judge slot IDs must be unique")
        if self.minimum_valid_slots > len(self.slot_ids):
            raise ValueError("minimum_valid_slots exceeds configured judge slots")
        return self


class SemanticPathEvaluationTask(FrozenEvaluationModel):
    """Read-only LLM evaluation of Reasoner semantics or Planner PlanIR."""

    schema_version: Literal["x-editor.semantic-path-evaluation-task/v1"] = (
        "x-editor.semantic-path-evaluation-task/v1"
    )
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    input_snapshot_ref: str = Field(min_length=1)
    focus: Literal[
        "artifact_measurements",
        "perception_evidence",
        "reasoner_semantics",
        "planner_decision",
        "plan_graph",
        "tool_strategy",
    ]
    evaluator_role: EvaluatorRole
    evaluation_use: EvaluationUse
    backend_id: str = Field(min_length=1)
    public_instruction: str = Field(min_length=1)
    subject_output: EvaluationEvidenceItem
    audio_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    context_evidence: tuple[EvaluationEvidenceItem, ...] = ()
    candidate_operation_cards: tuple[EvaluationEvidenceItem, ...] = ()
    rubrics: tuple[AtomicEvaluationRubric, ...] = Field(min_length=1)
    judge_policy: EvaluationJudgePolicy = Field(default_factory=EvaluationJudgePolicy)
    selection: FrozenEvaluationSelection
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False
    can_select_artifact: Literal[False] = False

    @model_validator(mode="after")
    def validate_focus_evidence(self) -> SemanticPathEvaluationTask:
        if self.selection.focus != self.focus:
            raise ValueError("selection focus does not match semantic task focus")
        expected_subject = {
            "artifact_measurements": "other",
            "perception_evidence": "reviewed_perception",
            "reasoner_semantics": "reasoner_memo",
            "planner_decision": "planner_decision",
            "plan_graph": "plan_ir",
            "tool_strategy": "plan_ir",
        }[self.focus]
        if self.subject_output.kind != expected_subject:
            raise ValueError(f"{self.focus} expects subject evidence kind {expected_subject!r}")
        if self.audio_artifacts and self.focus != "perception_evidence":
            raise ValueError("semantic audio attachments require perception_evidence focus")
        if self.focus == "perception_evidence":
            if self.candidate_operation_cards:
                raise ValueError("perception evaluation cannot consume operation candidates")
            _validate_unique_ids(self.audio_artifacts, "artifact_ref_id", "artifact")
            if any(item.role not in {"input", "source", "reference"} for item in self.audio_artifacts):
                raise ValueError("perception may only inspect its input audio")
            evidence_ids = {"public_instruction", self.subject_output.evidence_id,
                            *(item.evidence_id for item in self.context_evidence)}
            if evidence_ids.intersection(item.artifact_ref_id for item in self.audio_artifacts):
                raise ValueError("audio and semantic evidence namespaces must be disjoint")
        if self.focus == "reasoner_semantics" and self.candidate_operation_cards:
            raise ValueError("reasoner semantic evaluation cannot consume operation candidates")
        if self.focus == "tool_strategy" and not self.candidate_operation_cards:
            raise ValueError("tool strategy evaluation requires the visible candidate set")
        _validate_unique_ids(self.rubrics, "rubric_id", "rubric")
        _validate_unique_ids(
            (*self.context_evidence, *self.candidate_operation_cards, self.subject_output),
            "evidence_id",
            "evidence",
        )
        return self


class StepEvaluationSpec(FrozenEvaluationModel):
    """Pre-output rubric and artifact-input declaration for one graph node."""

    schema_version: Literal["x-editor.step-evaluation-spec/v1"] = "x-editor.step-evaluation-spec/v1"
    spec_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    input_snapshot_ref: str = Field(min_length=1)
    producer_node_id: str = Field(min_length=1)
    evaluator_role: EvaluatorRole
    evaluation_use: EvaluationUse
    backend_id: str = Field(min_length=1)
    public_instruction: str = Field(min_length=1)
    input_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    reference_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    effect_contract_ref: str = Field(min_length=1)
    intended_effects: tuple[EvaluationEffectAtom, ...] = Field(min_length=1)
    preservation_requirements: tuple[str, ...] = ()
    rubrics: tuple[AtomicEvaluationRubric, ...] = Field(min_length=1)
    context_padding_sec: float = Field(default=0.25, ge=0.0, le=0.3)
    judge_policy: EvaluationJudgePolicy = Field(default_factory=EvaluationJudgePolicy)
    selection: FrozenEvaluationSelection
    frozen_before_output: Literal[True] = True
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False

    @model_validator(mode="after")
    def validate_spec(self) -> StepEvaluationSpec:
        if self.selection.focus != "intermediate_step":
            raise ValueError("step evaluation selection must target intermediate_step")
        if not self.selection.selected_before_output or not self.selection.outcome_independent:
            raise ValueError("step evaluation spec requires a pre-output selection")
        _validate_unique_ids(
            (*self.input_artifacts, *self.reference_artifacts),
            "artifact_ref_id",
            "artifact",
        )
        _validate_unique_ids(self.intended_effects, "effect_id", "effect")
        _validate_unique_ids(self.rubrics, "rubric_id", "rubric")
        return self


class ArtifactQualityEvaluationTask(FrozenEvaluationModel):
    """Optional multimodal evaluation of an intermediate or delivered artifact."""

    schema_version: Literal["x-editor.artifact-quality-evaluation-task/v1"] = (
        "x-editor.artifact-quality-evaluation-task/v1"
    )
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    input_snapshot_ref: str = Field(min_length=1)
    focus: Literal["intermediate_step", "final_delivery"]
    evaluator_role: EvaluatorRole
    evaluation_use: EvaluationUse
    backend_id: str = Field(min_length=1)
    visibility: EvaluationVisibility = "blind_quality"
    public_instruction: str = Field(min_length=1)
    input_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    output_artifact: EvaluationArtifactRef
    reference_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    producer_node_id: str | None = None
    effect_contract_ref: str = Field(min_length=1)
    intended_effects: tuple[EvaluationEffectAtom, ...] = Field(min_length=1)
    preservation_requirements: tuple[str, ...] = ()
    regions: tuple[EvaluationAudioRegion, ...] = ()
    materialized_clips: tuple[MaterializedEvaluationAudioClip, ...] = ()
    rubrics: tuple[AtomicEvaluationRubric, ...] = Field(min_length=1)
    forensic_evidence: tuple[EvaluationEvidenceItem, ...] = ()
    measurement_evidence: tuple[EvaluationEvidenceItem, ...] = ()
    delivery_context: DeliveryReviewContext | None = None
    coverage_requirements: tuple[DeliveryCoverageRequirement, ...] = ()
    context_padding_sec: float = Field(default=0.25, ge=0.0, le=0.3)
    judge_policy: EvaluationJudgePolicy = Field(default_factory=EvaluationJudgePolicy)
    selection: FrozenEvaluationSelection
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False
    can_select_artifact: Literal[False] = False

    @model_validator(mode="after")
    def validate_artifact_task(self) -> ArtifactQualityEvaluationTask:
        if self.selection.focus != self.focus:
            raise ValueError("selection focus does not match artifact task focus")
        if self.focus == "intermediate_step" and not self.producer_node_id:
            raise ValueError("intermediate evaluation requires a producer_node_id")
        if self.output_artifact.role not in {"output", "intermediate"}:
            raise ValueError("output_artifact must have output or intermediate role")
        artifacts = (
            *self.input_artifacts,
            self.output_artifact,
            *self.reference_artifacts,
        )
        _validate_unique_ids(artifacts, "artifact_ref_id", "artifact")
        _validate_unique_ids(self.measurement_evidence, "evidence_id", "measurement")
        occupied = ({a.artifact_ref_id for a in artifacts} | {r.region_id for r in self.regions}
                    | {c.clip_id for c in self.materialized_clips} | {e.evidence_id for e in self.forensic_evidence})
        if any(e.visibility != "judge" or e.evidence_id in occupied for e in self.measurement_evidence):
            raise ValueError("measurement evidence must be visible and have a disjoint namespace")
        _validate_unique_ids(self.regions, "region_id", "region")
        _validate_unique_ids(self.materialized_clips, "clip_id", "clip")
        _validate_unique_ids(self.intended_effects, "effect_id", "effect")
        _validate_unique_ids(self.rubrics, "rubric_id", "rubric")
        artifact_ids = {item.artifact_ref_id for item in artifacts}
        if self.delivery_context is not None:
            context = self.delivery_context
            if self.focus != "final_delivery":
                raise ValueError("terminal delivery context requires final_delivery focus")
            if context.terminal_artifact_ref_id != self.output_artifact.artifact_ref_id:
                raise ValueError("delivery context must name the evaluated output")
            if context.current_request != self.public_instruction:
                raise ValueError("delivery context must bind the current public request")
            if any(
                state.artifact_ref_id not in artifact_ids
                for state in context.states if state.artifact_ref_id is not None
            ):
                raise ValueError("delivery state references an unattached artifact")
            if not self.coverage_requirements:
                raise ValueError("delivery review requires explicit coverage items")
        elif self.coverage_requirements:
            raise ValueError("coverage items require a delivery context")
        _validate_unique_ids(self.coverage_requirements, "coverage_id", "coverage")
        if any(region.artifact_ref_id not in artifact_ids for region in self.regions):
            raise ValueError("audio region references an unknown artifact")
        region_ids = {item.region_id for item in self.regions}
        rubric_ids = {item.rubric_id for item in self.rubrics}
        if any(
            item.rubric_id not in rubric_ids
            or not set(item.evidence_region_ids).issubset(region_ids)
            for item in self.coverage_requirements
        ):
            raise ValueError("delivery coverage references an unknown rubric or region")
        if any(
            clip.region_id not in region_ids or clip.source_artifact_ref_id not in artifact_ids
            for clip in self.materialized_clips
        ):
            raise ValueError("materialized clip references an unknown region or artifact")
        if len({item.region_id for item in self.materialized_clips}) != len(
            self.materialized_clips
        ):
            raise ValueError("each region can have at most one materialized clip")
        dangling_effect_regions = {
            region_id
            for effect in self.intended_effects
            for region_id in effect.region_ids
            if region_id not in region_ids
        }
        dangling_rubric_regions = {
            region_id
            for rubric in self.rubrics
            for region_id in rubric.evidence_region_ids
            if region_id not in region_ids
        }
        if dangling_effect_regions or dangling_rubric_regions:
            raise ValueError(
                "effect or rubric references unknown regions: "
                f"{sorted(dangling_effect_regions | dangling_rubric_regions)}"
            )
        if self.visibility == "blind_quality" and any(
            item.visibility != "forensic_only" for item in self.forensic_evidence
        ):
            raise ValueError("blind quality task must keep forensic evidence hidden")
        if self.selection.capacity_eligible and self.visibility != "blind_quality":
            raise ValueError("forensic evaluation is not eligible for capacity scores")
        return self


EvaluationTask = SemanticPathEvaluationTask | ArtifactQualityEvaluationTask


class AtomicRubricJudgement(FrozenEvaluationModel):
    """One judge slot's answer to one atomic rubric."""

    rubric_id: str = Field(min_length=1)
    applicability: RubricApplicability
    grade: int | None = Field(default=None, ge=0, le=3)
    abstain: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_grade(self) -> AtomicRubricJudgement:
        if self.applicability == "applicable" and not self.abstain:
            if self.grade is None:
                raise ValueError("an applicable non-abstaining judgement requires a grade")
        elif self.grade is not None:
            raise ValueError("N/A, unknown, and abstaining judgements cannot carry a grade")
        if self.applicability == "not_applicable" and self.abstain:
            raise ValueError("not_applicable is a decision, not an abstention")
        return self


class JudgeSlotResult(FrozenEvaluationModel):
    """Complete result of one preregistered judge slot."""

    slot_id: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    judge_family: str = ""
    sampling_mode: JudgeSamplingMode = "unknown"
    independence_claim: JudgeIndependenceClaim = "unspecified"
    independence_group: str = ""
    status: JudgeSlotStatus
    judgements: tuple[AtomicRubricJudgement, ...] = ()
    request_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    error_type: str | None = None
    error_message: str | None = None
    delivery_review: DeliveryReviewResult | None = None
    coverage_reference_bindings: tuple[DeliveryCoverageReferenceBinding, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> JudgeSlotResult:
        family = self.judge_family or f"{self.provider}:{self.model}"
        group = self.independence_group or (
            f"slot:{self.slot_id}"
            if self.independence_claim == "independent"
            else f"family:{family}"
        )
        object.__setattr__(self, "judge_family", family)
        object.__setattr__(self, "independence_group", group)
        if self.status == "completed" and not self.judgements:
            raise ValueError("completed judge slot must contain judgements")
        if self.status != "completed" and self.judgements:
            raise ValueError("failed judge slot cannot contain judgements")
        return self


class AtomicRubricAggregate(FrozenEvaluationModel):
    """Cross-slot aggregate for one rubric."""

    rubric_id: str
    weight: float = Field(gt=0.0)
    status: Literal["graded", "not_applicable", "unknown"]
    grade: float | None = Field(default=None, ge=0.0, le=3.0)
    normalized_score: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_slot_count: int = Field(ge=0)
    slot_grades: tuple[int, ...] = ()
    disagreement: float | None = Field(default=None, ge=0.0, le=3.0)
    capacity_passed: bool | None = None
    guidance_actionable: bool = False
    evidence_status: Literal["sufficient", "missing", "conflicting"] = "sufficient"
    model_grade: float | None = Field(default=None, ge=0.0, le=3.0)
    grade_source: Literal["model", "bounded_by_measured_constraint", "unknown_evidence"] = "model"
    fact_checks: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_aggregate(self) -> AtomicRubricAggregate:
        if self.status == "graded":
            if self.grade is None or self.normalized_score is None:
                raise ValueError("graded rubric requires grade and normalized_score")
            # A diagnostic median can exist without a unanimous capacity decision.
            if self.guidance_actionable and self.capacity_passed is None:
                raise ValueError("unknown capacity consensus cannot drive guidance")
        elif self.grade is not None or self.normalized_score is not None:
            raise ValueError("ungraded rubric cannot carry a score")
        elif self.capacity_passed is not None or self.guidance_actionable:
            raise ValueError("ungraded rubric cannot drive capacity or guidance")
        return self


class EvaluationResultV2(FrozenEvaluationModel):
    """Read-only ordinal evaluation result and uncertainty bounds."""

    schema_version: Literal["x-editor.evaluation-result/v2"] = "x-editor.evaluation-result/v2"
    result_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    focus: EvaluationFocus
    evaluator_role: EvaluatorRole
    evaluation_use: EvaluationUse
    selection_id: str = Field(min_length=1)
    slot_results: tuple[JudgeSlotResult, ...]
    rubric_results: tuple[AtomicRubricAggregate, ...]
    score_status: Literal["scored", "unknown", "not_applicable"]
    known_score: float | None = Field(default=None, ge=0.0, le=1.0)
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    known_weight_fraction: float = Field(ge=0.0, le=1.0)
    effective_judge_count: int = Field(default=0, ge=0)
    correlated_slot_count: int = Field(default=0, ge=0)
    independent_vote_claimed: bool = False
    capacity_eligible: bool
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False
    can_select_artifact: Literal[False] = False

    @model_validator(mode="after")
    def validate_bounds(self) -> EvaluationResultV2:
        if self.lower_bound > self.upper_bound:
            raise ValueError("evaluation lower_bound exceeds upper_bound")
        if self.known_score is not None and not (
            self.lower_bound <= self.known_score <= self.upper_bound
        ):
            raise ValueError("known score must fall inside uncertainty bounds")
        return self


class GuidancePacket(FrozenEvaluationModel):
    """Questions and constraints for a future attempt, never a replacement output."""

    schema_version: Literal["x-editor.guidance-packet/v1"] = "x-editor.guidance-packet/v1"
    guidance_id: str = Field(min_length=1)
    source_result_ids: tuple[str, ...] = Field(min_length=1)
    target_modules: tuple[GuidanceTargetModule, ...] = Field(min_length=1)
    failed_or_weak_rubric_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    questions_to_reconsider: tuple[str, ...] = Field(min_length=1)
    constraints_to_preserve: tuple[str, ...] = ()
    applies_to_new_attempt_only: Literal[True] = True
    contains_expected_answer: Literal[False] = False
    contains_direct_replacement_output: Literal[False] = False
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False
    can_override_verifier: Literal[False] = False
    guidance_fingerprint: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_unique_refs(self) -> GuidancePacket:
        for label, values in (
            ("source result", self.source_result_ids),
            ("target module", self.target_modules),
            ("rubric", self.failed_or_weak_rubric_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} values must be unique")
        expected = canonical_evaluation_fingerprint(
            {
                "target_modules": self.target_modules,
                "weak_rubrics": self.failed_or_weak_rubric_ids,
                "issues": tuple(_normalize_fingerprint_text(item) for item in self.questions_to_reconsider),
                "preservation": tuple(
                    _normalize_fingerprint_text(item) for item in self.constraints_to_preserve
                ),
            }
        )
        if self.guidance_fingerprint and self.guidance_fingerprint != expected:
            raise ValueError("guidance_fingerprint does not match guidance content")
        object.__setattr__(self, "guidance_fingerprint", expected)
        return self


class GuidedAttemptPolicy(FrozenEvaluationModel):
    """Fixed policy for a future loop arm; disabled by default."""

    schema_version: Literal["x-editor.guided-attempt-policy/v1"] = (
        "x-editor.guided-attempt-policy/v1"
    )
    policy_id: str = Field(min_length=1)
    mode: Literal["disabled", "audit", "enforced"] = "disabled"
    preregistered_feedback_arm: bool = False
    max_new_attempts: int = Field(default=0, ge=0, le=2)
    # Logical judge slots; transport attempts are budgeted separately.
    max_evaluator_calls: int = Field(default=0, ge=0, le=256)
    max_evaluator_http_attempts: int | None = Field(default=None, ge=0)
    max_guider_http_attempts: int | None = Field(default=None, ge=0)
    new_attempt_reenters_normal_forward_path: Literal[True] = True
    can_mutate_existing_attempt: Literal[False] = False
    can_override_verifier: Literal[False] = False
    can_select_hidden_best_attempt: Literal[False] = False
    terminal_selection: Literal["last_executed_attempt"] = "last_executed_attempt"
    stop_on_repeated_failure_fingerprint: Literal[True] = True
    stop_on_repeated_guidance_fingerprint: Literal[True] = True

    @model_validator(mode="after")
    def validate_enforcement(self) -> GuidedAttemptPolicy:
        if self.mode == "enforced" and (
            not self.preregistered_feedback_arm
            or self.max_new_attempts < 1
            or self.max_evaluator_calls < 1
        ):
            raise ValueError(
                "enforced guided attempts require a preregistered arm and positive budgets"
            )
        if self.mode == "disabled" and (
            self.max_new_attempts or self.max_evaluator_calls
            or self.max_evaluator_http_attempts or self.max_guider_http_attempts
        ):
            raise ValueError("disabled guided-attempt policy must have zero budgets")
        return self


class GuidedAttemptDecision(FrozenEvaluationModel):
    """Non-mutating controller decision about whether to open a new attempt."""

    schema_version: Literal["x-editor.guided-attempt-decision/v1"] = (
        "x-editor.guided-attempt-decision/v1"
    )
    decision_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    action: Literal["no_new_attempt", "shadow_recommendation", "open_new_attempt"]
    guidance_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = Field(min_length=1)
    new_attempt_authorized: bool
    proposed_attempt_index: int | None = Field(default=None, ge=1)
    existing_attempt_mutated: Literal[False] = False
    normal_forward_path_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_action(self) -> GuidedAttemptDecision:
        if self.action == "open_new_attempt":
            if not self.new_attempt_authorized or self.proposed_attempt_index is None:
                raise ValueError("open_new_attempt requires authorization and an index")
        elif self.new_attempt_authorized or self.proposed_attempt_index is not None:
            raise ValueError("non-opening decision cannot authorize a new attempt")
        return self


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_JUDGE_VISIBLE_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


def _normalize_fingerprint_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _validate_unique_ids(values: Any, field_name: str, label: str) -> None:
    ids = [getattr(item, field_name) for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} IDs must be unique")


__all__ = [
    "ArtifactQualityEvaluationTask",
    "AtomicEvaluationRubric",
    "AtomicRubricAggregate",
    "AtomicRubricJudgement",
    "EvaluationArtifactRef",
    "EvaluationAudioRegion",
    "EvaluationEffectAtom",
    "EvaluationEvidenceItem",
    "EvaluationFocus",
    "EvaluationInputSnapshot",
    "EvaluationJudgePolicy",
    "EvaluationMode",
    "EvaluationResultV2",
    "EvaluationSamplingPolicy",
    "EvaluationTask",
    "EvaluationUse",
    "EvaluationVisibility",
    "EvidenceState",
    "FrozenEvaluationSelection",
    "GuidancePacket",
    "GuidanceTargetModule",
    "GuidedAttemptDecision",
    "GuidedAttemptPolicy",
    "JudgeSlotResult",
    "MaterializedEvaluationAudioClip",
    "RubricEvidenceRequirement",
    "RubricEvidenceSource",
    "SemanticPathEvaluationTask",
    "StepEvaluationSpec",
    "canonical_evaluation_fingerprint",
]
