"""Auditable semantic contracts for the composite Planner/Reasoner gateway.

These models expose bounded decisions, evidence needs, and field provenance.
They intentionally do not expose prompts, model responses, or free-form chains
of thought. Runtime orchestration remains responsible for provider invocation,
evidence admission, and execution authority.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.schemas import (
    AudioEditPlan,
    AudioEditStep,
    PlannerCannotPlanReason,
    StrictModel,
)
from x_editor.perception.capability_feedback import PerceptionCapabilityFeedback
from x_editor.reasoner.contracts import (
    PriorTurnSemanticContext,
    ReasonedEditIntent,
    ReasonerAdmissionReport,
    SemanticReasoningMemo,
    SessionStateContext,
)
from x_editor.runtime.audio_asset_facts import AudioAssetFacts
from x_editor.runtime.audio_usage import WorkflowAudioUsageBinding

TemporalPrecision = Literal["coarse", "segment", "word"]
PerceptionTargetKind = Literal[
    "event_label",
    "speaker_hint",
    "quoted_text",
    "semantic_property",
    "audio_property",
]
ArgumentSourceKind = Literal[
    "user_request",
    "perception_evidence",
    "derived_from_evidence",
    "deterministic_context",
]
RequirementPriority = Literal["must", "should", "prefer"]

_AUTO_PERCEPTION_OPERATION = "auto"
_PERCEPTION_OPERATION_PREFIXES = ("perception.", "flexsed.")
_EVENT_QUERY_REQUIRED_OPERATIONS = {
    "perception.flexsed_events",
    "flexsed.detect_events",
}
_EVENT_QUERY_TARGET_KINDS = {
    "event_label",
    "semantic_property",
    "audio_property",
}
_REGISTERED_ARGUMENT_ALIASES = {
    ("inputs", "input_audio"): "input_audio_path",
    ("inputs", "input_audio_path"): "input_audio",
}
_EXACT_INFERENCE_ARGUMENT_LEAVES_BY_OPERATION = {
    "artifact_ops.extract_spans": {"start_sec", "end_sec"},
    "artifact_ops.remove_spans": {"start_sec", "end_sec"},
    "artifact_ops.mute_spans": {"start_sec", "end_sec"},
    "artifact_ops.beep_spans": {"start_sec", "end_sec"},
    "artifact_ops.speaker_scoped_edit": {"start_sec", "end_sec"},
}
_MAX_TEXT_LENGTH = 2048
_MAX_LIST_ITEM_LENGTH = 512
PLAN_STEP_AUDIT_LABEL_MAX_LENGTH = 1024
PLAN_STEP_TARGET_SUMMARY_MAX_LENGTH = 1024
PLANNER_DECISION_SUMMARY_MAX_LENGTH = 8192

WorkflowListText = Annotated[
    str,
    Field(min_length=1, max_length=_MAX_LIST_ITEM_LENGTH),
]
WorkflowObjectiveText = Annotated[
    str,
    Field(min_length=1, max_length=1024),
]


class PlannerPerceptionTarget(StrictModel):
    """One semantic target the Perception operation must ground."""

    kind: PerceptionTargetKind
    value: str = Field(min_length=1, max_length=512)

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class PlannerPerceptionJob(StrictModel):
    """One independently routed Perception request inside a gateway round."""

    schema_version: Literal["x-editor.planner-perception-job/v1"] = (
        "x-editor.planner-perception-job/v1"
    )
    job_id: str = Field(min_length=1, max_length=256)
    scope: Literal["general", "targeted"] = "targeted"
    perception_operation: str = Field(
        default=_AUTO_PERCEPTION_OPERATION,
        min_length=1,
        max_length=256,
        description=(
            "Optional compatibility hint. Use 'auto' or omit it to let the "
            "Perception Gateway select the concrete route."
        ),
    )
    primary_audio_ref: str = Field(min_length=1, max_length=4096)
    supporting_audio_refs: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Additional assets actually observed by a paired/multi-audio capability. "
            "Keep empty for a single-audio observation. Independent measurements of "
            "different assets or versions need separate jobs, one primary ref each."
        ),
    )
    requirements: list[WorkflowListText] = Field(default_factory=list, max_length=32)
    expected_evidence_fields: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=32,
    )
    perception_targets: list[PlannerPerceptionTarget] = Field(
        default_factory=list,
        max_length=32,
    )
    temporal_precision: TemporalPrecision
    target_hints: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    evidence_questions: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Planner-authored semantic questions for this observation need. Semantic "
            "span needs should normally provide at least one directly."
        ),
    )
    observation_need_ids: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    recognition_context: str | None = Field(
        default=None, max_length=2048,
        description=(
            "Optional unverified vocabulary/context for a fresh speech recognition. "
            "Use to investigate deficient ASR; not a replacement transcript or an edit instruction."
        ),
    )
    required: Literal[True] = True

    @field_validator("recognition_context", mode="before")
    @classmethod
    def normalize_recognition_context(cls, value: Any) -> Any:
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("job_id", "perception_operation", "primary_audio_ref", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "supporting_audio_refs",
        "requirements",
        "expected_evidence_fields",
        "target_hints",
        "evidence_questions",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if not item:
                    raise ValueError("Perception job text lists cannot contain blanks")
            normalized.append(item)
        return _dedupe_text_values(normalized)

    @model_validator(mode="after")
    def validate_job(self) -> PlannerPerceptionJob:
        if (
            self.perception_operation != _AUTO_PERCEPTION_OPERATION
            and not self.perception_operation.startswith(_PERCEPTION_OPERATION_PREFIXES)
        ):
            raise ValueError("perception_operation must use perception.* or flexsed.*")
        _require_unique(
            self.supporting_audio_refs,
            field_name="supporting_audio_refs",
        )
        if self.primary_audio_ref in self.supporting_audio_refs:
            raise ValueError("primary audio cannot also be a supporting audio")
        for field_path in self.expected_evidence_fields:
            _validate_field_path(field_path, label="expected evidence field")
        if (
            self.perception_operation == "perception.semantic_span_review"
            and not self.perception_targets
            and not self.evidence_questions
        ):
            raise ValueError(
                "semantic span review requires a typed target or explicit evidence question"
            )
        # perception_operation is only a compatibility hint. Operation-specific
        # requirements and audio shape are checked after the Gateway has selected
        # the concrete route, so an inaccurate hint cannot reject a coherent need.
        return self


class PlannerAudioObservationJob(StrictModel):
    """One audio-scoped observation requested before execution."""

    job_id: str = Field(min_length=1, max_length=256)
    audio_ref: str = Field(min_length=1, max_length=4096)
    supporting_audio_refs: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=8,
    )
    requirements: list[WorkflowListText] = Field(default_factory=list, max_length=32)
    expected_evidence_fields: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=32,
    )
    required: bool = True

    @model_validator(mode="after")
    def validate_job(self) -> PlannerAudioObservationJob:
        self.job_id = self.job_id.strip()
        self.audio_ref = self.audio_ref.strip()
        _require_unique(
            self.supporting_audio_refs,
            field_name="supporting_audio_refs",
        )
        if self.audio_ref in self.supporting_audio_refs:
            raise ValueError("observation audio cannot also be a supporting audio")
        return self


class PlannerPerceptionIntent(StrictModel):
    """First-round semantic product: what evidence the Planner needs and why."""

    schema_version: Literal["x-editor.planner-perception-intent/v1"] = (
        "x-editor.planner-perception-intent/v1"
    )
    request_id: str = Field(min_length=1, max_length=256)
    scope: Literal["general", "targeted"] = "targeted"
    perception_operation: str = Field(
        default=_AUTO_PERCEPTION_OPERATION,
        min_length=1,
        max_length=256,
        description=(
            "Optional compatibility hint. Use 'auto' or omit it to let the "
            "Perception Gateway select the concrete route."
        ),
    )
    edit_objective: str = Field(min_length=1, max_length=_MAX_TEXT_LENGTH)
    perception_targets: list[PlannerPerceptionTarget] = Field(
        default_factory=list,
        max_length=32,
    )
    target_hints: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    evidence_questions: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    expected_evidence_fields: list[WorkflowListText] = Field(default_factory=list, max_length=32)
    temporal_precision: TemporalPrecision
    candidate_operations: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    ambiguities: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    audio_refs: list[WorkflowListText] = Field(min_length=1, max_length=8)
    primary_audio_ref: str = Field(min_length=1, max_length=4096)
    requirements: list[WorkflowListText] = Field(default_factory=list, max_length=32)
    observation_jobs: list[PlannerAudioObservationJob] = Field(
        default_factory=list,
        max_length=16,
    )

    @field_validator(
        "request_id",
        "perception_operation",
        "edit_objective",
        "primary_audio_ref",
        mode="before",
    )
    @classmethod
    def normalize_required_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "target_hints",
        "evidence_questions",
        "expected_evidence_fields",
        "candidate_operations",
        "ambiguities",
        "audio_refs",
        "requirements",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if not item:
                    raise ValueError("reasoned workflow text lists cannot contain blanks")
                if len(item) > _MAX_LIST_ITEM_LENGTH:
                    raise ValueError("reasoned workflow text list item exceeds maximum length")
            normalized.append(item)
        return _dedupe_text_values(normalized)

    @field_validator("perception_targets", mode="before")
    @classmethod
    def normalize_targets(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[Any] = []
        seen: set[tuple[str, str]] = set()
        for item in value:
            if not isinstance(item, Mapping):
                normalized.append(item)
                continue
            kind = str(item.get("kind") or "").strip()
            target_value = str(item.get("value") or "").strip()
            key = (kind.casefold(), target_value.casefold())
            if key in seen:
                continue
            seen.add(key)
            normalized.append({**item, "kind": kind, "value": target_value})
        return normalized

    @model_validator(mode="after")
    def validate_perception_intent(self) -> PlannerPerceptionIntent:
        if (
            self.perception_operation != _AUTO_PERCEPTION_OPERATION
            and not self.perception_operation.startswith(_PERCEPTION_OPERATION_PREFIXES)
        ):
            raise ValueError("perception_operation must use perception.* or flexsed.*")
        semantic_span_has_projectable_query = bool(
            self.perception_operation == "perception.semantic_span_review"
            and (self.target_hints or self.edit_objective.strip())
        )
        if (
            self.perception_operation not in _EVENT_QUERY_REQUIRED_OPERATIONS
            and not self.perception_targets
            and not self.evidence_questions
            and not semantic_span_has_projectable_query
            and not self.expected_evidence_fields
        ):
            raise ValueError(
                "Perception intent requires a typed target or an explicit evidence question"
            )
        if self.primary_audio_ref not in self.audio_refs:
            raise ValueError("primary_audio_ref must be declared in audio_refs")
        if self.observation_jobs:
            _require_unique(
                [item.job_id for item in self.observation_jobs],
                field_name="observation_jobs.job_id",
            )
            _require_unique(
                [item.audio_ref for item in self.observation_jobs],
                field_name="observation_jobs.audio_ref",
            )
            job_refs = {
                ref
                for job in self.observation_jobs
                for ref in (job.audio_ref, *job.supporting_audio_refs)
            }
            if not job_refs.issubset(set(self.audio_refs)):
                raise ValueError("observation job refs must be declared in perception audio_refs")
            if self.observation_jobs[0].audio_ref != self.primary_audio_ref:
                raise ValueError("primary_audio_ref must equal the first observation job audio_ref")
        invalid_candidates = sorted(
            operation
            for operation in self.candidate_operations
            if operation.startswith(_PERCEPTION_OPERATION_PREFIXES)
        )
        if invalid_candidates:
            raise ValueError(
                "candidate_operations must name downstream edit operations, not "
                f"perception operations: {invalid_candidates}"
            )
        if self.perception_operation in _EVENT_QUERY_REQUIRED_OPERATIONS and not (
            _perception_event_queries(
                self.perception_targets,
                self.evidence_questions,
            )
        ):
            raise ValueError(
                f"{self.perception_operation} requires a typed event target or "
                "an explicit evidence question"
            )
        timestamp_jobs_are_unary = bool(self.observation_jobs) and all(
            not item.supporting_audio_refs for item in self.observation_jobs
        )
        if (
            self.perception_operation == "perception.timestamped_caption"
            and self.audio_refs != [self.primary_audio_ref]
            and not timestamp_jobs_are_unary
        ):
            raise ValueError(
                "timestamped caption handles exactly one primary audio ref per request"
            )
        for field_path in self.expected_evidence_fields:
            _validate_field_path(field_path, label="expected evidence field")
        return self


class EvidencePathReference(StrictModel):
    """One exact field admitted from one authoritative reviewed evidence item."""

    evidence_id: str = Field(min_length=1, max_length=256)
    field_path: str = Field(min_length=1, max_length=512)

    @field_validator("evidence_id", "field_path", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("field_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _validate_field_path(value, label="evidence field path")
        return value


class SemanticArgumentSource(StrictModel):
    """Field-level provenance for one concrete edit-step argument."""

    argument_path: str = Field(min_length=1, max_length=512)
    source_kind: ArgumentSourceKind
    evidence_id: str | None = Field(default=None, min_length=1, max_length=256)
    evidence_path: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("argument_path", "evidence_id", "evidence_path", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("argument_path", mode="after")
    @classmethod
    def normalize_numeric_list_indexes(cls, value: str) -> str:
        return re.sub(r"\[(0|[1-9][0-9]*)\]", r".\1", value)

    @model_validator(mode="after")
    def validate_source_reference(self) -> SemanticArgumentSource:
        _validate_argument_path(self.argument_path)
        evidence_backed = self.source_kind in {
            "perception_evidence",
            "derived_from_evidence",
        }
        if evidence_backed:
            if self.evidence_id is None or self.evidence_path is None:
                raise ValueError(
                    "evidence-backed argument sources require evidence_id and evidence_path"
                )
            _validate_field_path(self.evidence_path, label="argument evidence path")
        elif self.evidence_id is not None or self.evidence_path is not None:
            raise ValueError("non-evidence argument sources cannot carry evidence references")
        return self


class PlanStepSemanticBinding(StrictModel):
    """Second-round semantic binding for one evidence-dependent edit step."""

    plan_step_id: str = Field(min_length=1, max_length=256)
    objective: str = Field(
        min_length=1,
        max_length=PLAN_STEP_AUDIT_LABEL_MAX_LENGTH,
        description=(
            "Framework-projected audit label copied from the matching PlanIR step "
            "description; it is not an executable argument or semantic authority."
        ),
    )
    target: str = Field(
        min_length=1,
        max_length=PLAN_STEP_TARGET_SUMMARY_MAX_LENGTH,
    )
    evidence_ids: list[str] = Field(default_factory=list, max_length=32)
    evidence_paths: list[EvidencePathReference] = Field(default_factory=list, max_length=128)
    operation: str = Field(min_length=1, max_length=256)
    argument_sources: list[SemanticArgumentSource] = Field(default_factory=list, max_length=128)
    unresolved_requirements: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("plan_step_id", "objective", "target", "operation", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("evidence_ids", "unresolved_requirements", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if not item:
                    raise ValueError("semantic binding text lists cannot contain blanks")
                if len(item) > _MAX_LIST_ITEM_LENGTH:
                    raise ValueError("semantic binding list item exceeds maximum length")
            normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def validate_binding_shape(self) -> PlanStepSemanticBinding:
        _require_unique(self.evidence_ids, field_name="evidence_ids")
        _require_unique(
            self.unresolved_requirements,
            field_name="unresolved_requirements",
        )
        path_pairs = [(item.evidence_id, item.field_path) for item in self.evidence_paths]
        if len(path_pairs) != len(set(path_pairs)):
            raise ValueError("evidence_paths must be unique")
        path_evidence_ids = {item.evidence_id for item in self.evidence_paths}
        if path_evidence_ids != set(self.evidence_ids):
            raise ValueError(
                "every evidence_id must have an evidence path and every path must "
                "reference a declared evidence_id"
            )
        argument_source_keys = [
            (
                item.argument_path,
                item.source_kind,
                item.evidence_id,
                item.evidence_path,
            )
            for item in self.argument_sources
        ]
        if len(argument_source_keys) != len(set(argument_source_keys)):
            raise ValueError("argument_sources cannot contain duplicate sources")
        declared_paths = set(path_pairs)
        argument_evidence_paths = {
            (item.evidence_id, item.evidence_path)
            for item in self.argument_sources
            if item.evidence_id is not None and item.evidence_path is not None
        }
        undeclared = argument_evidence_paths - declared_paths
        if undeclared:
            raise ValueError(
                f"argument_sources contain undeclared evidence paths: {sorted(undeclared)}"
            )
        return self


class PlannerExecutionSemantics(StrictModel):
    """Complete second-round semantic product accompanying an executable plan."""

    schema_version: Literal["x-editor.planner-execution-semantics/v1"] = (
        "x-editor.planner-execution-semantics/v1"
    )
    bindings: list[PlanStepSemanticBinding] = Field(default_factory=list, max_length=256)
    audio_usage_bindings: list[WorkflowAudioUsageBinding] = Field(
        default_factory=list,
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_unique_steps(self) -> PlannerExecutionSemantics:
        step_ids = [binding.plan_step_id for binding in self.bindings]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("semantic bindings must use unique plan_step_id values")
        usage_keys = [(item.plan_step_id, item.argument_path) for item in self.audio_usage_bindings]
        if len(usage_keys) != len(set(usage_keys)):
            raise ValueError("audio usage bindings must use unique step and argument paths")
        return self


class PlannerIntentRequirement(StrictModel):
    """One bounded preservation or quality requirement inferred from the request."""

    dimension: str = Field(min_length=1, max_length=128)
    priority: RequirementPriority = "should"
    description: str = Field(min_length=1, max_length=1024)


class PlannerEditIntent(StrictModel):
    """Task semantics frozen before evidence and route selection."""

    schema_version: Literal["x-editor.planner-edit-intent/v1"] = "x-editor.planner-edit-intent/v1"
    effect_kinds: list[WorkflowListText] = Field(min_length=1, max_length=16)
    target_scopes: list[WorkflowListText] = Field(min_length=1, max_length=8)
    edit_objectives: list[WorkflowObjectiveText] = Field(min_length=1, max_length=16)
    preservation_requirements: list[PlannerIntentRequirement] = Field(
        default_factory=list,
        max_length=16,
    )
    quality_requirements: list[PlannerIntentRequirement] = Field(
        default_factory=list,
        max_length=16,
    )
    required_evidence: list[WorkflowListText] = Field(default_factory=list, max_length=32)
    candidate_operations: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    uncertainties: list[WorkflowListText] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_unique_values(self) -> PlannerEditIntent:
        for field_name in (
            "effect_kinds",
            "target_scopes",
            "edit_objectives",
            "required_evidence",
            "candidate_operations",
            "uncertainties",
        ):
            _require_unique(getattr(self, field_name), field_name=field_name)
        return self


class PlannerToolCandidateAssessment(StrictModel):
    """Auditable candidate comparison without private reasoning text."""

    operation: str = Field(min_length=1, max_length=256)
    semantic_fit: Literal["full", "partial", "none", "unknown"]
    preservation_fit: Literal["full", "partial", "none", "unknown"]
    quality_fit: Literal["full", "partial", "none", "unknown"]
    legal_inputs_available: bool
    rejection_codes: list[str] = Field(default_factory=list, max_length=32)


class PlannerParameterSelection(StrictModel):
    """One value selected from an operation card's declared legal domain."""

    argument_path: str = Field(min_length=1, max_length=256)
    selected_value: Any
    contract_operation: str = Field(min_length=1, max_length=256)

    @field_validator("argument_path")
    @classmethod
    def validate_argument_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith(("inputs.", "params.")):
            raise ValueError("selected parameter path must start with inputs. or params.")
        return normalized


class PlannerStepToolSelection(StrictModel):
    """Route and legal parameter declarations for one plan step."""

    plan_step_id: str = Field(min_length=1, max_length=256)
    selected_operation: str = Field(min_length=1, max_length=256)
    candidates: list[PlannerToolCandidateAssessment] = Field(
        min_length=1,
        max_length=32,
    )
    parameter_selections: list[PlannerParameterSelection] = Field(
        default_factory=list,
        max_length=128,
    )
    selection_summary: str = Field(default="", max_length=1024)

    @model_validator(mode="after")
    def validate_selection_shape(self) -> PlannerStepToolSelection:
        candidate_operations = [item.operation for item in self.candidates]
        _require_unique(candidate_operations, field_name="candidates.operation")
        if self.selected_operation not in candidate_operations:
            raise ValueError("selected_operation must appear in candidates")
        paths = [item.argument_path for item in self.parameter_selections]
        _require_unique(paths, field_name="parameter_selections.argument_path")
        mismatches = [
            item.argument_path
            for item in self.parameter_selections
            if item.contract_operation != self.selected_operation
        ]
        if mismatches:
            raise ValueError("every parameter selection must cite the selected operation contract")
        return self


class PlannerToolSelection(StrictModel):
    """All tool and parameter selections for an executable plan."""

    schema_version: Literal["x-editor.planner-tool-selection/v1"] = (
        "x-editor.planner-tool-selection/v1"
    )
    steps: list[PlannerStepToolSelection] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_unique_steps(self) -> PlannerToolSelection:
        _require_unique(
            [item.plan_step_id for item in self.steps],
            field_name="steps.plan_step_id",
        )
        return self


class PlannerLegalArgumentContract(StrictModel):
    """Provider-visible legal domain for one candidate operation argument."""

    argument_path: str = Field(min_length=1, max_length=256)
    required: bool = False
    value_schema: dict[str, Any] = Field(default_factory=dict)
    semantic_role: str = Field(default="", max_length=256)
    runtime_value_source: (
        Literal[
            "source_audio_refs",
            "reference_audio_refs",
            "source_or_reference_audio_refs",
        ]
        | None
    ) = None
    description: str = Field(default="", max_length=1024)

    @field_validator("argument_path")
    @classmethod
    def validate_argument_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith(("inputs.", "params.")):
            raise ValueError("legal argument path must start with inputs. or params.")
        return normalized


class PlannerCandidateOperationContract(StrictModel):
    """One shortlisted operation with capability and legal argument bounds."""

    operation: str = Field(min_length=1, max_length=256)
    semantic_effects: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=32,
    )
    supported_scopes: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=16,
    )
    requirements: dict[str, list[str]] = Field(default_factory=dict)
    arguments: list[PlannerLegalArgumentContract] = Field(
        default_factory=list,
        max_length=256,
    )
    framework_owned_arguments: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=256,
    )
    parameter_rules: dict[str, Any] = Field(default_factory=dict)
    preservation_profile: dict[str, Any] = Field(default_factory=dict)
    quality_profile: dict[str, Any] = Field(default_factory=dict)
    artifact_risk_profile: dict[str, Any] = Field(default_factory=dict)
    composition_profile: dict[str, Any] = Field(default_factory=dict)
    limitations: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_unique_arguments(self) -> PlannerCandidateOperationContract:
        _require_unique(
            [item.argument_path for item in self.arguments],
            field_name="arguments.argument_path",
        )
        _require_unique(
            self.framework_owned_arguments,
            field_name="framework_owned_arguments",
        )
        overlap = {item.argument_path for item in self.arguments}.intersection(
            self.framework_owned_arguments
        )
        if overlap:
            raise ValueError(
                "Planner-selectable and framework-owned arguments must not overlap: "
                f"{sorted(overlap)}"
            )
        return self


class QualifiedPlannerInput(StrictModel):
    """Semantic-authority-preserving input to a separated Planner module."""

    schema_version: Literal["x-editor.qualified-planner-input/v1"] = (
        "x-editor.qualified-planner-input/v1"
    )
    planner_request_id: str = Field(min_length=1, max_length=256)
    phase: Literal["evidence_planning", "execution_planning"]
    reasoned_intent: ReasonedEditIntent
    reasoner_admission: ReasonerAdmissionReport
    source_audio_refs: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=16,
    )
    reference_audio_refs: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=16,
    )
    prior_turn_contexts: list[PriorTurnSemanticContext] = Field(
        default_factory=list,
        max_length=32,
    )
    reviewed_evidence_ids: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=64,
    )
    reviewed_evidence: dict[str, Any] = Field(default_factory=dict)
    evidence_path_sources: dict[str, list[str]] = Field(default_factory=dict)
    perception_capability_catalog: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=64,
    )
    candidate_operation_contracts: list[PlannerCandidateOperationContract] = Field(
        default_factory=list,
        max_length=128,
    )
    intent_authority: Literal["reasoner"] = "reasoner"
    semantic_memo_visible: Literal[False] = False
    private_reasoning_visible: Literal[False] = False

    @model_validator(mode="after")
    def validate_qualified_input(self) -> QualifiedPlannerInput:
        operations = [item.operation for item in self.candidate_operation_contracts]
        _require_unique(operations, field_name="candidate_operation_contracts.operation")
        _require_unique(self.source_audio_refs, field_name="source_audio_refs")
        _require_unique(self.reference_audio_refs, field_name="reference_audio_refs")
        _require_unique(
            self.reviewed_evidence_ids,
            field_name="reviewed_evidence_ids",
        )
        if self.phase == "execution_planning" and not self.reasoner_admission.planning_ready:
            raise ValueError("execution planning requires a planning-ready Reasoner admission")
        return self


class MemoQualifiedPlannerInput(StrictModel):
    """Planner input that preserves the Reasoner's natural-language conclusion."""

    schema_version: Literal["x-editor.memo-qualified-planner-input/v1"] = (
        "x-editor.memo-qualified-planner-input/v1"
    )
    planner_request_id: str = Field(min_length=1, max_length=256)
    phase: Literal["evidence_planning", "execution_planning"]
    user_request: str = Field(min_length=1, max_length=8192)
    execution_request: str | None = Field(default=None, max_length=32768)
    source_text: str | None = Field(default=None, max_length=16384)
    semantic_memo: SemanticReasoningMemo
    source_audio_refs: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    reference_audio_refs: list[WorkflowListText] = Field(default_factory=list, max_length=16)
    audio_asset_facts: list[AudioAssetFacts] = Field(default_factory=list)
    prior_turn_contexts: list[PriorTurnSemanticContext] = Field(
        default_factory=list,
        max_length=32,
    )
    session_state: SessionStateContext | None = None
    reviewed_evidence_ids: list[WorkflowListText] = Field(default_factory=list, max_length=64)
    reviewed_evidence: dict[str, Any] = Field(default_factory=dict)
    evidence_path_sources: dict[str, list[str]] = Field(default_factory=dict)
    evidence_path_audio_refs: dict[str, str] = Field(default_factory=dict)
    future_attempt_guidance: FutureAttemptGuidance | None = None
    perception_capability_feedback: PerceptionCapabilityFeedback | None = None
    perception_capability_catalog: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=64,
    )
    candidate_operation_contracts: list[PlannerCandidateOperationContract] = Field(
        default_factory=list,
        max_length=128,
    )
    semantic_authority: Literal["reasoner_memo"] = "reasoner_memo"
    semantic_conflict_policy: Literal["report_do_not_silently_reinterpret"] = (
        "report_do_not_silently_reinterpret"
    )
    memo_normalization_applied: Literal[False] = False
    private_reasoning_visible: Literal[False] = False

    @model_validator(mode="after")
    def validate_memo_input(self) -> MemoQualifiedPlannerInput:
        if self.semantic_memo.request_id != self.planner_request_id:
            raise ValueError("semantic memo request_id must match planner_request_id")
        if self.semantic_memo.remaining_execution_request != self.execution_request:
            raise ValueError("execution_request must exactly match Runtime-projected remaining work")
        if self.execution_request is None and (
            self.semantic_memo.selected_base_state_node_id is None
            or not self.semantic_memo.changes_satisfied_by_base
        ):
            raise ValueError(
                "Planner input cannot treat an incomplete audit memo as a state-only completion"
            )
        operations = [item.operation for item in self.candidate_operation_contracts]
        _require_unique(operations, field_name="candidate_operation_contracts.operation")
        perception_operations = [
            str(item.get("operation") or "").strip() for item in self.perception_capability_catalog
        ]
        if any(not operation for operation in perception_operations):
            raise ValueError("perception capability entries require operation")
        _require_unique(
            perception_operations,
            field_name="perception_capability_catalog.operation",
        )
        _require_unique(self.source_audio_refs, field_name="source_audio_refs")
        _require_unique(self.reference_audio_refs, field_name="reference_audio_refs")
        _require_unique(self.reviewed_evidence_ids, field_name="reviewed_evidence_ids")
        unknown_audio_paths = sorted(
            set(self.evidence_path_audio_refs) - set(self.evidence_path_sources)
        )
        if unknown_audio_paths:
            raise ValueError(
                "evidence_path_audio_refs contains paths outside admitted evidence: "
                f"{unknown_audio_paths}"
            )
        bound_audio_refs = {*self.source_audio_refs, *self.reference_audio_refs}
        unbound_path_audio_refs = sorted(
            set(self.evidence_path_audio_refs.values()) - bound_audio_refs
        )
        if unbound_path_audio_refs:
            raise ValueError(
                f"evidence_path_audio_refs contains unbound audio refs: {unbound_path_audio_refs}"
            )
        return self


class MemoQualifiedPlannerTurn(StrictModel):
    """Structured Planner result bound to an unchanged semantic memo."""

    schema_version: Literal["x-editor.memo-qualified-planner-turn/v3"] = (
        "x-editor.memo-qualified-planner-turn/v3"
    )
    decision: Literal[
        "request_perception",
        "execute",
        "complete_from_session_state",
        "cannot_plan",
    ]
    semantic_memo_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_base_state_node_id: str | None = Field(default=None, max_length=256)
    selected_base_audio_ref: str | None = Field(default=None, max_length=256)
    state_transition_kind: (
        Literal[
            "edit_current",
            "branch_from_history",
            "restore_history",
        ]
        | None
    ) = None
    state_completion_kind: Literal["select_existing", "restore_history"] | None = None
    changes_satisfied_by_base: list[str] = Field(default_factory=list, max_length=32)
    perception_jobs: list[PlannerPerceptionJob] = Field(
        default_factory=list,
        max_length=16,
    )
    candidate_operations: list[WorkflowListText] = Field(
        default_factory=list,
        max_length=8,
    )
    plan: AudioEditPlan | None = None
    execution_semantics: PlannerExecutionSemantics | None = None
    tool_selection: PlannerToolSelection | None = None
    missing_information: list[str] = Field(default_factory=list, max_length=64)
    semantic_conflict: str | None = Field(default=None, max_length=2048)
    cannot_plan_reasons: list[PlannerCannotPlanReason] = Field(
        default_factory=list,
        max_length=32,
    )
    decision_summary: str = Field(
        default="",
        max_length=PLANNER_DECISION_SUMMARY_MAX_LENGTH,
    )

    @model_validator(mode="after")
    def validate_decision_payload(self) -> MemoQualifiedPlannerTurn:
        if self.semantic_conflict is not None and self.decision != "cannot_plan":
            raise ValueError("a semantic conflict must produce cannot_plan")
        if self.decision != "cannot_plan" and self.cannot_plan_reasons:
            raise ValueError("typed cannot_plan reasons are valid only for cannot_plan")
        if self.decision == "request_perception":
            if not self.perception_jobs:
                raise ValueError("request_perception requires perception_jobs")
            _require_unique(
                [item.job_id for item in self.perception_jobs],
                field_name="perception_jobs.job_id",
            )
            _require_unique(
                self.candidate_operations,
                field_name="candidate_operations",
            )
            if any(
                value is not None
                for value in (self.plan, self.execution_semantics, self.tool_selection)
            ):
                raise ValueError("request_perception cannot expose executable work products")
            if (self.selected_base_state_node_id is None) != (self.selected_base_audio_ref is None):
                raise ValueError("Perception base state and audio ref must be selected together")
            if self.state_transition_kind is not None:
                raise ValueError(
                    "request_perception selects an evidence base, not a final state transition"
                )
            if self.state_completion_kind is not None or self.changes_satisfied_by_base:
                raise ValueError("request_perception cannot expose a state-only completion")
        elif self.decision == "execute":
            if self.plan is None:
                raise ValueError("execute requires a plan")
            if self.perception_jobs or self.candidate_operations or self.missing_information:
                raise ValueError("execute cannot retain unresolved information")
            if (self.selected_base_state_node_id is None) != (self.selected_base_audio_ref is None):
                raise ValueError("execution base state and audio ref must be selected together")
            if self.state_transition_kind is None:
                raise ValueError("execute requires an explicit state_transition_kind")
            if self.state_completion_kind is not None or self.changes_satisfied_by_base:
                raise ValueError("execute cannot expose a state-only completion")
            if self.tool_selection is not None:
                selections = {item.plan_step_id: item for item in self.tool_selection.steps}
                plan_steps = {item.id: item for item in self.plan.steps}
                if set(selections) != set(plan_steps):
                    raise ValueError(
                        "tool_selection must declare exactly one selection for every plan step"
                    )
                for step_id, selection in selections.items():
                    step = plan_steps[step_id]
                    if selection.selected_operation != step.operation:
                        raise ValueError(
                            "tool_selection operation must match the plan step operation"
                        )
                    for parameter in selection.parameter_selections:
                        found, value = _plan_step_argument_value(
                            step,
                            parameter.argument_path,
                        )
                        if not found or value != parameter.selected_value:
                            raise ValueError(
                                "parameter selection must exactly match its plan argument"
                            )
        elif self.decision == "complete_from_session_state":
            if self.selected_base_state_node_id is None or self.selected_base_audio_ref is None:
                raise ValueError("state-only completion requires a Session Graph node/audio pair")
            if self.state_completion_kind is None:
                raise ValueError("state-only completion requires an explicit selection kind")
            if not self.changes_satisfied_by_base:
                raise ValueError("state-only completion requires satisfied changes")
            if any(
                value is not None
                for value in (
                    self.plan,
                    self.execution_semantics,
                    self.tool_selection,
                    self.state_transition_kind,
                )
            ):
                raise ValueError("state-only completion cannot expose executable work products")
            if self.perception_jobs or self.candidate_operations or self.missing_information:
                raise ValueError("state-only completion cannot retain unresolved work")
        else:
            if (
                not self.missing_information
                and self.semantic_conflict is None
                and not self.cannot_plan_reasons
            ):
                raise ValueError(
                    "cannot_plan requires missing_information, semantic_conflict, or typed reasons"
                )
            if any(
                value is not None
                for value in (
                    self.plan,
                    self.execution_semantics,
                    self.tool_selection,
                    self.selected_base_state_node_id,
                    self.selected_base_audio_ref,
                    self.state_transition_kind,
                    self.state_completion_kind,
                )
            ):
                raise ValueError("cannot_plan cannot expose work products")
            if self.perception_jobs or self.candidate_operations or self.changes_satisfied_by_base:
                raise ValueError("cannot_plan cannot expose Perception work products")
        return self


class QualifiedPlannerTurn(StrictModel):
    """Separated Planner output that cannot redefine Reasoner semantics."""

    schema_version: Literal["x-editor.qualified-planner-turn/v1"] = (
        "x-editor.qualified-planner-turn/v1"
    )
    decision: Literal["request_perception", "execute", "cannot_plan"]
    reasoned_intent_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    perception_intent: PlannerPerceptionIntent | None = None
    plan: AudioEditPlan | None = None
    execution_semantics: PlannerExecutionSemantics | None = None
    tool_selection: PlannerToolSelection | None = None
    missing_information: list[str] = Field(default_factory=list, max_length=64)
    decision_summary: str = Field(default="", max_length=1024)

    @model_validator(mode="after")
    def validate_decision_payload(self) -> QualifiedPlannerTurn:
        if self.decision == "request_perception":
            if self.perception_intent is None:
                raise ValueError("request_perception requires a perception_intent")
            if any(
                value is not None
                for value in (
                    self.plan,
                    self.execution_semantics,
                    self.tool_selection,
                )
            ):
                raise ValueError("request_perception cannot expose executable work products")
        elif self.decision == "execute":
            if any(
                value is None
                for value in (
                    self.plan,
                    self.execution_semantics,
                    self.tool_selection,
                )
            ):
                raise ValueError("execute requires plan, execution_semantics, and tool_selection")
            if self.perception_intent is not None or self.missing_information:
                raise ValueError(
                    "execute cannot retain unresolved perception or missing information"
                )
        else:
            if not self.missing_information:
                raise ValueError("cannot_plan requires missing_information")
            if any(
                value is not None
                for value in (
                    self.perception_intent,
                    self.plan,
                    self.execution_semantics,
                    self.tool_selection,
                )
            ):
                raise ValueError("cannot_plan cannot expose work products")
        return self


class PlannerReasonerTurn(StrictModel):
    """One bounded Planner/Reasoner decision without private chain of thought."""

    schema_version: Literal["x-editor.planner-reasoner-turn/v1"] = (
        "x-editor.planner-reasoner-turn/v1"
    )
    decision: Literal["request_perception", "execute", "cannot_plan"]
    edit_intent: PlannerEditIntent | None = None
    perception_intent: PlannerPerceptionIntent | None = None
    plan: AudioEditPlan | None = None
    execution_semantics: PlannerExecutionSemantics | None = None
    tool_selection: PlannerToolSelection | None = None
    missing_information: list[str] = Field(default_factory=list, max_length=64)
    decision_summary: str = Field(default="", max_length=1024)

    @model_validator(mode="after")
    def validate_decision_payload(self) -> PlannerReasonerTurn:
        if self.decision == "request_perception":
            if self.perception_intent is None:
                raise ValueError("request_perception requires a perception_intent")
            if (
                self.plan is not None
                or self.execution_semantics is not None
                or self.tool_selection is not None
            ):
                raise ValueError("request_perception cannot expose an executable plan")
        elif self.decision == "execute":
            if self.plan is None or self.execution_semantics is None:
                raise ValueError("execute requires a plan and execution_semantics")
            if self.perception_intent is not None or self.missing_information:
                raise ValueError(
                    "execute cannot retain unresolved perception or missing information"
                )
            if self.tool_selection is not None:
                selections = {item.plan_step_id: item for item in self.tool_selection.steps}
                plan_steps = {item.id: item for item in self.plan.steps}
                if set(selections) != set(plan_steps):
                    raise ValueError(
                        "tool_selection must declare exactly one selection for every plan step"
                    )
                for step_id, selection in selections.items():
                    step = plan_steps[step_id]
                    if selection.selected_operation != step.operation:
                        raise ValueError(
                            "tool_selection operation must match the plan step operation"
                        )
                    for parameter in selection.parameter_selections:
                        found, value = _plan_step_argument_value(
                            step,
                            parameter.argument_path,
                        )
                        if not found or value != parameter.selected_value:
                            raise ValueError(
                                "parameter selection must exactly match its plan argument"
                            )
        else:
            if not self.missing_information:
                raise ValueError("cannot_plan requires missing_information")
            if (
                self.perception_intent is not None
                or self.plan is not None
                or self.execution_semantics is not None
                or self.tool_selection is not None
                or self.edit_intent is not None
            ):
                raise ValueError("cannot_plan cannot expose work products")
        return self


class PlannerPostPerceptionExecuteTurn(StrictModel):
    """Second-round execute branch with structurally required work products."""

    schema_version: Literal["x-editor.planner-reasoner-turn/v1"] = (
        "x-editor.planner-reasoner-turn/v1"
    )
    decision: Literal["execute"]
    edit_intent: PlannerEditIntent
    perception_intent: None = None
    plan: AudioEditPlan
    execution_semantics: PlannerExecutionSemantics
    tool_selection: PlannerToolSelection
    missing_information: list[str] = Field(default_factory=list, max_length=0)
    decision_summary: str = Field(default="", max_length=1024)


class PlannerPrePerceptionRequestTurn(StrictModel):
    """First-round evidence request without second-round execution branches."""

    schema_version: Literal["x-editor.planner-reasoner-turn/v1"] = (
        "x-editor.planner-reasoner-turn/v1"
    )
    decision: Literal["request_perception"]
    edit_intent: PlannerEditIntent
    perception_intent: PlannerPerceptionIntent
    plan: None = None
    execution_semantics: None = None
    tool_selection: None = None
    missing_information: list[str] = Field(default_factory=list, max_length=0)
    decision_summary: str = Field(default="", max_length=1024)


class PlannerPrePerceptionCannotPlanTurn(StrictModel):
    """First-round decline when no safe evidence request can be formed."""

    schema_version: Literal["x-editor.planner-reasoner-turn/v1"] = (
        "x-editor.planner-reasoner-turn/v1"
    )
    decision: Literal["cannot_plan"]
    edit_intent: None = None
    perception_intent: None = None
    plan: None = None
    execution_semantics: None = None
    tool_selection: None = None
    missing_information: list[str] = Field(min_length=1, max_length=64)
    decision_summary: str = Field(default="", max_length=1024)


PlannerPrePerceptionTurn = Annotated[
    PlannerPrePerceptionRequestTurn | PlannerPrePerceptionCannotPlanTurn,
    Field(discriminator="decision"),
]


class PlannerPostPerceptionCannotPlanTurn(StrictModel):
    """Second-round decline branch used when reviewed evidence is insufficient."""

    schema_version: Literal["x-editor.planner-reasoner-turn/v1"] = (
        "x-editor.planner-reasoner-turn/v1"
    )
    decision: Literal["cannot_plan"]
    edit_intent: None = None
    perception_intent: None = None
    plan: None = None
    execution_semantics: None = None
    tool_selection: None = None
    missing_information: list[str] = Field(min_length=1, max_length=64)
    decision_summary: str = Field(default="", max_length=1024)


PlannerPostPerceptionTurn = Annotated[
    PlannerPostPerceptionExecuteTurn | PlannerPostPerceptionCannotPlanTurn,
    Field(discriminator="decision"),
]


def planner_reasoner_output_schema(*, post_perception: bool) -> dict[str, Any]:
    """Return the phase-specific provider output schema.

    The canonical runtime model remains ``PlannerReasonerTurn``. The narrower
    second-round schema makes the execute branch's required fields visible to
    JSON-schema-aware providers and to providers that only read the prompt.
    """

    if not post_perception:
        return TypeAdapter(PlannerPrePerceptionTurn).json_schema()
    return TypeAdapter(PlannerPostPerceptionTurn).json_schema()


def _plan_step_argument_value(
    step: AudioEditStep,
    path: str,
) -> tuple[bool, Any]:
    root, _, remainder = path.partition(".")
    current: Any = step.inputs if root == "inputs" else step.params
    if not remainder:
        return False, None
    for part in remainder.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def project_perception_job_to_workflow_request(
    job: PlannerPerceptionJob,
    *,
    edit_objective: str,
    candidate_operations: Collection[str],
) -> dict[str, Any]:
    """Project one provider-authored job without changing its route or meaning."""

    checked = PlannerPerceptionJob.model_validate(job.model_dump(mode="python"))
    checked_candidates = _dedupe_text_values(
        [str(item).strip() for item in candidate_operations if str(item).strip()]
    )
    supporting_audio_refs = list(checked.supporting_audio_refs)
    request_audio_refs = [checked.primary_audio_ref, *supporting_audio_refs]
    inputs: dict[str, Any] = {"input_audio_path": checked.primary_audio_ref}
    if len(supporting_audio_refs) == 1:
        inputs["reference_audio_path"] = supporting_audio_refs[0]
    elif supporting_audio_refs:
        inputs["reference_audio_paths"] = supporting_audio_refs
    job_objective = _perception_job_audit_objective(
        checked,
        parent_objective=edit_objective,
    )
    query_projection = _semantic_span_review_query_projection(
        operation=checked.perception_operation,
        perception_targets=checked.perception_targets,
        target_hints=checked.target_hints,
        evidence_questions=checked.evidence_questions,
        parent_objective=job_objective,
    )
    instruction = _perception_job_instruction(
        checked,
        edit_objective=job_objective,
        effective_query_projection=query_projection,
    )
    params = _perception_job_operation_params(
        checked,
        candidate_operations=checked_candidates,
    )
    if query_projection is not None:
        params["instruction"] = query_projection["effective_evidence_question"]
    metadata: dict[str, Any] = {
        "reasoned_contract": checked.schema_version,
        "primary_audio_ref": checked.primary_audio_ref,
        "perception_job": checked.model_dump(mode="json"),
        "edit_objective": job_objective,
        "target_hints": list(checked.target_hints),
        "perception_targets": [item.model_dump(mode="json") for item in checked.perception_targets],
        "evidence_questions": list(checked.evidence_questions),
        "expected_evidence_fields": list(checked.expected_evidence_fields),
        "temporal_precision": checked.temporal_precision,
        "candidate_operations": list(checked_candidates),
    }
    if query_projection is not None:
        metadata["effective_query_projection"] = query_projection
    return {
        "request_id": checked.job_id,
        "scope": checked.scope,
        "instruction": instruction,
        "source_step_id": None,
        "operation": checked.perception_operation,
        "inputs": inputs,
        "params": params,
        "expected_output": (
            "Reviewed evidence fields: " + ", ".join(checked.expected_evidence_fields)
        ),
        "audio_refs": request_audio_refs,
        "requirements": list(checked.requirements),
        "metadata": metadata,
        "required": True,
    }


def _perception_job_audit_objective(
    job: PlannerPerceptionJob,
    *,
    parent_objective: str,
) -> str:
    """Choose a concise Planner-authored label without rewriting its semantics."""

    if job.evidence_questions:
        return job.evidence_questions[0]
    if job.perception_targets:
        target = job.perception_targets[0]
        return f"Observe {target.kind}: {target.value}"
    if job.target_hints:
        return job.target_hints[0]
    if job.requirements:
        return job.requirements[0]
    return parent_objective.strip()


def project_perception_jobs_to_workflow_requests(
    jobs: Collection[PlannerPerceptionJob],
    *,
    edit_objective: str,
    candidate_operations: Collection[str],
) -> list[dict[str, Any]]:
    """Project one gateway batch into independently auditable requests."""

    checked_jobs = [
        PlannerPerceptionJob.model_validate(item.model_dump(mode="python")) for item in jobs
    ]
    _require_unique(
        [item.job_id for item in checked_jobs],
        field_name="perception_jobs.job_id",
    )
    return [
        project_perception_job_to_workflow_request(
            job,
            edit_objective=edit_objective,
            candidate_operations=candidate_operations,
        )
        for job in checked_jobs
    ]


def project_perception_intent_to_workflow_request(
    intent: PlannerPerceptionIntent,
    *,
    observation_job: PlannerAudioObservationJob | None = None,
) -> dict[str, Any]:
    """Project the legacy composite Planner's v1 Perception intent.

    Memo-qualified Planner output does not use this compatibility path. The
    projection copies the v1 route and fields exactly; it performs no semantic
    repair and remains covered for the legacy composite Planner only.
    """

    checked = PlannerPerceptionIntent.model_validate(intent.model_dump(mode="python"))
    job = (
        PlannerAudioObservationJob.model_validate(observation_job.model_dump(mode="python"))
        if observation_job is not None
        else None
    )
    primary_audio_ref = job.audio_ref if job is not None else checked.primary_audio_ref
    supporting_audio_refs = (
        list(job.supporting_audio_refs)
        if job is not None
        else [item for item in checked.audio_refs if item != checked.primary_audio_ref]
    )
    request_audio_refs = list(dict.fromkeys([primary_audio_ref, *supporting_audio_refs]))
    inputs: dict[str, Any] = {"input_audio_path": primary_audio_ref}
    if len(supporting_audio_refs) == 1:
        inputs["reference_audio_path"] = supporting_audio_refs[0]
    elif supporting_audio_refs:
        inputs["reference_audio_paths"] = supporting_audio_refs
    expected_fields = (
        list(job.expected_evidence_fields)
        if job is not None and job.expected_evidence_fields
        else list(checked.expected_evidence_fields)
    )
    query_projection = _semantic_span_review_query_projection(
        operation=checked.perception_operation,
        perception_targets=checked.perception_targets,
        target_hints=checked.target_hints,
        evidence_questions=checked.evidence_questions,
        parent_objective=checked.edit_objective,
    )
    instruction = _perception_instruction(
        checked,
        observation_job=job,
        effective_query_projection=query_projection,
    )
    params = _perception_operation_params(checked)
    if query_projection is not None:
        params["instruction"] = query_projection["effective_evidence_question"]
    metadata: dict[str, Any] = {
        "reasoned_contract": checked.schema_version,
        "primary_audio_ref": primary_audio_ref,
        "observation_job": (job.model_dump(mode="json") if job is not None else None),
        "edit_objective": checked.edit_objective,
        "target_hints": list(checked.target_hints),
        "perception_targets": [item.model_dump(mode="json") for item in checked.perception_targets],
        "evidence_questions": list(checked.evidence_questions),
        "expected_evidence_fields": expected_fields,
        "temporal_precision": checked.temporal_precision,
        "candidate_operations": list(checked.candidate_operations),
        "ambiguities": list(checked.ambiguities),
    }
    if query_projection is not None:
        metadata["effective_query_projection"] = query_projection
    return {
        "request_id": job.job_id if job is not None else checked.request_id,
        "scope": checked.scope,
        "instruction": instruction,
        "source_step_id": None,
        "operation": checked.perception_operation,
        "inputs": inputs,
        "params": params,
        "expected_output": (
            "Reviewed evidence fields: " + ", ".join(expected_fields)
            if expected_fields
            else "Reviewed evidence relevant to the declared perception targets."
        ),
        "audio_refs": request_audio_refs,
        "requirements": (
            list(job.requirements)
            if job is not None and job.requirements
            else list(checked.requirements)
        ),
        "metadata": metadata,
        "required": job.required if job is not None else True,
    }


def project_perception_intent_to_workflow_requests(
    intent: PlannerPerceptionIntent,
) -> list[dict[str, Any]]:
    """Project one intent into independently auditable audio-scoped requests."""

    checked = PlannerPerceptionIntent.model_validate(intent.model_dump(mode="python"))
    if not checked.observation_jobs:
        return [project_perception_intent_to_workflow_request(checked)]
    return [
        project_perception_intent_to_workflow_request(
            checked,
            observation_job=job,
        )
        for job in checked.observation_jobs
    ]


def validate_plan_semantic_bindings(
    semantics: PlannerExecutionSemantics,
    *,
    plan: AudioEditPlan,
    available_evidence_paths: Mapping[str, Collection[str]],
    evidence_required_step_ids: Collection[str],
    available_evidence_values: Mapping[str, Mapping[str, Any]] | None = None,
    binding_normalization_mode: Literal[
        "explicit_only",
        "canonical_alias",
        "exact_unique",
    ] = "exact_unique",
    irrelevant_binding_mode: Literal["validate", "drop"] = "validate",
) -> PlannerExecutionSemantics:
    """Validate second-round bindings against authoritative plan and evidence.

    ``evidence_required_step_ids`` must be computed by the runtime admission or
    tool contracts. It is intentionally not trusted from Planner output; doing
    so would let an omitted binding silently redefine which steps need evidence.
    """

    checked = PlannerExecutionSemantics.model_validate(semantics.model_dump(mode="python"))
    checked_plan = AudioEditPlan.model_validate(plan.model_dump(mode="python"))
    steps_by_id = _unique_plan_steps(checked_plan)
    required_step_ids = _normalized_required_step_ids(evidence_required_step_ids)
    unknown_required = sorted(set(required_step_ids) - set(steps_by_id))
    if unknown_required:
        raise ValueError(f"evidence-required step ids are absent from the plan: {unknown_required}")
    perception_required = sorted(
        step_id
        for step_id in required_step_ids
        if steps_by_id[step_id].operation.startswith(_PERCEPTION_OPERATION_PREFIXES)
    )
    if perception_required:
        raise ValueError(
            f"semantic bindings apply to edit steps, not perception steps: {perception_required}"
        )

    if irrelevant_binding_mode == "drop":
        checked = checked.model_copy(
            update={
                "bindings": [
                    binding
                    for binding in checked.bindings
                    if binding.plan_step_id in set(required_step_ids)
                ]
            },
            deep=True,
        )
    binding_ids = {binding.plan_step_id for binding in checked.bindings}
    unknown_bindings = sorted(binding_ids - set(steps_by_id))
    if unknown_bindings:
        raise ValueError(f"semantic bindings reference absent plan steps: {unknown_bindings}")
    missing = sorted(set(required_step_ids) - binding_ids)
    if missing:
        raise ValueError(f"semantic binding coverage mismatch: missing={missing}")

    evidence_index = _validated_evidence_index(available_evidence_paths)
    evidence_values = available_evidence_values or {}
    normalized_bindings: list[PlanStepSemanticBinding] = []
    for raw_binding in checked.bindings:
        is_required = raw_binding.plan_step_id in set(required_step_ids)
        binding = _normalize_binding_argument_sources(
            raw_binding,
            step=steps_by_id[raw_binding.plan_step_id],
            evidence_values=evidence_values,
            mode=binding_normalization_mode,
        )
        step = steps_by_id[binding.plan_step_id]
        if binding.operation != step.operation:
            raise ValueError(
                f"semantic binding operation mismatch for {binding.plan_step_id}: "
                f"{binding.operation!r} != {step.operation!r}"
            )
        if binding.unresolved_requirements and (
            is_required or irrelevant_binding_mode == "validate"
        ):
            raise ValueError(
                f"semantic binding for {binding.plan_step_id} has unresolved "
                f"requirements: {binding.unresolved_requirements}"
            )
        for reference in binding.evidence_paths:
            _validate_available_evidence_reference(reference, evidence_index)
        for source in binding.argument_sources:
            argument_value = _resolve_argument_path(step, source.argument_path)
            if source.evidence_id is not None and source.evidence_path is not None:
                _validate_available_evidence_reference(
                    EvidencePathReference(
                        evidence_id=source.evidence_id,
                        field_path=source.evidence_path,
                    ),
                    evidence_index,
                )
                if source.source_kind == "perception_evidence":
                    source_values = evidence_values.get(source.evidence_id, {})
                    if source.evidence_path not in source_values:
                        raise ValueError(
                            "direct evidence argument source has no authoritative "
                            f"value: {source.evidence_id}:{source.evidence_path}"
                        )
                    if not _same_json_value(
                        argument_value,
                        source_values[source.evidence_path],
                    ):
                        raise ValueError(
                            "plan argument does not equal its direct evidence value: "
                            f"{binding.plan_step_id}:{source.argument_path}"
                        )
        if is_required and not any(
            source.source_kind in {"perception_evidence", "derived_from_evidence"}
            for source in binding.argument_sources
        ):
            raise ValueError(
                "an evidence-dependent semantic binding requires at least one "
                f"unambiguous evidence-backed argument source: {binding.plan_step_id}"
            )
        unverified_derivations = sorted(
            source.argument_path
            for source in binding.argument_sources
            if source.source_kind == "derived_from_evidence"
        )
        if unverified_derivations:
            raise ValueError(
                "derived_from_evidence requires a registered deterministic "
                "transform contract before execution: "
                f"{unverified_derivations}"
            )
        required_argument_paths = _required_evidence_argument_paths(step)
        boundary_shape_errors = _local_span_boundary_shape_errors(step)
        if is_required and boundary_shape_errors:
            raise ValueError(
                "local span operation has incomplete boundaries for "
                f"{binding.plan_step_id}: {boundary_shape_errors}"
            )
        bound_argument_paths = {
            source.argument_path
            for source in binding.argument_sources
            if source.source_kind
            in {
                "perception_evidence",
                "derived_from_evidence",
            }
        }
        missing_argument_paths = sorted(required_argument_paths - bound_argument_paths)
        if is_required and missing_argument_paths:
            raise ValueError(
                "semantic binding omits required local boundary sources for "
                f"{binding.plan_step_id}: {missing_argument_paths}"
            )
        normalized_bindings.append(binding)
    return checked.model_copy(update={"bindings": normalized_bindings}, deep=True)


def _perception_job_instruction(
    job: PlannerPerceptionJob,
    *,
    edit_objective: str,
    effective_query_projection: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        f"Edit objective: {edit_objective}",
        f"Primary audio ref: {job.primary_audio_ref}",
        f"Temporal precision: {job.temporal_precision}",
        "Expected evidence fields: " + ", ".join(job.expected_evidence_fields),
    ]
    if job.perception_targets:
        lines.append(
            "Perception targets: "
            + "; ".join(f"{item.kind}={item.value}" for item in job.perception_targets)
        )
    if job.supporting_audio_refs:
        lines.append("Supporting audio refs: " + ", ".join(job.supporting_audio_refs))
    if job.target_hints:
        lines.append("Target clues: " + "; ".join(job.target_hints))
    if job.evidence_questions:
        lines.append("Evidence questions: " + "; ".join(job.evidence_questions))
    if effective_query_projection is not None:
        lines.extend(_effective_query_instruction_lines(effective_query_projection))
    if job.requirements:
        lines.append("Audio-scoped requirements: " + "; ".join(job.requirements))
    return "\n".join(lines)


def _perception_job_operation_params(
    job: PlannerPerceptionJob,
    *,
    candidate_operations: Collection[str],
) -> dict[str, Any]:
    """Derive only route-mechanical flags from explicitly authored job fields."""

    event_queries = _perception_event_queries(
        job.perception_targets,
        job.evidence_questions,
    )
    target_kinds = {item.kind for item in job.perception_targets}
    expected_evidence_roots = {
        _expected_evidence_root(field) for field in job.expected_evidence_fields
    }
    local_span_operations = {
        "artifact_ops.extract_spans",
        "artifact_ops.remove_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.beep_spans",
    }
    requires_local_boundaries = bool(
        job.scope == "targeted"
        and job.temporal_precision in {"segment", "word"}
        and set(candidate_operations) & local_span_operations
    )
    if job.perception_operation == "perception.speech_context":
        return {
            "need_source_text": "source_text" in expected_evidence_roots,
            "need_words": bool(
                job.temporal_precision == "word" or "word_timestamps" in expected_evidence_roots
            ),
            "need_speakers": bool(
                {"speakers", "speaker_segments", "speaker_profiles"} & expected_evidence_roots
            ),
            "allow_open_omni": False,
            **({"recognition_context": job.recognition_context} if job.recognition_context else {}),
        }
    if job.perception_operation == "perception.audio_fallback":
        coarse_audio_property_only = (
            job.temporal_precision == "coarse"
            and target_kinds == {"audio_property"}
        )
        return {
            "need_source_text": "source_text" in expected_evidence_roots,
            "need_words": bool(
                job.temporal_precision == "word" or "word_timestamps" in expected_evidence_roots
            ),
            "need_speakers": bool({"speakers", "speaker_segments"} & expected_evidence_roots),
            "need_speaker_profiles": "speaker_profiles" in expected_evidence_roots,
            "need_segment_descriptions": bool(
                not coarse_audio_property_only
                and {
                    "segments",
                    "segment_descriptions",
                    "candidate_segments",
                    "selected_spans",
                }
                & expected_evidence_roots
            ),
            "allow_open_omni": True,
        }
    if job.perception_operation == "perception.speaker_attributed_review":
        # Acquiring missing voice properties is independent of revising an
        # existing transcript. Do not silently disable it with candidate review.
        if "speaker_profiles" in expected_evidence_roots:
            return {"supplement_speaker_attributes": True}
        return {}
    if job.perception_operation in {
        "perception.flexsed_events",
        "flexsed.detect_events",
    }:
        return {"events": event_queries}
    if job.perception_operation == "perception.semantic_span_review":
        return {
            "need_speakers": "speaker_hint" in target_kinds,
            "semantic_targets": [item.model_dump(mode="json") for item in job.perception_targets],
        }
    if job.perception_operation != "perception.timestamped_caption":
        return {}
    params: dict[str, Any] = {
        "workflow_scope": job.scope,
        "temporal_precision": job.temporal_precision,
        "require_local_boundaries": requires_local_boundaries,
    }
    event_labels = [item.value for item in job.perception_targets if item.kind == "event_label"]
    if event_labels:
        params["event_labels"] = event_labels
    return params


def _perception_instruction(
    intent: PlannerPerceptionIntent,
    *,
    observation_job: PlannerAudioObservationJob | None = None,
    effective_query_projection: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        f"Edit objective: {intent.edit_objective}",
        "Perception targets: "
        + "; ".join(f"{item.kind}={item.value}" for item in intent.perception_targets),
        f"Temporal precision: {intent.temporal_precision}",
    ]
    if intent.target_hints:
        lines.append("Target clues: " + "; ".join(intent.target_hints))
    if intent.evidence_questions:
        lines.append("Evidence questions: " + "; ".join(intent.evidence_questions))
    if effective_query_projection is not None:
        lines.extend(_effective_query_instruction_lines(effective_query_projection))
    if intent.expected_evidence_fields:
        lines.append("Expected evidence fields: " + ", ".join(intent.expected_evidence_fields))
    if observation_job is not None:
        lines.append(f"Observed audio ref: {observation_job.audio_ref}")
        if observation_job.supporting_audio_refs:
            lines.append(
                "Supporting audio refs: " + ", ".join(observation_job.supporting_audio_refs)
            )
        if observation_job.requirements:
            lines.append("Audio-scoped requirements: " + "; ".join(observation_job.requirements))
    if intent.ambiguities:
        lines.append("Ambiguities to resolve: " + "; ".join(intent.ambiguities))
    return "\n".join(lines)


def _perception_operation_params(
    intent: PlannerPerceptionIntent,
) -> dict[str, Any]:
    event_queries = _perception_event_queries(
        intent.perception_targets,
        intent.evidence_questions,
    )
    target_kinds = {item.kind for item in intent.perception_targets}
    expected_evidence_roots = {
        _expected_evidence_root(field) for field in intent.expected_evidence_fields
    }
    local_span_operations = {
        "artifact_ops.extract_spans",
        "artifact_ops.remove_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.beep_spans",
    }
    requires_local_boundaries = bool(
        intent.scope == "targeted"
        and intent.temporal_precision in {"segment", "word"}
        and set(intent.candidate_operations) & local_span_operations
    )
    if intent.perception_operation == "perception.speech_context":
        return {
            "need_source_text": True,
            "need_words": bool(
                intent.temporal_precision == "word"
                or "word_timestamps" in expected_evidence_roots
                or set(intent.candidate_operations) & local_span_operations
            ),
            "need_speakers": False,
            "allow_open_omni": False,
        }
    if intent.perception_operation == "perception.speaker_attributed_review":
        return {}
    if intent.perception_operation == "perception.semantic_span_review":
        action_by_operation = {
            "artifact_ops.remove_spans": "remove_spans",
            "artifact_ops.mute_spans": "mute_spans",
            "artifact_ops.beep_spans": "beep_spans",
            "artifact_ops.speaker_scoped_edit": "speaker_scoped_edit",
        }
        action = next(
            (
                action_by_operation[operation]
                for operation in intent.candidate_operations
                if operation in action_by_operation
            ),
            "extract_spans",
        )
        return {
            "action": action,
            "need_speakers": "speaker_hint" in target_kinds,
        }
    if intent.perception_operation == "perception.audio_fallback":
        return {
            "need_source_text": bool({"quoted_text", "speaker_hint"} & target_kinds),
            "need_words": intent.temporal_precision == "word",
            "need_speakers": "speaker_hint" in target_kinds,
            "need_speaker_profiles": bool({"speaker_hint", "semantic_property"} & target_kinds),
            "need_segment_descriptions": bool(
                {"semantic_property", "audio_property", "event_label"} & target_kinds
            ),
            "allow_open_omni": True,
        }
    if intent.perception_operation in {
        "perception.flexsed_events",
        "flexsed.detect_events",
    }:
        return {"events": event_queries}
    if intent.perception_operation != "perception.timestamped_caption":
        return {}
    params: dict[str, Any] = {
        "workflow_scope": intent.scope,
        "temporal_precision": intent.temporal_precision,
        "require_local_boundaries": requires_local_boundaries,
    }
    event_labels = [item.value for item in intent.perception_targets if item.kind == "event_label"]
    if event_labels:
        params["event_labels"] = event_labels
    return params


def _perception_event_queries(
    targets: Collection[PlannerPerceptionTarget],
    evidence_questions: Collection[str],
) -> list[str]:
    """Return exact Planner-authored detector queries without semantic parsing."""

    queries = [item.value for item in targets if item.kind in _EVENT_QUERY_TARGET_KINDS]
    queries.extend(str(item).strip() for item in evidence_questions if str(item).strip())
    return _dedupe_text_values(queries)


def _semantic_span_review_query_projection(
    *,
    operation: str,
    perception_targets: Collection[PlannerPerceptionTarget],
    target_hints: Collection[str],
    evidence_questions: Collection[str],
    parent_objective: str,
) -> dict[str, Any] | None:
    """Expose one auditable semantic query without interpreting its content."""

    if operation != "perception.semantic_span_review":
        return None

    questions = [str(item) for item in evidence_questions]
    if questions:
        effective_question = "\n".join(questions)
        projection_kind = "planner_evidence_questions_verbatim"
        source_fields = ["evidence_questions"]
        planner_authored_question = True
        objective_copied_verbatim = False
    else:
        rendered_targets = [f"{item.kind}={item.value}" for item in perception_targets]
        hints = [str(item) for item in target_hints]
        if rendered_targets or hints:
            effective_question = "\n".join([*rendered_targets, *hints])
            projection_kind = "deterministic_planner_targets_and_hints"
            source_fields = [
                field
                for field, values in (
                    ("perception_targets", rendered_targets),
                    ("target_hints", hints),
                )
                if values
            ]
            planner_authored_question = False
            objective_copied_verbatim = False
        else:
            if not isinstance(parent_objective, str) or not parent_objective.strip():
                raise ValueError(
                    "semantic_span_review requires a non-empty explicit query or "
                    "parent edit objective"
                )
            effective_question = parent_objective
            projection_kind = "deterministic_parent_objective_verbatim"
            source_fields = ["parent_edit_objective"]
            planner_authored_question = False
            objective_copied_verbatim = True

    return {
        "schema_version": "x-editor.semantic-span-query-projection/v1",
        "effective_evidence_question": effective_question,
        "projection_kind": projection_kind,
        "source_fields": source_fields,
        "planner_authored_question": planner_authored_question,
        "objective_copied_verbatim": objective_copied_verbatim,
        "semantic_inference_applied": False,
    }


def _effective_query_instruction_lines(
    projection: Mapping[str, Any],
) -> list[str]:
    effective_question = str(projection["effective_evidence_question"])
    projection_kind = str(projection["projection_kind"])
    return [
        "Effective semantic evidence question: " + effective_question,
        (
            "Effective query provenance: "
            f"{projection_kind}; semantic_inference_applied=false; "
            "this field is a runtime projection and does not replace the Planner's "
            "perception_targets, target_hints, or evidence_questions."
        ),
    ]


def _expected_evidence_root(field_path: str) -> str:
    normalized = str(field_path).strip()
    for prefix in ("facts.", "perception_report."):
        normalized = normalized.removeprefix(prefix)
    return normalized.split(".", 1)[0].split("[", 1)[0]


def _require_unique(values: Collection[str], *, field_name: str) -> None:
    normalized = [value.casefold() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique values")


def _dedupe_text_values(values: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            deduped.append(value)
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_binding_argument_sources(
    binding: PlanStepSemanticBinding,
    *,
    step: AudioEditStep,
    evidence_values: Mapping[str, Mapping[str, Any]],
    mode: Literal[
        "explicit_only",
        "canonical_alias",
        "exact_unique",
    ] = "exact_unique",
) -> PlanStepSemanticBinding:
    """Apply only deterministic, unambiguous provenance corrections."""

    normalized_sources: list[SemanticArgumentSource] = []
    source_keys: set[tuple[str, str, str | None, str | None]] = set()
    for source in binding.argument_sources:
        argument_path = source.argument_path
        if mode in {"canonical_alias", "exact_unique"}:
            argument_path = _canonical_argument_path(
                step,
                source.argument_path,
            )
        if (
            mode == "exact_unique"
            and not _argument_path_exists(step, argument_path)
            and _is_non_executable_span_annotation_path(
                step,
                argument_path,
            )
        ):
            # A model may emit extra provenance for descriptive evidence that
            # is not an executor argument. Drop only that unusable row; the
            # required-boundary checks below still demand exact evidence for
            # every executable span leaf.
            continue
        normalized = source.model_copy(
            update={"argument_path": argument_path},
            deep=True,
        )
        key = (
            normalized.argument_path,
            normalized.source_kind,
            normalized.evidence_id,
            normalized.evidence_path,
        )
        if key in source_keys:
            continue
        source_keys.add(key)
        normalized_sources.append(normalized)

    if mode != "exact_unique":
        return binding.model_copy(
            update={"argument_sources": normalized_sources},
            deep=True,
        )

    bound_references = {
        (item.evidence_id, item.evidence_path)
        for item in normalized_sources
        if item.source_kind in {"perception_evidence", "derived_from_evidence"}
    }
    for reference in binding.evidence_paths:
        reference_key = (reference.evidence_id, reference.field_path)
        if reference_key in bound_references:
            continue
        values = evidence_values.get(reference.evidence_id, {})
        if reference.field_path not in values:
            continue
        matches = _matching_argument_paths(
            step,
            evidence_path=reference.field_path,
            evidence_value=values[reference.field_path],
        )
        if len(matches) != 1:
            continue
        candidate = SemanticArgumentSource(
            argument_path=matches[0],
            source_kind="perception_evidence",
            evidence_id=reference.evidence_id,
            evidence_path=reference.field_path,
        )
        key = (
            candidate.argument_path,
            candidate.source_kind,
            candidate.evidence_id,
            candidate.evidence_path,
        )
        if key not in source_keys:
            source_keys.add(key)
            normalized_sources.append(candidate)

    return binding.model_copy(
        update={"argument_sources": normalized_sources},
        deep=True,
    )


def _canonical_argument_path(step: AudioEditStep, argument_path: str) -> str:
    parts = argument_path.split(".")
    current: Any = step.inputs if parts[0] == "inputs" else step.params
    canonical = [parts[0]]
    for depth, part in enumerate(parts[1:], start=1):
        if isinstance(current, Mapping):
            selected = part
            if selected not in current and depth == 1:
                registered_alias = _REGISTERED_ARGUMENT_ALIASES.get((parts[0], part))
                if registered_alias in current:
                    selected = registered_alias
            canonical.append(selected)
            if selected not in current:
                canonical.extend(parts[len(canonical) :])
                break
            current = current[selected]
            continue
        if isinstance(current, list) and part.isdigit():
            canonical.append(part)
            index = int(part)
            if index >= len(current):
                canonical.extend(parts[len(canonical) :])
                break
            current = current[index]
            continue
        canonical.extend(parts[len(canonical) :])
        break
    return ".".join(canonical)


def _matching_argument_paths(
    step: AudioEditStep,
    *,
    evidence_path: str,
    evidence_value: Any,
) -> list[str]:
    inferable_leaves = _EXACT_INFERENCE_ARGUMENT_LEAVES_BY_OPERATION.get(
        step.operation,
        set(),
    )
    evidence_leaf = evidence_path.rsplit(".", 1)[-1].casefold()
    if evidence_leaf not in inferable_leaves:
        return []
    matches: list[str] = []
    for argument_path, argument_value in _argument_leaf_values(step):
        if _is_audio_argument_path(argument_path):
            continue
        argument_leaf = argument_path.rsplit(".", 1)[-1].casefold()
        if argument_leaf != evidence_leaf:
            continue
        if _same_json_value(argument_value, evidence_value):
            matches.append(argument_path)
    return matches


def _argument_leaf_values(step: AudioEditStep) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}")
            return
        values.append((path, value))

    visit(step.inputs, "inputs")
    visit(step.params, "params")
    return values


def _required_evidence_argument_paths(step: AudioEditStep) -> set[str]:
    inferable_leaves = _EXACT_INFERENCE_ARGUMENT_LEAVES_BY_OPERATION.get(
        step.operation,
        set(),
    )
    return {
        argument_path
        for argument_path, _ in _argument_leaf_values(step)
        if argument_path.rsplit(".", 1)[-1].casefold() in inferable_leaves
    }


def _local_span_boundary_shape_errors(
    step: AudioEditStep,
) -> list[str]:
    if step.operation not in _EXACT_INFERENCE_ARGUMENT_LEAVES_BY_OPERATION:
        return []
    spans = step.params.get("spans")
    if spans is None:
        spans = step.params.get("selected_spans")
    if spans is None:
        spans = step.inputs.get("spans")
    if spans is None:
        spans = step.inputs.get("selected_spans")
    if not isinstance(spans, list) or not spans:
        return ["spans must be a non-empty array"]
    errors: list[str] = []
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            errors.append(f"spans.{index} must be an object")
            continue
        if "start_sec" not in span or "end_sec" not in span:
            errors.append(f"spans.{index} requires start_sec and end_sec")
    return errors


def _is_audio_argument_path(argument_path: str) -> bool:
    leaf = argument_path.rsplit(".", 1)[-1].casefold()
    return "audio" in leaf and leaf.endswith(("path", "audio"))


def _same_json_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _validate_field_path(value: str, *, label: str) -> None:
    allowed_punctuation = "._-[](),= :"
    if (
        not value
        or value.startswith(".")
        or value.endswith(".")
        or ".." in value
        or any(character in "\r\n\t{}\\" for character in value)
        or any(
            not (
                character.isalnum()
                or character in allowed_punctuation
                or (character in "'\"" and _quote_is_inside_identity_selector(value, index))
            )
            for index, character in enumerate(value)
        )
    ):
        raise ValueError(f"{label} is not a valid dotted field path: {value!r}")


def _quote_is_inside_identity_selector(value: str, index: int) -> bool:
    left = value.rfind("[", 0, index)
    right = value.find("]", index + 1)
    if left < 0 or right < 0:
        return False
    selector = value[left + 1 : right]
    return "=" in selector and "(" in selector and ")" in selector


def _validate_argument_path(value: str) -> None:
    _validate_field_path(value, label="argument path")
    if not value.startswith(("inputs.", "params.")):
        raise ValueError("argument_path must be under inputs.* or params.*")


def _unique_plan_steps(plan: AudioEditPlan) -> dict[str, AudioEditStep]:
    steps_by_id = {step.id: step for step in plan.steps}
    if len(steps_by_id) != len(plan.steps):
        raise ValueError("plan step ids must be unique for semantic binding")
    return steps_by_id


def _normalized_required_step_ids(values: Collection[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("evidence_required_step_ids must be a collection of step ids")
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError("evidence_required_step_ids cannot contain blanks")
    if len(normalized) != len(set(normalized)):
        raise ValueError("evidence_required_step_ids must be unique")
    return normalized


def _validated_evidence_index(
    available_evidence_paths: Mapping[str, Collection[str]],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for raw_evidence_id, raw_paths in available_evidence_paths.items():
        evidence_id = str(raw_evidence_id).strip()
        if not evidence_id:
            raise ValueError("available evidence ids cannot be blank")
        if isinstance(raw_paths, (str, bytes)):
            raise TypeError(f"available paths for {evidence_id!r} must be a collection of paths")
        paths = {str(path).strip() for path in raw_paths}
        if not paths or any(not path for path in paths):
            raise ValueError(f"available evidence {evidence_id!r} must expose non-empty paths")
        for path in paths:
            _validate_field_path(path, label="available evidence path")
        index[evidence_id] = paths
    return index


def _validate_available_evidence_reference(
    reference: EvidencePathReference,
    evidence_index: Mapping[str, set[str]],
) -> None:
    paths = evidence_index.get(reference.evidence_id)
    if paths is None:
        raise ValueError(f"unknown evidence id: {reference.evidence_id}")
    if reference.field_path not in paths:
        raise ValueError(
            f"evidence path is outside {reference.evidence_id}: {reference.field_path}"
        )


def _resolve_argument_path(step: AudioEditStep, argument_path: str) -> Any:
    parts = argument_path.split(".")
    current: Any = step.inputs if parts[0] == "inputs" else step.params
    for part in parts[1:]:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        available = (
            sorted(str(key) for key in current)
            if isinstance(current, Mapping)
            else [str(index) for index in range(len(current))]
            if isinstance(current, list)
            else []
        )
        raise ValueError(
            f"argument path is outside plan step {step.id}: {argument_path}; "
            f"available children at {part!r}: {available}"
        )
    return current


def _argument_path_exists(step: AudioEditStep, argument_path: str) -> bool:
    try:
        _resolve_argument_path(step, argument_path)
    except ValueError:
        return False
    return True


def _is_non_executable_span_annotation_path(
    step: AudioEditStep,
    argument_path: str,
) -> bool:
    parts = argument_path.split(".")
    if (
        len(parts) != 4
        or parts[0] not in {"inputs", "params"}
        or parts[1] not in {"spans", "selected_spans"}
        or not parts[2].isdigit()
        or parts[3] in {"start_sec", "end_sec"}
    ):
        return False
    parent_path = ".".join(parts[:-1])
    try:
        parent = _resolve_argument_path(step, parent_path)
    except ValueError:
        return False
    return isinstance(parent, Mapping)


__all__ = [
    "ArgumentSourceKind",
    "EvidencePathReference",
    "MemoQualifiedPlannerInput",
    "MemoQualifiedPlannerTurn",
    "PlanStepSemanticBinding",
    "PlannerAudioObservationJob",
    "PlannerCandidateOperationContract",
    "PlannerExecutionSemantics",
    "PlannerLegalArgumentContract",
    "PlannerPerceptionIntent",
    "PlannerPerceptionJob",
    "PlannerPerceptionTarget",
    "PlannerReasonerTurn",
    "QualifiedPlannerInput",
    "QualifiedPlannerTurn",
    "SemanticArgumentSource",
    "TemporalPrecision",
    "project_perception_intent_to_workflow_request",
    "project_perception_intent_to_workflow_requests",
    "project_perception_job_to_workflow_request",
    "project_perception_jobs_to_workflow_requests",
    "validate_plan_semantic_bindings",
]
