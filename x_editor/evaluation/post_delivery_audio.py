"""Benchmark-neutral Qwen3-Omni scoring for committed audio deliveries.

This module is deliberately downstream of the unified benchmark runner. It
reads a completed run, verifies the frozen selection and delivery lineage, and
writes scores outside the run directory. It never imports or calls workflow,
Verifier, ArtifactStore, or delivery components.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import soundfile as sf
from pydantic import Field, ValidationError, field_validator, model_validator

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.core.agentic_mode import (
    DEFAULT_QWEN3_OMNI_MODEL,
    QWEN3_OMNI_PROVIDER,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.final_state_projection import (
    FinalStateProjection,
    project_final_session_payload,
)
from x_editor.core.schemas import StrictModel
from x_editor.evaluation.effect_profiles import (
    EvaluationEffectContract,
    EvaluationProperty,
    FrozenObjectiveEffect,
    OperationEvaluationEffectProfile,
    load_operation_evaluation_effect_catalog,
    resolve_evaluation_effect_contract,
)
from x_editor.evaluation.flow_reporting import build_post_delivery_flow_report
from x_editor.evaluation.interval_alignment import (
    DEFAULT_BOUNDED_MAXIMUM_EDIT_FRACTION,
    DEFAULT_BOUNDED_MINIMUM_TOKEN_COUNT,
    DEFAULT_MAXIMUM_BOUNDARY_PADDING_SEC,
    DEFAULT_MAXIMUM_WORD_OVERLAP_SEC,
    DEFAULT_MINIMUM_ALIGNMENT_COVERAGE,
    DEFAULT_MINIMUM_TOKEN_COUNT,
    ScorerInterval,
    recover_local_interval_alignment,
)
from x_editor.evaluation.roles import build_evaluator_role_contract
from x_editor.evaluation.wording_evidence import (
    IndependentWordingObservation,
    SpeechPerceptionWordingEvidenceProvider,
    WordingEvidenceProvider,
    WordingPairEvidence,
    build_wording_pair_evidence,
)

SIDECAR_SPEC_KEY = "post_delivery_audio_rubric"
AUDIO_TEMPORAL_LINEAGE_KEY = "audio_temporal_lineage"
DEFAULT_CROP_MARGIN_SEC = 0.25
RubricDimension = Literal[
    "edited_region_instruction_following",
    "edited_region_quality",
    "unedited_region_consistency",
]
AtomicRubricId = Literal[
    "edited_instruction_observable",
    "edited_intelligibility",
    "edited_naturalness",
    "edited_boundary_continuity",
    "edited_artifact_absence",
    "unedited_wording_consistency",
    "unedited_speaker_consistency",
    "unedited_background_consistency",
    "unedited_timing_consistency",
]
ScoreStatus = Literal[
    "dry_run",
    "pass",
    "fail",
    "abstain",
    "incomplete",
    "invalid",
    "provider_failure",
    "not_applicable",
]
_DIMENSIONS: tuple[RubricDimension, ...] = (
    "edited_region_instruction_following",
    "edited_region_quality",
    "unedited_region_consistency",
)
_INFERRED_UNEDITED_BOUNDARY_GUARD_SEC = 0.2
_PRESERVATION_PROPERTIES: dict[AtomicRubricId, EvaluationProperty] = {
    "unedited_wording_consistency": "wording",
    "unedited_speaker_consistency": "speaker_identity",
    "unedited_background_consistency": "acoustic_properties",
    "unedited_timing_consistency": "local_timing",
}
_WORDING_RUBRICS = frozenset(("edited_instruction_observable", "unedited_wording_consistency"))


@dataclass(frozen=True)
class _AtomicRubricDefinition:
    atomic_rubric_id: AtomicRubricId
    dimension: RubricDimension
    observable: str
    question: str


_ATOMIC_RUBRIC_CATALOG_VERSION = "x-editor.post-delivery-atomic-rubrics/v1"
SCORER_PROTOCOL_SCHEMA_VERSION = "x-editor.post-delivery-audio-scorer-protocol/v14"
SCORING_CONTEXT_PROTOCOL_VERSION = "x-editor.post-delivery-scoring-context/v13"

# Preserve the private name for older callers while keeping freeze manifests on
# the same public identity as the scorer implementation.
_SCORING_CONTEXT_PROTOCOL_VERSION = SCORING_CONTEXT_PROTOCOL_VERSION
_ATOMIC_RUBRICS: tuple[_AtomicRubricDefinition, ...] = (
    _AtomicRubricDefinition(
        atomic_rubric_id="edited_instruction_observable",
        dimension="edited_region_instruction_following",
        observable="requested audible change is present in the named output interval",
        question=(
            "Compared with the paired source interval, is the audible change requested "
            "by the public instruction present in the named output interval?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="edited_intelligibility",
        dimension="edited_region_quality",
        observable="edited speech is intelligible when the requested result contains speech",
        question=(
            "When the requested result contains speech, is that speech intelligible "
            "inside the named output interval?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="edited_naturalness",
        dimension="edited_region_quality",
        observable="edited output sounds natural for the requested effect",
        question=(
            "Does the named output interval sound natural for the effect requested by "
            "the public instruction?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="edited_boundary_continuity",
        dimension="edited_region_quality",
        observable="transitions at the named edited interval boundaries are continuous",
        question=(
            "Are the transitions at the exact start and end of the named output "
            "interval audibly continuous?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="edited_artifact_absence",
        dimension="edited_region_quality",
        observable="edited output has no unintended clipping, distortion, or noise",
        question=(
            "Is the named output interval free from unintended clipping, distortion, "
            "clicks, or added noise?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="unedited_wording_consistency",
        dimension="unedited_region_consistency",
        observable="spoken wording is preserved within each unedited source/output pair",
        question=(
            "Within each named unedited pair, is the spoken wording preserved from "
            "source to output?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="unedited_speaker_consistency",
        dimension="unedited_region_consistency",
        observable="speaker identity is preserved within each unedited pair",
        question=(
            "Within each named unedited pair, is the speaker identity preserved from "
            "source to output?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="unedited_background_consistency",
        dimension="unedited_region_consistency",
        observable="background acoustics are preserved within each unedited pair",
        question=(
            "Within each named unedited pair, are the background sound and acoustic "
            "environment preserved from source to output?"
        ),
    ),
    _AtomicRubricDefinition(
        atomic_rubric_id="unedited_timing_consistency",
        dimension="unedited_region_consistency",
        observable="local speech timing is preserved within each unedited pair",
        question=(
            "Within each named unedited pair, is the local speech timing preserved "
            "from source to output?"
        ),
    ),
)


class AudioEvaluationInterval(StrictModel):
    """One source-timeline interval frozen before execution begins."""

    interval_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> AudioEvaluationInterval:
        if not math.isfinite(self.start_sec) or not math.isfinite(self.end_sec):
            raise ValueError("audio evaluation interval bounds must be finite")
        if self.end_sec <= self.start_sec:
            raise ValueError("audio evaluation interval end_sec must exceed start_sec")
        return self


class FrozenRubricQuestion(StrictModel):
    """Optional pre-authored question without answers or scorer outcomes."""

    dimension: RubricDimension
    question: str = Field(min_length=1)


class AudioEvaluationTarget(StrictModel):
    """One operation-authored semantic target bound to one edited interval."""

    target_ref_id: str = Field(min_length=1)
    interval_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    derivation: Literal[
        "executed_structured_operation_order",
        "executed_single_target_operation",
        "executed_typed_result_order",
        "executed_operation_effect_scope",
    ]


class FrozenPostDeliveryAudioRubricSpec(StrictModel):
    """Prompt-hidden, pre-execution interval contract used by every benchmark."""

    schema_version: Literal["x-editor.post-delivery-audio-rubric-spec/v1"] = (
        "x-editor.post-delivery-audio-rubric-spec/v1"
    )
    source_asset_ref: str = Field(min_length=1)
    source_duration_sec: float | None = Field(default=None, gt=0.0)
    edit_scope: Literal["local", "global"]
    edited_intervals: list[AudioEvaluationInterval] = Field(default_factory=list)
    unedited_intervals: list[AudioEvaluationInterval] | None = None
    targets: list[AudioEvaluationTarget] = Field(default_factory=list)
    objective_effects: list[FrozenObjectiveEffect] = Field(default_factory=list)
    rubrics: list[FrozenRubricQuestion] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_spec(self) -> FrozenPostDeliveryAudioRubricSpec:
        if self.source_duration_sec is not None and not math.isfinite(self.source_duration_sec):
            raise ValueError("source_duration_sec must be finite")
        if self.edit_scope == "local" and not self.edited_intervals:
            raise ValueError("local scoring spec requires edited_intervals")
        dimensions = [item.dimension for item in self.rubrics]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("frozen rubric dimensions must be unique")
        _validate_non_overlapping(self.edited_intervals, name="edited_intervals")
        target_ids = [item.target_ref_id for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("audio evaluation target refs must be unique")
        edited_ids = {item.interval_id for item in self.edited_intervals}
        if any(item.interval_id not in edited_ids for item in self.targets):
            raise ValueError("audio evaluation target references an unknown edited interval")
        if len({item.interval_id for item in self.targets}) != len(self.targets):
            raise ValueError("each edited interval can have at most one evaluation target")
        objective_ids = [item.objective_ref_id for item in self.objective_effects]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("frozen objective effect refs must be unique")
        known_target_ids = set(target_ids)
        if any(
            item.target_ref_id is not None and item.target_ref_id not in known_target_ids
            for item in self.objective_effects
        ):
            raise ValueError("frozen objective effect references an unknown target")
        if self.unedited_intervals is not None:
            _validate_non_overlapping(
                self.unedited_intervals,
                name="unedited_intervals",
            )
        return self


class PostDeliveryAudioScoringOptions(StrictModel):
    """One read-only scoring invocation."""

    schema_version: Literal["x-editor.post-delivery-audio-scoring-options/v1"] = (
        "x-editor.post-delivery-audio-scoring-options/v1"
    )
    run_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    mode: Literal["dry-run", "score"] = "dry-run"
    arms: list[Literal["W0", "W1", "W2"]] = Field(default_factory=list)
    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    timeout_sec: float = Field(default=300.0, gt=0.0)
    valid_repetitions: int = Field(default=3, ge=1, le=9)
    required_positive_votes: int = Field(default=2, ge=1)
    max_attempts: int = Field(default=10, ge=1, le=30)
    crop_margin_sec: float = Field(
        default=DEFAULT_CROP_MARGIN_SEC,
        ge=0.0,
        le=DEFAULT_CROP_MARGIN_SEC,
    )
    minimum_crop_sec: float = Field(default=0.01, gt=0.0, le=1.0)
    timeline_tolerance_sec: float = Field(default=0.05, ge=0.0, le=0.5)
    missing_interval_policy: Literal["abstain", "public_full_clip_fallback"] = "abstain"
    wording_evidence_mode: Literal["disabled", "best_effort"] = "disabled"
    wording_evidence_min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    wording_evidence_timeout_sec: float = Field(default=300.0, gt=0.0)
    wording_asr_url: str | None = None
    wording_aligner_url: str | None = None
    wording_model_name: str | None = None
    resume: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> PostDeliveryAudioScoringOptions:
        if len(self.arms) != len(set(self.arms)):
            raise ValueError("scoring arms must be unique")
        if self.required_positive_votes > self.valid_repetitions:
            raise ValueError("required_positive_votes cannot exceed valid_repetitions")
        if self.max_attempts < self.valid_repetitions:
            raise ValueError("max_attempts cannot be smaller than valid_repetitions")
        return self


class AudioIntervalMapping(StrictModel):
    """Auditable mapping from a frozen source interval to output timebase."""

    interval_id: str = Field(min_length=1)
    source_start_sec: float = Field(ge=0.0)
    source_end_sec: float = Field(gt=0.0)
    output_start_sec: float = Field(ge=0.0)
    output_end_sec: float = Field(gt=0.0)
    mapping_source: Literal[
        "runtime_explicit_span_mapping",
        "runtime_identity_outside_boundary_anchored",
        "runtime_timeline_preserved",
        "runtime_composed_timeline_ledger",
        "runtime_typed_temporal_lineage",
        "runtime_typed_step_result",
        "runtime_typed_insertion_effect",
        "global_full_clip",
        "public_full_clip_fallback",
        "scorer_independent_alignment",
        "scorer_independent_exact_alignment",
        "scorer_independent_bounded_alignment",
    ]

    @model_validator(mode="after")
    def validate_mapping(self) -> AudioIntervalMapping:
        values = (
            self.source_start_sec,
            self.source_end_sec,
            self.output_start_sec,
            self.output_end_sec,
        )
        if any(not math.isfinite(item) for item in values):
            raise ValueError("audio interval mapping bounds must be finite")
        if self.source_end_sec <= self.source_start_sec:
            raise ValueError("source mapping interval has no positive duration")
        if self.output_end_sec <= self.output_start_sec:
            raise ValueError("output mapping interval has no positive duration")
        return self


class _TypedTemporalLineageSpan(StrictModel):
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> _TypedTemporalLineageSpan:
        if self.end_sec <= self.start_sec:
            raise ValueError("typed temporal lineage span has no positive duration")
        return self


class _TypedTemporalLineageMapping(StrictModel):
    mapping_id: str = Field(min_length=1)
    role: Literal[
        "structural_mapping",
        "semantic_anchor",
        "affected_support",
    ]
    source_span: _TypedTemporalLineageSpan
    output_span: _TypedTemporalLineageSpan
    provenance_paths: list[str] = Field(min_length=1)


class _TypedTemporalLineage(StrictModel):
    """Read-only mirror of the executor-authored lineage wire contract."""

    schema_version: Literal["x-editor.audio-temporal-lineage/v1"]
    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    status: Literal["complete", "partial", "unavailable"]
    edit_scope: Literal["local", "global", "unknown"]
    input_revision: int = Field(ge=0)
    output_revision: int = Field(ge=1)
    input_audio_paths: list[str]
    input_audio_refs: list[str]
    output_audio_path: str = Field(min_length=1)
    input_duration_sec: float | None = Field(default=None, gt=0.0)
    output_duration_sec: float | None = Field(default=None, gt=0.0)
    exact_timeline_duration_preserved: bool | None = None
    duration_change_within_quality_tolerance: bool | None = None
    quality_duration_tolerance_sec: float = Field(gt=0.0)
    structural_mappings: list[_TypedTemporalLineageMapping]
    semantic_anchors: list[_TypedTemporalLineageMapping]
    affected_support: list[_TypedTemporalLineageMapping]
    identity_outside_affected_support: bool | None = None
    derivation: str = Field(min_length=1)
    reason_code: str | None = None
    reason: str | None = None


class AudioCropEvidence(StrictModel):
    """One materialized WAV excerpt with complete source-file provenance."""

    crop_id: str = Field(min_length=1)
    interval_id: str = Field(min_length=1)
    timebase: Literal["source", "output"]
    label: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_path: str = Field(min_length=1)
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_start_sec: float = Field(ge=0.0)
    requested_end_sec: float = Field(gt=0.0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    original_duration_sec: float = Field(gt=0.0)


class AudioRubricAttempt(StrictModel):
    """One independent provider sample, including invalid and failed attempts."""

    attempt_index: int = Field(ge=1)
    status: Literal["valid", "abstain", "invalid", "provider_failure"]
    score: Literal[0, 1, 2, 3] | None = None
    positive_vote: bool | None = None
    reason: str = Field(min_length=1)
    provider_attempts: int = Field(default=0, ge=0)
    raw_response_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    raw_response_excerpt: str | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> AudioRubricAttempt:
        if self.status == "valid":
            if self.score is None or self.positive_vote is None:
                raise ValueError("valid rubric attempt requires score and vote")
            if self.positive_vote != (self.score >= 2):
                raise ValueError("positive_vote must equal score >= 2")
        elif self.score is not None or self.positive_vote is not None:
            raise ValueError("non-valid rubric attempt cannot contain a score")
        return self


class AudioRubricRequest(StrictModel):
    """Provider-visible rubric plus prompt-hidden immutable join fields."""

    schema_version: Literal[
        "x-editor.audio-rubric-request/v1",
        "x-editor.audio-rubric-request/v2",
        "x-editor.audio-rubric-request/v3",
        "x-editor.audio-rubric-request/v4",
        "x-editor.audio-rubric-request/v5",
    ] = "x-editor.audio-rubric-request/v5"
    assignment_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_projection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assigned_arm: Literal["W0", "W1", "W2"]
    rubric_scope_source: Literal[
        "frozen_sidecar",
        "runtime_temporal_transform",
        "runtime_typed_temporal_lineage",
        "runtime_graph_interval_projection",
        "runtime_reviewed_evidence",
        "public_full_clip_fallback",
        "unavailable",
    ]
    dimension: RubricDimension
    atomic_rubric_id: AtomicRubricId | None = None
    atomic_rubric_catalog_version: str | None = None
    observable: str | None = None
    public_user_instruction: str = Field(min_length=1)
    active_execution_instruction: str | None = None
    target_ref_id: str | None = None
    target_requirement: str | None = None
    target_derivation: str | None = None
    final_state_projection: FinalStateProjection | None = None
    final_state_projection_source: Literal[
        "persisted_verified",
        "legacy_graph_recomputed",
        "unavailable",
    ] = "unavailable"
    effect_contract: EvaluationEffectContract | None = None
    evaluation_region_kind: Literal[
        "edited_target",
        "unedited_complement",
        "global_edited_region_preservation",
        "unavailable",
    ] = "unavailable"
    applicability: Literal[
        "applicable",
        "edited_region_preservation",
        "not_applicable",
        "unknown",
    ] = "unknown"
    applicability_source: str | None = None
    applicability_reason: str | None = None
    source_asset_ref: str = Field(min_length=1)
    intervals: list[AudioEvaluationInterval]
    interval_mappings: list[AudioIntervalMapping]
    question: str = Field(min_length=1)
    crops: list[AudioCropEvidence] = Field(default_factory=list)
    wording_evidence: list[WordingPairEvidence] = Field(default_factory=list)
    preflight_status: Literal["ready", "abstain", "invalid", "not_applicable"]
    preflight_reason: str | None = None
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class AudioRubricScoreResult(StrictModel):
    """One terminal rubric result aggregated from independent provider samples."""

    schema_version: Literal["x-editor.audio-rubric-score-result/v1"] = (
        "x-editor.audio-rubric-score-result/v1"
    )
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignment_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assigned_arm: Literal["W0", "W1", "W2"]
    dimension: RubricDimension
    atomic_rubric_id: AtomicRubricId | None = None
    status: ScoreStatus
    valid_scores: list[Literal[0, 1, 2, 3]] = Field(default_factory=list)
    positive_votes: int = Field(default=0, ge=0)
    valid_sample_count: int = Field(default=0, ge=0)
    attempt_count: int = Field(default=0, ge=0)
    mean_score: float | None = Field(default=None, ge=0.0, le=3.0)
    passed: bool | None = None
    reason: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    provider_attempts: int = Field(default=0, ge=0)
    attempts: list[AudioRubricAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> AudioRubricScoreResult:
        if self.status in {"pass", "fail"}:
            if self.mean_score is None or self.passed is None:
                raise ValueError("scored result requires mean_score and passed")
            if self.status != ("pass" if self.passed else "fail"):
                raise ValueError("rubric status conflicts with passed")
        elif self.mean_score is not None or self.passed is not None:
            raise ValueError("non-scored result cannot contain mean_score or passed")
        if self.valid_sample_count != len(self.valid_scores):
            raise ValueError("valid_sample_count must match valid_scores")
        if self.positive_votes != sum(score >= 2 for score in self.valid_scores):
            raise ValueError("positive_votes must match valid_scores")
        if self.attempt_count != len(self.attempts):
            raise ValueError("attempt_count must match attempts")
        return self


class _JudgeResponse(StrictModel):
    score: Literal[0, 1, 2, 3] | None = None
    abstain: bool
    reason: str = Field(min_length=1)

    @field_validator("score", mode="before")
    @classmethod
    def require_integer_score(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("judge score must be an integer 0, 1, 2, or 3")
        return value

    @model_validator(mode="after")
    def validate_response(self) -> _JudgeResponse:
        if self.abstain == (self.score is not None):
            raise ValueError("judge must return either a score or abstain")
        return self


class _JudgeClient(Protocol):
    model_name: str
    last_transport_attempt_count: int

    def call(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class _PreparedJob:
    request: AudioRubricRequest
    audio_paths: tuple[str, ...]


@dataclass(frozen=True)
class _RunEvidence:
    root: Path
    run_fingerprint: str
    protected_fingerprint: str
    expected_assignments: int
    selected_by_identity: dict[tuple[str, str, int | None], dict[str, Any]]
    sidecar_by_identity: dict[tuple[str, str, int | None], dict[str, Any]]
    deliveries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _EvaluationExecutionContext:
    active_execution_instruction: str | None
    final_state_projection: FinalStateProjection | None
    final_state_projection_source: Literal[
        "persisted_verified",
        "legacy_graph_recomputed",
        "unavailable",
    ]
    operations: tuple[str, ...]
    active_turn_indices: frozenset[int] | None


@dataclass(frozen=True)
class _RuntimeIntervalProjection:
    """Read-only source/output pairs proven by the adopted execution branch."""

    source_asset_ref: str
    edit_scope: Literal["local", "global"]
    interval_mappings: tuple[AudioIntervalMapping, ...]
    identity_outside_regions: bool
    rubric_scope_source: Literal[
        "runtime_temporal_transform",
        "runtime_typed_temporal_lineage",
        "runtime_graph_interval_projection",
    ]
    projection_source: str
    step_records: tuple[dict[str, Any], ...]
    structural_mappings: tuple[AudioIntervalMapping, ...] = ()
    affected_support_mappings: tuple[AudioIntervalMapping, ...] = ()
    effects: tuple[_RuntimeEffectIntervalProjection, ...] = ()


@dataclass(frozen=True)
class _RuntimeEffectIntervalProjection:
    """One executed effect with its own auditable evaluation scope."""

    effect_ref_id: str
    step_id: str
    operation: str
    edit_scope: Literal["local", "global"]
    semantic_anchor_mappings: tuple[AudioIntervalMapping, ...]
    affected_support_mappings: tuple[AudioIntervalMapping, ...]
    step_record: dict[str, Any]


@dataclass(frozen=True)
class _ReviewedRuntimeTargetProjection:
    """Reviewed source-timeline targets from the adopted runtime branch."""

    source_asset_ref: str
    intervals: tuple[AudioEvaluationInterval, ...]


@dataclass(frozen=True)
class _AtomicIntervalGroup:
    """One atomic rubric/effect region after scope selection."""

    intervals: tuple[AudioEvaluationInterval, ...]
    target: AudioEvaluationTarget | None
    edit_scope: Literal["local", "global"]
    interval_mappings: tuple[AudioIntervalMapping, ...] | None = None
    effect_operations: tuple[str, ...] | None = None


class _WordingEvidenceCollector:
    """Deduplicate read-only ASR calls across atomic rubric requests."""

    def __init__(
        self,
        *,
        provider: WordingEvidenceProvider,
        minimum_confidence: float,
    ) -> None:
        self.provider = provider
        self.minimum_confidence = minimum_confidence
        self._observations: dict[
            str,
            IndependentWordingObservation | Exception,
        ] = {}

    def for_pairs(
        self,
        *,
        mappings: list[AudioIntervalMapping],
        crops: list[AudioCropEvidence],
    ) -> list[WordingPairEvidence]:
        result: list[WordingPairEvidence] = []
        for mapping in mappings:
            source_crop = next(
                (
                    crop
                    for crop in crops
                    if crop.interval_id == mapping.interval_id and crop.timebase == "source"
                ),
                None,
            )
            output_crop = next(
                (
                    crop
                    for crop in crops
                    if crop.interval_id == mapping.interval_id and crop.timebase == "output"
                ),
                None,
            )
            if source_crop is None or output_crop is None:
                result.append(
                    build_wording_pair_evidence(
                        pair_id=mapping.interval_id,
                        source=None,
                        output=None,
                        minimum_confidence=self.minimum_confidence,
                    )
                )
                continue
            source = self._observe(source_crop)
            output = self._observe(output_crop)
            errors = [
                f"{timebase}: {type(value).__name__}: {value}"
                for timebase, value in (("source", source), ("output", output))
                if isinstance(value, Exception)
            ]
            result.append(
                build_wording_pair_evidence(
                    pair_id=mapping.interval_id,
                    source=(source if isinstance(source, IndependentWordingObservation) else None),
                    output=(output if isinstance(output, IndependentWordingObservation) else None),
                    minimum_confidence=self.minimum_confidence,
                    provider_error="; ".join(errors) if errors else None,
                )
            )
        return result

    def recover_local_mappings(
        self,
        *,
        source_path: Path,
        output_path: Path,
        intervals: list[AudioEvaluationInterval],
    ) -> tuple[list[AudioIntervalMapping] | None, str]:
        source = self._observe_path(source_path, label="full source audio")
        output = self._observe_path(output_path, label="full delivered output audio")
        errors = [
            f"{label}: {type(value).__name__}: {value}"
            for label, value in (("source", source), ("output", output))
            if isinstance(value, Exception)
        ]
        if errors:
            return None, "; ".join(errors)
        assert isinstance(source, IndependentWordingObservation)
        assert isinstance(output, IndependentWordingObservation)
        alignment = recover_local_interval_alignment(
            source=source,
            output=output,
            intervals=[
                ScorerInterval(
                    interval_id=interval.interval_id,
                    start_sec=interval.start_sec,
                    end_sec=interval.end_sec,
                )
                for interval in intervals
            ],
        )
        if alignment.status != "ready":
            return None, alignment.reason
        return (
            [
                AudioIntervalMapping(
                    interval_id=mapping.interval_id,
                    source_start_sec=mapping.source_start_sec,
                    source_end_sec=mapping.source_end_sec,
                    output_start_sec=mapping.output_start_sec,
                    output_end_sec=mapping.output_end_sec,
                    mapping_source=(
                        "scorer_independent_exact_alignment"
                        if mapping.method == "exact_unique_sequence"
                        else "scorer_independent_bounded_alignment"
                    ),
                )
                for mapping in alignment.mappings
            ],
            alignment.reason,
        )

    def _observe(
        self,
        crop: AudioCropEvidence,
    ) -> IndependentWordingObservation | Exception:
        return self._observe_path(
            Path(crop.path),
            label=crop.label,
            expected_sha256=crop.sha256,
        )

    def _observe_path(
        self,
        path: Path,
        *,
        label: str,
        expected_sha256: str | None = None,
    ) -> IndependentWordingObservation | Exception:
        resolved = path.expanduser().resolve()
        audio_sha256 = expected_sha256 or _sha256_file(resolved)
        cached = self._observations.get(audio_sha256)
        if cached is not None:
            return cached
        try:
            raw = self.provider.observe(str(resolved), label=label)
            observation = IndependentWordingObservation.model_validate(raw)
            if observation.audio_sha256 != audio_sha256:
                raise ValueError("wording evidence audio hash does not match its crop")
            value: IndependentWordingObservation | Exception = observation
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            value = exc
        self._observations[audio_sha256] = value
        return value


_SYSTEM_PROMPT = """You are Qwen3-Omni acting only as a post-delivery audio rubric judge.
You cannot admit, reject, repair, rerun, or alter execution. Judge only the named
atomic observable, the public user instruction, the attached audio, and the
listed source/output interval mappings. Evaluate exactly one observable fact;
do not blend other quality dimensions into the score. Do not infer benchmark
answers or use any target audio. A Planner interpretation, selected operation,
tool capability, execution status, or Verifier result is never evidence that
the requested audible effect occurred. Judge the attached audio directly; do
not claim success merely because a suitable tool ran. For preservation-only
rubrics, the edit instruction is deliberately hidden: do not infer edited words
or operations from metadata. For local-timing preservation, absolute timeline
offsets only localize the paired excerpts and are not themselves audible timing
errors. If a speech-edit instruction describes a fact, attribute, relation,
quantity, date, or semantic referent in the spoken sentence, the requested
result must be audible in what the output says. Do not treat a non-acoustic
subject such as an object's color as a reason to leave speech unchanged.
Conversely, descriptive wording in an instruction is not automatically verbatim
replacement text unless the user explicitly requests or quotes it as such;
judge the intended listener-observable meaning from the public instruction and
source speech. If a pair declares sample_identity_evidence=identical_pcm,
identical PCM is conclusive evidence
that the named preservation observable did not change; do not invent words or
events to explain that fact. Independent ASR/alignment evidence is advisory and
must be judged at its stated confidence. When it is low-confidence or conflicts
with the attached audio, do not treat it as truth; abstain when the audio alone
cannot support a responsible wording judgement. Transcript, wording, ASR, or
alignment agreement can establish only spoken-content preservation; unlike
sample-identical PCM, it cannot establish whether speaker identity or vocal
timbre changed or stayed the same. Judge speaker identity and voice conversion
only from audible speaker and timbre characteristics. When a public voice
reference is attached, compare the source voice, output voice, and reference
voice directly; lexical or transcript overlap is not speaker-identity evidence.
Use this four-level scale:
0 = clearly violates the rubric;
1 = partly satisfies it but has material errors;
2 = acceptably satisfies it with only minor issues;
3 = strongly and clearly satisfies it.
Return exactly one JSON object with keys score, abstain, and reason. score must
be one of 0, 1, 2, 3 or null. Set abstain=true and score=null only when the
attached evidence cannot support a responsible judgement."""


def run_post_delivery_audio_scoring(
    options: PostDeliveryAudioScoringOptions,
    *,
    client: _JudgeClient | None = None,
    wording_evidence_provider: WordingEvidenceProvider | None = None,
) -> dict[str, Any]:
    """Prepare or score committed deliveries without mutating execution evidence."""

    run_root = Path(options.run_dir).expanduser().resolve()
    output_root = Path(options.output_dir).expanduser().resolve()
    _require_external_output_dir(run_root, output_root)
    if output_root.exists() and not options.resume:
        raise FileExistsError(f"scoring output directory already exists: {output_root}")
    if options.resume and not output_root.is_dir():
        raise FileNotFoundError(f"resume scoring output directory does not exist: {output_root}")

    evidence = _load_run_evidence(run_root, arms=set(options.arms))

    effective_base_url = options.base_url
    if options.mode == "score" and client is None:
        base_url = resolve_openai_compatible_base_url(
            QWEN3_OMNI_PROVIDER,
            options.base_url,
        )
        if not base_url:
            raise ValueError("score mode requires a Qwen3-Omni base URL")
        effective_base_url = base_url
        client = OpenAICompatibleChatClient(
            api_key=resolve_openai_compatible_api_key(
                QWEN3_OMNI_PROVIDER,
                options.api_key,
            ),
            base_url=base_url,
            model_name=resolve_openai_compatible_model(
                QWEN3_OMNI_PROVIDER,
                options.model_name,
            ),
            provider_name=QWEN3_OMNI_PROVIDER,
            temperature=0.7,
            max_tokens=512,
            timeout=options.timeout_sec,
            max_retries=0,
            allow_json_repair=False,
            require_api_key=False,
        )
    elif options.mode == "score":
        effective_base_url = getattr(client, "base_url", effective_base_url)

    if options.wording_evidence_mode == "disabled":
        if wording_evidence_provider is not None:
            raise ValueError("wording_evidence_provider requires wording_evidence_mode=best_effort")
        wording_collector = None
    else:
        provider = wording_evidence_provider or SpeechPerceptionWordingEvidenceProvider(
            asr_url=options.wording_asr_url,
            aligner_url=options.wording_aligner_url,
            timeout_sec=options.wording_evidence_timeout_sec,
            cache_dir=output_root / "wording_evidence_cache",
            model_name=options.wording_model_name,
        )
        wording_collector = _WordingEvidenceCollector(
            provider=provider,
            minimum_confidence=options.wording_evidence_min_confidence,
        )

    model_name = (
        getattr(client, "model_name", options.model_name or DEFAULT_QWEN3_OMNI_MODEL)
        if options.mode == "score"
        else None
    )
    protocol = _scorer_protocol(
        options,
        model_name=model_name,
        effective_base_url=effective_base_url,
        authentication_configured=bool(
            getattr(client, "api_key", None) if client is not None else options.api_key
        ),
        wording_evidence_provider=wording_evidence_provider
        or (wording_collector.provider if wording_collector is not None else None),
    )
    requests_path = output_root / "rubric_requests.jsonl"
    manifest_path = output_root / "score_manifest.json"
    if options.resume:
        manifest = _read_json(manifest_path)
        requests = [AudioRubricRequest.model_validate(item) for item in _read_jsonl(requests_path)]
        jobs = [
            _PreparedJob(
                request=request,
                audio_paths=tuple(crop.path for crop in request.crops),
            )
            for request in requests
        ]
        _verify_resume_identity(
            manifest,
            evidence=evidence,
            jobs=jobs,
            protocol=protocol,
        )
    else:
        output_root.mkdir(parents=True)
        jobs = _prepare_jobs(
            evidence,
            options=options,
            crop_root=output_root / "crops",
            wording_collector=wording_collector,
        )
        requests = [job.request for job in jobs]
        _write_jsonl(
            requests_path,
            [request.model_dump(mode="json") for request in requests],
        )
        manifest = _score_manifest(
            evidence=evidence,
            options=options,
            jobs=jobs,
            protocol=protocol,
            state="in_progress",
        )
        _write_json(manifest_path, manifest)

    _verify_jobs_match_run(jobs, evidence=evidence)
    audio_evidence_before = _job_audio_fingerprint(jobs)
    results_path = output_root / "results.jsonl"
    existing_results = _resume_results(results_path, jobs=jobs) if options.resume else {}

    results: list[AudioRubricScoreResult] = []
    for job in jobs:
        existing = existing_results.get(job.request.request_fingerprint)
        if existing is not None:
            results.append(existing)
            continue
        if job.request.preflight_status != "ready":
            result = _preflight_result(job.request)
        elif options.mode == "dry-run":
            result = _terminal_result(
                job.request,
                status="dry_run",
                reason="provider call intentionally skipped in dry-run mode",
            )
        else:
            assert client is not None
            result = _score_job(job, client=client, options=options)
        _append_jsonl(results_path, result.model_dump(mode="json"))
        results.append(result)

    protected_after = _protected_run_fingerprint(run_root)
    if protected_after != evidence.protected_fingerprint:
        raise RuntimeError("benchmark execution evidence changed during scoring")
    audio_evidence_after = _job_audio_fingerprint(jobs)
    if audio_evidence_after != audio_evidence_before:
        raise RuntimeError("public scoring audio evidence changed during scoring")

    serialized_results = [item.model_dump(mode="json") for item in results]
    flow_report = build_post_delivery_flow_report(
        expected_assignments=evidence.expected_assignments,
        delivered_assignments=len(evidence.deliveries),
        results=serialized_results,
        requests=[request.model_dump(mode="json") for request in requests],
    )
    _write_json(output_root / "flow_report.json", flow_report)
    _write_jsonl(
        output_root / "assignment_flow_reports.jsonl",
        list(flow_report["assignments"]),
    )
    summary = _score_summary(
        mode=options.mode,
        run_fingerprint=evidence.run_fingerprint,
        deliveries=len(evidence.deliveries),
        results=results,
        options=options,
        flow_report=flow_report,
    )
    _write_json(output_root / "summary.json", summary)
    completed_manifest = dict(manifest)
    completed_manifest.update(
        {
            "state": "complete",
            "execution_evidence_fingerprint_after": protected_after,
            "scoring_audio_fingerprint_before": audio_evidence_before,
            "scoring_audio_fingerprint_after": audio_evidence_after,
            "result_set_fingerprint": _fingerprint(serialized_results),
            "completed_rubric_count": len(results),
        }
    )
    _write_json(manifest_path, completed_manifest)
    return summary


def _load_run_evidence(root: Path, *, arms: set[str]) -> _RunEvidence:
    manifest = _read_json(root / "run_manifest.json")
    completion = _read_json(root / "run_completion.json")
    run_config = _read_json(root / "run_config.json")
    if manifest.get("schema_version") != "x-editor.benchmark-run-manifest/v1":
        raise ValueError("unsupported benchmark run manifest")
    if completion.get("state") != "complete" or completion.get("mode") != "executed":
        raise ValueError("post-delivery scoring requires a completed executed run")
    run_fingerprint = str(manifest.get("run_fingerprint") or "")
    if completion.get("run_fingerprint") != run_fingerprint:
        raise ValueError("run completion fingerprint does not match run manifest")
    immutable = manifest.get("immutable_artifacts")
    if not isinstance(immutable, dict) or not immutable:
        raise ValueError("run manifest has no immutable artifacts")
    for relative, expected_sha in immutable.items():
        path = _resolve_under(root, str(relative))
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise ValueError(f"frozen run artifact hash mismatch: {relative}")

    selected_rows = _read_jsonl(root / "selected_cases.jsonl")
    selected_by_identity: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for row in selected_rows:
        identity = _identity_key(row.get("audit_identity"))
        if identity in selected_by_identity:
            raise ValueError(f"duplicate selected identity: {identity}")
        selected_by_identity[identity] = row

    configured_arms = {
        str(arm) for arm in run_config.get("arms") or [] if str(arm) in {"W0", "W1", "W2"}
    }
    selected_arms = configured_arms & arms if arms else configured_arms
    expected_from_projection = len(selected_by_identity) * len(selected_arms)
    expected_from_completion = completion.get("expected_assignments")
    if (
        not arms
        and isinstance(expected_from_completion, int)
        and not isinstance(expected_from_completion, bool)
        and expected_from_completion >= 0
    ):
        expected_assignments = expected_from_completion
    else:
        expected_assignments = expected_from_projection

    sidecar_by_identity: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for path in sorted((root / "sidecars").glob("*.json")):
        row = _read_json(path)
        identity = _identity_key(row.get("audit_identity"))
        if identity in sidecar_by_identity:
            raise ValueError(f"duplicate sidecar identity: {identity}")
        sidecar_by_identity[identity] = row

    deliveries: list[dict[str, Any]] = []
    for marker_path in sorted((root / "samples").glob("*/*/completion.json")):
        marker = _read_json(marker_path)
        arm = str(marker.get("assigned_arm") or "")
        if arms and arm not in arms:
            continue
        if marker.get("state") != "complete" or marker.get("status") != "completed":
            continue
        identity = _identity_key(marker.get("audit_identity"))
        selected = selected_by_identity.get(identity)
        sidecar = sidecar_by_identity.get(identity)
        if selected is None or sidecar is None:
            raise ValueError(f"delivery has no frozen scenario/sidecar: {identity}")
        if selected.get("projection_status") != "ready":
            raise ValueError(f"completed delivery has a failed projection: {identity}")
        runtime_fingerprint = str(marker.get("runtime_projection_fingerprint") or "")
        if selected.get("runtime_projection_fingerprint") != runtime_fingerprint:
            raise ValueError(f"delivery projection fingerprint mismatch: {identity}")
        result_path = _resolve_under(root, str(marker.get("result_path") or ""))
        if _sha256_file(result_path) != marker.get("result_sha256"):
            raise ValueError(f"delivery result hash mismatch: {identity}/{arm}")
        result_row = _read_json(result_path)
        if result_row.get("status") != "completed":
            raise ValueError(f"delivery result is not completed: {identity}/{arm}")
        lineage = marker.get("artifact_lineage")
        if not isinstance(lineage, dict):
            raise TypeError(f"completed delivery has no artifact lineage: {identity}/{arm}")
        output_path = _resolve_under(
            root,
            str(lineage.get("canonical_output_path") or ""),
        )
        try:
            output_path.relative_to(root / "outputs")
        except ValueError as exc:
            raise ValueError("canonical delivery is outside outputs/") from exc
        if not output_path.is_file():
            raise ValueError(f"canonical delivery is missing: {output_path}")
        if _sha256_file(output_path) != lineage.get("sha256"):
            raise ValueError(f"canonical delivery hash mismatch: {identity}/{arm}")
        if output_path.stat().st_size != lineage.get("size_bytes"):
            raise ValueError(f"canonical delivery size mismatch: {identity}/{arm}")
        deliveries.append(
            {
                "marker": marker,
                "selected": selected,
                "sidecar": sidecar,
                "result": result_row,
                "output_path": str(output_path),
            }
        )
    return _RunEvidence(
        root=root,
        run_fingerprint=run_fingerprint,
        protected_fingerprint=_protected_run_fingerprint(root),
        expected_assignments=max(expected_assignments, len(deliveries)),
        selected_by_identity=selected_by_identity,
        sidecar_by_identity=sidecar_by_identity,
        deliveries=tuple(deliveries),
    )


def _prepare_jobs(
    evidence: _RunEvidence,
    *,
    options: PostDeliveryAudioScoringOptions,
    crop_root: Path,
    wording_collector: _WordingEvidenceCollector | None,
) -> list[_PreparedJob]:
    jobs: list[_PreparedJob] = []
    for delivery in evidence.deliveries:
        marker = delivery["marker"]
        selected = delivery["selected"]
        scenario = selected.get("runtime_scenario")
        if not isinstance(scenario, dict):
            jobs.extend(_invalid_delivery_jobs(marker, "runtime scenario is missing"))
            continue
        try:
            instruction = _public_instruction(scenario)
        except ValueError as exc:
            jobs.extend(_invalid_delivery_jobs(marker, str(exc)))
            continue
        assets = scenario.get("audio_assets")
        if not isinstance(assets, list) or not assets:
            jobs.extend(_invalid_delivery_jobs(marker, "runtime audio assets are missing"))
            continue
        source_asset = next(
            (item for item in assets if isinstance(item, dict) and item.get("role") == "source"),
            None,
        )
        if source_asset is None or not source_asset.get("asset_ref"):
            jobs.extend(
                _invalid_delivery_jobs(
                    marker,
                    "audio rubric scoring has no public source asset",
                    instruction=instruction,
                )
            )
            continue
        try:
            source_audio_aliases = _public_source_audio_aliases(assets)
        except ValueError as exc:
            jobs.extend(
                _invalid_delivery_jobs(
                    marker,
                    f"invalid public workflow audio assets: {exc}",
                    instruction=instruction,
                )
            )
            continue
        source_asset_ref = str(source_asset["asset_ref"])
        producer_output_path = str(
            (marker.get("artifact_lineage") or {}).get("producer_output_path") or ""
        )
        try:
            asset_rows = _verified_public_assets(assets)
            source_duration_by_ref = {
                str(item["asset_ref"]): _audio_duration_sec(
                    Path(str(item["path"])).expanduser().resolve()
                )
                for item in asset_rows
                if item.get("role") == "source" and item.get("asset_ref")
            }
            output_path = Path(str(delivery["output_path"])).resolve()
            output_duration_sec = _audio_duration_sec(output_path)
        except (OSError, ValueError, sf.SoundFileError) as exc:
            jobs.extend(
                _invalid_delivery_jobs(
                    marker,
                    f"audio rubric evidence is invalid: {type(exc).__name__}: {exc}",
                    instruction=instruction,
                )
            )
            continue
        try:
            execution_context = _evaluation_execution_context(
                delivery["result"],
                run_root=evidence.root,
                producer_output_path=producer_output_path,
            )
        except (OSError, TypeError, ValueError) as exc:
            jobs.extend(
                _invalid_delivery_jobs(
                    marker,
                    f"final-state projection is invalid: {type(exc).__name__}: {exc}",
                    instruction=instruction,
                )
            )
            continue
        runtime_interval_projection = _runtime_graph_interval_projection(
            delivery["result"],
            source_audio_aliases=source_audio_aliases,
            producer_output_path=producer_output_path,
            active_turn_indices=execution_context.active_turn_indices,
            source_duration_by_ref=source_duration_by_ref,
            output_duration_sec=output_duration_sec,
        )
        sidecar_spec = _sidecar_scoring_spec(delivery["sidecar"])
        if sidecar_spec is not None:
            try:
                spec = FrozenPostDeliveryAudioRubricSpec.model_validate(sidecar_spec)
            except ValidationError as exc:
                jobs.extend(
                    _invalid_delivery_jobs(
                        marker,
                        f"invalid frozen audio rubric spec: {_validation_summary(exc)}",
                        instruction=instruction,
                    )
                )
                continue
            default_scope_source = "frozen_sidecar"
        else:
            rubric_only_spec = _mmae_style_rubric_spec(
                delivery["sidecar"],
                source_asset_ref=source_asset_ref,
            )
            rubric_questions = (
                FrozenPostDeliveryAudioRubricSpec.model_validate(rubric_only_spec).rubrics
                if rubric_only_spec is not None
                else []
            )
            runtime_spec = (
                _rubric_spec_from_runtime_projection(
                    runtime_interval_projection,
                    rubrics=rubric_questions,
                )
                if runtime_interval_projection is not None
                else None
            )
            if runtime_spec is not None:
                spec = runtime_spec
                default_scope_source = runtime_interval_projection.rubric_scope_source
            else:
                runtime_spec = _runtime_reviewed_evidence_rubric_spec(
                    delivery["result"],
                    source_audio_aliases=source_audio_aliases,
                    rubrics=rubric_questions,
                    active_turn_indices=execution_context.active_turn_indices,
                )
                if runtime_spec is not None:
                    spec = runtime_spec
                    default_scope_source = "runtime_reviewed_evidence"
                else:
                    jobs.extend(
                        _abstained_delivery_jobs(
                            marker,
                            instruction=instruction,
                            reason=(
                                "no frozen sidecar or unambiguous interval projection "
                                "from the final adopted graph branch, complete timeline "
                                "lineage, typed successful result, or reviewed runtime "
                                "target is available"
                            ),
                        )
                    )
                    continue
        source = next(
            (
                item
                for item in assets
                if isinstance(item, dict)
                and item.get("asset_ref") == spec.source_asset_ref
                and item.get("role") == "source"
            ),
            None,
        )
        if source is None:
            jobs.extend(
                _invalid_delivery_jobs(
                    marker,
                    "frozen scoring source_asset_ref is not a public source asset",
                    instruction=instruction,
                )
            )
            continue
        try:
            source_path = Path(str(source["path"])).expanduser().resolve()
            source_duration_sec = source_duration_by_ref[spec.source_asset_ref]
            if (
                spec.source_duration_sec is not None
                and abs(spec.source_duration_sec - source_duration_sec)
                > options.timeline_tolerance_sec
            ):
                raise ValueError("frozen source duration does not match the decoded source audio")
            edited, unedited = _resolved_intervals(
                spec,
                duration_sec=source_duration_sec,
            )
        except (KeyError, OSError, ValueError, sf.SoundFileError) as exc:
            jobs.extend(
                _invalid_delivery_jobs(
                    marker,
                    f"audio rubric evidence is invalid: {type(exc).__name__}: {exc}",
                    instruction=instruction,
                )
            )
            continue
        target_by_interval = {item.interval_id: item for item in spec.targets}
        for atomic_rubric in _ATOMIC_RUBRICS:
            dimension = atomic_rubric.dimension
            if (
                sidecar_spec is None
                and runtime_interval_projection is not None
                and runtime_interval_projection.effects
            ):
                interval_groups = _typed_effect_interval_groups(
                    atomic_rubric,
                    projection=runtime_interval_projection,
                    edited_intervals=edited,
                    unedited_intervals=unedited,
                    source_duration_sec=source_duration_sec,
                    output_duration_sec=output_duration_sec,
                )
            elif dimension == "unedited_region_consistency":
                interval_groups = [
                    _AtomicIntervalGroup(
                        intervals=tuple(unedited),
                        target=None,
                        edit_scope=spec.edit_scope,
                    )
                ]
            elif target_by_interval:
                target_groups = [
                    _AtomicIntervalGroup(
                        intervals=(interval,),
                        target=target_by_interval[interval.interval_id],
                        edit_scope=spec.edit_scope,
                    )
                    for interval in edited
                    if interval.interval_id in target_by_interval
                ]
                interval_groups = (
                    target_groups
                    if len(target_groups) == len(edited)
                    else [
                        _AtomicIntervalGroup(
                            intervals=tuple(edited),
                            target=None,
                            edit_scope=spec.edit_scope,
                        )
                    ]
                )
            else:
                interval_groups = [
                    _AtomicIntervalGroup(
                        intervals=tuple(edited),
                        target=None,
                        edit_scope=spec.edit_scope,
                    )
                ]

            for interval_group in interval_groups:
                intervals = list(interval_group.intervals)
                target = interval_group.target
                effect_operations = (
                    interval_group.effect_operations
                    if interval_group.effect_operations is not None
                    else execution_context.operations
                )
                effect_contract = resolve_evaluation_effect_contract(
                    list(effect_operations),
                    objective_effects=spec.objective_effects,
                    target_ref_id=target.target_ref_id if target is not None else None,
                    final_state_node_id=(
                        execution_context.final_state_projection.current_node_id
                        if execution_context.final_state_projection is not None
                        else None
                    ),
                )
                applicability: Literal[
                    "applicable",
                    "edited_region_preservation",
                    "not_applicable",
                    "unknown",
                ] = "applicable"
                applicability_source = "frozen_local_scope"
                applicability_reason = "rubric evaluates the explicit target or local complement"
                region_kind: Literal[
                    "edited_target",
                    "unedited_complement",
                    "global_edited_region_preservation",
                    "unavailable",
                ] = (
                    "unedited_complement"
                    if dimension == "unedited_region_consistency"
                    else "edited_target"
                )
                preflight_status: Literal["ready", "abstain", "invalid", "not_applicable"] = "ready"
                preflight_reason: str | None = None

                typed_global_disposition = (
                    _typed_global_effect_disposition(
                        runtime_interval_projection,
                        atomic_rubric.atomic_rubric_id,
                    )
                    if dimension == "unedited_region_consistency"
                    and sidecar_spec is None
                    and runtime_interval_projection is not None
                    and runtime_interval_projection.effects
                    else None
                )
                if typed_global_disposition is not None:
                    property_name = _PRESERVATION_PROPERTIES[
                        atomic_rubric.atomic_rubric_id
                    ]
                    applicability_source = (
                        "typed_global_effects+operation_effect_catalog/v2"
                    )
                    applicability_reason = (
                        "global effects on the adopted branch classify "
                        f"{property_name} as {typed_global_disposition}"
                    )
                    if typed_global_disposition == "changed":
                        applicability = "not_applicable"
                        preflight_status = "not_applicable"
                        preflight_reason = applicability_reason
                    elif typed_global_disposition == "unknown":
                        applicability = "unknown"
                        preflight_status = "abstain"
                        preflight_reason = applicability_reason
                    elif interval_group.edit_scope == "global":
                        region_kind = "global_edited_region_preservation"
                        applicability = "edited_region_preservation"
                elif (
                    dimension == "unedited_region_consistency"
                    and interval_group.edit_scope == "global"
                ):
                    region_kind = "global_edited_region_preservation"
                    property_name = _PRESERVATION_PROPERTIES[atomic_rubric.atomic_rubric_id]
                    applicability_source = "frozen_objective+executed_operation_effect_catalog/v2"
                    if (
                        effect_contract.operation_chain is None
                        and not effect_contract.objective_effects
                    ):
                        applicability = "unknown"
                        preflight_status = "abstain"
                        applicability_reason = (
                            "no active executed operation chain is available to decide "
                            f"whether {property_name} should be preserved"
                        )
                        preflight_reason = applicability_reason
                    else:
                        disposition = effect_contract.dispositions[property_name]
                        applicability_reason = (
                            "frozen objective and active operation effects classify "
                            f"{property_name} as {disposition}"
                        )
                        if disposition == "changed":
                            applicability = "not_applicable"
                            preflight_status = "not_applicable"
                            preflight_reason = applicability_reason
                        elif disposition == "unknown":
                            applicability = "unknown"
                            preflight_status = "abstain"
                            preflight_reason = applicability_reason
                        else:
                            applicability = "edited_region_preservation"

                if (
                    interval_group.edit_scope == "global"
                    and sidecar_spec is None
                    and runtime_interval_projection is not None
                    and bool(runtime_interval_projection.effects)
                    and atomic_rubric.atomic_rubric_id
                    == "edited_boundary_continuity"
                ):
                    applicability = "not_applicable"
                    preflight_status = "not_applicable"
                    applicability_source = "typed_effect_temporal_scope"
                    applicability_reason = (
                        "a whole-audio effect has no internal local edit boundary"
                    )
                    preflight_reason = applicability_reason

                common_job_args = {
                    "marker": marker,
                    "rubric_scope_source": default_scope_source,
                    "dimension": dimension,
                    "instruction": instruction,
                    "active_execution_instruction": (
                        execution_context.active_execution_instruction
                    ),
                    "target": target,
                    "final_state_projection": execution_context.final_state_projection,
                    "final_state_projection_source": (
                        execution_context.final_state_projection_source
                    ),
                    "effect_contract": effect_contract,
                    "evaluation_region_kind": region_kind,
                    "applicability": applicability,
                    "applicability_source": applicability_source,
                    "applicability_reason": applicability_reason,
                    "source_asset_ref": spec.source_asset_ref,
                    "atomic_rubric": atomic_rubric,
                }
                if not intervals:
                    jobs.append(
                        _build_job(
                            **common_job_args,
                            intervals=[],
                            interval_mappings=[],
                            crops=[],
                            preflight_status="abstain",
                            preflight_reason=(
                                "no unedited interval exists for this frozen edit scope"
                                if dimension == "unedited_region_consistency"
                                else "no edited interval exists for this frozen edit scope"
                            ),
                        )
                    )
                    continue
                if preflight_status != "ready":
                    jobs.append(
                        _build_job(
                            **common_job_args,
                            intervals=intervals,
                            interval_mappings=[],
                            crops=[],
                            preflight_status=preflight_status,
                            preflight_reason=preflight_reason,
                        )
                    )
                    continue
                mappings = (
                    list(interval_group.interval_mappings)
                    if interval_group.interval_mappings is not None
                    else _resolve_interval_mappings(
                        intervals,
                        edit_scope=interval_group.edit_scope,
                        result_row=delivery["result"],
                        producer_output_path=producer_output_path or str(output_path),
                        source_duration_sec=source_duration_sec,
                        output_duration_sec=output_duration_sec,
                        tolerance_sec=options.timeline_tolerance_sec,
                        runtime_interval_projection=runtime_interval_projection,
                        active_turn_indices=execution_context.active_turn_indices,
                    )
                )
                recovery_reason: str | None = None
                if (
                    mappings is None
                    and wording_collector is not None
                    and interval_group.edit_scope == "local"
                    and dimension != "unedited_region_consistency"
                    and atomic_rubric.atomic_rubric_id != "edited_boundary_continuity"
                ):
                    mappings, recovery_reason = wording_collector.recover_local_mappings(
                        source_path=source_path,
                        output_path=output_path,
                        intervals=intervals,
                    )
                if mappings is None:
                    jobs.append(
                        _build_job(
                            **common_job_args,
                            intervals=intervals,
                            interval_mappings=[],
                            crops=[],
                            preflight_status="abstain",
                            preflight_reason=(
                                "no auditable source-to-output interval mapping exists"
                                + (
                                    f"; independent scorer localization: {recovery_reason}"
                                    if recovery_reason
                                    else ""
                                )
                            ),
                        )
                    )
                    continue
                if atomic_rubric.atomic_rubric_id == "edited_boundary_continuity":
                    missing_boundary_context = _missing_boundary_context_reason(
                        mappings,
                        source_duration_sec=source_duration_sec,
                        output_duration_sec=output_duration_sec,
                        tolerance_sec=options.timeline_tolerance_sec,
                    )
                    if missing_boundary_context is not None:
                        jobs.append(
                            _build_job(
                                **common_job_args,
                                intervals=intervals,
                                interval_mappings=mappings,
                                crops=[],
                                preflight_status="abstain",
                                preflight_reason=missing_boundary_context,
                            )
                        )
                        continue
                try:
                    crops = _materialize_job_crops(
                        marker=marker,
                        dimension=dimension,
                        atomic_rubric_id=atomic_rubric.atomic_rubric_id,
                        mappings=mappings,
                        source_path=source_path,
                        output_path=output_path,
                        public_assets=asset_rows,
                        source_asset_ref=spec.source_asset_ref,
                        crop_root=crop_root,
                        crop_margin_sec=options.crop_margin_sec,
                        minimum_crop_sec=options.minimum_crop_sec,
                    )
                except (OSError, ValueError, sf.SoundFileError) as exc:
                    jobs.append(
                        _build_job(
                            **common_job_args,
                            intervals=intervals,
                            interval_mappings=mappings,
                            crops=[],
                            preflight_status="invalid",
                            preflight_reason=(
                                f"audio crop materialization failed: {type(exc).__name__}: {exc}"
                            ),
                        )
                    )
                    continue
                wording_evidence = (
                    wording_collector.for_pairs(mappings=mappings, crops=crops)
                    if wording_collector is not None
                    and atomic_rubric.atomic_rubric_id in _WORDING_RUBRICS
                    else []
                )
                jobs.append(
                    _build_job(
                        **common_job_args,
                        intervals=intervals,
                        interval_mappings=mappings,
                        crops=crops,
                        wording_evidence=wording_evidence,
                        preflight_status="ready",
                        preflight_reason=None,
                    )
                )
    return jobs


def _typed_effect_interval_groups(
    atomic_rubric: _AtomicRubricDefinition,
    *,
    projection: _RuntimeIntervalProjection,
    edited_intervals: list[AudioEvaluationInterval],
    unedited_intervals: list[AudioEvaluationInterval],
    source_duration_sec: float,
    output_duration_sec: float,
) -> list[_AtomicIntervalGroup]:
    operations = tuple(effect.operation for effect in projection.effects)
    if atomic_rubric.dimension == "unedited_region_consistency":
        if projection.edit_scope == "global" and all(
            effect.edit_scope == "global" for effect in projection.effects
        ):
            interval = AudioEvaluationInterval(
                interval_id="typed-global-preservation",
                start_sec=0.0,
                end_sec=source_duration_sec,
            )
            mapping = _full_clip_mapping(
                source_duration_sec=source_duration_sec,
                output_duration_sec=output_duration_sec,
            ).model_copy(update={"interval_id": interval.interval_id})
            return [
                _AtomicIntervalGroup(
                    intervals=(interval,),
                    target=None,
                    edit_scope="global",
                    interval_mappings=(mapping,),
                    effect_operations=operations,
                )
            ]
        if unedited_intervals:
            return [
                _AtomicIntervalGroup(
                    intervals=tuple(unedited_intervals),
                    target=None,
                    edit_scope="local",
                    effect_operations=operations,
                )
            ]
        has_local_support = any(
            effect.affected_support_mappings
            for effect in projection.effects
            if effect.edit_scope == "local"
        )
        if not has_local_support and any(
            effect.edit_scope == "global" for effect in projection.effects
        ):
            interval = AudioEvaluationInterval(
                interval_id="typed-global-preservation",
                start_sec=0.0,
                end_sec=source_duration_sec,
            )
            mapping = _full_clip_mapping(
                source_duration_sec=source_duration_sec,
                output_duration_sec=output_duration_sec,
            ).model_copy(update={"interval_id": interval.interval_id})
            return [
                _AtomicIntervalGroup(
                    intervals=(interval,),
                    target=None,
                    edit_scope="global",
                    interval_mappings=(mapping,),
                    effect_operations=operations,
                )
            ]
        return [
            _AtomicIntervalGroup(
                intervals=(),
                target=None,
                edit_scope="local",
                effect_operations=operations,
            )
        ]

    groups: list[_AtomicIntervalGroup] = []
    for effect in projection.effects:
        if effect.edit_scope == "global":
            interval = AudioEvaluationInterval(
                interval_id=f"{effect.effect_ref_id}:global",
                start_sec=0.0,
                end_sec=source_duration_sec,
            )
            mapping = _full_clip_mapping(
                source_duration_sec=source_duration_sec,
                output_duration_sec=output_duration_sec,
            ).model_copy(update={"interval_id": interval.interval_id})
            groups.append(
                _AtomicIntervalGroup(
                    intervals=(interval,),
                    target=_generic_effect_target(effect, interval.interval_id),
                    edit_scope="global",
                    interval_mappings=(mapping,),
                    effect_operations=(effect.operation,),
                )
            )
            continue

        mappings = _typed_local_effect_mappings(atomic_rubric, effect)
        if not mappings:
            unavailable_interval_id = f"{effect.effect_ref_id}:anchor-unavailable"
            groups.append(
                _AtomicIntervalGroup(
                    intervals=(),
                    target=_generic_effect_target(effect, unavailable_interval_id),
                    edit_scope="local",
                    interval_mappings=(),
                    effect_operations=(effect.operation,),
                )
            )
            continue
        intervals = [
            AudioEvaluationInterval(
                interval_id=mapping.interval_id,
                start_sec=mapping.source_start_sec,
                end_sec=mapping.source_end_sec,
            )
            for mapping in mappings
        ]
        targets = _operation_authored_evaluation_targets(
            effect.step_record,
            intervals=intervals,
        )
        if len(targets) != len(intervals):
            targets = [
                _generic_effect_target(effect, interval.interval_id, index=index)
                for index, interval in enumerate(intervals)
            ]
        groups.extend(
            _AtomicIntervalGroup(
                intervals=(interval,),
                target=target,
                edit_scope="local",
                interval_mappings=(mapping,),
                effect_operations=(effect.operation,),
            )
            for interval, mapping, target in zip(
                intervals,
                mappings,
                targets,
                strict=True,
            )
        )
    return groups or [
        _AtomicIntervalGroup(
            intervals=tuple(edited_intervals),
            target=None,
            edit_scope=projection.edit_scope,
            effect_operations=operations,
        )
    ]


def _typed_local_effect_mappings(
    atomic_rubric: _AtomicRubricDefinition,
    effect: _RuntimeEffectIntervalProjection,
) -> tuple[AudioIntervalMapping, ...]:
    if atomic_rubric.atomic_rubric_id == "edited_instruction_observable":
        return effect.semantic_anchor_mappings
    if atomic_rubric.atomic_rubric_id == "edited_intelligibility":
        return effect.semantic_anchor_mappings or effect.affected_support_mappings
    return effect.affected_support_mappings


def _generic_effect_target(
    effect: _RuntimeEffectIntervalProjection,
    interval_id: str,
    *,
    index: int = 0,
) -> AudioEvaluationTarget:
    description = str(effect.step_record.get("description") or "").strip()
    requirement = description or (
        "Apply the requested audible effect associated with executed operation "
        f"{effect.operation!r}."
    )
    return AudioEvaluationTarget(
        target_ref_id=f"{effect.effect_ref_id}:target:{index:03d}",
        interval_id=interval_id,
        requirement=requirement,
        derivation="executed_operation_effect_scope",
    )


def _typed_global_effect_disposition(
    projection: _RuntimeIntervalProjection,
    atomic_rubric_id: AtomicRubricId,
) -> Literal["changed", "preserved", "unknown"] | None:
    property_name = _PRESERVATION_PROPERTIES.get(atomic_rubric_id)
    if property_name is None:
        return None
    operations = [
        effect.operation for effect in projection.effects if effect.edit_scope == "global"
    ]
    if not operations:
        return None
    return resolve_evaluation_effect_contract(operations).dispositions[property_name]


def _build_job(
    *,
    marker: dict[str, Any],
    rubric_scope_source: Literal[
        "frozen_sidecar",
        "runtime_temporal_transform",
        "runtime_typed_temporal_lineage",
        "runtime_graph_interval_projection",
        "runtime_reviewed_evidence",
        "public_full_clip_fallback",
        "unavailable",
    ],
    dimension: RubricDimension,
    instruction: str,
    source_asset_ref: str,
    intervals: list[AudioEvaluationInterval],
    interval_mappings: list[AudioIntervalMapping],
    atomic_rubric: _AtomicRubricDefinition,
    crops: list[AudioCropEvidence],
    preflight_status: Literal["ready", "abstain", "invalid", "not_applicable"],
    preflight_reason: str | None,
    active_execution_instruction: str | None = None,
    target: AudioEvaluationTarget | None = None,
    final_state_projection: FinalStateProjection | None = None,
    final_state_projection_source: Literal[
        "persisted_verified",
        "legacy_graph_recomputed",
        "unavailable",
    ] = "unavailable",
    effect_contract: EvaluationEffectContract | None = None,
    evaluation_region_kind: Literal[
        "edited_target",
        "unedited_complement",
        "global_edited_region_preservation",
        "unavailable",
    ] = "unavailable",
    applicability: Literal[
        "applicable",
        "edited_region_preservation",
        "not_applicable",
        "unknown",
    ] = "unknown",
    applicability_source: str | None = None,
    applicability_reason: str | None = None,
    wording_evidence: list[WordingPairEvidence] | None = None,
) -> _PreparedJob:
    request_body = {
        "assignment_fingerprint": marker["assignment_fingerprint"],
        "runtime_projection_fingerprint": marker["runtime_projection_fingerprint"],
        "assigned_arm": marker["assigned_arm"],
        "rubric_scope_source": rubric_scope_source,
        "dimension": dimension,
        "atomic_rubric_id": atomic_rubric.atomic_rubric_id,
        "atomic_rubric_catalog_version": _ATOMIC_RUBRIC_CATALOG_VERSION,
        "observable": atomic_rubric.observable,
        "public_user_instruction": instruction,
        "active_execution_instruction": active_execution_instruction,
        "target_ref_id": target.target_ref_id if target is not None else None,
        "target_requirement": target.requirement if target is not None else None,
        "target_derivation": target.derivation if target is not None else None,
        "final_state_projection": (
            final_state_projection.model_dump(mode="json")
            if final_state_projection is not None
            else None
        ),
        "final_state_projection_source": final_state_projection_source,
        "effect_contract": (
            effect_contract.model_dump(mode="json") if effect_contract is not None else None
        ),
        "evaluation_region_kind": evaluation_region_kind,
        "applicability": applicability,
        "applicability_source": applicability_source,
        "applicability_reason": applicability_reason,
        "source_asset_ref": source_asset_ref,
        "intervals": [item.model_dump(mode="json") for item in intervals],
        "interval_mappings": [item.model_dump(mode="json") for item in interval_mappings],
        "question": atomic_rubric.question,
        "crops": [item.model_dump(mode="json") for item in crops],
        "wording_evidence": [item.model_dump(mode="json") for item in wording_evidence or []],
        "preflight_status": preflight_status,
        "preflight_reason": preflight_reason,
    }
    request = AudioRubricRequest(
        **request_body,
        request_fingerprint=_request_fingerprint(request_body),
    )
    return _PreparedJob(
        request=request,
        audio_paths=tuple(crop.path for crop in crops),
    )


def _abstained_delivery_jobs(
    marker: dict[str, Any],
    *,
    instruction: str,
    reason: str,
) -> list[_PreparedJob]:
    return [
        _placeholder_job(
            marker,
            atomic_rubric=atomic_rubric,
            instruction=instruction,
            status="abstain",
            reason=reason,
        )
        for atomic_rubric in _ATOMIC_RUBRICS
    ]


def _invalid_delivery_jobs(
    marker: dict[str, Any],
    reason: str,
    *,
    instruction: str = "Unavailable public instruction",
) -> list[_PreparedJob]:
    return [
        _placeholder_job(
            marker,
            atomic_rubric=atomic_rubric,
            instruction=instruction,
            status="invalid",
            reason=reason,
        )
        for atomic_rubric in _ATOMIC_RUBRICS
    ]


def _placeholder_job(
    marker: dict[str, Any],
    *,
    atomic_rubric: _AtomicRubricDefinition,
    instruction: str,
    status: Literal["abstain", "invalid"],
    reason: str,
) -> _PreparedJob:
    body = {
        "assignment_fingerprint": marker["assignment_fingerprint"],
        "runtime_projection_fingerprint": marker["runtime_projection_fingerprint"],
        "assigned_arm": marker["assigned_arm"],
        "rubric_scope_source": "unavailable",
        "dimension": atomic_rubric.dimension,
        "atomic_rubric_id": atomic_rubric.atomic_rubric_id,
        "atomic_rubric_catalog_version": _ATOMIC_RUBRIC_CATALOG_VERSION,
        "observable": atomic_rubric.observable,
        "public_user_instruction": instruction,
        "active_execution_instruction": None,
        "target_ref_id": None,
        "target_requirement": None,
        "target_derivation": None,
        "final_state_projection": None,
        "final_state_projection_source": "unavailable",
        "effect_contract": None,
        "evaluation_region_kind": "unavailable",
        "applicability": "unknown",
        "applicability_source": None,
        "applicability_reason": reason,
        "source_asset_ref": "unavailable",
        "intervals": [],
        "interval_mappings": [],
        "question": atomic_rubric.question,
        "crops": [],
        "wording_evidence": [],
        "preflight_status": status,
        "preflight_reason": reason,
    }
    return _PreparedJob(
        request=AudioRubricRequest(
            **body,
            request_fingerprint=_request_fingerprint(body),
        ),
        audio_paths=(),
    )


def _resolve_interval_mappings(
    intervals: list[AudioEvaluationInterval],
    *,
    edit_scope: Literal["local", "global"],
    result_row: dict[str, Any],
    producer_output_path: str,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
    runtime_interval_projection: _RuntimeIntervalProjection | None = None,
    active_turn_indices: frozenset[int] | None = None,
) -> list[AudioIntervalMapping] | None:
    if edit_scope == "global":
        return [
            _full_clip_mapping(
                source_duration_sec=source_duration_sec,
                output_duration_sec=output_duration_sec,
            )
        ]

    if (
        runtime_interval_projection is not None
        and runtime_interval_projection.edit_scope == "local"
    ):
        projected = _mappings_from_runtime_interval_projection(
            intervals,
            projection=runtime_interval_projection,
            source_duration_sec=source_duration_sec,
            output_duration_sec=output_duration_sec,
            tolerance_sec=tolerance_sec,
        )
        if projected is not None:
            return projected

    step_results = _completed_step_results(
        result_row,
        producer_output_path=producer_output_path,
        active_turn_indices=active_turn_indices,
    )
    composed = _mappings_from_timeline_ledger(intervals, step_results)
    if composed is not None:
        return composed
    if _step_results_have_temporal_conflict(step_results):
        return None
    explicit = _explicit_runtime_span_mappings(step_results)
    identity_outside_regions = bool(step_results) and all(
        _result_identity_outside_regions(result) for result in step_results
    )
    timeline_preserved = (
        bool(step_results)
        and _result_timeline_duration_preserved(step_results[-1])
        and abs(source_duration_sec - output_duration_sec) <= tolerance_sec
    )
    resolved: list[AudioIntervalMapping] = []
    for interval in intervals:
        if timeline_preserved:
            output_start = interval.start_sec
            output_end = interval.end_sec
            source = "runtime_timeline_preserved"
        else:
            mapped = _map_interval_from_runtime_spans(
                interval,
                explicit,
                source_duration_sec=source_duration_sec,
                output_duration_sec=output_duration_sec,
                tolerance_sec=tolerance_sec,
                identity_outside_regions=identity_outside_regions,
            )
            if mapped is None:
                return None
            output_start, output_end, source = mapped
        if output_end > output_duration_sec + tolerance_sec:
            return None
        resolved.append(
            AudioIntervalMapping(
                interval_id=interval.interval_id,
                source_start_sec=interval.start_sec,
                source_end_sec=interval.end_sec,
                output_start_sec=max(0.0, output_start),
                output_end_sec=min(output_duration_sec, output_end),
                mapping_source=source,
            )
        )
    return resolved


def _mappings_from_runtime_interval_projection(
    intervals: list[AudioEvaluationInterval],
    *,
    projection: _RuntimeIntervalProjection,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
) -> list[AudioIntervalMapping] | None:
    explicit = list(projection.interval_mappings)
    if not explicit:
        return None
    if any(
        mapping.source_end_sec > source_duration_sec + tolerance_sec
        or mapping.output_end_sec > output_duration_sec + tolerance_sec
        for mapping in explicit
    ):
        return None
    raw = [
        (
            mapping.source_start_sec,
            mapping.source_end_sec,
            mapping.output_start_sec,
            mapping.output_end_sec,
        )
        for mapping in explicit
    ]
    resolved: list[AudioIntervalMapping] = []
    for interval in intervals:
        exact = next(
            (
                mapping
                for mapping in explicit
                if abs(interval.start_sec - mapping.source_start_sec) <= tolerance_sec
                and abs(interval.end_sec - mapping.source_end_sec) <= tolerance_sec
            ),
            None,
        )
        if exact is not None:
            resolved.append(
                exact.model_copy(
                    update={
                        "interval_id": interval.interval_id,
                        "source_start_sec": interval.start_sec,
                        "source_end_sec": interval.end_sec,
                    }
                )
            )
            continue
        mapped = _map_interval_from_runtime_spans(
            interval,
            raw,
            source_duration_sec=source_duration_sec,
            output_duration_sec=output_duration_sec,
            tolerance_sec=tolerance_sec,
            identity_outside_regions=projection.identity_outside_regions,
        )
        if mapped is None:
            return None
        output_start, output_end, mapping_source = mapped
        resolved.append(
            AudioIntervalMapping(
                interval_id=interval.interval_id,
                source_start_sec=interval.start_sec,
                source_end_sec=interval.end_sec,
                output_start_sec=max(0.0, output_start),
                output_end_sec=min(output_duration_sec, output_end),
                mapping_source=mapping_source,
            )
        )
    return resolved


def _mappings_from_timeline_ledger(
    intervals: list[AudioEvaluationInterval],
    step_results: list[dict[str, Any]],
) -> list[AudioIntervalMapping] | None:
    if not step_results:
        return None
    payload = step_results[-1].get("audio_timeline_ledger")
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != "x-editor.timeline-ledger/v1"
        or payload.get("status") != "complete"
        or not isinstance(payload.get("pieces"), list)
    ):
        return None
    resolved: list[AudioIntervalMapping] = []
    for interval in intervals:
        target = _map_interval_from_serialized_ledger(
            interval.start_sec,
            interval.end_sec,
            payload["pieces"],
        )
        if target is None:
            return None
        resolved.append(
            AudioIntervalMapping(
                interval_id=interval.interval_id,
                source_start_sec=interval.start_sec,
                source_end_sec=interval.end_sec,
                output_start_sec=target[0],
                output_end_sec=target[1],
                mapping_source="runtime_composed_timeline_ledger",
            )
        )
    return resolved


def _map_interval_from_serialized_ledger(
    start_sec: float,
    end_sec: float,
    pieces: list[Any],
) -> tuple[float, float] | None:
    return _map_interval_from_serialized_ledger_direction(
        start_sec,
        end_sec,
        pieces,
        forward=True,
    )


def _map_interval_from_serialized_ledger_direction(
    start_sec: float,
    end_sec: float,
    pieces: list[Any],
    *,
    forward: bool,
) -> tuple[float, float] | None:
    fragments: list[tuple[float, float, float, float]] = []
    for piece in pieces:
        if not isinstance(piece, dict):
            return None
        source_key, target_key = (
            ("base_interval", "current_interval")
            if forward
            else ("current_interval", "base_interval")
        )
        source = _span_bounds(piece.get(source_key))
        target = _span_bounds(piece.get(target_key))
        if source is None:
            continue
        overlap_start = max(start_sec, source[0])
        overlap_end = min(end_sec, source[1])
        if overlap_end <= overlap_start:
            continue
        if target is None:
            return None
        scale = (target[1] - target[0]) / (source[1] - source[0])
        mapped_start = target[0] + (overlap_start - source[0]) * scale
        mapped_end = target[0] + (overlap_end - source[0]) * scale
        fragments.append((overlap_start, overlap_end, mapped_start, mapped_end))
    fragments.sort()
    if not fragments:
        return None
    cursor = start_sec
    for source_start, source_end, _, _ in fragments:
        if abs(source_start - cursor) > 1e-6:
            return None
        cursor = source_end
    if abs(cursor - end_sec) > 1e-6:
        return None
    return fragments[0][2], fragments[-1][3]


def _full_clip_mapping(
    *,
    source_duration_sec: float,
    output_duration_sec: float,
) -> AudioIntervalMapping:
    return AudioIntervalMapping(
        interval_id="global-full-clip",
        source_start_sec=0.0,
        source_end_sec=source_duration_sec,
        output_start_sec=0.0,
        output_end_sec=output_duration_sec,
        mapping_source="global_full_clip",
    )


def _missing_boundary_context_reason(
    mappings: list[AudioIntervalMapping],
    *,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
) -> str | None:
    """Require real audio on both sides before asking about edit boundaries."""

    minimum_context_sec = max(0.001, tolerance_sec)
    for mapping in mappings:
        if (
            mapping.source_start_sec < minimum_context_sec
            or mapping.output_start_sec < minimum_context_sec
            or mapping.source_end_sec > source_duration_sec - minimum_context_sec
            or mapping.output_end_sec > output_duration_sec - minimum_context_sec
        ):
            return (
                "boundary continuity requires audible context on both sides of every "
                "named source/output boundary; at least one boundary touches or falls "
                "outside a media edge"
            )
    return None


def _completed_step_results(
    result_row: dict[str, Any],
    *,
    producer_output_path: str | None = None,
    active_turn_indices: frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    return [
        record["result"]
        for record in _completed_step_records(
            result_row,
            producer_output_path=producer_output_path,
            active_turn_indices=active_turn_indices,
        )
    ]


def _completed_step_records(
    result_row: dict[str, Any],
    *,
    producer_output_path: str | None = None,
    active_turn_indices: frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    arm_result_records = _completed_workflow_arm_result_records(result_row)
    if not arm_result_records:
        return []
    if active_turn_indices is not None:
        available = {turn_index for turn_index, _ in arm_result_records}
        if not active_turn_indices.issubset(available):
            return []
        arm_result_records = [
            item for item in arm_result_records if item[0] in active_turn_indices
        ]
    arm_results = [arm_result for _, arm_result in arm_result_records]
    records: list[dict[str, Any]] = []
    operation_graph_chains: list[list[dict[str, Any]]] = []
    for _, arm_result in arm_result_records:
        forward = arm_result.get("forward_result")
        if not isinstance(forward, dict):
            continue
        executed_plan = forward.get("executed_plan")
        if not isinstance(executed_plan, dict):
            continue
        steps = executed_plan.get("steps")
        if not isinstance(steps, list):
            continue
        arm_records: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict) or step.get("status") != "success":
                continue
            value = step.get("result")
            if isinstance(value, dict):
                transform = value.get("audio_temporal_transform")
                transform_operation = (
                    transform.get("operation") if isinstance(transform, dict) else None
                )
                arm_records.append(
                    {
                        "step_id": str(step.get("id") or ""),
                        "operation": str(step.get("operation") or transform_operation or ""),
                        "description": str(step.get("description") or ""),
                        "inputs": dict(step.get("inputs") or {}),
                        "params": dict(step.get("params") or {}),
                        "result": value,
                    }
                )
        graph_step_ids = _operation_graph_active_step_ids(
            executed_plan,
            producer_output_path=None if active_turn_indices is not None else producer_output_path,
        )
        if graph_step_ids is not None:
            if graph_step_ids:
                by_step_id = {record["step_id"]: record for record in arm_records}
                if len(by_step_id) != len(arm_records) or any(
                    step_id not in by_step_id for step_id in graph_step_ids
                ):
                    return []
                operation_graph_chains.append(
                    [by_step_id[step_id] for step_id in graph_step_ids]
                )
                if active_turn_indices is not None:
                    records.extend(operation_graph_chains[-1])
            continue
        records.extend(arm_records)

    if active_turn_indices is not None:
        if not records:
            return []
        if producer_output_path and not _step_record_produced_delivery(
            records[-1],
            producer_output_path=producer_output_path,
        ):
            return []
        return records
    if operation_graph_chains:
        return operation_graph_chains[0] if len(operation_graph_chains) == 1 else []
    if len(arm_results) == 1:
        return records
    if producer_output_path is None:
        return []
    active_results = _active_step_result_chain(
        [record["result"] for record in records],
        producer_output_path=producer_output_path,
    )
    active_result_ids = {id(result) for result in active_results}
    return [record for record in records if id(record["result"]) in active_result_ids]


def _step_record_produced_delivery(
    record: dict[str, Any],
    *,
    producer_output_path: str,
) -> bool:
    result = record.get("result")
    if not isinstance(result, dict):
        return False
    candidates = [str(result.get("output_audio_path") or "")]
    delivery = result.get("artifact_delivery_contract")
    if isinstance(delivery, dict):
        candidates.append(str(delivery.get("output_audio_path") or ""))
    for key in (AUDIO_TEMPORAL_LINEAGE_KEY, "audio_temporal_transform"):
        payload = result.get(key)
        if isinstance(payload, dict):
            candidates.append(str(payload.get("output_audio_path") or ""))
    resolved = [value for value in candidates if value.strip()]
    return bool(resolved) and all(
        _same_resolved_path(value, producer_output_path) for value in resolved
    )


def _operation_graph_active_step_ids(
    executed_plan: dict[str, Any],
    *,
    producer_output_path: str | None,
) -> tuple[str, ...] | None:
    """Read the actual OperationGraph only when it identifies the delivery node."""

    metadata = executed_plan.get("metadata")
    graph = metadata.get("operation_graph_actual") if isinstance(metadata, dict) else None
    if not isinstance(graph, dict):
        return None
    if graph.get("schema_version") != "x-editor.operation_graph.v1":
        return ()
    nodes = graph.get("nodes")
    operations = graph.get("operations")
    current_node_id = str(graph.get("current_node_id") or "")
    if not isinstance(nodes, list) or not isinstance(operations, list) or not current_node_id:
        return ()
    current_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and str(node.get("node_id") or "") == current_node_id
    ]
    if len(current_nodes) != 1:
        return ()
    current = current_nodes[0]
    current_path = str(current.get("path") or "")
    if producer_output_path and not _same_resolved_path(current_path, producer_output_path):
        return ()
    chain = current.get("operation_chain")
    if not isinstance(chain, list) or not chain:
        return ()
    step_ids = tuple(
        str(item.get("step_id") or "") for item in chain if isinstance(item, dict)
    )
    if not step_ids or any(not step_id for step_id in step_ids) or len(set(step_ids)) != len(step_ids):
        return ()
    successful_operations = {
        str(item.get("step_id") or ""): str(item.get("operation") or "")
        for item in operations
        if isinstance(item, dict) and item.get("status") == "success"
    }
    if any(step_id not in successful_operations for step_id in step_ids):
        return ()
    for item, step_id in zip(chain, step_ids, strict=True):
        operation = str(item.get("operation") or "")
        if operation and successful_operations[step_id] != operation:
            return ()
    return step_ids


def _active_step_result_chain(
    step_results: list[dict[str, Any]],
    *,
    producer_output_path: str,
) -> list[dict[str, Any]]:
    """Follow only transforms that contributed to the delivered artifact."""

    if not producer_output_path:
        return []

    indexed: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, result in enumerate(step_results):
        transform = result.get("audio_temporal_transform")
        if not isinstance(transform, dict):
            continue
        output_path = str(transform.get("output_audio_path") or "").strip()
        if output_path:
            indexed.append((index, result, transform))

    terminal = [
        item
        for item in indexed
        if _same_resolved_path(
            str(item[2].get("output_audio_path") or ""),
            producer_output_path,
        )
    ]
    if len(terminal) != 1:
        return []

    chain = [terminal[0]]
    seen = {terminal[0][0]}
    while True:
        input_paths = [
            str(value) for value in chain[0][2].get("input_audio_paths") or [] if str(value).strip()
        ]
        if not input_paths:
            break
        predecessors = [
            item
            for item in indexed
            if item[0] not in seen
            and any(
                _same_resolved_path(
                    str(item[2].get("output_audio_path") or ""),
                    input_path,
                )
                for input_path in input_paths
            )
        ]
        if not predecessors:
            break
        if len(predecessors) != 1:
            return []
        chain.insert(0, predecessors[0])
        seen.add(predecessors[0][0])
    return [result for _, result, _ in chain]


def _completed_workflow_arm_results(
    result_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Follow only the persisted benchmark -> scenario -> turn schemas."""

    return [
        arm_result
        for _, arm_result in _completed_workflow_arm_result_records(result_row)
    ]


def _completed_workflow_arm_result_records(
    result_row: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    """Return validated turn indices with their completed arm results."""

    if (
        result_row.get("schema_version") != "x-editor.benchmark-arm-result/v1"
        or result_row.get("status") != "completed"
    ):
        return []
    scenario = result_row.get("result")
    if (
        not isinstance(scenario, dict)
        or scenario.get("schema_version") != "x-editor.workflow-scenario-result/v1"
        or scenario.get("status") != "completed"
    ):
        return []
    turns = scenario.get("turn_results")
    if not isinstance(turns, list) or not turns:
        return []

    ordered: list[tuple[int, dict[str, Any]]] = []
    seen_turns: set[int] = set()
    for turn in turns:
        if (
            not isinstance(turn, dict)
            or turn.get("schema_version") != "x-editor.workflow-scenario-turn-result/v1"
        ):
            return []
        turn_index = turn.get("turn_index")
        if (
            isinstance(turn_index, bool)
            or not isinstance(turn_index, int)
            or turn_index < 0
            or turn_index in seen_turns
        ):
            return []
        arm_result = turn.get("result")
        if (
            not isinstance(arm_result, dict)
            or arm_result.get("schema_version") != "x-editor.workflow-arm-result/v1"
            or arm_result.get("status") != "completed"
        ):
            return []
        seen_turns.add(turn_index)
        ordered.append((turn_index, arm_result))
    if [index for index, _ in ordered] != sorted(seen_turns):
        return []
    return ordered


def _evaluation_execution_context(
    result_row: dict[str, Any],
    *,
    run_root: Path,
    producer_output_path: str,
) -> _EvaluationExecutionContext:
    arm_results = _completed_workflow_arm_results(result_row)
    active_instruction: str | None = None
    if arm_results:
        final_plan = arm_results[-1].get("final_plan")
        if isinstance(final_plan, dict):
            active_instruction = _optional_text(final_plan.get("request"))

    step_records = _completed_step_records(
        result_row,
        producer_output_path=producer_output_path,
    )
    operations = tuple(
        operation
        for record in step_records
        if (operation := _optional_text(record.get("operation"))) is not None
    )

    scenario_result = result_row.get("result")
    if not isinstance(scenario_result, dict):
        return _EvaluationExecutionContext(
            active_execution_instruction=active_instruction,
            final_state_projection=None,
            final_state_projection_source="unavailable",
            operations=operations,
            active_turn_indices=None,
        )
    graph_path_text = _optional_text(scenario_result.get("session_graph_path"))
    if graph_path_text is None:
        return _EvaluationExecutionContext(
            active_execution_instruction=active_instruction,
            final_state_projection=None,
            final_state_projection_source="unavailable",
            operations=operations,
            active_turn_indices=None,
        )
    graph_path = Path(graph_path_text).expanduser().resolve()
    try:
        graph_path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("session graph path is outside the frozen benchmark run") from exc
    graph_payload = _read_json(graph_path)
    recomputed = project_final_session_payload(graph_payload, graph_path=graph_path)
    if recomputed.final_audio_path and not _same_resolved_path(
        recomputed.final_audio_path,
        producer_output_path,
    ):
        raise ValueError("session graph final state does not match delivered producer output")
    persisted_raw = scenario_result.get("final_state_projection")
    if persisted_raw is None:
        source: Literal[
            "persisted_verified",
            "legacy_graph_recomputed",
            "unavailable",
        ] = "legacy_graph_recomputed"
    else:
        persisted = FinalStateProjection.model_validate(persisted_raw)
        if persisted != recomputed:
            raise ValueError("persisted final-state projection does not match session graph")
        source = "persisted_verified"
    active_turn_indices = frozenset(
        state.round_index - 1
        for state in recomputed.active_states
        if state.round_index > 0
    )
    step_records = _completed_step_records(
        result_row,
        producer_output_path=producer_output_path,
        active_turn_indices=active_turn_indices,
    )
    operations = tuple(
        operation
        for record in step_records
        if (operation := _optional_text(record.get("operation"))) is not None
    )
    active_session_operations = _active_session_branch_operations(
        graph_payload,
        final_state=recomputed,
    )
    final_round_operations = tuple(recomputed.final_operations)
    if (
        operations
        and final_round_operations
        and operations[-len(final_round_operations) :] != final_round_operations
    ):
        raise ValueError(
            "OperationGraph step chain does not match the final adopted Session Graph round"
        )
    if active_session_operations and operations != tuple(active_session_operations):
        raise ValueError(
            "OperationGraph step chain does not match the final adopted Session Graph branch"
        )
    if active_session_operations:
        operations = tuple(active_session_operations)
    return _EvaluationExecutionContext(
        active_execution_instruction=active_instruction,
        final_state_projection=recomputed,
        final_state_projection_source=source,
        operations=operations,
        active_turn_indices=active_turn_indices,
    )


def _active_session_branch_operations(
    graph_payload: dict[str, Any],
    *,
    final_state: FinalStateProjection,
) -> list[str]:
    """Flatten only edit operations on the final adopted Session Graph branch."""

    rounds = graph_payload.get("round_history")
    edges = graph_payload.get("edges")
    if not isinstance(rounds, list) or not isinstance(edges, dict):
        return []
    active_node_ids = {state.node_id for state in final_state.active_states}
    result: list[str] = []
    for record in rounds:
        if (
            not isinstance(record, dict)
            or record.get("status") != "completed"
            or str(record.get("result_node_id") or "") not in active_node_ids
        ):
            continue
        edge = edges.get(str(record.get("edge_id") or ""))
        if not isinstance(edge, dict):
            return []
        operations = edge.get("operations")
        if not isinstance(operations, list):
            return []
        result.extend(str(operation) for operation in operations if str(operation).strip())
    return result


def _explicit_runtime_span_mappings(
    step_results: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []

    for result in step_results:
        transform = result.get("audio_temporal_transform")
        mappings = (
            transform.get("timeline_mappings")
            if isinstance(transform, dict)
            else result.get("timeline_mappings")
        )
        if not isinstance(mappings, list):
            continue
        for value in mappings:
            if not isinstance(value, dict):
                continue
            source = _span_bounds(value.get("source_span") or value.get("source_interval"))
            output = _span_bounds(value.get("output_span") or value.get("output_interval"))
            if source is None or output is None:
                source = _direct_span_bounds(value, prefix="source")
                output = _direct_span_bounds(value, prefix="output")
            if source is not None and output is not None:
                row = (*source, *output)
                if row not in rows:
                    rows.append(row)
    return sorted(rows)


def _result_timeline_duration_preserved(result: dict[str, Any]) -> bool:
    transform = result.get("audio_temporal_transform")
    if isinstance(transform, dict):
        return transform.get("timeline_duration_preserved") is True
    return result.get("timeline_duration_preserved") is True


def _result_identity_outside_regions(result: dict[str, Any]) -> bool:
    transform = result.get("audio_temporal_transform")
    if isinstance(transform, dict):
        return transform.get("identity_outside_regions") is True
    return result.get("identity_outside_regions") is True


def _span_bounds(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        start_value = value.get("start_sec")
        end_value = value.get("end_sec")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start_value, end_value = value[0], value[1]
    else:
        return None
    return _finite_bounds(start_value, end_value)


def _direct_span_bounds(
    value: dict[str, Any],
    *,
    prefix: Literal["source", "output"],
) -> tuple[float, float] | None:
    return _finite_bounds(
        value.get(f"{prefix}_start_sec"),
        value.get(f"{prefix}_end_sec"),
    )


def _finite_bounds(start_value: Any, end_value: Any) -> tuple[float, float] | None:
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return None
    return start, end


def _map_interval_from_runtime_spans(
    interval: AudioEvaluationInterval,
    mappings: list[tuple[float, float, float, float]],
    *,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
    identity_outside_regions: bool,
) -> tuple[float, float, str] | None:
    for source_start, source_end, output_start, output_end in mappings:
        if (
            abs(interval.start_sec - source_start) <= tolerance_sec
            and abs(interval.end_sec - source_end) <= tolerance_sec
        ):
            return output_start, output_end, "runtime_explicit_span_mapping"

    piecewise_auditable = _piecewise_mapping_is_auditable(
        mappings,
        source_duration_sec=source_duration_sec,
        output_duration_sec=output_duration_sec,
        tolerance_sec=tolerance_sec,
    )
    boundary_anchored = identity_outside_regions and _boundary_anchors_are_auditable(
        mappings,
        source_duration_sec=source_duration_sec,
        output_duration_sec=output_duration_sec,
        tolerance_sec=tolerance_sec,
    )
    if not piecewise_auditable and not boundary_anchored:
        return None
    start = _map_unchanged_boundary(
        interval.start_sec,
        mappings,
        source_duration_sec=source_duration_sec,
        output_duration_sec=output_duration_sec,
        tolerance_sec=tolerance_sec,
    )
    end = _map_unchanged_boundary(
        interval.end_sec,
        mappings,
        source_duration_sec=source_duration_sec,
        output_duration_sec=output_duration_sec,
        tolerance_sec=tolerance_sec,
    )
    if start is None or end is None or end <= start:
        return None
    source = (
        "runtime_explicit_span_mapping"
        if piecewise_auditable
        else "runtime_identity_outside_boundary_anchored"
    )
    return start, end, source


def _piecewise_mapping_is_auditable(
    mappings: list[tuple[float, float, float, float]],
    *,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
) -> bool:
    if not mappings:
        return False
    ordered = sorted(mappings)
    for left, right in pairwise(ordered):
        if right[0] < left[1] - tolerance_sec:
            return False
        if right[2] < left[3] - tolerance_sec:
            return False
    final_delta = ordered[-1][3] - ordered[-1][1]
    return abs((source_duration_sec + final_delta) - output_duration_sec) <= tolerance_sec


def _boundary_anchors_are_auditable(
    mappings: list[tuple[float, float, float, float]],
    *,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
) -> bool:
    if not mappings:
        return False
    ordered = sorted(mappings)
    for source_start, source_end, output_start, output_end in ordered:
        if (
            source_end > source_duration_sec + tolerance_sec
            or output_end > output_duration_sec + tolerance_sec
            or source_start < 0.0
            or output_start < 0.0
        ):
            return False
    return all(
        right[0] >= left[1] - tolerance_sec and right[2] >= left[3] - tolerance_sec
        for left, right in pairwise(ordered)
    )


def _map_unchanged_boundary(
    value: float,
    mappings: list[tuple[float, float, float, float]],
    *,
    source_duration_sec: float,
    output_duration_sec: float,
    tolerance_sec: float,
) -> float | None:
    if abs(value) <= tolerance_sec:
        return 0.0
    if abs(value - source_duration_sec) <= tolerance_sec:
        return output_duration_sec
    delta = 0.0
    for source_start, source_end, output_start, output_end in sorted(mappings):
        if value < source_start - tolerance_sec:
            break
        if source_start + tolerance_sec < value < source_end - tolerance_sec:
            return None
        if value >= source_end - tolerance_sec:
            delta = output_end - source_end
        elif abs(value - source_start) <= tolerance_sec:
            return output_start
        else:
            return output_end
    return value + delta


def _materialize_job_crops(
    *,
    marker: dict[str, Any],
    dimension: RubricDimension,
    atomic_rubric_id: AtomicRubricId,
    mappings: list[AudioIntervalMapping],
    source_path: Path,
    output_path: Path,
    public_assets: list[dict[str, str]],
    source_asset_ref: str,
    crop_root: Path,
    crop_margin_sec: float,
    minimum_crop_sec: float,
) -> list[AudioCropEvidence]:
    destination = crop_root / marker["assignment_fingerprint"] / dimension / atomic_rubric_id
    margin = 0.0 if dimension == "unedited_region_consistency" else crop_margin_sec
    crops: list[AudioCropEvidence] = []
    for index, mapping in enumerate(mappings):
        token = f"{index:03d}-{_safe_token(mapping.interval_id)}"
        crops.append(
            _materialize_crop(
                source_path,
                requested_start_sec=mapping.source_start_sec,
                requested_end_sec=mapping.source_end_sec,
                margin_sec=margin,
                destination=destination / f"{token}-source.wav",
                crop_id=f"{token}-source",
                interval_id=mapping.interval_id,
                timebase="source",
                label=f"<audio source excerpt {mapping.interval_id}>",
                minimum_crop_sec=minimum_crop_sec,
            )
        )
        crops.append(
            _materialize_crop(
                output_path,
                requested_start_sec=mapping.output_start_sec,
                requested_end_sec=mapping.output_end_sec,
                margin_sec=margin,
                destination=destination / f"{token}-output.wav",
                crop_id=f"{token}-output",
                interval_id=mapping.interval_id,
                timebase="output",
                label=f"<audio output excerpt {mapping.interval_id}>",
                minimum_crop_sec=minimum_crop_sec,
            )
        )
    if dimension == "edited_region_instruction_following":
        for index, asset in enumerate(public_assets):
            if asset["asset_ref"] == source_asset_ref:
                continue
            reference_path = Path(asset["path"])
            duration = _audio_duration_sec(reference_path)
            crops.append(
                _materialize_crop(
                    reference_path,
                    requested_start_sec=0.0,
                    requested_end_sec=duration,
                    margin_sec=0.0,
                    destination=destination / f"reference-{index:03d}.wav",
                    crop_id=f"reference-{index:03d}",
                    interval_id=f"reference-{asset['asset_ref']}",
                    timebase="source",
                    label=f"<audio public reference {index + 1}>",
                    minimum_crop_sec=minimum_crop_sec,
                )
            )
    return crops


def _materialize_crop(
    original: Path,
    *,
    requested_start_sec: float,
    requested_end_sec: float,
    margin_sec: float,
    destination: Path,
    crop_id: str,
    interval_id: str,
    timebase: Literal["source", "output"],
    label: str,
    minimum_crop_sec: float,
) -> AudioCropEvidence:
    info = sf.info(str(original))
    duration = info.frames / info.samplerate
    start = max(0.0, requested_start_sec - margin_sec)
    end = min(duration, requested_end_sec + margin_sec)
    if end - start < minimum_crop_sec:
        raise ValueError(f"crop {crop_id} has no usable duration")
    start_frame = round(start * info.samplerate)
    end_frame = round(end * info.samplerate)
    audio, sample_rate = sf.read(
        str(original),
        start=start_frame,
        stop=end_frame,
        always_2d=True,
        dtype="float32",
    )
    if audio.size == 0:
        raise ValueError(f"crop {crop_id} decoded no samples")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(destination),
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        subtype="PCM_16",
    )
    actual_end = start + len(audio) / sample_rate
    return AudioCropEvidence(
        crop_id=crop_id,
        interval_id=interval_id,
        timebase=timebase,
        label=label,
        path=str(destination.resolve()),
        sha256=_sha256_file(destination),
        original_path=str(original.resolve()),
        original_sha256=_sha256_file(original),
        requested_start_sec=requested_start_sec,
        requested_end_sec=requested_end_sec,
        start_sec=start,
        end_sec=actual_end,
        original_duration_sec=duration,
    )


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return token[:80] or "interval"


def _score_job(
    job: _PreparedJob,
    *,
    client: _JudgeClient,
    options: PostDeliveryAudioScoringOptions,
) -> AudioRubricScoreResult:
    request = job.request
    attempts: list[AudioRubricAttempt] = []
    valid_scores: list[Literal[0, 1, 2, 3]] = []
    provider_attempts = 0
    while len(valid_scores) < options.valid_repetitions and len(attempts) < options.max_attempts:
        attempt_index = len(attempts) + 1
        user_text = _judge_user_prompt(request, sample_index=attempt_index)
        raw = ""
        _job_audio_fingerprint([job])
        try:
            raw = client.call(
                _SYSTEM_PROMPT,
                user_text,
                audio_paths=list(job.audio_paths),
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - failed attempts are evidence
            transport_attempts = _client_transport_attempts(client)
            provider_attempts += transport_attempts
            attempts.append(
                AudioRubricAttempt(
                    attempt_index=attempt_index,
                    status="provider_failure",
                    reason=f"{type(exc).__name__}: {str(exc)[:500]}",
                    provider_attempts=transport_attempts,
                )
            )
            _job_audio_fingerprint([job])
            continue
        transport_attempts = _client_transport_attempts(client)
        provider_attempts += transport_attempts
        _job_audio_fingerprint([job])
        raw_sha = _sha256_bytes(raw.encode("utf-8"))
        try:
            parsed = _JudgeResponse.model_validate(_parse_json_object(raw))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            attempts.append(
                AudioRubricAttempt(
                    attempt_index=attempt_index,
                    status="invalid",
                    reason=(f"invalid provider rubric response: {_error_summary(exc)}"),
                    provider_attempts=transport_attempts,
                    raw_response_sha256=raw_sha,
                    raw_response_excerpt=raw[:1000],
                )
            )
            continue
        if parsed.abstain:
            attempts.append(
                AudioRubricAttempt(
                    attempt_index=attempt_index,
                    status="abstain",
                    reason=parsed.reason,
                    provider_attempts=transport_attempts,
                    raw_response_sha256=raw_sha,
                    raw_response_excerpt=raw[:1000],
                )
            )
            continue
        assert parsed.score is not None
        valid_scores.append(parsed.score)
        attempts.append(
            AudioRubricAttempt(
                attempt_index=attempt_index,
                status="valid",
                score=parsed.score,
                positive_vote=parsed.score >= 2,
                reason=parsed.reason,
                provider_attempts=transport_attempts,
                raw_response_sha256=raw_sha,
                raw_response_excerpt=raw[:1000],
            )
        )

    positive_votes = sum(score >= 2 for score in valid_scores)
    complete = len(valid_scores) == options.valid_repetitions
    passed = complete and positive_votes >= options.required_positive_votes
    if complete:
        status: ScoreStatus = "pass" if passed else "fail"
        reason = f"{positive_votes}/{options.valid_repetitions} valid samples scored at least 2"
        mean_score = sum(valid_scores) / len(valid_scores)
    else:
        attempt_statuses = {item.status for item in attempts}
        if attempt_statuses == {"abstain"}:
            status = "abstain"
        elif attempt_statuses == {"provider_failure"}:
            status = "provider_failure"
        elif attempt_statuses == {"invalid"}:
            status = "invalid"
        else:
            status = "incomplete"
        reason = (
            f"collected {len(valid_scores)}/{options.valid_repetitions} valid "
            f"samples after {len(attempts)} attempts"
        )
        mean_score = None
    return AudioRubricScoreResult(
        request_fingerprint=request.request_fingerprint,
        assignment_fingerprint=request.assignment_fingerprint,
        assigned_arm=request.assigned_arm,
        dimension=request.dimension,
        atomic_rubric_id=request.atomic_rubric_id,
        status=status,
        valid_scores=valid_scores,
        positive_votes=positive_votes,
        valid_sample_count=len(valid_scores),
        attempt_count=len(attempts),
        mean_score=mean_score,
        passed=passed if complete else None,
        reason=reason,
        provider=QWEN3_OMNI_PROVIDER,
        model=getattr(client, "model_name", DEFAULT_QWEN3_OMNI_MODEL),
        provider_attempts=provider_attempts,
        attempts=attempts,
    )


def _client_transport_attempts(client: _JudgeClient) -> int:
    return max(
        0,
        int(getattr(client, "last_transport_attempt_count", 0) or 0),
    )


def _preflight_result(request: AudioRubricRequest) -> AudioRubricScoreResult:
    return _terminal_result(
        request,
        status=request.preflight_status,
        reason=request.preflight_reason or "rubric preflight did not pass",
    )


def _terminal_result(
    request: AudioRubricRequest,
    *,
    status: Literal[
        "dry_run",
        "abstain",
        "incomplete",
        "invalid",
        "provider_failure",
        "not_applicable",
    ],
    reason: str,
    provider: str | None = None,
    model: str | None = None,
    provider_attempts: int = 0,
) -> AudioRubricScoreResult:
    return AudioRubricScoreResult(
        request_fingerprint=request.request_fingerprint,
        assignment_fingerprint=request.assignment_fingerprint,
        assigned_arm=request.assigned_arm,
        dimension=request.dimension,
        atomic_rubric_id=request.atomic_rubric_id,
        status=status,
        reason=reason,
        provider=provider,
        model=model,
        provider_attempts=provider_attempts,
    )


def _judge_user_prompt(
    request: AudioRubricRequest,
    *,
    sample_index: int,
) -> str:
    pair_inventory, reference_inventory = _pair_prompt_inventory(request)
    attachment_inventory = "\n".join(
        (
            f"- attachment {index + 1}: {crop.label}; "
            f"timebase={crop.timebase}; pair_id={crop.interval_id}; "
            f"named_interval_sec=[{crop.requested_start_sec:.3f}, "
            f"{crop.requested_end_sec:.3f}]; actual_crop_sec=[{crop.start_sec:.3f}, "
            f"{crop.end_sec:.3f}]"
        )
        for index, crop in enumerate(request.crops)
    )
    atomic_rule = _atomic_evidence_rule(request)
    instruction_context = (
        "Omitted for this preservation-only rubric. Infer no edit history or "
        "additional expected difference; compare only the named source/output "
        "preservation pair and the stated applicability."
        if request.dimension == "unedited_region_consistency"
        else request.public_user_instruction
    )
    target_context = (
        (
            "Preserve only the atomic observable named by this rubric inside each "
            "listed source/output pair. No edit instruction applies inside this "
            "preservation-only comparison."
        )
        if request.dimension == "unedited_region_consistency"
        else (
            "This is one localized audio pair for the public request. Derive the "
            "expected audible change independently from the public user instruction "
            "and the attached source/output evidence. Planner-authored target values "
            "and operation choices are intentionally withheld because they may be "
            "wrong and are not scoring ground truth."
        )
    )
    preservation_only = request.dimension == "unedited_region_consistency"
    graph_context = (
        "Omitted for this preservation-only rubric; the frozen pair already reflects "
        "the selected Session Graph branch."
        if preservation_only
        else _final_state_prompt_context(request.final_state_projection)
    )
    effect_context = _effect_contract_prompt_context(
        request.effect_contract,
        preservation_only=preservation_only,
    )
    wording_context = _wording_evidence_prompt_context(request)
    return (
        f"Independent rubric sample: {sample_index}\n"
        f"Rubric dimension: {request.dimension}\n"
        f"Atomic rubric id: {request.atomic_rubric_id or 'legacy_dimension_rubric'}\n"
        f"Atomic observable:\n{request.observable or request.question}\n\n"
        f"Public user instruction:\n{instruction_context}\n\n"
        f"Target-specific requirement for this audio pair:\n{target_context}\n\n"
        "Final Session Graph projection (branch identity only; never evidence that an "
        "audible effect succeeded):\n"
        f"{graph_context}\n\n"
        "Rubric applicability:\n"
        f"region_kind={request.evaluation_region_kind}; "
        f"applicability={request.applicability}; "
        f"reason={request.applicability_reason or 'not recorded'}\n\n"
        f"Expected effect and preservation contract:\n{effect_context}\n\n"
        f"Independent wording evidence:\n{wording_context}\n\n"
        f"Atomic rubric question:\n{request.question}\n\n"
        f"Evidence rule:\n{atomic_rule}\n\n"
        "Boundary-context rule:\n"
        "An actual crop may extend before or after its named interval. That extra "
        "audio is boundary context and may contain unedited material. Do not score "
        "the context as part of the requested edit. Judge only the named interval; "
        "for the boundary-continuity rubric, use the context only to inspect the "
        "transition at the exact named start and end timestamps.\n\n"
        f"Timestamped evidence pairs:\n{pair_inventory}\n\n"
        f"Additional public references:\n{reference_inventory}\n\n"
        f"Attached audio inventory in submission order:\n{attachment_inventory}"
    )


def _effect_contract_prompt_context(
    contract: EvaluationEffectContract | None,
    *,
    preservation_only: bool,
) -> str:
    if contract is None:
        return "unavailable"
    return json.dumps(
        {
            "target_ref_id": contract.target_ref_id,
            "final_state_node_id": contract.final_state_node_id,
            "temporal_scope": contract.temporal_scope,
            "expected_dispositions": contract.dispositions,
            "conflicts": contract.conflicts,
            "interpretation_limit": (
                "Dispositions control rubric applicability only. They do not reveal "
                "the selected tool and cannot prove that any requested change occurred."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _wording_evidence_prompt_context(request: AudioRubricRequest) -> str:
    if request.atomic_rubric_id not in _WORDING_RUBRICS:
        return "not requested for this non-wording atomic rubric"
    if not request.wording_evidence:
        return (
            "not available; judge only the attached audio and abstain when wording "
            "cannot be identified responsibly"
        )
    return json.dumps(
        [
            {
                "pair_id": item.pair_id,
                "status": item.status,
                "minimum_confidence": item.minimum_confidence,
                "pair_confidence": item.pair_confidence,
                "source_transcript": (item.source.transcript if item.source is not None else None),
                "output_transcript": (item.output.transcript if item.output is not None else None),
                "source_language": (item.source.language if item.source is not None else None),
                "output_language": (item.output.language if item.output is not None else None),
                "transcript_similarity": item.transcript_similarity,
                "alignment_changes": [
                    change.model_dump(mode="json") for change in item.alignment_changes
                ],
                "reason": item.reason,
                "judge_rule": (
                    "ready evidence is supporting evidence, not an oracle"
                    if item.status == "ready"
                    else "do not use this evidence as proof; rely on audio or abstain"
                ),
            }
            for item in request.wording_evidence
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def _final_state_prompt_context(projection: FinalStateProjection | None) -> str:
    if projection is None:
        return "unavailable"
    excluded = [
        {
            "node_id": item.node_id,
            "round_index": item.round_index,
            "reason": item.reason,
        }
        for item in projection.excluded_round_states
    ]
    return json.dumps(
        {
            "authority": "session_graph",
            "current_node_id": projection.current_node_id,
            "active_node_ids": [item.node_id for item in projection.active_states],
            "final_selected_base_node_id": projection.final_selected_base_node_id,
            "final_round_entry_node_id": projection.final_round_entry_node_id,
            "excluded_round_states": excluded,
            "interpretation_limit": (
                "This proves branch selection, not perceptual quality. Do not require a "
                "narrow residual-edit crop to demonstrate changes located elsewhere."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _pair_prompt_inventory(request: AudioRubricRequest) -> tuple[str, str]:
    numbered_crops = list(enumerate(request.crops, start=1))
    pair_lines: list[str] = []
    mapped_interval_ids = {item.interval_id for item in request.interval_mappings}
    for mapping in request.interval_mappings:
        source_duration = mapping.source_end_sec - mapping.source_start_sec
        output_duration = mapping.output_end_sec - mapping.output_start_sec
        start_offset = mapping.output_start_sec - mapping.source_start_sec
        end_offset = mapping.output_end_sec - mapping.source_end_sec
        timing_lines: tuple[str, ...] = ()
        if request.atomic_rubric_id == "unedited_timing_consistency":
            timing_lines = (
                f"  source_local_duration_sec={source_duration:.3f}",
                f"  output_local_duration_sec={output_duration:.3f}",
                f"  local_duration_delta_sec={output_duration - source_duration:+.3f}",
                f"  absolute_start_offset_sec={start_offset:+.3f}",
                f"  absolute_end_offset_sec={end_offset:+.3f}",
                (
                    "  coordinate_rule=absolute offsets are localization metadata; "
                    "an equal start/end offset is a pure timeline translation and "
                    "must not lower the local-timing score"
                ),
            )
        source = next(
            (
                (index, crop)
                for index, crop in numbered_crops
                if crop.interval_id == mapping.interval_id and crop.timebase == "source"
            ),
            None,
        )
        output = next(
            (
                (index, crop)
                for index, crop in numbered_crops
                if crop.interval_id == mapping.interval_id and crop.timebase == "output"
            ),
            None,
        )
        source_attachment = (
            "not attached for this atomic rubric"
            if source is None
            else (
                f"attachment {source[0]}, actual_source_crop_sec="
                f"[{source[1].start_sec:.3f}, {source[1].end_sec:.3f}]"
            )
        )
        output_attachment = (
            "not attached"
            if output is None
            else (
                f"attachment {output[0]}, actual_output_crop_sec="
                f"[{output[1].start_sec:.3f}, {output[1].end_sec:.3f}]"
            )
        )
        sample_identical = (
            source is not None and output is not None and source[1].sha256 == output[1].sha256
        )
        identity_lines = ("  sample_identity_evidence=identical_pcm",) if sample_identical else ()
        pair_lines.append(
            "\n".join(
                (
                    f"- pair_id={mapping.interval_id}",
                    f"  mapping_source={mapping.mapping_source}",
                    (
                        "  named_source_interval_sec="
                        f"[{mapping.source_start_sec:.3f}, "
                        f"{mapping.source_end_sec:.3f}]"
                    ),
                    (
                        "  named_output_interval_sec="
                        f"[{mapping.output_start_sec:.3f}, "
                        f"{mapping.output_end_sec:.3f}]"
                    ),
                    f"  source_attachment={source_attachment}",
                    f"  output_attachment={output_attachment}",
                    *identity_lines,
                    *timing_lines,
                )
            )
        )
    reference_lines = [
        (
            f"- attachment {index}: {crop.label}; "
            f"actual_crop_sec=[{crop.start_sec:.3f}, {crop.end_sec:.3f}]"
        )
        for index, crop in numbered_crops
        if crop.interval_id not in mapped_interval_ids
    ]
    return (
        "\n".join(pair_lines) or "- no auditable interval pair",
        "\n".join(reference_lines) or "- none",
    )


def _atomic_evidence_rule(request: AudioRubricRequest) -> str:
    voice_change_rule = _voice_change_evidence_rule(request)
    atomic_rules: dict[AtomicRubricId, str] = {
        "edited_instruction_observable": (
            "Compare this one named source/output pair and independently derive the "
            "expected local change from the public request. Ignore public-request clauses "
            "whose effects clearly belong outside this interval. Planner-authored targets, "
            "selected tools, and execution success are unavailable by design and must not "
            "be inferred. Session Graph facts establish branch identity only; this narrow "
            "crop is responsible only for the public-request target audible here."
            f"{voice_change_rule}"
        ),
        "edited_intelligibility": (
            "Use each paired source attachment only for localization and comparison; "
            "score speech intelligibility inside the named output interval. Abstain "
            "when the requested result contains no speech to which intelligibility applies."
        ),
        "edited_naturalness": (
            "Use each paired source attachment only for localization and comparison; "
            "score whether the named output interval sounds natural for the requested "
            "effect. Do not score unrelated content or preservation properties."
        ),
        "edited_boundary_continuity": (
            "Use each paired source attachment only for localization and comparison. "
            "Inspect output transitions at the exact named start and end timestamps; "
            "surrounding crop audio exists only to make those boundaries audible. "
            "A named interval boundary is an internal edit boundary, not an audio "
            "endpoint or a requirement for silence. Speech may continue through and "
            "beyond that timestamp; do not call that continuation a cutoff. "
            "A changed interval duration or changed spoken content is not by itself a "
            "boundary defect. Score only audible clipping, pops, abrupt cuts, or an "
            "incoherent transition at the named boundary."
        ),
        "edited_artifact_absence": (
            "Use each paired source attachment only for localization and comparison; "
            "score unintended clipping, clicks, distortion, or added noise in the named "
            "output interval. Do not treat a requested effect as an artifact."
        ),
        "unedited_wording_consistency": (
            "Compare source and output within each named preservation pair separately and "
            "judge only whether audible spoken wording is preserved. Do not infer words "
            "from an edit instruction, operation name, timestamp, or silence. If the pair "
            "does not contain enough audible speech to identify wording responsibly, "
            "abstain unless sample_identity_evidence=identical_pcm already proves that the "
            "waveform did not "
            "change. For a sample-identical pair, cite waveform identity without inventing "
            "its linguistic content. The pair may cover an edited region when a different "
            "observable was intentionally changed."
        ),
        "unedited_speaker_consistency": (
            "Compare source and output within each named preservation pair separately and "
            "judge only whether speaker identity is preserved. Use audible speaker and "
            "timbre characteristics, not shared or differing words, transcript similarity, "
            "ASR, or alignment results. Matching spoken content proves only wording "
            "preservation, not speaker-identity preservation. Sample-identical PCM may "
            "still prove that every audible property in the pair is unchanged. The pair "
            "may cover an edited region when a different observable was intentionally "
            "changed."
        ),
        "unedited_background_consistency": (
            "Compare source and output within each named preservation pair separately and "
            "judge only whether background sound and acoustic environment are preserved. "
            "The pair may cover an edited region when another observable changed."
        ),
        "unedited_timing_consistency": (
            "Compare source and output within each named preservation pair separately and "
            "judge only whether local speech timing is preserved. The pair may cover an "
            "edited region when another observable changed. Ignore absolute start and end "
            "coordinates when assigning the score: they only localize the paired audio. An "
            "equal offset of both interval boundaries is a pure timeline translation and "
            "cannot be a local timing failure. Judge audible internal rhythm and local "
            "duration within the pair. If there is not enough audible speech to compare "
            "timing responsibly, abstain."
        ),
    }
    if request.atomic_rubric_id is not None:
        return atomic_rules[request.atomic_rubric_id]
    return {
        "edited_region_instruction_following": (
            "Compare each source/output edited pair and judge instruction following."
        ),
        "edited_region_quality": "Judge only the attached edited output excerpts.",
        "unedited_region_consistency": ("Compare each source/output unedited pair separately."),
    }[request.dimension]


def _voice_change_evidence_rule(request: AudioRubricRequest) -> str:
    contract = request.effect_contract
    if contract is None or not any(
        contract.dispositions.get(property_name) == "changed"
        for property_name in ("speaker_identity", "vocal_timbre")
    ):
        return ""

    rule = (
        " The effect contract declares speaker identity or vocal timbre as changed. "
        "For this voice-change judgement, matching source/output words, transcript, "
        "language, ASR, or alignment can establish only spoken-content preservation; "
        "none of them is evidence that speaker identity stayed the same or changed. "
        "Base the requested-effect score only on audible speaker and timbre "
        "characteristics."
    )
    has_public_reference = any(
        crop.label.startswith("<audio public reference ") for crop in request.crops
    )
    if has_public_reference:
        rule += (
            " A public voice reference is attached: make a three-way audible comparison "
            "of source, output, and reference. Judge whether the output voice moves away "
            "from the source voice toward the reference voice using only audible vocal "
            "timbre and speaker characteristics. Do not use matching or differing lexical "
            "content as evidence for or against the identity match."
        )
    return rule


def _generated_question(dimension: RubricDimension) -> str:
    if dimension == "unedited_region_consistency":
        return (
            "How consistently does the delivered audio preserve the source audio "
            "inside every frozen unedited interval?"
        )
    if dimension == "edited_region_quality":
        return (
            "How intelligible, natural, and artifact-free is the delivered audio "
            "inside every frozen edited interval?"
        )
    return (
        "How well does the delivered audio follow the public user instruction "
        "inside every frozen edited interval?"
    )


def _public_full_clip_rubric_spec(
    source_asset_ref: str,
) -> FrozenPostDeliveryAudioRubricSpec:
    return FrozenPostDeliveryAudioRubricSpec(
        source_asset_ref=source_asset_ref,
        edit_scope="global",
        rubrics=[
            FrozenRubricQuestion(
                dimension="unedited_region_consistency",
                question=(
                    "Across the full clip, how well does the output preserve all "
                    "speech, speaker traits, timing, and acoustics that the public "
                    "instruction does not ask to change?"
                ),
            ),
            FrozenRubricQuestion(
                dimension="edited_region_instruction_following",
                question=(
                    "Across the full clip, how well does the output follow the "
                    "public user instruction?"
                ),
            ),
            FrozenRubricQuestion(
                dimension="edited_region_quality",
                question=(
                    "Across the full clip, how intelligible, natural, and free of "
                    "audible artifacts is the delivered audio?"
                ),
            ),
        ],
    )


def _resolved_intervals(
    spec: FrozenPostDeliveryAudioRubricSpec,
    *,
    duration_sec: float,
) -> tuple[list[AudioEvaluationInterval], list[AudioEvaluationInterval]]:
    if spec.edit_scope == "global":
        edited = [
            AudioEvaluationInterval(
                interval_id="global-edit",
                start_sec=0.0,
                end_sec=duration_sec,
            )
        ]
        unedited = list(spec.unedited_intervals or []) or [
            AudioEvaluationInterval(
                interval_id="global-preservation",
                start_sec=0.0,
                end_sec=duration_sec,
            )
        ]
    else:
        edited = list(spec.edited_intervals)
        unedited = (
            list(spec.unedited_intervals)
            if spec.unedited_intervals is not None
            else _complement_intervals(
                edited,
                duration_sec=duration_sec,
                boundary_guard_sec=_INFERRED_UNEDITED_BOUNDARY_GUARD_SEC,
            )
        )
    for interval in [*edited, *unedited]:
        if interval.end_sec > duration_sec + 0.05:
            raise ValueError(
                f"interval {interval.interval_id!r} exceeds source duration {duration_sec:.3f}"
            )
    if spec.edit_scope != "global":
        _validate_disjoint_sets(edited, unedited)
    return edited, unedited


def _complement_intervals(
    edited: list[AudioEvaluationInterval],
    *,
    duration_sec: float,
    boundary_guard_sec: float = 0.0,
) -> list[AudioEvaluationInterval]:
    cursor = 0.0
    result: list[AudioEvaluationInterval] = []
    for index, interval in enumerate(sorted(edited, key=lambda item: item.start_sec)):
        preserved_end = max(0.0, interval.start_sec - boundary_guard_sec)
        if preserved_end > cursor:
            result.append(
                AudioEvaluationInterval(
                    interval_id=f"unedited-{index:03d}",
                    start_sec=cursor,
                    end_sec=preserved_end,
                )
            )
        cursor = max(
            cursor,
            min(duration_sec, interval.end_sec + boundary_guard_sec),
        )
    if cursor < duration_sec:
        result.append(
            AudioEvaluationInterval(
                interval_id=f"unedited-{len(result):03d}",
                start_sec=cursor,
                end_sec=duration_sec,
            )
        )
    return result


def _validate_non_overlapping(
    intervals: list[AudioEvaluationInterval],
    *,
    name: str,
) -> None:
    ordered = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    ids = [item.interval_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} interval ids must be unique")
    for left, right in pairwise(ordered):
        if right.start_sec < left.end_sec:
            raise ValueError(f"{name} intervals must not overlap")


def _validate_disjoint_sets(
    edited: list[AudioEvaluationInterval],
    unedited: list[AudioEvaluationInterval],
) -> None:
    for left in edited:
        for right in unedited:
            if max(left.start_sec, right.start_sec) < min(left.end_sec, right.end_sec):
                raise ValueError("edited and unedited intervals must not overlap")


def _verified_public_assets(assets: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if any(not isinstance(item, dict) for item in assets):
        raise ValueError("runtime audio asset is not an object")
    ordered = sorted(assets, key=lambda value: value.get("order", 0))
    for index, item in enumerate(ordered):
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"public runtime audio asset is missing: {path}")
        digest = _sha256_file(path)
        declared = item.get("content_sha256")
        if declared and digest != declared:
            raise ValueError(f"public runtime audio asset hash mismatch: {path}")
        rows.append(
            {
                "asset_ref": str(item.get("asset_ref") or ""),
                "role": str(item.get("role") or ""),
                "label": f"<audio input{index + 1}>",
                "path": str(path),
                "sha256": digest,
            }
        )
    return rows


def _public_instruction(scenario: dict[str, Any]) -> str:
    turns = scenario.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("runtime scenario has no public turns")
    requests = [
        str(item.get("user_request") or "").strip() for item in turns if isinstance(item, dict)
    ]
    if not requests or any(not item for item in requests):
        raise ValueError("runtime scenario has an empty public instruction")
    if len(requests) == 1:
        return requests[0]
    return "\n".join(f"Turn {index + 1}: {value}" for index, value in enumerate(requests))


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sidecar_scoring_spec(sidecar: dict[str, Any]) -> Any:
    evaluator_sidecar = sidecar.get("evaluator_sidecar")
    if not isinstance(evaluator_sidecar, dict):
        return None
    hidden_fields = evaluator_sidecar.get("hidden_fields")
    if not isinstance(hidden_fields, dict):
        return None
    # Intentionally do not inspect any other sidecar field.
    return hidden_fields.get(SIDECAR_SPEC_KEY)


def _mmae_style_rubric_spec(
    sidecar: dict[str, Any],
    *,
    source_asset_ref: str,
) -> dict[str, Any] | None:
    """Project frozen rubric questions without exposing choices or answers."""

    evaluator_sidecar = sidecar.get("evaluator_sidecar")
    if not isinstance(evaluator_sidecar, dict):
        return None
    hidden_fields = evaluator_sidecar.get("hidden_fields")
    if not isinstance(hidden_fields, dict):
        return None
    candidates: list[Any] = []
    direct = hidden_fields.get("rubrics")
    if isinstance(direct, list):
        candidates.extend(direct)
    evaluator_spec = hidden_fields.get("evaluator_spec")
    if isinstance(evaluator_spec, dict) and isinstance(evaluator_spec.get("rubrics"), list):
        candidates.extend(evaluator_spec["rubrics"])

    by_dimension: dict[RubricDimension, list[str]] = {
        "edited_region_instruction_following": [],
        "edited_region_quality": [],
        "unedited_region_consistency": [],
    }
    for item in candidates:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        category = str(item.get("category") or "").strip().lower()
        if not question or not category:
            continue
        if "quality" in category or "natural" in category or "artifact" in category:
            by_dimension["edited_region_quality"].append(question)
        elif "consistency" in category:
            by_dimension["unedited_region_consistency"].append(question)
        elif "instruction" in category or "following" in category:
            by_dimension["edited_region_instruction_following"].append(question)

    rubrics = [
        {
            "dimension": dimension,
            "question": "\n".join(
                f"{index + 1}. {question}" for index, question in enumerate(questions)
            ),
        }
        for dimension, questions in by_dimension.items()
        if questions
    ]
    if not rubrics:
        return None
    present = {item["dimension"] for item in rubrics}
    for dimension in _DIMENSIONS:
        if dimension not in present:
            rubrics.append(
                {
                    "dimension": dimension,
                    "question": _generated_question(dimension),
                }
            )
    return FrozenPostDeliveryAudioRubricSpec(
        source_asset_ref=source_asset_ref,
        edit_scope="global",
        rubrics=[FrozenRubricQuestion.model_validate(item) for item in rubrics],
    ).model_dump(mode="json")


def _runtime_temporal_transform_rubric_spec(
    result_row: dict[str, Any],
    *,
    source_audio_aliases: dict[str, frozenset[str]],
    rubrics: list[FrozenRubricQuestion],
    producer_output_path: str,
) -> FrozenPostDeliveryAudioRubricSpec | None:
    projection = _runtime_graph_interval_projection(
        result_row,
        source_audio_aliases=source_audio_aliases,
        producer_output_path=producer_output_path,
    )
    if projection is None:
        return None
    return _rubric_spec_from_runtime_projection(projection, rubrics=rubrics)


def _runtime_graph_interval_projection(
    result_row: dict[str, Any],
    *,
    source_audio_aliases: dict[str, frozenset[str]],
    producer_output_path: str,
    active_turn_indices: frozenset[int] | None = None,
    source_duration_by_ref: dict[str, float] | None = None,
    output_duration_sec: float | None = None,
) -> _RuntimeIntervalProjection | None:
    """Project intervals without consulting benchmark identity or expected answers."""

    step_records = _completed_step_records(
        result_row,
        producer_output_path=producer_output_path,
        active_turn_indices=active_turn_indices,
    )
    if not step_records:
        return None
    source_asset_ref = _source_asset_ref_for_step_records(
        step_records,
        source_audio_aliases=source_audio_aliases,
    )
    if source_asset_ref is None:
        return None
    reviewed_targets = _reviewed_runtime_target_projection(
        result_row,
        source_audio_aliases=source_audio_aliases,
        active_turn_indices=active_turn_indices,
    )

    typed_lineage_projection = _projection_from_typed_temporal_lineage(
        step_records,
        source_asset_ref=source_asset_ref,
        producer_output_path=producer_output_path,
    )
    if typed_lineage_projection is not None:
        return _attach_reviewed_semantic_anchors(
            typed_lineage_projection,
            reviewed_targets=reviewed_targets,
        )

    ledger_projection = _projection_from_complete_timeline_ledger(
        step_records,
        source_asset_ref=source_asset_ref,
    )
    if ledger_projection is not None:
        return ledger_projection

    if _step_records_have_temporal_conflict(step_records):
        return None

    transform_projection = _projection_from_complete_temporal_transform(
        step_records,
        source_asset_ref=source_asset_ref,
        producer_output_path=producer_output_path,
    )
    if transform_projection is not None:
        return transform_projection

    if len(step_records) != 1:
        return None
    record = step_records[0]
    profile = load_operation_evaluation_effect_catalog().by_operation().get(
        str(record.get("operation") or "")
    )
    if profile is None:
        return None
    typed_projection = _projection_from_typed_step_result(
        record,
        profile=profile,
        source_asset_ref=source_asset_ref,
        producer_output_path=producer_output_path,
        reviewed_targets=reviewed_targets,
        source_duration_sec=(source_duration_by_ref or {}).get(source_asset_ref),
        output_duration_sec=output_duration_sec,
    )
    if typed_projection is not None:
        return typed_projection
    if profile.temporal_scope == "global":
        return _global_step_interval_projection(
            record,
            source_asset_ref=source_asset_ref,
            producer_output_path=producer_output_path,
            rubric_scope_source="runtime_graph_interval_projection",
            projection_source="typed_operation_effect_global_scope",
        )
    return None


def _rubric_spec_from_runtime_projection(
    projection: _RuntimeIntervalProjection,
    *,
    rubrics: list[FrozenRubricQuestion],
) -> FrozenPostDeliveryAudioRubricSpec:
    local_scope_mappings = (
        projection.affected_support_mappings
        if projection.effects
        else projection.interval_mappings
    )
    if projection.edit_scope == "global" or not local_scope_mappings:
        return FrozenPostDeliveryAudioRubricSpec(
            source_asset_ref=projection.source_asset_ref,
            edit_scope="global",
            rubrics=rubrics,
        )
    intervals = (
        _merged_source_intervals(
            local_scope_mappings,
            interval_prefix="typed-lineage-support",
        )
        if projection.effects
        else [
            AudioEvaluationInterval(
                interval_id=mapping.interval_id,
                start_sec=mapping.source_start_sec,
                end_sec=mapping.source_end_sec,
            )
            for mapping in local_scope_mappings
        ]
    )
    targets = (
        []
        if projection.effects
        else _operation_authored_evaluation_targets(
            projection.step_records[-1],
            intervals=intervals,
        )
    )
    return FrozenPostDeliveryAudioRubricSpec(
        source_asset_ref=projection.source_asset_ref,
        edit_scope="local",
        edited_intervals=intervals,
        targets=targets,
        rubrics=rubrics,
    )


def _merged_source_intervals(
    mappings: tuple[AudioIntervalMapping, ...],
    *,
    interval_prefix: str,
) -> list[AudioEvaluationInterval]:
    ordered = sorted(
        (mapping.source_start_sec, mapping.source_end_sec) for mapping in mappings
    )
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [
        AudioEvaluationInterval(
            interval_id=f"{interval_prefix}:{index:03d}",
            start_sec=start,
            end_sec=end,
        )
        for index, (start, end) in enumerate(merged)
    ]


def _source_asset_ref_for_step_records(
    step_records: list[dict[str, Any]],
    *,
    source_audio_aliases: dict[str, frozenset[str]],
) -> str | None:
    first = step_records[0]
    result = first.get("result")
    candidates: list[str] = []
    if isinstance(result, dict):
        lineage = result.get(AUDIO_TEMPORAL_LINEAGE_KEY)
        if isinstance(lineage, dict):
            candidates.extend(str(item) for item in lineage.get("input_audio_refs") or [])
            candidates.extend(str(item) for item in lineage.get("input_audio_paths") or [])
        ledger = result.get("audio_timeline_ledger")
        if isinstance(ledger, dict):
            candidates.append(str(ledger.get("base_audio_ref") or ""))
        transform = result.get("audio_temporal_transform")
        if isinstance(transform, dict):
            candidates.extend(str(item) for item in transform.get("input_audio_refs") or [])
            candidates.extend(str(item) for item in transform.get("input_audio_paths") or [])
    candidates.extend(_nested_text_values(first.get("inputs")))
    matches = {
        source_asset_ref
        for value in candidates
        if (
            source_asset_ref := _source_asset_ref_for_alias(
                value,
                source_audio_aliases=source_audio_aliases,
            )
        )
        is not None
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _nested_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_text_values(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _nested_text_values(child)]
    return []


def _projection_from_typed_temporal_lineage(
    step_records: list[dict[str, Any]],
    *,
    source_asset_ref: str,
    producer_output_path: str,
) -> _RuntimeIntervalProjection | None:
    parsed: list[tuple[dict[str, Any], _TypedTemporalLineage]] = []
    for record in step_records:
        result = record.get("result")
        raw = result.get(AUDIO_TEMPORAL_LINEAGE_KEY) if isinstance(result, dict) else None
        if not isinstance(raw, dict):
            return None
        try:
            lineage = _TypedTemporalLineage.model_validate(raw)
        except ValidationError:
            return None
        if (
            lineage.status != "complete"
            or lineage.step_id != str(record.get("step_id") or "")
            or lineage.operation != str(record.get("operation") or "")
        ):
            return None
        parsed.append((record, lineage))
    if not parsed or not _same_resolved_path(
        parsed[-1][1].output_audio_path,
        producer_output_path,
    ):
        return None

    ledgers: list[dict[str, Any]] = []
    complete_composition = True
    previous_output_revision: int | None = None
    previous_output_duration: float | None = None
    for record, lineage in parsed:
        result = record.get("result")
        ledger = result.get("audio_timeline_ledger") if isinstance(result, dict) else None
        if (
            not _typed_lineage_structure_is_valid(lineage)
            or not isinstance(ledger, dict)
            or ledger.get("schema_version") != "x-editor.timeline-ledger/v1"
            or ledger.get("status") != "complete"
            or not isinstance(ledger.get("pieces"), list)
            or _ledger_revision(ledger) != lineage.output_revision
            or lineage.output_duration_sec is None
            or not _optional_duration_close(
                ledger.get("current_duration_sec"),
                lineage.output_duration_sec,
            )
            or (
                previous_output_revision is not None
                and lineage.input_revision != previous_output_revision
            )
            or (
                previous_output_duration is not None
                and not _optional_duration_close(
                    lineage.input_duration_sec,
                    previous_output_duration,
                )
            )
        ):
            complete_composition = False
            break
        ledgers.append(ledger)
        previous_output_revision = lineage.output_revision
        previous_output_duration = lineage.output_duration_sec

    structural_mappings: tuple[AudioIntervalMapping, ...] = ()
    if complete_composition:
        final_ledger = ledgers[-1]
        structural_mappings = tuple(
            AudioIntervalMapping(
                interval_id=f"typed-lineage-structural-{index:03d}",
                source_start_sec=source[0],
                source_end_sec=source[1],
                output_start_sec=output[0],
                output_end_sec=output[1],
                mapping_source="runtime_typed_temporal_lineage",
            )
            for index, piece in enumerate(final_ledger["pieces"])
            if isinstance(piece, dict)
            and piece.get("kind") != "identity"
            and (source := _span_bounds(piece.get("base_interval"))) is not None
            and (output := _span_bounds(piece.get("current_interval"))) is not None
        )

    effects: list[_RuntimeEffectIntervalProjection] = []
    affected_support: list[AudioIntervalMapping] = []
    for index, (record, lineage) in enumerate(parsed):
        if lineage.edit_scope == "global":
            effects.append(
                _RuntimeEffectIntervalProjection(
                    effect_ref_id=f"{lineage.step_id}:effect",
                    step_id=lineage.step_id,
                    operation=lineage.operation,
                    edit_scope="global",
                    semantic_anchor_mappings=(),
                    affected_support_mappings=(),
                    step_record=record,
                )
            )
            continue
        semantic = (
            _project_typed_lineage_mappings_to_final(
                lineage.semantic_anchors,
                before_step=ledgers[index - 1] if index else None,
                after_step=ledgers[index],
                final_ledger=ledgers[-1],
                first_input_duration_sec=parsed[0][1].input_duration_sec,
                interval_prefix=f"{lineage.step_id}:semantic",
            )
            if complete_composition
            else ()
        )
        support = (
            _project_typed_lineage_mappings_to_final(
                lineage.affected_support or lineage.structural_mappings,
                before_step=ledgers[index - 1] if index else None,
                after_step=ledgers[index],
                final_ledger=ledgers[-1],
                first_input_duration_sec=parsed[0][1].input_duration_sec,
                interval_prefix=f"{lineage.step_id}:support",
            )
            if complete_composition
            else ()
        )
        affected_support.extend(support)
        effects.append(
            _RuntimeEffectIntervalProjection(
                effect_ref_id=f"{lineage.step_id}:effect",
                step_id=lineage.step_id,
                operation=lineage.operation,
                edit_scope="local",
                semantic_anchor_mappings=semantic,
                affected_support_mappings=support,
                step_record=record,
            )
        )

    local_effects = [effect for effect in effects if effect.edit_scope == "local"]
    global_effects = [effect for effect in effects if effect.edit_scope == "global"]
    if not global_effects and (
        not complete_composition
        or not any(
            effect.semantic_anchor_mappings or effect.affected_support_mappings
            for effect in local_effects
        )
    ):
        return None
    primary_local_mappings = tuple(
        mapping
        for effect in local_effects
        for mapping in (
            effect.semantic_anchor_mappings or effect.affected_support_mappings
        )
    )
    return _RuntimeIntervalProjection(
        source_asset_ref=source_asset_ref,
        edit_scope="local" if local_effects else "global",
        interval_mappings=structural_mappings or primary_local_mappings,
        identity_outside_regions=complete_composition,
        rubric_scope_source="runtime_typed_temporal_lineage",
        projection_source="active_branch_typed_temporal_lineage",
        step_records=tuple(step_records),
        structural_mappings=structural_mappings,
        affected_support_mappings=tuple(affected_support),
        effects=tuple(effects),
    )


def _attach_reviewed_semantic_anchors(
    projection: _RuntimeIntervalProjection,
    *,
    reviewed_targets: _ReviewedRuntimeTargetProjection | None,
) -> _RuntimeIntervalProjection:
    """Map reviewed source targets through a complete, single-effect lineage."""

    if (
        reviewed_targets is None
        or reviewed_targets.source_asset_ref != projection.source_asset_ref
    ):
        return projection
    local_effects = [effect for effect in projection.effects if effect.edit_scope == "local"]
    if len(local_effects) != 1 or local_effects[0].semantic_anchor_mappings:
        return projection
    effect = local_effects[0]
    support = effect.affected_support_mappings or projection.structural_mappings
    if not support:
        return projection
    final_result = projection.step_records[-1].get("result")
    final_ledger = (
        final_result.get("audio_timeline_ledger")
        if isinstance(final_result, dict)
        else None
    )
    if (
        not isinstance(final_ledger, dict)
        or final_ledger.get("schema_version") != "x-editor.timeline-ledger/v1"
        or final_ledger.get("status") != "complete"
        or not isinstance(final_ledger.get("pieces"), list)
    ):
        return projection

    mappings: list[AudioIntervalMapping] = []
    for index, interval in enumerate(reviewed_targets.intervals):
        owners = [
            candidate
            for candidate in support
            if interval.start_sec >= candidate.source_start_sec - 1e-4
            and interval.end_sec <= candidate.source_end_sec + 1e-4
        ]
        if len(owners) != 1:
            return projection
        output = _map_interval_from_serialized_ledger_direction(
            interval.start_sec,
            interval.end_sec,
            final_ledger["pieces"],
            forward=True,
        )
        if output is None:
            return projection
        mappings.append(
            AudioIntervalMapping(
                interval_id=f"{effect.step_id}:reviewed-semantic:{index:03d}",
                source_start_sec=interval.start_sec,
                source_end_sec=interval.end_sec,
                output_start_sec=output[0],
                output_end_sec=output[1],
                mapping_source="runtime_typed_temporal_lineage",
            )
        )
    if not _projection_mappings_are_ordered(mappings):
        return projection
    updated_effect = replace(effect, semantic_anchor_mappings=tuple(mappings))
    return replace(
        projection,
        effects=tuple(
            updated_effect if item.effect_ref_id == effect.effect_ref_id else item
            for item in projection.effects
        ),
    )


def _typed_lineage_structure_is_valid(
    lineage: _TypedTemporalLineage,
) -> bool:
    input_duration = lineage.input_duration_sec
    output_duration = lineage.output_duration_sec
    if input_duration is None or output_duration is None:
        return False
    if lineage.edit_scope == "global":
        return (
            lineage.exact_timeline_duration_preserved is True
            and abs(input_duration - output_duration) <= 1e-4
        )
    if lineage.edit_scope != "local" or not lineage.structural_mappings:
        return False
    cumulative_delta = 0.0
    previous_source_end = 0.0
    for mapping in sorted(
        lineage.structural_mappings,
        key=lambda item: item.source_span.start_sec,
    ):
        source = mapping.source_span
        output = mapping.output_span
        if source.start_sec < previous_source_end or source.end_sec > input_duration + 1e-4:
            return False
        expected_output_start = source.start_sec + cumulative_delta
        if abs(output.start_sec - expected_output_start) > 1e-4:
            return False
        cumulative_delta += (output.end_sec - output.start_sec) - (
            source.end_sec - source.start_sec
        )
        previous_source_end = source.end_sec
    return abs((input_duration + cumulative_delta) - output_duration) <= 1e-4


def _project_typed_lineage_mappings_to_final(
    mappings: list[_TypedTemporalLineageMapping],
    *,
    before_step: dict[str, Any] | None,
    after_step: dict[str, Any],
    final_ledger: dict[str, Any],
    first_input_duration_sec: float | None,
    interval_prefix: str,
) -> tuple[AudioIntervalMapping, ...]:
    projected: list[AudioIntervalMapping] = []
    for index, mapping in enumerate(mappings):
        source_base = (
            (
                mapping.source_span.start_sec,
                mapping.source_span.end_sec,
            )
            if before_step is None
            and first_input_duration_sec is not None
            and mapping.source_span.end_sec <= first_input_duration_sec + 1e-4
            else _map_interval_from_serialized_ledger_direction(
                mapping.source_span.start_sec,
                mapping.source_span.end_sec,
                before_step.get("pieces", []) if isinstance(before_step, dict) else [],
                forward=False,
            )
        )
        output_base = _map_interval_from_serialized_ledger_direction(
            mapping.output_span.start_sec,
            mapping.output_span.end_sec,
            after_step.get("pieces", []),
            forward=False,
        )
        output_final = (
            _map_interval_from_serialized_ledger_direction(
                output_base[0],
                output_base[1],
                final_ledger.get("pieces", []),
                forward=True,
            )
            if output_base is not None
            else None
        )
        if source_base is None or output_final is None:
            continue
        projected.append(
            AudioIntervalMapping(
                interval_id=f"{interval_prefix}:{index:03d}",
                source_start_sec=source_base[0],
                source_end_sec=source_base[1],
                output_start_sec=output_final[0],
                output_end_sec=output_final[1],
                mapping_source="runtime_typed_temporal_lineage",
            )
        )
    return tuple(projected)


def _ledger_revision(ledger: dict[str, Any]) -> int | None:
    timebase = ledger.get("current_timebase")
    revision = timebase.get("revision") if isinstance(timebase, dict) else None
    return revision if isinstance(revision, int) and not isinstance(revision, bool) else None


def _optional_duration_close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-4
    except (TypeError, ValueError):
        return False


_TEMPORAL_CONFLICT_REASON_CODES = frozenset(
    {
        "explicit_mapping_disagrees_with_piecewise_delta",
        "invalid_piecewise_temporal_mapping",
        "temporal_mapping_exceeds_input",
        "legacy_pipeline_final_duration_mismatch",
        "legacy_pipeline_structural_output_mismatch",
    }
)


def _step_records_have_temporal_conflict(
    step_records: list[dict[str, Any]],
) -> bool:
    return _step_results_have_temporal_conflict(
        [
            result
            for record in step_records
            if isinstance((result := record.get("result")), dict)
        ]
    )


def _step_results_have_temporal_conflict(
    step_results: list[dict[str, Any]],
) -> bool:
    for result in step_results:
        raw_lineage = result.get(AUDIO_TEMPORAL_LINEAGE_KEY)
        if isinstance(raw_lineage, dict):
            try:
                lineage = _TypedTemporalLineage.model_validate(raw_lineage)
            except ValidationError:
                return True
            if (
                lineage.reason_code in _TEMPORAL_CONFLICT_REASON_CODES
                or (
                    lineage.status == "complete"
                    and lineage.edit_scope == "local"
                    and not _typed_lineage_structure_is_valid(lineage)
                )
            ):
                return True
        ledger = result.get("audio_timeline_ledger")
        if not isinstance(ledger, dict):
            continue
        if (
            ledger.get("status") != "complete"
            and str(ledger.get("reason_code") or "")
            in _TEMPORAL_CONFLICT_REASON_CODES
        ):
            return True
        transform = result.get("audio_temporal_transform")
        if (
            ledger.get("status") == "complete"
            and isinstance(ledger.get("pieces"), list)
            and all(
                isinstance(piece, dict) and piece.get("kind") == "identity"
                for piece in ledger["pieces"]
            )
            and isinstance(transform, dict)
            and transform.get("status") == "complete"
            and transform.get("edit_scope") == "local"
            and bool(transform.get("timeline_mappings"))
        ):
            return True
    return False


def _projection_from_complete_timeline_ledger(
    step_records: list[dict[str, Any]],
    *,
    source_asset_ref: str,
) -> _RuntimeIntervalProjection | None:
    result = step_records[-1].get("result")
    ledger = result.get("audio_timeline_ledger") if isinstance(result, dict) else None
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != "x-editor.timeline-ledger/v1"
        or ledger.get("status") != "complete"
        or not isinstance(ledger.get("pieces"), list)
    ):
        return None
    mappings: list[AudioIntervalMapping] = []
    for index, piece in enumerate(ledger["pieces"]):
        if not isinstance(piece, dict):
            return None
        if piece.get("kind") == "identity":
            continue
        source = _span_bounds(piece.get("base_interval"))
        output = _span_bounds(piece.get("current_interval"))
        if source is None or output is None:
            return None
        mappings.append(
            AudioIntervalMapping(
                interval_id=f"runtime-ledger-{index:03d}",
                source_start_sec=source[0],
                source_end_sec=source[1],
                output_start_sec=output[0],
                output_end_sec=output[1],
                mapping_source="runtime_composed_timeline_ledger",
            )
        )
    if not mappings or not _projection_mappings_are_ordered(mappings):
        return None
    return _RuntimeIntervalProjection(
        source_asset_ref=source_asset_ref,
        edit_scope="local",
        interval_mappings=tuple(mappings),
        identity_outside_regions=True,
        rubric_scope_source="runtime_graph_interval_projection",
        projection_source="final_active_branch_composed_timeline_ledger",
        step_records=tuple(step_records),
    )


def _projection_from_complete_temporal_transform(
    step_records: list[dict[str, Any]],
    *,
    source_asset_ref: str,
    producer_output_path: str,
) -> _RuntimeIntervalProjection | None:
    transforms: list[dict[str, Any]] = []
    for record in step_records:
        result = record.get("result")
        transform = result.get("audio_temporal_transform") if isinstance(result, dict) else None
        if (
            not isinstance(transform, dict)
            or transform.get("schema_version") != "x-editor.audio-temporal-transform/v1"
            or transform.get("status") != "complete"
        ):
            return None
        transforms.append(transform)
    final_transform = transforms[-1]
    if not _same_resolved_path(
        str(final_transform.get("output_audio_path") or ""),
        producer_output_path,
    ):
        return None
    if final_transform.get("edit_scope") == "global":
        if any(transform.get("edit_scope") != "global" for transform in transforms):
            return None
        return _RuntimeIntervalProjection(
            source_asset_ref=source_asset_ref,
            edit_scope="global",
            interval_mappings=(),
            identity_outside_regions=False,
            rubric_scope_source="runtime_temporal_transform",
            projection_source="executor_complete_global_temporal_transform",
            step_records=tuple(step_records),
            effects=tuple(
                _global_effect_projection(record)
                for record in step_records
            ),
        )
    if len(transforms) != 1 or final_transform.get("edit_scope") != "local":
        return None
    mappings = _audio_interval_mappings_from_rows(
        final_transform.get("timeline_mappings"),
        step_id=str(step_records[-1].get("step_id") or "step"),
        mapping_source="runtime_explicit_span_mapping",
        interval_prefix="runtime-transform",
    )
    if not mappings:
        return None
    return _RuntimeIntervalProjection(
        source_asset_ref=source_asset_ref,
        edit_scope="local",
        interval_mappings=tuple(mappings),
        identity_outside_regions=final_transform.get("identity_outside_regions") is True,
        rubric_scope_source="runtime_temporal_transform",
        projection_source="executor_complete_local_temporal_transform",
        step_records=tuple(step_records),
    )


def _global_effect_projection(
    step_record: dict[str, Any],
) -> _RuntimeEffectIntervalProjection:
    step_id = str(step_record.get("step_id") or "step")
    return _RuntimeEffectIntervalProjection(
        effect_ref_id=f"{step_id}:effect",
        step_id=step_id,
        operation=str(step_record.get("operation") or ""),
        edit_scope="global",
        semantic_anchor_mappings=(),
        affected_support_mappings=(),
        step_record=step_record,
    )


def _global_step_interval_projection(
    step_record: dict[str, Any],
    *,
    source_asset_ref: str,
    producer_output_path: str,
    rubric_scope_source: Literal[
        "runtime_temporal_transform",
        "runtime_typed_temporal_lineage",
        "runtime_graph_interval_projection",
    ],
    projection_source: str,
) -> _RuntimeIntervalProjection | None:
    if not _step_record_produced_delivery(
        step_record,
        producer_output_path=producer_output_path,
    ):
        return None
    return _RuntimeIntervalProjection(
        source_asset_ref=source_asset_ref,
        edit_scope="global",
        interval_mappings=(),
        identity_outside_regions=False,
        rubric_scope_source=rubric_scope_source,
        projection_source=projection_source,
        step_records=(step_record,),
        effects=(_global_effect_projection(step_record),),
    )


def _projection_from_typed_step_result(
    step_record: dict[str, Any],
    *,
    profile: OperationEvaluationEffectProfile,
    source_asset_ref: str,
    producer_output_path: str,
    reviewed_targets: _ReviewedRuntimeTargetProjection | None,
    source_duration_sec: float | None,
    output_duration_sec: float | None,
) -> _RuntimeIntervalProjection | None:
    result = step_record.get("result")
    if not isinstance(result, dict) or not _step_record_produced_delivery(
        step_record,
        producer_output_path=producer_output_path,
    ):
        return None
    contract = profile.local_interval_contract
    if contract == "aligned_edit_span_pairs":
        mappings = _aligned_edit_span_pair_mappings(step_record)
    elif contract == "pause_insertion_plan":
        mappings = _pause_insertion_plan_mappings(step_record)
    elif contract == "source_target_spans":
        mappings = _source_target_span_mappings(
            step_record,
            reviewed_targets=reviewed_targets,
            source_asset_ref=source_asset_ref,
        )
    elif contract == "selected_spans_extraction":
        return _selected_span_extraction_projection(
            step_record,
            source_asset_ref=source_asset_ref,
            producer_output_path=producer_output_path,
            reviewed_targets=reviewed_targets,
            source_duration_sec=source_duration_sec,
            output_duration_sec=output_duration_sec,
        )
    elif contract == "selected_spans_removal":
        return _selected_span_removal_projection(
            step_record,
            source_asset_ref=source_asset_ref,
            producer_output_path=producer_output_path,
            reviewed_targets=reviewed_targets,
            source_duration_sec=source_duration_sec,
            output_duration_sec=output_duration_sec,
        )
    elif contract == "whole_source_text_coverage":
        if not _typed_result_covers_whole_source(result):
            return None
        return _global_step_interval_projection(
            step_record,
            source_asset_ref=source_asset_ref,
            producer_output_path=producer_output_path,
            rubric_scope_source="runtime_graph_interval_projection",
            projection_source="typed_result_whole_source_coverage",
        )
    else:
        return None
    if not mappings:
        return None
    return _RuntimeIntervalProjection(
        source_asset_ref=source_asset_ref,
        edit_scope="local",
        interval_mappings=tuple(mappings),
        identity_outside_regions=profile.identity_outside_local_intervals,
        rubric_scope_source="runtime_graph_interval_projection",
        projection_source=f"typed_step_result:{contract}",
        step_records=(step_record,),
    )


def _selected_span_extraction_projection(
    step_record: dict[str, Any],
    *,
    source_asset_ref: str,
    producer_output_path: str,
    reviewed_targets: _ReviewedRuntimeTargetProjection | None,
    source_duration_sec: float | None,
    output_duration_sec: float | None,
) -> _RuntimeIntervalProjection | None:
    if source_duration_sec is None or output_duration_sec is None:
        return None
    selected = _typed_selected_operation_intervals(
        step_record,
        source_asset_ref=source_asset_ref,
        reviewed_targets=reviewed_targets,
        source_duration_sec=source_duration_sec,
    )
    if selected is None:
        return None
    result = step_record.get("result")
    if not isinstance(result, dict):
        return None
    crossfade_ms = _typed_crossfade_ms(result)
    if crossfade_ms is None:
        return None
    mappings, predicted_output_sec = _extracted_span_mappings(
        selected,
        step_id=str(step_record.get("step_id") or "step"),
        crossfade_ms=crossfade_ms,
    )
    if (
        mappings is None
        or not _typed_structural_duration_matches(
            predicted_output_sec,
            output_duration_sec,
            interval_count=len(selected),
        )
        or not _typed_result_span_summary_matches(
            result,
            selected,
            duration_key="selected_duration_sec",
        )
    ):
        return None
    return _RuntimeIntervalProjection(
        source_asset_ref=source_asset_ref,
        edit_scope="local",
        interval_mappings=tuple(mappings),
        identity_outside_regions=False,
        rubric_scope_source="runtime_graph_interval_projection",
        projection_source="typed_result_selected_spans_extraction",
        step_records=(step_record,),
    )


def _selected_span_removal_projection(
    step_record: dict[str, Any],
    *,
    source_asset_ref: str,
    producer_output_path: str,
    reviewed_targets: _ReviewedRuntimeTargetProjection | None,
    source_duration_sec: float | None,
    output_duration_sec: float | None,
) -> _RuntimeIntervalProjection | None:
    if source_duration_sec is None or output_duration_sec is None:
        return None
    selected = _typed_selected_operation_intervals(
        step_record,
        source_asset_ref=source_asset_ref,
        reviewed_targets=reviewed_targets,
        source_duration_sec=source_duration_sec,
    )
    if selected is None:
        return None
    result = step_record.get("result")
    if not isinstance(result, dict):
        return None
    crossfade_ms = _typed_crossfade_ms(result)
    if crossfade_ms is None:
        return None
    retained = _complement_bounds(selected, duration_sec=source_duration_sec)
    if not retained:
        return None
    predicted_output_sec = _joined_output_duration_sec(
        retained,
        crossfade_ms=crossfade_ms,
    )
    if (
        not _typed_structural_duration_matches(
            predicted_output_sec,
            output_duration_sec,
            interval_count=len(selected),
        )
        or not _typed_result_span_summary_matches(
            result,
            selected,
            duration_key="removed_duration_sec",
        )
    ):
        return None
    return _global_step_interval_projection(
        step_record,
        source_asset_ref=source_asset_ref,
        producer_output_path=producer_output_path,
        rubric_scope_source="runtime_graph_interval_projection",
        projection_source="typed_result_selected_spans_removal",
    )


def _typed_selected_operation_intervals(
    step_record: dict[str, Any],
    *,
    source_asset_ref: str,
    reviewed_targets: _ReviewedRuntimeTargetProjection | None,
    source_duration_sec: float,
) -> list[tuple[float, float]] | None:
    result = step_record.get("result")
    if not isinstance(result, dict):
        return None
    raw_selected = result.get("selected_spans")
    if raw_selected is None:
        if (
            reviewed_targets is None
            or reviewed_targets.source_asset_ref != source_asset_ref
        ):
            return None
        selected = [
            (interval.start_sec, interval.end_sec)
            for interval in reviewed_targets.intervals
        ]
    else:
        selected = _bounds_list(raw_selected)
        if selected is None:
            return None
    selected = sorted(selected)
    if any(end > source_duration_sec + 1e-3 for _, end in selected):
        return None
    if any(right[0] < left[1] for left, right in pairwise(selected)):
        return None
    return selected


def _typed_crossfade_ms(result: dict[str, Any]) -> int | None:
    value = result.get("crossfade_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or abs(number - round(number)) > 1e-6:
        return None
    return round(number)


def _rounded_millisecond_bounds(
    intervals: list[tuple[float, float]],
) -> list[tuple[int, int]] | None:
    result = [
        (round(start * 1000), round(end * 1000))
        for start, end in intervals
    ]
    return result if all(end > start for start, end in result) else None


def _join_crossfades_ms(
    piece_durations_ms: list[int],
    *,
    crossfade_ms: int,
) -> tuple[list[int], int]:
    incoming = [0] * len(piece_durations_ms)
    output_duration_ms = 0
    for index, piece_duration_ms in enumerate(piece_durations_ms):
        overlap = (
            0
            if index == 0
            else min(crossfade_ms, output_duration_ms, piece_duration_ms)
        )
        incoming[index] = overlap
        output_duration_ms += piece_duration_ms - overlap
    return incoming, output_duration_ms


def _extracted_span_mappings(
    intervals: list[tuple[float, float]],
    *,
    step_id: str,
    crossfade_ms: int,
) -> tuple[list[AudioIntervalMapping] | None, float]:
    rounded = _rounded_millisecond_bounds(intervals)
    if rounded is None:
        return None, 0.0
    durations = [end - start for start, end in rounded]
    incoming, output_duration_ms = _join_crossfades_ms(
        durations,
        crossfade_ms=crossfade_ms,
    )
    output_cursor_ms = 0
    mappings: list[AudioIntervalMapping] = []
    for index, ((source_start, source_end), duration_ms) in enumerate(
        zip(rounded, durations, strict=True)
    ):
        incoming_overlap = incoming[index]
        outgoing_overlap = incoming[index + 1] if index + 1 < len(incoming) else 0
        output_after_append = output_cursor_ms + duration_ms - incoming_overlap
        mapped_source_start = source_start + incoming_overlap
        mapped_source_end = source_end - outgoing_overlap
        mapped_output_start = output_cursor_ms
        mapped_output_end = output_after_append - outgoing_overlap
        if mapped_source_end <= mapped_source_start or mapped_output_end <= mapped_output_start:
            return None, output_duration_ms / 1000.0
        mappings.append(
            AudioIntervalMapping(
                interval_id=f"{step_id}:target:{index + 1:03d}",
                source_start_sec=mapped_source_start / 1000.0,
                source_end_sec=mapped_source_end / 1000.0,
                output_start_sec=mapped_output_start / 1000.0,
                output_end_sec=mapped_output_end / 1000.0,
                mapping_source="runtime_typed_step_result",
            )
        )
        output_cursor_ms = output_after_append
    if not _projection_mappings_are_ordered(mappings):
        return None, output_duration_ms / 1000.0
    return mappings, output_duration_ms / 1000.0


def _joined_output_duration_sec(
    intervals: list[tuple[float, float]],
    *,
    crossfade_ms: int,
) -> float:
    rounded = _rounded_millisecond_bounds(intervals)
    if rounded is None:
        return 0.0
    _, output_duration_ms = _join_crossfades_ms(
        [end - start for start, end in rounded],
        crossfade_ms=crossfade_ms,
    )
    return output_duration_ms / 1000.0


def _complement_bounds(
    intervals: list[tuple[float, float]],
    *,
    duration_sec: float,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_sec:
        result.append((cursor, duration_sec))
    return result


def _typed_structural_duration_matches(
    predicted_sec: float,
    observed_sec: float,
    *,
    interval_count: int,
) -> bool:
    tolerance = max(0.005, 0.001 * (interval_count + 2))
    return predicted_sec > 0 and abs(predicted_sec - observed_sec) <= tolerance


def _typed_result_span_summary_matches(
    result: dict[str, Any],
    intervals: list[tuple[float, float]],
    *,
    duration_key: str,
) -> bool:
    span_count = result.get("span_count")
    if span_count is not None and (
        isinstance(span_count, bool)
        or not isinstance(span_count, int)
        or span_count != len(intervals)
    ):
        return False
    declared_duration = result.get(duration_key)
    if declared_duration is None:
        return True
    if isinstance(declared_duration, bool) or not isinstance(
        declared_duration,
        (int, float),
    ):
        return False
    expected = sum(end - start for start, end in intervals)
    return math.isfinite(float(declared_duration)) and abs(
        float(declared_duration) - expected
    ) <= 0.01


def _aligned_edit_span_pair_mappings(
    step_record: dict[str, Any],
) -> list[AudioIntervalMapping] | None:
    result = step_record.get("result")
    edit_result = result.get("edit_result") if isinstance(result, dict) else None
    rows = edit_result.get("edits_applied") if isinstance(edit_result, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    return _audio_interval_mappings_from_rows(
        rows,
        step_id=str(step_record.get("step_id") or "step"),
        mapping_source="runtime_typed_step_result",
        source_keys=("source_span", "source_interval"),
        output_keys=("target_span", "output_span", "output_interval"),
    )


def _source_target_span_mappings(
    step_record: dict[str, Any],
    *,
    reviewed_targets: _ReviewedRuntimeTargetProjection | None,
    source_asset_ref: str,
) -> list[AudioIntervalMapping] | None:
    result = step_record.get("result")
    if not isinstance(result, dict):
        return None
    source_rows = result.get("source_spans")
    output_rows = result.get("target_spans") or result.get("target_aligned_spans")
    if (
        (not isinstance(source_rows, list) or not source_rows)
        and reviewed_targets is not None
        and reviewed_targets.source_asset_ref == source_asset_ref
    ):
        source_rows = [
            [interval.start_sec, interval.end_sec]
            for interval in reviewed_targets.intervals
        ]
    if (
        not isinstance(source_rows, list)
        or not isinstance(output_rows, list)
        or not source_rows
        or len(source_rows) != len(output_rows)
    ):
        return None
    rows = [
        {"source_span": source, "target_span": output}
        for source, output in zip(source_rows, output_rows, strict=True)
    ]
    return _audio_interval_mappings_from_rows(
        rows,
        step_id=str(step_record.get("step_id") or "step"),
        mapping_source="runtime_typed_step_result",
        source_keys=("source_span",),
        output_keys=("target_span",),
    )


def _audio_interval_mappings_from_rows(
    rows: Any,
    *,
    step_id: str,
    mapping_source: Literal[
        "runtime_explicit_span_mapping",
        "runtime_typed_step_result",
    ],
    source_keys: tuple[str, ...] = ("source_span", "source_interval"),
    output_keys: tuple[str, ...] = ("output_span", "output_interval"),
    interval_prefix: str | None = None,
) -> list[AudioIntervalMapping] | None:
    if not isinstance(rows, list) or not rows:
        return None
    mappings: list[AudioIntervalMapping] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        source = next(
            (bounds for key in source_keys if (bounds := _span_bounds(row.get(key))) is not None),
            None,
        )
        output = next(
            (bounds for key in output_keys if (bounds := _span_bounds(row.get(key))) is not None),
            None,
        )
        if source is None or output is None:
            return None
        mappings.append(
            AudioIntervalMapping(
                interval_id=(
                    f"{interval_prefix}-{index:03d}"
                    if interval_prefix is not None
                    else f"{step_id}:target:{index + 1:03d}"
                ),
                source_start_sec=source[0],
                source_end_sec=source[1],
                output_start_sec=output[0],
                output_end_sec=output[1],
                mapping_source=mapping_source,
            )
        )
    return mappings if _projection_mappings_are_ordered(mappings) else None


def _pause_insertion_plan_mappings(
    step_record: dict[str, Any],
) -> list[AudioIntervalMapping] | None:
    result = step_record.get("result")
    plan = result.get("pause_edit_plan") if isinstance(result, dict) else None
    if not isinstance(plan, dict):
        return None
    anchors = _finite_number_list(plan.get("insert_position_sec"), positive=False)
    durations = _finite_number_list(plan.get("pause_duration_sec"), positive=True)
    pause_intervals = _bounds_list(plan.get("pause_interval_sec"))
    if (
        anchors is None
        or durations is None
        or pause_intervals is None
        or not anchors
        or len(anchors) != len(durations)
        or len(anchors) != len(pause_intervals)
        or anchors != sorted(anchors)
    ):
        return None
    cumulative = 0.0
    for anchor, duration, output in zip(anchors, durations, pause_intervals, strict=True):
        if (
            abs(output[0] - (anchor + cumulative)) > 0.05
            or abs((output[1] - output[0]) - duration) > 0.05
        ):
            return None
        cumulative += duration

    smoothing = plan.get("smoothing_points")
    support_intervals: list[tuple[float, float]] = []
    if isinstance(smoothing, list) and len(smoothing) == len(anchors):
        for item in smoothing:
            bounds = _span_bounds(item.get("regen_span_sec")) if isinstance(item, dict) else None
            if bounds is None:
                return None
            support_intervals.append(bounds)
    elif len(anchors) == 1:
        bounds = _span_bounds(plan.get("merged_regen_expected_span"))
        if bounds is None:
            return None
        support_intervals = [bounds]
    else:
        return None

    mappings: list[AudioIntervalMapping] = []
    step_id = str(step_record.get("step_id") or "step")
    for index, (pause, support) in enumerate(
        zip(pause_intervals, support_intervals, strict=True)
    ):
        if support[0] > pause[0] + 0.05 or support[1] < pause[1] - 0.05:
            return None
        source_start = _inverse_insertion_boundary(
            support[0],
            anchors=anchors,
            durations=durations,
            output_intervals=pause_intervals,
        )
        source_end = _inverse_insertion_boundary(
            support[1],
            anchors=anchors,
            durations=durations,
            output_intervals=pause_intervals,
        )
        if source_start is None or source_end is None or source_end <= source_start:
            return None
        mappings.append(
            AudioIntervalMapping(
                interval_id=f"{step_id}:target:{index + 1:03d}",
                source_start_sec=source_start,
                source_end_sec=source_end,
                output_start_sec=support[0],
                output_end_sec=support[1],
                mapping_source="runtime_typed_insertion_effect",
            )
        )
    return mappings if _projection_mappings_are_ordered(mappings) else None


def _inverse_insertion_boundary(
    value: float,
    *,
    anchors: list[float],
    durations: list[float],
    output_intervals: list[tuple[float, float]],
) -> float | None:
    delta = 0.0
    for anchor, duration, output in zip(anchors, durations, output_intervals, strict=True):
        if value < output[0] - 0.05:
            break
        if output[0] + 0.05 < value < output[1] - 0.05:
            return None
        if abs(value - output[0]) <= 0.05 or abs(value - output[1]) <= 0.05:
            return anchor
        if value >= output[1] - 0.05:
            delta += duration
    mapped = value - delta
    return mapped if mapped >= 0.0 and math.isfinite(mapped) else None


def _finite_number_list(value: Any, *, positive: bool) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
            return None
        result.append(number)
    return result


def _bounds_list(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, list) or not value:
        return None
    result = [_span_bounds(item) for item in value]
    return None if any(item is None for item in result) else [item for item in result if item]


def _typed_result_covers_whole_source(result: dict[str, Any]) -> bool:
    source_text = str(result.get("source_text") or "").strip()
    if not source_text:
        return False
    metadata = result.get("metadata")
    spans = metadata.get("target_spans") if isinstance(metadata, dict) else None
    if isinstance(spans, list) and spans:
        bounds: list[tuple[int, int]] = []
        for item in spans:
            if not isinstance(item, dict):
                return False
            start = item.get("start_char")
            end = item.get("end_char")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                return False
            bounds.append((start, end))
        bounds.sort()
        if bounds[0][0] == 0 and bounds[-1][1] == len(source_text) and all(
            right[0] <= left[1] for left, right in pairwise(bounds)
        ):
            return True
    changed = result.get("changed_segments")
    return (
        isinstance(changed, list)
        and len(changed) == 1
        and isinstance(changed[0], dict)
        and str(changed[0].get("segment_text") or "").strip() == source_text
    )


def _projection_mappings_are_ordered(
    mappings: list[AudioIntervalMapping],
) -> bool:
    if not mappings:
        return False
    for left, right in pairwise(mappings):
        if (
            right.source_start_sec < left.source_end_sec
            or right.output_start_sec < left.output_end_sec
        ):
            return False
    return len({item.interval_id for item in mappings}) == len(mappings)


def _operation_authored_evaluation_targets(
    step_record: dict[str, Any],
    *,
    intervals: list[AudioEvaluationInterval],
) -> list[AudioEvaluationTarget]:
    """Bind targets only when the executed operation exposes an auditable order."""

    if not intervals:
        return []
    operation = str(step_record.get("operation") or "")
    params = step_record.get("params")
    if not isinstance(params, dict):
        return []
    step_id = str(step_record.get("step_id") or "step")
    requirements: list[str] = []
    derivation: Literal[
        "executed_structured_operation_order",
        "executed_single_target_operation",
        "executed_typed_result_order",
    ]
    if operation == "edit_data_pipeline.text_edit":
        requirements = _text_edit_target_requirements(params.get("instruction_st"))
        derivation = "executed_structured_operation_order"
    elif operation == "emphasis_stress_edit":
        target_phrase = str(params.get("target_phrase") or "").strip()
        if target_phrase:
            requirements = [f"Make the spoken target phrase {target_phrase!r} audibly stand out."]
        derivation = "executed_single_target_operation"
    else:
        profile = load_operation_evaluation_effect_catalog().by_operation().get(operation)
        requirements = (
            _typed_result_target_requirements(
                step_record,
                contract=profile.local_interval_contract,
            )
            if profile is not None
            else []
        )
        derivation = "executed_typed_result_order"
    if len(requirements) != len(intervals):
        return []
    return [
        AudioEvaluationTarget(
            target_ref_id=f"{step_id}:target:{index + 1:03d}",
            interval_id=interval.interval_id,
            requirement=requirement,
            derivation=derivation,
        )
        for index, (interval, requirement) in enumerate(zip(intervals, requirements, strict=True))
    ]


def _typed_result_target_requirements(
    step_record: dict[str, Any],
    *,
    contract: str,
) -> list[str]:
    result = step_record.get("result")
    if not isinstance(result, dict):
        return []
    if contract == "aligned_edit_span_pairs":
        edit_result = result.get("edit_result")
        rows = edit_result.get("edits_applied") if isinstance(edit_result, dict) else None
        if not isinstance(rows, list):
            return []
        requirements: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return []
            edit_type = str(row.get("edit_type") or "local prosody").replace("_", " ")
            words = str(row.get("words") or "").strip()
            target = f" spoken target {words!r}" if words else f" target {index + 1}"
            requirements.append(f"Apply the requested {edit_type} change to{target}.")
        return requirements
    if contract == "pause_insertion_plan":
        plan = result.get("pause_edit_plan")
        anchors = plan.get("insert_position_sec") if isinstance(plan, dict) else None
        if not isinstance(anchors, list):
            return []
        return [
            f"Insert the requested pause at named local boundary {index + 1}."
            for index in range(len(anchors))
        ]
    return []


def _text_edit_target_requirements(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        root = ET.fromstring(f"<root>{value}</root>")
    except ET.ParseError:
        return []
    requirements: list[str] = []
    for child in root.iter():
        if child is root:
            continue
        tag = child.tag.lower()
        source_text = "".join(child.itertext()).strip()
        if tag == "sub":
            target_text = str(child.attrib.get("targ") or "").strip()
            if not source_text or not target_text:
                return []
            requirements.append(f"Replace spoken {source_text!r} with spoken {target_text!r}.")
        elif tag == "del":
            if not source_text:
                return []
            requirements.append(f"Remove the spoken content {source_text!r}.")
        elif tag == "ins":
            if not source_text:
                return []
            requirements.append(f"Insert the spoken content {source_text!r}.")
    return requirements


def _edited_intervals_from_timeline_ledger(
    ledger: dict[str, Any],
) -> list[AudioEvaluationInterval] | None:
    if (
        ledger.get("schema_version") != "x-editor.timeline-ledger/v1"
        or ledger.get("status") != "complete"
        or not isinstance(ledger.get("pieces"), list)
    ):
        return None
    intervals: list[AudioEvaluationInterval] = []
    for index, piece in enumerate(ledger["pieces"]):
        if not isinstance(piece, dict):
            return None
        if piece.get("kind") == "identity":
            continue
        source_bounds = _span_bounds(piece.get("base_interval"))
        output_bounds = _span_bounds(piece.get("current_interval"))
        if source_bounds is None or output_bounds is None:
            return None
        intervals.append(
            AudioEvaluationInterval(
                interval_id=f"runtime-ledger-{index:03d}",
                start_sec=source_bounds[0],
                end_sec=source_bounds[1],
            )
        )
    return intervals


def _same_resolved_path(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except (OSError, RuntimeError):
        return left == right


def _runtime_reviewed_evidence_rubric_spec(
    result_row: dict[str, Any],
    *,
    source_audio_aliases: dict[str, frozenset[str]],
    rubrics: list[FrozenRubricQuestion],
    active_turn_indices: frozenset[int] | None = None,
) -> FrozenPostDeliveryAudioRubricSpec | None:
    """Derive target intervals only from reviewed evidence in a completed result."""

    projection = _reviewed_runtime_target_projection(
        result_row,
        source_audio_aliases=source_audio_aliases,
        active_turn_indices=active_turn_indices,
    )
    if projection is None:
        return None
    return FrozenPostDeliveryAudioRubricSpec(
        source_asset_ref=projection.source_asset_ref,
        edit_scope="local",
        edited_intervals=list(projection.intervals),
        rubrics=rubrics,
    )


def _reviewed_runtime_target_projection(
    result_row: dict[str, Any],
    *,
    source_audio_aliases: dict[str, frozenset[str]],
    active_turn_indices: frozenset[int] | None,
) -> _ReviewedRuntimeTargetProjection | None:
    """Collect only resolved, non-reference targets on the adopted branch."""

    containers = _reviewed_runtime_evidence_containers(
        result_row,
        active_turn_indices=active_turn_indices,
    )
    raw_spans: list[dict[str, Any]] = []
    raw_bindings: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if not isinstance(value, dict):
            return
        selected = value.get("selected_spans")
        if isinstance(selected, list):
            raw_spans.extend(item for item in selected if isinstance(item, dict))
        bindings = value.get("target_bindings")
        if isinstance(bindings, list):
            raw_bindings.extend(item for item in bindings if isinstance(item, dict))
        report = value.get("perception_report")
        if isinstance(report, dict):
            selected = report.get("selected_spans")
            if isinstance(selected, list):
                raw_spans.extend(item for item in selected if isinstance(item, dict))
            bindings = report.get("target_bindings")
            if isinstance(bindings, list):
                raw_bindings.extend(item for item in bindings if isinstance(item, dict))

    for container in containers:
        collect(container)

    resolved_span_ids = {
        str(span_id)
        for binding in raw_bindings
        if binding.get("status") == "resolved"
        for span_id in (
            binding.get("resolved_span_ids")
            if isinstance(binding.get("resolved_span_ids"), list)
            else []
        )
        if str(span_id).strip()
    }
    if raw_bindings and not resolved_span_ids:
        return None

    intervals_by_source: dict[str, list[AudioEvaluationInterval]] = {}
    seen: set[tuple[str, float, float]] = set()
    for index, span in enumerate(raw_spans):
        span_id = str(span.get("span_id") or "").strip()
        if resolved_span_ids and span_id not in resolved_span_ids:
            continue
        semantic_role = str(span.get("semantic_role") or "").strip().lower()
        if semantic_role in {"reference", "reference_target", "context", "preserve"}:
            continue
        audio_ref = str(span.get("audio_ref_id") or "").strip()
        source_asset_ref = _source_asset_ref_for_alias(
            audio_ref,
            source_audio_aliases=source_audio_aliases,
        )
        if source_asset_ref is None:
            continue
        bounds = _finite_bounds(span.get("start_sec"), span.get("end_sec"))
        seen_key = (source_asset_ref, *(bounds or (0.0, 0.0)))
        if bounds is None or seen_key in seen:
            continue
        seen.add(seen_key)
        intervals_by_source.setdefault(source_asset_ref, []).append(
            AudioEvaluationInterval(
                interval_id=span_id or f"runtime-selected-{index:03d}",
                start_sec=bounds[0],
                end_sec=bounds[1],
            )
        )
    if len(intervals_by_source) != 1:
        return None
    source_asset_ref, intervals = next(iter(intervals_by_source.items()))
    intervals.sort(key=lambda item: (item.start_sec, item.end_sec, item.interval_id))
    if not intervals:
        return None
    try:
        _validate_non_overlapping(intervals, name="runtime selected_spans")
    except ValueError:
        return None
    return _ReviewedRuntimeTargetProjection(
        source_asset_ref=source_asset_ref,
        intervals=tuple(intervals),
    )


def _reviewed_runtime_evidence_containers(
    result_row: dict[str, Any],
    *,
    active_turn_indices: frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    arm_result_records = _completed_workflow_arm_result_records(result_row)
    if active_turn_indices is not None:
        arm_result_records = [
            item for item in arm_result_records if item[0] in active_turn_indices
        ]
    for _, arm_result in arm_result_records:
        history = arm_result.get("perception_history")
        if isinstance(history, list):
            for record in history:
                if not isinstance(record, dict) or record.get("status") != "completed":
                    continue
                evidence = record.get("evidence")
                if not isinstance(evidence, dict) or evidence.get("usable") is False:
                    continue
                facts = evidence.get("facts")
                if isinstance(facts, dict):
                    containers.append(facts)
        projection = arm_result.get("evidence_projection")
        if isinstance(projection, dict):
            for key in ("execution_context", "planner_context"):
                value = projection.get(key)
                if isinstance(value, dict):
                    containers.append(value)
        for key in ("reviewed_perception", "reviewed_evidence", "perception_report"):
            value = arm_result.get(key)
            if isinstance(value, dict):
                containers.append(value)
    return containers


def _public_source_audio_aliases(
    assets: list[Any],
) -> dict[str, frozenset[str]]:
    """Build aliases only from serialized public WorkflowAudioAsset fields."""

    aliases_by_source: dict[str, frozenset[str]] = {}
    owners: dict[str, str] = {}
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("role") != "source":
            continue
        if asset.get("schema_version") != "x-editor.workflow-audio-asset/v1":
            raise ValueError("source asset has an unsupported schema_version")
        asset_ref = str(asset.get("asset_ref") or "").strip()
        order = asset.get("order")
        if not asset_ref:
            raise ValueError("source asset_ref is blank")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("source asset order must be a non-negative integer")
        logical_ref = f"aud:v1:s{order:04d}"
        aliases = frozenset((asset_ref, logical_ref))
        for alias in aliases:
            owner = owners.get(alias)
            if owner is not None and owner != asset_ref:
                raise ValueError(f"source audio alias collision: {alias}")
            owners[alias] = asset_ref
        aliases_by_source[asset_ref] = aliases
    if not aliases_by_source:
        raise ValueError("no public source WorkflowAudioAsset is available")
    return aliases_by_source


def _source_asset_ref_for_alias(
    audio_ref: str,
    *,
    source_audio_aliases: dict[str, frozenset[str]],
) -> str | None:
    if not audio_ref:
        return None
    matches = [
        asset_ref for asset_ref, aliases in source_audio_aliases.items() if audio_ref in aliases
    ]
    return matches[0] if len(matches) == 1 else None


def _scorer_protocol(
    options: PostDeliveryAudioScoringOptions,
    *,
    model_name: str | None,
    effective_base_url: str | None,
    authentication_configured: bool,
    wording_evidence_provider: WordingEvidenceProvider | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCORER_PROTOCOL_SCHEMA_VERSION,
        "evaluator_role": "post_delivery_development_diagnostic",
        "mode": options.mode,
        "arms": sorted(options.arms),
        "provider": QWEN3_OMNI_PROVIDER if options.mode == "score" else None,
        "model": model_name,
        "base_url": effective_base_url if options.mode == "score" else None,
        "authentication_mode": (
            "bearer" if options.mode == "score" and authentication_configured else "none"
        ),
        "temperature": 0.7,
        "max_tokens": 512,
        "timeout_sec": options.timeout_sec,
        "transport_retries": 0,
        "rubric_dimensions": list(_DIMENSIONS),
        "atomic_rubric_catalog_version": _ATOMIC_RUBRIC_CATALOG_VERSION,
        "scoring_context_protocol_version": _SCORING_CONTEXT_PROTOCOL_VERSION,
        "judge_prompt_fingerprint": _fingerprint(
            {
                "system_prompt": _SYSTEM_PROMPT,
                "context_protocol": _SCORING_CONTEXT_PROTOCOL_VERSION,
            }
        ),
        "operation_effect_catalog_fingerprint": _fingerprint(
            load_operation_evaluation_effect_catalog().model_dump(mode="json")
        ),
        "applicability_policy": {
            "global_changed": "not_applicable",
            "global_preserved": "edited_region_preservation",
            "global_unknown": "abstain",
            "local_complement": "applicable",
        },
        "target_atomization_policy": {
            "version": "x-editor.operation-authored-target-order/v1",
            "requires_exact_target_interval_count": True,
            "missing_or_ambiguous_lineage": "group_or_abstain_without_text_guess",
        },
        "interval_projection_policy": {
            "version": "x-editor.post-delivery-interval-projection/v5",
            "authority_order": [
                "final_adopted_session_graph_branch",
                "actual_operation_graph_output_chain",
                "complete_composed_timeline_ledger",
                "complete_audio_temporal_transform",
                "reviewed_runtime_targets",
                "typed_successful_step_result",
                "typed_global_operation_effect",
                "independent_full_audio_asr_alignment_localization",
            ],
            "benchmark_identity_or_expected_answer_allowed": False,
            "unavailable_lineage_is_a_conflict": False,
            "explicit_lineage_or_ledger_conflict": "abstain",
            "ambiguous_or_incomplete_projection": "abstain",
            "independent_alignment_scope": (
                "local edited rubrics except physical boundary continuity"
            ),
            "independent_alignment_is_semantic_evidence": False,
            "independent_alignment_thresholds": {
                "minimum_alignment_coverage": DEFAULT_MINIMUM_ALIGNMENT_COVERAGE,
                "minimum_lexical_tokens": DEFAULT_MINIMUM_TOKEN_COUNT,
                "maximum_total_boundary_padding_sec": (
                    DEFAULT_MAXIMUM_BOUNDARY_PADDING_SEC
                ),
                "maximum_word_overlap_sec": DEFAULT_MAXIMUM_WORD_OVERLAP_SEC,
                "bounded_minimum_lexical_tokens": (
                    DEFAULT_BOUNDED_MINIMUM_TOKEN_COUNT
                ),
                "bounded_maximum_edit_fraction": (
                    DEFAULT_BOUNDED_MAXIMUM_EDIT_FRACTION
                ),
                "candidate_must_be_unique": True,
            },
            "execution_feedback_allowed": False,
        },
        "final_state_policy": {
            "version": "x-editor.final-state-projection/v1",
            "authority": "session_graph",
            "model_claims_are_not_scoring_truth": True,
        },
        "wording_evidence_policy": {
            "mode": options.wording_evidence_mode,
            "provider": (
                getattr(wording_evidence_provider, "provider_name", None)
                if wording_evidence_provider is not None
                else None
            ),
            "model": (
                getattr(wording_evidence_provider, "model_name", None)
                if wording_evidence_provider is not None
                else None
            ),
            "minimum_confidence": options.wording_evidence_min_confidence,
            "timeout_sec": options.wording_evidence_timeout_sec,
            "low_confidence_policy": "advisory_only_and_judge_may_abstain",
            "execution_feedback_allowed": False,
        },
        "atomic_rubrics": [
            {
                "atomic_rubric_id": item.atomic_rubric_id,
                "dimension": item.dimension,
                "observable": item.observable,
                "question": item.question,
            }
            for item in _ATOMIC_RUBRICS
        ],
        "single_sample_positive_threshold": 2,
        "vote_policy": {
            "valid_repetitions": options.valid_repetitions,
            "required_positive_votes": options.required_positive_votes,
            "max_attempts": options.max_attempts,
        },
        "crop_policy": {
            "crop_margin_sec": options.crop_margin_sec,
            "minimum_crop_sec": options.minimum_crop_sec,
            "timeline_tolerance_sec": options.timeline_tolerance_sec,
            "missing_interval_policy": options.missing_interval_policy,
        },
    }


def _score_manifest(
    *,
    evidence: _RunEvidence,
    options: PostDeliveryAudioScoringOptions,
    jobs: list[_PreparedJob],
    protocol: dict[str, Any],
    state: Literal["in_progress", "complete"],
) -> dict[str, Any]:
    request_set_fingerprint = _request_set_fingerprint(jobs)
    protocol_fingerprint = _fingerprint(protocol)
    resume_identity = {
        "run_fingerprint": evidence.run_fingerprint,
        "execution_evidence_fingerprint": evidence.protected_fingerprint,
        "request_set_fingerprint": request_set_fingerprint,
        "scorer_protocol_fingerprint": protocol_fingerprint,
    }
    return {
        "schema_version": "x-editor.post-delivery-audio-score-manifest/v2",
        "state": state,
        "experiment_kind": "post_delivery_development_diagnostic",
        "evaluator_contract": build_evaluator_role_contract(
            "post_delivery_development_diagnostic"
        ).model_dump(mode="json"),
        "scientific_boundary": {
            "development_diagnostic_only": True,
            "may_update_tool_prior": False,
            "may_feedback_to_runtime": False,
            "may_trigger_new_attempt": False,
            "eligible_as_final_outcome": False,
        },
        "mode": options.mode,
        "run_dir": str(evidence.root),
        "run_fingerprint": evidence.run_fingerprint,
        "execution_evidence_fingerprint_before": evidence.protected_fingerprint,
        "execution_evidence_fingerprint_after": None,
        "score_output_outside_run": True,
        "sidecar_spec_key": SIDECAR_SPEC_KEY,
        "provider": protocol["provider"],
        "model": protocol["model"],
        "transport_retries": 0,
        "scorer_protocol": protocol,
        "scorer_protocol_fingerprint": protocol_fingerprint,
        "request_count": len(jobs),
        "request_set_fingerprint": request_set_fingerprint,
        "scoring_audio_fingerprint_before": _job_audio_fingerprint(jobs),
        "scoring_audio_fingerprint_after": None,
        "completed_rubric_count": 0,
        "result_set_fingerprint": None,
        "resume_identity_fingerprint": _fingerprint(resume_identity),
    }


def _verify_resume_identity(
    manifest: dict[str, Any],
    *,
    evidence: _RunEvidence,
    jobs: list[_PreparedJob],
    protocol: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != ("x-editor.post-delivery-audio-score-manifest/v2"):
        raise ValueError("resume requires a post-delivery score manifest v2")
    expected_contract = build_evaluator_role_contract(
        "post_delivery_development_diagnostic"
    ).model_dump(mode="json")
    if manifest.get("experiment_kind") != "post_delivery_development_diagnostic":
        raise ValueError("resume evaluator role does not match development diagnostic")
    if manifest.get("evaluator_contract") != expected_contract:
        raise ValueError("resume evaluator authority contract changed")
    request_set_fingerprint = _request_set_fingerprint(jobs)
    protocol_fingerprint = _fingerprint(protocol)
    resume_identity = {
        "run_fingerprint": evidence.run_fingerprint,
        "execution_evidence_fingerprint": evidence.protected_fingerprint,
        "request_set_fingerprint": request_set_fingerprint,
        "scorer_protocol_fingerprint": protocol_fingerprint,
    }
    checks = {
        "run_dir": str(evidence.root),
        "run_fingerprint": evidence.run_fingerprint,
        "execution_evidence_fingerprint_before": evidence.protected_fingerprint,
        "request_set_fingerprint": request_set_fingerprint,
        "scorer_protocol": protocol,
        "scorer_protocol_fingerprint": protocol_fingerprint,
        "resume_identity_fingerprint": _fingerprint(resume_identity),
        "scoring_audio_fingerprint_before": _job_audio_fingerprint(jobs),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"resume identity mismatch: {key}")


def _request_set_fingerprint(jobs: list[_PreparedJob]) -> str:
    return _fingerprint(
        [_request_fingerprint_payload(job.request.model_dump(mode="json")) for job in jobs]
    )


def _request_fingerprint(value: dict[str, Any]) -> str:
    return _fingerprint(_request_fingerprint_payload(value))


def _request_fingerprint_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Remove materialization-location fields from a rubric request identity."""

    body = dict(value)
    body.pop("schema_version", None)
    body.pop("request_fingerprint", None)
    body["crops"] = [
        {key: item for key, item in dict(crop).items() if key != "path"}
        for crop in body.get("crops", [])
    ]
    return body


def _verify_jobs_match_run(
    jobs: list[_PreparedJob],
    *,
    evidence: _RunEvidence,
) -> None:
    deliveries = {
        item["marker"]["assignment_fingerprint"]: item["marker"] for item in evidence.deliveries
    }
    expected_keys = {
        (assignment_fingerprint, atomic_rubric.atomic_rubric_id)
        for assignment_fingerprint in deliveries
        for atomic_rubric in _ATOMIC_RUBRICS
    }
    covered_keys: set[tuple[str, str]] = set()
    request_keys: set[tuple[str, str, str | None]] = set()
    for job in jobs:
        request = job.request
        marker = deliveries.get(request.assignment_fingerprint)
        if marker is None:
            raise ValueError("rubric request does not belong to the frozen run")
        if (
            request.runtime_projection_fingerprint != marker["runtime_projection_fingerprint"]
            or request.assigned_arm != marker["assigned_arm"]
        ):
            raise ValueError("rubric request runtime identity does not match delivery")
        rubric_key = request.atomic_rubric_id or request.dimension
        covered_keys.add((request.assignment_fingerprint, rubric_key))
        request_key = (
            request.assignment_fingerprint,
            rubric_key,
            request.target_ref_id,
        )
        if request_key in request_keys:
            raise ValueError("duplicate atomic rubric request for assignment and target")
        request_keys.add(request_key)
        body = request.model_dump(mode="json")
        if _request_fingerprint(body) != request.request_fingerprint:
            raise ValueError("rubric request fingerprint is invalid")
        if request.preflight_status == "ready" and not request.crops:
            raise ValueError("ready rubric request has no materialized audio crops")
        if request.preflight_status != "ready" and request.crops:
            raise ValueError("non-ready rubric request unexpectedly contains crops")
        if request.rubric_scope_source == "public_full_clip_fallback" and any(
            item.mapping_source != "public_full_clip_fallback" for item in request.interval_mappings
        ):
            raise ValueError("full-clip fallback request has a non-fallback mapping")
    if covered_keys != expected_keys:
        raise ValueError("rubric request set does not cover the frozen atomic catalog")


def _resume_results(
    path: Path,
    *,
    jobs: list[_PreparedJob],
) -> dict[str, AudioRubricScoreResult]:
    if not path.is_file():
        return {}
    requests = {job.request.request_fingerprint: job.request for job in jobs}
    results: dict[str, AudioRubricScoreResult] = {}
    for raw in _read_jsonl(path):
        result = AudioRubricScoreResult.model_validate(raw)
        request = requests.get(result.request_fingerprint)
        if request is None:
            raise ValueError("resume results contain an unknown rubric request")
        if result.request_fingerprint in results:
            raise ValueError("resume results contain a duplicate completed rubric")
        if (
            result.assignment_fingerprint != request.assignment_fingerprint
            or result.assigned_arm != request.assigned_arm
            or result.dimension != request.dimension
            or result.atomic_rubric_id != request.atomic_rubric_id
        ):
            raise ValueError("resume result identity does not match rubric request")
        results[result.request_fingerprint] = result
    return results


def _protected_run_fingerprint(root: Path) -> str:
    rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return _fingerprint(rows)


def _job_audio_fingerprint(jobs: list[_PreparedJob]) -> str:
    expected_by_path: dict[Path, str] = {}
    for job in jobs:
        if len(job.audio_paths) != len(job.request.crops):
            raise ValueError("prepared rubric crop inventory is inconsistent")
        for crop in job.request.crops:
            pairs = (
                (Path(crop.path).expanduser().resolve(), crop.sha256),
                (
                    Path(crop.original_path).expanduser().resolve(),
                    crop.original_sha256,
                ),
            )
            for path, expected_sha in pairs:
                previous = expected_by_path.setdefault(path, expected_sha)
                if previous != expected_sha:
                    raise ValueError("prepared rubric audio has conflicting hashes")
    rows: list[dict[str, Any]] = []
    for path, expected_sha in sorted(
        expected_by_path.items(),
        key=lambda item: str(item[0]),
    ):
        if not path.is_file():
            raise RuntimeError(f"public scoring audio disappeared: {path}")
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"public scoring audio hash changed: {path}")
        rows.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "size_bytes": path.stat().st_size,
            }
        )
    return _fingerprint(rows)


def _score_summary(
    *,
    mode: str,
    run_fingerprint: str,
    deliveries: int,
    results: list[AudioRubricScoreResult],
    options: PostDeliveryAudioScoringOptions,
    flow_report: dict[str, Any],
) -> dict[str, Any]:
    statuses = Counter(item.status for item in results)

    def summarize(rows: list[AudioRubricScoreResult]) -> dict[str, Any]:
        scored = [item for item in rows if item.mean_score is not None]
        return {
            "rubrics": len(rows),
            "status_counts": dict(sorted(Counter(item.status for item in rows).items())),
            "scored": len(scored),
            "passed": sum(item.passed is True for item in scored),
            "pass_rate": (
                sum(item.passed is True for item in scored) / len(scored) if scored else None
            ),
            "mean_score": (
                sum(float(item.mean_score) for item in scored) / len(scored) if scored else None
            ),
            "valid_provider_samples": sum(item.valid_sample_count for item in rows),
            "provider_attempts": sum(item.attempt_count for item in rows),
        }

    by_dimension: dict[str, dict[str, Any]] = {}
    for dimension in _DIMENSIONS:
        rows = [item for item in results if item.dimension == dimension]
        by_dimension[dimension] = summarize(rows)
    by_atomic_rubric = {
        atomic_rubric.atomic_rubric_id: summarize(
            [item for item in results if item.atomic_rubric_id == atomic_rubric.atomic_rubric_id]
        )
        for atomic_rubric in _ATOMIC_RUBRICS
    }
    return {
        "schema_version": "x-editor.post-delivery-audio-score-summary/v1",
        "mode": mode,
        "run_fingerprint": run_fingerprint,
        "delivered_assignments": deliveries,
        "rubric_results": len(results),
        "single_sample_positive_threshold": 2,
        "valid_repetitions": options.valid_repetitions,
        "required_positive_votes": options.required_positive_votes,
        "max_attempts": options.max_attempts,
        "status_counts": dict(sorted(statuses.items())),
        "dimensions": by_dimension,
        "atomic_rubrics": by_atomic_rubric,
        "execution_reachability": flow_report["execution"],
        "measurable_coverage": flow_report["measurement"],
        "interval_projection": flow_report["interval_projection"],
        "utility": flow_report["utility"],
        "disagreement_and_abstain": flow_report["uncertainty"],
        "chain_flow": flow_report["chain_flow"],
        "scorer_results_may_route_tools": False,
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    candidates = [raw.strip()]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("provider response does not contain one JSON object")


def _audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"audio has no positive duration: {path}")
    return info.frames / info.samplerate


def _identity_key(value: Any) -> tuple[str, str, int | None]:
    if not isinstance(value, dict):
        raise TypeError("benchmark audit identity is missing")
    benchmark = str(value.get("benchmark") or "").strip()
    record_key = str(value.get("record_key") or "").strip()
    source_index = value.get("source_index")
    if (
        not benchmark
        or not record_key
        or (source_index is not None and not isinstance(source_index, int))
    ):
        raise ValueError("benchmark audit identity is invalid")
    return benchmark, record_key, source_index


def _require_external_output_dir(run_root: Path, output_root: Path) -> None:
    if output_root == run_root:
        raise ValueError("scoring output directory must be outside the benchmark run")
    try:
        output_root.relative_to(run_root)
    except ValueError:
        return
    raise ValueError("scoring output directory must be outside the benchmark run")


def _resolve_under(root: Path, relative: str) -> Path:
    if not relative:
        raise ValueError("run-relative evidence path is empty")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("run evidence path escapes the run directory") from exc
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validation_summary(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}" for error in exc.errors()
    )[:1000]


def _error_summary(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return _validation_summary(exc)
    return str(exc)[:1000]


__all__ = [
    "SCORER_PROTOCOL_SCHEMA_VERSION",
    "SCORING_CONTEXT_PROTOCOL_VERSION",
    "SIDECAR_SPEC_KEY",
    "AudioCropEvidence",
    "AudioEvaluationInterval",
    "AudioIntervalMapping",
    "AudioRubricAttempt",
    "AudioRubricRequest",
    "AudioRubricScoreResult",
    "FrozenPostDeliveryAudioRubricSpec",
    "FrozenRubricQuestion",
    "PostDeliveryAudioScoringOptions",
    "run_post_delivery_audio_scoring",
]
