"""Versioned contracts for semantic reasoning before executable planning.

The Reasoner describes what the user means and what evidence is still needed.
It does not select operations, author tool parameters, or control execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.schemas import StrictModel
from x_editor.runtime.audio_asset_facts import AudioAssetFacts

ReasonerPhase = Literal["initial_understanding", "evidence_review"]
ReasonerShadowStrategy = Literal[
    "direct_json",
    "private_thinking_json",
    "memo_then_compile",
    "memo_only",
    "memo_only_brief",
    "prose_memo",
    "evidence_memo",
    "draft_memo",
]
ReasonerSemanticRevisionPolicy = Literal[
    "freeze_initial",
    "evidence_correctable",
]
RequestedChangeLedgerStatus = Literal[
    "complete",
    "provider_omitted",
    "provider_empty",
]
ReferentDomain = Literal[
    "spoken_proposition",
    "voice_signal",
    "acoustic_scene",
    "media_timeline",
]
ReasonerResolutionStatus = Literal[
    "resolved",
    "needs_evidence",
    "ambiguous",
    "unsupported",
]
SemanticTargetCandidateKind = Literal[
    "span",
    "speaker",
    "audio_event",
    "acoustic_region",
]
SemanticTargetBindingStatus = Literal["resolved", "ambiguous", "unresolved"]
SemanticValueSource = Literal[
    "user_request",
    "source_text",
    "perception_evidence",
    "deterministic_derivation",
]
SemanticValueStatus = Literal["known", "requires_evidence", "not_applicable"]
TransformationKind = Literal[
    "replace_spoken_content",
    "insert_spoken_content",
    "remove_spoken_content",
    "adjust_semantic_value",
    "adjust_voice_delivery",
    "change_voice_identity",
    "change_acoustic_scene",
    "insert_pause",
    "remove_pause",
    "reorder_media",
    "extract_media",
    "remove_media",
    "compose_media",
    "other",
]
ChangeDirection = Literal[
    "increase",
    "decrease",
    "later",
    "earlier",
    "replace",
    "insert",
    "remove",
    "swap",
    "reorder",
    "preserve",
    "none",
]
EvidenceNeedKind = Literal[
    "source_text",
    "word_timestamps",
    "segment_timestamps",
    "speaker_identity",
    "voice_attribute",
    "acoustic_attribute",
    "event_grounding",
    "reference_audio",
    "other",
]
SemanticEntityKind = Literal[
    "spoken_content",
    "speaker",
    "sound_event",
    "acoustic_layer",
    "asset",
    "session_state",
]
SemanticEntityRole = Literal[
    "foreground",
    "background",
    "reference",
    "edit_base",
]
SemanticTemporalExtent = Literal[
    "point",
    "span",
    "all_entity_segments",
    "whole_asset",
    "not_applicable",
]
ReasonerIssueSeverity = Literal["blocking", "advisory"]
CompletionProofKind = Literal[
    "accepted_transition",
    "rollback_before_transition",
]

_MAX_SHORT_TEXT = 512
_MAX_LONG_TEXT = 2048
SEMANTIC_MEMO_OUTPUT_MAX_LENGTH = 2048
_TRANSFORMATION_DOMAIN: dict[TransformationKind, ReferentDomain] = {
    "replace_spoken_content": "spoken_proposition",
    "insert_spoken_content": "spoken_proposition",
    "remove_spoken_content": "spoken_proposition",
    "adjust_semantic_value": "spoken_proposition",
    "adjust_voice_delivery": "voice_signal",
    "change_voice_identity": "voice_signal",
    "change_acoustic_scene": "acoustic_scene",
    "insert_pause": "voice_signal",
    "remove_pause": "voice_signal",
    "reorder_media": "media_timeline",
    "extract_media": "media_timeline",
    "remove_media": "media_timeline",
    "compose_media": "media_timeline",
}
_TARGET_VALUE_REQUIRED = {
    "replace_spoken_content",
    "insert_spoken_content",
    "adjust_semantic_value",
    "adjust_voice_delivery",
    "change_voice_identity",
    "change_acoustic_scene",
    "insert_pause",
}
_RELATIVE_DIRECTIONS = {"increase", "decrease", "later", "earlier"}


def _normalize_text(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _normalize_text_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized: list[Any] = []
    seen: set[str] = set()
    for item in value:
        item = _normalize_text(item)
        if isinstance(item, str):
            if not item:
                raise ValueError("text lists cannot contain blank values")
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
        normalized.append(item)
    return normalized


class SemanticValueReference(StrictModel):
    """One source or target value plus its auditable derivation."""

    value_status: SemanticValueStatus
    value: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    description: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    source: SemanticValueSource
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)
    derivation: str | None = Field(default=None, max_length=_MAX_LONG_TEXT)

    @field_validator("value", "description", "derivation", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def normalize_evidence_refs(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class ReasonerEvidenceNeed(StrictModel):
    """Evidence required to resolve one semantic objective."""

    need_id: str = Field(min_length=1, max_length=128)
    kind: EvidenceNeedKind
    description: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    required: bool = True
    resolved_by_evidence_ids: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("need_id", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("resolved_by_evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_refs(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class ObservationNeed(StrictModel):
    """One source fact the Reasoner needs, without selecting perception routing.

    This is intentionally a small evidence-request envelope. ``fact_kind`` and
    ``temporal_precision`` remain short open-text descriptions so a provider can
    express uncertainty or a new fact type without being forced into a concrete
    perception operation, provider, model, or executable parameter vocabulary.
    """

    fact_kind: str = Field(min_length=1, max_length=128)
    audio_refs: list[str] = Field(default_factory=list, max_length=16)
    temporal_precision: str | None = Field(default=None, max_length=128)
    reason: str = Field(
        min_length=1,
        max_length=_MAX_SHORT_TEXT,
        validation_alias=AliasChoices("reason", "why_needed"),
    )

    @field_validator("fact_kind", "temporal_precision", "reason", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("audio_refs", mode="before")
    @classmethod
    def normalize_audio_refs(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class ObservationRequirement(ObservationNeed):
    """A stable, source-scoped question and its current observation status."""

    need_id: str = Field(min_length=1, max_length=128)
    status: Literal["pending", "resolved", "observed", "not_detected", "conflicting"] = "pending"
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)
    finding: str = Field(default="", max_length=2048)

    @model_validator(mode="after")
    def require_observation_support(self) -> ObservationRequirement:
        if self.status in {"resolved", "observed", "not_detected", "conflicting"} and not self.evidence_refs:
            raise ValueError("an observation finding requires supplied evidence references")
        if not self.audio_refs:
            raise ValueError("an observation requirement must identify its audio scope")
        return self


class SemanticEditDraft(StrictModel):
    """Tool-neutral proposed content; never a measured source observation."""

    audio_ref: str = Field(min_length=1, max_length=4096)
    intended_change: str = Field(min_length=1, max_length=2048)
    source_excerpt: str | None = Field(default=None, max_length=4096)
    proposed_content: str = Field(min_length=1, max_length=4096)
    annotated_transcript: str | None = Field(default=None, max_length=8192)
    edit_scope: str = Field(min_length=1, max_length=2048)
    preservation_requirements: list[str] = Field(default_factory=list, max_length=16)
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)


class ReasonerTransformation(StrictModel):
    """Typed state transition without any tool or parameter decision."""

    kind: TransformationKind
    direction: ChangeDirection
    description: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: Any) -> Any:
        return _normalize_text(value)


class ExpectedObservableChange(StrictModel):
    """What must be observably different if the objective is satisfied."""

    domain: ReferentDomain
    description: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    must_change: list[str] = Field(min_length=1, max_length=16)
    must_preserve: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("must_change", "must_preserve", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class ReasonedEditObjective(StrictModel):
    """One atomic, ordered edit objective understood by the Reasoner."""

    objective_id: str = Field(min_length=1, max_length=128)
    requested_state_change: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    referent_domain: ReferentDomain
    target_description: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    transformation: ReasonerTransformation
    source_value_ref: SemanticValueReference = Field(
        description=(
            "Observed input state. When evidence is still required, keep the "
            "object present with value_status=requires_evidence and use value "
            "for a concise symbolic description of the missing observation."
        )
    )
    resolved_target_value: SemanticValueReference = Field(
        description=(
            "Requested output state. When evidence is still required, keep the "
            "object present with value_status=requires_evidence and use value "
            "for a concise symbolic description of the unresolved target."
        )
    )
    expected_observable_change: ExpectedObservableChange
    effect_kind: str = Field(min_length=1, max_length=256)
    dependency_ids: list[str] = Field(default_factory=list, max_length=32)
    preservation_requirements: list[str] = Field(default_factory=list, max_length=32)
    required_evidence: list[ReasonerEvidenceNeed] = Field(default_factory=list, max_length=32)
    resolution_status: ReasonerResolutionStatus
    uncertainties: list[str] = Field(default_factory=list, max_length=32)
    reason_basis: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)

    @field_validator(
        "objective_id",
        "requested_state_change",
        "target_description",
        "effect_kind",
        "reason_basis",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator(
        "dependency_ids",
        "preservation_requirements",
        "uncertainties",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class ReasonedEditIntent(StrictModel):
    """Complete semantic handoff from Reasoner to a later Planner."""

    schema_version: Literal["x-editor.reasoned-edit-intent/v1"] = "x-editor.reasoned-edit-intent/v1"
    request_id: str = Field(min_length=1, max_length=256)
    phase: ReasonerPhase
    request_summary: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    objectives: list[ReasonedEditObjective] = Field(min_length=1, max_length=16)
    global_preservation_requirements: list[str] = Field(
        default_factory=list,
        max_length=32,
    )
    ambiguities: list[str] = Field(default_factory=list, max_length=32)
    overall_resolution_status: ReasonerResolutionStatus

    @field_validator("request_id", "request_summary", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator(
        "global_preservation_requirements",
        "ambiguities",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> Any:
        return _normalize_text_list(value)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json"))


class SatisfiedChangeProofBinding(StrictModel):
    """Bind one claimed satisfied change to authoritative Session Graph proof."""

    change: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    proof_refs: list[str] = Field(min_length=1, max_length=8)

    @field_validator("change", mode="before")
    @classmethod
    def normalize_change(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("proof_refs", mode="before")
    @classmethod
    def normalize_proof_refs(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class RequestedSemanticChange(StrictModel):
    """One listener-observable difference plus optional object/scope hints.

    The natural-language memo remains authoritative. These orthogonal fields
    reduce later Planner rediscovery when the Reasoner is confident, while null
    values and an empty observation-needs list remain valid uncertainty.
    """

    change_id: str = Field(min_length=1, max_length=128)
    observable_domain: ReferentDomain
    requested_difference: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    preservation_requirements: list[str] = Field(default_factory=list, max_length=32)
    entity_kind: SemanticEntityKind | None = None
    entity_role: SemanticEntityRole | None = None
    temporal_extent: SemanticTemporalExtent | None = None
    observation_needs: list[ObservationNeed] = Field(default_factory=list, max_length=16)

    @field_validator("change_id", "requested_difference", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("preservation_requirements", mode="before")
    @classmethod
    def normalize_preservation(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class SemanticTargetRelation(StrictModel):
    """A semantic relation used to select a measured candidate without timing guesses."""

    relation_kind: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)
    anchor_description: str = Field(min_length=1, max_length=_MAX_SHORT_TEXT)

    @field_validator("relation_kind", "anchor_description", mode="before")
    @classmethod
    def normalize_relation_text(cls, value: Any) -> Any:
        return _normalize_text(value)


class SemanticTargetSelector(StrictModel):
    """Initial Reasoner semantics for choosing later Perception candidates."""

    target_id: str = Field(min_length=1, max_length=128)
    change_id: str = Field(min_length=1, max_length=128)
    candidate_kind: SemanticTargetCandidateKind
    description: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    selection_rule: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)
    quantifier: str | None = Field(default=None, max_length=_MAX_SHORT_TEXT)
    ordinal: int | None = Field(default=None, ge=1)
    target_count: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("target_count", "count"),
    )
    candidate_ref_count: int | None = Field(default=None, ge=1)
    relation: SemanticTargetRelation | None = None
    required_for_execution: Literal[True] = True

    @field_validator(
        "target_id",
        "change_id",
        "description",
        "selection_rule",
        "quantifier",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)


class SemanticCandidateRef(StrictModel):
    """Audio-scoped reference to one admitted Perception candidate."""

    audio_ref_id: str = Field(min_length=1, max_length=512)
    candidate_id: str = Field(min_length=1, max_length=512)

    @field_validator("audio_ref_id", "candidate_id", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)


class SemanticTargetBinding(StrictModel):
    """Evidence-review selection of audio-scoped candidates, never raw measurements."""

    target_id: str = Field(min_length=1, max_length=128)
    candidate_kind: SemanticTargetCandidateKind
    status: SemanticTargetBindingStatus
    candidate_refs: list[SemanticCandidateRef] = Field(default_factory=list, max_length=128)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)

    @field_validator("target_id", "reason", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @model_validator(mode="after")
    def validate_resolution(self) -> SemanticTargetBinding:
        if self.status == "resolved" and not self.candidate_refs:
            raise ValueError("a resolved semantic target requires candidate_refs")
        refs = [(item.audio_ref_id, item.candidate_id) for item in self.candidate_refs]
        if len(refs) != len(set(refs)):
            raise ValueError("semantic target candidate refs must be unique")
        return self


def _validate_semantic_target_shape(
    *,
    phase: ReasonerPhase,
    change_ids: list[str],
    target_selectors: list[SemanticTargetSelector],
    target_bindings: list[SemanticTargetBinding],
    allow_empty_evidence_bindings: bool = False,
) -> None:
    selector_ids = [item.target_id for item in target_selectors]
    if len(selector_ids) != len(set(selector_ids)):
        raise ValueError("semantic target selector ids must be unique")
    unknown_change_ids = sorted({item.change_id for item in target_selectors} - set(change_ids))
    if unknown_change_ids:
        raise ValueError(
            f"semantic target selectors reference unknown requested changes: {unknown_change_ids}"
        )

    binding_ids = [item.target_id for item in target_bindings]
    if len(binding_ids) != len(set(binding_ids)):
        raise ValueError("semantic target binding ids must be unique")
    if phase == "initial_understanding" and target_bindings:
        raise ValueError("initial semantic understanding cannot bind measured candidates")
    if (
        phase == "evidence_review"
        and selector_ids
        and binding_ids
        and not set(binding_ids).issubset(set(selector_ids))
    ):
        raise ValueError("evidence review target bindings must refer to known selectors")

    selectors_by_id = {item.target_id: item for item in target_selectors}
    unknown_target_ids = sorted(set(binding_ids) - set(selectors_by_id))
    if unknown_target_ids:
        raise ValueError(
            f"semantic target bindings reference unknown selectors: {unknown_target_ids}"
        )
    for binding in target_bindings:
        selector = selectors_by_id[binding.target_id]
        if binding.candidate_kind != selector.candidate_kind:
            raise ValueError(f"semantic target {binding.target_id!r} changed candidate kind")
        if (
            binding.status == "resolved"
            and selector.candidate_ref_count is not None
            and len(binding.candidate_refs) != selector.candidate_ref_count
        ):
            raise ValueError(
                f"semantic target {binding.target_id!r} requires exactly "
                f"{selector.candidate_ref_count} candidate refs"
            )


class SemanticReasoningMemo(StrictModel):
    """Reviewable semantic analysis, distinct from private model reasoning.

    The provider-authored ``content`` is the semantic authority. Optional typed
    object, scope, and observation hints live on each requested change and may
    remain absent or uncertain. The framework binds the handoff to the call,
    prior memo, and Graph proof inventory without persisting raw chain-of-thought.
    """

    schema_version: Literal["x-editor.semantic-reasoning-memo/v2"] = (
        "x-editor.semantic-reasoning-memo/v2"
    )
    request_id: str = Field(min_length=1, max_length=256)
    phase: ReasonerPhase
    requested_changes: list[RequestedSemanticChange] = Field(
        default_factory=list,
        max_length=16,
    )
    requested_change_ledger_status: RequestedChangeLedgerStatus = "complete"
    target_selectors: list[SemanticTargetSelector] = Field(default_factory=list, max_length=32)
    target_bindings: list[SemanticTargetBinding] = Field(default_factory=list, max_length=32)
    content: str = Field(min_length=1, max_length=16384)
    observation_ledger: list[ObservationRequirement] = Field(default_factory=list, max_length=16)
    semantic_drafts: list[SemanticEditDraft] = Field(default_factory=list, max_length=16)
    selected_base_state_node_id: str | None = Field(default=None, max_length=256)
    selected_base_audio_ref: str | None = Field(default=None, max_length=256)
    changes_satisfied_by_base: list[str] = Field(default_factory=list, max_length=32)
    satisfied_change_proof_bindings: list[SatisfiedChangeProofBinding] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "One binding per claimed satisfied change. Proof refs must come from the "
            "current call's Session Graph completion-proof inventory."
        ),
    )
    remaining_execution_request: str | None = Field(
        default=None,
        max_length=32768,
        description=(
            "Framework-owned compatibility rendering of requested_changes that are not "
            "covered by changes_satisfied_by_base. Providers do not author this field. "
            "It is null only when exact Session Graph proofs authorize every requested "
            "change at the selected base."
        ),
    )
    prior_memo_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator(
        "request_id",
        "content",
        "selected_base_state_node_id",
        "selected_base_audio_ref",
        "remaining_execution_request",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("changes_satisfied_by_base", mode="before")
    @classmethod
    def normalize_satisfied_changes(cls, value: Any) -> Any:
        return _normalize_text_list(value)

    @model_validator(mode="before")
    @classmethod
    def derive_requested_change_ledger_status(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "requested_change_ledger_status" in value:
            return value
        normalized = dict(value)
        normalized["requested_change_ledger_status"] = (
            "complete" if normalized.get("requested_changes") else "provider_omitted"
        )
        return normalized

    @model_validator(mode="after")
    def validate_base_pair(self) -> SemanticReasoningMemo:
        change_ids = [item.change_id for item in self.requested_changes]
        if len(change_ids) != len(set(change_ids)):
            raise ValueError("semantic memo requested change ids must be unique")
        if self.requested_change_ledger_status == "complete" and not change_ids:
            raise ValueError("a complete requested-change ledger cannot be empty")
        if self.requested_change_ledger_status != "complete" and change_ids:
            raise ValueError("an incomplete requested-change ledger must be empty")
        _validate_semantic_target_shape(
            phase=self.phase,
            change_ids=change_ids,
            target_selectors=self.target_selectors,
            target_bindings=self.target_bindings,
        )
        if self.requested_change_ledger_status != "complete" and (
            self.changes_satisfied_by_base or self.satisfied_change_proof_bindings
        ):
            raise ValueError("a missing requested-change ledger cannot authorize Graph completion")
        unknown_satisfied_change_ids = sorted(set(self.changes_satisfied_by_base) - set(change_ids))
        if unknown_satisfied_change_ids:
            raise ValueError(
                "changes_satisfied_by_base contains ids outside requested_changes: "
                f"{unknown_satisfied_change_ids}"
            )
        if (self.selected_base_state_node_id is None) != (self.selected_base_audio_ref is None):
            raise ValueError("semantic memo base node and audio ref must be selected together")
        if self.changes_satisfied_by_base and self.selected_base_state_node_id is None:
            raise ValueError("changes satisfied by a base require a selected Session Graph state")
        _validate_satisfied_change_binding_shape(
            self.changes_satisfied_by_base,
            self.satisfied_change_proof_bindings,
        )
        if (
            "remaining_execution_request" in self.model_fields_set
            and self.remaining_execution_request is None
            and (self.selected_base_state_node_id is None or not self.changes_satisfied_by_base)
        ):
            raise ValueError(
                "an empty remaining execution request requires a selected base and "
                "at least one change satisfied by that base"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json"))


class SemanticReasoningMemoOutput(StrictModel):
    """Small provider-authored handoff; framework call identity stays external."""

    requested_changes: list[RequestedSemanticChange] = Field(
        min_length=1,
        max_length=16,
    )
    target_selectors: list[SemanticTargetSelector] = Field(default_factory=list, max_length=32)
    target_bindings: list[SemanticTargetBinding] = Field(default_factory=list, max_length=32)
    content: str = Field(
        min_length=1,
        max_length=SEMANTIC_MEMO_OUTPUT_MAX_LENGTH,
    )
    selected_base_state_node_id: str | None = Field(default=None, max_length=256)
    selected_base_audio_ref: str | None = Field(default=None, max_length=256)
    changes_satisfied_by_base: list[str] = Field(default_factory=list, max_length=32)
    satisfied_change_proof_bindings: list[SatisfiedChangeProofBinding] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "One exact Session Graph completion-proof binding for every claimed "
            "satisfied change; empty when no change is claimed satisfied."
        ),
    )
    remaining_execution_request: str | None = Field(
        default=None,
        max_length=32768,
        description=(
            "Legacy compatibility input only. The provider response schema omits this "
            "field; Runtime derives the durable value from requested_changes and exact "
            "Session Graph completion claims."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def read_legacy_targetless_output(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        compatible = dict(value)
        compatible.setdefault("target_selectors", [])
        compatible.setdefault("target_bindings", [])
        return compatible

    @field_validator(
        "content",
        "selected_base_state_node_id",
        "selected_base_audio_ref",
        "remaining_execution_request",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("changes_satisfied_by_base", mode="before")
    @classmethod
    def normalize_satisfied_changes(cls, value: Any) -> Any:
        return _normalize_text_list(value)

    @model_validator(mode="after")
    def validate_base_pair(self) -> SemanticReasoningMemoOutput:
        change_ids = [item.change_id for item in self.requested_changes]
        if len(change_ids) != len(set(change_ids)):
            raise ValueError("semantic memo requested change ids must be unique")
        _validate_semantic_target_shape(
            phase=("evidence_review" if self.target_bindings else "initial_understanding"),
            change_ids=change_ids,
            target_selectors=self.target_selectors,
            target_bindings=self.target_bindings,
            allow_empty_evidence_bindings=True,
        )
        unknown_satisfied_change_ids = sorted(set(self.changes_satisfied_by_base) - set(change_ids))
        if unknown_satisfied_change_ids:
            raise ValueError(
                "changes_satisfied_by_base contains ids outside requested_changes: "
                f"{unknown_satisfied_change_ids}"
            )
        if (self.selected_base_state_node_id is None) != (self.selected_base_audio_ref is None):
            raise ValueError("semantic memo base node and audio ref must be selected together")
        if self.changes_satisfied_by_base and self.selected_base_state_node_id is None:
            raise ValueError("changes satisfied by a base require a selected Session Graph state")
        _validate_satisfied_change_binding_shape(
            self.changes_satisfied_by_base,
            self.satisfied_change_proof_bindings,
        )
        if self.changes_satisfied_by_base and not self.satisfied_change_proof_bindings:
            raise ValueError(
                "provider-authored satisfied changes require completion proof bindings"
            )
        if (
            "remaining_execution_request" in self.model_fields_set
            and self.remaining_execution_request is None
            and (
            self.selected_base_state_node_id is None or not self.changes_satisfied_by_base
            )
        ):
            raise ValueError(
                "an empty remaining execution request requires a selected base and "
                "at least one change satisfied by that base"
            )
        return self


class PriorTurnSemanticContext(StrictModel):
    """Benchmark-neutral history needed to resolve multi-turn references."""

    schema_version: Literal["x-editor.prior-turn-semantic-context/v1"] = (
        "x-editor.prior-turn-semantic-context/v1"
    )
    turn_index: int = Field(ge=0)
    user_request: str = Field(min_length=1, max_length=8192)
    source_text: str | None = Field(default=None, max_length=16384)
    authoritative_status: Literal["completed"] = "completed"
    result_is_ancestor_of_current_source: bool

    @field_validator("user_request", "source_text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)


class SessionStateAudioSlotContext(StrictModel):
    """One audio asset that can be selected inside a Session Graph state."""

    audio_ref: str = Field(min_length=1, max_length=256)
    role: str = Field(min_length=1, max_length=128)
    audio_id: str | None = Field(default=None, max_length=256)
    primary: bool = False

    @field_validator("audio_ref", "role", "audio_id", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)


class SessionStateNodeContext(StrictModel):
    """Prompt-safe state node projected from the authoritative Session Graph."""

    node_id: str = Field(min_length=1, max_length=256)
    audio_ref: str | None = Field(default=None, max_length=256)
    artifact_ref_id: str | None = Field(default=None, max_length=512)
    round_index: int = Field(ge=0)
    kind: str = Field(min_length=1, max_length=128)
    parent_node_ids: list[str] = Field(default_factory=list, max_length=32)
    operation_summary: str | None = Field(default=None, max_length=1024)
    session_eligible: bool
    is_current: bool = False
    audio_slots: list[SessionStateAudioSlotContext] = Field(
        default_factory=list,
        max_length=32,
    )

    @field_validator(
        "node_id",
        "audio_ref",
        "artifact_ref_id",
        "kind",
        "operation_summary",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("parent_node_ids", mode="before")
    @classmethod
    def normalize_parent_ids(cls, value: Any) -> Any:
        return _normalize_text_list(value)

    @model_validator(mode="after")
    def validate_audio_slots(self) -> SessionStateNodeContext:
        if not self.audio_slots and self.audio_ref is not None:
            self.audio_slots = [
                SessionStateAudioSlotContext(
                    audio_ref=self.audio_ref,
                    role="primary",
                    primary=True,
                )
            ]
        refs = [item.audio_ref for item in self.audio_slots]
        if len(refs) != len(set(refs)):
            raise ValueError("session state audio slot refs must be unique per node")
        primary = [item for item in self.audio_slots if item.primary]
        if self.audio_ref is None:
            if primary:
                raise ValueError("a node without primary audio_ref cannot mark a primary slot")
        elif len(primary) != 1 or primary[0].audio_ref != self.audio_ref:
            raise ValueError("session state primary audio slot must match node.audio_ref")
        return self

    def contains_audio_ref(self, audio_ref: str | None) -> bool:
        return audio_ref is not None and any(
            item.audio_ref == audio_ref for item in self.audio_slots
        )


class SessionStateTransitionContext(StrictModel):
    """Prompt-safe executed transition projected from the Session Graph."""

    edge_id: str = Field(min_length=1, max_length=256)
    source_node_ids: list[str] = Field(min_length=1, max_length=32)
    target_node_id: str = Field(min_length=1, max_length=256)
    round_index: int = Field(ge=0)
    instruction: str = Field(min_length=1, max_length=8192)
    operations: list[str] = Field(default_factory=list, max_length=64)
    operation_type: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=128)

    @field_validator(
        "edge_id",
        "target_node_id",
        "instruction",
        "operation_type",
        "status",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator("source_node_ids", "operations", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> Any:
        return _normalize_text_list(value)


class SessionStateContext(StrictModel):
    """Current state and history supplied by Session Graph, never inferred from prose."""

    schema_version: Literal["x-editor.session-state-context/v1"] = (
        "x-editor.session-state-context/v1"
    )
    current_node_id: str = Field(min_length=1, max_length=256)
    current_audio_ref: str = Field(min_length=1, max_length=256)
    nodes: list[SessionStateNodeContext] = Field(min_length=1, max_length=128)
    transitions: list[SessionStateTransitionContext] = Field(
        default_factory=list,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_current_node(self) -> SessionStateContext:
        node_ids = [item.node_id for item in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("session state node ids must be unique")
        current = [item for item in self.nodes if item.is_current]
        if len(current) != 1 or current[0].node_id != self.current_node_id:
            raise ValueError("session state requires exactly one matching current node")
        if current[0].audio_ref != self.current_audio_ref:
            raise ValueError("current session node must expose current_audio_ref")
        known_nodes = set(node_ids)
        for item in self.transitions:
            if item.target_node_id not in known_nodes or not set(item.source_node_ids).issubset(
                known_nodes
            ):
                raise ValueError("session transition references an unknown node")
        return self


class SessionCompletionProof(StrictModel):
    """One completion claim derived only from authoritative Session Graph structure."""

    proof_ref: str = Field(min_length=1, max_length=256)
    proof_kind: CompletionProofKind
    selected_node_id: str = Field(min_length=1, max_length=256)
    selected_audio_ref: str = Field(min_length=1, max_length=256)
    transition_edge_id: str = Field(min_length=1, max_length=256)
    transition_source_node_ids: list[str] = Field(min_length=1, max_length=32)
    transition_target_node_id: str = Field(min_length=1, max_length=256)
    transition_instruction: str = Field(min_length=1, max_length=8192)
    transition_operations: list[str] = Field(min_length=1, max_length=64)


class ReasonerCallRequest(StrictModel):
    """Prompt-visible, benchmark-neutral input for one Reasoner call."""

    schema_version: Literal["x-editor.reasoner-call-request/v1"] = (
        "x-editor.reasoner-call-request/v1"
    )
    request_id: str = Field(min_length=1, max_length=256)
    phase: ReasonerPhase
    user_request: str = Field(min_length=1, max_length=8192)
    source_text: str | None = Field(default=None, max_length=16384)
    source_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    reference_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    audio_asset_facts: list[AudioAssetFacts] = Field(default_factory=list)
    prior_turn_contexts: list[PriorTurnSemanticContext] = Field(
        default_factory=list,
        max_length=32,
    )
    session_state: SessionStateContext | None = None
    reviewed_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    reviewed_evidence: dict[str, Any] = Field(default_factory=dict)
    evidence_field_sources: dict[str, list[str]] = Field(default_factory=dict)
    evidence_path_sources: dict[str, list[str]] = Field(default_factory=dict)
    evidence_path_audio_refs: dict[str, str] = Field(default_factory=dict)
    future_attempt_guidance: FutureAttemptGuidance | None = None
    prior_intent: ReasonedEditIntent | None = None
    prior_memo: SemanticReasoningMemo | None = None
    semantic_revision_policy: ReasonerSemanticRevisionPolicy = "freeze_initial"
    semantic_effect_ontology: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=256,
    )

    @field_validator("request_id", "user_request", "source_text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        return _normalize_text(value)

    @field_validator(
        "source_audio_refs",
        "reference_audio_refs",
        "reviewed_evidence_ids",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> Any:
        return _normalize_text_list(value)

    @model_validator(mode="after")
    def validate_phase_inputs(self) -> ReasonerCallRequest:
        if self.phase == "initial_understanding":
            if self.prior_intent is not None or self.prior_memo is not None:
                raise ValueError("initial understanding cannot consume a prior intent or memo")
        elif self.prior_intent is None and self.prior_memo is None:
            raise ValueError("evidence review requires the prior Reasoner intent or memo")
        elif self.semantic_revision_policy == "evidence_correctable" and self.prior_memo is None:
            raise ValueError("evidence-correctable review requires the prior semantic memo")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json"))


def session_completion_proof_inventory(
    session_state: SessionStateContext | None,
) -> list[SessionCompletionProof]:
    """Derive completion authority from completed, executable Graph transitions.

    Ordinary Perception evidence is intentionally absent from this derivation. A
    completed edit can authorize its exact target state, while selecting one of
    its source states can authorize rolling back to before that transition.
    State-selection bookkeeping never creates recursive completion authority.
    """

    if session_state is None:
        return []
    nodes_by_id = {item.node_id: item for item in session_state.nodes}
    proofs: list[SessionCompletionProof] = []
    for transition in sorted(
        session_state.transitions,
        key=lambda item: (item.round_index, item.edge_id),
    ):
        if not _transition_has_completion_authority(transition):
            continue
        target = nodes_by_id[transition.target_node_id]
        if target.session_eligible and target.audio_ref is not None:
            proofs.append(
                _completion_proof(
                    proof_kind="accepted_transition",
                    selected_node=target,
                    transition=transition,
                )
            )
        for source_node_id in transition.source_node_ids:
            source = nodes_by_id[source_node_id]
            if not source.session_eligible or source.audio_ref is None:
                continue
            proofs.append(
                _completion_proof(
                    proof_kind="rollback_before_transition",
                    selected_node=source,
                    transition=transition,
                )
            )
    proof_refs = [item.proof_ref for item in proofs]
    if len(proof_refs) != len(set(proof_refs)):
        raise ValueError("Session Graph produced duplicate completion proof refs")
    return proofs


def validate_semantic_memo_completion_authority(
    memo: SemanticReasoningMemo,
    *,
    request: ReasonerCallRequest,
) -> None:
    """Reject state-satisfaction claims without exact Graph-scoped proof."""

    if request.phase == "evidence_review":
        if request.prior_memo is None:
            raise ValueError("evidence review requires the prior semantic memo")
        if memo.requested_changes != request.prior_memo.requested_changes:
            raise ValueError("evidence review must preserve the initial requested-change ledger")
        if memo.requested_change_ledger_status != request.prior_memo.requested_change_ledger_status:
            raise ValueError(
                "evidence review must preserve the initial requested-change ledger status"
            )

    selected_node = None
    if memo.selected_base_state_node_id is not None:
        if request.session_state is None:
            raise ValueError("selected base requires Session Graph state in the current call")
        selected_node = next(
            (
                item
                for item in request.session_state.nodes
                if item.node_id == memo.selected_base_state_node_id
            ),
            None,
        )
        if selected_node is None or not selected_node.session_eligible:
            raise ValueError("selected base is absent from the eligible Session Graph states")
        if not selected_node.contains_audio_ref(memo.selected_base_audio_ref):
            raise ValueError("selected base audio ref is not a selectable slot on the Session Graph node")

    bindings = memo.satisfied_change_proof_bindings
    if not memo.changes_satisfied_by_base:
        if bindings:
            raise ValueError("completion proof bindings require claimed satisfied changes")
        return
    if not bindings:
        raise ValueError(
            "each change satisfied by a base requires a Session Graph completion proof"
        )
    if selected_node is None:
        raise ValueError("completion proof bindings require an eligible selected base")

    inventory = {
        item.proof_ref: item for item in session_completion_proof_inventory(request.session_state)
    }
    for binding in bindings:
        for proof_ref in binding.proof_refs:
            proof = inventory.get(proof_ref)
            if proof is None:
                raise ValueError(f"unknown Session Graph completion proof ref: {proof_ref}")
            if proof.selected_node_id != selected_node.node_id:
                raise ValueError(
                    "Session Graph completion proof is scoped to a different selected node"
                )
            if proof.selected_audio_ref != memo.selected_base_audio_ref:
                raise ValueError(
                    "Session Graph completion proof is scoped to a different selected audio"
                )


def validate_semantic_memo_target_binding_authority(
    memo: SemanticReasoningMemo,
    *,
    request: ReasonerCallRequest,
) -> None:
    """Allow semantic choice only over stable candidates present in reviewed evidence."""

    if request.phase == "initial_understanding":
        if memo.target_bindings:
            raise ValueError("initial Reasoner output cannot bind Perception candidates")
        return
    if request.prior_memo is None:
        raise ValueError("evidence review requires the prior semantic memo")
    if memo.target_selectors != request.prior_memo.target_selectors:
        raise ValueError("evidence review must preserve the initial target selectors")

    admitted_refs = _reviewed_candidate_refs_by_kind(request.reviewed_evidence)
    for binding in memo.target_bindings:
        unknown = sorted(
            {
                (item.audio_ref_id, binding.candidate_kind, item.candidate_id)
                for item in binding.candidate_refs
            }
            - admitted_refs[binding.candidate_kind]
        )
        if unknown:
            raise ValueError(
                f"semantic target {binding.target_id!r} references unadmitted "
                f"{binding.candidate_kind} candidates: {unknown}"
            )


def required_semantic_target_binding_failures(
    memo: SemanticReasoningMemo,
) -> list[str]:
    """Return required selector ids that are not authoritatively resolved."""

    bindings = {item.target_id: item for item in memo.target_bindings}
    return [
        selector.target_id
        for selector in memo.target_selectors
        if selector.required_for_execution
        and (
            selector.target_id not in bindings or bindings[selector.target_id].status != "resolved"
        )
    ]


def _reviewed_candidate_refs_by_kind(
    value: Mapping[str, Any],
) -> dict[SemanticTargetCandidateKind, set[tuple[str, str, str]]]:
    candidates: dict[SemanticTargetCandidateKind, set[tuple[str, str, str]]] = {
        "span": set(),
        "speaker": set(),
        "audio_event": set(),
        "acoustic_region": set(),
    }
    candidate_collections: dict[str, tuple[SemanticTargetCandidateKind, str]] = {
        "candidate_segments": ("span", "candidate_id"),
        "segments": ("span", "segment_id"),
        "word_timestamps": ("span", "word_id"),
        "speaker_segments": ("span", "segment_id"),
        "event_segments": ("audio_event", "segment_id"),
        "prosody_segments": ("span", "segment_id"),
        "acoustic_segments": ("acoustic_region", "segment_id"),
        "audio_caption_segments": ("audio_event", "segment_id"),
        "selected_spans": ("span", "span_id"),
        "events": ("audio_event", "event_id"),
    }
    speaker_collections = {"speakers", "speaker_profiles", "speaker_segments"}

    def visit(
        item: Any,
        *,
        parent_key: str | None = None,
        inherited_audio_ref_id: str | None = None,
    ) -> None:
        if isinstance(item, Mapping):
            own_audio_ref_id = _normalized_optional_text(item.get("audio_ref_id"))
            if (
                own_audio_ref_id is not None
                and inherited_audio_ref_id is not None
                and own_audio_ref_id != inherited_audio_ref_id
            ):
                raise ValueError(
                    "reviewed candidate audio_ref_id conflicts with its containing namespace"
                )
            audio_ref_id = own_audio_ref_id or inherited_audio_ref_id
            if parent_key in candidate_collections:
                configured = candidate_collections[parent_key]
                if parent_key == "candidate_segments":
                    declared_kind = item.get("candidate_kind")
                    candidate_kind = (
                        declared_kind
                        if declared_kind in candidates
                        else "span"
                    )
                    candidate_id_field = "candidate_id"
                else:
                    candidate_kind, candidate_id_field = configured
                candidate_id = item.get(candidate_id_field)
                if isinstance(candidate_id, str) and candidate_id and audio_ref_id:
                    candidates[candidate_kind].add(
                        (audio_ref_id, candidate_kind, candidate_id)
                    )
            if parent_key in speaker_collections or parent_key == "candidate_segments":
                speaker_id = item.get("speaker_id")
                if isinstance(speaker_id, str) and speaker_id and audio_ref_id:
                    candidates["speaker"].add((audio_ref_id, "speaker", speaker_id))
            for key, child in item.items():
                visit(
                    child,
                    parent_key=str(key),
                    inherited_audio_ref_id=audio_ref_id,
                )
        elif isinstance(item, list):
            for child in item:
                visit(
                    child,
                    parent_key=parent_key,
                    inherited_audio_ref_id=inherited_audio_ref_id,
                )

    visit(value)
    return candidates


def reviewed_candidate_ref_inventory(
    value: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return the exact audio-scoped IDs the evidence-review Reasoner may select."""

    candidates = _reviewed_candidate_refs_by_kind(value)
    return [
        {
            "audio_ref_id": audio_ref_id,
            "candidate_kind": candidate_kind,
            "candidate_id": candidate_id,
        }
        for candidate_kind in (
            "span",
            "speaker",
            "audio_event",
            "acoustic_region",
        )
        for audio_ref_id, _, candidate_id in sorted(candidates[candidate_kind])
    ]


def _normalized_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


class ReasonerAdmissionIssue(StrictModel):
    """One principle-level semantic contract issue."""

    code: str = Field(min_length=1, max_length=256)
    severity: ReasonerIssueSeverity
    objective_id: str | None = None
    message: str = Field(min_length=1, max_length=_MAX_LONG_TEXT)


class ReasonerAdmissionReport(StrictModel):
    """Non-authoritative readiness report over one typed intent."""

    schema_version: Literal["x-editor.reasoner-admission-report/v1"] = (
        "x-editor.reasoner-admission-report/v1"
    )
    contract_valid: Literal[True] = True
    planning_ready: bool
    issues: list[ReasonerAdmissionIssue] = Field(default_factory=list)
    admitted_objective_ids: list[str] = Field(default_factory=list)
    blocked_objective_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_summary(self) -> ReasonerAdmissionReport:
        blocking = any(item.severity == "blocking" for item in self.issues)
        if self.planning_ready == blocking:
            raise ValueError("planning_ready must be false exactly when blockers exist")
        overlap = set(self.admitted_objective_ids) & set(self.blocked_objective_ids)
        if overlap:
            raise ValueError("admitted and blocked objective ids must be disjoint")
        return self


def admit_reasoned_edit_intent(
    intent: ReasonedEditIntent,
    *,
    request: ReasonerCallRequest,
    supported_effect_kinds: Collection[str] | None = None,
) -> ReasonerAdmissionReport:
    """Check internal semantics and evidence closure without judging quality."""

    issues: list[ReasonerAdmissionIssue] = []
    available_evidence = set(request.reviewed_evidence_ids)
    supported_effects = set(supported_effect_kinds or [])

    if intent.request_id != request.request_id:
        issues.append(
            _issue(
                "reasoner_request_id_mismatch",
                "blocking",
                None,
                "Reasoner output is not bound to the current gateway call.",
            )
        )
    if intent.phase != request.phase:
        issues.append(
            _issue(
                "reasoner_phase_mismatch",
                "blocking",
                None,
                "Reasoner output phase differs from the current call phase.",
            )
        )

    prior_by_id = (
        {item.objective_id: item for item in request.prior_intent.objectives}
        if request.prior_intent is not None
        else {}
    )
    if request.prior_intent is not None:
        prior_ids = [item.objective_id for item in request.prior_intent.objectives]
        current_ids = [item.objective_id for item in intent.objectives]
        if prior_ids != current_ids:
            correctable = request.semantic_revision_policy == "evidence_correctable"
            issues.append(
                _issue(
                    (
                        "reasoner_evidence_objective_ledger_revision"
                        if correctable
                        else "reasoner_objective_ledger_drift"
                    ),
                    "advisory" if correctable else "blocking",
                    None,
                    (
                        "Evidence changed the semantic objective ledger; the "
                        "revision is retained as an explicit audit fact."
                        if correctable
                        else "Evidence review must preserve objective count, ids, and order."
                    ),
                )
            )
        if (
            request.prior_intent.global_preservation_requirements
            != intent.global_preservation_requirements
        ):
            correctable = request.semantic_revision_policy == "evidence_correctable"
            issues.append(
                _issue(
                    (
                        "reasoner_evidence_global_preservation_revision"
                        if correctable
                        else "reasoner_global_preservation_drift"
                    ),
                    "advisory" if correctable else "blocking",
                    None,
                    (
                        "Evidence changed the interpreted preservation contract; "
                        "the revision is retained as an explicit audit fact."
                        if correctable
                        else "Evidence review changed frozen global preservation requirements."
                    ),
                )
            )

    objective_ids = [item.objective_id for item in intent.objectives]
    if len(objective_ids) != len(set(objective_ids)):
        issues.append(
            _issue(
                "reasoner_objective_id_duplicate",
                "blocking",
                None,
                "Objective ids must be unique within one intent.",
            )
        )
    expected_overall = _overall_resolution_status(intent.objectives)
    if intent.overall_resolution_status != expected_overall:
        issues.append(
            _issue(
                "reasoner_overall_resolution_status_mismatch",
                "blocking",
                None,
                "Overall resolution status does not summarize objective statuses.",
            )
        )

    blocked_ids: set[str] = set()
    prior_objective_ids: set[str] = set()
    for objective in intent.objectives:
        unknown_dependencies = set(objective.dependency_ids) - set(objective_ids)
        non_prior_dependencies = (
            set(objective.dependency_ids) - prior_objective_ids - {objective.objective_id}
        )
        if objective.objective_id in objective.dependency_ids:
            issues.append(
                _issue(
                    "reasoner_objective_self_dependency",
                    "blocking",
                    objective.objective_id,
                    "An objective cannot depend on itself.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if unknown_dependencies:
            issues.append(
                _issue(
                    "reasoner_objective_dependency_unknown",
                    "blocking",
                    objective.objective_id,
                    "Objective dependencies must reference known objective ids.",
                )
            )
            blocked_ids.add(objective.objective_id)
        elif non_prior_dependencies:
            issues.append(
                _issue(
                    "reasoner_objective_dependency_not_prior",
                    "blocking",
                    objective.objective_id,
                    "Dependencies must reference earlier objectives in the ledger.",
                )
            )
            blocked_ids.add(objective.objective_id)
        prior_objective_ids.add(objective.objective_id)
        expected_domain = _TRANSFORMATION_DOMAIN.get(objective.transformation.kind)
        if expected_domain is not None and expected_domain != objective.referent_domain:
            issues.append(
                _issue(
                    "reasoner_transformation_domain_mismatch",
                    "blocking",
                    objective.objective_id,
                    "Transformation kind is inconsistent with the referent domain.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if objective.expected_observable_change.domain != objective.referent_domain:
            issues.append(
                _issue(
                    "reasoner_observable_domain_mismatch",
                    "blocking",
                    objective.objective_id,
                    "Expected observable change must use the referent domain.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if all(
            value.strip().casefold() in {"none", "no change", "nothing"}
            for value in objective.expected_observable_change.must_change
        ):
            issues.append(
                _issue(
                    "reasoner_objective_has_no_observable_delta",
                    "blocking",
                    objective.objective_id,
                    "A change objective must identify at least one observable delta.",
                )
            )
            blocked_ids.add(objective.objective_id)
        for value_ref, label in (
            (objective.source_value_ref, "source"),
            (objective.resolved_target_value, "target"),
        ):
            if (
                value_ref.value_status == "known"
                and value_ref.source == "perception_evidence"
                and not value_ref.evidence_refs
            ):
                issues.append(
                    _issue(
                        f"reasoner_{label}_value_evidence_refs_missing",
                        "blocking",
                        objective.objective_id,
                        "Perception-derived values require reviewed evidence refs.",
                    )
                )
                blocked_ids.add(objective.objective_id)
            if value_ref.source == "deterministic_derivation" and not value_ref.derivation:
                issues.append(
                    _issue(
                        f"reasoner_{label}_value_derivation_missing",
                        "blocking",
                        objective.objective_id,
                        "Deterministically derived values require a derivation summary.",
                    )
                )
                blocked_ids.add(objective.objective_id)
        unresolved_value_refs = [
            value_ref
            for value_ref in (
                objective.source_value_ref,
                objective.resolved_target_value,
            )
            if value_ref.value_status == "requires_evidence"
        ]
        unresolved_required_needs = [
            need
            for need in objective.required_evidence
            if need.required and not need.resolved_by_evidence_ids
        ]
        if objective.resolution_status == "needs_evidence" and not unresolved_required_needs:
            issues.append(
                _issue(
                    "reasoner_missing_evidence_need_not_declared",
                    "blocking",
                    objective.objective_id,
                    "needs_evidence requires a concrete unresolved evidence need.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if objective.resolution_status == "resolved" and unresolved_required_needs:
            issues.append(
                _issue(
                    "reasoner_resolved_with_unmet_evidence",
                    "blocking",
                    objective.objective_id,
                    "Resolved objective retains an unmet required evidence need.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if unresolved_value_refs and not unresolved_required_needs:
            issues.append(
                _issue(
                    "reasoner_value_evidence_need_not_declared",
                    "blocking",
                    objective.objective_id,
                    "Values marked requires_evidence need a concrete unresolved evidence request.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if objective.resolution_status == "resolved" and unresolved_value_refs:
            issues.append(
                _issue(
                    "reasoner_resolved_with_unresolved_value",
                    "blocking",
                    objective.objective_id,
                    "Resolved objectives cannot retain values marked requires_evidence.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if objective.resolution_status == "ambiguous" and not objective.uncertainties:
            issues.append(
                _issue(
                    "reasoner_ambiguity_not_described",
                    "blocking",
                    objective.objective_id,
                    "Ambiguous objective requires an explicit uncertainty.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if (
            objective.resolution_status == "resolved"
            and objective.transformation.kind in _TARGET_VALUE_REQUIRED
            and objective.resolved_target_value.value_status != "known"
        ):
            issues.append(
                _issue(
                    "reasoner_resolved_target_value_missing",
                    "blocking",
                    objective.objective_id,
                    "A resolved value-changing objective requires a target value.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if (
            objective.transformation.direction in _RELATIVE_DIRECTIONS
            and objective.source_value_ref.value_status != "known"
        ):
            issues.append(
                _issue(
                    "reasoner_relative_source_value_missing",
                    "blocking",
                    objective.objective_id,
                    "Relative change requires an explicit source value reference.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if objective.resolution_status != "resolved":
            issues.append(
                _issue(
                    f"reasoner_objective_{objective.resolution_status}",
                    "blocking",
                    objective.objective_id,
                    "Objective is not resolved enough to compile into an executable plan.",
                )
            )
            blocked_ids.add(objective.objective_id)
        missing_refs = sorted(
            {
                evidence_id
                for need in objective.required_evidence
                if need.required
                for evidence_id in need.resolved_by_evidence_ids
                if evidence_id not in available_evidence
            }
        )
        if missing_refs:
            issues.append(
                _issue(
                    "reasoner_evidence_reference_unavailable",
                    "blocking",
                    objective.objective_id,
                    "Objective cites reviewed evidence ids unavailable to this call.",
                )
            )
            blocked_ids.add(objective.objective_id)
        if supported_effects and objective.effect_kind not in supported_effects:
            issues.append(
                _issue(
                    "reasoner_effect_outside_runtime_ontology",
                    "advisory",
                    objective.objective_id,
                    "The semantic effect is not represented in the current runtime catalog.",
                )
            )

        prior = prior_by_id.get(objective.objective_id)
        if prior is not None:
            frozen_current = _frozen_objective_fields(objective)
            frozen_prior = _frozen_objective_fields(prior)
            if frozen_current != frozen_prior:
                correctable = request.semantic_revision_policy == "evidence_correctable"
                issues.append(
                    _issue(
                        (
                            "reasoner_evidence_semantics_revision"
                            if correctable
                            else "reasoner_frozen_semantics_drift"
                        ),
                        "advisory" if correctable else "blocking",
                        objective.objective_id,
                        (
                            "New evidence revised domain, transformation, direction, "
                            "or dependencies; the revision remains visible for audit."
                            if correctable
                            else "Evidence review changed domain, transformation, direction, or dependencies."
                        ),
                    )
                )
                if not correctable:
                    blocked_ids.add(objective.objective_id)
            if objective.requested_state_change != prior.requested_state_change:
                issues.append(
                    _issue(
                        "reasoner_state_change_rephrased",
                        "advisory",
                        objective.objective_id,
                        "Evidence review rephrased the frozen requested state change.",
                    )
                )

    if request.phase == "initial_understanding" and not request.reviewed_evidence_ids:
        for objective in intent.objectives:
            if any(
                reference.source == "perception_evidence"
                and reference.value_status == "known"
                and (bool(reference.value) or bool(reference.evidence_refs))
                for reference in (
                    objective.source_value_ref,
                    objective.resolved_target_value,
                )
            ):
                issues.append(
                    _issue(
                        "reasoner_initial_evidence_hallucination",
                        "blocking",
                        objective.objective_id,
                        "Initial understanding cannot claim unavailable perception evidence.",
                    )
                )
                blocked_ids.add(objective.objective_id)

    all_ids = [item.objective_id for item in intent.objectives]
    blocking_exists = any(item.severity == "blocking" for item in issues)
    return ReasonerAdmissionReport(
        planning_ready=not blocking_exists,
        issues=issues,
        admitted_objective_ids=[item for item in all_ids if item not in blocked_ids],
        blocked_objective_ids=[item for item in all_ids if item in blocked_ids],
    )


def reasoned_edit_intent_output_schema() -> dict[str, Any]:
    """Return the exact provider-visible Reasoner output schema."""

    return ReasonedEditIntent.model_json_schema()


def _validate_satisfied_change_binding_shape(
    changes: list[str],
    bindings: list[SatisfiedChangeProofBinding],
) -> None:
    if bindings and [item.change for item in bindings] != changes:
        raise ValueError(
            "completion proof bindings must match satisfied changes exactly and in order"
        )


def _transition_has_completion_authority(
    transition: SessionStateTransitionContext,
) -> bool:
    return (
        transition.status.casefold() == "completed"
        and transition.operation_type.casefold() not in {"state_selection", "rollback"}
        and bool(transition.operations)
    )


def _completion_proof(
    *,
    proof_kind: CompletionProofKind,
    selected_node: SessionStateNodeContext,
    transition: SessionStateTransitionContext,
) -> SessionCompletionProof:
    identity = {
        "proof_kind": proof_kind,
        "selected_node_id": selected_node.node_id,
        "transition_edge_id": transition.edge_id,
    }
    digest = _fingerprint(identity)
    return SessionCompletionProof(
        proof_ref=f"graph-completion:{proof_kind}:{digest}",
        proof_kind=proof_kind,
        selected_node_id=selected_node.node_id,
        selected_audio_ref=selected_node.audio_ref or "",
        transition_edge_id=transition.edge_id,
        transition_source_node_ids=list(transition.source_node_ids),
        transition_target_node_id=transition.target_node_id,
        transition_instruction=transition.instruction,
        transition_operations=list(transition.operations),
    )


def _overall_resolution_status(
    objectives: list[ReasonedEditObjective],
) -> ReasonerResolutionStatus:
    statuses = {item.resolution_status for item in objectives}
    for status in ("unsupported", "ambiguous", "needs_evidence"):
        if status in statuses:
            return status
    return "resolved"


def _frozen_objective_fields(objective: ReasonedEditObjective) -> dict[str, Any]:
    return {
        "objective_id": objective.objective_id,
        "referent_domain": objective.referent_domain,
        "transformation_kind": objective.transformation.kind,
        "direction": objective.transformation.direction,
        "dependency_ids": list(objective.dependency_ids),
    }


def _issue(
    code: str,
    severity: ReasonerIssueSeverity,
    objective_id: str | None,
    message: str,
) -> ReasonerAdmissionIssue:
    return ReasonerAdmissionIssue(
        code=code,
        severity=severity,
        objective_id=objective_id,
        message=message,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ChangeDirection",
    "CompletionProofKind",
    "ExpectedObservableChange",
    "ObservationNeed",
    "PriorTurnSemanticContext",
    "ReasonedEditIntent",
    "ReasonedEditObjective",
    "ReasonerAdmissionIssue",
    "ReasonerAdmissionReport",
    "ReasonerCallRequest",
    "ReasonerEvidenceNeed",
    "ReasonerPhase",
    "ReasonerResolutionStatus",
    "ReasonerSemanticRevisionPolicy",
    "ReasonerShadowStrategy",
    "ReasonerTransformation",
    "ReferentDomain",
    "RequestedSemanticChange",
    "SatisfiedChangeProofBinding",
    "SemanticCandidateRef",
    "SemanticEntityKind",
    "SemanticEntityRole",
    "SemanticReasoningMemo",
    "SemanticReasoningMemoOutput",
    "SemanticTargetBinding",
    "SemanticTargetBindingStatus",
    "SemanticTargetCandidateKind",
    "SemanticTargetRelation",
    "SemanticTargetSelector",
    "SemanticTemporalExtent",
    "SemanticValueReference",
    "SessionCompletionProof",
    "SessionStateAudioSlotContext",
    "SessionStateContext",
    "SessionStateNodeContext",
    "SessionStateTransitionContext",
    "admit_reasoned_edit_intent",
    "reasoned_edit_intent_output_schema",
    "required_semantic_target_binding_failures",
    "reviewed_candidate_ref_inventory",
    "session_completion_proof_inventory",
    "validate_semantic_memo_completion_authority",
    "validate_semantic_memo_target_binding_authority",
]
