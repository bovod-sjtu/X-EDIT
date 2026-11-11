"""Typed Perception capabilities exposed to workflow Planners.

The catalog describes evidence contracts and admissible implementation routes.
It deliberately does not prescribe one model for every request: Planner skills
select the evidence need, while the runtime records the backend route actually
used to satisfy it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal

PerceptionTargetKind = Literal[
    "event_label",
    "speaker_hint",
    "quoted_text",
    "semantic_property",
    "audio_property",
]


@dataclass(frozen=True)
class WorkflowPerceptionCapability:
    """One Planner-visible evidence contract with replaceable backend routes."""

    operation: str
    description: str
    skill_ids: tuple[str, ...]
    target_kinds: tuple[PerceptionTargetKind, ...]
    evidence_types: tuple[str, ...]
    requirement_rules: tuple[str, ...]
    backend_routes: tuple[str, ...]
    temporal_precision: tuple[str, ...]
    entity_kinds: tuple[str, ...] = ()
    output_role: Literal[
        "authoritative_specialist",
        "reviewed_evidence",
        "candidate_prior",
    ] = "reviewed_evidence"
    relative_cost: int = 100
    audio_input_mode: Literal[
        "single_audio",
        "paired_audio",
        "multi_audio",
    ] = "single_audio"
    target_kind_companions: tuple[
        tuple[PerceptionTargetKind, tuple[PerceptionTargetKind, ...]], ...
    ] = ()
    target_kind_evidence_companions: tuple[tuple[PerceptionTargetKind, tuple[str, ...]], ...] = ()
    candidate_rubric: bool = False
    flexsed_policy: Literal["not_used", "optional", "required"] = "not_used"
    supports_recognition_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowPerceptionCapabilityResolution:
    """Catalog match plus requested evidence that no registered route exposes."""

    compatible_operations: tuple[str, ...]
    recognized_evidence_fields: tuple[str, ...]
    unsupported_evidence_fields: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowPerceptionRoutePart:
    """One lossless part of a uniquely decomposable mixed evidence job."""

    operation: str
    expected_evidence_fields: tuple[str, ...]
    target_kinds: tuple[str, ...]


WORKFLOW_PERCEPTION_CAPABILITIES: tuple[WorkflowPerceptionCapability, ...] = (
    WorkflowPerceptionCapability(
        operation="perception.speech_context",
        description=(
            "Ground verbatim transcript content and word timing with specialist "
            "ASR and forced alignment; Planner/Reasoner may classify transcript "
            "content, while audio-dependent classes use semantic_span_review. "
            "An explicit recognition_context requests fresh contextual recognition "
            "and new word alignment, allowing a deficient prior transcript to be revised."
        ),
        skill_ids=(
            "timeline_span_grounding",
            "speech_like_vocalization_span_grounding",
            "edit_data_pipeline_local_edit",
            "censor_masking_span_edit",
        ),
        target_kinds=("quoted_text",),
        evidence_types=("source_text", "word_timestamps"),
        requirement_rules=(
            "source_text is required for transcript or exact lexical-content grounding",
            "word_timestamps are required only for word-precision or local phrase boundaries",
            "semantic selection is a downstream Planner/Reasoner responsibility unless audio cues require reviewed selected_spans",
        ),
        backend_routes=("qwen3_asr", "forced_alignment"),
        temporal_precision=("coarse", "segment", "word"),
        entity_kinds=("spoken_content",),
        output_role="authoritative_specialist",
        relative_cost=10,
        supports_recognition_context=True,
    ),
    WorkflowPerceptionCapability(
        operation="perception.speaker_attributed_review",
        description=(
            "Ground speaker IDs, order, overlap, and speaker-scoped speech from "
            "speaker-aware timestamped ASR evidence with optional transcript review; "
            "a separate audio observation supplements task-relevant voice attributes."
        ),
        skill_ids=("speaker_attributed_speech_span_grounding",),
        target_kinds=(
            "speaker_hint",
            "audio_property",
            "quoted_text",
            "semantic_property",
        ),
        evidence_types=(
            "source_text",
            "word_timestamps",
            "speaker_segments",
            "speaker_profiles",
            "speaker_attributed_transcript",
            "selected_speaker_refs",
            "selected_speaker_ids",
            "selected_spans",
            "target_bindings",
        ),
        requirement_rules=(
            "timestamped speaker segments are required for speaker-scoped edits",
            "speaker ids are scoped by audio_ref_id and must remain stable within that audio",
            "transcript evidence is required only when speech content constrains the target",
            "task-required speaker profiles need supplement_speaker_attributes even when transcript review is disabled",
            "attribute observation preserves measured transcript and timestamps; unknown properties remain unresolved",
            "Perception may publish candidate speakers with uncertainty; the evidence-review Reasoner owns semantic role selection",
            "only framework-materialized resolved bindings may become audio-scoped selected_speaker_refs or selected_spans for execution",
        ),
        backend_routes=(
            "qwen3_asr_alignment_diarization",
            "speaker_aware_ts_asr",
            "qwen3_omni_evidence_review",
        ),
        temporal_precision=("coarse", "segment", "word"),
        entity_kinds=("speaker",),
        output_role="reviewed_evidence",
        relative_cost=24,
        target_kind_companions=(
            ("semantic_property", ("speaker_hint",)),
        ),
        target_kind_evidence_companions=(
            ("semantic_property", ("speaker_segments", "speaker_profiles", "speaker_attributed_transcript")),
        ),
    ),
    WorkflowPerceptionCapability(
        operation="perception.semantic_span_review",
        description=(
            "Build aligned speech candidates with TS-ASR evidence, then use an "
            "Omni yes/no rubric only when deterministic grounding is insufficient."
        ),
        skill_ids=(
            "perceptual_property_span_grounding",
            "timeline_span_grounding",
            "speech_like_vocalization_span_grounding",
        ),
        target_kinds=(
            "speaker_hint",
            "quoted_text",
            "semantic_property",
            "audio_property",
        ),
        evidence_types=(
            "source_text",
            "word_timestamps",
            "candidate_segments",
            "selected_spans",
            "candidate_rubric_judgements",
            "target_bindings",
        ),
        requirement_rules=(
            "aligned candidate segments may be admitted before final semantic selection",
            "candidate rubric judgements are required only when deterministic candidate matching is insufficient",
            "the evidence-review Reasoner binds semantic targets to admitted audio-scoped candidate refs",
            "only framework-materialized resolved bindings may become selected_spans for execution",
        ),
        backend_routes=(
            "qwen3_asr_alignment",
            "deterministic_candidate_match",
            "qwen3_omni_candidate_rubric",
        ),
        temporal_precision=("segment", "word"),
        entity_kinds=("spoken_content",),
        output_role="reviewed_evidence",
        relative_cost=20,
        candidate_rubric=True,
    ),
    WorkflowPerceptionCapability(
        operation="perception.audio_fallback",
        description=(
            "Obtain open semantic audio observations directly from Omni, optionally "
            "seeded by specialist speech evidence."
        ),
        skill_ids=(
            "perceptual_property_span_grounding",
            "timeline_span_grounding",
        ),
        target_kinds=(
            "speaker_hint",
            "quoted_text",
            "semantic_property",
            "audio_property",
            "event_label",
        ),
        evidence_types=(
            "source_text",
            "speaker_segments",
            "speaker_profiles",
            "segment_descriptions",
            "audio_properties",
        ),
        requirement_rules=(
            "at least one reviewed fact relevant to the declared target is required",
            "speaker profiles are audio-scoped observations in speakers and segment descriptions; uncertain voice attributes remain unknown",
            "a local destructive edit still requires confirmed local boundaries",
        ),
        backend_routes=(
            "specialist_speech_evidence",
            "qwen3_omni_direct_review",
        ),
        temporal_precision=("coarse", "segment"),
        entity_kinds=("acoustic_layer", "asset"),
        output_role="reviewed_evidence",
        relative_cost=18,
    ),
    WorkflowPerceptionCapability(
        operation="perception.timestamped_caption",
        description=(
            "Ground open-ended audio captions or non-speech events with Omni; a "
            "FlexSED prior is optional and never implied by this operation alone."
        ),
        skill_ids=("audio_caption_span_grounding",),
        target_kinds=("event_label", "semantic_property", "audio_property"),
        evidence_types=(
            "audio_caption_segments",
            "selected_spans",
            "candidate_rubric_judgements",
        ),
        requirement_rules=(
            "timestamped caption segments are required for caption grounding",
            "confirmed selected_spans are required for a targeted local edit",
            "a candidate rubric is required only for unresolved model-only candidates",
        ),
        backend_routes=(
            "qwen3_omni_direct_grounding",
            "optional_flexsed_candidate_prior",
            "qwen3_omni_candidate_rubric",
        ),
        temporal_precision=("coarse", "segment"),
        entity_kinds=("sound_event", "acoustic_layer"),
        output_role="reviewed_evidence",
        relative_cost=26,
        candidate_rubric=True,
        flexsed_policy="optional",
    ),
    WorkflowPerceptionCapability(
        operation="perception.flexsed_events",
        description=(
            "Ground an explicit audible-event query with FlexSED when event-detector "
            "evidence is the selected route."
        ),
        skill_ids=("audio_event_span_grounding",),
        target_kinds=("event_label", "semantic_property", "audio_property"),
        evidence_types=("event_labels", "event_timestamps", "event_heatmap"),
        requirement_rules=(
            (
                "an event_label, typed event property, or explicit atomic evidence "
                "question is required for this detector route"
            ),
            "the authored query is forwarded without extracting a label by string rules",
            "event timestamps are required for local editing",
        ),
        backend_routes=("flexsed_event_detection",),
        temporal_precision=("segment",),
        entity_kinds=("sound_event",),
        output_role="candidate_prior",
        relative_cost=12,
        flexsed_policy="required",
    ),
)

_BY_OPERATION = {
    capability.operation: capability for capability in WORKFLOW_PERCEPTION_CAPABILITIES
}


def workflow_perception_capability(
    operation: str,
) -> WorkflowPerceptionCapability | None:
    return _BY_OPERATION.get(operation)


def workflow_perception_capabilities(
    available_operations: Iterable[str] | None = None,
) -> list[WorkflowPerceptionCapability]:
    if available_operations is None:
        return list(WORKFLOW_PERCEPTION_CAPABILITIES)
    available = {str(operation) for operation in available_operations}
    return [
        capability
        for capability in WORKFLOW_PERCEPTION_CAPABILITIES
        if capability.operation in available
    ]


def compatible_workflow_perception_operations(
    *,
    target_kinds: Iterable[str],
    expected_evidence_fields: Iterable[str],
    temporal_precision: str,
    available_operations: Iterable[str] | None = None,
    allow_unique_evidence_completion: bool = False,
) -> list[str]:
    """Return contracts that can satisfy all declared, catalog-known needs.

    Unknown evidence field names are left for typed result validation. They do
    not make every catalog entry incompatible merely because a Planner supplied
    an additional backend-specific or nested field name.
    """

    return list(
        resolve_workflow_perception_capabilities(
            target_kinds=target_kinds,
            expected_evidence_fields=expected_evidence_fields,
            temporal_precision=temporal_precision,
            available_operations=available_operations,
            allow_unique_evidence_completion=allow_unique_evidence_completion,
        ).compatible_operations
    )


def resolve_workflow_perception_capabilities(
    *,
    target_kinds: Iterable[str],
    expected_evidence_fields: Iterable[str],
    temporal_precision: str,
    available_operations: Iterable[str] | None = None,
    allow_unique_evidence_completion: bool = False,
) -> WorkflowPerceptionCapabilityResolution:
    """Resolve routes while exposing, rather than silently dropping, unknown fields."""

    capabilities = workflow_perception_capabilities(available_operations)
    requested_kinds = {str(kind).strip() for kind in target_kinds if str(kind).strip()}
    known_evidence_types = {
        evidence_type for capability in capabilities for evidence_type in capability.evidence_types
    }
    field_rows = [
        (str(field_path), _catalog_evidence_type(str(field_path)))
        for field_path in expected_evidence_fields
        if str(field_path).strip()
    ]
    requested_evidence_types = {
        normalized for _field_path, normalized in field_rows if normalized in known_evidence_types
    }
    operations = [
        capability.operation
        for capability in capabilities
        if requested_kinds.issubset(capability.target_kinds)
        and _target_companions_satisfied(
            capability,
            requested_kinds=requested_kinds,
            requested_evidence_types=requested_evidence_types,
        )
        and requested_evidence_types.issubset(capability.evidence_types)
        and temporal_precision in capability.temporal_precision
    ]
    if not operations and allow_unique_evidence_completion:
        operations = [
            capability.operation
            for capability in capabilities
            if requested_kinds.issubset(capability.target_kinds)
            and _target_companions_satisfied(
                capability,
                requested_kinds=requested_kinds,
                requested_evidence_types=requested_evidence_types,
                allow_unique_evidence_completion=True,
            )
            and requested_evidence_types.issubset(capability.evidence_types)
            and temporal_precision in capability.temporal_precision
        ]
    return WorkflowPerceptionCapabilityResolution(
        compatible_operations=tuple(operations),
        recognized_evidence_fields=tuple(
            field_path
            for field_path, normalized in field_rows
            if normalized in known_evidence_types
        ),
        unsupported_evidence_fields=tuple(
            field_path
            for field_path, normalized in field_rows
            if normalized not in known_evidence_types
        ),
    )


def unique_workflow_perception_evidence_companion_additions(
    *,
    operation: str,
    target_kinds: Iterable[str],
    expected_evidence_fields: Iterable[str],
) -> tuple[str, ...]:
    """Return only unambiguous, catalog-declared evidence companion additions.

    This is a contract completion, not semantic repair: it may only add the
    sole evidence type that the already-selected route declares for a target
    kind. Ambiguous alternatives remain a Planner error.
    """

    capability = workflow_perception_capability(operation)
    if capability is None:
        return ()
    requested_kinds = {str(kind).strip() for kind in target_kinds if str(kind).strip()}
    requested_evidence_types = {
        _catalog_evidence_type(str(field_path))
        for field_path in expected_evidence_fields
        if str(field_path).strip()
    }
    evidence_alternatives = {
        target_kind: tuple(evidence_types)
        for target_kind, evidence_types in capability.target_kind_evidence_companions
    }
    additions: list[str] = []
    for target_kind, companions in capability.target_kind_companions:
        if target_kind not in requested_kinds:
            continue
        if set(companions).issubset(requested_kinds):
            continue
        alternatives = evidence_alternatives.get(target_kind, ())
        if set(alternatives) & requested_evidence_types:
            continue
        if len(alternatives) != 1:
            return ()
        addition = alternatives[0]
        if addition not in additions:
            additions.append(addition)
    return tuple(additions)


def uniquely_decompose_workflow_perception_requirements(
    *,
    selected_operation: str,
    target_kinds: Iterable[str],
    expected_evidence_fields: Iterable[str],
    temporal_precision: str,
    available_operations: Iterable[str] | None = None,
) -> tuple[WorkflowPerceptionRoutePart, ...]:
    """Split a mixed job only when its minimum catalog cover is unique.

    The Planner-selected route is retained when it contributes any declared
    evidence. Supplemental routes must uniquely cover every remaining known
    evidence type and target kind. No declared field is removed or rewritten.
    """

    capabilities = [
        capability
        for capability in workflow_perception_capabilities(available_operations)
        if temporal_precision in capability.temporal_precision
    ]
    capability_by_operation = {
        capability.operation: capability for capability in capabilities
    }
    selected = capability_by_operation.get(selected_operation)
    if selected is None:
        return ()

    fields = tuple(
        dict.fromkeys(
            str(field_path)
            for field_path in expected_evidence_fields
            if str(field_path).strip()
        )
    )
    kinds = tuple(
        dict.fromkeys(str(kind).strip() for kind in target_kinds if str(kind).strip())
    )
    known_evidence_types = {
        evidence_type for capability in capabilities for evidence_type in capability.evidence_types
    }
    field_type = {field: _catalog_evidence_type(field) for field in fields}
    unsupported_fields = [
        field for field, evidence_type in field_type.items() if evidence_type not in known_evidence_types
    ]

    selected_coverage = {
        *(f"evidence:{evidence_type}" for evidence_type in selected.evidence_types),
        *(f"target:{kind}" for kind in selected.target_kinds),
    }
    required_coverage = {
        *(f"evidence:{field_type[field]}" for field in fields if field not in unsupported_fields),
        *(f"target:{kind}" for kind in kinds),
    }
    selected_contribution = required_coverage & selected_coverage
    if not selected_contribution:
        return ()
    remaining = required_coverage - selected_coverage

    supplemental_candidates = [
        capability
        for capability in capabilities
        if capability.operation != selected.operation
        and remaining
        & {
            *(f"evidence:{evidence_type}" for evidence_type in capability.evidence_types),
            *(f"target:{kind}" for kind in capability.target_kinds),
        }
    ]
    solutions: list[tuple[WorkflowPerceptionCapability, ...]] = []
    if not remaining:
        solutions = [()]
    else:
        for size in range(1, len(supplemental_candidates) + 1):
            for candidate_set in combinations(supplemental_candidates, size):
                covered = set(selected_coverage)
                for capability in candidate_set:
                    covered.update(
                        f"evidence:{evidence_type}"
                        for evidence_type in capability.evidence_types
                    )
                    covered.update(f"target:{kind}" for kind in capability.target_kinds)
                if required_coverage.issubset(covered):
                    solutions.append(candidate_set)
            if solutions:
                break
    if len(solutions) != 1 or not solutions[0]:
        return ()

    chosen = (selected, *solutions[0])
    fields_by_operation: dict[str, list[str]] = {
        capability.operation: [] for capability in chosen
    }
    targets_by_operation: dict[str, list[str]] = {
        capability.operation: [] for capability in chosen
    }
    for field in fields:
        evidence_type = field_type[field]
        if field in unsupported_fields:
            fields_by_operation[selected.operation].append(field)
            continue
        owner = next(
            (
                capability
                for capability in chosen
                if evidence_type in capability.evidence_types
            ),
            None,
        )
        if owner is None:
            return ()
        fields_by_operation[owner.operation].append(field)
    for kind in kinds:
        owner = next(
            (capability for capability in chosen if kind in capability.target_kinds),
            None,
        )
        if owner is None:
            return ()
        targets_by_operation[owner.operation].append(kind)

    for capability in chosen:
        if fields_by_operation[capability.operation]:
            continue
        shared_field = next(
            (
                field
                for field in fields
                if field_type[field] in capability.evidence_types
            ),
            None,
        )
        if shared_field is None:
            return ()
        fields_by_operation[capability.operation].append(shared_field)

    return tuple(
        WorkflowPerceptionRoutePart(
            operation=capability.operation,
            expected_evidence_fields=tuple(fields_by_operation[capability.operation]),
            target_kinds=tuple(targets_by_operation[capability.operation]),
        )
        for capability in chosen
    )


def _target_companions_satisfied(
    capability: WorkflowPerceptionCapability,
    *,
    requested_kinds: set[str],
    requested_evidence_types: set[str],
    allow_unique_evidence_completion: bool = False,
) -> bool:
    evidence_alternatives = {
        target_kind: set(evidence_types)
        for target_kind, evidence_types in capability.target_kind_evidence_companions
    }
    for target_kind, companions in capability.target_kind_companions:
        if target_kind not in requested_kinds:
            continue
        if set(companions).issubset(requested_kinds):
            continue
        if evidence_alternatives.get(target_kind, set()) & requested_evidence_types:
            continue
        if (
            allow_unique_evidence_completion
            and len(evidence_alternatives.get(target_kind, set())) == 1
        ):
            continue
        return False
    return True


def _catalog_evidence_type(field_path: str) -> str:
    normalized = field_path.strip()
    for prefix in ("facts.", "perception_report."):
        normalized = normalized.removeprefix(prefix)
    return normalized.split(".", 1)[0].split("[", 1)[0]


def workflow_perception_catalog_payload(
    available_operations: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        capability.to_dict()
        for capability in workflow_perception_capabilities(available_operations)
    ]


def workflow_perception_catalog_fingerprint(
    available_operations: Iterable[str] | None = None,
) -> str:
    payload = workflow_perception_catalog_payload(available_operations)
    material = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


__all__ = [
    "WORKFLOW_PERCEPTION_CAPABILITIES",
    "WorkflowPerceptionCapability",
    "WorkflowPerceptionCapabilityResolution",
    "WorkflowPerceptionRoutePart",
    "compatible_workflow_perception_operations",
    "resolve_workflow_perception_capabilities",
    "unique_workflow_perception_evidence_companion_additions",
    "uniquely_decompose_workflow_perception_requirements",
    "workflow_perception_capabilities",
    "workflow_perception_capability",
    "workflow_perception_catalog_fingerprint",
    "workflow_perception_catalog_payload",
]
