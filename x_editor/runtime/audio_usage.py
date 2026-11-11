"""Per-turn audio usage contracts shared across workflow components.

This module deliberately has no Planner, Reasoner, or workflow-case imports so
the contracts can be consumed at every boundary without creating import cycles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

AudioUsageKind = Literal[
    "edit_input",
    "reference_input",
    "speaker_grounded_reference",
    "assembly_member",
    "comparison_input",
]
AudioEvidenceRequirement = Literal["existence_decodability", "reviewed_facts"]
ReviewedEvidencePathKind = Literal["speaker_fact", "other"]

_INTRINSIC_SPEAKER_FACT_FIELDS = {
    "selected_speaker_ids",
    "speaker_profiles",
    "speaker_segments",
    "speakers",
}
_SPEAKER_ATTRIBUTED_COLLECTION_FIELDS = {
    "candidate_segments",
    "segments",
    "selected_spans",
    "word_timestamps",
}


class WorkflowAudioEvidenceReference(StrictModel):
    """One exact reviewed-evidence path supporting an audio usage decision."""

    evidence_id: str = Field(min_length=1, max_length=256)
    evidence_path: str = Field(min_length=1, max_length=512)


class WorkflowAudioUsageBinding(StrictModel):
    """Planner-authored per-step use of one immutable audio asset."""

    schema_version: Literal["x-editor.workflow-audio-usage-binding/v1"] = (
        "x-editor.workflow-audio-usage-binding/v1"
    )
    plan_step_id: str = Field(min_length=1, max_length=256)
    argument_path: str = Field(min_length=1, max_length=512)
    audio_ref: str = Field(min_length=1, max_length=4096)
    usage: AudioUsageKind
    evidence_requirement: AudioEvidenceRequirement = "existence_decodability"
    evidence: list[WorkflowAudioEvidenceReference] = Field(
        default_factory=list,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_binding(self) -> WorkflowAudioUsageBinding:
        self.plan_step_id = self.plan_step_id.strip()
        self.argument_path = self.argument_path.strip()
        self.audio_ref = self.audio_ref.strip()
        if not self.argument_path.startswith(("inputs.", "params.")):
            raise ValueError(
                "audio usage argument_path must start with inputs. or params."
            )
        evidence_keys = [
            (item.evidence_id, item.evidence_path) for item in self.evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("audio usage evidence references must be unique")
        if self.evidence_requirement == "reviewed_facts" and not self.evidence:
            raise ValueError(
                "reviewed_facts audio usage requires at least one evidence reference"
            )
        if (
            self.usage == "speaker_grounded_reference"
            and self.evidence_requirement != "reviewed_facts"
        ):
            raise ValueError(
                "speaker_grounded_reference usage requires reviewed speaker facts"
            )
        return self


def audio_evidence_requirement_for_usage(
    usage: AudioUsageKind,
) -> AudioEvidenceRequirement:
    """Return the framework-owned evidence level implied by one usage kind."""

    if usage == "speaker_grounded_reference":
        return "reviewed_facts"
    return "existence_decodability"


def reviewed_evidence_path_kind(path: str) -> ReviewedEvidencePathKind:
    """Classify one canonical reviewed-evidence path by typed schema fields.

    Identity selector payloads are deliberately ignored. Classification uses
    exact reviewed-evidence collection and leaf names, never words contained in
    a transcript, question, selector value, or benchmark record.
    """

    components = _canonical_field_path_components(path)
    for index, component in enumerate(components):
        if component in _INTRINSIC_SPEAKER_FACT_FIELDS:
            return "speaker_fact"
        if component in _SPEAKER_ATTRIBUTED_COLLECTION_FIELDS and (
            "speaker_id" in components[index + 1 :]
        ):
            return "speaker_fact"
        if component == "target_bindings" and (
            "resolved_speaker_ids" in components[index + 1 :]
        ):
            return "speaker_fact"
    return "other"


def canonical_speaker_evidence_references(
    *,
    audio_ref: str,
    evidence_path_sources: Mapping[str, Sequence[str]],
    evidence_path_audio_refs: Mapping[str, str],
) -> list[WorkflowAudioEvidenceReference]:
    """Select a stable same-audio speaker fact from admitted typed evidence.

    The Planner decides that an asset is a speaker-grounded reference. The
    framework then performs the mechanical provenance join over admitted
    ``path -> evidence ids`` and ``path -> audio ref`` indexes. No semantic
    interpretation or fallback evidence is introduced here.
    """

    candidates: list[tuple[int, int, str, str]] = []
    for raw_path, raw_evidence_ids in evidence_path_sources.items():
        path = str(raw_path).strip()
        if (
            not path
            or evidence_path_audio_refs.get(path) != audio_ref
            or reviewed_evidence_path_kind(path) != "speaker_fact"
        ):
            continue
        components = _canonical_field_path_components(path)
        for raw_evidence_id in raw_evidence_ids:
            evidence_id = str(raw_evidence_id).strip()
            if evidence_id:
                candidates.append((len(components), len(path), path, evidence_id))
    if not candidates:
        return []
    _, _, path, evidence_id = min(candidates)
    return [
        WorkflowAudioEvidenceReference(
            evidence_id=evidence_id,
            evidence_path=path,
        )
    ]


def _canonical_field_path_components(path: str) -> tuple[str, ...]:
    """Return field names while excluding canonical identity selectors."""

    components: list[str] = []
    current: list[str] = []
    selector_depth = 0
    for character in str(path).strip():
        if character == "[":
            if selector_depth == 0 and current:
                components.append("".join(current))
                current = []
            selector_depth += 1
            continue
        if character == "]":
            selector_depth = max(0, selector_depth - 1)
            continue
        if character == "." and selector_depth == 0:
            if current:
                components.append("".join(current))
                current = []
            continue
        if selector_depth == 0:
            current.append(character)
    if current:
        components.append("".join(current))
    return tuple(item for item in components if item)


__all__ = [
    "AudioEvidenceRequirement",
    "AudioUsageKind",
    "ReviewedEvidencePathKind",
    "WorkflowAudioEvidenceReference",
    "WorkflowAudioUsageBinding",
    "audio_evidence_requirement_for_usage",
    "canonical_speaker_evidence_references",
    "reviewed_evidence_path_kind",
]
