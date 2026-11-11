"""Benchmark-neutral inputs for the shared workflow runtime."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import Field, model_validator

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.schemas import StrictModel
from x_editor.reasoner.contracts import PriorTurnSemanticContext, SessionStateContext
from x_editor.runtime.audio_usage import (
    AudioEvidenceRequirement,
    AudioUsageKind,
    WorkflowAudioEvidenceReference,
    WorkflowAudioUsageBinding,
)

AudioAssetProvenance = Literal["user_provided", "generated", "session_output"]


class HistoricalReviewedEvidenceContext(StrictModel):
    """Framework-authored, state-bound evidence retained across session turns."""

    schema_version: Literal[
        "x-editor.historical-reviewed-evidence-context/v1"
    ] = "x-editor.historical-reviewed-evidence-context/v1"
    authority: Literal["session_graph_reviewed_evidence_history"] = (
        "session_graph_reviewed_evidence_history"
    )
    temporal_reuse_policy: str = Field(min_length=1)
    snapshots: list[dict[str, object]] = Field(min_length=1, max_length=128)


class WorkflowAudioAsset(StrictModel):
    """One typed audio input available to a workflow scenario."""

    schema_version: Literal["x-editor.workflow-audio-asset/v1"] = (
        "x-editor.workflow-audio-asset/v1"
    )
    asset_ref: str = Field(min_length=1)
    path: str = Field(min_length=1)
    provenance: AudioAssetProvenance = "user_provided"
    role: Literal["source", "reference"]
    order: int = Field(ge=0)
    content_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @property
    def logical_ref(self) -> str:
        """Return a storage- and dataset-independent runtime-visible reference."""

        prefix = "s" if self.role == "source" else "r"
        return f"aud:v1:{prefix}{self.order:04d}"


class RuntimeAudioBinding(StrictModel):
    """Runtime-private resolution from one logical audio ref to local storage."""

    schema_version: Literal["x-editor.runtime-audio-binding/v1"] = (
        "x-editor.runtime-audio-binding/v1"
    )
    audio_ref: str = Field(min_length=1)
    materialized_path: str = Field(min_length=1)
    provenance: AudioAssetProvenance | None = None
    role: Literal["source", "reference", "authoritative_output"]
    order: int = Field(ge=0)
    origin: Literal[
        "scenario_asset",
        "artifact_store_commit",
        "session_graph_state",
    ]
    content_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    origin_artifact_ref_id: str | None = None
    origin_state_node_id: str | None = None

    @model_validator(mode="after")
    def bind_provenance(self) -> RuntimeAudioBinding:
        expected = (
            "session_output"
            if self.origin in {"artifact_store_commit", "session_graph_state"}
            else "user_provided"
        )
        if self.provenance is None:
            self.provenance = expected
        elif (
            self.origin in {"artifact_store_commit", "session_graph_state"}
            and self.provenance != expected
        ):
            raise ValueError(
                "authoritative session audio bindings require session_output provenance"
            )
        if self.origin == "session_graph_state" and not self.origin_state_node_id:
            raise ValueError("Session Graph state bindings require origin_state_node_id")
        return self


class WorkflowRuntimeCase(StrictModel):
    """Materialized, single-turn input accepted by the workflow runtime."""

    schema_version: Literal["x-editor.workflow-runtime-case/v1"] = (
        "x-editor.workflow-runtime-case/v1"
    )
    user_request: str = Field(min_length=1)
    source_audio_refs: list[str] = Field(min_length=1, max_length=16)
    reference_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    source_text: str | None = None
    prior_turn_contexts: list[PriorTurnSemanticContext] = Field(
        default_factory=list,
        max_length=32,
    )
    session_state: SessionStateContext | None = None
    historical_reviewed_evidence: HistoricalReviewedEvidenceContext | None = Field(
        default=None,
        exclude=True,
    )
    future_attempt_guidance: FutureAttemptGuidance | None = Field(
        default=None,
        exclude=True,
    )
    planning_context: dict[str, object] = Field(default_factory=dict)
    audio_resolution_mode: Literal["legacy_path", "opaque_ref_v1"] = "legacy_path"
    audio_bindings: list[RuntimeAudioBinding] = Field(
        default_factory=list,
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_refs(self) -> WorkflowRuntimeCase:
        self.source_audio_refs = _normalized_unique_refs(
            self.source_audio_refs,
            field_name="source_audio_refs",
        )
        self.reference_audio_refs = _normalized_unique_refs(
            self.reference_audio_refs,
            field_name="reference_audio_refs",
        )
        if self.source_text is not None:
            self.source_text = self.source_text.strip() or None
        if self.historical_reviewed_evidence is not None and self.session_state is None:
            raise ValueError(
                "historical reviewed evidence requires authoritative Session Graph state"
            )
        if self.audio_resolution_mode == "opaque_ref_v1":
            self._validate_opaque_bindings()
        return self

    def _validate_opaque_bindings(self) -> None:
        refs = [item.audio_ref for item in self.audio_bindings]
        invalid_refs = sorted(
            ref
            for ref in refs
            if re.fullmatch(
                r"aud:v1:[A-Za-z0-9][A-Za-z0-9._-]*",
                ref,
                flags=re.IGNORECASE,
            )
            is None
            or not ref.startswith("aud:v1:")
        )
        if invalid_refs:
            raise ValueError("opaque runtime audio bindings require canonical refs")
        if len(refs) != len(set(refs)):
            raise ValueError("runtime audio binding refs must be unique")
        orders = [item.order for item in self.audio_bindings]
        if len(orders) != len(set(orders)):
            raise ValueError("runtime audio binding order values must be unique")
        expected = set(self.source_audio_refs) | set(self.reference_audio_refs)
        if set(refs) != expected:
            raise ValueError(
                "opaque runtime audio refs require one exact private binding each"
            )
        by_ref = {item.audio_ref: item for item in self.audio_bindings}
        wrong_sources = sorted(
            ref
            for ref in self.source_audio_refs
            if by_ref[ref].role not in {"source", "authoritative_output"}
        )
        wrong_references = sorted(
            ref
            for ref in self.reference_audio_refs
            if by_ref[ref].role != "reference"
        )
        if wrong_sources or wrong_references:
            raise ValueError("opaque runtime audio refs have incompatible binding roles")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema_version": self.schema_version,
                "user_request": self.user_request,
                "source_audio_refs": self.source_audio_refs,
                "reference_audio_refs": self.reference_audio_refs,
                "source_text": self.source_text,
                "prior_turn_contexts": [
                    item.model_dump(mode="json") for item in self.prior_turn_contexts
                ],
                "session_state": (
                    self.session_state.model_dump(mode="json")
                    if self.session_state is not None
                    else None
                ),
                "historical_reviewed_evidence": (
                    self.historical_reviewed_evidence.model_dump(mode="json")
                    if self.historical_reviewed_evidence is not None
                    else None
                ),
                "future_attempt_guidance_fingerprint": (
                    self.future_attempt_guidance.guidance_fingerprint
                    if self.future_attempt_guidance is not None
                    else None
                ),
                "materialized_input_fingerprint": (
                    self.materialized_input_fingerprint
                ),
            }
        )

    @property
    def materialized_input_fingerprint(self) -> str | None:
        """Hash prompt-hidden input identity without exposing storage paths."""

        if self.audio_resolution_mode != "opaque_ref_v1":
            return None
        return _fingerprint(
            [
                {
                    "audio_ref": item.audio_ref,
                    "provenance": item.provenance,
                    "role": item.role,
                    "order": item.order,
                    "content_sha256": item.content_sha256,
                    "origin": item.origin,
                    "origin_artifact_ref_id": item.origin_artifact_ref_id,
                }
                for item in sorted(self.audio_bindings, key=lambda value: value.order)
            ]
        )

    def resolve_audio_ref(self, audio_ref: str) -> str:
        """Resolve a bound ref without exposing the binding in public payloads."""

        if self.audio_resolution_mode == "legacy_path":
            return audio_ref
        for binding in self.audio_bindings:
            if binding.audio_ref == audio_ref:
                return binding.materialized_path
        raise ValueError(f"unknown runtime audio ref: {audio_ref}")


class WorkflowRuntimeTurn(StrictModel):
    """One public user turn before paths are materialized for execution."""

    schema_version: Literal["x-editor.workflow-runtime-turn/v1"] = (
        "x-editor.workflow-runtime-turn/v1"
    )
    turn_index: int = Field(ge=0)
    user_request: str = Field(min_length=1)
    source_asset_refs: list[str] = Field(default_factory=list, max_length=16)
    reference_asset_refs: list[str] = Field(default_factory=list, max_length=16)
    source_text: str | None = None
    use_previous_output: bool = False

    @model_validator(mode="after")
    def validate_turn(self) -> WorkflowRuntimeTurn:
        self.source_asset_refs = _normalized_unique_refs(
            self.source_asset_refs,
            field_name="source_asset_refs",
        )
        self.reference_asset_refs = _normalized_unique_refs(
            self.reference_asset_refs,
            field_name="reference_asset_refs",
        )
        if not self.use_previous_output and not self.source_asset_refs:
            raise ValueError(
                "workflow turn requires source assets or use_previous_output"
            )
        if self.source_text is not None:
            self.source_text = self.source_text.strip() or None
        return self


class WorkflowRuntimeScenario(StrictModel):
    """A benchmark-independent sequence of one or more workflow turns."""

    schema_version: Literal["x-editor.workflow-runtime-scenario/v1"] = (
        "x-editor.workflow-runtime-scenario/v1"
    )
    audio_assets: list[WorkflowAudioAsset] = Field(min_length=1, max_length=32)
    turns: list[WorkflowRuntimeTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scenario(self) -> WorkflowRuntimeScenario:
        asset_refs = [item.asset_ref for item in self.audio_assets]
        if len(asset_refs) != len(set(asset_refs)):
            raise ValueError("workflow audio asset refs must be unique")
        asset_orders = [item.order for item in self.audio_assets]
        if len(asset_orders) != len(set(asset_orders)):
            raise ValueError("workflow audio asset order values must be unique")
        turn_indexes = [item.turn_index for item in self.turns]
        if turn_indexes != list(range(len(self.turns))):
            raise ValueError("workflow turn indexes must be contiguous from zero")
        if self.turns[0].use_previous_output:
            raise ValueError("the first workflow turn cannot use previous output")

        by_ref = {item.asset_ref: item for item in self.audio_assets}
        for turn in self.turns:
            unknown = sorted(
                (set(turn.source_asset_refs) | set(turn.reference_asset_refs))
                - set(by_ref)
            )
            if unknown:
                raise ValueError(
                    f"workflow turn references unknown audio assets: {unknown}"
                )
            wrong_sources = sorted(
                ref
                for ref in turn.source_asset_refs
                if by_ref[ref].role != "source"
            )
            wrong_references = sorted(
                ref
                for ref in turn.reference_asset_refs
                if by_ref[ref].role != "reference"
            )
            if wrong_sources:
                raise ValueError(
                    f"workflow turn source refs have the wrong role: {wrong_sources}"
                )
            if wrong_references:
                raise ValueError(
                    "workflow turn reference refs have the wrong role: "
                    f"{wrong_references}"
                )
        return self

    @property
    def fingerprint(self) -> str:
        return self.semantic_fingerprint

    @property
    def semantic_fingerprint(self) -> str:
        """Hash only runtime-visible semantics, never storage paths or dataset ids."""

        ordered_assets = sorted(self.audio_assets, key=lambda value: value.order)
        return _fingerprint(
            {
                "model_visible": self.model_visible_projection(),
                "audio_content_sha256": [
                    item.content_sha256 for item in ordered_assets
                ],
            }
        )

    def model_visible_projection(self) -> dict[str, object]:
        """Project the scenario contract that Planner/Perception may observe."""

        by_ref = {item.asset_ref: item.logical_ref for item in self.audio_assets}
        return {
            "schema_version": "x-editor.workflow-runtime-scenario-visible/v1",
            "audio_assets": [
                {
                    "audio_ref": item.logical_ref,
                    "provenance": item.provenance,
                    "role": item.role,
                    "order": item.order,
                }
                for item in sorted(self.audio_assets, key=lambda value: value.order)
            ],
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "user_request": turn.user_request,
                    "source_audio_refs": [
                        by_ref[ref] for ref in turn.source_asset_refs
                    ],
                    "reference_audio_refs": [
                        by_ref[ref] for ref in turn.reference_asset_refs
                    ],
                    "source_text": turn.source_text,
                    "use_previous_output": turn.use_previous_output,
                }
                for turn in self.turns
            ],
        }

    @property
    def materialization_fingerprint(self) -> str:
        """Hash the private scenario materialization for resume/readiness checks."""

        return _fingerprint(self.model_dump(mode="json"))

    def visible_audio_assets(self, turn_index: int) -> list[WorkflowAudioAsset]:
        """Return inputs first disclosed by this turn, preserving stable identities."""
        if not 0 <= turn_index < len(self.turns):
            raise IndexError("workflow turn index is out of range")
        disclosed = {
            ref
            for turn in self.turns[: turn_index + 1]
            for ref in (*turn.source_asset_refs, *turn.reference_asset_refs)
        }
        return sorted(
            (asset for asset in self.audio_assets if asset.asset_ref in disclosed),
            key=lambda asset: asset.order,
        )

    def materialize_turn(
        self,
        turn_index: int,
        *,
        previous_output: str | None = None,
        previous_output_ref: str | None = None,
        previous_output_path: str | None = None,
        previous_output_artifact_ref_id: str | None = None,
        previous_output_state_node_id: str | None = None,
        prior_turn_contexts: list[PriorTurnSemanticContext] | None = None,
        session_state: SessionStateContext | None = None,
        historical_reviewed_evidence: HistoricalReviewedEvidenceContext | None = None,
        future_attempt_guidance: FutureAttemptGuidance | None = None,
        planning_context: dict[str, object] | None = None,
        session_audio_bindings: list[RuntimeAudioBinding] | None = None,
    ) -> WorkflowRuntimeCase:
        turn = self.turns[turn_index]
        by_ref = {item.asset_ref: item for item in self.audio_assets}
        sources: list[str] = []
        bindings: list[RuntimeAudioBinding] = []
        if turn.use_previous_output:
            materialized_previous = previous_output_path or previous_output
            if not materialized_previous:
                raise ValueError(
                    "previous_output is required for this workflow turn"
                )
            authoritative_ref = f"aud:v1:o{turn_index - 1:04d}"
            current_binding = None
            if (
                previous_output_ref is not None
                and previous_output_ref != authoritative_ref
            ):
                matches = [
                    item for item in session_audio_bindings or []
                    if item.audio_ref == previous_output_ref
                    and item.materialized_path == str(materialized_previous)
                    and item.role in {"source", "authoritative_output"}
                ]
                if (
                    len(matches) != 1
                    or session_state is None
                    or session_state.current_audio_ref != previous_output_ref
                ):
                    raise ValueError(
                        "previous_output_ref must match the framework canonical output ref "
                        "or an explicitly bound accepted current state"
                    )
                current_binding = matches[0]
                authoritative_ref = current_binding.audio_ref
            previous_origin = (
                "session_graph_state"
                if str(previous_output_state_node_id or "").strip()
                else "artifact_store_commit"
            )
            if (
                previous_origin == "artifact_store_commit"
                and not str(previous_output_artifact_ref_id or "").strip()
                and current_binding is None
            ):
                raise ValueError(
                    "previous_output_artifact_ref_id is required for authoritative output"
                )
            sources.append(authoritative_ref)
            bindings.append(
                current_binding.model_copy(update={"order": 0}, deep=True)
                if current_binding is not None
                else RuntimeAudioBinding(
                    audio_ref=authoritative_ref,
                    materialized_path=str(materialized_previous),
                    role="authoritative_output",
                    order=0,
                    origin=previous_origin,
                    origin_artifact_ref_id=(
                        str(previous_output_artifact_ref_id).strip()
                        if previous_output_artifact_ref_id
                        else None
                    ),
                    origin_state_node_id=(
                        str(previous_output_state_node_id).strip()
                        if previous_output_state_node_id
                        else None
                    ),
                )
            )
        for asset_ref in turn.source_asset_refs:
            asset = by_ref[asset_ref]
            sources.append(asset.logical_ref)
            bindings.append(
                RuntimeAudioBinding(
                    audio_ref=asset.logical_ref,
                    materialized_path=asset.path,
                    provenance=asset.provenance,
                    role="source",
                    order=len(bindings),
                    origin="scenario_asset",
                )
            )
        references: list[str] = []
        for asset_ref in turn.reference_asset_refs:
            asset = by_ref[asset_ref]
            references.append(asset.logical_ref)
            bindings.append(
                RuntimeAudioBinding(
                    audio_ref=asset.logical_ref,
                    materialized_path=asset.path,
                    provenance=asset.provenance,
                    role="reference",
                    order=len(bindings),
                    origin="scenario_asset",
                )
            )
        bound_refs = {item.audio_ref for item in bindings}
        visible_asset_refs = {
            asset.logical_ref for asset in self.visible_audio_assets(turn_index)
        }
        for session_binding in session_audio_bindings or []:
            if (
                session_binding.origin == "scenario_asset"
                and session_binding.audio_ref not in visible_asset_refs
            ):
                raise ValueError("session audio binding precedes its first public turn")
            if session_binding.audio_ref in bound_refs:
                existing = next(
                    item
                    for item in bindings
                    if item.audio_ref == session_binding.audio_ref
                )
                if (
                    existing.materialized_path != session_binding.materialized_path
                    or existing.role != session_binding.role
                    or existing.origin != session_binding.origin
                ):
                    raise ValueError(
                        "session audio binding conflicts with materialized turn binding"
                    )
                continue
            destination = references if session_binding.role == "reference" else sources
            destination.append(session_binding.audio_ref)
            bindings.append(
                session_binding.model_copy(
                    update={"order": len(bindings)},
                    deep=True,
                )
            )
            bound_refs.add(session_binding.audio_ref)
        return WorkflowRuntimeCase(
            user_request=turn.user_request,
            source_audio_refs=list(dict.fromkeys(sources)),
            reference_audio_refs=references,
            source_text=turn.source_text,
            prior_turn_contexts=[
                item.model_copy(deep=True) for item in (prior_turn_contexts or [])
            ],
            session_state=(
                session_state.model_copy(deep=True)
                if session_state is not None
                else None
            ),
            historical_reviewed_evidence=(
                historical_reviewed_evidence.model_copy(deep=True)
                if historical_reviewed_evidence is not None
                else None
            ),
            future_attempt_guidance=(
                future_attempt_guidance.model_copy(deep=True)
                if future_attempt_guidance is not None
                else None
            ),
            planning_context=dict(planning_context or {}),
            audio_resolution_mode="opaque_ref_v1",
            audio_bindings=bindings,
        )


def _normalized_unique_refs(values: list[str], *, field_name: str) -> list[str]:
    normalized = [str(item).strip() for item in values]
    if any(not item for item in normalized):
        raise ValueError(f"{field_name} cannot contain blank values")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be unique")
    return normalized


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AudioAssetProvenance",
    "AudioEvidenceRequirement",
    "AudioUsageKind",
    "HistoricalReviewedEvidenceContext",
    "PriorTurnSemanticContext",
    "RuntimeAudioBinding",
    "WorkflowAudioAsset",
    "WorkflowAudioEvidenceReference",
    "WorkflowAudioUsageBinding",
    "WorkflowRuntimeCase",
    "WorkflowRuntimeScenario",
    "WorkflowRuntimeTurn",
]
