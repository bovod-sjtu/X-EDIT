"""Versioned operation effects used only to decide rubric applicability."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

EvaluationProperty = Literal[
    "wording",
    "language",
    "speaker_identity",
    "vocal_timbre",
    "acoustic_properties",
    "duration",
    "local_timing",
]
EffectDisposition = Literal["changed", "preserved", "unknown"]
OperationTemporalScope = Literal[
    "global",
    "local",
    "local_or_global",
    "structural",
    "unknown",
]
EvaluationTemporalScope = Literal[
    "global",
    "local",
    "local_or_global",
    "structural",
    "mixed",
    "unknown",
]
LocalIntervalContract = Literal[
    "none",
    "source_target_spans",
    "aligned_edit_span_pairs",
    "pause_insertion_plan",
    "whole_source_text_coverage",
    "selected_spans_extraction",
    "selected_spans_removal",
]

_PROPERTIES: tuple[EvaluationProperty, ...] = (
    "wording",
    "language",
    "speaker_identity",
    "vocal_timbre",
    "acoustic_properties",
    "duration",
    "local_timing",
)


class OperationEvaluationEffectProfile(StrictModel):
    operation: str = Field(min_length=1)
    changed: list[EvaluationProperty] = Field(default_factory=list)
    preserved: list[EvaluationProperty] = Field(default_factory=list)
    unknown: list[EvaluationProperty] = Field(default_factory=list)
    temporal_scope: OperationTemporalScope = "unknown"
    local_interval_contract: LocalIntervalContract = "none"
    identity_outside_local_intervals: bool = False

    @model_validator(mode="after")
    def validate_partition(self) -> OperationEvaluationEffectProfile:
        groups = [self.changed, self.preserved, self.unknown]
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("effect profile property lists must be unique")
        flattened = [item for group in groups for item in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("effect profile property lists must be disjoint")
        if set(flattened) != set(_PROPERTIES):
            raise ValueError("effect profile must classify every evaluation property")
        if self.local_interval_contract != "none" and self.temporal_scope not in {
            "local",
            "local_or_global",
            "structural",
        }:
            raise ValueError(
                "result interval contracts require a local-capable or structural scope"
            )
        if (
            self.identity_outside_local_intervals
            and self.local_interval_contract == "none"
        ):
            raise ValueError(
                "identity outside local intervals requires a typed interval contract"
            )
        return self

    def disposition(self, property_name: EvaluationProperty) -> EffectDisposition:
        if property_name in self.changed:
            return "changed"
        if property_name in self.preserved:
            return "preserved"
        return "unknown"


class OperationEvaluationEffectCatalog(StrictModel):
    schema_version: Literal[
        "x-editor.operation-evaluation-effects/v5",
        "x-editor.operation-evaluation-effects/v6",
    ] = (
        "x-editor.operation-evaluation-effects/v6"
    )
    profiles: list[OperationEvaluationEffectProfile]

    @model_validator(mode="after")
    def validate_operations(self) -> OperationEvaluationEffectCatalog:
        operations = [item.operation for item in self.profiles]
        if len(operations) != len(set(operations)):
            raise ValueError("operation evaluation effect profiles must be unique")
        return self

    def by_operation(self) -> dict[str, OperationEvaluationEffectProfile]:
        return {item.operation: item for item in self.profiles}


class OperationEffectChain(StrictModel):
    operations: list[str] = Field(min_length=1)
    dispositions: dict[EvaluationProperty, EffectDisposition]
    temporal_scope: EvaluationTemporalScope
    missing_operations: list[str] = Field(default_factory=list)


class FrozenObjectiveEffect(StrictModel):
    """Optional effect semantics frozen with the user objective before scoring."""

    objective_ref_id: str = Field(min_length=1)
    target_ref_id: str | None = None
    changed: list[EvaluationProperty] = Field(default_factory=list)
    preserved: list[EvaluationProperty] = Field(default_factory=list)
    unknown: list[EvaluationProperty] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_effects(self) -> FrozenObjectiveEffect:
        groups = [self.changed, self.preserved, self.unknown]
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("objective effect property lists must be unique")
        flattened = [item for group in groups for item in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("objective effect property lists must be disjoint")
        return self

    def explicit_dispositions(self) -> dict[EvaluationProperty, EffectDisposition]:
        return {
            **{item: "changed" for item in self.changed},
            **{item: "preserved" for item in self.preserved},
            **{item: "unknown" for item in self.unknown},
        }


class EvaluationEffectContract(StrictModel):
    """Auditable combination of objective intent, operations, and final graph state."""

    schema_version: Literal["x-editor.evaluation-effect-contract/v2"] = (
        "x-editor.evaluation-effect-contract/v2"
    )
    target_ref_id: str | None = None
    final_state_node_id: str | None = None
    operation_chain: OperationEffectChain | None = None
    objective_effects: list[FrozenObjectiveEffect] = Field(default_factory=list)
    dispositions: dict[EvaluationProperty, EffectDisposition]
    temporal_scope: EvaluationTemporalScope = "unknown"
    conflicts: list[str] = Field(default_factory=list)
    authority_order: list[str] = Field(
        default_factory=lambda: [
            "frozen_objective_effect",
            "admitted_executed_operation_effect",
            "unknown",
        ]
    )


@lru_cache(maxsize=1)
def load_operation_evaluation_effect_catalog() -> OperationEvaluationEffectCatalog:
    path = Path(__file__).with_name("operation_effect_profiles.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OperationEvaluationEffectCatalog.model_validate(raw)


def resolve_operation_effect_chain(
    operations: list[str],
    *,
    catalog: OperationEvaluationEffectCatalog | None = None,
) -> OperationEffectChain:
    """Compose exact operation profiles without inspecting instruction text."""

    if not operations:
        raise ValueError("operation effect chain requires at least one operation")
    catalog = catalog or load_operation_evaluation_effect_catalog()
    profiles = catalog.by_operation()
    missing = sorted({operation for operation in operations if operation not in profiles})
    dispositions: dict[EvaluationProperty, EffectDisposition] = {}
    for property_name in _PROPERTIES:
        values = [
            profiles[operation].disposition(property_name)
            for operation in operations
            if operation in profiles
        ]
        if any(value == "changed" for value in values):
            disposition: EffectDisposition = "changed"
        elif missing or any(value == "unknown" for value in values) or not values:
            disposition = "unknown"
        elif all(value == "preserved" for value in values):
            disposition = "preserved"
        else:
            disposition = "unknown"
        dispositions[property_name] = disposition
    temporal_scope = _compose_temporal_scope(
        [profiles[operation].temporal_scope for operation in operations if operation in profiles],
        has_missing=bool(missing),
    )
    return OperationEffectChain(
        operations=list(operations),
        dispositions=dispositions,
        temporal_scope=temporal_scope,
        missing_operations=missing,
    )


def _compose_temporal_scope(
    scopes: list[OperationTemporalScope],
    *,
    has_missing: bool,
) -> EvaluationTemporalScope:
    if has_missing or not scopes or "unknown" in scopes:
        return "unknown"
    unique = set(scopes)
    if len(unique) == 1:
        return scopes[0]
    if unique <= {"local", "local_or_global"}:
        return "local_or_global"
    if unique <= {"global", "local_or_global"}:
        return "local_or_global"
    if "structural" in unique:
        return "structural"
    return "mixed"


def resolve_evaluation_effect_contract(
    operations: list[str],
    *,
    objective_effects: list[FrozenObjectiveEffect] | None = None,
    target_ref_id: str | None = None,
    final_state_node_id: str | None = None,
    catalog: OperationEvaluationEffectCatalog | None = None,
) -> EvaluationEffectContract:
    """Resolve rubric semantics without parsing or rewriting instruction text."""

    operation_chain = (
        resolve_operation_effect_chain(operations, catalog=catalog) if operations else None
    )
    relevant_objectives = [
        item
        for item in objective_effects or []
        if item.target_ref_id is None or item.target_ref_id == target_ref_id
    ]
    dispositions: dict[EvaluationProperty, EffectDisposition] = {}
    conflicts: list[str] = []
    for property_name in _PROPERTIES:
        explicit = {
            value
            for item in relevant_objectives
            if (value := item.explicit_dispositions().get(property_name)) is not None
        }
        if len(explicit) > 1:
            dispositions[property_name] = "unknown"
            conflicts.append(
                f"frozen objectives disagree about {property_name}: {sorted(explicit)!r}"
            )
            continue
        operation_value = (
            operation_chain.dispositions[property_name]
            if operation_chain is not None
            else "unknown"
        )
        if explicit:
            objective_value = next(iter(explicit))
            dispositions[property_name] = objective_value
            if (
                objective_value != "unknown"
                and operation_value != "unknown"
                and objective_value != operation_value
            ):
                conflicts.append(
                    f"frozen objective expects {property_name}={objective_value} but "
                    f"the active operation chain declares {operation_value}"
                )
        else:
            dispositions[property_name] = operation_value
    return EvaluationEffectContract(
        target_ref_id=target_ref_id,
        final_state_node_id=final_state_node_id,
        operation_chain=operation_chain,
        objective_effects=relevant_objectives,
        dispositions=dispositions,
        temporal_scope=(
            operation_chain.temporal_scope
            if operation_chain is not None
            else "unknown"
        ),
        conflicts=conflicts,
    )


__all__ = [
    "EffectDisposition",
    "EvaluationEffectContract",
    "EvaluationProperty",
    "EvaluationTemporalScope",
    "FrozenObjectiveEffect",
    "LocalIntervalContract",
    "OperationEffectChain",
    "OperationEvaluationEffectCatalog",
    "OperationEvaluationEffectProfile",
    "OperationTemporalScope",
    "load_operation_evaluation_effect_catalog",
    "resolve_evaluation_effect_contract",
    "resolve_operation_effect_chain",
]
