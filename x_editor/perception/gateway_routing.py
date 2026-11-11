"""Gateway-owned routing from observation needs to Perception capabilities.

Reasoners and Planners describe the object and evidence they need.  This module
selects a registered capability without inspecting user wording, benchmark
identity, provider names, or evaluator data.  Unresolved fields remain visible
and never become a protocol exception by themselves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from x_editor.perception.workflow_catalog import (
    WorkflowPerceptionRoutePart,
    resolve_workflow_perception_capabilities,
    uniquely_decompose_workflow_perception_requirements,
    workflow_perception_capabilities,
    workflow_perception_capability,
)

GatewayRouteStatus = Literal["not_required", "routed", "partial", "unresolved"]

@dataclass(frozen=True)
class GatewayPerceptionRouteDecision:
    """A route decision plus needs the current catalog cannot yet satisfy."""

    status: GatewayRouteStatus
    route_parts: tuple[WorkflowPerceptionRoutePart, ...] = ()
    unsupported_evidence_fields: tuple[str, ...] = ()
    alternative_operations: tuple[str, ...] = ()
    alternative_decompositions: tuple[tuple[WorkflowPerceptionRoutePart, ...], ...] = ()
    warnings: tuple[str, ...] = ()
    selected_operation_hint: str | None = None
    hint_used: bool = False

    @property
    def blocks_workflow(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "x-editor.perception-gateway-route/v1",
            "status": self.status,
            "route_parts": [
                {
                    "operation": part.operation,
                    "expected_evidence_fields": list(part.expected_evidence_fields),
                    "target_kinds": list(part.target_kinds),
                }
                for part in self.route_parts
            ],
            "unsupported_evidence_fields": list(self.unsupported_evidence_fields),
            "alternative_operations": list(self.alternative_operations),
            "alternative_decompositions": [
                [
                    {
                        "operation": part.operation,
                        "expected_evidence_fields": list(part.expected_evidence_fields),
                        "target_kinds": list(part.target_kinds),
                    }
                    for part in parts
                ]
                for parts in self.alternative_decompositions
            ],
            "warnings": list(self.warnings),
            "selected_operation_hint": self.selected_operation_hint,
            "hint_used": self.hint_used,
            "blocks_workflow": False,
        }


def route_perception_observation_need(
    *,
    target_kinds: Iterable[str] = (),
    expected_evidence_fields: Iterable[str] = (),
    temporal_precision: str = "coarse",
    entity_kind: str | None = None,
    selected_operation_hint: str | None = None,
    available_operations: Iterable[str] | None = None,
) -> GatewayPerceptionRouteDecision:
    """Route one typed observation need without trusting a model-selected route."""

    kinds = tuple(dict.fromkeys(_clean_values(target_kinds)))
    fields = tuple(dict.fromkeys(_clean_values(expected_evidence_fields)))
    entity = str(entity_kind or "").strip().lower()
    hint = str(selected_operation_hint or "").strip() or None
    available = tuple(
        capability.operation
        for capability in workflow_perception_capabilities(available_operations)
    )

    if entity == "session_state":
        return GatewayPerceptionRouteDecision(
            status="not_required",
            selected_operation_hint=hint,
            warnings=("session_state_is_resolved_by_the_runtime_graph",),
        )

    resolution = resolve_workflow_perception_capabilities(
        target_kinds=kinds,
        expected_evidence_fields=fields,
        temporal_precision=temporal_precision,
        available_operations=available,
        allow_unique_evidence_completion=True,
    )
    if (
        fields
        and not resolution.recognized_evidence_fields
        and not kinds
        and not entity
    ):
        return GatewayPerceptionRouteDecision(
            status="unresolved",
            unsupported_evidence_fields=resolution.unsupported_evidence_fields,
            warnings=("no_registered_capability_produces_the_requested_evidence_fields",),
            selected_operation_hint=hint,
        )
    compatible = tuple(resolution.compatible_operations)
    warnings: list[str] = []
    if hint is not None and hint not in available:
        warnings.append("selected_operation_hint_is_not_available")
    elif hint is not None and compatible and hint not in compatible:
        warnings.append("selected_operation_hint_is_incompatible_and_was_ignored")

    selected = _select_single_route(
        compatible,
        entity_kind=entity,
        selected_operation_hint=hint,
        expected_evidence_fields=fields,
        target_kinds=kinds,
    )
    if selected is not None:
        status: GatewayRouteStatus = (
            "partial" if resolution.unsupported_evidence_fields else "routed"
        )
        return GatewayPerceptionRouteDecision(
            status=status,
            route_parts=(
                WorkflowPerceptionRoutePart(
                    operation=selected,
                    expected_evidence_fields=fields,
                    target_kinds=kinds,
                ),
            ),
            unsupported_evidence_fields=resolution.unsupported_evidence_fields,
            alternative_operations=tuple(item for item in compatible if item != selected),
            warnings=tuple(warnings),
            selected_operation_hint=hint,
            hint_used=hint == selected,
        )

    decompositions = _candidate_decompositions(
        available_operations=available,
        target_kinds=kinds,
        expected_evidence_fields=fields,
        temporal_precision=temporal_precision,
    )
    decomposition = _select_decomposition(
        decompositions,
        entity_kind=entity,
        selected_operation_hint=hint,
    )
    if decomposition:
        return GatewayPerceptionRouteDecision(
            status="partial" if resolution.unsupported_evidence_fields else "routed",
            route_parts=decomposition,
            unsupported_evidence_fields=resolution.unsupported_evidence_fields,
            alternative_operations=compatible,
            warnings=tuple(warnings),
            selected_operation_hint=hint,
            hint_used=bool(hint and any(part.operation == hint for part in decomposition)),
        )

    entity_default = _entity_default_route(
        entity_kind=entity,
        available_operations=available,
        temporal_precision=temporal_precision,
    )
    recognized_fields = tuple(resolution.recognized_evidence_fields)
    if entity_default is not None and not recognized_fields and not fields:
        return GatewayPerceptionRouteDecision(
            status="routed",
            route_parts=(
                WorkflowPerceptionRoutePart(
                    operation=entity_default,
                    expected_evidence_fields=(),
                    target_kinds=kinds,
                ),
            ),
            alternative_operations=compatible,
            warnings=tuple(warnings),
            selected_operation_hint=hint,
            hint_used=hint == entity_default,
        )

    if not kinds and not fields and not entity:
        warnings.append("observation_need_has_no_typed_object_or_evidence_field")
    elif fields and not recognized_fields:
        warnings.append("no_registered_capability_produces_the_requested_evidence_fields")
    else:
        warnings.append("observation_need_requires_clarification_or_catalog_extension")
    return GatewayPerceptionRouteDecision(
        status="unresolved",
        unsupported_evidence_fields=resolution.unsupported_evidence_fields,
        alternative_operations=compatible,
        alternative_decompositions=decompositions,
        warnings=tuple(warnings),
        selected_operation_hint=hint,
    )


def _select_single_route(
    operations: tuple[str, ...],
    *,
    entity_kind: str,
    selected_operation_hint: str | None,
    expected_evidence_fields: tuple[str, ...],
    target_kinds: tuple[str, ...],
) -> str | None:
    if not operations:
        return None
    if len(operations) == 1:
        return operations[0]
    scores = {
        operation: _single_route_score(
            operation,
            entity_kind=entity_kind,
            expected_evidence_fields=expected_evidence_fields,
            target_kinds=target_kinds,
        )
        for operation in operations
    }
    best_score = min(scores.values())
    best = [operation for operation, score in scores.items() if score == best_score]
    if len(best) == 1:
        return best[0]
    if selected_operation_hint in best:
        return selected_operation_hint
    return None


def _single_route_score(
    operation: str,
    *,
    entity_kind: str,
    expected_evidence_fields: tuple[str, ...],
    target_kinds: tuple[str, ...],
) -> tuple[int, int, int, int, int]:
    capability = workflow_perception_capability(operation)
    if capability is None:
        return (999, 999, 999, 999, 999)
    entity_rank = 0 if entity_kind and entity_kind in capability.entity_kinds else 1
    output_role_rank = 2 if capability.output_role == "candidate_prior" else 0
    requested_evidence = {_evidence_root(field) for field in expected_evidence_fields}
    requested_targets = set(target_kinds)
    extra_evidence = len(set(capability.evidence_types) - requested_evidence)
    extra_targets = len(set(capability.target_kinds) - requested_targets)
    return (
        entity_rank,
        output_role_rank,
        extra_evidence,
        extra_targets,
        capability.relative_cost,
    )


def _candidate_decompositions(
    *,
    available_operations: tuple[str, ...],
    target_kinds: tuple[str, ...],
    expected_evidence_fields: tuple[str, ...],
    temporal_precision: str,
) -> tuple[tuple[WorkflowPerceptionRoutePart, ...], ...]:
    unique: dict[tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...], Any] = {}
    for operation in available_operations:
        parts = uniquely_decompose_workflow_perception_requirements(
            selected_operation=operation,
            target_kinds=target_kinds,
            expected_evidence_fields=expected_evidence_fields,
            temporal_precision=temporal_precision,
            available_operations=available_operations,
        )
        if not parts:
            continue
        key = tuple(
            sorted(
                (
                    part.operation,
                    tuple(part.expected_evidence_fields),
                    tuple(part.target_kinds),
                )
                for part in parts
            )
        )
        unique[key] = parts
    return tuple(unique.values())


def _select_decomposition(
    decompositions: tuple[tuple[WorkflowPerceptionRoutePart, ...], ...],
    *,
    entity_kind: str,
    selected_operation_hint: str | None,
) -> tuple[WorkflowPerceptionRoutePart, ...]:
    if not decompositions:
        return ()
    scores: dict[tuple[WorkflowPerceptionRoutePart, ...], tuple[int, int, int]] = {}
    for parts in decompositions:
        capabilities = [
            workflow_perception_capability(part.operation) for part in parts
        ]
        entity_penalty = min(
            (
                0
                for capability in capabilities
                if capability is not None and entity_kind in capability.entity_kinds
            ),
            default=1,
        )
        prior_penalty = sum(
            2
            for capability in capabilities
            if capability is not None and capability.output_role == "candidate_prior"
        )
        scores[parts] = (
            len(parts),
            entity_penalty + prior_penalty,
            sum(
                capability.relative_cost
                for capability in capabilities
                if capability is not None
            )
            + 100 * sum(capability is None for capability in capabilities),
        )
    best_score = min(scores.values())
    best = [parts for parts, score in scores.items() if score == best_score]
    if len(best) == 1:
        return best[0]
    hinted = [
        parts
        for parts in best
        if selected_operation_hint
        and any(part.operation == selected_operation_hint for part in parts)
    ]
    return hinted[0] if len(hinted) == 1 else ()


def _entity_default_route(
    *,
    entity_kind: str,
    available_operations: tuple[str, ...],
    temporal_precision: str,
) -> str | None:
    candidates = [
        capability
        for capability in workflow_perception_capabilities(available_operations)
        if entity_kind in capability.entity_kinds
        and temporal_precision in capability.temporal_precision
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda capability: (
            capability.output_role == "candidate_prior",
            capability.relative_cost,
        )
    )
    best_key = (
        candidates[0].output_role == "candidate_prior",
        candidates[0].relative_cost,
    )
    best = [
        capability
        for capability in candidates
        if (
            capability.output_role == "candidate_prior",
            capability.relative_cost,
        )
        == best_key
    ]
    return best[0].operation if len(best) == 1 else None


def _clean_values(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _evidence_root(field_path: str) -> str:
    normalized = str(field_path).strip()
    for prefix in ("facts.", "perception_report."):
        normalized = normalized.removeprefix(prefix)
    return normalized.split(".", 1)[0].split("[", 1)[0]
