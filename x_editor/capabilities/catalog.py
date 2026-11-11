"""Discovery and Planner projection for operation capability cards."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from x_editor.capabilities.priors import FrozenToolPriorSnapshot
from x_editor.capabilities.schemas import OperationCapabilityCard

PRIOR_GUIDANCE_PAYLOAD_KEY = "tool_prior_guidance"


class OperationCapabilityCatalog:
    def __init__(self, cards: list[OperationCapabilityCard]) -> None:
        by_operation: dict[str, OperationCapabilityCard] = {}
        for card in cards:
            if card.operation in by_operation:
                raise ValueError(f"duplicate operation capability card: {card.operation}")
            by_operation[card.operation] = card
        self._cards = by_operation

    def get(self, operation: str) -> OperationCapabilityCard | None:
        return self._cards.get(operation)

    def operations(self) -> list[str]:
        return sorted(self._cards)

    def planner_payload(self, operations: list[str] | set[str]) -> list[dict[str, Any]]:
        return [
            self._cards[operation].planner_payload()
            for operation in sorted(set(operations))
            if operation in self._cards
        ]


@lru_cache(maxsize=1)
def load_operation_capability_catalog() -> OperationCapabilityCatalog:
    root = Path(__file__).with_name("cards")
    cards: list[OperationCapabilityCard] = []
    if root.exists():
        for path in sorted(root.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            try:
                card = OperationCapabilityCard.model_validate(raw)
            except Exception as exc:
                raise ValueError(f"invalid operation capability card {path}: {exc}") from exc
            cards.append(card)
    return OperationCapabilityCatalog(cards)


def enrich_operation_descriptions(
    operations: list[dict[str, Any]],
    *,
    catalog: OperationCapabilityCatalog | None = None,
) -> list[dict[str, Any]]:
    """Attach one canonical card without altering registry operation metadata."""

    catalog = catalog or load_operation_capability_catalog()
    enriched: list[dict[str, Any]] = []
    for raw in operations:
        item = copy.deepcopy(raw)
        operation = str(item.get("operation") or "")
        card = catalog.get(operation)
        if card is not None:
            item["operation_capability"] = card.planner_payload()
        enriched.append(item)
    return enriched


def attach_tool_prior_guidance(
    planner_payload: Mapping[str, Any],
    *,
    frozen_prior: FrozenToolPriorSnapshot | None = None,
) -> dict[str, Any]:
    """Attach an explicitly frozen development prior without selecting a route."""

    payload = copy.deepcopy(dict(planner_payload))
    if frozen_prior is None:
        return payload
    if PRIOR_GUIDANCE_PAYLOAD_KEY in payload:
        raise ValueError(f"planner payload already contains {PRIOR_GUIDANCE_PAYLOAD_KEY!r}")

    snapshot = frozen_prior.snapshot
    guidance_payload: list[dict[str, Any]] = []
    for item in snapshot.guidance:
        routes_by_id = {
            route.route_id: {"operations": list(route.operations)}
            for route in sorted(item.routes, key=lambda route: route.route_id)
        }
        projected: dict[str, Any] = {
            "guidance_id": item.guidance_id,
            "comparison_group": item.comparison_group,
            "task_scope": item.task_scope,
            "decision": item.decision,
            "ordering_semantics": item.ordering_semantics,
            "routes_by_id": routes_by_id,
            "decision_reason": item.decision_reason,
        }
        if item.decision == "prefer":
            preferred = next(
                route for route in item.routes if route.route_id == item.preferred_route_id
            )
            projected["preferred_route_id"] = item.preferred_route_id
            projected["preferred_operations"] = list(preferred.operations)
            projected["development_evidence"] = [
                evidence.model_dump(mode="json") for evidence in item.pairwise_evidence
            ]
        else:
            projected["preferred_route_id"] = None
            projected["prohibition"] = "do_not_rank_or_apply_an_implicit_tie_break"
        guidance_payload.append(projected)

    payload[PRIOR_GUIDANCE_PAYLOAD_KEY] = {
        "schema_version": "x-editor.tool-prior-guidance/v1",
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_version": snapshot.snapshot_version,
        "snapshot_fingerprint": frozen_prior.binding.snapshot_fingerprint,
        "source_role": snapshot.source.evaluator_role,
        "source_data_role": snapshot.source.data_role,
        "eligible_as_final_outcome": False,
        "guidance_scope": snapshot.usage_policy.guidance_scope,
        "guidance": guidance_payload,
    }
    return payload
