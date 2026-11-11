"""Auditable projection of the final active branch in an edit session graph."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import Field

from x_editor.core.schemas import StrictModel


class ActiveSessionState(StrictModel):
    node_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    parent_node_id: str | None = None
    primary_audio_path: str | None = None
    audio_ref: str | None = None
    operation_summary: str | None = None


class ExcludedSessionState(StrictModel):
    node_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    primary_audio_path: str | None = None
    operation_summary: str | None = None
    reason: Literal["not_on_final_active_branch"] = "not_on_final_active_branch"


class FinalStateProjection(StrictModel):
    """Graph-authored facts about the state that produced the final delivery."""

    schema_version: Literal["x-editor.final-state-projection/v1"] = (
        "x-editor.final-state-projection/v1"
    )
    graph_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_node_id: str = Field(min_length=1)
    root_node_ids: list[str] = Field(min_length=1)
    active_states: list[ActiveSessionState] = Field(min_length=1)
    final_audio_path: str | None = None
    final_audio_ref: str | None = None
    final_selected_base_node_id: str | None = None
    final_round_entry_node_id: str | None = None
    final_operation_type: str | None = None
    final_operations: list[str] = Field(default_factory=list)
    excluded_round_states: list[ExcludedSessionState] = Field(default_factory=list)


def project_final_session_state(
    graph: Any,
    *,
    graph_path: str | Path | None = None,
) -> FinalStateProjection:
    """Follow graph node IDs, never audio-path heuristics or model claims."""

    current_node_id = str(graph.current_node_id or "").strip()
    if not current_node_id or current_node_id not in graph.nodes:
        raise ValueError("session graph has no valid current node")
    root_ids = list(graph.root_node_ids)
    if not root_ids or any(node_id not in graph.nodes for node_id in root_ids):
        raise ValueError("session graph has invalid root nodes")

    active_reversed: list[ActiveSessionState] = []
    active_ids: set[str] = set()
    node_id = current_node_id
    while True:
        if node_id in active_ids:
            raise ValueError("session graph active branch contains a cycle")
        node = graph.nodes[node_id]
        active_ids.add(node_id)
        parent_id = _selected_parent_id(node_id=node_id, graph=graph)
        active_reversed.append(
            ActiveSessionState(
                node_id=node.node_id,
                round_index=node.round_index,
                parent_node_id=parent_id,
                primary_audio_path=node.primary_audio_path,
                audio_ref=_optional_text(node.metadata.get("audio_ref")),
                operation_summary=node.operation_summary,
            )
        )
        if parent_id is None:
            if node_id not in root_ids:
                raise ValueError("session graph active branch does not terminate at a root")
            break
        node_id = parent_id

    active_states = list(reversed(active_reversed))
    final_record = next(
        (
            record
            for record in reversed(graph.round_history)
            if record.status == "completed" and record.result_node_id == current_node_id
        ),
        None,
    )
    final_edge = (
        graph.edges.get(str(final_record.edge_id or ""))
        if final_record is not None
        else None
    )
    excluded = [
        ExcludedSessionState(
            node_id=node.node_id,
            round_index=node.round_index,
            primary_audio_path=node.primary_audio_path,
            operation_summary=node.operation_summary,
        )
        for node in graph.nodes.values()
        if node.is_round_result
        and node.kind != "initial"
        and node.node_id not in active_ids
    ]
    excluded.sort(key=lambda item: (item.round_index, item.node_id))
    current = graph.nodes[current_node_id]
    return FinalStateProjection(
        graph_sha256=_sha256_file(Path(graph_path)) if graph_path is not None else None,
        current_node_id=current_node_id,
        root_node_ids=root_ids,
        active_states=active_states,
        final_audio_path=current.primary_audio_path,
        final_audio_ref=_optional_text(current.metadata.get("audio_ref")),
        final_selected_base_node_id=(
            _optional_text(final_record.input_node_id) if final_record is not None else None
        ),
        final_round_entry_node_id=(
            _optional_text(final_record.metadata.get("round_entry_node_id"))
            if final_record is not None
            else None
        ),
        final_operation_type=final_edge.operation_type if final_edge is not None else None,
        final_operations=list(final_edge.operations) if final_edge is not None else [],
        excluded_round_states=excluded,
    )


def project_final_session_payload(
    payload: dict[str, Any],
    *,
    graph_path: str | Path | None = None,
) -> FinalStateProjection:
    """Project the serialized graph without importing runtime control code."""

    if payload.get("schema_version") != "x-editor.edit_session_graph.v1":
        raise ValueError("unsupported edit session graph schema")
    nodes_raw = payload.get("nodes")
    edges_raw = payload.get("edges")
    rounds_raw = payload.get("round_history")
    if not isinstance(nodes_raw, dict) or not isinstance(edges_raw, dict):
        raise TypeError("serialized session graph has invalid node or edge maps")
    if not isinstance(rounds_raw, list):
        raise TypeError("serialized session graph has invalid round history")
    graph = SimpleNamespace(
        current_node_id=payload.get("current_node_id"),
        root_node_ids=list(payload.get("root_node_ids") or []),
        nodes={
            str(node_id): SimpleNamespace(
                node_id=str(node.get("node_id") or node_id),
                round_index=int(node.get("round_index") or 0),
                kind=str(node.get("kind") or ""),
                primary_audio_path=node.get("primary_audio_path"),
                parent_node_ids=list(node.get("parent_node_ids") or []),
                operation_summary=node.get("operation_summary"),
                is_round_result=bool(node.get("is_round_result")),
                metadata=dict(node.get("metadata") or {}),
            )
            for node_id, node in nodes_raw.items()
            if isinstance(node, dict)
        },
        edges={
            str(edge_id): SimpleNamespace(
                operation_type=str(edge.get("operation_type") or ""),
                operations=list(edge.get("operations") or []),
            )
            for edge_id, edge in edges_raw.items()
            if isinstance(edge, dict)
        },
        round_history=[
            SimpleNamespace(
                status=str(item.get("status") or ""),
                result_node_id=item.get("result_node_id"),
                edge_id=item.get("edge_id"),
                input_node_id=item.get("input_node_id"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in rounds_raw
            if isinstance(item, dict)
        ],
    )
    if len(graph.nodes) != len(nodes_raw) or len(graph.edges) != len(edges_raw):
        raise ValueError("serialized session graph contains non-object nodes or edges")
    return project_final_session_state(graph, graph_path=graph_path)


def _selected_parent_id(*, node_id: str, graph: Any) -> str | None:
    node = graph.nodes[node_id]
    parents = list(node.parent_node_ids)
    selected = _optional_text(node.metadata.get("selected_source_node_id"))
    if selected is not None:
        if selected not in parents:
            raise ValueError(
                f"session graph node {node_id!r} selected source is not a parent"
            )
        return selected
    if not parents:
        return None
    if len(parents) != 1:
        raise ValueError(f"session graph node {node_id!r} has ambiguous parents")
    if parents[0] not in graph.nodes:
        raise ValueError(f"session graph node {node_id!r} references a missing parent")
    return parents[0]


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"session graph is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ActiveSessionState",
    "ExcludedSessionState",
    "FinalStateProjection",
    "project_final_session_payload",
    "project_final_session_state",
]
