"""Session-level directed graph for multi-round audio edits."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from x_editor.reasoner.contracts import (
    SessionStateAudioSlotContext,
    SessionStateContext,
    SessionStateNodeContext,
    SessionStateTransitionContext,
)

SCHEMA_VERSION = "x-editor.edit_session_graph.v1"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class SessionAudioAsset:
    """One audio path attached to a session graph node."""

    asset_id: str
    path: str
    role: str
    label: str | None = None
    audio_id: str | None = None
    source: str | None = None
    round_index: int | None = None
    step_id: str | None = None
    operation: str | None = None
    span: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditSessionNode:
    """A concrete audio version in a session."""

    node_id: str
    round_index: int
    created_at: str
    kind: str
    audio_paths: list[SessionAudioAsset] = field(default_factory=list)
    primary_audio_path: str | None = None
    parent_node_ids: list[str] = field(default_factory=list)
    incoming_edge_ids: list[str] = field(default_factory=list)
    outgoing_edge_ids: list[str] = field(default_factory=list)
    edit_count: int = 0
    operation_summary: str | None = None
    operation_history: list[dict[str, Any]] = field(default_factory=list)
    stage: str | None = None
    is_round_result: bool = False
    rollback_stack: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditSessionEdge:
    """A directed edit operation from one or more parent nodes to a target node."""

    edge_id: str
    round_index: int
    created_at: str
    source_node_ids: list[str]
    target_node_id: str
    operation_type: str
    instruction: str
    status: str
    plan_step_ids: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    input_audio_paths: list[str] = field(default_factory=list)
    output_audio_paths: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditRoundRecord:
    """One user-facing edit turn."""

    round_index: int
    request: str
    status: str
    started_at: str
    input_node_id: str | None = None
    result_node_id: str | None = None
    edge_id: str | None = None
    ended_at: str | None = None
    evidence_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditSessionGraph:
    """Directed graph that preserves audio lineage within one editing session."""

    session_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    round_index: int = 0
    current_node_id: str | None = None
    root_node_ids: list[str] = field(default_factory=list)
    nodes: dict[str, EditSessionNode] = field(default_factory=dict)
    edges: dict[str, EditSessionEdge] = field(default_factory=dict)
    round_history: list[EditRoundRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, session_id: str | None = None) -> "EditSessionGraph":
        return cls(session_id=session_id or uuid.uuid4().hex[:12])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditSessionGraph":
        graph = cls(
            session_id=str(payload.get("session_id") or uuid.uuid4().hex[:12]),
            schema_version=str(payload.get("schema_version") or SCHEMA_VERSION),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
            round_index=int(payload.get("round_index") or 0),
            current_node_id=payload.get("current_node_id"),
            root_node_ids=[str(item) for item in payload.get("root_node_ids") or []],
            metadata=dict(payload.get("metadata") or {}),
        )
        for node_id, raw_node in (payload.get("nodes") or {}).items():
            if not isinstance(raw_node, dict):
                continue
            assets = [
                SessionAudioAsset(**item)
                for item in raw_node.get("audio_paths") or []
                if isinstance(item, dict)
            ]
            graph.nodes[str(node_id)] = EditSessionNode(
                node_id=str(raw_node.get("node_id") or node_id),
                round_index=int(raw_node.get("round_index") or 0),
                created_at=str(raw_node.get("created_at") or _now()),
                kind=str(raw_node.get("kind") or "edit_result"),
                audio_paths=assets,
                primary_audio_path=raw_node.get("primary_audio_path"),
                parent_node_ids=[str(item) for item in raw_node.get("parent_node_ids") or []],
                incoming_edge_ids=[str(item) for item in raw_node.get("incoming_edge_ids") or []],
                outgoing_edge_ids=[str(item) for item in raw_node.get("outgoing_edge_ids") or []],
                edit_count=int(raw_node.get("edit_count") or 0),
                operation_summary=raw_node.get("operation_summary"),
                operation_history=[dict(item) for item in raw_node.get("operation_history") or [] if isinstance(item, dict)],
                stage=raw_node.get("stage"),
                is_round_result=bool(raw_node.get("is_round_result")),
                rollback_stack=[dict(item) for item in raw_node.get("rollback_stack") or [] if isinstance(item, dict)],
                metadata=dict(raw_node.get("metadata") or {}),
            )
        for edge_id, raw_edge in (payload.get("edges") or {}).items():
            if not isinstance(raw_edge, dict):
                continue
            graph.edges[str(edge_id)] = EditSessionEdge(
                edge_id=str(raw_edge.get("edge_id") or edge_id),
                round_index=int(raw_edge.get("round_index") or 0),
                created_at=str(raw_edge.get("created_at") or _now()),
                source_node_ids=[str(item) for item in raw_edge.get("source_node_ids") or []],
                target_node_id=str(raw_edge.get("target_node_id") or ""),
                operation_type=str(raw_edge.get("operation_type") or "edit"),
                instruction=str(raw_edge.get("instruction") or ""),
                status=str(raw_edge.get("status") or "recorded"),
                plan_step_ids=[str(item) for item in raw_edge.get("plan_step_ids") or []],
                operations=[str(item) for item in raw_edge.get("operations") or []],
                input_audio_paths=[str(item) for item in raw_edge.get("input_audio_paths") or []],
                output_audio_paths=[str(item) for item in raw_edge.get("output_audio_paths") or []],
                evidence=dict(raw_edge.get("evidence") or {}),
                rollback=dict(raw_edge.get("rollback") or {}),
                metadata=dict(raw_edge.get("metadata") or {}),
            )
        graph.round_history = [
            EditRoundRecord(
                round_index=int(item.get("round_index") or 0),
                request=str(item.get("request") or ""),
                status=str(item.get("status") or "recorded"),
                started_at=str(item.get("started_at") or _now()),
                input_node_id=item.get("input_node_id"),
                result_node_id=item.get("result_node_id"),
                edge_id=item.get("edge_id"),
                ended_at=item.get("ended_at"),
                evidence_path=item.get("evidence_path"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in payload.get("round_history") or []
            if isinstance(item, dict)
        ]
        return graph

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def ensure_initial_node(
        self,
        *,
        input_audio: str | None = None,
        input_audios: list[str] | None = None,
        reference_audio_items: list[dict[str, Any]] | None = None,
        reference_audio_paths: list[str] | None = None,
    ) -> None:
        """Create the root node once, preserving all source/reference audio paths."""

        if self.root_node_ids and self.current_node_id:
            self._merge_external_assets(
                input_audios=input_audios,
                reference_audio_items=reference_audio_items,
                reference_audio_paths=reference_audio_paths,
            )
            return
        paths = _dedupe(
            [
                *(input_audios or []),
                *([input_audio] if input_audio else []),
                *(reference_audio_paths or []),
                *[
                    str(item.get("path") or item.get("source") or "")
                    for item in reference_audio_items or []
                    if isinstance(item, dict)
                ],
            ]
        )
        assets: list[SessionAudioAsset] = []
        primary = input_audio or (paths[0] if paths else None)
        for index, path in enumerate(paths):
            if not path:
                continue
            role = "source" if _same_path(path, primary) or index == 0 else "reference"
            assets.append(
                SessionAudioAsset(
                    asset_id=f"asset_initial_{index + 1}",
                    path=str(path),
                    role=role,
                    label="primary source" if role == "source" else f"reference {index}",
                    audio_id="source" if role == "source" else _reference_audio_id(path, reference_audio_items, index),
                    source="session_initial_inputs",
                    round_index=0,
                    metadata=_reference_item_metadata(path, reference_audio_items) if role == "reference" else {},
                )
            )
        node_id = "node_initial"
        node = EditSessionNode(
            node_id=node_id,
            round_index=0,
            created_at=_now(),
            kind="initial",
            audio_paths=assets,
            primary_audio_path=primary,
            edit_count=0,
            operation_summary="Initial source/reference audio set.",
            stage="initial",
            is_round_result=True,
            metadata={"input_audio_count": len(paths)},
        )
        self.nodes[node_id] = node
        reference_node_ids = [
            self._ensure_reference_source_node(asset)
            for asset in assets
            if asset.role == "reference"
        ]
        self.root_node_ids = [node_id, *[item for item in reference_node_ids if item]]
        self.current_node_id = node_id
        self.updated_at = _now()

    def register_external_input(
        self, *, audio_ref: str, path: str, role: str, round_index: int,
    ) -> None:
        """Record a newly disclosed attachment without mutating an old state."""
        node_id = f"node_input_{_safe_id(audio_ref)}"
        if node_id in self.nodes:
            existing = self.nodes[node_id]
            if existing.primary_audio_path != path or existing.metadata.get("input_role") != role:
                raise ValueError("external input identity conflicts with an existing asset")
            return
        self.nodes[node_id] = EditSessionNode(
            node_id=node_id, round_index=round_index, created_at=_now(),
            kind="external_input",
            audio_paths=[SessionAudioAsset(
                asset_id=f"asset_{_safe_id(audio_ref)}", path=path, role=role,
                audio_id=audio_ref, source="session_round_inputs", round_index=round_index,
            )],
            primary_audio_path=path,
            operation_summary="User supplied audio input.",
            metadata={
                "audio_ref": audio_ref, "input_role": role,
                "session_eligible": role == "source",
            },
        )
        self.root_node_ids.append(node_id)
        self.updated_at = _now()

    def start_round(self, request: str, *, metadata: dict[str, Any] | None = None) -> int:
        self.round_index += 1
        record = EditRoundRecord(
            round_index=self.round_index,
            request=request,
            status="planning",
            started_at=_now(),
            input_node_id=self.current_node_id,
            metadata=dict(metadata or {}),
        )
        self.round_history.append(record)
        self.updated_at = _now()
        return self.round_index

    def complete_round(
        self,
        *,
        round_index: int,
        request: str,
        status: str,
        plan: Any,
        final_output_path: str | None,
        evidence_path: str | None = None,
        evaluation: dict[str, Any] | None = None,
        trace_result: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        advance_current_audio: bool = True,
        failure_registry: dict[str, Any] | None = None,
        suffix_resume_checkpoint: dict[str, Any] | None = None,
        suffix_resume_execution_admission: dict[str, Any] | None = None,
        final_delivery_manifest_shadow: dict[str, Any] | None = None,
        session_transition: dict[str, Any] | None = None,
    ) -> None:
        """Append one round while advancing current audio only when authorized."""

        round_entry_node_id = _round_input_node_id(self, round_index) or self.current_node_id
        metadata = _model_metadata(plan)
        session_meta = (
            dict(session_transition)
            if session_transition is not None
            else metadata.get("session_graph")
            if isinstance(metadata.get("session_graph"), dict)
            else {}
        )
        terminal_failure_stage = (
            str(trace_result.get("failure_stage") or "")
            if isinstance(trace_result, dict)
            else ""
        )
        terminal_failure_codes = (
            [str(item) for item in trace_result.get("failure_codes") or []]
            if isinstance(trace_result, dict)
            else []
        )
        selected_source_node_id = str(
            session_meta.get("selected_source_node_id") or round_entry_node_id or ""
        )
        if selected_source_node_id and selected_source_node_id not in self.nodes:
            raise ValueError(
                f"selected Session Graph source node does not exist: {selected_source_node_id}"
            )
        parent_id = selected_source_node_id or round_entry_node_id
        parent = self.nodes.get(parent_id or "")
        rollback_to_node_id = str(session_meta.get("rollback_to_node_id") or "")
        operations = _plan_operations(plan)
        plan_step_ids = _plan_step_ids(plan)
        operation_type = _operation_type(request, session_meta, rollback_to_node_id)
        if advance_current_audio and operation_type == "rollback" and not rollback_to_node_id:
            rollback_to_node_id = (
                _previous_round_node_id(self, before_round_index=round_index) or ""
            )
        if (
            advance_current_audio
            and operation_type == "rollback"
            and rollback_to_node_id
            and not final_output_path
        ):
            final_output_path = self.nodes.get(
                rollback_to_node_id, EditSessionNode("", 0, "", "")
            ).primary_audio_path
        result_path = (
            final_output_path or (parent.primary_audio_path if parent else None)
            if advance_current_audio
            else None
        )
        node_id = f"node_round_{round_index}"
        edge_id = f"edge_round_{round_index}"
        context_input_audios = (context or {}).get("input_audios")
        if not isinstance(context_input_audios, list):
            context_input_audios = []
        input_audio_paths = _dedupe(
            [
                *context_input_audios,
                (context or {}).get("input_audio") or "",
                *[item.get("path", "") for item in _audio_artifacts(plan) if isinstance(item, dict) and item.get("kind") == "input"],
            ]
        )
        output_audio_paths = _dedupe(
            [
                result_path or "",
                *[item.get("path", "") for item in _audio_artifacts(plan) if isinstance(item, dict) and item.get("kind") == "step_output"],
            ]
        )
        rollback_stack = list(parent.rollback_stack if parent else [])
        rollback_payload: dict[str, Any] = {}
        if advance_current_audio and operation_type == "rollback":
            rollback_payload = {
                "from_node_id": round_entry_node_id,
                "to_node_id": rollback_to_node_id,
                "requested_by_round": round_index,
                "replayed_edge_ids": session_meta.get("replayed_edge_ids") or [],
                "reason": session_meta.get("rollback_reason") or "user requested rollback/reverse/transform-back behavior",
            }
            rollback_stack.append(rollback_payload)
        node = EditSessionNode(
            node_id=node_id,
            round_index=round_index,
            created_at=_now(),
            kind=(
                "failed_edit_attempt"
                if not advance_current_audio
                else "rollback_result"
                if operation_type == "rollback"
                else "state_selection_result"
                if operation_type == "state_selection"
                else "edit_result"
            ),
            audio_paths=[
                SessionAudioAsset(
                    asset_id=f"asset_round_{round_index}_primary",
                    path=str(result_path),
                    role="round_result",
                    label=f"round {round_index} result",
                    source=f"edge:{edge_id}",
                    round_index=round_index,
                    metadata={"status": status},
                )
            ]
            if result_path
            else [],
            primary_audio_path=result_path,
            parent_node_ids=[item for item in [parent_id] if item],
            incoming_edge_ids=[edge_id],
            edit_count=(parent.edit_count if parent else 0)
            + (
                0
                if operation_type in {"rollback", "state_selection"}
                or not advance_current_audio
                else 1
            ),
            operation_summary=_operation_summary(operations, request),
            operation_history=[
                *(parent.operation_history if parent else []),
                {
                    "round_index": round_index,
                    "request": request,
                    "operations": operations,
                    "status": status,
                    "operation_type": operation_type,
                },
            ],
            stage=f"round_{round_index}",
            is_round_result=advance_current_audio,
            rollback_stack=rollback_stack,
            metadata={
                "evaluation": evaluation or {},
                "selected_source_node_id": selected_source_node_id or None,
                "rollback_to_node_id": rollback_to_node_id or None,
                "current_audio_advanced": advance_current_audio,
                "preserved_current_node_id": round_entry_node_id
                if not advance_current_audio
                else None,
                "failure_registry": failure_registry or {},
                "terminal_failure_stage": terminal_failure_stage or None,
                "terminal_failure_codes": terminal_failure_codes,
                **(
                    {"state_only_completion": dict(session_transition)}
                    if session_transition is not None
                    else {}
                ),
                "suffix_resume_checkpoint": suffix_resume_checkpoint or {},
                "suffix_resume_execution_admission": (
                    suffix_resume_execution_admission or {}
                ),
                **(
                    {
                        "final_delivery_manifest_shadow": final_delivery_manifest_shadow
                    }
                    if final_delivery_manifest_shadow
                    else {}
                ),
            },
        )
        edge = EditSessionEdge(
            edge_id=edge_id,
            round_index=round_index,
            created_at=_now(),
            source_node_ids=[item for item in [parent_id] if item],
            target_node_id=node_id,
            operation_type=operation_type,
            instruction=request,
            status=status,
            plan_step_ids=plan_step_ids,
            operations=operations,
            input_audio_paths=input_audio_paths,
            output_audio_paths=output_audio_paths,
            evidence={
                "execution_summary": _metadata_value(plan, "execution_summary"),
                "audio_artifacts": _audio_artifacts(plan),
                "executor_step_records": _metadata_value(plan, "executor_step_records") or [],
                "tool_invocations": _metadata_value(plan, "tool_invocations") or [],
                "localization": _localization_from_trace(trace_result),
                "terminal_failure_stage": terminal_failure_stage or None,
                "terminal_failure_codes": terminal_failure_codes,
                "evidence_path": evidence_path,
                "failure_registry": failure_registry or {},
                "suffix_resume_checkpoint": suffix_resume_checkpoint or {},
                "suffix_resume_execution_admission": (
                    suffix_resume_execution_admission or {}
                ),
                **(
                    {
                        "final_delivery_manifest_shadow": final_delivery_manifest_shadow
                    }
                    if final_delivery_manifest_shadow
                    else {}
                ),
            },
            rollback=rollback_payload,
            metadata={
                "session_graph_plan_metadata": session_meta,
                "session_eligible": advance_current_audio,
            },
        )
        self.nodes[node_id] = node
        self.edges[edge_id] = edge
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].outgoing_edge_ids.append(edge_id)
        step_lineage = self._append_step_asset_lineage(
            round_index=round_index,
            request=request,
            status=status,
            plan=plan,
            round_edge_id=edge_id,
            round_result_node_id=node_id,
            input_node_id=parent_id,
            result_path=result_path,
            session_eligible=advance_current_audio,
        )
        if step_lineage:
            node.metadata["step_lineage"] = step_lineage
            edge.metadata["step_lineage"] = step_lineage
        if advance_current_audio:
            self.current_node_id = node_id
        for record in self.round_history:
            if record.round_index == round_index:
                record.status = status
                record.result_node_id = node_id
                record.edge_id = edge_id
                record.ended_at = _now()
                record.evidence_path = evidence_path
                record.input_node_id = parent_id
                record.metadata["round_entry_node_id"] = round_entry_node_id
                record.metadata["current_audio_advanced"] = advance_current_audio
                if not advance_current_audio:
                    record.metadata["preserved_current_node_id"] = round_entry_node_id
                if failure_registry:
                    record.metadata["failure_registry"] = failure_registry
                if suffix_resume_checkpoint:
                    record.metadata["suffix_resume_checkpoint"] = (
                        suffix_resume_checkpoint
                    )
                if suffix_resume_execution_admission:
                    record.metadata["suffix_resume_execution_admission"] = (
                        suffix_resume_execution_admission
                    )
                if final_delivery_manifest_shadow:
                    record.metadata["final_delivery_manifest_shadow"] = (
                        final_delivery_manifest_shadow
                    )
                if step_lineage:
                    record.metadata["step_edge_ids"] = step_lineage.get("step_edge_ids", [])
                    record.metadata["step_node_ids"] = step_lineage.get("step_node_ids", [])
                break
        self.updated_at = _now()

    def current_node(self) -> EditSessionNode | None:
        return self.nodes.get(self.current_node_id or "")

    def current_audio_path(self) -> str | None:
        node = self.current_node()
        return node.primary_audio_path if node else None

    def semantic_state_context(
        self,
        *,
        audio_ref_by_path: dict[str, str],
    ) -> SessionStateContext:
        """Project authoritative round history without exposing filesystem paths."""

        if not self.current_node_id or self.current_node_id not in self.nodes:
            raise ValueError("session graph has no current state node")
        projected_nodes: list[SessionStateNodeContext] = []
        included_node_ids: set[str] = set()
        ordered_nodes = sorted(
            self.nodes.values(),
            key=lambda item: (item.round_index, item.created_at, item.node_id),
        )
        for node in ordered_nodes:
            if not (node.kind in {"initial", "external_input"} or node.is_round_result):
                continue
            audio_ref = _optional_text(node.metadata.get("audio_ref")) or (
                _audio_ref_for_path(
                    node.primary_audio_path,
                    audio_ref_by_path=audio_ref_by_path,
                )
            )
            if node.node_id == self.current_node_id and audio_ref is None:
                raise ValueError("current session state has no canonical audio ref")
            audio_slots = _session_state_audio_slots(
                node,
                primary_audio_ref=audio_ref,
                audio_ref_by_path=audio_ref_by_path,
            )
            projected_nodes.append(
                SessionStateNodeContext(
                    node_id=node.node_id,
                    audio_ref=audio_ref,
                    artifact_ref_id=_optional_text(
                        node.metadata.get("artifact_ref_id")
                    ),
                    round_index=node.round_index,
                    kind=node.kind,
                    parent_node_ids=list(node.parent_node_ids),
                    operation_summary=node.operation_summary,
                    session_eligible=_session_eligible(node),
                    is_current=node.node_id == self.current_node_id,
                    audio_slots=audio_slots,
                )
            )
            included_node_ids.add(node.node_id)
        current_audio_ref = next(
            item.audio_ref
            for item in projected_nodes
            if item.node_id == self.current_node_id
        )
        assert current_audio_ref is not None
        transitions = [
            SessionStateTransitionContext(
                edge_id=edge.edge_id,
                source_node_ids=list(edge.source_node_ids),
                target_node_id=edge.target_node_id,
                round_index=edge.round_index,
                instruction=edge.instruction,
                operations=list(edge.operations),
                operation_type=edge.operation_type,
                status=edge.status,
            )
            for edge in sorted(
                self.edges.values(),
                key=lambda item: (item.round_index, item.created_at, item.edge_id),
            )
            if edge.target_node_id in included_node_ids
            and set(edge.source_node_ids).issubset(included_node_ids)
        ]
        return SessionStateContext(
            current_node_id=self.current_node_id,
            current_audio_ref=current_audio_ref,
            nodes=projected_nodes,
            transitions=transitions,
        )

    def planner_context(self, *, max_nodes: int = 12, max_edges: int = 12) -> dict[str, Any]:
        aliases = self.audio_aliases()
        nodes = sorted(self.nodes.values(), key=lambda item: (item.round_index, item.created_at))[-max_nodes:]
        edges = sorted(self.edges.values(), key=lambda item: (item.round_index, item.created_at))[-max_edges:]
        current = self.current_node()
        return _without_empty(
            {
                "schema_version": self.schema_version,
                "session_id": self.session_id,
                "current_round_index": self.round_index,
                "current_node_id": self.current_node_id,
                "current_audio_path": self.current_audio_path(),
                "current_edit_count": current.edit_count if current else 0,
                "current_rollback_stack": current.rollback_stack if current else [],
                "audio_aliases": aliases,
                "reference_landmarks": self.reference_landmarks(),
                "normalized_refs": _normalized_refs_for_planner(self),
                "nodes": [_node_summary(node) for node in nodes],
                "edges": [_edge_summary(edge) for edge in edges],
                "round_history": [_round_summary(item) for item in self.round_history[-max_nodes:]],
                "planner_protocol": {
                    "default_input_alias": "session_current",
                    "node_alias_format": "session_node:<node_id>",
                    "round_result_alias_format": "session_round:<round_index>:result",
                    "rollback_metadata": (
                        "For undo/transform back/reverse plans, set plan.metadata.session_graph."
                        "rollback_to_node_id and selected_source_node_id, then operate on that alias or export it."
                    ),
                    "multi_audio_policy": (
                        "Use audio_id/reference aliases only for reference/context paths; edit steps should consume "
                        "session_current unless the instruction explicitly targets another session node or input audio."
                    ),
                    "step_lineage_policy": (
                        "Step-level operation edges and audio/evidence asset nodes live in the same graph. "
                        "Use reference landmark nodes only when a later operation explicitly needs reference audio."
                    ),
                    "normalized_refs_policy": (
                        "normalized_refs contains stable ArtifactRef/SpanRef/ObjectiveRef ids. Bind selected spans "
                        "to their audio_ref_id and keep objective coverage tied to executable step ids."
                    ),
                },
            }
        )

    def audio_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        current = self.current_audio_path()
        if current:
            aliases["session_current"] = current
            aliases["current_session_audio"] = current
        for node in self.nodes.values():
            if not node.primary_audio_path or not _session_eligible(node):
                continue
            aliases[f"session_node:{node.node_id}"] = node.primary_audio_path
            aliases[f"node:{node.node_id}"] = node.primary_audio_path
            if node.is_round_result:
                aliases[f"session_round:{node.round_index}:result"] = node.primary_audio_path
                aliases[f"round_{node.round_index}_result"] = node.primary_audio_path
            if node.kind == "initial":
                aliases["session_initial"] = node.primary_audio_path
                aliases["session_original"] = node.primary_audio_path
            if node.metadata.get("is_reference_landmark"):
                aliases[f"session_reference_node:{node.node_id}"] = node.primary_audio_path
                aliases[f"reference_node:{node.node_id}"] = node.primary_audio_path
                audio_id = node.metadata.get("audio_id")
                if audio_id:
                    aliases[f"session_reference:{audio_id}"] = node.primary_audio_path
                    aliases[f"reference:{audio_id}"] = node.primary_audio_path
                    aliases.setdefault(str(audio_id), node.primary_audio_path)
        for node in self.nodes.values():
            if not _session_eligible(node):
                continue
            for asset in node.audio_paths:
                if asset.audio_id and asset.path:
                    aliases[f"session_audio:{asset.audio_id}"] = asset.path
                    aliases.setdefault(asset.audio_id, asset.path)
        return aliases

    def reference_landmarks(self) -> list[dict[str, Any]]:
        landmarks: list[dict[str, Any]] = []
        for node in self.nodes.values():
            if not node.metadata.get("is_reference_landmark"):
                continue
            landmarks.append(
                _without_empty(
                    {
                        "node_id": node.node_id,
                        "kind": node.kind,
                        "audio_id": node.metadata.get("audio_id"),
                        "primary_audio_path": node.primary_audio_path,
                        "round_index": node.round_index,
                        "step_id": node.metadata.get("step_id"),
                        "operation": node.metadata.get("operation"),
                        "reference_role": node.metadata.get("reference_role"),
                        "span_set": node.metadata.get("span_set"),
                        "speaker_ids": node.metadata.get("speaker_ids"),
                        "requested_time_range": node.metadata.get("requested_time_range"),
                    }
                )
            )
        return sorted(landmarks, key=lambda item: (int(item.get("round_index") or 0), str(item.get("node_id") or "")))

    def _merge_external_assets(
        self,
        *,
        input_audios: list[str] | None = None,
        reference_audio_items: list[dict[str, Any]] | None = None,
        reference_audio_paths: list[str] | None = None,
    ) -> None:
        node = self.current_node()
        if node is None:
            return
        known = {_path_key(asset.path) for asset in node.audio_paths}
        new_paths = _dedupe(
            [
                *(input_audios or []),
                *(reference_audio_paths or []),
                *[
                    str(item.get("path") or item.get("source") or "")
                    for item in reference_audio_items or []
                    if isinstance(item, dict)
                ],
            ]
        )
        for path in new_paths:
            if not path or _path_key(path) in known:
                continue
            known.add(_path_key(path))
            role = _external_asset_role(path, reference_audio_items, reference_audio_paths)
            node.audio_paths.append(
                asset := SessionAudioAsset(
                    asset_id=f"asset_round_{self.round_index}_external_{len(node.audio_paths) + 1}",
                    path=path,
                    role=role,
                    label="newly supplied session audio",
                    audio_id=_reference_audio_id(path, reference_audio_items, len(node.audio_paths)) if role == "reference" else None,
                    source="session_round_inputs",
                    round_index=self.round_index,
                    metadata=_reference_item_metadata(path, reference_audio_items),
                )
            )
            if asset.role == "reference":
                ref_node_id = self._ensure_reference_source_node(asset)
                if ref_node_id and ref_node_id not in self.root_node_ids:
                    self.root_node_ids.append(ref_node_id)
        self.updated_at = _now()

    def _ensure_reference_source_node(self, asset: SessionAudioAsset) -> str | None:
        if not asset.path:
            return None
        for node in self.nodes.values():
            if node.kind == "reference_source" and _same_path(node.primary_audio_path, asset.path):
                return node.node_id
        audio_id = asset.audio_id or _safe_id(Path(asset.path).stem)
        node_id = _unique_id(f"node_ref_{_safe_id(audio_id)}", self.nodes)
        metadata = dict(asset.metadata or {})
        metadata.update(
            {
                "asset_role": "reference_source",
                "reference_role": "reference_source",
                "is_reference_landmark": True,
                "audio_id": audio_id,
                "requested_time_range": metadata.get("time_range") or metadata.get("requested_time_range"),
            }
        )
        node = EditSessionNode(
            node_id=node_id,
            round_index=asset.round_index or 0,
            created_at=_now(),
            kind="reference_source",
            audio_paths=[
                SessionAudioAsset(
                    asset_id=f"{asset.asset_id}_source_node",
                    path=asset.path,
                    role="reference_source",
                    label=asset.label or "reference source",
                    audio_id=asset.audio_id,
                    source=asset.source,
                    round_index=asset.round_index,
                    span=asset.span,
                    metadata=dict(asset.metadata or {}),
                )
            ],
            primary_audio_path=asset.path,
            operation_summary="Reference audio source.",
            stage="reference",
            metadata=_without_empty(metadata),
        )
        self.nodes[node_id] = node
        return node_id

    def _append_step_asset_lineage(
        self,
        *,
        round_index: int,
        request: str,
        status: str,
        plan: Any,
        round_edge_id: str,
        round_result_node_id: str,
        input_node_id: str | None,
        result_path: str | None,
        session_eligible: bool,
    ) -> dict[str, Any]:
        steps = _plan_steps(plan)
        if not steps:
            return {}
        artifacts = _audio_artifacts(plan)
        records = _executor_step_records(plan)
        records_by_step = {
            str(record.get("step_id")): record
            for record in records
            if isinstance(record, dict) and record.get("step_id")
        }
        artifacts_by_step: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifacts:
            step_id = artifact.get("step_id")
            if step_id:
                artifacts_by_step.setdefault(str(step_id), []).append(artifact)
        path_nodes = _path_to_node_ids(self)
        step_output_nodes: dict[str, list[str]] = {}
        step_edge_ids: list[str] = []
        step_node_ids: list[str] = []
        last_editable_node_id = input_node_id

        for index, step in enumerate(steps, start=1):
            step_id = _step_id(step) or f"step_{index}"
            operation = _step_operation(step)
            step_status = _step_status(step) or str(records_by_step.get(step_id, {}).get("status") or status)
            record = records_by_step.get(step_id, {})
            step_artifacts = artifacts_by_step.get(step_id, [])
            step_inputs = _step_inputs(step)
            step_params = _step_params(step)
            depends_on = _step_depends_on(step)
            input_node_ids = _resolve_step_input_node_ids(
                step_inputs=step_inputs,
                step_params=step_params,
                record=record,
                depends_on=depends_on,
                path_nodes=path_nodes,
                step_output_nodes=step_output_nodes,
                fallback_node_id=last_editable_node_id,
            )
            output_node_ids: list[str] = []
            primary_target_node_id = ""
            step_result = _step_result(step)
            span_evidence = _span_evidence_from_step(step_result, step_params)
            speaker_ids = _speaker_ids_from_spans(span_evidence)
            operation_type = _step_operation_type(
                operation=operation,
                request=request,
                description=_step_description(step),
                params=step_params,
                artifacts=step_artifacts,
            )

            for artifact_index, artifact in enumerate(step_artifacts, start=1):
                artifact_path = str(artifact.get("path") or "")
                if not artifact_path:
                    continue
                if result_path and _same_path(artifact_path, result_path):
                    node_id = round_result_node_id
                    node = self.nodes.get(node_id)
                    if node is not None:
                        node.metadata.setdefault("step_id", step_id)
                        node.metadata.setdefault("operation", operation)
                        node.metadata.setdefault("step_index", index)
                        node.metadata.setdefault("span_set", span_evidence)
                        if speaker_ids:
                            node.metadata.setdefault("speaker_ids", speaker_ids)
                else:
                    role = _artifact_role(artifact, operation_type)
                    node_id = _unique_id(
                        f"node_round_{round_index}_{_safe_id(step_id)}_{_safe_id(role)}",
                        self.nodes,
                    )
                    node_metadata = _without_empty(
                        {
                            "asset_role": role,
                            "round_index": round_index,
                            "step_index": index,
                            "step_id": step_id,
                            "operation": operation,
                            "source_round_edge_id": round_edge_id,
                            "span_set": span_evidence,
                            "speaker_ids": speaker_ids,
                            "is_reference_landmark": role in {"reference_clip", "reference_source"},
                            "reference_role": role if role.startswith("reference") else None,
                            "audio_artifact": dict(artifact),
                            "session_eligible": session_eligible,
                        }
                    )
                    node = EditSessionNode(
                        node_id=node_id,
                        round_index=round_index,
                        created_at=_now(),
                        kind=role,
                        audio_paths=[
                            SessionAudioAsset(
                                asset_id=f"asset_round_{round_index}_{_safe_id(step_id)}_{artifact_index}",
                                path=artifact_path,
                                role=role,
                                label=str(artifact.get("role") or role),
                                audio_id=_artifact_audio_id(artifact, role, step_id),
                                source=f"step:{step_id}",
                                round_index=round_index,
                                step_id=step_id,
                                operation=operation,
                                span=_single_span_or_none(span_evidence),
                                metadata=_without_empty(
                                    {
                                        "span_set": span_evidence,
                                        "speaker_ids": speaker_ids,
                                        "artifact": dict(artifact),
                                    }
                                ),
                            )
                        ],
                        primary_audio_path=artifact_path,
                        parent_node_ids=input_node_ids,
                        edit_count=(self.nodes.get(last_editable_node_id or "", EditSessionNode("", 0, "", "")).edit_count if last_editable_node_id else 0),
                        operation_summary=operation,
                        stage=f"round_{round_index}_step_{index}",
                        metadata=node_metadata,
                    )
                    self.nodes[node_id] = node
                    step_node_ids.append(node_id)
                output_node_ids.append(node_id)
                path_nodes.setdefault(_path_key(artifact_path), []).append(node_id)

            if not output_node_ids:
                node_id = _unique_id(f"node_round_{round_index}_{_safe_id(step_id)}_evidence", self.nodes)
                node = EditSessionNode(
                    node_id=node_id,
                    round_index=round_index,
                    created_at=_now(),
                    kind="operation_evidence",
                    parent_node_ids=input_node_ids,
                    edit_count=(self.nodes.get(last_editable_node_id or "", EditSessionNode("", 0, "", "")).edit_count if last_editable_node_id else 0),
                    operation_summary=operation,
                    stage=f"round_{round_index}_step_{index}",
                    metadata=_without_empty(
                        {
                            "asset_role": "operation_evidence",
                            "round_index": round_index,
                            "step_index": index,
                            "step_id": step_id,
                            "operation": operation,
                            "result_keys": sorted(step_result.keys()) if isinstance(step_result, dict) else [],
                            "span_set": span_evidence,
                            "speaker_ids": speaker_ids,
                            "session_eligible": session_eligible,
                        }
                    ),
                )
                self.nodes[node_id] = node
                output_node_ids.append(node_id)
                step_node_ids.append(node_id)

            primary_target_node_id = output_node_ids[0]
            edge_id = _unique_id(f"edge_round_{round_index}_{_safe_id(step_id)}", self.edges)
            edge = EditSessionEdge(
                edge_id=edge_id,
                round_index=round_index,
                created_at=_now(),
                source_node_ids=input_node_ids,
                target_node_id=primary_target_node_id,
                operation_type=operation_type,
                instruction=_step_description(step) or request,
                status=step_status,
                plan_step_ids=[step_id],
                operations=[operation] if operation else [],
                input_audio_paths=_paths_for_nodes(self, input_node_ids),
                output_audio_paths=_dedupe([str(item.get("path") or "") for item in step_artifacts]),
                evidence=_without_empty(
                    {
                        "executor_step_record": record,
                        "audio_artifacts": step_artifacts,
                        "selected_spans": span_evidence,
                        "tool_invocations": _tool_invocations_for_step(plan, step_id),
                    }
                ),
                metadata=_without_empty(
                    {
                        "round_summary_edge_id": round_edge_id,
                        "step_index": index,
                        "step_id": step_id,
                        "depends_on": depends_on,
                        "input_node_ids": input_node_ids,
                        "output_node_ids": output_node_ids,
                        "inputs": step_inputs,
                        "params_summary": _params_summary(step_params),
                        "is_reference_operation": operation_type == "reference_prepare",
                        "speaker_ids": speaker_ids,
                        "session_eligible": session_eligible,
                    }
                ),
            )
            self.edges[edge_id] = edge
            step_edge_ids.append(edge_id)
            for source_id in input_node_ids:
                if source_id in self.nodes:
                    self.nodes[source_id].outgoing_edge_ids.append(edge_id)
            for target_id in output_node_ids:
                if target_id in self.nodes:
                    self.nodes[target_id].incoming_edge_ids.append(edge_id)
            step_output_nodes[step_id] = output_node_ids
            if operation_type != "reference_prepare":
                last_audio = _last_audio_node(self.nodes, output_node_ids)
                if last_audio:
                    last_editable_node_id = last_audio

        return _without_empty(
            {
                "lineage_model": "single_graph_step_audio_asset_dag",
                "round_summary_edge_id": round_edge_id,
                "step_edge_ids": step_edge_ids,
                "step_node_ids": step_node_ids,
            }
        )


def load_edit_session_graph(path: str | Path) -> EditSessionGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("edit session graph JSON must be an object")
    return EditSessionGraph.from_dict(payload)


def save_edit_session_graph(graph: EditSessionGraph, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def graph_from_context(context: dict[str, Any], *, session_id: str | None = None) -> EditSessionGraph:
    raw = context.get("edit_session_graph")
    if isinstance(raw, EditSessionGraph):
        return raw
    if isinstance(raw, dict):
        return EditSessionGraph.from_dict(raw)
    path = context.get("edit_session_graph_path")
    if path:
        candidate = Path(str(path))
        if candidate.exists():
            return load_edit_session_graph(candidate)
    return EditSessionGraph.create(session_id=session_id or _context_session_id(context))


def _context_session_id(context: dict[str, Any]) -> str | None:
    value = context.get("edit_session_id") or context.get("session_id")
    return str(value) if value else None


def _node_summary(node: EditSessionNode) -> dict[str, Any]:
    return _without_empty(
        {
            "node_id": node.node_id,
            "round_index": node.round_index,
            "kind": node.kind,
            "primary_audio_path": node.primary_audio_path,
            "asset_count": len(node.audio_paths),
            "audio_paths": [
                _without_empty(
                    {
                        "asset_id": asset.asset_id,
                        "path": asset.path,
                        "role": asset.role,
                        "audio_id": asset.audio_id,
                        "label": asset.label,
                        "span": asset.span,
                    }
                )
                for asset in node.audio_paths[:8]
            ],
            "parent_node_ids": node.parent_node_ids,
            "edit_count": node.edit_count,
            "operation_summary": node.operation_summary,
            "is_round_result": node.is_round_result,
            "rollback_stack_depth": len(node.rollback_stack),
            "step_id": node.metadata.get("step_id"),
            "step_index": node.metadata.get("step_index"),
            "asset_role": node.metadata.get("asset_role"),
            "is_reference_landmark": node.metadata.get("is_reference_landmark"),
            "speaker_ids": node.metadata.get("speaker_ids"),
            "span_set": node.metadata.get("span_set"),
            "session_eligible": _session_eligible(node),
            "current_audio_advanced": node.metadata.get("current_audio_advanced"),
            "failure_summary": _failure_summary(
                node.metadata.get("failure_registry")
            ),
            "suffix_resume_summary": _suffix_resume_summary(
                node.metadata.get("suffix_resume_checkpoint")
            ),
        }
    )


def _edge_summary(edge: EditSessionEdge) -> dict[str, Any]:
    return _without_empty(
        {
            "edge_id": edge.edge_id,
            "round_index": edge.round_index,
            "source_node_ids": edge.source_node_ids,
            "target_node_id": edge.target_node_id,
            "operation_type": edge.operation_type,
            "instruction": edge.instruction,
            "status": edge.status,
            "operations": edge.operations,
            "output_audio_paths": edge.output_audio_paths[:4],
            "rollback": edge.rollback,
            "step_id": edge.metadata.get("step_id"),
            "step_index": edge.metadata.get("step_index"),
            "output_node_ids": edge.metadata.get("output_node_ids"),
            "is_reference_operation": edge.metadata.get("is_reference_operation"),
            "session_eligible": edge.metadata.get("session_eligible"),
            "failure_summary": _failure_summary(
                edge.evidence.get("failure_registry")
            ),
            "suffix_resume_summary": _suffix_resume_summary(
                edge.evidence.get("suffix_resume_checkpoint")
            ),
        }
    )


def _round_summary(record: EditRoundRecord) -> dict[str, Any]:
    return _without_empty(
        {
            "round_index": record.round_index,
            "request": record.request,
            "status": record.status,
            "input_node_id": record.input_node_id,
            "result_node_id": record.result_node_id,
            "edge_id": record.edge_id,
            "current_audio_advanced": record.metadata.get(
                "current_audio_advanced"
            ),
            "preserved_current_node_id": record.metadata.get(
                "preserved_current_node_id"
            ),
            "failure_summary": _failure_summary(
                record.metadata.get("failure_registry")
            ),
            "suffix_resume_summary": _suffix_resume_summary(
                record.metadata.get("suffix_resume_checkpoint")
            ),
        }
    )


def _session_eligible(node: EditSessionNode) -> bool:
    if node.metadata.get("session_eligible") is False:
        return False
    return node.metadata.get("current_audio_advanced") is not False


def _failure_summary(value: Any) -> dict[str, Any]:
    registry = value if isinstance(value, dict) else {}
    if not registry:
        return {}
    return _without_empty(
        {
            "status": registry.get("status"),
            "terminal_failure_stage": registry.get("terminal_failure_stage"),
            "terminal_failed_step_ids": registry.get("terminal_failed_step_ids") or [],
            "unresolved_failure_codes": registry.get("unresolved_failure_codes") or [],
            "failure_count": len(registry.get("failure_history") or []),
            "repair_count": len(registry.get("repair_history") or []),
        }
    )


def _suffix_resume_summary(value: Any) -> dict[str, Any]:
    checkpoint = value if isinstance(value, dict) else {}
    if not checkpoint:
        return {}
    return _without_empty(
        {
            "status": checkpoint.get("status"),
            "failed_step_ids": checkpoint.get("failed_step_ids") or [],
            "restart_step_ids": checkpoint.get("restart_step_ids") or [],
            "rerun_step_ids": checkpoint.get("rerun_step_ids") or [],
            "reusable_step_ids": checkpoint.get("reusable_step_ids") or [],
            "reusable_artifact_refs": [
                item.get("artifact_ref_id")
                for item in checkpoint.get("reusable_artifacts") or []
                if isinstance(item, dict) and item.get("artifact_ref_id")
            ],
            "behavior_changed": checkpoint.get("behavior_changed"),
        }
    )


def _normalized_refs_for_planner(graph: EditSessionGraph) -> dict[str, Any]:
    try:
        from x_editor.session import normalized_refs_for_planner

        return normalized_refs_for_planner(graph)
    except Exception as exc:
        return {
            "schema_version": "x-editor.normalized_session_record.v1",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        key = _path_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _path_key(value: Any) -> str:
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return _path_key(left) == _path_key(right)


def _audio_ref_for_path(
    path: str | None,
    *,
    audio_ref_by_path: dict[str, str],
) -> str | None:
    if not path:
        return None
    matches = [
        audio_ref
        for candidate_path, audio_ref in audio_ref_by_path.items()
        if _same_path(path, candidate_path)
    ]
    unique = list(dict.fromkeys(matches))
    if len(unique) > 1:
        raise ValueError("session audio path maps to multiple canonical refs")
    return unique[0] if unique else None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _reference_audio_id(path: str, items: list[dict[str, Any]] | None, index: int) -> str:
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_path = item.get("path") or item.get("source")
        if item_path and _same_path(path, item_path):
            return str(item.get("audio_id") or item.get("id") or f"ref{index + 1}")
    return f"ref{index + 1}"


def _round_input_node_id(graph: EditSessionGraph, round_index: int) -> str | None:
    for record in graph.round_history:
        if record.round_index == round_index:
            return record.input_node_id
    return None


def _session_state_audio_slots(
    node: EditSessionNode,
    *,
    primary_audio_ref: str | None,
    audio_ref_by_path: dict[str, str],
) -> list[SessionStateAudioSlotContext]:
    slots_by_ref: dict[str, SessionStateAudioSlotContext] = {}
    for asset in node.audio_paths:
        audio_ref = _audio_ref_for_path(
            asset.path,
            audio_ref_by_path=audio_ref_by_path,
        )
        if audio_ref is None and asset.path == node.primary_audio_path:
            audio_ref = primary_audio_ref
        if audio_ref is None:
            continue
        is_primary = audio_ref == primary_audio_ref
        existing = slots_by_ref.get(audio_ref)
        if existing is None or (is_primary and not existing.primary):
            slots_by_ref[audio_ref] = SessionStateAudioSlotContext(
                audio_ref=audio_ref,
                role=asset.role or ("primary" if is_primary else "member"),
                audio_id=_optional_text(asset.audio_id),
                primary=is_primary,
            )
    if primary_audio_ref is not None and primary_audio_ref not in slots_by_ref:
        slots_by_ref[primary_audio_ref] = SessionStateAudioSlotContext(
            audio_ref=primary_audio_ref,
            role="primary",
            primary=True,
        )
    return list(slots_by_ref.values())


def _previous_round_node_id(graph: EditSessionGraph, *, before_round_index: int) -> str | None:
    candidates = [
        record.result_node_id
        for record in graph.round_history
        if record.round_index < before_round_index and record.result_node_id
    ]
    if len(candidates) >= 2:
        return candidates[-2]
    if candidates:
        return graph.root_node_ids[0] if graph.root_node_ids else candidates[0]
    return graph.root_node_ids[0] if graph.root_node_ids else None


def _model_metadata(plan: Any) -> dict[str, Any]:
    if hasattr(plan, "metadata"):
        value = getattr(plan, "metadata")
        return value if isinstance(value, dict) else {}
    if isinstance(plan, dict):
        value = plan.get("metadata")
        return value if isinstance(value, dict) else {}
    return {}


def _metadata_value(plan: Any, key: str) -> Any:
    return _model_metadata(plan).get(key)


def _plan_steps(plan: Any) -> list[Any]:
    if hasattr(plan, "steps"):
        steps = getattr(plan, "steps")
        return list(steps) if isinstance(steps, list) else []
    if isinstance(plan, dict):
        steps = plan.get("steps")
        return steps if isinstance(steps, list) else []
    return []


def _plan_operations(plan: Any) -> list[str]:
    values: list[str] = []
    for step in _plan_steps(plan):
        operation = getattr(step, "operation", None) if not isinstance(step, dict) else step.get("operation")
        if operation:
            values.append(str(operation))
    return values


def _plan_step_ids(plan: Any) -> list[str]:
    values: list[str] = []
    for step in _plan_steps(plan):
        step_id = getattr(step, "id", None) if not isinstance(step, dict) else step.get("id")
        if step_id:
            values.append(str(step_id))
    return values


def _audio_artifacts(plan: Any) -> list[dict[str, Any]]:
    value = _metadata_value(plan, "audio_artifacts")
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _executor_step_records(plan: Any) -> list[dict[str, Any]]:
    value = _metadata_value(plan, "executor_step_records")
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _tool_invocations_for_step(plan: Any, step_id: str) -> list[dict[str, Any]]:
    value = _metadata_value(plan, "tool_invocations")
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if str(item.get("step_id") or "") == step_id:
            rows.append(dict(item))
            continue
        attempts = item.get("attempts")
        if isinstance(attempts, list) and any(isinstance(attempt, dict) and str(attempt.get("step_id") or "") == step_id for attempt in attempts):
            rows.append(dict(item))
    return rows


def _step_id(step: Any) -> str | None:
    value = getattr(step, "id", None) if not isinstance(step, dict) else step.get("id")
    return str(value) if value else None


def _step_operation(step: Any) -> str:
    value = getattr(step, "operation", None) if not isinstance(step, dict) else step.get("operation")
    return str(value or "")


def _step_description(step: Any) -> str:
    value = getattr(step, "description", None) if not isinstance(step, dict) else step.get("description")
    return str(value or "")


def _step_status(step: Any) -> str | None:
    value = getattr(step, "status", None) if not isinstance(step, dict) else step.get("status")
    return str(value) if value else None


def _step_inputs(step: Any) -> dict[str, Any]:
    value = getattr(step, "inputs", None) if not isinstance(step, dict) else step.get("inputs")
    return dict(value) if isinstance(value, dict) else {}


def _step_params(step: Any) -> dict[str, Any]:
    value = getattr(step, "params", None) if not isinstance(step, dict) else step.get("params")
    return dict(value) if isinstance(value, dict) else {}


def _step_depends_on(step: Any) -> list[str]:
    value = getattr(step, "depends_on", None) if not isinstance(step, dict) else step.get("depends_on")
    return [str(item) for item in value] if isinstance(value, list) else []


def _step_result(step: Any) -> dict[str, Any]:
    value = getattr(step, "result", None) if not isinstance(step, dict) else step.get("result")
    return dict(value) if isinstance(value, dict) else {}


def _path_to_node_ids(graph: EditSessionGraph) -> dict[str, list[str]]:
    by_path: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        if node.primary_audio_path:
            by_path.setdefault(_path_key(node.primary_audio_path), []).append(node.node_id)
        for asset in node.audio_paths:
            if asset.path:
                by_path.setdefault(_path_key(asset.path), []).append(node.node_id)
    for key, node_ids in list(by_path.items()):
        by_path[key] = sorted(
            _dedupe(node_ids),
            key=lambda node_id: (
                0 if graph.nodes.get(node_id) and graph.nodes[node_id].kind == "reference_source" else 1,
                0 if graph.nodes.get(node_id) and graph.nodes[node_id].primary_audio_path else 1,
                node_id,
            ),
        )
    return by_path


def _resolve_step_input_node_ids(
    *,
    step_inputs: dict[str, Any],
    step_params: dict[str, Any],
    record: dict[str, Any],
    depends_on: list[str],
    path_nodes: dict[str, list[str]],
    step_output_nodes: dict[str, list[str]],
    fallback_node_id: str | None,
) -> list[str]:
    node_ids: list[str] = []
    for dep_id in depends_on:
        node_ids.extend(step_output_nodes.get(dep_id) or [])
    for path in _step_input_paths(step_inputs, step_params, record):
        node_ids.extend(path_nodes.get(_path_key(path)) or [])
    if not node_ids and fallback_node_id:
        node_ids.append(fallback_node_id)
    return _dedupe(node_ids)


def _step_input_paths(step_inputs: dict[str, Any], step_params: dict[str, Any], record: dict[str, Any]) -> list[str]:
    values: list[Any] = [
        record.get("current_audio_before"),
        record.get("input_audio_path"),
    ]
    for key in ("inputs_after_normalization", "inputs_before"):
        if isinstance(record.get(key), dict):
            values.append(record[key])
    values.append(step_inputs)
    reference_subset = {
        key: step_params.get(key)
        for key in (
            "input_audio",
            "input_audio_path",
            "source_audio",
            "source_audio_path",
            "reference_audio",
            "reference_audio_path",
            "ref_audio",
            "ref_audio_path",
            "speaker_audio",
            "speaker_audio_path",
        )
        if key in step_params
    }
    values.append(reference_subset)
    return [path for path in _iter_audio_like_values(values) if path]


def _iter_audio_like_values(value: Any) -> list[str]:
    results: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if isinstance(item, (str, Path)) and (
                "audio" in key_text
                or "path" in key_text
                or "reference" in key_text
                or "source" in key_text
                or "speaker" in key_text
            ):
                results.append(str(item))
            else:
                results.extend(_iter_audio_like_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            results.extend(_iter_audio_like_values(item))
    elif isinstance(value, (str, Path)):
        text = str(value)
        if text and (Path(text).suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"} or "/" in text):
            results.append(text)
    return results


def _span_evidence_from_step(result: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for source in (result, params):
        for key in ("selected_spans", "target_spans", "spans", "segments", "speaker_segments"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, list):
                candidates.extend(value)
    spans: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if not any(key in item for key in ("start_sec", "end_sec", "start", "end", "speaker_id", "speaker_ids", "text", "transcript")):
            continue
        spans.append(dict(item))
    return spans


def _speaker_ids_from_spans(spans: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for span in spans:
        for key in ("speaker_id", "speaker", "speaker_label"):
            value = span.get(key)
            if value not in (None, ""):
                values.append(str(value))
        value = span.get("speaker_ids")
        if isinstance(value, list):
            values.extend(str(item) for item in value if item not in (None, ""))
    return _dedupe(values)


def _single_span_or_none(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    return dict(spans[0]) if len(spans) == 1 else None


def _step_operation_type(
    *,
    operation: str,
    request: str,
    description: str,
    params: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> str:
    op_lower = operation.lower()
    lowered = " ".join(
        str(value or "").lower()
        for value in [
            operation,
            request,
            description,
            params.get("selection_instruction"),
            params.get("selector"),
            params.get("target_speaker_description"),
            params.get("speaker_description"),
            *[item.get("role") for item in artifacts if isinstance(item, dict)],
            *[item.get("result_key") for item in artifacts if isinstance(item, dict)],
        ]
    )
    reference_prepare_op = (
        "extract" in op_lower
        or op_lower.startswith("perception.")
        or op_lower in {"slice_audio", "artifact_ops.extract_spans"}
    )
    if reference_prepare_op and any(term in lowered for term in ("reference", "ref_audio", "voice clone", "clone reference", "enrollment", "参考")):
            return "reference_prepare"
    if operation.startswith("perception."):
        return "perception"
    if "reconstruct" in operation:
        return "reconstruct"
    return "edit"


def _artifact_role(artifact: dict[str, Any], operation_type: str) -> str:
    text = " ".join(
        str(artifact.get(key) or "").lower()
        for key in ("artifact_id", "role", "source", "result_key", "description", "operation")
    )
    if operation_type == "reference_prepare" or any(term in text for term in ("reference", "ref_audio", "enrollment", "voice_clone")):
        return "reference_clip"
    if artifact.get("is_final_output"):
        return "round_result"
    if any(term in text for term in ("input_segment", "selected_span", "target_span", "source_segment")):
        return "target_clip"
    return "intermediate_clip"


def _artifact_audio_id(artifact: dict[str, Any], role: str, step_id: str) -> str | None:
    if role.startswith("reference"):
        return str(artifact.get("audio_id") or f"ref_clip_{step_id}")
    return artifact.get("audio_id")


def _paths_for_nodes(graph: EditSessionGraph, node_ids: list[str]) -> list[str]:
    return _dedupe([graph.nodes[node_id].primary_audio_path for node_id in node_ids if node_id in graph.nodes and graph.nodes[node_id].primary_audio_path])


def _params_summary(params: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "selector",
        "selection_instruction",
        "target_speaker_description",
        "speaker_description",
        "reference_audio_path",
        "ref_audio_path",
        "edit_op",
        "operation",
    ):
        if key in params and params[key] not in (None, "", [], {}):
            summary[key] = params[key]
    for key in ("selected_spans", "spans", "target_spans"):
        value = params.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
    return summary


def _last_audio_node(nodes: dict[str, EditSessionNode], node_ids: list[str]) -> str | None:
    for node_id in reversed(node_ids):
        node = nodes.get(node_id)
        if node and node.primary_audio_path and node.kind not in {"reference_clip", "reference_source", "operation_evidence"}:
            return node_id
    for node_id in reversed(node_ids):
        node = nodes.get(node_id)
        if node and node.primary_audio_path:
            return node_id
    return None


def _reference_item_metadata(path: str, items: list[dict[str, Any]] | None) -> dict[str, Any]:
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_path = item.get("path") or item.get("source") or item.get("audio_path")
        if item_path and _same_path(path, item_path):
            return _without_empty(
                {
                    "audio_id": item.get("audio_id") or item.get("id"),
                    "note": item.get("note") or item.get("description"),
                    "time_range": item.get("time_range") or item.get("segment") or item.get("part"),
                    "requested_time_range": item.get("time_range") or item.get("segment") or item.get("part"),
                    "source": item.get("source"),
                }
            )
    return {}


def _external_asset_role(path: str, items: list[dict[str, Any]] | None, reference_audio_paths: list[str] | None) -> str:
    if any(_same_path(path, item) for item in reference_audio_paths or []):
        return "reference"
    for item in items or []:
        if not isinstance(item, dict):
            continue
        item_path = item.get("path") or item.get("source") or item.get("audio_path")
        if item_path and _same_path(path, item_path):
            return "reference"
    return "external_context"


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
    return safe.strip("_") or "item"


def _unique_id(prefix: str, existing: dict[str, Any]) -> str:
    candidate = prefix
    index = 2
    while candidate in existing:
        candidate = f"{prefix}_{index}"
        index += 1
    return candidate


def _operation_type(request: str, session_meta: dict[str, Any], rollback_to_node_id: str) -> str:
    explicit = session_meta.get("operation_type")
    if explicit:
        return str(explicit)
    if rollback_to_node_id:
        return "rollback"
    text = " ".join(str(request or "").lower().split())
    rollback_markers = (
        "undo",
        "roll back",
        "rollback",
        "revert",
        "restore",
        "transform back",
        "reverse the edit",
        "撤销",
        "回退",
        "还原",
        "恢复到",
        "变回",
    )
    return "rollback" if any(marker in text for marker in rollback_markers) else "edit"


def _operation_summary(operations: list[str], request: str) -> str:
    if operations:
        return " -> ".join(operations)
    return str(request or "")[:160]


def _localization_from_trace(trace_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trace_result, dict):
        return {}
    evidence = trace_result.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("localization"), dict):
        return evidence["localization"]
    localization = trace_result.get("localization")
    return localization if isinstance(localization, dict) else {}
