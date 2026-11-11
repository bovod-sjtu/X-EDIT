"""Sidecar operation-level audio DAG projections.

The operation graph is intentionally a projection: it does not drive execution
or replace ``AudioEditPlan``.  It gives planner/executor traces a compact DAG
of useful audio artifacts inside one edit round.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from x_editor.session.schemas import SessionEvent, SessionEventLog, SessionModel, ValidationIssue


OPERATION_GRAPH_SCHEMA_VERSION = "x-editor.operation_graph.v1"
OPERATION_GRAPH_VALIDATION_SCHEMA_VERSION = "x-editor.operation_graph_validation.v1"
OPERATION_GRAPH_RECONCILIATION_SCHEMA_VERSION = "x-editor.operation_graph_reconciliation.v1"
ROUND_OPERATION_GRAPH_STORE_SCHEMA_VERSION = "x-editor.round_operation_graph_store.v1"
ROUND_OPERATION_GRAPH_VALIDATION_SCHEMA_VERSION = "x-editor.round_operation_graph_validation.v1"

_PRIMARY_RESULT_KEYS = {
    "output_audio_path",
    "final_output_path",
    "audio_path",
    "path",
}

_PROCESS_ARTIFACT_TOKENS = {
    "alignment_report",
    "event_log",
    "judge",
    "perception",
    "perception_report",
    "process",
    "report",
    "rubric",
    "score",
    "scoring",
    "verification",
    "verification_report",
    "verifier",
}


class OperationAudioNode(SessionModel):
    """One useful planned or actual audio artifact inside an edit operation DAG."""

    node_id: str
    role: str = "audio"
    kind: str = "audio"
    path: str | None = None
    audio_ref_id: str | None = None
    artifact_id: str | None = None
    round_index: int | None = None
    step_id: str | None = None
    operation: str | None = None
    result_key: str | None = None
    parent_node_ids: list[str] = Field(default_factory=list)
    ancestor_node_ids: list[str] = Field(default_factory=list)
    operation_chain: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None
    objective_ids: list[str] = Field(default_factory=list)
    segment: dict[str, Any] | None = None
    split_group_id: str | None = None
    split_index: int | None = None
    split_role: str | None = None
    parallel_group_id: str | None = None
    barrier_reason: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationRecord(SessionModel):
    """Provenance record for an edit operation that produced or consumed nodes."""

    op_id: str
    step_id: str
    operation: str
    round_index: int | None = None
    reason: str | None = None
    objective_ids: list[str] = Field(default_factory=list)
    input_node_ids: list[str] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)
    status: str | None = None
    result_keys: list[str] = Field(default_factory=list)
    parallel_group_id: str | None = None
    barrier_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationGraphRecord(SessionModel):
    """Operation-level sidecar DAG for one edit plan or executed plan."""

    schema_version: str = OPERATION_GRAPH_SCHEMA_VERSION
    graph_id: str
    graph_kind: Literal["planned", "actual", "reconciled"]
    source: str | None = None
    input_node_ids: list[str] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)
    current_node_id: str | None = None
    nodes: list[OperationAudioNode] = Field(default_factory=list)
    operations: list[OperationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationStateNode(SessionModel):
    """A round-local audio state after zero or more operation outputs.

    The state node owns graph topology. Audio payload remains in
    ``OperationAudioNode`` so reports, verifier output, and scoring records do
    not become audio DAG nodes.
    """

    state_node_id: str
    round_index: int
    role: str = "state"
    parent_state_node_ids: list[str] = Field(default_factory=list)
    audio_node_ids: list[str] = Field(default_factory=list)
    audio_segments: list[dict[str, Any]] = Field(default_factory=list)
    producer_step_id: str | None = None
    operation: str | None = None
    reason: str | None = None
    status: str | None = None
    split_group_id: str | None = None
    branch_role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoundOperationGraphRecord(SessionModel):
    """One dialogue round's operation graph, with explicit source and target."""

    graph_id: str
    graph_kind: Literal["planned", "actual", "reconciled"]
    round_index: int
    source_state_node_id: str
    target_state_node_id: str
    state_nodes: list[OperationStateNode] = Field(default_factory=list)
    audio_nodes: list[OperationAudioNode] = Field(default_factory=list)
    operations: list[OperationRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoundOperationGraphStore(SessionModel):
    """Storage sidecar for round-scoped operation graphs."""

    schema_version: str = ROUND_OPERATION_GRAPH_STORE_SCHEMA_VERSION
    store_id: str
    graph_kind: Literal["planned", "actual", "reconciled"]
    rounds: list[RoundOperationGraphRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationGraphValidationReport(SessionModel):
    """Validation report for an operation graph projection."""

    schema_version: str = OPERATION_GRAPH_VALIDATION_SCHEMA_VERSION
    valid: bool
    issue_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)


class RoundOperationGraphValidationReport(SessionModel):
    """Validation report for round-scoped operation graph storage."""

    schema_version: str = ROUND_OPERATION_GRAPH_VALIDATION_SCHEMA_VERSION
    valid: bool
    issue_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)


def build_planned_operation_graph(plan: Any) -> OperationGraphRecord:
    """Build the planner-side pseudo graph, preserving LLM supplied reasons."""

    plan_data = _object_dump(plan)
    metadata = _mapping(plan_data.get("metadata"))
    raw = metadata.get("operation_graph_plan")
    if isinstance(raw, dict) and raw:
        return _normalize_raw_operation_graph(raw, plan_data=plan_data, graph_kind="planned")
    return _infer_planned_operation_graph(plan_data)


def build_actual_operation_graph(plan: Any) -> OperationGraphRecord:
    """Project executed step records and audio artifacts into an actual DAG."""

    plan_data = _object_dump(plan)
    metadata = _mapping(plan_data.get("metadata"))
    round_index = _round_index_from_plan_data(plan_data)
    planned = build_planned_operation_graph(plan_data) if isinstance(metadata.get("operation_graph_plan"), dict) else None
    planned_by_key = _planned_nodes_by_match_key(planned)
    step_rationales = _step_rationales_by_id(metadata)
    artifacts = _dict_list(metadata.get("audio_artifacts"))
    step_records = {str(item.get("step_id")): item for item in _dict_list(metadata.get("executor_step_records")) if item.get("step_id")}
    steps = _dict_list(plan_data.get("steps"))

    input_nodes: list[OperationAudioNode] = []
    path_to_node: dict[str, str] = {}
    seen_node_ids: set[str] = set()
    input_paths = _dedupe(
        [
            plan_data.get("input_audio"),
            *[
                item.get("path")
                for item in artifacts
                if item.get("kind") == "input" and item.get("path")
            ],
        ]
    )
    for index, path in enumerate(input_paths, start=1):
        node_id = "op_node:input" if index == 1 else f"op_node:input_{index}"
        node = OperationAudioNode(
            node_id=node_id,
            role="input_audio" if index == 1 else f"input_audio_{index}",
            kind="input",
            path=str(path),
            round_index=round_index,
            reason="Original input audio.",
            status="available",
            metadata=_without_empty({"source": "plan.input_audio" if index == 1 else "metadata.audio_artifacts"}),
        )
        input_nodes.append(node)
        seen_node_ids.add(node.node_id)
        path_to_node[_path_key(path)] = node.node_id

    useful_path_keys, useful_match_keys = _useful_artifact_keys(plan_data, artifacts, planned)
    nodes: list[OperationAudioNode] = list(input_nodes)
    operations: list[OperationRecord] = []
    node_by_id: dict[str, OperationAudioNode] = {node.node_id: node for node in nodes}
    step_output_nodes: dict[str, list[str]] = {}
    current_node_ids = [node.node_id for node in input_nodes[:1]]
    omitted_artifacts: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps, start=1):
        step_id = str(step.get("id") or f"step_{step_index}")
        operation = str(step.get("operation") or "")
        step_artifacts = [
            item for item in artifacts
            if item.get("kind") == "step_output" and str(item.get("step_id") or "") == step_id
        ]
        selected_artifacts: list[dict[str, Any]] = []
        for artifact in step_artifacts:
            path_key = _path_key(artifact.get("path"))
            if not useful_path_keys and not useful_match_keys:
                selected_artifacts.append(artifact)
            elif any(key in useful_match_keys for key in _artifact_match_keys(artifact)) or path_key in useful_path_keys:
                selected_artifacts.append(artifact)
            else:
                omitted_artifacts.append(_artifact_summary(artifact, reason="unused_split_output"))

        parent_node_ids = _parent_node_ids_for_step(
            step,
            step_output_nodes=step_output_nodes,
            path_to_node=path_to_node,
            current_node_ids=current_node_ids,
            input_node_ids=[node.node_id for node in input_nodes],
        )
        output_node_ids: list[str] = []
        op_reason = _step_reason(step, step_rationales.get(step_id))
        objective_ids = _step_objective_ids(step_rationales.get(step_id))
        split_group_id = f"split:{step_id}" if len(selected_artifacts) > 1 else None

        for output_index, artifact in enumerate(selected_artifacts, start=1):
            result_key = _string_or_none(artifact.get("result_key"))
            role = str(artifact.get("role") or result_key or "audio")
            planned_node = _match_planned_node(planned_by_key, artifact)
            planned_node_id = planned_node.node_id if planned_node else None
            node_id = planned_node_id or _unique_node_id(
                f"op_node:{_safe_id(step_id)}:{_safe_id(result_key or role or output_index)}",
                seen_node_ids,
            )
            seen_node_ids.add(node_id)
            chain_item = _operation_chain_item(
                step_id=step_id,
                operation=operation,
                reason=planned_node.reason if planned_node and planned_node.reason else op_reason,
                objective_ids=planned_node.objective_ids if planned_node and planned_node.objective_ids else objective_ids,
            )
            node = OperationAudioNode(
                node_id=node_id,
                role=role,
                kind=str(artifact.get("kind") or "step_output"),
                path=_string_or_none(artifact.get("path")),
                artifact_id=_string_or_none(artifact.get("artifact_id")),
                round_index=round_index,
                step_id=step_id,
                operation=operation,
                result_key=result_key,
                parent_node_ids=parent_node_ids,
                ancestor_node_ids=_ancestor_node_ids(parent_node_ids, node_by_id),
                operation_chain=_operation_chain(parent_node_ids, node_by_id, chain_item),
                reason=planned_node.reason if planned_node and planned_node.reason else op_reason,
                objective_ids=planned_node.objective_ids if planned_node and planned_node.objective_ids else objective_ids,
                segment=_segment_from_artifact_or_step(artifact, step),
                split_group_id=(planned_node.split_group_id if planned_node else None) or split_group_id,
                split_index=(planned_node.split_index if planned_node and planned_node.split_index is not None else (output_index if split_group_id else None)),
                split_role=(planned_node.split_role if planned_node else None) or (role if split_group_id else None),
                parallel_group_id=planned_node.parallel_group_id if planned_node else None,
                barrier_reason=planned_node.barrier_reason if planned_node else None,
                status=_string_or_none(step_records.get(step_id, {}).get("status") or step.get("status")),
                metadata=_without_empty(
                    {
                        "planned_node_id": planned_node_id,
                        "source": artifact.get("source"),
                        "expected_output": artifact.get("expected_output"),
                        "is_final_output": artifact.get("is_final_output"),
                        "duration_sec": artifact.get("duration_sec"),
                    }
                ),
            )
            nodes.append(node)
            node_by_id[node_id] = node
            output_node_ids.append(node_id)
            if artifact.get("path"):
                path_to_node[_path_key(artifact.get("path"))] = node_id

        record = step_records.get(step_id, {})
        result_keys = _string_list(record.get("result_keys"))
        if not result_keys and isinstance(step.get("result"), dict):
            result_keys = sorted(str(key) for key in step["result"].keys())
        operations.append(
            OperationRecord(
                op_id=f"op:{step_id}",
                step_id=step_id,
                operation=operation,
                round_index=round_index,
                reason=op_reason,
                objective_ids=objective_ids,
                input_node_ids=parent_node_ids,
                output_node_ids=output_node_ids,
                status=_string_or_none(record.get("status") or step.get("status")),
                result_keys=result_keys,
                parallel_group_id=_parallel_group_for_step(step, metadata),
                barrier_reason=_barrier_reason_for_step(step, metadata),
                metadata=_without_empty(
                    {
                        "step_index": step_index,
                        "description": step.get("description"),
                        "current_audio_before": record.get("current_audio_before"),
                        "current_audio_after": record.get("current_audio_after"),
                    }
                ),
            )
        )
        step_output_nodes[step_id] = output_node_ids
        if output_node_ids:
            current_node_ids = output_node_ids

    output_node_ids = _graph_output_node_ids(nodes, operations, explicit=_string_list(metadata.get("operation_graph_outputs")))
    current_node_id = output_node_ids[-1] if output_node_ids else (current_node_ids[-1] if current_node_ids else None)
    return OperationGraphRecord(
        graph_id=_graph_id(plan_data, "actual"),
        graph_kind="actual",
        source="executor.audio_artifacts",
        input_node_ids=[node.node_id for node in input_nodes],
        output_node_ids=output_node_ids,
        current_node_id=current_node_id,
        nodes=nodes,
        operations=operations,
        metadata=_without_empty(
            {
                "request_hash": _request_hash(plan_data),
                "round_index": round_index,
                "omitted_audio_artifacts": omitted_artifacts,
                "omitted_audio_artifact_count": len(omitted_artifacts),
            }
        ),
    )


def build_round_operation_graph_store(
    record: OperationGraphRecord | dict[str, Any],
    *,
    round_index: int | None = None,
    graph_kind: Literal["planned", "actual", "reconciled"] | None = None,
    source: str | None = None,
) -> RoundOperationGraphStore:
    """Build round-scoped state storage from a single-round operation graph.

    ``OperationGraphRecord`` remains the executor's lightweight dependency DAG.
    This store is a separate projection for round source/target boundaries,
    state nodes, frontend/debug views, and stronger validation.
    """

    graph = _coerce_graph(record)
    round_id = int(round_index or _round_index_from_graph(graph) or 1)
    source_state_id = f"round_{round_id}:source"
    target_state_id = f"round_{round_id}:target"
    graph_kind_value = graph_kind or graph.graph_kind
    audio_nodes = [
        node.model_copy(update={"round_index": node.round_index if node.round_index is not None else round_id})
        for node in graph.nodes
    ]
    audio_by_id = {node.node_id: node for node in audio_nodes}
    state_nodes: list[OperationStateNode] = []
    state_by_id: dict[str, OperationStateNode] = {}
    audio_to_state: dict[str, str] = {}

    input_node_ids = [node_id for node_id in graph.input_node_ids if node_id in audio_by_id]
    source_state = OperationStateNode(
        state_node_id=source_state_id,
        round_index=round_id,
        role="source",
        audio_node_ids=input_node_ids,
        audio_segments=[_audio_segment_summary(audio_by_id[node_id]) for node_id in input_node_ids],
        status="available",
        metadata=_without_empty({"operation_graph_id": graph.graph_id}),
    )
    state_nodes.append(source_state)
    state_by_id[source_state.state_node_id] = source_state
    for node_id in input_node_ids:
        audio_to_state[node_id] = source_state_id

    for operation in graph.operations:
        output_node_ids = [node_id for node_id in operation.output_node_ids if node_id in audio_by_id]
        if not output_node_ids:
            continue
        parent_state_ids = _dedupe(
            audio_to_state.get(node_id) or source_state_id
            for node_id in operation.input_node_ids
            if node_id in audio_by_id or node_id in audio_to_state
        ) or [source_state_id]
        split_group_id = f"split:{operation.step_id}" if len(output_node_ids) > 1 else None
        for output_node_id in output_node_ids:
            audio_node = audio_by_id[output_node_id]
            state_id = _state_id_for_audio_node(
                round_index=round_id,
                operation=operation,
                audio_node=audio_node,
                existing_ids=set(state_by_id),
            )
            state = OperationStateNode(
                state_node_id=state_id,
                round_index=round_id,
                role="state",
                parent_state_node_ids=parent_state_ids,
                audio_node_ids=[output_node_id],
                audio_segments=[_audio_segment_summary(audio_node)],
                producer_step_id=operation.step_id,
                operation=operation.operation,
                reason=operation.reason or audio_node.reason,
                status=operation.status or audio_node.status,
                split_group_id=audio_node.split_group_id or split_group_id,
                branch_role=audio_node.split_role or audio_node.role,
                metadata=_without_empty(
                    {
                        "operation_graph_node_id": output_node_id,
                        "operation_graph_id": graph.graph_id,
                    }
                ),
            )
            state_nodes.append(state)
            state_by_id[state.state_node_id] = state
            audio_to_state[output_node_id] = state_id

    target_parent_ids = _dedupe(
        audio_to_state.get(node_id)
        for node_id in graph.output_node_ids
        if audio_to_state.get(node_id)
    )
    if not target_parent_ids:
        leaf_state_ids = _round_leaf_state_ids(state_nodes, source_state_id)
        target_parent_ids = leaf_state_ids or [source_state_id]
    target_audio_node_ids = [node_id for node_id in graph.output_node_ids if node_id in audio_by_id]
    target_state = OperationStateNode(
        state_node_id=target_state_id,
        round_index=round_id,
        role="target",
        parent_state_node_ids=target_parent_ids,
        audio_node_ids=target_audio_node_ids,
        audio_segments=[_audio_segment_summary(audio_by_id[node_id]) for node_id in target_audio_node_ids],
        status="available" if target_audio_node_ids else None,
        metadata=_without_empty({"operation_graph_id": graph.graph_id, "current_node_id": graph.current_node_id}),
    )
    state_nodes.append(target_state)

    round_record = RoundOperationGraphRecord(
        graph_id=f"{graph.graph_id}:round:{round_id}",
        graph_kind=graph_kind_value,
        round_index=round_id,
        source_state_node_id=source_state_id,
        target_state_node_id=target_state_id,
        state_nodes=state_nodes,
        audio_nodes=audio_nodes,
        operations=[
            operation.model_copy(
                update={"round_index": operation.round_index if operation.round_index is not None else round_id}
            )
            for operation in graph.operations
        ],
        metadata=_without_empty(
            {
                "operation_graph_id": graph.graph_id,
                "operation_graph_source": graph.source,
                "source": source,
            }
        ),
    )
    return RoundOperationGraphStore(
        store_id=f"{graph.graph_id}:round_store",
        graph_kind=graph_kind_value,
        rounds=[round_record],
        metadata=_without_empty(
            {
                "source": source or "operation_graph_projection",
                "operation_graph_id": graph.graph_id,
            }
        ),
    )


def build_operation_graph_event_log(
    plan: Any,
    *,
    round_store: RoundOperationGraphStore | dict[str, Any] | None = None,
    actual_graph: OperationGraphRecord | dict[str, Any] | None = None,
) -> SessionEventLog:
    """Build an append-only event log for execution facts and projections."""

    plan_data = _object_dump(plan)
    metadata = _mapping(plan_data.get("metadata"))
    round_index = _round_index_from_plan_data(plan_data) or 1
    request_hash = _request_hash(plan_data)
    store = _coerce_round_store(round_store) if round_store is not None else None
    graph = _coerce_graph(actual_graph) if actual_graph is not None else None
    events: list[SessionEvent] = []

    def append(event_type: str, **kwargs: Any) -> None:
        event_id = f"event:{request_hash}:r{round_index}:{len(events) + 1:04d}:{event_type}"
        events.append(SessionEvent(event_id=event_id, event_type=event_type, **kwargs))

    append(
        "user_intent_recorded",
        round_index=round_index,
        actor="user",
        summary=_string_or_none(plan_data.get("request")),
        payload=_without_empty({"input_audio": plan_data.get("input_audio"), "output_audio": plan_data.get("output_audio")}),
    )
    append(
        "plan_recorded",
        round_index=round_index,
        actor="planner",
        summary="AudioEditPlan recorded.",
        refs={"steps": [str(step.get("id")) for step in _dict_list(plan_data.get("steps")) if step.get("id")]},
        payload=_without_empty(
            {
                "step_count": len(_dict_list(plan_data.get("steps"))),
                "has_operation_graph_plan": isinstance(metadata.get("operation_graph_plan"), dict),
            }
        ),
    )
    execution = _mapping(metadata.get("operation_graph_execution"))
    if execution:
        append(
            "operation_graph_execution_recorded",
            round_index=round_index,
            actor="executor",
            summary="OperationGraph scheduler execution recorded.",
            payload=_without_empty(
                {
                    "scheduler": execution.get("scheduler"),
                    "fallback_to_original": execution.get("fallback_to_original"),
                    "fallback_reason": execution.get("fallback_reason"),
                    "binding_count": len(_dict_list(execution.get("bindings"))),
                    "node_path_update_count": len(_dict_list(execution.get("node_path_updates"))),
                    "round_execution_plan": execution.get("round_execution_plan"),
                    "scheduler_dependencies": execution.get("scheduler_dependencies"),
                }
            ),
        )
    for loop in _dict_list(metadata.get("executor_loops")):
        append(
            "executor_loop_recorded",
            round_index=round_index,
            actor="executor",
            summary=f"Executor loop {loop.get('loop_index')} recorded.",
            payload=_without_empty(
                {
                    "loop_index": loop.get("loop_index"),
                    "scheduler": loop.get("scheduler"),
                    "ready_steps": loop.get("ready_steps"),
                    "executed_steps": loop.get("executed_steps"),
                    "failed_steps": loop.get("failed_steps"),
                    "pseudo_parallel": loop.get("pseudo_parallel"),
                }
            ),
        )
    for record in _dict_list(metadata.get("executor_step_records")):
        step_id = _string_or_none(record.get("step_id"))
        append(
            "operation_step_recorded",
            round_index=round_index,
            step_id=step_id,
            actor="executor",
            summary=_string_or_none(record.get("operation") or step_id),
            payload=_without_empty(
                {
                    "status": record.get("status"),
                    "operation": record.get("operation"),
                    "current_audio_before": record.get("current_audio_before"),
                    "current_audio_after": record.get("current_audio_after"),
                    "output_path": record.get("output_path"),
                    "result_keys": record.get("result_keys"),
                    "error_message": record.get("error_message"),
                }
            ),
        )
    for artifact in _dict_list(metadata.get("audio_artifacts")):
        append(
            "audio_artifact_recorded",
            round_index=round_index,
            step_id=_string_or_none(artifact.get("step_id")),
            actor="executor",
            summary=_string_or_none(artifact.get("role") or artifact.get("artifact_id")),
            refs={"artifacts": _dedupe([_string_or_none(artifact.get("artifact_id"))])},
            payload=_without_empty(
                {
                    "artifact_id": artifact.get("artifact_id"),
                    "kind": artifact.get("kind"),
                    "role": artifact.get("role"),
                    "path": artifact.get("path"),
                    "result_key": artifact.get("result_key"),
                    "is_final_output": artifact.get("is_final_output"),
                }
            ),
        )
    if graph is not None:
        append(
            "operation_graph_actual_projected",
            round_index=round_index,
            actor="runtime",
            summary="OperationGraph actual projection recorded.",
            refs={"operation_nodes": [node.node_id for node in graph.nodes]},
            payload=_without_empty(
                {
                    "graph_id": graph.graph_id,
                    "node_count": len(graph.nodes),
                    "operation_count": len(graph.operations),
                    "output_node_ids": graph.output_node_ids,
                    "current_node_id": graph.current_node_id,
                }
            ),
        )
    if store is not None:
        round_records = store.rounds
        append(
            "round_operation_graph_projected",
            round_index=round_index,
            actor="runtime",
            summary="Round-scoped OperationGraph store projected.",
            refs={
                "state_nodes": [
                    state.state_node_id
                    for round_record in round_records
                    for state in round_record.state_nodes
                ],
                "operation_nodes": [
                    audio.node_id
                    for round_record in round_records
                    for audio in round_record.audio_nodes
                ],
            },
            payload=_without_empty(
                {
                    "store_id": store.store_id,
                    "round_count": len(round_records),
                    "round_indexes": [round_record.round_index for round_record in round_records],
                }
            ),
        )
    for key, event_type in (
        ("operation_graph_validation", "operation_graph_validation_recorded"),
        ("operation_graph_round_validation", "operation_graph_round_validation_recorded"),
        ("operation_graph_reconciliation", "operation_graph_reconciliation_recorded"),
    ):
        report = _mapping(metadata.get(key))
        if not report:
            continue
        append(
            event_type,
            round_index=round_index,
            actor="runtime",
            summary=f"{key} recorded.",
            payload=_without_empty(
                {
                    "valid": report.get("valid"),
                    "issue_counts": report.get("issue_counts"),
                    "matched_nodes": report.get("matched_nodes"),
                    "planned_only_nodes": report.get("planned_only_nodes"),
                    "actual_only_nodes": report.get("actual_only_nodes"),
                    "path_drift": report.get("path_drift"),
                    "issues": report.get("issues"),
                }
            ),
        )
    for key, event_type, actor in (
        ("verification_reports", "verification_recorded", "verifier"),
        ("verification_repair_plans", "verification_repair_plan_recorded", "verifier"),
        ("bounded_repair_attempts", "bounded_repair_recorded", "executor"),
        ("service_bootstrap", "service_bootstrap_recorded", "runtime"),
    ):
        rows = _dict_list(metadata.get(key))
        if not rows:
            continue
        append(
            event_type,
            round_index=round_index,
            actor=actor,
            summary=f"{key} recorded.",
            payload={key: rows},
        )
    return SessionEventLog(
        event_log_id=f"event_log:{request_hash}:round:{round_index}",
        round_index=round_index,
        events=events,
        metadata=_without_empty(
            {
                "request_hash": request_hash,
                "source": "executor.operation_graph_metadata",
                "event_count": len(events),
            }
        ),
    )


def validate_operation_graph(
    record: OperationGraphRecord | dict[str, Any],
    *,
    plan: Any | None = None,
) -> OperationGraphValidationReport:
    """Validate operation graph DAG shape and planner/step references."""

    graph = _coerce_graph(record)
    issues: list[ValidationIssue] = []
    node_ids = [node.node_id for node in graph.nodes]
    counts = Counter(node_ids)
    for node_id, count in counts.items():
        if count > 1:
            issues.append(_issue("error", "duplicate_node_id", "Operation graph node_id appears more than once.", location="nodes", ref_id=node_id))
    node_by_id = {node.node_id: node for node in graph.nodes}
    known_step_ids = _step_ids_from_plan(plan)

    for index, node in enumerate(graph.nodes):
        process_issue = _process_artifact_issue(node, location=f"nodes[{index}]")
        if process_issue is not None:
            issues.append(process_issue)
        if graph.graph_kind == "planned" and not node.reason:
            issues.append(_issue("warning", "planned_node_reason_missing", "Planned operation graph node has no reason.", location=f"nodes[{index}].reason", ref_id=node.node_id))
        if graph.graph_kind == "actual" and node.kind != "input" and not node.path:
            issues.append(_issue("warning", "actual_node_path_missing", "Actual operation graph node has no audio path.", location=f"nodes[{index}].path", ref_id=node.node_id))
        if known_step_ids and node.step_id and node.step_id not in known_step_ids:
            issues.append(_issue("warning", "node_unknown_step", "Operation graph node references a step not present in the plan.", location=f"nodes[{index}].step_id", ref_id=node.node_id, metadata={"step_id": node.step_id}))
        for parent_id in node.parent_node_ids:
            if parent_id not in node_by_id:
                issues.append(_issue("error", "dangling_parent_node", "Operation graph node references a missing parent node.", location=f"nodes[{index}].parent_node_ids", ref_id=node.node_id, metadata={"parent_node_id": parent_id}))
            if parent_id == node.node_id:
                issues.append(_issue("error", "self_parent_node", "Operation graph node cannot be its own parent.", location=f"nodes[{index}].parent_node_ids", ref_id=node.node_id))
            parent_node = node_by_id.get(parent_id)
            if (
                parent_node is not None
                and node.round_index is not None
                and parent_node.round_index is not None
                and node.round_index != parent_node.round_index
            ):
                issues.append(
                    _issue(
                        "error",
                        "cross_round_parent_node",
                        "Operation graph audio node cannot depend on a parent from another dialogue round.",
                        location=f"nodes[{index}].parent_node_ids",
                        ref_id=node.node_id,
                        metadata={"parent_node_id": parent_id, "round_index": node.round_index, "parent_round_index": parent_node.round_index},
                    )
                )

    for index, operation in enumerate(graph.operations):
        if graph.graph_kind == "planned" and not operation.reason:
            issues.append(_issue("warning", "planned_operation_reason_missing", "Planned operation has no reason.", location=f"operations[{index}].reason", ref_id=operation.op_id))
        if known_step_ids and operation.step_id not in known_step_ids:
            issues.append(_issue("warning", "operation_unknown_step", "Operation references a step not present in the plan.", location=f"operations[{index}].step_id", ref_id=operation.op_id, metadata={"step_id": operation.step_id}))
        for input_id in operation.input_node_ids:
            if input_id not in node_by_id:
                issues.append(_issue("error", "operation_input_node_missing", "Operation input_node_ids contains a missing node.", location=f"operations[{index}].input_node_ids", ref_id=operation.op_id, metadata={"input_node_id": input_id}))
        for output_id in operation.output_node_ids:
            if output_id not in node_by_id:
                issues.append(_issue("error", "operation_output_node_missing", "Operation output_node_ids contains a missing node.", location=f"operations[{index}].output_node_ids", ref_id=operation.op_id, metadata={"output_node_id": output_id}))

    for output_id in graph.output_node_ids:
        if output_id not in node_by_id:
            issues.append(_issue("error", "graph_output_node_missing", "Graph output_node_ids contains a missing node.", location="output_node_ids", ref_id=output_id))
    if graph.current_node_id and graph.current_node_id not in node_by_id:
        issues.append(_issue("error", "graph_current_node_missing", "Graph current_node_id is not present in nodes.", location="current_node_id", ref_id=graph.current_node_id))
    issues.extend(_cycle_issues(graph))

    severity_counts = Counter(issue.severity for issue in issues)
    return OperationGraphValidationReport(
        valid=severity_counts.get("error", 0) == 0,
        issue_counts=dict(sorted(severity_counts.items())),
        issues=issues,
    )


def validate_round_operation_graph_store(
    record: RoundOperationGraphStore | dict[str, Any],
) -> RoundOperationGraphValidationReport:
    """Validate round-local state graph boundaries and audio refs."""

    store = _coerce_round_store(record)
    issues: list[ValidationIssue] = []
    round_indexes = [round_record.round_index for round_record in store.rounds]
    for round_index, count in Counter(round_indexes).items():
        if count > 1:
            issues.append(_issue("error", "duplicate_round_index", "Round operation graph store has duplicate round_index.", location="rounds", ref_id=str(round_index)))

    for round_pos, round_record in enumerate(store.rounds):
        location = f"rounds[{round_pos}]"
        state_by_id = {node.state_node_id: node for node in round_record.state_nodes}
        audio_by_id = {node.node_id: node for node in round_record.audio_nodes}
        if round_record.source_state_node_id not in state_by_id:
            issues.append(_issue("error", "round_source_state_missing", "Round operation graph is missing its source state node.", location=f"{location}.source_state_node_id", ref_id=round_record.source_state_node_id))
        if round_record.target_state_node_id not in state_by_id:
            issues.append(_issue("error", "round_target_state_missing", "Round operation graph is missing its target state node.", location=f"{location}.target_state_node_id", ref_id=round_record.target_state_node_id))

        for audio_index, audio_node in enumerate(round_record.audio_nodes):
            process_issue = _process_artifact_issue(audio_node, location=f"{location}.audio_nodes[{audio_index}]")
            if process_issue is not None:
                issues.append(process_issue)
            if audio_node.round_index is not None and audio_node.round_index != round_record.round_index:
                issues.append(
                    _issue(
                        "error",
                        "audio_node_round_mismatch",
                        "Round operation graph audio node belongs to a different dialogue round.",
                        location=f"{location}.audio_nodes[{audio_index}].round_index",
                        ref_id=audio_node.node_id,
                        metadata={"round_index": round_record.round_index, "audio_node_round_index": audio_node.round_index},
                    )
                )

        for state_index, state_node in enumerate(round_record.state_nodes):
            state_location = f"{location}.state_nodes[{state_index}]"
            if state_node.round_index != round_record.round_index:
                issues.append(
                    _issue(
                        "error",
                        "state_node_round_mismatch",
                        "Round operation graph state node belongs to a different dialogue round.",
                        location=f"{state_location}.round_index",
                        ref_id=state_node.state_node_id,
                        metadata={"round_index": round_record.round_index, "state_node_round_index": state_node.round_index},
                    )
                )
            if state_node.state_node_id == round_record.source_state_node_id and state_node.parent_state_node_ids:
                issues.append(_issue("error", "round_source_has_parent", "Round source state node must not have parents.", location=f"{state_location}.parent_state_node_ids", ref_id=state_node.state_node_id))
            for parent_id in state_node.parent_state_node_ids:
                parent = state_by_id.get(parent_id)
                if parent is None:
                    issues.append(
                        _issue(
                            "error",
                            "dangling_parent_state_node",
                            "Round operation graph state node references a missing parent state node.",
                            location=f"{state_location}.parent_state_node_ids",
                            ref_id=state_node.state_node_id,
                            metadata={"parent_state_node_id": parent_id},
                        )
                    )
                    continue
                if parent.round_index != state_node.round_index:
                    issues.append(
                        _issue(
                            "error",
                            "cross_round_state_parent",
                            "Round operation graph state edge cannot cross dialogue rounds.",
                            location=f"{state_location}.parent_state_node_ids",
                            ref_id=state_node.state_node_id,
                            metadata={"parent_state_node_id": parent_id, "round_index": state_node.round_index, "parent_round_index": parent.round_index},
                        )
                    )
                if parent_id == state_node.state_node_id:
                    issues.append(_issue("error", "self_parent_state_node", "Round operation graph state node cannot be its own parent.", location=f"{state_location}.parent_state_node_ids", ref_id=state_node.state_node_id))
            for audio_node_id in state_node.audio_node_ids:
                if audio_node_id not in audio_by_id:
                    issues.append(
                        _issue(
                            "error",
                            "state_audio_node_missing",
                            "Round operation graph state node references a missing audio node.",
                            location=f"{state_location}.audio_node_ids",
                            ref_id=state_node.state_node_id,
                            metadata={"audio_node_id": audio_node_id},
                        )
                    )

        for op_index, operation in enumerate(round_record.operations):
            op_location = f"{location}.operations[{op_index}]"
            if operation.round_index is not None and operation.round_index != round_record.round_index:
                issues.append(
                    _issue(
                        "error",
                        "operation_round_mismatch",
                        "Round operation graph operation belongs to a different dialogue round.",
                        location=f"{op_location}.round_index",
                        ref_id=operation.op_id,
                        metadata={"round_index": round_record.round_index, "operation_round_index": operation.round_index},
                    )
                )
            for node_id in [*operation.input_node_ids, *operation.output_node_ids]:
                if node_id not in audio_by_id:
                    issues.append(
                        _issue(
                            "error",
                            "operation_audio_node_missing",
                            "Round operation graph operation references a missing audio node.",
                            location=op_location,
                            ref_id=operation.op_id,
                            metadata={"audio_node_id": node_id},
                        )
                    )
        issues.extend(_round_state_cycle_issues(round_record, location=location))

    severity_counts = Counter(issue.severity for issue in issues)
    return RoundOperationGraphValidationReport(
        valid=severity_counts.get("error", 0) == 0,
        issue_counts=dict(sorted(severity_counts.items())),
        issues=issues,
    )


def reconcile_planned_actual_graph(
    planned: OperationGraphRecord | dict[str, Any],
    actual: OperationGraphRecord | dict[str, Any],
) -> dict[str, Any]:
    """Align a planned pseudo graph with actual artifacts and keep useful diffs."""

    planned_graph = _coerce_graph(planned)
    actual_graph = _coerce_graph(actual)
    planned_by_key = _planned_nodes_by_match_key(planned_graph)
    matched: list[dict[str, str]] = []
    actual_nodes: list[OperationAudioNode] = []
    matched_planned_ids: set[str] = set()
    path_drift: list[dict[str, Any]] = []

    for node in actual_graph.nodes:
        planned_node = _match_planned_node(planned_by_key, _node_as_artifact(node))
        if planned_node:
            matched_planned_ids.add(planned_node.node_id)
            matched.append({"planned_node_id": planned_node.node_id, "actual_node_id": node.node_id})
            if planned_node.path and node.path and _path_key(planned_node.path) != _path_key(node.path):
                path_drift.append(
                    {
                        "planned_node_id": planned_node.node_id,
                        "actual_node_id": node.node_id,
                        "planned_path": planned_node.path,
                        "actual_path": node.path,
                    }
                )
            node = node.model_copy(
                update={
                    "reason": node.reason or planned_node.reason,
                    "objective_ids": node.objective_ids or planned_node.objective_ids,
                    "parallel_group_id": node.parallel_group_id or planned_node.parallel_group_id,
                    "barrier_reason": node.barrier_reason or planned_node.barrier_reason,
                    "metadata": {
                        **node.metadata,
                        **_without_empty({"planned_node_id": planned_node.node_id}),
                    },
                }
            )
        actual_nodes.append(node)

    enriched_actual = actual_graph.model_copy(update={"nodes": actual_nodes})
    planned_only = [
        node.node_id
        for node in planned_graph.nodes
        if node.node_id not in matched_planned_ids and node.kind != "input"
    ]
    actual_only = [
        node.node_id
        for node in enriched_actual.nodes
        if not _match_planned_node(planned_by_key, _node_as_artifact(node)) and node.kind != "input"
    ]
    issues: list[ValidationIssue] = []
    for node_id in planned_only:
        issues.append(_issue("warning", "planned_only_node", "Planned operation graph node was not produced by execution.", ref_id=node_id))
    for node_id in actual_only:
        issues.append(_issue("info", "actual_only_node", "Actual operation graph node was not present in the planner pseudo graph.", ref_id=node_id))
    for item in path_drift:
        issues.append(_issue("warning", "path_drift", "Planned and actual node paths differ.", ref_id=item["actual_node_id"], metadata=item))
    for item in _mapping(enriched_actual.metadata).get("omitted_audio_artifacts") or []:
        if isinstance(item, dict):
            issues.append(_issue("info", "unused_split_output", "Execution produced an audio artifact that was omitted from the useful operation graph.", metadata=item))

    severity_counts = Counter(issue.severity for issue in issues)
    return {
        "schema_version": OPERATION_GRAPH_RECONCILIATION_SCHEMA_VERSION,
        "valid": severity_counts.get("error", 0) == 0,
        "issue_counts": dict(sorted(severity_counts.items())),
        "issues": [_model_row(issue) for issue in issues],
        "matched_nodes": matched,
        "planned_only_nodes": planned_only,
        "actual_only_nodes": actual_only,
        "path_drift": path_drift,
        "actual_graph": _model_row(enriched_actual),
    }


def _normalize_raw_operation_graph(
    raw: dict[str, Any],
    *,
    plan_data: dict[str, Any],
    graph_kind: Literal["planned", "actual", "reconciled"],
) -> OperationGraphRecord:
    steps = _dict_list(plan_data.get("steps"))
    step_by_id = {str(step.get("id") or f"step_{index + 1}"): step for index, step in enumerate(steps)}
    rationales = _step_rationales_by_id(_mapping(plan_data.get("metadata")))
    default_round_index = _int_or_none(raw.get("round_index")) or _round_index_from_plan_data(plan_data)
    raw_nodes = raw.get("nodes") or raw.get("audio_nodes") or raw.get("operation_nodes") or []
    if isinstance(raw_nodes, dict):
        node_rows = [item for item in raw_nodes.values() if isinstance(item, dict)]
    else:
        node_rows = _dict_list(raw_nodes)
    raw_ops = raw.get("operations") or raw.get("operation_records") or []
    if isinstance(raw_ops, dict):
        op_rows = [item for item in raw_ops.values() if isinstance(item, dict)]
    else:
        op_rows = _dict_list(raw_ops)

    nodes: list[OperationAudioNode] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(node_rows, start=1):
        step_id = _string_or_none(row.get("step_id") or row.get("producer_step_id") or row.get("expected_producer_step_id"))
        step = step_by_id.get(step_id or "")
        result_key = _string_or_none(row.get("result_key") or row.get("expected_result_key"))
        node_id = _unique_node_id(
            str(row.get("node_id") or row.get("id") or f"op_node:{_safe_id(step_id or 'planned')}:{_safe_id(result_key or index)}"),
            seen_ids,
        )
        seen_ids.add(node_id)
        rationale = rationales.get(step_id or "")
        nodes.append(
            OperationAudioNode(
                node_id=node_id,
                role=str(row.get("role") or row.get("planned_role") or result_key or "audio"),
                kind=str(row.get("kind") or "audio"),
                path=_string_or_none(row.get("path") or row.get("expected_path")),
                audio_ref_id=_string_or_none(row.get("audio_ref_id")),
                artifact_id=_string_or_none(row.get("artifact_id")),
                round_index=_int_or_none(row.get("round_index")),
                step_id=step_id,
                operation=_string_or_none(row.get("operation") or (step or {}).get("operation")),
                result_key=result_key,
                parent_node_ids=_string_list(row.get("parent_node_ids") or row.get("parents") or row.get("input_node_ids")),
                ancestor_node_ids=_string_list(row.get("ancestor_node_ids")),
                operation_chain=_dict_list(row.get("operation_chain")),
                reason=_string_or_none(row.get("reason") or row.get("node_reason") or row.get("producer_reason")),
                objective_ids=_string_list(row.get("objective_ids") or _step_objective_ids(rationale)),
                segment=_mapping_or_none(row.get("segment") or row.get("span")),
                split_group_id=_string_or_none(row.get("split_group_id")),
                split_index=_int_or_none(row.get("split_index")),
                split_role=_string_or_none(row.get("split_role")),
                parallel_group_id=_string_or_none(row.get("parallel_group_id")),
                barrier_reason=_string_or_none(row.get("barrier_reason")),
                status=_string_or_none(row.get("status")),
                metadata=_mapping(row.get("metadata")),
            )
        )
    raw_input_node_ids = _dedupe(
        [
            *_string_list(raw.get("input_node_ids") or raw.get("inputs")),
            *[
                node_id
                for node_id in _referenced_operation_graph_node_ids(node_rows=node_rows, op_rows=op_rows)
                if _looks_like_input_node_id(node_id)
            ],
        ]
    )
    synthesized_inputs: list[OperationAudioNode] = []
    for index, node_id in enumerate(raw_input_node_ids, start=1):
        if node_id in seen_ids:
            continue
        path = _input_audio_path_for_index(plan_data, index - 1)
        synthesized_inputs.append(
            OperationAudioNode(
                node_id=node_id,
                role="input_audio" if index == 1 else f"input_audio_{index}",
                kind="input",
                path=path,
                round_index=default_round_index,
                reason="Original input audio.",
                status="planned" if graph_kind == "planned" else None,
                metadata=_without_empty({"source": "operation_graph.input_node_ids"}),
            )
        )
        seen_ids.add(node_id)
    if synthesized_inputs:
        nodes = [*synthesized_inputs, *nodes]

    operations = [
        OperationRecord(
            op_id=str(row.get("op_id") or row.get("id") or f"op:{row.get('step_id') or index}"),
            step_id=str(row.get("step_id") or ""),
            operation=str(row.get("operation") or (step_by_id.get(str(row.get("step_id") or "")) or {}).get("operation") or ""),
            round_index=_int_or_none(row.get("round_index")),
            reason=_string_or_none(row.get("reason")),
            objective_ids=_string_list(row.get("objective_ids")),
            input_node_ids=_string_list(row.get("input_node_ids") or row.get("inputs")),
            output_node_ids=_string_list(row.get("output_node_ids") or row.get("outputs")),
            status=_string_or_none(row.get("status")),
            result_keys=_string_list(row.get("result_keys")),
            parallel_group_id=_string_or_none(row.get("parallel_group_id")),
            barrier_reason=_string_or_none(row.get("barrier_reason")),
            metadata=_mapping(row.get("metadata")),
        )
        for index, row in enumerate(op_rows, start=1)
    ]
    if not operations:
        operations = _operations_from_nodes_and_steps(nodes, steps, rationales)

    node_by_id = {node.node_id: node for node in nodes}
    nodes = [_node_with_derived_lineage(node, node_by_id, operations) for node in nodes]
    input_node_ids = raw_input_node_ids or [
        node.node_id for node in nodes if not node.parent_node_ids
    ]
    output_node_ids = _string_list(raw.get("output_node_ids") or raw.get("outputs")) or _graph_output_node_ids(nodes, operations)
    return OperationGraphRecord(
        graph_id=str(raw.get("graph_id") or _graph_id(plan_data, graph_kind)),
        graph_kind=graph_kind,
        source=_string_or_none(raw.get("source") or "planner.operation_graph_plan"),
        input_node_ids=input_node_ids,
        output_node_ids=output_node_ids,
        current_node_id=_string_or_none(raw.get("current_node_id")) or (output_node_ids[-1] if output_node_ids else None),
        nodes=nodes,
        operations=operations,
        metadata=_without_empty({"round_index": default_round_index, **_mapping(raw.get("metadata"))}),
    )


def _infer_planned_operation_graph(plan_data: dict[str, Any]) -> OperationGraphRecord:
    metadata = _mapping(plan_data.get("metadata"))
    round_index = _round_index_from_plan_data(plan_data)
    steps = _dict_list(plan_data.get("steps"))
    rationales = _step_rationales_by_id(metadata)
    nodes: list[OperationAudioNode] = []
    operations: list[OperationRecord] = []
    node_by_id: dict[str, OperationAudioNode] = {}
    step_output_nodes: dict[str, list[str]] = {}
    current_node_ids: list[str] = []
    if plan_data.get("input_audio"):
        input_node = OperationAudioNode(
            node_id="op_node:input",
            role="input_audio",
            kind="input",
            path=str(plan_data["input_audio"]),
            round_index=round_index,
            reason="Original input audio.",
            status="planned",
        )
        nodes.append(input_node)
        node_by_id[input_node.node_id] = input_node
        current_node_ids = [input_node.node_id]

    for index, step in enumerate(steps, start=1):
        step_id = str(step.get("id") or f"step_{index}")
        rationale = rationales.get(step_id)
        parent_node_ids = _dedupe(
            [
                parent
                for dep in _string_list(step.get("depends_on"))
                for parent in step_output_nodes.get(dep, [])
            ]
        ) or list(current_node_ids)
        operation = str(step.get("operation") or "")
        result_key = "output_audio_path"
        node_id = _unique_node_id(f"op_node:{_safe_id(step_id)}:{result_key}", set(node_by_id))
        reason = _step_reason(step, rationale)
        objective_ids = _step_objective_ids(rationale)
        chain_item = _operation_chain_item(step_id=step_id, operation=operation, reason=reason, objective_ids=objective_ids)
        node = OperationAudioNode(
            node_id=node_id,
            role=str(step.get("expected_output") or result_key),
            kind="planned_output",
            step_id=step_id,
            operation=operation,
            round_index=round_index,
            result_key=result_key,
            parent_node_ids=parent_node_ids,
            ancestor_node_ids=_ancestor_node_ids(parent_node_ids, node_by_id),
            operation_chain=_operation_chain(parent_node_ids, node_by_id, chain_item),
            reason=reason,
            objective_ids=objective_ids,
            status=str(step.get("status") or "pending"),
        )
        nodes.append(node)
        node_by_id[node_id] = node
        step_output_nodes[step_id] = [node_id]
        current_node_ids = [node_id]
        operations.append(
            OperationRecord(
                op_id=f"op:{step_id}",
                step_id=step_id,
                operation=operation,
                round_index=round_index,
                reason=reason,
                objective_ids=objective_ids,
                input_node_ids=parent_node_ids,
                output_node_ids=[node_id],
                status=str(step.get("status") or "pending"),
                result_keys=[result_key],
                parallel_group_id=_parallel_group_for_step(step, metadata),
                barrier_reason=_barrier_reason_for_step(step, metadata),
            )
        )

    output_node_ids = _graph_output_node_ids(nodes, operations)
    return OperationGraphRecord(
        graph_id=_graph_id(plan_data, "planned"),
        graph_kind="planned",
        source="planner.inferred_from_steps",
        input_node_ids=[node.node_id for node in nodes if node.kind == "input"],
        output_node_ids=output_node_ids,
        current_node_id=output_node_ids[-1] if output_node_ids else None,
        nodes=nodes,
        operations=operations,
        metadata=_without_empty({"request_hash": _request_hash(plan_data), "round_index": round_index, "inferred": True}),
    )


def _operations_from_nodes_and_steps(
    nodes: list[OperationAudioNode],
    steps: list[dict[str, Any]],
    rationales: dict[str, dict[str, Any]],
) -> list[OperationRecord]:
    nodes_by_step: dict[str, list[OperationAudioNode]] = {}
    for node in nodes:
        if node.step_id:
            nodes_by_step.setdefault(node.step_id, []).append(node)
    operations: list[OperationRecord] = []
    for index, step in enumerate(steps, start=1):
        step_id = str(step.get("id") or f"step_{index}")
        step_nodes = nodes_by_step.get(step_id, [])
        parent_ids = _dedupe(parent for node in step_nodes for parent in node.parent_node_ids)
        rationale = rationales.get(step_id)
        operations.append(
            OperationRecord(
                op_id=f"op:{step_id}",
                step_id=step_id,
                operation=str(step.get("operation") or ""),
                reason=_step_reason(step, rationale),
                objective_ids=_step_objective_ids(rationale),
                input_node_ids=parent_ids,
                output_node_ids=[node.node_id for node in step_nodes],
                status=str(step.get("status") or "pending"),
                result_keys=_dedupe(node.result_key for node in step_nodes if node.result_key),
            )
        )
    return operations


def _useful_artifact_keys(
    plan_data: dict[str, Any],
    artifacts: list[dict[str, Any]],
    planned: OperationGraphRecord | None,
) -> tuple[set[str], set[tuple[str, str, str]]]:
    useful_paths = {
        _path_key(path)
        for path in [
            _mapping(plan_data.get("metadata")).get("final_output_path"),
            plan_data.get("output_audio"),
            *_audio_path_values_from_step_inputs(plan_data),
        ]
        if path
    }
    useful_match_keys = _referenced_step_output_match_keys(plan_data)
    if planned:
        for node in planned.nodes:
            if node.path:
                useful_paths.add(_path_key(node.path))
            if node.step_id:
                useful_match_keys.add((node.step_id, node.result_key or "", node.role or ""))
                useful_match_keys.add((node.step_id, node.result_key or "", ""))
                useful_match_keys.add((node.step_id, "", node.role or ""))
    if not useful_paths and not useful_match_keys:
        for artifact in artifacts:
            if artifact.get("kind") == "step_output" and _audio_artifact_key_leaf(artifact.get("result_key")) in _PRIMARY_RESULT_KEYS:
                useful_paths.add(_path_key(artifact.get("path")))
    return useful_paths, useful_match_keys


def _referenced_step_output_match_keys(
    plan_data: dict[str, Any],
) -> set[tuple[str, str, str]]:
    """Identify exact upstream audio outputs named by structured step bindings."""

    references: set[tuple[str, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            from_step = _string_or_none(value.get("from_step"))
            from_output = _string_or_none(value.get("from_output"))
            if from_step and from_output:
                references.add((from_step, from_output))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for step in _dict_list(plan_data.get("steps")):
        visit(step.get("inputs"))
        visit(step.get("params"))

    keys: set[tuple[str, str, str]] = set()
    for step_id, output_name in references:
        keys.add((step_id, output_name, ""))
        keys.add((step_id, "", output_name))
    return keys


def _parent_node_ids_for_step(
    step: dict[str, Any],
    *,
    step_output_nodes: dict[str, list[str]],
    path_to_node: dict[str, str],
    current_node_ids: list[str],
    input_node_ids: list[str],
) -> list[str]:
    parents = _dedupe(
        node_id
        for dep in _string_list(step.get("depends_on"))
        for node_id in step_output_nodes.get(dep, [])
    )
    for path in _audio_path_values_from_mapping(_mapping(step.get("inputs"))):
        node_id = path_to_node.get(_path_key(path))
        if node_id:
            parents.append(node_id)
    for path in _audio_path_values_from_mapping(_mapping(step.get("params"))):
        node_id = path_to_node.get(_path_key(path))
        if node_id:
            parents.append(node_id)
    parents = _dedupe(parents)
    return parents or list(current_node_ids) or list(input_node_ids)


def _planned_nodes_by_match_key(planned: OperationGraphRecord | None) -> dict[tuple[str, str, str], OperationAudioNode]:
    rows: dict[tuple[str, str, str], OperationAudioNode] = {}
    if not planned:
        return rows
    for node in planned.nodes:
        if not node.step_id:
            continue
        keys = [
            (node.step_id, node.result_key or "", node.role or ""),
            (node.step_id, node.result_key or "", ""),
            (node.step_id, "", node.role or ""),
        ]
        for key in keys:
            rows.setdefault(key, node)
    return rows


def _match_planned_node(
    planned_by_key: dict[tuple[str, str, str], OperationAudioNode],
    artifact: dict[str, Any],
) -> OperationAudioNode | None:
    for key in _artifact_match_keys(artifact):
        node = planned_by_key.get(key)
        if node:
            return node
    return None


def _artifact_match_key(artifact: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(artifact.get("step_id") or ""),
        str(artifact.get("result_key") or ""),
        str(artifact.get("role") or ""),
    )


def _artifact_match_keys(artifact: dict[str, Any]) -> list[tuple[str, str, str]]:
    step_id = str(artifact.get("step_id") or "")
    result_key = str(artifact.get("result_key") or "")
    role = str(artifact.get("role") or "")
    return [
        (step_id, result_key, role),
        (step_id, result_key, ""),
        (step_id, "", role),
    ]


def _node_as_artifact(node: OperationAudioNode) -> dict[str, Any]:
    return {
        "step_id": node.step_id,
        "result_key": node.result_key,
        "role": node.role,
        "path": node.path,
    }


def _node_with_derived_lineage(
    node: OperationAudioNode,
    node_by_id: dict[str, OperationAudioNode],
    operations: list[OperationRecord],
) -> OperationAudioNode:
    op = next((item for item in operations if item.step_id == node.step_id), None)
    chain_item = (
        _operation_chain_item(
            step_id=op.step_id,
            operation=op.operation,
            reason=op.reason or node.reason,
            objective_ids=op.objective_ids or node.objective_ids,
        )
        if op
        else None
    )
    return node.model_copy(
        update={
            "ancestor_node_ids": node.ancestor_node_ids or _ancestor_node_ids(node.parent_node_ids, node_by_id),
            "operation_chain": node.operation_chain or _operation_chain(node.parent_node_ids, node_by_id, chain_item),
        }
    )


def _operation_chain_item(
    *,
    step_id: str,
    operation: str,
    reason: str | None,
    objective_ids: list[str],
) -> dict[str, Any]:
    return _without_empty(
        {
            "op_id": f"op:{step_id}",
            "step_id": step_id,
            "operation": operation,
            "reason": reason,
            "objective_ids": objective_ids,
        }
    )


def _operation_chain(
    parent_node_ids: list[str],
    node_by_id: dict[str, OperationAudioNode],
    chain_item: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent_id in parent_node_ids:
        parent = node_by_id.get(parent_id)
        if not parent:
            continue
        for item in parent.operation_chain:
            key = str(item.get("op_id") or item.get("step_id") or item)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
    if chain_item:
        key = str(chain_item.get("op_id") or chain_item.get("step_id") or chain_item)
        if key not in seen:
            result.append(chain_item)
    return result


def _ancestor_node_ids(parent_node_ids: list[str], node_by_id: dict[str, OperationAudioNode]) -> list[str]:
    ancestors: list[str] = []
    for parent_id in parent_node_ids:
        if parent_id not in ancestors:
            ancestors.append(parent_id)
        parent = node_by_id.get(parent_id)
        if not parent:
            continue
        for ancestor in parent.ancestor_node_ids:
            if ancestor not in ancestors:
                ancestors.append(ancestor)
    return ancestors


def _graph_output_node_ids(
    nodes: list[OperationAudioNode],
    operations: list[OperationRecord],
    *,
    explicit: list[str] | None = None,
) -> list[str]:
    if explicit:
        return explicit
    consumed = {node_id for op in operations for node_id in op.input_node_ids}
    leaves = [node.node_id for node in nodes if node.node_id not in consumed and node.kind != "input"]
    return leaves or ([nodes[-1].node_id] if nodes else [])


def _cycle_issues(graph: OperationGraphRecord) -> list[ValidationIssue]:
    node_by_id = {node.node_id: node for node in graph.nodes}
    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[ValidationIssue] = []

    def visit(node_id: str, path: list[str]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            issues.append(_issue("error", "cycle_detected", "Operation graph must be acyclic.", ref_id=node_id, metadata={"path": [*path, node_id]}))
            return
        visiting.add(node_id)
        node = node_by_id.get(node_id)
        if node:
            for parent_id in node.parent_node_ids:
                if parent_id in node_by_id:
                    visit(parent_id, [*path, node_id])
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_by_id:
        visit(node_id, [])
    return issues


def _round_state_cycle_issues(round_record: RoundOperationGraphRecord, *, location: str) -> list[ValidationIssue]:
    state_by_id = {node.state_node_id: node for node in round_record.state_nodes}
    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[ValidationIssue] = []

    def visit(state_node_id: str, path: list[str]) -> None:
        if state_node_id in visited:
            return
        if state_node_id in visiting:
            issues.append(
                _issue(
                    "error",
                    "state_cycle_detected",
                    "Round operation graph state graph must be acyclic.",
                    location=f"{location}.state_nodes",
                    ref_id=state_node_id,
                    metadata={"path": [*path, state_node_id]},
                )
            )
            return
        visiting.add(state_node_id)
        state_node = state_by_id.get(state_node_id)
        if state_node:
            for parent_id in state_node.parent_state_node_ids:
                if parent_id in state_by_id:
                    visit(parent_id, [*path, state_node_id])
        visiting.remove(state_node_id)
        visited.add(state_node_id)

    for state_node_id in state_by_id:
        visit(state_node_id, [])
    return issues


def _process_artifact_issue(node: OperationAudioNode, *, location: str) -> ValidationIssue | None:
    tokens = {
        _normalized_token(node.kind),
        _normalized_token(node.role),
        _normalized_token(node.result_key),
    }
    metadata_type = _mapping(node.metadata).get("artifact_type") or _mapping(node.metadata).get("type")
    if metadata_type:
        tokens.add(_normalized_token(metadata_type))
    if any(token in _PROCESS_ARTIFACT_TOKENS for token in tokens if token):
        return _issue(
            "error",
            "non_audio_process_node",
            "OperationAudioNode must describe a useful audio artifact/span; process reports belong in event log or metadata.",
            location=location,
            ref_id=node.node_id,
            metadata={"kind": node.kind, "role": node.role, "result_key": node.result_key},
        )
    if node.path and not _looks_like_audio_path(str(node.path)):
        return _issue(
            "warning",
            "audio_node_path_not_audio",
            "OperationAudioNode path does not look like an audio file.",
            location=f"{location}.path",
            ref_id=node.node_id,
            metadata={"path": node.path},
        )
    return None


def _normalized_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _round_index_from_graph(graph: OperationGraphRecord) -> int | None:
    metadata = _mapping(graph.metadata)
    for value in (
        metadata.get("round_index"),
        metadata.get("dialogue_round_index"),
        metadata.get("session_round_index"),
    ):
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    for node in graph.nodes:
        if node.round_index is not None:
            return node.round_index
    for operation in graph.operations:
        if operation.round_index is not None:
            return operation.round_index
    return None


def _round_index_from_plan_data(plan_data: dict[str, Any]) -> int | None:
    metadata = _mapping(plan_data.get("metadata"))
    global_params = _mapping(plan_data.get("global_params"))
    candidates = [
        metadata.get("round_index"),
        metadata.get("dialogue_round_index"),
        _mapping(metadata.get("session_graph")).get("round_index"),
        _mapping(metadata.get("session_graph")).get("edit_session_round_index"),
        global_params.get("round_index"),
        global_params.get("dialogue_round_index"),
        global_params.get("edit_session_round_index"),
    ]
    for value in candidates:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _state_id_for_audio_node(
    *,
    round_index: int,
    operation: OperationRecord,
    audio_node: OperationAudioNode,
    existing_ids: set[str],
) -> str:
    if audio_node.node_id in existing_ids:
        return _unique_node_id(f"round_{round_index}:state:{_safe_id(audio_node.node_id)}", existing_ids)
    base = f"round_{round_index}:{_safe_id(operation.step_id)}:{_safe_id(audio_node.role or audio_node.result_key or 'state')}"
    return _unique_node_id(base, existing_ids)


def _audio_segment_summary(node: OperationAudioNode) -> dict[str, Any]:
    return _without_empty(
        {
            "audio_node_id": node.node_id,
            "role": node.role,
            "kind": node.kind,
            "path": node.path,
            "artifact_id": node.artifact_id,
            "segment": node.segment,
            "split_group_id": node.split_group_id,
            "split_index": node.split_index,
            "split_role": node.split_role,
        }
    )


def _round_leaf_state_ids(state_nodes: list[OperationStateNode], source_state_id: str) -> list[str]:
    consumed = {
        parent_id
        for node in state_nodes
        for parent_id in node.parent_state_node_ids
    }
    return [
        node.state_node_id
        for node in state_nodes
        if node.state_node_id != source_state_id and node.state_node_id not in consumed
    ]


def _step_rationales_by_id(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in _dict_list(metadata.get("step_rationales")):
        step_id = item.get("step_id")
        if step_id:
            rows[str(step_id)] = item
    return rows


def _step_reason(step: dict[str, Any], rationale: dict[str, Any] | None = None) -> str | None:
    for value in (
        (rationale or {}).get("decision_reason"),
        (rationale or {}).get("input_reason"),
        step.get("description"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _step_objective_ids(rationale: dict[str, Any] | None) -> list[str]:
    return _string_list((rationale or {}).get("objective_ids"))


def _parallel_group_for_step(step: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    value = _mapping(step.get("metadata")).get("parallel_group_id")
    if value:
        return str(value)
    for item in _dict_list(metadata.get("operation_graph_parallel_groups")):
        step_ids = _string_list(item.get("step_ids"))
        if str(step.get("id") or "") in step_ids and item.get("parallel_group_id"):
            return str(item["parallel_group_id"])
    return None


def _barrier_reason_for_step(step: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    value = _mapping(step.get("metadata")).get("barrier_reason")
    if value:
        return str(value)
    for item in _dict_list(metadata.get("operation_graph_barriers")):
        step_ids = _string_list(item.get("step_ids"))
        if str(step.get("id") or "") in step_ids and item.get("barrier_reason"):
            return str(item["barrier_reason"])
    return None


def _segment_from_artifact_or_step(artifact: dict[str, Any], step: dict[str, Any]) -> dict[str, Any] | None:
    for value in (
        _mapping(artifact.get("metadata")).get("span"),
        _mapping(artifact.get("metadata")).get("segment"),
        artifact.get("span"),
        _first_span(_mapping(step.get("params")).get("selected_spans")),
        _first_span(_mapping(step.get("result")).get("selected_spans")),
    ):
        if isinstance(value, dict):
            return value
    return None


def _first_span(value: Any) -> dict[str, Any] | None:
    rows = _dict_list(value)
    return rows[0] if len(rows) == 1 else None


def _audio_path_values_from_step_inputs(plan_data: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for step in _dict_list(plan_data.get("steps")):
        paths.extend(_audio_path_values_from_mapping(_mapping(step.get("inputs"))))
        paths.extend(_audio_path_values_from_mapping(_mapping(step.get("params"))))
    return paths


def _audio_path_values_from_mapping(value: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for _path, item in _iter_values(value):
        if isinstance(item, str) and _looks_like_audio_path(item):
            rows.append(item)
    return rows


def _iter_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_iter_values(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_iter_values(item, f"{prefix}[{index}]"))
    else:
        rows.append((prefix, value))
    return rows


def _looks_like_audio_path(value: str) -> bool:
    text = str(value)
    lowered = text.lower().split("?", 1)[0]
    return lowered.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"))


def _audio_artifact_key_leaf(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return text.rsplit(".", 1)[-1].split("[", 1)[0]


def _artifact_summary(artifact: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return _without_empty(
        {
            "reason": reason,
            "artifact_id": artifact.get("artifact_id"),
            "step_id": artifact.get("step_id"),
            "role": artifact.get("role"),
            "result_key": artifact.get("result_key"),
            "path": artifact.get("path"),
        }
    )


def _coerce_graph(record: OperationGraphRecord | dict[str, Any]) -> OperationGraphRecord:
    if isinstance(record, OperationGraphRecord):
        return record
    return OperationGraphRecord(**record)


def _coerce_round_store(record: RoundOperationGraphStore | dict[str, Any]) -> RoundOperationGraphStore:
    if isinstance(record, RoundOperationGraphStore):
        return record
    return RoundOperationGraphStore(**record)


def _step_ids_from_plan(plan: Any | None) -> set[str]:
    plan_data = _object_dump(plan)
    return {str(step.get("id")) for step in _dict_list(plan_data.get("steps")) if step.get("id")}


def _input_audio_path_for_index(plan_data: dict[str, Any], index: int) -> str | None:
    global_params = _mapping(plan_data.get("global_params"))
    paths = _dedupe(
        [
            plan_data.get("input_audio"),
            *_string_list(global_params.get("input_audios") or global_params.get("input_audio_paths")),
        ]
    )
    if 0 <= index < len(paths):
        return paths[index]
    return None


def _referenced_operation_graph_node_ids(
    *,
    node_rows: list[dict[str, Any]],
    op_rows: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    for row in node_rows:
        ids.extend(_string_list(row.get("parent_node_ids") or row.get("parents") or row.get("input_node_ids")))
    for row in op_rows:
        ids.extend(_string_list(row.get("input_node_ids") or row.get("inputs")))
    return _dedupe(ids)


def _looks_like_input_node_id(node_id: str) -> bool:
    text = str(node_id or "").lower()
    return text in {"input", "input_audio", "op_node:input", "op_node:input_audio"} or text.startswith("op_node:input_")


def _object_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        dumped = value.to_dict()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _model_row(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _without_empty(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return _without_empty(value)
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _path_key(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    safe = "".join(char if char.isalnum() or char in {"_", "-", ":"} else "_" for char in text)
    return safe.strip("_") or "item"


def _unique_node_id(base: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    index = 2
    while f"{base}_{index}" in seen:
        index += 1
    return f"{base}_{index}"


def _request_hash(plan_data: dict[str, Any]) -> str:
    request = str(plan_data.get("request") or "")
    return hashlib.sha1(request.encode("utf-8")).hexdigest()[:12]


def _graph_id(plan_data: dict[str, Any], kind: str) -> str:
    return f"operation_graph:{kind}:{_request_hash(plan_data)}"


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    location: str | None = None,
    ref_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        location=location,
        ref_id=ref_id,
        metadata=metadata or {},
    )
