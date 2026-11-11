"""Integrity-bound, single-use continuation of an accepted workflow history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from x_editor.core.final_state_projection import project_final_session_state
from x_editor.core.schemas import StrictModel
from x_editor.runtime.edit_session_graph import EditSessionGraph, load_edit_session_graph
from x_editor.runtime.workflow_case import RuntimeAudioBinding, WorkflowRuntimeScenario
from x_editor.runtime.workflow_orchestration import WorkflowArm
from x_editor.runtime.workflow_session import (
    WorkflowScenarioResult,
    WorkflowScenarioTurnResult,
    _initial_session_bindings,
)


class WorkflowContinuation(StrictModel):
    """Private checkpoint; never expose its paths or results to a provider."""

    schema_version: Literal["x-editor.workflow-continuation/v1"] = (
        "x-editor.workflow-continuation/v1"
    )
    scenario: WorkflowRuntimeScenario
    result: WorkflowScenarioResult
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_sha256: dict[str, str]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validated_graph(
    scenario: WorkflowRuntimeScenario, result: WorkflowScenarioResult
) -> EditSessionGraph:
    if result.scenario_fingerprint != scenario.fingerprint:
        raise ValueError("continuation result does not match its scenario")
    if [row.turn_index for row in result.turn_results] != list(range(len(scenario.turns))):
        raise ValueError("continuation requires a terminal result for every previous turn")
    if not result.session_graph_path or result.final_state_projection is None:
        raise ValueError("continuation requires a persisted graph and final state projection")
    graph = load_edit_session_graph(result.session_graph_path)
    if graph.round_index != len(scenario.turns) or len(graph.round_history) != len(scenario.turns):
        raise ValueError("continuation graph round history is incomplete")
    for index, record in enumerate(graph.round_history):
        if record.round_index != index + 1 or record.request != scenario.turns[index].user_request:
            raise ValueError("continuation graph does not match the previous requests")
        if record.status not in {"completed", "blocked", "failed"}:
            raise ValueError("continuation cannot resume an unfinished round")
    if graph.round_history[-1].status != result.status:
        raise ValueError("continuation graph and terminal result disagree")
    if (
        project_final_session_state(graph, graph_path=result.session_graph_path)
        != result.final_state_projection
    ):
        raise ValueError("continuation graph integrity does not match the terminal projection")
    if graph.current_node() is None or not graph.current_audio_path():
        raise ValueError("continuation has no accepted current audio")
    return graph


def _asset_hashes(scenario: WorkflowRuntimeScenario, graph: EditSessionGraph) -> dict[str, str]:
    paths = {asset.path for asset in scenario.audio_assets}
    for node in graph.nodes.values():
        # Hash available intermediate evidence too, without granting it authority.
        paths.update(asset.path for asset in node.audio_paths if Path(asset.path).is_file())
        # Accepted history must remain available, even when it is not current.
        if node.kind == "initial" or node.is_round_result:
            if node.primary_audio_path:
                paths.add(node.primary_audio_path)
            paths.update(asset.path for asset in node.audio_paths if asset.path)
    hashes = {path: _file_sha256(path) for path in sorted(paths)}
    for asset in scenario.audio_assets:
        if asset.content_sha256 is not None and hashes[asset.path] != asset.content_sha256:
            raise ValueError("continuation source asset content hash changed")
    return hashes


def create_workflow_continuation(
    scenario: WorkflowRuntimeScenario, result: WorkflowScenarioResult
) -> WorkflowContinuation:
    """Bind a completed or failed request to its last accepted graph state.

    This performs no execution and does not claim the checkpoint. Exceptions
    without a terminal scenario result cannot be resumed through this API.
    """
    scenario = WorkflowRuntimeScenario.model_validate(scenario)
    result = WorkflowScenarioResult.model_validate(result)
    graph = _validated_graph(scenario, result)
    payload = {
        "schema_version": "x-editor.workflow-continuation/v1",
        "scenario": scenario.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "graph_sha256": _file_sha256(result.session_graph_path),
        "asset_sha256": _asset_hashes(scenario, graph),
    }
    return WorkflowContinuation.model_validate({**payload, "fingerprint": _fingerprint(payload)})


@dataclass
class RestoredWorkflowSession:
    graph: EditSessionGraph
    turn_results: list[WorkflowScenarioTurnResult]
    audio_ref_by_path: dict[str, str]
    session_bindings: list[RuntimeAudioBinding]
    previous_output_path: str
    previous_output_ref: str
    previous_output_artifact_ref_id: str | None
    previous_output_state_node_id: str | None
    failed_turn_indices: set[int]


def restore_workflow_continuation(
    continuation: WorkflowContinuation | dict,
    scenario: WorkflowRuntimeScenario,
    *,
    arm: WorkflowArm,
    output_dir: Path,
) -> RestoredWorkflowSession:
    """Validate and atomically claim one checkpoint before any provider call."""
    checkpoint = WorkflowContinuation.model_validate(continuation)
    payload = checkpoint.model_dump(mode="json", exclude={"fingerprint"})
    if _fingerprint(payload) != checkpoint.fingerprint:
        raise ValueError("continuation checkpoint fingerprint changed")
    previous = checkpoint.scenario
    if checkpoint.result.assigned_arm != arm:
        raise ValueError("continuation workflow arm cannot change")
    if len(scenario.turns) <= len(previous.turns):
        raise ValueError("continuation requires at least one new user turn")
    if scenario.turns[: len(previous.turns)] != previous.turns:
        raise ValueError("continuation cannot change previous user turns")
    if scenario.audio_assets[: len(previous.audio_assets)] != previous.audio_assets:
        raise ValueError("continuation cannot change or reorder previous audio assets")
    if any(
        asset.order <= max(item.order for item in previous.audio_assets)
        for asset in scenario.audio_assets[len(previous.audio_assets) :]
    ):
        raise ValueError("continuation new assets must follow the previous asset order")
    graph = _validated_graph(previous, checkpoint.result)
    if _file_sha256(checkpoint.result.session_graph_path) != checkpoint.graph_sha256:
        raise ValueError("continuation graph content changed")
    if _asset_hashes(previous, graph) != checkpoint.asset_sha256:
        raise ValueError("continuation historical asset content changed")
    _asset_hashes(scenario, graph)  # Also validate newly appended input assets.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("continuation output directory must be new or empty")

    bindings = _initial_session_bindings(
        scenario, through_turn_index=len(previous.turns) - 1,
    )
    refs = {
        asset.path: asset.logical_ref
        for asset in scenario.visible_audio_assets(len(previous.turns) - 1)
    }
    for record in graph.round_history:
        if record.status != "completed":
            continue
        node = graph.nodes.get(record.result_node_id or "")
        if node is None or not node.is_round_result or not node.primary_audio_path:
            raise ValueError("continuation accepted round has no graph artifact")
        audio_ref = node.metadata.get("audio_ref")
        if not isinstance(audio_ref, str):
            raise TypeError("continuation accepted artifact has no canonical audio ref")
        selected = node.metadata.get("selected_state_node_id")
        artifact_ref = node.metadata.get("artifact_ref_id")
        if not selected and not artifact_ref:
            raise ValueError("continuation accepted output has no commit authority")
        bindings.append(
            RuntimeAudioBinding(
                audio_ref=audio_ref,
                materialized_path=node.primary_audio_path,
                role="authoritative_output",
                order=len(bindings),
                origin="session_graph_state" if selected else "artifact_store_commit",
                origin_artifact_ref_id=artifact_ref,
                origin_state_node_id=selected,
            )
        )
        if not selected:
            refs[node.primary_audio_path] = audio_ref
    current = graph.current_node()
    assert current is not None and current.primary_audio_path
    current_ref = current.metadata.get("audio_ref") or refs.get(current.primary_audio_path)
    if not isinstance(current_ref, str):
        raise TypeError("continuation current state has no bound audio ref")
    # The claim is separate from the immutable source graph/result. Recreating a
    # checkpoint for the same parent produces the same key, including across processes.
    claims = Path(checkpoint.result.session_graph_path).parent.parent / ".workflow_continuations"
    claims.mkdir(parents=True, exist_ok=True)
    try:
        with (claims / f"{checkpoint.fingerprint}.json").open("x", encoding="utf-8") as stream:
            json.dump({"parent": checkpoint.fingerprint, "output_dir": str(output_dir)}, stream)
    except FileExistsError as exc:
        raise ValueError("continuation checkpoint has already been claimed") from exc
    return RestoredWorkflowSession(
        graph=graph,
        turn_results=[row.model_copy(deep=True) for row in checkpoint.result.turn_results],
        audio_ref_by_path=refs,
        session_bindings=bindings,
        previous_output_path=current.primary_audio_path,
        previous_output_ref=current_ref,
        previous_output_artifact_ref_id=current.metadata.get("artifact_ref_id"),
        previous_output_state_node_id=current.metadata.get("selected_state_node_id"),
        failed_turn_indices={
            row.round_index - 1 for row in graph.round_history if row.status != "completed"
        },
    )


__all__ = ["WorkflowContinuation", "create_workflow_continuation"]
