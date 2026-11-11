"""Verify an explicit history selection against the persisted Session Graph.

An unchanged waveform is not evidence of an edit. It can be a valid delivery
when the runtime completed a history restore. This proof is derived from the
graph and the completed turn, never from request wording or an evaluator.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from x_editor.core.final_state_projection import (
    FinalStateProjection,
    project_final_session_payload,
)


def history_delivery_proof(
    result: dict[str, Any] | None,
    *,
    graph_path: Path,
    output_path: Path,
) -> dict[str, Any] | None:
    """Return a content-bound receipt, or no authority for source-identical output."""
    if not isinstance(result, dict) or result.get("status") != "completed":
        return None
    projection = result.get("final_state_projection") or {}
    if not isinstance(projection, dict) or projection.get("final_operation_type") != "rollback":
        return None
    try:
        return _verify_history_delivery(result, graph_path=graph_path, output_path=output_path)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _verify_history_delivery(
    result: dict[str, Any], *, graph_path: Path, output_path: Path
) -> dict[str, Any] | None:
    if Path(result["session_graph_path"]).resolve() != graph_path.resolve():
        return None
    projection = FinalStateProjection.model_validate(result["final_state_projection"])
    graph_bytes = graph_path.read_bytes()
    graph_sha256 = hashlib.sha256(graph_bytes).hexdigest()
    if graph_sha256 != projection.graph_sha256:
        return None
    graph = json.loads(graph_bytes)
    observed = project_final_session_payload(graph)
    if observed.model_copy(update={"graph_sha256": graph_sha256}) != projection:
        return None

    turns = result["turn_results"]
    if not turns or any(turn["result"]["status"] != "completed" for turn in turns):
        return None
    final_turn = turns[-1]["result"]
    completion = final_turn.get("state_only_completion") or {}
    if (
        completion.get("selection_kind") != "restore_history"
        or final_turn.get("forward_result") is not None
        or final_turn.get("final_plan") is not None
    ):
        return None

    selected_id = completion["selected_node_id"]
    selected = graph["nodes"][selected_id]
    if any(selected.get("metadata", {}).get(key) is False for key in (
        "session_eligible", "current_audio_advanced"
    )):
        return None
    current = graph["nodes"][projection.current_node_id]
    record = graph["round_history"][-1]
    edge = graph["edges"][record["edge_id"]]
    selection = current["metadata"].get("state_only_completion") or {}
    if not (
        record["status"] == edge["status"] == "completed"
        and record["result_node_id"] == edge["target_node_id"] == projection.current_node_id
        and record["input_node_id"] == selected_id
        and record["metadata"].get("round_entry_node_id") not in (None, selected_id)
        and edge["operation_type"] == "rollback"
        and edge["source_node_ids"] == current["parent_node_ids"] == [selected_id]
        and not edge["operations"] and not edge["plan_step_ids"]
        and selection.get("selection_kind") == "restore_history"
        and selection.get("selected_source_node_id") == selected_id
        and selection.get("selected_source_audio_ref") == completion["selected_audio_ref"]
    ):
        return None
    reusable = selected_id in graph["root_node_ids"] or any(
        prior["status"] == "completed" and prior["result_node_id"] == selected_id
        for prior in graph["round_history"][:-1]
    )
    paths = (result["final_output_path"], projection.final_audio_path, selected["primary_audio_path"])
    if not reusable or any(Path(path).resolve() != output_path.resolve() for path in paths):
        return None
    output = output_path.read_bytes()
    if not output:
        return None
    return {
        "schema_version": "x-editor.history-delivery-proof/v1",
        "graph_sha256": graph_sha256,
        "selected_node_id": selected_id,
        "selected_audio_ref": completion["selected_audio_ref"],
        "result_node_id": projection.current_node_id,
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "output_size_bytes": len(output),
    }
