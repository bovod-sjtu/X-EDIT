"""Graph-selected edit bases and narrowly scoped initial audio materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from x_editor.core.schemas import AudioEditPlan
from x_editor.reasoner.contracts import SessionStateContext

INITIAL_AUDIO_AUTHORITY_KEY = "_runtime_admitted_initial_audio"


@dataclass(frozen=True)
class AdmittedInitialAudio:
    """Forward-owned permission for one initial plan input resolution."""

    audio_ref: str


def plan_selects_bound_reference_base(
    plan: AudioEditPlan,
    *,
    reference_audio_refs: list[str],
    session_state: SessionStateContext | None,
) -> bool:
    """Require a bound asset and an explicit, consistent eligible graph base."""

    if session_state is None or plan.input_audio not in reference_audio_refs:
        return False
    selection = plan.metadata.get("session_graph")
    if not isinstance(selection, dict) or (
        selection.get("selection_kind") != "semantic_state_base"
        or selection.get("selected_source_audio_ref") != plan.input_audio
    ):
        return False
    selected_node_id = selection.get("selected_source_node_id")
    matching = [
        node
        for node in session_state.nodes
        if node.node_id == selected_node_id
        and node.session_eligible
        and node.contains_audio_ref(plan.input_audio)
    ]
    if len(matching) != 1:
        return False
    operation_type = selection.get("operation_type")
    if selected_node_id == session_state.current_node_id:
        return operation_type == "edit"
    return operation_type == "branch_edit"


def resolve_initial_audio_reference(value: Any, state: dict[str, Any]) -> Any:
    """Consume Forward's graph-base grant only for this exact plan input."""

    from x_editor.runtime.audio_references import resolve_audio_reference

    context = state.get("context")
    grant = context.pop(INITIAL_AUDIO_AUTHORITY_KEY, None) if isinstance(context, dict) else None
    plan = state.get("plan")
    admitted = (
        isinstance(grant, AdmittedInitialAudio)
        and isinstance(plan, AudioEditPlan)
        and value == plan.input_audio == grant.audio_ref
    )
    return resolve_audio_reference(
        value,
        state,
        expected_role=None if admitted else "source",
        allow_runtime_paths=False,
    )
