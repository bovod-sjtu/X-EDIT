"""Bind request provenance without treating a model paraphrase as authority.

The model owns its interpretation and executable steps. The caller owns which
request those steps are being proposed for. This projection does not establish
semantic correctness and does not change any operation, parameter or evidence.
"""
from __future__ import annotations

from x_editor.planner.reasoned_workflow import (
    MemoQualifiedPlannerInput,
    MemoQualifiedPlannerTurn,
)


def bind_execution_request(
    turn: MemoQualifiedPlannerTurn, planner_input: MemoQualifiedPlannerInput
) -> tuple[MemoQualifiedPlannerTurn, dict]:
    """Preserve the proposal in an audit receipt; bind trusted call provenance.

    Both separated and integrated planning use this boundary. A missing active
    request is left for the ordinary state-completion validator to reject; this
    function never supplies a request when the caller has authorized no edit.
    """
    if turn.decision != "execute" or turn.plan is None:
        return turn, {"applied": False, "reason": "no_execution_plan"}
    if planner_input.execution_request is None:
        return turn, {"applied": False, "reason": "no_active_execution_request"}
    original = turn.plan.request
    authoritative = planner_input.execution_request
    receipt = {
        "schema_version": "x-editor.plan-request-binding/v1",
        "applied": True,
        "changed": original != authoritative,
        "source": "planner_input.execution_request",
        "planner_request_id": planner_input.planner_request_id,
        "original_user_request": planner_input.user_request,
        "active_execution_request": authoritative,
        "provider_request_description": original,
        "semantic_equivalence_asserted": False,
        "execution_arguments_modified": False,
    }
    return turn.model_copy(update={
        "plan": turn.plan.model_copy(update={"request": authoritative})
    }), receipt
