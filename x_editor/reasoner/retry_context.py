"""Opt-in retry context: retain typed state while retiring prior prose."""

from typing import Any, Literal

from x_editor.reasoner.contracts import SemanticReasoningMemo

ReasonerRetryContextMode = Literal["prior_memo", "current_state/v1"]


def project_retry_semantic_state(memo: SemanticReasoningMemo) -> dict[str, Any]:
    """Keep all typed semantic/observation state; identities stay in the audit."""
    return memo.model_dump(mode="json", exclude={
        "content", "request_id", "schema_version", "prior_memo_fingerprint", "phase",
    })


def share_retry_observation_ledger(payload: dict[str, Any]) -> None:
    """Use the review envelope's ledger once, retaining an explicit reference.

    Only current-state retries contain this projection. The original typed
    request and ordinary first-pass prompts remain unchanged.
    """
    state = payload.get("current_semantic_state")
    if state is None:
        return
    if state.get("observation_ledger") != payload["pending_observation_history"]:
        raise ValueError("retry state and review observation ledger disagree")
    state.pop("observation_ledger")
    state["observation_ledger_ref"] = "/pending_observation_history"
