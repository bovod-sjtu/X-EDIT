"""Planner-facing composition rules and graph-authorized completion choices.

These are input contracts, not a repair pass: the provider still chooses the
requested result, and normal admission validates every returned decision.
"""

from __future__ import annotations

from typing import Any

from x_editor.reasoner.contracts import (
    SessionStateContext,
    session_completion_proof_inventory,
)


COMPOSITION_RULES = """Composition and completion contract:
- Decide whether the requested result requires new processing or selecting an existing state. If an authorized existing state satisfies the WHOLE request, return complete_from_session_state with state_completion_kind and changes_satisfied_by_base. Never represent this as execute with steps=[], a fake identity tool, or a plan output whose from_step names a Session node. A completion proof authorizes a node/audio pair; it does not establish that this pair satisfies the current request.
- For a historical state followed by NEW edits, use execute with branch_from_history and real nonempty steps. For selecting that history without new edits, use complete_from_session_state with restore_history. Only selection of the current node uses select_existing. Never mix state_completion_kind and state_transition_kind.
- A clip whose samples enter concatenation, replacement, or mixing is an assembly_member, even if uploaded as a reference. A clip used only to condition another tool is reference_input. Declare each external audio's exact scalar argument path, including a list index. Whole-clip assembly needs existence/decodability, not speaker attribution or lexical target grounding unless the request selects such a target.
- Voice conversion preserves spoken content from the explicitly bound source and conditions speaker timbre on the explicitly bound reference. Determine this direction from the user's intended content and voice roles, independently of upload order, filenames, or numeric aliases. Declare both audio bindings and the scope: a full-recording conversion and a speaker/region-scoped conversion are different plans. For a local conversion retain its grounded selection and reconstruction; do not widen it to the whole asset because another audio supplies the reference voice.
- Choose an offered operation by its effect, preservation constraints, and cost, and encode that choice explicitly. For a softer delivery, whisper_style can be a reasonable audio_effects approximation when it fits the intended effect; explicitly select preset and strength instead of expecting the executor to infer them from wording. Whisper coloration, level change, emotion/style editing, and generative delivery have different effects and fidelity limits. No one of them is mandatory merely because a request describes softness. Use only available capabilities and report material approximation limits.
- A foreground/background wrapper does not choose the nested tool from prose. Where the offered capability explicitly permits edit_foreground, supply edit_operation (or edit_op) and the corresponding edit_params (or edit_op_params), including scope and source/reference bindings. An explicit audio_effects choice requires its preset and desired strength; a tagged edit requires its valid source_text and instruction_st contract. If the offered wrapper only supports removal/extraction, use only those actions and express other supported edits as explicit admitted steps; do not send unavailable nested parameters.
- Dependent edits consume the previous step's output through from_step/from_output plus depends_on. The final output must retain all earlier requested effects. Independent parts may branch, but assemble them only when the requested deliverable is one audio; do not flatten multiple requested deliverables into one.
- Operation scope is relative to its bound input artifact. A global operation may process an extracted region when that region satisfies its input requirements. Before declaring a local objective unsupported, consider whether offered extraction, processing, and assembly capabilities can jointly satisfy it. Preserve unselected regions and their order, account for changed output duration, and respect context requirements and boundary-quality limits; do not invent local parameters for a global operation or assume every operation is safely composable.
- Relative level changes require an offered gain_db capability; target_db is an absolute level and cannot represent a delta. For local changes, use only supplied explicit intervals or reviewed target spans and preserve every unselected interval. If the requested gain cannot fit available headroom, report the boundary instead of silently lowering the gain.
- Perception jobs describe missing observations. A single_audio capability observes one asset: request separate jobs to observe additional assets. Supporting audio is for a capability that explicitly accepts paired_audio or multi_audio; it does not make a single-audio route observe a pair. Cross-audio relationships require evidence from the relevant assets, not an assumed shared index.
"""


def session_completion_choices(
    state: SessionStateContext | None,
    *,
    bound_source_refs: list[str],
) -> list[dict[str, Any]]:
    """Expose exact wire identifiers only for existing, bound proof targets.

    No requested-effect labels or completion decision are inferred here. Multiple
    proofs for one node are retained as provenance without duplicating choices.
    """

    if state is None:
        return []
    choices: dict[tuple[str, str], dict[str, Any]] = {}
    for proof in session_completion_proof_inventory(state):
        if proof.selected_audio_ref not in bound_source_refs:
            continue
        key = (proof.selected_node_id, proof.selected_audio_ref)
        choice = choices.setdefault(key, {
            "selected_base_state_node_id": key[0],
            "selected_base_audio_ref": key[1],
            "state_completion_kind": (
                "select_existing" if key[0] == state.current_node_id else "restore_history"
            ),
            "proof_refs": [],
        })
        if proof.proof_ref not in choice["proof_refs"]:
            choice["proof_refs"].append(proof.proof_ref)
    return list(choices.values())
