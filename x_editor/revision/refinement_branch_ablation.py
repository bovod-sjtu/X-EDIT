"""Production review support extracted without changing function bodies."""

from copy import deepcopy


BRANCH_ARMS = {
    "semantic_feedback": "semantic",
    "multimodal_feedback": "multimodal",
    "joint_feedback": "joint",
}


def semantic_tool_observations(observations):
    """Keep invocation and generation-plan facts, not output listening results.

    Pipeline metadata mixes planned synthesis durations with optional output
    recognition. Project the declared generation-plan fields so a semantic
    reviewer can discover a zero allocation without receiving output ASR.
    This is an experimental information boundary, never a tool routing rule.
    """
    fields = {"step_id", "operation", "status", "actual_inputs", "actual_params", "error",
              "reported_source_spans", "reported_target_spans", "reported_temporal_transform",
              "reported_target_text", "reported_generation_plans", "authority"}
    projected = []
    for row in observations:
        result = {k: deepcopy(v) for k, v in row.items() if k in fields}
        metadata = row.get("reported_metadata") or {}
        generations = metadata.get("generations", {}) if isinstance(metadata, dict) else {}
        plans = {}
        for key, generation in generations.items() if isinstance(generations, dict) else []:
            if not isinstance(generation, dict):
                continue
            span = generation.get("span_plan") or {}
            estimates = span.get("metadata") or {}
            synthesis = generation.get("synthesis") or {}
            plans[key] = {
                "source_spans": deepcopy(span.get("source_spans")),
                "target_spans": deepcopy(span.get("target_spans")),
                "duration_estimates": {k: deepcopy(estimates[k]) for k in
                    ("raw_fix_durations", "expanded_fix_durations", "speech_rate") if k in estimates},
                "synthesis_inputs": {k: deepcopy(synthesis[k]) for k in
                    ("fix_durations_sec", "target_condition_duration",
                     "text_payload", "seed", "cfg_strength", "synthesis_mode") if k in synthesis},
            }
            if isinstance(synthesis.get("planned_spans"), list):
                plans[key]["synthesis_inputs"]["planned_spans"] = [
                    {k: deepcopy(item[k]) for k in ("kind", "source_text", "target_text",
                        "source_char_span", "target_char_span", "source_span", "target_span",
                        "fix_duration") if k in item}
                    for item in synthesis["planned_spans"] if isinstance(item, dict)]
        if plans:
            result["reported_generation_plans"] = plans
        projected.append(result)
    return projected


def project_focus_payload(payload, branch):
    """Apply the source intervention before diagnostic scope selection as well."""
    result = deepcopy(payload)
    if branch == "semantic":
        result["terminal_audio_review"] = None
        result["terminal_coordinate_measurements"] = None
        for turn in result["actual_turns"]:
            turn["tool_observations"] = semantic_tool_observations(turn["tool_observations"])
        result["feedback_information_policy"] = (
            "Only review pre-execution Reasoner/Planner evidence use and executable contracts. "
            "Post-output listening/ASR/PCM findings are withheld; do not infer their judgments. "
            "No terminal-quality requirement checks are requested or certified by this branch.")
    elif branch == "multimodal":
        result["feedback_information_policy"] = (
            "Initiate recovery only from observed audio requirement violations or missing acoustic evidence. "
            "Raw module handoffs help address that acoustic issue; no dedicated semantic module scores are available. "
            "Do not invent an independent semantic defect from an unreviewed handoff.")
    return result


def project_feedback_payload(payload, branch):
    if branch not in BRANCH_ARMS.values():
        raise ValueError(f"Unknown feedback branch: {branch}")
    # Allowlist rather than deleting a few known fields: future additions to the
    # ordinary reviewer must not silently change an ablation's information set.
    common = ("schema", "public_prefixes", "audio_inventory", "available_question_ids",
              "diagnostic_scope", "current_evidence_state", "recorded_tool_observations",
              "available_tool_capabilities", "recorded_plan_dependencies")
    keys = set(common)
    if branch in {"semantic", "joint"}:
        keys.update(("atomic_reviews", "recorded_tool_observations", "decision_review_gate"))
    if branch in {"multimodal", "joint"}:
        keys.update(("terminal_audio_review", "new_targeted_observations", "execution_purpose_observations"))
    result = {key: deepcopy(value) for key, value in payload.items() if key in keys}
    if branch == "semantic":
        result["audio_inventory"] = []
        result["recorded_tool_observations"] = semantic_tool_observations(
            result.get("recorded_tool_observations", []))
    result["feedback_information_policy"] = {
        "branch": branch,
        "missing_branch": "unobserved, not passed and not failed",
        "guidance_targets": ["reasoner", "planner"],
        "perception_request": "Ask Reasoner to request relevant perception through its normal interface.",
        "observation_checks_allowed": branch != "semantic",
        "attribution_scope": (
            "The output-side channel includes listening, measured signal facts and tool-return telemetry; "
            "its effect must not be reported as the isolated effect of the Omni model. "
            "Both branches use the same controller and current-state handoff. "
            "Semantic-only retains invocation inputs/status/errors and declared generation-plan fields, "
            "including duration allocations; opaque post-output telemetry is withheld.")
    }
    return result


def scope_history(branch, history):
    # Each episode already owns its own history. Do not reintroduce raw module
    # entries or an unmasked selector judgment through the scope-review prompt.
    return {"branch": branch, "earlier_same_arm_feedback": deepcopy(history)}
