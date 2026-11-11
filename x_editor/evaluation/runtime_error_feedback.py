"""Recorded execution failures remain usable when an advisory review fails.

These are runtime observations, not acoustic judgments or replacement plans.
Only an explicitly configured outer controller may deliver them to a new attempt.
"""

import json

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint


def runtime_failure_facts(result):
    facts = {}
    for turn in result.turn_results:
        actual = turn.result
        if actual.status == "completed":
            continue
        forward = actual.forward_result
        calls, findings = [], []
        if forward is not None:
            if forward.executed_plan is not None:
                calls = [dict(step_id=s.id, operation=s.operation, status=s.status,
                              actual_inputs=s.inputs, actual_params=s.params, error=s.error)
                         for s in forward.executed_plan.steps if s.status == "failed" or s.error]
            if forward.admission is not None:
                findings = [f.model_dump(mode="json") for f in forward.admission.findings
                            if f.severity == "blocking"]
                implicated = {f["step_id"] for f in findings if f.get("step_id")}
                if forward.input_plan is not None:
                    calls.extend(dict(step_id=s.id, operation=s.operation,
                                      status="not_executed_admission_rejected",
                                      actual_inputs=s.inputs, actual_params=s.params)
                                 for s in forward.input_plan.steps if s.id in implicated)
        facts[turn.turn_index] = dict(
            turn_index=turn.turn_index, status=actual.status,
            failure_stage=actual.failure_stage, failure_codes=list(actual.failure_codes),
            terminal_error_message=actual.terminal_error_message,
            admission_findings=findings, failed_tool_calls=calls,
        )
    return facts


def runtime_error_guidance(facts, *, attempt_id):
    text = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    if len(text) > 29000:
        raise ValueError("raw errors exceed guidance budget; retained without truncation")
    chunks = [text[i:i + 1800] for i in range(0, len(text), 1800)]
    identity = "runtime-error:" + canonical_evaluation_fingerprint(facts)
    return FutureAttemptGuidance(
        guidance_id=identity, source_attempt_id=attempt_id,
        source_evaluation_result_ids=(identity,), diagnostic_evidence_refs=(identity,),
        target_modules=("reasoner", "planner", "perception"),
        source_failure_stage=facts["failure_stage"] or "runtime",
        source_failure_codes=tuple(c for c in facts["failure_codes"] if len(c) <= 256),
        questions_to_reconsider=(
            "The previous attempt failed with the following unmodified runtime errors. "
            "Reconsider your own inputs using the current request, source evidence and tool contracts. "
            "No evaluator has diagnosed these errors or checked the output sound.",
            *(f"Raw error JSON part {i + 1}/{len(chunks)}: {chunk}" for i, chunk in enumerate(chunks)),
        ),
        constraints_to_preserve=("Preserve the user's edit goals and all non-target content.",),
    )


def retain_runtime_errors(result, *, attempt_id, reviewed_guidance):
    """Return current failed-turn facts plus qualified guidance, with provenance.

    Never carries errors from completed turns or previous attempts. If a merged
    packet exceeds the existing context contract, keep the bounded raw packet
    and explicitly record that the advisory packet was not forwarded.
    """
    facts = runtime_failure_facts(result)
    guidance, dispositions = dict(reviewed_guidance), {}
    for turn_index, value in facts.items():
        try:
            raw = runtime_error_guidance(value, attempt_id=attempt_id)
        except ValueError as exc:
            dispositions[turn_index] = {"action": "not_forwarded", "error": str(exc)}
            continue
        previous = guidance.get(turn_index)
        if previous is None:
            guidance[turn_index] = raw
            dispositions[turn_index] = {"action": "raw_errors_without_model_diagnosis"}
            continue
        if previous.source_attempt_id != attempt_id:
            guidance[turn_index] = raw
            dispositions[turn_index] = {"action": "raw_errors_only_review_attempt_mismatch"}
            continue
        payload = previous.model_dump(mode="json")
        payload.update(guidance_fingerprint="", source_attempt_id=attempt_id,
                       guidance_id="runtime-plus-review:" + canonical_evaluation_fingerprint(
                           [raw.guidance_fingerprint, previous.guidance_fingerprint]))
        for key in ("source_evaluation_result_ids", "diagnostic_evidence_refs", "target_modules",
                    "source_failure_codes", "constraints_to_preserve", "questions_to_reconsider"):
            payload[key] = list(dict.fromkeys([*getattr(raw, key), *getattr(previous, key)]))
        payload["source_failure_stage"] = raw.source_failure_stage
        try:
            guidance[turn_index] = FutureAttemptGuidance.model_validate(payload)
            dispositions[turn_index] = {"action": "raw_errors_and_reviewed_guidance"}
        except ValueError as exc:
            guidance[turn_index] = raw
            dispositions[turn_index] = {"action": "raw_errors_only_context_budget", "error": str(exc)}
    return guidance, {"facts": facts, "dispositions": dispositions}
