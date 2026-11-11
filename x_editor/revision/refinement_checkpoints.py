"""Production review support extracted without changing function bodies."""

from pathlib import Path


from x_editor.revision.atomic_module_review import public_context, read


from x_editor.revision.natural_atomic_checkpoints import (
    jsonl,
    link_checkpoint,
    planner_wire_index,
    read_output,
)


from x_editor.evaluation.atomic_module_contract_audit import audit_module_contract


from x_editor.runtime.module_checkpoints import ModuleCheckpoint, _fingerprint


from x_editor.evaluation.adviser_projection import absent_adviser, actual_adviser_io, planner_advice_from_wire


def recorded_tool_observations(turn_result):
    """Actual tool returns for the next-attempt critic, not acoustic truth.

    Kept outside the earlier module's grading input: a later backend defect
    does not automatically make an earlier reasoning decision incorrect.
    """
    forward = turn_result.forward_result
    if forward is None or forward.executed_plan is None:
        return []
    observations = []
    for step in forward.executed_plan.steps:
        result = step.result if isinstance(step.result, dict) else {}
        observations.append(
            {
                "step_id": step.id,
                "operation": step.operation,
                "status": step.status,
                "actual_inputs": step.inputs,
                "actual_params": step.params,
                "error": step.error,
                "reported_metadata": result.get("metadata"),
                "reported_source_spans": result.get("source_spans"),
                "reported_target_spans": result.get("target_spans"),
                "reported_temporal_transform": result.get("audio_temporal_transform"),
                "reported_duration_sec": result.get("duration_seconds"),
                "reported_target_text": result.get("target_text"),
                "authority": "recorded_tool_report_not_acoustic_truth",
            }
        )
    return observations


def latest_module_sources(scenario, result, root: Path):
    checkpoints, rows, errors = [], {}, []
    turns = {turn.turn_index: turn for turn in result.turn_results}
    for path in sorted((root / "turns").glob("*/execution/module_checkpoints.jsonl")):
        for payload in jsonl(path):
            checkpoint = ModuleCheckpoint.model_validate(payload)
            if checkpoint.module not in {"reasoner", "planner", "planner_reasoner"}:
                continue
            turn = turns.get(checkpoint.context.turn_index)
            if (
                turn is None
                or checkpoint.context.workflow_run_id != turn.result.run_id
                or checkpoint.context.runtime_case_fingerprint != turn.runtime_case_fingerprint
                or _fingerprint(checkpoint.output) != checkpoint.output_fingerprint
            ):
                raise ValueError("recorded checkpoint identity differs from this attempt")
            checkpoints.append((checkpoint, str(path)))
    wire = planner_wire_index(root)
    for module in ("reasoner", "planner"):
        for call in sorted((root / "module_invocations" / module).glob("*")):
            if not call.is_dir():
                continue
            module_input = {}
            try:
                module_input = read(call / "input.json")["args"][0]
                receipt = read(call / "receipt.json")
                checkpoint, _ = link_checkpoint(module, module_input, receipt, checkpoints)
                turn = checkpoint.context.turn_index
                if module == 'reasoner' and absent_adviser(receipt.get('invocation_trace')):
                    continue  # Adapter compatibility memo is not an independent module.
                context = public_context(scenario.model_dump(mode="json"), turn)
                if module_input.get("user_request") != scenario.turns[turn].user_request:
                    raise ValueError("recorded module input belongs to a different public request")
                module_output, accepted, representation = read_output(
                    call, module, module_input, wire
                )
                if module == 'reasoner':
                    actual = actual_adviser_io(module_input, receipt.get('invocation_trace'))
                    if actual is not None:
                        module_input, module_output = actual
                        accepted = True
                        representation = {
                            'representation': 'actual_independent_adviser',
                            'source': str(call / 'receipt.json'),
                            'adapter_summary_used': False,
                        }
                audit = audit_module_contract(
                    module, module_input, module_output, accepted=accepted, receipt=receipt
                )
                audit["output_representation"] = representation
                # Admission can reject a well-shaped plan after the module returns.
                # Preserve that symptom, without turning it into a semantic grade.
                audit["observed_turn_terminal"] = {
                    "status": turns[turn].result.status,
                    "failure_stage": turns[turn].result.failure_stage,
                    "failure_codes": turns[turn].result.failure_codes,
                    "terminal_error_message": turns[turn].result.terminal_error_message,
                }
                forward = turns[turn].result.forward_result
                if forward is not None:
                    audit["observed_turn_terminal"]["forward_admission"] = (
                        forward.admission.model_dump(mode="json") if forward.admission else None
                    )
                    audit["observed_turn_terminal"]["admission_input_plan"] = (
                        forward.input_plan.model_dump(mode="json") if forward.input_plan else None
                    )
                # Module acceptance and graph compilation are narrower checks
                # than the recorded operation-specific admission decision.
                audit["module_acceptance_scope"] = (
                    "accepted_in_recorded_run refers to this module response only. "
                    "Compare observed_turn_terminal.forward_admission against its actual "
                    "admission_input_plan before claiming executable parameter validity."
                )
                trace_meta = (receipt.get('invocation_trace') or {}).get('metadata') or {}
                if module == 'planner' and trace_meta.get('architecture') == 'common_planner_advisory':
                    advice, prompt_path = planner_advice_from_wire(module_input['planner_request_id'], wire)
                    audit['actual_planner_prompt_source'] = prompt_path
                    # Schema checks above use the original typed compiler input.
                    # The extra field below is actual LLM-visible advice, not the
                    # adapter's synthetic memo or reconstructed reasoning.
                    module_input = {**module_input, 'independent_guidance_actually_supplied': advice}
                rows[(turn, module)] = {
                    "public_context": context.model_dump(mode="json"),
                    "module_input": module_input,
                    "module_output": module_output,
                    "contract_audit": audit,
                }
            except (ValueError, KeyError, OSError, IndexError) as exc:
                # Do not fall back to an earlier accepted output on a later failure.
                request = module_input
                request_id = request.get(
                    "request_id" if module == "reasoner" else "planner_request_id", ""
                )
                for index, turn in turns.items():
                    if request_id.startswith(turn.result.run_id + ":semantic:"):
                        rows.pop((index, module), None)
                errors.append({"module": module, "call": str(call), "error": str(exc)})
    return rows, errors
