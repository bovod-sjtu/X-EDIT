"""Discard previous execution facts before an explicitly requested replay."""
from x_editor.core.schemas import AudioEditPlan

def reset_candidate_execution_state(plan: AudioEditPlan) -> None:
    """Remove prior-attempt state when a returned plan is used as repair input."""

    for step in plan.steps:
        step.status = "pending"
        step.result = None
        step.error = None
    runtime_metadata_keys = {
        "artifact_event_span_grounding",
        "artifact_ops_followup_span_reuse",
        "artifact_ops_input_bindings",
        "artifact_ops_input_repairs",
        "artifact_ops_reconstruct_repairs",
        "audio_artifacts",
        "bounded_repair_attempts",
        "debug_audio_artifacts",
        "emphasis_stress_fallbacks",
        "emphasis_stress_perception_repairs",
        "evaluator_repair_best_effort",
        "execution_error",
        "execution_summary",
        "executor_gemini_call_metrics",
        "executor_llm_call_metrics",
        "executor_loops",
        "executor_reference_repairs",
        "executor_step_records",
        "executor_tool_call_errors",
        "executor_tool_call_repairs",
        "executor_tool_calls",
        "final_output_path",
        "flexsed_window_repairs",
        "intermediate_audio_artifacts",
        "intermediate_audios",
        "last_executor_verification_error",
        "last_tool_contract_error",
        "need_repair",
        "need_tool_contract_replan",
        "operation_contract_failures",
        "operation_graph_execution",
        "operation_normalization",
        "operation_normalization_errors",
        "operation_rewrites",
        "perception_audio_repairs",
        "perception_requirement_relaxations",
        "reference_audio_extraction_grounding",
        "runtime_rewrite_rule_hits",
        "runtime_verification_best_effort",
        "service_bootstrap",
        "source_text_repairs",
        "span_selection_param_filters",
        "step_result_reference_resolutions",
        "suffix_resume_reuse_error",
        "suffix_resume_reuse_receipts",
        "suffix_resume_reused_step_ids",
        "suffix_resume_skipped_tool_call_count",
        "text_edit_contract_failures",
        "text_state_updates",
        "tool_contract_best_effort",
        "tool_contract_repair_attempts",
        "tool_contract_violations",
        "tool_invocations",
        "verification_repair_plans",
        "verification_reports",
    }
    for key in runtime_metadata_keys:
        plan.metadata.pop(key, None)

