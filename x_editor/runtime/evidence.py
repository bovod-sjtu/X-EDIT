"""Evidence bundle helpers for X-Editor planning/execution/evaluation runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def build_execution_evidence(
    *,
    sample_id: str | None = None,
    request: str | None = None,
    status: str | None = None,
    input_audios: list[str] | None = None,
    output_audio_path: str | None = None,
    source_text: str | None = None,
    perception: dict[str, Any] | None = None,
    planner_name: str | None = None,
    executor_name: str | None = None,
    plan: Any | None = None,
    trace_result: dict[str, Any] | None = None,
    evaluation: Any | None = None,
    evaluation_history: list[Any] | None = None,
    selected_tools: list[str] | None = None,
    routing_policy_candidates: list[Any] | None = None,
    available_operations: list[Any] | None = None,
    context: dict[str, Any] | None = None,
    error_message: str | None = None,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    """Build a frontend-friendly evidence bundle without invoking new services."""

    context = _mapping(context)
    result = _mapping(trace_result)
    plan_data = _model_dump(plan)
    if not plan_data and isinstance(result.get("plan"), dict):
        plan_data = _model_dump(result.get("plan"))
    if not request:
        request = str(
            result.get("instruction")
            or result.get("request")
            or plan_data.get("request")
            or context.get("user_prompt")
            or ""
        )
    input_audios = _string_list(input_audios or result.get("input_audios") or context.get("input_audios"))
    if not input_audios:
        input_audio = context.get("input_audio") or plan_data.get("input_audio")
        if input_audio:
            input_audios = [str(input_audio)]
    failure_registry = _mapping(result.get("workflow_failure_registry"))
    suffix_resume_checkpoint = _mapping(result.get("suffix_resume_checkpoint"))
    suffix_resume_execution_admission = _mapping(
        result.get("suffix_resume_execution_admission")
    )
    final_delivery_manifest_shadow = _mapping(
        result.get("final_delivery_manifest_shadow")
    )
    if failure_registry.get("status") == "unresolved":
        output_audio_path = failure_registry.get("authoritative_final_output_path")
    else:
        output_audio_path = (
            output_audio_path
            or result.get("output_audio_path")
            or _metadata(plan_data).get("final_output_path")
            or _final_output_from_plan(plan_data)
        )
    perception = perception if isinstance(perception, dict) else _mapping(result.get("perception"))
    source_text = source_text or result.get("source_text") or perception.get("source_text")
    status = status or str(result.get("status") or _status_from_plan(plan_data) or "")
    error_message = error_message or result.get("error_message") or _metadata(plan_data).get("execution_error")

    steps = _plan_steps(plan_data)
    if not steps:
        steps = _dict_list(result.get("plan"))
    metadata = _metadata(plan_data)
    tool_trace = _dict_list(result.get("tool_trace")) or _tool_trace_from_plan_steps(steps)
    tool_call_outputs = _dict_list(result.get("tool_call_outputs")) or _tool_call_outputs_from_trace(tool_trace)
    audio_artifacts = _dedupe_artifacts(
        _dict_list(result.get("audio_artifacts"))
        or _dict_list(metadata.get("audio_artifacts"))
        or _audio_artifacts_from_tool_outputs(tool_call_outputs)
    )
    intermediate_audios = _dedupe_artifacts(
        _dict_list(result.get("intermediate_audio_artifacts"))
        or _dict_list(result.get("intermediate_audios"))
        or _dict_list(metadata.get("intermediate_audio_artifacts"))
        or [item for item in audio_artifacts if item.get("kind") == "step_output"]
    )
    selected_tools = _string_list(selected_tools or result.get("selected_tools")) or _tools_from_steps_and_trace(
        steps,
        tool_trace,
    )
    planner_skill_trace = extract_planner_skill_trace(result if result else plan_data)
    routing_policy_candidates = (
        (routing_policy_candidates if isinstance(routing_policy_candidates, list) else [])
        or _list_or_empty(result.get("routing_policy_candidates"))
        or planner_skill_trace["routing_policy_candidates"]
    )
    planner_skill_ids = planner_skill_trace["planner_skill_ids"]
    evaluation_data = _model_dump(evaluation)
    if not evaluation_data and isinstance(result.get("evaluation"), dict):
        evaluation_data = _model_dump(result.get("evaluation"))
    evaluation_history_data = [_model_dump(item) for item in evaluation_history or result.get("evaluation_history") or []]
    executor_step_records = _dict_list(metadata.get("executor_step_records"))
    executor_loops = _dict_list(metadata.get("executor_loops"))
    tool_invocations = _dict_list(metadata.get("tool_invocations"))
    evaluation_revisions = _int_or_zero(result.get("evaluation_revisions"))
    if evaluation_history_data:
        evaluation_revisions = max(evaluation_revisions, len(evaluation_history_data) - 1)
    localization = _localization_evidence(
        perception=perception,
        tool_call_outputs=tool_call_outputs,
        tool_trace=tool_trace,
        result=result,
    )
    planner_perception_context = _planner_perception_context(context, result)
    session_graph = _mapping(result.get("edit_session_graph")) or _mapping(context.get("edit_session_graph"))
    session_graph_context = _mapping(context.get("edit_session_graph_context")) or _mapping(
        result.get("edit_session_graph_context")
    )
    multi_round = _mapping(result.get("multi_round")) or _mapping(context.get("multi_round"))

    tool_counts = Counter(selected_tools)
    for row in tool_trace:
        tool_id = row.get("tool_id") or row.get("operation")
        if tool_id:
            tool_counts[str(tool_id)] += 1
    for row in tool_invocations:
        tool_id = row.get("tool_id")
        if tool_id:
            tool_counts[str(tool_id)] += 1

    evaluator_metadata = _mapping(evaluation_data.get("metadata"))
    flow = {
        "sample_id": sample_id,
        "status": status or None,
        "loop_count": {
            "planner_executor_attempts": len(evaluation_history_data) or (1 if steps else 0),
            "evaluation_revisions": evaluation_revisions,
            "executor_scheduler_loops": len(executor_loops),
            "edit_rounds": _int_or_zero(multi_round.get("round_count")),
        },
        "multi_round": _json_safe(multi_round),
        "modules": _without_empty(
            {
                "planner": planner_name or result.get("planner") or _planner_from_context(context),
                "executor": executor_name or result.get("executor") or _executor_from_result(result, plan_data),
                "evaluator": evaluator_metadata.get("evaluator_display_name")
                or evaluator_metadata.get("evaluator")
                or _evaluator_from_evaluation(evaluation_data),
            }
        ),
        "stages": [
            _stage("perception", "completed" if perception else "not_recorded", details=_perception_summary(perception)),
            _stage(
                "localization",
                "completed" if localization.get("selected_spans") or localization.get("sources") else "not_recorded",
                details={
                    "selected_span_count": len(localization.get("selected_spans") or []),
                    "localization_source_count": len(localization.get("sources") or []),
                    "perception_segment_count": len(localization.get("perception_segments") or []),
                },
            ),
            _stage(
                "planning",
                "completed" if steps or result.get("plan") else "not_recorded",
                module=planner_name or result.get("planner"),
                details={
                    "step_count": len(steps),
                    "selected_tools": selected_tools,
                    "routing_policy_candidate_count": len(routing_policy_candidates),
                    "planner_skill_ids": planner_skill_ids,
                },
            ),
            _stage(
                "execution",
                "completed" if str(status).lower() == "completed" else ("failed" if error_message else "recorded"),
                module=executor_name or result.get("executor"),
                details={
                    "tool_trace_steps": len(tool_trace),
                    "tool_invocations": len(tool_invocations),
                    "intermediate_audio_count": len(intermediate_audios),
                    "error_message": error_message,
                },
            ),
            *(
                [
                    _stage(
                        "finalization",
                        "failed"
                        if failure_registry.get("status") == "unresolved"
                        else "completed",
                        details={
                            "terminal_failure_stage": failure_registry.get(
                                "terminal_failure_stage"
                            ),
                            "failed_step_ids": failure_registry.get(
                                "terminal_failed_step_ids"
                            )
                            or [],
                            "failure_codes": failure_registry.get(
                                "unresolved_failure_codes"
                            )
                            or [],
                            "repair_count": len(
                                failure_registry.get("repair_history") or []
                            ),
                            "suffix_resume_status": suffix_resume_checkpoint.get(
                                "status"
                            ),
                            "restart_step_ids": suffix_resume_checkpoint.get(
                                "restart_step_ids"
                            )
                            or [],
                            "reusable_step_ids": suffix_resume_checkpoint.get(
                                "reusable_step_ids"
                            )
                            or [],
                            "suffix_resume_execution_status": (
                                suffix_resume_execution_admission.get("status")
                            ),
                            "actually_reused_step_ids": (
                                suffix_resume_execution_admission.get(
                                    "reused_step_ids"
                                )
                                or []
                            ),
                            "skipped_tool_call_count": (
                                suffix_resume_execution_admission.get(
                                    "skipped_tool_call_count"
                                )
                                or 0
                            ),
                            "final_delivery_manifest_status": (
                                final_delivery_manifest_shadow.get("status")
                            ),
                            "authoritative_delivery_artifact_ref_id": (
                                final_delivery_manifest_shadow.get(
                                    "committed_final_artifact_ref_id"
                                )
                            ),
                            "ordered_delivery_component_count": len(
                                final_delivery_manifest_shadow.get(
                                    "ordered_components"
                                )
                                or []
                            ),
                        },
                    )
                ]
                if failure_registry
                else []
            ),
            _stage(
                "evaluation",
                _evaluation_status(evaluation_data),
                details={
                    "passed": evaluation_data.get("passed"),
                    "score": evaluation_data.get("score"),
                    "rubric_count": len(evaluator_metadata.get("rubrics") or []),
                    "answer_count": len(evaluator_metadata.get("answers") or []),
                    "need_replan": evaluation_data.get("need_replan"),
                },
            ),
        ],
    }

    evidence = {
        "schema_version": "x-editor.execution_evidence.v1",
        "sample_id": sample_id,
        "request": request or None,
        "status": status or None,
        "source_text": source_text,
        "files": _without_empty(
            {
                "input_audios": input_audios,
                "output_audio_path": output_audio_path,
                "evidence_path": evidence_path,
                "edit_session_graph_path": result.get("edit_session_graph_path") or context.get("edit_session_graph_path"),
            }
        ),
        "run_flow": flow,
        "planner": _without_empty(
            {
                "name": planner_name or result.get("planner"),
                "request": request or None,
                "available_operations": _json_safe(available_operations or context.get("available_operations") or []),
                "routing_policy_candidates": _json_safe(routing_policy_candidates),
                "planner_skill_ids": planner_skill_ids,
                "perception_context": _json_safe(planner_perception_context),
                "session_graph_context": _json_safe(session_graph_context),
                "gemini_planner_call": result.get("gemini_planner_call") or metadata.get("gemini_planner_call"),
                "selected_tools": selected_tools,
                "plan": _json_safe(plan_data or result.get("plan")),
            }
        ),
        "executor": _without_empty(
            {
                "name": executor_name or result.get("executor"),
                "execution_contract": result.get("execution_contract")
                or metadata.get("execution_contract")
                or ("AudioEditPlan" if steps else None),
                "execution_summary": result.get("execution_summary") or metadata.get("execution_summary"),
                "scheduler_loops": executor_loops,
                "step_records": executor_step_records,
                "tool_trace": _json_safe(tool_trace),
                "tool_invocations": _json_safe(tool_invocations),
                "executor_tool_calls": metadata.get("executor_tool_calls"),
                "executor_tool_call_errors": metadata.get("executor_tool_call_errors"),
                "executor_gemini_call_metrics": metadata.get("executor_gemini_call_metrics"),
                "operation_normalization": metadata.get("operation_normalization"),
                "operation_normalization_errors": metadata.get("operation_normalization_errors"),
                "operation_graph_execution": metadata.get("operation_graph_execution"),
                "operation_graph_actual": metadata.get("operation_graph_actual"),
                "operation_graph_validation": metadata.get("operation_graph_validation"),
                "operation_graph_round_store": metadata.get("operation_graph_round_store"),
                "operation_graph_round_validation": metadata.get("operation_graph_round_validation"),
                "operation_graph_reconciliation": metadata.get("operation_graph_reconciliation"),
                "session_event_log": metadata.get("session_event_log"),
                "operation_rewrites": metadata.get("operation_rewrites"),
                "span_selection_param_filters": metadata.get("span_selection_param_filters"),
                "service_bootstrap": metadata.get("service_bootstrap") or result.get("service_bootstrap"),
                "suffix_resume_execution_admission": _json_safe(
                    suffix_resume_execution_admission
                ),
                "suffix_resume_reuse_receipts": _json_safe(
                    metadata.get("suffix_resume_reuse_receipts") or []
                ),
                "error_message": error_message,
            }
        ),
        "tools": _without_empty(
            {
                "selected_tools": selected_tools,
                "tool_counts": dict(sorted(tool_counts.items())),
                "tool_call_outputs": _json_safe(tool_call_outputs),
                "audio_artifacts": _json_safe(audio_artifacts),
                "intermediate_audio_artifacts": _json_safe(intermediate_audios),
            }
        ),
        "evaluator": _without_empty(
            {
                "evaluation": _json_safe(evaluation_data),
                "evaluation_history": _json_safe(evaluation_history_data),
                "rubrics": _json_safe(evaluator_metadata.get("rubrics") or []),
                "answers": _json_safe(evaluator_metadata.get("answers") or []),
                "rubric_evidence": _json_safe(evaluator_metadata.get("rubric_evidence")),
                "judge_service_bootstrap": _json_safe(evaluator_metadata.get("judge_service_bootstrap")),
            }
        ),
        "perception": _json_safe(perception),
        "localization": _json_safe(localization),
        "multi_round": _json_safe(multi_round),
        "session_graph": _json_safe(session_graph),
        "failure_registry": _json_safe(failure_registry),
        "suffix_resume_checkpoint": _json_safe(suffix_resume_checkpoint),
        "suffix_resume_execution_admission": _json_safe(
            suffix_resume_execution_admission
        ),
        "final_delivery_manifest_shadow": _json_safe(
            final_delivery_manifest_shadow
        ),
    }
    return _json_safe(_without_empty(evidence))


def extract_planner_skill_trace(result_or_plan: Any) -> dict[str, Any]:
    """Return planner skill evidence already carried by a runtime result or plan."""

    result = _mapping(result_or_plan)
    if result.get("steps") is not None and result.get("metadata") is not None:
        plan_data = result
    elif result:
        plan_data = _model_dump(result.get("plan"))
    else:
        plan_data = _model_dump(result_or_plan)
    metadata = _metadata(plan_data)
    candidates = _list_or_empty(result.get("routing_policy_candidates")) or _list_or_empty(
        metadata.get("routing_policy_candidates")
    )
    skill_ids = _string_list(result.get("planner_skill_ids") or metadata.get("planner_skill_ids"))
    if not skill_ids:
        skill_ids = _planner_skill_ids_from_candidates(candidates)
    return {
        "routing_policy_candidates": _json_safe(candidates),
        "planner_skill_ids": skill_ids,
    }


def write_evidence_bundle(
    evidence: dict[str, Any],
    output_dir: str | Path,
    *,
    filename: str = "evidence.json",
) -> str:
    """Persist an evidence bundle and update its own ``files.evidence_path``."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    updated = dict(evidence)
    files = dict(_mapping(updated.get("files")))
    files["evidence_path"] = str(path)
    updated["files"] = files
    run_flow = dict(_mapping(updated.get("run_flow")))
    if run_flow:
        run_flow.setdefault("sample_id", updated.get("sample_id"))
        updated["run_flow"] = run_flow
    path.write_text(json.dumps(_json_safe(updated), ensure_ascii=False, indent=2), encoding="utf-8")
    evidence.clear()
    evidence.update(updated)
    return str(path)


def load_or_build_trace_evidence(trace: dict[str, Any]) -> dict[str, Any]:
    """Return persisted/inline evidence, falling back to a derived legacy bundle."""

    inline = trace.get("evidence")
    if isinstance(inline, dict):
        return _augment_evidence_with_trace(inline, trace)
    path = trace.get("evidence_path")
    if isinstance(path, str) and path:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return _augment_evidence_with_trace(data, trace)
        except (OSError, json.JSONDecodeError):
            pass
    return build_execution_evidence(
        sample_id=str(trace.get("id") or trace.get("sample_id") or "") or None,
        request=str(trace.get("instruction") or trace.get("request") or "") or None,
        status=str(trace.get("status") or "") or None,
        input_audios=_string_list(trace.get("input_audios")),
        output_audio_path=trace.get("output_audio_path"),
        source_text=trace.get("source_text"),
        perception=_mapping(trace.get("perception")),
        plan=trace.get("plan"),
        trace_result=trace,
        evaluation=trace.get("evaluation"),
        evaluation_history=_list_or_empty(trace.get("evaluation_history")),
        selected_tools=_string_list(trace.get("selected_tools")),
        routing_policy_candidates=_list_or_empty(trace.get("routing_policy_candidates")),
        error_message=trace.get("error_message"),
        evidence_path=path if isinstance(path, str) else None,
    )


def _augment_evidence_with_trace(evidence: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Backfill newly supported evidence sections for older persisted bundles."""

    updated = _json_safe(evidence)
    if not isinstance(updated, dict):
        updated = {}
    if not updated.get("localization"):
        localization = _localization_evidence(
            perception=_mapping(trace.get("perception")) or _mapping(updated.get("perception")),
            tool_call_outputs=_dict_list(trace.get("tool_call_outputs"))
            or _dict_list(_mapping(updated.get("tools")).get("tool_call_outputs")),
            tool_trace=_dict_list(trace.get("tool_trace")) or _dict_list(_mapping(updated.get("executor")).get("tool_trace")),
            result=trace,
        )
        if localization:
            updated["localization"] = localization
            _upsert_localization_stage(updated, localization)
    return updated


def _upsert_localization_stage(evidence: dict[str, Any], localization: dict[str, Any]) -> None:
    flow = _mapping(evidence.get("run_flow"))
    stages = _dict_list(flow.get("stages"))
    if not flow or any(stage.get("name") == "localization" for stage in stages):
        return
    details = {
        "selected_span_count": len(localization.get("selected_spans") or []),
        "localization_source_count": len(localization.get("sources") or []),
        "perception_segment_count": len(localization.get("perception_segments") or []),
    }
    stage = _stage(
        "localization",
        "completed" if details["selected_span_count"] or details["localization_source_count"] else "not_recorded",
        details=details,
    )
    insert_at = 1 if stages and stages[0].get("name") == "perception" else len(stages)
    stages.insert(insert_at, stage)
    flow["stages"] = stages
    evidence["run_flow"] = flow


def evidence_filename_for_sample(sample_id: str | None) -> str:
    if not sample_id:
        return "evidence.json"
    return f"{_safe_filename(sample_id)}.evidence.json"


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _metadata(plan_data: dict[str, Any]) -> dict[str, Any]:
    return _mapping(plan_data.get("metadata"))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if value is None:
        return []
    return [str(value)]


def _planner_skill_ids_from_candidates(candidates: list[Any]) -> list[str]:
    skill_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        policy_id = candidate.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            continue
        if policy_id not in skill_ids:
            skill_ids.append(policy_id)
    return skill_ids


def _plan_steps(plan_data: dict[str, Any]) -> list[dict[str, Any]]:
    return _dict_list(plan_data.get("steps"))


def _status_from_plan(plan_data: dict[str, Any]) -> str | None:
    steps = _plan_steps(plan_data)
    if not steps:
        return None
    if any(step.get("status") == "failed" for step in steps):
        return "failed"
    if all(step.get("status") == "success" for step in steps):
        return "completed"
    return "executed"


def _final_output_from_plan(plan_data: dict[str, Any]) -> str | None:
    for step in reversed(_plan_steps(plan_data)):
        result = _mapping(step.get("result"))
        for key in ("output_audio_path", "target_audio_path", "output_path", "generated_audio_path"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    value = plan_data.get("output_audio")
    return str(value) if value else None


def _tool_trace_from_plan_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for step in steps:
        rows.append(
            _without_empty(
                {
                    "step_id": step.get("id"),
                    "tool_id": step.get("operation"),
                    "operation": step.get("operation"),
                    "success": step.get("status") == "success",
                    "output": step.get("result") if isinstance(step.get("result"), dict) else {},
                    "error_message": step.get("error"),
                    "execution_contract": "AudioEditPlan",
                }
            )
        )
    return rows


def _tool_call_outputs_from_trace(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(tool_trace):
        output = _mapping(item.get("output"))
        rows.append(
            _without_empty(
                {
                    "index": index,
                    "step_id": item.get("step_id"),
                    "tool_id": item.get("tool_id") or item.get("operation"),
                    "operation": item.get("operation") or item.get("tool_id"),
                    "success": item.get("success"),
                    "arguments": item.get("arguments"),
                    "output": output,
                    "error_message": item.get("error_message"),
                    "audio_artifacts": _audio_artifacts_from_mapping(
                        output,
                        step_id=item.get("step_id"),
                        operation=item.get("operation") or item.get("tool_id"),
                        source_prefix=f"tool_trace[{index}].output",
                    ),
                }
            )
        )
    return rows


def _audio_artifacts_from_tool_outputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for row in rows:
        artifacts.extend(_dict_list(row.get("audio_artifacts")))
        artifacts.extend(
            _audio_artifacts_from_mapping(
                row.get("output"),
                step_id=row.get("step_id"),
                operation=row.get("operation") or row.get("tool_id"),
                source_prefix=f"tool_call_outputs[{row.get('index')}].output",
            )
        )
    return _dedupe_artifacts(artifacts)


def _audio_artifacts_from_mapping(
    value: Any,
    *,
    step_id: Any,
    operation: Any,
    source_prefix: str,
) -> list[dict[str, Any]]:
    artifacts = []
    for key_path, path_text in _iter_audio_paths(value):
        path = Path(str(path_text))
        artifacts.append(
            {
                "artifact_id": f"{step_id or 'step'}:{_safe_filename(key_path)}",
                "step_id": step_id,
                "operation": operation,
                "role": key_path.split(".")[-1].split("[", 1)[0],
                "kind": "step_output",
                "path": str(path),
                "exists": path.exists(),
                "source": f"{source_prefix}.{key_path}",
                "result_key": key_path,
            }
        )
    return artifacts


def _iter_audio_paths(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_iter_audio_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_iter_audio_paths(item, child))
    elif isinstance(value, str) and prefix and Path(value).suffix.lower() in _AUDIO_EXTENSIONS:
        paths.append((prefix, value))
    return paths


def _tools_from_steps_and_trace(steps: list[dict[str, Any]], trace: list[dict[str, Any]]) -> list[str]:
    tools: list[str] = []
    for step in steps:
        operation = step.get("operation")
        if operation:
            tools.append(str(operation))
    for row in trace:
        value = row.get("tool_id") or row.get("operation")
        if value:
            tools.append(str(value))
    deduped: list[str] = []
    for tool in tools:
        if tool not in deduped:
            deduped.append(tool)
    return deduped


def _perception_summary(perception: dict[str, Any]) -> dict[str, Any]:
    if not perception:
        return {}
    return _without_empty(
        {
            "has_source_text": bool(perception.get("source_text") or perception.get("text")),
            "word_timestamp_count": len(perception.get("word_timestamps") or []),
            "speaker_segment_count": len(perception.get("speaker_segments") or perception.get("segments") or []),
            "attempt_count": len(perception.get("perception_attempts") or []),
            "errors": perception.get("errors"),
            "services": perception.get("services"),
        }
    )


def _planner_perception_context(context: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Expose planner-side perception guidance without changing execution data."""

    keys = (
        "enable_planner_perception_guidance",
        "planner_perception_guidance",
        "span_selection_analysis",
        "perception_gap_fill_guidance",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        value = context.get(key)
        if value in (None, "", [], {}):
            value = result.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    return _without_empty(payload)


def _localization_evidence(
    *,
    perception: dict[str, Any],
    tool_call_outputs: list[dict[str, Any]],
    tool_trace: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Collect audio-span selection evidence from perception and tool outputs."""

    sources: list[dict[str, Any]] = []
    selected_spans: list[dict[str, Any]] = []
    for index, row in enumerate(tool_call_outputs):
        output = _mapping(row.get("output"))
        for source in _collect_localization_sources(
            output,
            source_prefix=f"tool_call_outputs[{index}].output",
            step_id=row.get("step_id"),
            operation=row.get("operation") or row.get("tool_id"),
        ):
            sources.append(source)
            selected_spans.extend(_selected_spans_from_source(source))

    if not tool_call_outputs:
        for index, row in enumerate(tool_trace):
            output = _mapping(row.get("output"))
            for source in _collect_localization_sources(
                output,
                source_prefix=f"tool_trace[{index}].output",
                step_id=row.get("step_id"),
                operation=row.get("operation") or row.get("tool_id"),
            ):
                sources.append(source)
                selected_spans.extend(_selected_spans_from_source(source))

    for source in _collect_localization_sources(
        result,
        source_prefix="trace_result",
        step_id=None,
        operation=None,
        skip_keys={
            "tool_call_outputs",
            "tool_trace",
            "evidence",
            "plan",
            "evaluation",
            "evaluation_history",
            "second_pass",
            "metadata",
        },
    ):
        sources.append(source)
        selected_spans.extend(_selected_spans_from_source(source))

    perception_segments = _perception_segments(perception)
    selected_spans = _dedupe_span_rows(selected_spans)
    sources = _dedupe_localization_sources(sources)
    if not sources and not selected_spans and not perception_segments:
        return {}
    return _without_empty(
        {
            "summary": {
                "selected_span_count": len(selected_spans),
                "localization_source_count": len(sources),
                "perception_segment_count": len(perception_segments),
            },
            "selected_spans": selected_spans,
            "sources": sources,
            "perception_segments": perception_segments[:50],
        }
    )


def _collect_localization_sources(
    value: Any,
    *,
    source_prefix: str,
    step_id: Any,
    operation: Any,
    skip_keys: set[str] | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    if depth > 7:
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if skip_keys and str(key) in skip_keys:
                continue
            child = f"{source_prefix}.{key}"
            normalized_key = str(key).lower()
            if isinstance(item, dict) and normalized_key in {
                "perception_segment_selection",
                "span_selection_evidence",
                "segment_selection",
                "selection_evidence",
                "timeline_selection",
            }:
                rows.append(_localization_source(item, child, step_id=step_id, operation=operation))
            elif isinstance(item, list) and normalized_key in {
                "selected_spans",
                "target_spans",
                "timeline_spans",
                "edited_spans",
            } and _looks_like_span_list(item):
                rows.append(
                    _localization_source(
                        {"selected_spans": item, "selected_count": len(item)},
                        child,
                        step_id=step_id,
                        operation=operation,
                    )
                )
            rows.extend(
                _collect_localization_sources(
                    item,
                    source_prefix=child,
                    step_id=step_id,
                    operation=operation,
                    skip_keys=skip_keys,
                    depth=depth + 1,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(
                _collect_localization_sources(
                    item,
                    source_prefix=f"{source_prefix}[{index}]",
                    step_id=step_id,
                    operation=operation,
                    skip_keys=skip_keys,
                    depth=depth + 1,
                )
            )
    return rows


def _localization_source(
    payload: dict[str, Any],
    source_path: str,
    *,
    step_id: Any,
    operation: Any,
) -> dict[str, Any]:
    judgements = _dict_list(payload.get("judgements"))
    selected = (
        _dict_list(payload.get("selected_spans"))
        or _dict_list(payload.get("target_spans"))
        or _dict_list(payload.get("timeline_spans"))
        or _dict_list(payload.get("spans"))
    )
    selected_ids = _string_list(payload.get("selected_segment_ids"))
    return _without_empty(
        {
            "source": source_path,
            "step_id": step_id,
            "operation": operation or payload.get("action") or payload.get("operation"),
            "status": payload.get("status"),
            "provider": payload.get("provider"),
            "instruction": payload.get("instruction"),
            "action": payload.get("action"),
            "candidate_count": payload.get("candidate_count"),
            "selected_count": payload.get("selected_count") or len(selected) or len(selected_ids),
            "selected_segment_ids": selected_ids,
            "reason": payload.get("reason"),
            "selected_spans": selected,
            "judgements": judgements,
            "raw": payload,
        }
    )


def _selected_spans_from_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    operation = source.get("operation")
    step_id = source.get("step_id")
    selected_ids = {str(item) for item in source.get("selected_segment_ids") or []}
    for span in _dict_list(source.get("selected_spans")):
        spans.append(_span_row(span, source=source.get("source"), step_id=step_id, operation=operation))
    for judgement in _dict_list(source.get("judgements")):
        segment_id = judgement.get("segment_id") or judgement.get("id")
        if judgement.get("matches") is True or (segment_id is not None and str(segment_id) in selected_ids):
            spans.append(_span_row(judgement, source=source.get("source"), step_id=step_id, operation=operation))
    return spans


def _span_row(value: dict[str, Any], *, source: Any, step_id: Any, operation: Any) -> dict[str, Any]:
    return _without_empty(
        {
            "segment_id": value.get("segment_id") or value.get("id"),
            "start_sec": _number_or_none(_first_present(value.get("start_sec"), value.get("start"))),
            "end_sec": _number_or_none(_first_present(value.get("end_sec"), value.get("end"))),
            "speaker": value.get("speaker") or value.get("speaker_id"),
            "text": value.get("text") or value.get("transcript"),
            "label": value.get("label"),
            "confidence": _number_or_none(value.get("confidence")),
            "reason": value.get("reason"),
            "detected_properties": value.get("detected_properties"),
            "source": source,
            "step_id": step_id,
            "operation": operation,
        }
    )


def _perception_segments(perception: dict[str, Any]) -> list[dict[str, Any]]:
    segments = (
        _dict_list(perception.get("speaker_segments"))
        or _dict_list(perception.get("segments"))
        or _dict_list(perception.get("timeline_segments"))
    )
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        rows.append(
            _without_empty(
                {
                    "segment_id": segment.get("segment_id") or segment.get("id") or f"perception_{index:03d}",
                    "start_sec": _number_or_none(_first_present(segment.get("start_sec"), segment.get("start"))),
                    "end_sec": _number_or_none(_first_present(segment.get("end_sec"), segment.get("end"))),
                    "speaker": segment.get("speaker") or segment.get("speaker_id"),
                    "text": segment.get("text") or segment.get("transcript"),
                    "label": segment.get("label") or segment.get("description"),
                    "source": "perception",
                }
            )
        )
    return rows


def _looks_like_span_list(values: list[Any]) -> bool:
    for item in values:
        if not isinstance(item, dict):
            continue
        if any(key in item for key in ("start_sec", "end_sec", "start", "end", "segment_id", "span")):
            return True
    return False


def _dedupe_span_rows(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in values:
        key = (
            str(item.get("segment_id") or ""),
            str(item.get("start_sec") or ""),
            str(item.get("end_sec") or ""),
            str(item.get("operation") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(_json_safe(item))
    return rows


def _dedupe_localization_sources(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        key = str(item.get("source") or json.dumps(_json_safe(item), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        rows.append(_json_safe(item))
    return rows


def _number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _stage(
    name: str,
    status: str,
    *,
    module: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _without_empty({"name": name, "status": status, "module": module, "details": _without_empty(details or {})})


def _planner_from_context(context: dict[str, Any]) -> str | None:
    planner = context.get("planner")
    return str(planner) if planner else None


def _executor_from_result(result: dict[str, Any], plan_data: dict[str, Any]) -> str | None:
    if result.get("execution_contract") == "AudioEditPlan" or _plan_steps(plan_data):
        return "AudioEditExecutor"
    return None


def _evaluator_from_evaluation(evaluation: dict[str, Any]) -> str | None:
    if evaluation:
        metadata = _mapping(evaluation.get("metadata"))
        return str(metadata.get("evaluator") or metadata.get("evaluator_display_name") or "Evaluator")
    return None


def _evaluation_status(evaluation: dict[str, Any]) -> str:
    if not evaluation:
        return "not_recorded"
    if _mapping(evaluation.get("metadata")).get("verification_unavailable"):
        return "unavailable"
    return "passed" if evaluation.get("passed") is True else "failed"


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any, *, depth: int = 10) -> Any:
    if depth <= 0:
        return _truncate(str(value), 500)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_safe(value.model_dump(), depth=depth - 1)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth - 1)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth - 1) for item in value]
    return _truncate(str(value), 500)


def _dedupe_artifacts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str, str]] = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        key = (
            str(item.get("kind") or ""),
            str(item.get("step_id") or ""),
            _resolved_path(path) if path else json.dumps(_json_safe(item), sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(_json_safe(item))
    return rows


def _resolved_path(value: Any) -> str:
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("_")[:160] or "sample"


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
