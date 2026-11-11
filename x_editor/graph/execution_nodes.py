"""Execution nodes for the tool-aware graph."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from x_editor.core.config import runtime_artifact_path
from x_editor.core.schemas import ToolAction, ToolManual, ValidationReport
from x_editor.core.state import AudioEditState
from x_editor.planner import ToolAwarePlanner
from x_editor.runtime.audio_plan_executor import AudioEditExecutor, execution_summary
from x_editor.runtime.accent_conversion import execute_accent_conversion, requests_reference_accent_match
from x_editor.runtime.artifact_ops import execute_artifact_operation
from x_editor.runtime.foreground_background_remix import execute_foreground_background_remix
from x_editor.runtime.intra_audio_voice_transfer import execute_intra_audio_voice_transfer
from x_editor.runtime.music_vocal_timeline import execute_music_vocal_timeline_edit
from x_editor.runtime.segmented_edit import maybe_execute_segmented_edit_pipeline
from x_editor.runtime.tool_action_adapter import ToolActionAdapterError, tool_actions_to_audio_edit_plan
from x_editor.tools.executor import ToolExecutor
from x_editor.tools.loader import load_manual


def create_tool_execution_node(
    planner: ToolAwarePlanner,
    executor: ToolExecutor | None = None,
    *,
    use_structured_executor: bool = False,
):
    async def node(state: AudioEditState) -> dict[str, Any]:
        tool_executor = executor or ToolExecutor()
        close_executor = executor is None
        artifact_paths = {"input_audio": state["base_audio_path"]}
        artifacts: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        output_dir = str(_absolute_path(state.get("output_dir") or runtime_artifact_path()))

        try:
            if use_structured_executor:
                structured = await _try_structured_executor_path(state, tool_executor, output_dir)
                if structured is not None:
                    return structured
            for item in state.get("plan") or []:
                action = ToolAction(**item)
                if action.tool_id == "artifact_ops":
                    try:
                        result = _execute_artifact_ops(action, artifact_paths, dict(state), state["base_audio_path"], output_dir)
                    except Exception as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "success": False,
                            "output": {},
                            "error_message": error_message,
                        })
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": error_message,
                        }
                    output_path = result["output_audio_path"]
                    traces.append({"step_id": action.step_id, "tool_id": action.tool_id, "success": True, "output": result})
                elif action.tool_id == "foreground_background_remix":
                    try:
                        input_audio = artifact_paths.get(action.input_artifacts[0], state["base_audio_path"])

                        async def execute_existing_tool(
                            tool_id: str,
                            arguments: dict[str, Any],
                            runtime_state: dict[str, Any],
                        ) -> dict[str, Any]:
                            segmented_result = None
                            if tool_id == "edit_data_pipeline":
                                segmented_result = await maybe_execute_segmented_edit_pipeline(
                                    tool_executor=tool_executor,
                                    arguments=arguments,
                                    state=runtime_state,
                                    output_dir=output_dir,
                                )
                            tool_result = segmented_result or await tool_executor.execute(tool_id, arguments)
                            if not tool_result.success:
                                raise RuntimeError(tool_result.error_message or f"Tool failed: {tool_id}")
                            return tool_result.output

                        result = await execute_foreground_background_remix(
                            execute_tool=execute_existing_tool,
                            input_audio_path=input_audio,
                            output_path=str(Path(output_dir) / f"{action.step_id}.wav"),
                            output_dir=str(Path(output_dir) / action.step_id),
                            instruction=str(state.get("user_prompt") or action.objective),
                            source_text=state.get("source_text"),
                            params={},
                            state=dict(state),
                        )
                    except Exception as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "success": False,
                            "output": {},
                            "error_message": error_message,
                        })
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": error_message,
                        }
                    output_path = result["output_audio_path"]
                    traces.append({"step_id": action.step_id, "tool_id": action.tool_id, "success": True, "output": result})
                elif action.tool_id == "ace_step_music":
                    try:
                        input_audio = artifact_paths.get(action.input_artifacts[0], state["base_audio_path"])

                        async def execute_existing_tool(
                            tool_id: str,
                            arguments: dict[str, Any],
                            runtime_state: dict[str, Any],
                        ) -> dict[str, Any]:
                            tool_result = await tool_executor.execute(tool_id, arguments)
                            if not tool_result.success:
                                raise RuntimeError(tool_result.error_message or f"Tool failed: {tool_id}")
                            return tool_result.output

                        result = await execute_music_vocal_timeline_edit(
                            execute_tool=execute_existing_tool,
                            input_audio_path=input_audio,
                            output_path=str(Path(output_dir) / f"{action.step_id}.wav"),
                            output_dir=str(Path(output_dir) / action.step_id),
                            instruction=str(state.get("user_prompt") or action.objective),
                            source_text=state.get("source_text"),
                            perception_report=state.get("perception_report") or {},
                            reference_audio_paths=state.get("reference_audio_paths") or [],
                            params={},
                            state=dict(state),
                        )
                    except Exception as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "success": False,
                            "output": {},
                            "error_message": error_message,
                        })
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": error_message,
                        }
                    output_path = result["output_audio_path"]
                    traces.append({"step_id": action.step_id, "tool_id": action.tool_id, "success": True, "output": result})
                elif action.tool_id == "intra_audio_voice_transfer":
                    result = execute_intra_audio_voice_transfer(
                        state=dict(state),
                        output_path=str(Path(output_dir) / f"{action.step_id}.wav"),
                        output_dir=output_dir,
                        instruction=state.get("user_prompt"),
                    )
                    traces.append({
                        "step_id": action.step_id,
                        "tool_id": action.tool_id,
                        "mcp_tool_name": result.tool_name,
                        "arguments": {
                            "input_audio_path": state["base_audio_path"],
                            "output_path": str(Path(output_dir) / f"{action.step_id}.wav"),
                            "instruction": state.get("user_prompt"),
                        },
                        "success": result.success,
                        "output": result.output,
                        "error_message": result.error_message,
                    })
                    if not result.success:
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": result.error_message or "Intra-audio voice transfer failed",
                        }
                    output_path = _extract_output_path(result.output)
                elif action.tool_id == "accent_conversion":
                    try:
                        input_audio = artifact_paths.get(action.input_artifacts[0], state["base_audio_path"])

                        async def execute_existing_tool(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
                            tool_result = await tool_executor.execute(tool_id, arguments)
                            if not tool_result.success:
                                raise RuntimeError(tool_result.error_message or f"Tool failed: {tool_id}")
                            return tool_result.output

                        reference_audio_path = None
                        if requests_reference_accent_match(str(state.get("user_prompt") or action.objective)):
                            refs = state.get("reference_audio_paths") or []
                            reference_audio_path = refs[0] if refs else None

                        result = await execute_accent_conversion(
                            execute_tool=execute_existing_tool,
                            input_audio_path=input_audio,
                            output_path=str(Path(output_dir) / f"{action.step_id}.wav"),
                            output_dir=output_dir,
                            instruction=str(state.get("user_prompt") or action.objective),
                            source_text=state.get("source_text"),
                            perception_report=state.get("perception_report") or state.get("audio_perception") or {},
                            reference_audio_path=reference_audio_path,
                        )
                    except Exception as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "success": False,
                            "output": {},
                            "error_message": error_message,
                        })
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": error_message,
                        }
                    output_path = result["output_audio_path"]
                    traces.append({"step_id": action.step_id, "tool_id": action.tool_id, "success": True, "output": result})
                elif action.tool_id in {"paralinguistic_event_add", "paralinguistic_event_remove"}:
                    try:
                        input_audio = artifact_paths.get(action.input_artifacts[0], state["base_audio_path"])
                        structured_plan = tool_actions_to_audio_edit_plan(
                            [action],
                            state={
                                **dict(state),
                                "base_audio_path": input_audio,
                                "input_audio": input_audio,
                            },
                        )
                        structured_plan.input_audio = input_audio
                        structured_executor = AudioEditExecutor(
                            tool_executor=tool_executor,
                            output_dir=output_dir,
                            enable_gemini_executor=False,
                        )
                        structured_context = {
                            "input_audio": input_audio,
                            "output_dir": output_dir,
                            "source_text": state.get("source_text"),
                            "perception_report": state.get("perception_report") or state.get("audio_perception") or {},
                            "reference_audio_paths": state.get("reference_audio_paths") or [],
                            **(state.get("runtime_settings") or {}),
                        }
                        executed = await structured_executor.aexecute(structured_plan, context=structured_context)
                        step = executed.steps[0]
                        result = step.result or {}
                        output_path = _extract_output_path(result)
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "success": step.status == "success",
                            "output": result,
                            "error_message": step.error,
                            "execution_contract": "AudioEditPlan",
                        })
                        if step.status != "success" or not output_path:
                            return {
                                "status": "failed",
                                "tool_trace": traces,
                                "artifacts": artifacts,
                                "validation_reports": validations,
                                "error_message": step.error or f"Tool did not return output path: {action.tool_id}",
                            }
                    except Exception as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "success": False,
                            "output": {},
                            "error_message": error_message,
                            "execution_contract": "AudioEditPlan",
                        })
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": error_message,
                        }
                else:
                    try:
                        manual = load_manual(action.tool_id)
                        call = planner.create_tool_calls([action], {action.tool_id: manual}, dict(state), artifact_paths)[0]
                        call.arguments.update(_normalize_tool_arguments(action, call.arguments, artifact_paths, state["base_audio_path"], output_dir))
                        call.arguments = _prepare_derived_reference_arguments(call.arguments)
                        segmented_result = None
                        if action.tool_id == "edit_data_pipeline":
                            segmented_result = await maybe_execute_segmented_edit_pipeline(
                                tool_executor=tool_executor,
                                arguments=call.arguments,
                                state=dict(state),
                                output_dir=output_dir,
                            )
                        result = segmented_result or await tool_executor.execute(action.tool_id, call.arguments)
                    except Exception as exc:
                        error_message = f"{type(exc).__name__}: {exc}"
                        traces.append({
                            "step_id": action.step_id,
                            "tool_id": action.tool_id,
                            "mcp_tool_name": None,
                            "arguments": {},
                            "success": False,
                            "output": {},
                            "error_message": error_message,
                        })
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": error_message,
                        }
                    traces.append({
                        "step_id": action.step_id,
                        "tool_id": action.tool_id,
                        "mcp_tool_name": call.mcp_tool_name,
                        "arguments": call.arguments,
                        "success": result.success,
                        "output": result.output,
                        "error_message": result.error_message,
                    })
                    if not result.success:
                        repaired_arguments = planner.repair_tool_call(
                            action,
                            call.arguments,
                            dict(state),
                            artifact_paths,
                            result.error_message or "",
                        )
                        if repaired_arguments is not None:
                            repaired_arguments.update(
                                _normalize_tool_arguments(
                                    action,
                                    repaired_arguments,
                                    artifact_paths,
                                    state["base_audio_path"],
                                    output_dir,
                                )
                            )
                            repaired_arguments = _prepare_repair_splice_arguments(repaired_arguments)
                            repaired_result = await tool_executor.execute(action.tool_id, repaired_arguments)
                            if repaired_result.success:
                                try:
                                    repaired_result = _finalize_repair_splice(repaired_result, repaired_arguments)
                                except Exception as exc:
                                    repaired_result = repaired_result.model_copy(
                                        update={
                                            "success": False,
                                            "error_message": f"Repair splice failed: {exc}",
                                        }
                                    )
                            traces.append({
                                "phase": "repair_attempt",
                                "step_id": action.step_id,
                                "tool_id": action.tool_id,
                                "mcp_tool_name": call.mcp_tool_name,
                                "repair_context": {
                                    "failed_error_message": result.error_message,
                                    "failed_arguments": call.arguments,
                                },
                                "arguments": repaired_arguments,
                                "success": repaired_result.success,
                                "output": repaired_result.output,
                                "error_message": repaired_result.error_message,
                            })
                            if repaired_result.success:
                                result = repaired_result
                                output_path = _extract_output_path(result.output)
                            else:
                                return {
                                    "status": "failed",
                                    "tool_trace": traces,
                                    "artifacts": artifacts,
                                    "validation_reports": validations,
                                    "error_message": repaired_result.error_message or result.error_message or f"Tool failed: {action.tool_id}",
                                }
                        else:
                            return {
                                "status": "failed",
                                "tool_trace": traces,
                                "artifacts": artifacts,
                                "validation_reports": validations,
                                "error_message": result.error_message or f"Tool failed: {action.tool_id}",
                            }
                    else:
                        output_path = _extract_output_path(result.output)

                    if not output_path:
                        return {
                            "status": "failed",
                            "tool_trace": traces,
                            "artifacts": artifacts,
                            "validation_reports": validations,
                            "error_message": result.error_message or f"Tool did not return output path: {action.tool_id}",
                        }

                artifact_id = action.output_artifact_role
                if output_path:
                    artifact_paths[artifact_id] = output_path
                    artifacts.append({
                        "artifact_id": artifact_id,
                        "path": output_path,
                        "role": action.output_artifact_role,
                        "parent_id": action.input_artifacts[0] if action.input_artifacts else None,
                        "description": action.objective,
                    })
                validation = _validate_output(action, output_path)
                validations.append(validation.model_dump())
                if not validation.passed:
                    return {
                        "status": "failed",
                        "tool_trace": traces,
                        "artifacts": artifacts,
                        "validation_reports": validations,
                        "error_message": validation.required_revision or f"Validation failed: {action.step_id}",
                    }

            final_path = (
                artifact_paths.get("spatial_reverb_audio")
                or artifact_paths.get("final_audio")
                or artifact_paths.get("foreground_background_remix_audio")
                or artifact_paths.get("intra_audio_voice_transfer_audio")
                or artifact_paths.get("accent_converted_audio")
                or artifact_paths.get("separated_speech_audio")
                or artifact_paths.get("edit_pipeline_audio")
                or artifact_paths.get("translated_speech_audio")
                or artifact_paths.get("music_vocal_audio")
                or artifact_paths.get("paralinguistic_event_added_audio")
                or artifact_paths.get("paralinguistic_event_removed_audio")
                or artifact_paths.get("reconstructed_audio")
                or artifact_paths.get("speaker_scoped_edit_audio")
                or artifact_paths.get("extracted_audio")
                or artifact_paths.get("removed_audio")
                or artifact_paths.get("muted_audio")
                or artifact_paths.get("concatenated_audio")
                or artifact_paths.get("artifact_audio")
                or artifact_paths.get("converted_audio")
                or artifact_paths.get("synthesized_audio")
                or state["base_audio_path"]
            )
            return {
                "status": "completed",
                "output_audio_path": final_path,
                "tool_trace": traces,
                "artifacts": artifacts,
                "validation_reports": validations,
            }
        finally:
            if close_executor:
                await tool_executor.shutdown()
    return node


async def _try_structured_executor_path(
    state: AudioEditState,
    tool_executor: ToolExecutor,
    output_dir: str,
) -> dict[str, Any] | None:
    """Execute graph actions through AudioEditExecutor when they are adaptable."""

    actions = state.get("plan") or []
    if not actions:
        return None
    try:
        plan = tool_actions_to_audio_edit_plan(actions, state=dict(state))
    except (ToolActionAdapterError, ValueError, TypeError):
        return None
    executor = AudioEditExecutor(
        tool_executor=tool_executor,
        output_dir=output_dir,
        enable_gemini_executor=False,
    )
    context = {
        "input_audio": state["base_audio_path"],
        "output_dir": output_dir,
        "source_text": state.get("source_text"),
        "perception_report": state.get("perception_report") or state.get("audio_perception") or {},
        "reference_audio_paths": state.get("reference_audio_paths") or [],
        **(state.get("runtime_settings") or {}),
    }
    executed = await executor.aexecute(
        plan,
        context=context,
    )
    evaluation = _evaluate_structured_execution(executed, context)
    summary = execution_summary(executed)
    traces = []
    artifacts = []
    validations = []
    for step in executed.steps:
        output_path = _extract_output_path(step.result or {}) if step.result else None
        traces.append(
            {
                "step_id": step.id,
                "tool_id": step.operation,
                "success": step.status == "success",
                "output": step.result or {},
                "error_message": step.error,
                "execution_contract": "AudioEditPlan",
            }
        )
        if output_path:
            artifacts.append(
                {
                    "artifact_id": step.expected_output or step.id,
                    "path": output_path,
                    "role": step.expected_output or step.operation,
                    "parent_id": None,
                    "description": step.description,
                }
            )
        validations.append(
            ValidationReport(
                step_id=step.id,
                passed=step.status == "success",
                score=1.0 if step.status == "success" else 0.0,
                findings=[step.error or "structured executor step completed"],
                required_revision=None if step.status == "success" else step.error,
            ).model_dump()
        )
    final_path = executed.metadata.get("final_output_path")
    audio_artifacts = executed.metadata.get("audio_artifacts") or artifacts
    intermediate_audio_artifacts = executed.metadata.get("intermediate_audio_artifacts") or []
    failed = [step for step in executed.steps if step.status == "failed"]
    result = {
        "status": "completed" if summary["completed"] else "failed",
        "output_audio_path": final_path,
        "tool_trace": traces,
        "artifacts": audio_artifacts,
        "audio_artifacts": audio_artifacts,
        "debug_audio_artifacts": executed.metadata.get("debug_audio_artifacts") or intermediate_audio_artifacts,
        "intermediate_audio_artifacts": intermediate_audio_artifacts,
        "intermediate_audios": executed.metadata.get("intermediate_audios") or intermediate_audio_artifacts,
        "validation_reports": validations,
        "evaluation": evaluation,
        "evaluation_history": [{"attempt": 0, "evaluation": evaluation}] if evaluation else [],
        "runtime_verification_revisions": 0,
        "runtime_verification_repair_history": [],
        "runtime_verification_best_effort": executed.metadata.get("runtime_verification_best_effort"),
        "error_message": failed[0].error if failed else None,
        "execution_summary": summary,
        "execution_contract": "AudioEditPlan",
    }
    repair = await _try_evaluator_repair(
        state=state,
        tool_executor=tool_executor,
        context=context,
        evaluation=evaluation,
        execution_summary=summary,
        final_output_path=final_path,
        output_dir=output_dir,
    )
    if repair:
        result["evaluator_repair"] = repair
        if repair.get("accepted") and isinstance(repair.get("result"), dict):
            repaired = _structured_result_from_runner(repair["result"], output_dir=output_dir)
            repaired["evaluator_repair"] = repair
            repaired["evaluation_history"] = [
                {"attempt": 0, "evaluation": evaluation},
                *repair["result"].get("evaluation_history", []),
            ]
            repaired["evaluation_revisions"] = 1 + int(repair["result"].get("evaluation_revisions") or 0)
            return repaired
    return result


def _evaluate_structured_execution(executed: Any, context: dict[str, Any]) -> dict[str, Any] | None:
    provider = str(context.get("audio_evaluator") or "").strip().lower()
    if not provider or provider in {"stub", "none", "off", "disabled"}:
        return None
    try:
        from x_editor.runtime.evaluator import build_default_evaluator

        return build_default_evaluator(context).evaluate(executed, context=context).model_dump()
    except Exception as exc:
        return {
            "passed": False,
            "score": None,
            "feedback": f"Structured evaluator failed: {type(exc).__name__}: {exc}",
            "need_replan": False,
            "failed_steps": [],
            "metadata": {
                "verification_error": f"{type(exc).__name__}: {exc}",
                "verification_unavailable": True,
            },
        }


async def _try_evaluator_repair(
    *,
    state: AudioEditState,
    tool_executor: ToolExecutor,
    context: dict[str, Any],
    evaluation: dict[str, Any] | None,
    execution_summary: dict[str, Any],
    final_output_path: str | None,
    output_dir: str,
) -> dict[str, Any] | None:
    if not evaluation or not evaluation.get("need_replan"):
        return None
    if not _context_bool(context, "enable_evaluator_repair", default=False):
        return None
    max_revisions = int(context.get("max_evaluation_revisions") or 0)
    if max_revisions <= 0:
        return None
    try:
        from x_editor.planner.gemini_planner import GeminiPlanner
        from x_editor.runner import async_run_audio_edit

        repair_dir = Path(output_dir) / "evaluator_repair"
        repair_dir.mkdir(parents=True, exist_ok=True)
        executor = AudioEditExecutor(
            tool_executor=tool_executor,
            output_dir=str(repair_dir),
            enable_gemini_executor=bool(context.get("use_gemini_executor", False)),
        )
        available_operations = executor.registry.describe_operations()
        planner = GeminiPlanner(available_operations=available_operations)
        repair_context = dict(context)
        repair_context.update(
            {
                "output_dir": str(repair_dir),
                "output_audio": str(repair_dir / "final.wav"),
                "available_operations": available_operations,
                "previous_failure": {
                    "stage": "evaluation",
                    "evaluation": evaluation,
                    "execution_summary": execution_summary,
                    "final_output_path": final_output_path,
                },
                "retry_guidance": evaluation.get("feedback"),
                "max_evaluation_revisions": max(0, max_revisions - 1),
                "enable_evaluator_repair": max_revisions > 1,
            }
        )
        candidate = await async_run_audio_edit(
            user_request=str(state.get("user_prompt") or ""),
            input_audio=state["base_audio_path"],
            output_audio=str(repair_dir / "final.wav"),
            context=repair_context,
            planner=planner,
            executor=executor,
        )
        accepted = _repair_candidate_accepted(evaluation, candidate)
        return {
            "attempt": 1,
            "accepted": accepted,
            "quality_before": _evaluation_quality(evaluation),
            "quality_after": _evaluation_quality(candidate.get("evaluation")),
            "final_output_path_before": final_output_path,
            "candidate_final_output_path": candidate.get("final_output_path"),
            "result": candidate,
        }
    except Exception as exc:
        return {
            "attempt": 1,
            "accepted": False,
            "quality_before": _evaluation_quality(evaluation),
            "quality_after": None,
            "error_message": f"{type(exc).__name__}: {exc}",
        }


def _repair_candidate_accepted(original_evaluation: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
    final_path = candidate.get("final_output_path")
    if not isinstance(final_path, str) or not Path(final_path).exists():
        return False
    summary = candidate.get("execution_summary") if isinstance(candidate.get("execution_summary"), dict) else {}
    if summary and not summary.get("completed"):
        return False
    before = _evaluation_quality(original_evaluation)
    after = _evaluation_quality(candidate.get("evaluation"))
    return after > before + 1e-6


def _evaluation_quality(evaluation: Any) -> float:
    if not isinstance(evaluation, dict):
        return -1.0
    metadata = evaluation.get("metadata") if isinstance(evaluation.get("metadata"), dict) else {}
    if metadata.get("verification_unavailable"):
        return -1.0
    score = evaluation.get("score")
    if isinstance(score, (int, float)):
        return float(score) + (0.25 if evaluation.get("passed") else 0.0)
    return 1.0 if evaluation.get("passed") else 0.0


def _structured_result_from_runner(result: dict[str, Any], *, output_dir: str) -> dict[str, Any]:
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    tool_trace = _tool_trace_from_plan_payload(plan)
    final_output_path = result.get("final_output_path")
    summary = result.get("execution_summary") if isinstance(result.get("execution_summary"), dict) else {}
    audio_artifacts = result.get("audio_artifacts") if isinstance(result.get("audio_artifacts"), list) else []
    intermediate = (
        result.get("intermediate_audio_artifacts")
        if isinstance(result.get("intermediate_audio_artifacts"), list)
        else []
    )
    error_message = _first_failed_step_error(plan)
    return {
        "status": "completed" if summary.get("completed") and final_output_path else "failed",
        "output_audio_path": final_output_path,
        "tool_trace": tool_trace,
        "artifacts": audio_artifacts,
        "audio_artifacts": audio_artifacts,
        "debug_audio_artifacts": result.get("debug_audio_artifacts") or intermediate,
        "intermediate_audio_artifacts": intermediate,
        "intermediate_audios": result.get("intermediate_audios") or intermediate,
        "validation_reports": _validation_reports_from_plan_payload(plan),
        "evaluation": result.get("evaluation"),
        "runtime_verification_revisions": result.get("runtime_verification_revisions"),
        "runtime_verification_repair_history": result.get("runtime_verification_repair_history"),
        "runtime_verification_best_effort": result.get("runtime_verification_best_effort"),
        "error_message": error_message,
        "execution_summary": summary,
        "execution_contract": "AudioEditPlan",
        "runner_output_dir": str(output_dir),
    }


def _tool_trace_from_plan_payload(plan: dict[str, Any]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        traces.append(
            {
                "step_id": step.get("id"),
                "tool_id": step.get("operation"),
                "success": step.get("status") == "success",
                "output": step.get("result") if isinstance(step.get("result"), dict) else {},
                "error_message": step.get("error"),
                "execution_contract": "AudioEditPlan",
            }
        )
    return traces


def _validation_reports_from_plan_payload(plan: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        reports.append(
            ValidationReport(
                step_id=str(step.get("id") or ""),
                passed=step.get("status") == "success",
                score=1.0 if step.get("status") == "success" else 0.0,
                findings=[step.get("error") or "structured executor step completed"],
                required_revision=None if step.get("status") == "success" else step.get("error"),
            ).model_dump()
        )
    return reports


def _first_failed_step_error(plan: dict[str, Any]) -> str | None:
    for step in plan.get("steps") or []:
        if isinstance(step, dict) and step.get("status") == "failed" and step.get("error"):
            return str(step["error"])
    return None


def _context_bool(context: dict[str, Any], key: str, *, default: bool = False) -> bool:
    if key not in context:
        return default
    value = context.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prepare_repair_splice_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    splice = arguments.get("_x_editor_repair_splice")
    if not isinstance(splice, dict):
        return arguments

    full_audio = str(splice.get("full_input_audio_path") or arguments.get("input_audio_path") or "")
    final_output = str(splice.get("final_output_path") or arguments.get("output_path") or "")
    source_len = int(splice.get("source_text_length") or 0)
    char_start = int(splice.get("source_char_start") or 0)
    char_end = int(splice.get("source_char_end") or source_len)
    if not full_audio or not final_output or source_len <= 0:
        return arguments

    duration = _audio_duration_sec(full_audio)
    pad_sec = 0.35
    start_sec = max(0.0, duration * char_start / source_len - pad_sec)
    end_sec = min(duration, duration * char_end / source_len + pad_sec)
    if end_sec <= start_sec:
        return arguments

    final_path = Path(final_output)
    slice_path = final_path.with_name(f"{final_path.stem}_input_slice.wav")
    edited_slice_path = final_path.with_name(f"{final_path.stem}_edited_slice.wav")
    _slice_audio(full_audio, str(slice_path), start_sec, end_sec)

    prepared = dict(arguments)
    prepared["input_audio_path"] = str(slice_path)
    prepared["output_path"] = str(edited_slice_path)
    prepared["_x_editor_repair_splice"] = {
        **splice,
        "full_input_audio_path": full_audio,
        "final_output_path": final_output,
        "slice_audio_path": str(slice_path),
        "edited_slice_audio_path": str(edited_slice_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "full_duration_sec": duration,
    }
    return prepared


def _prepare_derived_reference_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    reference = arguments.get("_x_editor_derived_reference")
    if not isinstance(reference, dict):
        return arguments

    mixture = arguments.get("mixture_audio_path") or arguments.get("input_audio_path")
    reference_path = arguments.get("reference_audio_path")
    start_sec = reference.get("start_sec")
    end_sec = reference.get("end_sec")
    if not mixture or not reference_path or start_sec is None or end_sec is None:
        return arguments

    enroll_duration = float(arguments.get("enroll_duration") or 5.0)
    start = max(0.0, float(start_sec))
    end = min(float(end_sec), start + enroll_duration)
    if end <= start:
        return arguments

    _slice_audio(str(mixture), str(reference_path), start, end)
    prepared = dict(arguments)
    prepared["_x_editor_derived_reference"] = {
        **reference,
        "reference_audio_path": str(reference_path),
        "reference_start_sec": start,
        "reference_end_sec": end,
    }
    return prepared


def _finalize_repair_splice(result, arguments: dict[str, Any]):
    splice = arguments.get("_x_editor_repair_splice")
    if not result.success or not isinstance(splice, dict):
        return result

    full_audio = str(splice["full_input_audio_path"])
    edited_slice = _extract_output_path(result.output) or str(splice["edited_slice_audio_path"])
    final_output = str(splice["final_output_path"])
    start_sec = float(splice["start_sec"])
    end_sec = float(splice["end_sec"])
    duration = float(splice["full_duration_sec"])
    _splice_audio(full_audio, edited_slice, final_output, start_sec, end_sec, duration)

    output = dict(result.output)
    output["output_audio_path"] = final_output
    output["repair_splice"] = splice
    return result.model_copy(update={"output": output})


def _audio_duration_sec(audio_path: str) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _slice_audio(input_audio: str, output_audio: str, start_sec: float, end_sec: float) -> None:
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            input_audio,
            "-t",
            f"{end_sec - start_sec:.3f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio,
        ],
        check=True,
    )


def _splice_audio(
    full_audio: str,
    edited_slice: str,
    output_audio: str,
    start_sec: float,
    end_sec: float,
    duration: float,
) -> None:
    Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
    if start_sec <= 0.001 and end_sec >= duration - 0.001:
        Path(output_audio).write_bytes(Path(edited_slice).read_bytes())
        return

    filters = []
    labels = []
    if start_sec > 0.001:
        filters.append(f"[0:a]atrim=start=0:end={start_sec:.3f},asetpts=PTS-STARTPTS[a0]")
        labels.append("[a0]")
    filters.append("[1:a]asetpts=PTS-STARTPTS[a1]")
    labels.append("[a1]")
    if end_sec < duration - 0.001:
        filters.append(f"[0:a]atrim=start={end_sec:.3f},asetpts=PTS-STARTPTS[a2]")
        labels.append("[a2]")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            full_audio,
            "-i",
            edited_slice,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            output_audio,
        ],
        check=True,
    )


def _execute_artifact_ops(
    action: ToolAction,
    artifact_paths: dict[str, str],
    state: dict[str, Any],
    base_audio_path: str,
    output_dir: str,
) -> dict[str, Any]:
    return execute_artifact_operation(
        action=action,
        artifact_paths=artifact_paths,
        state=state,
        base_audio_path=base_audio_path,
        output_dir=output_dir,
    )


def _absolute_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _extract_output_path(output: dict[str, Any]) -> str | None:
    for key in ["output_audio_path", "target_audio_path", "output_path", "generated_audio_path"]:
        value = output.get(key)
        if isinstance(value, str):
            return value
    return None


def _normalize_tool_arguments(
    action: ToolAction,
    arguments: dict[str, Any],
    artifact_paths: dict[str, str],
    base_audio_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Resolve model-generated artifact placeholders into concrete paths."""

    normalized = dict(arguments)
    for key, value in list(normalized.items()):
        if not isinstance(value, str) or not value.startswith("<artifact:"):
            continue
        artifact_id = value.removeprefix("<artifact:").removesuffix(">")
        if key in {"output_path", "output_audio_path", "target_audio_path"}:
            normalized[key] = str(Path(output_dir) / f"{action.step_id}.wav")
        elif artifact_id == "input_audio":
            normalized[key] = base_audio_path
        elif artifact_id in artifact_paths:
            normalized[key] = artifact_paths[artifact_id]

    if action.tool_id in {
        "seamless_expressive_s2st",
        "ace_step_music",
        "qwen3_tts",
        "seed_vc",
        "speech_enhancement",
        "tse",
        "speech_separation",
        "edit_data_pipeline",
        "intra_audio_voice_transfer",
        "accent_conversion",
        "audio_effects",
        "spatial_reverb",
    }:
        output_key = "output_path"
        if not normalized.get(output_key) or str(normalized[output_key]).startswith("<artifact:"):
            normalized[output_key] = str(Path(output_dir) / f"{action.step_id}.wav")

    return normalized


def _validate_output(action: ToolAction, output_path: str | None) -> ValidationReport:
    if not output_path:
        return ValidationReport(
            step_id=action.step_id,
            passed=False,
            findings=["tool did not return an output audio path"],
            required_revision="Tool output must include an audio path",
        )
    if output_path.startswith("<artifact:"):
        return ValidationReport(
            step_id=action.step_id,
            passed=False,
            findings=[f"unresolved artifact reference: {output_path}"],
            required_revision="Resolve previous artifact path before executing this step",
        )
    if not Path(output_path).exists():
        return ValidationReport(
            step_id=action.step_id,
            passed=False,
            findings=[f"output path does not exist: {output_path}"],
            required_revision="Tool must create the declared output artifact",
        )
    return ValidationReport(
        step_id=action.step_id,
        passed=True,
        score=1.0,
        findings=["output artifact exists; perceptual validation not yet enabled"],
    )
