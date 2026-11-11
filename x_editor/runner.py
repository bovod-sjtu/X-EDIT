"""Planner -> Executor -> Evaluator runner for structured audio edits."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from x_editor.core.agentic_mode import resolve_agentic_mode
from x_editor.core.config import runtime_artifact_path
from x_editor.core.schemas import AudioEditPlan
from x_editor.planner import build_audio_edit_planner
from x_editor.runtime.audio_plan_executor import AudioEditExecutor, execution_summary
from x_editor.runtime.edit_session_graph import (
    EditSessionGraph,
    graph_from_context,
    save_edit_session_graph,
)
from x_editor.runtime.evaluator import (
    AudioJudgeEvaluator,
    Evaluator,
    build_default_evaluator,
)
from x_editor.runtime.evidence import (
    build_execution_evidence,
    extract_planner_skill_trace,
    write_evidence_bundle,
)
from x_editor.workflow_control import (
    ArtifactFinalizationAudit,
    ArtifactFinalizationAuditor,
    ArtifactShadowObserver,
    ArtifactStore,
    ArtifactStoreCommit,
    FinalAssemblerShadow,
    FinalDeliveryManifest,
    PlanAdmissionGate,
    PlanAdmissionResult,
    RepairController,
    RepairSignal,
    SuffixResumeAdmissionGate,
    SuffixResumeAuditor,
    SuffixResumeCheckpoint,
    SuffixResumeExecutionAdmission,
    SuffixResumeExecutionRequest,
    WorkflowControlSession,
    WorkflowFailureRegistry,
    assemble_final_output,
    build_workflow_control_session,
    full_replay_admission,
    project_workflow_failure_registry,
)
from x_editor.workflow_control.schemas import (
    ArtifactAuditAttemptOutcome,
    RepairAction,
    RepairDecision,
    RepairOutcome,
    RepairScope,
)


def run_audio_edit(
    user_request: str,
    input_audio: str | None = None,
    output_audio: str | None = None,
    context: dict[str, Any] | None = None,
    *,
    planner: Any | None = None,
    executor: AudioEditExecutor | None = None,
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    """Synchronous convenience wrapper for the full structured edit flow."""

    return asyncio.run(
        async_run_audio_edit(
            user_request=user_request,
            input_audio=input_audio,
            output_audio=output_audio,
            context=context,
            planner=planner,
            executor=executor,
            evaluator=evaluator,
        )
    )


async def async_run_audio_edit(
    user_request: str,
    input_audio: str | None = None,
    output_audio: str | None = None,
    context: dict[str, Any] | None = None,
    *,
    planner: Any | None = None,
    executor: AudioEditExecutor | None = None,
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    """Run user request through planning, execution, evaluation, and optional repair."""

    context = dict(context or {})
    if input_audio is not None:
        context["input_audio"] = input_audio
    if output_audio is not None:
        context["output_audio"] = output_audio
    context.setdefault(
        "agentic_mode", resolve_agentic_mode(context.get("agentic_mode"))
    )

    session_graph: EditSessionGraph | None = None
    session_round_index: int | None = None
    if _edit_session_graph_enabled(context):
        session_graph = graph_from_context(context)
        session_graph.ensure_initial_node(
            input_audio=input_audio or context.get("input_audio"),
            input_audios=_string_list(
                context.get("input_audios") or context.get("input_audio_paths")
            ),
            reference_audio_items=_dict_list(context.get("reference_audio_items")),
            reference_audio_paths=_string_list(context.get("reference_audio_paths")),
        )
        if _context_bool(context, "edit_session_use_current_as_input", default=True):
            current_audio = session_graph.current_audio_path()
            if current_audio:
                if input_audio and input_audio != current_audio:
                    context.setdefault("original_request_input_audio", input_audio)
                input_audio = current_audio
                context["input_audio"] = current_audio
                context["input_audios"] = _session_input_audios(session_graph, context)
        session_round_index = session_graph.start_round(
            user_request,
            metadata={
                "requested_input_audio": context.get("original_request_input_audio")
                or input_audio,
                "output_audio": output_audio or context.get("output_audio"),
            },
        )
        context["edit_session_id"] = session_graph.session_id
        context["edit_session_graph"] = session_graph.to_dict()
        context["edit_session_graph_context"] = session_graph.planner_context()
        context["edit_session_audio_aliases"] = session_graph.audio_aliases()

    executor = executor or AudioEditExecutor(agentic_mode=context.get("agentic_mode"))
    available_operations = executor.registry.describe_operations()
    context["available_operations"] = available_operations
    planner = planner or build_audio_edit_planner(
        agentic_mode=context.get("agentic_mode"),
        available_operations=available_operations,
    )
    evaluator = evaluator or build_default_evaluator(context)
    workflow_control = build_workflow_control_session(
        context,
        output_dir=getattr(executor, "output_dir", None) or context.get("output_dir"),
    )
    repair_controller = RepairController()
    repair_controller_history: list[dict[str, Any]] = []
    artifact_finalization_auditor = ArtifactFinalizationAuditor()
    artifact_finalization_history: list[dict[str, Any]] = []
    selected_artifact_audit: ArtifactFinalizationAudit | None = None
    workflow_control.emit(
        stage="runner",
        kind="run_started",
        status="started",
        payload={
            "agentic_mode": context.get("agentic_mode"),
            "input_audio": input_audio,
            "output_audio": output_audio,
            "available_operation_count": len(available_operations),
        },
    )

    plan: AudioEditPlan = planner.plan(user_request, context=context)
    if input_audio is not None:
        plan.input_audio = input_audio
    if output_audio is not None:
        plan.output_audio = output_audio
    workflow_control.emit(
        stage="planner",
        kind="plan_created",
        status="completed",
        subject_refs=[step.id for step in plan.steps],
        payload={"plan": plan.model_dump(mode="json")},
    )
    plan_admission_history: list[dict[str, Any]] = []
    plan_admission_result = _audit_plan_admission(
        workflow_control=workflow_control,
        plan=plan,
        available_operations=available_operations,
        context=context,
        planner_stage="initial",
        attempt_id="plan:0",
    )
    if plan_admission_result is not None:
        plan_admission_history.append(
            {
                "planner_stage": "initial",
                "attempt_id": "plan:0",
                "admission": plan_admission_result.model_dump(mode="json"),
            }
        )

    suffix_resume_execution_admission = _prepare_suffix_resume_execution(
        context=context,
        workflow_control=workflow_control,
        candidate_plan=plan,
    )
    if suffix_resume_execution_admission is not None:
        if suffix_resume_execution_admission.status != "admitted":
            _emit_suffix_resume_execution_decision(
                workflow_control,
                suffix_resume_execution_admission,
            )
        _reset_candidate_execution_state(plan)

    executed_plan = await executor.aexecute(plan, context=context)
    if (
        suffix_resume_execution_admission is not None
        and suffix_resume_execution_admission.status == "admitted"
        and executed_plan.metadata.get("suffix_resume_reuse_error")
    ):
        suffix_resume_execution_admission = full_replay_admission(
            plan,
            reason_code="executor_reuse_binding_failed",
            plain_summary=(
                "Executor could not bind the revalidated artifacts and replayed "
                "the complete candidate plan."
            ),
            checkpoint_id=suffix_resume_execution_admission.checkpoint_id,
        )
        _emit_suffix_resume_execution_decision(
            workflow_control,
            suffix_resume_execution_admission,
            executor_error=executed_plan.metadata.get("suffix_resume_reuse_error"),
        )
    elif (
        suffix_resume_execution_admission is not None
        and suffix_resume_execution_admission.status == "admitted"
    ):
        _emit_suffix_resume_execution_decision(
            workflow_control,
            suffix_resume_execution_admission,
        )
        _emit_suffix_resume_reuse_completed(
            workflow_control,
            suffix_resume_execution_admission,
            executed_plan,
        )
    repair_controller_history.extend(
        _observe_step_local_repairs(
            workflow_control=workflow_control,
            controller=repair_controller,
            plan=executed_plan,
            attempt_prefix="execution:initial",
        )
    )
    initial_artifact_audit = _audit_artifact_finalization(
        workflow_control=workflow_control,
        auditor=artifact_finalization_auditor,
        plan=executed_plan,
        attempt_id="execution:initial",
        attempt_outcome="selected",
    )
    if initial_artifact_audit is not None:
        selected_artifact_audit = initial_artifact_audit
        artifact_finalization_history.append(
            {
                "attempt_id": "execution:initial",
                "attempt_outcome": "selected",
                "audit": initial_artifact_audit.model_dump(mode="json"),
            }
        )
    workflow_control.emit(
        stage="executor",
        kind="initial_execution_completed",
        status="completed",
        subject_refs=[step.id for step in executed_plan.steps],
        payload={
            "execution_summary": execution_summary(executed_plan),
            "final_output_path": executed_plan.metadata.get("final_output_path")
            or _last_output(executed_plan),
        },
    )
    tool_contract_repair_history: list[dict[str, Any]] = []
    tool_contract_revision = 0
    max_tool_contract_revisions = _max_tool_contract_revisions(context)
    best_tool_contract_plan = executed_plan
    best_tool_contract_quality = _tool_contract_quality(executed_plan)
    while (
        tool_contract_revision < max_tool_contract_revisions
        and _needs_tool_contract_replan(best_tool_contract_plan)
    ):
        tool_contract_revision += 1
        repair_context = dict(context)
        violations = _tool_contract_violations(best_tool_contract_plan)
        repair_context["previous_failure"] = {
            "stage": "tool_contract",
            "attempt": tool_contract_revision - 1,
            "execution_summary": execution_summary(best_tool_contract_plan),
            "final_output_path": best_tool_contract_plan.metadata.get(
                "final_output_path"
            )
            or _last_output(best_tool_contract_plan),
            "tool_contract_violations": violations,
            "operation_normalization_errors": best_tool_contract_plan.metadata.get(
                "operation_normalization_errors"
            )
            or {},
            "failed_tool_trace": best_tool_contract_plan.metadata.get(
                "tool_invocations"
            )
            or [],
        }
        repair_context["retry_guidance"] = _tool_contract_retry_guidance(
            best_tool_contract_plan
        )
        repair_context["failed_tool_trace"] = (
            best_tool_contract_plan.metadata.get("tool_invocations") or []
        )
        repair_attempt_id = f"tool_contract:{tool_contract_revision}"
        repair_budget = {"planner_calls": 1.0, "executor_runs": 1.0}
        repair_signal: RepairSignal | None = None
        repair_decision: RepairDecision | None = None
        if workflow_control.enabled:
            repair_signal = repair_controller.signal_from_tool_contract(violations)
            repair_decision = _observe_controller_recommendation(
                workflow_control=workflow_control,
                controller=repair_controller,
                signal=repair_signal,
                legacy_action="replan_global",
                budget_delta=repair_budget,
                attempt_id=repair_attempt_id,
            )
        candidate_plan: AudioEditPlan = planner.plan(
            user_request, context=repair_context
        )
        if input_audio is not None:
            candidate_plan.input_audio = input_audio
        if output_audio is not None:
            candidate_plan.output_audio = output_audio
        candidate_admission = _audit_plan_admission(
            workflow_control=workflow_control,
            plan=candidate_plan,
            available_operations=available_operations,
            context=repair_context,
            planner_stage="tool_contract_repair",
            attempt_id=repair_attempt_id,
        )
        if candidate_admission is not None:
            plan_admission_history.append(
                {
                    "planner_stage": "tool_contract_repair",
                    "attempt_id": repair_attempt_id,
                    "admission": candidate_admission.model_dump(mode="json"),
                }
            )
        candidate_executed = await executor.aexecute(
            candidate_plan, context=repair_context
        )
        repair_controller_history.extend(
            _observe_step_local_repairs(
                workflow_control=workflow_control,
                controller=repair_controller,
                plan=candidate_executed,
                attempt_prefix=f"{repair_attempt_id}:execution",
            )
        )
        candidate_quality = _tool_contract_quality(candidate_executed)
        accepted = (
            not _needs_tool_contract_replan(candidate_executed)
            or candidate_quality > best_tool_contract_quality + 1e-6
        )
        candidate_artifact_audit = _audit_artifact_finalization(
            workflow_control=workflow_control,
            auditor=artifact_finalization_auditor,
            plan=candidate_executed,
            attempt_id=f"{repair_attempt_id}:execution",
            attempt_outcome="selected" if accepted else "rejected",
        )
        if candidate_artifact_audit is not None:
            if accepted:
                selected_artifact_audit = candidate_artifact_audit
            artifact_finalization_history.append(
                {
                    "attempt_id": f"{repair_attempt_id}:execution",
                    "attempt_outcome": "selected" if accepted else "rejected",
                    "audit": candidate_artifact_audit.model_dump(mode="json"),
                }
            )
        tool_contract_repair_history.append(
            {
                "attempt": tool_contract_revision,
                "accepted": accepted,
                "quality_before": best_tool_contract_quality,
                "quality_after": candidate_quality,
                "previous_violations": violations,
                "retry_guidance": repair_context["retry_guidance"],
                "candidate_execution_summary": execution_summary(candidate_executed),
                "candidate_tool_contract_violations": _tool_contract_violations(
                    candidate_executed
                ),
                "candidate_final_output_path": candidate_executed.metadata.get(
                    "final_output_path"
                )
                or _last_output(candidate_executed),
            }
        )
        workflow_control.emit(
            stage="repair",
            kind="legacy_repair_attempt_completed",
            status="accepted" if accepted else "rejected",
            attempt_id=repair_attempt_id,
            payload={
                "repair_stage": "tool_contract",
                **tool_contract_repair_history[-1],
            },
        )
        if repair_signal is not None and repair_decision is not None:
            repair_controller_history.append(
                _record_controller_comparison(
                    workflow_control=workflow_control,
                    controller=repair_controller,
                    signal=repair_signal,
                    decision=repair_decision,
                    legacy_action="replan_global",
                    legacy_scope="plan",
                    outcome="accepted" if accepted else "rejected",
                    budget_delta=repair_budget,
                    attempt_id=repair_attempt_id,
                    metadata={
                        "quality_before": best_tool_contract_quality,
                        "quality_after": candidate_quality,
                    },
                )
            )
        if accepted:
            best_tool_contract_plan = candidate_executed
            best_tool_contract_quality = candidate_quality
            if not _needs_tool_contract_replan(best_tool_contract_plan):
                break
        else:
            break

    executed_plan = best_tool_contract_plan
    if _needs_tool_contract_replan(executed_plan):
        executed_plan.metadata["tool_contract_best_effort"] = {
            "returned_best_available_version": True,
            "quality": best_tool_contract_quality,
            "violations": _tool_contract_violations(executed_plan),
            "message": (
                "Tool contract verification still reports invalid tool calls after repair. "
                "Returning the best available execution with the blocking call evidence."
            ),
        }

    evaluation = evaluator.evaluate(executed_plan, context=context)
    workflow_control.emit(
        stage="evaluator",
        kind="initial_evaluation_completed",
        status="completed",
        subject_refs=list(evaluation.failed_steps),
        payload={"evaluation": evaluation.model_dump(mode="json")},
    )
    evaluation_history = [
        _attempt_record(
            attempt=0,
            plan=executed_plan,
            evaluation=evaluation,
        )
    ]
    runtime_repair_history: list[dict[str, Any]] = []
    runtime_revision = 0
    max_runtime_revisions = _max_runtime_verification_revisions(context)
    best_plan = executed_plan
    best_evaluation = evaluation
    best_quality = _runtime_verification_quality(executed_plan)
    while (
        runtime_revision < max_runtime_revisions
        and _needs_runtime_verification_repair(best_plan)
    ):
        runtime_revision += 1
        repair_context = dict(context)
        defects = _runtime_verification_defects(best_plan)
        repair_context["previous_failure"] = {
            "stage": "runtime_verification",
            "attempt": runtime_revision - 1,
            "execution_summary": execution_summary(best_plan),
            "final_output_path": best_plan.metadata.get("final_output_path")
            or _last_output(best_plan),
            "verification_reports": best_plan.metadata.get("verification_reports")
            or [],
            "verification_repair_plans": best_plan.metadata.get(
                "verification_repair_plans"
            )
            or [],
            "bounded_repair_attempts": best_plan.metadata.get("bounded_repair_attempts")
            or [],
            "defects": defects,
        }
        repair_context["retry_guidance"] = _runtime_verification_retry_guidance(
            best_plan
        )
        repair_context["runtime_verification_repair"] = {
            "prefer_major_actions": True,
            "return_best_effort_if_not_improved": True,
        }
        repair_attempt_id = f"runtime_verification:{runtime_revision}"
        repair_budget = {
            "planner_calls": 1.0,
            "executor_runs": 1.0,
            "evaluator_calls": 1.0,
        }
        repair_signal = None
        repair_decision = None
        if workflow_control.enabled:
            repair_signal = repair_controller.signal_from_runtime_verification(
                defects,
                best_plan.metadata.get("verification_repair_plans") or [],
            )
            repair_decision = _observe_controller_recommendation(
                workflow_control=workflow_control,
                controller=repair_controller,
                signal=repair_signal,
                legacy_action="replan_global",
                budget_delta=repair_budget,
                attempt_id=repair_attempt_id,
            )
        candidate_plan: AudioEditPlan = planner.plan(
            user_request, context=repair_context
        )
        if input_audio is not None:
            candidate_plan.input_audio = input_audio
        if output_audio is not None:
            candidate_plan.output_audio = output_audio
        candidate_admission = _audit_plan_admission(
            workflow_control=workflow_control,
            plan=candidate_plan,
            available_operations=available_operations,
            context=repair_context,
            planner_stage="runtime_verification_repair",
            attempt_id=repair_attempt_id,
        )
        if candidate_admission is not None:
            plan_admission_history.append(
                {
                    "planner_stage": "runtime_verification_repair",
                    "attempt_id": repair_attempt_id,
                    "admission": candidate_admission.model_dump(mode="json"),
                }
            )
        candidate_executed = await executor.aexecute(
            candidate_plan, context=repair_context
        )
        repair_controller_history.extend(
            _observe_step_local_repairs(
                workflow_control=workflow_control,
                controller=repair_controller,
                plan=candidate_executed,
                attempt_prefix=f"{repair_attempt_id}:execution",
            )
        )
        candidate_evaluation = evaluator.evaluate(
            candidate_executed, context=repair_context
        )
        candidate_quality = _runtime_verification_quality(candidate_executed)
        improved = (
            not _needs_runtime_verification_repair(candidate_executed)
            or candidate_quality > best_quality + 1e-6
        )
        candidate_artifact_audit = _audit_artifact_finalization(
            workflow_control=workflow_control,
            auditor=artifact_finalization_auditor,
            plan=candidate_executed,
            attempt_id=f"{repair_attempt_id}:execution",
            attempt_outcome="selected" if improved else "rejected",
        )
        if candidate_artifact_audit is not None:
            if improved:
                selected_artifact_audit = candidate_artifact_audit
            artifact_finalization_history.append(
                {
                    "attempt_id": f"{repair_attempt_id}:execution",
                    "attempt_outcome": "selected" if improved else "rejected",
                    "audit": candidate_artifact_audit.model_dump(mode="json"),
                }
            )
        runtime_repair_history.append(
            {
                "attempt": runtime_revision,
                "accepted": improved,
                "quality_before": best_quality,
                "quality_after": candidate_quality,
                "previous_defects": defects,
                "candidate_final_output_path": candidate_executed.metadata.get(
                    "final_output_path"
                )
                or _last_output(candidate_executed),
                "candidate_execution_summary": execution_summary(candidate_executed),
                "candidate_verification_reports": candidate_executed.metadata.get(
                    "verification_reports"
                )
                or [],
                "candidate_verification_repair_plans": candidate_executed.metadata.get(
                    "verification_repair_plans"
                )
                or [],
            }
        )
        workflow_control.emit(
            stage="repair",
            kind="legacy_repair_attempt_completed",
            status="accepted" if improved else "rejected",
            attempt_id=repair_attempt_id,
            payload={
                "repair_stage": "runtime_verification",
                **runtime_repair_history[-1],
            },
        )
        if repair_signal is not None and repair_decision is not None:
            repair_controller_history.append(
                _record_controller_comparison(
                    workflow_control=workflow_control,
                    controller=repair_controller,
                    signal=repair_signal,
                    decision=repair_decision,
                    legacy_action="replan_global",
                    legacy_scope="plan",
                    outcome="accepted" if improved else "rejected",
                    budget_delta=repair_budget,
                    attempt_id=repair_attempt_id,
                    metadata={
                        "quality_before": best_quality,
                        "quality_after": candidate_quality,
                    },
                )
            )
        evaluation_history.append(
            _attempt_record(
                attempt=len(evaluation_history),
                plan=candidate_executed,
                evaluation=candidate_evaluation,
            )
        )
        if improved:
            best_plan = candidate_executed
            best_evaluation = candidate_evaluation
            best_quality = candidate_quality
            if not _needs_runtime_verification_repair(best_plan):
                break
        else:
            break

    executed_plan = best_plan
    evaluation = best_evaluation
    if _needs_runtime_verification_repair(executed_plan):
        executed_plan.metadata["runtime_verification_best_effort"] = {
            "returned_best_available_version": True,
            "quality": best_quality,
            "defects": _runtime_verification_defects(executed_plan),
            "message": (
                "Runtime verification still reports defects after bounded repair. "
                "Returning the best available artifact with known limitations."
            ),
        }

    max_revisions = _max_evaluation_revisions(context, evaluator)
    revision = 0
    evaluator_repair_history: list[dict[str, Any]] = []
    best_eval_plan = executed_plan
    best_eval_evaluation = evaluation
    best_eval_quality = _evaluation_repair_quality(executed_plan, evaluation)
    while (
        revision < max_revisions
        and best_eval_evaluation.need_replan
        and not best_eval_evaluation.passed
    ):
        revision += 1
        repair_context = dict(context)
        repair_context["previous_failure"] = {
            "stage": "evaluation",
            "attempt": revision - 1,
            "evaluation": best_eval_evaluation.model_dump(),
            "execution_summary": execution_summary(best_eval_plan),
            "final_output_path": best_eval_plan.metadata.get("final_output_path")
            or _last_output(best_eval_plan),
        }
        repair_context["retry_guidance"] = best_eval_evaluation.feedback
        repair_context["evaluation_feedback"] = best_eval_evaluation.model_dump()
        repair_attempt_id = f"evaluation:{revision}"
        repair_budget = {
            "planner_calls": 1.0,
            "executor_runs": 1.0,
            "evaluator_calls": 1.0,
        }
        repair_signal = None
        repair_decision = None
        if workflow_control.enabled:
            repair_signal = repair_controller.signal_from_evaluation(
                best_eval_evaluation.model_dump(mode="json")
            )
            repair_decision = _observe_controller_recommendation(
                workflow_control=workflow_control,
                controller=repair_controller,
                signal=repair_signal,
                legacy_action="replan_global",
                budget_delta=repair_budget,
                attempt_id=repair_attempt_id,
            )
        repair_plan: AudioEditPlan = planner.plan(user_request, context=repair_context)
        if input_audio is not None:
            repair_plan.input_audio = input_audio
        if output_audio is not None:
            repair_plan.output_audio = output_audio
        candidate_admission = _audit_plan_admission(
            workflow_control=workflow_control,
            plan=repair_plan,
            available_operations=available_operations,
            context=repair_context,
            planner_stage="evaluation_repair",
            attempt_id=repair_attempt_id,
        )
        if candidate_admission is not None:
            plan_admission_history.append(
                {
                    "planner_stage": "evaluation_repair",
                    "attempt_id": repair_attempt_id,
                    "admission": candidate_admission.model_dump(mode="json"),
                }
            )
        candidate_plan = await executor.aexecute(repair_plan, context=repair_context)
        repair_controller_history.extend(
            _observe_step_local_repairs(
                workflow_control=workflow_control,
                controller=repair_controller,
                plan=candidate_plan,
                attempt_prefix=f"{repair_attempt_id}:execution",
            )
        )
        candidate_evaluation = evaluator.evaluate(
            candidate_plan, context=repair_context
        )
        candidate_quality = _evaluation_repair_quality(
            candidate_plan, candidate_evaluation
        )
        accepted = (
            candidate_quality > best_eval_quality + 1e-6 or candidate_evaluation.passed
        )
        candidate_artifact_audit = _audit_artifact_finalization(
            workflow_control=workflow_control,
            auditor=artifact_finalization_auditor,
            plan=candidate_plan,
            attempt_id=f"{repair_attempt_id}:execution",
            attempt_outcome="selected" if accepted else "rejected",
        )
        if candidate_artifact_audit is not None:
            if accepted:
                selected_artifact_audit = candidate_artifact_audit
            artifact_finalization_history.append(
                {
                    "attempt_id": f"{repair_attempt_id}:execution",
                    "attempt_outcome": "selected" if accepted else "rejected",
                    "audit": candidate_artifact_audit.model_dump(mode="json"),
                }
            )
        evaluator_repair_history.append(
            {
                "attempt": revision,
                "accepted": accepted,
                "quality_before": best_eval_quality,
                "quality_after": candidate_quality,
                "candidate_execution_summary": execution_summary(candidate_plan),
                "candidate_final_output_path": candidate_plan.metadata.get(
                    "final_output_path"
                )
                or _last_output(candidate_plan),
                "candidate_evaluation": candidate_evaluation.model_dump(),
            }
        )
        workflow_control.emit(
            stage="repair",
            kind="legacy_repair_attempt_completed",
            status="accepted" if accepted else "rejected",
            attempt_id=repair_attempt_id,
            payload={"repair_stage": "evaluation", **evaluator_repair_history[-1]},
        )
        if repair_signal is not None and repair_decision is not None:
            repair_controller_history.append(
                _record_controller_comparison(
                    workflow_control=workflow_control,
                    controller=repair_controller,
                    signal=repair_signal,
                    decision=repair_decision,
                    legacy_action="replan_global",
                    legacy_scope="plan",
                    outcome="accepted" if accepted else "rejected",
                    budget_delta=repair_budget,
                    attempt_id=repair_attempt_id,
                    metadata={
                        "quality_before": best_eval_quality,
                        "quality_after": candidate_quality,
                    },
                )
            )
        evaluation_history.append(
            _attempt_record(
                attempt=revision,
                plan=candidate_plan,
                evaluation=candidate_evaluation,
            )
        )
        if accepted:
            best_eval_plan = candidate_plan
            best_eval_evaluation = candidate_evaluation
            best_eval_quality = candidate_quality
            if candidate_evaluation.passed:
                break
        else:
            break

    executed_plan = best_eval_plan
    evaluation = best_eval_evaluation
    if evaluator_repair_history and not evaluator_repair_history[-1].get("accepted"):
        executed_plan.metadata["evaluator_repair_best_effort"] = {
            "returned_best_available_version": True,
            "quality": best_eval_quality,
            "message": (
                "Evaluator repair did not improve the completed artifact. "
                "Returning the best available execution instead of replacing it with a worse retry."
            ),
        }

    summary = execution_summary(executed_plan)
    legacy_final_output_path = executed_plan.metadata.get(
        "final_output_path"
    ) or _last_output(
        executed_plan
    )
    final_artifact_audit = selected_artifact_audit
    artifact_store_commit: ArtifactStoreCommit | None = None
    final_delivery_manifest_shadow: FinalDeliveryManifest | None = None
    if workflow_control.artifact_store_mode == "enforced":
        if final_artifact_audit is None:
            raise RuntimeError(
                "enforced artifact finalization requires an artifact audit"
            )
        artifact_store_commit = ArtifactStore().commit(final_artifact_audit)
        final_output_path = assemble_final_output(artifact_store_commit)
        workflow_control.emit(
            stage="finalization",
            kind=(
                "artifact_store_committed"
                if artifact_store_commit.status == "committed"
                else "artifact_store_blocked"
            ),
            status=(
                "accepted"
                if artifact_store_commit.status == "committed"
                else "failed"
            ),
            attempt_id=artifact_store_commit.attempt_id,
            subject_refs=(
                [artifact_store_commit.final_artifact_ref_id]
                if artifact_store_commit.final_artifact_ref_id
                else list(artifact_store_commit.rejected_artifact_refs)
                + list(artifact_store_commit.candidate_artifact_refs)
            ),
            payload={
                "commit": artifact_store_commit.model_dump(mode="json"),
            },
        )
        if workflow_control.final_assembler_mode == "shadow":
            final_delivery_manifest_shadow = FinalAssemblerShadow().project(
                executed_plan,
                final_artifact_audit,
                artifact_store_commit,
            )
            workflow_control.emit(
                stage="finalization",
                kind="final_delivery_manifest_observed",
                status="observed",
                attempt_id=final_delivery_manifest_shadow.attempt_id,
                subject_refs=[
                    ref
                    for ref in [
                        final_delivery_manifest_shadow.committed_final_artifact_ref_id,
                        *[
                            item.artifact_ref_id
                            for item in final_delivery_manifest_shadow.ordered_components
                        ],
                    ]
                    if ref
                ],
                payload={
                    "manifest": final_delivery_manifest_shadow.model_dump(mode="json"),
                    "behavior_changed": False,
                },
            )
    else:
        final_output_path = legacy_final_output_path
    artifact_delivery_passed = (
        artifact_store_commit is None or artifact_store_commit.status == "committed"
    )
    run_completed = bool(summary.get("completed")) and artifact_delivery_passed
    run_status = "completed" if run_completed else "failed"
    audio_debug = _debug_audio_metadata(executed_plan)
    planner_skill_trace = extract_planner_skill_trace(executed_plan)

    execution_perception = _execution_perception(context, executed_plan)
    execution_source_text = (
        context.get("source_text")
        or executed_plan.global_params.get("source_text")
        or execution_perception.get("source_text")
    )

    result = {
        "plan": executed_plan.model_dump(),
        "execution_summary": summary,
        "evaluation": evaluation.model_dump(),
        "evaluation_history": evaluation_history,
        "verification_passed": evaluation.passed,
        "tool_contract_revisions": tool_contract_revision,
        "tool_contract_repair_history": tool_contract_repair_history,
        "tool_contract_best_effort": executed_plan.metadata.get(
            "tool_contract_best_effort"
        ),
        "evaluation_revisions": revision,
        "evaluator_repair_history": evaluator_repair_history,
        "evaluator_repair_best_effort": executed_plan.metadata.get(
            "evaluator_repair_best_effort"
        ),
        "runtime_verification_revisions": runtime_revision,
        "runtime_verification_repair_history": runtime_repair_history,
        "runtime_verification_best_effort": executed_plan.metadata.get(
            "runtime_verification_best_effort"
        ),
        "final_output_path": final_output_path,
        "source_text": execution_source_text,
        "perception": execution_perception,
        "available_operations": available_operations,
        "routing_policy_candidates": planner_skill_trace["routing_policy_candidates"],
        "planner_skill_ids": planner_skill_trace["planner_skill_ids"],
        **audio_debug,
    }
    if suffix_resume_execution_admission is not None:
        result["suffix_resume_execution_admission"] = (
            suffix_resume_execution_admission.model_dump(mode="json")
        )
    if artifact_store_commit is not None:
        result.update(
            {
                "run_status": run_status,
                "artifact_delivery_passed": artifact_delivery_passed,
                "artifact_store_commit": artifact_store_commit.model_dump(
                    mode="json"
                ),
            }
        )
    if final_delivery_manifest_shadow is not None:
        result["final_delivery_manifest_shadow"] = (
            final_delivery_manifest_shadow.model_dump(mode="json")
        )
    run_completed_payload = {
        "execution_summary": summary,
        "evaluation": evaluation.model_dump(mode="json"),
        "final_output_path": final_output_path,
        "tool_contract_revisions": tool_contract_revision,
        "runtime_verification_revisions": runtime_revision,
        "evaluation_revisions": revision,
    }
    if artifact_store_commit is not None:
        run_completed_payload.update(
            {
                "legacy_final_output_path": legacy_final_output_path,
                "artifact_delivery_passed": artifact_delivery_passed,
                "artifact_store_commit_id": artifact_store_commit.commit_id,
            }
        )
    if final_delivery_manifest_shadow is not None:
        run_completed_payload.update(
            {
                "final_delivery_manifest_id": final_delivery_manifest_shadow.manifest_id,
                "final_delivery_manifest_status": final_delivery_manifest_shadow.status,
            }
        )
    workflow_control.emit(
        stage="runner",
        kind="run_completed",
        status=run_status,
        payload=run_completed_payload,
    )
    workflow_failure_registry: WorkflowFailureRegistry | None = None
    suffix_resume_checkpoint: SuffixResumeCheckpoint | None = None
    if artifact_store_commit is not None:
        workflow_failure_registry = project_workflow_failure_registry(
            workflow_control.events()
        )
        registry_payload = workflow_failure_registry.model_dump(mode="json")
        workflow_control.emit(
            stage="session",
            kind="workflow_failure_registry_projected",
            status="completed",
            attempt_id=artifact_store_commit.attempt_id,
            subject_refs=[
                *workflow_failure_registry.terminal_failed_step_ids,
                *workflow_failure_registry.terminal_failed_artifact_refs,
            ],
            payload={"failure_registry": registry_payload},
        )
        result["workflow_failure_registry"] = registry_payload
        if artifact_store_commit.status == "blocked":
            if final_artifact_audit is None:
                raise RuntimeError("suffix resume audit requires artifact finalization")
            suffix_resume_checkpoint = SuffixResumeAuditor().audit(
                executed_plan,
                final_artifact_audit,
                workflow_failure_registry,
            )
            resume_payload = suffix_resume_checkpoint.model_dump(mode="json")
            workflow_control.emit(
                stage="repair",
                kind="suffix_resume_checkpoint_audited",
                status="observed",
                attempt_id=suffix_resume_checkpoint.attempt_id,
                subject_refs=[
                    *suffix_resume_checkpoint.restart_step_ids,
                    *[
                        item.artifact_ref_id
                        for item in suffix_resume_checkpoint.reusable_artifacts
                    ],
                ],
                payload={
                    "checkpoint": resume_payload,
                    "behavior_changed": False,
                },
            )
            result["suffix_resume_checkpoint"] = resume_payload
    if workflow_control.enabled:
        control_summary = workflow_control.summary()
        comparison_count = sum(
            item.get("record_kind") == "comparison"
            for item in repair_controller_history
        )
        enforcement_count = sum(
            item.get("record_kind") == "enforcement"
            for item in repair_controller_history
        )
        control_summary["repair_controller"] = {
            "policy_version": repair_controller.policy_version,
            "shadow_only": enforcement_count == 0,
            "comparison_count": comparison_count,
            "enforcement_count": enforcement_count,
            "history": repair_controller_history,
        }
        if final_artifact_audit is not None:
            control_summary["artifact_finalization"] = (
                final_artifact_audit.model_dump(mode="json")
            )
            control_summary["artifact_finalization_history"] = (
                artifact_finalization_history
            )
        if artifact_store_commit is not None:
            control_summary["artifact_store_commit"] = (
                artifact_store_commit.model_dump(mode="json")
            )
        if final_delivery_manifest_shadow is not None:
            control_summary["final_delivery_manifest_shadow"] = (
                final_delivery_manifest_shadow.model_dump(mode="json")
            )
        if workflow_failure_registry is not None:
            control_summary["failure_registry"] = (
                workflow_failure_registry.model_dump(mode="json")
            )
        if suffix_resume_checkpoint is not None:
            control_summary["suffix_resume_checkpoint"] = (
                suffix_resume_checkpoint.model_dump(mode="json")
            )
        if suffix_resume_execution_admission is not None:
            control_summary["suffix_resume_execution_admission"] = (
                suffix_resume_execution_admission.model_dump(mode="json")
            )
        if plan_admission_result is not None:
            control_summary["plan_admission"] = plan_admission_result.model_dump(
                mode="json"
            )
            control_summary["plan_admission_history"] = plan_admission_history
        result["workflow_control"] = control_summary
    anticipated_evidence_path = None
    if _context_bool(context, "write_evidence", default=True):
        evidence_dir = _evidence_output_dir(
            context, executed_plan, final_output_path, executor
        )
        evidence_filename = str(
            context.get("evidence_filename") or "execution_evidence.json"
        )
        anticipated_evidence_path = str(evidence_dir / evidence_filename)
    if session_graph is not None and session_round_index is not None:
        session_graph.complete_round(
            round_index=session_round_index,
            request=user_request,
            status=run_status,
            plan=executed_plan,
            final_output_path=final_output_path,
            evidence_path=anticipated_evidence_path,
            evaluation=evaluation.model_dump(),
            trace_result=result,
            context=context,
            advance_current_audio=(
                artifact_store_commit is None
                or artifact_store_commit.status == "committed"
            ),
            failure_registry=(
                workflow_failure_registry.model_dump(mode="json")
                if workflow_failure_registry is not None
                else None
            ),
            suffix_resume_checkpoint=(
                suffix_resume_checkpoint.model_dump(mode="json")
                if suffix_resume_checkpoint is not None
                else None
            ),
            suffix_resume_execution_admission=(
                suffix_resume_execution_admission.model_dump(mode="json")
                if suffix_resume_execution_admission is not None
                else None
            ),
            final_delivery_manifest_shadow=(
                final_delivery_manifest_shadow.model_dump(mode="json")
                if final_delivery_manifest_shadow is not None
                else None
            ),
        )
        context["edit_session_graph"] = session_graph.to_dict()
        context["edit_session_graph_context"] = session_graph.planner_context()
        context["edit_session_audio_aliases"] = session_graph.audio_aliases()
        result["edit_session_id"] = session_graph.session_id
        result["edit_session_round_index"] = session_round_index
        result["edit_session_graph"] = session_graph.to_dict()
        graph_path = context.get("edit_session_graph_path")
        if graph_path:
            result["edit_session_graph_path"] = save_edit_session_graph(
                session_graph, str(graph_path)
            )
    evidence = build_execution_evidence(
        request=user_request,
        status=run_status,
        input_audios=[input_audio]
        if input_audio
        else context.get("input_audios") or [],
        output_audio_path=final_output_path,
        source_text=execution_source_text,
        perception=execution_perception,
        planner_name=planner.__class__.__name__,
        executor_name=executor.__class__.__name__,
        plan=executed_plan,
        trace_result=result,
        evaluation=evaluation,
        evaluation_history=evaluation_history,
        available_operations=available_operations,
        context=context,
        error_message=(
            _artifact_store_failure_message(artifact_store_commit)
            or _first_failed_step_error(executed_plan)
        ),
    )
    if _context_bool(context, "write_evidence", default=True):
        evidence_dir = _evidence_output_dir(
            context, executed_plan, final_output_path, executor
        )
        evidence_path = write_evidence_bundle(
            evidence,
            evidence_dir,
            filename=str(context.get("evidence_filename") or "execution_evidence.json"),
        )
        result["evidence_path"] = evidence_path
        if session_graph is not None and context.get("edit_session_graph_path"):
            result["edit_session_graph_path"] = save_edit_session_graph(
                session_graph, str(context["edit_session_graph_path"])
            )
    result["evidence"] = evidence
    return result


def _prepare_suffix_resume_execution(
    *,
    context: dict[str, Any],
    workflow_control: WorkflowControlSession,
    candidate_plan: AudioEditPlan,
) -> SuffixResumeExecutionAdmission | None:
    mode = str(
        context.get("workflow_suffix_resume_mode") or "disabled"
    ).strip().lower()
    if mode == "disabled":
        return None
    if mode != "enforced":
        return full_replay_admission(
            candidate_plan,
            reason_code="suffix_resume_mode_invalid",
            plain_summary=(
                "The suffix-resume mode is invalid, so the complete plan will run."
            ),
        )
    if (
        workflow_control.mode != "enforced"
        or workflow_control.artifact_store_mode != "enforced"
    ):
        return full_replay_admission(
            candidate_plan,
            reason_code="suffix_resume_authority_not_enforced",
            plain_summary=(
                "Suffix reuse requires enforced workflow control and ArtifactStore."
            ),
        )
    for key in (
        "max_tool_contract_revisions",
        "max_runtime_verification_revisions",
        "max_evaluation_revisions",
    ):
        try:
            value = int(context[key])
        except (KeyError, TypeError, ValueError):
            value = -1
        if value != 0:
            return full_replay_admission(
                candidate_plan,
                reason_code="suffix_resume_repair_budget_not_frozen",
                plain_summary=(
                    "Suffix reuse requires all Planner/runtime/Evaluator repair "
                    "revision budgets to be explicitly fixed at zero."
                ),
            )

    raw_request = context.get("workflow_suffix_resume_request")
    if raw_request is None:
        raw_request = {
            "checkpoint": context.get("workflow_suffix_resume_checkpoint"),
            "original_plan": context.get("workflow_suffix_resume_original_plan"),
        }
    try:
        request = SuffixResumeExecutionRequest.model_validate(raw_request)
        original_plan = AudioEditPlan.model_validate(request.original_plan)
        admission = SuffixResumeAdmissionGate().audit(
            request.checkpoint,
            original_plan,
            candidate_plan,
        )
    except (KeyError, TypeError, ValueError):
        return full_replay_admission(
            candidate_plan,
            reason_code="suffix_resume_request_invalid",
            plain_summary=(
                "The supplied checkpoint or original plan is incomplete, so the "
                "complete candidate plan will run."
            ),
        )
    if admission.status == "admitted":
        context["_workflow_suffix_resume_reuse_receipts"] = [
            item.model_dump(mode="json") for item in admission.receipts
        ]
    else:
        context.pop("_workflow_suffix_resume_reuse_receipts", None)
    return admission


def _reset_candidate_execution_state(plan: AudioEditPlan) -> None:
    from x_editor.workflow_control.execution_state import reset_candidate_execution_state
    reset_candidate_execution_state(plan)


def _emit_suffix_resume_execution_decision(
    workflow_control: WorkflowControlSession,
    admission: SuffixResumeExecutionAdmission,
    *,
    executor_error: Any | None = None,
) -> None:
    admitted = admission.status == "admitted"
    payload = {
        "admission": admission.model_dump(mode="json"),
        "behavior_changed": admission.behavior_changed,
    }
    if executor_error:
        payload["executor_error"] = str(executor_error)
    workflow_control.emit(
        stage="repair",
        kind=(
            "suffix_resume_execution_admitted"
            if admitted
            else "suffix_resume_execution_fell_back"
        ),
        status="accepted" if admitted else "observed",
        attempt_id="execution:initial",
        subject_refs=[
            *admission.reused_step_ids,
            *[item.artifact_ref_id for item in admission.receipts],
        ],
        payload=payload,
    )


def _emit_suffix_resume_reuse_completed(
    workflow_control: WorkflowControlSession,
    admission: SuffixResumeExecutionAdmission,
    executed_plan: AudioEditPlan,
) -> None:
    actual_steps = set(
        executed_plan.metadata.get("suffix_resume_reused_step_ids") or []
    )
    for receipt in admission.receipts:
        if receipt.producer_step_id not in actual_steps:
            continue
        workflow_control.emit(
            stage="executor",
            kind="suffix_resume_step_reused",
            status="completed",
            attempt_id="execution:initial",
            subject_refs=[receipt.producer_step_id, receipt.artifact_ref_id],
            payload={
                "receipt": receipt.model_dump(mode="json"),
                "tool_call_skipped": receipt.skipped_tool_call_count,
                "behavior_changed": True,
            },
        )


def _audit_artifact_finalization(
    *,
    workflow_control: WorkflowControlSession,
    auditor: ArtifactFinalizationAuditor,
    plan: AudioEditPlan,
    attempt_id: str,
    attempt_outcome: ArtifactAuditAttemptOutcome,
) -> ArtifactFinalizationAudit | None:
    if not workflow_control.enabled:
        return None

    audit = auditor.audit(
        plan,
        attempt_id=attempt_id,
        attempt_outcome=attempt_outcome,
    )
    shadow_observation = ArtifactShadowObserver().observe(plan, audit)
    for artifact in audit.artifacts:
        subject_refs = [artifact.artifact_ref_id, artifact.producer_step_id]
        workflow_control.emit(
            stage="finalization",
            kind="artifact_candidate_observed",
            status="observed",
            attempt_id=attempt_id,
            subject_refs=subject_refs,
            payload={"artifact": artifact.model_dump(mode="json")},
        )
        workflow_control.emit(
            stage="finalization",
            kind="artifact_disposition_audited",
            status={
                "accepted": "accepted",
                "candidate": "observed",
                "rejected": "rejected",
            }[artifact.disposition],
            attempt_id=attempt_id,
            subject_refs=subject_refs,
            payload={
                "artifact_ref_id": artifact.artifact_ref_id,
                "disposition": artifact.disposition,
                "reasons": artifact.disposition_reasons,
                "behavior_changed": False,
            },
        )
    workflow_control.emit(
        stage="finalization",
        kind="artifact_shadow_observed",
        status="observed",
        attempt_id=attempt_id,
        subject_refs=[item.artifact_ref_id for item in audit.artifacts],
        payload={
            "observation": shadow_observation.model_dump(mode="json"),
            "behavior_changed": False,
        },
    )
    workflow_control.emit(
        stage="finalization",
        kind="artifact_finalization_audited",
        status={
            "would_finalize": "accepted",
            "blocked": "failed",
            "insufficient_evidence": "observed",
        }[audit.status],
        attempt_id=attempt_id,
        subject_refs=[item.artifact_ref_id for item in audit.artifacts],
        payload={"audit": audit.model_dump(mode="json")},
    )
    return audit


def _artifact_store_failure_message(
    commit: ArtifactStoreCommit | None,
) -> str | None:
    if commit is None or commit.status == "committed":
        return None
    codes = ", ".join(commit.failure_codes) or "accepted_final_artifact_missing"
    return f"ArtifactStore blocked final delivery: {codes}."


def _observe_controller_recommendation(
    *,
    workflow_control: WorkflowControlSession,
    controller: RepairController,
    signal: RepairSignal,
    legacy_action: RepairAction,
    budget_delta: dict[str, float],
    attempt_id: str,
) -> RepairDecision:
    decision = controller.recommend(
        signal,
        legacy_action=legacy_action,
        budget_delta=budget_delta,
    )
    workflow_control.observe_repair(decision, attempt_id=attempt_id)
    return decision


def _record_controller_comparison(
    *,
    workflow_control: WorkflowControlSession,
    controller: RepairController,
    signal: RepairSignal,
    decision: RepairDecision,
    legacy_action: RepairAction,
    legacy_scope: RepairScope,
    outcome: RepairOutcome,
    budget_delta: dict[str, float],
    attempt_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparison = controller.compare(
        signal,
        decision,
        legacy_action=legacy_action,
        legacy_scope=legacy_scope,
        outcome=outcome,
        budget_delta=budget_delta,
        metadata=metadata,
    )
    workflow_control.compare_repair(
        comparison,
        attempt_id=attempt_id,
        subject_refs=signal.subject_refs,
    )
    return {
        "record_kind": "comparison",
        "attempt_id": attempt_id,
        "signal": signal.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "comparison": comparison.model_dump(mode="json"),
    }


def _observe_step_local_repairs(
    *,
    workflow_control: WorkflowControlSession,
    controller: RepairController,
    plan: AudioEditPlan,
    attempt_prefix: str,
) -> list[dict[str, Any]]:
    if not workflow_control.enabled:
        return []

    history: list[dict[str, Any]] = []
    for index, attempt in enumerate(
        item
        for item in plan.metadata.get("tool_contract_repair_attempts") or []
        if isinstance(item, dict)
    ):
        attempt_id = f"{attempt_prefix}:tool_contract_local:{index}"
        if attempt.get("controller_enforced"):
            raw_signal = attempt.get("signal")
            raw_decision = attempt.get("decision")
            if not isinstance(raw_signal, dict) or not isinstance(raw_decision, dict):
                continue
            signal = RepairSignal.model_validate(raw_signal)
            decision = RepairDecision.model_validate(raw_decision)
            workflow_control.observe_repair(decision, attempt_id=attempt_id)
            workflow_control.emit(
                stage="repair",
                kind="argument_repair_enforcement_selected",
                status="observed",
                attempt_id=attempt_id,
                subject_refs=signal.subject_refs,
                payload={
                    "signal_id": signal.signal_id,
                    "decision_id": decision.decision_id,
                    "action": decision.action,
                    "scope": decision.scope,
                },
            )
            workflow_control.emit(
                stage="repair",
                kind="argument_repair_enforcement_completed",
                status=_workflow_event_status(attempt.get("status")),
                attempt_id=attempt_id,
                subject_refs=signal.subject_refs,
                payload={
                    "status": attempt.get("status"),
                    "proposal": attempt.get("proposal"),
                    "admission": attempt.get("admission"),
                    "remaining_violations": attempt.get("remaining_violations"),
                    "rejection_reason": attempt.get("rejection_reason"),
                    "error": attempt.get("error"),
                    "legacy_fallback_action": attempt.get(
                        "legacy_fallback_action"
                    ),
                },
            )
            history.append(
                {
                    "record_kind": "enforcement",
                    "attempt_id": attempt_id,
                    "signal": signal.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                    "enforcement": dict(attempt),
                }
            )
            continue
        signal = controller.signal_from_local_tool_contract_attempt(attempt)
        budget = {"executor_llm_calls": 1.0}
        decision = _observe_controller_recommendation(
            workflow_control=workflow_control,
            controller=controller,
            signal=signal,
            legacy_action="repair_step_arguments",
            budget_delta=budget,
            attempt_id=attempt_id,
        )
        history.append(
            _record_controller_comparison(
                workflow_control=workflow_control,
                controller=controller,
                signal=signal,
                decision=decision,
                legacy_action="repair_step_arguments",
                legacy_scope="argument",
                outcome=_repair_outcome(attempt.get("status")),
                budget_delta=budget,
                attempt_id=attempt_id,
            )
        )

    for index, attempt in enumerate(
        item
        for item in plan.metadata.get("bounded_repair_attempts") or []
        if isinstance(item, dict)
    ):
        attempt_id = f"{attempt_prefix}:bounded:{index}"
        signal = controller.signal_from_bounded_attempt(attempt)
        budget = (
            {"executor_step_calls": 0.0, "verifier_calls": 0.0}
            if attempt.get("status") == "skipped"
            else {"executor_step_calls": 1.0, "verifier_calls": 1.0}
        )
        decision = _observe_controller_recommendation(
            workflow_control=workflow_control,
            controller=controller,
            signal=signal,
            legacy_action="repair_step_arguments",
            budget_delta=budget,
            attempt_id=attempt_id,
        )
        history.append(
            _record_controller_comparison(
                workflow_control=workflow_control,
                controller=controller,
                signal=signal,
                decision=decision,
                legacy_action="repair_step_arguments",
                legacy_scope="argument",
                outcome=_repair_outcome(attempt.get("status")),
                budget_delta=budget,
                attempt_id=attempt_id,
                metadata={
                    "score_before": attempt.get("score_before"),
                    "score_after": attempt.get("score_after"),
                },
            )
        )

    runtime_repairs = [
        item
        for item in plan.metadata.get("tool_invocations") or []
        if isinstance(item, dict) and item.get("runtime_repair")
    ]
    for index, invocation in enumerate(runtime_repairs):
        attempt_id = f"{attempt_prefix}:runtime_local:{index}"
        signal = controller.signal_from_runtime_local_invocation(invocation)
        budget = {"tool_calls": 1.0}
        decision = _observe_controller_recommendation(
            workflow_control=workflow_control,
            controller=controller,
            signal=signal,
            legacy_action="rerun_step",
            budget_delta=budget,
            attempt_id=attempt_id,
        )
        history.append(
            _record_controller_comparison(
                workflow_control=workflow_control,
                controller=controller,
                signal=signal,
                decision=decision,
                legacy_action="rerun_step",
                legacy_scope="step",
                outcome="accepted" if invocation.get("success") else "failed",
                budget_delta=budget,
                attempt_id=attempt_id,
            )
        )
    return history


def _repair_outcome(status: Any) -> RepairOutcome:
    normalized = str(status or "").strip().lower()
    if normalized in {"accepted", "rejected", "failed"}:
        return normalized  # type: ignore[return-value]
    if normalized == "skipped":
        return "not_run"
    return "unknown"


def _workflow_event_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"accepted", "rejected", "failed"}:
        return normalized
    return "observed"


def _audit_plan_admission(
    *,
    workflow_control: WorkflowControlSession,
    plan: AudioEditPlan,
    available_operations: list[dict[str, Any]],
    context: dict[str, Any],
    planner_stage: str,
    attempt_id: str,
) -> PlanAdmissionResult | None:
    if not workflow_control.enabled:
        return None

    result = PlanAdmissionGate().audit(
        plan,
        available_operations=available_operations,
        context=context,
    )
    workflow_control.emit(
        stage="planner",
        kind="plan_admission_audited",
        status="completed",
        attempt_id=attempt_id,
        subject_refs=[step.id for step in plan.steps],
        payload={
            "planner_stage": planner_stage,
            "admission": result.model_dump(mode="json"),
        },
    )
    return result


def _attempt_record(
    *, attempt: int, plan: AudioEditPlan, evaluation: Any
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "plan": plan.model_dump(),
        "evaluation": evaluation.model_dump(),
        "final_output_path": plan.metadata.get("final_output_path")
        or _last_output(plan),
        **_debug_audio_metadata(plan),
    }


def _debug_audio_metadata(plan: AudioEditPlan) -> dict[str, Any]:
    intermediate = plan.metadata.get("intermediate_audio_artifacts") or []
    return {
        "audio_artifacts": plan.metadata.get("audio_artifacts") or [],
        "debug_audio_artifacts": plan.metadata.get("debug_audio_artifacts")
        or intermediate,
        "intermediate_audio_artifacts": intermediate,
        "intermediate_audios": plan.metadata.get("intermediate_audios") or intermediate,
    }


def _execution_perception(
    context: dict[str, Any], plan: AudioEditPlan
) -> dict[str, Any]:
    perception: dict[str, Any] = {}
    for candidate in [
        context.get("perception_report"),
        plan.global_params.get("perception_report"),
    ]:
        if isinstance(candidate, dict):
            perception = _merge_non_empty_dicts(perception, candidate)
    for step in plan.steps:
        result = step.result if isinstance(step.result, dict) else {}
        candidate = result.get("perception_report")
        if isinstance(candidate, dict):
            perception = _merge_non_empty_dicts(perception, candidate)
    return perception


def _merge_non_empty_dicts(
    base: dict[str, Any], update: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if value in (None, "", [], {}):
            continue
        merged[key] = value
    return merged


def _last_output(plan: AudioEditPlan) -> str | None:
    for step in reversed(plan.steps):
        if isinstance(step.result, dict):
            for key in (
                "output_audio_path",
                "output_path",
                "target_audio_path",
                "generated_audio_path",
            ):
                value = step.result.get(key)
                if isinstance(value, str) and Path(value).exists():
                    return value
    return plan.output_audio or plan.input_audio


def _first_failed_step_error(plan: AudioEditPlan) -> str | None:
    for step in plan.steps:
        if step.status == "failed" and step.error:
            return step.error
    return None


def _evidence_output_dir(
    context: dict[str, Any],
    plan: AudioEditPlan,
    final_output_path: str | None,
    executor: AudioEditExecutor,
) -> Path:
    for value in (
        context.get("evidence_output_dir"),
        context.get("output_dir"),
        plan.global_params.get("output_dir"),
        getattr(executor, "output_dir", None),
    ):
        if value:
            return Path(value)
    if final_output_path:
        return Path(final_output_path).expanduser().parent
    return runtime_artifact_path("evidence")


def _max_evaluation_revisions(context: dict[str, Any], evaluator: Evaluator) -> int:
    if "max_evaluation_revisions" in context:
        return max(0, int(context["max_evaluation_revisions"]))
    if isinstance(evaluator, AudioJudgeEvaluator) and _context_bool(
        context,
        "enable_evaluator_repair",
        default=False,
    ):
        return max(0, int(os.getenv("X_EDITOR_EVALUATOR_MAX_REVISIONS", "1")))
    return max(0, int(os.getenv("X_EDITOR_EVALUATOR_MAX_REVISIONS", "0")))


def _max_tool_contract_revisions(context: dict[str, Any]) -> int:
    if "max_tool_contract_revisions" in context:
        return max(0, int(context["max_tool_contract_revisions"]))
    if _context_bool(context, "enable_tool_contract_repair", default=True):
        return max(0, int(os.getenv("X_EDITOR_TOOL_CONTRACT_MAX_REVISIONS", "1")))
    return 0


def _max_runtime_verification_revisions(context: dict[str, Any]) -> int:
    if "max_runtime_verification_revisions" in context:
        return max(0, int(context["max_runtime_verification_revisions"]))
    if _context_bool(context, "enable_runtime_verification_repair", default=False):
        return max(0, int(os.getenv("X_EDITOR_RUNTIME_REPAIR_MAX_REVISIONS", "1")))
    return max(0, int(os.getenv("X_EDITOR_RUNTIME_REPAIR_MAX_REVISIONS", "0")))


def _needs_tool_contract_replan(plan: AudioEditPlan) -> bool:
    if not plan.metadata.get("need_tool_contract_replan"):
        return False
    return any(
        isinstance(item, dict) and item.get("severity") == "blocking"
        for item in plan.metadata.get("tool_contract_violations") or []
    )


def _tool_contract_violations(plan: AudioEditPlan) -> list[dict[str, Any]]:
    return [
        item
        for item in plan.metadata.get("tool_contract_violations") or []
        if isinstance(item, dict)
    ]


def _tool_contract_quality(plan: AudioEditPlan) -> float:
    violations = _tool_contract_violations(plan)
    blocking = sum(1 for item in violations if item.get("severity") == "blocking")
    summary = execution_summary(plan)
    failed = len(summary.get("failed_steps") or [])
    if not violations:
        return 1.0 if not failed else max(0.0, 0.8 - 0.1 * failed)
    return round(max(0.0, 1.0 - 0.35 * blocking - 0.05 * failed), 6)


def _tool_contract_retry_guidance(plan: AudioEditPlan) -> list[str]:
    guidance: list[str] = []
    for violation in _tool_contract_violations(plan):
        for item in violation.get("retry_guidance") or []:
            guidance.append(str(item))
        suggestions = violation.get("suggested_operations") or []
        if suggestions:
            guidance.append(
                f"Suggested operations: {', '.join(str(item) for item in suggestions)}"
            )
    return _dedupe_strings(guidance)


def _needs_runtime_verification_repair(plan: AudioEditPlan) -> bool:
    if not plan.metadata.get("need_repair"):
        return False
    reports = plan.metadata.get("verification_reports") or []
    return any(
        isinstance(report, dict) and report.get("passed") is False for report in reports
    )


def _runtime_verification_quality(plan: AudioEditPlan) -> float:
    reports = [
        report
        for report in (plan.metadata.get("verification_reports") or [])
        if isinstance(report, dict)
    ]
    if not reports:
        return 1.0 if not plan.metadata.get("need_repair") else 0.0
    values = []
    for report in reports:
        score = report.get("score")
        if isinstance(score, (int, float)):
            values.append(float(score))
        else:
            values.append(1.0 if report.get("passed") else 0.0)
    quality = sum(values) / max(1, len(values))
    quality -= 0.2 * sum(
        1
        for report in reports
        if report.get("passed") is False and report.get("severity") == "blocking"
    )
    quality -= 0.05 * sum(
        1
        for report in reports
        if report.get("passed") is False and report.get("severity") == "warning"
    )
    return round(max(0.0, quality), 6)


def _evaluation_repair_quality(plan: AudioEditPlan, evaluation: Any) -> float:
    summary = execution_summary(plan)
    if not summary.get("completed"):
        return -1.0
    output_path = plan.metadata.get("final_output_path") or _last_output(plan)
    if not output_path or not Path(str(output_path)).exists():
        return -0.5
    score = getattr(evaluation, "score", None)
    if isinstance(score, (int, float)):
        quality = float(score)
    else:
        quality = 1.0 if getattr(evaluation, "passed", False) else 0.0
    if getattr(evaluation, "passed", False):
        quality += 1.0
    return round(quality, 6)


def _runtime_verification_defects(plan: AudioEditPlan) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for report in plan.metadata.get("verification_reports") or []:
        if not isinstance(report, dict) or report.get("passed") is not False:
            continue
        defects.append(
            {
                "component": report.get("component"),
                "step_id": report.get("step_id"),
                "operation": report.get("operation"),
                "score": report.get("score"),
                "severity": report.get("severity"),
                "findings": report.get("findings") or [],
                "repair_hints": report.get("repair_hints") or [],
                "repair_actions": report.get("repair_actions") or [],
            }
        )
    return defects


def _runtime_verification_retry_guidance(plan: AudioEditPlan) -> list[str]:
    guidance: list[str] = []
    for repair_plan in plan.metadata.get("verification_repair_plans") or []:
        if not isinstance(repair_plan, dict):
            continue
        for action in repair_plan.get("major_actions") or []:
            if not isinstance(action, dict):
                continue
            rationale = action.get("rationale")
            strategy = action.get("strategy")
            params = action.get("params")
            guidance.append(
                f"Major repair action {strategy}: {rationale}. Params: {params}"
            )
        for hint in repair_plan.get("repair_hints") or []:
            guidance.append(str(hint))
    return _dedupe_strings(guidance)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _edit_session_graph_enabled(context: dict[str, Any]) -> bool:
    return bool(
        context.get("enable_edit_session_graph")
        or context.get("edit_session_id")
        or context.get("session_id")
        or context.get("edit_session_graph")
        or context.get("edit_session_graph_path")
    )


def _session_input_audios(
    graph: EditSessionGraph, context: dict[str, Any]
) -> list[str]:
    values: list[str] = []
    current = graph.current_audio_path()
    if current:
        values.append(current)
    values.extend(
        _string_list(
            context.get("input_audios") or context.get("input_audio_paths")
        )
    )
    values.extend(_string_list(context.get("reference_audio_paths")))
    for node in graph.nodes.values():
        if node.primary_audio_path:
            values.append(node.primary_audio_path)
        for asset in node.audio_paths:
            if asset.path:
                values.append(asset.path)
    return _dedupe_strings([str(item) for item in values if item])


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _context_bool(context: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in context:
        return default
    value = context[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
