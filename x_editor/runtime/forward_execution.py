"""Evaluator-free single-pass execution path for controlled workflow studies."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.core.invocation_trace import (
    ComponentInvocationAdmissionPolicy,
    ComponentInvocationTrace,
    validate_component_invocation_trace,
)
from x_editor.core.schemas import AudioEditPlan, StrictModel
from x_editor.reasoner.contracts import SessionStateContext
from x_editor.runtime.audio_base_selection import (
    INITIAL_AUDIO_AUTHORITY_KEY,
    AdmittedInitialAudio,
    plan_selects_bound_reference_base,
)
from x_editor.runtime.audio_plan_executor import AudioEditExecutor, execution_summary
from x_editor.runtime.execution_graph import (
    ExecutionGraph,
    ExecutionGraphCompileError,
    compile_execution_graph,
)
from x_editor.runtime.execution_events import (
    ExecutionEventObserver,
    ExecutionEventRecorder,
)
from x_editor.runtime.workflow_case import RuntimeAudioBinding
from x_editor.workflow_control import (
    ArtifactFinalizationAudit,
    ArtifactFinalizationAuditor,
    ArtifactStore,
    ArtifactStoreCommit,
    FinalAssemblerShadow,
    FinalDeliveryManifest,
    PlanAdmissionGate,
    PlanAdmissionResult,
    WorkflowEvent,
    assemble_final_output,
)
from x_editor.workflow_control.schemas import new_workflow_id
from x_editor.workflow_control.forward_resume import RestrictedForwardResume, admit_forward_resume
from x_editor.workflow_control.execution_state import reset_candidate_execution_state

ForwardExecutionStatus = Literal["completed", "blocked", "failed"]
ForwardFailureStage = Literal[
    "none",
    "configuration",
    "plan_admission",
    "executor",
    "verifier",
    "artifact_store",
]


class ForwardExecutionPolicy(StrictModel):
    """Frozen v1 policy: one strict forward attempt with no repair authority."""

    schema_version: Literal["x-editor.forward-execution-policy/v1"] = (
        "x-editor.forward-execution-policy/v1"
    )
    plan_admission_mode: Literal["enforced"] = "enforced"
    tool_contract_mode: Literal["enforced"] = "enforced"
    runtime_verification_mode: Literal["strict"] = "strict"
    artifact_store_mode: Literal["enforced"] = "enforced"
    final_assembler_mode: Literal["shadow"] = "shadow"
    repair_mode: Literal["disabled"] = "disabled"
    executor_llm_mode: Literal["disabled"] = "disabled"
    operation_selection_mode: Literal["preserve"] = "preserve"
    tool_fallback_mode: Literal["disabled"] = "disabled"


class ForwardExecutionRequest(StrictModel):
    """Complete input contract for one evaluator-free execution attempt."""

    schema_version: Literal["x-editor.forward-execution-request/v1"] = (
        "x-editor.forward-execution-request/v1"
    )
    run_id: str = Field(default_factory=lambda: new_workflow_id("forward"))
    plan: AudioEditPlan
    context: dict[str, Any] = Field(default_factory=dict)
    suffix_resume: RestrictedForwardResume | None = Field(default=None, exclude=True)
    audio_resolution_mode: Literal["legacy_path", "opaque_ref_v1"] = (
        "legacy_path"
    )
    session_state: SessionStateContext | None = Field(default=None, exclude=True)
    audio_bindings: list[RuntimeAudioBinding] = Field(
        default_factory=list,
        exclude=True,
    )
    runtime_output_dir: str | None = Field(default=None, exclude=True)
    event_log_path: str | None = Field(default=None, exclude=True)
    policy: ForwardExecutionPolicy = Field(default_factory=ForwardExecutionPolicy)
    planned_perception_invocation_policy: (
        ComponentInvocationAdmissionPolicy | None
    ) = None
    planned_perception_evidence_mode: Literal["disabled", "reviewed"] = (
        "disabled"
    )

    @model_validator(mode="after")
    def validate_private_audio_bindings(self) -> ForwardExecutionRequest:
        bound_refs = [item.audio_ref for item in self.audio_bindings]
        if len(bound_refs) != len(set(bound_refs)):
            raise ValueError("runtime audio binding refs must be unique")
        binding_orders = [item.order for item in self.audio_bindings]
        if len(binding_orders) != len(set(binding_orders)):
            raise ValueError("runtime audio binding orders must be unique")
        if self.audio_resolution_mode == "opaque_ref_v1" and not bound_refs:
            raise ValueError(
                "opaque forward execution requires private audio bindings"
            )
        return self


class ForwardExecutionResult(StrictModel):
    """Terminal facts from one forward attempt; no Evaluator verdict is present."""

    schema_version: Literal["x-editor.forward-execution-result/v1"] = (
        "x-editor.forward-execution-result/v1"
    )
    run_id: str
    status: ForwardExecutionStatus
    failure_stage: ForwardFailureStage = "none"
    failure_codes: list[str] = Field(default_factory=list)
    policy: ForwardExecutionPolicy
    input_plan: AudioEditPlan
    admission: PlanAdmissionResult | None = None
    execution_graph: ExecutionGraph | None = None
    executed_plan: AudioEditPlan | None = None
    execution_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_audio_bindings: list[RuntimeAudioBinding] = Field(default_factory=list)
    recovery_session_state: dict[str, Any] | None = None
    artifact_audit: ArtifactFinalizationAudit | None = None
    artifact_store_commit: ArtifactStoreCommit | None = None
    final_delivery_manifest_shadow: FinalDeliveryManifest | None = None
    final_output_path: str | None = None
    executor_run_count: int = Field(default=0, ge=0, le=1)
    planner_call_count: Literal[0] = 0
    evaluator_call_count: Literal[0] = 0
    repair_attempt_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    planned_perception_invocation_traces: list[ComponentInvocationTrace] = Field(
        default_factory=list
    )
    planned_perception_logical_provider_call_count: int = Field(default=0, ge=0)
    planned_perception_provider_call_count: int = Field(default=0, ge=0)
    planned_perception_provider_retry_count: int = Field(default=0, ge=0)
    planned_perception_logical_retry_count: int = Field(default=0, ge=0)
    planned_perception_json_repair_count: int = Field(default=0, ge=0)
    planned_perception_schema_repair_count: int = Field(default=0, ge=0)
    planned_perception_fallback_count: int = Field(default=0, ge=0)
    event_log_path: str | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)
    observation_errors: list[dict[str, Any]] = Field(default_factory=list)


class ForwardExecutionPath:
    """Run an already-frozen plan through one strict, evaluator-free attempt."""

    def __init__(self, executor: AudioEditExecutor) -> None:
        self.executor = executor

    def run(
        self,
        request: ForwardExecutionRequest,
        *,
        observer: ExecutionEventObserver | None = None,
    ) -> ForwardExecutionResult:
        """Synchronous wrapper for callers outside an event loop."""

        return asyncio.run(self.arun(request, observer=observer))

    async def arun(
        self,
        request: ForwardExecutionRequest,
        *,
        observer: ExecutionEventObserver | None = None,
    ) -> ForwardExecutionResult:
        """Execute one plan without constructing Planner, Evaluator, or repair loops."""

        if request.suffix_resume is not None:
            request = request.model_copy(deep=True)
            reset_candidate_execution_state(request.plan)

        recorder = ExecutionEventRecorder(
            run_id=request.run_id,
            event_log_path=request.event_log_path,
            observer=observer,
        )

        recorder.emit(
            stage="runner",
            kind="forward_execution_started",
            status="started",
            attempt_id="execution:0",
            subject_refs=[step.id for step in request.plan.steps],
            payload={
                "policy": request.policy.model_dump(mode="json"),
                "planned_perception_invocation_policy": (
                    request.planned_perception_invocation_policy.model_dump(
                        mode="json"
                    )
                    if request.planned_perception_invocation_policy is not None
                    else None
                ),
                "planned_perception_evidence_mode": (
                    request.planned_perception_evidence_mode
                ),
                "planner_calls": 0,
                "evaluator_calls": 0,
            },
        )

        context, configuration_errors = _frozen_forward_context(request.context)
        if configuration_errors:
            recorder.emit(
                stage="runner",
                kind="forward_execution_configuration_rejected",
                status="failed",
                attempt_id="execution:0",
                payload={"errors": configuration_errors},
            )
            return self._result(
                request=request,
                recorder=recorder,
                status="failed",
                failure_stage="configuration",
                failure_codes=[item["code"] for item in configuration_errors],
            )

        if request.runtime_output_dir is not None:
            context["output_dir"] = request.runtime_output_dir
        if request.audio_resolution_mode == "opaque_ref_v1":
            context["_runtime_audio_resolution_mode"] = "opaque_ref_v1"
            context["_runtime_audio_bindings"] = [
                item.model_copy(deep=True) for item in request.audio_bindings
            ]

        context.pop(INITIAL_AUDIO_AUTHORITY_KEY, None)
        operations = self.executor.registry.describe_operations()
        admission = PlanAdmissionGate().audit(
            request.plan,
            available_operations=operations,
            context=context,
        )
        forward_admission_blockers = _forward_admission_blockers(
            admission,
            request.plan,
        )
        recorder.emit(
            stage="verifier",
            kind="forward_plan_admission_completed",
            status="accepted" if not forward_admission_blockers else "rejected",
            attempt_id="execution:0",
            subject_refs=[step.id for step in request.plan.steps],
            payload={
                "admission": admission.model_dump(mode="json"),
                "forward_blocking_codes": forward_admission_blockers,
            },
        )
        if forward_admission_blockers:
            failure_codes = forward_admission_blockers
            recorder.emit(
                stage="runner",
                kind="forward_execution_blocked",
                status="failed",
                attempt_id="execution:0",
                subject_refs=[step.id for step in request.plan.steps],
                payload={
                    "failure_stage": "plan_admission",
                    "failure_codes": failure_codes,
                },
            )
            return self._result(
                request=request,
                recorder=recorder,
                status="blocked",
                failure_stage="plan_admission",
                failure_codes=failure_codes,
                admission=admission,
            )
        try:
            execution_graph = compile_execution_graph(
                request.plan,
                graph_id=f"{request.run_id}:execution:0",
            )
        except ExecutionGraphCompileError as exc:
            failure_code = f"execution_graph_compile_rejected:{exc.code}"
            recorder.emit(
                stage="verifier",
                kind="forward_execution_graph_rejected",
                status="rejected",
                attempt_id="execution:0",
                subject_refs=[step.id for step in request.plan.steps],
                payload={
                    "failure_code": failure_code,
                    "step_id": exc.step_id,
                    "field_path": exc.field_path,
                    "message": str(exc),
                },
            )
            return self._result(
                request=request,
                recorder=recorder,
                status="blocked",
                failure_stage="plan_admission",
                failure_codes=[failure_code],
                admission=admission,
            )
        context["_forward_execution_graph"] = execution_graph.model_dump(mode="python")
        recorder.emit(
            stage="runner",
            kind="forward_execution_graph_compiled",
            status="accepted",
            attempt_id="execution:0",
            subject_refs=[node.plan_step_id for node in execution_graph.nodes],
            payload={
                "graph_id": execution_graph.graph_id,
                "compiler_version": execution_graph.compiler_version,
                "topological_order": list(execution_graph.topological_order),
                "primary_output": execution_graph.primary_output.model_dump(mode="json"),
            },
        )
        authored_audio_usage = request.plan.metadata.get(
            "workflow_audio_usage_bindings"
        )
        if isinstance(authored_audio_usage, list):
            context["_runtime_admitted_audio_usage_bindings"] = [
                dict(item) for item in authored_audio_usage if isinstance(item, dict)
            ]
        if plan_selects_bound_reference_base(
            request.plan,
            reference_audio_refs=[
                item.audio_ref for item in request.audio_bindings if item.role == "reference"
            ],
            session_state=request.session_state,
        ):
            context[INITIAL_AUDIO_AUTHORITY_KEY] = AdmittedInitialAudio(request.plan.input_audio)
        if request.planned_perception_invocation_policy is not None:
            context["_forward_planned_perception_invocation_policy"] = (
                request.planned_perception_invocation_policy.model_dump(
                    mode="json"
                )
            )
        context["_forward_planned_perception_evidence_mode"] = (
            request.planned_perception_evidence_mode
        )

        if request.suffix_resume is not None:
            resume = request.suffix_resume
            reuse = admit_forward_resume(
                resume, request.plan, request.audio_bindings,
                session_state=request.session_state.model_dump(mode="json") if request.session_state else None,
            )
            if reuse.status == "admitted":
                context["_workflow_suffix_resume_reuse_receipts"] = [
                    item.model_dump(mode="json") for item in reuse.receipts
                ]
            recorder.emit(
                stage="runner", kind="forward_suffix_resume_decision",
                status="accepted" if reuse.status == "admitted" else "rejected",
                attempt_id=f"recovery:{resume.previous_attempt + 1}",
                payload={"prior_run_id": resume.prior_run_id,
                         "attempt_number": resume.previous_attempt + 1,
                         "max_total_attempts": resume.max_total_attempts,
                         "remaining_attempts": resume.max_total_attempts - resume.previous_attempt - 1,
                         "mechanical_reuse_not_quality_certification": True,
                         "admission": reuse.model_dump(mode="json")},
            )

        recorder.emit(
            stage="executor",
            kind="forward_executor_started",
            status="started",
            attempt_id="execution:0",
            subject_refs=[step.id for step in request.plan.steps],
        )
        try:
            executed_plan = await self.executor.aexecute(
                request.plan,
                context=context,
                observer=recorder,
            )
        except Exception as exc:
            recorder.emit(
                stage="executor",
                kind="forward_executor_failed",
                status="failed",
                attempt_id="execution:0",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return self._result(
                request=request,
                recorder=recorder,
                status="failed",
                failure_stage="executor",
                failure_codes=["executor_raised"],
                admission=admission,
                execution_graph=execution_graph,
                executor_run_count=1,
            )

        summary = execution_summary(executed_plan)
        implicit_repair_facts = _implicit_repair_facts(executed_plan)
        recorder.emit(
            stage="executor",
            kind="forward_executor_completed",
            status="completed" if summary.get("completed") else "failed",
            attempt_id="execution:0",
            subject_refs=[step.id for step in executed_plan.steps],
            payload={"execution_summary": summary},
        )

        try:
            audit = ArtifactFinalizationAuditor().audit(
                executed_plan,
                attempt_id="execution:0",
                attempt_outcome=(
                    "rejected" if implicit_repair_facts else "selected"
                ),
            )
        except Exception as exc:
            recorder.emit(
                stage="finalization",
                kind="forward_artifact_finalization_audit_failed",
                status="failed",
                attempt_id="execution:0",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return self._result(
                request=request,
                recorder=recorder,
                status="failed",
                failure_stage="artifact_store",
                failure_codes=["artifact_finalization_audit_raised"],
                admission=admission,
                executed_plan=executed_plan,
                summary=summary,
                executor_run_count=1,
            )
        recorder.emit(
            stage="finalization",
            kind="forward_artifact_finalization_audited",
            status=("accepted" if audit.status == "would_finalize" else "rejected"),
            attempt_id="execution:0",
            subject_refs=[item.artifact_ref_id for item in audit.artifacts],
            payload={"audit": audit.model_dump(mode="json")},
        )
        try:
            commit = ArtifactStore().commit(audit)
        except Exception as exc:
            recorder.emit(
                stage="finalization",
                kind="forward_artifact_store_failed",
                status="failed",
                attempt_id="execution:0",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return self._result(
                request=request,
                recorder=recorder,
                status="failed",
                failure_stage="artifact_store",
                failure_codes=["artifact_store_raised"],
                admission=admission,
                executed_plan=executed_plan,
                summary=summary,
                audit=audit,
                executor_run_count=1,
            )
        recorder.emit(
            stage="finalization",
            kind=(
                "forward_artifact_store_committed"
                if commit.status == "committed"
                else "forward_artifact_store_blocked"
            ),
            status="accepted" if commit.status == "committed" else "failed",
            attempt_id="execution:0",
            subject_refs=[
                ref
                for ref in [
                    commit.final_artifact_ref_id,
                    *commit.rejected_artifact_refs,
                    *commit.candidate_artifact_refs,
                ]
                if ref
            ],
            payload={"commit": commit.model_dump(mode="json")},
        )
        manifest: FinalDeliveryManifest | None = None
        try:
            manifest = FinalAssemblerShadow().project(executed_plan, audit, commit)
        except Exception as exc:
            recorder.record_observation_error(
                component="final_assembler_shadow",
                operation="project",
                error=exc,
            )
            recorder.emit(
                stage="finalization",
                kind="forward_final_delivery_manifest_observation_failed",
                status="failed",
                attempt_id="execution:0",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "behavior_changed": False,
                },
            )
        else:
            recorder.emit(
                stage="finalization",
                kind="forward_final_delivery_manifest_observed",
                status="observed",
                attempt_id="execution:0",
                payload={
                    "manifest": manifest.model_dump(mode="json"),
                    "behavior_changed": False,
                },
            )

        failure_stage, failure_codes = _terminal_failure(
            executed_plan=executed_plan,
            summary=summary,
            commit=commit,
            implicit_repair_facts=implicit_repair_facts,
        )
        status: ForwardExecutionStatus
        if failure_stage == "none":
            status = "completed"
        elif failure_stage == "executor":
            status = "failed"
        else:
            status = "blocked"
        recorder.emit(
            stage="runner",
            kind={
                "completed": "forward_execution_completed",
                "blocked": "forward_execution_blocked",
                "failed": "forward_execution_failed",
            }[status],
            status="completed" if status == "completed" else "failed",
            attempt_id="execution:0",
            subject_refs=[step.id for step in executed_plan.steps],
            payload={
                "failure_stage": failure_stage,
                "failure_codes": failure_codes,
                "final_output_path": assemble_final_output(commit),
                "planner_calls": 0,
                "evaluator_calls": 0,
            },
        )
        return self._result(
            request=request,
            recorder=recorder,
            status=status,
            failure_stage=failure_stage,
            failure_codes=failure_codes,
            admission=admission,
            executed_plan=executed_plan,
            summary=summary,
            audit=audit,
            commit=commit,
            manifest=manifest,
            executor_run_count=1,
        )

    def _result(
        self,
        *,
        request: ForwardExecutionRequest,
        recorder: ExecutionEventRecorder,
        status: ForwardExecutionStatus,
        failure_stage: ForwardFailureStage,
        failure_codes: list[str],
        admission: PlanAdmissionResult | None = None,
        execution_graph: ExecutionGraph | None = None,
        executed_plan: AudioEditPlan | None = None,
        summary: dict[str, Any] | None = None,
        audit: ArtifactFinalizationAudit | None = None,
        commit: ArtifactStoreCommit | None = None,
        manifest: FinalDeliveryManifest | None = None,
        executor_run_count: int = 0,
    ) -> ForwardExecutionResult:
        metadata = executed_plan.metadata if executed_plan is not None else {}
        planned_perception_traces = _planned_perception_invocation_traces(
            executed_plan
        )
        repair_attempt_count = sum(
            _record_count(metadata.get(key))
            for key in _IMPLICIT_REPAIR_METADATA_KEYS
        )
        if executed_plan is not None:
            raw_graph = executed_plan.metadata.get("execution_graph")
            if isinstance(raw_graph, dict):
                execution_graph = ExecutionGraph.model_validate(raw_graph)
        return ForwardExecutionResult(
            run_id=request.run_id,
            runtime_audio_bindings=[b.model_copy(deep=True) for b in request.audio_bindings],
            recovery_session_state=request.session_state.model_dump(mode="json") if request.session_state else None,
            status=status,
            failure_stage=failure_stage,
            failure_codes=list(dict.fromkeys(failure_codes)),
            policy=request.policy,
            input_plan=request.plan,
            admission=admission,
            execution_graph=execution_graph,
            executed_plan=executed_plan,
            execution_summary=dict(summary or {}),
            artifact_audit=audit,
            artifact_store_commit=commit,
            final_delivery_manifest_shadow=manifest,
            final_output_path=assemble_final_output(commit) if commit else None,
            executor_run_count=executor_run_count,
            repair_attempt_count=repair_attempt_count,
            tool_call_count=len(metadata.get("tool_invocations") or []),
            planned_perception_invocation_traces=planned_perception_traces,
            planned_perception_logical_provider_call_count=sum(
                item.logical_provider_call_count
                for item in planned_perception_traces
            ),
            planned_perception_provider_call_count=sum(
                item.provider_call_count for item in planned_perception_traces
            ),
            planned_perception_provider_retry_count=sum(
                item.retry_count for item in planned_perception_traces
            ),
            planned_perception_logical_retry_count=sum(
                item.logical_retry_count for item in planned_perception_traces
            ),
            planned_perception_json_repair_count=sum(
                item.json_repair_count for item in planned_perception_traces
            ),
            planned_perception_schema_repair_count=sum(
                item.schema_repair_count for item in planned_perception_traces
            ),
            planned_perception_fallback_count=sum(
                len(item.fallback_events) for item in planned_perception_traces
            ),
            event_log_path=recorder.event_log_path,
            events=list(recorder.events),
            observation_errors=[
                *recorder.observation_errors,
                *list(metadata.get("execution_observer_errors") or []),
            ],
        )


_FROZEN_CONTEXT: dict[str, Any] = {
    "workflow_control_mode": "disabled",
    "workflow_artifact_store_mode": "disabled",
    "workflow_final_assembler_mode": "disabled",
    "workflow_enforced_repair_actions": [],
    "enable_tool_contract_verification": True,
    "enable_runtime_verification": True,
    "enable_model_verification": False,
    "verify_model": False,
    "runtime_verification_strict": True,
    "enable_tool_contract_repair": False,
    "enable_runtime_verification_repair": False,
    "enable_evaluator_repair": False,
    "enable_bounded_repair": False,
    "max_bounded_micro_repairs": 0,
    "max_tool_contract_revisions": 0,
    "max_runtime_verification_revisions": 0,
    "max_evaluation_revisions": 0,
    "use_executor_tool_call_planner": False,
    "use_gemini_executor": False,
    "executor_llm_repair": False,
    "allow_implicit_executor_repair": False,
    "allow_speech_rate_method_fallback": False,
    "preserve_explicit_operation_selection": True,
    "max_transient_tool_attempts": 1,
    "continue_on_error": False,
    "enable_operation_graph_execution": False,
}

_FORBIDDEN_CONTEXT_PREFIXES = (
    "_workflow_",
    "_execution_",
    "workflow_suffix_resume",
)

_IMPLICIT_REPAIR_METADATA_KEYS = (
    "operation_rewrites",
    "bounded_repair_attempts",
    "tool_contract_repair_attempts",
    "executor_tool_call_repairs",
    "source_text_repairs",
    "emphasis_stress_perception_repairs",
    "emphasis_stress_fallbacks",
    "perception_audio_repairs",
    "flexsed_window_repairs",
    "artifact_ops_input_repairs",
    "artifact_ops_reconstruct_repairs",
    "artifact_ops_followup_span_reuse",
    "perception_requirement_relaxations",
    "executor_reference_repairs",
)


def _frozen_forward_context(
    context: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    for key, actual in context.items():
        if not key.startswith(_FORBIDDEN_CONTEXT_PREFIXES):
            continue
        errors.append(
            {
                "code": f"forward_private_context_forbidden:{key}",
                "field": key,
                "actual": actual,
            }
        )
    for key, expected in _FROZEN_CONTEXT.items():
        if key not in context:
            continue
        actual = context[key]
        if _equivalent_setting(actual, expected):
            continue
        errors.append(
            {
                "code": f"forward_policy_conflict:{key}",
                "field": key,
                "expected": expected,
                "actual": actual,
            }
        )
    frozen = dict(context)
    frozen.update(_FROZEN_CONTEXT)
    return frozen, errors


def _equivalent_setting(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str):
            normalized = actual.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                actual = True
            elif normalized in {"0", "false", "no", "off"}:
                actual = False
        return actual is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return actual in (None, "", [], (), set(), frozenset())
    return str(actual).strip().lower() == str(expected).strip().lower()


def _forward_admission_blockers(
    admission: PlanAdmissionResult,
    plan: AudioEditPlan,
) -> list[str]:
    strict_warning_codes = {
        "plan_has_no_steps",
        "required_input_not_explicit",
        "required_param_not_explicit",
    }
    blockers = [
        finding.code
        for finding in admission.findings
        if finding.severity == "blocking"
        or finding.code in strict_warning_codes
    ]
    if _equivalent_setting(plan.global_params.get("continue_on_error"), True):
        blockers.append("forward_policy_conflict:global_params.continue_on_error")
    blockers.extend(
        f"forward_policy_conflict:steps.{step.id}.continue_on_error"
        for step in plan.steps
        if step.continue_on_error is True
    )
    return list(dict.fromkeys(blockers))


def _terminal_failure(
    *,
    executed_plan: AudioEditPlan,
    summary: dict[str, Any],
    commit: ArtifactStoreCommit,
    implicit_repair_facts: list[str],
) -> tuple[ForwardFailureStage, list[str]]:
    if implicit_repair_facts:
        return "executor", [
            f"implicit_executor_repair_observed:{item}"
            for item in implicit_repair_facts
        ]
    if not summary.get("completed"):
        blocking_reports = [
            report
            for report in executed_plan.metadata.get("verification_reports") or []
            if isinstance(report, dict)
            and report.get("passed") is False
            and report.get("severity") == "blocking"
        ]
        if blocking_reports:
            failure_codes: list[str] = []
            for report in blocking_reports:
                if report.get("component") == "workflow.component_invocation_trace":
                    failure_codes.extend(
                        str(item) for item in report.get("findings") or []
                    )
                else:
                    failure_codes.append(
                        str(
                            report.get("component")
                            or "runtime_verification_failed"
                        )
                    )
            return "verifier", list(dict.fromkeys(failure_codes))
        failed_steps = [str(item) for item in summary.get("failed_steps") or []]
        required_skipped_steps = [
            str(item)
            for item in summary.get("required_skipped_steps") or []
        ]
        if failed_steps:
            return "executor", [
                f"step_failed:{step_id}" for step_id in failed_steps
            ]
        if required_skipped_steps:
            return "executor", [
                f"required_step_skipped:{step_id}"
                for step_id in required_skipped_steps
            ]
        return "executor", ["execution_incomplete"]
    if commit.status != "committed":
        return "artifact_store", list(commit.failure_codes) or [
            "artifact_store_blocked"
        ]
    return "none", []


def _implicit_repair_facts(plan: AudioEditPlan) -> list[str]:
    return [
        key
        for key in _IMPLICIT_REPAIR_METADATA_KEYS
        if _record_count(plan.metadata.get(key)) > 0
    ]


def _record_count(value: Any) -> int:
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return len(value)
    return int(value not in (None, "", False))


def _planned_perception_invocation_traces(
    plan: AudioEditPlan | None,
) -> list[ComponentInvocationTrace]:
    if plan is None:
        return []
    traces: list[ComponentInvocationTrace] = []
    for step in plan.steps:
        if not step.operation.startswith(("perception.", "flexsed.")):
            continue
        result = step.result if isinstance(step.result, dict) else {}
        raw_trace = result.get("component_invocation_trace")
        if raw_trace is None:
            continue
        try:
            trace = validate_component_invocation_trace(raw_trace)
        except Exception:
            continue
        if trace.component == "perception":
            traces.append(trace.model_copy(deep=True))
    return traces


__all__ = [
    "ForwardExecutionPath",
    "ForwardExecutionPolicy",
    "ForwardExecutionRequest",
    "ForwardExecutionResult",
]
