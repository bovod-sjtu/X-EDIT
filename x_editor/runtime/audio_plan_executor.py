"""Structured-plan executor built on the existing X-Editor tool framework."""

from __future__ import annotations

import asyncio
import copy
import difflib
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from html import unescape
from itertools import pairwise
from pathlib import Path
from typing import Any

from x_editor.agents.base import GeminiClient
from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.capabilities.catalog import enrich_operation_descriptions
from x_editor.core.agentic_mode import (
    GEMINI_MODE,
    GLM_PROVIDER,
    OPEN_GLM_QWEN_OMNI_MODE,
    resolve_agentic_mode,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.config import AudioEditConfig
from x_editor.core.invocation_trace import (
    ComponentInvocationAdmissionPolicy,
    ComponentInvocationTrace,
    component_invocation_trace_blockers,
    validate_component_invocation_trace,
)
from x_editor.core.reviewed_evidence import (
    sanitize_reviewed_perception_result,
)
from x_editor.core.schemas import (
    AudioEditPlan,
    AudioEditStep,
    ExecutorToolCall,
    ToolManual,
)
from x_editor.perception import (
    PerceptionRequirements,
    SpeechPerceptionClient,
    build_audio_perception_fallback,
    build_event_candidates,
    run_speaker_attributed_review,
    run_timestamped_caption_review,
)
from x_editor.perception.invocation_trace import (
    finalize_frontend_invocation_trace,
    reset_frontend_invocation_trace,
)
from x_editor.runtime.accent_conversion import (
    execute_accent_conversion,
    requests_reference_accent_match,
)
from x_editor.runtime.artifact_delivery import (
    ARTIFACT_DELIVERY_CONTRACT_KEY,
    attach_artifact_delivery_contract,
    collect_consumed_audio_paths,
)
from x_editor.runtime.artifact_ops import execute_artifact_step, select_timeline_spans
from x_editor.runtime.audio_base_selection import resolve_initial_audio_reference
from x_editor.runtime.audio_effects import canonical_audio_effect_preset
from x_editor.runtime.audio_references import (
    AudioReferenceResolutionError,
    bound_audio_refs,
    canonical_audio_ref_for_runtime_path,
    canonicalize_perception_audio_bindings,
    canonicalize_runtime_audio_paths,
    collect_input_audios,
    opaque_audio_resolution_enabled,
    register_runtime_audio_paths,
    require_existing_audio_path,
    resolve_audio_reference,
    resolve_audio_references_in_mapping,
    restore_canonical_audio_references,
    validate_materialized_audio_reference_roles,
)
from x_editor.runtime.audio_temporal_lineage import (
    AUDIO_TEMPORAL_LINEAGE_KEY,
    AUDIO_TEMPORAL_TRANSFORM_KEY,
    AUDIO_TIMELINE_LEDGER_KEY,
    attach_audio_temporal_transform,
)
from x_editor.runtime.execution_events import ExecutionEventObserver
from x_editor.runtime.execution_graph import ExecutionGraph
from x_editor.runtime.executor_tool_call_planner import ExecutorToolCallPlanner
from x_editor.runtime.foreground_background_remix import (
    execute_foreground_background_remix,
)
from x_editor.runtime.gemini_executor import (
    GeminiExecutorToolCallPlanner,
    apply_executor_tool_call,
    validate_executor_tool_call,
)
from x_editor.runtime.intra_audio_voice_transfer import (
    execute_intra_audio_voice_transfer,
)
from x_editor.runtime.music_vocal_timeline import execute_music_vocal_timeline_edit
from x_editor.runtime.normalization_audit import (
    build_normalization_audit,
    normalization_snapshot,
)
from x_editor.runtime.openai_compatible_executor import GLMExecutorToolCallPlanner
from x_editor.runtime.operation_normalizers import (
    NonEquivalentNormalizationError,
    OperationBoundaryError,
    build_edit_data_pipeline_tool_arguments,
    canonicalize_language,
    edit_data_pipeline_label_contract,
    edit_data_pipeline_label_requirements,
    filter_edit_pipeline_service_overrides,
    normalize_step_for_operation,
)
from x_editor.runtime.paralinguistic_events import (
    execute_paralinguistic_event_add,
    execute_paralinguistic_event_remove,
)
from x_editor.runtime.rewrite_rules import (
    absolute_timeline_input_binding_rule,
    apply_runtime_rewrite_rules,
    default_post_gemini_rewrite_rules,
    default_pre_gemini_rewrite_rules,
    record_runtime_rewrite_rule_hit,
)
from x_editor.runtime.segmented_edit import maybe_execute_segmented_edit_pipeline
from x_editor.runtime.timeline_ledger import TimeInterval, TimelineLedger
from x_editor.runtime.tool_contract_verifier import (
    ToolContractViolationError,
    build_tool_contract_violation,
    verify_step_tool_contract,
)
from x_editor.runtime.verification import (
    EvidenceTimebaseBinding,
    RuntimeRepairAction,
    RuntimeRepairPlan,
    RuntimeVerificationRegistry,
    RuntimeVerificationReport,
    StepVerificationContext,
    VerificationArtifact,
    build_default_verification_registry,
    build_runtime_repair_plan,
)
from x_editor.runtime.artifact_timeline import evidence_timeline, crosses_unreviewed_content_transition
from x_editor.tools.executor import ToolExecutor
from x_editor.tools.loader import load_all_capabilities, load_manual
from x_editor.tools.mcp.schemas import ToolResult
from x_editor.workflow_control.argument_repair import (
    ArgumentRepairAdmissionGate,
    argument_repair_evidence_sources,
    deterministic_argument_repair_candidate,
    proposal_from_executor_call,
)
from x_editor.workflow_control.policy import (
    argument_repair_max_attempts,
    repair_action_is_enforced,
)
from x_editor.workflow_control.repair_controller import RepairController
from x_editor.workflow_control.schemas import SuffixResumeReuseReceipt

OperationHandler = Callable[
    [AudioEditStep, dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]
]


_AUDIO_ARTIFACT_EXTENSIONS = {
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
_PRIMARY_OUTPUT_PATH_KEYS = {
    "output_audio_path",
    "target_audio_path",
    "output_path",
    "generated_audio_path",
}
_RUNTIME_ARTIFACT_COMMIT_KEY = "runtime_artifact_commit"
_EVIDENCE_BOUNDARY_TOLERANCE_SEC = 0.05
_VOICE_DESIGN_MEMORY_PATH = Path(__file__).with_name("voice_design_memory.yaml")
_REFERENCE_AUDIO_PATH_KEYS = {
    "input_audio",
    "input_audio_path",
    "reference_audio",
    "reference_audio_path",
    "reference_audio_paths",
    "source_audio_path",
    "target_speaker_reference_path",
}


@dataclass
class AudioOperationSpec:
    """Planner-facing operation metadata backed by an existing X-Editor tool."""

    operation: str
    description: str
    tool_id: str | None = None
    tool_operation: str | None = None
    required_inputs: list[str] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)
    default_params: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "description": self.description,
            "tool_id": self.tool_id,
            "tool_operation": self.tool_operation,
            "required_inputs": self.required_inputs,
            "required_params": self.required_params,
            "default_params": self.default_params,
        }


class AudioOperationRegistry:
    """Map planner operation ids to executable adapters."""

    def __init__(self) -> None:
        self._handlers: dict[str, OperationHandler] = {}
        self._specs: dict[str, AudioOperationSpec] = {}

    def register(
        self,
        operation: str,
        handler: OperationHandler,
        *,
        description: str = "",
        tool_id: str | None = None,
        tool_operation: str | None = None,
        required_inputs: list[str] | None = None,
        required_params: list[str] | None = None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self._handlers[operation] = handler
        self._specs[operation] = AudioOperationSpec(
            operation=operation,
            description=description,
            tool_id=tool_id,
            tool_operation=tool_operation,
            required_inputs=required_inputs or [],
            required_params=required_params or [],
            default_params=default_params or {},
        )

    def get(self, operation: str) -> OperationHandler:
        if operation not in self._handlers:
            raise KeyError(f"Unsupported audio operation: {operation}")
        return self._handlers[operation]

    def list_operations(self) -> list[str]:
        return sorted(self._handlers)

    def restrict(self, operations: set[str]) -> None:
        self._handlers = {name: handler for name, handler in self._handlers.items() if name in operations}
        self._specs = {name: spec for name, spec in self._specs.items() if name in operations}

    def describe_operations(self) -> list[dict[str, Any]]:
        return enrich_operation_descriptions(
            [self._specs[key].model_dump() for key in sorted(self._specs)]
        )

    def describe_operation(self, operation: str) -> dict[str, Any]:
        if operation not in self._specs:
            raise KeyError(f"Unsupported audio operation: {operation}")
        return enrich_operation_descriptions([self._specs[operation].model_dump()])[0]


class AudioEditExecutor:
    """Execute ``AudioEditPlan`` steps through existing X-Editor tools/runtime."""

    def __init__(
        self,
        *,
        registry: AudioOperationRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        output_dir: str | Path | None = None,
        gemini_executor: GeminiExecutorToolCallPlanner | None = None,
        executor_tool_call_planner: ExecutorToolCallPlanner | None = None,
        enable_gemini_executor: bool | None = None,
        agentic_mode: str | None = None,
        verification_registry: RuntimeVerificationRegistry | None = None,
        perception_frontends: dict[str, Any] | None = None,
        speech_perception_client: SpeechPerceptionClient | None = None,
    ) -> None:
        self.registry = registry or AudioOperationRegistry()
        self.tool_executor = tool_executor or ToolExecutor()
        self._owns_tool_executor = tool_executor is None
        self.output_dir = Path(output_dir or AudioEditConfig().temp_dir / "planner_executor")
        self.agentic_mode = resolve_agentic_mode(agentic_mode)
        self.perception_frontends = dict(perception_frontends or {})
        self.speech_perception_client = speech_perception_client
        self.verification_registry = verification_registry or build_default_verification_registry()
        self.enable_gemini_executor = (
            _env_bool("X_EDITOR_ENABLE_GEMINI_EXECUTOR", default=False)
            if enable_gemini_executor is None
            else enable_gemini_executor
        )
        self._gemini_executor_provided = (
            gemini_executor is not None or executor_tool_call_planner is not None
        )
        self._gemini_executor_timeout: float | None = None
        self.executor_tool_call_planner = executor_tool_call_planner or gemini_executor
        self.gemini_executor = self.executor_tool_call_planner
        self.pre_gemini_rewrite_rules = default_pre_gemini_rewrite_rules()
        self.post_gemini_rewrite_rules = default_post_gemini_rewrite_rules()
        if self.enable_gemini_executor and self.executor_tool_call_planner is None:
            self._gemini_executor_timeout = _env_float("X_EDITOR_GEMINI_EXECUTOR_TIMEOUT", 60.0)
            self.executor_tool_call_planner = self._build_executor_tool_call_planner(
                timeout=self._gemini_executor_timeout
            )
            self.gemini_executor = self.executor_tool_call_planner
        if registry is None:
            self._register_default_operations()
        from x_editor.tools.selection import current_tool_selection

        selection = current_tool_selection()
        if selection is not None:
            self.registry.restrict(selection.operations())

    def _configure_gemini_executor_for_context(self, context: dict[str, Any]) -> None:
        if self._gemini_executor_provided:
            return
        self.agentic_mode = resolve_agentic_mode(context.get("agentic_mode") or self.agentic_mode)
        use_executor_planner = _context_bool(
            context,
            "use_executor_tool_call_planner",
            default=_context_bool(
                context, "use_gemini_executor", default=self.enable_gemini_executor
            ),
        )
        if not use_executor_planner:
            return
        timeout = _safe_float(context.get("gemini_executor_timeout"))
        if timeout is None:
            timeout = _env_float("X_EDITOR_GEMINI_EXECUTOR_TIMEOUT", 60.0)
        timeout = max(1.0, float(timeout))
        if (
            self.executor_tool_call_planner is not None
            and self._gemini_executor_timeout == timeout
            and _executor_planner_matches_mode(self.executor_tool_call_planner, self.agentic_mode)
        ):
            return
        self._gemini_executor_timeout = timeout
        self.executor_tool_call_planner = self._build_executor_tool_call_planner(timeout=timeout)
        self.gemini_executor = self.executor_tool_call_planner

    def _build_executor_tool_call_planner(self, *, timeout: float) -> ExecutorToolCallPlanner:
        if self.agentic_mode == OPEN_GLM_QWEN_OMNI_MODE:
            return GLMExecutorToolCallPlanner(timeout=timeout)
        return GeminiExecutorToolCallPlanner(timeout=timeout)

    def execute(
        self,
        plan: AudioEditPlan,
        context: dict[str, Any] | None = None,
        *,
        observer: ExecutionEventObserver | None = None,
    ) -> AudioEditPlan:
        """Synchronous wrapper for callers that are not already in an event loop."""

        return asyncio.run(self.aexecute(plan, context, observer=observer))

    async def aexecute(
        self,
        plan: AudioEditPlan,
        context: dict[str, Any] | None = None,
        *,
        observer: ExecutionEventObserver | None = None,
    ) -> AudioEditPlan:
        """Execute a plan and return the same plan shape with step statuses filled."""

        context = dict(context or {})
        plan = plan.model_copy(deep=True)
        self.output_dir = Path(
            context.get("output_dir") or plan.global_params.get("output_dir") or self.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_gemini_executor_for_context(context)
        execution_graph, execution_graph_error = _authoritative_execution_graph(
            plan,
            context,
        )
        if execution_graph_error is not None:
            return _fail_entire_plan(plan, execution_graph_error)

        state = {
            "plan": plan,
            "context": context,
            "output_dir": str(self.output_dir),
            "input_audio": context.get("input_audio") or plan.input_audio,
            "input_audios": context.get("input_audios") or context.get("input_audio_paths") or [],
            "output_audio": context.get("output_audio") or plan.output_audio,
            "current_audio_path": context.get("input_audio") or plan.input_audio,
            "source_text_audio_path": context.get("source_text_audio_path")
            or plan.global_params.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index")
            or plan.global_params.get("source_text_audio_index"),
            "artifacts": {},
            "audio_artifacts": [],
            "audio_artifact_seen": set(),
            "step_outputs": {},
            "step_text_outputs": {},
            "user_prompt": plan.request,
            "source_text": context.get("source_text") or plan.global_params.get("source_text"),
            "perception_report": context.get("perception_report")
            or plan.global_params.get("perception_report")
            or {},
            "perception_reports_by_audio_path": {},
            "reviewed_perception_bindings": [],
            "reference_audio_paths": context.get("reference_audio_paths")
            or plan.global_params.get("reference_audio_paths")
            or [],
            "reference_audio_items": context.get("reference_audio_items")
            or plan.global_params.get("reference_audio_items")
            or [],
            "reference_audio_aliases": context.get("reference_audio_aliases")
            or plan.global_params.get("reference_audio_aliases")
            or {},
            "edit_session_audio_aliases": context.get("edit_session_audio_aliases")
            or plan.global_params.get("edit_session_audio_aliases")
            or {},
            "edit_session_graph_context": context.get("edit_session_graph_context")
            or plan.global_params.get("edit_session_graph_context")
            or {},
            "executor_tool_calls": {},
            "executor_tool_call_errors": {},
            "executor_gemini_call_metrics": {},
            "executor_llm_call_metrics": {},
            "operation_normalization": {},
            "operation_normalization_errors": {},
            "operation_contract_failures": {},
            "text_edit_contract_failures": {},
            "flexsed_window_repairs": {},
            "tool_contract_violations": [],
            "tool_contract_repair_attempts": [],
            "last_tool_contract_error": None,
            "executor_loops": [],
            "executor_step_records": [],
            "tool_invocations": [],
            "last_executor_normalization_error": None,
            "verification_reports": [],
            "verification_repair_plans": [],
            "bounded_repair_attempts": [],
            "execution_observer": observer,
            "execution_observer_errors": [],
            "execution_graph": execution_graph,
            "runtime_artifact_revisions": {},
            "current_runtime_artifact_id": None,
            "perception_evidence_timebase": None,
            "verification_contexts": [],
            "rejected_step_artifacts": [],
            "timeline_span_rebindings": [],
            "last_executor_verification_error": None,
            "service_bootstrap_report_start": len(
                getattr(self.tool_executor, "bootstrap_reports", [])
            ),
        }
        if opaque_audio_resolution_enabled(state):
            _materialize_opaque_audio_state(state)
        _initialize_runtime_artifact_authority(state, execution_graph)
        _bind_initial_perception_evidence_timebase(state)
        _index_namespaced_reviewed_perception(state)
        if state["input_audio"]:
            input_path = Path(str(state["input_audio"]))
            if not input_path.exists():
                return _fail_entire_plan(plan, f"Input audio path does not exist: {input_path}")
            state["artifacts"]["input_audio"] = str(input_path)
            state["artifacts"]["base"] = str(input_path)
            _record_audio_artifact(
                state,
                artifact_id="input_audio",
                path=str(input_path),
                role="input_audio",
                kind="input",
                source="context.input_audio",
                description="Original input audio.",
            )
        if not state["input_audios"]:
            state["input_audios"] = collect_input_audios(state)
        for index, audio_path in enumerate(collect_input_audios(state)):
            state["artifacts"].setdefault(f"input:{index}", str(audio_path))
            state["artifacts"].setdefault(f"audio{index + 1}", str(audio_path))
            _record_audio_artifact(
                state,
                artifact_id=f"input:{index}",
                path=str(audio_path),
                role=f"input_audio_{index + 1}",
                kind="input",
                source=f"context.input_audios[{index}]",
                description="Additional input audio.",
            )

        _apply_suffix_resume_reuse(plan, state)

        try:
            if plan.metadata.get("command_program"):
                from x_editor.runtime.command_execution import execute_commands
                if execution_graph is not None:
                    execution_graph.status = "running"
                await execute_commands(plan, state, self._execute_one_step,
                    (lambda step: _sync_execution_graph_step(execution_graph, step))
                    if execution_graph is not None else None)
            elif execution_graph is not None:
                execution_graph.status = "running"
                await self._execute_steps(
                    plan,
                    state,
                    execution_graph=execution_graph,
                )
            elif _operation_graph_execution_enabled(state):
                await self._execute_steps_with_operation_graph(plan, state)
            else:
                await self._execute_steps(plan, state)
        finally:
            if self._owns_tool_executor:
                await self.tool_executor.shutdown()
        if execution_graph is not None:
            _finalize_authoritative_execution_graph(execution_graph, plan)
            plan.metadata["execution_graph"] = execution_graph.model_dump(mode="json")
        if state["executor_tool_calls"]:
            plan.metadata["executor_tool_calls"] = state["executor_tool_calls"]
        if state["executor_tool_call_errors"]:
            plan.metadata["executor_tool_call_errors"] = state["executor_tool_call_errors"]
        if state["executor_gemini_call_metrics"]:
            plan.metadata["executor_gemini_call_metrics"] = state["executor_gemini_call_metrics"]
        if state["executor_llm_call_metrics"]:
            plan.metadata["executor_llm_call_metrics"] = state["executor_llm_call_metrics"]
        if state.get("executor_tool_call_repairs"):
            plan.metadata["executor_tool_call_repairs"] = state["executor_tool_call_repairs"]
        if state.get("executor_reference_repairs"):
            plan.metadata["executor_reference_repairs"] = state["executor_reference_repairs"]
        if state["operation_normalization"]:
            plan.metadata["operation_normalization"] = state["operation_normalization"]
        if state["operation_normalization_errors"]:
            plan.metadata["operation_normalization_errors"] = state[
                "operation_normalization_errors"
            ]
        if state.get("operation_contract_failures"):
            plan.metadata["operation_contract_failures"] = state["operation_contract_failures"]
        if state.get("text_edit_contract_failures"):
            plan.metadata["text_edit_contract_failures"] = state["text_edit_contract_failures"]
        if state.get("need_tool_contract_replan"):
            plan.metadata["need_tool_contract_replan"] = True
        if state.get("tool_contract_violations"):
            plan.metadata["tool_contract_violations"] = state["tool_contract_violations"]
            plan.metadata["need_tool_contract_replan"] = bool(
                plan.metadata.get("need_tool_contract_replan")
            ) or any(
                item.get("severity") == "blocking"
                for item in state["tool_contract_violations"]
                if isinstance(item, dict)
            )
        if state.get("tool_contract_repair_attempts"):
            plan.metadata["tool_contract_repair_attempts"] = state["tool_contract_repair_attempts"]
        if state.get("last_tool_contract_error"):
            plan.metadata["last_tool_contract_error"] = state["last_tool_contract_error"]
        if state.get("operation_rewrites"):
            plan.metadata["operation_rewrites"] = state["operation_rewrites"]
        if state.get("runtime_rewrite_rule_hits"):
            plan.metadata["runtime_rewrite_rule_hits"] = state["runtime_rewrite_rule_hits"]
        if state.get("artifact_ops_input_repairs"):
            plan.metadata["artifact_ops_input_repairs"] = state["artifact_ops_input_repairs"]
        if state.get("artifact_ops_input_bindings"):
            plan.metadata["artifact_ops_input_bindings"] = state["artifact_ops_input_bindings"]
        if state.get("artifact_ops_reconstruct_repairs"):
            plan.metadata["artifact_ops_reconstruct_repairs"] = state[
                "artifact_ops_reconstruct_repairs"
            ]
        if state.get("artifact_ops_reconstruct_normalizations"):
            plan.metadata["artifact_ops_reconstruct_normalizations"] = state[
                "artifact_ops_reconstruct_normalizations"
            ]
        if state.get("artifact_ops_followup_span_reuse"):
            plan.metadata["artifact_ops_followup_span_reuse"] = state[
                "artifact_ops_followup_span_reuse"
            ]
        if state.get("span_selection_param_filters"):
            plan.metadata["span_selection_param_filters"] = state["span_selection_param_filters"]
        if state.get("step_result_reference_resolutions"):
            plan.metadata["step_result_reference_resolutions"] = state[
                "step_result_reference_resolutions"
            ]
        if state.get("perception_audio_repairs"):
            plan.metadata["perception_audio_repairs"] = state["perception_audio_repairs"]
        if state.get("reviewed_perception_bindings"):
            plan.metadata["reviewed_perception_bindings"] = state["reviewed_perception_bindings"]
        if state.get("flexsed_window_repairs"):
            plan.metadata["flexsed_window_repairs"] = state["flexsed_window_repairs"]
        if state.get("source_text_repairs"):
            plan.metadata["source_text_repairs"] = state["source_text_repairs"]
        if state.get("emphasis_stress_perception_repairs"):
            plan.metadata["emphasis_stress_perception_repairs"] = state[
                "emphasis_stress_perception_repairs"
            ]
        if state.get("emphasis_stress_fallbacks"):
            plan.metadata["emphasis_stress_fallbacks"] = state["emphasis_stress_fallbacks"]
        if state.get("reference_audio_extraction_grounding"):
            plan.metadata["reference_audio_extraction_grounding"] = state[
                "reference_audio_extraction_grounding"
            ]
        if state.get("artifact_event_span_grounding"):
            plan.metadata["artifact_event_span_grounding"] = state["artifact_event_span_grounding"]
        if state.get("perception_requirement_relaxations"):
            plan.metadata["perception_requirement_relaxations"] = state[
                "perception_requirement_relaxations"
            ]
        if state.get("executor_loops"):
            plan.metadata["executor_loops"] = state["executor_loops"]
        if state.get("executor_step_records"):
            plan.metadata["executor_step_records"] = state["executor_step_records"]
        if state.get("operation_graph_execution"):
            plan.metadata["operation_graph_execution"] = state["operation_graph_execution"]
        if state.get("text_state_updates"):
            plan.metadata["text_state_updates"] = state["text_state_updates"]
        if state.get("tool_invocations"):
            plan.metadata["tool_invocations"] = state["tool_invocations"]
        if state.get("verification_reports"):
            plan.metadata["verification_reports"] = state["verification_reports"]
        if state.get("verification_contexts"):
            plan.metadata["verification_contexts"] = state["verification_contexts"]
        if state.get("rejected_step_artifacts"):
            plan.metadata["rejected_step_artifacts"] = state["rejected_step_artifacts"]
        if state.get("timeline_span_rebindings"):
            plan.metadata["timeline_span_rebindings"] = state["timeline_span_rebindings"]
        if state.get("suffix_resume_reuse_receipts"):
            plan.metadata["suffix_resume_reuse_receipts"] = state["suffix_resume_reuse_receipts"]
            plan.metadata["suffix_resume_reused_step_ids"] = state["suffix_resume_reused_step_ids"]
            plan.metadata["suffix_resume_skipped_tool_call_count"] = sum(
                int(item.get("skipped_tool_call_count") or 0)
                for item in state["suffix_resume_reuse_receipts"]
                if isinstance(item, dict)
            )
        if state.get("suffix_resume_reuse_error"):
            plan.metadata["suffix_resume_reuse_error"] = state["suffix_resume_reuse_error"]
        if state.get("verification_repair_plans"):
            plan.metadata["verification_repair_plans"] = state["verification_repair_plans"]
        if state.get("bounded_repair_attempts"):
            plan.metadata["bounded_repair_attempts"] = state["bounded_repair_attempts"]
        if state.get("execution_observer_errors"):
            plan.metadata["execution_observer_errors"] = state["execution_observer_errors"]
        if state.get("last_executor_verification_error"):
            plan.metadata["need_repair"] = True
            plan.metadata["last_executor_verification_error"] = state[
                "last_executor_verification_error"
            ]
        bootstrap_reports = _tool_bootstrap_reports(
            self.tool_executor,
            int(state.get("service_bootstrap_report_start") or 0),
        )
        if bootstrap_reports:
            plan.metadata["service_bootstrap"] = bootstrap_reports
        declared_outputs, declared_output_errors = _declared_output_artifacts(plan, state)
        if plan.output_artifacts:
            plan.metadata["output_artifacts"] = declared_outputs
            if declared_output_errors:
                plan.metadata["output_artifact_errors"] = declared_output_errors
                plan.metadata["need_repair"] = True
                final_output_path = None
            else:
                final_output_path = next(
                    item["path"] for item in declared_outputs if item["primary"]
                )
            for item in declared_outputs:
                _mark_final_audio_artifacts(state, item["path"])
        else:
            final_output_path = _final_output_path(plan, state)
            _mark_final_audio_artifacts(state, final_output_path)
        plan.metadata["final_output_path"] = final_output_path
        audio_artifacts = _public_audio_artifacts(state)
        if audio_artifacts:
            intermediate_audio_artifacts = [
                item for item in audio_artifacts if item.get("kind") == "step_output"
            ]
            plan.metadata["audio_artifacts"] = audio_artifacts
            plan.metadata["debug_audio_artifacts"] = intermediate_audio_artifacts
            plan.metadata["intermediate_audio_artifacts"] = intermediate_audio_artifacts
            plan.metadata["intermediate_audios"] = intermediate_audio_artifacts
        _attach_operation_graph_metadata(plan, state)
        plan.metadata["execution_summary"] = execution_summary(plan)
        return plan

    async def _execute_steps(
        self,
        plan: AudioEditPlan,
        state: dict[str, Any],
        *,
        execution_graph: ExecutionGraph | None = None,
    ) -> None:
        by_id = {step.id: step for step in plan.steps}
        graph_nodes_by_step = (
            {node.plan_step_id: node for node in execution_graph.nodes}
            if execution_graph is not None
            else {}
        )
        ordered_steps = (
            [
                by_id[next(node.plan_step_id for node in execution_graph.nodes if node.node_id == node_id)]
                for node_id in execution_graph.topological_order
            ]
            if execution_graph is not None
            else plan.steps
        )
        reused_step_ids = set(state.get("suffix_resume_reused_step_ids") or [])
        for step_id in reused_step_ids:
            if execution_graph is not None and step_id in by_id:
                _sync_execution_graph_step(execution_graph, by_id[step_id])
        pending = {step.id for step in plan.steps if step.id not in reused_step_ids}
        stop_after_failure = False
        loop_index = 0

        while pending:
            loop_index += 1
            loop_record = {
                "loop_index": loop_index,
                "pending_before": sorted(pending),
                "executed_steps": [],
                "skipped_steps": [],
                "failed_steps": [],
            }
            state.setdefault("executor_loops", []).append(loop_record)
            progressed = False
            for step in ordered_steps:
                if step.id not in pending:
                    continue
                dependencies = (
                    graph_nodes_by_step[step.id].depends_on
                    if execution_graph is not None
                    else step.depends_on
                )
                missing_deps = [dep for dep in dependencies if dep not in by_id]
                if missing_deps:
                    _mark_failed(step, f"Missing dependency step(s): {', '.join(missing_deps)}")
                    if execution_graph is not None:
                        _sync_execution_graph_step(execution_graph, step)
                    loop_record["failed_steps"].append(step.id)
                    pending.remove(step.id)
                    stop_after_failure = not _continue_on_error(step, state)
                    progressed = True
                    continue
                if any(dep in pending for dep in dependencies):
                    continue
                failed_deps = [dep for dep in dependencies if by_id[dep].status != "success"]
                if failed_deps:
                    _mark_skipped(step, f"Dependency failed or skipped: {', '.join(failed_deps)}")
                    if execution_graph is not None:
                        _sync_execution_graph_step(execution_graph, step)
                    loop_record["skipped_steps"].append(step.id)
                    pending.remove(step.id)
                    progressed = True
                    continue
                if stop_after_failure:
                    _mark_skipped(step, "Skipped after a previous required step failed")
                    if execution_graph is not None:
                        _sync_execution_graph_step(execution_graph, step)
                    loop_record["skipped_steps"].append(step.id)
                    pending.remove(step.id)
                    progressed = True
                    continue
                await self._execute_one_step(step, state)
                if execution_graph is not None:
                    _sync_execution_graph_step(execution_graph, step)
                loop_record["executed_steps"].append(step.id)
                if step.status == "failed":
                    loop_record["failed_steps"].append(step.id)
                elif step.status == "skipped":
                    loop_record["skipped_steps"].append(step.id)
                pending.remove(step.id)
                if step.status == "failed" and not _continue_on_error(step, state):
                    stop_after_failure = True
                progressed = True
            if not progressed:
                for step_id in list(pending):
                    _mark_failed(by_id[step_id], "Cyclic or unsatisfied step dependencies")
                    if execution_graph is not None:
                        _sync_execution_graph_step(execution_graph, by_id[step_id])
                    loop_record["failed_steps"].append(step_id)
                    pending.remove(step_id)
            loop_record["pending_after"] = sorted(pending)

    async def _execute_steps_with_operation_graph(
        self, plan: AudioEditPlan, state: dict[str, Any]
    ) -> None:
        execution_record: dict[str, Any] = {
            "enabled": True,
            "scheduler": "operation_graph_v1",
            "fallback_to_original": False,
            "bindings": [],
            "node_path_updates": [],
        }
        state["operation_graph_execution"] = execution_record
        try:
            from x_editor.session.operation_graph import (
                build_planned_operation_graph,
                build_round_operation_graph_store,
                validate_operation_graph,
                validate_round_operation_graph_store,
            )

            planned_graph = build_planned_operation_graph(plan)
            validation = validate_operation_graph(planned_graph, plan=plan)
            planned_round_store = build_round_operation_graph_store(
                planned_graph,
                round_index=_operation_graph_execution_round_index(plan, state),
                graph_kind="planned",
                source="executor.planned_operation_graph",
            )
            round_validation = validate_round_operation_graph_store(planned_round_store)
        except Exception as exc:
            execution_record.update(
                {
                    "fallback_to_original": True,
                    "fallback_reason": f"planned_graph_error:{type(exc).__name__}: {exc}",
                }
            )
            await self._execute_steps(plan, state)
            return

        execution_record["planned_graph"] = {
            "graph_id": planned_graph.graph_id,
            "source": planned_graph.source,
            "node_count": len(planned_graph.nodes),
            "operation_count": len(planned_graph.operations),
        }
        execution_record["planned_graph_validation"] = validation.model_dump(exclude_none=True)
        execution_record["planned_round_store"] = {
            "store_id": planned_round_store.store_id,
            "round_count": len(planned_round_store.rounds),
            "round_indexes": [
                round_record.round_index for round_record in planned_round_store.rounds
            ],
        }
        round_execution_plan = _operation_graph_round_execution_plan(planned_round_store)
        execution_record["round_execution_plan"] = round_execution_plan
        execution_record["planned_round_validation"] = round_validation.model_dump(
            exclude_none=True
        )
        if not validation.valid:
            execution_record.update(
                {
                    "fallback_to_original": True,
                    "fallback_reason": "planned_graph_validation_failed",
                }
            )
            await self._execute_steps(plan, state)
            return
        if not round_validation.valid:
            execution_record.update(
                {
                    "fallback_to_original": True,
                    "fallback_reason": "planned_round_graph_validation_failed",
                }
            )
            await self._execute_steps(plan, state)
            return

        by_id = {step.id: step for step in plan.steps}
        reused_step_ids = set(state.get("suffix_resume_reused_step_ids") or [])
        pending = {step.id for step in plan.steps if step.id not in reused_step_ids}
        step_order = _operation_graph_step_order(planned_graph, plan)
        if not step_order:
            execution_record.update(
                {
                    "fallback_to_original": True,
                    "fallback_reason": "planned_graph_has_no_executable_steps",
                }
            )
            await self._execute_steps(plan, state)
            return

        step_ops = _operation_graph_operations_by_step(planned_graph)
        graph_dependencies = _operation_graph_step_dependencies(planned_graph, by_id)
        round_dependencies = _operation_graph_round_step_dependencies(round_execution_plan, by_id)
        scheduler_dependencies = _operation_graph_merge_step_dependencies(
            graph_dependencies, round_dependencies
        )
        execution_record["scheduler_dependencies"] = scheduler_dependencies
        node_paths = _operation_graph_initial_node_paths(planned_graph, state)
        state["_operation_graph_node_paths"] = node_paths
        for step_id in state.get("suffix_resume_reused_step_ids") or []:
            step = by_id.get(step_id)
            if step is None:
                continue
            updates = _update_operation_graph_node_paths_after_step(
                step,
                planned_graph=planned_graph,
                step_ops=step_ops,
                node_paths=node_paths,
            )
            if updates:
                execution_record["node_path_updates"].extend(updates)
        stop_after_failure = False
        loop_index = 0

        while pending:
            loop_index += 1
            loop_record = {
                "loop_index": loop_index,
                "scheduler": "operation_graph_v1",
                "pending_before": sorted(pending),
                "ready_steps": [],
                "executed_steps": [],
                "skipped_steps": [],
                "failed_steps": [],
            }
            state.setdefault("executor_loops", []).append(loop_record)
            progressed = False
            ready_steps: list[AudioEditStep] = []

            for step_id in step_order:
                if step_id not in pending:
                    continue
                step = by_id[step_id]
                required_deps = _operation_graph_required_step_dependencies(
                    step, scheduler_dependencies
                )
                missing_deps = [dep for dep in required_deps if dep not in by_id]
                if missing_deps:
                    _mark_failed(step, f"Missing dependency step(s): {', '.join(missing_deps)}")
                    loop_record["failed_steps"].append(step.id)
                    pending.remove(step.id)
                    stop_after_failure = not _continue_on_error(step, state)
                    progressed = True
                    continue
                if any(dep in pending for dep in required_deps):
                    continue
                failed_deps = [dep for dep in required_deps if by_id[dep].status != "success"]
                if failed_deps:
                    _mark_skipped(step, f"Dependency failed or skipped: {', '.join(failed_deps)}")
                    loop_record["skipped_steps"].append(step.id)
                    pending.remove(step.id)
                    progressed = True
                    continue
                ready_steps.append(step)

            loop_record["ready_steps"] = [step.id for step in ready_steps]
            if len(ready_steps) > 1:
                loop_record["pseudo_parallel"] = True
                loop_record["pseudo_parallel_group_id"] = f"operation_graph_loop_{loop_index}"

            for step in ready_steps:
                if step.id not in pending:
                    continue
                if stop_after_failure:
                    _mark_skipped(step, "Skipped after a previous required step failed")
                    loop_record["skipped_steps"].append(step.id)
                    pending.remove(step.id)
                    progressed = True
                    continue
                binding = _bind_step_inputs_from_operation_graph(
                    step,
                    state,
                    planned_graph=planned_graph,
                    step_ops=step_ops,
                    node_paths=node_paths,
                )
                if binding:
                    execution_record["bindings"].append(binding)
                await self._execute_one_step(step, state)
                loop_record["executed_steps"].append(step.id)
                if step.status == "failed":
                    loop_record["failed_steps"].append(step.id)
                elif step.status == "skipped":
                    loop_record["skipped_steps"].append(step.id)
                pending.remove(step.id)
                updates = _update_operation_graph_node_paths_after_step(
                    step,
                    planned_graph=planned_graph,
                    step_ops=step_ops,
                    node_paths=node_paths,
                )
                if updates:
                    execution_record["node_path_updates"].extend(updates)
                if step.status == "failed" and not _continue_on_error(step, state):
                    stop_after_failure = True
                progressed = True

            if not progressed:
                for step_id in list(pending):
                    _mark_failed(
                        by_id[step_id],
                        "Cyclic or unsatisfied operation graph dependencies",
                    )
                    loop_record["failed_steps"].append(step_id)
                    pending.remove(step_id)
            loop_record["pending_after"] = sorted(pending)

        execution_record["final_node_paths"] = dict(sorted(node_paths.items()))

    async def _execute_one_step(self, step: AudioEditStep, state: dict[str, Any]) -> None:
        step.status = "running"
        opaque_audio_refs = opaque_audio_resolution_enabled(state)
        canonical_inputs: dict[str, Any] | None = None
        canonical_params: dict[str, Any] | None = None
        record = {
            "step_id": step.id,
            "operation": step.operation,
            "description": step.description,
            "status": "running",
            "current_audio_before": _public_runtime_audio_ref(
                state.get("current_audio_path"),
                state,
            ),
            "inputs_before": dict(step.inputs),
            "params_before": dict(step.params),
        }
        record_appended = False
        state["_current_step_id"] = step.id
        _emit_execution_observation(
            state,
            stage="executor",
            kind="step_execution_started",
            status="started",
            step=step,
            payload={"step": step.model_dump(mode="python")},
        )
        try:
            preserve_route_identity = _context_bool(
                state["context"],
                "preserve_explicit_operation_selection",
                default=False,
            )
            if not preserve_route_identity:
                apply_runtime_rewrite_rules(
                    executor=self,
                    step=step,
                    state=state,
                    rules=self.pre_gemini_rewrite_rules,
                )
            if not _argument_repair_enforced(state):
                self._maybe_normalize_step_with_gemini(step, state)
            if not preserve_route_identity:
                apply_runtime_rewrite_rules(
                    executor=self,
                    step=step,
                    state=state,
                    rules=self.post_gemini_rewrite_rules,
                )
            if opaque_audio_refs:
                canonical_inputs = copy.deepcopy(step.inputs)
                canonical_params = copy.deepcopy(step.params)
                _resolve_step_audio_aliases(step, state)
                _resolve_step_result_references_in_step(step, state)
            else:
                _resolve_step_result_references_in_step(step, state)
                _resolve_step_audio_aliases(step, state)
            if opaque_audio_refs:
                register_runtime_audio_paths(state, step.inputs)
                register_runtime_audio_paths(state, step.params)
                state["_runtime_audio_boundary_materialized"] = True
            perception_binding = _activate_reviewed_perception_for_step(step, state)
            if perception_binding is not None:
                state["reviewed_perception_bindings"].append(perception_binding)
            _rebind_explicit_step_spans_to_current_artifact(step, state)
            self._normalize_step_before_handler(step, state)
            normalization_audit = state["operation_normalization"].get(step.id)
            if isinstance(normalization_audit, dict):
                record["normalization_audit"] = copy.deepcopy(normalization_audit)
            if opaque_audio_refs:
                validate_materialized_audio_reference_roles(step.inputs, state)
                validate_materialized_audio_reference_roles(step.params, state)
                register_runtime_audio_paths(state, step.inputs)
                register_runtime_audio_paths(state, step.params)
                normalization = state["operation_normalization"].get(step.id)
                if isinstance(normalization, dict):
                    restored_normalization = dict(normalization)
                    for audit_key, canonical_mapping in (
                        ("inputs_before", canonical_inputs or {}),
                        ("params_before", canonical_params or {}),
                        ("inputs", canonical_inputs or {}),
                        ("params", canonical_params or {}),
                    ):
                        restored_normalization[audit_key] = restore_canonical_audio_references(
                            dict(normalization.get(audit_key) or {}),
                            canonical_mapping,
                        )
                    state["operation_normalization"][step.id] = restored_normalization
                    record["normalization_audit"] = copy.deepcopy(restored_normalization)
            if opaque_audio_refs:
                record["inputs_after_normalization"] = copy.deepcopy(canonical_inputs)
                record["params_after_normalization"] = copy.deepcopy(canonical_params)
            else:
                record["inputs_after_normalization"] = dict(step.inputs)
                record["params_after_normalization"] = dict(step.params)
            self._verify_tool_contract_before_handler(step, state)
            handler = self.registry.get(step.operation)
            record["handler"] = getattr(handler, "__name__", str(handler))
            output = await self._run_step_handler(step, state)
            if opaque_audio_refs and step.operation.startswith(("perception.", "flexsed.")):
                output = canonicalize_perception_audio_bindings(output, state)
            output = _review_planned_perception_output(step, state, output)
            step.status = "success"
            output_path = _extract_output_path(output)
            record["output_path"] = (
                f"{step.id}:output_audio_path" if opaque_audio_refs and output_path else output_path
            )
            record["result_keys"] = sorted(output.keys()) if isinstance(output, dict) else []
            consumed_audio_paths = collect_consumed_audio_paths(
                step=step,
                state=state,
                output=output,
            )
            record["consumed_audio_paths_observed"] = True
            if opaque_audio_refs:
                record["consumed_audio_refs"] = [
                    audio_ref
                    for audio_ref in (
                        canonical_audio_ref_for_runtime_path(path, state)
                        for path in consumed_audio_paths
                    )
                    if audio_ref is not None
                ]
            else:
                record["consumed_audio_paths"] = consumed_audio_paths
            step.result = output
            verification_context = _build_step_verification_context(
                step=step,
                state=state,
                output=output,
                consumed_audio_paths=consumed_audio_paths,
            )
            _stage_step_candidate(step, state, verification_context)
            if opaque_audio_refs:
                register_runtime_audio_paths(state, output_path)
            verification_context = await self._verify_step_after_success(
                step,
                state,
                verification_context=verification_context,
            )
            if step.status == "success":
                _accept_step_candidate(step, state, verification_context)
                self._commit_step_output(step, state, step.result or output)
            else:
                _reject_step_candidate(step, state, verification_context)
            record["status"] = step.status or "success"
            if step.error:
                record["error_message"] = step.error
            record["current_audio_after"] = _public_runtime_audio_ref(
                state.get("current_audio_path"),
                state,
            )
        except Exception as exc:
            if isinstance(exc, ToolContractViolationError):
                _mark_failed(step, f"tool contract violation: {exc}")
                record["status"] = "failed"
                record["error_message"] = step.error
                record["current_audio_after"] = _public_runtime_audio_ref(
                    state.get("current_audio_path"),
                    state,
                )
                state.setdefault("executor_step_records", []).append(_without_empty(record))
                record_appended = True
                state.pop("_current_step_id", None)
                return
            repaired_error = await self._maybe_repair_runtime_text_edit_failure(step, state, exc)
            if repaired_error is None and step.status == "success":
                record["status"] = "success"
                record["repaired_from_error"] = f"{type(exc).__name__}: {exc}"
                record["current_audio_after"] = _public_runtime_audio_ref(
                    state.get("current_audio_path"),
                    state,
                )
                state.setdefault("executor_step_records", []).append(_without_empty(record))
                record_appended = True
                state.pop("_current_step_id", None)
                return
            if repaired_error is not None:
                _mark_failed(step, repaired_error)
                record["status"] = "failed"
                record["error_message"] = repaired_error
                record["current_audio_after"] = _public_runtime_audio_ref(
                    state.get("current_audio_path"),
                    state,
                )
                state.setdefault("executor_step_records", []).append(_without_empty(record))
                record_appended = True
                state.pop("_current_step_id", None)
                return
            _mark_failed(step, f"{type(exc).__name__}: {exc}")
            record["status"] = "failed"
            record["error_message"] = step.error
            record["current_audio_after"] = _public_runtime_audio_ref(
                state.get("current_audio_path"),
                state,
            )
        finally:
            if canonical_inputs is not None and canonical_params is not None:
                step.inputs = restore_canonical_audio_references(
                    step.inputs,
                    canonical_inputs,
                )
                step.params = restore_canonical_audio_references(
                    step.params,
                    canonical_params,
                )
            state.pop("_runtime_audio_boundary_materialized", None)
            if step.status != "running" and not record_appended:
                state["executor_step_records"].append(_without_empty(record))
            _emit_execution_observation(
                state,
                stage="executor",
                kind="step_execution_completed",
                status=_workflow_status_for_step(step),
                step=step,
                payload={
                    "step": step.model_dump(mode="python"),
                    "record": _without_empty(record),
                    "behavior_changed": bool(
                        (record.get("normalization_audit") or {}).get("behavior_changed", False)
                    ),
                },
            )
            verification = (
                step.result.get("verification") if isinstance(step.result, dict) else None
            )
            if verification:
                _emit_execution_observation(
                    state,
                    stage="verifier",
                    kind="step_verification_completed",
                    status=("failed" if step.status == "failed" else "completed"),
                    step=step,
                    payload={
                        "verification": verification,
                        "step_status": step.status,
                        "behavior_changed": False,
                    },
                )
            state.pop("_current_step_id", None)

    async def _run_step_handler(self, step: AudioEditStep, state: dict[str, Any]) -> dict[str, Any]:
        handler = self.registry.get(step.operation)
        invocation_rows = state.get("tool_invocations")
        invocation_start = len(invocation_rows) if isinstance(invocation_rows, list) else 0
        output = handler(step, state)
        if inspect.isawaitable(output):
            output = await output
        output = attach_audio_temporal_transform(
            step=step,
            state=state,
            output=output,
            input_audio_paths=collect_consumed_audio_paths(
                step=step,
                state=state,
                output=output,
            ),
            commit_to_state=False,
        )
        return attach_artifact_delivery_contract(
            step=step,
            state=state,
            output=output,
            handler_name=getattr(handler, "__name__", str(handler)),
            invocation_start=invocation_start,
        )

    def _commit_step_output(
        self, step: AudioEditStep, state: dict[str, Any], output: dict[str, Any]
    ) -> None:
        step.result = output
        timeline_ledger = output.get(AUDIO_TIMELINE_LEDGER_KEY)
        if isinstance(timeline_ledger, dict):
            state[AUDIO_TIMELINE_LEDGER_KEY] = copy.deepcopy(timeline_ledger)
        output_path = _extract_output_path(output)
        if output_path:
            state["current_audio_path"] = output_path
            state["artifacts"][step.id] = output_path
            if step.expected_output:
                state["artifacts"][step.expected_output] = output_path
        state["step_outputs"][step.id] = output
        _record_step_audio_artifacts(step, state, output)
        _set_recorded_step_artifact_status(state, step.id, "accepted")
        _apply_step_state_updates(output, state)
        _apply_step_text_state_updates(step, output, state)

    async def _verify_step_after_success(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        *,
        verification_context: StepVerificationContext | None = None,
    ) -> StepVerificationContext | None:
        if not _runtime_verification_enabled(state):
            return verification_context
        reports: list[RuntimeVerificationReport] = []
        invocation_report = _planned_perception_invocation_report(step, state)
        if invocation_report is not None:
            reports.append(invocation_report)
        reports.extend(
            self.verification_registry.verify_step(
                step,
                step.result,
                state,
                verification_context=verification_context,
            )
        )
        if not reports:
            return verification_context
        failed_reports = [report for report in reports if not report.passed]
        repair_plan = (
            build_runtime_repair_plan(step, failed_reports, state) if failed_reports else None
        )
        initial_repair_plan_payload = (
            repair_plan.to_dict() if repair_plan and repair_plan.should_repair else None
        )
        if repair_plan and repair_plan.should_repair:
            state["verification_repair_plans"].append(initial_repair_plan_payload)
        if repair_plan and repair_plan.micro_actions and _bounded_repair_enabled(state):
            repair_result = await self._try_bounded_micro_repair(step, state, reports, repair_plan)
            if repair_result.get("attempts"):
                step.result = step.result if isinstance(step.result, dict) else {}
                step.result.setdefault("repair_attempts", []).extend(repair_result["attempts"])
                state.setdefault("bounded_repair_attempts", []).extend(repair_result["attempts"])
            if repair_result.get("accepted"):
                if isinstance(
                    repair_result.get("verification_context"), StepVerificationContext
                ):
                    verification_context = repair_result["verification_context"]
                reports = repair_result["reports"]
                failed_reports = [report for report in reports if not report.passed]
                repair_plan = (
                    build_runtime_repair_plan(step, failed_reports, state)
                    if failed_reports
                    else None
                )
        report_payloads = [report.to_dict() for report in reports]
        if isinstance(step.result, dict):
            existing = step.result.get("verification")
            if isinstance(existing, list):
                step.result["verification"] = [*existing, *report_payloads]
            elif existing:
                step.result["verification"] = [existing, *report_payloads]
            else:
                step.result["verification"] = report_payloads
            if repair_plan and repair_plan.should_repair:
                step.result["repair_plan"] = repair_plan.to_dict()
        state["verification_reports"].extend(report_payloads)
        if repair_plan and repair_plan.should_repair:
            final_repair_plan_payload = repair_plan.to_dict()
            if final_repair_plan_payload != initial_repair_plan_payload:
                state["verification_repair_plans"].append(final_repair_plan_payload)
        blocking = [
            report for report in reports if not report.passed and report.severity == "blocking"
        ]
        if not blocking:
            return verification_context
        verification_error = {
            "step_id": step.id,
            "operation": step.operation,
            "reports": [report.to_dict() for report in blocking],
            "repair_plan": repair_plan.to_dict() if repair_plan else None,
            "repair_hints": [hint for report in blocking for hint in report.repair_hints]
            + (
                [
                    action.rationale
                    for action in [
                        *(repair_plan.micro_actions if repair_plan else []),
                        *(repair_plan.major_actions if repair_plan else []),
                    ]
                ]
            ),
        }
        state["last_executor_verification_error"] = verification_error
        state["plan"].metadata["need_repair"] = True
        state["plan"].metadata["last_executor_verification_error"] = verification_error
        if _runtime_verification_strict(state):
            joined = "; ".join(finding for report in blocking for finding in report.findings)
            _mark_failed(step, f"runtime verification failed: {joined or blocking[0].component}")
        return verification_context

    async def _try_bounded_micro_repair(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        reports: list[RuntimeVerificationReport],
        repair_plan: RuntimeRepairPlan,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        accepted_any = False
        accepted_context: StepVerificationContext | None = None
        current_reports = list(reports)
        max_attempts = _bounded_repair_max_micro_actions(state)
        for attempt_index, action in enumerate(repair_plan.micro_actions[:max_attempts]):
            components = set(action.expected_verification or [action.component])
            baseline_reports = _reports_for_components(current_reports, components)
            score_before = _aggregate_verification_score(baseline_reports)
            candidate_step = self._candidate_step_for_micro_action(
                step, state, action, attempt_index
            )
            if candidate_step is None:
                attempts.append(
                    {
                        "action": action.to_dict(),
                        "status": "skipped",
                        "reason": "No deterministic bounded repair executor is registered for this micro action.",
                        "score_before": score_before,
                    }
                )
                continue
            try:
                output = await self._run_step_handler(candidate_step, state)
                candidate_context = _build_step_verification_context(
                    step=candidate_step,
                    state=state,
                    output=output,
                    consumed_audio_paths=collect_consumed_audio_paths(
                        step=candidate_step,
                        state=state,
                        output=output,
                    ),
                )
                candidate_reports = self.verification_registry.verify_step(
                    candidate_step,
                    output,
                    state,
                    components=components,
                    verification_context=candidate_context,
                )
                score_after = _aggregate_verification_score(candidate_reports)
                improved = score_after > score_before + 1e-6
                attempt_record = {
                    "action": action.to_dict(),
                    "status": "accepted" if improved else "rejected",
                    "score_before": score_before,
                    "score_after": score_after,
                    "output_audio_path": _extract_output_path(output),
                    "reports": [report.to_dict() for report in candidate_reports],
                }
                attempts.append(attempt_record)
                if not improved:
                    continue
                accepted_any = True
                step.result = output
                _stage_step_candidate(step, state, candidate_context)
                current_reports = _replace_reports_for_components(
                    current_reports, candidate_reports, components
                )
                accepted_context = candidate_context
                if candidate_reports and all(report.passed for report in candidate_reports):
                    break
            except Exception as exc:
                attempts.append(
                    {
                        "action": action.to_dict(),
                        "status": "failed",
                        "score_before": score_before,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "accepted": accepted_any,
            "attempts": attempts,
            "reports": current_reports,
            "verification_context": (
                accepted_context if accepted_any else None
            ),
        }

    def _candidate_step_for_micro_action(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        action: RuntimeRepairAction,
        attempt_index: int,
    ) -> AudioEditStep | None:
        candidate = step.model_copy(deep=True)
        result = step.result if isinstance(step.result, dict) else {}
        candidate.params = dict(candidate.params)
        candidate.inputs = dict(candidate.inputs)
        input_audio = (
            result.get("input_audio_path")
            or candidate.inputs.get("input_audio_path")
            or candidate.params.get("input_audio_path")
            or state.get("input_audio")
        )
        if input_audio:
            candidate.inputs["input_audio_path"] = str(input_audio)

        strategy = action.strategy
        if strategy == "shift_span_to_phrase_occurrence":
            spans = _repair_shifted_spans(result.get("selected_spans"), action)
            if not spans:
                return None
            candidate.params["selected_spans"] = spans
            candidate.params["output_path"] = _repair_output_path(
                step, state, strategy, attempt_index
            )
            return candidate
        if strategy == "expand_span_boundaries_and_rescore":
            spans = _repair_expanded_spans(result.get("selected_spans"), action, result, state)
            if not spans:
                return None
            candidate.params["selected_spans"] = spans
            candidate.params["output_path"] = _repair_output_path(
                step, state, strategy, attempt_index
            )
            return candidate
        if (
            strategy == "retry_perception_with_refresh_and_language_hints"
            and step.operation.startswith("perception.")
        ):
            candidate.params.update(action.params)
            candidate.params["refresh_cache"] = True
            candidate.params["refresh_perception"] = True
            return candidate
        return None

    def _normalize_step_before_handler(self, step: AudioEditStep, state: dict[str, Any]) -> None:
        from x_editor.runtime.crop_text_binding import CropTextBindingError, bind_crop_text

        before = normalization_snapshot(
            operation=step.operation,
            inputs=copy.deepcopy(step.inputs),
            params=copy.deepcopy(step.params),
        )
        crop_binding = None
        try:
            _resolve_step_audio_aliases(step, state)
            crop_binding = bind_crop_text(step, state)
            if crop_binding is not None:
                # Scope projection is separately proven by the accepted producer
                # receipt; ordinary normalization must still preserve this call.
                before = normalization_snapshot(
                    operation=step.operation, inputs=copy.deepcopy(step.inputs),
                    params=copy.deepcopy(step.params),
                )
            self._maybe_repair_missing_source_text_for_speech_rate(step, state)
            normalize_step_for_operation(step, state)
        except (OperationBoundaryError, CropTextBindingError) as exc:
            state["operation_normalization_errors"][step.id] = str(exc)
            _record_operation_contract_failure(step, state, str(exc))
            state["need_tool_contract_replan"] = True
            if _tool_contract_verification_enabled(state):
                self._record_tool_contract_violation(
                    step,
                    state,
                    build_tool_contract_violation(
                        step,
                        state,
                        operation_spec=self._operation_spec_for_step(step),
                        stage="normalization",
                        reason=str(exc),
                    ),
                )
            raise
        after = normalization_snapshot(
            operation=step.operation,
            inputs=copy.deepcopy(step.inputs),
            params=copy.deepcopy(step.params),
        )
        audit = build_normalization_audit(before, after)
        if not audit.semantics_preserved:
            step.operation = before.operation
            step.inputs = copy.deepcopy(before.inputs)
            step.params = copy.deepcopy(before.params)
            state["operation_normalization"][step.id] = audit.as_dict()
            state["need_tool_contract_replan"] = True
            raise NonEquivalentNormalizationError(
                "executor normalization boundary would change Planner semantics; replan required"
            )
        state["operation_normalization"][step.id] = audit.as_dict()
        if crop_binding is not None:
            state["operation_normalization"][step.id]["crop_text_binding"] = crop_binding

    def _maybe_repair_missing_source_text_for_speech_rate(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
    ) -> None:
        operation = step.operation.removeprefix("edit_data_pipeline.").replace("-", "_")
        if operation != "speech_rate":
            return
        method = str(step.params.get("method") or "wsola+indextts2").strip().lower()
        if method == "wsola":
            return
        if step.params.get("reference_pacing_policy") == "duration_ratio_approximation":
            return
        if _non_empty_text(
            step.params.get("text") or step.params.get("source_text") or state.get("source_text")
        ):
            return
        if not _implicit_executor_repair_enabled(state):
            raise OperationBoundaryError(
                "edit_data_pipeline.speech_rate requires source_text; strict forward "
                "execution forbids implicit perception repair"
            )

        try:
            input_audio = _resolve_input_audio(step, state)
        except Exception as exc:
            raise OperationBoundaryError(
                "edit_data_pipeline.speech_rate requires source_text for the quality "
                f"wsola+indextts2 path, but input audio could not be resolved for ASR repair: {exc}"
            ) from exc

        alternate_audio_paths = _alternate_source_text_audio_paths(input_audio, state)
        language_hints = _source_text_repair_language_hints(step, state)
        repair_record: dict[str, Any] = {
            "strategy": "perception_source_text_before_speech_rate",
            "step_id": step.id,
            "operation": step.operation,
            "input_audio_path": input_audio,
            "alternate_audio_paths": alternate_audio_paths,
            "language_hints": language_hints,
            "reason": (
                "speech_rate quality path requires transcript text; run Qwen3ASR/specialized "
                "speech perception first, with the configured audio fallback provider if specialized "
                "perception is insufficient."
            ),
            "agentic_mode": self.agentic_mode,
        }
        state.setdefault("source_text_repairs", {})[step.id] = repair_record

        try:
            result = build_audio_perception_fallback(
                agentic_mode=self.agentic_mode,
                speech_client=self.speech_perception_client,
            ).perceive(
                user_prompt=str(
                    step.params.get("instruction")
                    or step.description
                    or state.get("user_prompt")
                    or "Recover source text before speech-rate editing."
                ),
                audio_path=input_audio,
                base_perception=state.get("perception_report")
                if _perception_matches_audio(input_audio, state)
                else {},
                source_text=None,
                reference_audio_paths=list(state.get("reference_audio_paths") or []),
                alternate_audio_paths=alternate_audio_paths,
                output_dir=str(Path(state["output_dir"]) / f"{step.id}_source_text_repair"),
                requirements=PerceptionRequirements(source_text=True),
                language_hints=language_hints,
                run_specialized=True,
                allow_model_fallback=True,
                force_model_fallback=False,
                refresh_cache=True,
                strict=True,
            )
        except Exception as exc:
            repair_record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise OperationBoundaryError(
                "edit_data_pipeline.speech_rate requires source_text for wsola+indextts2; "
                "automatic ASR/model perception repair failed. Do not downgrade to pure wsola "
                f"unless the planner explicitly marks a non-resynthesis approximation. Error: {exc}"
            ) from exc

        perception = result.perception
        source_text = str(perception.get("source_text") or "").strip()
        repair_record.update(
            {
                "status": "completed" if source_text else "insufficient",
                "satisfied": result.satisfied,
                "missing": result.missing,
                "attempts": result.attempts,
                "source_text_length": len(source_text),
                "source_text_preview": source_text[:160],
                "source_text_audio_path": perception.get("source_text_audio_path"),
                "source_text_audio_index": perception.get("source_text_audio_index"),
            }
        )
        if not source_text:
            raise OperationBoundaryError(
                "edit_data_pipeline.speech_rate requires source_text for wsola+indextts2; "
                "ASR/model perception repair completed but did not produce transcript text."
            )

        state["source_text"] = source_text
        if perception.get("source_text_audio_path"):
            state["source_text_audio_path"] = perception.get("source_text_audio_path")
        if perception.get("source_text_audio_index") is not None:
            state["source_text_audio_index"] = perception.get("source_text_audio_index")
        state["perception_report"] = perception
        reports = state.setdefault("perception_reports_by_audio_path", {})
        if isinstance(reports, dict):
            source_audio_path = perception.get("source_text_audio_path") or input_audio
            reports[str(source_audio_path)] = perception
            try:
                reports[str(Path(str(source_audio_path)).expanduser().resolve())] = perception
            except Exception:
                pass
        step.params.setdefault("source_text", source_text)

    def _verify_tool_contract_before_handler(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> None:
        if not _tool_contract_verification_enabled(state):
            return
        operation_spec = self._operation_spec_for_step(step)
        if operation_spec is None:
            violation = build_tool_contract_violation(
                step,
                state,
                operation_spec=None,
                stage="pre_tool_invocation",
                reason=f"Unsupported audio operation: {step.operation}",
            )
            self._record_tool_contract_violation(step, state, violation)
            raise ToolContractViolationError([violation])
        violations = verify_step_tool_contract(
            step,
            state,
            operation_spec=operation_spec,
            output_path=_output_path(step, state),
        )
        if not violations:
            return
        repair_attempt_count = len(state.get("tool_contract_repair_attempts") or [])
        if _argument_repair_enforced(state):
            repaired = self._maybe_enforce_step_argument_repair(
                step,
                state,
                violations,
                operation_spec=operation_spec,
            )
        else:
            repaired = self._maybe_repair_step_tool_contract_with_executor_llm(
                step, state, violations
            )
        if repaired:
            repaired_violations = verify_step_tool_contract(
                step,
                state,
                operation_spec=operation_spec,
                output_path=_output_path(step, state),
            )
            attempts = state.get("tool_contract_repair_attempts") or []
            if len(attempts) > repair_attempt_count:
                attempt = attempts[-1]
                attempt["status"] = "accepted" if not repaired_violations else "rejected"
                attempt["remaining_violations"] = [
                    item.to_dict() if hasattr(item, "to_dict") else dict(item)
                    for item in repaired_violations
                ]
            if not repaired_violations:
                return
            violations = repaired_violations
        for violation in violations:
            self._record_tool_contract_violation(step, state, violation)
        raise ToolContractViolationError(violations)

    def _operation_spec_for_step(self, step: AudioEditStep) -> dict[str, Any] | None:
        try:
            return self.registry.describe_operation(step.operation)
        except KeyError:
            return None

    def _record_tool_contract_violation(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        violation: Any,
    ) -> None:
        payload = violation.to_dict() if hasattr(violation, "to_dict") else dict(violation)
        existing = state.setdefault("tool_contract_violations", [])
        duplicate_key = (
            payload.get("step_id"),
            payload.get("operation"),
            payload.get("stage"),
            payload.get("reason"),
        )
        for item in existing:
            if not isinstance(item, dict):
                continue
            item_key = (
                item.get("step_id"),
                item.get("operation"),
                item.get("stage"),
                item.get("reason"),
            )
            if item_key == duplicate_key:
                state["last_tool_contract_error"] = item
                return
        existing.append(payload)
        state["last_tool_contract_error"] = payload
        state["plan"].metadata["need_tool_contract_replan"] = True
        state["plan"].metadata["last_tool_contract_error"] = payload

    def _maybe_repair_step_with_gemini(
        self, step: AudioEditStep, state: dict[str, Any], error: str
    ) -> bool:
        use_gemini = _context_bool(
            state["context"],
            "use_executor_tool_call_planner",
            default=_context_bool(
                state["context"],
                "use_gemini_executor",
                default=self.enable_gemini_executor,
            ),
        )
        if not use_gemini:
            return False
        if not _context_bool(state["context"], "executor_llm_repair", default=True):
            return False
        if state.get("_executor_llm_repair_in_progress"):
            return False
        state["last_executor_normalization_error"] = error
        state["_executor_llm_repair_in_progress"] = True
        try:
            self._maybe_normalize_step_with_gemini(step, state)
            state["operation_normalization_errors"][f"{step.id}:mechanical"] = error
            return True
        except Exception as exc:
            state["executor_tool_call_errors"][f"{step.id}:repair"] = f"{type(exc).__name__}: {exc}"
            return False
        finally:
            state["_executor_llm_repair_in_progress"] = False

    def _maybe_repair_step_tool_contract_with_executor_llm(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        violations: list[Any],
    ) -> bool:
        use_executor_planner = _context_bool(
            state["context"],
            "use_executor_tool_call_planner",
            default=_context_bool(
                state["context"],
                "use_gemini_executor",
                default=self.enable_gemini_executor,
            ),
        )
        if not use_executor_planner:
            return False
        if not _context_bool(state["context"], "executor_llm_repair", default=True):
            return False
        if state.get("_executor_tool_contract_repair_in_progress"):
            return False
        state["_executor_tool_contract_repair_in_progress"] = True
        verifier_error = {
            "stage": "tool_contract_verifier",
            "violations": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in violations
            ],
        }
        state["last_tool_contract_error"] = verifier_error
        before = {
            "inputs": dict(step.inputs),
            "params": dict(step.params),
            "operation": step.operation,
        }
        try:
            self._maybe_normalize_step_with_gemini(step, state, verifier_error=verifier_error)
            after = {
                "inputs": dict(step.inputs),
                "params": dict(step.params),
                "operation": step.operation,
            }
            state.setdefault("tool_contract_repair_attempts", []).append(
                {
                    "step_id": step.id,
                    "operation": step.operation,
                    "provider": _executor_provider_name(self.executor_tool_call_planner),
                    "status": "attempted",
                    "before": before,
                    "after": after,
                    "verifier_error": verifier_error,
                }
            )
            return True
        except Exception as exc:
            state.setdefault("tool_contract_repair_attempts", []).append(
                {
                    "step_id": step.id,
                    "operation": step.operation,
                    "provider": _executor_provider_name(self.executor_tool_call_planner),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "verifier_error": verifier_error,
                }
            )
            state["executor_tool_call_errors"][f"{step.id}:tool_contract_repair"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False
        finally:
            state["_executor_tool_contract_repair_in_progress"] = False

    def _maybe_enforce_step_argument_repair(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        violations: list[Any],
        *,
        operation_spec: dict[str, Any],
    ) -> bool:
        """Apply one admitted field patch without mutating the failed step first."""

        attempts = state.setdefault("tool_contract_repair_attempts", [])
        prior_attempts = [
            item
            for item in attempts
            if isinstance(item, dict)
            and item.get("controller_enforced")
            and item.get("step_id") == step.id
        ]
        if len(prior_attempts) >= argument_repair_max_attempts(state["context"]):
            return False

        violation_payloads = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in violations
        ]
        verifier_error = {
            "stage": "tool_contract_verifier",
            "violations": violation_payloads,
        }
        before = step.model_copy(deep=True)
        deterministic = deterministic_argument_repair_candidate(
            step=before,
            violations=violation_payloads,
            operation_spec=operation_spec,
        )
        selected_budget = (
            {"deterministic_repairs": 1.0}
            if deterministic is not None
            else {"executor_llm_calls": 1.0}
        )
        controller = RepairController()
        signal = controller.signal_from_tool_contract(violation_payloads)
        decision = controller.recommend(
            signal,
            legacy_action="replan_global",
            budget_delta=selected_budget,
            shadow_only=False,
        )
        decision = decision.model_copy(update={"budget_delta": selected_budget})
        attempt: dict[str, Any] = {
            "step_id": step.id,
            "operation": step.operation,
            "provider": _executor_provider_name(self.executor_tool_call_planner),
            "status": "selected",
            "controller_enforced": True,
            "legacy_fallback_action": "replan_global",
            "before": {
                "operation": before.operation,
                "inputs": dict(before.inputs),
                "params": dict(before.params),
            },
            "verifier_error": verifier_error,
            "signal": signal.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }
        attempts.append(attempt)
        state["last_tool_contract_error"] = verifier_error

        if decision.action != "repair_step_arguments":
            attempt["status"] = "rejected"
            attempt["rejection_reason"] = (
                f"repair controller selected {decision.action!r}, not argument repair"
            )
            return False
        if not repair_action_is_enforced(state["context"], decision.action):
            attempt["status"] = "rejected"
            attempt["rejection_reason"] = "repair action is not explicitly allowlisted"
            return False
        state["_executor_tool_contract_repair_in_progress"] = True
        try:
            if deterministic is not None:
                call, candidate = deterministic
                validate_executor_tool_call(
                    call=call,
                    step=before,
                    state=state,
                    operation_spec=operation_spec,
                    manual=None,
                )
                producer_kind = "deterministic"
                producer = "deterministic_argument_repair_candidate"
                provider = None
                model = None
                prompt_version = "x-editor.argument-repair/mechanical-v1"
                attempt["provider"] = "deterministic"
            else:
                if not _executor_argument_repair_enabled(self, state):
                    attempt["status"] = "failed"
                    attempt["error"] = "executor semantic argument-repair proposer is disabled"
                    return False
                call, candidate = self._propose_step_argument_repair(
                    before,
                    state,
                    operation_spec=operation_spec,
                    verifier_error=verifier_error,
                )
                producer_kind = "executor_llm"
                producer = self.executor_tool_call_planner.__class__.__name__
                provider = _executor_provider_name(self.executor_tool_call_planner)
                model = getattr(self.executor_tool_call_planner, "model_name", None)
                prompt_version = getattr(self.executor_tool_call_planner, "prompt_version", None)
                attempt["provider"] = provider
            proposal = proposal_from_executor_call(
                signal=signal,
                before=before,
                candidate=candidate,
                call=call,
                violations=violation_payloads,
                producer=producer,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                producer_kind=producer_kind,
            )
            admission = ArgumentRepairAdmissionGate().audit(
                proposal,
                evidence_sources=argument_repair_evidence_sources(before, state),
            )
            attempt["proposal"] = proposal.model_dump(mode="json")
            attempt["admission"] = admission.model_dump(mode="json")
            attempt["candidate"] = {
                "operation": candidate.operation,
                "inputs": dict(candidate.inputs),
                "params": dict(candidate.params),
            }
            if not admission.admitted:
                attempt["status"] = "rejected"
                attempt["rejection_reason"] = "argument repair admission rejected"
                return False

            remaining = verify_step_tool_contract(
                candidate,
                state,
                operation_spec=operation_spec,
                output_path=_output_path(candidate, state),
            )
            attempt["remaining_violations"] = [
                item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in remaining
            ]
            if remaining:
                attempt["status"] = "rejected"
                attempt["rejection_reason"] = (
                    "candidate still violates the deterministic tool contract"
                )
                return False

            step.inputs = dict(candidate.inputs)
            step.params = dict(candidate.params)
            attempt["status"] = "accepted"
            attempt["after"] = {
                "operation": step.operation,
                "inputs": dict(step.inputs),
                "params": dict(step.params),
            }
            admission.behavior_changed = True
            attempt["admission"] = admission.model_dump(mode="json")
            return True
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            state["executor_tool_call_errors"][f"{step.id}:argument_repair"] = attempt["error"]
            return False
        finally:
            state["_executor_tool_contract_repair_in_progress"] = False

    def _propose_step_argument_repair(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        *,
        operation_spec: dict[str, Any],
        verifier_error: dict[str, Any],
    ) -> tuple[ExecutorToolCall, AudioEditStep]:
        if self.executor_tool_call_planner is None:
            self.executor_tool_call_planner = self._build_executor_tool_call_planner(
                timeout=_env_float("X_EDITOR_GEMINI_EXECUTOR_TIMEOUT", 60.0)
            )
            self.gemini_executor = self.executor_tool_call_planner
        manual = None
        tool_id = operation_spec.get("tool_id")
        if tool_id:
            try:
                manual = load_manual(str(tool_id))
            except FileNotFoundError:
                manual = None
        call = _plan_executor_tool_call(
            self.executor_tool_call_planner,
            step=step,
            state=state,
            operation_spec=operation_spec,
            manual=manual,
            verifier_error=verifier_error,
        )
        _record_executor_gemini_metrics(state, step, self.executor_tool_call_planner)
        validate_executor_tool_call(
            call=call,
            step=step,
            state=state,
            operation_spec=operation_spec,
            manual=manual,
        )
        state["executor_tool_calls"][f"{step.id}:argument_repair"] = call.model_dump(mode="json")
        if not call.allow_execute:
            return call, step.model_copy(deep=True)
        return call, apply_executor_tool_call(step, call, state=state)

    def _maybe_normalize_step_with_gemini(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        *,
        verifier_error: dict[str, Any] | str | None = None,
    ) -> None:
        use_gemini = _context_bool(
            state["context"],
            "use_executor_tool_call_planner",
            default=_context_bool(
                state["context"],
                "use_gemini_executor",
                default=self.enable_gemini_executor,
            ),
        )
        if not use_gemini:
            return
        if self.agentic_mode == OPEN_GLM_QWEN_OMNI_MODE and not (
            state.get("_executor_llm_repair_in_progress")
            or state.get("_executor_tool_contract_repair_in_progress")
        ):
            return
        if _has_self_contained_structured_runtime_params(step):
            return
        if step.operation == "emphasis_stress_edit":
            return
        if step.params.get("reference_pacing_policy") == "duration_ratio_approximation":
            return
        if step.operation in {"artifact_ops.reorder_spans", "export_wav"}:
            return
        if self.executor_tool_call_planner is None:
            self.executor_tool_call_planner = self._build_executor_tool_call_planner(
                timeout=_env_float("X_EDITOR_GEMINI_EXECUTOR_TIMEOUT", 60.0)
            )
            self.gemini_executor = self.executor_tool_call_planner

        try:
            spec = self.registry.describe_operation(step.operation)
            manual = None
            tool_id = spec.get("tool_id")
            if tool_id:
                try:
                    manual = load_manual(str(tool_id))
                except FileNotFoundError:
                    manual = None
            call = _plan_executor_tool_call(
                self.executor_tool_call_planner,
                step=step,
                state=state,
                operation_spec=spec,
                manual=manual,
                verifier_error=verifier_error,
            )
            _record_executor_gemini_metrics(state, step, self.executor_tool_call_planner)
            validate_executor_tool_call(
                call=call,
                step=step,
                state=state,
                operation_spec=spec,
                manual=manual,
            )
            state["executor_tool_calls"][step.id] = call.model_dump()
            if not call.allow_execute:
                if self._maybe_allow_speech_rate_source_text_repair(step, state, call):
                    return
                if self._maybe_allow_intra_audio_transfer_with_gemini_segments(step, state, call):
                    return
                if self._maybe_allow_text_edit_with_deterministic_repair(step, state, call):
                    return
                if self._maybe_allow_audio_effect_with_deterministic_preset(step, state, call):
                    return
                if self._maybe_rewrite_blocked_step_to_audio_effect(step, state, call):
                    return
                missing = f" missing_inputs={call.missing_inputs}" if call.missing_inputs else ""
                raise RuntimeError(
                    f"Gemini executor blocked step: {call.blocked_reason or 'not executable'}{missing}"
                )
            normalized = apply_executor_tool_call(step, call)
            step.inputs = normalized.inputs
            step.params = normalized.params
            _resolve_step_audio_aliases(step, state)
        except Exception as exc:
            _record_executor_gemini_metrics(state, step, self.executor_tool_call_planner)
            state["executor_tool_call_errors"][step.id] = f"{type(exc).__name__}: {exc}"
            if _context_bool(state["context"], "gemini_executor_fallback", default=False):
                return
            raise

    def _maybe_allow_speech_rate_source_text_repair(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        call: ExecutorToolCall,
    ) -> bool:
        if not _executor_blocked_on_source_text(call):
            return False
        operation = step.operation.removeprefix("edit_data_pipeline.").replace("-", "_")
        if operation == "speech_rate":
            self._maybe_repair_missing_source_text_for_speech_rate(step, state)
            state.setdefault("executor_tool_call_repairs", {})[step.id] = {
                "strategy": "perception_source_text_before_speech_rate_after_gemini_block",
                "blocked_reason": call.blocked_reason,
                "missing_inputs": list(call.missing_inputs or []),
                "reason": (
                    "Gemini executor blocked the quality speech_rate path because source_text "
                    "was missing; source_text is repairable via ASR/Gemini perception, so the "
                    "step should continue instead of downgrading or rewriting to audio effects."
                ),
            }
            return True

        nested_operation = _nested_edit_operation_name(step.params)
        if operation == "foreground_background_remix" and nested_operation == "speech_rate":
            state.setdefault("executor_tool_call_repairs", {})[step.id] = {
                "strategy": "defer_nested_speech_rate_source_text_repair_to_foreground_handler",
                "blocked_reason": call.blocked_reason,
                "missing_inputs": list(call.missing_inputs or []),
                "nested_operation": "edit_data_pipeline.speech_rate",
                "reason": (
                    "Nested foreground speech_rate needs transcript text for wsola+indextts2, "
                    "but the correct transcript target is the selected foreground stem after "
                    "source separation. Defer to foreground_background_remix source_text repair "
                    "rather than rewriting the composite operation to audio_effects."
                ),
            }
            return True
        return False

    def _maybe_allow_text_edit_with_deterministic_repair(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        call: ExecutorToolCall,
    ) -> bool:
        if step.operation != "edit_data_pipeline.text_edit":
            return False
        source_text = state.get("source_text")
        if not isinstance(source_text, str) or not source_text.strip():
            return False
        instruction = str(
            step.params.get("instruction") or step.description or state.get("user_prompt") or ""
        ).strip()
        if not instruction:
            return False
        input_audio = (
            step.inputs.get("input_audio_path")
            or step.inputs.get("input_audio")
            or state.get("current_audio_path")
            or state.get("input_audio")
            or "current"
        )
        output_path = str(step.params.get("output_path") or _output_path(step, state))
        language = (
            step.params.get("language")
            or state.get("language")
            or (state.get("context") or {}).get("language")
        )
        try:
            args = build_edit_data_pipeline_tool_arguments(
                operation="text_edit",
                instruction=instruction,
                source_text=source_text,
                input_audio_path=str(input_audio),
                output_path=output_path,
                output_dir=str(state.get("output_dir") or ""),
                language=str(language) if language else None,
                params={**step.params, "instruction": instruction},
            )
        except OperationBoundaryError as exc:
            state["executor_tool_call_errors"][f"{step.id}:deterministic_text_edit_repair"] = str(
                exc
            )
            return False
        except Exception as exc:  # noqa: BLE001
            state["executor_tool_call_errors"][f"{step.id}:deterministic_text_edit_repair"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False

        step.inputs["input_audio_path"] = str(input_audio)
        step.params.update(args)
        state.setdefault("executor_tool_call_repairs", {})[step.id] = {
            "strategy": "deterministic_text_edit_after_gemini_block",
            "blocked_reason": call.blocked_reason,
            "missing_inputs": list(call.missing_inputs or []),
            "instruction_st": args.get("instruction_st"),
            "target_text": args.get("target_text"),
        }
        return True

    def _maybe_allow_audio_effect_with_deterministic_preset(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        call: ExecutorToolCall,
    ) -> bool:
        if step.operation != "audio_effects":
            return False
        params = dict(step.params)
        instruction = str(
            params.get("instruction")
            or params.get("preset")
            or params.get("effect")
            or step.description
            or state.get("user_prompt")
            or ""
        ).strip()
        preset = canonical_audio_effect_preset(
            " ".join(
                str(item)
                for item in [params.get("preset"), params.get("effect"), instruction]
                if item
            )
        )
        if not preset:
            return False
        input_audio = (
            step.inputs.get("input_audio_path")
            or step.inputs.get("input_audio")
            or state.get("current_audio_path")
            or state.get("input_audio")
            or "current"
        )
        step.inputs["input_audio_path"] = str(input_audio)
        params["instruction"] = instruction
        params["preset"] = preset
        params.setdefault("strength", 0.75)
        params.setdefault("preserve_duration", True)
        step.params = params
        state.setdefault("executor_tool_call_repairs", {})[step.id] = {
            "strategy": "deterministic_audio_effect_after_gemini_block",
            "blocked_reason": call.blocked_reason,
            "missing_inputs": list(call.missing_inputs or []),
            "preset": preset,
        }
        return True

    def _maybe_rewrite_blocked_step_to_audio_effect(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        call: ExecutorToolCall,
    ) -> bool:
        if step.operation == "audio_effects":
            return False
        if "audio_effects" not in self.registry.list_operations():
            return False
        instruction = str(
            step.params.get("instruction")
            or step.params.get("effect")
            or step.params.get("preset")
            or step.description
            or state.get("user_prompt")
            or ""
        ).strip()
        evidence = " ".join(
            str(value)
            for value in [
                instruction,
                step.operation,
                call.blocked_reason,
                " ".join(str(item) for item in (call.missing_inputs or [])),
                state.get("user_prompt"),
            ]
            if value
        )
        preset = canonical_audio_effect_preset(evidence)
        if not preset:
            return False
        original_operation = step.operation
        input_audio = (
            step.inputs.get("input_audio_path")
            or step.inputs.get("input_audio")
            or state.get("current_audio_path")
            or state.get("input_audio")
            or "current"
        )
        step.operation = "audio_effects"
        step.inputs = {"input_audio_path": str(input_audio)}
        step.params = {
            "instruction": instruction or str(state.get("user_prompt") or ""),
            "preset": preset,
            "strength": 0.75,
            "preserve_duration": True,
        }
        state.setdefault("executor_tool_call_repairs", {})[step.id] = {
            "strategy": "blocked_step_rewritten_to_audio_effect",
            "original_operation": call.operation
            or call.tool_id
            or call.mcp_tool_name
            or original_operation,
            "blocked_reason": call.blocked_reason,
            "missing_inputs": list(call.missing_inputs or []),
            "preset": preset,
        }
        return True

    def _maybe_allow_intra_audio_transfer_with_gemini_segments(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        call: ExecutorToolCall,
    ) -> bool:
        if step.operation != "intra_audio_voice_transfer":
            return False
        missing_text = " ".join(str(item) for item in (call.missing_inputs or [])).casefold()
        blocked_text = str(call.blocked_reason or "").casefold()
        if "speaker" not in f"{missing_text} {blocked_text}":
            return False
        perception = state.get("perception_report") or {}
        gemini_speaker_ids = _gemini_perception_speaker_ids(perception)
        if len(gemini_speaker_ids) < 2:
            return False
        step.inputs = {
            **step.inputs,
            "input_audio_path": step.inputs.get("input_audio_path")
            or step.inputs.get("input_audio")
            or "current",
        }
        params = dict(step.params)
        params.setdefault("instruction", step.description or state.get("user_prompt"))
        source_text = state.get("source_text") or (
            perception.get("source_text") if isinstance(perception, dict) else None
        )
        if isinstance(source_text, str) and source_text.strip():
            params.setdefault("source_text", source_text)
        params["perception_report"] = perception
        step.params = params
        state.setdefault("executor_tool_call_repairs", {})[step.id] = {
            "operation": "intra_audio_voice_transfer",
            "repair": "allow_with_gemini_fallback_speaker_segments",
            "reason": call.blocked_reason,
            "missing_inputs": list(call.missing_inputs or []),
            "gemini_speaker_ids": gemini_speaker_ids,
        }
        return True

    def _maybe_rewrite_cross_file_intra_voice_transfer(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> bool:
        if step.operation != "intra_audio_voice_transfer":
            return False
        reference_audio = _reference_vc_fallback_audio(step, state)
        if not reference_audio:
            return False
        explicit_reference = any(
            step.inputs.get(key) or step.params.get(key)
            for key in (
                "reference_audio_path",
                "reference_audio",
                "ref_audio_path",
                "speaker_audio_path",
            )
        )
        prompt = str(state.get("user_prompt") or step.description or "")
        if not explicit_reference and not re.search(
            r"\baudio\s*[12]\b|\binput\s*[12]\b|\breference audio\b", prompt, re.IGNORECASE
        ):
            return False

        source_selector = _selector_text(
            step.params.get("source_selector")
            or step.params.get("target_selector")
            or step.params.get("selector")
            or step.params.get("source_speaker")
            or step.params.get("target_speaker")
        )
        input_audio = (
            step.inputs.get("input_audio")
            or step.inputs.get("input_audio_path")
            or step.params.get("input_audio")
            or step.params.get("input_audio_path")
            or "current"
        )
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "intra_audio_voice_transfer",
            "to": "artifact_ops.speaker_scoped_edit" if source_selector else "seed_vc",
            "reason": "external reference audio was provided; use cross-file SeedVC instead of intra-audio speaker mapping",
            "input_audio": resolve_audio_reference(input_audio, state),
            "reference_audio_path": reference_audio,
            "source_selector": source_selector,
        }
        if source_selector:
            step.operation = "artifact_ops.speaker_scoped_edit"
            step.inputs = {"input_audio": input_audio}
            step.params = {
                "selector": source_selector,
                "edit_op": "seed_vc",
                "edit_op_params": {
                    "reference_audio_path": reference_audio,
                },
                "preserve_timeline": True,
                "instruction": step.description or prompt,
            }
            return True

        step.operation = "seed_vc"
        step.inputs = {
            "input_audio": input_audio,
            "reference_audio_path": reference_audio,
        }
        step.params = {"reference_audio_path": reference_audio}
        return True

    def _maybe_rewrite_reference_free_tse_to_timeline_extract(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> bool:
        if step.operation != "tse":
            return False
        params = dict(step.params)
        if any(
            step.inputs.get(key) or params.get(key)
            for key in (
                "reference_audio_path",
                "reference_audio",
                "ref_audio_path",
                "speaker_audio_path",
            )
        ):
            return False
        if _tse_has_external_reference_audio(step, state, params):
            return False
        if len(collect_input_audios(state)) > 1 or state.get("reference_audio_paths"):
            return False
        text = " ".join(
            str(value)
            for value in [
                state.get("user_prompt"),
                step.description,
                params.get("instruction"),
                params.get("target_description"),
            ]
            if value
        ).lower()
        if not _tse_text_is_speech_speaker_extract(text):
            return False
        input_audio = (
            step.inputs.get("mixture_audio_path")
            or step.inputs.get("mixture_audio")
            or step.inputs.get("input_audio_path")
            or step.inputs.get("input_audio")
            or "current"
        )
        selector = (
            params.get("instruction")
            or params.get("target_description")
            or step.description
            or state.get("user_prompt")
        )
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "tse",
            "to": "artifact_ops.extract_spans",
            "reason": (
                "TSE requires an enrollment reference; this step only has a natural-language speaker/speech selector, "
                "so use timeline extraction with perception evidence instead"
            ),
            "selector": selector,
        }
        step.operation = "artifact_ops.extract_spans"
        step.inputs = {"input_audio": input_audio}
        step.params = {
            "instruction": selector,
            "selection_instruction": selector,
        }
        return True

    def _maybe_rewrite_voice_order_swap_to_timeline_reorder(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
    ) -> bool:
        if step.operation not in {
            "intra_audio_voice_transfer",
            "artifact_ops.speaker_scoped_edit",
        }:
            return False
        request = str(
            state.get("user_prompt") or step.description or step.params.get("instruction") or ""
        )
        lowered = request.lower()
        request_reorders_speakers = (
            (
                any(
                    marker in lowered
                    for marker in ["swap", "switch", "exchange", "reorder", "reverse"]
                )
                and "order" in lowered
            )
            or (
                "order" in lowered
                and any(marker in lowered for marker in ["change", "alter", "modify"])
            )
        ) and any(
            marker in lowered for marker in ["voice", "voices", "speaker", "speakers", "human"]
        )
        nested_reorder_operation = _is_timeline_reorder_operation_name(
            step.params.get("operation")
            or step.params.get("edit_op")
            or step.params.get("edit_operation")
            or step.params.get("sub_operation")
        )
        if step.operation == "intra_audio_voice_transfer" and not request_reorders_speakers:
            return False
        if step.operation == "artifact_ops.speaker_scoped_edit" and not (
            request_reorders_speakers or nested_reorder_operation
        ):
            return False
        input_audio = (
            step.inputs.get("input_audio_path")
            or step.inputs.get("input_audio")
            or step.params.get("input_audio_path")
            or step.params.get("input_audio")
            or "current"
        )
        from_operation = step.operation
        params = {
            "instruction": request or step.description or "Reorder selected voice segments.",
            "preserve_duration": True,
        }
        for key in (
            "selected_spans",
            "spans",
            "segments",
            "order",
            "target_speaker_order",
        ):
            value = step.params.get(key)
            if value:
                params[key] = value
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": from_operation,
            "to": "artifact_ops.reorder_spans",
            "reason": "request asks to swap/reorder the temporal order of voices, not to transfer speaker timbre",
        }
        step.operation = "artifact_ops.reorder_spans"
        step.description = step.description or request or "Reorder selected voice segments."
        step.inputs = {"input_audio_path": input_audio}
        step.params = params
        return True

    def _maybe_rewrite_long_text_edit_as_voice_clone_tts(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        call: Any,
    ) -> bool:
        # Existing-audio lexical edits must not be rewritten to TTS. This
        # method is retained as a no-op compatibility hook so older callers do
        # not regain the harmful qwen3_tts fallback path.
        return False

    def _maybe_rewrite_unsupported_nonhuman_voice_request(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
    ) -> bool:
        if step.operation != "unsupported":
            return False
        if "audio_effects" not in self.registry.list_operations():
            return False
        request = str(state.get("user_prompt") or step.description or "")
        reason = str(step.params.get("reason") or step.description or "")
        lowered = f"{request} {reason}".lower()
        if not _is_nonhuman_voice_request(lowered):
            return False
        input_audio = _nonhuman_voice_best_effort_input_audio(request, state)
        if not input_audio or not Path(str(input_audio)).exists():
            return False
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "unsupported",
            "to": "audio_effects",
            "strategy": "best_effort_nonhuman_vocal_coloration",
            "reason": (
                "No true non-human/animal voice conversion model is registered. "
                "Using deterministic animal_vocal_style DSP as a best-effort audible output "
                "while preserving the selected input timeline."
            ),
            "original_reason": reason,
            "input_audio_path": str(input_audio),
        }
        step.operation = "audio_effects"
        step.description = step.description or "Best-effort non-human vocal style coloration."
        step.inputs = {"input_audio_path": str(input_audio)}
        step.params = {
            "preset": "animal_vocal_style",
            "strength": 0.78,
            "instruction": request or step.description,
            "best_effort": True,
            "capability_boundary": (
                "animal_vocal_style is a DSP approximation, not true animal voice conversion "
                "or semantic speech synthesis from animal vocalizations."
            ),
        }
        return True

    def _maybe_rewrite_unsupported_audio_effect_request(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
    ) -> bool:
        if step.operation != "unsupported":
            return False
        if "audio_effects" not in self.registry.list_operations():
            return False
        request = str(state.get("user_prompt") or step.description or "")
        reason = str(step.params.get("reason") or step.description or "")
        preset = canonical_audio_effect_preset(f"{request} {reason}")
        if not preset:
            return False
        input_audio = state.get("current_audio_path") or state.get("input_audio")
        if not input_audio or not Path(str(input_audio)).exists():
            return False
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "unsupported",
            "to": "audio_effects",
            "strategy": "deterministic_audio_effect_for_supported_preset",
            "original_reason": reason,
            "preset": preset,
        }
        step.operation = "audio_effects"
        step.description = step.description or request or f"Apply {preset}."
        step.inputs = {"input_audio_path": str(input_audio)}
        step.params = {
            "preset": preset,
            "strength": 0.75,
            "instruction": request or reason or step.description,
            "preserve_duration": True,
        }
        return True

    async def _maybe_repair_runtime_text_edit_failure(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        original_exc: Exception,
    ) -> str | None:
        # Text-edit runtime failures must stay on the text-edit boundary. Do
        # not regenerate existing-source lexical edits with qwen3_tts.
        message = f"{type(original_exc).__name__}: {original_exc}"
        if not _implicit_executor_repair_enabled(state):
            return message
        if step.operation != "edit_data_pipeline.text_edit":
            return message
        if not _is_f5_text_edit_duration_limit_error(message):
            return message
        perception = state.get("perception_report") or {}
        word_timestamps = (
            perception.get("word_timestamps") if isinstance(perception, dict) else None
        )
        if not word_timestamps:
            return (
                "text_edit local repair requires word_timestamps or full aligned transcript; "
                f"original error: {message}"
            )

        params = dict(step.params)
        instruction = params.pop("instruction", state["user_prompt"])
        input_audio = _resolve_input_audio(step, state)
        output_path = _output_path(step, state)
        source_text = params.pop("source_text", None) or state.get("source_text")
        language = canonicalize_language(
            params.pop("language", None) or _language_from_perception(state)
        )
        args = build_edit_data_pipeline_tool_arguments(
            operation="text_edit",
            instruction=str(instruction),
            source_text=str(source_text) if source_text is not None else None,
            input_audio_path=input_audio,
            output_path=output_path,
            output_dir=state["output_dir"],
            language=language,
            params=params,
        )
        result = await maybe_execute_segmented_edit_pipeline(
            tool_executor=self.tool_executor,
            arguments=args,
            state={
                "user_prompt": state["user_prompt"],
                "source_text": state.get("source_text"),
                "perception_report": perception,
            },
            output_dir=state["output_dir"],
            force_local_text_edit=True,
        )
        state.setdefault("tool_invocations", []).append(
            _without_empty(
                {
                    "step_id": step.id,
                    "tool_id": "edit_data_pipeline",
                    "arguments": args,
                    "runtime_repair": "local_phrase_text_edit_after_f5_limit",
                    "success": bool(result and result.success),
                    "error_message": result.error_message
                    if result
                    else "segmented repair did not run",
                    "output": result.output if result else None,
                }
            )
        )
        if result is None:
            return (
                "text_edit local repair requires word_timestamps or full aligned transcript and a "
                f"single alignable <sub>/<del> instruction_st; original error: {message}"
            )
        if not result.success:
            return f"{result.error_message}; original error: {message}"
        output = result.output if isinstance(result.output, dict) else {}
        step.status = "success"
        step.error = None
        self._commit_step_output(step, state, output)
        try:
            await self._verify_step_after_success(step, state)
        except Exception as verify_exc:
            return f"{type(verify_exc).__name__}: {verify_exc}"
        return None

    def _maybe_rewrite_emphasis_pitch_rate_to_p1_resynthesis(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
    ) -> None:
        if step.operation != "edit_data_pipeline.pitch_rate_edit":
            return
        if "emphasis_stress_edit" not in self.registry.list_operations():
            return
        params = step.params if isinstance(step.params, dict) else {}
        if str(params.get("emphasis_strategy") or "").strip().lower() in {
            "pitch_rate",
            "pitch-rate",
            "legacy_pitch_rate",
            "legacy",
        }:
            return
        evidence = " ".join(
            str(value or "")
            for value in [
                state.get("user_prompt"),
                step.description,
                params.get("instruction"),
                params.get("edit_instruction"),
            ]
        )
        if not _requests_emphasis_stress_edit(evidence):
            return
        target_phrase = _emphasis_target_phrase(
            params,
            instruction=evidence,
            source_text=params.get("source_text") or state.get("source_text"),
        )
        if not target_phrase:
            return
        original_operation = step.operation
        original_params = dict(params)
        step.operation = "emphasis_stress_edit"
        step.params = {
            **original_params,
            "instruction": str(
                params.get("instruction")
                or step.description
                or state.get("user_prompt")
                or evidence
            ),
            "target_phrase": target_phrase,
            "_pitch_rate_fallback_params": original_params,
            "emphasis_strategy": "p1_local_resynthesis",
        }
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": original_operation,
            "to": "emphasis_stress_edit",
            "reason": (
                "emphasis/stress edits now prefer P1 local resynthesis plus SeedVC speaker recovery "
                "and audio_assembly splice normalization over pitch-only DSP"
            ),
            "target_phrase": target_phrase,
            "fallback_operation": "edit_data_pipeline.pitch_rate_edit",
        }

    def _register_default_operations(self) -> None:
        capabilities = load_all_capabilities()
        if "auk" in capabilities:
            from x_editor.runtime.auk_handler import register_auk

            register_auk(self)
        if "speech_enhancement" in capabilities:
            self.registry.register(
                "speech_enhancement",
                self._handle_speech_enhancement,
                description="Improve speech clarity, denoise, dereverb, or clean final artifacts with ClearerVoice.",
                tool_id="speech_enhancement",
                required_inputs=["input_audio"],
                default_params={"mode": "general"},
            )
        if "edit_data_pipeline" in capabilities:
            manual = load_manual("edit_data_pipeline")
            enum_values = (
                manual.input_schema.get("properties", {}).get("operation", {}).get("enum", [])
            )
            for operation in enum_values:
                op_id = f"edit_data_pipeline.{operation}"
                default_params = filter_edit_pipeline_service_overrides(
                    operation,
                    manual.defaults,
                )
                if operation == "text_edit":
                    default_params["text_edit_max_input_sec"] = float(
                        manual.defaults.get("text_edit_max_input_sec", 30.0)
                    )
                elif operation == "volume_normalize":
                    default_params["method"] = "rms"
                elif operation == "pitch_shift":
                    default_params["method"] = "psola"
                label_contract = edit_data_pipeline_label_contract(operation)
                if label_contract:
                    default_params["label_contract"] = label_contract
                self.registry.register(
                    op_id,
                    self._handle_edit_data_pipeline,
                    description=f"Run edit-data-pipeline operation `{operation}` through its MCP wrapper.",
                    tool_id="edit_data_pipeline",
                    tool_operation=operation,
                    required_inputs=["input_audio"],
                    default_params=default_params,
                )
        for tool_id, handler, description in [
            (
                "audio_effects",
                self._handle_audio_effects,
                "Apply deterministic speech-preserving acoustic/channel effects.",
            ),
            (
                "seamless_expressive_s2st",
                self._handle_seamless_expressive_s2st,
                "Translate source speech to target-language speech with SeamlessExpressive/SeamlessM4T.",
            ),
            (
                "seed_vc",
                self._handle_seed_vc,
                "Convert source speech to a reference speaker with SeedVC.",
            ),
            (
                "qwen3_tts",
                self._handle_qwen3_tts,
                "Synthesize speech from text with Qwen3-TTS.",
            ),
            (
                "tse",
                self._handle_tse,
                "Extract a target speaker using reference enrollment audio.",
            ),
            (
                "speech_separation",
                self._handle_speech_separation,
                "Separate mixed or overlapping speech into stems.",
            ),
            (
                "spatial_reverb",
                self._handle_spatial_reverb,
                "Apply room impulse response reverb for a requested acoustic space.",
            ),
            (
                "intra_audio_voice_transfer",
                self._handle_intra_audio_voice_transfer,
                "Transfer timbre between speakers inside the same audio.",
            ),
        ]:
            if tool_id in capabilities:
                self.registry.register(
                    tool_id,
                    handler,
                    description=description,
                    tool_id=tool_id,
                    required_params=["target_language"]
                    if tool_id == "seamless_expressive_s2st"
                    else None,
                )
        if "seed_vc" in capabilities and (
            "qwen3_tts" in capabilities
        ):
            self.registry.register(
                "no_reference_voice_conversion",
                self._handle_no_reference_voice_conversion,
                description=(
                    "Convert existing/source speech to a natural-language-designed target voice by first "
                    "synthesizing an independent target-voice reference with Qwen3-TTS, "
                    "then using SeedVC to convert the source waveform to that reference voice."
                ),
                tool_id="seed_vc",
                required_inputs=["input_audio"],
                default_params={
                    "voice_design_backend": "auto",
                    "reference_text_policy": "independent_neutral_carrier",
                },
            )
        if {"edit_data_pipeline", "qwen3_tts", "seed_vc"}.issubset(capabilities):
            self.registry.register(
                "emphasis_stress_edit",
                self._handle_emphasis_stress_edit,
                description=(
                    "Apply emphasis or lexical stress to quoted words/phrases by generating a strongly "
                    "stressed full-sentence target with Qwen3-TTS, converting it back to the source "
                    "speaker with SeedVC, then splicing only the target phrase through "
                    "edit_data_pipeline.audio_assembly with F5 boundary normalization. Prefer this for "
                    "emphasis/stress; use edit_data_pipeline.pitch_rate_edit only as a fallback."
                ),
                tool_id="edit_data_pipeline",
                tool_operation="audio_assembly",
                required_inputs=["input_audio"],
                default_params={
                    "preserve_original_speaker": True,
                    "preserve_outer_boundaries": True,
                    "fallback_operation": "edit_data_pipeline.pitch_rate_edit",
                },
            )
        if "ace_step_music" in capabilities:
            self.registry.register(
                "ace_step_music.timeline_edit",
                self._handle_ace_step_music_timeline_edit,
                description=(
                    "Resolve the requested speech speaker/line spans, send each selected span to ACE-Step "
                    "as singing/rap music vocals using the span as source/reference audio, then fit and "
                    "splice generated segments back into the original timeline. Use for generalized "
                    "speaker-scoped singing or rap edits; exact speaker identity, lyrics, and timing are best-effort."
                ),
                tool_id="ace_step_music",
                tool_operation="timeline_edit",
                required_inputs=["input_audio"],
                default_params={
                    "mode": "cover",
                    "reference_strategy": "selected_segment",
                    "preserve_timeline": True,
                    "allow_missing_lyrics": False,
                },
            )
        self.registry.register(
            "accent_conversion",
            self._handle_accent_conversion,
            description=(
                "Resolve an accent/dialect request with the accent library, convert the selected "
                "accent reference to the target speaker timbre with SeedVC, synthesize the target "
                "content with Qwen3-TTS voice_clone, and splice generated chunks back into the timeline."
            ),
            tool_id="accent_conversion",
            required_inputs=["input_audio"],
            default_params={"preserve_timeline": True},
        )
        self.registry.register(
            "perception.speech_context",
            self._handle_audio_perception_fallback,
            description=(
                "Run lightweight speech-context perception for transcript-conditioned edits: "
                "use specialized Qwen3ASR by default, optionally alignment/diarization when "
                "requested, and only escalate to the configured audio perception fallback when "
                "allow_gemini or force_gemini compatibility flags explicitly enable it. Updates runtime source_text/perception "
                "for downstream grounding."
            ),
            required_inputs=["input_audio"],
            default_params={
                "force_gemini": False,
                "strict": True,
                "need_source_text": True,
                "need_words": False,
                "need_speakers": False,
                "need_speaker_profiles": False,
                "allow_gemini": False,
            },
        )
        self.registry.register(
            "perception.audio_fallback",
            self._handle_audio_perception_fallback,
            description=(
                "Run framework-level audio perception fallback: use specialized ASR/alignment/"
                "diarization when available, then escalate to the configured agentic fallback "
                "provider (Gemini 2.5 Pro or Qwen3-Omni in open mode) when role/speaker/"
                "phrase/location evidence is missing or no specialized perception tool fits. "
                "Prefer perception.speech_context for simple transcript-only content edits."
            ),
            required_inputs=["input_audio"],
            default_params={
                "force_gemini": False,
                "strict": True,
                "need_words": True,
                "need_speakers": True,
            },
        )
        self.registry.register(
            "perception.speaker_attributed_review",
            self._handle_speaker_attributed_review,
            description=(
                "Run the multi-speaker speech perception subagent: collect Qwen3ASR wording, "
                "DiariZen speaker turns, optional SE-DiCoW speaker-attributed ASR hints, then "
                "use the configured speaker review frontend to produce verified "
                "speaker-attributed transcript spans. This is for speech spans only; audio "
                "caption/event grounding is a separate future module."
            ),
            required_inputs=["input_audio"],
            default_params={
                "strict": True,
                "run_specialized": True,
                "use_sedicow": True,
                "max_review_attempts": 2,
            },
        )
        self.registry.register(
            "perception.semantic_span_review",
            self._handle_semantic_span_review,
            description=(
                "Ground quoted speech, roles, or perceptual properties from "
                "TS-ASR/aligned candidate spans; use an Omni yes/no candidate "
                "rubric only when deterministic grounding is insufficient."
            ),
            required_inputs=["input_audio"],
            default_params={
                "action": "extract_spans",
                "need_speakers": False,
                "strict": True,
            },
        )
        self.registry.register(
            "perception.timestamped_caption",
            self._handle_timestamped_caption,
            description=(
                "Run timestamped audio caption perception with the configured "
                "agentic perception provider; optionally use FlexSED as a concrete-event "
                "candidate prior, cut diagnostic slices, then "
                "ask the provider to produce timestamped non-speech/audio-event captions with "
                "verification and repair feedback."
            ),
            required_inputs=["input_audio"],
            default_params={
                "use_flexsed_prior": False,
                "strict": True,
                "max_review_attempts": 2,
            },
        )
        if "flexsed" in capabilities:
            self.registry.register(
                "perception.flexsed_events",
                self._handle_flexsed_events,
                description=(
                    "Run FlexSED open-vocabulary sound event detection to ground concrete "
                    "non-speech or paralinguistic event labels into timestamp spans and "
                    "heatmap evidence. Use this for audio-caption/event span grounding, "
                    "not ASR, diarization, or word/speaker selection."
                ),
                tool_id="flexsed",
                tool_operation="detect_events",
                required_inputs=["input_audio", "events"],
                default_params={
                    "threshold": 0.5,
                    "min_duration_sec": 0.08,
                    "merge_gap_sec": 0.08,
                    "save_heatmap_png": True,
                    "save_heatmap_npy": True,
                },
            )
            self.registry.register(
                "flexsed.detect_events",
                self._handle_flexsed_events,
                description="Compatibility operation alias for perception.flexsed_events.",
                tool_id="flexsed",
                tool_operation="detect_events",
                required_inputs=["input_audio", "events"],
            )
        self.registry.register(
            "perception.gemini_audio",
            self._handle_audio_perception_fallback,
            description=(
                "Compatibility alias for speech perception when a plan explicitly asks "
                "for Gemini-backed fallback perception."
            ),
            required_inputs=["input_audio"],
            default_params={
                "force_gemini": True,
                "strict": True,
                "need_words": True,
                "need_speakers": True,
            },
        )
        self.registry.register(
            "perception.qwen3_omni_audio",
            self._handle_audio_perception_fallback,
            description=(
                "Compatibility alias for perception.audio_fallback when a plan explicitly asks "
                "for Qwen3-Omni/open semantic fallback perception."
            ),
            required_inputs=["input_audio"],
            default_params={
                "force_gemini": True,
                "strict": True,
                "need_words": True,
                "need_speakers": True,
            },
        )
        self.registry.register(
            "perception.open_omni_audio",
            self._handle_audio_perception_fallback,
            description="Alias for Qwen3-Omni/open semantic fallback perception.",
            required_inputs=["input_audio"],
            default_params={
                "force_gemini": True,
                "strict": True,
                "need_words": True,
                "need_speakers": True,
            },
        )
        if "speech_separation" in capabilities:
            self.registry.register(
                "foreground_background_remix",
                self._handle_foreground_background_remix,
                description=(
                    "Use BS-RoFormer/MelBand-RoFormer speech_separation to split foreground speech/vocals "
                    "from background, optionally edit or enhance the selected foreground stem through existing "
                    "X-Editor tools, then remix with the preserved background at the original timeline length. "
                    "Use this for requests such as changing speech rate while keeping background music unchanged, "
                    "or removing voice and keeping background only."
                ),
                required_inputs=["input_audio"],
                default_params={
                    "foreground_stem_policy": "speech",
                    "background_stem_policy": "background",
                    "foreground_action": "edit_foreground",
                    "preserve_duration": True,
                },
            )
        self.registry.register(
            "paralinguistic_event_add",
            self._handle_paralinguistic_event_add,
            description=(
                "Add a nonverbal paralinguistic event (breath, laugh, cough, or sigh) to existing speech "
                "by inserting an offline event asset at a detected pause or safe boundary. This preserves "
                "the original speech waveform around the insert and must not be implemented as text_edit "
                "or full-sentence TTS."
            ),
            required_inputs=["input_audio"],
            default_params={
                "event": "<breath|laugh|cough|sigh>",
                "enable_boundary_smooth": False,
                "enable_seedvc_self_conversion": False,
            },
        )
        self.registry.register(
            "paralinguistic_event_remove",
            self._handle_paralinguistic_event_remove,
            description=(
                "Remove a nonverbal paralinguistic event (breath, laugh, cough, or sigh) from existing speech "
                "by grounding local event spans with FlexSED or runtime evidence, then locally inpainting/"
                "ducking only those spans. AudioSep query separation is not assumed available; never route "
                "this to text_edit or whole-utterance TTS."
            ),
            required_inputs=["input_audio"],
            default_params={
                "event": "<breath|laugh|cough|sigh>",
                "preserve_duration": True,
            },
        )
        self.registry.register(
            "export_wav",
            self._handle_export_wav,
            description="Export/copy the current audio artifact to the requested WAV path.",
            required_inputs=["input_audio"],
        )
        self.registry.register(
            "artifact_ops.apply_gain",
            self._handle_relative_gain,
            description=(
                "Apply an exact relative gain_db to the whole input or explicit selected_spans. "
                "Preserves timeline and unselected samples; fails when headroom is insufficient. "
                "This is relative multiplication, not absolute loudness normalization."
            ),
            tool_id="artifact_ops",
            required_inputs=["input_audio"],
        )
        for operation, description in [
            (
                "artifact_ops.extract_spans",
                "Select timeline spans and export only those regions.",
            ),
            (
                "artifact_ops.remove_spans",
                "Remove selected timeline spans and stitch the remaining audio.",
            ),
            (
                "artifact_ops.mute_spans",
                "Mute selected timeline spans while preserving duration.",
            ),
            (
                "artifact_ops.beep_spans",
                "Replace selected timeline spans with beep tones while preserving duration.",
            ),
            (
                "artifact_ops.silence",
                "Generate a deterministic silent audio artifact of a requested duration.",
            ),
            (
                "artifact_ops.concat",
                "Concatenate input/reference audio artifacts in requested order, optionally inserting a fixed pause between sources.",
            ),
            (
                "artifact_ops.reorder_spans",
                "Reverse or reorder selected timeline spans while preserving surrounding audio.",
            ),
            (
                "artifact_ops.reconstruct",
                "Reconstruct a timeline from base and edited artifacts when spans are available.",
            ),
            (
                "artifact_ops.speaker_scoped_edit",
                "Diarize/select a speaker, edit selected segments via edit-data-pipeline, and stitch back.",
            ),
        ]:
            self.registry.register(
                operation,
                self._handle_artifact_ops,
                description=description,
                tool_id="artifact_ops",
                required_inputs=["input_audio"],
            )
        self.registry.register(
            "unsupported",
            self._handle_unsupported,
            description="Explicit marker for requests outside current X-Editor capabilities.",
        )

    async def _execute_tool(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "step_id": state.get("_current_step_id"),
            "tool_id": tool_id,
            "arguments": dict(arguments),
        }
        result: ToolResult | None = None
        try:
            attempts: list[dict[str, Any]] = []
            max_attempts = _transient_tool_max_attempts(tool_id)
            configured_max_attempts = state["context"].get("max_transient_tool_attempts")
            if configured_max_attempts is not None:
                max_attempts = min(
                    max_attempts,
                    max(1, int(configured_max_attempts)),
                )
            for attempt_index in range(1, max_attempts + 1):
                result = None
                attempt_record: dict[str, Any] = {"attempt": attempt_index}
                try:
                    if tool_id == "edit_data_pipeline":
                        segmentation_state = dict(state)
                        segmentation_state.setdefault("user_prompt", state["user_prompt"])
                        segmentation_state.setdefault("source_text", state.get("source_text"))
                        segmentation_state.setdefault(
                            "perception_report", state.get("perception_report") or {}
                        )
                        result = await maybe_execute_segmented_edit_pipeline(
                            tool_executor=self.tool_executor,
                            arguments=arguments,
                            state=segmentation_state,
                            output_dir=state["output_dir"],
                        )
                    result = result or await self.tool_executor.execute(tool_id, arguments)
                    attempt_record["success"] = bool(result.success)
                    attempt_record["error_message"] = result.error_message
                    if not result.success:
                        raise RuntimeError(result.error_message or f"Tool failed: {tool_id}")
                    record["success"] = True
                    record["output"] = result.output
                    if attempts:
                        attempts.append(_without_empty(attempt_record))
                        record["attempts"] = attempts
                    return result.output
                except Exception as exc:
                    message = (
                        result.error_message
                        if result is not None and result.error_message
                        else f"{type(exc).__name__}: {exc}"
                    )
                    attempt_record["success"] = False
                    attempt_record["error_message"] = message
                    if result is not None and result.output:
                        attempt_record["output"] = result.output
                    should_retry = attempt_index < max_attempts and _is_transient_tool_failure(
                        tool_id, message
                    )
                    attempt_record["retrying"] = should_retry
                    attempts.append(_without_empty(attempt_record))
                    if should_retry:
                        await asyncio.sleep(_transient_tool_retry_delay_sec())
                        continue
                    record["success"] = False
                    record["error_message"] = message
                    if result is not None:
                        record["output"] = result.output
                    if len(attempts) > 1 or should_retry:
                        record["attempts"] = attempts
                    raise
        except Exception as exc:
            record["success"] = False
            record["error_message"] = record.get("error_message") or f"{type(exc).__name__}: {exc}"
            raise
        finally:
            state.setdefault("tool_invocations", []).append(_without_empty(record))

    async def _handle_speech_enhancement(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        input_audio = _resolve_input_audio(step, state)
        args = {
            "input_audio_path": input_audio,
            "output_path": _output_path(step, state),
            "mode": params.pop("mode", "general"),
            "preserve_background": bool(params.pop("preserve_background", False)),
            "enhancement_goal": params.pop(
                "enhancement_goal", step.description or state["user_prompt"]
            ),
            **params,
        }
        result = dict(await self._execute_tool("speech_enhancement", args, state))
        result["speech_enhancement_contract"] = {
            "input_audio_path": str(args["input_audio_path"]),
            "output_path": str(args["output_path"]),
            "mode": str(args["mode"]),
            "preserve_background": bool(args["preserve_background"]),
            "enhancement_goal": str(args["enhancement_goal"]),
        }
        return result

    async def _handle_edit_data_pipeline(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        operation = step.operation.split(".", 1)[1]
        params.pop("operation", None)
        instruction = params.pop("instruction", state["user_prompt"])
        input_audio = _resolve_input_audio(step, state)
        output_path = _output_path(step, state)
        source_text = params.pop("source_text", None) or state.get("source_text")
        language = canonicalize_language(
            params.pop("language", None) or _language_from_perception(state)
        )
        if operation == "audio_assembly":
            _merge_audio_assembly_input_fields(step, state, params)
        args = build_edit_data_pipeline_tool_arguments(
            operation=operation,
            instruction=str(instruction),
            source_text=str(source_text) if source_text is not None else None,
            input_audio_path=input_audio,
            output_path=output_path,
            output_dir=state["output_dir"],
            language=language,
            params=params,
        )
        if operation in {
            "text_edit",
            "pitch_rate_edit",
            "emotion_style_edit",
            "pause_edit",
        } and not args.get("instruction_st"):
            raise RuntimeError(
                f"edit_data_pipeline.{operation} requires a tagged instruction_st; "
                "the current request could not be converted into the service-layer edit format"
            )
        if operation == "volume_normalize":
            args.setdefault("method", "rms")
            args.setdefault("target_db", -20.0)
            args.setdefault("headroom_db", -1.0)
        if operation == "speech_rate":
            args.setdefault("method", "wsola+indextts2")
            args.setdefault("speaker_audio_path", args["input_audio_path"])
            args["allow_method_fallback"] = _context_bool(
                state["context"],
                "allow_speech_rate_method_fallback",
                default=False,
            )
            if source_text:
                args.setdefault("text", source_text)
        if operation == "pitch_rate_edit":
            args.setdefault("pitch_method", "psola+indextts2_cco")
            args.setdefault("smooth_policy", "never")
        result = await self._execute_tool("edit_data_pipeline", args, state)
        if (
            operation == "text_edit"
            and isinstance(result, dict)
            and result.get("source_spans")
            and result.get("target_spans")
        ):
            result = dict(result)
            result["identity_outside_regions"] = True
        if operation == "volume_normalize":
            result = dict(result)
            result["volume_normalize_contract"] = {
                "input_audio_path": str(args["input_audio_path"]),
                "method": str(args["method"]),
                "target_db": float(args["target_db"]),
                "headroom_db": float(args["headroom_db"]),
            }
        return result

    async def _handle_emphasis_stress_edit(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or state.get("user_prompt") or step.description or ""
        )
        input_audio = _resolve_input_audio(step, state)
        output_path = _output_path(step, state)
        work_dir = Path(state["output_dir"]) / step.id
        work_dir.mkdir(parents=True, exist_ok=True)

        perception = state.get("perception_report") or {}
        source_text = (
            _pop_nonempty_string(params, "source_text")
            or state.get("source_text")
            or perception.get("source_text")
        )
        if not source_text or not _timestamp_words_from_perception(perception):
            if not _implicit_executor_repair_enabled(state):
                raise RuntimeError(
                    "emphasis_stress_edit requires source_text and word timestamps; "
                    "strict forward execution forbids implicit perception repair"
                )
            perception_result = build_audio_perception_fallback(
                agentic_mode=_perception_provider_mode(step, state, self.agentic_mode),
                speech_client=self.speech_perception_client,
            ).perceive(
                user_prompt=instruction,
                audio_path=input_audio,
                base_perception=perception,
                source_text=source_text,
                reference_audio_paths=list(state.get("reference_audio_paths") or []),
                output_dir=str(work_dir / "perception"),
                requirements=PerceptionRequirements(
                    source_text=True,
                    word_timestamps=True,
                    exact_word_timestamps=True,
                    speaker_segments=False,
                    speaker_profiles=False,
                    segment_descriptions=False,
                ),
                run_specialized=bool(params.pop("run_specialized_perception", True)),
                allow_gemini=bool(params.pop("allow_gemini_perception", True)),
                force_gemini=bool(params.pop("force_gemini_perception", False)),
                refresh_cache=bool(params.pop("refresh_perception_cache", False)),
                strict=False,
            )
            perception = perception_result.perception
            source_text = source_text or perception.get("source_text")
            state["perception_report"] = perception
            if source_text:
                state["source_text"] = source_text
            state.setdefault("emphasis_stress_perception_repairs", {})[step.id] = {
                "status": "refreshed",
                "source_text": source_text,
                "word_timestamp_count": len(_timestamp_words_from_perception(perception)),
                "satisfied": perception_result.satisfied,
                "missing": perception_result.missing,
            }

        if not isinstance(source_text, str) or not source_text.strip():
            raise RuntimeError(
                "emphasis_stress_edit requires source_text from params, context, or perception"
            )
        source_text = source_text.strip()

        target_phrase = _emphasis_target_phrase(
            params, instruction=instruction, source_text=source_text
        )
        if not target_phrase:
            raise RuntimeError(
                "emphasis_stress_edit requires a quoted or tagged target word/phrase to stress"
            )
        all_occurrences = _requests_all_target_occurrences(
            " ".join(str(value or "") for value in (state.get("user_prompt"), instruction)),
            params,
        )
        edit_tagged_text = _target_tagged_text_for_emphasis(
            source_text,
            target_phrase,
            all_occurrences=all_occurrences,
        )
        if edit_tagged_text == source_text:
            raise RuntimeError(
                f"emphasis_stress_edit could not find target phrase {target_phrase!r} in source_text"
            )

        language = canonicalize_language(
            params.pop("language", None) or _language_from_perception(state)
        )
        target_tts_path = str(work_dir / "01_qwen3_strong_emphasis_target.wav")
        seedvc_target_path = str(work_dir / "02_seedvc_source_speaker_target.wav")
        qwen_prompt = str(
            params.pop("qwen3_tts_instruct", None)
            or params.pop("target_voice_prompt", None)
            or _qwen3_emphasis_prompt(target_phrase, instruction)
        )
        qwen_args = {
            "text": source_text,
            "output_path": target_tts_path,
            "mode": "custom_voice",
            "instruct": qwen_prompt,
            "language": language or "Auto",
        }
        for key in (
            "speaker",
            "voice_design_backend",
            "temperature",
            "top_p",
            "max_new_tokens",
            "seed",
        ):
            if key in params:
                qwen_args[key] = params.pop(key)
        qwen_result = await self._execute_tool("qwen3_tts", qwen_args, state)
        generated_target = _extract_output_path(qwen_result) or target_tts_path

        seed_vc_args = {
            "source_audio_path": generated_target,
            "reference_audio_path": input_audio,
            "output_path": seedvc_target_path,
        }
        seed_vc_params = params.pop("seed_vc_params", None)
        if isinstance(seed_vc_params, dict):
            seed_vc_args.update(seed_vc_params)
        for key in ("diffusion_steps", "length_adjust", "inference_cfg_rate"):
            if key in params:
                seed_vc_args[key] = params.pop(key)
        seedvc_result = await self._execute_tool("seed_vc", seed_vc_args, state)
        speaker_matched_target = _extract_output_path(seedvc_result) or seedvc_target_path

        source_spans = _emphasis_source_target_spans(
            perception, target_phrase, all_occurrences=all_occurrences
        )
        assembly_args: dict[str, Any] = {
            "operation": "audio_assembly",
            "source_audio_path": input_audio,
            "target_audio_path": speaker_matched_target,
            "output_path": output_path,
            "edit_tagged_text": edit_tagged_text,
            "preserve_outer_boundaries": bool(params.pop("preserve_outer_boundaries", True)),
            "aligner_backend": str(params.pop("aligner_backend", "qwen3")),
            "language": language,
            "smoothing": params.pop("smoothing", _default_emphasis_assembly_smoothing()),
            "debug": bool(params.pop("debug", False)),
            "trace_id": str(params.pop("trace_id", step.id)),
        }
        debug_dir = params.pop("debug_dir", None)
        if debug_dir:
            assembly_args["debug_dir"] = str(debug_dir)
        elif assembly_args["debug"]:
            assembly_args["debug_dir"] = str(work_dir / "audio_assembly_debug")
        manual_source = params.pop("time_spans_from_source", None)
        manual_target = params.pop("time_spans_from_target", None)
        if manual_source and manual_target:
            assembly_args["time_spans_from_source"] = manual_source
            assembly_args["time_spans_from_target"] = manual_target
        elif params.pop("allow_source_span_metadata_only", True):
            # The audio-assembly service can align both source and target from
            # edit_tagged_text. Keep source spans in metadata for audit rather
            # than passing a partial manual span set that would be ignored.
            pass
        assembly_args.update({key: value for key, value in params.items() if value is not None})

        try:
            assembly_result = await self._execute_tool("edit_data_pipeline", assembly_args, state)
        except Exception as exc:
            if not _implicit_executor_repair_enabled(state):
                raise
            fallback_enabled = str(
                step.params.get("fallback_operation") or "edit_data_pipeline.pitch_rate_edit"
            )
            if fallback_enabled != "edit_data_pipeline.pitch_rate_edit":
                raise
            fallback_result = await self._execute_emphasis_pitch_rate_fallback(
                step=step,
                state=state,
                input_audio=input_audio,
                output_path=output_path,
                source_text=source_text,
                target_phrase=target_phrase,
                instruction=instruction,
                language=language,
                original_error=f"{type(exc).__name__}: {exc}",
            )
            return fallback_result

        assembled_output = _extract_output_path(assembly_result) or output_path
        assembly_source_spans, assembly_output_spans = _audio_assembly_temporal_spans(
            assembly_result,
            source_duration_sec=_audio_duration_seconds(Path(input_audio)),
            output_duration_sec=_audio_duration_seconds(Path(assembled_output)),
        )

        return {
            "operation": "emphasis_stress_edit",
            "output_audio_path": assembled_output,
            "source_audio_path": input_audio,
            "source_text": source_text,
            "target_phrase": target_phrase,
            "edit_tagged_text": edit_tagged_text,
            "source_target_spans": source_spans,
            "source_spans": assembly_source_spans,
            "target_spans": assembly_output_spans,
            "identity_outside_regions": True,
            "qwen3_tts_target": qwen_result,
            "seed_vc_conversion": seedvc_result,
            "audio_assembly": assembly_result,
            "method": (
                "P1 local resynthesis + SeedVC speaker recovery + "
                "edit_data_pipeline.audio_assembly F5 splice normalization"
            ),
        }

    async def _execute_emphasis_pitch_rate_fallback(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        input_audio: str,
        output_path: str,
        source_text: str,
        target_phrase: str,
        instruction: str,
        language: str | None,
        original_error: str,
    ) -> dict[str, Any]:
        fallback_params = step.params.get("_pitch_rate_fallback_params")
        if isinstance(fallback_params, dict):
            instruction_st = fallback_params.get("instruction_st")
        else:
            instruction_st = None
        if not isinstance(instruction_st, str) or not instruction_st.strip():
            instruction_st = _pitch_rate_fallback_instruction_st(source_text, target_phrase)
        args = build_edit_data_pipeline_tool_arguments(
            operation="pitch_rate_edit",
            instruction=instruction,
            source_text=source_text,
            input_audio_path=input_audio,
            output_path=output_path,
            output_dir=state["output_dir"],
            language=language,
            params={
                "instruction_st": instruction_st,
                "pitch_method": "psola+indextts2_cco",
                "smooth_policy": "never",
            },
        )
        state.setdefault("emphasis_stress_fallbacks", {})[step.id] = {
            "from": "emphasis_stress_edit",
            "to": "edit_data_pipeline.pitch_rate_edit",
            "reason": original_error,
            "target_phrase": target_phrase,
        }
        result = await self._execute_tool("edit_data_pipeline", args, state)
        return {
            "operation": "emphasis_stress_edit_fallback_pitch_rate",
            "output_audio_path": _extract_output_path(result) or output_path,
            "source_audio_path": input_audio,
            "source_text": source_text,
            "target_phrase": target_phrase,
            "fallback_reason": original_error,
            "pitch_rate_fallback": result,
        }

    async def _handle_seed_vc(self, step: AudioEditStep, state: dict[str, Any]) -> dict[str, Any]:
        params = dict(step.params)
        refs = state.get("reference_audio_paths") or []
        source_audio = _resolve_input_audio(step, state)
        reference_audio = (
            params.pop("reference_audio_path", None)
            or params.pop("reference_audio", None)
            or step.inputs.get("reference_audio_path")
            or step.inputs.get("reference_audio")
            or (refs[0] if refs else None)
        )
        if not reference_audio and _requests_original_voice_reference(step, state):
            reference_audio = _original_voice_reference_audio(state, exclude=source_audio)
        if not reference_audio:
            raise RuntimeError("seed_vc requires reference_audio_path")
        reference_audio = require_existing_audio_path(
            reference_audio, "seed_vc reference_audio_path", state
        )
        if _same_audio_path(reference_audio, source_audio) and _requests_original_voice_reference(
            step, state
        ):
            original_reference = _original_voice_reference_audio(state, exclude=source_audio)
            if original_reference:
                reference_audio = require_existing_audio_path(
                    original_reference, "seed_vc reference_audio_path", state
                )
        args = {
            "source_audio_path": source_audio,
            "reference_audio_path": reference_audio,
            "output_path": _output_path(step, state),
            **params,
        }
        return await self._execute_tool("seed_vc", args, state)

    async def _handle_no_reference_voice_conversion(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._execute_no_reference_voice_conversion(
            step, state, params=dict(step.params)
        )

    async def _handle_qwen3_tts(self, step: AudioEditStep, state: dict[str, Any]) -> dict[str, Any]:
        params = dict(step.params)
        mode = params.pop("mode", "custom_voice")
        if mode == "voice_clone":
            ref_audio = _pop_qwen3_tts_voice_clone_reference(params, step.inputs)
            if ref_audio:
                params["ref_audio_path"] = ref_audio
                params.setdefault("x_vector_only_mode", True)
        voice_design_backend = params.pop("voice_design_backend", None)
        voice_class_context = _voice_class_context(step, state, params)
        voice_class = (
            _voice_class_profile(voice_class_context, params) if mode == "custom_voice" else None
        )
        voice_class_postprocess = _voice_class_audio_postprocess_options(params)
        if mode == "custom_voice" and not voice_design_backend:
            voice_design_backend = os.getenv("QWEN3_TTS_VOICE_DESIGN_BACKEND") or os.getenv(
                "X_EDITOR_VOICE_DESIGN_BACKEND"
            )
        if mode == "custom_voice" and _requests_existing_audio_voice_conversion(
            step, state, params
        ):
            conversion_params = dict(params)
            conversion_params["voice_design_backend"] = voice_design_backend or "qwen3_tts"
            conversion_params.setdefault(
                "voice_prompt",
                conversion_params.get("instruct") or step.description or state["user_prompt"],
            )
            return await self._execute_no_reference_voice_conversion(
                step, state, params=conversion_params
            )
        text = params.pop("text", None) or state.get("source_text")
        if not text:
            raise RuntimeError("qwen3_tts requires text")
        args = {
            "text": text,
            "output_path": _output_path(step, state),
            "mode": mode,
            "instruct": params.pop("instruct", step.description or state["user_prompt"]),
            "language": params.pop("language", "Auto"),
            **params,
        }
        if voice_design_backend:
            args["voice_design_backend"] = voice_design_backend
        result = await self._execute_tool("qwen3_tts", args, state)
        return _apply_voice_class_postprocess_to_result(
            result,
            state=state,
            step=step,
            voice_class=voice_class,
            options=voice_class_postprocess,
        )


    async def _execute_no_reference_voice_conversion(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        source_audio = _resolve_input_audio(step, state)
        output_path = _output_path(step, state)
        work_dir = Path(state["output_dir"]) / step.id
        work_dir.mkdir(parents=True, exist_ok=True)
        backend = _voice_design_reference_backend(params, state)
        request = str(
            params.pop("instruction", None)
            or params.get("voice_prompt")
            or params.get("instruct")
            or step.description
            or state["user_prompt"]
        )
        request_context = _voice_class_context(step, state, {**params, "instruction": request})
        reference_path = str(
            Path(
                params.pop("reference_output_path", None)
                or (work_dir / f"{backend}_target_voice_reference.wav")
            )
        )
        seed_vc_params = _seed_vc_params_from_no_ref_voice_params(params)
        voice_design_params = _voice_design_params_from_no_ref_voice_params(params)
        raw_prompt_hint = str(params.get("voice_prompt") or params.get("instruct") or "")
        prompt_context = f"{raw_prompt_hint} {request_context} {_voice_design_prompt_hint_from_params(voice_design_params)}"
        voice_class = _voice_class_profile(prompt_context, voice_design_params)
        voice_class_reference_postprocess = _voice_class_audio_postprocess_options(
            params, voice_design_params
        )
        _apply_target_voice_reference_defaults(
            prompt_context, backend, voice_design_params, voice_class=voice_class
        )
        source_gender_preservation = _source_gender_preservation_for_voice_design(
            state,
            prompt_context=prompt_context,
            params=voice_design_params,
            voice_class=voice_class,
        )
        reference_text = str(
            params.pop("reference_text", None)
            or params.pop("target_voice_reference_text", None)
            or _independent_voice_reference_text(
                backend, request=request_context, voice_class=voice_class
            )
        )
        raw_qwen_prompt = params.pop("instruct", None) or params.pop("voice_prompt", None)
        qwen_prompt = (
            str(raw_qwen_prompt)
            if _is_descriptive_voice_prompt(raw_qwen_prompt)
            else _qwen3_reference_voice_prompt(
                f"{raw_qwen_prompt or ''} {request_context} {_voice_design_prompt_hint_from_params(voice_design_params)}"
            )
        )
        qwen_prompt = _strengthen_voice_class_prompt(
            qwen_prompt,
            backend="qwen3_tts",
            voice_class=voice_class,
            gender_preservation=source_gender_preservation,
        )
        reference_result = await self._execute_tool(
            "qwen3_tts",
            {
                "text": reference_text,
                "output_path": reference_path,
                "mode": "custom_voice",
                "instruct": qwen_prompt,
                "language": voice_design_params.pop("language", "Auto"),
                **voice_design_params,
            },
            state,
        )
        voice_design_prompt = qwen_prompt
        target_reference = _extract_output_path(reference_result) or reference_path
        raw_target_reference = target_reference
        reference_postprocess_result: dict[str, Any] | None = None
        if voice_class in {"child", "elderly"}:
            reference_postprocess_result = _maybe_postprocess_voice_class_audio(
                target_reference,
                work_dir=work_dir,
                voice_class=voice_class,
                options=voice_class_reference_postprocess,
            )
            target_reference = (
                reference_postprocess_result.get("output_audio_path") or target_reference
            )
        reference_resynthesis_result: dict[str, Any] | None = None
        if reference_postprocess_result and reference_postprocess_result.get("output_audio_path"):
            reference_resynthesis_result = await self._maybe_resynthesize_voice_class_reference(
                reference_postprocess_result["output_audio_path"],
                work_dir=work_dir,
                voice_class=voice_class,
                reference_text=reference_text,
                options=voice_class_reference_postprocess,
                state=state,
            )
            target_reference = (
                reference_resynthesis_result.get("output_audio_path") or target_reference
            )
        converted_result = await self._execute_tool(
            "seed_vc",
            {
                "source_audio_path": source_audio,
                "reference_audio_path": target_reference,
                "output_path": output_path,
                **seed_vc_params,
            },
            state,
        )
        converted_output = _extract_output_path(converted_result) or output_path
        return {
            "output_audio_path": converted_output,
            "source_audio_path": source_audio,
            "target_voice_reference_audio_path": target_reference,
            "target_voice_reference_raw_audio_path": raw_target_reference,
            "target_voice_reference_text": reference_text,
            "voice_design_backend": backend,
            "voice_design_prompt": voice_design_prompt,
            "target_voice_class": voice_class,
            "source_gender_preservation": source_gender_preservation,
            "voice_design_memory": _voice_design_memory_evidence(voice_class),
            "source_audio_independent_reference": True,
            "target_voice_reference_postprocess": reference_postprocess_result,
            "target_voice_reference_resynthesis": reference_resynthesis_result,
            "reference_generation": reference_result,
            "seed_vc_conversion": converted_result,
        }

    async def _maybe_resynthesize_voice_class_reference(
        self,
        input_audio_path: str,
        *,
        work_dir: Path,
        voice_class: str | None,
        reference_text: str,
        options: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        input_path = Path(str(input_audio_path)).expanduser()
        backend = _voice_class_reference_resynthesis_backend(options)
        result: dict[str, Any] = {
            "operation": "voice_class_reference_resynthesis",
            "voice_class": voice_class,
            "enabled": bool(options.get("resynthesis_enabled", True)),
            "backend": backend,
            "input_audio_path": str(input_path),
            "text": reference_text,
        }
        if voice_class not in {"child", "elderly"}:
            result["skipped"] = True
            result["reason"] = "unsupported_voice_class"
            return result
        if not result["enabled"]:
            result["skipped"] = True
            result["reason"] = "disabled"
            return result
        if backend in {"none", "off", "disabled"}:
            result["skipped"] = True
            result["reason"] = "backend_disabled"
            return result
        if backend not in {
            "edit_data_pipeline_speech_rate",
            "speech_rate",
            "indextts2",
            "wsola_indextts2",
        }:
            result["skipped"] = True
            result["reason"] = f"unsupported_backend:{backend}"
            return result
        if not input_path.exists():
            result["skipped"] = True
            result["reason"] = "input_audio_missing"
            return result
        if not str(reference_text or "").strip():
            result["skipped"] = True
            result["reason"] = "reference_text_missing"
            return result

        method = (
            str(options.get("resynthesis_method") or "wsola+indextts2").strip() or "wsola+indextts2"
        )
        output_path = work_dir / f"{input_path.stem}_{voice_class}_neural_resynth.wav"
        args = {
            "operation": "speech_rate",
            "input_audio_path": str(input_path),
            "output_path": str(output_path),
            "rate": 1.0,
            "method": method,
            "text": reference_text,
            "speaker_audio_path": str(input_path),
            "instruction": "Resynthesize the DSP-adjusted target voice reference to improve naturalness before SeedVC.",
        }
        if options.get("resynthesis_service_url"):
            args["service_url"] = options["resynthesis_service_url"]
        result.update(
            {
                "output_audio_path": str(output_path),
                "method": method,
                "rate": 1.0,
                "tool_id": "edit_data_pipeline",
                "tool_operation": "speech_rate",
            }
        )
        try:
            tool_result = await self._execute_tool("edit_data_pipeline", args, state)
        except Exception as exc:
            if bool(options.get("resynthesis_required", False)):
                raise
            result.pop("output_audio_path", None)
            result["skipped"] = True
            result["reason"] = f"resynthesis_tool_failed: {type(exc).__name__}: {exc}"
            return result
        output = _extract_output_path(tool_result) or str(output_path)
        if not Path(str(output)).exists():
            if bool(options.get("resynthesis_required", False)):
                raise RuntimeError(
                    f"voice-class reference resynthesis did not produce output audio: {output}"
                )
            result.pop("output_audio_path", None)
            result["skipped"] = True
            result["reason"] = "resynthesis_output_missing"
            result["tool_result"] = tool_result
            return result
        result["output_audio_path"] = str(output)
        result["tool_result"] = tool_result
        result["skipped"] = False
        return result

    async def _handle_audio_effects(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = params.pop("instruction", None) or step.description or state["user_prompt"]
        refs = state.get("reference_audio_paths") or []
        wants_reference_match = _requests_reference_channel_match(
            " ".join(str(value) for value in [instruction, state.get("user_prompt")] if value)
        )
        reference_audio = (
            params.pop("reference_audio_path", None)
            or step.inputs.get("reference_audio_path")
            or step.inputs.get("reference_audio")
            or (refs[0] if wants_reference_match and refs else None)
        )
        if reference_audio:
            reference_audio = require_existing_audio_path(
                reference_audio, "audio_effects reference_audio_path", state
            )
        if wants_reference_match and reference_audio:
            params.setdefault("preset", "match_channel_reference")
            params.setdefault("strength", 0.88)
        if not params.get("preset"):
            preset = canonical_audio_effect_preset(
                " ".join(
                    str(value)
                    for value in [
                        params.get("effect"),
                        instruction,
                        step.description,
                        state.get("user_prompt"),
                    ]
                    if value
                )
            )
            if preset:
                params["preset"] = preset
        args = {
            "input_audio_path": _resolve_input_audio(step, state),
            "output_path": _output_path(step, state),
            "instruction": instruction,
            "reference_audio_path": reference_audio,
            **params,
        }
        return await self._execute_tool("audio_effects", args, state)

    async def _handle_seamless_expressive_s2st(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        target_language = _seamless_language_code(
            params.pop("target_language", None)
            or params.pop("target_lang", None)
            or params.pop("language", None)
            or _target_language_from_request(state.get("user_prompt") or step.description)
        )
        if not target_language:
            raise RuntimeError("seamless_expressive_s2st requires target_language")
        source_language = _seamless_language_code(
            params.pop("source_language", None)
            or params.pop("source_lang", None)
            or _language_from_perception(state)
        )
        args = {
            "input_audio_path": _resolve_input_audio(step, state),
            "target_language": target_language,
            "output_path": _output_path(step, state),
            "mode": params.pop("mode", "auto"),
            "preserve_vocal_style": bool(params.pop("preserve_vocal_style", True)),
            "preserve_speech_rate": bool(params.pop("preserve_speech_rate", True)),
            "preserve_pauses": bool(params.pop("preserve_pauses", True)),
            "allow_fallback_to_m4t": bool(params.pop("allow_fallback_to_m4t", False)),
            "duration_factor": params.pop("duration_factor", 1.0),
            "timeout_seconds": params.pop("timeout_seconds", 1800.0),
            **params,
        }
        if source_language:
            args["source_language"] = source_language
        try:
            return await self._execute_tool("seamless_expressive_s2st", args, state)
        except RuntimeError as exc:
            if not _implicit_executor_repair_enabled(
                state
            ) or not _seamless_error_allows_text_tts_fallback(exc):
                raise
            return await self._fallback_seamless_to_qwen_voice_clone(
                step,
                state,
                args=args,
                target_language=target_language,
                source_language=source_language,
                unavailable_error=exc,
            )

    async def _fallback_seamless_to_qwen_voice_clone(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
        *,
        args: dict[str, Any],
        target_language: str,
        source_language: str | None,
        unavailable_error: Exception,
    ) -> dict[str, Any]:
        if "qwen3_tts" not in self.registry.list_operations():
            raise RuntimeError(
                "seamless_expressive_s2st is unavailable and qwen3_tts fallback is not registered"
            ) from unavailable_error
        input_audio = str(args["input_audio_path"])
        target_text = (
            step.params.get("target_text")
            or step.params.get("text")
            or args.get("target_text")
            or _target_text_from_step_params(step)
        )
        if not isinstance(target_text, str) or not target_text.strip():
            target_text = _translate_audio_or_text_for_s2st_fallback(
                input_audio_path=input_audio,
                source_text=state.get("source_text"),
                target_language=target_language,
                source_language=source_language,
                request=state.get("user_prompt") or step.description,
                agentic_mode=self.agentic_mode,
            )
        if not isinstance(target_text, str) or not target_text.strip():
            raise RuntimeError(
                "seamless_expressive_s2st is unavailable and fallback could not obtain target_text "
                "from planner params, perception transcript, or configured provider translation"
            ) from unavailable_error

        language = _language_name_from_code(
            target_language
        ) or _qwen3_tts_language_from_text_and_request(
            target_text,
            state.get("user_prompt") or step.description,
        )
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "seamless_expressive_s2st",
            "to": "qwen3_tts",
            "reason": f"Seamless runtime unavailable: {type(unavailable_error).__name__}: {unavailable_error}",
            "target_language": target_language,
            "source_language": source_language,
            "fallback_target_text": target_text,
        }
        result = await self._execute_tool(
            "qwen3_tts",
            {
                "text": target_text,
                "output_path": args["output_path"],
                "mode": "voice_clone",
                "ref_audio_path": input_audio,
                "language": language,
                "instruct": (
                    "Translate the source speech into the requested target language and synthesize it "
                    "using the input audio as the speaker timbre reference. Preserve the same speaker "
                    "identity as much as this fallback path allows."
                ),
                "x_vector_only_mode": True,
            },
            state,
        )
        output_audio_path = _extract_output_path(result) or args["output_path"]
        return {
            "output_audio_path": output_audio_path,
            "operation": "seamless_expressive_s2st_fallback_qwen3_tts",
            "target_text": target_text,
            "target_language": target_language,
            "source_language": source_language,
            "fallback_reason": str(unavailable_error),
            "qwen3_tts_result": result,
        }

    async def _handle_tse(self, step: AudioEditStep, state: dict[str, Any]) -> dict[str, Any]:
        params = dict(step.params)
        if _tse_step_should_use_timeline_vocal_extraction(step, state, params):
            instruction = str(
                params.pop("instruction", None) or step.description or state["user_prompt"]
            )
            base_audio = _base_input_audio_path(state) or _resolve_input_audio(step, state)
            selected_spans = _selected_spans_from_step_outputs(state)
            extract_params = {
                "instruction": instruction,
            }
            if selected_spans:
                extract_params["selected_spans"] = selected_spans
            state.setdefault("operation_rewrites", {})[step.id] = {
                "from": "tse",
                "to": "artifact_ops.extract_spans",
                "reason": (
                    "existing singing/vocal extraction from the same audio should use timeline spans; "
                    "TSE is reference-enrollment extraction and can hang or overfit when the reference was "
                    "itself cut from the same source"
                ),
                "input_audio_path": base_audio,
                "selected_span_count": len(selected_spans),
            }
            return execute_artifact_step(
                operation="extract_spans",
                instruction=instruction,
                input_audio_path=base_audio,
                output_path=_output_path(step, state),
                output_dir=str(
                    Path(state["output_dir"]) / f"{step.id}_tse_vocal_extraction_rewrite"
                ),
                perception_report=state.get("perception_report") or {},
                source_text=state.get("source_text"),
                reference_audio_paths=state.get("reference_audio_paths") or [],
                params={**extract_params, "_agentic_mode": self.agentic_mode},
            )
        prior_extract = _tse_prior_speech_extract_passthrough(step, state, params)
        if prior_extract:
            source_audio = str(prior_extract["output_audio_path"])
            output_path = _output_path(step, state)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if Path(source_audio).resolve() != Path(output_path).resolve():
                shutil.copy2(source_audio, output_path)
            state.setdefault("operation_rewrites", {})[step.id] = {
                "from": "tse",
                "to": "prior_artifact_ops.extract_spans_output",
                "reason": prior_extract["reason"],
                "prior_output_audio_path": source_audio,
                "selected_span_count": prior_extract.get("selected_span_count", 0),
            }
            return {
                "output_audio_path": output_path,
                "operation": "tse_reused_prior_speech_extract",
                "source_audio_path": source_audio,
                "skipped": True,
                "skip_reason": prior_extract["reason"],
                "selected_spans": prior_extract.get("selected_spans", []),
            }
        refs = state.get("reference_audio_paths") or []
        reference_audio = (
            params.pop("reference_audio_path", None)
            or step.inputs.get("reference_audio_path")
            or (refs[0] if refs else None)
        )
        if not reference_audio:
            raise RuntimeError("tse requires reference_audio_path")
        reference_audio = require_existing_audio_path(
            reference_audio, "tse reference_audio_path", state
        )
        mixture_audio = _resolve_input_audio(step, state)
        base_audio = _base_input_audio_path(state)
        if base_audio:
            try:
                if Path(mixture_audio).resolve() == Path(reference_audio).resolve():
                    mixture_audio = str(base_audio)
                    state.setdefault("operation_normalization", {}).setdefault(step.id, {})[
                        "tse_mixture_audio_repaired_from_reference_artifact"
                    ] = mixture_audio
            except OSError:
                pass
        args = {
            "mixture_audio_path": mixture_audio,
            "reference_audio_path": reference_audio,
            "output_path": _output_path(step, state),
            "enroll_duration": params.pop("enroll_duration", 5.0),
            **params,
        }
        return await self._execute_tool("tse", args, state)

    async def _handle_speech_separation(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        args = {
            "input_audio_path": _resolve_input_audio(step, state),
            "output_dir": params.pop(
                "output_dir", str(Path(state["output_dir"]) / f"{step.id}_stems")
            ),
            "output_path": _output_path(step, state),
            "model_family": params.pop("model_family", "bs_roformer"),
            "output_format": params.pop("output_format", "WAV"),
            "stem_policy": params.pop("stem_policy", "vocals"),
            **params,
        }
        result = dict(await self._execute_tool("speech_separation", args, state))
        result["speech_separation_contract"] = {
            "input_audio_path": str(args["input_audio_path"]),
            "output_dir": str(args["output_dir"]),
            "output_path": str(args["output_path"]),
            "model_family": str(args["model_family"]),
            "output_format": str(args["output_format"]),
            "stem_policy": str(args["stem_policy"]),
        }
        if args.get("model_filename") is not None:
            result["speech_separation_contract"]["model_filename"] = str(args["model_filename"])
        return result

    async def _handle_foreground_background_remix(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        params.setdefault("_agentic_mode", self.agentic_mode)
        instruction = str(
            params.pop("instruction", None) or state["user_prompt"] or step.description
        )
        return await execute_foreground_background_remix(
            execute_tool=self._execute_tool,
            input_audio_path=_resolve_input_audio(step, state),
            output_path=_output_path(step, state),
            output_dir=str(Path(state["output_dir"]) / step.id),
            instruction=instruction,
            source_text=params.pop("source_text", None) or state.get("source_text"),
            params=params,
            state=state,
        )

    async def _handle_paralinguistic_event_add(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        return await execute_paralinguistic_event_add(
            execute_tool=self._execute_tool,
            input_audio_path=_resolve_input_audio(step, state),
            output_path=_output_path(step, state),
            output_dir=state["output_dir"],
            instruction=instruction,
            source_text=params.get("source_text") or state.get("source_text"),
            params=params,
            state=state,
        )

    async def _handle_paralinguistic_event_remove(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        return await execute_paralinguistic_event_remove(
            execute_tool=self._execute_tool,
            input_audio_path=_resolve_input_audio(step, state),
            output_path=_output_path(step, state),
            output_dir=state["output_dir"],
            instruction=instruction,
            params=params,
            state=state,
        )

    async def _handle_ace_step_music_timeline_edit(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None)
            or params.get("selection_instruction")
            or _selector_text(params.get("selector"))
            or step.description
            or state["user_prompt"]
        )
        return await execute_music_vocal_timeline_edit(
            execute_tool=self._execute_tool,
            input_audio_path=_resolve_input_audio(step, state),
            output_path=_output_path(step, state),
            output_dir=str(Path(state["output_dir"]) / step.id),
            instruction=instruction,
            source_text=params.pop("source_text", None) or state.get("source_text"),
            perception_report=state.get("perception_report") or {},
            reference_audio_paths=state.get("reference_audio_paths") or [],
            params={**params, "_agentic_mode": self.agentic_mode},
            state=state,
        )

    def _handle_audio_perception_fallback(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        input_audio = _resolve_input_audio(step, state)
        force_gemini = bool(params.pop("force_gemini", step.operation == "perception.gemini_audio"))
        strict = bool(params.pop("strict", True))
        lightweight_speech_context = step.operation == "perception.speech_context"
        need_words = bool(params.pop("need_words", False if lightweight_speech_context else True))
        require_exact_word_timestamps = bool(params.pop("require_exact_word_timestamps", False))
        need_speakers = bool(
            params.pop("need_speakers", False if lightweight_speech_context else True)
        )
        need_source_text = bool(
            params.pop("need_source_text", True if lightweight_speech_context else need_words)
        )
        observe_source_text = bool(
            params.pop("observe_source_text", need_source_text)
        )
        need_profiles = bool(params.pop("need_speaker_profiles", need_speakers))
        need_segment_descriptions = bool(params.pop("need_segment_descriptions", False))
        if (
            need_words
            and not require_exact_word_timestamps
            and need_speakers
            and _perception_step_can_use_segment_grounding(instruction)
        ):
            state.setdefault("perception_requirement_relaxations", {})[step.id] = {
                "from": {"need_words": True},
                "to": {"need_words": False},
                "reason": (
                    "speaker/role/accent scoped extraction can be grounded by speaker segments "
                    "and profile descriptions; exact word timestamps are not required"
                ),
                "instruction": instruction,
            }
            need_words = False
        reference_audio_paths = (
            params.pop("reference_audio_paths", None) or state.get("reference_audio_paths") or []
        )
        reference_audio_paths = resolve_audio_reference(reference_audio_paths, state)
        base_audio_ref, reference_audio_refs, reference_audio_paths = (
            _perception_invocation_audio_bindings(
                state,
                base_audio_path=input_audio,
                fallback_reference_audio_paths=list(reference_audio_paths),
            )
        )
        reuse_existing = bool(params.pop("reuse_existing_perception", False))
        isolated_gateway_request = bool(
            (state.get("context") or {}).get(
                "_workflow_perception_isolated_request",
                False,
            )
        )
        refresh_for_new_audio = not reuse_existing and (
            isolated_gateway_request or _perception_input_differs_from_base(input_audio, state)
        )
        base_perception = {} if refresh_for_new_audio else (state.get("perception_report") or {})
        source_text = None if refresh_for_new_audio else state.get("source_text")
        allow_model_perception = params.pop("allow_gemini", None)
        if allow_model_perception is None:
            allow_model_perception = params.pop(
                "allow_open_omni",
                force_gemini or not lightweight_speech_context,
            )
        frontend = params.pop("frontend", None) or self.perception_frontends.get(step.operation)
        run_specialized = bool(params.pop("run_specialized", True))
        recognition_context = params.pop("recognition_context", None)
        result = build_audio_perception_fallback(
            agentic_mode=_perception_provider_mode(step, state, self.agentic_mode),
            speech_client=self.speech_perception_client,
        ).perceive(
            user_prompt=instruction,
            audio_path=input_audio,
            base_perception=base_perception,
            source_text=source_text,
            reference_audio_paths=list(reference_audio_paths),
            base_audio_ref=base_audio_ref,
            reference_audio_refs=reference_audio_refs,
            output_dir=str(Path(state["output_dir"]) / step.id),
            requirements=PerceptionRequirements(
                source_text=need_source_text,
                observe_source_text=observe_source_text,
                word_timestamps=need_words,
                exact_word_timestamps=require_exact_word_timestamps,
                speaker_segments=need_speakers,
                speaker_profiles=need_profiles,
                segment_descriptions=need_segment_descriptions,
            ),
            run_specialized=run_specialized,
            allow_gemini=bool(allow_model_perception),
            force_gemini=force_gemini,
            refresh_cache=bool(params.pop("refresh_cache", refresh_for_new_audio)),
            strict=strict,
            frontend=frontend,
            **({"recognition_context": recognition_context} if recognition_context else {}),
        )
        merged = result.perception
        output = {
            "operation": step.operation,
            "input_audio_path": input_audio,
            "perception_report": merged,
            "source_text": merged.get("source_text"),
            "satisfied": result.satisfied,
            "missing": result.missing,
            "attempts": result.attempts,
            "speaker_count": len(merged.get("speakers") or []),
            "segment_count": len(merged.get("speaker_segments") or merged.get("segments") or []),
            "perception_services": merged.get("services", {}),
            "warnings": merged.get("uncertainty", []),
            "perception_route_id": (
                "ts_asr_grounding"
                if lightweight_speech_context
                else (
                    f"{_reviewer_provider_id(frontend)}_direct"
                    if not run_specialized and bool(allow_model_perception)
                    else f"specialist_evidence_optional_{_reviewer_provider_id(frontend)}"
                )
            ),
        }
        trace = getattr(frontend, "last_invocation_trace", None)
        if trace is not None:
            output["component_invocation_trace"] = trace.model_dump(mode="json")
        elif lightweight_speech_context:
            output["component_invocation_trace"] = ComponentInvocationTrace.build(
                component="perception",
                component_name=step.operation,
                route="deterministic",
                status="completed",
                metadata={
                    "operation": step.operation,
                    "actual_route_id": "ts_asr_grounding",
                    "backend_attempts": _safe_perception_backend_attempts(result.attempts),
                },
            ).model_dump(mode="json")
        return output

    def _handle_speaker_attributed_review(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        input_audio = _resolve_input_audio(step, state)
        language = canonicalize_language(
            params.pop("language", None)
            or _language_from_perception(state)
            or state.get("language")
            or (state.get("context") or {}).get("language")
        )
        isolated_gateway_request = bool(
            (state.get("context") or {}).get(
                "_workflow_perception_isolated_request",
                False,
            )
        )
        review_kwargs: dict[str, Any] = {
            "audio_path": input_audio,
            "instruction": instruction,
            "output_dir": Path(state["output_dir"]) / step.id,
            "base_perception": (
                {} if isolated_gateway_request else (state.get("perception_report") or {})
            ),
            "run_specialized": bool(params.pop("run_specialized", True)),
            "use_sedicow": bool(params.pop("use_sedicow", True)),
            "max_review_attempts": int(params.pop("max_review_attempts", 2)),
            "strict": bool(params.pop("strict", True)),
            "refresh_cache": bool(params.pop("refresh_cache", False)),
            "language": language,
            "num_speakers": params.pop("num_speakers", None),
            "agentic_mode": _perception_provider_mode(step, state, self.agentic_mode),
            "use_temporal_reasoning": bool(params.pop("use_temporal_reasoning", False)),
            "use_specialist_prior": bool(params.pop("use_specialist_prior", True)),
            "enable_omni_review": bool(params.pop("enable_omni_review", True)),
            "supplement_speaker_attributes": bool(params.pop("supplement_speaker_attributes", False)),
            "observation_needs": params.pop("observation_needs", None),
        }
        explicit_backend = params.pop("speaker_evidence_backend", None)
        if explicit_backend is not None:
            review_kwargs["speaker_evidence_backend"] = str(explicit_backend)
        reviewer = params.pop("reviewer", None) or self.perception_frontends.get(step.operation)
        if reviewer is not None:
            reset_frontend_invocation_trace(reviewer)
            review_kwargs["reviewer"] = reviewer
        try:
            result = run_speaker_attributed_review(**review_kwargs)
        except Exception:
            if reviewer is not None:
                finalize_frontend_invocation_trace(reviewer, status="failed")
            raise
        trace = (
            finalize_frontend_invocation_trace(reviewer, status="completed")
            if reviewer is not None
            else None
        )
        review = result.get("speaker_attributed_review") or {}
        speaker_evidence = review.get("evidence") or {}
        perception = result.get("perception_report") or {}
        review_status = str(review.get("status") or "unknown")
        review_completed = review_status.strip().lower() in {
            "success",
            "completed",
            "passed",
        }
        specialist_only = review_status == "specialist_only"
        task_observation = review.get("task_observation")
        if specialist_only and not review_kwargs["enable_omni_review"] and trace is None:
            # No Omni call is expected in this explicit observation mode.
            # Record the completed specialist projection and its real backend
            # attempts, just as speech_context does; do not invent a review.
            trace = ComponentInvocationTrace.build(
                component="perception",
                component_name=step.operation,
                route="deterministic",
                status="completed",
                metadata={
                    "operation": step.operation,
                    "actual_route_id": "speaker_specialists_without_omni_review",
                    "backend_attempts": _safe_perception_backend_attempts(
                        speaker_evidence.get("attempts") or []
                    ),
                    "omni_review_performed": False,
                },
            )
        selected_spans = perception.get("selected_spans", []) if task_observation is not None else (
            _reviewed_speaker_selection(
                perception=perception,
                instruction=instruction,
                audio_path=Path(input_audio),
            )
            if review_completed
            else []
        )
        selected_speaker_ids = list(
            dict.fromkeys(
                str(item["speaker_id"]) for item in selected_spans if item.get("speaker_id")
            )
        )
        perception = dict(perception)
        perception.setdefault("source_text_source", "speaker_attributed_review")
        if selected_spans and task_observation is None:
            perception["selected_spans"] = selected_spans
            perception["selected_speaker_ids"] = selected_speaker_ids
            perception["target_bindings"] = [
                {
                    "target_id": "target_000",
                    "target_kind": "speaker",
                    "target_value": instruction,
                    "status": "resolved",
                    "resolved_span_ids": [str(item["span_id"]) for item in selected_spans],
                    "resolved_speaker_ids": selected_speaker_ids,
                    "resolved_event_ids": [],
                    "confidence": None,
                    "reason": "resolved from speaker-attributed review",
                    "source_refs": [str(item["span_id"]) for item in selected_spans],
                }
            ]
        uncertainties: list[str] = []
        if review.get("task_observation_error"):
            uncertainties.append("Task-specific speaker observation failed: " + str(review["task_observation_error"]))
        if task_observation is not None:
            uncertainties.extend(task_observation.get("unresolved_questions") or [])
        elif not review_completed:
            uncertainties.append(
                "Specialist speaker observations are available; Omni review was disabled by policy."
                if specialist_only else
                "Speaker evidence is available, but Omni review did not reach a passing quality status."
            )
            quality_history = review.get("quality_check_history")
            if isinstance(quality_history, list) and quality_history:
                latest_quality = quality_history[-1]
                if isinstance(latest_quality, dict):
                    uncertainties.extend(
                        "Unverified diagnostic from failed speaker review; not an accepted "
                        "identity/count/absence fact: " + str(issue)
                        for issue in latest_quality.get("issues") or [] if str(issue)
                    )
        output = {
            "operation": step.operation,
            "input_audio_path": input_audio,
            "perception_report": perception,
            "source_text": result.get("source_text") or perception.get("source_text"),
            "source_text_source": perception.get("source_text_source"),
            "selected_spans": selected_spans,
            "selected_speaker_ids": selected_speaker_ids,
            "target_bindings": perception.get("target_bindings") or [],
            "speaker_attributed_review": review,
            "speaker_count": len(perception.get("speakers") or []),
            "segment_count": len(
                perception.get("speaker_segments") or perception.get("segments") or []
            ),
            "slice_audio_calls": review.get("slice_audio_calls", []),
            "required_slice_calls": review.get("required_slice_calls", 0),
            "attempts": speaker_evidence.get("attempts") or [],
            "perception_route_id": (
                f"speaker_evidence_plus_{_reviewer_provider_id(reviewer)}_task_observation" if task_observation is not None else
                "speaker_specialists_without_omni_review" if specialist_only else
                f"speaker_specialists_plus_{_reviewer_provider_id(reviewer)}_review"
                if review_kwargs["use_specialist_prior"] else
                f"speaker_{_reviewer_provider_id(reviewer)}_without_specialist_prior"
            ),
            "review_status": review_status,
            "review_completed": review_completed,
            "task_observation_completed": task_observation is not None,
            "task_observation_error": review.get("task_observation_error"),
            "target_grounding_satisfied": task_observation.get("selection_status") == "supported" if task_observation else None,
            "satisfied": review_completed or specialist_only,
            "needs_review": task_observation["selection_status"] == "unknown" if task_observation else not review_completed,
            "uncertainties": uncertainties,
            "speaker_evidence_backend": speaker_evidence.get("speaker_evidence_backend"),
        }
        if trace is not None:
            output["component_invocation_trace"] = trace.model_dump(mode="json")
        return output

    def _handle_semantic_span_review(
        self,
        step: AudioEditStep,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from x_editor.perception.semantic_span_review import (
            run_semantic_span_review,
        )

        params = dict(step.params)
        input_audio = _resolve_input_audio(step, state)
        base_audio_ref, reference_audio_refs, reference_audio_paths = (
            _perception_invocation_audio_bindings(
                state,
                base_audio_path=input_audio,
                fallback_reference_audio_paths=list(
                    resolve_audio_reference(
                        state.get("reference_audio_paths") or [],
                        state,
                    )
                ),
            )
        )
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        return run_semantic_span_review(
            audio_path=input_audio,
            audio_ref_id=base_audio_ref,
            reference_audio_paths=reference_audio_paths,
            reference_audio_refs=reference_audio_refs,
            instruction=instruction,
            semantic_targets=params.pop("semantic_targets", None),
            output_dir=Path(state["output_dir"]) / step.id,
            action=str(params.pop("action", "extract_spans")),
            need_speakers=bool(params.pop("need_speakers", False)),
            reviewer=self.perception_frontends.get(step.operation),
            agentic_mode=_perception_provider_mode(
                step,
                state,
                self.agentic_mode,
            ),
            speech_client=self.speech_perception_client,
            strict=bool(params.pop("strict", True)),
        )

    async def _handle_timestamped_caption(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        input_audio = _resolve_input_audio(step, state)
        event_labels = _flexsed_event_list_from_step(step, params)
        use_flexsed_prior = bool(params.pop("use_flexsed_prior", False))
        require_flexsed_prior_success = bool(params.pop("require_flexsed_prior_success", False))
        if require_flexsed_prior_success and not use_flexsed_prior:
            raise RuntimeError(
                "timestamped caption requires FlexSED prior success but the prior is disabled"
            )
        flexsed_prior: dict[str, Any] | None = None
        flexsed_error: str | None = None
        if use_flexsed_prior:
            flexsed_events = build_event_candidates(instruction, event_labels)
            try:
                flexsed_prior = await self._execute_tool(
                    "flexsed",
                    {
                        "input_audio_path": input_audio,
                        "events": flexsed_events,
                        "output_dir": str(Path(state["output_dir"]) / step.id / "flexsed_prior"),
                        "threshold": float(params.pop("flexsed_threshold", 0.5)),
                        "min_duration_sec": float(params.pop("flexsed_min_duration_sec", 0.08)),
                        "merge_gap_sec": float(params.pop("flexsed_merge_gap_sec", 0.08)),
                        "save_heatmap_png": bool(params.pop("save_heatmap_png", True)),
                        "save_heatmap_npy": bool(params.pop("save_heatmap_npy", True)),
                        "include_scores": bool(params.pop("include_scores", False)),
                    },
                    state,
                )
            except Exception as exc:
                flexsed_error = f"{type(exc).__name__}: {exc}"
                state.setdefault("timestamped_caption_flexsed_prior_errors", {})[step.id] = (
                    flexsed_error
                )
                if require_flexsed_prior_success:
                    raise RuntimeError(
                        "required FlexSED prior failed before timestamped caption review: "
                        f"{flexsed_error}"
                    ) from exc
        reviewer = (
            params.pop("reviewer", None)
            or self.perception_frontends.get("perception.timestamped_caption")
        )
        isolated_gateway_request = bool(
            (state.get("context") or {}).get("_workflow_perception_isolated_request", False)
        )
        result = run_timestamped_caption_review(
            audio_path=input_audio,
            instruction=instruction,
            output_dir=Path(state["output_dir"]) / step.id,
            event_labels=event_labels,
            use_flexsed_prior=use_flexsed_prior,
            flexsed_prior=flexsed_prior,
            flexsed_error=flexsed_error,
            # Prior reviewed reports belong to orchestration's namespace, not
            # this independent observation (even when the audio ref matches).
            base_perception=(
                {} if isolated_gateway_request else (state.get("perception_report") or {})
            ),
            reviewer=reviewer,
            max_review_attempts=int(params.pop("max_review_attempts", 2)),
            enable_candidate_rubric=bool(params.pop("enable_candidate_rubric", True)),
            allow_slice_completion=bool(params.pop("allow_slice_completion", True)),
            workflow_scope=str(params.pop("workflow_scope", "targeted")),
            temporal_precision=str(params.pop("temporal_precision", "segment")),
            require_local_boundaries=params.pop(
                "require_local_boundaries",
                None,
            ),
            slice_shortfall_mode=params.pop("slice_shortfall_mode", "warn"),
            strict=bool(params.pop("strict", True)),
            agentic_mode=_perception_provider_mode(step, state, self.agentic_mode),
        )
        review = result.get("timestamped_caption_review") or {}
        perception = result.get("perception_report") or {}
        selected_spans = _caption_selected_spans(perception)
        output = {
            "operation": step.operation,
            "input_audio_path": input_audio,
            "perception_report": perception,
            "timestamped_caption_review": review,
            "flexsed_prior": flexsed_prior or {},
            "flexsed_prior_error": flexsed_error,
            "audio_caption_segments": perception.get("audio_caption_segments") or [],
            "selected_spans": selected_spans,
            "slice_audio_calls": review.get("slice_audio_calls", []),
            "review_attempts": review.get("review_attempts", []),
            "verification_history": review.get("verification_history", []),
            "needs_review": bool((perception.get("timestamped_caption") or {}).get("needs_review")),
            "perception_route_id": _timestamped_caption_route_id(
                reviewer,
                use_flexsed_prior=use_flexsed_prior,
            ),
        }
        if result.get("component_invocation_trace") is not None:
            output["component_invocation_trace"] = result["component_invocation_trace"]
        return output

    async def _handle_flexsed_events(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        input_audio = _resolve_input_audio(step, state)
        events = _flexsed_event_list_from_step(step, params)
        if not events:
            raise RuntimeError(
                "FlexSED event grounding requires concrete audible event labels in params.events, "
                "params.event_labels, params.event_list, or inputs.events."
            )
        output_dir = str(Path(state["output_dir"]) / step.id)
        args: dict[str, Any] = {
            "input_audio_path": input_audio,
            "events": events,
            "output_dir": params.pop("output_dir", output_dir),
            "threshold": params.pop("threshold", 0.5),
            "min_duration_sec": params.pop("min_duration_sec", 0.08),
            "merge_gap_sec": params.pop("merge_gap_sec", 0.08),
            "save_heatmap_png": params.pop("save_heatmap_png", True),
            "save_heatmap_npy": params.pop("save_heatmap_npy", True),
            "include_scores": params.pop("include_scores", False),
        }
        for key in (
            "norm_audio",
            "max_scores_json_frames",
            "fname",
            "timeout_seconds",
            "start_sec",
            "end_sec",
            "p",
            "score_mode",
            "min_score",
        ):
            if key in params:
                args[key] = params.pop(key)
        _repair_flexsed_window_args(args, input_audio=input_audio, step=step, state=state)
        result = await self._execute_tool("flexsed", args, state)
        flexsed_report = _without_empty(
            {
                "tool": "flexsed",
                "input_audio_path": input_audio,
                "events": result.get("events") or events,
                "timestamps": result.get("timestamps") or {},
                "heatmap": result.get("heatmap") or {},
                "frame_rate_hz": result.get("frame_rate_hz"),
                "frame_step_sec": result.get("frame_step_sec"),
                "threshold": result.get("threshold", args["threshold"]),
                "min_duration_sec": result.get("min_duration_sec", args["min_duration_sec"]),
                "merge_gap_sec": result.get("merge_gap_sec", args["merge_gap_sec"]),
                "audio_duration_sec": result.get("audio_duration_sec"),
                "window_event_summary": result.get("window_event_summary"),
                "window_event_line": result.get("window_event_line"),
            }
        )
        selected_spans = _flexsed_selected_spans(flexsed_report.get("timestamps") or {})
        perception = dict(state.get("perception_report") or {})
        audio_events = list(perception.get("audio_events") or [])
        audio_events.append(flexsed_report)
        perception["audio_events"] = audio_events
        notable = list(perception.get("notable_events") or [])
        for event in events:
            if event not in notable:
                notable.append(event)
        perception["notable_events"] = notable
        if selected_spans:
            perception.setdefault("event_segments", []).extend(selected_spans)
        return {
            "operation": step.operation,
            "input_audio_path": input_audio,
            "perception_report": perception,
            "flexsed_events": flexsed_report,
            "timestamps": flexsed_report.get("timestamps") or {},
            "heatmap": flexsed_report.get("heatmap") or {},
            "selected_spans": selected_spans,
            "perception_route_id": "flexsed_event_detection",
        }

    async def _handle_spatial_reverb(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        args = {
            "input_audio_path": _resolve_input_audio(step, state),
            "output_path": _output_path(step, state),
            "instruction": instruction,
            "room_preset": params.pop("room_preset", None),
            "room_size_m": params.pop("room_size_m", None),
            "t60": params.pop("t60", None),
            "dry_wet": params.pop("dry_wet", None),
            "source_distance_m": params.pop("source_distance_m", None),
            "receiver_position_m": params.pop("receiver_position_m", None),
            "source_position_m": params.pop("source_position_m", None),
            "rir_length_sec": params.pop("rir_length_sec", None),
            "preserve_duration": bool(params.pop("preserve_duration", False)),
            "backend": str(params.pop("backend", "auto")),
            "wet_gain_db": float(params.pop("wet_gain_db", 0.0)),
            "output_dir": str(Path(state["output_dir"]) / f"{step.id}_spatial_reverb"),
            **params,
        }
        args = {key: value for key, value in args.items() if value is not None}
        return await self._execute_tool("spatial_reverb", args, state)

    async def _handle_intra_audio_voice_transfer(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        step_perception = params.get("perception_report")
        perception_report = (
            step_perception
            if isinstance(step_perception, dict)
            else state.get("perception_report") or {}
        )
        source_text = (
            params.get("source_text")
            if isinstance(params.get("source_text"), str)
            else state.get("source_text")
        )
        output_path = _output_path(step, state)
        result = await self._execute_intra_audio_voice_transfer_with_gateway(
            runtime_state=state,
            state={
                "base_audio_path": _resolve_input_audio(step, state),
                "user_prompt": state["user_prompt"],
                "source_text": source_text,
                "perception_report": perception_report,
            },
            output_path=output_path,
            output_dir=state["output_dir"],
            instruction=params.get("instruction") or step.description or state["user_prompt"],
            mapping_params=params,
        )
        if not result.success:
            if not _implicit_executor_repair_enabled(state):
                raise RuntimeError(result.error_message or "intra_audio_voice_transfer failed")
            fallback = _reference_vc_fallback_audio(step, state)
            if (
                fallback
                and "needs at least two diarized speakers"
                in str(result.error_message or "").lower()
            ):
                source_audio = _resolve_input_audio(step, state)
                state.setdefault("operation_rewrites", {})[step.id] = {
                    "from": "intra_audio_voice_transfer",
                    "to": "seed_vc",
                    "reason": result.error_message
                    or "intra-audio transfer could not resolve two speakers",
                    "reference_audio_path": fallback,
                }
                seed_result = await self._execute_tool(
                    "seed_vc",
                    {
                        "source_audio_path": source_audio,
                        "reference_audio_path": fallback,
                        "output_path": output_path,
                    },
                    state,
                )
                return {
                    "output_audio_path": _extract_output_path(seed_result) or output_path,
                    "operation": "intra_audio_voice_transfer_fallback_seed_vc",
                    "source_audio_path": source_audio,
                    "reference_audio_path": fallback,
                    "seed_vc_result": seed_result,
                    "fallback_reason": result.error_message,
                }
            text_edit_fallback = self._maybe_fallback_intra_phrase_replacement_to_text_edit(
                step=step,
                state=state,
                params=params,
                output_path=output_path,
                error_message=result.error_message or "intra_audio_voice_transfer failed",
                perception_report=perception_report,
                source_text=source_text,
            )
            if text_edit_fallback is not None:
                return text_edit_fallback
            repeated_phrase_fallback = (
                self._maybe_fallback_repeated_phrase_group_style_to_speaker_scoped_edit(
                    step=step,
                    state=state,
                    params=params,
                    output_path=output_path,
                    error_message=result.error_message or "intra_audio_voice_transfer failed",
                    perception_report=perception_report,
                    source_text=source_text,
                )
            )
            if repeated_phrase_fallback is not None:
                return repeated_phrase_fallback
            span_style_fallback = self._maybe_fallback_intra_span_style_to_speaker_scoped_edit(
                step=step,
                state=state,
                params=params,
                output_path=output_path,
                error_message=result.error_message or "intra_audio_voice_transfer failed",
                perception_report=perception_report,
                source_text=source_text,
            )
            if span_style_fallback is not None:
                return span_style_fallback
            raise RuntimeError(result.error_message or "intra_audio_voice_transfer failed")
        return result.output

    async def _execute_intra_audio_voice_transfer_with_gateway(
        self,
        *,
        runtime_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolResult:
        loop = asyncio.get_running_loop()
        nested_workspace = (
            Path(str(kwargs["output_dir"]))
            / "intra_audio_voice_transfer"
            / Path(str(kwargs["output_path"])).stem
        )

        def tool_gateway(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self._register_nested_gateway_audio_paths(
                arguments=arguments,
                state=runtime_state,
                workspace=nested_workspace,
            )
            self._verify_nested_tool_contract(
                tool_id=tool_id,
                arguments=arguments,
                state=runtime_state,
            )
            future = asyncio.run_coroutine_threadsafe(
                self._execute_tool(tool_id, arguments, runtime_state), loop
            )
            return future.result()

        return await asyncio.to_thread(
            execute_intra_audio_voice_transfer,
            **kwargs,
            tool_gateway=tool_gateway,
        )

    def _maybe_fallback_repeated_phrase_group_style_to_speaker_scoped_edit(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        params: dict[str, Any],
        output_path: str,
        error_message: str,
        perception_report: dict[str, Any],
        source_text: str | None,
    ) -> dict[str, Any] | None:
        request = str(
            params.get("instruction") or step.description or state.get("user_prompt") or ""
        )
        if not _looks_like_repeated_phrase_group_style_request(request):
            return None
        phrase_plan = _repeated_phrase_group_rate_plan(
            perception_report=perception_report,
            source_text=source_text or state.get("source_text"),
        )
        selected_spans = (
            phrase_plan.get("selected_spans") if isinstance(phrase_plan, dict) else None
        )
        if not isinstance(selected_spans, list) or not selected_spans:
            return None
        base_audio = _base_input_audio_path(state) or _resolve_input_audio(step, state)
        if not base_audio:
            return None

        rate_factor = float(phrase_plan.get("rate_factor") or 0.8)
        style_instruction = (
            "Slow the selected repeated phrase occurrences to match the first clear occurrence "
            "in each phrase group while preserving the original voice and wording."
        )
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "intra_audio_voice_transfer",
            "to": "artifact_ops.speaker_scoped_edit",
            "reason": error_message,
            "fallback_strategy": "repeated_phrase_group_rate_match",
            "selected_spans": selected_spans,
            "reference_spans": phrase_plan.get("reference_spans", []),
            "rate_factor": rate_factor,
            "degraded_constraints": [
                "fallback matches repeated phrase pacing with segmented speech-rate editing; exact style transfer is not guaranteed"
            ],
        }
        result = execute_artifact_step(
            operation="speaker_scoped_edit",
            instruction=style_instruction,
            input_audio_path=base_audio,
            output_path=output_path,
            output_dir=str(Path(state["output_dir"]) / f"{step.id}_repeated_phrase_fallback"),
            perception_report=perception_report,
            reference_audio_paths=state.get("reference_audio_paths") or [],
            source_text=source_text,
            params={
                "selection_instruction": "all repeated fast phrase occurrences after the first clear occurrence",
                "selected_spans": selected_spans,
                "edit_op": "edit_data_pipeline.speech_rate",
                "edit_op_params": {
                    "rate": rate_factor,
                    "method": "wsola+indextts2",
                },
                "preserve_timeline": True,
            },
        )
        return {
            **result,
            "operation": "intra_audio_voice_transfer_fallback_repeated_phrase_rate_edit",
            "fallback_reason": error_message,
            "source_audio_path": base_audio,
            "selected_spans": selected_spans,
            "reference_spans": phrase_plan.get("reference_spans", []),
            "rate_factor": rate_factor,
            "degraded_constraints": [
                "fallback matches repeated phrase pacing with segmented speech-rate editing; exact style transfer is not guaranteed"
            ],
        }

    def _maybe_fallback_intra_span_style_to_speaker_scoped_edit(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        params: dict[str, Any],
        output_path: str,
        error_message: str,
        perception_report: dict[str, Any],
        source_text: str | None,
    ) -> dict[str, Any] | None:
        source_span = _span_pair_from_value(
            params.get("source_span_sec") or params.get("source_span")
        )
        reference_span = _span_pair_from_value(
            params.get("reference_span_sec") or params.get("reference_span")
        )
        if source_span is None or reference_span is None:
            return None
        request = str(
            params.get("instruction") or step.description or state.get("user_prompt") or ""
        )
        if not _has_any(
            request.lower(),
            ["tone", "style", "speaking style", "prosody", "intonation"],
        ):
            return None
        base_audio = _base_input_audio_path(state) or _resolve_input_audio(step, state)
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "intra_audio_voice_transfer",
            "to": "artifact_ops.speaker_scoped_edit",
            "reason": error_message,
            "fallback_strategy": "source_span_pitch_rate_style_edit",
            "source_span_sec": source_span,
            "reference_span_sec": reference_span,
            "degraded_constraints": [
                "reference span is used as textual style guidance; exact reference prosody transfer is not guaranteed"
            ],
        }
        style_instruction = (
            f"Edit the selected source sentence to better match the tone, speaking style, and intonation of "
            f"the reference span {reference_span[0]:.3g}-{reference_span[1]:.3g}s."
        )
        result = execute_artifact_step(
            operation="speaker_scoped_edit",
            instruction=style_instruction,
            input_audio_path=base_audio,
            output_path=output_path,
            output_dir=str(Path(state["output_dir"]) / f"{step.id}_span_style_fallback"),
            perception_report=perception_report,
            reference_audio_paths=state.get("reference_audio_paths") or [],
            source_text=source_text,
            params={
                "selected_spans": [
                    {
                        "start_sec": source_span[0],
                        "end_sec": source_span[1],
                        "label": "source_span_for_style_transfer",
                    }
                ],
                "edit_op": "pitch_rate_edit",
                "edit_op_params": {
                    "instruction": style_instruction,
                    "reference_span_sec": list(reference_span),
                },
                "preserve_timeline": True,
                "_agentic_mode": self.agentic_mode,
            },
        )
        return {
            **result,
            "operation": "intra_audio_voice_transfer_fallback_span_style_edit",
            "fallback_reason": error_message,
            "source_span_sec": list(source_span),
            "reference_span_sec": list(reference_span),
            "degraded_constraints": [
                "reference span is used as textual style guidance; exact reference prosody transfer is not guaranteed"
            ],
        }

    def _maybe_fallback_intra_phrase_replacement_to_text_edit(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        params: dict[str, Any],
        output_path: str,
        error_message: str,
        perception_report: dict[str, Any],
        source_text: str | None,
    ) -> dict[str, Any] | None:
        if "needs at least two diarized speakers" not in error_message.lower():
            return None
        replacement_pair = _phrase_replacement_pair_from_request(step, state, params)
        if replacement_pair is None:
            return None
        base_audio = _base_input_audio_path(state)
        if not base_audio:
            return None
        source_phrase, target_phrase = replacement_pair
        text_edit_instruction = (
            f'Replace every instance of "{source_phrase}" with "{target_phrase}".'
        )
        request = str(
            state.get("user_prompt") or step.description or params.get("instruction") or ""
        )
        requested_voice_gender = _normalize_binary_gender(request)
        degraded_constraints: list[str] = []
        if requested_voice_gender:
            degraded_constraints.append(
                f"requested {requested_voice_gender} target voice was not reliably applied; "
                "fallback preserves timeline and performs text replacement on the original audio"
            )
        state.setdefault("operation_rewrites", {})[step.id] = {
            "from": "intra_audio_voice_transfer",
            "to": "artifact_ops.speaker_scoped_edit",
            "reason": error_message,
            "fallback_strategy": "phrase_replacement_text_edit_on_original_audio",
            "source_phrase": source_phrase,
            "target_phrase": target_phrase,
            "input_audio_path": base_audio,
            "degraded_constraints": degraded_constraints,
        }
        result = execute_artifact_step(
            operation="speaker_scoped_edit",
            instruction=text_edit_instruction,
            input_audio_path=base_audio,
            output_path=output_path,
            output_dir=str(Path(state["output_dir"]) / f"{step.id}_phrase_replacement_fallback"),
            perception_report=perception_report,
            reference_audio_paths=state.get("reference_audio_paths") or [],
            source_text=source_text,
            params={
                "selection_instruction": f'all occurrences of the spoken phrase "{source_phrase}"',
                "edit_op": "text_edit",
                "edit_op_params": {
                    "instruction": text_edit_instruction,
                },
                "preserve_timeline": True,
                "_agentic_mode": self.agentic_mode,
            },
        )
        return {
            **result,
            "operation": "intra_audio_voice_transfer_fallback_speaker_scoped_text_edit",
            "fallback_reason": error_message,
            "source_audio_path": base_audio,
            "source_phrase": source_phrase,
            "target_phrase": target_phrase,
            "degraded_constraints": degraded_constraints,
        }

    async def _handle_accent_conversion(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        explicit_instruction = params.pop("instruction", None)
        user_prompt = str(state.get("user_prompt") or "")
        if explicit_instruction:
            instruction = str(explicit_instruction)
        elif user_prompt:
            instruction = user_prompt
        else:
            instruction = str(step.description or user_prompt)
        if _accent_plan_is_tts_generation(instruction, params):
            input_audio = _resolve_input_audio(step, state)
            target_text = _target_text_from_accent_params_or_instruction(params, instruction)
            if not target_text:
                raise RuntimeError("accent_conversion reroute to qwen3_tts requires target_text")
            state.setdefault("operation_rewrites", {})[step.id] = {
                "from": "accent_conversion",
                "to": "qwen3_tts",
                "reason": "request is target-sentence voice/pronunciation cloning, not accent conversion",
            }
            return await self._execute_tool(
                "qwen3_tts",
                {
                    "text": target_text,
                    "output_path": _output_path(step, state),
                    "mode": "voice_clone",
                    "ref_audio_path": input_audio,
                    "language": _qwen3_tts_language_from_text_and_request(target_text, instruction),
                    "instruct": instruction,
                    "x_vector_only_mode": True,
                },
                state,
            )
        if _accent_plan_is_native_fluency_rewrite(instruction, params):
            state.setdefault("operation_rewrites", {})[step.id] = {
                "from": "accent_conversion",
                "to": "artifact_ops.speaker_scoped_edit",
                "reason": "native/fluent same-language rewrite is text editing, not accent conversion",
            }
            return await self._execute_artifact_step_with_gateway(
                state=state,
                operation="speaker_scoped_edit",
                instruction=instruction,
                input_audio_path=_resolve_input_audio(step, state),
                output_path=_output_path(step, state),
                output_dir=str(Path(state["output_dir"]) / step.id),
                perception_report=state.get("perception_report") or {},
                source_text=params.pop("source_text", None) or state.get("source_text"),
                reference_audio_paths=state.get("reference_audio_paths") or [],
                params={
                    "selection_instruction": instruction,
                    "edit_op": "text_edit",
                    "edit_op_params": {"instruction_st": instruction},
                    "_agentic_mode": self.agentic_mode,
                    **_speaker_selector_params_from_accent_params(params),
                },
            )
        input_audio_path = _resolve_input_audio(step, state)
        reference_audio_path = (
            params.pop("reference_audio_path", None)
            or step.inputs.get("reference_audio_path")
            or step.inputs.get("reference_audio")
        )
        if not reference_audio_path and requests_reference_accent_match(instruction):
            refs = state.get("reference_audio_paths") or []
            reference_audio_path = (
                refs[0]
                if refs
                else _original_session_audio_reference(state, exclude=input_audio_path)
            )
            if reference_audio_path:
                state.setdefault("executor_reference_repairs", {})[step.id] = {
                    "operation": "accent_conversion",
                    "reference_audio_path": reference_audio_path,
                    "strategy": "restore_reference_accent_from_session_original",
                    "reason": "instruction requests restoring/matching a previous or original accent",
                }
        reference_audio_path = resolve_audio_reference(reference_audio_path, state)
        target_text_override = _pop_nonempty_string(params, "target_text")
        dependency_text = _text_from_text_edit_dependencies(step, state)
        current_text_override = _current_text_edit_text(state)
        explicit_source_text = params.pop("source_text", None)
        source_text = (
            target_text_override
            or dependency_text
            or current_text_override
            or explicit_source_text
            or state.get("source_text")
        )
        perception_report = state.get("perception_report") or {}
        text_override = target_text_override or dependency_text or current_text_override
        if text_override:
            perception_report = _perception_report_with_text_override(
                perception_report,
                text_override,
                source="previous_text_edit"
                if dependency_text or current_text_override
                else "accent_conversion.target_text",
            )

        async def execute_existing_tool(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return await self._execute_tool(tool_id, arguments, state)

        return await execute_accent_conversion(
            execute_tool=execute_existing_tool,
            input_audio_path=input_audio_path,
            output_path=_output_path(step, state),
            output_dir=str(Path(state["output_dir"]) / step.id),
            instruction=instruction,
            source_text=source_text,
            perception_report=perception_report,
            target_accent=params.pop("target_accent", None),
            reference_audio_path=str(reference_audio_path) if reference_audio_path else None,
            reference_perception_report=params.pop("reference_perception_report", None),
            agentic_mode=self.agentic_mode,
        )

    async def _execute_artifact_step_with_gateway(
        self,
        *,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        operation = str(kwargs.get("operation") or "")
        if operation.removeprefix("artifact_ops.") != "speaker_scoped_edit":
            return execute_artifact_step(**kwargs)

        loop = asyncio.get_running_loop()
        nested_workspace = (
            Path(str(kwargs["output_dir"])) / "artifact_ops" / Path(str(kwargs["output_path"])).stem
        )

        def tool_gateway(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self._register_nested_gateway_audio_paths(
                arguments=arguments,
                state=state,
                workspace=nested_workspace,
            )
            self._verify_nested_tool_contract(
                tool_id=tool_id,
                arguments=arguments,
                state=state,
            )
            future = asyncio.run_coroutine_threadsafe(
                self._execute_tool(tool_id, arguments, state), loop
            )
            return future.result()

        return await asyncio.to_thread(
            execute_artifact_step,
            **kwargs,
            tool_gateway=tool_gateway,
        )

    def _register_nested_gateway_audio_paths(
        self,
        *,
        arguments: dict[str, Any],
        state: dict[str, Any],
        workspace: Path,
    ) -> None:
        """Trust only composite-tool audio files created inside its workspace."""

        if not opaque_audio_resolution_enabled(state):
            return
        runtime_output_dir = state.get("output_dir")
        if not isinstance(runtime_output_dir, str) or not runtime_output_dir:
            return
        runtime_root = Path(runtime_output_dir).expanduser().resolve(strict=False)
        workspace_root = workspace.expanduser().resolve(strict=False)
        if workspace_root != runtime_root and runtime_root not in workspace_root.parents:
            return

        trusted_paths = state.setdefault("_runtime_trusted_audio_paths", set())
        if not isinstance(trusted_paths, set):
            trusted_paths = set()
            state["_runtime_trusted_audio_paths"] = trusted_paths

        for key, value in arguments.items():
            key_text = str(key).casefold()
            is_audio_path = key_text in {
                "audio_path",
                "audio_paths",
                "output_path",
            } or key_text.endswith(("_audio_path", "_audio_paths"))
            if not is_audio_path:
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if not isinstance(item, str) or not item:
                    continue
                candidate = Path(item).expanduser()
                resolved = candidate.resolve(strict=False)
                if resolved != workspace_root and workspace_root not in resolved.parents:
                    continue
                is_output = key_text in {"output_path", "output_audio_path"}
                if not is_output and not candidate.is_file():
                    continue
                trusted_paths.add(item)

    def _verify_nested_tool_contract(
        self,
        *,
        tool_id: str,
        arguments: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        operation = (
            f"edit_data_pipeline.{arguments.get('operation')}"
            if tool_id == "edit_data_pipeline" and arguments.get("operation")
            else tool_id
        )
        nested_step = AudioEditStep(
            id=f"{state.get('_current_step_id') or 'nested'}:{tool_id}",
            operation=operation,
            params=dict(arguments),
        )
        operation_spec = self._operation_spec_for_step(nested_step)
        violations = verify_step_tool_contract(
            nested_step,
            state,
            operation_spec=operation_spec,
            output_path=str(arguments.get("output_path") or "") or None,
        )
        if not violations:
            return
        for violation in violations:
            self._record_tool_contract_violation(nested_step, state, violation)
        raise ToolContractViolationError(violations)

    async def _handle_relative_gain(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._execute_artifact_step_with_gateway(
            state=state, operation="apply_gain", instruction=step.description or "",
            input_audio_path=_resolve_input_audio(step, state),
            output_path=_output_path(step, state), output_dir=state["output_dir"],
            params=dict(step.params),
        )

    async def _handle_artifact_ops(
        self, step: AudioEditStep, state: dict[str, Any]
    ) -> dict[str, Any]:
        params = dict(step.params)
        for key in ("edited_spans_audio_paths", "edited_spans_audio", "edits"):
            if key not in params and key in step.inputs:
                params[key] = step.inputs[key]
        operation = step.operation.split(".", 1)[1] if "." in step.operation else step.operation
        if operation == "reconstruct":
            _merge_reconstruct_input_and_param_edits(step=step, params=params, state=state)
            _normalize_reconstruct_step_reference_edits(step=step, params=params, state=state)
        if operation == "concat" and not (params.get("audio_paths") or params.get("sources")):
            for key in ("audio_paths", "input_audio_paths", "input_audios"):
                value = step.inputs.get(key) or step.params.get(key)
                if isinstance(value, list) and value:
                    params["audio_paths"] = value
                    break
            value = step.inputs.get("input_audio_path")
            if isinstance(value, list) and value:
                params.setdefault("audio_paths", value)
        span_filter = _filter_non_target_span_params(params, operation=operation)
        if span_filter:
            state.setdefault("span_selection_param_filters", {})[step.id] = span_filter
            if operation == "reconstruct" and span_filter.get("removed_all_explicit_targets"):
                raise RuntimeError(
                    "artifact_ops.reconstruct received only non-target context/reference spans; "
                    "context spans are evidence only and must not be reconstructed as edits"
                )
        input_audio_path = _resolve_input_audio(step, state)
        input_audio_path = _maybe_use_base_audio_for_absolute_spans(
            step=step,
            operation=operation,
            params=params,
            input_audio_path=input_audio_path,
            state=state,
        )
        instruction = str(
            params.pop("instruction", None) or step.description or state["user_prompt"]
        )
        reused_spans = _maybe_reuse_previous_selected_spans_for_followup(
            step=step,
            state=state,
            operation=operation,
            params=params,
            instruction=instruction,
        )
        if reused_spans:
            state.setdefault("artifact_ops_followup_span_reuse", {})[step.id] = reused_spans
        input_audio_path = _maybe_use_base_audio_for_original_speaker_selector(
            step=step,
            operation=operation,
            params=params,
            instruction=instruction,
            input_audio_path=input_audio_path,
            state=state,
        )
        grounded_reference = await self._reference_extract_perception_for_input_audio(
            step=step,
            state=state,
            operation=operation,
            instruction=instruction,
            input_audio_path=input_audio_path,
            params=params,
        )
        if grounded_reference:
            perception_report = grounded_reference["perception_report"]
            step_source_text = grounded_reference.get("source_text") or state.get("source_text")
        else:
            grounded_event = await self._event_grounding_for_artifact_step(
                step=step,
                state=state,
                operation=operation,
                instruction=instruction,
                input_audio_path=input_audio_path,
                params=params,
            )
            if grounded_event:
                perception_report = grounded_event["perception_report"]
                step_source_text = grounded_event.get("source_text") or state.get("source_text")
            else:
                perception_report, step_source_text = (
                    self._artifact_step_perception_for_input_audio(
                        step=step,
                        state=state,
                        operation=operation,
                        instruction=instruction,
                        input_audio_path=input_audio_path,
                        params=params,
                    )
                )
        edited_audio_path = params.pop("edited_audio_path", None)
        if edited_audio_path is not None:
            edited_audio_path = resolve_audio_reference(edited_audio_path, state)
        return await self._execute_artifact_step_with_gateway(
            state=state,
            operation=operation,
            instruction=instruction,
            input_audio_path=input_audio_path,
            output_path=_output_path(step, state),
            output_dir=state["output_dir"],
            perception_report=perception_report,
            reference_audio_paths=state.get("reference_audio_paths") or [],
            source_text=step_source_text,
            artifact_paths=state.get("artifacts") or {},
            edited_audio_path=edited_audio_path,
            params={**params, "_agentic_mode": self.agentic_mode},
        )

    async def _event_grounding_for_artifact_step(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        operation: str,
        instruction: str,
        input_audio_path: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not _artifact_event_grounding_is_needed(step, operation, instruction, params):
            return None

        event_labels = _artifact_event_target_labels(step, instruction, params)
        if not event_labels:
            return None

        if operation == "extract_spans" and _event_labels_are_negative_extract_filters(
            step, instruction, params, event_labels
        ):
            record = {
                "status": "skipped_negative_event_filter",
                "strategy": "speaker_or_speech_selector_with_negative_event_constraint",
                "input_audio_path": input_audio_path,
                "operation": operation,
                "target_event_labels": event_labels,
                "reason": (
                    "Event labels appear only as negative constraints for a speech/speaker extraction; "
                    "letting artifact_ops select the target speech spans instead of extracting the excluded events."
                ),
            }
            state.setdefault("artifact_event_span_grounding", {})[step.id] = _without_empty(record)
            return None

        prior_spans = _event_spans_from_prior_step_outputs(state, event_labels)
        if prior_spans:
            if operation == "speaker_scoped_edit":
                params["event_removal_spans"] = prior_spans
                params.setdefault(
                    "event_removal_policy",
                    "mute_grounded_non_speech_event_spans_after_speaker_edit",
                )
            else:
                params["selected_spans"] = prior_spans
                params.setdefault("selection_instruction", instruction)
            record = {
                "status": "grounded",
                "strategy": "reuse_prior_event_grounding",
                "input_audio_path": input_audio_path,
                "operation": operation,
                "target_event_labels": event_labels,
                "selected_span_count": len(prior_spans),
                "selected_spans": prior_spans,
            }
            state.setdefault("artifact_event_span_grounding", {})[step.id] = _without_empty(record)
            return {
                "perception_report": state.get("perception_report") or {},
                "selected_spans": prior_spans,
            }

        use_flexsed_prior = bool(params.get("use_flexsed_prior", True))
        flexsed_prior: dict[str, Any] | None = None
        flexsed_error: str | None = None
        if use_flexsed_prior:
            try:
                flexsed_prior = await self._execute_tool(
                    "flexsed",
                    {
                        "input_audio_path": input_audio_path,
                        "events": build_event_candidates(instruction, event_labels),
                        "output_dir": str(
                            Path(state["output_dir"]) / step.id / "artifact_event_flexsed_prior"
                        ),
                        "threshold": float(params.get("flexsed_threshold", 0.5)),
                        "min_duration_sec": float(params.get("flexsed_min_duration_sec", 0.08)),
                        "merge_gap_sec": float(params.get("flexsed_merge_gap_sec", 0.08)),
                        "save_heatmap_png": bool(params.get("save_heatmap_png", True)),
                        "save_heatmap_npy": bool(params.get("save_heatmap_npy", True)),
                        "include_scores": bool(params.get("include_scores", False)),
                    },
                    state,
                )
            except Exception as exc:
                flexsed_error = f"{type(exc).__name__}: {exc}"
                state.setdefault("timestamped_caption_flexsed_prior_errors", {})[step.id] = (
                    flexsed_error
                )

        caption_result: dict[str, Any] = {}
        caption_error: str | None = None
        try:
            caption_result = run_timestamped_caption_review(
                audio_path=input_audio_path,
                instruction=instruction,
                output_dir=Path(state["output_dir"]) / f"{step.id}_artifact_timestamped_caption",
                event_labels=event_labels,
                use_flexsed_prior=use_flexsed_prior,
                flexsed_prior=flexsed_prior,
                flexsed_error=flexsed_error,
                base_perception=state.get("perception_report") or {},
                reviewer=params.get("reviewer"),
                max_review_attempts=int(params.get("max_review_attempts", 2)),
                strict=False,
                agentic_mode=_perception_provider_mode(step, state, self.agentic_mode),
            )
        except Exception as exc:
            caption_error = f"{type(exc).__name__}: {exc}"

        perception = caption_result.get("perception_report") or state.get("perception_report") or {}
        caption_spans = _filter_event_grounded_spans(
            caption_result.get("selected_spans") or _caption_selected_spans(perception),
            event_labels,
            destructive=operation in {"remove_spans", "mute_spans", "beep_spans"},
        )
        flexsed_spans = _filter_event_grounded_spans(
            _flexsed_selected_spans((flexsed_prior or {}).get("timestamps") or {}),
            event_labels,
            destructive=False,
        )
        selected_spans = _merge_event_grounded_spans([*caption_spans, *flexsed_spans])
        if selected_spans:
            padding = float(
                params.get(
                    "event_span_padding_sec",
                    0.08 if operation in {"remove_spans", "mute_spans", "beep_spans"} else 0.0,
                )
            )
            if padding > 0:
                selected_spans = _pad_event_spans(selected_spans, padding)
            if operation == "speaker_scoped_edit":
                params["event_removal_spans"] = selected_spans
                params.setdefault(
                    "event_removal_policy",
                    "mute_grounded_non_speech_event_spans_after_speaker_edit",
                )
            else:
                params["selected_spans"] = selected_spans
                params.setdefault("selection_instruction", instruction)

        record = {
            "status": "grounded" if selected_spans else "failed_no_event_spans",
            "strategy": "flexsed_prior_timestamped_caption",
            "input_audio_path": input_audio_path,
            "operation": operation,
            "target_event_labels": event_labels,
            "use_flexsed_prior": use_flexsed_prior,
            "flexsed_error": flexsed_error,
            "timestamped_caption_error": caption_error,
            "caption_span_count": len(caption_spans),
            "flexsed_span_count": len(flexsed_spans),
            "selected_span_count": len(selected_spans),
            "selected_spans": selected_spans,
            "timestamped_caption_review": caption_result.get("timestamped_caption_review"),
            "flexsed_prior": flexsed_prior,
        }
        state.setdefault("artifact_event_span_grounding", {})[step.id] = _without_empty(record)
        if not selected_spans and operation != "speaker_scoped_edit":
            raise RuntimeError(
                "artifact_ops event edit could not resolve grounded event spans via "
                "FlexSED-prior timestamped caption; refusing broad transcript/speaker span fallback"
            )
        return {
            "perception_report": perception,
            "source_text": perception.get("source_text") or state.get("source_text"),
            "selected_spans": selected_spans,
        }

    async def _reference_extract_perception_for_input_audio(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        operation: str,
        instruction: str,
        input_audio_path: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        if operation != "extract_spans":
            return None
        if _has_explicit_span_params(params):
            return None
        try:
            input_path = Path(str(input_audio_path)).expanduser().resolve()
        except Exception:
            return None
        if not _is_reference_extract_step(
            step=step,
            state=state,
            input_path=input_path,
            instruction=instruction,
            params=params,
        ):
            return None

        evidence_text = _reference_extract_evidence_text(step, instruction, params)
        needs_speaker = _reference_extract_needs_speaker_review(evidence_text)
        needs_caption = _reference_extract_needs_timestamp_caption(evidence_text)
        if not needs_speaker and not needs_caption:
            needs_speaker = True

        base_perception = _perception_report_for_audio_path(state, input_path) or {}
        operations: list[str] = []
        reports: list[dict[str, Any]] = []
        selected_by_operation: dict[str, list[dict[str, Any]]] = {}
        source_text: str | None = None

        if needs_speaker:
            speaker_kwargs: dict[str, Any] = {
                "audio_path": input_path,
                "instruction": instruction,
                "output_dir": Path(state["output_dir"]) / f"{step.id}_reference_speaker_review",
                "base_perception": base_perception,
                "run_specialized": bool(params.get("run_specialized", True)),
                "use_sedicow": bool(params.get("use_sedicow", True)),
                "max_review_attempts": int(params.get("max_review_attempts", 2)),
                "strict": bool(
                    params.get("reference_perception_strict", params.get("strict", True))
                ),
                "language": params.get("language"),
                "num_speakers": params.get("num_speakers"),
                "agentic_mode": _perception_provider_mode(step, state, self.agentic_mode),
            }
            if params.get("speaker_evidence_backend") is not None:
                speaker_kwargs["speaker_evidence_backend"] = str(
                    params.get("speaker_evidence_backend")
                )
            speaker_result = run_speaker_attributed_review(**speaker_kwargs)
            speaker_perception = speaker_result.get("perception_report") or {}
            reports.append(speaker_perception)
            operations.append("perception.speaker_attributed_review")
            source_text = (
                speaker_result.get("source_text")
                or speaker_perception.get("source_text")
                or source_text
            )
            speaker_spans = _speaker_review_selected_spans(
                perception=speaker_perception,
                instruction=instruction,
                audio_path=input_path,
            )
            if speaker_spans:
                selected_by_operation["perception.speaker_attributed_review"] = speaker_spans

        if needs_caption:
            event_labels = _reference_event_labels(params)
            use_flexsed_prior = bool(params.get("use_flexsed_prior", True))
            flexsed_prior: dict[str, Any] | None = None
            flexsed_error: str | None = None
            if use_flexsed_prior:
                try:
                    flexsed_prior = await self._execute_tool(
                        "flexsed",
                        {
                            "input_audio_path": str(input_path),
                            "events": build_event_candidates(instruction, event_labels),
                            "output_dir": str(
                                Path(state["output_dir"]) / step.id / "reference_flexsed_prior"
                            ),
                            "threshold": float(params.get("flexsed_threshold", 0.5)),
                            "min_duration_sec": float(params.get("flexsed_min_duration_sec", 0.08)),
                            "merge_gap_sec": float(params.get("flexsed_merge_gap_sec", 0.08)),
                            "save_heatmap_png": bool(params.get("save_heatmap_png", True)),
                            "save_heatmap_npy": bool(params.get("save_heatmap_npy", True)),
                            "include_scores": bool(params.get("include_scores", False)),
                        },
                        state,
                    )
                except Exception as exc:
                    flexsed_error = f"{type(exc).__name__}: {exc}"
                    state.setdefault("timestamped_caption_flexsed_prior_errors", {})[step.id] = (
                        flexsed_error
                    )
            caption_result = run_timestamped_caption_review(
                audio_path=input_path,
                instruction=instruction,
                output_dir=Path(state["output_dir"]) / f"{step.id}_reference_timestamped_caption",
                event_labels=event_labels,
                use_flexsed_prior=use_flexsed_prior,
                flexsed_prior=flexsed_prior,
                flexsed_error=flexsed_error,
                base_perception=base_perception,
                reviewer=params.get("reviewer"),
                max_review_attempts=int(params.get("max_review_attempts", 2)),
                strict=bool(params.get("reference_perception_strict", params.get("strict", True))),
                agentic_mode=_perception_provider_mode(step, state, self.agentic_mode),
            )
            caption_perception = caption_result.get("perception_report") or {}
            reports.append(caption_perception)
            operations.append("perception.timestamped_caption")
            caption_spans = caption_result.get("selected_spans") or _caption_selected_spans(
                caption_perception
            )
            if caption_spans:
                selected_by_operation["perception.timestamped_caption"] = [
                    dict(span) for span in caption_spans if isinstance(span, dict)
                ]

        merged_perception = _merge_reference_perception_reports(reports)
        selected_spans = _combine_reference_grounded_spans(selected_by_operation)
        record = {
            "status": "grounded" if selected_spans else "failed_no_grounded_spans",
            "input_audio_path": str(input_path),
            "perception_operations": operations,
            "span_counts_by_operation": {
                key: len(value) for key, value in selected_by_operation.items()
            },
            "selected_span_count": len(selected_spans),
            "selection_policy": "speaker_event_intersection"
            if len(selected_by_operation) > 1
            else "single_perception_source",
            "source_text": source_text,
        }
        if not selected_spans:
            record["reason"] = (
                "reference audio extraction requires speaker_attributed_review or "
                "timestamped_caption to return concrete selected_spans"
            )
            state.setdefault("reference_audio_extraction_grounding", {})[step.id] = _without_empty(
                record
            )
            raise RuntimeError(
                "reference audio extraction could not resolve grounded spans via "
                f"{', '.join(operations) or 'perception'}"
            )

        params["selected_spans"] = selected_spans
        params.setdefault("selection_instruction", instruction)
        step.params["selected_spans"] = selected_spans
        step.params.setdefault("selection_instruction", instruction)
        record["selected_spans"] = selected_spans
        state.setdefault("reference_audio_extraction_grounding", {})[step.id] = _without_empty(
            record
        )
        _cache_perception_report_for_audio_path(state, input_path, merged_perception)
        return {
            "perception_report": merged_perception,
            "source_text": source_text
            or merged_perception.get("source_text")
            or state.get("source_text"),
            "selected_spans": selected_spans,
        }

    def _artifact_step_perception_for_input_audio(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        operation: str,
        instruction: str,
        input_audio_path: str,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        base_perception = state.get("perception_report") or {}
        base_source_text = state.get("source_text")
        if not _artifact_operation_needs_input_specific_perception(operation, instruction, params):
            return base_perception, base_source_text
        if not _perception_input_differs_from_base(input_audio_path, state):
            return base_perception, base_source_text
        try:
            input_path = Path(str(input_audio_path)).expanduser().resolve()
        except Exception:
            return base_perception, base_source_text
        if not _matches_known_input_or_reference_audio(input_path, state):
            return base_perception, base_source_text

        cached_perception = _perception_report_for_audio_path(state, input_path)
        if cached_perception:
            cached_source_text = cached_perception.get("source_text")
            return cached_perception, cached_source_text if isinstance(
                cached_source_text, str
            ) else base_source_text

        if not _implicit_executor_repair_enabled(state):
            raise OperationBoundaryError(
                "the selected audio input needs its own perception evidence; strict "
                "forward execution forbids implicit perception repair"
            )

        other_audios = []
        for audio in collect_input_audios(state):
            try:
                resolved = Path(str(audio)).expanduser().resolve()
            except Exception:
                continue
            if resolved != input_path:
                other_audios.append(str(resolved))
        try:
            result = build_audio_perception_fallback(
                agentic_mode=_perception_provider_mode(step, state, self.agentic_mode),
                speech_client=self.speech_perception_client,
            ).perceive(
                user_prompt=instruction,
                audio_path=str(input_path),
                base_perception={},
                reference_audio_paths=other_audios,
                output_dir=str(Path(state["output_dir"]) / f"{step.id}_input_audio_perception"),
                requirements=PerceptionRequirements(
                    source_text=True,
                    word_timestamps=True,
                    speaker_segments=True,
                    speaker_profiles=True,
                    segment_descriptions=True,
                ),
                run_specialized=True,
                allow_gemini=True,
                force_gemini=True,
                refresh_cache=False,
                strict=False,
            )
        except Exception as exc:
            state.setdefault("perception_audio_repairs", {})[step.id] = {
                "status": "failed",
                "input_audio_path": str(input_path),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            return base_perception, base_source_text

        perception = result.perception
        state.setdefault("perception_audio_repairs", {})[step.id] = {
            "status": "refreshed",
            "input_audio_path": str(input_path),
            "speaker_segment_count": len(
                perception.get("speaker_segments") or perception.get("segments") or []
            ),
            "speaker_count": len(perception.get("speakers") or []),
            "source_text": perception.get("source_text"),
            "satisfied": result.satisfied,
            "missing": result.missing,
        }
        return perception, perception.get("source_text") or base_source_text

    def _handle_export_wav(self, step: AudioEditStep, state: dict[str, Any]) -> dict[str, Any]:
        current = _resolve_input_audio(step, state)
        output_path = _output_path(step, state, prefer_plan_output=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(current).resolve() != Path(output_path).resolve():
            shutil.copy2(current, output_path)
        return {
            "output_audio_path": output_path,
            "operation": "export_wav",
            "source_audio_path": current,
        }

    def _handle_unsupported(self, step: AudioEditStep, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            step.params.get("reason") or step.description or "Unsupported audio edit request"
        )


def _resolve_input_audio(step: AudioEditStep, state: dict[str, Any]) -> str:
    value = (
        step.inputs.get("input_audio_path")
        or step.inputs.get("input_audio")
        or step.inputs.get("audio_path")
        or step.inputs.get("audio")
        or step.inputs.get("source_audio_path")
        or step.inputs.get("source_audio")
        or step.inputs.get("mixture_audio_path")
        or step.inputs.get("mixture_audio")
        or step.params.get("input_audio_path")
        or step.params.get("input_audio")
        or step.params.get("audio_path")
        or step.params.get("audio")
        or step.params.get("source_audio_path")
        or step.params.get("source_audio")
        or step.params.get("mixture_audio_path")
        or step.params.get("mixture_audio")
        or "current"
    )
    value = resolve_audio_reference(value, state)
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item), None)
    if not value:
        raise RuntimeError(f"Step {step.id} requires an input audio path")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Input audio path does not exist for step {step.id}: {path}")
    return str(path)


def _perception_invocation_audio_bindings(
    state: dict[str, Any],
    *,
    base_audio_path: str,
    fallback_reference_audio_paths: list[str],
) -> tuple[str | None, list[str] | None, list[str]]:
    """Bind an executor Perception invocation to its frozen logical namespace."""

    plan = state.get("plan")
    metadata = getattr(plan, "metadata", {}) if plan is not None else {}
    declared_refs = list(metadata.get("workflow_perception_audio_refs") or [])
    if not declared_refs or not opaque_audio_resolution_enabled(state):
        return None, None, fallback_reference_audio_paths

    base_ref = canonical_audio_ref_for_runtime_path(base_audio_path, state)
    if base_ref is None or base_ref not in declared_refs:
        raise AudioReferenceResolutionError(
            "Perception primary audio does not map to its frozen request namespace"
        )
    ordered_reference_refs = [item for item in declared_refs if item != base_ref]
    ordered_reference_paths: list[str] = []
    for audio_ref in ordered_reference_refs:
        resolved = resolve_audio_reference(audio_ref, state)
        if not isinstance(resolved, str) or not Path(resolved).is_file():
            raise AudioReferenceResolutionError(
                "Perception audio ref does not have one materialized invocation input"
            )
        ordered_reference_paths.append(resolved)
    return base_ref, ordered_reference_refs, ordered_reference_paths


def _merge_audio_assembly_input_fields(
    step: AudioEditStep, state: dict[str, Any], params: dict[str, Any]
) -> None:
    if params.get("target_audio_path"):
        return
    for key in (
        "target_audio_path",
        "target_audio",
        "generated_audio_path",
        "generated_audio",
        "replacement_audio_path",
        "replacement_audio",
        "insert_audio_path",
        "insert_audio",
    ):
        value = step.inputs.get(key)
        if value:
            params["target_audio_path"] = resolve_audio_reference(value, state)
            return


def _requests_original_voice_reference(step: AudioEditStep, state: dict[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in [
            state.get("user_prompt"),
            step.description,
            step.params.get("instruction"),
            step.params.get("voice_prompt"),
        ]
    ).lower()
    return any(
        phrase in text
        for phrase in [
            "preserve his original voice",
            "preserve her original voice",
            "preserve the original voice",
            "preserve original voice",
            "original speaker",
            "same speaker",
            "source speaker",
            "clone the speaker",
            "clone his voice",
            "clone her voice",
        ]
    )


def _original_voice_reference_audio(
    state: dict[str, Any], *, exclude: str | None = None
) -> str | None:
    candidates: list[Any] = [
        state.get("source_text_audio_path"),
        state.get("input_audio"),
    ]
    candidates.extend(collect_input_audios(state))
    for candidate in candidates:
        resolved = resolve_audio_reference(candidate, state)
        if not isinstance(resolved, str) or not resolved:
            continue
        if exclude and _same_audio_path(resolved, exclude):
            continue
        if Path(resolved).exists():
            return resolved
    return None


def _original_session_audio_reference(
    state: dict[str, Any], *, exclude: str | None = None
) -> str | None:
    artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), dict) else {}
    candidates: list[Any] = [
        artifacts.get("base") if isinstance(artifacts, dict) else None,
        artifacts.get("input_audio") if isinstance(artifacts, dict) else None,
        state.get("source_text_audio_path"),
        state.get("input_audio"),
        "original",
        "base",
        "input_audio",
    ]
    candidates.extend(collect_input_audios(state))
    for candidate in candidates:
        resolved = resolve_audio_reference(candidate, state)
        if not isinstance(resolved, str) or not resolved:
            continue
        if exclude and _same_audio_path(resolved, exclude):
            continue
        if Path(resolved).exists():
            return resolved
    return None


def _same_audio_path(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return str(left) == str(right)


def _normalize_reconstruct_step_reference_edits(
    *,
    step: AudioEditStep,
    params: dict[str, Any],
    state: dict[str, Any],
) -> None:
    edits = params.get("edits")
    if not isinstance(edits, list) or not edits:
        return
    normalized: list[Any] = []
    repairs: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    for index, item in enumerate(edits):
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        replacement_path = _reconstruct_replacement_path_from_edit(item, state)
        spans = _reconstruct_spans_from_edit(item, state)
        has_explicit_span = item.get("start_sec") is not None or item.get("end_sec") is not None
        span_repair_strategy: str | None = None
        if replacement_path and has_explicit_span:
            replacement_spans = _reconstruct_replacement_linked_spans(item, replacement_path, state)
            if replacement_spans and not _span_dicts_cover_same_targets(spans, replacement_spans):
                spans = replacement_spans
                span_repair_strategy = "replace_explicit_span_with_linked_extract_span"
        rewritten = False
        if replacement_path and spans and (not has_explicit_span or span_repair_strategy):
            for span in spans:
                start = span.get("start_sec", span.get("start"))
                end = span.get("end_sec", span.get("end"))
                if start is None or end is None:
                    continue
                normalized.append(
                    {
                        **_strip_reconstruct_step_reference_keys(item),
                        "audio_path": replacement_path,
                        "start_sec": start,
                        "end_sec": end,
                        "mode": item.get("mode", "replace"),
                    }
                )
                rewritten = True
            if rewritten:
                repairs.append(
                    {
                        "edit_index": index,
                        "strategy": span_repair_strategy
                        or "resolve_step_reference_replacement_and_spans",
                        "replacement_audio_path": replacement_path,
                        "span_count": len(spans),
                        "original_start_sec": item.get("start_sec"),
                        "original_end_sec": item.get("end_sec"),
                    }
                )
                continue
        if replacement_path and not _reconstruct_edit_has_audio_path(item):
            item = {**item, "audio_path": replacement_path}
            rewritten = True
        normalized.append(item)
        if rewritten:
            normalizations.append(
                {
                    "edit_index": index,
                    "strategy": "resolve_explicit_step_reference_replacement_audio",
                    "replacement_audio_path": replacement_path,
                }
            )
    if not repairs and not normalizations:
        return
    params["edits"] = normalized
    step.params["edits"] = normalized
    if repairs:
        state.setdefault("artifact_ops_reconstruct_repairs", {})[step.id] = repairs
    if normalizations:
        state.setdefault("artifact_ops_reconstruct_normalizations", {}).setdefault(
            step.id, []
        ).extend(normalizations)


def _merge_reconstruct_input_and_param_edits(
    *,
    step: AudioEditStep,
    params: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Merge split reconstruct specs when spans and replacement audio arrive separately."""

    param_edits = params.get("edits")
    input_edits = step.inputs.get("edits")
    if not isinstance(param_edits, list) or not isinstance(input_edits, list):
        return
    if not param_edits or not input_edits:
        return

    normalized: list[Any] = []
    repairs: list[dict[str, Any]] = []
    for index, item in enumerate(param_edits):
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        input_item = input_edits[index] if index < len(input_edits) else None
        if not isinstance(input_item, dict):
            normalized.append(item)
            continue
        merged = dict(item)
        copied: list[str] = []
        for key in (
            "input_audio_path",
            "audio_path",
            "path",
            "replacement_audio",
            "replacement_audio_path",
            "edited_audio",
            "edited_audio_path",
            "input_from_step",
            "replacement_from_step",
            "edited_audio_from_step",
            "audio_from_step",
        ):
            if merged.get(key):
                continue
            value = input_item.get(key)
            if value:
                merged[key] = value
                copied.append(key)
        if copied:
            repairs.append(
                {
                    "edit_index": index,
                    "strategy": "merge_input_edits_replacement_audio_into_param_edits",
                    "copied_keys": copied,
                }
            )
        normalized.append(merged)

    if not repairs:
        return
    params["edits"] = normalized
    step.params["edits"] = normalized
    state.setdefault("artifact_ops_reconstruct_repairs", {}).setdefault(step.id, []).extend(repairs)


def _reconstruct_replacement_linked_spans(
    item: dict[str, Any],
    replacement_path: str,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    for key in (
        "replace_span_from_step",
        "span_from_step",
        "source_span_from_step",
        "selected_span_from_step",
        "replace_span_step",
    ):
        spans = _selected_spans_from_step_output(_step_output_from_reference(item.get(key), state))
        if spans:
            return spans
    latest_extract = _latest_artifact_extract_output(state)
    spans = _selected_spans_from_step_output(latest_extract)
    if not spans:
        return []
    return spans


def _span_dicts_cover_same_targets(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    left_pairs = [_span_pair_from_value(item) for item in left if isinstance(item, dict)]
    right_pairs = [_span_pair_from_value(item) for item in right if isinstance(item, dict)]
    left_pairs = [item for item in left_pairs if item is not None]
    right_pairs = [item for item in right_pairs if item is not None]
    if len(left_pairs) != len(right_pairs):
        return False
    for (left_start, left_end), (right_start, right_end) in zip(
        sorted(left_pairs), sorted(right_pairs)
    ):
        intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
        union = max(left_end, right_end) - min(left_start, right_start)
        if union <= 0.0 or intersection / union < 0.7:
            return False
    return True


def _reconstruct_edit_has_audio_path(item: dict[str, Any]) -> bool:
    return any(
        item.get(key)
        for key in ("input_audio_path", "audio_path", "path", "replacement_audio_path")
    )


def _strip_reconstruct_step_reference_keys(item: dict[str, Any]) -> dict[str, Any]:
    drop = {
        "input_from_step",
        "replacement_from_step",
        "edited_audio_from_step",
        "audio_from_step",
        "replace_span_from_step",
        "span_from_step",
        "spans_from_step",
        "spans_from_artifact_id",
        "source_span_from_step",
        "source_spans_from_step",
        "source_spans_from_artifact_id",
        "selected_span_from_step",
        "selected_spans_from_step",
        "selected_spans_from_artifact_id",
        "replace_span_step",
    }
    return {key: value for key, value in item.items() if key not in drop}


def _reconstruct_replacement_path_from_edit(
    item: dict[str, Any], state: dict[str, Any]
) -> str | None:
    for key in (
        "input_audio_path",
        "audio_path",
        "path",
        "replacement_audio_path",
        "replacement_audio",
        "edited_audio_path",
        "edited_audio",
        "input_from_step",
        "replacement_from_step",
        "edited_audio_from_step",
        "audio_from_step",
    ):
        value = item.get(key)
        path = _audio_path_from_step_reference(value, state)
        if path:
            return path
    return None


def _reconstruct_spans_from_edit(
    item: dict[str, Any], state: dict[str, Any]
) -> list[dict[str, Any]]:
    for key in ("selected_spans", "spans"):
        value = item.get(key)
        if isinstance(value, list) and value:
            return [dict(span) for span in value if isinstance(span, dict)]
    if item.get("start_sec") is not None or item.get("end_sec") is not None:
        return [
            {
                "start_sec": item.get("start_sec", 0.0),
                "end_sec": item.get("end_sec", item.get("start_sec", 0.0)),
            }
        ]
    for key in (
        "replace_span_from_step",
        "span_from_step",
        "spans_from_step",
        "spans_from_artifact_id",
        "source_span_from_step",
        "source_spans_from_step",
        "source_spans_from_artifact_id",
        "selected_span_from_step",
        "selected_spans_from_step",
        "selected_spans_from_artifact_id",
        "replace_span_step",
    ):
        output = _step_output_from_reference(item.get(key), state)
        spans = _selected_spans_from_step_output(output)
        if spans:
            return spans
    return []


def _audio_path_from_step_reference(value: Any, state: dict[str, Any]) -> str | None:
    if not value:
        return None
    output = _step_output_from_reference(value, state)
    path = _extract_output_path(output) if output else None
    if path and Path(path).is_file():
        return path
    if isinstance(value, str):
        resolved = resolve_audio_reference(value, state)
        if isinstance(resolved, str) and Path(resolved).is_file():
            return resolved
    return None


def _step_output_from_reference(value: Any, state: dict[str, Any]) -> dict[str, Any] | None:
    if not value:
        return None
    step_outputs = state.get("step_outputs") if isinstance(state.get("step_outputs"), dict) else {}
    if isinstance(value, str) and isinstance(step_outputs.get(value), dict):
        return step_outputs[value]
    resolved = resolve_audio_reference(value, state)
    if isinstance(resolved, str):
        for output in step_outputs.values():
            if not isinstance(output, dict):
                continue
            output_path = _extract_output_path(output)
            if output_path and _same_audio_reference_path(resolved, output_path):
                return output
    return None


def _selected_spans_from_step_output(
    output: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    spans = output.get("selected_spans")
    if isinstance(spans, list) and spans:
        return [dict(span) for span in spans if isinstance(span, dict)]
    manifest_path = output.get("manifest_path")
    if isinstance(manifest_path, str) and Path(manifest_path).is_file():
        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, json.JSONDecodeError):
            return []
        spans = manifest.get("selected_spans")
        if isinstance(spans, list):
            return [dict(span) for span in spans if isinstance(span, dict)]
    return []


def _same_audio_reference_path(left: str, right: str) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return str(left) == str(right)


def _maybe_reuse_previous_selected_spans_for_followup(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    operation: str,
    params: dict[str, Any],
    instruction: str,
) -> dict[str, Any] | None:
    if operation != "speaker_scoped_edit":
        return None
    if any(
        params.get(key)
        for key in (
            "selected_spans",
            "spans",
            "time_spans",
            "segments",
            "segments_to_edit",
        )
    ):
        return None
    selector = _artifact_followup_selector_text(step=step, params=params, instruction=instruction)
    if not _requests_previous_selected_spans(selector):
        return None
    prior = _latest_artifact_selected_span_output(state)
    spans = _selected_spans_from_step_output(prior)
    if not spans:
        return None
    params["selected_spans"] = spans
    params.setdefault("selection_instruction", selector or instruction)
    params.setdefault("preserve_timeline", True)
    return {
        "source": "previous_artifact_selected_spans",
        "selector": selector,
        "span_count": len(spans),
        "prior_operation": prior.get("operation") if isinstance(prior, dict) else None,
        "prior_output_audio_path": prior.get("output_audio_path")
        if isinstance(prior, dict)
        else None,
    }


def _artifact_followup_selector_text(
    *,
    step: AudioEditStep,
    params: dict[str, Any],
    instruction: str,
) -> str:
    parts: list[str] = [instruction, step.description or ""]
    for key in (
        "selection_instruction",
        "selector_instruction",
        "select_instruction",
        "span_instruction",
        "selector",
        "target_description",
    ):
        value = params.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for nested_key in (
                "selection_instruction",
                "selector_instruction",
                "description",
                "target",
            ):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    parts.append(nested)
    return " ".join(part for part in parts if part).strip()


def _requests_previous_selected_spans(text: str) -> bool:
    lowered = text.lower()
    if not lowered:
        return False
    return bool(
        re.search(
            r"\b(?:previously|already|just|last|earlier)\s+(?:edited|selected|processed|made|converted)\b",
            lowered,
        )
        or re.search(
            r"\b(?:previous|prior|same|those|these|the)\s+(?:edited|selected|processed)\s+(?:segment|segments|span|spans|speech)\b",
            lowered,
        )
        or re.search(
            r"\b(?:segment|segments|span|spans|speech)\s+(?:that|which)\s+(?:were|was)\s+(?:just\s+)?(?:edited|selected|processed|made|converted)\b",
            lowered,
        )
        or "just made" in lowered
        or "were just made" in lowered
    )


def _latest_artifact_selected_span_output(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    step_outputs = state.get("step_outputs")
    if not isinstance(step_outputs, dict):
        return None
    preferred: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for output in reversed(list(step_outputs.values())):
        if not isinstance(output, dict) or not _selected_spans_from_step_output(output):
            continue
        operation = str(output.get("operation") or "").lower()
        if operation in {"speaker_scoped_edit", "artifact_ops.speaker_scoped_edit"}:
            preferred = output
            break
        if fallback is None and operation in {
            "extract_spans",
            "artifact_ops.extract_spans",
            "remove_spans",
            "artifact_ops.remove_spans",
            "mute_spans",
            "artifact_ops.mute_spans",
            "beep_spans",
            "artifact_ops.beep_spans",
        }:
            fallback = output
    return preferred or fallback


def _filter_non_target_span_params(
    params: dict[str, Any], *, operation: str
) -> dict[str, Any] | None:
    """Drop planner-review context spans from executable span params.

    PROMPT2 span decisions may contain context, excluded, or reference-only
    spans. Those are useful evidence, but artifact operations must only execute
    on true target spans.
    """

    records: list[dict[str, Any]] = []
    removed_all = False
    for key in ("selected_spans", "spans"):
        value = params.get(key)
        if not _span_list_has_decision_labels(value):
            continue
        kept, dropped = _target_span_items(value)
        params[key] = kept
        removed_all = removed_all or (bool(value) and not kept)
        records.append(
            {
                "param": key,
                "before_count": len(value),
                "after_count": len(kept),
                "dropped": [_span_filter_record(item) for item in dropped],
            }
        )

    edits = params.get("edits")
    if _span_list_has_decision_labels(edits):
        kept, dropped = _target_span_items(edits)
        params["edits"] = kept
        removed_all = removed_all or (bool(edits) and not kept)
        records.append(
            {
                "param": "edits",
                "before_count": len(edits),
                "after_count": len(kept),
                "dropped": [_span_filter_record(item) for item in dropped],
            }
        )

    if not records:
        return None
    return {
        "operation": operation,
        "reason": "filtered planner span decisions so context/reference/excluded spans are evidence only",
        "removed_all_explicit_targets": removed_all,
        "filters": records,
    }


def _span_list_has_decision_labels(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, dict)
        and any(
            key in item for key in ("decision", "operation_role", "span_decision", "selection_role")
        )
        for item in value
    )


def _target_span_items(items: list[Any]) -> tuple[list[Any], list[Any]]:
    kept: list[Any] = []
    dropped: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        if _span_item_is_executable_target(item):
            kept.append(item)
        else:
            dropped.append(item)
    return kept, dropped


def _span_item_is_executable_target(item: dict[str, Any]) -> bool:
    decision = str(item.get("decision") or item.get("span_decision") or "").strip().lower()
    role = str(item.get("operation_role") or item.get("selection_role") or "").strip().lower()
    if decision in {
        "exclude",
        "preserve_context",
        "needs_more_perception",
        "reference_span",
    }:
        return False
    if role in {
        "context",
        "preserve_context",
        "reference_voice",
        "reference_accent",
        "reference_audio",
        "unknown",
    }:
        return False
    if decision in {"edit_span", "target_span", "selected_target"}:
        return True
    if role.startswith("target_") or role in {
        "target",
        "edit",
        "extract",
        "remove",
        "mute",
        "beep",
    }:
        return True
    return not decision and not role


def _span_filter_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "candidate_id",
            "segment_id",
            "decision",
            "operation_role",
            "start_sec",
            "end_sec",
            "speaker_id",
            "text",
            "reason",
        )
        if item.get(key) not in (None, "", [], {})
    }


def _maybe_use_base_audio_for_absolute_spans(
    *,
    step: AudioEditStep,
    operation: str,
    params: dict[str, Any],
    input_audio_path: str,
    state: dict[str, Any],
) -> str:
    if operation not in {
        "extract_spans",
        "remove_spans",
        "mute_spans",
        "beep_spans",
        "reconstruct",
    }:
        return input_audio_path
    raw_timeline_reference = params.pop(
        "timeline_reference",
        params.pop("span_timeline", None),
    )
    timeline_reference = str(raw_timeline_reference or "").strip().lower()
    if timeline_reference in {"current", "intermediate", "relative"}:
        return input_audio_path
    explicit_base_timeline = timeline_reference in {
        "base",
        "original",
        "source",
        "absolute",
    }
    span_specs = params.get("edits") if operation == "reconstruct" else None
    span_value = span_specs or params.get("selected_spans") or params.get("spans")
    max_end = _max_explicit_span_end(span_value)
    min_start = _min_explicit_span_start(span_value)
    current_duration = _audio_duration_seconds(Path(input_audio_path))
    base_audio = state.get("input_audio")
    if (
        not explicit_base_timeline
        and max_end is None
        and not (
            min_start is not None
            and current_duration is not None
            and min_start > current_duration + 0.05
        )
    ):
        return input_audio_path
    if (
        not base_audio
        or str(base_audio) == str(input_audio_path)
        or not Path(str(base_audio)).exists()
    ):
        return input_audio_path
    if not explicit_base_timeline and (
        current_duration is None or (max_end is not None and max_end <= current_duration + 0.05)
    ):
        return input_audio_path
    base_duration = _audio_duration_seconds(Path(str(base_audio)))
    if base_duration is None or (min_start is not None and min_start >= base_duration + 0.05):
        return input_audio_path
    rule = absolute_timeline_input_binding_rule()
    state.setdefault("artifact_ops_input_bindings", {})[step.id] = {
        **rule.record(step_id=step.id),
        "strategy": "bind_absolute_timeline_spans_to_base_audio",
        "operation": step.operation,
        "previous_input_audio_path": input_audio_path,
        "base_audio_path": str(base_audio),
        "timeline_reference": "base",
        "decision_basis": (
            "explicit_timeline_reference"
            if explicit_base_timeline
            else "span_outside_current_inside_base"
        ),
        "max_span_end_sec": max_end if max_end is not None else "open_ended",
        "previous_input_duration_sec": current_duration,
        "base_duration_sec": base_duration,
    }
    step.inputs["input_audio_path"] = str(base_audio)
    for alias in ("input_audio", "audio_path", "audio"):
        if alias in step.inputs:
            step.inputs[alias] = str(base_audio)
    record_runtime_rewrite_rule_hit(rule, step=step, state=state)
    return str(base_audio)


def _maybe_use_base_audio_for_original_speaker_selector(
    *,
    step: AudioEditStep,
    operation: str,
    params: dict[str, Any],
    instruction: str,
    input_audio_path: str,
    state: dict[str, Any],
) -> str:
    if operation not in {"extract_spans", "remove_spans", "mute_spans", "beep_spans"}:
        return input_audio_path
    base_audio = state.get("input_audio")
    if (
        not base_audio
        or str(base_audio) == str(input_audio_path)
        or not Path(str(base_audio)).exists()
    ):
        return input_audio_path
    perception = state.get("perception_report") or {}
    if not isinstance(perception, dict) or not (
        perception.get("speaker_segments") or perception.get("segments")
    ):
        return input_audio_path
    current_duration = _audio_duration_seconds(Path(input_audio_path))
    base_duration = _audio_duration_seconds(Path(str(base_audio)))
    if current_duration is None or base_duration is None:
        return input_audio_path
    selector = _selector_text(
        params.get("selector")
        or params.get("source_selector")
        or params.get("target_selector")
        or params.get("speaker_selector")
    )
    candidates = [
        item for item in [selector, instruction] if isinstance(item, str) and item.strip()
    ]
    selected = []
    selected_from = None
    for candidate in candidates:
        spans = select_timeline_spans(
            candidate, perception, duration_sec=base_duration, action=operation
        )
        if spans:
            selected = spans
            selected_from = candidate
            break
    if not selected:
        return input_audio_path
    max_end = max(span.end_sec for span in selected)
    if max_end <= current_duration + 0.05:
        return input_audio_path
    if not _implicit_executor_repair_enabled(state):
        raise RuntimeError(
            "Implicit executor repair is disabled: "
            f"step {step.id!r} ({step.operation}) requires "
            "use_base_audio_for_original_speaker_selector before tool execution "
            f"because selector {selected_from!r} resolves through {max_end:.6f}s "
            f"but the current input duration is {current_duration:.6f}s"
        )
    state.setdefault("artifact_ops_input_repairs", {})[step.id] = {
        "strategy": "use_base_audio_for_original_speaker_selector",
        "operation": step.operation,
        "selector": selected_from,
        "previous_input_audio_path": input_audio_path,
        "base_audio_path": str(base_audio),
        "selected_span_count": len(selected),
        "max_span_end_sec": round(max_end, 6),
        "previous_input_duration_sec": current_duration,
        "base_duration_sec": base_duration,
    }
    return str(base_audio)


def _max_explicit_span_end(spans_value: Any) -> float | None:
    if not isinstance(spans_value, list):
        return None
    max_end: float | None = None
    for item in spans_value:
        end_value = None
        if isinstance(item, dict):
            end_value = item.get("end_sec", item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            end_value = item[1]
        end = _safe_float(end_value)
        if end is None:
            continue
        max_end = end if max_end is None else max(max_end, end)
    return max_end


def _min_explicit_span_start(spans_value: Any) -> float | None:
    if not isinstance(spans_value, list):
        return None
    min_start: float | None = None
    for item in spans_value:
        start_value = None
        if isinstance(item, dict):
            start_value = item.get("start_sec", item.get("start"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start_value = item[0]
        start = _safe_float(start_value)
        if start is None:
            continue
        min_start = start if min_start is None else min(min_start, start)
    return min_start


def _has_self_contained_structured_runtime_params(step: AudioEditStep) -> bool:
    if step.operation == "qwen3_tts" and _step_requests_synthetic_silence(step):
        return True
    if step.operation.startswith("edit_data_pipeline."):
        return bool(
            step.params.get("source_text")
            and (
                step.params.get("instruction")
                or step.params.get("edit_instruction")
                or step.params.get("instruction_st")
                or step.params.get("target_text")
            )
        )
    if step.operation == "no_reference_voice_conversion":
        return True
    if step.operation == "audio_effects":
        has_audio = any(
            step.inputs.get(key) or step.params.get(key)
            for key in (
                "input_audio_path",
                "input_audio",
                "audio_path",
                "audio",
                "source_audio_path",
                "source_audio",
            )
        )
        has_effect = bool(
            step.params.get("preset")
            or step.params.get("effect")
            or step.params.get("effect_name")
            or step.params.get("best_effort")
        )
        return has_audio and has_effect
    if step.operation == "seed_vc":
        return any(
            step.params.get(key) or step.inputs.get(key)
            for key in (
                "reference_audio_path",
                "reference_audio",
                "ref_audio_path",
                "speaker_audio_path",
            )
        )
    if step.operation == "intra_audio_voice_transfer":
        has_source = any(
            key in step.params
            for key in (
                "source_speaker_ids",
                "source_speaker_id",
                "source_speakers",
                "source_speaker",
            )
        )
        has_reference = any(
            key in step.params
            for key in (
                "reference_speaker_id",
                "target_speaker_id",
                "target_voice_speaker_id",
                "reference_speaker",
                "target_speaker",
                "target_voice_speaker",
            )
        )
        return has_source and has_reference
    if step.operation == "artifact_ops.speaker_scoped_edit":
        nested_operation = str(
            step.params.get("edit_op")
            or step.params.get("edit_operation")
            or step.params.get("sub_operation")
            or ""
        ).lower()
        nested_params = (
            step.params.get("edit_op_params")
            if isinstance(step.params.get("edit_op_params"), dict)
            else {}
        )
        has_reference = any(
            step.params.get(key) or step.inputs.get(key) or nested_params.get(key)
            for key in (
                "reference_audio_path",
                "reference_audio",
                "ref_audio_path",
                "speaker_audio_path",
            )
        )
        if has_reference and any(
            token in nested_operation for token in ("seed_vc", "voice_conversion", "voice_clone")
        ):
            return True
    if step.operation in {
        "artifact_ops.extract_spans",
        "artifact_ops.remove_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.beep_spans",
    }:
        return bool(step.params.get("instruction") or step.description)
    if step.operation == "artifact_ops.reconstruct" and isinstance(step.params.get("edits"), list):
        return True
    if step.operation == "seamless_expressive_s2st":
        has_target_language = any(
            step.params.get(key) for key in ("target_language", "target_lang", "language")
        ) or bool(_target_language_from_request(step.description))
        has_audio = any(
            step.inputs.get(key) or step.params.get(key)
            for key in (
                "input_audio_path",
                "input_audio",
                "audio_path",
                "audio",
                "source_audio_path",
                "source_audio",
            )
        )
        return has_target_language and has_audio
    return False


def _record_executor_gemini_metrics(
    state: dict[str, Any],
    step: AudioEditStep,
    gemini_executor: Any | None,
) -> None:
    client = getattr(gemini_executor, "client", None)
    metrics = getattr(client, "last_call_metrics", None)
    if not isinstance(metrics, dict):
        return
    provider = _executor_provider_name(gemini_executor)
    payload = {"provider": provider, **dict(metrics)}
    state.setdefault("executor_llm_call_metrics", {})[step.id] = payload
    if provider == GEMINI_MODE or str(provider).startswith("gemini"):
        state.setdefault("executor_gemini_call_metrics", {})[step.id] = dict(metrics)


def _plan_executor_tool_call(
    planner: Any,
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    operation_spec: dict[str, Any],
    manual: ToolManual | None,
    verifier_error: dict[str, Any] | str | None,
) -> ExecutorToolCall:
    kwargs = {
        "step": step,
        "state": state,
        "operation_spec": operation_spec,
        "manual": manual,
    }
    try:
        signature = inspect.signature(planner.plan_tool_call)
    except (TypeError, ValueError):
        signature = None
    if (
        signature is None
        or "verifier_error" in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    ):
        kwargs["verifier_error"] = verifier_error
    return planner.plan_tool_call(**kwargs)


def _executor_provider_name(planner: Any | None) -> str:
    if planner is None:
        return "none"
    provider = getattr(planner, "provider_name", None)
    if isinstance(provider, str) and provider:
        return provider
    name = planner.__class__.__name__
    if "Gemini" in name:
        return GEMINI_MODE
    if "GLM" in name:
        return "glm_4_7_flash"
    return name


def _executor_planner_matches_mode(planner: Any | None, agentic_mode: str) -> bool:
    provider = _executor_provider_name(planner)
    if agentic_mode == OPEN_GLM_QWEN_OMNI_MODE:
        return provider == "glm_4_7_flash"
    return provider == GEMINI_MODE or str(provider).startswith("gemini")


def _perception_provider_mode(step: AudioEditStep, state: dict[str, Any], default_mode: str) -> str:
    if step.operation == "perception.gemini_audio":
        return GEMINI_MODE
    if step.operation in {"perception.qwen3_omni_audio", "perception.open_omni_audio"}:
        return OPEN_GLM_QWEN_OMNI_MODE
    return resolve_agentic_mode(state.get("context", {}).get("agentic_mode") or default_mode)


def _timestamped_caption_route_id(
    reviewer: Any | None,
    *,
    use_flexsed_prior: bool,
) -> str:
    prefix = "flexsed_prior_plus_" if use_flexsed_prior else ""
    return f"{prefix}{_reviewer_provider_id(reviewer)}_timestamped_caption"


def _reviewer_provider_id(reviewer: Any | None) -> str:
    provider = str(getattr(reviewer, "provider_name", None) or "model")
    return provider.strip().lower().replace("-", "_")


def _emphasis_target_phrase_from_params(params: dict[str, Any], request: Any) -> str | None:
    for key in ("target_phrase", "phrase", "target_word", "word"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    instruction_st = str(params.get("instruction_st") or "")
    match = re.search(
        r"<(?:pitch|rate|stress|emphasis)\b[^>]*>(.*?)</(?:pitch|rate|stress|emphasis)>",
        instruction_st,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = re.sub(r"<[^>]+>", "", match.group(1))
        text = unescape(text).strip()
        if text:
            return text
    request_text = str(request or "")
    quoted = _first_quoted_phrase(request_text)
    return quoted.strip() if quoted else None


def _target_phrase_tagged_text(source_text: str, target_phrase: str) -> str:
    source = str(source_text)
    phrase = str(target_phrase).strip()
    if not phrase:
        return source
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return pattern.sub(lambda match: f"<target>{match.group(0)}</target>", source, count=1)


def _maybe_rewrite_volume_prosody_as_scoped_gain(
    step: AudioEditStep, state: dict[str, Any]
) -> bool:
    if step.operation not in {
        "edit_data_pipeline.pitch_rate_edit",
        "edit_data_pipeline.text_edit",
    }:
        return False
    params = dict(step.params)
    request = str(state.get("user_prompt") or step.description or params.get("instruction") or "")
    instruction_st = str(params.get("instruction_st") or "")
    combined = f"{request} {instruction_st}".lower()
    if "volume" not in combined and "loudness" not in combined and "prosody volume" not in combined:
        return False
    phrase = _phrase_from_volume_prosody_tag(instruction_st) or _first_quoted_phrase(request)
    selection = f'the spoken phrase "{phrase}"' if phrase else request
    target_db = -18.0
    if any(
        marker in combined
        for marker in ["decrease", "reduce", "lower", "quieter", "soft", "weaken"]
    ):
        target_db = -26.0
    elif any(marker in combined for marker in ["raise", "increase", "boost", "louder", "loud"]):
        target_db = -15.0
    state.setdefault("operation_rewrites", {})[step.id] = {
        "from": step.operation,
        "to": "artifact_ops.speaker_scoped_edit",
        "reason": "planner supplied volume/prosody tags to an edit-data-pipeline text/prosody step; route phrase-scoped gain through artifact_ops",
        "selection_instruction": selection,
        "target_db": target_db,
    }
    step.operation = "artifact_ops.speaker_scoped_edit"
    step.params = {
        "selection_instruction": selection,
        "edit_op": "volume_normalize",
        "edit_op_params": {"target_db": target_db, "method": "rms"},
        "preserve_timeline": True,
    }
    return True


def _phrase_from_volume_prosody_tag(instruction_st: str) -> str | None:
    match = re.search(
        r"<prosody\s+[^>]*volume=[\"'][^\"']+[\"'][^>]*>(.*?)</prosody>",
        instruction_st,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    phrase = re.sub(r"<[^>]+>", "", match.group(1)).strip()
    return phrase or None


def _first_quoted_phrase(text: str) -> str | None:
    for value in _quoted_phrases(text):
        if value.strip():
            return value.strip()
    return None


def _quoted_phrases(text: str | None) -> list[str]:
    values: list[str] = []
    single_quoted = r"(?<!\w)'((?:[^']|(?<=\w)'(?=\w))*)'(?!\w)"
    for match in re.finditer(rf"“([^”]+)”|{single_quoted}|\"([^\"]+)\"", text or ""):
        value = next((group for group in match.groups() if group is not None), "")
        if value.strip():
            values.append(value.strip())
    return values


def _phrase_replacement_pair_from_request(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> tuple[str, str] | None:
    direct_source = _first_string_value(
        params,
        "source_phrase",
        "source_word",
        "old_text",
        "from_text",
        "replace_from",
    )
    direct_target = _first_string_value(
        params,
        "target_phrase",
        "target_word",
        "new_text",
        "to_text",
        "replace_to",
        "replacement_text",
    )
    evidence = " ".join(
        str(value)
        for value in [
            state.get("user_prompt"),
            step.description,
            params.get("instruction"),
            params.get("edit_instruction"),
        ]
        if value
    )
    if direct_source and direct_target and _looks_like_replacement_request(evidence):
        return direct_source, direct_target

    quoted = _quoted_phrases(evidence)
    if len(quoted) >= 2 and _looks_like_replacement_request(evidence):
        return quoted[0], quoted[1]

    patterns = [
        r"\breplace\s+(?:every\s+|all\s+|each\s+)?(?:instances?\s+of\s+|occurrences?\s+of\s+)?(?:the\s+)?(?:word|phrase|term)?\s*(?P<old>[A-Za-z0-9_.+-]+)\s+(?:with|by)\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
        r"\bchange\s+(?:the\s+)?(?:word|phrase|term)?\s*(?P<old>[A-Za-z0-9_.+-]+)\s+(?:to|into)\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
        r"(?:把|将)(?P<old>[\w\u4e00-\u9fff_.+-]+)(?:替换成|替换为|换成|改成|改为)(?P<new>[\w\u4e00-\u9fff_.+-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, evidence, flags=re.IGNORECASE)
        if not match:
            continue
        source = match.group("old").strip(" :，,。.;!?\"'")
        target = match.group("new").strip(" :，,。.;!?\"'")
        if source and target:
            return source, target

    direct_target = direct_target or _first_string_value(params, "target_text", "text")
    if direct_target and len(quoted) == 1 and _looks_like_replacement_request(evidence):
        return quoted[0], direct_target
    return None


def _looks_like_repeated_phrase_group_style_request(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    repeated_phrase_signal = (
        "phrase group" in lowered
        or "first occurrence" in lowered
        or "repeated phrase" in lowered
        or "repeated segment" in lowered
        or "all fast" in lowered
        or ("fast english" in lowered and "basic english" in lowered)
    )
    style_signal = _has_any(
        lowered,
        [
            "clear style",
            "clearer style",
            "basic english",
            "fast english",
            "clear occurrence",
            "speech rate",
            "speaking rate",
            "pace",
            "pacing",
            "style",
        ],
    )
    return repeated_phrase_signal and style_signal


def _repeated_phrase_group_rate_plan(
    *,
    perception_report: dict[str, Any],
    source_text: str | None,
) -> dict[str, Any]:
    occurrences = _sentence_occurrences_from_perception(perception_report, source_text)
    if not occurrences:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        key = str(occurrence.get("key") or "")
        if len(key.split()) > 8:
            continue
        groups.setdefault(key, []).append(occurrence)

    selected_spans: list[dict[str, Any]] = []
    reference_spans: list[dict[str, Any]] = []
    rate_ratios: list[float] = []
    for key, group in groups.items():
        valid_group = [
            item
            for item in sorted(
                group,
                key=lambda value: (float(value["start_sec"]), float(value["end_sec"])),
            )
            if float(item["end_sec"]) > float(item["start_sec"])
        ]
        if len(valid_group) < 2:
            continue
        reference = valid_group[0]
        reference_duration = max(0.05, float(reference["end_sec"]) - float(reference["start_sec"]))
        reference_spans.append(
            {
                "start_sec": round(float(reference["start_sec"]), 6),
                "end_sec": round(float(reference["end_sec"]), 6),
                "text": str(reference.get("text") or key),
                "label": "first_clear_phrase_occurrence",
                "phrase_group": key,
            }
        )
        for occurrence in valid_group[1:]:
            duration = max(0.01, float(occurrence["end_sec"]) - float(occurrence["start_sec"]))
            selected_spans.append(
                {
                    "start_sec": round(float(occurrence["start_sec"]), 6),
                    "end_sec": round(float(occurrence["end_sec"]), 6),
                    "text": str(occurrence.get("text") or key),
                    "label": "repeated_phrase_occurrence_after_first",
                    "phrase_group": key,
                    "reference_start_sec": round(float(reference["start_sec"]), 6),
                    "reference_end_sec": round(float(reference["end_sec"]), 6),
                }
            )
            if duration < reference_duration * 0.98:
                rate_ratios.append(duration / reference_duration)

    if not selected_spans:
        return {}
    if rate_ratios:
        sorted_ratios = sorted(rate_ratios)
        median_ratio = sorted_ratios[len(sorted_ratios) // 2]
        rate_factor = max(0.55, min(0.95, median_ratio))
    else:
        rate_factor = 0.8
    return {
        "selected_spans": selected_spans,
        "reference_spans": reference_spans,
        "rate_factor": round(rate_factor, 3),
    }


def _sentence_occurrences_from_perception(
    perception_report: dict[str, Any],
    source_text: str | None,
) -> list[dict[str, Any]]:
    words = _timestamp_words_from_perception(perception_report)
    if not words:
        return []
    sentence_texts = _split_source_text_sentences(
        source_text
        or perception_report.get("source_text")
        or (perception_report.get("asr") or {}).get("text")
    )
    if sentence_texts:
        aligned = _align_sentence_texts_to_words(sentence_texts, words)
        if aligned:
            return aligned
    return _gap_segment_occurrences_from_words(words)


def _timestamp_words_from_perception(
    perception_report: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_words = perception_report.get("word_timestamps")
    if not isinstance(raw_words, list):
        raw_words = []
    words: list[dict[str, Any]] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        token = str(item.get("text") or item.get("word") or item.get("char") or "").strip()
        normalized = _normalize_phrase_key(token)
        if not normalized:
            continue
        start = _safe_float(item.get("start_sec", item.get("start")))
        end = _safe_float(item.get("end_sec", item.get("end")))
        if start is None or end is None:
            continue
        if end < start:
            continue
        words.append(
            {
                "text": token,
                "key": normalized,
                "start_sec": float(start),
                "end_sec": float(end),
            }
        )
    return words


def _split_source_text_sentences(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.findall(r"[^.!?。！？]+(?:[.!?。！？]+|$)", text)
    return [
        part.strip(" \t\r\n,，;；:：.!?。！？")
        for part in parts
        if part.strip(" \t\r\n,，;；:：.!?。！？")
    ]


def _align_sentence_texts_to_words(
    sentence_texts: list[str], words: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    cursor = 0
    for sentence in sentence_texts:
        sentence_words = _phrase_words(sentence)
        if not sentence_words:
            continue
        start_index = _find_word_sequence(words, sentence_words, start=cursor)
        if start_index is None and cursor < len(words):
            start_index = _find_word_sequence(words, sentence_words, start=0)
        if start_index is None:
            continue
        end_index = start_index + len(sentence_words) - 1
        if end_index >= len(words):
            continue
        start_sec = float(words[start_index]["start_sec"])
        end_sec = float(words[end_index]["end_sec"])
        if end_sec <= start_sec:
            end_sec = start_sec + 0.08
        occurrences.append(
            {
                "text": sentence,
                "key": " ".join(sentence_words),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
        cursor = end_index + 1
    return occurrences


def _gap_segment_occurrences_from_words(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not words:
        return []
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if current:
            gap = float(word["start_sec"]) - float(current[-1]["end_sec"])
            if gap > 0.75:
                segments.append(current)
                current = []
        current.append(word)
    if current:
        segments.append(current)
    occurrences: list[dict[str, Any]] = []
    for segment in segments:
        key = " ".join(str(item["key"]) for item in segment if item.get("key"))
        if not key:
            continue
        occurrences.append(
            {
                "text": " ".join(str(item["text"]) for item in segment if item.get("text")),
                "key": key,
                "start_sec": float(segment[0]["start_sec"]),
                "end_sec": max(
                    float(segment[-1]["end_sec"]), float(segment[0]["start_sec"]) + 0.08
                ),
            }
        )
    return occurrences


def _find_word_sequence(
    words: list[dict[str, Any]], sequence: list[str], *, start: int
) -> int | None:
    if not sequence or len(sequence) > len(words):
        return None
    max_start = len(words) - len(sequence)
    for index in range(max(0, start), max_start + 1):
        if all(
            words[index + offset].get("key") == expected for offset, expected in enumerate(sequence)
        ):
            return index
    return None


def _find_fuzzy_word_sequence(
    words: list[dict[str, Any]],
    sequence: list[str],
    *,
    start: int,
) -> tuple[int, int] | None:
    if not sequence or not words:
        return None
    target = " ".join(sequence)
    best: tuple[float, int, int] | None = None
    min_len = max(1, len(sequence) - 1)
    max_len = min(len(words), len(sequence) + 1)
    for index in range(max(0, start), len(words)):
        for length in range(min_len, max_len + 1):
            if index + length > len(words):
                continue
            window = [str(item.get("key") or "") for item in words[index : index + length]]
            if not any(window):
                continue
            candidate = " ".join(window)
            score = difflib.SequenceMatcher(a=target, b=candidate, autojunk=False).ratio()
            if len(sequence) == length:
                token_scores = [
                    difflib.SequenceMatcher(a=expected, b=actual, autojunk=False).ratio()
                    for expected, actual in zip(sequence, window)
                ]
                if token_scores:
                    score = max(score, sum(token_scores) / len(token_scores))
            if score < 0.82:
                continue
            ranking = (score, -abs(length - len(sequence)), -index)
            if best is None or ranking > (
                best[0],
                -abs(best[2] - len(sequence)),
                -best[1],
            ):
                best = (score, index, length)
    if best is None:
        return None
    return best[1], best[2]


def _fuzzy_phrase_char_span(source_text: str, target_phrase: str) -> tuple[int, int] | None:
    source_tokens = list(re.finditer(r"[\w']+", source_text, flags=re.UNICODE))
    target_tokens = _phrase_words(target_phrase)
    if not source_tokens or not target_tokens:
        return None
    normalized_source = [
        {
            "key": _normalize_phrase_key(match.group(0)),
            "start": match.start(),
            "end": match.end(),
        }
        for match in source_tokens
    ]
    fuzzy = _find_fuzzy_word_sequence(normalized_source, target_tokens, start=0)
    if fuzzy is None:
        return None
    index, length = fuzzy
    start = int(normalized_source[index]["start"])
    end = int(normalized_source[index + length - 1]["end"])
    return (start, end) if end > start else None


def _phrase_words(text: str) -> list[str]:
    return [
        word
        for word in (
            _normalize_phrase_key(match.group(0)) for match in re.finditer(r"[\w']+", text)
        )
        if word
    ]


def _normalize_phrase_key(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^\w']+", "", lowered)
    return lowered.strip("'_")


def _requests_emphasis_stress_edit(text: str) -> bool:
    lowered = str(text or "").casefold()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in [
            "emphasis",
            "emphasize",
            "emphasise",
            "stress",
            "add stress",
            "stress to",
            "stress the",
            "word stress",
            "speech stress",
            "stressed",
            "more prominent",
            "pronunciation of",
            "重音",
            "重读",
            "强调",
            "着重",
        ]
    )


def _emphasis_target_phrase(
    params: dict[str, Any],
    *,
    instruction: str,
    source_text: Any = None,
) -> str | None:
    for key in ("target_phrase", "target_word", "phrase", "word", "target_text"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().strip("\"'“”‘’")
    tagged = _emphasis_phrase_from_tagged_text(
        params.get("edit_tagged_text")
    ) or _emphasis_phrase_from_tagged_text(params.get("instruction_st"))
    if tagged:
        return tagged
    quoted = _quoted_phrases(str(instruction or ""))
    if quoted:
        return quoted[0]
    if _requests_last_word(str(instruction or "")) and isinstance(source_text, str):
        words = re.findall(r"[\w']+", source_text)
        if words:
            return words[-1]
    return None


def _emphasis_phrase_from_tagged_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for tag in ("target", "pitch", "rate"):
        pattern = (
            r"<target>(?P<body>[^<>]+)</target>"
            if tag == "target"
            else rf"<{tag}\b[^>]*>(?P<body>[^<>]+)</{tag}>"
        )
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            body = unescape(match.group("body")).strip()
            if body:
                return body
    return None


def _quoted_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for pattern in [
        r'"([^"\n]+)"',
        r"'([^'\n]+)'",
        r"“([^”\n]+)”",
        r"‘([^’\n]+)’",
    ]:
        for match in re.finditer(pattern, text):
            phrase = match.group(1).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _requests_all_target_occurrences(instruction: str, params: dict[str, Any]) -> bool:
    explicit = params.get("all_occurrences")
    if isinstance(explicit, bool):
        return explicit
    lowered = str(instruction or "").casefold()
    return any(
        token in lowered
        for token in [
            "all instances",
            "all occurrences",
            "all the words",
            "all words",
            "every instance",
            "every occurrence",
            "every matching word",
            "each matching word",
            "each instance",
            "each occurrence",
            "所有",
            "全部",
            "每个",
            "每一次",
        ]
    )


def _requests_last_word(instruction: str) -> bool:
    lowered = str(instruction or "").casefold()
    return any(
        token in lowered
        for token in [
            "last word",
            "final word",
            "ending word",
            "the word at the end",
            "最后一个词",
            "最后一个字",
            "句尾",
            "末尾",
        ]
    )


def _target_tagged_text_for_emphasis(
    source_text: str,
    target_phrase: str,
    *,
    all_occurrences: bool,
) -> str:
    if not target_phrase:
        return source_text
    pattern = re.compile(re.escape(target_phrase), flags=re.IGNORECASE)
    count = 0 if all_occurrences else 1

    def repl(match: re.Match[str]) -> str:
        return f"<target>{match.group(0)}</target>"

    tagged, replacements = pattern.subn(repl, source_text, count=count)
    if replacements:
        return tagged
    compact_target = re.sub(r"\s+", " ", target_phrase).strip()
    if compact_target != target_phrase:
        return _target_tagged_text_for_emphasis(
            source_text,
            compact_target,
            all_occurrences=all_occurrences,
        )
    fuzzy_span = _fuzzy_phrase_char_span(source_text, target_phrase)
    if fuzzy_span is not None:
        start, end = fuzzy_span
        return f"{source_text[:start]}<target>{source_text[start:end]}</target>{source_text[end:]}"
    return source_text


def _emphasis_source_target_spans(
    perception: dict[str, Any],
    target_phrase: str,
    *,
    all_occurrences: bool,
) -> list[dict[str, Any]]:
    words = _timestamp_words_from_perception(perception)
    sequence = _phrase_words(target_phrase)
    if not words or not sequence:
        return []
    spans: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(words):
        index = _find_word_sequence(words, sequence, start=cursor)
        length = len(sequence)
        if index is None:
            fuzzy = _find_fuzzy_word_sequence(words, sequence, start=cursor)
            if fuzzy is None:
                break
            index, length = fuzzy
        matched = words[index : index + length]
        spans.append(
            {
                "start_sec": round(float(matched[0]["start_sec"]), 6),
                "end_sec": round(float(matched[-1]["end_sec"]), 6),
                "text": " ".join(str(item.get("text") or "") for item in matched).strip(),
                "label": "emphasis_stress_target",
                "source": "perception.word_timestamps",
            }
        )
        cursor = index + max(1, length)
        if not all_occurrences:
            break
    return spans


def _audio_assembly_temporal_spans(
    assembly_result: dict[str, Any],
    *,
    source_duration_sec: float | None,
    output_duration_sec: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = assembly_result.get("target_spans")
    if (
        not isinstance(rows, list)
        or not rows
        or source_duration_sec is None
        or output_duration_sec is None
    ):
        return [], []

    aligned: list[tuple[float, float, float, float, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            return [], []
        source = row.get("source_answer")
        target = row.get("target_answer")
        if not isinstance(source, dict) or not isinstance(target, dict):
            return [], []
        try:
            source_start = float(source.get("start"))
            source_end = float(source.get("end"))
            target_start = float(target.get("start"))
            target_end = float(target.get("end"))
        except (TypeError, ValueError):
            return [], []
        if (
            source_start < 0
            or target_start < 0
            or source_end <= source_start
            or target_end <= target_start
        ):
            return [], []
        aligned.append(
            (
                source_start,
                source_end,
                target_start,
                target_end,
                str(row.get("text") or ""),
            )
        )

    aligned.sort(key=lambda item: (item[0], item[1]))
    if any(right[0] < left[1] for left, right in pairwise(aligned)):
        return [], []

    source_spans: list[dict[str, Any]] = []
    output_spans: list[dict[str, Any]] = []
    cumulative_delta = 0.0
    for source_start, source_end, target_start, target_end, text in aligned:
        output_start = source_start + cumulative_delta
        output_end = output_start + (target_end - target_start)
        source_spans.append(
            {
                "start_sec": round(source_start, 6),
                "end_sec": round(source_end, 6),
                "text": text,
                "source": "audio_assembly.source_answer",
            }
        )
        output_spans.append(
            {
                "start_sec": round(output_start, 6),
                "end_sec": round(output_end, 6),
                "text": text,
                "source": "audio_assembly.composed_output_timeline",
            }
        )
        cumulative_delta = output_end - source_end

    expected_output_duration = source_duration_sec + cumulative_delta
    if abs(expected_output_duration - output_duration_sec) > 0.05:
        return [], []
    return source_spans, output_spans


def _qwen3_emphasis_prompt(target_phrase: str, instruction: str) -> str:
    return (
        "Speak the full sentence naturally, preserving the exact words, but make the word or phrase "
        f"'{target_phrase}' clearly emphasized with stronger lexical stress, slightly higher energy, "
        "and a natural spoken focus. Do not add or remove words. "
        f"User instruction: {instruction}"
    )


def _default_emphasis_assembly_smoothing() -> dict[str, Any]:
    return {
        "regen_margin_sec": 0.42,
        "smooth_continuation_boundaries": True,
        "f5_steps": 32,
        "f5_cfg_strength": 2.0,
        "f5_sway_sampling_coef": -1.0,
        "f5_seed": 42,
    }


def _pitch_rate_fallback_instruction_st(source_text: str, target_phrase: str) -> str:
    pattern = re.compile(re.escape(target_phrase), flags=re.IGNORECASE)
    tagged, replacements = pattern.subn(
        lambda match: f"<pitch, semitones=8>{match.group(0)}</pitch>",
        source_text,
        count=1,
    )
    if replacements:
        return tagged
    return f"<pitch, semitones=8>{source_text}</pitch>"


def _looks_like_replacement_request(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(re.search(r"\b(replace|change|modify|substitut|relace)\b", lowered)) or any(
        marker in lowered for marker in ["替换", "换成", "改成", "改为"]
    )


def _first_string_value(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _pop_nonempty_string(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.pop(key, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _base_input_audio_path(state: dict[str, Any]) -> str | None:
    artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), dict) else {}
    for candidate in (
        state.get("input_audio"),
        artifacts.get("base") if isinstance(artifacts, dict) else None,
        artifacts.get("input_audio") if isinstance(artifacts, dict) else None,
        "base",
        "original",
        "input_audio",
    ):
        if not candidate:
            continue
        resolved = resolve_audio_reference(candidate, state)
        if isinstance(resolved, str) and Path(resolved).exists():
            return str(resolved)
    return None


def _reference_vc_fallback_audio(step: AudioEditStep, state: dict[str, Any]) -> str | None:
    refs = state.get("reference_audio_paths") or []
    candidates: list[Any] = []
    for key in (
        "reference_audio_path",
        "reference_audio",
        "ref_audio_path",
        "target_voice_reference_audio_path",
    ):
        candidates.append(step.params.get(key))
        candidates.append(step.inputs.get(key))
    candidates.extend(refs if isinstance(refs, list) else [refs])
    audios = collect_input_audios(state)
    if len(audios) >= 2:
        candidates.append(audios[1])
    for candidate in candidates:
        if not candidate:
            continue
        resolved = resolve_audio_reference(candidate, state)
        if isinstance(resolved, str) and Path(resolved).exists():
            return str(resolved)
    return None


def _tse_step_should_use_timeline_vocal_extraction(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    text = " ".join(
        str(value)
        for value in [
            state.get("user_prompt"),
            step.description,
            params.get("instruction"),
            params.get("target_description"),
        ]
        if value
    ).lower()
    if not (
        _has_any(
            text,
            ["extract", "isolate", "keep only", "remove everything else", "separate"],
        )
        and _has_any(
            text,
            [
                "singing voice",
                "singing vocal",
                "singing vocals",
                "singer",
                "vocal stem",
                "vocals",
                "clean vocal",
                "clean vocals",
                "a cappella",
                "acappella",
                "歌声",
                "演唱声",
                "唱歌声",
            ],
        )
    ):
        return False
    if len(collect_input_audios(state)) > 1:
        return False
    if state.get("reference_audio_paths"):
        return False
    return not _tse_has_external_reference_audio(step, state, params)


def _tse_has_external_reference_audio(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    output_dir = Path(str(state.get("output_dir") or "")).expanduser()
    base_audio = _base_input_audio_path(state)
    candidates: list[Any] = []
    for key in (
        "reference_audio_path",
        "reference_audio",
        "ref_audio_path",
        "speaker_audio_path",
    ):
        candidates.append(params.get(key))
        candidates.append(step.inputs.get(key))
    for candidate in candidates:
        if not candidate:
            continue
        resolved = resolve_audio_reference(candidate, state)
        if not isinstance(resolved, str) or not resolved:
            continue
        path = Path(resolved).expanduser()
        if base_audio:
            try:
                if path.resolve() == Path(base_audio).expanduser().resolve():
                    continue
            except OSError:
                pass
        try:
            path.relative_to(output_dir)
            continue
        except (ValueError, OSError):
            return True
    return False


def _tse_prior_speech_extract_passthrough(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    text = " ".join(
        str(value)
        for value in [
            state.get("user_prompt"),
            step.description,
            params.get("instruction"),
            params.get("target_description"),
        ]
        if value
    ).lower()
    if not _tse_text_is_speech_speaker_extract(text):
        return None
    if len(collect_input_audios(state)) > 1 or state.get("reference_audio_paths"):
        return None
    if _tse_has_external_reference_audio(step, state, params):
        return None
    prior = _latest_artifact_extract_output(state)
    if not prior:
        return None
    output_audio_path = prior.get("output_audio_path")
    selected_spans = prior.get("selected_spans")
    if (
        not isinstance(output_audio_path, str)
        or not Path(output_audio_path).exists()
        or not isinstance(selected_spans, list)
    ):
        return None
    return {
        "output_audio_path": output_audio_path,
        "selected_spans": selected_spans,
        "selected_span_count": len(selected_spans),
        "reason": (
            "prior artifact_ops.extract_spans already produced the requested speech/speaker/language-only audio; "
            "skipping redundant TSE reference extraction to avoid long-running same-source separation"
        ),
    }


def _tse_text_is_speech_speaker_extract(text: str) -> bool:
    if not _has_any(
        text,
        [
            "extract",
            "isolate",
            "keep only",
            "remove everything else",
            "removing all other",
            "separated",
        ],
    ):
        return False
    if not _has_any(
        text,
        [
            "speech",
            "speaker",
            "person",
            "man",
            "woman",
            "voice",
            "language",
            "russian",
            "english",
            "chinese",
            "mandarin",
            "spanish",
            "french",
            "japanese",
            "korean",
            "male",
            "female",
            "child",
            "adult",
        ],
    ):
        return False
    return not _has_any(
        text,
        [
            "singing",
            "singer",
            "song",
            "music",
            "musical",
            "instrument",
            "instrumental",
            "accompaniment",
            "vocal stem",
            "vocals",
            "a cappella",
            "acappella",
        ],
    )


def _span_pair_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        start = _safe_float(value.get("start_sec", value.get("start")))
        end = _safe_float(value.get("end_sec", value.get("end")))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start = _safe_float(value[0])
        end = _safe_float(value[1])
    else:
        return None
    if start is None or end is None or end <= start:
        return None
    return float(start), float(end)


def _latest_artifact_extract_output(state: dict[str, Any]) -> dict[str, Any] | None:
    step_outputs = state.get("step_outputs")
    if not isinstance(step_outputs, dict):
        return None
    for output in reversed(list(step_outputs.values())):
        if not isinstance(output, dict):
            continue
        operation = str(output.get("operation") or "").lower()
        if operation not in {"extract_spans", "artifact_ops.extract_spans"}:
            continue
        if output.get("output_audio_path") and output.get("selected_spans"):
            return output
    return None


def _selected_spans_from_step_outputs(state: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    step_outputs = state.get("step_outputs")
    if not isinstance(step_outputs, dict):
        return spans
    for output in step_outputs.values():
        if not isinstance(output, dict):
            continue
        value = output.get("selected_spans")
        if isinstance(value, list):
            spans.extend(dict(item) for item in value if isinstance(item, dict))
    return spans


def _selector_text(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in (
            "instruction",
            "description",
            "selector",
            "speaker",
            "speaker_id",
            "target",
        ):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _input_audio_is_secondary_source(value: Any, state: dict[str, Any]) -> bool:
    resolved = resolve_audio_reference(value, state)
    if not isinstance(resolved, str) or not resolved.strip():
        return False
    for candidate in collect_input_audios(state)[1:]:
        if isinstance(candidate, str) and _same_audio_reference_path(resolved, candidate):
            return True
    return False


def _materialize_opaque_audio_state(state: dict[str, Any]) -> None:
    # Bindings define a namespace; they do not implicitly select media for a plan.
    bound_audio_refs(state)
    primary_ref = state.get("input_audio")
    if primary_ref is not None:
        state["input_audio"] = resolve_initial_audio_reference(primary_ref, state)
    input_refs = state.get("input_audios")
    if input_refs:
        state["input_audios"] = resolve_audio_reference(
            input_refs,
            state,
            expected_role="source",
            allow_runtime_paths=False,
        )
    raw_reference_refs = state.get("reference_audio_paths")
    if raw_reference_refs:
        state["reference_audio_paths"] = resolve_audio_reference(
            raw_reference_refs,
            state,
            expected_role="reference",
            allow_runtime_paths=False,
        )
    source_text_audio = state.get("source_text_audio_path")
    if source_text_audio is not None:
        state["source_text_audio_path"] = resolve_audio_reference(
            source_text_audio,
            state,
            expected_role="source",
            allow_runtime_paths=False,
        )
    state["current_audio_path"] = state.get("input_audio")
    register_runtime_audio_paths(
        state,
        {
            "input_audio": state["input_audio"],
            "input_audios": state["input_audios"],
            "reference_audio_paths": state["reference_audio_paths"],
            "source_text_audio_path": state.get("source_text_audio_path"),
        },
    )


def _public_runtime_audio_ref(value: Any, state: dict[str, Any]) -> Any:
    if not opaque_audio_resolution_enabled(state):
        return value
    return canonical_audio_ref_for_runtime_path(value, state) or (
        "runtime_output" if value else value
    )


def _resolve_step_audio_aliases(step: AudioEditStep, state: dict[str, Any]) -> None:
    if step.operation == "artifact_ops.reconstruct":
        step.inputs = _resolve_reconstruct_mapping_audio_aliases(step.inputs, state)
        step.params = _resolve_reconstruct_mapping_audio_aliases(step.params, state)
        return
    step.inputs = resolve_audio_references_in_mapping(step.inputs, state)
    step.params = resolve_audio_references_in_mapping(step.params, state)


def _record_operation_contract_failure(
    step: AudioEditStep, state: dict[str, Any], reason: str
) -> None:
    contract = edit_data_pipeline_label_contract(step.operation)
    if not contract and not step.operation.startswith("edit_data_pipeline."):
        return
    normalized_operation = step.operation.removeprefix("edit_data_pipeline.").replace("-", "_")
    record = _without_empty(
        {
            "step_id": step.id,
            "operation": step.operation,
            "reason": reason,
            "stage": "operation_normalization",
            "required_field": contract.get("field") if contract else None,
            "allowed_tags": contract.get("allowed_tags") if contract else None,
            "supported_labels": contract.get("supported_labels") if contract else None,
            "tool_requirements": edit_data_pipeline_label_requirements(step.operation),
            "source_text_present": bool(step.params.get("source_text") or state.get("source_text")),
            "target_text_present": bool(step.params.get("target_text")),
            "retry_guidance": _operation_contract_retry_guidance(normalized_operation),
        }
    )
    state.setdefault("operation_contract_failures", {})[step.id] = record
    if normalized_operation == "text_edit" or "instruction_st" in reason:
        state.setdefault("text_edit_contract_failures", {})[step.id] = record


def _operation_contract_retry_guidance(normalized_operation: str) -> str:
    if normalized_operation == "text_edit":
        return (
            "Replan edit_data_pipeline.text_edit with valid instruction_st: use "
            '<sub targ="new phrase">old phrase</sub> for replacement, <ins>...</ins> '
            "for pure insertion, or <del>...</del> for pure deletion. Provide source_text "
            "and target_text when available; do not switch lexical edits to qwen3_tts."
        )
    return "Replan with the operation-specific edit_data_pipeline label contract before invoking the tool."


_STEP_RESULT_REF_RE = re.compile(
    r"(?P<ref>(?P<step_id>[A-Za-z_][A-Za-z0-9_:\-]*)"
    r"(?:\.result)?\."
    r"(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)*))"
)


def _resolve_step_result_references_in_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    """Resolve planner symbolic refs such as step_1.result.events[0].start_sec."""

    if not isinstance(state.get("step_outputs"), dict) or not state["step_outputs"]:
        return
    records: list[dict[str, Any]] = []
    step.inputs = _resolve_step_result_references(step.inputs, state, records)
    step.params = _resolve_step_result_references(step.params, state, records)
    if records:
        state.setdefault("step_result_reference_resolutions", {})[step.id] = records


def _resolve_step_result_references(
    value: Any, state: dict[str, Any], records: list[dict[str, Any]]
) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_step_result_references(item, state, records)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_step_result_references(item, state, records) for item in value]
    if not isinstance(value, str) or (".result." not in value and not _looks_like_step_ref(value)):
        return value

    matches = list(_STEP_RESULT_REF_RE.finditer(value))
    if not matches:
        return value
    if len(matches) == 1 and matches[0].group("ref") == value:
        resolved, record = _resolve_single_step_result_reference(matches[0], state)
        records.append(record)
        return resolved if record.get("status") == "resolved" else value

    replaced = value
    for match in matches:
        resolved, record = _resolve_single_step_result_reference(match, state)
        records.append(record)
        if record.get("status") == "resolved":
            replaced = replaced.replace(match.group("ref"), str(resolved))
    return replaced


def _looks_like_step_ref(value: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_])step[_:\-]?[A-Za-z0-9_:\-]*\.", value))


def _resolve_single_step_result_reference(
    match: re.Match[str], state: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    ref = match.group("ref")
    step_id = match.group("step_id")
    path = match.group("path")
    step_outputs = state.get("step_outputs") if isinstance(state.get("step_outputs"), dict) else {}
    output = step_outputs.get(step_id)
    if not isinstance(output, dict):
        return None, {
            "ref": ref,
            "step_id": step_id,
            "path": path,
            "status": "unresolved",
            "reason": "unknown_step_id",
        }
    resolved = _resolve_result_path(output, path)
    if resolved is _UNRESOLVED_STEP_RESULT_REF:
        return None, {
            "ref": ref,
            "step_id": step_id,
            "path": path,
            "status": "unresolved",
            "reason": "missing_path",
        }
    return resolved, {
        "ref": ref,
        "step_id": step_id,
        "path": path,
        "status": "resolved",
        "value_preview": _step_result_ref_preview(resolved),
    }


_UNRESOLVED_STEP_RESULT_REF = object()


def _resolve_result_path(output: dict[str, Any], path: str) -> Any:
    current: Any = output
    for raw_segment in path.split("."):
        name, index = _parse_result_path_segment(raw_segment)
        current = _resolve_result_path_name(current, name, output)
        if current is _UNRESOLVED_STEP_RESULT_REF:
            return current
        if index is not None:
            if not isinstance(current, list) or index >= len(current):
                return _UNRESOLVED_STEP_RESULT_REF
            current = current[index]
    return current


def _parse_result_path_segment(segment: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", segment)
    if not match:
        return segment, None
    return match.group(1), int(match.group(2)) if match.group(2) is not None else None


def _resolve_result_path_name(current: Any, name: str, root_output: dict[str, Any]) -> Any:
    if isinstance(current, dict) and name in current:
        return current[name]
    if isinstance(current, dict) and name in {
        "events",
        "event_segments",
        "audio_caption_segments",
        "selected_spans",
    }:
        return _event_like_segments_from_mapping(current) or _event_like_segments_from_mapping(
            root_output
        )
    if isinstance(current, dict) and isinstance(current.get("perception_report"), dict):
        perception = current["perception_report"]
        if name in perception:
            return perception[name]
        if name in {
            "events",
            "event_segments",
            "audio_caption_segments",
            "selected_spans",
        }:
            return _event_like_segments_from_mapping(
                perception
            ) or _event_like_segments_from_mapping(root_output)
    return _UNRESOLVED_STEP_RESULT_REF


def _event_like_segments_from_mapping(value: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("selected_spans", "event_segments", "audio_caption_segments"):
        segments = value.get(key)
        if isinstance(segments, list) and segments:
            return [dict(item) for item in segments if isinstance(item, dict)]
    flexsed_events = value.get("flexsed_events")
    if isinstance(flexsed_events, dict):
        spans = _flexsed_selected_spans(flexsed_events.get("timestamps") or {})
        if spans:
            return spans
    timestamps = value.get("timestamps")
    if isinstance(timestamps, dict):
        spans = _flexsed_selected_spans(timestamps)
        if spans:
            return spans
    for event_report in value.get("audio_events") or []:
        if isinstance(event_report, dict):
            spans = _flexsed_selected_spans(event_report.get("timestamps") or {})
            if spans:
                return spans
    return []


def _step_result_ref_preview(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value.keys())[:8]}
    return str(type(value).__name__)


def _resolve_reconstruct_mapping_audio_aliases(
    mapping: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    values = dict(mapping)
    edits = values.pop("edits", None)
    replacement_audio = {
        key: values.pop(key)
        for key in (
            "edited_spans_audio_paths",
            "edited_spans_audio",
            "edited_audio_path",
            "edited_audio",
            "replacement_audio_path",
            "replacement_audio",
        )
        if key in values
    }
    resolved = resolve_audio_references_in_mapping(values, state)
    allow_runtime_paths = (
        not opaque_audio_resolution_enabled(state)
        or bool(state.get("_runtime_audio_boundary_materialized"))
    )
    for key, value in replacement_audio.items():
        resolved[key] = resolve_audio_reference(
            value,
            state,
            expected_role="reference",
            allow_runtime_paths=allow_runtime_paths,
        )
    if edits is not None:
        resolved["edits"] = _resolve_reconstruct_edits_audio_aliases(edits, state)
    return resolved


def _resolve_reconstruct_edits_audio_aliases(edits: Any, state: dict[str, Any]) -> Any:
    if not isinstance(edits, list):
        return resolve_audio_reference(
            edits,
            state,
            allow_runtime_paths=(
                not opaque_audio_resolution_enabled(state)
                or bool(state.get("_runtime_audio_boundary_materialized"))
            ),
        )
    return [_resolve_reconstruct_edit_audio_aliases(item, state) for item in edits]


def _resolve_reconstruct_edit_audio_aliases(item: Any, state: dict[str, Any]) -> Any:
    if not isinstance(item, dict):
        return resolve_audio_reference(item, state)
    step_reference_keys = {
        "input_from_step",
        "replacement_from_step",
        "edited_audio_from_step",
        "audio_from_step",
        "replace_span_from_step",
        "span_from_step",
        "spans_from_step",
        "spans_from_artifact_id",
        "source_span_from_step",
        "source_spans_from_step",
        "source_spans_from_artifact_id",
        "selected_span_from_step",
        "selected_spans_from_step",
        "selected_spans_from_artifact_id",
        "replace_span_step",
    }
    source_audio_path_keys = {
        "input_audio",
        "input_audio_path",
        "audio",
        "audio_path",
        "path",
    }
    replacement_audio_path_keys = {
        "replacement_audio",
        "replacement_audio_path",
        "edited_audio",
        "edited_audio_path",
    }
    resolved: dict[str, Any] = {}
    for key, value in item.items():
        if key in step_reference_keys:
            resolved[key] = value
        elif key in source_audio_path_keys | replacement_audio_path_keys:
            resolved[key] = resolve_audio_reference(
                value,
                state,
                expected_role=(
                    "reference" if key in replacement_audio_path_keys else "source"
                ),
                allow_runtime_paths=(
                    not opaque_audio_resolution_enabled(state)
                    or bool(state.get("_runtime_audio_boundary_materialized"))
                ),
            )
        else:
            resolved[key] = value
    return resolved


def _output_path(
    step: AudioEditStep, state: dict[str, Any], *, prefer_plan_output: bool = False
) -> str:
    candidates: list[Any] = []
    if prefer_plan_output:
        candidates.append(state.get("output_audio"))
    candidates.extend(
        [
            step.params.get("output_path"),
            step.inputs.get("output_path"),
            state.get("output_audio") if _is_last_step(step, state) else None,
        ]
    )
    for candidate in candidates:
        explicit_path = _materialized_output_path(candidate, state)
        if explicit_path is not None:
            return explicit_path
    safe_operation = step.operation.replace(".", "_").replace("/", "_")
    return str(Path(state["output_dir"]) / f"{step.id}_{safe_operation}.wav")


def _materialized_output_path(value: Any, state: dict[str, Any]) -> str | None:
    """Keep concrete destinations while refusing to materialize logical audio refs."""

    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        return None
    token = value.strip()
    try:
        resolved = resolve_audio_reference(token, state)
    except AudioReferenceResolutionError:
        resolved = token
    if isinstance(resolved, str) and resolved != token:
        return None

    path = Path(token).expanduser()
    if path.is_absolute() or path.parent != Path(".") or bool(path.suffix):
        return token
    return None


def _is_last_step(step: AudioEditStep, state: dict[str, Any]) -> bool:
    steps = state["plan"].steps
    return bool(steps) and steps[-1].id == step.id


def _language_from_perception(state: dict[str, Any]) -> str | None:
    perception = state.get("perception_report")
    if not isinstance(perception, dict):
        return None
    language = perception.get("language") or (perception.get("asr") or {}).get("language")
    return str(language) if language else None


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _executor_blocked_on_source_text(call: ExecutorToolCall) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            call.blocked_reason,
            " ".join(str(item) for item in (call.missing_inputs or [])),
            " ".join(str(item) for item in (call.validation_notes or [])),
        ]
        if value
    ).casefold()
    return "source_text" in evidence or "source text" in evidence or "transcript" in evidence


def _nested_edit_operation_name(params: dict[str, Any]) -> str:
    raw_edit = params.get("edit_op")
    if raw_edit is None:
        raw_edit = params.get("edit_operation")
    operation = ""
    if isinstance(raw_edit, dict):
        operation = str(
            raw_edit.get("operation") or raw_edit.get("tool") or raw_edit.get("tool_id") or ""
        ).strip()
    elif isinstance(raw_edit, str):
        operation = raw_edit.strip()
    operation = operation.removeprefix("edit_data_pipeline.").replace("-", "_")
    return operation


def _perception_matches_audio(input_audio: str, state: dict[str, Any]) -> bool:
    perception = state.get("perception_report")
    if not isinstance(perception, dict):
        return False
    candidates = [
        perception.get("audio_path"),
        perception.get("source_text_audio_path"),
        state.get("source_text_audio_path"),
    ]
    try:
        input_path = Path(str(input_audio)).expanduser().resolve()
    except Exception:
        return False
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if input_path == Path(str(candidate)).expanduser().resolve():
                return True
        except Exception:
            continue
    return False


def _alternate_source_text_audio_paths(input_audio: str, state: dict[str, Any]) -> list[str]:
    try:
        input_path = Path(str(input_audio)).expanduser().resolve()
    except Exception:
        input_path = None
    alternates: list[str] = []
    seen: set[str] = set()
    for audio in [
        *collect_input_audios(state),
        *(state.get("reference_audio_paths") or []),
    ]:
        if not audio:
            continue
        try:
            resolved = Path(str(audio)).expanduser().resolve()
        except Exception:
            continue
        if input_path is not None and resolved == input_path:
            continue
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        alternates.append(key)
    return alternates


def _source_text_repair_language_hints(
    step: AudioEditStep, state: dict[str, Any]
) -> list[str | None]:
    values = [
        step.params.get("expected_language"),
        step.params.get("language"),
        state.get("language"),
        (state.get("context") or {}).get("language")
        if isinstance(state.get("context"), dict)
        else None,
        _language_from_perception(state),
    ]
    hints: list[str | None] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        hints.append(text)
    return hints


def _seamless_language_code(value: Any) -> str | None:
    code = canonicalize_language(value)
    if not code:
        return None
    return {"zho": "cmn", "fre": "fra", "ger": "deu"}.get(code, code)


def _seamless_error_allows_text_tts_fallback(exc: Exception) -> bool:
    text = str(exc).lower()
    return "seamless" in text and (
        "not available" in text
        or "service error 503" in text
        or "missing expressivity_predict" in text
        or "missing" in text
        and "model" in text
        or "gated artifacts" in text
    )


def _translate_audio_or_text_for_s2st_fallback(
    *,
    input_audio_path: str,
    source_text: Any,
    target_language: str,
    source_language: str | None,
    request: str | None,
    agentic_mode: str | None = None,
) -> str | None:
    target_name = _language_name_from_code(target_language) or target_language
    source_name = _language_name_from_code(source_language) if source_language else None
    system_prompt = (
        "You are the translation fallback for an audio editing runtime. "
        "Produce a target-language transcript for speech synthesis. "
        "Translate every spoken utterance in chronological order. "
        "Do not include timestamps, speaker labels, explanations, markdown, or quotes."
    )
    payload = {
        "task": "translate_speech_for_voice_clone_tts_fallback",
        "request": request,
        "source_text_hint": source_text if isinstance(source_text, str) else None,
        "source_language": source_name or source_language,
        "target_language": target_name,
        "required_json_schema": {
            "target_text": "complete translated speech transcript only",
            "source_used": "audio | source_text_hint | both",
            "reason": "brief note",
        },
    }
    mode = resolve_agentic_mode(agentic_mode)
    audio_paths = (
        [input_audio_path] if input_audio_path and Path(str(input_audio_path)).exists() else None
    )
    try:
        timeout = float(os.getenv("X_EDITOR_S2ST_FALLBACK_TRANSLATION_TIMEOUT", "90"))
    except ValueError:
        timeout = 90.0
    try:
        if mode == OPEN_GLM_QWEN_OMNI_MODE:
            if not str(source_text or "").strip():
                return None
            response = OpenAICompatibleChatClient(
                api_key=resolve_openai_compatible_api_key(GLM_PROVIDER),
                base_url=resolve_openai_compatible_base_url(GLM_PROVIDER),
                model_name=resolve_openai_compatible_model(GLM_PROVIDER),
                provider_name=GLM_PROVIDER,
                timeout=timeout,
                max_tokens=4096,
            ).call_json(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                response_format={"type": "json_object"},
                repair_schema_name="SpeechTranslationFallback",
            )
        else:
            response = GeminiClient(timeout=timeout).call_json(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                audio_paths=audio_paths,
            )
    except Exception:
        return None
    target = response.get("target_text") if isinstance(response, dict) else None
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _language_name_from_code(code: Any) -> str | None:
    if not code:
        return None
    normalized = str(code).strip().lower().replace("_", "-")
    mapping = {
        "en": "English",
        "eng": "English",
        "english": "English",
        "zh": "Chinese",
        "zho": "Chinese",
        "cmn": "Chinese",
        "mandarin": "Chinese",
        "chinese": "Chinese",
        "ja": "Japanese",
        "jpn": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "kor": "Korean",
        "korean": "Korean",
        "es": "Spanish",
        "spa": "Spanish",
        "spanish": "Spanish",
        "fr": "French",
        "fra": "French",
        "fre": "French",
        "french": "French",
        "de": "German",
        "deu": "German",
        "ger": "German",
        "german": "German",
        "it": "Italian",
        "ita": "Italian",
        "italian": "Italian",
        "pt": "Portuguese",
        "por": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "rus": "Russian",
        "russian": "Russian",
        "th": "Thai",
        "tha": "Thai",
        "thai": "Thai",
    }
    return mapping.get(normalized)


def _target_language_from_request(text: str | None) -> str | None:
    raw_text = text or ""
    lowered = raw_text.lower()
    mapping = [
        ("mandarin", "cmn"),
        ("chinese", "cmn"),
        ("中文", "cmn"),
        ("汉语", "cmn"),
        ("普通话", "cmn"),
        ("english", "eng"),
        ("英语", "eng"),
        ("英文", "eng"),
        ("french", "fra"),
        ("法语", "fra"),
        ("spanish", "spa"),
        ("西班牙语", "spa"),
        ("german", "deu"),
        ("德语", "deu"),
        ("japanese", "jpn"),
        ("日语", "jpn"),
        ("korean", "kor"),
        ("韩语", "kor"),
        ("portuguese", "por"),
        ("葡萄牙语", "por"),
        ("italian", "ita"),
        ("意大利语", "ita"),
        ("russian", "rus"),
        ("俄语", "rus"),
        ("thai", "tha"),
        ("泰语", "tha"),
    ]
    for needle, code in mapping:
        needle_lower = needle.lower()
        if re.search(rf"\b(?:to|into|in|as)\s+(?:the\s+)?{re.escape(needle_lower)}\b", lowered):
            return code
        if any(f"{prefix}{needle}" in raw_text for prefix in ("成", "到", "为", "至")):
            return code
    for needle, code in mapping:
        if needle in lowered or needle in (text or ""):
            return code
    return None


def _extract_output_path(output: dict[str, Any] | None) -> str | None:
    if not isinstance(output, dict):
        return None
    for key in (
        "output_audio_path",
        "target_audio_path",
        "output_path",
        "generated_audio_path",
    ):
        value = output.get(key)
        if isinstance(value, str):
            return value
    return None


def _pop_qwen3_tts_voice_clone_reference(params: dict[str, Any], inputs: dict[str, Any]) -> Any:
    aliases = (
        "ref_audio_path",
        "voice_clone_reference_path",
        "voice_clone_reference_audio_path",
        "voice_clone_reference",
        "speaker_audio_path",
        "speaker_audio",
        "reference_audio_path",
        "reference_audio",
    )
    for key in aliases:
        value = params.pop(key, None)
        if value:
            return value
    for key in aliases:
        value = inputs.get(key)
        if value:
            return value
    return None


def _requests_existing_audio_voice_conversion(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    if (
        params.get("reference_only")
        or params.get("standalone_tts")
        or params.get("synthesize_only")
    ):
        return False
    if not (state.get("current_audio_path") or state.get("input_audio")):
        return False
    text = " ".join(
        str(value)
        for value in [
            state.get("user_prompt"),
            step.description,
            params.get("instruction"),
            params.get("instruct"),
            params.get("voice_prompt"),
        ]
        if value
    ).lower()
    if not text:
        return False
    voice_terms = [
        "音色",
        "声音",
        "童声",
        "男声",
        "女声",
        "voice",
        "timbre",
        "speaker",
        "vocal",
    ]
    target_terms = [
        "小孩",
        "儿童",
        "孩子",
        "男孩",
        "女孩",
        "年轻",
        "年老",
        "老人",
        "男性",
        "女性",
        "低沉",
        "沙哑",
        "明亮",
        "child",
        "kid",
        "boy",
        "girl",
        "male",
        "female",
        "young",
        "elderly",
        "old",
        "deep",
        "raspy",
        "bright",
    ]
    edit_terms = [
        "改",
        "变",
        "转换",
        "转化",
        "成为",
        "变成",
        "convert",
        "transform",
        "change",
        "make",
    ]
    if not any(term in text for term in voice_terms):
        return False
    return any(term in text for term in target_terms) and any(term in text for term in edit_terms)


def _voice_design_reference_backend(params: dict[str, Any], state: dict[str, Any]) -> str:
    value = (
        params.pop("voice_design_backend", None)
        or params.pop("reference_synthesis_backend", None)
        or params.pop("reference_voice_backend", None)
        or _voice_design_backend_from_context(state)
        or os.getenv("QWEN3_TTS_VOICE_DESIGN_BACKEND")
        or os.getenv("X_EDITOR_VOICE_DESIGN_BACKEND")
        or "qwen3_tts"
    )
    text = str(value).strip().lower().replace("-", "_")
    if text not in {"qwen", "qwen3", "qwen3tts", "qwen3_tts", "auto"}:
        raise ValueError(f"Unsupported voice-design backend: {value!r}; install an extension separately")
    return "qwen3_tts"


def _voice_design_backend_from_context(state: dict[str, Any]) -> str | None:
    context = state.get("context") or {}
    preferences = context.get("tool_preferences")
    if isinstance(preferences, dict):
        value = (
            preferences.get("no_reference_voice_conversion")
            or preferences.get("no_reference_voice_design")
            or preferences.get("voice_design_tool")
        )
        if value:
            return str(value)
    for key in (
        "no_reference_voice_conversion_backend",
        "voice_design_backend",
        "voice_design_tool",
    ):
        if context.get(key):
            return str(context[key])
    plan = state.get("plan")
    global_params = getattr(plan, "global_params", None)
    if isinstance(global_params, dict):
        plan_preferences = global_params.get("tool_preferences")
        if isinstance(plan_preferences, dict):
            value = (
                plan_preferences.get("no_reference_voice_conversion")
                or plan_preferences.get("no_reference_voice_design")
                or plan_preferences.get("voice_design_tool")
            )
            if value:
                return str(value)
        for key in (
            "no_reference_voice_conversion_backend",
            "voice_design_backend",
            "voice_design_tool",
        ):
            if global_params.get(key):
                return str(global_params[key])
    return None


def _voice_design_memory_path() -> Path:
    return Path(
        os.getenv("X_EDITOR_VOICE_DESIGN_MEMORY_PATH") or _VOICE_DESIGN_MEMORY_PATH
    ).expanduser()


def _load_voice_design_memory() -> dict[str, Any]:
    path = _voice_design_memory_path()
    if not path.exists():
        return {"_source_path": str(path), "_load_error": "missing"}
    try:
        import yaml
    except Exception as exc:
        return {
            "_source_path": str(path),
            "_load_error": f"yaml_unavailable: {type(exc).__name__}: {exc}",
        }
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {
            "_source_path": str(path),
            "_load_error": f"read_failed: {type(exc).__name__}: {exc}",
        }
    if not isinstance(data, dict):
        return {"_source_path": str(path), "_load_error": "not_a_mapping"}
    memory = dict(data)
    memory["_source_path"] = str(path)
    return memory


def _voice_design_memory_profile(voice_class: str | None) -> dict[str, Any]:
    if not voice_class:
        return {}
    memory = _load_voice_design_memory()
    profiles = memory.get("voice_classes")
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(voice_class)
    return dict(profile) if isinstance(profile, dict) else {}


def _voice_design_memory_profile_value(voice_class: str | None, *keys: str) -> Any:
    value: Any = _voice_design_memory_profile(voice_class)
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _voice_design_memory_postprocess_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for voice_class in ("child", "elderly"):
        profile_postprocess = _voice_design_memory_profile_value(voice_class, "postprocess")
        if isinstance(profile_postprocess, dict):
            defaults.update(profile_postprocess)
    return defaults


def _voice_design_memory_evidence(voice_class: str | None) -> dict[str, Any] | None:
    if not voice_class:
        return None
    memory = _load_voice_design_memory()
    profile = _voice_design_memory_profile(voice_class)
    if not profile:
        return {
            "source_path": memory.get("_source_path"),
            "applied_voice_class": voice_class,
            "loaded": False,
            "reason": memory.get("_load_error") or "profile_not_found",
        }
    return {
        "source_path": memory.get("_source_path"),
        "version": memory.get("version"),
        "loaded": True,
        "applied_voice_class": voice_class,
        "summary": profile.get("summary"),
        "reference_text": profile.get("reference_text"),
        "design_defaults": profile.get("design_defaults"),
        "postprocess": profile.get("postprocess"),
    }


def _independent_voice_reference_text(
    backend: str, *, request: str, voice_class: str | None = None
) -> str:
    # A short carrier sentence keeps the synthetic reference independent of the source audio content.
    profile = voice_class or _voice_class_profile(request, {})
    memory_text = _voice_design_memory_profile_value(profile, "reference_text")
    if isinstance(memory_text, str) and memory_text.strip():
        return memory_text.strip()
    if profile == "child":
        return "小朋友们好，今天我们开心地读一段简短的话。"
    if profile == "elderly":
        return "您好，今天我慢慢读一段简短的话。"
    return "今天我们用清晰自然的声音读一段简短的参考语音。"


def _seed_vc_params_from_no_ref_voice_params(params: dict[str, Any]) -> dict[str, Any]:
    nested = params.pop("seed_vc_params", None)
    seed_params = dict(nested) if isinstance(nested, dict) else {}
    for key in ("diffusion_steps", "length_adjust", "inference_cfg_rate"):
        if key in params:
            seed_params[key] = params.pop(key)
    return seed_params


def _voice_design_params_from_no_ref_voice_params(
    params: dict[str, Any],
) -> dict[str, Any]:
    nested = params.pop("voice_design_params", None)
    design_params = dict(nested) if isinstance(nested, dict) else {}
    for key in (
        "age",
        "gender",
        "pitch",
        "pitch_variation",
        "volume",
        "speaking_rate",
        "emotion",
        "temperature",
        "top_p",
        "max_new_tokens",
        "seed",
        "language",
        "allow_prompt_rewrite",
        "strict_language",
        "speaker",
    ):
        if key in params:
            design_params[key] = params.pop(key)
    return design_params


def _voice_design_prompt_hint_from_params(params: dict[str, Any]) -> str:
    return " ".join(str(value) for value in params.values() if value is not None)


def _voice_class_context(step: AudioEditStep, state: dict[str, Any], params: dict[str, Any]) -> str:
    step_specific = " ".join(
        str(value)
        for value in [
            step.description,
            params.get("instruction"),
            params.get("instruct"),
            params.get("voice_prompt"),
            params.get("age"),
            params.get("gender"),
            params.get("pitch"),
            params.get("speaking_rate"),
        ]
        if value
    )
    if step_specific.strip() and _voice_class_profile(step_specific, params):
        return step_specific
    state_prompt = str(state.get("user_prompt") or "")
    return f"{step_specific} {state_prompt}".strip() if step_specific else state_prompt


def _voice_class_audio_postprocess_options(*dicts: dict[str, Any]) -> dict[str, Any]:
    memory_defaults = _voice_design_memory_postprocess_defaults()
    options: dict[str, Any] = {
        "enabled": _env_bool("X_EDITOR_VOICE_CLASS_AUDIO_POSTPROCESS", default=True),
        "child_pitch_shift_semitones": _env_float(
            "X_EDITOR_CHILD_VOICE_AUDIO_PITCH_SEMITONES",
            float(memory_defaults.get("child_pitch_shift_semitones", 3.5)),
        ),
        "child_highpass_hz": float(memory_defaults.get("child_highpass_hz", 100.0)),
        "child_eq_3300_gain_db": float(memory_defaults.get("child_eq_3300_gain_db", 0.8)),
        "child_eq_5200_gain_db": float(memory_defaults.get("child_eq_5200_gain_db", 0.4)),
        "elderly_pitch_shift_semitones": _env_float(
            "X_EDITOR_ELDERLY_VOICE_AUDIO_PITCH_SEMITONES",
            float(memory_defaults.get("elderly_pitch_shift_semitones", -7.0)),
        ),
        "elderly_preserve_duration": _env_bool(
            "X_EDITOR_ELDERLY_VOICE_PRESERVE_DURATION", default=False
        ),
        "elderly_vibrato_hz": float(memory_defaults.get("elderly_vibrato_hz", 4.8)),
        "elderly_vibrato_depth": float(memory_defaults.get("elderly_vibrato_depth", 0.24)),
        "elderly_tremolo_hz": float(memory_defaults.get("elderly_tremolo_hz", 4.4)),
        "elderly_tremolo_depth": float(memory_defaults.get("elderly_tremolo_depth", 0.22)),
        "elderly_lowpass_hz": float(memory_defaults.get("elderly_lowpass_hz", 3200.0)),
        "elderly_highpass_hz": float(memory_defaults.get("elderly_highpass_hz", 60.0)),
        "elderly_eq_900_gain_db": float(memory_defaults.get("elderly_eq_900_gain_db", -3.0)),
        "resynthesis_enabled": _env_bool(
            "X_EDITOR_VOICE_CLASS_REFERENCE_RESYNTHESIS",
            default=bool(memory_defaults.get("resynthesis_enabled", True)),
        ),
        "resynthesis_backend": os.getenv(
            "X_EDITOR_VOICE_CLASS_REFERENCE_RESYNTHESIS_BACKEND",
            str(memory_defaults.get("resynthesis_backend") or "edit_data_pipeline_speech_rate"),
        ),
        "resynthesis_method": os.getenv(
            "X_EDITOR_VOICE_CLASS_REFERENCE_RESYNTHESIS_METHOD",
            str(memory_defaults.get("resynthesis_method") or "wsola+indextts2"),
        ),
        "resynthesis_required": _env_bool(
            "X_EDITOR_VOICE_CLASS_REFERENCE_RESYNTHESIS_REQUIRED", default=False
        ),
    }
    for values in dicts:
        for key in (
            "voice_class_audio_postprocess",
            "voice_class_postprocess",
            "target_voice_audio_postprocess",
            "target_voice_reference_postprocess",
            "child_reference_postprocess",
            "postprocess_child_reference",
        ):
            if key in values:
                options["enabled"] = _truthy(values.pop(key))
        for key in (
            "child_reference_pitch_shift_semitones",
            "target_voice_reference_pitch_shift_semitones",
            "reference_pitch_shift_semitones",
            "child_voice_pitch_shift_semitones",
        ):
            if key in values:
                options["child_pitch_shift_semitones"] = _safe_float(values.pop(key)) or 3.5
        for key in (
            "elderly_voice_pitch_shift_semitones",
            "elderly_reference_pitch_shift_semitones",
        ):
            if key in values:
                options["elderly_pitch_shift_semitones"] = _safe_float(values.pop(key)) or -7.0
        for key in ("elderly_preserve_duration", "preserve_elderly_reference_duration"):
            if key in values:
                options["elderly_preserve_duration"] = _truthy(values.pop(key))
        for option_key, aliases in {
            "child_highpass_hz": (
                "child_voice_highpass_hz",
                "child_reference_highpass_hz",
            ),
            "child_eq_3300_gain_db": (
                "child_voice_eq_3300_gain_db",
                "child_reference_eq_3300_gain_db",
            ),
            "child_eq_5200_gain_db": (
                "child_voice_eq_5200_gain_db",
                "child_reference_eq_5200_gain_db",
            ),
            "elderly_vibrato_hz": (
                "elderly_voice_vibrato_hz",
                "elderly_reference_vibrato_hz",
            ),
            "elderly_vibrato_depth": (
                "elderly_voice_vibrato_depth",
                "elderly_reference_vibrato_depth",
            ),
            "elderly_tremolo_hz": (
                "elderly_voice_tremolo_hz",
                "elderly_reference_tremolo_hz",
            ),
            "elderly_tremolo_depth": (
                "elderly_voice_tremolo_depth",
                "elderly_reference_tremolo_depth",
            ),
            "elderly_lowpass_hz": (
                "elderly_voice_lowpass_hz",
                "elderly_reference_lowpass_hz",
            ),
            "elderly_highpass_hz": (
                "elderly_voice_highpass_hz",
                "elderly_reference_highpass_hz",
            ),
            "elderly_eq_900_gain_db": (
                "elderly_voice_eq_900_gain_db",
                "elderly_reference_eq_900_gain_db",
            ),
        }.items():
            for key in aliases:
                if key in values:
                    parsed = _safe_float(values.pop(key))
                    if parsed is not None:
                        options[option_key] = parsed
        for key in (
            "voice_class_reference_resynthesis",
            "target_voice_reference_resynthesis",
            "voice_class_audio_resynthesis",
            "resynthesize_voice_class_reference",
            "resynthesize_target_voice_reference",
            "codec_resynthesis",
            "neural_resynthesis",
        ):
            if key in values:
                options["resynthesis_enabled"] = _truthy(values.pop(key))
        for key in (
            "voice_class_reference_resynthesis_backend",
            "target_voice_reference_resynthesis_backend",
            "voice_class_resynthesis_backend",
            "codec_resynthesis_backend",
            "neural_resynthesis_backend",
        ):
            if key in values:
                options["resynthesis_backend"] = str(values.pop(key) or "").strip()
        for key in (
            "voice_class_reference_resynthesis_method",
            "target_voice_reference_resynthesis_method",
            "voice_class_resynthesis_method",
            "codec_resynthesis_method",
            "neural_resynthesis_method",
        ):
            if key in values:
                options["resynthesis_method"] = str(values.pop(key) or "").strip()
        for key in (
            "voice_class_reference_resynthesis_required",
            "target_voice_reference_resynthesis_required",
            "codec_resynthesis_required",
            "neural_resynthesis_required",
        ):
            if key in values:
                options["resynthesis_required"] = _truthy(values.pop(key))
        for key in (
            "voice_class_reference_resynthesis_service_url",
            "target_voice_reference_resynthesis_service_url",
        ):
            if key in values:
                options["resynthesis_service_url"] = str(values.pop(key) or "").strip()
    return options


def _voice_class_reference_resynthesis_backend(options: dict[str, Any]) -> str:
    value = (
        str(options.get("resynthesis_backend") or "edit_data_pipeline_speech_rate")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if value in {"auto", "default", "edit_data_pipeline"}:
        return "edit_data_pipeline_speech_rate"
    if value in {
        "wsola+indextts2",
        "wsola_indextts2",
        "indextts2",
        "speech_rate",
        "edit_data_pipeline_speech_rate",
    }:
        return value.replace("+", "_")
    return value


def _child_voice_reference_postprocess_options(
    *dicts: dict[str, Any],
) -> dict[str, Any]:
    options = _voice_class_audio_postprocess_options(*dicts)
    options["pitch_shift_semitones"] = options.get("child_pitch_shift_semitones", 3.5)
    return options


def _is_child_voice_request(text: Any) -> bool:
    return _voice_class_profile(text, {}) == "child"


def _voice_class_profile(text: Any, params: dict[str, Any] | None = None) -> str | None:
    params = params or {}
    lowered = str(text or "").lower()
    age = str(params.get("age") or "").strip().lower()
    gender = str(params.get("gender") or "").strip().lower()
    if (
        age in {"child", "kid", "children", "小孩", "儿童", "孩童"}
        or _has_any(
            lowered,
            [
                "童声",
                "儿童声",
                "儿童音色",
                "小孩",
                "孩子",
                "孩童",
                "未变声",
                "child voice",
                "children voice",
                "kid voice",
                "prepubescent",
                "little boy",
                "little girl",
                "young child",
            ],
        )
    ) and not _negates_child_voice(lowered):
        return "child"
    if age in {"elderly", "old", "senior", "老年", "老人"} or _has_any(
        lowered,
        [
            "老人",
            "老年",
            "年老",
            "苍老",
            "高龄",
            "老爷爷",
            "老奶奶",
            "elderly",
            "old voice",
            "older voice",
            "old man",
            "old woman",
            "senior voice",
            "aged voice",
        ],
    ):
        return "elderly"
    if age in {"young", "young_adult", "青年", "年轻"} or _has_any(
        lowered,
        ["年轻", "青年", "young voice", "younger voice", "young adult"],
    ):
        return "young_adult"
    if age in {"middle", "middle_aged", "中年"} or _has_any(
        lowered, ["中年", "middle-aged", "middle aged"]
    ):
        return "middle_aged"
    target_profile = _target_voice_class_from_conversion_request(lowered)
    if target_profile:
        return target_profile
    if gender in {"female", "woman", "女性", "女"} or _has_any(
        lowered,
        ["女声", "女性", "女人", "female voice", "female speaker", "woman voice"],
    ):
        return "adult_female"
    if gender in {"male", "man", "男性", "男"} or _has_any(
        lowered,
        ["男声", "男性", "男人", "male voice", "male speaker", "man voice"],
    ):
        return "adult_male"
    return None


def _target_voice_class_from_conversion_request(text: str) -> str | None:
    target = _target_voice_phrase(text)
    if not target:
        return None
    if _voice_phrase_has_any(
        target,
        [
            "童声",
            "儿童声",
            "儿童音色",
            "小孩",
            "孩子",
            "孩童",
            "child voice",
            "kid voice",
            "little boy",
            "little girl",
        ],
    ) and not _negates_child_voice(target):
        return "child"
    if _voice_phrase_has_any(
        target,
        [
            "老人",
            "老年",
            "苍老",
            "高龄",
            "elderly",
            "old voice",
            "old man",
            "old woman",
        ],
    ):
        return "elderly"
    if _voice_phrase_has_any(
        target, ["年轻", "青年", "young voice", "younger voice", "young adult"]
    ):
        return "young_adult"
    if _voice_phrase_has_any(target, ["中年", "middle-aged", "middle aged"]):
        return "middle_aged"
    if _voice_phrase_has_any(
        target,
        [
            "女声",
            "女性",
            "女人",
            "female voice",
            "female speaker",
            "female voices",
            "woman voice",
        ],
    ):
        return "adult_female"
    if _voice_phrase_has_any(
        target,
        [
            "男声",
            "男性",
            "男人",
            "male voice",
            "male speaker",
            "male voices",
            "man voice",
        ],
    ):
        return "adult_male"
    return None


def _target_voice_phrase(text: str) -> str | None:
    markers = [
        r"\bto\b",
        r"\binto\b",
        r"\bas\b",
        r"\bbecome\b",
        r"\bsound like\b",
        "改成",
        "变成",
        "换成",
        "转换为",
        "转换成",
        "转成",
        "变为",
        "成为",
    ]
    best: tuple[int, str] | None = None
    for marker in markers:
        if marker.startswith("\\"):
            for match in re.finditer(marker, text):
                candidate = text[match.end() :]
                if _voice_phrase_has_voice_class(candidate):
                    best = (match.start(), candidate)
        else:
            start = 0
            while True:
                index = text.find(marker, start)
                if index < 0:
                    break
                candidate = text[index + len(marker) :]
                if _voice_phrase_has_voice_class(candidate):
                    best = (index, candidate)
                start = index + len(marker)
    if best is None:
        return None
    phrase = re.split(r"[.;。；,，\n]", best[1], maxsplit=1)[0]
    return phrase.strip() or None


def _voice_phrase_has_voice_class(text: str) -> bool:
    return _voice_phrase_has_any(
        text,
        [
            "童声",
            "儿童",
            "小孩",
            "老人",
            "老年",
            "年轻",
            "青年",
            "中年",
            "女声",
            "女性",
            "男声",
            "男性",
            "child voice",
            "kid voice",
            "elderly",
            "old voice",
            "young voice",
            "middle-aged",
            "female voice",
            "female speaker",
            "female voices",
            "male voice",
            "male speaker",
            "male voices",
        ],
    )


def _voice_phrase_has_any(text: str, terms: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(term.lower() in lowered for term in terms)


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _is_timeline_reorder_operation_name(value: Any) -> bool:
    if isinstance(value, dict):
        value = (
            value.get("operation")
            or value.get("op")
            or value.get("edit_op")
            or value.get("edit_operation")
        )
    normalized = str(value or "").strip().lower().removeprefix("artifact_ops.").replace("-", "_")
    return normalized in {
        "reorder",
        "reorder_spans",
        "reorder_segments",
        "swap_order",
        "swap_segments",
    }


def _negates_child_voice(text: str) -> bool:
    return any(
        pattern in text
        for pattern in [
            "不要童声",
            "不要变成童声",
            "不要童声化",
            "不是童声",
            "不是儿童",
            "不是小孩",
            "非童声",
            "非儿童",
            "not a child",
            "not child",
            "not childlike",
            "not child-like",
            "avoid child",
        ]
    )


def _voice_design_params_request_child(params: dict[str, Any]) -> bool:
    age = str(params.get("age") or "").strip().lower()
    return age in {"child", "kid", "children", "小孩", "儿童", "孩童"}


def _apply_target_voice_reference_defaults(
    prompt_context: str,
    backend: str,
    params: dict[str, Any],
    *,
    voice_class: str | None = None,
) -> None:
    profile = voice_class or _voice_class_profile(prompt_context, params)
    if not profile:
        return
    defaults = _voice_class_default_params(profile)
    for key, value in defaults.items():
        params.setdefault(key, value)
    if any(term in prompt_context.lower() for term in ["男孩", "little boy", "boy voice"]):
        params.setdefault("gender", "male")
    elif any(term in prompt_context.lower() for term in ["女孩", "little girl", "girl voice"]):
        params.setdefault("gender", "female")
    elif any(term in prompt_context.lower() for term in ["老爷爷", "grandpa", "grandfather"]):
        params.setdefault("gender", "male")
    elif any(term in prompt_context.lower() for term in ["老奶奶", "grandma", "grandmother"]):
        params.setdefault("gender", "female")


def _voice_class_default_params(voice_class: str) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {
        "child": {
            "age": "child",
            "pitch": "very_high",
            "pitch_variation": "very_strong",
            "speaking_rate": "medium",
            "volume": "medium",
            "max_new_tokens": 1024,
        },
        "elderly": {
            "age": "elderly",
            "pitch": "low",
            "pitch_variation": "weak",
            "speaking_rate": "slow",
            "volume": "quiet",
            "max_new_tokens": 1024,
        },
        "young_adult": {
            "age": "young",
            "pitch": "high",
            "pitch_variation": "strong",
            "speaking_rate": "medium",
            "volume": "medium",
        },
        "middle_aged": {
            "age": "middle_aged",
            "pitch": "medium",
            "pitch_variation": "medium",
            "speaking_rate": "medium",
        },
        "adult_female": {
            "gender": "female",
            "pitch": "high",
            "pitch_variation": "medium",
        },
        "adult_male": {
            "gender": "male",
            "pitch": "low",
            "pitch_variation": "medium",
        },
    }
    values = dict(defaults.get(voice_class) or {})
    memory_defaults = _voice_design_memory_profile_value(voice_class, "design_defaults")
    if isinstance(memory_defaults, dict):
        values.update({str(key): value for key, value in memory_defaults.items()})
    return values


def _strengthen_voice_class_prompt(
    prompt: str,
    *,
    backend: str,
    voice_class: str | None,
    gender_preservation: dict[str, Any] | None = None,
) -> str:
    if not voice_class:
        return str(prompt or "").strip()
    required = _voice_class_prompt_constraint(voice_class, backend=backend)
    gender_required = _source_gender_preservation_prompt_constraint(
        gender_preservation, backend=backend
    )
    if gender_required:
        required = f"{required} {gender_required}".strip()
    return _append_prompt_constraint(prompt, required) if required else str(prompt or "").strip()


def _strengthen_child_voice_prompt(prompt: str, *, backend: str) -> str:
    return _strengthen_voice_class_prompt(prompt, backend=backend, voice_class="child")


def _source_gender_preservation_for_voice_design(
    state: dict[str, Any],
    *,
    prompt_context: str,
    params: dict[str, Any],
    voice_class: str | None,
) -> dict[str, Any] | None:
    if voice_class not in {"child", "elderly", "young_adult", "middle_aged"}:
        return None
    explicit_gender = _explicit_target_gender_from_voice_request(prompt_context, params)
    if explicit_gender:
        params.setdefault("gender", explicit_gender)
        return {
            "mode": "explicit_target_gender",
            "target_gender": explicit_gender,
            "preserve_source_gender": False,
            "reason": "user_or_step_explicitly_specified_target_gender",
        }
    inferred = _infer_source_gender_for_voice_design(state)
    if inferred.get("gender"):
        params.setdefault("gender", inferred["gender"])
        return {
            "mode": "preserve_source_gender",
            "target_gender": inferred["gender"],
            "preserve_source_gender": True,
            "source": inferred.get("source"),
            "votes": inferred.get("votes"),
        }
    return {
        "mode": "source_gender_unknown",
        "preserve_source_gender": False,
        "reason": inferred.get("reason") or "source_gender_not_available",
    }


def _source_gender_preservation_prompt_constraint(
    preservation: dict[str, Any] | None,
    *,
    backend: str,
) -> str:
    if not isinstance(preservation, dict) or preservation.get("mode") != "preserve_source_gender":
        return ""
    gender = _normalize_binary_gender(preservation.get("target_gender"))
    if not gender:
        return ""
    return (
        f"Preserve the source speaker's perceived gender as {gender}; change only the requested age/timbre class "
        "and do not switch to an opposite-gender voice."
    )


def _explicit_target_gender_from_voice_request(
    prompt_context: str, params: dict[str, Any]
) -> str | None:
    explicit_param = _normalize_binary_gender(params.get("gender"))
    if explicit_param:
        return explicit_param
    text = str(prompt_context or "").lower()
    female_terms = [
        "女孩",
        "女童",
        "小女孩",
        "女孩子",
        "老奶奶",
        "奶奶",
        "女性",
        "女声",
        "女人",
        "female",
        "woman",
        "girl",
        "grandma",
        "grandmother",
    ]
    male_terms = [
        "男孩",
        "男童",
        "小男孩",
        "男孩子",
        "老爷爷",
        "爷爷",
        "男性",
        "男声",
        "男人",
        "male",
        "man",
        "boy",
        "grandpa",
        "grandfather",
    ]
    female = _has_any(text, female_terms)
    male = _has_any(text, male_terms)
    if female and not male:
        return "female"
    if male and not female:
        return "male"
    return None


def _infer_source_gender_for_voice_design(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    for key in ("source_gender", "perceived_gender", "speaker_gender"):
        gender = _normalize_binary_gender(context.get(key))
        if gender:
            return {
                "gender": gender,
                "source": f"context.{key}",
                "votes": {gender: 1.0},
            }

    perception = state.get("perception_report") or state.get("audio_perception") or {}
    if not isinstance(perception, dict):
        return {"gender": None, "reason": "perception_report_not_dict"}
    for key in ("source_gender", "perceived_gender", "gender"):
        gender = _normalize_binary_gender(perception.get(key))
        if gender:
            return {
                "gender": gender,
                "source": f"perception_report.{key}",
                "votes": {gender: 1.0},
            }

    speaker_profiles: dict[str, str] = {}
    for item in perception.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        speaker_id = (
            item.get("speaker_id") or item.get("speaker") or item.get("label") or item.get("id")
        )
        if speaker_id is None:
            continue
        gender = _normalize_binary_gender(
            item.get("perceived_gender") or item.get("gender") or item.get("description")
        )
        if gender:
            speaker_profiles[str(speaker_id)] = gender

    votes = {"female": 0.0, "male": 0.0}
    sources: list[str] = []
    for index, item in enumerate(perception.get("speakers") or []):
        if not isinstance(item, dict):
            continue
        gender = _normalize_binary_gender(
            item.get("perceived_gender") or item.get("gender") or item.get("description")
        )
        if gender:
            votes[gender] += 1.0
            sources.append(f"perception_report.speakers[{index}]")
    for collection_name in ("speaker_segments", "segments"):
        for index, item in enumerate(perception.get(collection_name) or []):
            if not isinstance(item, dict):
                continue
            gender = _normalize_binary_gender(
                item.get("perceived_gender") or item.get("gender") or item.get("description")
            )
            speaker_id = (
                item.get("speaker_id") or item.get("speaker") or item.get("label") or item.get("id")
            )
            if not gender and speaker_id is not None:
                gender = speaker_profiles.get(str(speaker_id))
            if not gender:
                continue
            duration = _segment_duration_weight(item)
            votes[gender] += duration
            sources.append(f"perception_report.{collection_name}[{index}]")
    if votes["female"] <= 0 and votes["male"] <= 0:
        return {"gender": None, "reason": "no_gender_evidence_in_perception"}
    if abs(votes["female"] - votes["male"]) < 1e-6:
        return {"gender": None, "reason": "ambiguous_source_gender", "votes": votes}
    gender = "female" if votes["female"] > votes["male"] else "male"
    return {"gender": gender, "source": ",".join(sources[:4]), "votes": votes}


def _segment_duration_weight(item: dict[str, Any]) -> float:
    start = _safe_float(_first_present(item, "start_sec", "start", "begin"))
    end = _safe_float(_first_present(item, "end_sec", "end", "stop"))
    if start is not None and end is not None and end > start:
        return max(0.1, float(end) - float(start))
    duration = _safe_float(_first_present(item, "duration_sec", "duration"))
    if duration is not None and duration > 0:
        return max(0.1, float(duration))
    return 1.0


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _normalize_binary_gender(value: Any) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if not lowered:
        return None
    if re.search(r"\b(female|woman|women|girl|girls)\b", lowered) or _has_any(
        lowered, ["女性", "女声", "女说话", "女孩", "女童"]
    ):
        return "female"
    if re.search(r"\b(male|man|men|boy|boys)\b", lowered) or _has_any(
        lowered, ["男性", "男声", "男说话", "男孩", "男童"]
    ):
        return "male"
    return None


def _voice_class_prompt_constraint(voice_class: str, *, backend: str) -> str:
    memory_constraint = _voice_design_memory_profile_value(
        voice_class, "prompt_constraints", backend
    )
    if isinstance(memory_constraint, str) and memory_constraint.strip():
        return memory_constraint.strip()
    constraints = {
        "child": (
            "Strong child-voice constraint: sound like a 5- to 8-year-old prepubescent child, "
            "not a teenager and not an adult female or adult male. Use a very light, small vocal-tract timbre, "
            "clearly high fundamental pitch, bright head-voice quality, minimal chest resonance, no mature adult resonance, "
            "and playful but controlled articulation."
        ),
        "elderly": (
            "Strong elderly-voice constraint: sound like an 80-years-or-older speaker, not a middle-aged adult. "
            "Use weak breath support, slight vocal tremor, thinner low pitch, slower pacing, and mildly leaky or softened "
            "dental consonants as if some teeth are missing, while keeping the words intelligible."
        ),
        "young_adult": (
            "Strong young-adult voice constraint: sound like an 18- to 25-year-old adult, not a child. "
            "Use bright energetic timbre, stable breath, clear articulation, and lively but controlled pitch movement."
        ),
        "middle_aged": (
            "Strong middle-aged voice constraint: sound like a 40- to 55-year-old adult with mature, steady resonance, "
            "moderate pitch, natural pacing, and clear articulation."
        ),
        "adult_female": (
            "Strong adult-female voice constraint: sound like a 25- to 40-year-old adult female, not a child. "
            "Use a clear smooth timbre, naturally higher pitch range, and stable articulation."
        ),
        "adult_male": (
            "Strong adult-male voice constraint: sound like a 25- to 45-year-old adult male. "
            "Use a steady timbre with moderate chest resonance, naturally lower pitch range, and clear articulation."
        ),
    }
    return constraints.get(voice_class, "")


def _append_prompt_constraint(prompt: str, required: str) -> str:
    text = str(prompt or "").strip()
    if required in text:
        return text
    return f"{text} {required}".strip()


def _apply_voice_class_postprocess_to_result(
    result: dict[str, Any],
    *,
    state: dict[str, Any],
    step: AudioEditStep,
    voice_class: str | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, dict) or voice_class not in {"child", "elderly"}:
        return result
    raw_output = _extract_output_path(result)
    if not raw_output:
        return result
    postprocess = _maybe_postprocess_voice_class_audio(
        raw_output,
        work_dir=Path(state["output_dir"]) / step.id,
        voice_class=voice_class,
        options=options,
    )
    updated = dict(result)
    updated["target_voice_class"] = voice_class
    updated["voice_design_memory"] = _voice_design_memory_evidence(voice_class)
    updated["voice_class_audio_postprocess"] = postprocess
    updated["raw_output_audio_path"] = raw_output
    if postprocess.get("output_audio_path"):
        updated["output_audio_path"] = postprocess["output_audio_path"]
    return updated


def _maybe_postprocess_voice_class_audio(
    input_audio_path: str,
    *,
    work_dir: Path,
    voice_class: str | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    input_path = Path(str(input_audio_path)).expanduser()
    result: dict[str, Any] = {
        "operation": "voice_class_audio_postprocess",
        "voice_class": voice_class,
        "enabled": bool(options.get("enabled", True)),
        "input_audio_path": str(input_path),
    }
    if voice_class not in {"child", "elderly"}:
        result["skipped"] = True
        result["reason"] = "unsupported_voice_class"
        return result
    if not result["enabled"]:
        result["skipped"] = True
        result["reason"] = "disabled"
        return result
    if not input_path.exists():
        result["skipped"] = True
        result["reason"] = "input_audio_missing"
        return result
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        result["skipped"] = True
        result["reason"] = "ffmpeg_not_found"
        return result
    try:
        with wave.open(str(input_path), "rb") as wav:
            sample_rate = int(wav.getframerate())
    except (wave.Error, OSError) as exc:
        result["skipped"] = True
        result["reason"] = f"could_not_read_wav_sample_rate: {exc}"
        return result
    filters, semitones, shifted_rate, tempo = _voice_class_postprocess_filters(
        voice_class,
        sample_rate=sample_rate,
        options=options,
    )
    result["pitch_shift_semitones"] = semitones
    if abs(semitones) < 0.01:
        result["skipped"] = True
        result["reason"] = "zero_pitch_shift"
        return result
    if not filters:
        result["skipped"] = True
        result["reason"] = "empty_filter_chain"
        return result
    safe_suffix = str(semitones).replace("-", "m").replace(".", "p")
    output_path = (
        work_dir / f"{input_path.stem}_{voice_class}_voice_class_postprocessed_{safe_suffix}st.wav"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        ",".join(filters),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result.update(
        {
            "output_audio_path": str(output_path),
            "sample_rate": sample_rate,
            "shifted_sample_rate": shifted_rate,
            "tempo_factor": tempo,
            "filter_chain": filters,
            "backend": "ffmpeg_voice_class_dsp",
        }
    )
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        result.pop("output_audio_path", None)
        result["skipped"] = True
        result["reason"] = f"ffmpeg_failed: {exc}"
        return result
    if completed.returncode != 0 or not output_path.exists():
        result.pop("output_audio_path", None)
        result["skipped"] = True
        result["reason"] = "ffmpeg_nonzero_exit"
        stderr = (completed.stderr or "").strip()
        if stderr:
            result["stderr"] = stderr[-1000:]
        return result
    result["skipped"] = False
    return result


def _maybe_postprocess_child_voice_reference(
    input_audio_path: str,
    *,
    work_dir: Path,
    options: dict[str, Any],
) -> dict[str, Any]:
    return _maybe_postprocess_voice_class_audio(
        input_audio_path,
        work_dir=work_dir,
        voice_class="child",
        options=options,
    )


def _voice_class_postprocess_filters(
    voice_class: str,
    *,
    sample_rate: int,
    options: dict[str, Any],
) -> tuple[list[str], float, int, float | None]:
    if voice_class == "child":
        semitones = float(options.get("child_pitch_shift_semitones", 3.5))
        factor = 2 ** (semitones / 12.0)
        shifted_rate = max(8000, min(192000, int(round(sample_rate * factor))))
        tempo = 1.0 / factor
        highpass_hz = float(options.get("child_highpass_hz", 100.0))
        eq_3300_gain = float(options.get("child_eq_3300_gain_db", 0.8))
        eq_5200_gain = float(options.get("child_eq_5200_gain_db", 0.4))
        filters = [
            f"asetrate={shifted_rate}",
            f"aresample={sample_rate}",
            *_atempo_filters(tempo),
            f"highpass=f={_ffmpeg_number(highpass_hz)}",
            f"equalizer=f=3300:t=q:w=1.2:g={_ffmpeg_number(eq_3300_gain)}",
            f"equalizer=f=5200:t=q:w=1.4:g={_ffmpeg_number(eq_5200_gain)}",
        ]
        return filters, semitones, shifted_rate, tempo
    if voice_class == "elderly":
        semitones = float(options.get("elderly_pitch_shift_semitones", -7.0))
        factor = 2 ** (semitones / 12.0)
        shifted_rate = max(8000, min(192000, int(round(sample_rate * factor))))
        preserve_duration = bool(options.get("elderly_preserve_duration", False))
        tempo = 1.0 / factor if preserve_duration else None
        vibrato_hz = float(options.get("elderly_vibrato_hz", 4.8))
        vibrato_depth = float(options.get("elderly_vibrato_depth", 0.24))
        tremolo_hz = float(options.get("elderly_tremolo_hz", 4.4))
        tremolo_depth = float(options.get("elderly_tremolo_depth", 0.22))
        lowpass_hz = float(options.get("elderly_lowpass_hz", 3200.0))
        highpass_hz = float(options.get("elderly_highpass_hz", 60.0))
        eq_900_gain = float(options.get("elderly_eq_900_gain_db", -3.0))
        filters = [
            f"asetrate={shifted_rate}",
            f"aresample={sample_rate}",
            *(_atempo_filters(tempo) if tempo else []),
            f"vibrato=f={_ffmpeg_number(vibrato_hz)}:d={_ffmpeg_number(vibrato_depth)}",
            f"tremolo=f={_ffmpeg_number(tremolo_hz)}:d={_ffmpeg_number(tremolo_depth)}",
            f"lowpass=f={_ffmpeg_number(lowpass_hz)}",
            f"highpass=f={_ffmpeg_number(highpass_hz)}",
            f"equalizer=f=900:t=q:w=1.2:g={_ffmpeg_number(eq_900_gain)}",
        ]
        return filters, semitones, shifted_rate, tempo
    return [], 0.0, sample_rate, None


def _ffmpeg_number(value: float) -> str:
    return f"{float(value):g}"


def _atempo_filters(value: float | None) -> list[str]:
    if value is None:
        return []
    tempo = float(value)
    filters: list[str] = []
    while tempo < 0.5:
        filters.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        filters.append("atempo=2.0")
        tempo /= 2.0
    filters.append(f"atempo={tempo:.6f}")
    return filters


def _qwen3_reference_voice_prompt(request: str) -> str:
    from x_editor.runtime.operation_normalizers import _qwen3_voice_design_prompt

    return _qwen3_voice_design_prompt(request=request, source_text="", perception={})


def _is_descriptive_voice_prompt(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < 24:
        return False
    descriptive_markers = [
        "Gender/age:",
        "Pitch/range:",
        "Timbre texture:",
        "情绪",
        "语速",
        "音调",
        "嗓音",
        "发音",
        "timbre",
        "pitch",
        "speaking rate",
        "emotion",
    ]
    return any(marker.lower() in text.lower() for marker in descriptive_markers)




def _record_step_audio_artifacts(
    step: AudioEditStep, state: dict[str, Any], output: dict[str, Any] | None
) -> None:
    if not isinstance(output, dict):
        return
    delivery_intermediate_paths = _delivery_intermediate_audio_path_keys(output)
    candidates: dict[str, tuple[int, int, str, str]] = {}
    for ordinal, (key_path, path_text) in enumerate(_iter_audio_path_values(output)):
        if _is_executor_evidence_path(key_path):
            continue
        if not _should_collect_debug_audio_path(key_path, path_text, state):
            continue
        resolved = _resolved_path_key(path_text)
        if resolved in delivery_intermediate_paths:
            continue
        priority = _audio_artifact_candidate_priority(key_path)
        current = candidates.get(resolved)
        if current is None or (priority, ordinal) < (current[0], current[1]):
            candidates[resolved] = (priority, ordinal, key_path, str(path_text))
    for _, _, key_path, path_text in sorted(candidates.values()):
        leaf = _audio_artifact_key_leaf(key_path)
        role = (
            step.expected_output
            if leaf in _PRIMARY_OUTPUT_PATH_KEYS and step.expected_output
            else leaf
        )
        _record_audio_artifact(
            state,
            artifact_id=f"{step.id}:{_safe_artifact_key(key_path)}",
            path=path_text,
            role=role,
            kind="step_output",
            source=f"step_result.{step.id}.{key_path}",
            step=step,
            description=step.description,
            expected_output=step.expected_output,
            result_key=key_path,
        )
    _record_delivery_tool_audio_artifacts(step, state, output)


def _apply_suffix_resume_reuse(
    plan: AudioEditPlan,
    state: dict[str, Any],
) -> None:
    raw_receipts = state["context"].get("_workflow_suffix_resume_reuse_receipts")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        return
    try:
        receipts = [SuffixResumeReuseReceipt.model_validate(item) for item in raw_receipts]
        receipt_by_step = {item.producer_step_id: item for item in receipts}
        if len(receipt_by_step) != len(receipts):
            raise ValueError("duplicate suffix-resume receipt step")
        steps = {step.id: step for step in plan.steps}
        for receipt in receipts:
            step = steps.get(receipt.producer_step_id)
            if step is None or step.operation != receipt.operation:
                raise ValueError(
                    f"suffix-resume receipt does not match step {receipt.producer_step_id!r}"
                )
            path = Path(receipt.path).expanduser()
            if not path.is_file() or path.stat().st_size != receipt.size_bytes:
                raise ValueError(
                    f"suffix-resume artifact changed before executor binding: {receipt.path}"
                )
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != receipt.content_sha256:
                raise ValueError(
                    f"suffix-resume artifact hash changed before executor binding: {receipt.path}"
                )
    except (OSError, ValueError, TypeError) as exc:
        state["suffix_resume_reuse_error"] = f"{type(exc).__name__}: {exc}"
        return

    applied_receipts: list[dict[str, Any]] = []
    reused_step_ids: list[str] = []
    for step in plan.steps:
        receipt = receipt_by_step.get(step.id)
        if receipt is None:
            continue
        current_before = state.get("current_audio_path")
        output = {
            **copy.deepcopy(receipt.retained_output),
            receipt.result_key: receipt.path,
            "suffix_resume_reused": True,
            "suffix_resume_receipt_id": receipt.receipt_id,
        }
        step.status = "success"
        step.error = None
        step.result = output
        timeline_ledger = output.get(AUDIO_TIMELINE_LEDGER_KEY)
        if isinstance(timeline_ledger, dict):
            state[AUDIO_TIMELINE_LEDGER_KEY] = copy.deepcopy(timeline_ledger)
        state["current_audio_path"] = receipt.path
        state["artifacts"][step.id] = receipt.path
        if step.expected_output:
            state["artifacts"][step.expected_output] = receipt.path
        state["step_outputs"][step.id] = output
        _record_step_audio_artifacts(step, state, output)
        _apply_step_state_updates(output, state)
        _apply_step_text_state_updates(step, output, state)
        state["verification_reports"].append(dict(receipt.verification_report))
        state["executor_step_records"].append(
            _without_empty(
                {
                    "step_id": step.id,
                    "operation": step.operation,
                    "description": step.description,
                    "status": "reused",
                    "step_status": "success",
                    "suffix_resume_reused": True,
                    "suffix_resume_receipt_id": receipt.receipt_id,
                    "artifact_ref_id": receipt.artifact_ref_id,
                    "current_audio_before": current_before,
                    "current_audio_after": receipt.path,
                    "output_path": receipt.path,
                    "result_keys": [receipt.result_key],
                    "consumed_audio_paths_observed": True,
                    "consumed_audio_paths": list(receipt.parent_paths),
                    "skipped_tool_call_count": receipt.skipped_tool_call_count,
                }
            )
        )
        reused_step_ids.append(step.id)
        applied_receipts.append(receipt.model_dump(mode="json"))
    state["suffix_resume_reused_step_ids"] = reused_step_ids
    state["suffix_resume_reuse_receipts"] = applied_receipts


def _delivery_intermediate_audio_path_keys(output: dict[str, Any]) -> set[str]:
    contract = output.get(ARTIFACT_DELIVERY_CONTRACT_KEY)
    if not isinstance(contract, dict):
        return set()
    invocations = contract.get("tool_invocations")
    if not isinstance(invocations, list):
        return set()
    primary_output = _extract_output_path(output)
    primary_key = _resolved_path_key(primary_output) if primary_output else None
    return {
        resolved
        for invocation in invocations
        if isinstance(invocation, dict) and invocation.get("success") is True
        for path_text in [invocation.get("output_audio_path")]
        if isinstance(path_text, str) and path_text.strip()
        for resolved in [_resolved_path_key(path_text)]
        if resolved != primary_key
    }


def _record_delivery_tool_audio_artifacts(
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
) -> None:
    contract = output.get(ARTIFACT_DELIVERY_CONTRACT_KEY)
    if not isinstance(contract, dict):
        return
    invocations = contract.get("tool_invocations")
    if not isinstance(invocations, list):
        return
    primary_output = _extract_output_path(output)
    primary_key = _resolved_path_key(primary_output) if primary_output else None
    for index, invocation in enumerate(invocations):
        if not isinstance(invocation, dict) or invocation.get("success") is not True:
            continue
        path_text = invocation.get("output_audio_path")
        if not isinstance(path_text, str) or not path_text.strip():
            continue
        if primary_key is not None and _resolved_path_key(path_text) == primary_key:
            continue
        key_path = f"{ARTIFACT_DELIVERY_CONTRACT_KEY}.tool_invocations[{index}].output_audio_path"
        if not _should_collect_debug_audio_path(key_path, path_text, state):
            continue
        tool_id = str(invocation.get("tool_id") or "tool")
        _record_audio_artifact(
            state,
            artifact_id=f"{step.id}:delivery_tool:{index}:{_safe_artifact_key(tool_id)}",
            path=path_text,
            role=f"{tool_id}_intermediate_output",
            kind="tool_intermediate_output",
            source=f"step_result.{step.id}.{key_path}",
            step=step,
            description=step.description,
            expected_output=step.expected_output,
            result_key=key_path,
        )


def _is_executor_evidence_path(key_path: str) -> bool:
    return any(
        key_path == root or key_path.startswith(f"{root}.")
        for root in (
            ARTIFACT_DELIVERY_CONTRACT_KEY,
            AUDIO_TEMPORAL_LINEAGE_KEY,
            AUDIO_TEMPORAL_TRANSFORM_KEY,
            AUDIO_TIMELINE_LEDGER_KEY,
            _RUNTIME_ARTIFACT_COMMIT_KEY,
        )
    )


def _iter_audio_path_values(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            paths.extend(_iter_audio_path_values(item, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_iter_audio_path_values(item, child_prefix))
    elif isinstance(value, str) and prefix and _looks_like_audio_path(value):
        paths.append((prefix, value))
    return paths


def _should_collect_debug_audio_path(key_path: str, path_text: str, state: dict[str, Any]) -> bool:
    if not _looks_like_audio_path(path_text):
        return False
    path = Path(str(path_text))
    if not path.exists():
        return False
    leaf = _audio_artifact_key_leaf(key_path)
    if leaf in _REFERENCE_AUDIO_PATH_KEYS:
        return False
    under_output_dir = _path_is_under(path, Path(str(state.get("output_dir") or ".")))
    if _matches_known_input_or_reference_audio(path, state) and not under_output_dir:
        return False
    if leaf in _PRIMARY_OUTPUT_PATH_KEYS:
        return True
    if leaf in {"source_audio_copy"}:
        return True
    joined = key_path.lower()
    if leaf.endswith("_audio_path") or leaf.endswith("_audio_paths"):
        return True
    if leaf.endswith("_path") or leaf.endswith("_paths") or leaf in {"audio_path", "audio_paths"}:
        return any(
            marker in joined
            for marker in (
                "audio",
                "background",
                "converted",
                "fitted",
                "foreground",
                "generated",
                "segment",
                "source_copy",
                "stem",
                "synthesized",
            )
        )
    return False


def _record_audio_artifact(
    state: dict[str, Any],
    *,
    artifact_id: str,
    path: str,
    role: str,
    kind: str,
    source: str,
    step: AudioEditStep | None = None,
    description: str | None = None,
    expected_output: str | None = None,
    result_key: str | None = None,
) -> None:
    if not _looks_like_audio_path(path):
        return
    path_obj = Path(str(path))
    if not path_obj.exists():
        return
    resolved = _resolved_path_key(path)
    seen = state.setdefault("audio_artifact_seen", set())
    step_id = step.id if step is not None else None
    seen_key = (kind, step_id, resolved)
    if seen_key in seen:
        return
    seen.add(seen_key)
    stat = path_obj.stat()
    duration = _audio_duration_seconds(path_obj)
    record = _without_empty(
        {
            "artifact_id": artifact_id,
            "step_id": step_id,
            "operation": step.operation if step is not None else None,
            "role": role,
            "kind": kind,
            "path": str(path_obj),
            "exists": True,
            "size_bytes": stat.st_size,
            "duration_sec": duration,
            "source": source,
            "description": description,
            "expected_output": expected_output,
            "result_key": result_key,
            "is_final_output": False,
            "_resolved_path": resolved,
        }
    )
    state.setdefault("audio_artifacts", []).append(record)


def _mark_final_audio_artifacts(state: dict[str, Any], final_output_path: str | None) -> None:
    if not final_output_path:
        return
    final_resolved = _resolved_path_key(final_output_path)
    for item in state.get("audio_artifacts", []) or []:
        if isinstance(item, dict) and item.get("_resolved_path") == final_resolved:
            item["is_final_output"] = True


def _public_audio_artifacts(state: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for item in state.get("audio_artifacts", []) or []:
        if isinstance(item, dict):
            artifacts.append(
                {key: value for key, value in item.items() if not str(key).startswith("_")}
            )
    return artifacts


def _attach_operation_graph_metadata(plan: AudioEditPlan, state: dict[str, Any]) -> None:
    try:
        from x_editor.session.operation_graph import (
            build_actual_operation_graph,
            build_operation_graph_event_log,
            build_planned_operation_graph,
            build_round_operation_graph_store,
            reconcile_planned_actual_graph,
            validate_operation_graph,
            validate_round_operation_graph_store,
        )

        actual = build_actual_operation_graph(plan)
        actual_payload = actual.model_dump(exclude_none=True)
        if isinstance(plan.metadata.get("operation_graph_plan"), dict):
            planned = build_planned_operation_graph(plan)
            reconciliation = reconcile_planned_actual_graph(planned, actual)
            actual_payload = reconciliation.pop("actual_graph", actual_payload)
            plan.metadata["operation_graph_reconciliation"] = reconciliation
        plan.metadata["operation_graph_actual"] = actual_payload
        plan.metadata["operation_graph_validation"] = validate_operation_graph(
            actual_payload,
            plan=plan,
        ).model_dump(exclude_none=True)
        round_store = build_round_operation_graph_store(
            actual_payload,
            round_index=_operation_graph_execution_round_index(plan, state),
            graph_kind="actual",
            source="executor.operation_graph_actual",
        )
        round_store_payload = round_store.model_dump(exclude_none=True)
        plan.metadata["operation_graph_round_store"] = round_store_payload
        plan.metadata["operation_graph_round_validation"] = validate_round_operation_graph_store(
            round_store_payload,
        ).model_dump(exclude_none=True)
        plan.metadata["session_event_log"] = build_operation_graph_event_log(
            plan,
            round_store=round_store,
            actual_graph=actual_payload,
        ).model_dump(exclude_none=True)
    except Exception as exc:
        plan.metadata["operation_graph_error"] = f"{type(exc).__name__}: {exc}"


def _audio_artifact_candidate_priority(key_path: str) -> int:
    leaf = _audio_artifact_key_leaf(key_path)
    if leaf in _PRIMARY_OUTPUT_PATH_KEYS:
        return 0
    if any(
        marker in leaf
        for marker in ("foreground", "background", "converted", "synthesized", "fitted")
    ):
        return 1
    if any(marker in leaf for marker in ("segment", "stem", "generated")):
        return 2
    if leaf in _REFERENCE_AUDIO_PATH_KEYS:
        return 9
    return 5


def _matches_known_input_or_reference_audio(path: Path, state: dict[str, Any]) -> bool:
    resolved = _resolved_path_key(path)
    candidates: list[Any] = [
        state.get("input_audio"),
        *(state.get("input_audios") or []),
        *(state.get("reference_audio_paths") or []),
    ]
    return any(resolved == _resolved_path_key(candidate) for candidate in candidates if candidate)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _resolved_path_key(path: Any) -> str:
    try:
        return str(Path(str(path)).resolve())
    except Exception:
        return str(path)


def _looks_like_audio_path(path_text: str) -> bool:
    return Path(str(path_text)).suffix.lower() in _AUDIO_ARTIFACT_EXTENSIONS


def _audio_artifact_key_leaf(key_path: str) -> str:
    leaf = key_path.split(".")[-1].lower()
    return re.sub(r"\[\d+\]$", "", leaf)


def _safe_artifact_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_")
    return cleaned or "audio"


def _transient_tool_max_attempts(tool_id: str) -> int:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(tool_id).upper())
    raw = os.getenv(f"X_EDITOR_TOOL_TRANSIENT_RETRIES_{normalized}") or os.getenv(
        "X_EDITOR_TOOL_TRANSIENT_RETRIES"
    )
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    if tool_id == "edit_data_pipeline":
        return 2
    return 1


def _transient_tool_retry_delay_sec() -> float:
    raw = os.getenv("X_EDITOR_TOOL_TRANSIENT_RETRY_DELAY_SEC")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 1.0
    return 1.0


def _is_transient_tool_failure(tool_id: str, message: str | None) -> bool:
    if not message:
        return False
    text = str(message).lower()
    transient_markers = (
        "readtimeout",
        "read timed out",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connectionerror",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "502",
        "503",
        "504",
    )
    if not any(marker in text for marker in transient_markers):
        return False
    # edit_data_pipeline occasionally delegates to external model services; a
    # single whole-tool retry removes transient upstream timeouts without
    # hiding deterministic operation-boundary failures.
    return tool_id == "edit_data_pipeline" or bool(os.getenv("X_EDITOR_TOOL_TRANSIENT_RETRIES"))


def _is_f5_text_edit_duration_limit_error(message: str | None) -> bool:
    if not message:
        return False
    text = str(message).lower()
    return (
        "target timeline exceeds f5 edit limit" in text
        or "f5-tts edit request exceeds the single-call duration limit" in text
    )


def _without_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _gemini_perception_speaker_ids(perception: Any) -> list[str]:
    if not isinstance(perception, dict):
        return []
    gemini = perception.get("gemini_perception")
    if not isinstance(gemini, dict):
        return []
    ids: list[str] = []
    for collection_key in ("segments", "speakers"):
        collection = gemini.get(collection_key)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            speaker_id = item.get("speaker_id")
            if speaker_id is None:
                continue
            text = str(speaker_id).strip()
            if text and text not in ids:
                ids.append(text)
    return ids


def _flexsed_event_list_from_step(step: AudioEditStep, params: dict[str, Any]) -> list[str]:
    values: list[str] = []
    raw_candidates = [
        params.pop("events", None),
        params.pop("event_labels", None),
        params.pop("event_label", None),
        params.pop("target_events", None),
        params.pop("target_event", None),
        step.inputs.get("events"),
        step.inputs.get("event_labels"),
        step.inputs.get("event_label"),
        step.inputs.get("target_events"),
        step.inputs.get("target_event"),
    ]
    event_list = params.pop("event_list", None) or step.inputs.get("event_list")
    if event_list:
        raw_candidates.append(event_list)
    for raw in raw_candidates:
        if isinstance(raw, str):
            values.extend(item.strip() for item in re.split(r"[,\n;]+", raw) if item.strip())
        elif isinstance(raw, (list, tuple, set)):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _flexsed_selected_spans(timestamps: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    if not isinstance(timestamps, dict):
        return spans
    for event, items in timestamps.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            start = _safe_float(item.get("start_sec"))
            end = _safe_float(item.get("end_sec"))
            if start is None or end is None or end <= start:
                continue
            spans.append(
                _without_empty(
                    {
                        "start_sec": float(start),
                        "end_sec": float(end),
                        "event": str(item.get("event") or event),
                        "reason": f"FlexSED detected {item.get('event') or event}",
                        "source": "flexsed",
                        "confidence": item.get("peak_score"),
                        "mean_score": item.get("mean_score"),
                        "frame_start": item.get("frame_start"),
                        "frame_end": item.get("frame_end"),
                    }
                )
            )
    spans.sort(
        key=lambda item: (
            float(item.get("start_sec", 0.0)),
            float(item.get("end_sec", 0.0)),
        )
    )
    return spans


def _caption_selected_spans(perception: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    raw = perception.get("audio_caption_segments") if isinstance(perception, dict) else None
    if not isinstance(raw, list):
        return spans
    has_explicit_target_match = any(
        isinstance(item, dict)
        and str(item.get("target_match") or "").strip().lower()
        in {"yes", "no", "uncertain"}
        for item in raw
    )
    for item in raw:
        if not isinstance(item, dict):
            continue
        if (
            has_explicit_target_match
            and str(item.get("target_match") or "uncertain").strip().lower() != "yes"
        ):
            continue
        start = _safe_float(item.get("start_sec"))
        end = _safe_float(item.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        events = item.get("events") if isinstance(item.get("events"), list) else []
        spans.append(
            _without_empty(
                {
                    "start_sec": float(start),
                    "end_sec": float(end),
                    "event": ", ".join(str(event) for event in events if str(event).strip()),
                    "caption": item.get("caption"),
                    "reason": item.get("caption") or "Timestamped audio caption segment",
                    "source": "timestamped_caption",
                    "confidence": item.get("confidence"),
                    "needs_review": item.get("needs_review"),
                    "evidence_source": item.get("evidence_source"),
                    "target_match": item.get("target_match"),
                }
            )
        )
    spans.sort(
        key=lambda item: (
            float(item.get("start_sec", 0.0)),
            float(item.get("end_sec", 0.0)),
        )
    )
    return spans


_ARTIFACT_EVENT_GROUNDING_OPERATIONS = {
    "extract_spans",
    "remove_spans",
    "mute_spans",
    "beep_spans",
    "speaker_scoped_edit",
}
_DESTRUCTIVE_EVENT_GROUNDING_OPERATIONS = {"remove_spans", "mute_spans", "beep_spans"}
_SPEECH_EVENT_LABEL_BLOCKLIST = {
    "speech",
    "male speech",
    "female speech",
    "speaker",
    "voice",
    "spoken words",
    "question",
    "answer",
    "sentence",
    "phrase",
    "utterance",
}
_ARTIFACT_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "laughter": (
        "laughter",
        "laughing",
        "laugh",
        "laughs",
        "giggle",
        "giggles",
        "giggling",
        "chuckle",
        "chuckles",
        "chuckling",
        "笑声",
        "发笑",
        "笑",
    ),
    "whisper": (
        "whisper",
        "whispering",
        "whispered",
        "soft whisper",
        "hushed voice",
        "低语",
        "耳语",
        "轻声",
    ),
    "singing": (
        "singing",
        "singing voice",
        "soft singing",
        "song vocal",
        "vocals",
        "humming",
        "hum",
        "rap",
        "rapping",
        "chanting",
        "歌声",
        "唱歌",
        "哼唱",
        "说唱",
    ),
    "cough": ("cough", "coughing", "coughs", "咳嗽"),
    "sneeze": ("sneeze", "sneezing", "sneezes", "喷嚏", "打喷嚏"),
    "applause": ("applause", "clapping", "clap", "claps", "掌声", "鼓掌"),
    "siren": ("siren", "alarm", "warning tone", "警报", "警笛"),
    "phone ring": (
        "phone ring",
        "ringtone",
        "telephone ring",
        "ringing phone",
        "电话铃",
    ),
    "footsteps": ("footsteps", "footstep", "steps", "walking sound", "脚步"),
    "music": (
        "music",
        "background music",
        "song",
        "singing",
        "音乐",
        "背景音乐",
        "歌声",
    ),
    "wind": ("wind", "wind noise", "strong wind", "风声"),
    "crying": ("crying", "cry", "sobbing", "weeping", "哭声", "哭泣"),
    "scream": (
        "scream",
        "screaming",
        "shout",
        "shouting",
        "yell",
        "yelling",
        "尖叫",
        "喊叫",
    ),
    "breathing": ("breathing", "breath", "inhale", "exhale", "呼吸"),
    "dog bark": ("dog bark", "barking", "bark", "dog", "狗叫"),
    "cat meow": ("cat meow", "meow", "cat", "猫叫"),
    "bird": ("bird", "birds", "birdsong", "chirping", "鸟叫"),
    "crowd noise": ("crowd noise", "crowd", "people talking", "人群"),
    "knock": ("knock", "knocking", "door knock", "敲门", "敲击"),
}


def _artifact_event_grounding_is_needed(
    step: AudioEditStep,
    operation: str,
    instruction: str,
    params: dict[str, Any],
) -> bool:
    if operation not in _ARTIFACT_EVENT_GROUNDING_OPERATIONS:
        return False
    if operation != "speaker_scoped_edit" and _has_explicit_span_params(params):
        return False
    labels = _artifact_event_target_labels(step, instruction, params)
    if not labels:
        return False
    if operation in _DESTRUCTIVE_EVENT_GROUNDING_OPERATIONS:
        return True
    explicit_labels = _flexsed_event_list_from_step(step, dict(params))
    text = " ".join(
        str(value)
        for value in [
            instruction,
            step.description,
            params.get("selection_instruction"),
        ]
        if value
    )
    if operation == "speaker_scoped_edit":
        return _has_any(
            text.lower(),
            ["remove", "delete", "drop", "get rid of", "去掉", "删除", "移除"],
        )
    return bool(explicit_labels or _has_any(text.lower(), _ARTIFACT_EVENT_TRIGGER_TERMS))


_ARTIFACT_EVENT_TRIGGER_TERMS = [
    "sound",
    "event",
    "caption",
    "non-speech",
    "non speech",
    "paralinguistic",
    "laughter",
    "laughing",
    "giggle",
    "whisper",
    "whispering",
    "hushed",
    "singing",
    "singing voice",
    "soft singing",
    "humming",
    "rap",
    "cough",
    "sneeze",
    "siren",
    "applause",
    "music",
    "breathing",
    "crying",
    "scream",
    "footstep",
    "footsteps",
    "wind",
    "rain",
    "background noise",
    "room tone",
    "环境音",
    "背景音",
    "笑声",
    "耳语",
    "低语",
    "歌声",
    "哼唱",
    "呼吸",
    "咳嗽",
    "掌声",
]


def _artifact_event_target_labels(
    step: AudioEditStep, instruction: str, params: dict[str, Any]
) -> list[str]:
    labels: list[str] = []
    for label in _flexsed_event_list_from_step(step, dict(params)):
        if _event_label_allowed_for_artifact_grounding(label):
            labels.append(label)
    text = " ".join(
        str(value)
        for value in [
            instruction,
            step.description,
            params.get("selection_instruction"),
            params.get("selector_instruction"),
            params.get("target_event"),
            params.get("target_events"),
        ]
        if value
    )
    lowered = text.casefold()
    for canonical, aliases in _ARTIFACT_EVENT_ALIASES.items():
        if any(_text_mentions_event_alias(lowered, alias) for alias in aliases):
            labels.append(canonical)
            labels.extend(_event_contextual_labels(canonical, lowered))
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = str(label).strip()
        key = normalized.casefold()
        if (
            not normalized
            or key in seen
            or not _event_label_allowed_for_artifact_grounding(normalized)
        ):
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _event_labels_are_negative_extract_filters(
    step: AudioEditStep,
    instruction: str,
    params: dict[str, Any],
    event_labels: list[str],
) -> bool:
    text = " ".join(
        str(value)
        for value in [
            instruction,
            step.description,
            params.get("selection_instruction"),
            params.get("selector_instruction"),
        ]
        if value
    ).casefold()
    if not text or not event_labels:
        return False
    if not _looks_like_speech_or_speaker_extraction(text):
        return False
    mentions: list[tuple[int, str]] = []
    for alias in _event_alias_terms(event_labels):
        lowered_alias = str(alias).casefold()
        if not lowered_alias:
            continue
        if re.search(r"[a-z0-9]", lowered_alias):
            pattern = rf"(?<![a-z0-9]){re.escape(lowered_alias)}(?![a-z0-9])"
        else:
            pattern = re.escape(lowered_alias)
        mentions.extend((match.start(), lowered_alias) for match in re.finditer(pattern, text))
    if not mentions:
        return False
    return all(_event_mention_is_in_negative_context(text, start) for start, _alias in mentions)


def _looks_like_speech_or_speaker_extraction(text: str) -> bool:
    if not _has_any(
        text,
        [
            "extract",
            "isolate",
            "keep only",
            "retain only",
            "only keep",
            "提取",
            "只保留",
            "仅保留",
        ],
    ):
        return False
    return _has_any(
        text,
        [
            "speech spoken by",
            "speech from",
            "spoken by",
            "speaker",
            "speaker's",
            "voice",
            "talking",
            "utterance",
            "dialogue",
            "baby",
            "child",
            "kid",
            "girl",
            "boy",
            "woman",
            "man",
            "说话人",
            "人声",
            "语音",
            "讲话",
        ],
    )


def _event_mention_is_in_negative_context(text: str, start: int) -> bool:
    window = text[max(0, start - 96) : start + 48]
    return _has_any(
        window,
        [
            "exclude",
            "excluding",
            "excluded",
            "without",
            "not include",
            "not including",
            "except",
            "other than",
            "rather than",
            "non-speech vocalization",
            "non speech vocalization",
            "non-speech sounds",
            "non speech sounds",
            "remove",
            "removing",
            "drop",
            "filter out",
            "排除",
            "不包含",
            "不要",
            "去除",
            "去掉",
            "删除",
            "过滤",
            "非语音",
        ],
    )


def _event_contextual_labels(canonical: str, lowered_instruction: str) -> list[str]:
    labels: list[str] = []
    if _mentions_female_target(lowered_instruction):
        labels.append(f"female {canonical}")
    if _mentions_male_target(lowered_instruction):
        labels.append(f"male {canonical}")
    if (
        "group" in lowered_instruction
        or "multiple people" in lowered_instruction
        or "多人" in lowered_instruction
    ):
        labels.append(f"group {canonical}")
    return labels


def _mentions_female_target(text: str) -> bool:
    return (
        bool(re.search(r"(?<![a-z0-9])(?:female|woman|women|girl|her)(?![a-z0-9])", text))
        or "女" in text
    )


def _mentions_male_target(text: str) -> bool:
    return (
        bool(re.search(r"(?<![a-z0-9])(?:male|man|men|boy|his|him)(?![a-z0-9])", text))
        or "男" in text
    )


def _event_label_allowed_for_artifact_grounding(label: str) -> bool:
    normalized = str(label).strip().casefold()
    if not normalized:
        return False
    return normalized not in _SPEECH_EVENT_LABEL_BLOCKLIST


def _text_mentions_event_alias(lowered_text: str, alias: str) -> bool:
    lowered_alias = str(alias).casefold()
    if not lowered_alias:
        return False
    if re.search(r"[a-z0-9]", lowered_alias):
        return bool(
            re.search(rf"(?<![a-z0-9]){re.escape(lowered_alias)}(?![a-z0-9])", lowered_text)
        )
    return lowered_alias in lowered_text


def _event_spans_from_prior_step_outputs(
    state: dict[str, Any], event_labels: list[str]
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for output in (state.get("step_outputs") or {}).values():
        if not isinstance(output, dict):
            continue
        spans.extend(_selected_spans_from_step_output(output))
        perception = output.get("perception_report")
        if isinstance(perception, dict):
            spans.extend(_caption_selected_spans(perception))
            for event_report in perception.get("audio_events") or []:
                if isinstance(event_report, dict):
                    spans.extend(_flexsed_selected_spans(event_report.get("timestamps") or {}))
    return _merge_event_grounded_spans(
        _filter_event_grounded_spans(spans, event_labels, destructive=True)
    )


def _filter_event_grounded_spans(
    spans: list[dict[str, Any]],
    event_labels: list[str],
    *,
    destructive: bool,
) -> list[dict[str, Any]]:
    aliases = _event_alias_terms(event_labels)
    filtered: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        start = _safe_float(span.get("start_sec", span.get("start")))
        end = _safe_float(span.get("end_sec", span.get("end")))
        if start is None or end is None or end <= start:
            continue
        if destructive and _event_span_is_model_only(span):
            continue
        haystack = " ".join(
            str(value)
            for value in [
                span.get("event"),
                span.get("events"),
                span.get("caption"),
                span.get("reason"),
                span.get("description"),
                span.get("label"),
            ]
            if value
        ).casefold()
        if not any(_text_mentions_event_alias(haystack, alias) for alias in aliases):
            continue
        normalized = dict(span)
        normalized["start_sec"] = float(start)
        normalized["end_sec"] = float(end)
        normalized.setdefault("source", span.get("source") or "event_grounding")
        filtered.append(_without_empty(normalized))
    filtered.sort(
        key=lambda item: (
            float(item.get("start_sec", 0.0)),
            float(item.get("end_sec", 0.0)),
        )
    )
    return filtered


def _event_span_is_model_only(span: dict[str, Any]) -> bool:
    evidence_source = str(span.get("evidence_source") or "").casefold()
    if evidence_source in {"model_only", "omni_only", "qwen3_omni_only", "gemini_only"}:
        return True
    return (
        bool(span.get("needs_review"))
        and str(span.get("source") or "").casefold() == "timestamped_caption"
    )


def _event_alias_terms(event_labels: list[str]) -> list[str]:
    terms: list[str] = []
    for label in event_labels:
        text = str(label).strip()
        if not text:
            continue
        terms.append(text)
        key = text.casefold()
        for canonical, aliases in _ARTIFACT_EVENT_ALIASES.items():
            if key == canonical or any(key == alias.casefold() for alias in aliases):
                terms.append(canonical)
                terms.extend(aliases)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = str(term).casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(str(term))
    return deduped


def _merge_event_grounded_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return []
    normalized = []
    for span in spans:
        start = _safe_float(span.get("start_sec", span.get("start")))
        end = _safe_float(span.get("end_sec", span.get("end")))
        if start is None or end is None or end <= start:
            continue
        normalized.append({**span, "start_sec": float(start), "end_sec": float(end)})
    normalized.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    merged: list[dict[str, Any]] = []
    for span in normalized:
        if not merged or float(span["start_sec"]) > float(merged[-1]["end_sec"]) + 0.12:
            merged.append(dict(span))
            continue
        current = merged[-1]
        current["end_sec"] = max(float(current["end_sec"]), float(span["end_sec"]))
        current["source"] = _join_unique_values(current.get("source"), span.get("source"))
        current["event"] = _join_unique_values(current.get("event"), span.get("event"))
        current["reason"] = _join_unique_values(current.get("reason"), span.get("reason"))
    return [_without_empty(span) for span in merged]


def _pad_event_spans(spans: list[dict[str, Any]], padding_sec: float) -> list[dict[str, Any]]:
    padded = []
    for span in spans:
        start = max(0.0, float(span["start_sec"]) - padding_sec)
        end = float(span["end_sec"]) + padding_sec
        padded.append(
            {
                **span,
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "event_span_padding_sec": padding_sec,
            }
        )
    return _merge_event_grounded_spans(padded)


def _join_unique_values(left: Any, right: Any) -> str:
    values: list[str] = []
    for raw in (left, right):
        if raw is None:
            continue
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item).strip())
        else:
            values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return ", ".join(deduped)


def _apply_step_state_updates(output: dict[str, Any] | None, state: dict[str, Any]) -> None:
    if not isinstance(output, dict):
        return
    perception = output.get("perception_report")
    if isinstance(perception, dict):
        input_audio_path = output.get("input_audio_path")
        if isinstance(input_audio_path, str) and input_audio_path.strip():
            # Reviewed output uses public refs; runtime caches and artifact
            # identity must use the path admitted by the audio binding contract.
            input_audio_path = resolve_audio_reference(input_audio_path, state)
            reports = state.setdefault("perception_reports_by_audio_path", {})
            if isinstance(reports, dict):
                reports[str(input_audio_path)] = perception
                try:
                    reports[str(Path(input_audio_path).expanduser().resolve())] = perception
                except Exception:
                    pass
        state["perception_report"] = perception
        evidence_path = (
            str(input_audio_path)
            if isinstance(input_audio_path, str) and input_audio_path.strip()
            else str(state.get("current_audio_path") or "").strip()
        )
        if evidence_path:
            _bind_perception_evidence_timebase(
                state,
                perception,
                _runtime_artifact_for_path(state, evidence_path),
            )
        source_text = perception.get("source_text")
        if isinstance(source_text, str) and source_text.strip():
            state["source_text"] = source_text
        if perception.get("source_text_audio_path"):
            state["source_text_audio_path"] = perception.get("source_text_audio_path")
        if perception.get("source_text_audio_index") is not None:
            state["source_text_audio_index"] = perception.get("source_text_audio_index")
    source_text = output.get("source_text")
    if isinstance(source_text, str) and source_text.strip():
        state["source_text"] = source_text


def _apply_step_text_state_updates(
    step: AudioEditStep,
    output: dict[str, Any] | None,
    state: dict[str, Any],
) -> None:
    if step.operation != "edit_data_pipeline.text_edit":
        return
    output = output if isinstance(output, dict) else {}
    target_text = (
        _first_string_value(output, "target_text", "edited_text", "text")
        or _first_string_value(step.params, "target_text")
        or _target_text_from_instruction_st(step.params.get("instruction_st"))
    )
    if not target_text:
        return
    source_text = (
        _first_string_value(output, "source_text")
        or _first_string_value(step.params, "source_text")
        or (state.get("source_text") if isinstance(state.get("source_text"), str) else None)
    )
    text_record = _without_empty(
        {
            "operation": step.operation,
            "source_text": source_text,
            "target_text": target_text,
            "output_audio_path": _extract_output_path(output),
        }
    )
    state.setdefault("step_text_outputs", {})[step.id] = text_record
    state["source_text"] = target_text
    state["current_text"] = target_text
    state["current_text_source"] = step.operation
    state.setdefault("text_state_updates", []).append(
        _without_empty(
            {
                "step_id": step.id,
                "operation": step.operation,
                "source_text": source_text,
                "target_text": target_text,
            }
        )
    )


def _text_from_text_edit_dependencies(step: AudioEditStep, state: dict[str, Any]) -> str | None:
    text_outputs = state.get("step_text_outputs")
    if not isinstance(text_outputs, dict):
        return None
    for dependency in reversed(step.depends_on or []):
        record = text_outputs.get(dependency)
        if not isinstance(record, dict):
            continue
        target_text = record.get("target_text")
        if isinstance(target_text, str) and target_text.strip():
            return target_text.strip()
    return None


def _current_text_edit_text(state: dict[str, Any]) -> str | None:
    if state.get("current_text_source") != "edit_data_pipeline.text_edit":
        return None
    current_text = state.get("current_text")
    if isinstance(current_text, str) and current_text.strip():
        return current_text.strip()
    return None


def _perception_report_with_text_override(
    perception_report: dict[str, Any],
    text: str,
    *,
    source: str,
) -> dict[str, Any]:
    normalized_text = text.strip()
    if not normalized_text:
        return perception_report
    updated = copy.deepcopy(perception_report) if isinstance(perception_report, dict) else {}
    updated["source_text"] = normalized_text
    updated["text_override_source"] = source
    asr = updated.get("asr")
    if isinstance(asr, dict):
        asr["text"] = normalized_text
    for key in ("word_timestamps", "words", "alignment", "alignments"):
        updated.pop(key, None)
    for key in ("speaker_segments", "segments"):
        segment = _merged_text_override_segment(updated.get(key), normalized_text)
        if segment:
            updated[key] = [segment]
        else:
            updated.pop(key, None)
    return updated


def _merged_text_override_segment(raw_segments: Any, text: str) -> dict[str, Any] | None:
    if not isinstance(raw_segments, list):
        return None
    starts: list[float] = []
    ends: list[float] = []
    speaker_ids: set[str] = set()
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start_sec", item.get("start")))
        end = _safe_float(item.get("end_sec", item.get("end")))
        if start is None or end is None or end <= start:
            continue
        starts.append(start)
        ends.append(end)
        speaker_id = item.get("speaker_id")
        if speaker_id is not None:
            speaker_ids.add(str(speaker_id))
    if not starts or not ends:
        return None
    segment: dict[str, Any] = {
        "start_sec": min(starts),
        "end_sec": max(ends),
        "text": text,
        "transcript": text,
        "source": "text_edit_target_text_override",
    }
    if len(speaker_ids) == 1:
        segment["speaker_id"] = next(iter(speaker_ids))
    return segment


def _perception_report_for_audio_path(
    state: dict[str, Any], input_path: Path
) -> dict[str, Any] | None:
    reports = state.get("perception_reports_by_audio_path")
    if not isinstance(reports, dict):
        return None
    keys = [str(input_path)]
    try:
        keys.append(str(input_path.expanduser().resolve()))
    except Exception:
        pass
    for key in keys:
        report = reports.get(key)
        if isinstance(report, dict):
            return report
    return None


def _index_namespaced_reviewed_perception(state: dict[str, Any]) -> None:
    """Index reviewed reports by the materialized audio they actually describe."""

    container = state.get("perception_report")
    if not isinstance(container, dict) or container.get("schema_version") != (
        "x-editor.reviewed-perception/by-audio-ref/v1"
    ):
        return
    entries = container.get("audio_reports")
    if not isinstance(entries, list):
        return
    state["_namespaced_reviewed_perception"] = True
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        audio_ref = str(entry.get("audio_ref_id") or "").strip()
        report = entry.get("report")
        if not audio_ref or not isinstance(report, dict) or not report:
            continue
        try:
            materialized = resolve_audio_reference(audio_ref, state)
        except AudioReferenceResolutionError:
            continue
        if not isinstance(materialized, str) or not materialized:
            continue
        _cache_perception_report_for_audio_path(
            state,
            Path(materialized),
            copy.deepcopy(report),
        )


def _activate_reviewed_perception_for_step(
    step: AudioEditStep,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Expose only evidence whose audio namespace matches this step's input."""

    if not state.get("_namespaced_reviewed_perception"):
        return None
    input_audio = _resolve_input_audio(step, state)
    input_path = Path(input_audio)
    report = _perception_report_for_audio_path(state, input_path)
    audio_ref = canonical_audio_ref_for_runtime_path(input_audio, state)
    if report is None:
        state["perception_report"] = {}
        state["perception_evidence_timebase"] = None
        return {
            "step_id": step.id,
            "operation": step.operation,
            "input_audio_ref": audio_ref,
            "status": "missing_matching_report",
        }
    state["perception_report"] = copy.deepcopy(report)
    evidence_artifact = _runtime_artifact_for_path(state, str(input_path))
    _bind_perception_evidence_timebase(
        state,
        state["perception_report"],
        evidence_artifact,
    )
    source_text = report.get("source_text")
    if isinstance(source_text, str) and source_text.strip():
        state["source_text"] = source_text.strip()
    return {
        "step_id": step.id,
        "operation": step.operation,
        "input_audio_ref": audio_ref,
        "report_audio_ref": report.get("audio_ref_id"),
        "status": "matched",
    }


def _cache_perception_report_for_audio_path(
    state: dict[str, Any], input_path: Path, perception: dict[str, Any]
) -> None:
    if not isinstance(perception, dict) or not perception:
        return
    reports = state.setdefault("perception_reports_by_audio_path", {})
    if not isinstance(reports, dict):
        return
    reports[str(input_path)] = perception
    try:
        reports[str(input_path.expanduser().resolve())] = perception
    except Exception:
        pass


def _has_explicit_span_params(params: dict[str, Any]) -> bool:
    for key in (
        "spans",
        "selected_spans",
        "time_spans",
        "segments",
        "segments_to_extract",
    ):
        value = params.get(key)
        if isinstance(value, (list, tuple)) and value:
            return True
    return any(params.get(key) is not None for key in ("start_sec", "end_sec", "start", "end"))


def _is_reference_extract_step(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    input_path: Path,
    instruction: str,
    params: dict[str, Any],
) -> bool:
    evidence = _reference_extract_evidence_text(step, instruction, params)
    wants_reference_clip = _has_any(
        evidence,
        [
            "reference audio",
            "reference voice",
            "reference clip",
            "reference speaker",
            "voice clone reference",
            "clone reference",
            "enrollment",
            "audio2",
            "audio 2",
            "input2",
            "input 2",
            "second audio",
            "参考音频",
            "参考声音",
            "参考片段",
            "注册音频",
        ],
    )
    return wants_reference_clip or _input_path_is_reference_or_secondary(input_path, state)


def _input_path_is_reference_or_secondary(input_path: Path, state: dict[str, Any]) -> bool:
    resolved = _resolved_path_key(input_path)
    for item in state.get("reference_audio_paths") or []:
        if resolved == _resolved_path_key(item):
            return True
    for item in state.get("reference_audio_items") or []:
        if not isinstance(item, dict):
            continue
        for key in ("path", "audio_path", "reference_audio_path"):
            value = item.get(key)
            if value and resolved == _resolved_path_key(value):
                return True
    input_audios = collect_input_audios(state)
    for index, item in enumerate(input_audios):
        if index > 0 and resolved == _resolved_path_key(item):
            return True
    return False


def _reference_extract_evidence_text(
    step: AudioEditStep, instruction: str, params: dict[str, Any]
) -> str:
    values: list[Any] = [
        instruction,
        step.description,
        params.get("selection_instruction"),
        params.get("selector"),
        params.get("target_description"),
        params.get("target_speaker_description"),
        params.get("speaker_selector"),
        params.get("reference_selector"),
        step.inputs.get("input_audio"),
        step.inputs.get("input_audio_path"),
    ]
    return " ".join(str(value).lower() for value in values if value not in (None, "", [], {}))


def _reference_extract_needs_speaker_review(evidence: str) -> bool:
    speech_markers = [
        "speaker",
        "voice",
        "speech",
        "spoken",
        "utterance",
        "question",
        "answer",
        "phrase",
        "word",
        "sentence",
        "transcript",
        "pronunciation",
        "accent",
        "dialect",
        "language",
        "male",
        "female",
        "woman",
        "man",
        "child",
        "kid",
        "boy",
        "girl",
        "elderly",
        "role",
        "customer",
        "respondent",
        "interviewer",
        "narrator",
        "singing voice",
        "vocal",
        "说话",
        "说话人",
        "人声",
        "语音",
        "声音",
        "男声",
        "女声",
        "男人",
        "女人",
        "儿童",
        "口音",
        "方言",
    ]
    return _has_any(evidence, speech_markers) or bool(_quoted_phrases(evidence))


def _reference_extract_needs_timestamp_caption(evidence: str) -> bool:
    event_markers = [
        "sound event",
        "audio event",
        "caption",
        "background",
        "music",
        "noise",
        "environment",
        "ambient",
        "room",
        "reverb",
        "echo",
        "applause",
        "laughter",
        "laughing",
        "crying",
        "scream",
        "shout",
        "whisper",
        "telephone",
        "phone",
        "radio",
        "underwater",
        "wind",
        "muffled",
        "rain",
        "traffic",
        "engine",
        "siren",
        "beep",
        "tone",
        "事件",
        "背景",
        "音乐",
        "噪声",
        "环境",
        "混响",
        "回声",
        "电话",
        "广播",
        "水下",
        "风声",
        "雨声",
    ]
    if not _has_any(evidence, event_markers):
        return False
    speech_only_markers = [
        "phrase",
        "word",
        "sentence",
        "question",
        "answer",
        "speaker",
        "说话人",
    ]
    return (
        True
        if not _has_any(evidence, speech_only_markers)
        else _has_any(
            evidence,
            [
                "background",
                "music",
                "noise",
                "environment",
                "reverb",
                "echo",
                "事件",
                "背景",
                "音乐",
                "噪声",
                "环境",
            ],
        )
    )


def _reference_event_labels(params: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "events",
        "event_labels",
        "event_label",
        "target_events",
        "target_event",
        "event_list",
    ):
        raw = params.get(key)
        if isinstance(raw, str):
            values.extend(item.strip() for item in re.split(r"[,\n;]+", raw) if item.strip())
        elif isinstance(raw, (list, tuple, set)):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _speaker_review_selected_spans(
    *,
    perception: dict[str, Any],
    instruction: str,
    audio_path: Path,
) -> list[dict[str, Any]]:
    duration = _reference_grounding_duration(perception, audio_path)
    if duration is None or duration <= 0:
        return []
    spans = select_timeline_spans(
        instruction, perception, duration_sec=duration, action="extract_spans"
    )
    if spans:
        return _dedupe_span_dicts(
            _span_dicts_from_time_spans(spans, source_suffix="speaker_attributed_review")
        )
    # An observed speaker inventory does not resolve the requested speaker.
    # Leave observations available for reasoning or another perception request;
    # never promote every segment merely because the request mentions a voice.
    return []


def _reviewed_speaker_selection(
    *,
    perception: dict[str, Any],
    instruction: str,
    audio_path: Path,
) -> list[dict[str, Any]]:
    raw_spans = _speaker_review_selected_spans(
        perception=perception,
        instruction=instruction,
        audio_path=audio_path,
    )
    reviewed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_spans):
        reviewed.append(
            {
                "span_id": f"speaker_span_{index:03d}",
                "audio_ref_id": str(audio_path),
                "start_sec": raw.get("start_sec"),
                "end_sec": raw.get("end_sec"),
                "speaker_id": raw.get("speaker_id"),
                "text": raw.get("text") or raw.get("transcript") or "",
                "label": raw.get("label") or "speaker_target",
                "semantic_role": "edit_target",
                "target_id": "target_000",
                "source_refs": [
                    str(raw.get("segment_id") or raw.get("source") or "speaker_attributed_review")
                ],
                "confidence": raw.get("confidence"),
                "reason": "resolved from speaker-attributed evidence",
                "source": raw.get("source") or "speaker_attributed_review",
            }
        )
    return reviewed


def _reference_grounding_duration(perception: dict[str, Any], audio_path: Path) -> float | None:
    duration = _audio_duration_seconds(audio_path)
    if duration is not None:
        return duration
    candidates: list[float] = []
    for key in (
        "speaker_segments",
        "segments",
        "audio_caption_segments",
        "event_segments",
    ):
        raw = perception.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            end = _safe_float(item.get("end_sec", item.get("end")))
            if end is not None:
                candidates.append(end)
    return max(candidates) if candidates else None


def _span_dicts_from_time_spans(spans: list[Any], *, source_suffix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for span in spans:
        if hasattr(span, "to_dict"):
            item = span.to_dict()
        else:
            item = dict(span) if isinstance(span, dict) else {}
        start = _safe_float(item.get("start_sec", item.get("start")))
        end = _safe_float(item.get("end_sec", item.get("end")))
        if start is None or end is None or end <= start:
            continue
        source = str(item.get("source") or "").strip()
        item["start_sec"] = float(start)
        item["end_sec"] = float(end)
        item["source"] = f"{source}+{source_suffix}" if source else source_suffix
        result.append(_without_empty(item))
    return result


def _span_dicts_from_segment_dicts(
    segments: list[Any],
    duration_sec: float,
    *,
    source_suffix: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start_sec", item.get("start")))
        end = _safe_float(item.get("end_sec", item.get("end")))
        if start is None or end is None:
            continue
        start = max(0.0, min(start, duration_sec))
        end = max(0.0, min(end, duration_sec))
        if end <= start:
            continue
        result.append(
            _without_empty(
                {
                    "start_sec": float(start),
                    "end_sec": float(end),
                    "duration_sec": round(end - start, 6),
                    "label": item.get("label") or item.get("description"),
                    "speaker_id": item.get("speaker_id"),
                    "text": item.get("text") or item.get("transcript"),
                    "source": source_suffix,
                    "reason": item.get("description"),
                }
            )
        )
    return result


def _merge_reference_perception_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        for key, value in report.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                current = merged.setdefault(key, [])
                if isinstance(current, list):
                    current.extend(item for item in value if item not in current)
                else:
                    merged[key] = list(value)
            elif isinstance(value, dict):
                current = merged.get(key)
                if isinstance(current, dict):
                    merged[key] = {**current, **value}
                else:
                    merged[key] = dict(value)
            elif key not in merged:
                merged[key] = value
    return merged


def _combine_reference_grounded_spans(
    selected_by_operation: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    span_lists = [
        _dedupe_span_dicts(value)
        for value in selected_by_operation.values()
        if isinstance(value, list) and value
    ]
    if not span_lists:
        return []
    if len(span_lists) == 1:
        return span_lists[0]
    combined = span_lists[0]
    for next_spans in span_lists[1:]:
        combined = _intersect_span_dict_lists(combined, next_spans)
        if not combined:
            return []
    return _dedupe_span_dicts(combined)


def _intersect_span_dict_lists(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    intersections: list[dict[str, Any]] = []
    for left_span in left:
        left_start = _safe_float(left_span.get("start_sec"))
        left_end = _safe_float(left_span.get("end_sec"))
        if left_start is None or left_end is None:
            continue
        for right_span in right:
            right_start = _safe_float(right_span.get("start_sec"))
            right_end = _safe_float(right_span.get("end_sec"))
            if right_start is None or right_end is None:
                continue
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end <= start:
                continue
            intersections.append(
                _without_empty(
                    {
                        "start_sec": float(start),
                        "end_sec": float(end),
                        "duration_sec": round(end - start, 6),
                        "speaker_id": left_span.get("speaker_id") or right_span.get("speaker_id"),
                        "text": left_span.get("text") or left_span.get("transcript"),
                        "event": right_span.get("event"),
                        "caption": right_span.get("caption"),
                        "label": left_span.get("label") or right_span.get("label"),
                        "reason": "intersection of speaker-aware reference evidence and timestamped caption evidence",
                        "source": "reference_grounding_intersection",
                    }
                )
            )
    return intersections


def _dedupe_span_dicts(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str, str]] = set()
    for item in spans:
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start_sec"))
        end = _safe_float(item.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        key = (
            round(start, 3),
            round(end, 3),
            str(item.get("speaker_id") or ""),
            str(item.get("event") or item.get("label") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        clean = dict(item)
        clean["start_sec"] = float(start)
        clean["end_sec"] = float(end)
        clean["duration_sec"] = round(end - start, 6)
        deduped.append(_without_empty(clean))
    deduped.sort(
        key=lambda span: (
            float(span.get("start_sec", 0.0)),
            float(span.get("end_sec", 0.0)),
        )
    )
    return deduped


def _mark_failed(step: AudioEditStep, error: str) -> None:
    step.status = "failed"
    step.error = error


def _mark_skipped(step: AudioEditStep, error: str) -> None:
    step.status = "skipped"
    step.error = error


def _continue_on_error(step: AudioEditStep, state: dict[str, Any]) -> bool:
    if step.continue_on_error is not None:
        return step.continue_on_error
    return bool(
        state["context"].get("continue_on_error")
        or state["plan"].global_params.get("continue_on_error")
    )


def _operation_graph_execution_enabled(state: dict[str, Any]) -> bool:
    context = state.get("context") or {}
    plan = state.get("plan")
    if "enable_operation_graph_execution" in context:
        return _context_bool(context, "enable_operation_graph_execution", default=False)
    if plan is not None and "enable_operation_graph_execution" in plan.global_params:
        return bool(plan.global_params.get("enable_operation_graph_execution"))
    return _env_bool("X_EDITOR_ENABLE_OPERATION_GRAPH_EXECUTION", default=False)


def _operation_graph_execution_round_index(
    plan: AudioEditPlan | None, state: dict[str, Any]
) -> int:
    context = state.get("context") or {}
    context_session_graph = (
        context.get("session_graph") if isinstance(context.get("session_graph"), dict) else {}
    )
    candidates = [
        context.get("operation_graph_round_index"),
        context.get("edit_session_round_index"),
        context_session_graph.get("round_index"),
    ]
    if plan is not None:
        plan_session_graph = (
            plan.metadata.get("session_graph")
            if isinstance(plan.metadata.get("session_graph"), dict)
            else {}
        )
        candidates.extend(
            [
                plan.metadata.get("round_index"),
                plan_session_graph.get("round_index"),
                plan.global_params.get("operation_graph_round_index"),
                plan.global_params.get("edit_session_round_index"),
            ]
        )
    for value in candidates:
        try:
            if value is not None and value != "":
                return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return 1


def _operation_graph_step_order(graph: Any, plan: AudioEditPlan) -> list[str]:
    plan_step_ids = [step.id for step in plan.steps]
    known = set(plan_step_ids)
    ordered: list[str] = []
    for operation in getattr(graph, "operations", []) or []:
        step_id = str(getattr(operation, "step_id", "") or "")
        if step_id in known and step_id not in ordered:
            ordered.append(step_id)
    for step_id in plan_step_ids:
        if step_id not in ordered:
            ordered.append(step_id)
    return ordered


def _operation_graph_operations_by_step(graph: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for operation in getattr(graph, "operations", []) or []:
        step_id = str(getattr(operation, "step_id", "") or "")
        if step_id and step_id not in rows:
            rows[step_id] = operation
    return rows


def _operation_graph_step_dependencies(
    graph: Any, by_id: dict[str, AudioEditStep]
) -> dict[str, list[str]]:
    node_producers: dict[str, str] = {}
    for node in getattr(graph, "nodes", []) or []:
        node_id = str(getattr(node, "node_id", "") or "")
        step_id = str(getattr(node, "step_id", "") or "")
        if node_id and step_id:
            node_producers.setdefault(node_id, step_id)
    for operation in getattr(graph, "operations", []) or []:
        step_id = str(getattr(operation, "step_id", "") or "")
        for node_id in _string_values(getattr(operation, "output_node_ids", []) or []):
            if step_id:
                node_producers[node_id] = step_id

    dependencies: dict[str, list[str]] = {}
    for operation in getattr(graph, "operations", []) or []:
        step_id = str(getattr(operation, "step_id", "") or "")
        if step_id not in by_id:
            continue
        deps = []
        for node_id in _string_values(getattr(operation, "input_node_ids", []) or []):
            producer = node_producers.get(node_id)
            if producer and producer != step_id:
                deps.append(producer)
        dependencies[step_id] = _dedupe_string_list(deps)
    return dependencies


def _operation_graph_round_execution_plan(store: Any) -> dict[str, Any]:
    rounds = list(getattr(store, "rounds", []) or [])
    if not rounds:
        return {
            "round_count": 0,
            "state_edges": [],
            "step_state_outputs": {},
            "state_step_dependencies": {},
        }
    round_record = rounds[0]
    state_nodes = list(getattr(round_record, "state_nodes", []) or [])
    state_by_id = {
        str(getattr(state, "state_node_id", "") or ""): state
        for state in state_nodes
        if getattr(state, "state_node_id", None)
    }
    state_edges: list[dict[str, str]] = []
    step_state_outputs: dict[str, list[str]] = {}
    state_step_dependencies: dict[str, list[str]] = {}
    source_state_id = str(getattr(round_record, "source_state_node_id", "") or "")
    target_state_id = str(getattr(round_record, "target_state_node_id", "") or "")

    for state in state_nodes:
        state_id = str(getattr(state, "state_node_id", "") or "")
        if not state_id:
            continue
        producer_step_id = str(getattr(state, "producer_step_id", "") or "")
        if producer_step_id:
            step_state_outputs.setdefault(producer_step_id, []).append(state_id)
        for parent_id in _string_values(getattr(state, "parent_state_node_ids", []) or []):
            state_edges.append({"from_state_node_id": parent_id, "to_state_node_id": state_id})
            parent = state_by_id.get(parent_id)
            parent_step_id = (
                str(getattr(parent, "producer_step_id", "") or "") if parent is not None else ""
            )
            if producer_step_id and parent_step_id and parent_step_id != producer_step_id:
                state_step_dependencies.setdefault(producer_step_id, []).append(parent_step_id)

    return {
        "round_count": len(rounds),
        "round_index": getattr(round_record, "round_index", None),
        "source_state_node_id": source_state_id,
        "target_state_node_id": target_state_id,
        "state_node_count": len(state_nodes),
        "audio_node_count": len(getattr(round_record, "audio_nodes", []) or []),
        "operation_count": len(getattr(round_record, "operations", []) or []),
        "state_edges": state_edges,
        "step_state_outputs": {
            step_id: _dedupe_string_list(state_ids)
            for step_id, state_ids in sorted(step_state_outputs.items())
        },
        "state_step_dependencies": {
            step_id: _dedupe_string_list(deps)
            for step_id, deps in sorted(state_step_dependencies.items())
        },
    }


def _operation_graph_round_step_dependencies(
    round_execution_plan: dict[str, Any],
    by_id: dict[str, AudioEditStep],
) -> dict[str, list[str]]:
    dependencies: dict[str, list[str]] = {}
    for step_id, deps in (round_execution_plan.get("state_step_dependencies") or {}).items():
        if step_id in by_id:
            dependencies[str(step_id)] = [
                dep for dep in _dedupe_string_list(deps) if dep in by_id and dep != step_id
            ]
    return dependencies


def _operation_graph_merge_step_dependencies(
    *dependency_sets: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for dependencies in dependency_sets:
        for step_id, deps in dependencies.items():
            merged[step_id] = _dedupe_string_list([*(merged.get(step_id) or []), *deps])
    return dict(sorted(merged.items()))


def _operation_graph_required_step_dependencies(
    step: AudioEditStep,
    graph_dependencies: dict[str, list[str]],
) -> list[str]:
    return _dedupe_string_list([*(graph_dependencies.get(step.id) or []), *step.depends_on])


def _operation_graph_initial_node_paths(graph: Any, state: dict[str, Any]) -> dict[str, str]:
    node_paths: dict[str, str] = {}
    nodes_by_id = _operation_graph_nodes_by_id(graph)
    input_paths = _dedupe_string_list(
        [
            state.get("input_audio"),
            state.get("current_audio_path"),
            *collect_input_audios(state),
        ]
    )
    for index, node_id in enumerate(_string_values(getattr(graph, "input_node_ids", []) or [])):
        node = nodes_by_id.get(node_id)
        path = _operation_graph_resolve_existing_path(
            getattr(node, "path", None) if node else None, state
        )
        if not path and index < len(input_paths):
            path = (
                _operation_graph_resolve_existing_path(input_paths[index], state)
                or input_paths[index]
            )
        if path:
            node_paths[node_id] = path
    for node in getattr(graph, "nodes", []) or []:
        node_id = str(getattr(node, "node_id", "") or "")
        if not node_id or node_id in node_paths:
            continue
        kind = str(getattr(node, "kind", "") or "")
        if kind != "input" and getattr(node, "parent_node_ids", None):
            continue
        path = _operation_graph_resolve_existing_path(getattr(node, "path", None), state)
        if path:
            node_paths[node_id] = path
    return node_paths


def _bind_step_inputs_from_operation_graph(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    planned_graph: Any,
    step_ops: dict[str, Any],
    node_paths: dict[str, str],
) -> dict[str, Any] | None:
    operation = step_ops.get(step.id)
    if operation is None:
        return None
    input_node_ids = _string_values(getattr(operation, "input_node_ids", []) or [])
    if not input_node_ids:
        return None
    nodes_by_id = _operation_graph_nodes_by_id(planned_graph)
    parent_paths: list[str] = []
    missing_node_paths: list[str] = []
    for node_id in input_node_ids:
        path = node_paths.get(node_id)
        if not path:
            node = nodes_by_id.get(node_id)
            path = _operation_graph_resolve_existing_path(
                getattr(node, "path", None) if node else None, state
            )
            if path:
                node_paths[node_id] = path
        if path:
            parent_paths.append(path)
        else:
            missing_node_paths.append(node_id)
    parent_paths = _dedupe_string_list(parent_paths)
    if not parent_paths:
        return {
            "step_id": step.id,
            "status": "missing_parent_paths",
            "input_node_ids": input_node_ids,
            "missing_node_paths": missing_node_paths,
        }

    before_inputs = dict(step.inputs)
    before_params = dict(step.params)
    current_before = state.get("current_audio_path")
    state["current_audio_path"] = parent_paths[0]
    primary_changed = _set_operation_graph_primary_input(step, parent_paths[0])
    multi_changed = False
    if len(parent_paths) > 1:
        multi_changed = _set_operation_graph_multi_input(
            step,
            parent_paths,
            force=_operation_graph_should_force_multi_input(step.operation),
        )
    if not (primary_changed or multi_changed or current_before != state.get("current_audio_path")):
        return None
    return _without_empty(
        {
            "step_id": step.id,
            "input_node_ids": input_node_ids,
            "parent_paths": parent_paths,
            "missing_node_paths": missing_node_paths,
            "current_audio_before": current_before,
            "current_audio_after": state.get("current_audio_path"),
            "inputs_before": before_inputs if before_inputs != step.inputs else None,
            "inputs_after": dict(step.inputs) if before_inputs != step.inputs else None,
            "params_before": before_params if before_params != step.params else None,
            "params_after": dict(step.params) if before_params != step.params else None,
        }
    )


def _update_operation_graph_node_paths_after_step(
    step: AudioEditStep,
    *,
    planned_graph: Any,
    step_ops: dict[str, Any],
    node_paths: dict[str, str],
) -> list[dict[str, Any]]:
    operation = step_ops.get(step.id)
    if operation is None or not isinstance(step.result, dict):
        return []
    nodes_by_id = _operation_graph_nodes_by_id(planned_graph)
    output_node_ids = _string_values(getattr(operation, "output_node_ids", []) or [])
    all_output_paths = _operation_graph_all_output_paths(step.result)
    updates: list[dict[str, Any]] = []
    for index, node_id in enumerate(output_node_ids):
        node = nodes_by_id.get(node_id)
        path = _operation_graph_output_path_for_node(step.result, node, fallback_index=index)
        if not path and len(output_node_ids) == 1:
            path = _extract_output_path(step.result)
        if not path and index < len(all_output_paths):
            path = all_output_paths[index][1]
        if not path:
            continue
        node_paths[node_id] = str(path)
        updates.append(
            _without_empty(
                {
                    "step_id": step.id,
                    "node_id": node_id,
                    "path": str(path),
                    "result_key": getattr(node, "result_key", None) if node is not None else None,
                    "role": getattr(node, "role", None) if node is not None else None,
                }
            )
        )
    return updates


def _operation_graph_nodes_by_id(graph: Any) -> dict[str, Any]:
    return {
        str(getattr(node, "node_id", "") or ""): node
        for node in getattr(graph, "nodes", []) or []
        if getattr(node, "node_id", None)
    }


_OPERATION_GRAPH_PRIMARY_INPUT_KEYS = (
    "input_audio_path",
    "input_audio",
    "audio_path",
    "audio",
    "source_audio_path",
    "source_audio",
    "mixture_audio_path",
    "mixture_audio",
)

_OPERATION_GRAPH_MULTI_INPUT_KEYS = (
    "audio_paths",
    "input_audio_paths",
    "input_audios",
    "sources",
)


def _set_operation_graph_primary_input(step: AudioEditStep, path: str) -> bool:
    for key in _OPERATION_GRAPH_PRIMARY_INPUT_KEYS:
        if key in step.inputs:
            if _operation_graph_value_is_replaceable(step.inputs.get(key)):
                step.inputs[key] = path
                return True
            return False
    for key in _OPERATION_GRAPH_PRIMARY_INPUT_KEYS:
        if key in step.params:
            if _operation_graph_value_is_replaceable(step.params.get(key)):
                step.params[key] = path
                return True
            return False
    step.inputs["input_audio_path"] = path
    return True


def _set_operation_graph_multi_input(step: AudioEditStep, paths: list[str], *, force: bool) -> bool:
    for mapping in (step.inputs, step.params):
        for key in _OPERATION_GRAPH_MULTI_INPUT_KEYS:
            if key not in mapping:
                continue
            if force or _operation_graph_value_is_replaceable(mapping.get(key)):
                mapping[key] = list(paths)
                return True
            return False
    if force:
        step.inputs["audio_paths"] = list(paths)
        return True
    return False


def _operation_graph_should_force_multi_input(operation: str) -> bool:
    lowered = str(operation or "").lower()
    return any(token in lowered for token in ("concat", "reconstruct", "merge", "mix", "remix"))


def _operation_graph_value_is_replaceable(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, list):
        return not value or all(_operation_graph_value_is_replaceable(item) for item in value)
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if lowered in {
        "current",
        "current_audio",
        "current_audio_path",
        "previous",
        "previous_audio",
        "input",
        "input_audio",
        "input_audio_path",
        "base",
        "base_audio",
        "source",
        "source_audio",
        "source_audio_path",
        "session_current",
    }:
        return True
    return lowered.startswith("session_current") or "{{current" in lowered


def _operation_graph_resolve_existing_path(value: Any, state: dict[str, Any]) -> str | None:
    if value in (None, ""):
        return None
    resolved = resolve_audio_reference(value, state)
    if isinstance(resolved, list):
        resolved = next((item for item in resolved if isinstance(item, str) and item), None)
    if not isinstance(resolved, str) or not resolved:
        return None
    path = Path(str(resolved))
    return str(path) if path.exists() else None


def _operation_graph_output_path_for_node(
    output: dict[str, Any], node: Any, *, fallback_index: int
) -> str | None:
    if node is None:
        return None
    result_key = str(getattr(node, "result_key", "") or "")
    role = str(getattr(node, "role", "") or "")
    split_index = getattr(node, "split_index", None)
    for key in (result_key, role):
        if not key:
            continue
        path = _operation_graph_output_path_for_key(output, key, split_index=split_index)
        if path:
            return path
    all_paths = _operation_graph_all_output_paths(output)
    if fallback_index < len(all_paths):
        return all_paths[fallback_index][1]
    return None


def _operation_graph_output_path_for_key(
    output: dict[str, Any],
    key: str,
    *,
    split_index: Any,
) -> str | None:
    candidates = [
        path
        for key_path, path in _iter_audio_path_values(output)
        if key_path == key
        or key_path.endswith(f".{key}")
        or _audio_artifact_key_leaf(key_path) == _audio_artifact_key_leaf(key)
    ]
    if not candidates:
        return None
    try:
        index = int(split_index) - 1 if split_index is not None else 0
    except (TypeError, ValueError):
        index = 0
    if 0 <= index < len(candidates):
        return candidates[index]
    return candidates[0]


def _operation_graph_all_output_paths(output: dict[str, Any]) -> list[tuple[str, str]]:
    return [(key_path, path) for key_path, path in _iter_audio_path_values(output)]


def _string_values(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(item) for item in values if item not in (None, "")]
    if values in (None, ""):
        return []
    return [str(values)]


def _dedupe_string_list(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _emit_execution_observation(
    state: dict[str, Any],
    *,
    stage: str,
    kind: str,
    status: str,
    step: AudioEditStep,
    payload: dict[str, Any],
) -> None:
    """Publish a one-way fact while keeping observer failures non-authoritative."""

    observer = state.get("execution_observer")
    if observer is None:
        return
    try:
        observer.emit(
            stage=stage,
            kind=kind,
            status=status,
            attempt_id=f"step:{step.id}",
            subject_refs=[step.id],
            payload=_json_safe_observation_value(canonicalize_runtime_audio_paths(payload, state)),
        )
    except Exception as exc:
        state.setdefault("execution_observer_errors", []).append(
            {
                "step_id": step.id,
                "stage": stage,
                "kind": kind,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )


def _json_safe_observation_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return {"non_serializable_type": type(value).__name__}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_observation_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_observation_value(item, depth=depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe_observation_value(
                value.model_dump(mode="python"),
                depth=depth + 1,
            )
        except Exception:
            pass
    return {"non_serializable_type": (f"{type(value).__module__}.{type(value).__qualname__}")}


def _workflow_status_for_step(step: AudioEditStep) -> str:
    if step.status == "failed":
        return "failed"
    if step.status == "running":
        return "started"
    return "completed"


def _safe_perception_backend_attempts(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep route accounting without persisting audio paths or ASR content."""

    safe: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        item = {
            key: attempt.get(key)
            for key in (
                "provider",
                "status",
                "audio_index",
                "language",
                "refresh_cache",
            )
            if isinstance(attempt.get(key), (str, int, float, bool))
        }
        if item:
            safe.append(item)
    return safe


def _review_planned_perception_output(
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    if not step.operation.startswith(("perception.", "flexsed.")):
        return output
    context = state.get("context") or {}
    if context.get("_forward_planned_perception_evidence_mode") != "reviewed":
        return output
    output = copy.deepcopy(output)
    raw_trace = output.get("component_invocation_trace")
    if raw_trace is not None:
        try:
            trace = validate_component_invocation_trace(raw_trace)
        except Exception:
            output.pop("component_invocation_trace", None)
            output["component_invocation_trace_invalid"] = True
        else:
            output["component_invocation_trace"] = trace.model_dump(mode="json")
    audio_refs = collect_consumed_audio_paths(
        step=step,
        state=state,
        output={},
    )
    if opaque_audio_resolution_enabled(state):
        audio_refs = [
            audio_ref
            for audio_ref in (
                canonical_audio_ref_for_runtime_path(path, state) for path in audio_refs
            )
            if audio_ref is not None
        ]
    return sanitize_reviewed_perception_result(
        output,
        audio_refs=audio_refs,
    )


def _planned_perception_invocation_report(
    step: AudioEditStep,
    state: dict[str, Any],
) -> RuntimeVerificationReport | None:
    if not step.operation.startswith(("perception.", "flexsed.")):
        return None
    context = state.get("context") or {}
    raw_policy = context.get("_forward_planned_perception_invocation_policy")
    if raw_policy is None:
        return None
    try:
        policy = ComponentInvocationAdmissionPolicy.model_validate(raw_policy)
    except Exception as exc:
        return RuntimeVerificationReport(
            component="workflow.component_invocation_trace",
            step_id=step.id,
            operation=step.operation,
            passed=False,
            confidence="high",
            severity="blocking",
            findings=[
                "perception_invocation_policy_invalid",
                f"{type(exc).__name__}: {exc}",
            ],
        )

    result = step.result if isinstance(step.result, dict) else {}
    raw_trace = result.get("component_invocation_trace")
    trace: ComponentInvocationTrace | None = None
    trace_error: Exception | None = None
    if result.get("component_invocation_trace_invalid") is True:
        trace_error = ValueError("planned perception trace is invalid")
    elif raw_trace is not None:
        try:
            trace = validate_component_invocation_trace(raw_trace)
        except Exception as exc:
            trace_error = exc
    gateway_call_id = f"planned_perception:{step.id}"
    if trace is not None:
        if trace.gateway_call_id is not None and trace.gateway_call_id != gateway_call_id:
            trace_error = ValueError("planned perception trace is bound to a different step")
        else:
            trace = trace.model_copy(
                update={"gateway_call_id": gateway_call_id},
                deep=True,
            )
            result["component_invocation_trace"] = trace.model_dump(mode="json")

    if trace_error is not None:
        blockers = ["perception_invocation_trace_invalid"]
    else:
        blockers = component_invocation_trace_blockers(
            policy,
            trace,
            component="perception",
            gateway_call_id=gateway_call_id,
        )
    metrics = (
        {
            "logical_provider_call_count": trace.logical_provider_call_count,
            "provider_call_count": trace.provider_call_count,
            "retry_count": trace.retry_count,
            "logical_retry_count": trace.logical_retry_count,
            "json_repair_count": trace.json_repair_count,
            "schema_repair_count": trace.schema_repair_count,
            "fallback_count": len(trace.fallback_events),
        }
        if trace is not None
        else {}
    )
    return RuntimeVerificationReport(
        component="workflow.component_invocation_trace",
        step_id=step.id,
        operation=step.operation,
        passed=not blockers,
        confidence="high",
        severity="blocking" if blockers else "info",
        findings=blockers or ["planned_perception_invocation_trace_accepted"],
        metrics=metrics,
    )


def _implicit_executor_repair_enabled(state: dict[str, Any]) -> bool:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    return _context_bool(context, "allow_implicit_executor_repair", default=True)


def _runtime_verification_enabled(state: dict[str, Any]) -> bool:
    context = state.get("context") or {}
    plan = state.get("plan")
    if "enable_runtime_verification" in context:
        return _context_bool(context, "enable_runtime_verification", default=True)
    if plan is not None and "enable_runtime_verification" in plan.global_params:
        return bool(plan.global_params.get("enable_runtime_verification"))
    return _env_bool("X_EDITOR_ENABLE_RUNTIME_VERIFICATION", default=True)


def _tool_contract_verification_enabled(state: dict[str, Any]) -> bool:
    context = state.get("context") or {}
    plan = state.get("plan")
    if "enable_tool_contract_verification" in context:
        return _context_bool(context, "enable_tool_contract_verification", default=True)
    if plan is not None and "enable_tool_contract_verification" in plan.global_params:
        return bool(plan.global_params.get("enable_tool_contract_verification"))
    return _env_bool("X_EDITOR_ENABLE_TOOL_CONTRACT_VERIFICATION", default=True)


def _argument_repair_enforced(state: dict[str, Any]) -> bool:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    return repair_action_is_enforced(context, "repair_step_arguments")


def _executor_argument_repair_enabled(executor: AudioEditExecutor, state: dict[str, Any]) -> bool:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    return _context_bool(
        context,
        "use_executor_tool_call_planner",
        default=_context_bool(
            context,
            "use_gemini_executor",
            default=executor.enable_gemini_executor,
        ),
    ) and _context_bool(context, "executor_llm_repair", default=True)


def _runtime_verification_strict(state: dict[str, Any]) -> bool:
    context = state.get("context") or {}
    plan = state.get("plan")
    if "runtime_verification_strict" in context:
        return _context_bool(context, "runtime_verification_strict", default=False)
    if plan is not None and "runtime_verification_strict" in plan.global_params:
        return bool(plan.global_params.get("runtime_verification_strict"))
    return _env_bool("X_EDITOR_VERIFY_STRICT", default=False)


def _bounded_repair_enabled(state: dict[str, Any]) -> bool:
    context = state.get("context") or {}
    plan = state.get("plan")
    if "enable_bounded_repair" in context:
        return _context_bool(context, "enable_bounded_repair", default=True)
    if plan is not None and "enable_bounded_repair" in plan.global_params:
        return bool(plan.global_params.get("enable_bounded_repair"))
    return _env_bool("X_EDITOR_ENABLE_BOUNDED_REPAIR", default=True)


def _bounded_repair_max_micro_actions(state: dict[str, Any]) -> int:
    context = state.get("context") or {}
    plan = state.get("plan")
    if "max_bounded_micro_repairs" in context:
        return max(0, int(context["max_bounded_micro_repairs"]))
    if plan is not None and "max_bounded_micro_repairs" in plan.global_params:
        return max(0, int(plan.global_params["max_bounded_micro_repairs"]))
    return max(0, int(os.getenv("X_EDITOR_BOUNDED_REPAIR_MAX_MICRO_ACTIONS", "2")))


def _reports_for_components(
    reports: list[RuntimeVerificationReport],
    components: set[str],
) -> list[RuntimeVerificationReport]:
    return [report for report in reports if report.component in components]


def _replace_reports_for_components(
    current: list[RuntimeVerificationReport],
    replacement: list[RuntimeVerificationReport],
    components: set[str],
) -> list[RuntimeVerificationReport]:
    return [report for report in current if report.component not in components] + list(replacement)


def _aggregate_verification_score(reports: list[RuntimeVerificationReport]) -> float:
    if not reports:
        return 0.0
    values = []
    for report in reports:
        if report.score is not None:
            values.append(float(report.score))
        else:
            values.append(1.0 if report.passed else 0.0)
    blocking_penalty = 0.15 * sum(
        1 for report in reports if not report.passed and report.severity == "blocking"
    )
    return round(max(0.0, sum(values) / len(values) - blocking_penalty), 6)


def _repair_shifted_spans(
    spans_value: Any, action: RuntimeRepairAction
) -> list[dict[str, Any]] | None:
    if not isinstance(spans_value, list):
        return None
    spans = [dict(item) for item in spans_value if isinstance(item, dict)]
    if not spans:
        return None
    params = action.params
    try:
        index = int(params.get("span_index", 0))
        start = float(params["suggested_start_sec"])
        end = float(params["suggested_end_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    if index < 0 or index >= len(spans) or end <= start:
        return None
    spans[index].update(
        {
            "start_sec": start,
            "end_sec": end,
            "duration_sec": round(end - start, 6),
            "source": f"{spans[index].get('source') or 'span'}+bounded_repair_shift",
        }
    )
    return spans


def _repair_expanded_spans(
    spans_value: Any,
    action: RuntimeRepairAction,
    result: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not isinstance(spans_value, list):
        return None
    spans = [dict(item) for item in spans_value if isinstance(item, dict)]
    if not spans:
        return None
    before = float(action.params.get("expand_before_sec", 0.2))
    after = float(action.params.get("expand_after_sec", 0.2))
    duration = _audio_duration_seconds(
        Path(str(result.get("input_audio_path") or state.get("input_audio") or ""))
    )
    for span in spans:
        start = _safe_float(span.get("start_sec"))
        end = _safe_float(span.get("end_sec"))
        if start is None or end is None:
            return None
        start = max(0.0, start - before)
        end = end + after
        if duration is not None:
            end = min(duration, end)
        if end <= start:
            return None
        span.update(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 6),
                "source": f"{span.get('source') or 'span'}+bounded_repair_expand",
            }
        )
    return spans


def _repair_output_path(
    step: AudioEditStep, state: dict[str, Any], strategy: str, attempt_index: int
) -> str:
    safe_strategy = re.sub(r"[^A-Za-z0-9_.-]+", "_", strategy).strip("_") or "repair"
    safe_operation = step.operation.replace(".", "_").replace("/", "_")
    return str(
        Path(state["output_dir"])
        / "bounded_repair"
        / f"{step.id}_{safe_operation}_{attempt_index}_{safe_strategy}.wav"
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _context_bool(context: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in context:
        return default
    value = context[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _audio_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                return None
            return handle.getnframes() / float(frame_rate)
    except Exception:
        return None


def _repair_flexsed_window_args(
    args: dict[str, Any],
    *,
    input_audio: str,
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    has_start = args.get("start_sec") is not None
    has_end = args.get("end_sec") is not None
    if has_start == has_end:
        return

    default_window_sec = _env_float("X_EDITOR_FLEXSED_DEFAULT_WINDOW_SEC", 30.0)
    before = {key: args.get(key) for key in ("start_sec", "end_sec") if args.get(key) is not None}
    if has_start:
        start = _safe_float(args.get("start_sec"))
        if start is None:
            return
        duration = _audio_duration_seconds(Path(str(input_audio)))
        end = duration if duration is not None and duration > start else start + default_window_sec
        if end <= start:
            end = start + max(default_window_sec, 0.1)
        args["start_sec"] = float(start)
        args["end_sec"] = float(end)
        reason = "filled_missing_end_sec_from_audio_duration_or_default_window"
    else:
        end = _safe_float(args.get("end_sec"))
        if end is None:
            return
        args["start_sec"] = 0.0
        args["end_sec"] = float(end if end > 0 else max(default_window_sec, 0.1))
        reason = "filled_missing_start_sec_with_zero"

    state.setdefault("flexsed_window_repairs", {})[step.id] = {
        "step_id": step.id,
        "input_audio_path": input_audio,
        "before": before,
        "after": {"start_sec": args.get("start_sec"), "end_sec": args.get("end_sec")},
        "reason": reason,
    }


def _perception_input_differs_from_base(input_audio: str, state: dict[str, Any]) -> bool:
    try:
        input_path = Path(str(input_audio)).expanduser().resolve()
    except Exception:
        return False
    candidates = [
        state.get("source_text_audio_path"),
        state.get("input_audio"),
        state.get("input_audios", [None])[0]
        if isinstance(state.get("input_audios"), list)
        else None,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if input_path == Path(str(candidate)).expanduser().resolve():
                return False
        except Exception:
            continue
    return True


def _artifact_operation_needs_input_specific_perception(
    operation: str,
    instruction: str,
    params: dict[str, Any],
) -> bool:
    if operation not in {
        "extract_spans",
        "remove_spans",
        "mute_spans",
        "beep_spans",
        "speaker_scoped_edit",
    }:
        return False
    for key in ("spans", "selected_spans", "time_spans", "segments"):
        value = params.get(key)
        if isinstance(value, list) and value:
            return False
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            params.get("selector"),
            params.get("speaker_selector"),
            params.get("target_speaker_description"),
            params.get("target_description"),
        ]
        if value
    ).lower()
    if operation == "speaker_scoped_edit":
        return True
    return any(
        token in evidence
        for token in [
            "speaker",
            "voice",
            "speech",
            "male",
            "female",
            "woman",
            "man",
            "child",
            "kid",
            "girl",
            "boy",
            "phrase",
            "word",
            "sentence",
            "说话",
            "人声",
            "男",
            "女",
        ]
    )


def _perception_step_can_use_segment_grounding(instruction: str) -> bool:
    text = (instruction or "").lower()
    if not text.strip():
        return False
    if _quoted_phrases(instruction):
        return False
    exact_markers = [
        "word",
        "phrase",
        "term",
        "sentence",
        "entity",
        "proper noun",
        "name",
        "exact moment",
        "exact moments",
        "timestamp",
        "timestamps",
        "time stamp",
        "time stamps",
        "单词",
        "词语",
        "短语",
        "句子",
        "实体",
        "名字",
        "时间戳",
    ]
    if _has_any(text, exact_markers):
        return False
    segment_markers = [
        "speaker",
        "voice",
        "speech with",
        "speech by",
        "spoken by",
        "accent",
        "accented",
        "dialect",
        "male",
        "female",
        "woman",
        "women",
        "man",
        "men",
        "child",
        "kid",
        "boy",
        "girl",
        "elderly",
        "old person",
        "young person",
        "role",
        "character",
        "说话人",
        "人声",
        "声音",
        "口音",
        "方言",
        "男声",
        "女声",
        "男人",
        "女人",
        "儿童",
        "小孩",
        "老人",
        "角色",
    ]
    return _has_any(text, segment_markers)


def _requests_reference_channel_match(text: str | None) -> bool:
    lowered = (text or "").lower()
    return (
        "same acoustic channel" in lowered
        or "acoustic channel condition" in lowered
        or "same channel condition" in lowered
        or "reference channel" in lowered
        or "channel environment" in lowered
        or "recording characteristics" in lowered
        or "recording characteristic" in lowered
        or "recorded in the same acoustic" in lowered
        or "recorded with the same acoustic" in lowered
        or "match acoustic channel" in lowered
        or "match channel" in lowered
    )


def _is_nonhuman_voice_request(lowered: str) -> bool:
    animal_terms = [
        "animal",
        "non-human",
        "nonhuman",
        "cat",
        "kitten",
        "meow",
        "dog",
        "puppy",
        "bark",
        "鸟",
        "猫",
        "狗",
        "动物",
        "非人",
    ]
    voice_terms = [
        "voice",
        "timbre",
        "speak",
        "spoken",
        "say ",
        "saying",
        "words",
        "speech",
        "vocal",
        "音色",
        "声音",
        "说",
        "语音",
    ]
    return any(term in lowered for term in animal_terms) and any(
        term in lowered for term in voice_terms
    )


def _nonhuman_voice_best_effort_input_audio(request: str, state: dict[str, Any]) -> str | None:
    audios = [
        str(audio) for audio in collect_input_audios(state) if isinstance(audio, str) and audio
    ]
    lowered = str(request or "").lower()
    source_patterns = [
        (
            2,
            r"\b(?:words|content|speech|spoken words)\b.{0,100}\b(?:second audio|audio\s*2|input\s*2)\b",
        ),
        (
            2,
            r"\b(?:second audio|audio\s*2|input\s*2)\b.{0,100}\b(?:words|content|speech|spoken words)\b",
        ),
        (
            1,
            r"\b(?:words|content|speech|spoken words)\b.{0,100}\b(?:first audio|audio\s*1|input\s*1)\b",
        ),
        (
            1,
            r"\b(?:first audio|audio\s*1|input\s*1)\b.{0,100}\b(?:words|content|speech|spoken words)\b",
        ),
    ]
    for number, pattern in source_patterns:
        if re.search(pattern, lowered) and len(audios) >= number:
            return audios[number - 1]
    current = resolve_audio_reference(
        state.get("current_audio_path") or state.get("input_audio") or "current", state
    )
    if isinstance(current, str) and current:
        return current
    return audios[0] if audios else None


def _qwen3_tts_target_language(call: Any, target_text: str, request: str | None = None) -> str:
    values = []
    for mapping in (
        getattr(call, "arguments", {}) or {},
        getattr(call, "normalized_params", {}) or {},
    ):
        if isinstance(mapping, dict):
            values.extend(
                mapping.get(key) for key in ("target_language", "target_lang") if mapping.get(key)
            )
    for value in values:
        normalized = str(value).strip().lower().replace("_", "-")
        if normalized in {
            "zh",
            "zho",
            "cmn",
            "cn",
            "chinese",
            "mandarin",
            "中文",
            "汉语",
            "普通话",
        }:
            return "Chinese"
        if normalized in {"en", "eng", "english", "英语"}:
            return "English"
        if normalized in {"ko", "kor", "korean", "韩语", "韩国语"}:
            return "Korean"
        if normalized in {"ja", "jpn", "japanese", "日语", "日本语"}:
            return "Japanese"
        if normalized in {"es", "spa", "spanish", "西班牙语"}:
            return "Spanish"
        if normalized in {"fr", "fra", "fre", "french", "法语"}:
            return "French"
        if normalized in {"de", "deu", "ger", "german", "德语"}:
            return "German"
        if normalized in {"it", "ita", "italian", "意大利语"}:
            return "Italian"
        if normalized in {"pt", "por", "portuguese", "葡萄牙语"}:
            return "Portuguese"
        if normalized in {"ru", "rus", "russian", "俄语"}:
            return "Russian"
    return _qwen3_tts_language_from_text_and_request(target_text, request)


def _accent_plan_is_tts_generation(instruction: str, params: dict[str, Any]) -> bool:
    target_text = params.get("target_text") or params.get("text") or ""
    text = f"{instruction} {target_text}".lower()
    if _mentions_accent_or_dialect(text) and not any(
        marker in text for marker in ["pronunciation", "say", "saying"]
    ):
        return False
    return any(marker in text for marker in ["generate", "saying", "say ", "synthesize"]) and any(
        marker in text for marker in ["pronunciation", "voice", "speaker", "clone"]
    )


def _accent_plan_is_native_fluency_rewrite(instruction: str, params: dict[str, Any]) -> bool:
    text = f"{instruction} {params}".lower()
    target = str(params.get("target_accent") or params.get("accent") or "").lower()
    if _mentions_accent_or_dialect(text) and target not in {
        "native chinese",
        "standard chinese",
        "native mandarin",
    }:
        return False
    return any(
        marker in text
        for marker in [
            "fluent",
            "native chinese",
            "natural",
            "word-by-word",
            "unnatural",
            "grammatical",
        ]
    )


def _mentions_accent_or_dialect(text: str) -> bool:
    return "accent" in text or "dialect" in text or "口音" in text or "方言" in text


def _mentions_named_accent_or_dialect(text: str) -> bool:
    named = [
        "british",
        "american",
        "australian",
        "indian",
        "french accent",
        "thai accent",
        "cantonese",
        "sichuan",
        "shanghai",
        "suzhou",
        "wuyu",
        "dialect",
        "口音",
        "方言",
        "粤语",
        "四川",
        "上海",
        "苏州",
        "吴语",
    ]
    return any(marker in text for marker in named)


def _target_text_from_accent_params_or_instruction(
    params: dict[str, Any], instruction: str
) -> str | None:
    target = params.get("target_text") or params.get("text")
    if isinstance(target, str) and target.strip():
        return target.strip()
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"|“([^”]+)”", instruction)
    for groups in quoted:
        value = next((item for item in groups if item), "")
        if value and len(value.split()) >= 2:
            return value.strip()
    return None


def _speaker_selector_params_from_accent_params(
    params: dict[str, Any],
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in (
        "speaker_id",
        "target_speaker_id",
        "speaker_ids",
        "target_speaker_ids",
        "spans",
        "selector",
    ):
        value = params.get(key)
        if value is not None:
            selected[key] = value
    return selected


def _qwen3_tts_language_from_text_and_request(target_text: str, request: str | None) -> str:
    request_lower = (request or "").lower()
    if "english" in request_lower or "英语" in request_lower or "英文" in request_lower:
        return "English"
    if (
        "chinese" in request_lower
        or "mandarin" in request_lower
        or "中文" in request_lower
        or "普通话" in request_lower
    ):
        return "Chinese"
    if "japanese" in request_lower or "日语" in request_lower or "日本语" in request_lower:
        return "Japanese"
    if "korean" in request_lower or "韩语" in request_lower or "韩国语" in request_lower:
        return "Korean"
    if "spanish" in request_lower or "西班牙语" in request_lower:
        return "Spanish"
    if "french" in request_lower or "法语" in request_lower:
        return "French"
    if "german" in request_lower or "德语" in request_lower:
        return "German"
    if "italian" in request_lower or "意大利语" in request_lower:
        return "Italian"
    if "portuguese" in request_lower or "葡萄牙语" in request_lower:
        return "Portuguese"
    if "russian" in request_lower or "俄语" in request_lower:
        return "Russian"
    if any("\u4e00" <= char <= "\u9fff" for char in target_text):
        return "Chinese"
    if any("\u3040" <= char <= "\u30ff" for char in target_text):
        return "Japanese"
    if any("\uac00" <= char <= "\ud7af" for char in target_text):
        return "Korean"
    letters = [char for char in target_text if char.isalpha()]
    ascii_letters = [char for char in letters if char.isascii()]
    if letters and len(ascii_letters) / max(1, len(letters)) > 0.85:
        return "English"
    return "Auto"


def _step_requests_synthetic_silence(step: AudioEditStep) -> bool:
    if step.params.get("text"):
        return False
    text = " ".join(
        str(value)
        for value in [
            step.description,
            step.params.get("instruction"),
            step.params.get("instruct"),
        ]
        if value
    ).lower()
    return any(
        marker in text for marker in ["silence", "silent", "pause", "gap", "静音", "停顿"]
    ) and any(
        marker in text
        for marker in [
            "generate",
            "create",
            "insert",
            "synthetic",
            "生成",
            "创建",
            "插入",
        ]
    )


def _target_text_from_request(request: str) -> str | None:
    quoted = re.findall(r"['\"]([^'\"]{1,240})['\"]", request)
    for item in quoted:
        candidate = item.strip()
        if candidate and not _looks_like_audio_reference_text(candidate):
            return candidate
    patterns = [
        r"\bsaying\s+([^.;]+)",
        r"\bsay\s+([^.;]+)",
        r"说\s*([^。；.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, request, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" :，,。.;'\"")
            if candidate:
                return candidate
    return None


def _looks_like_audio_reference_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "audio" in lowered
        or "input" in lowered
        or re.search(r"\[[0-9.]+\s*s?\s*:[0-9.]+\s*s?\]", lowered) is not None
    )


def _is_pronunciation_text_edit_boundary_failure(
    step: AudioEditStep,
    state: dict[str, Any],
    exc: Exception,
) -> bool:
    error = str(exc).lower()
    if "instruction_st" not in error and "text_edit" not in error:
        return False
    text = " ".join(
        str(value)
        for value in [
            state.get("user_prompt"),
            step.description,
            step.params.get("instruction"),
            step.params.get("instruction_st"),
            step.params.get("edit_instruction"),
        ]
        if value
    ).lower()
    pronunciation_markers = [
        "pronunciation",
        "pronounce",
        "phoneme",
        "vowel",
        "diphthong",
        "front vowel",
        "back vowel",
        "consonant",
        "syllable",
        "ipa",
        "from /",
        "to /",
        "发音",
        "音素",
        "元音",
        "辅音",
    ]
    correction_markers = [
        "correct",
        "change",
        "modify",
        "adjust",
        "replace",
        "from",
        "to",
        "改",
        "纠正",
    ]
    return any(marker in text for marker in pronunciation_markers) and any(
        marker in text for marker in correction_markers
    )


def _target_text_from_step_params(step: AudioEditStep) -> str | None:
    target = step.params.get("target_text")
    if isinstance(target, str) and target.strip():
        return target.strip()
    target = _target_text_from_instruction_st(step.params.get("instruction_st"))
    if target:
        return target
    return None


def _target_text_from_instruction_st(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    def repl(match: re.Match[str]) -> str:
        if match.group("sub_attrs") is not None:
            attrs = match.group("sub_attrs") or ""
            target = re.search(r'targ="([^"]*)"', attrs) or re.search(r"targ='([^']*)'", attrs)
            return unescape(target.group(1)) if target else unescape(match.group("sub_body") or "")
        if match.group("ins_body") is not None:
            return unescape(match.group("ins_body") or "")
        if match.group("del_body") is not None:
            return ""
        return match.group(0)

    rendered = re.sub(
        r"<sub\s+(?P<sub_attrs>[^>]*)>(?P<sub_body>.*?)</sub>"
        r"|<ins\b[^>]*>(?P<ins_body>.*?)</ins>"
        r"|<del\b[^>]*>(?P<del_body>.*?)</del>",
        repl,
        value,
        flags=re.DOTALL,
    )
    rendered = re.sub(r"<[^>]+>", "", rendered)
    rendered = unescape(rendered).strip()
    return rendered or None


def _initialize_runtime_artifact_authority(
    state: dict[str, Any],
    execution_graph: ExecutionGraph | None,
) -> None:
    """Register immutable input revisions before any step can mutate runtime state."""

    paths = collect_input_audios(state)
    primary_path = str(state.get("current_audio_path") or "").strip()
    if primary_path and primary_path not in paths:
        paths.insert(0, primary_path)
    graph_input = None
    if execution_graph is not None:
        graph_input = next(
            (item for item in execution_graph.artifacts if item.source == "plan_input"),
            None,
        )
    for index, path in enumerate(paths):
        artifact_id = (
            graph_input.artifact_id
            if index == 0 and graph_input is not None
            else f"runtime:artifact:input:{index:04d}"
        )
        artifact = VerificationArtifact(
            artifact_id=artifact_id,
            path=str(path),
            duration_sec=_audio_duration_seconds(Path(str(path))),
            timeline_revision=0,
            status="available",
            audio_ref_id=canonical_audio_ref_for_runtime_path(str(path), state),
        )
        state["runtime_artifact_revisions"][artifact_id] = artifact
        if index == 0:
            state["current_runtime_artifact_id"] = artifact_id
        if graph_input is not None and artifact_id == graph_input.artifact_id:
            graph_input.duration_sec = artifact.duration_sec
            graph_input.timeline_revision = 0


def _bind_initial_perception_evidence_timebase(state: dict[str, Any]) -> None:
    report = state.get("perception_report")
    if not isinstance(report, dict) or not report:
        return
    artifact = _current_runtime_artifact(state)
    if artifact is not None:
        _bind_perception_evidence_timebase(state, report, artifact)


def _bind_perception_evidence_timebase(
    state: dict[str, Any],
    report: dict[str, Any],
    artifact: VerificationArtifact,
) -> None:
    binding = {
        "schema_version": "x-editor.evidence-timebase-binding/v1",
        "observed_artifact_id": artifact.artifact_id,
        "observed_audio_path": artifact.path,
        "observed_timeline_revision": artifact.timeline_revision,
        "audio_ref_id": artifact.audio_ref_id or report.get("audio_ref_id"),
    }
    state["perception_evidence_timebase"] = binding
    if not isinstance(report.get("timebase"), dict):
        report["timebase"] = copy.deepcopy(binding)


def _current_runtime_artifact(state: dict[str, Any]) -> VerificationArtifact | None:
    artifact_id = state.get("current_runtime_artifact_id")
    artifact = (state.get("runtime_artifact_revisions") or {}).get(artifact_id)
    return artifact if isinstance(artifact, VerificationArtifact) else None


def _runtime_artifact_for_path(
    state: dict[str, Any],
    path: str,
    *,
    default_revision: int | None = None,
) -> VerificationArtifact:
    resolved = _resolved_path_key(path)
    for artifact in (state.get("runtime_artifact_revisions") or {}).values():
        if (
            isinstance(artifact, VerificationArtifact)
            and artifact.path
            and _resolved_path_key(artifact.path) == resolved
        ):
            return artifact
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    artifact = VerificationArtifact(
        artifact_id=f"runtime:artifact:path:{digest}",
        path=str(path),
        duration_sec=_audio_duration_seconds(Path(str(path))),
        timeline_revision=default_revision,
        status="available",
        audio_ref_id=canonical_audio_ref_for_runtime_path(str(path), state),
    )
    state.setdefault("runtime_artifact_revisions", {})[artifact.artifact_id] = artifact
    return artifact


def _graph_output_artifact_id(state: dict[str, Any], step: AudioEditStep) -> str:
    graph = state.get("execution_graph")
    if isinstance(graph, ExecutionGraph):
        node = next((item for item in graph.nodes if item.plan_step_id == step.id), None)
        if node is not None:
            preferred = next(
                (item for item in node.output_ports if item.name == "output_audio_path"),
                None,
            )
            if preferred is None and node.output_ports:
                preferred = node.output_ports[0]
            if preferred is not None:
                return preferred.artifact_id
    return f"runtime:artifact:step:{step.id}:output_audio_path"


def _build_step_verification_context(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
    consumed_audio_paths: list[str],
) -> StepVerificationContext:
    lineage = output.get(AUDIO_TEMPORAL_LINEAGE_KEY)
    lineage = lineage if isinstance(lineage, dict) else {}
    input_revision = _safe_int(lineage.get("input_revision"))
    output_revision = _safe_int(lineage.get("output_revision"))
    timeline_inputs = [
        str(item)
        for item in lineage.get("input_audio_paths") or []
        if isinstance(item, str) and item.strip()
    ]
    primary_path = (
        timeline_inputs[0]
        if timeline_inputs
        else str(state.get("current_audio_path") or "").strip()
    )
    ordered_paths = list(dict.fromkeys([*timeline_inputs, *consumed_audio_paths]))
    if primary_path and primary_path not in ordered_paths:
        ordered_paths.insert(0, primary_path)
    input_artifacts = tuple(
        _runtime_artifact_for_path(
            state,
            path,
            default_revision=(input_revision if path == primary_path else None),
        )
        for path in ordered_paths
    )
    primary_input = next(
        (
            item
            for item in input_artifacts
            if item.path
            and primary_path
            and _resolved_path_key(item.path) == _resolved_path_key(primary_path)
        ),
        _current_runtime_artifact(state),
    )
    if primary_input is not None and input_revision is not None and (
        primary_input.timeline_revision != input_revision
    ):
        primary_input = VerificationArtifact(
            artifact_id=primary_input.artifact_id,
            path=primary_input.path,
            duration_sec=(
                _safe_float(lineage.get("input_duration_sec"))
                or primary_input.duration_sec
            ),
            timeline_revision=input_revision,
            status=primary_input.status,
            parent_artifact_ids=primary_input.parent_artifact_ids,
            audio_ref_id=primary_input.audio_ref_id,
        )
    output_path = _extract_output_path(output)
    candidate_output = None
    if output_path:
        candidate_output = VerificationArtifact(
            artifact_id=_graph_output_artifact_id(state, step),
            path=output_path,
            duration_sec=(
                _safe_float(lineage.get("output_duration_sec"))
                or _audio_duration_seconds(Path(output_path))
            ),
            timeline_revision=output_revision,
            status="candidate",
            parent_artifact_ids=tuple(item.artifact_id for item in input_artifacts),
            audio_ref_id=canonical_audio_ref_for_runtime_path(output_path, state),
        )
    evidence_binding = _build_evidence_timebase_binding(
        state=state,
        primary_input=primary_input,
    )
    return StepVerificationContext(
        step_id=step.id,
        input_artifacts=input_artifacts,
        primary_input_artifact=primary_input,
        candidate_output_artifact=candidate_output,
        source_timebase_artifact_id=(
            primary_input.artifact_id if primary_input is not None else None
        ),
        target_timebase_artifact_id=(
            candidate_output.artifact_id if candidate_output is not None else None
        ),
        evidence_timebase=evidence_binding,
    )


def _build_evidence_timebase_binding(
    *,
    state: dict[str, Any],
    primary_input: VerificationArtifact | None,
) -> EvidenceTimebaseBinding:
    perception = state.get("perception_report")
    perception = perception if isinstance(perception, dict) else {}
    raw_words = [
        dict(item)
        for item in perception.get("word_timestamps") or []
        if isinstance(item, dict)
    ]
    binding = state.get("perception_evidence_timebase")
    if not raw_words:
        return EvidenceTimebaseBinding(
            status="absent",
            verifier_artifact_id=(primary_input.artifact_id if primary_input else None),
            verifier_timeline_revision=(
                primary_input.timeline_revision if primary_input else None
            ),
        )
    if not isinstance(binding, dict) or primary_input is None:
        return EvidenceTimebaseBinding(
            status="unbound",
            verifier_artifact_id=(primary_input.artifact_id if primary_input else None),
            verifier_timeline_revision=(
                primary_input.timeline_revision if primary_input else None
            ),
            reason="Reviewed word timestamps do not declare an observed artifact revision.",
        )
    observed_id = str(binding.get("observed_artifact_id") or "").strip() or None
    observed_revision = _safe_int(binding.get("observed_timeline_revision"))
    if observed_id == primary_input.artifact_id and observed_revision == (
        primary_input.timeline_revision
    ):
        return EvidenceTimebaseBinding(
            status="current",
            observed_artifact_id=observed_id,
            observed_timeline_revision=observed_revision,
            verifier_artifact_id=primary_input.artifact_id,
            verifier_timeline_revision=primary_input.timeline_revision,
            word_timestamps=tuple(raw_words),
        )
    if crosses_unreviewed_content_transition(
        state, observed_artifact_id=observed_id, input_artifact=primary_input,
    ):
        return EvidenceTimebaseBinding(
            status="stale_unmappable", observed_artifact_id=observed_id,
            observed_timeline_revision=observed_revision,
            verifier_artifact_id=primary_input.artifact_id,
            verifier_timeline_revision=primary_input.timeline_revision,
            omitted_word_count=len(raw_words),
            reason="The time axis is mapped, but the processing contract does not preserve old lexical observations.",
        )
    if observed_revision == 0 and primary_input.timeline_revision not in {None, 0}:
        try:
            ledger = evidence_timeline(
                state,
                observed_artifact_id=observed_id,
                observed_revision=observed_revision,
                input_artifact=primary_input,
            )
            if ledger is not None:
                projected, omitted = _project_words_from_base_timeline(raw_words, ledger)
                if projected:
                    return EvidenceTimebaseBinding(
                        status=(
                            "mapped_from_base"
                            if omitted == 0
                            else "partially_mapped_from_base"
                        ),
                        observed_artifact_id=observed_id,
                        observed_timeline_revision=observed_revision,
                        verifier_artifact_id=primary_input.artifact_id,
                        verifier_timeline_revision=primary_input.timeline_revision,
                        word_timestamps=tuple(projected),
                        omitted_word_count=omitted,
                        reason=(
                            None
                            if omitted == 0
                            else "Words crossing edited regions were omitted from exact projection."
                        ),
                    )
        except (TypeError, ValueError):
            pass
    return EvidenceTimebaseBinding(
        status="stale_unmappable",
        observed_artifact_id=observed_id,
        observed_timeline_revision=observed_revision,
        verifier_artifact_id=primary_input.artifact_id,
        verifier_timeline_revision=primary_input.timeline_revision,
        omitted_word_count=len(raw_words),
        reason="Reviewed word timestamps could not be mapped safely to the step input revision.",
    )


def _project_words_from_base_timeline(
    words: list[dict[str, Any]],
    ledger: TimelineLedger,
) -> tuple[list[dict[str, Any]], int]:
    projected: list[dict[str, Any]] = []
    omitted = 0
    for word in words:
        start = _safe_float(word.get("start_sec", word.get("start")))
        end = _safe_float(word.get("end_sec", word.get("end")))
        if start is None or end is None or end <= start:
            omitted += 1
            continue
        boundary_adjustment_sec = 0.0
        if end > ledger.base_duration_sec:
            boundary_adjustment_sec = end - ledger.base_duration_sec
            if (
                boundary_adjustment_sec <= _EVIDENCE_BOUNDARY_TOLERANCE_SEC
                and start < ledger.base_duration_sec
            ):
                end = ledger.base_duration_sec
            else:
                omitted += 1
                continue
        try:
            mapping = ledger.map_base_interval(TimeInterval(start_sec=start, end_sec=end))
        except ValueError:
            omitted += 1
            continue
        envelope = mapping.target_envelope
        if (
            mapping.status != "complete"
            or envelope is None
            or any(fragment.correspondence != "exact" for fragment in mapping.fragments)
        ):
            omitted += 1
            continue
        item = dict(word)
        item["start_sec"] = envelope.start_sec
        item["end_sec"] = envelope.end_sec
        item["timebase_projection"] = "timeline_ledger_exact"
        if boundary_adjustment_sec:
            item["timebase_boundary_clamped"] = True
            item["timebase_boundary_adjustment_sec"] = boundary_adjustment_sec
        projected.append(item)
    return projected, omitted


def _rebind_explicit_step_spans_to_current_artifact(
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    """Bind explicit spans to the actual step input, including independent branches."""

    containers = (("inputs", step.inputs), ("params", step.params))
    span_keys = ("selected_spans", "spans", "time_spans", "segments_to_extract")
    targets = [
        (container_name, container, key, container.get(key))
        for container_name, container in containers
        for key in span_keys
        if isinstance(container.get(key), list) and container.get(key)
    ]
    if not targets:
        return
    if step.span_timebase == "step_input":
        from x_editor.runtime.span_coordinates import validate_input_coordinates
        input_path = _resolve_input_audio(step, state)
        validate_input_coordinates(step, input_path, [value for _, _, _, value in targets])
        state.setdefault("timeline_span_rebindings", []).append({
            "step_id": step.id, "status": "bound_to_step_input",
            "input_audio_path": str(input_path), "span_timebase": step.span_timebase,
        })
        return
    binding = state.get("perception_evidence_timebase")
    if not isinstance(binding, dict):
        return
    current = _runtime_artifact_for_path(state, _resolve_input_audio(step, state))
    observed_id = str(binding.get("observed_artifact_id") or "").strip()
    observed_revision = _safe_int(binding.get("observed_timeline_revision"))
    if observed_id == current.artifact_id and observed_revision == current.timeline_revision:
        return
    ledger = None
    try:
        ledger = evidence_timeline(
            state,
            observed_artifact_id=observed_id,
            observed_revision=observed_revision,
            input_artifact=current,
        )
    except (TypeError, ValueError):
        pass
    if (
        ledger is None
        or current.timeline_revision is None
        or ledger.current_timebase.revision != current.timeline_revision
    ):
        state.setdefault("timeline_span_rebindings", []).append(
            {
                "step_id": step.id,
                "status": "requires_reperception",
                "observed_artifact_id": observed_id or None,
                "observed_timeline_revision": observed_revision,
                "current_artifact_id": current.artifact_id,
                "current_timeline_revision": current.timeline_revision,
                "reason": "no exact timeline mapping is available for the evidence revision",
            }
        )
        raise RuntimeError(
            "stale_timeline_evidence_requires_reperception: explicit spans are bound "
            "to an older artifact revision without an exact mapping"
        )
    for container_name, container, key, value in targets:
        mapped, changed, failure = _map_explicit_span_list(value, ledger)
        if failure is not None:
            state.setdefault("timeline_span_rebindings", []).append(
                {
                    "step_id": step.id,
                    "field_path": f"{container_name}.{key}",
                    "status": "requires_reperception",
                    "observed_artifact_id": observed_id or None,
                    "observed_timeline_revision": observed_revision,
                    "current_artifact_id": current.artifact_id,
                    "current_timeline_revision": current.timeline_revision,
                    "reason": failure,
                }
            )
            raise RuntimeError(
                "stale_timeline_evidence_requires_reperception: " + failure
            )
        if not changed:
            continue
        container[key] = mapped
        state.setdefault("timeline_span_rebindings", []).append(
            {
                "step_id": step.id,
                "field_path": f"{container_name}.{key}",
                "status": "mapped_exactly",
                "observed_artifact_id": observed_id or None,
                "observed_timeline_revision": observed_revision,
                "current_artifact_id": current.artifact_id,
                "current_timeline_revision": current.timeline_revision,
                "source": "timeline_ledger",
            }
        )


def _map_explicit_span_list(
    spans: list[Any],
    ledger: TimelineLedger,
) -> tuple[list[Any], bool, str | None]:
    mapped: list[Any] = []
    changed = False
    for index, raw in enumerate(spans):
        if isinstance(raw, dict):
            start = _safe_float(raw.get("start_sec", raw.get("start")))
            end = _safe_float(raw.get("end_sec", raw.get("end")))
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            start = _safe_float(raw[0])
            end = _safe_float(raw[1])
        else:
            return spans, False, f"span[{index}] has no typed start/end coordinates"
        if start is None or end is None or end <= start:
            return spans, False, f"span[{index}] has invalid start/end coordinates"
        try:
            result = ledger.map_base_interval(TimeInterval(start_sec=start, end_sec=end))
        except ValueError as exc:
            return spans, False, f"span[{index}] cannot be mapped: {exc}"
        envelope = result.target_envelope
        if (
            result.status != "complete"
            or envelope is None
            or any(fragment.correspondence != "exact" for fragment in result.fragments)
        ):
            return (
                spans,
                False,
                f"span[{index}] intersects an edited or unmapped timeline region",
            )
        if isinstance(raw, dict):
            item = dict(raw)
            start_key = "start_sec" if "start_sec" in item or "start" not in item else "start"
            end_key = "end_sec" if "end_sec" in item or "end" not in item else "end"
            item[start_key] = envelope.start_sec
            item[end_key] = envelope.end_sec
            item["timebase_projection"] = "timeline_ledger_exact"
        else:
            item = [envelope.start_sec, envelope.end_sec, *list(raw[2:])]
        changed = changed or abs(envelope.start_sec - start) > 1e-7 or abs(
            envelope.end_sec - end
        ) > 1e-7
        mapped.append(item)
    return mapped, changed, None


def _stage_step_candidate(
    step: AudioEditStep,
    state: dict[str, Any],
    context: StepVerificationContext,
) -> None:
    if not isinstance(step.result, dict):
        return
    step.result[_RUNTIME_ARTIFACT_COMMIT_KEY] = {
        "schema_version": "x-editor.runtime-artifact-commit/v1",
        "status": "candidate",
        "verification_context": context.to_dict(),
    }
    state.setdefault("verification_contexts", []).append(context.to_dict())
    candidate = context.candidate_output_artifact
    if candidate is not None:
        state.setdefault("runtime_artifact_revisions", {})[
            candidate.artifact_id
        ] = candidate


def _accept_step_candidate(
    step: AudioEditStep,
    state: dict[str, Any],
    context: StepVerificationContext,
) -> None:
    candidate = context.candidate_output_artifact
    _set_step_candidate_status(step, "accepted")
    if candidate is None:
        return
    accepted = VerificationArtifact(
        artifact_id=candidate.artifact_id,
        path=candidate.path,
        duration_sec=candidate.duration_sec,
        timeline_revision=candidate.timeline_revision,
        status="accepted",
        parent_artifact_ids=candidate.parent_artifact_ids,
        audio_ref_id=candidate.audio_ref_id,
    )
    state.setdefault("runtime_artifact_revisions", {})[accepted.artifact_id] = accepted
    time_axis = (step.result or {}).get("source_time_axis")
    if isinstance(time_axis, dict) and time_axis.get("semantic_evidence_preserved") is False:
        state.setdefault("runtime_semantic_evidence_boundaries", {})[accepted.artifact_id] = {
            "step_id": step.id, "reason": "time_coordinates_do_not_preserve_lexical_evidence",
        }
    state["current_runtime_artifact_id"] = accepted.artifact_id


def _reject_step_candidate(
    step: AudioEditStep,
    state: dict[str, Any],
    context: StepVerificationContext,
) -> None:
    candidate = context.candidate_output_artifact
    _set_step_candidate_status(step, "rejected")
    if candidate is None:
        return
    rejected = VerificationArtifact(
        artifact_id=candidate.artifact_id,
        path=candidate.path,
        duration_sec=candidate.duration_sec,
        timeline_revision=candidate.timeline_revision,
        status="rejected",
        parent_artifact_ids=candidate.parent_artifact_ids,
        audio_ref_id=candidate.audio_ref_id,
    )
    state.setdefault("runtime_artifact_revisions", {})[rejected.artifact_id] = rejected
    _record_step_audio_artifacts(step, state, step.result)
    _set_recorded_step_artifact_status(state, step.id, "rejected")
    state.setdefault("rejected_step_artifacts", []).append(
        {
            **rejected.to_dict(),
            "step_id": step.id,
            "operation": step.operation,
            "error": step.error,
        }
    )


def _set_step_candidate_status(step: AudioEditStep, status: str) -> None:
    if not isinstance(step.result, dict):
        return
    payload = step.result.get(_RUNTIME_ARTIFACT_COMMIT_KEY)
    if isinstance(payload, dict):
        payload["status"] = status


def _set_recorded_step_artifact_status(
    state: dict[str, Any], step_id: str, status: str
) -> None:
    for record in state.get("audio_artifacts") or []:
        if isinstance(record, dict) and record.get("step_id") == step_id:
            record["artifact_status"] = status


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _authoritative_execution_graph(
    plan: AudioEditPlan,
    context: dict[str, Any],
) -> tuple[ExecutionGraph | None, str | None]:
    """Load the Forward Runtime graph and prove it still represents ``plan``.

    Direct executor callers remain supported without a graph. Once the Forward
    Runtime supplies one, however, the executor must not silently fall back to
    the plan list or accept a graph that describes different work.
    """

    raw_graph = context.get("_forward_execution_graph")
    if raw_graph is None:
        return None, None
    try:
        graph = (
            raw_graph.model_copy(deep=True)
            if isinstance(raw_graph, ExecutionGraph)
            else ExecutionGraph.model_validate(raw_graph)
        )
    except Exception as exc:
        return None, f"Invalid authoritative execution graph: {exc}"

    if graph.status != "pending":
        return None, (
            "Invalid authoritative execution graph: expected pending graph, "
            f"got {graph.status}"
        )
    if graph.request != plan.request:
        return None, "Authoritative execution graph request does not match PlanIR"

    plan_steps = {step.id: step for step in plan.steps}
    graph_nodes = {node.plan_step_id: node for node in graph.nodes}
    if len(graph_nodes) != len(graph.nodes):
        return None, "Authoritative execution graph contains duplicate PlanIR step IDs"
    if set(graph_nodes) != set(plan_steps):
        return None, (
            "Authoritative execution graph step set does not match PlanIR: "
            f"graph={sorted(graph_nodes)}, plan={sorted(plan_steps)}"
        )

    node_ids = {node.node_id for node in graph.nodes}
    if len(node_ids) != len(graph.nodes) or set(graph.topological_order) != node_ids:
        return None, "Authoritative execution graph has an invalid topological node set"
    if len(graph.topological_order) != len(graph.nodes):
        return None, "Authoritative execution graph topological order contains duplicates"
    topological_position = {
        node_id: index for index, node_id in enumerate(graph.topological_order)
    }

    for step_id, step in plan_steps.items():
        node = graph_nodes[step_id]
        if (
            node.operation != step.operation
            or node.description != step.description
            or node.inputs != step.inputs
            or node.params != step.params
            or node.depends_on != step.depends_on
            or node.expected_output != step.expected_output
            or node.required != step.required
            or node.continue_on_error != step.continue_on_error
        ):
            return None, (
                "Authoritative execution graph node does not match PlanIR step "
                f"{step_id}"
            )
        if node.dependency_node_ids != [graph_nodes[item].node_id for item in step.depends_on]:
            return None, (
                "Authoritative execution graph dependency bindings do not match "
                f"PlanIR step {step_id}"
            )
        node_position = topological_position[node.node_id]
        if any(
            topological_position[dependency_node_id] >= node_position
            for dependency_node_id in node.dependency_node_ids
        ):
            return None, (
                "Authoritative execution graph topological order violates dependency "
                f"for PlanIR step {step_id}"
            )
    return graph, None


def _sync_execution_graph_step(
    graph: ExecutionGraph,
    step: AudioEditStep,
) -> None:
    node = next(item for item in graph.nodes if item.plan_step_id == step.id)
    node.status = {
        "pending": "pending",
        "running": "running",
        "success": "succeeded",
        "failed": "failed",
        "skipped": "skipped",
    }[step.status]

    artifacts_by_id = {item.artifact_id: item for item in graph.artifacts}
    commit = (
        step.result.get(_RUNTIME_ARTIFACT_COMMIT_KEY)
        if isinstance(step.result, dict)
        else None
    )
    commit = commit if isinstance(commit, dict) else {}
    verification_context = commit.get("verification_context")
    verification_context = (
        verification_context if isinstance(verification_context, dict) else {}
    )
    candidate = verification_context.get("candidate_output_artifact")
    candidate = candidate if isinstance(candidate, dict) else {}
    for port in node.output_ports:
        artifact = artifacts_by_id[port.artifact_id]
        if step.status == "success":
            value = _execution_graph_output_value(step.result, port.name)
            port.status = "produced" if value is not None else "failed"
            artifact.status = port.status
            artifact.value = value
            artifact.verification_status = (
                "accepted" if value is not None else "not_required"
            )
            if value is not None:
                artifact.duration_sec = _safe_float(candidate.get("duration_sec"))
                artifact.timeline_revision = _safe_int(candidate.get("timeline_revision"))
                artifact.parent_artifact_ids = [
                    str(item)
                    for item in candidate.get("parent_artifact_ids") or []
                    if str(item)
                ]
        elif step.status in {"failed", "skipped"}:
            rejected_value = (
                _execution_graph_output_value(step.result, port.name)
                if commit.get("status") == "rejected"
                else None
            )
            if rejected_value is not None:
                port.status = "rejected"
                artifact.status = "rejected"
                artifact.value = rejected_value
                artifact.verification_status = "rejected"
                artifact.duration_sec = _safe_float(candidate.get("duration_sec"))
                artifact.timeline_revision = _safe_int(candidate.get("timeline_revision"))
                artifact.parent_artifact_ids = [
                    str(item)
                    for item in candidate.get("parent_artifact_ids") or []
                    if str(item)
                ]
            else:
                port.status = "failed"
                artifact.status = "failed"
                artifact.value = None
                artifact.verification_status = "not_required"
        else:
            port.status = "pending"
            artifact.status = "pending"
            artifact.value = None
            artifact.verification_status = "pending"


def _execution_graph_output_value(result: dict[str, Any] | None, path: str) -> Any:
    value: Any = result
    for component in path.split("."):
        if isinstance(value, dict):
            value = value.get(component)
        elif isinstance(value, list) and component.isdigit():
            index = int(component)
            value = value[index] if index < len(value) else None
        else:
            return None
        if value is None:
            return None
    return value


def _finalize_authoritative_execution_graph(
    graph: ExecutionGraph,
    plan: AudioEditPlan,
) -> None:
    for step in plan.steps:
        _sync_execution_graph_step(graph, step)
    artifacts_by_id = {item.artifact_id: item for item in graph.artifacts}
    primary_artifact = artifacts_by_id.get(graph.primary_output.source_artifact_id)
    required_nodes_succeeded = all(
        not node.required or node.status == "succeeded" for node in graph.nodes
    )
    graph.status = (
        "succeeded"
        if required_nodes_succeeded
        and primary_artifact is not None
        and primary_artifact.status == "produced"
        else "failed"
    )


def _full_text_replacement_is_long_context(
    call: Any, state: dict[str, Any], target_text: str
) -> bool:
    source_text = ""
    for mapping in (
        getattr(call, "normalized_params", {}) or {},
        getattr(call, "arguments", {}) or {},
    ):
        if isinstance(mapping, dict) and isinstance(mapping.get("source_text"), str):
            source_text = mapping["source_text"]
            break
    if not source_text and isinstance(state.get("source_text"), str):
        source_text = state["source_text"]
    source_words = len(source_text.split())
    return len(source_text) >= 220 or source_words >= 38 or len(target_text) >= 90


def _fail_entire_plan(plan: AudioEditPlan, error: str) -> AudioEditPlan:
    plan = plan.model_copy(deep=True)
    plan.metadata["execution_error"] = error
    for step in plan.steps:
        _mark_skipped(step, error)
    plan.metadata["execution_summary"] = execution_summary(plan)
    return plan


def _tool_bootstrap_reports(tool_executor: Any, start_index: int) -> list[dict[str, Any]]:
    reports = getattr(tool_executor, "bootstrap_reports", None)
    if not isinstance(reports, list):
        return []
    return [item for item in reports[start_index:] if isinstance(item, dict)]


def _final_output_path(plan: AudioEditPlan, state: dict[str, Any]) -> str | None:
    if state.get("current_audio_path"):
        return str(state["current_audio_path"])
    for step in reversed(plan.steps):
        output = _extract_output_path(step.result)
        if output:
            return output
    return plan.output_audio or plan.input_audio


def _declared_output_artifacts(
    plan: AudioEditPlan,
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not plan.output_artifacts:
        return [], []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    step_outputs = state.get("step_outputs") or {}
    for declaration in plan.output_artifacts:
        result = step_outputs.get(declaration.from_step)
        value: Any = result
        for component in declaration.from_output.split("."):
            if isinstance(value, dict):
                value = value.get(component)
            elif isinstance(value, list) and component.isdigit():
                index = int(component)
                value = value[index] if index < len(value) else None
            else:
                value = None
            if value is None:
                break
        if not isinstance(value, str) or not Path(value).is_file():
            errors.append(
                f"{declaration.artifact_id}: {declaration.from_step}."
                f"{declaration.from_output} did not resolve to an audio file"
            )
            continue
        records.append(
            {
                "artifact_id": declaration.artifact_id,
                "from_step": declaration.from_step,
                "from_output": declaration.from_output,
                "role": declaration.role,
                "primary": declaration.primary,
                "path": str(value),
            }
        )
    return records, errors


def execution_summary(plan: AudioEditPlan) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for step in plan.steps:
        counts[step.status] = counts.get(step.status, 0) + 1
    failed = [step.id for step in plan.steps if step.status == "failed"]
    skipped = [step.id for step in plan.steps if step.status == "skipped"]
    required_skipped = [
        step.id for step in plan.steps if step.required and step.status == "skipped"
    ]
    return {
        "step_count": len(plan.steps),
        "status_counts": counts,
        "failed_steps": failed,
        "skipped_steps": skipped,
        "required_skipped_steps": required_skipped,
        "completed": (
            not failed
            and not required_skipped
            and not any(step.status in {"pending", "running"} for step in plan.steps)
        ),
    }


def build_default_audio_operation_registry() -> AudioOperationRegistry:
    """Return a registry populated with the current catalog-backed operations."""

    executor = AudioEditExecutor()
    return executor.registry
