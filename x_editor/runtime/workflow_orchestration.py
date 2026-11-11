"""Typed W0/W1/W2 orchestration over the shared strict forward path."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.invocation_trace import (
    ComponentInvocationAdmissionPolicy,
    ComponentInvocationTrace,
    component_invocation_trace_blockers,
    validate_component_invocation_trace,
)
from x_editor.core.perception_evidence_authority import ObservationKind
from x_editor.core.reviewed_evidence import (
    PROJECTED_SCALAR_FIELDS,
    sanitize_reviewed_perception_facts,
)
from x_editor.core.schemas import AudioEditPlan, PlannerCannotPlanReason, StrictModel
from x_editor.perception.capability_feedback import PerceptionCapabilityFeedback
from x_editor.reasoner.contracts import PriorTurnSemanticContext, SessionStateContext
from x_editor.reasoner.shadow import (
    ReasonerShadowCallRecord,
    ReasonerShadowInputSnapshot,
)
from x_editor.runtime.audio_asset_facts import AudioAssetFacts
from x_editor.runtime.audio_base_selection import plan_selects_bound_reference_base
from x_editor.runtime.execution_events import (
    ExecutionEventObserver,
    ExecutionEventRecorder,
)
from x_editor.runtime.forward_execution import (
    ForwardExecutionPath,
    ForwardExecutionPolicy,
    ForwardExecutionRequest,
    ForwardExecutionResult,
)
from x_editor.runtime.workflow_case import (
    HistoricalReviewedEvidenceContext,
    RuntimeAudioBinding,
)
from x_editor.workflow_control import WorkflowEvent
from x_editor.workflow_control.schemas import new_workflow_id

WorkflowArm = Literal["W0", "W1", "W2"]
WorkflowPlannerDecision = Literal[
    "execute",
    "request_perception",
    "request_replan",
    "complete_from_session_state",
    "cannot_plan",
]
WorkflowOrchestrationStatus = Literal["completed", "blocked", "failed"]
WorkflowOrchestrationFailureStage = Literal[
    "none",
    "configuration",
    "reasoner",
    "planner",
    "perception",
    "workflow_protocol",
    "plan_admission",
    "executor",
    "verifier",
    "artifact_store",
    "forward_execution",
]

_ARM_OWNED_CONTEXT_PREFIXES = ("workflow_", "_workflow_")
_ARM_OWNED_CONTEXT_KEYS = {
    "enable_planner_perception_guidance",
    "planner_perception_guidance",
    "perception_attempts",
    "perception_gap_fill_guidance",
    "perception_report",
    "span_selection_analysis",
    "previous_failure",
    "failed_route",
    "failed_tool_trace",
    "retry_guidance",
}
_REVIEWED_PLANNER_PROJECTION_FIELDS = {
    "perception_report",
    *PROJECTED_SCALAR_FIELDS,
}
_REVIEWED_EXECUTION_PROJECTION_FIELDS = {
    "perception_report",
    *PROJECTED_SCALAR_FIELDS,
}


class WorkflowPlanner(Protocol):
    """Planner boundary used by all workflow arms."""

    def plan(self, user_request: str, context: dict[str, Any]) -> Any:
        """Return an AudioEditPlan or a WorkflowPlannerOutput, possibly awaitable."""


class WorkflowPerceptionGateway(Protocol):
    """Explicit perception boundary used before a plan is frozen."""

    def perceive(
        self,
        request: WorkflowPerceptionRequest,
        *,
        context: dict[str, Any],
    ) -> Any:
        """Return WorkflowPerceptionEvidence, possibly awaitable."""


class WorkflowPerceptionRequest(StrictModel):
    """One explicit, required perception request issued before forward execution."""

    schema_version: Literal["x-editor.workflow-perception-request/v1"] = (
        "x-editor.workflow-perception-request/v1"
    )
    request_id: str
    scope: Literal["general", "targeted"]
    instruction: str
    source_step_id: str | None = None
    operation: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    audio_refs: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    required: Literal[True] = True

    @model_validator(mode="after")
    def validate_execution_contract(self) -> WorkflowPerceptionRequest:
        if self.operation is not None and not self.operation.startswith(
            ("perception.", "flexsed.")
        ):
            raise ValueError("workflow perception operation must use perception.* or flexsed.*")
        if self.operation is None and (self.inputs or self.params):
            raise ValueError("perception inputs and params require an explicit operation")
        return self


class WorkflowMissingAudioEvidence(StrictModel):
    """Explicit record that one requested audio produced no reviewed facts."""

    schema_version: Literal["x-editor.workflow-missing-audio-evidence/v1"] = (
        "x-editor.workflow-missing-audio-evidence/v1"
    )
    audio_ref: str = Field(min_length=1, max_length=4096)
    missing_requirements: list[str] = Field(default_factory=list, max_length=64)
    reason: str = Field(min_length=1, max_length=2048)
    blocking: bool = True


class WorkflowToolCallSummary(StrictModel):
    """Safe accounting for one tool call without arguments or outputs."""

    schema_version: Literal["x-editor.workflow-tool-call-summary/v1"] = (
        "x-editor.workflow-tool-call-summary/v1"
    )
    tool_id: str
    step_id: str | None = None
    success: bool
    attempt_count: int = Field(default=1, ge=1)


class WorkflowPerceptionEvidence(StrictModel):
    """Gateway evidence bound to one request and pending reviewed projection."""

    schema_version: Literal["x-editor.workflow-perception-evidence/v1"] = (
        "x-editor.workflow-perception-evidence/v1"
    )
    evidence_id: str
    request_id: str
    scope: Literal["general", "targeted"]
    source: str
    operation: str | None = None
    expected_evidence_fields: list[str] = Field(default_factory=list, max_length=64)
    requested_audio_refs: list[str] = Field(default_factory=list)
    audio_refs: list[str] = Field(default_factory=list)
    missing_audio_evidence: list[WorkflowMissingAudioEvidence] = Field(
        default_factory=list,
        max_length=16,
    )
    facts: dict[str, Any] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)
    usable: bool = True
    invocation_trace: ComponentInvocationTrace | None = None
    tool_calls: list[WorkflowToolCallSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_usable_evidence(self) -> WorkflowPerceptionEvidence:
        if self.operation is not None and not self.operation.startswith(
            ("perception.", "flexsed.")
        ):
            raise ValueError("perception evidence operation must use perception.* or flexsed.*")
        self.expected_evidence_fields = list(
            dict.fromkeys(
                str(item).strip() for item in self.expected_evidence_fields if str(item).strip()
            )
        )
        self.requested_audio_refs = list(
            dict.fromkeys(
                str(item).strip()
                for item in (self.requested_audio_refs or self.audio_refs)
                if str(item).strip()
            )
        )
        self.audio_refs = list(
            dict.fromkeys(str(item).strip() for item in self.audio_refs if str(item).strip())
        )
        requested = set(self.requested_audio_refs)
        covered = set(self.audio_refs)
        missing = {item.audio_ref for item in self.missing_audio_evidence}
        if not covered.issubset(requested):
            raise ValueError("covered perception audio refs must have been requested")
        if not missing.issubset(requested):
            raise ValueError("missing perception audio refs must have been requested")
        if covered.intersection(missing):
            raise ValueError("one audio ref cannot be both covered and missing")
        if len(missing) != len(self.missing_audio_evidence):
            raise ValueError("missing audio evidence refs must be unique")
        if self.usable and not self.facts:
            raise ValueError("usable perception evidence requires at least one fact")
        if self.invocation_trace is not None:
            if self.invocation_trace.component != "perception":
                raise ValueError("perception evidence requires a perception trace")
            if self.invocation_trace.status != "completed":
                raise ValueError("returned perception evidence requires completed trace")
        return self


class WorkflowReviewedEvidenceSource(StrictModel):
    """Immutable sanitized source report retained outside model-facing context."""

    schema_version: Literal["x-editor.reviewed-evidence-source/v1"] = (
        "x-editor.reviewed-evidence-source/v1"
    )
    evidence_id: str
    request_id: str
    source: str
    operation: str | None = None
    audio_ref_id: str | None = None
    observation_kinds: list[ObservationKind] = Field(default_factory=list)
    authority_by_observation_kind: dict[
        ObservationKind, Literal["incidental", "declared", "primary"]
    ] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvidenceAuthorityAssignment(StrictModel):
    """Typed canonical-source decision for one audio observation kind."""

    schema_version: Literal["x-editor.evidence-authority-assignment/v1"] = (
        "x-editor.evidence-authority-assignment/v1"
    )
    audio_ref_id: str | None = None
    observation_kind: ObservationKind
    status: Literal["canonical", "conflict"]
    authoritative_evidence_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    candidate_authority: dict[str, Literal["incidental", "declared", "primary"]] = Field(
        default_factory=dict
    )


class WorkflowEvidenceConflict(StrictModel):
    """Auditable disagreement withheld from the canonical Planner projection."""

    schema_version: Literal["x-editor.evidence-conflict/v1"] = "x-editor.evidence-conflict/v1"
    conflict_id: str
    code: str
    observation_kind: ObservationKind
    audio_ref_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    scope_ref: str | None = None
    message: str
    blocking: bool
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowEvidenceProjection(StrictModel):
    """Reviewed evidence fields admitted to Planner and Executor contexts."""

    schema_version: Literal["x-editor.workflow-evidence-projection/v1"] = (
        "x-editor.workflow-evidence-projection/v1"
    )
    evidence_ids: list[str] = Field(default_factory=list)
    requested_audio_refs: list[str] = Field(default_factory=list)
    source_audio_refs: list[str] = Field(default_factory=list)
    missing_audio_refs: list[str] = Field(default_factory=list)
    planner_context: dict[str, Any] = Field(default_factory=dict)
    execution_context: dict[str, Any] = Field(default_factory=dict)
    field_sources: dict[str, list[str]] = Field(default_factory=dict)
    path_sources: dict[str, list[str]] = Field(default_factory=dict)
    path_audio_refs: dict[str, str] = Field(default_factory=dict)
    reviewed_sources: list[WorkflowReviewedEvidenceSource] = Field(default_factory=list)
    authority_assignments: list[WorkflowEvidenceAuthorityAssignment] = Field(default_factory=list)
    evidence_conflicts: list[WorkflowEvidenceConflict] = Field(default_factory=list)
    projection_fingerprint: str

    @model_validator(mode="after")
    def validate_authority_provenance(self) -> WorkflowEvidenceProjection:
        evidence_ids = set(self.evidence_ids)
        referenced_ids = (
            {source.evidence_id for source in self.reviewed_sources}
            | {
                evidence_id
                for assignment in self.authority_assignments
                for evidence_id in (
                    *assignment.authoritative_evidence_ids,
                    *assignment.supporting_evidence_ids,
                    *assignment.candidate_authority,
                )
            }
            | {
                evidence_id
                for conflict in self.evidence_conflicts
                for evidence_id in conflict.evidence_ids
            }
        )
        unknown = sorted(referenced_ids - evidence_ids)
        if unknown:
            raise ValueError(
                f"evidence authority provenance references unknown evidence ids: {unknown}"
            )
        return self


class WorkflowEvidenceProjector(Protocol):
    """Project typed evidence without exposing arbitrary facts to components."""

    def project(
        self,
        evidence: list[WorkflowPerceptionEvidence],
    ) -> WorkflowEvidenceProjection:
        """Return a deterministic reviewed projection, possibly raising."""


class WorkflowPlannedPerceptionStep(StrictModel):
    """Auditable W2 declaration for one planned fine-grained gap step."""

    schema_version: Literal["x-editor.workflow-planned-perception-step/v1"] = (
        "x-editor.workflow-planned-perception-step/v1"
    )
    step_id: str
    role: Literal["fine_grained_gap"] = "fine_grained_gap"
    gap_description: str = Field(min_length=1)
    evidence_requirements: list[str] = Field(min_length=1)
    based_on_evidence_ids: list[str] = Field(min_length=1)


class WorkflowStateOnlyCompletion(StrictModel):
    """Planner-selected existing Session Graph state requiring no tool execution."""

    schema_version: Literal["x-editor.workflow-state-only-completion/v1"] = (
        "x-editor.workflow-state-only-completion/v1"
    )
    selected_node_id: str = Field(min_length=1, max_length=256)
    selected_audio_ref: str = Field(min_length=1, max_length=256)
    selection_kind: Literal["select_existing", "restore_history"]
    changes_satisfied_by_base: list[str] = Field(min_length=1, max_length=32)


class WorkflowPlannerOutput(StrictModel):
    """One Planner turn that either freezes a plan or requests evidence."""

    schema_version: Literal["x-editor.workflow-planner-output/v1"] = (
        "x-editor.workflow-planner-output/v1"
    )
    decision: WorkflowPlannerDecision
    plan: AudioEditPlan | None = None
    perception_requests: list[WorkflowPerceptionRequest] = Field(default_factory=list)
    capability_feedback: PerceptionCapabilityFeedback | None = None
    planned_perception_steps: list[WorkflowPlannedPerceptionStep] = Field(default_factory=list)
    state_only_completion: WorkflowStateOnlyCompletion | None = None
    missing_information: list[str] = Field(default_factory=list)
    cannot_plan_reasons: list[PlannerCannotPlanReason] = Field(default_factory=list)
    reason: str = ""
    frozen_edit_intent: dict[str, Any] | None = None
    invocation_trace: ComponentInvocationTrace | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> WorkflowPlannerOutput:
        if (self.decision == "request_replan") != (self.capability_feedback is not None):
            raise ValueError("catalog replan requires its own capability feedback")
        if self.decision != "cannot_plan" and self.cannot_plan_reasons:
            raise ValueError("typed cannot_plan reasons are valid only for cannot_plan")
        if self.invocation_trace is not None:
            if self.invocation_trace.component != "planner":
                raise ValueError("Planner output requires a planner trace")
            if self.invocation_trace.status != "completed":
                raise ValueError("returned Planner output requires completed trace")
        request_ids = [item.request_id for item in self.perception_requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("perception request ids must be unique within a Planner turn")
        declared_step_ids = [item.step_id for item in self.planned_perception_steps]
        if len(declared_step_ids) != len(set(declared_step_ids)):
            raise ValueError("planned perception step ids must be unique")
        if self.decision == "execute":
            if self.plan is None:
                raise ValueError("execute decision requires a plan")
            if self.state_only_completion is not None:
                raise ValueError("execute decision cannot expose a state-only completion")
            if self.perception_requests:
                raise ValueError("execute decision cannot retain unresolved perception requests")
            planned_step_ids = set(_planned_perception_step_ids(self.plan))
            if not set(declared_step_ids).issubset(planned_step_ids):
                raise ValueError("planned perception declarations must reference perception steps")
        elif self.decision == "request_replan":
            if self.plan or self.perception_requests or self.planned_perception_steps or self.state_only_completion:
                raise ValueError("catalog replan cannot expose execution or observation results")
        elif self.decision == "request_perception":
            if not self.perception_requests:
                raise ValueError("request_perception decision requires perception requests")
            if self.state_only_completion is not None:
                raise ValueError(
                    "request_perception decision cannot expose a state-only completion"
                )
            if self.planned_perception_steps:
                raise ValueError("request_perception cannot declare executable perception steps")
        elif self.decision == "complete_from_session_state":
            if self.state_only_completion is None:
                raise ValueError(
                    "complete_from_session_state requires an explicit state completion"
                )
            if self.plan is not None or self.perception_requests or self.planned_perception_steps:
                raise ValueError("state-only completion cannot expose a plan or perception request")
            if self.missing_information:
                raise ValueError("state-only completion cannot retain missing information")
        elif (
            self.plan is not None
            or self.perception_requests
            or self.planned_perception_steps
            or self.state_only_completion is not None
        ):
            raise ValueError("cannot_plan decision cannot expose a plan or perception requests")
        elif not self.cannot_plan_reasons:
            self.cannot_plan_reasons = [
                PlannerCannotPlanReason(
                    code="other",
                    message=(
                        self.reason
                        or "; ".join(self.missing_information)
                        or "Planner could not produce a legal plan."
                    ),
                )
            ]
        return self


class WorkflowArmPolicy(StrictModel):
    """Frozen orchestration controls for the first offline comparison contract."""

    schema_version: Literal["x-editor.workflow-arm-policy/v1"] = "x-editor.workflow-arm-policy/v1"
    w0_planner_call_limit: Literal[1] = 1
    w1_planner_call_limit: int = Field(default=3, ge=1, le=10)
    w1_perception_call_limit: int = Field(
        default=2,
        ge=0,
        le=20,
        description=(
            "Maximum W1 Planner-to-Perception gateway rounds. One round may contain "
            "multiple independently audited perception requests. Zero disables additional "
            "gateway calls without disabling source bootstrap observations."
        ),
    )
    w1_stop_on_no_evidence_progress: bool = False
    w2_initial_perception_call_limit: Literal[1] = 1
    w2_planner_call_limit: Literal[1] = 1
    planner_internal_perception_guidance: Literal["disabled"] = "disabled"
    fallback_mode: Literal["disabled"] = "disabled"
    repair_mode: Literal["disabled"] = "disabled"
    evaluator_mode: Literal["disabled"] = "disabled"
    component_invocation_trace_mode: Literal["optional", "required"] = "optional"
    evidence_projection_mode: Literal["disabled", "reviewed"] = "disabled"
    planner_provider_call_limit_per_gateway: int = Field(
        default=1,
        ge=0,
        le=20,
    )
    reasoner_provider_call_limit_per_gateway: int = Field(
        default=1,
        ge=0,
        le=20,
    )
    perception_provider_call_limit_per_gateway: int = Field(
        default=8,
        ge=0,
        le=100,
    )
    provider_retry_mode: Literal["disabled"] = "disabled"
    json_repair_mode: Literal["disabled"] = "disabled"
    schema_repair_mode: Literal["disabled"] = "disabled"


class WorkflowArmRequest(StrictModel):
    """Complete input contract for one assigned workflow arm."""

    schema_version: Literal["x-editor.workflow-arm-request/v1"] = "x-editor.workflow-arm-request/v1"
    run_id: str = Field(default_factory=lambda: new_workflow_id("workflow_arm"))
    arm: WorkflowArm
    user_request: str
    source_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    reference_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    audio_resolution_mode: Literal["legacy_path", "opaque_ref_v1"] = "legacy_path"
    audio_bindings: list[RuntimeAudioBinding] = Field(
        default_factory=list,
        exclude=True,
    )
    runtime_output_dir: str | None = Field(default=None, exclude=True)
    source_text: str | None = None
    prior_turn_contexts: list[PriorTurnSemanticContext] = Field(
        default_factory=list,
        max_length=32,
    )
    session_state: SessionStateContext | None = None
    historical_reviewed_evidence: HistoricalReviewedEvidenceContext | None = Field(
        default=None,
        exclude=True,
    )
    future_attempt_guidance: FutureAttemptGuidance | None = Field(
        default=None,
        exclude=True,
    )
    planner_prompt_profile: str | None = None
    planning_context: dict[str, Any] = Field(default_factory=dict)
    audio_asset_facts: list[AudioAssetFacts] = Field(default_factory=list)
    execution_context: dict[str, Any] = Field(default_factory=dict)
    bootstrap_perception_requests: list[WorkflowPerceptionRequest] = Field(
        default_factory=list,
        max_length=16,
    )
    initial_perception_request: WorkflowPerceptionRequest | None = None
    event_log_path: str | None = Field(default=None, exclude=True)
    policy: WorkflowArmPolicy = Field(default_factory=WorkflowArmPolicy)
    forward_policy: ForwardExecutionPolicy = Field(default_factory=ForwardExecutionPolicy)

    @model_validator(mode="after")
    def validate_arm_inputs(self) -> WorkflowArmRequest:
        normalized_refs = [str(item).strip() for item in self.source_audio_refs]
        if any(not item for item in normalized_refs):
            raise ValueError("source_audio_refs cannot contain blank values")
        if len(normalized_refs) != len(set(normalized_refs)):
            raise ValueError("source_audio_refs must be unique")
        self.source_audio_refs = normalized_refs
        normalized_reference_refs = [str(item).strip() for item in self.reference_audio_refs]
        if any(not item for item in normalized_reference_refs):
            raise ValueError("reference_audio_refs cannot contain blank values")
        if len(normalized_reference_refs) != len(set(normalized_reference_refs)):
            raise ValueError("reference_audio_refs must be unique")
        self.reference_audio_refs = normalized_reference_refs
        if self.source_text is not None:
            self.source_text = self.source_text.strip() or None
        if self.historical_reviewed_evidence is not None and self.session_state is None:
            raise ValueError(
                "historical reviewed evidence requires authoritative Session Graph state"
            )
        bootstrap_ids = [item.request_id for item in self.bootstrap_perception_requests]
        if len(bootstrap_ids) != len(set(bootstrap_ids)):
            raise ValueError("bootstrap perception request ids must be unique")
        frozen_audio_refs = set(self.source_audio_refs) | set(self.reference_audio_refs)
        for item in self.bootstrap_perception_requests:
            if item.scope != "general":
                raise ValueError("bootstrap perception requests must use general scope")
            if item.operation is None:
                raise ValueError("bootstrap perception requests require an explicit operation")
            if len(item.audio_refs) != 1:
                raise ValueError("bootstrap perception requests must isolate exactly one audio ref")
            if frozen_audio_refs and not set(item.audio_refs).issubset(frozen_audio_refs):
                raise ValueError(
                    "bootstrap perception request is outside frozen source/reference audio refs"
                )
        if self.audio_resolution_mode == "opaque_ref_v1":
            expected_refs = set(self.source_audio_refs) | set(self.reference_audio_refs)
            bound_refs = [item.audio_ref for item in self.audio_bindings]
            if len(bound_refs) != len(set(bound_refs)):
                raise ValueError("runtime audio binding refs must be unique")
            binding_orders = [item.order for item in self.audio_bindings]
            if len(binding_orders) != len(set(binding_orders)):
                raise ValueError("runtime audio binding orders must be unique")
            if set(bound_refs) != expected_refs:
                raise ValueError("opaque workflow arm refs require one exact private binding each")
            by_ref = {item.audio_ref: item for item in self.audio_bindings}
            if any(
                by_ref[ref].role not in {"source", "authoritative_output"}
                for ref in self.source_audio_refs
            ):
                raise ValueError("opaque source refs have incompatible binding roles")
            if any(by_ref[ref].role != "reference" for ref in self.reference_audio_refs):
                raise ValueError("opaque reference refs have incompatible binding roles")
        if self.arm == "W2":
            if self.initial_perception_request is None:
                raise ValueError("W2 requires one explicit initial perception request")
            if self.initial_perception_request.scope != "general":
                raise ValueError("W2 initial perception request must use general scope")
            frozen_audio_refs = set(self.source_audio_refs) | set(self.reference_audio_refs)
            if frozen_audio_refs and not set(self.initial_perception_request.audio_refs).issubset(
                frozen_audio_refs
            ):
                raise ValueError(
                    "W2 initial perception request is outside frozen source/reference audio refs"
                )
        elif self.initial_perception_request is not None:
            raise ValueError("only W2 accepts an initial perception request")
        return self


class WorkflowPlannerCallRecord(StrictModel):
    """Immutable facts for one Planner turn."""

    call_index: int = Field(ge=1)
    call_id: str
    status: Literal["completed", "failed"]
    input_fingerprint: str
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_fingerprint: str
    decision: WorkflowPlannerDecision | None = None
    planner_route: Literal["direct", "deterministic_preflight", "fallback"] | None = None
    requested_perception_ids: list[str] = Field(default_factory=list)
    planned_perception_step_ids: list[str] = Field(default_factory=list)
    plan_fingerprint: str | None = None
    previous_plan_fingerprint: str | None = None
    plan_changed: bool | None = None
    plan_snapshot: AudioEditPlan | None = None
    invocation_trace: ComponentInvocationTrace | None = None
    error_type: str | None = None
    error_message: str | None = None


class WorkflowReasonerCallRecord(StrictModel):
    """One authoritative semantic Reasoner call preceding a separated Planner."""

    schema_version: Literal["x-editor.workflow-reasoner-call/v1"] = (
        "x-editor.workflow-reasoner-call/v1"
    )
    call_index: int = Field(ge=1)
    call_id: str
    status: Literal["completed", "failed"]
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    memo: dict[str, Any] | None = None
    invocation_trace: ComponentInvocationTrace | None = None
    error_type: str | None = None
    error_message: str | None = None


class WorkflowPerceptionCallRecord(StrictModel):
    """Immutable facts for one pre-plan perception call and its evidence delta."""

    call_index: int = Field(ge=1)
    call_id: str
    status: Literal["completed", "failed"]
    request: WorkflowPerceptionRequest
    evidence: WorkflowPerceptionEvidence | None = None
    evidence_ids_before: list[str] = Field(default_factory=list)
    evidence_ids_after: list[str] = Field(default_factory=list)
    evidence_fingerprint_before: str
    evidence_fingerprint_after: str
    content_progressed: bool = False
    invocation_trace: ComponentInvocationTrace | None = None
    tool_calls: list[WorkflowToolCallSummary] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class WorkflowArmResult(StrictModel):
    """Terminal orchestration facts with a nested authoritative forward result."""

    schema_version: Literal["x-editor.workflow-arm-result/v1"] = "x-editor.workflow-arm-result/v1"
    run_id: str
    assigned_arm: WorkflowArm
    source_audio_refs: list[str] = Field(default_factory=list)
    reference_audio_refs: list[str] = Field(default_factory=list)
    status: WorkflowOrchestrationStatus
    failure_stage: WorkflowOrchestrationFailureStage = "none"
    failure_codes: list[str] = Field(default_factory=list)
    terminal_error_type: str | None = None
    terminal_error_message: str | None = None
    policy: WorkflowArmPolicy
    forward_policy: ForwardExecutionPolicy
    planning_context_fingerprint: str
    execution_context_fingerprint: str
    actual_path: list[str] = Field(default_factory=list)
    final_plan: AudioEditPlan | None = None
    forward_result: ForwardExecutionResult | None = None
    reasoner_history: list[WorkflowReasonerCallRecord] = Field(default_factory=list)
    planner_history: list[WorkflowPlannerCallRecord] = Field(default_factory=list)
    perception_history: list[WorkflowPerceptionCallRecord] = Field(default_factory=list)
    planner_call_count: int = Field(default=0, ge=0)
    perception_gateway_round_count: int = Field(default=0, ge=0)
    orchestration_perception_call_count: int = Field(default=0, ge=0)
    planned_perception_step_ids: list[str] = Field(default_factory=list)
    planned_perception_steps: list[WorkflowPlannedPerceptionStep] = Field(default_factory=list)
    planned_perception_invocation_traces: list[ComponentInvocationTrace] = Field(
        default_factory=list
    )
    evidence_projection: WorkflowEvidenceProjection | None = None
    state_only_completion: WorkflowStateOnlyCompletion | None = None
    planner_logical_provider_call_count: int = Field(default=0, ge=0)
    reasoner_logical_provider_call_count: int = Field(default=0, ge=0)
    perception_logical_provider_call_count: int = Field(default=0, ge=0)
    planner_provider_call_count: int = Field(default=0, ge=0)
    reasoner_provider_call_count: int = Field(default=0, ge=0)
    perception_provider_call_count: int = Field(default=0, ge=0)
    perception_tool_call_count: int = Field(default=0, ge=0)
    perception_tool_attempt_count: int = Field(default=0, ge=0)
    provider_retry_count: int = Field(default=0, ge=0)
    logical_provider_retry_count: int = Field(default=0, ge=0)
    json_repair_count: int = Field(default=0, ge=0)
    schema_repair_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    evaluator_call_count: Literal[0] = 0
    reasoner_shadow_history: list[ReasonerShadowCallRecord] = Field(default_factory=list)
    reasoner_shadow_input_count: int = Field(default=0, ge=0)
    reasoner_shadow_logical_provider_call_count: int = Field(default=0, ge=0)
    reasoner_shadow_provider_call_count: int = Field(default=0, ge=0)
    reasoner_shadow_retry_count: int = Field(default=0, ge=0)
    reasoner_shadow_event_log_path: str | None = None
    reasoner_shadow_input_bundle_path: str | None = None
    reasoner_shadow_observation_errors: list[dict[str, Any]] = Field(default_factory=list)
    reasoner_shadow_inputs: list[ReasonerShadowInputSnapshot] = Field(
        default_factory=list,
        exclude=True,
    )
    event_log_path: str | None = None
    module_checkpoint_path: str | None = None
    module_checkpoint_count: int = Field(default=0, ge=0)
    module_checkpoint_errors: list[dict[str, str]] = Field(default_factory=list)
    events: list[WorkflowEvent] = Field(default_factory=list)
    observation_errors: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state_only_terminal(self) -> WorkflowArmResult:
        if self.state_only_completion is None:
            return self
        if self.status != "completed" or self.failure_stage != "none" or self.failure_codes:
            raise ValueError("state-only completion must be a successful terminal result")
        if self.final_plan is not None or self.forward_result is not None:
            raise ValueError("state-only completion cannot include plan or forward execution")
        return self


@dataclass
class _OrchestrationState:
    request: WorkflowArmRequest
    recorder: ExecutionEventRecorder
    checkpoint_observer: ExecutionEventObserver | None = None
    evidence: list[WorkflowPerceptionEvidence] = field(default_factory=list)
    reasoner_history: list[WorkflowReasonerCallRecord] = field(default_factory=list)
    planner_history: list[WorkflowPlannerCallRecord] = field(default_factory=list)
    perception_history: list[WorkflowPerceptionCallRecord] = field(default_factory=list)
    perception_gateway_round_count: int = 0
    capability_feedback: PerceptionCapabilityFeedback | None = None
    reasoner_shadow_inputs: list[ReasonerShadowInputSnapshot] = field(default_factory=list)
    reasoner_shadow_capture_errors: list[dict[str, Any]] = field(default_factory=list)
    actual_path: list[str] = field(default_factory=list)
    frozen_edit_intent: dict[str, Any] | None = None
    final_plan: AudioEditPlan | None = None
    planned_perception_steps: list[WorkflowPlannedPerceptionStep] = field(default_factory=list)
    evidence_projection: WorkflowEvidenceProjection | None = None
    state_only_completion: WorkflowStateOnlyCompletion | None = None
    forward_result: ForwardExecutionResult | None = None


class _WorkflowAbort(RuntimeError):
    def __init__(
        self,
        *,
        status: WorkflowOrchestrationStatus,
        stage: WorkflowOrchestrationFailureStage,
        codes: list[str],
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(", ".join(codes))
        self.status = status
        self.stage = stage
        self.codes = codes
        self.error_type = error_type
        self.error_message = error_message


class WorkflowArmOrchestrator:
    """Run one W0/W1/W2 arm and delegate only its frozen plan downstream."""

    def __init__(
        self,
        *,
        planner: WorkflowPlanner,
        perception: WorkflowPerceptionGateway,
        forward_path: ForwardExecutionPath,
        evidence_projector: WorkflowEvidenceProjector | None = None,
    ) -> None:
        self.planner = planner
        self.perception = perception
        self.forward_path = forward_path
        self.evidence_projector = evidence_projector

    def run(
        self,
        request: WorkflowArmRequest,
        *,
        observer: ExecutionEventObserver | None = None,
    ) -> WorkflowArmResult:
        return asyncio.run(self.arun(request, observer=observer))

    async def arun(
        self,
        request: WorkflowArmRequest,
        *,
        observer: ExecutionEventObserver | None = None,
    ) -> WorkflowArmResult:
        recorder = ExecutionEventRecorder(
            run_id=request.run_id,
            event_log_path=request.event_log_path,
            observer=observer,
        )
        state = _OrchestrationState(
            request=request,
            recorder=recorder,
            checkpoint_observer=observer,
        )
        recorder.emit(
            stage="runner",
            kind="workflow_arm_started",
            status="started",
            subject_refs=[request.arm],
            payload={
                "assigned_arm": request.arm,
                "policy": request.policy.model_dump(mode="json"),
                "forward_policy": request.forward_policy.model_dump(mode="json"),
                "evaluator_calls": 0,
            },
        )
        try:
            self._validate_configuration(request)
            await self._run_bootstrap_perception(state)
            self._refresh_evidence_projection(state)
            if request.arm == "W0":
                plan = await self._run_w0(state)
            elif request.arm == "W1":
                plan = await self._run_w1(state)
            else:
                plan = await self._run_w2(state)
            if state.state_only_completion is not None:
                completion = state.state_only_completion
                state.actual_path.extend(
                    [
                        f"session_state:{completion.selection_kind}",
                        "forward:skipped_state_only",
                    ]
                )
                recorder.emit(
                    stage="runner",
                    kind="workflow_state_only_completion_accepted",
                    status="completed",
                    subject_refs=[completion.selected_node_id, completion.selected_audio_ref],
                    payload={
                        "assigned_arm": request.arm,
                        "state_only_completion": completion.model_dump(mode="json"),
                        "actual_path": list(state.actual_path),
                        "evaluator_calls": 0,
                        "forward_execution_skipped": True,
                    },
                )
                recorder.emit(
                    stage="runner",
                    kind="workflow_arm_completed",
                    status="completed",
                    subject_refs=[request.arm],
                    payload={
                        "assigned_arm": request.arm,
                        "actual_path": list(state.actual_path),
                        "failure_stage": "none",
                        "failure_codes": [],
                        "evaluator_calls": 0,
                    },
                )
                return self._result(
                    state,
                    status="completed",
                    failure_stage="none",
                    failure_codes=[],
                )
            assert plan is not None
            _validate_plan_source_audio_refs(
                plan,
                request.source_audio_refs,
                reference_audio_refs=request.reference_audio_refs,
                session_state=request.session_state,
            )
            state.final_plan = plan.model_copy(deep=True)
            planned_perception_ids = _planned_perception_step_ids(plan)
            recorder.emit(
                stage="planner",
                kind="workflow_plan_frozen",
                status="accepted",
                subject_refs=[step.id for step in plan.steps],
                payload={
                    "assigned_arm": request.arm,
                    "plan": plan.model_dump(mode="json"),
                    "plan_fingerprint": _fingerprint(plan),
                    "planned_perception_step_ids": planned_perception_ids,
                    "planned_perception_steps": [
                        item.model_dump(mode="json") for item in state.planned_perception_steps
                    ],
                    "forward_run_id": f"{request.run_id}:forward",
                },
            )
            state.actual_path.append("forward:started")
            forward_run_id = f"{request.run_id}:forward"
            execution_context = self._execution_context(state)
            try:
                state.forward_result = await self.forward_path.arun(
                    ForwardExecutionRequest(
                        run_id=forward_run_id,
                        plan=state.final_plan,
                        context=execution_context,
                        session_state=request.session_state,
                        audio_resolution_mode=request.audio_resolution_mode,
                        audio_bindings=[
                            item.model_copy(deep=True) for item in request.audio_bindings
                        ],
                        runtime_output_dir=request.runtime_output_dir,
                        policy=request.forward_policy,
                        planned_perception_invocation_policy=(
                            _component_admission_policy(
                                request.policy,
                                component="perception",
                            )
                        ),
                        planned_perception_evidence_mode=(request.policy.evidence_projection_mode),
                    ),
                    observer=recorder,
                )
            except Exception as exc:
                state.actual_path.append("forward:raised")
                recorder.emit(
                    stage="runner",
                    kind="workflow_forward_execution_raised",
                    status="failed",
                    subject_refs=[forward_run_id],
                    payload={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                raise _WorkflowAbort(
                    status="failed",
                    stage="forward_execution",
                    codes=["forward_execution_raised"],
                    error_type=type(exc).__name__,
                    error_message=_safe_error_message(exc),
                ) from exc
            state.actual_path.extend(_forward_step_path(state.forward_result))
            state.actual_path.append(f"forward:{state.forward_result.status}")
            recorder.emit(
                stage="runner",
                kind={
                    "completed": "workflow_arm_completed",
                    "blocked": "workflow_arm_blocked",
                    "failed": "workflow_arm_failed",
                }[state.forward_result.status],
                status=("completed" if state.forward_result.status == "completed" else "failed"),
                subject_refs=[request.arm],
                payload={
                    "assigned_arm": request.arm,
                    "actual_path": list(state.actual_path),
                    "failure_stage": state.forward_result.failure_stage,
                    "failure_codes": list(state.forward_result.failure_codes),
                    "evaluator_calls": 0,
                },
            )
            return self._result(
                state,
                status=state.forward_result.status,
                failure_stage=state.forward_result.failure_stage,
                failure_codes=state.forward_result.failure_codes,
            )
        except _WorkflowAbort as exc:
            recorder.emit(
                stage="runner",
                kind=("workflow_arm_blocked" if exc.status == "blocked" else "workflow_arm_failed"),
                status="failed",
                subject_refs=[request.arm],
                payload={
                    "assigned_arm": request.arm,
                    "actual_path": list(state.actual_path),
                    "failure_stage": exc.stage,
                    "failure_codes": exc.codes,
                    "terminal_error_type": exc.error_type,
                    "terminal_error_message": exc.error_message,
                    "evaluator_calls": 0,
                },
            )
            return self._result(
                state,
                status=exc.status,
                failure_stage=exc.stage,
                failure_codes=exc.codes,
                terminal_error_type=exc.error_type,
                terminal_error_message=exc.error_message,
            )

    async def _run_bootstrap_perception(self, state: _OrchestrationState) -> None:
        """Acquire explicit framework-owned source observations before planning."""

        requests = state.request.bootstrap_perception_requests
        if not requests:
            return
        request_ids = [item.request_id for item in requests]
        state.recorder.emit(
            stage="perception",
            kind="workflow_bootstrap_perception_gateway_started",
            status="started",
            attempt_id="perception_bootstrap:1",
            subject_refs=request_ids,
            payload={
                "assigned_arm": state.request.arm,
                "request_count": len(requests),
                "request_ids": request_ids,
            },
        )
        progressed = False
        for perception_request in requests:
            content_before = _workflow_evidence_content_fingerprint(state)
            await self._call_perception(state, perception_request)
            content_after = _workflow_evidence_content_fingerprint(state)
            progressed = progressed or content_after != content_before
        state.recorder.emit(
            stage="perception",
            kind="workflow_bootstrap_perception_gateway_completed",
            status="completed",
            attempt_id="perception_bootstrap:1",
            subject_refs=request_ids,
            payload={
                "assigned_arm": state.request.arm,
                "request_count": len(requests),
                "request_ids": request_ids,
                "content_progressed": progressed,
            },
        )
        if not progressed:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["bootstrap_perception_made_no_evidence_progress"],
            )

    async def _run_w0(self, state: _OrchestrationState) -> AudioEditPlan | None:
        output = await self._call_planner(state)
        if output.decision == "cannot_plan":
            self._abort_cannot_plan(output)
        if output.decision == "complete_from_session_state":
            assert output.state_only_completion is not None
            state.state_only_completion = output.state_only_completion.model_copy(deep=True)
            return None
        if output.decision != "execute" or output.plan is None:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["w0_external_perception_loop_forbidden"],
            )
        if output.planned_perception_steps:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["w0_w2_gap_declarations_forbidden"],
            )
        topology_codes = _planned_perception_dependency_codes(
            output.plan,
            set(_planned_perception_step_ids(output.plan)),
            arm="w0",
        )
        if topology_codes:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=topology_codes,
            )
        return output.plan

    async def _run_w1(self, state: _OrchestrationState) -> AudioEditPlan | None:
        policy = state.request.policy
        for planner_index in range(1, policy.w1_planner_call_limit + 1):
            output = await self._call_planner(state)
            state.capability_feedback = None
            if output.frozen_edit_intent is not None:
                if state.frozen_edit_intent is None:
                    state.frozen_edit_intent = copy.deepcopy(output.frozen_edit_intent)
                elif not _edit_intent_preserves_frozen_semantics(
                    state.frozen_edit_intent,
                    output.frozen_edit_intent,
                    allow_effect_revision=(
                        state.request.planner_prompt_profile
                        in {
                            "robust_contract_v4",
                            "robust_contract_v5",
                        }
                    ),
                ):
                    raise _WorkflowAbort(
                        status="blocked",
                        stage="workflow_protocol",
                        codes=["w1_frozen_edit_intent_changed"],
                    )
            if output.decision == "execute" and output.plan is not None:
                if _planned_perception_step_ids(output.plan):
                    raise _WorkflowAbort(
                        status="blocked",
                        stage="workflow_protocol",
                        codes=["w1_final_plan_contains_perception_steps"],
                    )
                return output.plan
            if output.decision == "complete_from_session_state":
                assert output.state_only_completion is not None
                state.state_only_completion = output.state_only_completion.model_copy(deep=True)
                return None
            if output.decision == "cannot_plan":
                self._abort_cannot_plan(output)
            if planner_index == policy.w1_planner_call_limit:
                raise _WorkflowAbort(
                    status="blocked",
                    stage="workflow_protocol",
                    codes=["w1_planner_loop_limit_reached"],
                )
            if output.decision == "request_replan":
                assert output.capability_feedback is not None
                state.capability_feedback = output.capability_feedback.model_copy(deep=True)
                state.recorder.emit(
                    stage="planner",
                    kind="workflow_perception_capability_feedback",
                    status="observed",
                    attempt_id=f"planner:{planner_index}",
                    payload=state.capability_feedback.model_dump(mode="json"),
                )
                continue
            gateway_round = state.perception_gateway_round_count + 1
            if gateway_round > policy.w1_perception_call_limit:
                raise _WorkflowAbort(
                    status="blocked",
                    stage="workflow_protocol",
                    codes=["w1_perception_call_limit_exceeded"],
                )
            if len(_bound_audio_refs(state.request)) > 1 and any(
                not item.audio_refs for item in output.perception_requests
            ):
                raise _WorkflowAbort(
                    status="blocked",
                    stage="workflow_protocol",
                    codes=["multi_audio_perception_request_missing_audio_refs"],
                )
            state.perception_gateway_round_count = gateway_round
            request_ids = [item.request_id for item in output.perception_requests]
            state.recorder.emit(
                stage="perception",
                kind="workflow_perception_gateway_round_started",
                status="started",
                attempt_id=f"perception_gateway:{gateway_round}",
                subject_refs=request_ids,
                payload={
                    "assigned_arm": state.request.arm,
                    "gateway_round": gateway_round,
                    "request_count": len(request_ids),
                    "request_ids": request_ids,
                },
            )
            progressed = False
            for perception_request in output.perception_requests:
                content_before = _workflow_evidence_content_fingerprint(state)
                try:
                    await self._call_perception(
                        state,
                        perception_request,
                    )
                except _WorkflowAbort as exc:
                    decomposition = _perception_decomposition_metadata(
                        perception_request
                    )
                    if decomposition is None:
                        raise
                    # A gateway decomposition is one collective observation job.
                    # Preserve the failed sub-route, then let sibling routes run so
                    # the next Planner call can judge the evidence actually obtained.
                    state.recorder.emit(
                        stage="perception",
                        kind="workflow_perception_decomposition_part_failed",
                        status="observed",
                        attempt_id=f"perception_gateway:{gateway_round}",
                        subject_refs=[
                            str(decomposition["source_job_id"]),
                            perception_request.request_id,
                        ],
                        payload={
                            "assigned_arm": state.request.arm,
                            "gateway_round": gateway_round,
                            "source_job_id": decomposition["source_job_id"],
                            "part_index": decomposition["part_index"],
                            "part_count": decomposition["part_count"],
                            "request_id": perception_request.request_id,
                            "failure_stage": exc.stage,
                            "failure_codes": list(exc.codes),
                            "next_action": "continue_collective_observation_job",
                            "admission_and_verifier_bypass": False,
                        },
                    )
                content_after = _workflow_evidence_content_fingerprint(state)
                progressed = progressed or (content_after != content_before)
            state.recorder.emit(
                stage="perception",
                kind="workflow_perception_gateway_round_completed",
                status="completed",
                attempt_id=f"perception_gateway:{gateway_round}",
                subject_refs=request_ids,
                payload={
                    "assigned_arm": state.request.arm,
                    "gateway_round": gateway_round,
                    "request_count": len(request_ids),
                    "request_ids": request_ids,
                    "content_progressed": progressed,
                },
            )
            if not progressed:
                state.recorder.emit(
                    stage="perception",
                    kind="workflow_perception_gateway_no_new_evidence",
                    status="observed",
                    attempt_id=f"perception_gateway:{gateway_round}",
                    subject_refs=request_ids,
                    payload={
                        "assigned_arm": state.request.arm,
                        "gateway_round": gateway_round,
                        "request_ids": request_ids,
                        "next_action": (
                            "stop_by_policy"
                            if policy.w1_stop_on_no_evidence_progress
                            else "return_existing_evidence_to_planner"
                        ),
                    },
                )
                if policy.w1_stop_on_no_evidence_progress:
                    raise _WorkflowAbort(
                        status="blocked",
                        stage="workflow_protocol",
                        codes=["w1_perception_made_no_evidence_progress"],
                    )
        raise _WorkflowAbort(
            status="blocked",
            stage="workflow_protocol",
            codes=["w1_planner_loop_limit_reached"],
        )

    async def _run_w2(self, state: _OrchestrationState) -> AudioEditPlan | None:
        initial_request = state.request.initial_perception_request
        assert initial_request is not None
        await self._call_perception(state, initial_request)
        output = await self._call_planner(state)
        if output.decision == "cannot_plan":
            self._abort_cannot_plan(output)
        if output.decision == "complete_from_session_state":
            assert output.state_only_completion is not None
            state.state_only_completion = output.state_only_completion.model_copy(deep=True)
            return None
        if output.decision != "execute" or output.plan is None:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["w2_external_perception_loop_forbidden"],
            )
        planned_step_ids = set(_planned_perception_step_ids(output.plan))
        declared_step_ids = {item.step_id for item in output.planned_perception_steps}
        if planned_step_ids != declared_step_ids:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["w2_perception_gap_declarations_incomplete"],
            )
        evidence_ids = {item.evidence_id for item in state.evidence}
        if any(
            not set(item.based_on_evidence_ids).issubset(evidence_ids)
            for item in output.planned_perception_steps
        ):
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["w2_gap_declaration_evidence_unknown"],
            )
        topology_codes = _planned_perception_dependency_codes(
            output.plan,
            planned_step_ids,
            arm="w2",
        )
        if topology_codes:
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=topology_codes,
            )
        state.planned_perception_steps = [
            item.model_copy(deep=True) for item in output.planned_perception_steps
        ]
        return output.plan

    async def _call_planner(
        self,
        state: _OrchestrationState,
    ) -> WorkflowPlannerOutput:
        call_index = len(state.planner_history) + 1
        call_id = f"planner:{call_index}"
        context = self._planner_context(state, call_index=call_index)
        input_fingerprint = _fingerprint(context)
        evidence_ids = [item.evidence_id for item in state.evidence]
        evidence_fingerprint = _fingerprint(
            [item.model_dump(mode="json") for item in state.evidence]
        )
        planner_started_event = state.recorder.emit(
            stage="planner",
            kind="workflow_planner_call_started",
            status="started",
            attempt_id=call_id,
            subject_refs=evidence_ids,
            payload={
                "assigned_arm": state.request.arm,
                "call_index": call_index,
                "input_fingerprint": input_fingerprint,
                "evidence_ids": evidence_ids,
                "evidence_fingerprint": evidence_fingerprint,
            },
        )
        try:
            state.reasoner_shadow_inputs.append(
                _reasoner_shadow_input_snapshot(
                    state,
                    context=context,
                    call_index=call_index,
                    planner_input_fingerprint=input_fingerprint,
                    planner_started_event=planner_started_event,
                )
            )
        except Exception as exc:  # noqa: BLE001 - observation cannot own control
            state.reasoner_shadow_capture_errors.append(
                {
                    "component": "reasoner_shadow_input_tap",
                    "operation": "capture",
                    "error_type": type(exc).__name__,
                    "error_message": _safe_error_message(exc),
                    "planner_call_index": call_index,
                }
            )
        try:
            raw_output = await _invoke_component_callable(
                self.planner.plan,
                state.request.user_request,
                copy.deepcopy(context),
            )
        except Exception as exc:
            _capture_authoritative_reasoner_record(
                state,
                planner=self.planner,
                call_index=call_index,
            )
            invocation_trace = _component_trace_from_error_or_component(
                exc,
                self.planner,
            )
            record = WorkflowPlannerCallRecord(
                call_index=call_index,
                call_id=call_id,
                status="failed",
                input_fingerprint=input_fingerprint,
                evidence_ids=evidence_ids,
                evidence_fingerprint=evidence_fingerprint,
                invocation_trace=invocation_trace,
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )
            state.planner_history.append(record)
            state.actual_path.append(f"planner:{call_index}:failed")
            state.recorder.emit(
                stage="planner",
                kind="workflow_planner_call_failed",
                status="failed",
                attempt_id=call_id,
                payload={"record": record.model_dump(mode="json")},
            )
            reasoner_record = next(
                (
                    item
                    for item in reversed(state.reasoner_history)
                    if item.call_index == call_index and item.status == "failed"
                ),
                None,
            )
            if reasoner_record is not None:
                contract_rejection = "ContractError" in str(reasoner_record.error_type or "")
                completion_budget_exhausted = (
                    reasoner_record.error_type == "OpenAICompatibleReasoningNonConvergenceError"
                )
                raise _WorkflowAbort(
                    status="blocked" if contract_rejection else "failed",
                    stage="reasoner",
                    codes=[
                        (
                            "reasoner_contract_rejected"
                            if contract_rejection
                            else "reasoner_completion_budget_exhausted"
                            if completion_budget_exhausted
                            else "reasoner_raised"
                        )
                    ],
                    error_type=reasoner_record.error_type,
                    error_message=reasoner_record.error_message,
                ) from exc
            protocol_rejection = getattr(exc, "failure_kind", None) == "protocol"
            raise _WorkflowAbort(
                status="blocked" if protocol_rejection else "failed",
                stage=("workflow_protocol" if protocol_rejection else "planner"),
                codes=["planner_contract_rejected" if protocol_rejection else "planner_raised"],
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            ) from exc
        _capture_authoritative_reasoner_record(
            state,
            planner=self.planner,
            call_index=call_index,
        )
        reasoner_record = next(
            (item for item in reversed(state.reasoner_history) if item.call_index == call_index),
            None,
        )
        if reasoner_record is not None:
            reasoner_blockers = _component_trace_blockers(
                state.request.policy,
                reasoner_record.invocation_trace,
                component="reasoner",
                gateway_call_id=f"reasoner:{call_index}",
            )
            if reasoner_blockers:
                state.actual_path.append(f"reasoner:{call_index}:invocation_rejected")
                state.recorder.emit(
                    stage="reasoner",
                    kind="workflow_reasoner_invocation_rejected",
                    status="rejected",
                    attempt_id=f"reasoner:{call_index}",
                    payload={
                        "record": reasoner_record.model_dump(mode="json"),
                        "failure_codes": reasoner_blockers,
                    },
                )
                raise _WorkflowAbort(
                    status="blocked",
                    stage="workflow_protocol",
                    codes=reasoner_blockers,
                )
        try:
            output = _normalize_planner_output(raw_output)
        except Exception as exc:
            invocation_trace = _component_trace_from_error_or_component(
                exc,
                self.planner,
            )
            record = WorkflowPlannerCallRecord(
                call_index=call_index,
                call_id=call_id,
                status="failed",
                input_fingerprint=input_fingerprint,
                evidence_ids=evidence_ids,
                evidence_fingerprint=evidence_fingerprint,
                invocation_trace=invocation_trace,
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )
            state.planner_history.append(record)
            state.actual_path.append(f"planner:{call_index}:invalid")
            state.recorder.emit(
                stage="planner",
                kind="workflow_planner_output_rejected",
                status="rejected",
                attempt_id=call_id,
                payload={"record": record.model_dump(mode="json")},
            )
            raise _WorkflowAbort(
                status="failed",
                stage="workflow_protocol",
                codes=["planner_output_invalid"],
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            ) from exc
        planner_route = _planner_route(
            output.plan,
            output.invocation_trace,
        )
        plan_fingerprint = _fingerprint(output.plan) if output.plan else None
        previous_plan_fingerprint = next(
            (
                item.plan_fingerprint
                for item in reversed(state.planner_history)
                if item.plan_fingerprint is not None
            ),
            None,
        )
        record = WorkflowPlannerCallRecord(
            call_index=call_index,
            call_id=call_id,
            status="completed",
            input_fingerprint=input_fingerprint,
            evidence_ids=evidence_ids,
            evidence_fingerprint=evidence_fingerprint,
            decision=output.decision,
            planner_route=planner_route,
            requested_perception_ids=[item.request_id for item in output.perception_requests],
            planned_perception_step_ids=[item.step_id for item in output.planned_perception_steps],
            plan_fingerprint=plan_fingerprint,
            previous_plan_fingerprint=previous_plan_fingerprint,
            plan_changed=(
                None
                if plan_fingerprint is None or previous_plan_fingerprint is None
                else plan_fingerprint != previous_plan_fingerprint
            ),
            plan_snapshot=(output.plan.model_copy(deep=True) if output.plan is not None else None),
            invocation_trace=(
                output.invocation_trace.model_copy(deep=True)
                if output.invocation_trace is not None
                else None
            ),
        )
        state.planner_history.append(record)
        invocation_blockers = _component_trace_blockers(
            state.request.policy,
            output.invocation_trace,
            component="planner",
            gateway_call_id=call_id,
        )
        if invocation_blockers:
            state.actual_path.append(f"planner:{call_index}:invocation_rejected")
            state.recorder.emit(
                stage="planner",
                kind="workflow_planner_invocation_rejected",
                status="rejected",
                attempt_id=call_id,
                payload={
                    "record": record.model_dump(mode="json"),
                    "failure_codes": invocation_blockers,
                },
            )
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=invocation_blockers,
            )
        if planner_route == "fallback":
            state.actual_path.append(f"planner:{call_index}:fallback_rejected")
            state.recorder.emit(
                stage="planner",
                kind="workflow_planner_fallback_rejected",
                status="rejected",
                attempt_id=call_id,
                payload={"record": record.model_dump(mode="json")},
            )
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["planner_fallback_forbidden"],
            )
        state.actual_path.append(f"planner:{call_index}:{output.decision}")
        separated_planner = any(item.call_index == call_index for item in state.reasoner_history)
        state.recorder.emit(
            stage="planner",
            kind=(
                "workflow_separated_planner_call_completed"
                if separated_planner
                else "workflow_planner_call_completed"
            ),
            status="completed",
            attempt_id=call_id,
            subject_refs=record.requested_perception_ids,
            payload={"record": record.model_dump(mode="json")},
        )
        return output

    async def _call_perception(
        self,
        state: _OrchestrationState,
        request: WorkflowPerceptionRequest,
    ) -> WorkflowPerceptionEvidence:
        bound_audio_refs = _bound_audio_refs(state.request)
        if bound_audio_refs and not set(request.audio_refs).issubset(set(bound_audio_refs)):
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=["perception_request_audio_ref_outside_bound_inputs"],
            )
        if any(item.request.request_id == request.request_id for item in state.perception_history):
            raise _WorkflowAbort(
                status="failed",
                stage="workflow_protocol",
                codes=[f"duplicate_perception_request_id:{request.request_id}"],
            )
        call_index = len(state.perception_history) + 1
        call_id = f"perception:{call_index}"
        before_ids = [item.evidence_id for item in state.evidence]
        before_fingerprint = _fingerprint([item.model_dump(mode="json") for item in state.evidence])
        state.recorder.emit(
            stage="perception",
            kind="workflow_perception_call_started",
            status="started",
            attempt_id=call_id,
            subject_refs=[request.request_id],
            payload={
                "assigned_arm": state.request.arm,
                "call_index": call_index,
                "request": request.model_dump(mode="json"),
                "evidence_ids_before": before_ids,
                "evidence_fingerprint_before": before_fingerprint,
            },
        )
        context = self._perception_context(state, call_index=call_index)
        try:
            raw_evidence = await _invoke_component_callable(
                self.perception.perceive,
                request.model_copy(deep=True),
                context=copy.deepcopy(context),
            )
        except Exception as exc:
            invocation_trace = _component_trace_from_error_or_component(
                exc,
                self.perception,
            )
            tool_calls = _tool_calls_from_error_or_component(
                exc,
                self.perception,
            )
            self._record_perception_failure(
                state,
                request=request,
                call_index=call_index,
                call_id=call_id,
                before_ids=before_ids,
                before_fingerprint=before_fingerprint,
                error=exc,
                invocation_trace=invocation_trace,
                tool_calls=tool_calls,
            )
            if getattr(exc, "failure_kind", None) == "protocol":
                raise _WorkflowAbort(
                    status="failed",
                    stage="workflow_protocol",
                    codes=["perception_output_invalid"],
                    error_type=type(exc).__name__,
                    error_message=_safe_error_message(exc),
                ) from exc
            raise _WorkflowAbort(
                status="failed",
                stage="perception",
                codes=["perception_raised"],
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            ) from exc
        evidence: WorkflowPerceptionEvidence | None = None
        try:
            evidence = _normalize_perception_evidence(raw_evidence)
            if (
                len(request.audio_refs) == 1
                and evidence.facts
                and not evidence.requested_audio_refs
                and not evidence.audio_refs
                and not evidence.missing_audio_evidence
            ):
                # Legacy single-audio gateways predate explicit per-audio
                # coverage. Binding facts to the sole requested asset is
                # unambiguous; multi-audio evidence must always declare its
                # partitions and is never inferred here.
                evidence = evidence.model_copy(
                    update={
                        "requested_audio_refs": list(request.audio_refs),
                        "audio_refs": list(request.audio_refs),
                    },
                    deep=True,
                )
            if evidence.request_id != request.request_id:
                raise ValueError("perception evidence request_id does not match request")
            if evidence.scope != request.scope:
                raise ValueError("perception evidence scope does not match request")
            if evidence.requested_audio_refs != request.audio_refs:
                raise ValueError(
                    "perception evidence requested_audio_refs do not match the request"
                )
            if not set(evidence.audio_refs).issubset(set(request.audio_refs)):
                raise ValueError("perception evidence covered audio_refs exceed the request")
            if any(item.evidence_id == evidence.evidence_id for item in state.evidence):
                raise ValueError("perception evidence_id must be unique within a run")
            if not evidence.usable:
                raise ValueError("required perception evidence is unusable")
            if state.request.policy.evidence_projection_mode == "reviewed":
                reviewed_facts = sanitize_reviewed_perception_facts(
                    evidence.facts,
                    audio_refs=evidence.requested_audio_refs,
                )
                if not reviewed_facts:
                    raise ValueError("perception evidence contains no reviewed facts")
                evidence = evidence.model_copy(
                    update={"facts": reviewed_facts},
                    deep=True,
                )
        except Exception as exc:
            invocation_trace = (
                evidence.invocation_trace
                if evidence is not None
                else _component_trace_from_error_or_component(
                    exc,
                    self.perception,
                )
            )
            tool_calls = (
                evidence.tool_calls
                if evidence is not None
                else _tool_calls_from_error_or_component(
                    exc,
                    self.perception,
                )
            )
            self._record_perception_failure(
                state,
                request=request,
                call_index=call_index,
                call_id=call_id,
                before_ids=before_ids,
                before_fingerprint=before_fingerprint,
                error=exc,
                invocation_trace=invocation_trace,
                tool_calls=tool_calls,
            )
            raise _WorkflowAbort(
                status="failed",
                stage="workflow_protocol",
                codes=["perception_output_invalid"],
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            ) from exc
        invocation_blockers = _component_trace_blockers(
            state.request.policy,
            evidence.invocation_trace,
            component="perception",
            gateway_call_id=call_id,
        )
        if invocation_blockers:
            rejection = RuntimeError(", ".join(invocation_blockers))
            self._record_perception_failure(
                state,
                request=request,
                call_index=call_index,
                call_id=call_id,
                before_ids=before_ids,
                before_fingerprint=before_fingerprint,
                error=rejection,
                invocation_trace=evidence.invocation_trace,
                tool_calls=evidence.tool_calls,
            )
            state.actual_path.append(f"perception:{request.request_id}:invocation_rejected")
            state.recorder.emit(
                stage="perception",
                kind="workflow_perception_invocation_rejected",
                status="rejected",
                attempt_id=call_id,
                subject_refs=[request.request_id],
                payload={"failure_codes": invocation_blockers},
            )
            raise _WorkflowAbort(
                status="blocked",
                stage="workflow_protocol",
                codes=invocation_blockers,
            )
        prior_content = {_evidence_content_fingerprint(item) for item in state.evidence}
        content_progressed = _evidence_content_fingerprint(evidence) not in prior_content
        state.evidence.append(evidence.model_copy(deep=True))
        after_ids = [item.evidence_id for item in state.evidence]
        after_fingerprint = _fingerprint([item.model_dump(mode="json") for item in state.evidence])
        record = WorkflowPerceptionCallRecord(
            call_index=call_index,
            call_id=call_id,
            status="completed",
            request=request,
            evidence=evidence,
            evidence_ids_before=before_ids,
            evidence_ids_after=after_ids,
            evidence_fingerprint_before=before_fingerprint,
            evidence_fingerprint_after=after_fingerprint,
            content_progressed=content_progressed,
            invocation_trace=(
                evidence.invocation_trace.model_copy(deep=True)
                if evidence.invocation_trace is not None
                else None
            ),
            tool_calls=[item.model_copy(deep=True) for item in evidence.tool_calls],
        )
        state.perception_history.append(record)
        state.actual_path.append(f"perception:{request.request_id}:completed")
        state.recorder.emit(
            stage="perception",
            kind="workflow_perception_call_completed",
            status="completed",
            attempt_id=call_id,
            subject_refs=[request.request_id, evidence.evidence_id],
            payload={"record": record.model_dump(mode="json")},
        )
        self._refresh_evidence_projection(state)
        return evidence

    def _record_perception_failure(
        self,
        state: _OrchestrationState,
        *,
        request: WorkflowPerceptionRequest,
        call_index: int,
        call_id: str,
        before_ids: list[str],
        before_fingerprint: str,
        error: Exception,
        invocation_trace: ComponentInvocationTrace | None = None,
        tool_calls: list[WorkflowToolCallSummary] | None = None,
    ) -> None:
        record = WorkflowPerceptionCallRecord(
            call_index=call_index,
            call_id=call_id,
            status="failed",
            request=request,
            evidence_ids_before=before_ids,
            evidence_ids_after=before_ids,
            evidence_fingerprint_before=before_fingerprint,
            evidence_fingerprint_after=before_fingerprint,
            error_type=type(error).__name__,
            error_message=_safe_error_message(error),
            invocation_trace=(
                invocation_trace.model_copy(deep=True) if invocation_trace is not None else None
            ),
            tool_calls=[item.model_copy(deep=True) for item in (tool_calls or [])],
        )
        state.perception_history.append(record)
        state.actual_path.append(f"perception:{request.request_id}:failed")
        state.recorder.emit(
            stage="perception",
            kind="workflow_perception_call_failed",
            status="failed",
            attempt_id=call_id,
            subject_refs=[request.request_id],
            payload={"record": record.model_dump(mode="json")},
        )

    def _planner_context(
        self,
        state: _OrchestrationState,
        *,
        call_index: int,
    ) -> dict[str, Any]:
        context = copy.deepcopy(state.request.planning_context)
        context.update(
            {
                "enable_planner_perception_guidance": False,
                "audio_asset_facts": [
                    item.model_dump(mode="json") for item in state.request.audio_asset_facts
                ],
                "workflow_run_id": state.request.run_id,
                "workflow_arm": state.request.arm,
                "workflow_planner_call_index": call_index,
                "workflow_perception_evidence_ids": [item.evidence_id for item in state.evidence],
                "workflow_source_audio_refs": list(state.request.source_audio_refs),
                "workflow_reference_audio_refs": list(state.request.reference_audio_refs),
                "workflow_prior_turn_semantic_context": [
                    item.model_dump(mode="json") for item in state.request.prior_turn_contexts
                ],
                "workflow_session_state": (
                    state.request.session_state.model_dump(mode="json")
                    if state.request.session_state is not None
                    else None
                ),
            }
        )
        if state.request.source_audio_refs:
            source_refs = list(state.request.source_audio_refs)
            context["input_audio"] = source_refs[0]
            context["input_audios"] = source_refs
            context["input_audio_paths"] = source_refs
        if state.request.reference_audio_refs:
            context["reference_audio_paths"] = list(state.request.reference_audio_refs)
        if state.request.source_text:
            context["source_text"] = state.request.source_text
        if state.request.historical_reviewed_evidence is not None:
            context["workflow_historical_reviewed_evidence"] = (
                state.request.historical_reviewed_evidence.model_dump(mode="json")
            )
        if state.request.future_attempt_guidance is not None:
            context["workflow_future_attempt_guidance"] = (
                state.request.future_attempt_guidance.model_dump(mode="json")
            )
        if state.request.policy.evidence_projection_mode == "disabled":
            context["workflow_perception_evidence"] = [
                item.model_dump(mode="json") for item in state.evidence
            ]
        if state.evidence_projection is not None:
            context["workflow_evidence_field_sources"] = copy.deepcopy(
                state.evidence_projection.field_sources
            )
            context["workflow_evidence_path_sources"] = copy.deepcopy(
                state.evidence_projection.path_sources
            )
            context["workflow_evidence_path_audio_refs"] = copy.deepcopy(
                state.evidence_projection.path_audio_refs
            )
            context["workflow_evidence_projection_fingerprint"] = (
                state.evidence_projection.projection_fingerprint
            )
            context.update(copy.deepcopy(state.evidence_projection.planner_context))
        if state.frozen_edit_intent is not None:
            context["workflow_frozen_edit_intent"] = copy.deepcopy(state.frozen_edit_intent)
        if state.capability_feedback is not None:
            context["workflow_perception_capability_feedback"] = (
                state.capability_feedback.model_dump(mode="json")
            )
        return context

    def _execution_context(
        self,
        state: _OrchestrationState,
    ) -> dict[str, Any]:
        context = copy.deepcopy(state.request.execution_context)
        if state.evidence_projection is not None:
            context.update(copy.deepcopy(state.evidence_projection.execution_context))
            context["workflow_evidence_path_sources"] = copy.deepcopy(
                state.evidence_projection.path_sources
            )
            context["workflow_evidence_path_audio_refs"] = copy.deepcopy(
                state.evidence_projection.path_audio_refs
            )
        return context

    def _perception_context(
        self,
        state: _OrchestrationState,
        *,
        call_index: int,
    ) -> dict[str, Any]:
        context = copy.deepcopy(state.request.planning_context)
        context.update(
            {
                "enable_planner_perception_guidance": False,
                "workflow_arm": state.request.arm,
                "workflow_perception_call_index": call_index,
                "workflow_perception_evidence_ids": [item.evidence_id for item in state.evidence],
                "workflow_source_audio_refs": list(state.request.source_audio_refs),
                "workflow_reference_audio_refs": list(state.request.reference_audio_refs),
            }
        )
        if state.request.source_audio_refs:
            source_refs = list(state.request.source_audio_refs)
            context["input_audio"] = source_refs[0]
            context["input_audios"] = source_refs
            context["input_audio_paths"] = source_refs
        if state.request.reference_audio_refs:
            context["reference_audio_paths"] = list(state.request.reference_audio_refs)
        if state.request.source_text:
            context["source_text"] = state.request.source_text
        if state.request.policy.evidence_projection_mode == "disabled":
            context["workflow_perception_evidence"] = [
                item.model_dump(mode="json") for item in state.evidence
            ]
        if state.evidence_projection is not None:
            context["workflow_evidence_field_sources"] = copy.deepcopy(
                state.evidence_projection.field_sources
            )
            context["workflow_evidence_path_sources"] = copy.deepcopy(
                state.evidence_projection.path_sources
            )
            context["workflow_evidence_projection_fingerprint"] = (
                state.evidence_projection.projection_fingerprint
            )
            context.update(copy.deepcopy(state.evidence_projection.planner_context))
        if state.request.runtime_output_dir is not None:
            context["output_dir"] = state.request.runtime_output_dir
        if state.request.audio_resolution_mode == "opaque_ref_v1":
            context["_runtime_audio_resolution_mode"] = "opaque_ref_v1"
            context["_runtime_audio_bindings"] = [
                item.model_copy(deep=True) for item in state.request.audio_bindings
            ]
        return context

    def _refresh_evidence_projection(
        self,
        state: _OrchestrationState,
    ) -> None:
        if self.evidence_projector is None:
            state.evidence_projection = None
            return
        try:
            projection = self.evidence_projector.project(
                [item.model_copy(deep=True) for item in state.evidence]
            )
            projection = WorkflowEvidenceProjection.model_validate(projection)
            unexpected_planner = sorted(
                set(projection.planner_context) - _REVIEWED_PLANNER_PROJECTION_FIELDS
            )
            unexpected_execution = sorted(
                set(projection.execution_context) - _REVIEWED_EXECUTION_PROJECTION_FIELDS
            )
            if unexpected_planner or unexpected_execution:
                raise ValueError(
                    "evidence projector emitted unreviewed fields: "
                    f"planner={unexpected_planner}, "
                    f"execution={unexpected_execution}"
                )
            evidence_ids = [item.evidence_id for item in state.evidence]
            if projection.evidence_ids != evidence_ids:
                raise ValueError("evidence projection must preserve evidence ids in order")
            projected_fields = {
                *projection.planner_context,
                *projection.execution_context,
            }
            unexpected_sources = sorted(set(projection.field_sources) - projected_fields)
            missing_sources = sorted(projected_fields - set(projection.field_sources))
            unknown_source_ids = sorted(
                {
                    source_id
                    for source_ids in projection.field_sources.values()
                    for source_id in source_ids
                    if source_id not in evidence_ids
                }
            )
            unknown_path_source_ids = sorted(
                {
                    source_id
                    for source_ids in projection.path_sources.values()
                    for source_id in source_ids
                    if source_id not in evidence_ids
                }
            )
            expected_requested_refs = list(
                dict.fromkeys(
                    audio_ref for item in state.evidence for audio_ref in item.requested_audio_refs
                )
            )
            expected_covered_refs = list(
                dict.fromkeys(audio_ref for item in state.evidence for audio_ref in item.audio_refs)
            )
            expected_missing_refs = [
                audio_ref
                for audio_ref in expected_requested_refs
                if audio_ref not in set(expected_covered_refs)
            ]
            if projection.requested_audio_refs != expected_requested_refs:
                raise ValueError("evidence projection requested audio refs do not match evidence")
            if projection.source_audio_refs != expected_covered_refs:
                raise ValueError("evidence projection covered audio refs do not match evidence")
            if projection.missing_audio_refs != expected_missing_refs:
                raise ValueError("evidence projection missing audio refs do not match evidence")
            unknown_path_audio_refs = sorted(
                set(projection.path_audio_refs.values()) - set(projection.source_audio_refs)
            )
            if (
                unexpected_sources
                or missing_sources
                or unknown_source_ids
                or unknown_path_source_ids
                or unknown_path_audio_refs
            ):
                raise ValueError(
                    "evidence projection provenance is invalid: "
                    f"unexpected_fields={unexpected_sources}, "
                    f"missing_fields={missing_sources}, "
                    f"sources={unknown_source_ids}, "
                    f"path_sources={unknown_path_source_ids}"
                    f", path_audio_refs={unknown_path_audio_refs}"
                )
            expected_fingerprint = workflow_evidence_projection_fingerprint(projection)
            if projection.projection_fingerprint != expected_fingerprint:
                raise ValueError("evidence projection fingerprint mismatch")
        except Exception as exc:
            state.recorder.emit(
                stage="perception",
                kind="workflow_evidence_projection_rejected",
                status="rejected",
                subject_refs=[item.evidence_id for item in state.evidence],
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise _WorkflowAbort(
                status="failed",
                stage="workflow_protocol",
                codes=["evidence_projection_invalid"],
            ) from exc
        state.evidence_projection = projection.model_copy(deep=True)
        state.recorder.emit(
            stage="perception",
            kind="workflow_evidence_projection_completed",
            status="completed",
            subject_refs=list(projection.evidence_ids),
            payload={"projection": projection.model_dump(mode="json")},
        )

    def _validate_configuration(self, request: WorkflowArmRequest) -> None:
        if (
            request.policy.evidence_projection_mode == "reviewed"
            and self.evidence_projector is None
        ):
            raise _WorkflowAbort(
                status="failed",
                stage="configuration",
                codes=["reviewed_evidence_projector_required"],
            )
        if (
            request.policy.evidence_projection_mode == "disabled"
            and self.evidence_projector is not None
        ):
            raise _WorkflowAbort(
                status="failed",
                stage="configuration",
                codes=["evidence_projector_forbidden_when_disabled"],
            )
        conflicts = [
            f"planning_context.{key}"
            for key in _arm_owned_context_conflicts(request.planning_context)
        ]
        conflicts.extend(
            f"execution_context.{key}"
            for key in _arm_owned_context_conflicts(request.execution_context)
        )
        if request.policy.evidence_projection_mode == "reviewed":
            conflicts.extend(
                f"planning_context.{key}"
                for key in _reviewed_context_conflicts(
                    request.planning_context,
                    allowed_keys=set(),
                )
            )
            conflicts.extend(
                f"execution_context.{key}"
                for key in _reviewed_context_conflicts(
                    request.execution_context,
                    allowed_keys={"output_audio", "output_dir"},
                )
            )
        if conflicts:
            raise _WorkflowAbort(
                status="failed",
                stage="configuration",
                codes=[f"workflow_arm_policy_conflict:{key}" for key in dict.fromkeys(conflicts)],
            )

    def _abort_cannot_plan(self, output: WorkflowPlannerOutput) -> None:
        codes = ["planner_cannot_plan"]
        codes.extend(
            f"planner_cannot_plan_reason:{item.code}"
            for item in output.cannot_plan_reasons
        )
        codes.extend(f"planner_missing_information:{item}" for item in output.missing_information)
        raise _WorkflowAbort(
            status="blocked",
            stage="planner",
            codes=codes,
        )

    def _result(
        self,
        state: _OrchestrationState,
        *,
        status: WorkflowOrchestrationStatus,
        failure_stage: WorkflowOrchestrationFailureStage,
        failure_codes: list[str],
        terminal_error_type: str | None = None,
        terminal_error_message: str | None = None,
    ) -> WorkflowArmResult:
        forward_errors = (
            state.forward_result.observation_errors if state.forward_result is not None else []
        )
        planner_traces = [
            item.invocation_trace
            for item in state.planner_history
            if item.invocation_trace is not None
        ]
        reasoner_traces = [
            item.invocation_trace
            for item in state.reasoner_history
            if item.invocation_trace is not None
        ]
        perception_traces = [
            item.invocation_trace
            for item in state.perception_history
            if item.invocation_trace is not None
        ]
        planned_perception_traces = (
            list(state.forward_result.planned_perception_invocation_traces)
            if state.forward_result is not None
            else []
        )
        perception_traces.extend(planned_perception_traces)
        component_traces = [*reasoner_traces, *planner_traces, *perception_traces]
        return WorkflowArmResult(
            run_id=state.request.run_id,
            assigned_arm=state.request.arm,
            source_audio_refs=list(state.request.source_audio_refs),
            reference_audio_refs=list(state.request.reference_audio_refs),
            status=status,
            failure_stage=failure_stage,
            failure_codes=list(dict.fromkeys(failure_codes)),
            terminal_error_type=terminal_error_type,
            terminal_error_message=(
                _safe_error_message(terminal_error_message) if terminal_error_message else None
            ),
            policy=state.request.policy,
            forward_policy=state.request.forward_policy,
            planning_context_fingerprint=_fingerprint(
                {
                    "caller_context": state.request.planning_context,
                    "historical_reviewed_evidence": (
                        state.request.historical_reviewed_evidence.model_dump(mode="json")
                        if state.request.historical_reviewed_evidence is not None
                        else None
                    ),
                }
            ),
            execution_context_fingerprint=_fingerprint(self._execution_context(state)),
            actual_path=list(state.actual_path),
            final_plan=(
                state.final_plan.model_copy(deep=True) if state.final_plan is not None else None
            ),
            forward_result=state.forward_result,
            reasoner_history=list(state.reasoner_history),
            planner_history=list(state.planner_history),
            perception_history=list(state.perception_history),
            planner_call_count=len(state.planner_history),
            perception_gateway_round_count=state.perception_gateway_round_count,
            orchestration_perception_call_count=len(state.perception_history),
            planned_perception_step_ids=(
                _planned_perception_step_ids(state.final_plan)
                if state.final_plan is not None
                else []
            ),
            planned_perception_steps=[
                item.model_copy(deep=True) for item in state.planned_perception_steps
            ],
            planned_perception_invocation_traces=[
                item.model_copy(deep=True) for item in planned_perception_traces
            ],
            evidence_projection=(
                state.evidence_projection.model_copy(deep=True)
                if state.evidence_projection is not None
                else None
            ),
            state_only_completion=(
                state.state_only_completion.model_copy(deep=True)
                if state.state_only_completion is not None
                else None
            ),
            planner_logical_provider_call_count=sum(
                item.logical_provider_call_count for item in planner_traces
            ),
            reasoner_logical_provider_call_count=sum(
                item.logical_provider_call_count for item in reasoner_traces
            ),
            perception_logical_provider_call_count=sum(
                item.logical_provider_call_count for item in perception_traces
            ),
            planner_provider_call_count=sum(item.provider_call_count for item in planner_traces),
            reasoner_provider_call_count=sum(item.provider_call_count for item in reasoner_traces),
            perception_provider_call_count=sum(
                item.provider_call_count for item in perception_traces
            ),
            perception_tool_call_count=sum(
                len(item.tool_calls) for item in state.perception_history
            ),
            perception_tool_attempt_count=sum(
                tool_call.attempt_count
                for item in state.perception_history
                for tool_call in item.tool_calls
            ),
            provider_retry_count=sum(item.retry_count for item in component_traces),
            logical_provider_retry_count=sum(item.logical_retry_count for item in component_traces),
            json_repair_count=sum(item.json_repair_count for item in component_traces),
            schema_repair_count=sum(item.schema_repair_count for item in component_traces),
            fallback_count=sum(len(item.fallback_events) for item in component_traces),
            reasoner_shadow_input_count=len(state.reasoner_shadow_inputs),
            reasoner_shadow_inputs=[
                item.model_copy(deep=True) for item in state.reasoner_shadow_inputs
            ],
            reasoner_shadow_observation_errors=list(state.reasoner_shadow_capture_errors),
            event_log_path=state.recorder.event_log_path,
            module_checkpoint_path=getattr(
                state.checkpoint_observer,
                "checkpoint_path",
                None,
            ),
            module_checkpoint_count=int(getattr(state.checkpoint_observer, "checkpoint_count", 0)),
            module_checkpoint_errors=list(getattr(state.checkpoint_observer, "errors", [])),
            events=list(state.recorder.events),
            observation_errors=[
                *state.recorder.observation_errors,
                *forward_errors,
            ],
        )


def _reasoner_shadow_input_snapshot(
    state: _OrchestrationState,
    *,
    context: dict[str, Any],
    call_index: int,
    planner_input_fingerprint: str,
    planner_started_event: WorkflowEvent,
) -> ReasonerShadowInputSnapshot:
    """Freeze only reviewed semantic inputs; no Reasoner call occurs here."""

    evidence_ids = [item.evidence_id for item in state.evidence]
    reviewed_evidence = (
        {
            key: copy.deepcopy(context[key])
            for key in sorted(_REVIEWED_PLANNER_PROJECTION_FIELDS)
            if key != "source_text" and key in context
        }
        if evidence_ids
        else {}
    )
    return ReasonerShadowInputSnapshot(
        request_id=f"reasoner-shadow:{call_index}",
        audio_asset_facts=[item.model_copy(deep=True) for item in state.request.audio_asset_facts],
        planner_call_index=call_index,
        phase=("evidence_review" if state.reasoner_shadow_inputs else "initial_understanding"),
        user_request=state.request.user_request,
        source_text=state.request.source_text,
        source_audio_refs=list(state.request.source_audio_refs),
        reference_audio_refs=list(state.request.reference_audio_refs),
        prior_turn_contexts=[
            item.model_copy(deep=True) for item in state.request.prior_turn_contexts
        ],
        session_state=(
            state.request.session_state.model_copy(deep=True)
            if state.request.session_state is not None
            else None
        ),
        reviewed_evidence_ids=evidence_ids,
        reviewed_evidence=reviewed_evidence,
        evidence_field_sources=copy.deepcopy(context.get("workflow_evidence_field_sources") or {}),
        evidence_path_sources=copy.deepcopy(context.get("workflow_evidence_path_sources") or {}),
        planner_input_fingerprint=planner_input_fingerprint,
        planner_source_event_ref={
            "event_id": planner_started_event.event_id,
            "run_id": planner_started_event.run_id,
            "sequence": planner_started_event.sequence,
        },
    )


def _capture_authoritative_reasoner_record(
    state: _OrchestrationState,
    *,
    planner: Any,
    call_index: int,
) -> None:
    """Capture a separated Reasoner product once, without fabricating one."""

    raw = getattr(planner, "last_reasoner_record", None)
    if raw is None:
        return
    try:
        record = WorkflowReasonerCallRecord.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - malformed accounting cannot gain authority
        state.reasoner_shadow_capture_errors.append(
            {
                "component": "authoritative_reasoner_record",
                "operation": "capture",
                "error_type": type(exc).__name__,
                "error_message": _safe_error_message(exc),
                "planner_call_index": call_index,
            }
        )
        return
    expected_id = f"reasoner:{call_index}"
    if record.call_index != call_index or record.call_id != expected_id:
        state.reasoner_shadow_capture_errors.append(
            {
                "component": "authoritative_reasoner_record",
                "operation": "capture",
                "error_type": "ReasonerCallIdentityMismatch",
                "error_message": (
                    f"expected {expected_id}, got {record.call_id} at index {record.call_index}"
                ),
                "planner_call_index": call_index,
            }
        )
        return
    if any(item.call_id == record.call_id for item in state.reasoner_history):
        return
    state.reasoner_history.append(record)
    state.actual_path.append(f"reasoner:{call_index}:{record.status}")
    state.recorder.emit(
        stage="reasoner",
        kind=(
            "workflow_reasoner_call_completed"
            if record.status == "completed"
            else "workflow_reasoner_call_failed"
        ),
        status=("completed" if record.status == "completed" else "failed"),
        attempt_id=record.call_id,
        payload={"record": record.model_dump(mode="json")},
    )


_ERROR_URL_PATTERN = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
_ERROR_CREDENTIAL_PATTERN = re.compile(
    r"\b(api[_ -]?key|authorization|bearer|token|secret|password)\b"
    r"\s*(?::|=|\s)\s*(?:bearer\s+)?[^\s,;]+",
    re.IGNORECASE,
)


def _safe_error_message(error: BaseException | str) -> str:
    message = str(error)
    message = _ERROR_URL_PATTERN.sub("<redacted-url>", message)
    message = _ERROR_CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group(1)}=<redacted>",
        message,
    )
    if len(message) > 1024:
        return f"{message[:1000]}...<truncated>"
    return message


def _tool_calls_from_error_or_component(
    error: BaseException,
    component: Any,
) -> list[WorkflowToolCallSummary]:
    raw_calls = getattr(error, "tool_calls", None)
    if raw_calls is None:
        raw_calls = getattr(component, "last_tool_calls", None)
    if not isinstance(raw_calls, list):
        return []
    calls: list[WorkflowToolCallSummary] = []
    for raw in raw_calls:
        try:
            call = (
                raw
                if isinstance(raw, WorkflowToolCallSummary)
                else WorkflowToolCallSummary.model_validate(raw)
            )
        except Exception:
            continue
        calls.append(call.model_copy(deep=True))
    return calls


def _normalize_planner_output(value: Any) -> WorkflowPlannerOutput:
    if isinstance(value, AudioEditPlan):
        return WorkflowPlannerOutput(
            decision="execute",
            plan=value.model_copy(deep=True),
        )
    if isinstance(value, WorkflowPlannerOutput):
        return WorkflowPlannerOutput.model_validate(value.model_dump(mode="python"))
    return WorkflowPlannerOutput.model_validate(value)


def _normalize_perception_evidence(value: Any) -> WorkflowPerceptionEvidence:
    if isinstance(value, WorkflowPerceptionEvidence):
        return WorkflowPerceptionEvidence.model_validate(value.model_dump(mode="python"))
    return WorkflowPerceptionEvidence.model_validate(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _invoke_component_callable(callback: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Invoke async components directly and move synchronous I/O off the event loop."""

    if inspect.iscoroutinefunction(callback):
        return await callback(*args, **kwargs)
    value = await asyncio.to_thread(callback, *args, **kwargs)
    return await _maybe_await(value)


def _fingerprint(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workflow_evidence_projection_fingerprint(
    projection: WorkflowEvidenceProjection,
) -> str:
    """Fingerprint projection content without recursively hashing itself."""

    return _fingerprint(
        {
            "schema_version": projection.schema_version,
            "evidence_ids": projection.evidence_ids,
            "source_audio_refs": projection.source_audio_refs,
            "planner_context": projection.planner_context,
            "execution_context": projection.execution_context,
            "field_sources": projection.field_sources,
            "path_sources": projection.path_sources,
            "reviewed_sources": [
                item.model_dump(mode="json") for item in projection.reviewed_sources
            ],
            "authority_assignments": [
                item.model_dump(mode="json") for item in projection.authority_assignments
            ],
            "evidence_conflicts": [
                item.model_dump(mode="json") for item in projection.evidence_conflicts
            ],
        }
    )


def _evidence_content_fingerprint(evidence: WorkflowPerceptionEvidence) -> str:
    return _fingerprint(
        {
            "scope": evidence.scope,
            "facts": evidence.facts,
        }
    )


def _workflow_evidence_content_fingerprint(
    state: _OrchestrationState,
) -> str:
    if (
        state.request.policy.evidence_projection_mode == "reviewed"
        and state.evidence_projection is not None
    ):
        return _fingerprint(state.evidence_projection.planner_context)
    return _fingerprint(sorted({_evidence_content_fingerprint(item) for item in state.evidence}))


def _planned_perception_step_ids(plan: AudioEditPlan | None) -> list[str]:
    if plan is None:
        return []
    return [
        step.id for step in plan.steps if step.operation.startswith(("perception.", "flexsed."))
    ]


def _bound_audio_refs(request: WorkflowArmRequest) -> list[str]:
    return list(dict.fromkeys([*request.source_audio_refs, *request.reference_audio_refs]))


def _perception_decomposition_metadata(
    request: WorkflowPerceptionRequest,
) -> dict[str, int | str] | None:
    """Return validated collective-job metadata for one routed sub-request."""

    raw = request.metadata.get("perception_job_decomposition")
    if not isinstance(raw, dict):
        return None
    source_job_id = str(raw.get("source_job_id") or "").strip()
    try:
        part_index = int(raw.get("part_index"))
        part_count = int(raw.get("part_count"))
    except (TypeError, ValueError):
        return None
    if not source_job_id or part_count < 2 or part_index < 0 or part_index >= part_count:
        return None
    return {
        "source_job_id": source_job_id,
        "part_index": part_index,
        "part_count": part_count,
    }


def _validate_plan_source_audio_refs(
    plan: AudioEditPlan,
    source_audio_refs: list[str],
    *,
    reference_audio_refs: list[str] | None = None,
    session_state: SessionStateContext | None = None,
) -> None:
    if not source_audio_refs:
        return
    if not plan.input_audio:
        raise _WorkflowAbort(
            status="blocked",
            stage="workflow_protocol",
            codes=["final_plan_input_audio_required"],
        )
    if plan.input_audio not in source_audio_refs:
        if plan_selects_bound_reference_base(
            plan,
            reference_audio_refs=reference_audio_refs or [],
            session_state=session_state,
        ):
            return
        raise _WorkflowAbort(
            status="blocked",
            stage="workflow_protocol",
            codes=["final_plan_input_audio_outside_source_inputs"],
        )


def _planned_perception_dependency_codes(
    plan: AudioEditPlan,
    planned_perception_ids: set[str],
    *,
    arm: Literal["w0", "w2"],
) -> list[str]:
    if not planned_perception_ids:
        return []
    edit_steps = [step for step in plan.steps if step.id not in planned_perception_ids]
    if not edit_steps:
        noun = "gap_perception" if arm == "w2" else "planned_perception"
        return [f"{arm}_{noun}_requires_downstream_edit"]
    by_id = {step.id: step for step in plan.steps}
    unconsumed = [
        step_id
        for step_id in sorted(planned_perception_ids)
        if not any(step_id in _step_ancestors(edit_step.id, by_id) for edit_step in edit_steps)
    ]
    noun = "gap_perception" if arm == "w2" else "planned_perception"
    return [f"{arm}_{noun}_not_consumed:{step_id}" for step_id in unconsumed]


def _step_ancestors(step_id: str, by_id: dict[str, Any]) -> set[str]:
    ancestors: set[str] = set()
    pending = list(getattr(by_id.get(step_id), "depends_on", []))
    while pending:
        dependency_id = pending.pop()
        if dependency_id in ancestors:
            continue
        ancestors.add(dependency_id)
        dependency = by_id.get(dependency_id)
        if dependency is not None:
            pending.extend(dependency.depends_on)
    return ancestors


def _planner_route(
    plan: AudioEditPlan | None,
    invocation_trace: ComponentInvocationTrace | None,
) -> Literal["direct", "deterministic_preflight", "fallback"] | None:
    if invocation_trace is not None and invocation_trace.fallback_used:
        return "fallback"
    if invocation_trace is not None and invocation_trace.route == "deterministic_preflight":
        return "deterministic_preflight"
    if plan is None:
        return None
    if plan.metadata.get("planner_fallback"):
        return "fallback"
    if plan.metadata.get("planner_preflight"):
        return "deterministic_preflight"
    return "direct"


def _component_trace_from_error_or_component(
    error: Exception,
    component: Any,
) -> ComponentInvocationTrace | None:
    candidates = [
        getattr(error, "invocation_trace", None),
        getattr(component, "last_invocation_trace", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return validate_component_invocation_trace(candidate).model_copy(deep=True)
        except Exception:
            continue
    return None


_FROZEN_EDIT_INTENT_SEMANTIC_FIELDS = (
    "schema_version",
    "target_scopes",
    "edit_objectives",
    "preservation_requirements",
    "quality_requirements",
    "required_evidence",
)


def _edit_intent_preserves_frozen_semantics(
    frozen: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_effect_revision: bool = False,
) -> bool:
    """Allow route revision without allowing post-evidence objective drift."""

    if any(
        frozen.get(field) != current.get(field) for field in _FROZEN_EDIT_INTENT_SEMANTIC_FIELDS
    ):
        return False
    current_effects = list(current.get("effect_kinds") or [])
    if not current_effects:
        return False
    if not allow_effect_revision and not _is_ordered_subset(
        list(frozen.get("effect_kinds") or []),
        current_effects,
    ):
        return False
    frozen_uncertainties = list(frozen.get("uncertainties") or [])
    current_uncertainties = list(current.get("uncertainties") or [])
    return _is_ordered_subset(frozen_uncertainties, current_uncertainties)


def _is_ordered_subset(frozen: list[Any], current: list[Any]) -> bool:
    iterator = iter(frozen)
    for current_item in current:
        if not any(frozen_item == current_item for frozen_item in iterator):
            return False
    return True


def _component_trace_blockers(
    policy: WorkflowArmPolicy,
    trace: ComponentInvocationTrace | None,
    *,
    component: Literal["reasoner", "planner", "perception"],
    gateway_call_id: str,
) -> list[str]:
    return component_invocation_trace_blockers(
        _component_admission_policy(policy, component=component),
        trace,
        component=component,
        gateway_call_id=gateway_call_id,
    )


def _component_admission_policy(
    policy: WorkflowArmPolicy,
    *,
    component: Literal["reasoner", "planner", "perception"],
) -> ComponentInvocationAdmissionPolicy:
    provider_limit = {
        "reasoner": policy.reasoner_provider_call_limit_per_gateway,
        "planner": policy.planner_provider_call_limit_per_gateway,
        "perception": policy.perception_provider_call_limit_per_gateway,
    }[component]
    return ComponentInvocationAdmissionPolicy(
        trace_mode=policy.component_invocation_trace_mode,
        provider_call_limit=provider_limit,
        provider_retry_mode=policy.provider_retry_mode,
        json_repair_mode=policy.json_repair_mode,
        schema_repair_mode=policy.schema_repair_mode,
        fallback_mode=policy.fallback_mode,
    )


def _forward_step_path(result: ForwardExecutionResult) -> list[str]:
    path: list[str] = []
    for event in result.events:
        if event.kind != "step_execution_completed":
            continue
        step = event.payload.get("step")
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "unknown")
        operation = str(step.get("operation") or "unknown")
        prefix = (
            "planned_perception"
            if operation.startswith(("perception.", "flexsed."))
            else "executor"
        )
        path.append(f"{prefix}:{step_id}:{operation}:{event.status}")
    return path


def _arm_owned_context_conflicts(context: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key in context
        if key in _ARM_OWNED_CONTEXT_KEYS or key.startswith(_ARM_OWNED_CONTEXT_PREFIXES)
    )


def _reviewed_context_conflicts(
    value: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> list[str]:
    conflicts = [str(key) for key in value if str(key) not in allowed_keys]
    conflicts.extend(
        str(key)
        for key, item in value.items()
        if str(key) in allowed_keys and item is not None and not isinstance(item, str)
    )
    return sorted(dict.fromkeys(conflicts))


__all__ = [
    "WorkflowArm",
    "WorkflowArmOrchestrator",
    "WorkflowArmPolicy",
    "WorkflowArmRequest",
    "WorkflowArmResult",
    "WorkflowEvidenceProjection",
    "WorkflowEvidenceProjector",
    "WorkflowPerceptionCallRecord",
    "WorkflowPerceptionEvidence",
    "WorkflowPerceptionGateway",
    "WorkflowPerceptionRequest",
    "WorkflowPlannedPerceptionStep",
    "WorkflowPlanner",
    "WorkflowPlannerCallRecord",
    "WorkflowPlannerOutput",
    "WorkflowReasonerCallRecord",
    "WorkflowStateOnlyCompletion",
    "WorkflowToolCallSummary",
    "workflow_evidence_projection_fingerprint",
]
