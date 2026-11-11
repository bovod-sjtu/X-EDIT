"""Adapters from real X-Editor components to typed workflow-arm contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from x_editor.capabilities.schemas import (
    OperationCapabilityCard,
    validate_operation_arguments,
)
from x_editor.core.invocation_trace import (
    ComponentInvocationTrace,
    validate_component_invocation_trace,
)
from x_editor.core.perception_evidence_authority import (
    EvidenceAuthorityAssignment,
    ObservationKind,
    ReviewedEvidenceSource,
    append_conflict_uncertainties,
    build_reviewed_evidence_source,
    competing_observations,
    reconcile_canonical_reviewed_report,
    resolve_evidence_authority,
    scope_report_to_authority,
)
from x_editor.core.reviewed_evidence import (
    PROJECTED_SCALAR_FIELDS as _PROJECTED_SCALAR_FIELDS,
)
from x_editor.core.reviewed_evidence import (
    namespace_reviewed_local_ids as _namespace_reviewed_local_ids,
)
from x_editor.core.reviewed_evidence import (
    sanitize_reviewed_perception_facts as _sanitize_perception_facts,
)
from x_editor.core.reviewed_evidence import (
    sanitize_reviewed_perception_report as _sanitize_perception_report,
)
from x_editor.core.reviewed_evidence import (
    sanitize_reviewed_value as _sanitize_reviewed_value,
)
from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.core.transcript_revision import REVISION_OPERATION, bind_transcript_revision
from x_editor.perception.workflow_catalog import (
    compatible_workflow_perception_operations,
    resolve_workflow_perception_capabilities,
    workflow_perception_capability,
)
from x_editor.planner.reasoned_workflow import (
    PlannerReasonerTurn,
    project_perception_intent_to_workflow_requests,
    validate_plan_semantic_bindings,
)
from x_editor.runtime.audio_plan_executor import AudioEditExecutor
from x_editor.runtime.audio_references import (
    AUDIO_REFERENCE_KEYS,
    canonical_audio_ref_for_runtime_path,
    canonicalize_perception_audio_bindings,
    opaque_audio_resolution_enabled,
)
from x_editor.runtime.workflow_orchestration import (
    WorkflowEvidenceProjection,
    WorkflowPerceptionEvidence,
    WorkflowPerceptionRequest,
    WorkflowPlannedPerceptionStep,
    WorkflowPlannerOutput,
    WorkflowToolCallSummary,
    workflow_evidence_projection_fingerprint,
)
from x_editor.runtime.workflow_protocol import (
    WorkflowProtocolPolicy,
    resolve_workflow_protocol_policy,
)

_PERCEPTION_PREFIXES = ("perception.", "flexsed.")
_SYMBOLIC_AUDIO_REFS = {"base", "current", "input", "input_audio"}
_WORKFLOW_SOURCE_AUDIO_ALIASES = {
    "base",
    "input",
    "input_audio",
    "source",
    "source_audio",
    "workflow_source",
    "workflow_source_audio",
    "workflow_source_audio_ref",
    "workflow_source_audio_refs",
    "inputs.input_audio",
    "inputs.input_audio_path",
    "plan.input_audio",
}
_AUDIO_REF_KEYS = set(AUDIO_REFERENCE_KEYS)
_MULTI_AUDIO_REPORT_SCHEMA_VERSION = "x-editor.reviewed-perception/by-audio-ref/v1"
_MULTI_AUDIO_REPORT_ENTRY_SCHEMA_VERSION = "x-editor.reviewed-perception/audio-report/v1"
_LOCAL_DESTRUCTIVE_SPAN_OPERATIONS = {
    "artifact_ops.remove_spans",
    "artifact_ops.mute_spans",
    "artifact_ops.beep_spans",
}


class WorkflowComponentAdapterError(RuntimeError):
    """Component failure that preserves any terminal invocation trace."""

    def __init__(
        self,
        message: str,
        *,
        invocation_trace: ComponentInvocationTrace | None = None,
        failure_kind: Literal["component", "protocol"] = "component",
        tool_calls: list[WorkflowToolCallSummary] | None = None,
    ) -> None:
        super().__init__(message)
        self.invocation_trace = invocation_trace
        self.failure_kind = failure_kind
        self.tool_calls = [item.model_copy(deep=True) for item in (tool_calls or [])]


class TraceableAudioEditPlannerAdapter:
    """Map an AudioEditPlan-producing Planner onto W0/W1/W2 decisions."""

    def __init__(
        self,
        planner: Any,
        *,
        allowed_perception_operations: set[str],
        allowed_edit_operations: set[str] | None = None,
        evidence_required_operations: set[str] | None = None,
        perception_operation_params: dict[str, dict[str, Any]] | None = None,
        perception_route_policies: dict[str, dict[str, Any]] | None = None,
        edit_operation_params: dict[str, dict[str, Any]] | None = None,
        edit_operation_capability_cards: dict[str, dict[str, Any]] | None = None,
        tool_prior_guidance: dict[str, Any] | None = None,
        workflow_protocol_policy: WorkflowProtocolPolicy | str | None = None,
        planner_prompt_profile: str = "objective_coverage_v2",
    ) -> None:
        self.planner = planner
        self.allowed_perception_operations = set(allowed_perception_operations)
        self.allowed_edit_operations = (
            set(allowed_edit_operations) if allowed_edit_operations is not None else None
        )
        self.evidence_required_operations = set(
            evidence_required_operations
            or {
                "artifact_ops.extract_spans",
                "artifact_ops.remove_spans",
                "artifact_ops.mute_spans",
                "artifact_ops.beep_spans",
            }
        )
        self.perception_operation_params = copy.deepcopy(perception_operation_params or {})
        self.perception_route_policies = copy.deepcopy(perception_route_policies or {})
        self.edit_operation_params = copy.deepcopy(edit_operation_params or {})
        self.edit_operation_capability_cards = {
            operation: OperationCapabilityCard.model_validate(card)
            for operation, card in (edit_operation_capability_cards or {}).items()
        }
        self.tool_prior_guidance = copy.deepcopy(tool_prior_guidance)
        self.workflow_protocol_policy = resolve_workflow_protocol_policy(workflow_protocol_policy)
        self.planner_prompt_profile = planner_prompt_profile
        self.last_invocation_trace: ComponentInvocationTrace | None = None
        self._call_lock = threading.Lock()

    def plan(
        self,
        user_request: str,
        context: dict[str, Any],
    ) -> WorkflowPlannerOutput:
        self.last_invocation_trace = None
        gateway_call_id = f"planner:{int(context.get('workflow_planner_call_index') or 1)}"
        with self._call_lock:
            if hasattr(self.planner, "last_invocation_trace"):
                self.planner.last_invocation_trace = None
            try:
                component_context = copy.deepcopy(context)
                if self.tool_prior_guidance is not None:
                    component_context["workflow_tool_prior_guidance"] = copy.deepcopy(
                        self.tool_prior_guidance
                    )
                if str(context.get("workflow_arm") or "") == "W1" and callable(
                    getattr(self.planner, "plan_workflow_turn", None)
                ):
                    component_context["workflow_frozen_edit_operation_params"] = copy.deepcopy(
                        self.edit_operation_params
                    )
                    component_context["workflow_require_capability_selection"] = bool(
                        self.edit_operation_capability_cards
                    )
                    if self.edit_operation_capability_cards:
                        component_context["workflow_routing_boundary_mode"] = "shadow"
                    component_context["workflow_evidence_required_operations"] = sorted(
                        self.evidence_required_operations
                    )
                    component_context["workflow_protocol_policy"] = (
                        self.workflow_protocol_policy.model_dump(mode="json")
                    )
                    component_context["workflow_planner_prompt_profile"] = (
                        self.planner_prompt_profile
                    )
                    raw_plan = self.planner.plan_workflow_turn(
                        user_request,
                        component_context,
                    )
                else:
                    raw_plan = self.planner.plan(
                        user_request,
                        component_context,
                    )
            except Exception as exc:
                trace = _component_trace(
                    self.planner,
                    gateway_call_id=gateway_call_id,
                )
                self.last_invocation_trace = trace
                raise WorkflowComponentAdapterError(
                    str(exc),
                    invocation_trace=trace,
                    failure_kind=getattr(
                        exc,
                        "failure_kind",
                        "component",
                    ),
                ) from exc
            trace = _component_trace(
                self.planner,
                gateway_call_id=gateway_call_id,
            )
        self.last_invocation_trace = trace
        if isinstance(raw_plan, PlannerReasonerTurn):
            try:
                return self._reasoned_w1_output(
                    raw_plan,
                    context=context,
                    trace=trace,
                )
            except WorkflowComponentAdapterError as exc:
                exc.failure_kind = "protocol"
                raise
        if isinstance(raw_plan, WorkflowPlannerOutput):
            output = raw_plan.model_copy(deep=True)
            output.invocation_trace = trace
            return output
        try:
            plan = AudioEditPlan.model_validate(raw_plan).model_copy(deep=True)
        except Exception as exc:
            raise WorkflowComponentAdapterError(
                f"Planner returned an invalid AudioEditPlan: {exc}",
                invocation_trace=trace,
            ) from exc
        plan, frozen_param_normalizations = _normalize_frozen_edit_operation_params(
            plan,
            frozen_params=self.edit_operation_params,
        )
        if frozen_param_normalizations:
            plan.metadata["workflow_frozen_edit_param_normalizations"] = frozen_param_normalizations

        if plan.missing_inputs:
            return WorkflowPlannerOutput(
                decision="cannot_plan",
                missing_information=list(plan.missing_inputs),
                cannot_plan_reasons=[
                    {
                        "code": "other",
                        "message": "Planner reported unresolved required inputs.",
                    }
                ],
                reason="Planner reported unresolved required inputs.",
                invocation_trace=trace,
            )
        if not plan.steps:
            return WorkflowPlannerOutput(
                decision="cannot_plan",
                missing_information=["executable edit steps"],
                cannot_plan_reasons=[
                    {
                        "code": "other",
                        "message": "Planner returned no executable steps.",
                    }
                ],
                reason="Planner returned no executable steps.",
                invocation_trace=trace,
            )

        arm = str(context.get("workflow_arm") or "")
        if arm == "W1":
            return self._w1_output(plan, context=context, trace=trace)
        if arm == "W2":
            return self._w2_output(plan, context=context, trace=trace)
        if arm != "W0":
            raise WorkflowComponentAdapterError(
                f"Unknown workflow arm: {arm or '<missing>'}",
                invocation_trace=trace,
            )
        return WorkflowPlannerOutput(
            decision="execute",
            plan=plan,
            invocation_trace=trace,
        )

    def _reasoned_w1_output(
        self,
        turn: PlannerReasonerTurn,
        *,
        context: dict[str, Any],
        trace: ComponentInvocationTrace | None,
    ) -> WorkflowPlannerOutput:
        turn = PlannerReasonerTurn.model_validate(turn.model_dump(mode="python"))
        if str(context.get("workflow_arm") or "") != "W1":
            raise WorkflowComponentAdapterError(
                "Native Planner/Reasoner turns are currently admitted only for W1.",
                invocation_trace=trace,
            )
        if turn.decision == "cannot_plan":
            return WorkflowPlannerOutput(
                decision="cannot_plan",
                missing_information=list(turn.missing_information),
                cannot_plan_reasons=[
                    {
                        "code": "other",
                        "message": (
                            turn.decision_summary
                            or "; ".join(turn.missing_information)
                            or "Planner could not produce a legal plan."
                        ),
                    }
                ],
                reason=turn.decision_summary,
                invocation_trace=trace,
            )
        if turn.decision == "request_perception":
            assert turn.perception_intent is not None
            intent = turn.perception_intent
            if self.edit_operation_capability_cards and turn.edit_intent is None:
                raise WorkflowComponentAdapterError(
                    "Planner/Reasoner must freeze a typed edit_intent before Perception.",
                    invocation_trace=trace,
                )
            if intent.perception_operation not in self.allowed_perception_operations:
                raise WorkflowComponentAdapterError(
                    "Planner/Reasoner selected a Perception operation outside the frozen catalog.",
                    invocation_trace=trace,
                )
            source_refs, reference_refs, bound_refs = _workflow_bound_audio_ref_namespace(context)
            if not source_refs or not set(intent.audio_refs).issubset(bound_refs):
                raise WorkflowComponentAdapterError(
                    "Planner/Reasoner Perception intent is outside bound workflow "
                    "source/reference audio refs.",
                    invocation_trace=trace,
                )
            if self.allowed_edit_operations is not None and not set(
                intent.candidate_operations
            ).issubset(self.allowed_edit_operations):
                raise WorkflowComponentAdapterError(
                    "Planner/Reasoner proposed an edit operation outside the frozen catalog.",
                    invocation_trace=trace,
                )
            capability_resolution = resolve_workflow_perception_capabilities(
                target_kinds=(item.kind for item in intent.perception_targets),
                expected_evidence_fields=intent.expected_evidence_fields,
                temporal_precision=intent.temporal_precision,
                available_operations=self.allowed_perception_operations,
            )
            compatible_operations = list(capability_resolution.compatible_operations)
            unsupported_evidence_fields = list(capability_resolution.unsupported_evidence_fields)
            if unsupported_evidence_fields and not capability_resolution.recognized_evidence_fields:
                raise WorkflowComponentAdapterError(
                    "Planner/Reasoner requested evidence fields that no frozen "
                    "Perception route exposes; "
                    f"unsupported_evidence_fields={unsupported_evidence_fields}",
                    invocation_trace=trace,
                )
            route_normalization: dict[str, Any] | None = None
            if intent.perception_operation not in compatible_operations:
                overlap_match = _unique_perception_route_by_evidence_overlap(
                    target_kinds=(item.kind for item in intent.perception_targets),
                    expected_evidence_fields=intent.expected_evidence_fields,
                    temporal_precision=intent.temporal_precision,
                    available_operations=self.allowed_perception_operations,
                )
                if (
                    self.workflow_protocol_policy.perception_route_normalization_mode
                    == "exact_unique"
                    and (len(compatible_operations) == 1 or overlap_match is not None)
                ):
                    normalized_operation = (
                        compatible_operations[0]
                        if len(compatible_operations) == 1
                        else overlap_match
                    )
                    assert normalized_operation is not None
                    capability = workflow_perception_capability(normalized_operation)
                    supported_evidence = set(
                        capability.evidence_types if capability is not None else ()
                    )
                    normalized_evidence_fields = [
                        field
                        for field in intent.expected_evidence_fields
                        if _evidence_field_root(field) in supported_evidence
                    ]
                    route_normalization = {
                        "source_operation": intent.perception_operation,
                        "normalized_operation": normalized_operation,
                        "reason": (
                            "selected operation cannot satisfy all declared target kinds, "
                            "evidence types, and temporal precision; catalog match is unique"
                            if len(compatible_operations) == 1
                            else "selected operation carried incompatible route-specific "
                            "evidence fields; target, temporal precision, and supported-field "
                            "overlap identify one catalog route"
                        ),
                    }
                    removed_fields = [
                        field
                        for field in intent.expected_evidence_fields
                        if field not in normalized_evidence_fields
                    ]
                    if removed_fields:
                        route_normalization["removed_incompatible_evidence_fields"] = removed_fields
                    intent = type(intent).model_validate(
                        {
                            **intent.model_dump(mode="python"),
                            "perception_operation": normalized_operation,
                            "expected_evidence_fields": normalized_evidence_fields,
                        }
                    )
                else:
                    declared_target_kinds = [item.kind for item in intent.perception_targets]
                    raise WorkflowComponentAdapterError(
                        "Planner/Reasoner selected a Perception operation that cannot "
                        "satisfy its declared targets, evidence fields, and temporal "
                        "precision; "
                        f"selected={intent.perception_operation}; "
                        f"target_kinds={declared_target_kinds}; "
                        f"expected_evidence_fields={intent.expected_evidence_fields}; "
                        f"temporal_precision={intent.temporal_precision}; "
                        f"compatible catalog operations={compatible_operations}",
                        invocation_trace=trace,
                    )
            elif unsupported_evidence_fields:
                recognized_fields = list(capability_resolution.recognized_evidence_fields)
                route_normalization = {
                    "source_operation": intent.perception_operation,
                    "normalized_operation": intent.perception_operation,
                    "reason": (
                        "catalog has no registered producer for supplemental "
                        "evidence fields; continue with the recognized evidence "
                        "contract and expose the unsupported fields explicitly"
                    ),
                    "removed_unsupported_evidence_fields": (unsupported_evidence_fields),
                }
                intent = type(intent).model_validate(
                    {
                        **intent.model_dump(mode="python"),
                        "expected_evidence_fields": recognized_fields,
                    }
                )
            capability = workflow_perception_capability(intent.perception_operation)
            frozen_params = self.perception_operation_params.get(
                intent.perception_operation,
                {},
            )
            route_policy = self.perception_route_policies.get(intent.perception_operation)
            requests: list[WorkflowPerceptionRequest] = []
            for request_payload in project_perception_intent_to_workflow_requests(intent):
                request_audio_refs = list(request_payload.get("audio_refs") or [])
                request_metadata = dict(request_payload.get("metadata") or {})
                primary_audio_ref = str(request_metadata.get("primary_audio_ref") or "")
                request_payload["inputs"] = _reasoned_perception_audio_inputs(
                    request_audio_refs,
                    primary_audio_ref=primary_audio_ref,
                    source_refs=source_refs,
                    reference_refs=reference_refs,
                )
                if capability is not None:
                    request_metadata["perception_capability"] = capability.to_dict()
                    request_metadata["perception_skill_ids"] = list(capability.skill_ids)
                    request_metadata["supported_evidence_types"] = list(capability.evidence_types)
                    request_metadata["evidence_requirement_rules"] = list(
                        capability.requirement_rules
                    )
                    if route_normalization is not None:
                        request_metadata["perception_route_normalization"] = route_normalization
                    if unsupported_evidence_fields:
                        request_metadata["unsupported_requested_evidence_fields"] = (
                            unsupported_evidence_fields
                        )
                request_params = dict(request_payload.get("params") or {})
                for key, value in frozen_params.items():
                    if (
                        intent.perception_operation == "perception.timestamped_caption"
                        and key == "require_flexsed_prior_success"
                        and not request_params.get("event_labels")
                    ):
                        request_params[key] = False
                        continue
                    if key in request_params and request_params[key] != value:
                        raise WorkflowComponentAdapterError(
                            "Planner/Reasoner Perception param conflicts with "
                            f"frozen policy: {key}",
                            invocation_trace=trace,
                        )
                    request_params[key] = copy.deepcopy(value)
                request_payload["params"] = request_params
                if route_policy is not None:
                    request_metadata["perception_route_policy"] = copy.deepcopy(route_policy)
                    request_metadata["perception_route_policy_fingerprint"] = _stable_fingerprint(
                        route_policy
                    )
                request_payload["metadata"] = request_metadata
                requests.append(WorkflowPerceptionRequest.model_validate(request_payload))
            return WorkflowPlannerOutput(
                decision="request_perception",
                perception_requests=requests,
                missing_information=list(intent.requirements),
                reason=turn.decision_summary,
                frozen_edit_intent=(
                    turn.edit_intent.model_dump(mode="json")
                    if turn.edit_intent is not None
                    else None
                ),
                invocation_trace=trace,
            )

        assert turn.plan is not None
        assert turn.execution_semantics is not None
        if self.edit_operation_capability_cards and turn.edit_intent is None:
            raise WorkflowComponentAdapterError(
                "Planner/Reasoner executable output is missing typed edit_intent.",
                invocation_trace=trace,
            )
        if self.edit_operation_capability_cards and turn.tool_selection is None:
            raise WorkflowComponentAdapterError(
                "Planner/Reasoner executable output is missing tool_selection and "
                "legal parameter declarations.",
                invocation_trace=trace,
            )
        if _perception_steps(turn.plan):
            raise WorkflowComponentAdapterError(
                "Native Planner/Reasoner final plan cannot contain Perception steps.",
                invocation_trace=trace,
            )
        operations = {step.operation for step in turn.plan.steps}
        if self.allowed_edit_operations is not None and not operations.issubset(
            self.allowed_edit_operations
        ):
            raise WorkflowComponentAdapterError(
                "Planner/Reasoner final plan contains an edit operation outside "
                "the frozen catalog.",
                invocation_trace=trace,
            )
        source_ref_values, _reference_ref_values, bound_refs = _workflow_bound_audio_ref_namespace(
            context
        )
        plan, frozen_param_normalizations = _normalize_frozen_edit_operation_params(
            turn.plan,
            frozen_params=self.edit_operation_params,
        )
        if frozen_param_normalizations:
            plan.metadata["workflow_frozen_edit_param_normalizations"] = frozen_param_normalizations
        tool_selection = turn.tool_selection
        normalized_turn = turn.model_copy(
            update={"plan": plan, "tool_selection": tool_selection},
            deep=True,
        )
        capability_errors = _capability_selection_contract_errors(
            normalized_turn,
            plan=plan,
            capability_cards=self.edit_operation_capability_cards,
            frozen_params=self.edit_operation_params,
        )
        if capability_errors:
            raise WorkflowComponentAdapterError(
                "Planner/Reasoner tool or parameter selection rejected: "
                + "; ".join(capability_errors),
                invocation_trace=trace,
            )
        plan, objective_coverage_normalizations = _normalize_validated_single_objective_step_set(
            plan,
            tool_selection=tool_selection,
        )
        if objective_coverage_normalizations:
            plan.metadata.setdefault(
                "objective_coverage_normalizations",
                [],
            ).extend(objective_coverage_normalizations)
        semantics = turn.execution_semantics
        path_sources = context.get("workflow_evidence_path_sources") or {}
        evidence_paths = _evidence_paths_by_id(path_sources)
        evidence_values = _evidence_values_by_id(
            context,
            path_sources,
        )
        if self.workflow_protocol_policy.binding_normalization_mode == "exact_unique":
            plan, span_container_plan_normalizations = _normalize_plan_span_input_containers(plan)
            if span_container_plan_normalizations:
                plan.metadata["workflow_span_container_normalizations"] = (
                    span_container_plan_normalizations
                )
            plan, direct_boundary_normalizations = _normalize_declared_direct_boundary_values(
                plan,
                semantics,
                evidence_values=evidence_values,
                evidence_required_operations=self.evidence_required_operations,
            )
            plan, semantics, span_binding_normalizations = _normalize_unique_reviewed_span_binding(
                plan,
                semantics,
                context=context,
                evidence_values=evidence_values,
                evidence_required_operations=self.evidence_required_operations,
            )
            semantics, span_container_normalizations = _normalize_span_argument_container_aliases(
                plan, semantics
            )
            if span_container_normalizations:
                plan.metadata.setdefault(
                    "workflow_span_container_normalizations",
                    [],
                ).extend(span_container_normalizations)
            semantics, unique_boundary_source_normalizations = (
                _normalize_unique_local_boundary_argument_sources(
                    plan,
                    semantics,
                    evidence_values=evidence_values,
                    evidence_required_operations=self.evidence_required_operations,
                )
            )
            span_binding_normalizations = [
                *direct_boundary_normalizations,
                *span_binding_normalizations,
                *unique_boundary_source_normalizations,
            ]
            if span_binding_normalizations:
                plan.metadata["workflow_span_binding_normalizations"] = span_binding_normalizations
            if self.workflow_protocol_policy.irrelevant_binding_mode == "drop":
                semantics, dropped_nonboundary_sources = (
                    _drop_mismatched_nonboundary_direct_evidence_sources(
                        plan,
                        semantics,
                        evidence_values=evidence_values,
                        evidence_required_operations=self.evidence_required_operations,
                    )
                )
                if dropped_nonboundary_sources:
                    plan.metadata["workflow_nonboundary_binding_normalizations"] = (
                        dropped_nonboundary_sources
                    )
        audio_ref_normalizations: list[dict[str, str]] = []
        if self.workflow_protocol_policy.binding_normalization_mode == "exact_unique":
            plan, audio_ref_normalizations = _normalize_evidence_audio_aliases(
                plan,
                source_refs=source_ref_values,
                evidence_ids=set(context.get("workflow_perception_evidence_ids") or []),
            )
            if audio_ref_normalizations:
                plan.metadata["workflow_audio_ref_normalizations"] = audio_ref_normalizations
            plan, dependency_output_normalizations = _normalize_dependency_audio_output_aliases(
                plan
            )
            if dependency_output_normalizations:
                audio_ref_normalizations.extend(dependency_output_normalizations)
                plan.metadata["workflow_audio_ref_normalizations"] = audio_ref_normalizations
            semantics, source_role_normalizations = _normalize_exact_argument_source_roles(
                plan,
                semantics,
                evidence_values=evidence_values,
            )
            if source_role_normalizations:
                plan.metadata["workflow_argument_source_normalizations"] = (
                    source_role_normalizations
                )
            semantics, nonleaf_evidence_path_normalizations = (
                _drop_nonleaf_evidence_path_declarations(
                    semantics,
                    available_evidence_paths=evidence_paths,
                )
            )
            if nonleaf_evidence_path_normalizations:
                plan.metadata["workflow_evidence_path_normalizations"] = (
                    nonleaf_evidence_path_normalizations
                )
        invalid_audio_refs = _invalid_final_plan_audio_refs(
            plan,
            source_refs=bound_refs,
        )
        if invalid_audio_refs:
            raise WorkflowComponentAdapterError(
                "Planner/Reasoner final plan contains audio refs outside workflow "
                "source/reference bindings or declared dependencies: "
                f"{invalid_audio_refs}",
                invocation_trace=trace,
            )
        required_step_ids = [
            step.id for step in plan.steps if step.operation in self.evidence_required_operations
        ]
        local_boundary_error = _local_destructive_boundary_error(
            plan,
            context=context,
        )
        if local_boundary_error is not None:
            raise WorkflowComponentAdapterError(
                local_boundary_error,
                invocation_trace=trace,
            )
        try:
            semantics = validate_plan_semantic_bindings(
                semantics,
                plan=plan,
                available_evidence_paths=evidence_paths,
                evidence_required_step_ids=required_step_ids,
                available_evidence_values=evidence_values,
                binding_normalization_mode=(
                    self.workflow_protocol_policy.binding_normalization_mode
                ),
                irrelevant_binding_mode=(self.workflow_protocol_policy.irrelevant_binding_mode),
            )
        except Exception as exc:
            raise WorkflowComponentAdapterError(
                f"Planner/Reasoner semantic binding rejected: {exc}",
                invocation_trace=trace,
            ) from exc
        plan = plan.model_copy(deep=True)
        plan.metadata["planner_reasoner_contract"] = {
            "schema_version": turn.schema_version,
            "decision_summary": turn.decision_summary,
            "execution_semantics": semantics.model_dump(mode="json"),
            "edit_intent": (
                turn.edit_intent.model_dump(mode="json") if turn.edit_intent is not None else None
            ),
            "tool_selection": (
                turn.tool_selection.model_dump(mode="json")
                if turn.tool_selection is not None
                else None
            ),
            "evidence_projection_fingerprint": context.get(
                "workflow_evidence_projection_fingerprint"
            ),
            "workflow_protocol_policy": (self.workflow_protocol_policy.model_dump(mode="json")),
            "semantic_binding_audit": _semantic_binding_protocol_audit(
                turn.execution_semantics,
                semantics,
            ),
        }
        plan.metadata["workflow_audio_usage_bindings"] = [
            item.model_dump(mode="json") for item in semantics.audio_usage_bindings
        ]
        return WorkflowPlannerOutput(
            decision="execute",
            plan=plan,
            reason=turn.decision_summary,
            frozen_edit_intent=(
                turn.edit_intent.model_dump(mode="json") if turn.edit_intent is not None else None
            ),
            invocation_trace=trace,
        )

    def _w1_output(
        self,
        plan: AudioEditPlan,
        *,
        context: dict[str, Any],
        trace: ComponentInvocationTrace | None,
    ) -> WorkflowPlannerOutput:
        perception_steps = _perception_steps(plan)
        if not perception_steps:
            return WorkflowPlannerOutput(
                decision="execute",
                plan=plan,
                invocation_trace=trace,
            )
        step_ids = [step.id for step in perception_steps]
        if len(step_ids) != len(set(step_ids)):
            return _cannot_plan(
                trace,
                "unique perception step ids",
                "W1 cannot create stable requests from duplicate step ids.",
            )
        perception_ids = set(step_ids)
        if any(
            dependency_id not in perception_ids
            for step in perception_steps
            for dependency_id in step.depends_on
        ):
            return _cannot_plan(
                trace,
                "pre-plan perception independent of edit artifacts",
                "W1 external perception requests cannot consume draft-plan artifacts.",
            )
        runnable_steps = [step for step in perception_steps if not step.depends_on]
        if not runnable_steps:
            return _cannot_plan(
                trace,
                "acyclic runnable perception roots",
                "W1 found no perception root that can run before draft edits.",
            )
        unknown_operations = self._unknown_perception_operations(perception_steps)
        if unknown_operations:
            return _cannot_plan(
                trace,
                f"registered perception operations: {', '.join(unknown_operations)}",
                "W1 will not issue requests for operations absent from the frozen catalog.",
            )
        if not _all_perception_steps_feed_edits(
            plan,
            set(step_ids),
        ):
            return _cannot_plan(
                trace,
                "perception evidence consumed by a downstream edit",
                "W1 will not spend calls on an unconsumed perception step.",
            )

        call_index = int(context.get("workflow_planner_call_index") or 1)
        requests: list[WorkflowPerceptionRequest] = []
        for step in runnable_steps:
            audio_refs = _audio_refs(step, plan=plan, context=context)
            if not audio_refs:
                return _cannot_plan(
                    trace,
                    f"audio reference for {step.id}",
                    "The planned perception operation has no grounded audio input.",
                )
            requests.append(
                WorkflowPerceptionRequest(
                    request_id=f"w1:{call_index}:{step.id}",
                    scope="targeted",
                    instruction=step.description or plan.request,
                    source_step_id=step.id,
                    operation=step.operation,
                    inputs=copy.deepcopy(step.inputs),
                    params=copy.deepcopy(step.params),
                    expected_output=step.expected_output,
                    audio_refs=audio_refs,
                    requirements=_step_requirements(step),
                    metadata={"planner_call_index": call_index},
                )
            )
        return WorkflowPlannerOutput(
            decision="request_perception",
            plan=plan,
            perception_requests=requests,
            missing_information=[f"perception evidence from {step.id}" for step in runnable_steps],
            reason="Draft plan requires explicit external perception before replanning.",
            invocation_trace=trace,
        )

    def _w2_output(
        self,
        plan: AudioEditPlan,
        *,
        context: dict[str, Any],
        trace: ComponentInvocationTrace | None,
    ) -> WorkflowPlannerOutput:
        plan, prerequisite_normalizations = _normalize_w2_edit_evidence_prerequisites(
            plan,
            context=context,
            allowed_perception_operations=self.allowed_perception_operations,
            perception_operation_params=self.perception_operation_params,
        )
        if prerequisite_normalizations:
            plan.metadata["workflow_evidence_prerequisite_normalizations"] = (
                prerequisite_normalizations
            )
        perception_steps = _perception_steps(plan)
        unknown_operations = self._unknown_perception_operations(perception_steps)
        if unknown_operations:
            return _cannot_plan(
                trace,
                f"registered perception operations: {', '.join(unknown_operations)}",
                "W2 will not execute unregistered fine-grained perception steps.",
            )
        field_sources = context.get("workflow_evidence_field_sources") or {}
        evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for source_ids in field_sources.values()
                if isinstance(source_ids, list)
                for evidence_id in source_ids
                if str(evidence_id)
            )
        )
        if perception_steps and not evidence_ids:
            return _cannot_plan(
                trace,
                "general perception evidence ids",
                "W2 gap steps must cite the general evidence they refine.",
            )
        declarations = [
            WorkflowPlannedPerceptionStep(
                step_id=step.id,
                gap_description=step.description
                or f"Resolve the remaining evidence gap for {step.operation}.",
                evidence_requirements=_step_requirements(step),
                based_on_evidence_ids=evidence_ids,
            )
            for step in perception_steps
        ]
        return WorkflowPlannerOutput(
            decision="execute",
            plan=plan,
            planned_perception_steps=declarations,
            invocation_trace=trace,
        )

    def _unknown_perception_operations(
        self,
        steps: list[AudioEditStep],
    ) -> list[str]:
        return sorted(
            {
                step.operation
                for step in steps
                if step.operation not in self.allowed_perception_operations
            }
        )


PerceptionTraceBuilder = Callable[
    [WorkflowPerceptionRequest, AudioEditPlan],
    ComponentInvocationTrace,
]


class AudioEditExecutorPerceptionAdapter:
    """Execute one explicit perception-only step through AudioEditExecutor."""

    def __init__(
        self,
        executor: AudioEditExecutor,
        *,
        allowed_perception_operations: set[str],
        trace_builder: PerceptionTraceBuilder | None = None,
        deterministic_operations: set[str] | None = None,
        workflow_protocol_policy: WorkflowProtocolPolicy | str | None = None,
    ) -> None:
        self.executor = executor
        self.allowed_perception_operations = set(allowed_perception_operations)
        self.trace_builder = trace_builder
        self.deterministic_operations = set(deterministic_operations or set())
        self.workflow_protocol_policy = resolve_workflow_protocol_policy(workflow_protocol_policy)
        self.last_invocation_trace: ComponentInvocationTrace | None = None
        self.last_tool_calls: list[WorkflowToolCallSummary] = []

    async def perceive(
        self,
        request: WorkflowPerceptionRequest,
        *,
        context: dict[str, Any],
    ) -> WorkflowPerceptionEvidence:
        self.last_invocation_trace = None
        self.last_tool_calls = []
        if request.operation is None:
            raise WorkflowComponentAdapterError(
                "Executor-backed perception requires request.operation"
            )
        if request.operation not in self.allowed_perception_operations:
            raise WorkflowComponentAdapterError(
                f"Perception operation is outside the frozen catalog: {request.operation}"
            )
        if not request.audio_refs:
            raise WorkflowComponentAdapterError(
                "Executor-backed perception requires at least one audio_refs entry"
            )
        if opaque_audio_resolution_enabled({"context": context}):
            try:
                canonicalize_perception_audio_bindings(
                    {},
                    {"context": context},
                    audio_refs=request.audio_refs,
                )
            except Exception as exc:
                raise WorkflowComponentAdapterError(str(exc)) from exc
        input_refs: list[str] = []
        _collect_audio_refs(request.inputs, input_refs)
        undeclared_refs = sorted(
            {
                item
                for item in input_refs
                if item not in _SYMBOLIC_AUDIO_REFS and item not in request.audio_refs
            }
        )
        if undeclared_refs:
            if opaque_audio_resolution_enabled({"context": context}):
                raise WorkflowComponentAdapterError(
                    "Perception inputs contain an undeclared opaque audio ref"
                )
            raise WorkflowComponentAdapterError(
                f"Perception inputs contain undeclared audio refs: {undeclared_refs}"
            )
        plan = AudioEditPlan(
            request=request.instruction,
            input_audio=request.audio_refs[0] if request.audio_refs else None,
            steps=[
                AudioEditStep(
                    id=request.source_step_id or request.request_id,
                    operation=request.operation,
                    description=request.instruction,
                    inputs=copy.deepcopy(request.inputs),
                    params=copy.deepcopy(request.params),
                    expected_output=request.expected_output,
                )
            ],
            metadata={
                "workflow_perception_request_id": request.request_id,
                "workflow_perception_only": True,
                "workflow_perception_audio_refs": list(request.audio_refs),
            },
        )
        if request.operation == "perception.speech_context":
            # A gateway query may return a measured lack of evidence to Reasoner.
            # This never relaxes an editing tool's executable prerequisites.
            plan.steps[0].params["strict"] = False
        execution_context = copy.deepcopy(context)
        execution_context["use_executor_tool_call_planner"] = False
        execution_context["use_gemini_executor"] = False
        primary_audio_ref = str(
            request.inputs.get("input_audio_path")
            or request.inputs.get("input_audio")
            or request.metadata.get("primary_audio_ref")
            or request.audio_refs[0]
            or ""
        ).strip()
        if not primary_audio_ref or primary_audio_ref not in request.audio_refs:
            raise WorkflowComponentAdapterError(
                "Perception request requires one bound primary audio ref."
            )
        execution_context["input_audio"] = primary_audio_ref
        execution_context["input_audios"] = [primary_audio_ref]
        execution_context["input_audio_paths"] = [primary_audio_ref]
        # Each gateway request owns one immutable observation result. Prior
        # reviewed evidence remains available to orchestration, but must not be
        # merged into a different request's frontend report implicitly.
        execution_context["_workflow_perception_isolated_request"] = True
        execution_context["_runtime_perception_audio_refs"] = list(request.audio_refs)
        frontend = self.executor.perception_frontends.get(request.operation)
        if frontend is not None and hasattr(frontend, "last_invocation_trace"):
            frontend.last_invocation_trace = None
        try:
            executed = await self.executor.aexecute(plan, execution_context)
        except Exception as exc:
            trace = self._trace(request, plan, context=context)
            self.last_invocation_trace = trace
            raise WorkflowComponentAdapterError(
                str(exc),
                invocation_trace=trace,
                tool_calls=self.last_tool_calls,
            ) from exc

        tool_calls = _safe_tool_call_summaries(executed)
        self.last_tool_calls = tool_calls
        trace = self._trace(request, executed, context=context)
        self.last_invocation_trace = trace
        step = executed.steps[0]
        if step.status != "success" or not isinstance(step.result, dict):
            raise WorkflowComponentAdapterError(
                step.error or "Perception executor step did not complete.",
                invocation_trace=trace,
                tool_calls=tool_calls,
            )
        result = copy.deepcopy(step.result)
        try:
            result = _canonicalize_perception_result_uncertainties(result)
        except (TypeError, ValueError) as exc:
            raise WorkflowComponentAdapterError(
                str(exc),
                invocation_trace=trace,
                failure_kind="protocol",
                tool_calls=tool_calls,
            ) from exc
        result_for_facts = result
        if opaque_audio_resolution_enabled({"context": context}):
            try:
                # Canonicalize only fields eligible for reviewed evidence. Raw
                # provider/service audit metadata is not an execution input and
                # must not invalidate an otherwise typed observation.
                result_for_facts = _sanitize_perception_facts(
                    result,
                    audio_refs=[],
                    validate_audio_bindings=False,
                )
                result_for_facts = canonicalize_perception_audio_bindings(
                    result_for_facts,
                    {"context": context},
                    audio_refs=request.audio_refs,
                )
                report = result_for_facts.get("perception_report")
                declared_primary_audio_ref = request.metadata.get("primary_audio_ref")
                if (
                    isinstance(report, dict)
                    and not report.get("audio_ref_id")
                    and not report.get("audio_index")
                    and isinstance(declared_primary_audio_ref, str)
                    and declared_primary_audio_ref in request.audio_refs
                ):
                    report["audio_ref_id"] = declared_primary_audio_ref
                    report["audio_index"] = request.audio_refs.index(declared_primary_audio_ref)
            except Exception as exc:
                raise WorkflowComponentAdapterError(
                    str(exc),
                    invocation_trace=trace,
                    failure_kind="protocol",
                    tool_calls=tool_calls,
                ) from exc
        final_output_path = executed.metadata.get("final_output_path")
        final_output_ref = final_output_path
        expected_input_ref = executed.input_audio or ""
        if opaque_audio_resolution_enabled({"context": context}):
            final_output_ref = canonical_audio_ref_for_runtime_path(
                final_output_path,
                {"context": context},
            )
            if final_output_path and final_output_ref is None:
                final_output_ref = "unbound_runtime_output"
            expected_input_ref = primary_audio_ref
        if result.get("output_audio_path") or (
            final_output_ref and str(final_output_ref) != str(expected_input_ref)
        ):
            raise WorkflowComponentAdapterError(
                "External perception gateway cannot publish an edited audio artifact.",
                invocation_trace=trace,
                tool_calls=tool_calls,
            )
        try:
            facts = _evidence_facts(
                result_for_facts,
                audio_refs=request.audio_refs,
            )
        except (TypeError, ValueError) as exc:
            raise WorkflowComponentAdapterError(
                f"Perception evidence namespace rejected: {exc}",
                invocation_trace=trace,
                failure_kind="protocol",
                tool_calls=tool_calls,
            ) from exc
        if request.operation == REVISION_OPERATION and request.params.get("recognition_context"):
            completed_contextual_observation = any(
                isinstance(attempt, dict)
                and attempt.get("status") in {"completed", "insufficient"}
                and not attempt.get("errors")
                and attempt.get("recognition_context") == request.params["recognition_context"]
                for attempt in result.get("attempts") or []
            )
            if not completed_contextual_observation:
                raise WorkflowComponentAdapterError(
                    "Contextual recognition lacks a completed observation receipt.",
                    invocation_trace=trace, tool_calls=tool_calls,
                )
            report = facts.get("perception_report")
            if isinstance(report, dict):
                revision = bind_transcript_revision(
                    report, context=context, audio_ref=primary_audio_ref,
                    request_id=request.request_id,
                )
                if revision is not None:
                    report["transcript_revision"] = revision
        contract_failure = _reviewed_perception_contract_failure(
            request,
            result=result,
            facts=facts,
        )
        if contract_failure is not None:
            raise WorkflowComponentAdapterError(
                contract_failure,
                invocation_trace=trace,
                tool_calls=tool_calls,
            )
        result_failure = _perception_result_failure(
            result,
            has_reviewable_facts=bool(facts),
            policy=self.workflow_protocol_policy,
        )
        if result_failure is not None:
            raise WorkflowComponentAdapterError(
                result_failure,
                invocation_trace=trace,
                tool_calls=tool_calls,
            )
        if not facts:
            raise WorkflowComponentAdapterError(
                "Perception executor returned no reviewable facts.",
                invocation_trace=trace,
                tool_calls=tool_calls,
            )
        covered_audio_refs = _covered_audio_refs_from_reviewed_facts(
            facts,
            requested_audio_refs=request.audio_refs,
        )
        if not covered_audio_refs:
            raise WorkflowComponentAdapterError(
                "Perception executor returned reviewed facts without an audio namespace.",
                invocation_trace=trace,
                tool_calls=tool_calls,
            )
        missing_audio_refs = [
            audio_ref
            for audio_ref in request.audio_refs
            if audio_ref not in set(covered_audio_refs)
        ]
        observation_job = request.metadata.get("observation_job")
        job_requirements = (
            list(observation_job.get("requirements") or [])
            if isinstance(observation_job, dict)
            else list(request.requirements)
        )
        evidence_id = f"evidence:{request.request_id}:{_stable_fingerprint(facts)[:12]}"
        return WorkflowPerceptionEvidence(
            evidence_id=evidence_id,
            request_id=request.request_id,
            scope=request.scope,
            source=f"audio_edit_executor:{request.operation}",
            operation=request.operation,
            expected_evidence_fields=list(request.metadata.get("expected_evidence_fields") or []),
            requested_audio_refs=list(request.audio_refs),
            audio_refs=covered_audio_refs,
            missing_audio_evidence=[
                {
                    "audio_ref": audio_ref,
                    "missing_requirements": job_requirements,
                    "reason": (
                        "The completed Perception result contained no reviewed "
                        "facts bound to this requested audio."
                    ),
                    "blocking": bool(request.required),
                }
                for audio_ref in missing_audio_refs
            ],
            facts=facts,
            uncertainties=_uncertainties(result),
            usable=True,
            invocation_trace=trace,
            tool_calls=tool_calls,
        )

    def _trace(
        self,
        request: WorkflowPerceptionRequest,
        plan: AudioEditPlan,
        *,
        context: dict[str, Any],
    ) -> ComponentInvocationTrace | None:
        result = plan.steps[0].result if plan.steps else None
        raw_trace = result.get("component_invocation_trace") if isinstance(result, dict) else None
        if raw_trace is None:
            frontend = self.executor.perception_frontends.get(request.operation)
            raw_trace = getattr(frontend, "last_invocation_trace", None)
        if raw_trace is not None:
            trace = validate_component_invocation_trace(raw_trace)
        elif self.trace_builder is not None:
            trace = validate_component_invocation_trace(self.trace_builder(request, plan))
        elif request.operation in self.deterministic_operations:
            trace = ComponentInvocationTrace.build(
                component="perception",
                component_name=type(self).__name__,
                route="deterministic",
                status=(
                    "completed" if plan.steps and plan.steps[0].status == "success" else "failed"
                ),
                metadata={
                    "operation": request.operation,
                    "backend_attempts": _safe_backend_attempts(result),
                },
            )
        else:
            trace = None
        if trace is not None and trace.component != "perception":
            raise WorkflowComponentAdapterError(
                "Perception trace has the wrong component.",
                invocation_trace=trace,
            )
        if trace is not None:
            result_mapping = result if isinstance(result, dict) else {}
            route_policy = request.metadata.get("perception_route_policy")
            safe_backend_attempts = _safe_backend_attempts(result)
            trace_metadata = {
                **trace.metadata,
                "operation": request.operation,
                "planned_route_id": (
                    route_policy.get("route_id") if isinstance(route_policy, dict) else None
                ),
                "actual_route_id": (
                    result_mapping.get("perception_route_id")
                    or result_mapping.get("grounding_route")
                    or trace.metadata.get("actual_route_id")
                ),
                "route_policy_fingerprint": request.metadata.get(
                    "perception_route_policy_fingerprint"
                ),
                "backend_attempts": (
                    safe_backend_attempts or trace.metadata.get("backend_attempts")
                ),
            }
            trace_metadata = {
                key: value for key, value in trace_metadata.items() if value not in (None, [], {})
            }
            gateway_call_id = (
                f"perception:{int(context.get('workflow_perception_call_index') or 1)}"
            )
            if trace.gateway_call_id is not None and trace.gateway_call_id != gateway_call_id:
                raise WorkflowComponentAdapterError(
                    "Perception trace is bound to a different gateway call.",
                    invocation_trace=trace,
                )
            trace = trace.model_copy(
                update={
                    "gateway_call_id": gateway_call_id,
                    "metadata": trace_metadata,
                },
                deep=True,
            )
        return trace


def _safe_backend_attempts(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    raw_attempts = result.get("attempts")
    if not isinstance(raw_attempts, list):
        return []
    safe: list[dict[str, Any]] = []
    for raw in raw_attempts:
        if not isinstance(raw, dict):
            continue
        item = {
            key: raw.get(key)
            for key in (
                "provider",
                "status",
                "audio_index",
                "language",
                "refresh_cache",
            )
            if isinstance(raw.get(key), (str, int, float, bool))
        }
        if item:
            safe.append(item)
    return safe


class ReviewedWorkflowEvidenceProjector:
    """Admit only reviewed, provenance-bound evidence into component contexts."""

    def project(
        self,
        evidence: list[WorkflowPerceptionEvidence],
    ) -> WorkflowEvidenceProjection:
        for item in evidence:
            _validate_current_evidence_audio_namespace(item)
        evidence_ids = [item.evidence_id for item in evidence]
        planner_context: dict[str, Any] = {}
        execution_context: dict[str, Any] = {}
        field_sources: dict[str, list[str]] = {}
        path_sources: dict[str, list[str]] = {}
        path_audio_refs: dict[str, str] = {}

        audio_namespaces = {
            tuple(sorted(dict.fromkeys(item.requested_audio_refs)))
            for item in evidence
            if item.requested_audio_refs
        }
        requested_audio_refs = list(
            dict.fromkeys(audio_ref for item in evidence for audio_ref in item.requested_audio_refs)
        )
        source_audio_refs = list(
            dict.fromkeys(audio_ref for item in evidence for audio_ref in item.audio_refs)
        )
        if len(requested_audio_refs) > 1:
            return _project_multi_audio_reviewed_evidence(
                evidence,
                requested_audio_refs=requested_audio_refs,
            )
        if len(audio_namespaces) > 1:
            raise ValueError("multiple audio namespaces require a namespaced evidence projection")
        if audio_namespaces and any(not item.audio_refs for item in evidence):
            raise ValueError("all merged evidence must declare its source audio namespace")
        prepared_sources: list[tuple[WorkflowPerceptionEvidence, ReviewedEvidenceSource]] = []
        for item in evidence:
            facts = _sanitize_perception_facts(
                item.facts,
                audio_refs=item.requested_audio_refs,
            )
            report = facts.get("perception_report")
            if not isinstance(report, dict):
                continue
            sanitized_report = _sanitize_perception_report(report)
            if not sanitized_report:
                continue
            report_audio_ref = sanitized_report.get("audio_ref_id")
            if not isinstance(report_audio_ref, str) or not report_audio_ref:
                item_audio_refs = item.audio_refs or item.requested_audio_refs
                report_audio_ref = item_audio_refs[0] if len(item_audio_refs) == 1 else None
            prepared_sources.append(
                (
                    item,
                    build_reviewed_evidence_source(
                        evidence_id=item.evidence_id,
                        request_id=item.request_id,
                        source=item.source,
                        operation=item.operation,
                        expected_evidence_fields=item.expected_evidence_fields,
                        audio_ref_id=report_audio_ref,
                        report=sanitized_report,
                    ),
                )
            )

        source_records = [source for _item, source in prepared_sources]
        authority_assignments, authority_conflicts = resolve_evidence_authority(source_records)
        reports: list[tuple[str, dict[str, Any]]] = []
        scalar_candidates: dict[str, list[tuple[str, Any]]] = {
            field: [] for field in _PROJECTED_SCALAR_FIELDS
        }
        for item, source in prepared_sources:
            # Bind local references before authority filtering removes supporting
            # definitions. Canonical reconciliation below remaps timed word IDs.
            namespaced_report = _namespace_reviewed_local_ids(
                source.report,
                evidence_id=item.evidence_id,
                audio_ref_id=source.audio_ref_id,
            )
            namespaced_report = scope_report_to_authority(
                source, authority_assignments, bound_report=namespaced_report,
            )
            if not namespaced_report:
                continue
            reports.append((item.evidence_id, namespaced_report))
            for path in _reviewed_projection_paths(
                namespaced_report,
                prefix="perception_report",
            ):
                path_sources.setdefault(path, []).append(item.evidence_id)
                if len(item.audio_refs) == 1:
                    path_audio_refs[path] = item.audio_refs[0]
            for field in _PROJECTED_SCALAR_FIELDS:
                if field in namespaced_report:
                    scalar_candidates[field].append((item.evidence_id, namespaced_report[field]))

        if reports:
            merged_report: dict[str, Any] = {}
            for evidence_id, report in reports:
                merged_report = _merge_reviewed_values(
                    merged_report,
                    report,
                    path="perception_report",
                )
                field_sources.setdefault("perception_report", []).append(evidence_id)
            merged_report = append_conflict_uncertainties(
                merged_report,
                [
                    conflict
                    for conflict in authority_conflicts
                    if conflict.get("audio_ref_id") in {None, merged_report.get("audio_ref_id")}
                ],
            )
            merged_report, reconciliation_conflicts = reconcile_canonical_reviewed_report(
                merged_report,
                authority_evidence_ids=_authority_evidence_ids_by_kind(
                    authority_assignments,
                    audio_ref_id=merged_report.get("audio_ref_id"),
                ),
            )
            authority_conflicts.extend(reconciliation_conflicts)
            planner_context["perception_report"] = merged_report
            execution_context["perception_report"] = copy.deepcopy(merged_report)

        for field, candidates in scalar_candidates.items():
            if not candidates:
                continue
            value = _one_consistent_value(candidates, field=field)
            value = _sanitize_reviewed_value(value)
            planner_context[field] = value
            execution_context[field] = copy.deepcopy(value)
            field_sources[field] = list(dict.fromkeys(evidence_id for evidence_id, _ in candidates))
            path_sources[field] = list(field_sources[field])
            if len(source_audio_refs) == 1:
                path_audio_refs[field] = source_audio_refs[0]

        if evidence and not planner_context:
            raise ValueError("perception evidence contains no reviewed Planner/Executor fields")

        projection = WorkflowEvidenceProjection(
            evidence_ids=evidence_ids,
            requested_audio_refs=requested_audio_refs,
            source_audio_refs=source_audio_refs,
            missing_audio_refs=[
                item for item in requested_audio_refs if item not in set(source_audio_refs)
            ],
            planner_context=planner_context,
            execution_context=execution_context,
            field_sources=field_sources,
            path_sources={
                path: list(dict.fromkeys(source_ids)) for path, source_ids in path_sources.items()
            },
            path_audio_refs=path_audio_refs,
            reviewed_sources=[source.audit_record() for source in source_records],
            authority_assignments=[
                assignment.audit_record() for assignment in authority_assignments
            ],
            evidence_conflicts=_dedupe_evidence_conflicts(authority_conflicts),
            projection_fingerprint="pending",
        )
        _attach_competing_observations(projection, source_records, authority_assignments)
        projection.projection_fingerprint = workflow_evidence_projection_fingerprint(projection)
        return projection


def _validate_current_evidence_audio_namespace(
    evidence: WorkflowPerceptionEvidence,
) -> None:
    """Recheck mutable evidence before trusting its requested/covered split."""

    requested = list(evidence.requested_audio_refs)
    covered = list(evidence.audio_refs)
    missing = [item.audio_ref for item in evidence.missing_audio_evidence]
    for field, values in (
        ("requested_audio_refs", requested),
        ("audio_refs", covered),
        ("missing_audio_evidence", missing),
    ):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"perception evidence {field} contains a blank audio ref")
        if len(values) != len(set(values)):
            raise ValueError(f"perception evidence {field} contains duplicate audio refs")

    requested_set = set(requested)
    covered_set = set(covered)
    missing_set = set(missing)
    if not covered_set.issubset(requested_set):
        unexpected = sorted(covered_set - requested_set)
        raise ValueError(f"covered perception audio refs must have been requested: {unexpected}")
    if not missing_set.issubset(requested_set):
        unexpected = sorted(missing_set - requested_set)
        raise ValueError(f"missing perception audio refs must have been requested: {unexpected}")
    overlap = sorted(covered_set.intersection(missing_set))
    if overlap:
        raise ValueError(f"one audio ref cannot be both covered and missing: {overlap}")


def _project_multi_audio_reviewed_evidence(
    evidence: list[WorkflowPerceptionEvidence],
    *,
    requested_audio_refs: list[str],
) -> WorkflowEvidenceProjection:
    evidence_ids = [item.evidence_id for item in evidence]
    reports_by_audio_ref: dict[str, dict[str, Any]] = {}
    contributing_evidence_ids: list[str] = []
    path_sources: dict[str, list[str]] = {}
    path_audio_refs: dict[str, str] = {}
    covered_audio_refs = list(
        dict.fromkeys(audio_ref for item in evidence for audio_ref in item.audio_refs)
    )
    missing_by_audio_ref = {
        missing.audio_ref: missing for item in evidence for missing in item.missing_audio_evidence
    }
    prepared_sources: list[tuple[WorkflowPerceptionEvidence, ReviewedEvidenceSource]] = []

    for item in evidence:
        audio_refs = _validated_multi_audio_evidence_refs(item)
        if not audio_refs:
            continue
        facts = _sanitize_perception_facts(
            item.facts,
            audio_refs=item.requested_audio_refs,
        )
        report = facts.get("perception_report")
        if not isinstance(report, dict):
            continue
        partitions = _partition_reviewed_report_by_audio_ref(
            report,
            evidence_audio_refs=item.requested_audio_refs,
            projection_audio_refs=requested_audio_refs,
        )
        unexpected_partitions = sorted(set(partitions) - set(audio_refs))
        if unexpected_partitions:
            raise ValueError(
                "reviewed facts claim coverage outside completed audio refs: "
                f"{unexpected_partitions}"
            )
        unsubstantiated_refs = sorted(set(audio_refs) - set(partitions))
        if unsubstantiated_refs:
            raise ValueError(
                "completed perception audio refs have no admitted reviewed facts: "
                f"{unsubstantiated_refs}"
            )
        if partitions:
            contributing_evidence_ids.append(item.evidence_id)
        for audio_ref_id, partition in partitions.items():
            prepared_sources.append(
                (
                    item,
                    build_reviewed_evidence_source(
                        evidence_id=item.evidence_id,
                        request_id=item.request_id,
                        source=item.source,
                        operation=item.operation,
                        expected_evidence_fields=item.expected_evidence_fields,
                        audio_ref_id=audio_ref_id,
                        report=partition,
                    ),
                )
            )

    source_records = [source for _item, source in prepared_sources]
    authority_assignments, authority_conflicts = resolve_evidence_authority(source_records)
    all_conflicts = list(authority_conflicts)
    for item, source in prepared_sources:
        audio_ref_id = source.audio_ref_id
        if audio_ref_id is None:
            raise ValueError("multi-audio reviewed evidence requires an audio_ref_id")
        partition = _namespace_reviewed_local_ids(
            source.report,
            evidence_id=item.evidence_id,
            audio_ref_id=audio_ref_id,
        )
        partition = scope_report_to_authority(
            source, authority_assignments, bound_report=partition,
        )
        report_path = _multi_audio_report_path(audio_ref_id)
        if audio_ref_id in reports_by_audio_ref:
            reports_by_audio_ref[audio_ref_id] = _merge_reviewed_values(
                reports_by_audio_ref[audio_ref_id],
                partition,
                path=report_path,
            )
        else:
            reports_by_audio_ref[audio_ref_id] = copy.deepcopy(partition)
        entry_path = _multi_audio_report_entry_path(audio_ref_id)
        for path in _reviewed_projection_paths(partition, prefix=report_path):
            path_sources.setdefault(path, []).append(item.evidence_id)
            path_audio_refs[path] = audio_ref_id
        for path in (
            f"{entry_path}.schema_version",
            f"{entry_path}.audio_ref_id",
            f"{entry_path}.status",
        ):
            path_sources.setdefault(path, []).append(item.evidence_id)
            path_audio_refs[path] = audio_ref_id

    for audio_ref_id, report in list(reports_by_audio_ref.items()):
        report = append_conflict_uncertainties(
            report,
            [
                conflict
                for conflict in authority_conflicts
                if conflict.get("audio_ref_id") == audio_ref_id
            ],
        )
        report, reconciliation_conflicts = reconcile_canonical_reviewed_report(
            report,
            authority_evidence_ids=_authority_evidence_ids_by_kind(
                authority_assignments,
                audio_ref_id=audio_ref_id,
            ),
        )
        all_conflicts.extend(reconciliation_conflicts)
        reports_by_audio_ref[audio_ref_id] = report

    if evidence and not reports_by_audio_ref:
        raise ValueError("perception evidence contains no reviewed Planner/Executor fields")

    audio_reports: list[dict[str, Any]] = []
    for audio_ref_id in requested_audio_refs:
        if audio_ref_id in reports_by_audio_ref:
            audio_reports.append(
                {
                    "schema_version": _MULTI_AUDIO_REPORT_ENTRY_SCHEMA_VERSION,
                    "audio_ref_id": audio_ref_id,
                    "status": "completed",
                    "report": reports_by_audio_ref[audio_ref_id],
                }
            )
            continue
        missing = missing_by_audio_ref.get(audio_ref_id)
        audio_reports.append(
            {
                "schema_version": _MULTI_AUDIO_REPORT_ENTRY_SCHEMA_VERSION,
                "audio_ref_id": audio_ref_id,
                "status": "missing_evidence",
                "missing_requirements": (
                    list(missing.missing_requirements) if missing is not None else []
                ),
                "reason": (
                    missing.reason
                    if missing is not None
                    else "No completed reviewed facts were returned for this audio."
                ),
                "blocking": bool(missing.blocking) if missing is not None else True,
                "report": {},
            }
        )
    namespaced_report = {
        "schema_version": _MULTI_AUDIO_REPORT_SCHEMA_VERSION,
        "audio_reports": audio_reports,
    }
    report_source_ids = list(dict.fromkeys(contributing_evidence_ids))
    projection = WorkflowEvidenceProjection(
        evidence_ids=evidence_ids,
        requested_audio_refs=requested_audio_refs,
        source_audio_refs=covered_audio_refs,
        missing_audio_refs=[
            audio_ref
            for audio_ref in requested_audio_refs
            if audio_ref not in set(covered_audio_refs)
        ],
        planner_context={"perception_report": namespaced_report},
        execution_context={
            "perception_report": copy.deepcopy(namespaced_report),
        },
        field_sources={"perception_report": report_source_ids},
        path_sources={
            path: list(dict.fromkeys(source_ids)) for path, source_ids in path_sources.items()
        },
        path_audio_refs=path_audio_refs,
        reviewed_sources=[source.audit_record() for source in source_records],
        authority_assignments=[assignment.audit_record() for assignment in authority_assignments],
        evidence_conflicts=_dedupe_evidence_conflicts(all_conflicts),
        projection_fingerprint="pending",
    )
    _attach_competing_observations(projection, source_records, authority_assignments)
    projection.projection_fingerprint = workflow_evidence_projection_fingerprint(projection)
    return projection


def _attach_competing_observations(
    projection: WorkflowEvidenceProjection,
    sources: Sequence[ReviewedEvidenceSource],
    assignments: Sequence[EvidenceAuthorityAssignment],
) -> None:
    """Expose diagnostics after executable contexts and paths have been frozen."""

    report = projection.planner_context.get("perception_report", {})
    reports = (
        [entry.get("report", {}) for entry in report.get("audio_reports", [])]
        if "audio_reports" in report
        else [report]
    )
    observations = competing_observations(sources, assignments)
    for audio_report in reports:
        scoped = [
            item for item in observations
            if item["audio_ref_id"] == audio_report.get("audio_ref_id")
        ]
        if scoped:
            audio_report["competing_observations"] = scoped


def _validated_multi_audio_evidence_refs(
    evidence: WorkflowPerceptionEvidence,
) -> list[str]:
    refs = list(evidence.audio_refs)
    if not evidence.requested_audio_refs:
        raise ValueError("all multi-audio evidence must declare its audio_ref_id namespace")
    if any(not str(audio_ref).strip() for audio_ref in refs):
        raise ValueError("multi-audio evidence contains a blank audio_ref_id")
    if len(refs) != len(set(refs)):
        raise ValueError("multi-audio evidence contains duplicate audio_ref_id values")
    return refs


def _partition_reviewed_report_by_audio_ref(
    report: dict[str, Any],
    *,
    evidence_audio_refs: list[str],
    projection_audio_refs: list[str],
) -> dict[str, dict[str, Any]]:
    report_audio_ref = _mapping_audio_ref_id(
        report,
        audio_refs=evidence_audio_refs,
        path="perception_report",
    )
    if report_audio_ref is not None or len(evidence_audio_refs) == 1:
        audio_ref_id = report_audio_ref or evidence_audio_refs[0]
        normalized = _normalize_reviewed_audio_bindings(
            report,
            evidence_audio_refs=evidence_audio_refs,
            projection_audio_refs=projection_audio_refs,
            expected_audio_ref=audio_ref_id,
            path="perception_report",
        )
        normalized["audio_ref_id"] = audio_ref_id
        normalized["audio_index"] = projection_audio_refs.index(audio_ref_id)
        return {audio_ref_id: normalized}

    partitions: dict[str, dict[str, Any]] = {}
    source_text_fields = {
        "source_text",
        "source_text_audio_index",
        "source_text_audio_path",
        "source_text_source",
    }
    present_source_text_fields = source_text_fields.intersection(report)
    if present_source_text_fields:
        source_text_audio_ref = _source_text_audio_ref_id(
            report,
            audio_refs=evidence_audio_refs,
            path="perception_report",
        )
        if source_text_audio_ref is None:
            raise ValueError(
                "multi-audio source_text evidence requires "
                "source_text_audio_path or source_text_audio_index"
            )
        partition = partitions.setdefault(source_text_audio_ref, {})
        for field in source_text_fields:
            if field in report:
                partition[field] = copy.deepcopy(report[field])
        partition["source_text_audio_path"] = source_text_audio_ref
        partition["source_text_audio_index"] = projection_audio_refs.index(source_text_audio_ref)

    ignored_fields = {
        "schema_version",
        "audio_ref_id",
        "audio_index",
        *source_text_fields,
    }
    for field, value in report.items():
        if field in ignored_fields:
            continue
        field_path = f"perception_report.{field}"
        if isinstance(value, list):
            for index, raw_item in enumerate(value):
                if not isinstance(raw_item, dict):
                    raise TypeError(f"ambiguous multi-audio namespace at {field_path}[{index}]")
                audio_ref_id = _mapping_audio_ref_id(
                    raw_item,
                    audio_refs=evidence_audio_refs,
                    path=f"{field_path}[{index}]",
                )
                if audio_ref_id is None:
                    raise ValueError(f"missing audio_ref_id namespace at {field_path}[{index}]")
                normalized_item = _normalize_reviewed_audio_bindings(
                    raw_item,
                    evidence_audio_refs=evidence_audio_refs,
                    projection_audio_refs=projection_audio_refs,
                    expected_audio_ref=audio_ref_id,
                    path=f"{field_path}[{index}]",
                )
                partitions.setdefault(audio_ref_id, {}).setdefault(
                    field,
                    [],
                ).append(normalized_item)
            continue
        if isinstance(value, dict):
            audio_ref_id = _mapping_audio_ref_id(
                value,
                audio_refs=evidence_audio_refs,
                path=field_path,
            )
            if audio_ref_id is None:
                raise ValueError(f"missing audio_ref_id namespace at {field_path}")
            normalized_value = _normalize_reviewed_audio_bindings(
                value,
                evidence_audio_refs=evidence_audio_refs,
                projection_audio_refs=projection_audio_refs,
                expected_audio_ref=audio_ref_id,
                path=field_path,
            )
            partitions.setdefault(audio_ref_id, {})[field] = normalized_value
            continue
        raise ValueError(f"ambiguous multi-audio namespace at {field_path}")

    for audio_ref_id, partition in partitions.items():
        partition["audio_ref_id"] = audio_ref_id
        partition["audio_index"] = projection_audio_refs.index(audio_ref_id)
        partition["schema_version"] = report["schema_version"]
    return partitions


def _normalize_reviewed_audio_bindings(
    value: Any,
    *,
    evidence_audio_refs: list[str],
    projection_audio_refs: list[str],
    expected_audio_ref: str,
    path: str,
) -> Any:
    if isinstance(value, list):
        return [
            _normalize_reviewed_audio_bindings(
                item,
                evidence_audio_refs=evidence_audio_refs,
                projection_audio_refs=projection_audio_refs,
                expected_audio_ref=expected_audio_ref,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    normalized = copy.deepcopy(value)
    audio_ref_id = _mapping_audio_ref_id(
        normalized,
        audio_refs=evidence_audio_refs,
        path=path,
    )
    if audio_ref_id is not None:
        if audio_ref_id != expected_audio_ref:
            raise ValueError(
                f"audio_ref_id namespace at {path} conflicts with {expected_audio_ref!r}"
            )
        normalized["audio_ref_id"] = audio_ref_id
        normalized["audio_index"] = projection_audio_refs.index(audio_ref_id)

    source_text_audio_ref = _source_text_audio_ref_id(
        normalized,
        audio_refs=evidence_audio_refs,
        path=path,
    )
    if source_text_audio_ref is not None:
        if source_text_audio_ref != expected_audio_ref:
            raise ValueError(
                f"source-text audio namespace at {path} conflicts with {expected_audio_ref!r}"
            )
        normalized["source_text_audio_path"] = source_text_audio_ref
        normalized["source_text_audio_index"] = projection_audio_refs.index(source_text_audio_ref)

    for key, child in list(normalized.items()):
        if key in {
            "audio_ref_id",
            "audio_index",
            "source_text_audio_path",
            "source_text_audio_index",
        }:
            continue
        normalized[key] = _normalize_reviewed_audio_bindings(
            child,
            evidence_audio_refs=evidence_audio_refs,
            projection_audio_refs=projection_audio_refs,
            expected_audio_ref=expected_audio_ref,
            path=f"{path}.{key}",
        )
    return normalized


def _mapping_audio_ref_id(
    value: dict[str, Any],
    *,
    audio_refs: list[str],
    path: str,
) -> str | None:
    raw_ref = value.get("audio_ref_id")
    raw_index = value.get("audio_index")
    if raw_ref is None and raw_index is None:
        return None
    if raw_index is not None:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(f"audio_index at {path} must be an integer")
        if raw_index < 0 or raw_index >= len(audio_refs):
            raise ValueError(f"audio_index at {path} is outside the evidence namespace")
    audio_ref_id = str(raw_ref) if raw_ref is not None else audio_refs[raw_index]
    if audio_ref_id not in audio_refs:
        raise ValueError(f"audio_ref_id at {path} is outside the evidence namespace")
    if raw_index is not None and audio_refs[raw_index] != audio_ref_id:
        raise ValueError(f"audio_ref_id and audio_index at {path} disagree")
    return audio_ref_id


def _source_text_audio_ref_id(
    value: dict[str, Any],
    *,
    audio_refs: list[str],
    path: str,
) -> str | None:
    raw_path = value.get("source_text_audio_path")
    raw_index = value.get("source_text_audio_index")
    if raw_path is None and raw_index is None:
        return None
    if raw_index is not None:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(f"source_text_audio_index at {path} must be an integer")
        if raw_index < 0 or raw_index >= len(audio_refs):
            raise ValueError(f"source_text_audio_index at {path} is outside the evidence namespace")
    audio_ref_id = str(raw_path) if raw_path is not None else audio_refs[raw_index]
    if audio_ref_id not in audio_refs:
        raise ValueError(f"source_text_audio_path at {path} is outside the evidence namespace")
    if raw_index is not None and audio_refs[raw_index] != audio_ref_id:
        raise ValueError(f"source_text audio path and index at {path} disagree")
    return audio_ref_id


def _multi_audio_report_entry_path(audio_ref_id: str) -> str:
    return f"perception_report.audio_reports[audio_ref_id={(audio_ref_id,)}]"


def _multi_audio_report_path(audio_ref_id: str) -> str:
    return f"{_multi_audio_report_entry_path(audio_ref_id)}.report"


def _unique_perception_route_by_evidence_overlap(
    *,
    target_kinds: Any,
    expected_evidence_fields: list[str],
    temporal_precision: str,
    available_operations: set[str],
) -> str | None:
    """Recover one route when Planner copied fields from the wrong catalog row."""

    target_kind_values = list(target_kinds)
    target_compatible = compatible_workflow_perception_operations(
        target_kinds=target_kind_values,
        expected_evidence_fields=(),
        temporal_precision=temporal_precision,
        available_operations=available_operations,
    )
    requested_roots = {_evidence_field_root(field) for field in expected_evidence_fields}
    if "speaker_hint" not in target_kind_values:
        # Stale speaker fields copied from another catalog row should not make
        # a non-speaker semantic request prefer or tie with a speaker route.
        requested_roots.difference_update(
            {
                "speaker_segments",
                "speaker_attributed_transcript",
                "selected_speaker_refs",
                "selected_speaker_ids",
            }
        )
    scores: list[tuple[int, str]] = []
    for operation in target_compatible:
        capability = workflow_perception_capability(operation)
        if capability is None:
            continue
        score = len(requested_roots.intersection(capability.evidence_types))
        scores.append((score, operation))
    if not scores:
        return None
    best_score = max(score for score, _ in scores)
    if best_score <= 0:
        return None
    best = [operation for score, operation in scores if score == best_score]
    return best[0] if len(best) == 1 else None


def _capability_selection_contract_errors(
    turn: PlannerReasonerTurn,
    *,
    plan: AudioEditPlan,
    capability_cards: dict[str, OperationCapabilityCard],
    frozen_params: dict[str, dict[str, Any]],
) -> list[str]:
    """Cross-check Planner declarations against cards and the concrete plan."""

    if not capability_cards:
        return []
    if turn.tool_selection is None:
        return ["tool_selection is required when operation capability cards are active"]
    selections = {item.plan_step_id: item for item in turn.tool_selection.steps}
    errors: list[str] = []
    for step in plan.steps:
        card = capability_cards.get(step.operation)
        if card is None:
            errors.append(f"{step.id}: no capability card for {step.operation}")
            continue
        selection = selections.get(step.id)
        if selection is None:
            errors.append(f"{step.id}: missing tool selection declaration")
            continue
        unknown_candidates = sorted(
            candidate.operation
            for candidate in selection.candidates
            if candidate.operation not in capability_cards
        )
        if unknown_candidates:
            errors.append(
                f"{step.id}: candidate operations lack capability cards {unknown_candidates}"
            )
        frozen_values = {
            f"params.{key}": value
            for key, value in (frozen_params.get(step.operation) or {}).items()
        }
        violations = validate_operation_arguments(
            card,
            inputs=step.inputs,
            params=step.params,
            framework_values=frozen_values,
        )
        errors.extend(
            f"{step.id}: {item.code} at {item.path or '<operation>'}" for item in violations
        )
        declared_paths = {item.argument_path for item in selection.parameter_selections}
        selectable_specs = {item.path: item for item in card.parameter_contract.planner_selectable}
        selectable_paths = set(selectable_specs)
        framework_paths = set(card.parameter_contract.framework_owned)
        extra_declarations = sorted(
            path
            for path in declared_paths - selectable_paths - framework_paths
            if not any(
                path.startswith((f"{container}.", f"{container}["))
                and spec.value_type in {"object", "array"}
                for container, spec in selectable_specs.items()
            )
        )
        if extra_declarations:
            errors.append(
                f"{step.id}: parameter declarations are outside the legal set {extra_declarations}"
            )
    return errors


def _normalize_validated_single_objective_step_set(
    plan: AudioEditPlan,
    *,
    tool_selection: Any,
) -> tuple[AudioEditPlan, list[dict[str, Any]]]:
    """Repair structural coverage only after every required step is card-valid."""

    metadata = plan.metadata if isinstance(plan.metadata, dict) else {}
    objectives = [
        item
        for item in metadata.get("high_level_objectives") or []
        if isinstance(item, dict) and item.get("required", True)
    ]
    coverage = metadata.get("objective_coverage")
    required_steps = [
        step
        for step in plan.steps
        if step.required
        and step.operation not in {"", "unsupported"}
        and not step.operation.startswith(("perception.", "flexsed."))
    ]
    if (
        len(objectives) != 1
        or not required_steps
        or not isinstance(coverage, list)
        or tool_selection is None
    ):
        return plan, []

    selections = {item.plan_step_id: item for item in tool_selection.steps}
    for step in required_steps:
        selection = selections.get(step.id)
        if selection is None or selection.selected_operation != step.operation:
            return plan, []
        selected_candidate = next(
            (
                candidate
                for candidate in selection.candidates
                if candidate.operation == step.operation
            ),
            None,
        )
        if (
            selected_candidate is None
            or selected_candidate.semantic_fit != "full"
            or not selected_candidate.legal_inputs_available
        ):
            return plan, []

    objective_id = str(objectives[0].get("id") or "")
    step_ids = [step.id for step in required_steps]
    normalized = plan.model_copy(deep=True)
    normalized_coverage = normalized.metadata.get("objective_coverage") or []
    changed = False
    for item in normalized_coverage:
        if (
            not isinstance(item, dict)
            or str(item.get("objective_id") or "") != objective_id
            or item.get("status") != "uncovered"
        ):
            continue
        item["covered_by_steps"] = step_ids
        item["status"] = "planned"
        item["coverage_reason"] = (
            "The card-validated required step set is structurally responsible "
            "for the only required objective; outcome quality remains unevaluated."
        )
        changed = True
    if not changed:
        return plan, []
    return normalized, [
        {
            "objective_id": objective_id,
            "covered_by_steps": step_ids,
            "reason": "validated_single_objective_required_step_set",
        }
    ]


def _evidence_field_root(field_path: str) -> str:
    normalized = str(field_path).strip()
    for prefix in ("facts.", "perception_report."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized.split(".", 1)[0].split("[", 1)[0]


def _semantic_binding_protocol_audit(
    authored: Any,
    admitted: Any,
) -> dict[str, Any]:
    authored_by_step = {binding.plan_step_id: binding for binding in authored.bindings}
    admitted_by_step = {binding.plan_step_id: binding for binding in admitted.bindings}
    dropped = [
        {
            "plan_step_id": step_id,
            "authored_binding_fingerprint": _stable_fingerprint(binding.model_dump(mode="json")),
            "reason": "binding_does_not_control_an_evidence_required_step",
        }
        for step_id, binding in authored_by_step.items()
        if step_id not in admitted_by_step
    ]
    normalizations: list[dict[str, Any]] = []
    dropped_argument_sources: list[dict[str, Any]] = []
    for step_id, admitted_binding in admitted_by_step.items():
        authored_binding = authored_by_step.get(step_id)
        if authored_binding is None:
            continue
        authored_sources = {
            (
                item.argument_path,
                item.source_kind,
                item.evidence_id,
                item.evidence_path,
            )
            for item in authored_binding.argument_sources
        }
        admitted_references = {
            (
                item.source_kind,
                item.evidence_id,
                item.evidence_path,
            )
            for item in admitted_binding.argument_sources
        }
        for source in authored_binding.argument_sources:
            if (
                source.source_kind,
                source.evidence_id,
                source.evidence_path,
            ) in admitted_references:
                continue
            dropped_argument_sources.append(
                {
                    "plan_step_id": step_id,
                    "argument_path": source.argument_path,
                    "source_kind": source.source_kind,
                    "evidence_id": source.evidence_id,
                    "evidence_path": source.evidence_path,
                    "reason": "argument_path_not_present_in_executable_step",
                }
            )
        for source in admitted_binding.argument_sources:
            source_key = (
                source.argument_path,
                source.source_kind,
                source.evidence_id,
                source.evidence_path,
            )
            if source_key in authored_sources:
                continue
            matching_reference = next(
                (
                    item
                    for item in authored_binding.argument_sources
                    if item.source_kind == source.source_kind
                    and item.evidence_id == source.evidence_id
                    and item.evidence_path == source.evidence_path
                ),
                None,
            )
            normalizations.append(
                {
                    "plan_step_id": step_id,
                    "kind": (
                        "canonicalized_registered_argument_alias"
                        if matching_reference is not None
                        else "inferred_exact_registered_span_source"
                    ),
                    "from_argument_path": (
                        matching_reference.argument_path if matching_reference is not None else None
                    ),
                    "argument_path": source.argument_path,
                    "evidence_id": source.evidence_id,
                    "evidence_path": source.evidence_path,
                }
            )
    return {
        "authored_binding_count": len(authored_by_step),
        "admitted_binding_count": len(admitted_by_step),
        "dropped_bindings": dropped,
        "dropped_argument_sources": dropped_argument_sources,
        "normalizations": normalizations,
    }


def _local_destructive_boundary_error(
    plan: AudioEditPlan,
    *,
    context: dict[str, Any],
) -> str | None:
    report = context.get("perception_report")
    caption = report.get("timestamped_caption") if isinstance(report, dict) else None
    duration_value = caption.get("audio_duration_sec") if isinstance(caption, dict) else None
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        return None
    if duration <= 5.0:
        return None
    for step in plan.steps:
        if step.operation not in _LOCAL_DESTRUCTIVE_SPAN_OPERATIONS:
            continue
        raw_spans = step.params.get("spans")
        if raw_spans is None:
            raw_spans = step.params.get("selected_spans")
        if raw_spans is None:
            raw_spans = step.inputs.get("spans")
        if raw_spans is None:
            raw_spans = step.inputs.get("selected_spans")
        if not isinstance(raw_spans, list):
            continue
        intervals: list[tuple[float, float]] = []
        for raw_span in raw_spans:
            if not isinstance(raw_span, dict):
                continue
            try:
                start = max(0.0, float(raw_span["start_sec"]))
                end = min(duration, float(raw_span["end_sec"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                intervals.append((start, end))
        covered = _merged_interval_duration(intervals)
        if covered >= 0.9 * duration:
            return (
                "Planner/Reasoner local destructive edit covers nearly the "
                f"whole audio without local boundaries: {step.id}"
            )
    return None


def _merged_interval_duration(
    intervals: list[tuple[float, float]],
) -> float:
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for start, end in sorted(intervals):
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def _component_trace(
    component: Any,
    *,
    gateway_call_id: str,
) -> ComponentInvocationTrace | None:
    raw = getattr(component, "last_invocation_trace", None)
    if raw is None:
        return None
    trace = validate_component_invocation_trace(raw)
    if trace.gateway_call_id is not None and trace.gateway_call_id != gateway_call_id:
        raise WorkflowComponentAdapterError(
            "Component trace is bound to a different gateway call.",
            invocation_trace=trace,
        )
    return trace.model_copy(
        update={"gateway_call_id": gateway_call_id},
        deep=True,
    )


def _safe_tool_call_summaries(
    plan: AudioEditPlan,
) -> list[WorkflowToolCallSummary]:
    raw_calls = plan.metadata.get("tool_invocations")
    if not isinstance(raw_calls, list):
        return []
    summaries: list[WorkflowToolCallSummary] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        tool_id = str(raw.get("tool_id") or "").strip()
        if not tool_id:
            continue
        attempts = raw.get("attempts")
        attempt_count = len(attempts) if isinstance(attempts, list) else 1
        summaries.append(
            WorkflowToolCallSummary(
                tool_id=tool_id,
                step_id=(str(raw["step_id"]) if raw.get("step_id") is not None else None),
                success=bool(raw.get("success", False)),
                attempt_count=max(1, attempt_count),
            )
        )
    return summaries


def _evidence_paths_by_id(
    path_sources: Any,
) -> dict[str, set[str]]:
    if not isinstance(path_sources, dict):
        raise ValueError("workflow evidence path sources must be an object")
    result: dict[str, set[str]] = {}
    for raw_path, raw_ids in path_sources.items():
        path = str(raw_path)
        if not isinstance(raw_ids, list):
            raise ValueError("workflow evidence path source ids must be arrays")
        for raw_id in raw_ids:
            evidence_id = str(raw_id)
            result.setdefault(evidence_id, set()).add(path)
    return result


def _evidence_values_by_id(
    context: dict[str, Any],
    path_sources: Any,
) -> dict[str, dict[str, Any]]:
    by_id = _evidence_paths_by_id(path_sources)
    leaf_values: dict[str, Any] = {}
    for field in (
        "perception_report",
        "source_text",
        "source_text_audio_index",
        "source_text_audio_path",
        "span_selection_analysis",
    ):
        if field in context:
            leaf_values.update(_reviewed_leaf_values(context[field], prefix=field))
    values: dict[str, dict[str, Any]] = {evidence_id: {} for evidence_id in by_id}
    for evidence_id, paths in by_id.items():
        for path in paths:
            if path in leaf_values:
                value = leaf_values[path]
            else:
                value = _resolve_context_path(context, path)
            values[evidence_id][path] = copy.deepcopy(value)
    return values


def _normalize_unique_reviewed_span_binding(
    plan: AudioEditPlan,
    semantics: Any,
    *,
    context: dict[str, Any],
    evidence_values: dict[str, dict[str, Any]],
    evidence_required_operations: set[str],
) -> tuple[AudioEditPlan, Any, list[dict[str, Any]]]:
    """Copy or bind one unambiguous reviewed span set to local steps."""

    candidate_steps = [
        step for step in plan.steps if step.operation in evidence_required_operations
    ]
    if not candidate_steps:
        return plan, semantics, []
    report = context.get("perception_report")
    selected_spans = report.get("selected_spans") if isinstance(report, dict) else None
    source_collection = "selected_spans"
    identity_field = "span_id"
    normalization_reason = "unique_reviewed_span_exact_copy"
    if not isinstance(selected_spans, list) or not selected_spans:
        speaker_segments = report.get("speaker_segments") if isinstance(report, dict) else None
        if not isinstance(speaker_segments, list) or len(speaker_segments) != 1:
            return plan, semantics, []
        selected_spans = speaker_segments
        source_collection = "speaker_segments"
        identity_field = "segment_id"
        normalization_reason = "unique_reviewed_speaker_segment_exact_copy"

    copied_spans: list[dict[str, float]] = []
    boundary_sources: list[tuple[str, str, str]] = []
    for index, span in enumerate(selected_spans):
        if not isinstance(span, dict):
            return plan, semantics, []
        start_value = span.get("start_sec")
        end_value = span.get("end_sec")
        try:
            start_sec = float(start_value)
            end_sec = float(end_value)
        except (TypeError, ValueError):
            return plan, semantics, []
        if end_sec <= start_sec:
            return plan, semantics, []
        span_id = str(span.get(identity_field) or "")
        candidates: list[tuple[str, str, str]] = []
        for evidence_id, values in evidence_values.items():
            id_paths = [
                path
                for path, value in values.items()
                if path.startswith(f"perception_report.{source_collection}[")
                and path.endswith(f".{identity_field}")
                and span_id
                and value == span_id
            ]
            for id_path in id_paths:
                prefix = id_path.removesuffix(f".{identity_field}")
                start_path = f"{prefix}.start_sec"
                end_path = f"{prefix}.end_sec"
                if (
                    start_path in values
                    and end_path in values
                    and values[start_path] == span.get("start_sec")
                    and values[end_path] == span.get("end_sec")
                ):
                    candidates.append((evidence_id, start_path, end_path))
        if len(candidates) != 1:
            return plan, semantics, []
        evidence_id, start_path, end_path = candidates[0]
        copied_spans.append(
            {
                "start_sec": copy.deepcopy(start_value),
                "end_sec": copy.deepcopy(end_value),
            }
        )
        boundary_sources.extend(
            [
                (f"params.selected_spans.{index}.start_sec", evidence_id, start_path),
                (f"params.selected_spans.{index}.end_sec", evidence_id, end_path),
            ]
        )

    if len({source[1] for source in boundary_sources}) != 1:
        return plan, semantics, []
    plan_data = plan.model_dump(mode="python")
    semantics_data = semantics.model_dump(mode="python")
    bindings_by_step = {
        str(item.get("plan_step_id") or ""): item
        for item in semantics_data.get("bindings") or []
        if isinstance(item, dict)
    }
    evidence_id = boundary_sources[0][1]
    normalizations: list[dict[str, Any]] = []
    for candidate_step in candidate_steps:
        target_step_id = candidate_step.id
        binding = bindings_by_step.get(target_step_id)
        if not isinstance(binding, dict) or evidence_id not in (binding.get("evidence_ids") or []):
            continue
        target_step = next(step for step in plan_data["steps"] if step.get("id") == target_step_id)
        params = target_step.setdefault("params", {})
        span_field = "selected_spans"
        existing_spans = params.get(span_field)
        if not existing_spans:
            span_field = "spans"
            existing_spans = params.get(span_field)
        if existing_spans:
            comparable_spans = (
                [
                    {
                        "start_sec": item.get("start_sec"),
                        "end_sec": item.get("end_sec"),
                    }
                    for item in existing_spans
                    if isinstance(item, dict)
                ]
                if isinstance(existing_spans, list)
                else []
            )
            if comparable_spans != copied_spans:
                continue
            reason = normalization_reason.replace("_exact_copy", "_exact_binding")
        else:
            span_field = "selected_spans"
            params[span_field] = copy.deepcopy(copied_spans)
            reason = normalization_reason
        evidence_ids = binding.setdefault("evidence_ids", [])
        evidence_paths = binding.setdefault("evidence_paths", [])
        argument_sources = binding.setdefault("argument_sources", [])
        argument_sources[:] = [
            source
            for source in argument_sources
            if not str(source.get("argument_path") or "").startswith("inputs.selected_spans")
        ]
        added_source = False
        for base_argument_path, source_evidence_id, evidence_path in boundary_sources:
            argument_path = base_argument_path.replace(
                "params.selected_spans.",
                f"params.{span_field}.",
                1,
            )
            if source_evidence_id not in evidence_ids:
                evidence_ids.append(source_evidence_id)
            path_item = {
                "evidence_id": source_evidence_id,
                "field_path": evidence_path,
            }
            if path_item not in evidence_paths:
                evidence_paths.append(path_item)
            source_item = {
                "argument_path": argument_path,
                "source_kind": "perception_evidence",
                "evidence_id": source_evidence_id,
                "evidence_path": evidence_path,
            }
            if source_item not in argument_sources:
                argument_sources.append(source_item)
                added_source = True
        if not added_source and existing_spans:
            continue
        normalizations.append(
            {
                "step_id": target_step_id,
                "span_count": len(copied_spans),
                "evidence_id": evidence_id,
                "reason": reason,
            }
        )

    return (
        AudioEditPlan.model_validate(plan_data),
        type(semantics).model_validate(semantics_data),
        normalizations,
    )


def _normalize_span_argument_container_aliases(
    plan: AudioEditPlan,
    semantics: Any,
) -> tuple[Any, list[dict[str, str]]]:
    plan_steps = {step.id: step.model_dump(mode="python") for step in plan.steps}
    semantics_data = semantics.model_dump(mode="python")
    records: list[dict[str, str]] = []
    for binding in semantics_data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        step_id = str(binding.get("plan_step_id") or "")
        step = plan_steps.get(step_id)
        if step is None:
            continue
        for source in binding.get("argument_sources") or []:
            if not isinstance(source, dict):
                continue
            argument_path = str(source.get("argument_path") or "")
            if not argument_path.startswith("inputs.selected_spans."):
                continue
            normalized_path = argument_path.replace(
                "inputs.selected_spans.",
                "params.selected_spans.",
                1,
            )
            if _resolve_step_argument_value(step, normalized_path) is _MISSING_ARGUMENT_VALUE:
                continue
            source["argument_path"] = normalized_path
            records.append(
                {
                    "step_id": step_id,
                    "from_argument_path": argument_path,
                    "argument_path": normalized_path,
                    "reason": "canonicalized_selected_spans_container",
                }
            )
    return type(semantics).model_validate(semantics_data), records


def _normalize_plan_span_input_containers(
    plan: AudioEditPlan,
) -> tuple[AudioEditPlan, list[dict[str, str]]]:
    """Move unambiguous local span selectors into the canonical params container."""

    normalized = plan.model_copy(deep=True)
    records: list[dict[str, str]] = []
    for step in normalized.steps:
        for field_name in ("selected_spans", "spans"):
            if field_name not in step.inputs:
                continue
            input_value = step.inputs[field_name]
            param_value = step.params.get(field_name)
            if param_value not in (None, [], {}) and param_value != input_value:
                continue
            step.params[field_name] = copy.deepcopy(input_value)
            step.inputs.pop(field_name, None)
            records.append(
                {
                    "step_id": step.id,
                    "from_argument_path": f"inputs.{field_name}",
                    "argument_path": f"params.{field_name}",
                    "reason": "canonicalized_local_span_container",
                }
            )
    return normalized, records


def _normalize_declared_direct_boundary_values(
    plan: AudioEditPlan,
    semantics: Any,
    *,
    evidence_values: dict[str, dict[str, Any]],
    evidence_required_operations: set[str],
) -> tuple[AudioEditPlan, list[dict[str, Any]]]:
    """Make declared direct boundaries equal their reviewed evidence values."""

    normalized = plan.model_copy(deep=True)
    steps_by_id = {
        step.id: step for step in normalized.steps if step.operation in evidence_required_operations
    }
    records: list[dict[str, Any]] = []
    for binding in semantics.bindings:
        step = steps_by_id.get(binding.plan_step_id)
        if step is None:
            continue
        for source in binding.argument_sources:
            if source.source_kind != "perception_evidence":
                continue
            argument_path = source.argument_path
            boundary = argument_path.rsplit(".", 1)[-1]
            if boundary not in {"start_sec", "end_sec"}:
                continue
            if source.evidence_id is None or source.evidence_path is None:
                continue
            evidence_value = evidence_values.get(source.evidence_id, {}).get(
                source.evidence_path,
                _MISSING_ARGUMENT_VALUE,
            )
            if not isinstance(evidence_value, (int, float)) or isinstance(evidence_value, bool):
                continue
            step_data = step.model_dump(mode="python")
            argument_value = _resolve_step_argument_value(
                step_data,
                argument_path,
            )
            if argument_value is _MISSING_ARGUMENT_VALUE:
                continue
            if type(argument_value) is type(evidence_value) and (argument_value == evidence_value):
                continue
            if not _set_step_argument_value(step, argument_path, evidence_value):
                continue
            records.append(
                {
                    "step_id": step.id,
                    "argument_path": argument_path,
                    "evidence_id": source.evidence_id,
                    "evidence_path": source.evidence_path,
                    "from": copy.deepcopy(argument_value),
                    "to": copy.deepcopy(evidence_value),
                    "reason": "canonicalized_declared_direct_boundary_value",
                }
            )
    return normalized, records


def _normalize_unique_local_boundary_argument_sources(
    plan: AudioEditPlan,
    semantics: Any,
    *,
    evidence_values: dict[str, dict[str, Any]],
    evidence_required_operations: set[str],
) -> tuple[Any, list[dict[str, str]]]:
    """Bind an omitted local boundary only to one exact reviewed leaf."""

    plan_steps = {
        step.id: step.model_dump(mode="python")
        for step in plan.steps
        if step.operation in evidence_required_operations
    }
    semantics_data = semantics.model_dump(mode="python")
    records: list[dict[str, str]] = []
    for binding in semantics_data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        step_id = str(binding.get("plan_step_id") or "")
        step = plan_steps.get(step_id)
        if step is None:
            continue
        declared_evidence_ids = [
            str(evidence_id)
            for evidence_id in binding.get("evidence_ids") or []
            if str(evidence_id) in evidence_values
        ]
        if not declared_evidence_ids:
            continue
        argument_sources = binding.setdefault("argument_sources", [])
        bound_argument_paths = {
            str(source.get("argument_path") or "")
            for source in argument_sources
            if isinstance(source, dict)
            and source.get("source_kind") in {"perception_evidence", "derived_from_evidence"}
        }
        for argument_path, argument_value in _local_boundary_argument_values(step):
            if argument_path in bound_argument_paths:
                continue
            boundary_name = argument_path.rsplit(".", 1)[-1]
            candidates = [
                (evidence_id, evidence_path)
                for evidence_id in declared_evidence_ids
                for evidence_path, evidence_value in evidence_values[evidence_id].items()
                if evidence_path.rsplit(".", 1)[-1] == boundary_name
                and _same_exact_json_number(argument_value, evidence_value)
            ]
            if len(candidates) != 1:
                continue
            evidence_id, evidence_path = candidates[0]
            source_item = {
                "argument_path": argument_path,
                "source_kind": "perception_evidence",
                "evidence_id": evidence_id,
                "evidence_path": evidence_path,
            }
            argument_sources.append(source_item)
            bound_argument_paths.add(argument_path)
            path_item = {
                "evidence_id": evidence_id,
                "field_path": evidence_path,
            }
            evidence_paths = binding.setdefault("evidence_paths", [])
            if path_item not in evidence_paths:
                evidence_paths.append(path_item)
            records.append(
                {
                    "step_id": step_id,
                    "argument_path": argument_path,
                    "evidence_id": evidence_id,
                    "evidence_path": evidence_path,
                    "reason": "unique_exact_reviewed_boundary_source",
                }
            )
    return type(semantics).model_validate(semantics_data), records


def _local_boundary_argument_values(step: dict[str, Any]) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for owner in ("inputs", "params"):
        container = step.get(owner)
        if not isinstance(container, dict):
            continue
        for field in ("selected_spans", "spans"):
            spans = container.get(field)
            if not isinstance(spans, list):
                continue
            for index, span in enumerate(spans):
                if not isinstance(span, dict):
                    continue
                for boundary_name in ("start_sec", "end_sec"):
                    if boundary_name in span:
                        values.append(
                            (
                                f"{owner}.{field}.{index}.{boundary_name}",
                                span[boundary_name],
                            )
                        )
    return values


def _same_exact_json_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and type(left) is type(right)
        and left == right
    )


def _drop_mismatched_nonboundary_direct_evidence_sources(
    plan: AudioEditPlan,
    semantics: Any,
    *,
    evidence_values: dict[str, dict[str, Any]],
    evidence_required_operations: set[str],
) -> tuple[Any, list[dict[str, str]]]:
    """Drop false direct-evidence claims without weakening span boundaries."""

    plan_steps = {
        step.id: step.model_dump(mode="python")
        for step in plan.steps
        if step.operation in evidence_required_operations
    }
    semantics_data = semantics.model_dump(mode="python")
    records: list[dict[str, str]] = []
    for binding in semantics_data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        step_id = str(binding.get("plan_step_id") or "")
        step = plan_steps.get(step_id)
        if step is None:
            continue
        retained: list[Any] = []
        for source in binding.get("argument_sources") or []:
            if not isinstance(source, dict) or source.get("source_kind") != "perception_evidence":
                retained.append(source)
                continue
            argument_path = str(source.get("argument_path") or "")
            if argument_path.rsplit(".", 1)[-1] in {"start_sec", "end_sec"}:
                retained.append(source)
                continue
            evidence_id = str(source.get("evidence_id") or "")
            evidence_path = str(source.get("evidence_path") or "")
            source_values = evidence_values.get(evidence_id, {})
            if evidence_path not in source_values:
                retained.append(source)
                continue
            argument_value = _resolve_step_argument_value(step, argument_path)
            if argument_value is _MISSING_ARGUMENT_VALUE:
                records.append(
                    {
                        "step_id": step_id,
                        "argument_path": argument_path,
                        "evidence_id": evidence_id,
                        "evidence_path": evidence_path,
                        "reason": "dropped_absent_nonboundary_argument_source",
                    }
                )
                continue
            evidence_value = source_values[evidence_path]
            if type(argument_value) is type(evidence_value) and argument_value == evidence_value:
                retained.append(source)
                continue
            records.append(
                {
                    "step_id": step_id,
                    "argument_path": argument_path,
                    "evidence_id": evidence_id,
                    "evidence_path": evidence_path,
                    "reason": "dropped_nonboundary_direct_evidence_mismatch",
                }
            )
        binding["argument_sources"] = retained
    return type(semantics).model_validate(semantics_data), records


def _normalize_exact_argument_source_roles(
    plan: AudioEditPlan,
    semantics: Any,
    *,
    evidence_values: dict[str, dict[str, Any]],
) -> tuple[Any, list[dict[str, str]]]:
    """Repair source-role labels only when the executable value proves them."""

    plan_steps = {step.id: step.model_dump(mode="python") for step in plan.steps}
    step_ids = [step.id for step in plan.steps]
    semantics_data = semantics.model_dump(mode="python")
    records: list[dict[str, str]] = []
    for binding in semantics_data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        step_id = str(binding.get("plan_step_id") or "")
        step = plan_steps.get(step_id)
        if step is None:
            continue
        converted_plan_paths: set[tuple[str, str]] = set()
        for source in binding.get("argument_sources") or []:
            if not isinstance(source, dict) or source.get("source_kind") not in {
                "perception_evidence",
                "derived_from_evidence",
            }:
                continue
            argument_path = str(source.get("argument_path") or "")
            evidence_id = str(source.get("evidence_id") or "")
            evidence_path = str(source.get("evidence_path") or "")
            argument_value = _resolve_step_argument_value(step, argument_path)
            evidence_value = evidence_values.get(evidence_id, {}).get(
                evidence_path,
                _MISSING_ARGUMENT_VALUE,
            )
            if (
                source.get("source_kind") == "derived_from_evidence"
                and argument_path.rsplit(".", 1)[-1] in _AUDIO_REF_KEYS
                and argument_value is not _MISSING_ARGUMENT_VALUE
                and evidence_value is not _MISSING_ARGUMENT_VALUE
                and type(argument_value) is type(evidence_value)
                and argument_value == evidence_value
            ):
                source["source_kind"] = "perception_evidence"
                records.append(
                    {
                        "step_id": step_id,
                        "argument_path": argument_path,
                        "evidence_id": evidence_id,
                        "evidence_path": evidence_path,
                        "reason": "exact_evidence_value_is_not_a_derivation",
                    }
                )
                continue

            context_source = _deterministic_audio_argument_context(
                argument_path,
                argument_value,
                step=step,
                plan_input_audio=plan.input_audio,
            )
            if context_source is None:
                continue
            source["source_kind"] = "deterministic_context"
            source.pop("evidence_id", None)
            source.pop("evidence_path", None)
            if (
                _plan_output_evidence_alias_step_id(
                    evidence_path,
                    step_ids=step_ids,
                )
                is not None
            ):
                converted_plan_paths.add((evidence_id, evidence_path))
            records.append(
                {
                    "step_id": step_id,
                    "argument_path": argument_path,
                    "context_source": context_source,
                    "reason": "audio_argument_is_deterministic_plan_context",
                }
            )

        if not converted_plan_paths:
            continue
        used_paths = {
            (
                str(source.get("evidence_id") or ""),
                str(source.get("evidence_path") or ""),
            )
            for source in binding.get("argument_sources") or []
            if isinstance(source, dict)
            and source.get("source_kind")
            in {
                "perception_evidence",
                "derived_from_evidence",
            }
        }
        binding["evidence_paths"] = [
            item
            for item in binding.get("evidence_paths") or []
            if not (
                isinstance(item, dict)
                and (
                    str(item.get("evidence_id") or ""),
                    str(item.get("field_path") or ""),
                )
                in converted_plan_paths - used_paths
            )
        ]
        binding["evidence_ids"] = list(
            dict.fromkeys(
                str(item.get("evidence_id") or "")
                for item in binding["evidence_paths"]
                if isinstance(item, dict) and item.get("evidence_id")
            )
        )
    return type(semantics).model_validate(semantics_data), records


def _drop_nonleaf_evidence_path_declarations(
    semantics: Any,
    *,
    available_evidence_paths: dict[str, set[str]],
) -> tuple[Any, list[dict[str, str]]]:
    """Drop descriptive collection paths while preserving exact source refs."""

    semantics_data = semantics.model_dump(mode="python")
    records: list[dict[str, str]] = []
    for binding in semantics_data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        step_id = str(binding.get("plan_step_id") or "")
        source_refs = {
            (
                str(source.get("evidence_id") or ""),
                str(source.get("evidence_path") or ""),
            )
            for source in binding.get("argument_sources") or []
            if isinstance(source, dict)
            and source.get("source_kind") in {"perception_evidence", "derived_from_evidence"}
        }
        retained: list[Any] = []
        for reference in binding.get("evidence_paths") or []:
            if not isinstance(reference, dict):
                retained.append(reference)
                continue
            evidence_id = str(reference.get("evidence_id") or "")
            field_path = str(reference.get("field_path") or "")
            available_paths = available_evidence_paths.get(evidence_id, set())
            if (
                field_path in available_paths
                or (evidence_id, field_path) in source_refs
                or not any(
                    path.startswith((f"{field_path}.", f"{field_path}["))
                    for path in available_paths
                )
            ):
                retained.append(reference)
                continue
            records.append(
                {
                    "step_id": step_id,
                    "evidence_id": evidence_id,
                    "evidence_path": field_path,
                    "reason": "dropped_nonleaf_evidence_collection_declaration",
                }
            )
        binding["evidence_paths"] = retained
    return type(semantics).model_validate(semantics_data), records


def _plan_output_evidence_alias_step_id(
    evidence_path: str,
    *,
    step_ids: list[str],
) -> str | None:
    match = re.fullmatch(
        r"plan\.steps(?:\[(\d+)\]|\.(\d+))\."
        r"(?:output|output_audio|output_audio_path|audio|audio_path)",
        evidence_path,
    )
    if match is None:
        return None
    index = int(match.group(1) or match.group(2))
    return step_ids[index] if index < len(step_ids) else None


def _deterministic_audio_argument_context(
    argument_path: str,
    value: Any,
    *,
    step: dict[str, Any],
    plan_input_audio: str | None,
) -> str | None:
    if argument_path.rsplit(".", 1)[-1] not in _AUDIO_REF_KEYS:
        return None
    dependencies = {str(item) for item in step.get("depends_on") or [] if str(item)}
    if isinstance(value, dict):
        from_step = str(value.get("from_step") or "")
        return from_step if from_step in dependencies else None
    if isinstance(value, str):
        if value == plan_input_audio:
            return "workflow_source"
        if value in _SYMBOLIC_AUDIO_REFS:
            return value
        for dependency in dependencies:
            if value == dependency or value.startswith((f"{dependency}:", f"{dependency}.")):
                return dependency
    return None


_MISSING_ARGUMENT_VALUE = object()


def _resolve_step_argument_value(step: dict[str, Any], argument_path: str) -> Any:
    parts = argument_path.split(".")
    if not parts or parts[0] not in {"inputs", "params"}:
        return _MISSING_ARGUMENT_VALUE
    current: Any = step.get(parts[0])
    for part in parts[1:]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return _MISSING_ARGUMENT_VALUE
    return current


def _set_step_argument_value(
    step: AudioEditStep,
    argument_path: str,
    value: Any,
) -> bool:
    parts = argument_path.split(".")
    if not parts or parts[0] not in {"inputs", "params"}:
        return False
    current: Any = step.inputs if parts[0] == "inputs" else step.params
    for part in parts[1:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False
    leaf = parts[-1]
    if isinstance(current, dict) and leaf in current:
        current[leaf] = copy.deepcopy(value)
        return True
    if isinstance(current, list) and leaf.isdigit():
        index = int(leaf)
        if 0 <= index < len(current):
            current[index] = copy.deepcopy(value)
            return True
    return False


def _reviewed_leaf_values(value: Any, *, prefix: str) -> dict[str, Any]:
    if isinstance(value, dict):
        values: dict[str, Any] = {}
        for key, item in value.items():
            values.update(_reviewed_leaf_values(item, prefix=f"{prefix}.{key}"))
        return values or {prefix: copy.deepcopy(value)}
    if isinstance(value, list):
        values: dict[str, Any] = {}
        for index, item in enumerate(value):
            identity = _list_item_identity(item)
            selector = f"[{identity[0]}={identity[1:]}]" if identity is not None else f"[{index}]"
            values.update(_reviewed_leaf_values(item, prefix=f"{prefix}{selector}"))
        return values or {prefix: copy.deepcopy(value)}
    return {prefix: copy.deepcopy(value)}


def _resolve_context_path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in _split_reviewed_context_path(path):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, dict) and "[" in part and part.endswith("]"):
            field_name, selector_suffix = part.split("[", 1)
            collection = current.get(field_name)
            selector = f"[{selector_suffix}"
            if isinstance(collection, list):
                matches = []
                for index, item in enumerate(collection):
                    identity = _list_item_identity(item)
                    item_selector = (
                        f"[{identity[0]}={identity[1:]}]" if identity is not None else f"[{index}]"
                    )
                    if item_selector == selector:
                        matches.append(item)
                if len(matches) == 1:
                    current = matches[0]
                    continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise ValueError(f"reviewed evidence path is absent from context: {path}")
    return current


def _split_reviewed_context_path(path: str) -> list[str]:
    """Split dotted paths without splitting dots inside list selectors."""

    parts: list[str] = []
    start = 0
    bracket_depth = 0
    for index, character in enumerate(path):
        if character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif character == "." and bracket_depth == 0:
            parts.append(path[start:index])
            start = index + 1
    parts.append(path[start:])
    return parts


def _cannot_plan(
    trace: ComponentInvocationTrace | None,
    missing: str,
    reason: str,
) -> WorkflowPlannerOutput:
    return WorkflowPlannerOutput(
        decision="cannot_plan",
        missing_information=[missing],
        cannot_plan_reasons=[{"code": "other", "message": reason}],
        reason=reason,
        invocation_trace=trace,
    )


def _perception_steps(plan: AudioEditPlan) -> list[AudioEditStep]:
    return [step for step in plan.steps if step.operation.startswith(_PERCEPTION_PREFIXES)]


def _normalize_frozen_edit_operation_params(
    plan: AudioEditPlan,
    *,
    frozen_params: dict[str, dict[str, Any]],
) -> tuple[AudioEditPlan, list[dict[str, Any]]]:
    """Apply framework-owned deployment choices without model authorship."""

    normalized = plan.model_copy(deep=True)
    normalizations: list[dict[str, Any]] = []
    missing = object()
    for step in normalized.steps:
        for key, expected in frozen_params.get(step.operation, {}).items():
            previous = step.params.get(key, missing)
            if previous == expected:
                continue
            step.params[key] = copy.deepcopy(expected)
            record: dict[str, Any] = {
                "step_id": step.id,
                "operation": step.operation,
                "param": key,
                "value": copy.deepcopy(expected),
                "action": "injected" if previous is missing else "overridden",
                "reason": "framework_owned_frozen_runtime_parameter",
            }
            if previous is not missing:
                record["planner_value"] = copy.deepcopy(previous)
            normalizations.append(record)
    return normalized, normalizations


def _normalize_w2_edit_evidence_prerequisites(
    plan: AudioEditPlan,
    *,
    context: dict[str, Any],
    allowed_perception_operations: set[str],
    perception_operation_params: dict[str, dict[str, Any]],
) -> tuple[AudioEditPlan, list[dict[str, Any]]]:
    """Make deterministic edit prerequisites explicit in the frozen W2 plan."""

    if _reviewed_context_has_word_timestamps(context):
        return plan, []

    normalized = plan.model_copy(deep=True)
    normalizations: list[dict[str, Any]] = []
    step_ids = {step.id for step in normalized.steps}
    index = 0
    while index < len(normalized.steps):
        edit_step = normalized.steps[index]
        if edit_step.operation != "emphasis_stress_edit":
            index += 1
            continue
        if _step_has_word_timestamp_perception_dependency(
            edit_step,
            plan=normalized,
        ):
            index += 1
            continue
        operation = "perception.speech_context"
        if operation not in allowed_perception_operations:
            index += 1
            continue

        base_id = f"{edit_step.id}__word_timestamps"
        perception_step_id = base_id
        suffix = 2
        while perception_step_id in step_ids:
            perception_step_id = f"{base_id}_{suffix}"
            suffix += 1
        step_ids.add(perception_step_id)
        params = copy.deepcopy(perception_operation_params.get(operation) or {})
        params.update(
            {
                "need_source_text": True,
                "need_words": True,
                "require_exact_word_timestamps": True,
            }
        )
        perception_step = AudioEditStep(
            id=perception_step_id,
            operation=operation,
            description=(
                f"Recover reviewed source text and word timestamps required by {edit_step.id}."
            ),
            inputs=copy.deepcopy(edit_step.inputs),
            params=params,
            expected_output="Reviewed source_text and word_timestamps.",
            depends_on=list(edit_step.depends_on),
            status="pending",
            required=True,
        )
        edit_step.depends_on = list(dict.fromkeys([*edit_step.depends_on, perception_step_id]))
        normalized.steps.insert(index, perception_step)
        normalizations.append(
            {
                "edit_step_id": edit_step.id,
                "inserted_step_id": perception_step_id,
                "operation": operation,
                "required_evidence_fields": ["source_text", "word_timestamps"],
                "reason": "emphasis_stress_edit_requires_word_timestamps",
            }
        )
        index += 2
    return normalized, normalizations


def _reviewed_context_has_word_timestamps(context: dict[str, Any]) -> bool:
    perception = context.get("perception_report")
    if not isinstance(perception, dict):
        return False
    words = perception.get("word_timestamps")
    return isinstance(words, list) and any(
        isinstance(item, dict)
        and item.get("start_sec", item.get("start")) is not None
        and item.get("end_sec", item.get("end")) is not None
        for item in words
    )


def _step_has_word_timestamp_perception_dependency(
    edit_step: AudioEditStep,
    *,
    plan: AudioEditPlan,
) -> bool:
    by_id = {step.id: step for step in plan.steps}
    pending = list(edit_step.depends_on)
    seen: set[str] = set()
    while pending:
        step_id = pending.pop()
        if step_id in seen:
            continue
        seen.add(step_id)
        step = by_id.get(step_id)
        if step is None:
            continue
        if step.operation == "perception.speech_context" and bool(step.params.get("need_words")):
            return True
        pending.extend(step.depends_on)
    return False


def _all_perception_steps_feed_edits(
    plan: AudioEditPlan,
    perception_ids: set[str],
) -> bool:
    edit_steps = [step for step in plan.steps if step.id not in perception_ids]
    if not edit_steps:
        return False
    by_id = {step.id: step for step in plan.steps}
    consumed: set[str] = set()
    for edit_step in edit_steps:
        pending = list(edit_step.depends_on)
        seen: set[str] = set()
        while pending:
            step_id = pending.pop()
            if step_id in seen:
                continue
            seen.add(step_id)
            if step_id in perception_ids:
                consumed.add(step_id)
            dependency = by_id.get(step_id)
            if dependency is not None:
                pending.extend(dependency.depends_on)
    return consumed == perception_ids


def _audio_refs(
    step: AudioEditStep,
    *,
    plan: AudioEditPlan,
    context: dict[str, Any],
) -> list[str]:
    refs: list[str] = []
    _collect_audio_refs(plan.input_audio, refs, key="input_audio")
    _collect_audio_refs(step.inputs, refs)
    for key in sorted(_AUDIO_REF_KEYS):
        _collect_audio_refs(context.get(key), refs, key=key)
    return list(dict.fromkeys(refs))


def _workflow_bound_audio_ref_namespace(
    context: dict[str, Any],
) -> tuple[list[str], list[str], set[str]]:
    """Return the frozen Planner-visible source and reference namespaces."""

    source_refs = _context_audio_ref_list(context.get("workflow_source_audio_refs"))
    explicit_reference_refs = context.get("workflow_reference_audio_refs")
    reference_refs = _context_audio_ref_list(
        explicit_reference_refs
        if explicit_reference_refs is not None
        else context.get("reference_audio_paths")
    )
    return source_refs, reference_refs, set((*source_refs, *reference_refs))


def _context_audio_ref_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return list(dict.fromkeys(text for item in values if (text := str(item).strip())))


def _reasoned_perception_audio_inputs(
    declared_refs: list[str],
    *,
    primary_audio_ref: str,
    source_refs: list[str],
    reference_refs: list[str],
) -> dict[str, Any]:
    """Project one analyzed audio plus optional comparison references.

    Perception handlers are unary unless their contract explicitly consumes
    supporting references. ``input_audio_path`` therefore means "the audio
    observed by this call", independent of the asset's benchmark source or
    reference role.
    """

    declared = set(declared_refs)
    if primary_audio_ref not in declared:
        raise ValueError("primary Perception audio ref must be declared")
    known_refs = {*source_refs, *reference_refs}
    if primary_audio_ref not in known_refs:
        raise ValueError("primary Perception audio ref is outside workflow inputs")
    supporting_refs = [item for item in declared_refs if item != primary_audio_ref]
    inputs: dict[str, Any] = {"input_audio_path": primary_audio_ref}
    if len(supporting_refs) == 1:
        inputs["reference_audio_path"] = supporting_refs[0]
    elif supporting_refs:
        inputs["reference_audio_paths"] = supporting_refs
    return inputs


def _invalid_final_plan_audio_refs(
    plan: AudioEditPlan,
    *,
    source_refs: set[str],
) -> list[str]:
    steps_by_id = {step.id: step for step in plan.steps}
    invalid: list[str] = []
    for step in plan.steps:
        input_refs: list[str] = []
        _collect_audio_refs(step.inputs, input_refs)
        _collect_audio_refs(step.params, input_refs)
        dependency_ids = {step_id for step_id in step.depends_on if step_id in steps_by_id}
        for audio_ref in input_refs:
            if audio_ref in source_refs:
                continue
            if any(
                _matches_dependency_audio_ref(audio_ref, steps_by_id[step_id])
                for step_id in dependency_ids
            ):
                continue
            invalid.append(f"{step.id}:{audio_ref}")
        structured_refs: list[tuple[str, str | None]] = []
        _collect_structured_step_audio_refs(step.inputs, structured_refs)
        _collect_structured_step_audio_refs(step.params, structured_refs)
        for from_step, from_output in structured_refs:
            if from_step not in dependency_ids:
                invalid.append(f"{step.id}:from_step_without_dependency:{from_step}")
                continue
            if from_output not in {
                None,
                "audio",
                "audio_path",
                "output_audio",
                "output_audio_path",
            }:
                invalid.append(f"{step.id}:unsupported_from_output:{from_step}:{from_output}")
    return sorted(set(invalid))


def _matches_dependency_audio_ref(
    audio_ref: str,
    dependency: AudioEditStep,
) -> bool:
    if audio_ref == dependency.id or audio_ref.startswith(
        (f"{dependency.id}:", f"{dependency.id}.")
    ):
        return True
    return False


def _collect_audio_refs(value: Any, refs: list[str], *, key: str = "") -> None:
    if isinstance(value, str):
        if key in _AUDIO_REF_KEYS:
            if value not in _SYMBOLIC_AUDIO_REFS:
                refs.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_audio_refs(item, refs, key=key)
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _collect_audio_refs(child_value, refs, key=child_key)


def _collect_structured_step_audio_refs(
    value: Any,
    refs: list[tuple[str, str | None]],
) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_structured_step_audio_refs(item, refs)
        return
    if not isinstance(value, dict):
        return
    from_step = value.get("from_step")
    if isinstance(from_step, str) and from_step.strip():
        from_output = value.get("from_output")
        refs.append(
            (
                from_step.strip(),
                str(from_output).strip() if from_output is not None else None,
            )
        )
        return
    for child_value in value.values():
        _collect_structured_step_audio_refs(child_value, refs)


def _normalize_evidence_audio_aliases(
    plan: AudioEditPlan,
    *,
    source_refs: list[str],
    evidence_ids: set[str],
) -> tuple[AudioEditPlan, list[dict[str, str]]]:
    if len(source_refs) != 1:
        return plan, []
    normalized = plan.model_copy(deep=True)
    source_ref = source_refs[0]
    plan_step_ids = {step.id for step in normalized.steps}
    records: list[dict[str, str]] = []
    if normalized.input_audio in (None, ""):
        records.append(
            {
                "field": "input_audio",
                "from": "<missing>",
                "to": source_ref,
                "reason": "plan_has_one_workflow_source_audio",
            }
        )
        normalized.input_audio = source_ref
    elif normalized.input_audio in evidence_ids:
        records.append(
            {
                "field": "input_audio",
                "from": str(normalized.input_audio),
                "to": source_ref,
                "reason": "evidence_id_has_one_workflow_source_audio",
            }
        )
        normalized.input_audio = source_ref
    if not evidence_ids:
        return normalized, records
    for step in normalized.steps:
        step.inputs = _normalize_evidence_audio_aliases_in_value(
            step.inputs,
            key="",
            path=f"steps.{step.id}.inputs",
            evidence_ids=evidence_ids,
            source_ref=source_ref,
            plan_step_ids=plan_step_ids,
            records=records,
        )
        step.params = _normalize_evidence_audio_aliases_in_value(
            step.params,
            key="",
            path=f"steps.{step.id}.params",
            evidence_ids=evidence_ids,
            source_ref=source_ref,
            plan_step_ids=plan_step_ids,
            records=records,
        )
    return normalized, records


def _normalize_dependency_audio_output_aliases(
    plan: AudioEditPlan,
) -> tuple[AudioEditPlan, list[dict[str, str]]]:
    """Canonicalize one declared dependency's descriptive audio output alias."""

    normalized = plan.model_copy(deep=True)
    step_ids = {step.id for step in normalized.steps}
    step_order = {step.id: index for index, step in enumerate(normalized.steps)}
    alias_producers: dict[str, set[str]] = {}
    for producer in normalized.steps:
        aliases = {producer.id}
        if isinstance(producer.expected_output, str) and producer.expected_output.strip():
            aliases.add(producer.expected_output.strip())
        for alias in aliases:
            alias_producers.setdefault(alias, set()).add(producer.id)
    records: list[dict[str, str]] = []
    for step in normalized.steps:
        structured_refs: list[tuple[str, str | None]] = []
        _collect_structured_step_audio_refs(step.inputs, structured_refs)
        _collect_structured_step_audio_refs(step.params, structured_refs)
        for from_step, _ in structured_refs:
            if from_step in step_ids and from_step != step.id and from_step not in step.depends_on:
                step.depends_on.append(from_step)
                records.append(
                    {
                        "field": f"steps.{step.id}.depends_on",
                        "from": "<missing>",
                        "to": from_step,
                        "reason": "structured_audio_ref_declares_unique_producer",
                    }
                )
        plain_refs: list[str] = []
        _collect_audio_refs(step.inputs, plain_refs)
        _collect_audio_refs(step.params, plain_refs)
        for audio_ref in plain_refs:
            explicit_producer = _from_step_string_producer(audio_ref)
            if (
                explicit_producer in step_ids
                and explicit_producer != step.id
                and step_order[explicit_producer] < step_order[step.id]
            ):
                if explicit_producer not in step.depends_on:
                    step.depends_on.append(explicit_producer)
                    records.append(
                        {
                            "field": f"steps.{step.id}.depends_on",
                            "from": "<missing>",
                            "to": explicit_producer,
                            "reason": "exact_from_step_string_declares_producer",
                        }
                    )
                continue
            candidates = {
                producer_id
                for producer_id in alias_producers.get(audio_ref, set())
                if step_order[producer_id] < step_order[step.id]
            }
            if len(candidates) != 1:
                continue
            producer_id = candidates.pop()
            if producer_id in step.depends_on:
                continue
            step.depends_on.append(producer_id)
            records.append(
                {
                    "field": f"steps.{step.id}.depends_on",
                    "from": "<missing>",
                    "to": producer_id,
                    "reason": "unique_expected_output_alias_declares_producer",
                }
            )
        dependency_ids = {dependency for dependency in step.depends_on if dependency in step_ids}
        dependency_aliases = {
            alias: next(iter(producers))
            for alias, producers in alias_producers.items()
            if len(producers) == 1 and next(iter(producers)) in dependency_ids
        }
        step.inputs = _normalize_dependency_audio_output_aliases_in_value(
            step.inputs,
            key="",
            path=f"steps.{step.id}.inputs",
            dependency_ids=dependency_ids,
            dependency_aliases=dependency_aliases,
            records=records,
        )
        step.params = _normalize_dependency_audio_output_aliases_in_value(
            step.params,
            key="",
            path=f"steps.{step.id}.params",
            dependency_ids=dependency_ids,
            dependency_aliases=dependency_aliases,
            records=records,
        )
    return normalized, records


def _normalize_dependency_audio_output_aliases_in_value(
    value: Any,
    *,
    key: str,
    path: str,
    dependency_ids: set[str],
    dependency_aliases: dict[str, str],
    records: list[dict[str, str]],
) -> Any:
    if isinstance(value, str):
        explicit_producer = _from_step_string_producer(value) if key in _AUDIO_REF_KEYS else None
        if explicit_producer in dependency_ids:
            records.append(
                {
                    "field": path,
                    "from": value,
                    "to": explicit_producer,
                    "reason": "exact_from_step_string_normalized_to_step_ref",
                }
            )
            return {
                "from_step": explicit_producer,
                "from_output": "output_audio_path",
            }
        producer_id = dependency_aliases.get(value) if key in _AUDIO_REF_KEYS else None
        if producer_id is None:
            return value
        records.append(
            {
                "field": path,
                "from": value,
                "to": producer_id,
                "reason": "unique_expected_output_alias_normalized_to_step_ref",
            }
        )
        return {
            "from_step": producer_id,
            "from_output": "output_audio_path",
        }
    if isinstance(value, list):
        return [
            _normalize_dependency_audio_output_aliases_in_value(
                item,
                key=key,
                path=f"{path}.{index}",
                dependency_ids=dependency_ids,
                dependency_aliases=dependency_aliases,
                records=records,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value
    from_step = value.get("from_step")
    from_output = value.get("from_output")
    if from_step == "current":
        records.append(
            {
                "field": path,
                "from": "from_step:current",
                "to": "current",
                "reason": "current_is_a_symbolic_audio_alias_not_a_step_id",
            }
        )
        return "current"
    accepted_outputs = {
        None,
        "audio",
        "audio_path",
        "output_audio",
        "output_audio_path",
    }
    normalized_from_output = str(from_output).strip() if from_output is not None else None
    if (
        isinstance(from_step, str)
        and from_step in dependency_ids
        and normalized_from_output not in accepted_outputs
    ):
        updated = copy.deepcopy(value)
        updated["from_output"] = "output_audio_path"
        records.append(
            {
                "field": path,
                "from": str(normalized_from_output),
                "to": "output_audio_path",
                "reason": "declared_dependency_has_one_primary_audio_output",
            }
        )
        return updated
    return {
        child_key: _normalize_dependency_audio_output_aliases_in_value(
            child_value,
            key=child_key,
            path=f"{path}.{child_key}",
            dependency_ids=dependency_ids,
            dependency_aliases=dependency_aliases,
            records=records,
        )
        for child_key, child_value in value.items()
    }


def _from_step_string_producer(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"from_step:([A-Za-z0-9_.-]+)", value.strip())
    return match.group(1) if match is not None else None


def _normalize_evidence_audio_aliases_in_value(
    value: Any,
    *,
    key: str,
    path: str,
    evidence_ids: set[str],
    source_ref: str,
    plan_step_ids: set[str],
    records: list[dict[str, str]],
) -> Any:
    if isinstance(value, str):
        if (
            key in _AUDIO_REF_KEYS
            and not _is_reference_audio_ref_key(key)
            and value in evidence_ids
        ):
            records.append(
                {
                    "field": path,
                    "from": value,
                    "to": source_ref,
                    "reason": "evidence_id_has_one_workflow_source_audio",
                }
            )
            return source_ref
        if (
            key in _AUDIO_REF_KEYS
            and not _is_reference_audio_ref_key(key)
            and value in _WORKFLOW_SOURCE_AUDIO_ALIASES
            and value not in plan_step_ids
        ):
            records.append(
                {
                    "field": path,
                    "from": value,
                    "to": source_ref,
                    "reason": "symbolic_workflow_source_has_one_audio",
                }
            )
            return source_ref
        return value
    if isinstance(value, list):
        return [
            _normalize_evidence_audio_aliases_in_value(
                item,
                key=key,
                path=f"{path}.{index}",
                evidence_ids=evidence_ids,
                source_ref=source_ref,
                plan_step_ids=plan_step_ids,
                records=records,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if key in _AUDIO_REF_KEYS:
            from_step = str(value.get("from_step") or "").strip()
            artifact_ref_id = str(value.get("artifact_ref_id") or "").strip()
            alias = from_step or artifact_ref_id
            if alias in _WORKFLOW_SOURCE_AUDIO_ALIASES and alias not in plan_step_ids:
                records.append(
                    {
                        "field": path,
                        "from": alias,
                        "to": source_ref,
                        "reason": "symbolic_workflow_source_has_one_audio",
                    }
                )
                return source_ref
        if "from_step" in value:
            return value
        return {
            child_key: _normalize_evidence_audio_aliases_in_value(
                child_value,
                key=child_key,
                path=f"{path}.{child_key}",
                evidence_ids=evidence_ids,
                source_ref=source_ref,
                plan_step_ids=plan_step_ids,
                records=records,
            )
            for child_key, child_value in value.items()
        }
    return value


def _is_reference_audio_ref_key(key: str) -> bool:
    lowered = key.casefold()
    return "reference" in lowered or lowered.startswith(("ref_audio", "speaker_audio"))


def _step_requirements(step: AudioEditStep) -> list[str]:
    requirements = [
        value
        for value in (step.expected_output, step.description)
        if isinstance(value, str) and value.strip()
    ]
    return requirements or [f"Ground evidence needed by downstream edits after {step.operation}."]


def _evidence_facts(
    result: dict[str, Any],
    *,
    audio_refs: list[str],
) -> dict[str, Any]:
    return _sanitize_perception_facts(result, audio_refs=audio_refs)


def _canonicalize_perception_result_uncertainties(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Make the reviewed report the sole source of typed uncertainties.

    Perception handlers may expose operational or quality state through the
    legacy top-level field while authority reconciliation records observation
    conflicts inside ``perception_report``. Those are complementary claims, so
    normalize both representations and merge them into the reviewed report.
    Only contradictory fields on the same typed uncertainty are protocol
    errors; independent uncertainties remain independently reviewable.
    """

    canonical = copy.deepcopy(result)
    report = canonical.get("perception_report")
    if not isinstance(report, dict):
        return canonical

    nested_present = "uncertainties" in report
    top_level_present = "uncertainties" in canonical
    if not nested_present and not top_level_present:
        return canonical

    nested_items = _canonical_uncertainty_items(report.get("uncertainties"))
    top_level_items = (
        _canonical_uncertainty_items(canonical.get("uncertainties"))
        if top_level_present
        else []
    )
    merged_items = _merge_canonical_uncertainty_items(nested_items, top_level_items)
    if merged_items:
        report["uncertainties"] = merged_items
    else:
        report.pop("uncertainties", None)
    canonical.pop("uncertainties", None)
    return canonical


def _canonical_uncertainty_items(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", [], {}):
        return []
    raw_items = value if isinstance(value, list) else [value]
    reviewed = _sanitize_perception_report({"uncertainties": raw_items})
    items = reviewed.get("uncertainties")
    return [copy.deepcopy(item) for item in items] if isinstance(items, list) else []


def _merge_canonical_uncertainty_items(
    nested_items: list[dict[str, Any]],
    compatibility_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    item_indexes: dict[tuple[str, ...], int] = {}
    for item in [*nested_items, *compatibility_items]:
        identity = _uncertainty_identity(item)
        existing_index = item_indexes.get(identity)
        if existing_index is None:
            item_indexes[identity] = len(merged)
            merged.append(copy.deepcopy(item))
            continue
        merged[existing_index] = _merge_same_uncertainty(
            merged[existing_index],
            item,
            identity=identity,
        )
    return merged


def _uncertainty_identity(item: dict[str, Any]) -> tuple[str, ...]:
    """Return the stable identity of one typed uncertainty claim.

    ``message`` distinguishes separate claims that happen to share a broad
    code. This keeps independently reported operational and observation details
    instead of forcing one to overwrite the other.
    """

    fields = ("code", "scope_ref", "message", "audio_ref_id", "audio_index")
    identity = tuple(_stable_uncertainty_value(item.get(field)) for field in fields)
    if any(identity):
        return ("typed", *identity)
    return ("content", _stable_uncertainty_value(item))


def _stable_uncertainty_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _merge_same_uncertainty(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    identity: tuple[str, ...],
) -> dict[str, Any]:
    merged = copy.deepcopy(existing)
    for field, incoming_value in incoming.items():
        if field not in merged:
            merged[field] = copy.deepcopy(incoming_value)
            continue
        existing_value = merged[field]
        if existing_value == incoming_value:
            continue
        if field == "missing_evidence":
            merged[field] = _merge_uncertainty_missing_evidence(
                existing_value,
                incoming_value,
            )
            continue
        raise ValueError(
            "reviewed perception result conflicts with "
            "perception_report.uncertainties for the same uncertainty "
            f"identity {identity!r}: field {field!r} has mutually exclusive values"
        )
    return merged


def _merge_uncertainty_missing_evidence(existing: Any, incoming: Any) -> list[Any]:
    existing_items = existing if isinstance(existing, list) else [existing]
    incoming_items = incoming if isinstance(incoming, list) else [incoming]
    merged: list[Any] = []
    fingerprints: set[str] = set()
    for item in [*existing_items, *incoming_items]:
        fingerprint = _stable_uncertainty_value(item)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        merged.append(copy.deepcopy(item))
    return merged


def _covered_audio_refs_from_reviewed_facts(
    facts: dict[str, Any],
    *,
    requested_audio_refs: list[str],
) -> list[str]:
    """Derive coverage from admitted facts, never from the request envelope."""

    requested = set(requested_audio_refs)
    covered: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("audio_ref_id", "source_text_audio_path"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate in requested:
                    covered.add(candidate)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(facts)
    return [audio_ref for audio_ref in requested_audio_refs if audio_ref in covered]


def _reviewed_perception_contract_failure(
    request: WorkflowPerceptionRequest,
    *,
    result: dict[str, Any],
    facts: dict[str, Any],
) -> str | None:
    report = facts.get("perception_report")
    reviewed_report = report if isinstance(report, dict) else {}
    critical_fields = (
        "selected_spans",
        "candidate_segments",
        "candidate_rubric_judgements",
        "selected_speaker_refs",
        "selected_speaker_ids",
        "target_bindings",
    )
    raw_report = result.get("perception_report")
    raw_report = raw_report if isinstance(raw_report, dict) else {}
    nested_semantic = raw_report.get("semantic_span_review")
    nested_semantic = nested_semantic if isinstance(nested_semantic, dict) else {}
    for field in critical_fields:
        raw_value = result.get(field)
        if raw_value in (None, [], {}):
            raw_value = raw_report.get(field)
        if raw_value in (None, [], {}):
            raw_value = nested_semantic.get(field)
        if raw_value not in (None, [], {}) and reviewed_report.get(field) in (
            None,
            [],
            {},
        ):
            return (
                "Perception produced a required reviewed field that was not admitted: "
                f"perception_report.{field}"
            )

    if request.operation == "perception.semantic_span_review":
        has_selected_binding = bool(
            reviewed_report.get("selected_spans")
            and reviewed_report.get("target_bindings")
        )
        has_measured_candidates = bool(reviewed_report.get("candidate_segments"))
        if not has_selected_binding and not has_measured_candidates:
            return (
                "Perception semantic-span evidence is incomplete after reviewed "
                "projection: expected either measured candidate_segments or a "
                "resolved selected_spans/target_bindings pair"
            )
    return None


def _uncertainties(result: dict[str, Any]) -> list[str]:
    values: list[str] = []

    def append(raw: Any) -> None:
        if isinstance(raw, dict):
            message = str(raw.get("message") or raw.get("code") or "").strip()
            if message:
                values.append(message)
            return
        if str(raw):
            values.append(str(raw))

    for key in ("warnings", "missing", "uncertainties"):
        raw = result.get(key)
        if isinstance(raw, list):
            for item in raw:
                append(item)
        elif raw:
            append(raw)
    report = result.get("perception_report")
    nested_uncertainties = report.get("uncertainties") if isinstance(report, dict) else None
    if isinstance(nested_uncertainties, list):
        for item in nested_uncertainties:
            append(item)
    elif nested_uncertainties:
        append(nested_uncertainties)
    if result.get("needs_review") is True:
        details = _perception_review_details(result)
        values.append("Perception evidence remains reviewable and was admitted with uncertainty.")
        values.extend(details)
    if result.get("satisfied") is False:
        values.append(
            "Perception did not fully satisfy every declared requirement; "
            "available facts were retained."
        )
    return list(dict.fromkeys(values))


def _perception_result_failure(
    result: dict[str, Any],
    *,
    has_reviewable_facts: bool = False,
    policy: WorkflowProtocolPolicy | str | None = None,
) -> str | None:
    resolved_policy = resolve_workflow_protocol_policy(policy)
    required_missing = result.get("required_missing") or result.get("blocking_missing")
    if required_missing:
        return f"Perception operation reports missing required evidence: {required_missing}"
    if result.get("unsafe_for_execution") is True:
        return "Perception evidence is explicitly unsafe for execution."
    if (
        resolved_policy.reviewable_uncertainty_mode == "block"
        and result.get("needs_review") is True
    ):
        return "Perception evidence still requires review under the selected protocol."
    if resolved_policy.reviewable_uncertainty_mode == "block" and result.get("satisfied") is False:
        return "Perception did not satisfy every declared requirement under the selected protocol."
    if result.get("satisfied") is False and not has_reviewable_facts:
        return "Perception operation produced no usable facts for its requirements."
    for key in ("speaker_attributed_review", "timestamped_caption_review"):
        review = result.get(key)
        if not isinstance(review, dict):
            continue
        status = str(review.get("status") or "").strip().lower()
        if status and status not in {"success", "completed", "passed"}:
            verification = review.get("verification")
            errors = verification.get("errors") if isinstance(verification, dict) else None
            if errors or not has_reviewable_facts:
                return f"Perception review did not pass: {key}.{status}"
    return None


def _perception_review_details(result: dict[str, Any]) -> list[str]:
    details: list[str] = []
    review = result.get("timestamped_caption_review")
    verification = review.get("verification") if isinstance(review, dict) else None
    if isinstance(verification, dict):
        for key in ("errors", "warnings"):
            values = verification.get(key)
            if isinstance(values, list):
                details.extend(str(item) for item in values if str(item))
        normalizations = verification.get("normalizations")
        if isinstance(normalizations, list):
            details.extend(
                str(item.get("reason"))
                for item in normalizations
                if isinstance(item, dict) and item.get("reason")
            )
    report = result.get("perception_report")
    timestamped_caption = report.get("timestamped_caption") if isinstance(report, dict) else None
    if isinstance(timestamped_caption, dict):
        warnings = timestamped_caption.get("warnings")
        if isinstance(warnings, list):
            details.extend(str(item) for item in warnings if str(item))
    return [item[:300] for item in dict.fromkeys(details)][:8]


def _one_consistent_value(
    candidates: list[tuple[str, Any]],
    *,
    field: str,
) -> Any:
    first = candidates[0][1]
    if any(candidate != first for _, candidate in candidates[1:]):
        sources = [source for source, _ in candidates]
        raise ValueError(f"conflicting {field} values from evidence {sources}")
    return copy.deepcopy(first)


def _authority_evidence_ids_by_kind(
    assignments: Sequence[EvidenceAuthorityAssignment],
    *,
    audio_ref_id: str | None,
) -> dict[ObservationKind, list[str]]:
    result: dict[ObservationKind, list[str]] = {}
    for assignment in assignments:
        if assignment.audio_ref_id != audio_ref_id:
            continue
        evidence_ids = list(assignment.authoritative_evidence_ids)
        if not evidence_ids:
            evidence_ids = [evidence_id for evidence_id, _level in assignment.candidate_levels]
        result[assignment.observation_kind] = list(dict.fromkeys(evidence_ids))
    return result


def _dedupe_evidence_conflicts(
    conflicts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for conflict in conflicts:
        conflict_id = str(conflict.get("conflict_id") or "")
        if not conflict_id:
            continue
        deduped.setdefault(conflict_id, copy.deepcopy(dict(conflict)))
    return list(deduped.values())


def _merge_reviewed_values(left: Any, right: Any, *, path: str) -> Any:
    # A missing observation is not a contradictory observation. Later evidence
    # may enrich an unknown field, while two distinct known values remain a
    # hard conflict below.
    if left is None:
        return copy.deepcopy(right)
    if right is None:
        return copy.deepcopy(left)
    if isinstance(left, dict) and isinstance(right, dict):
        merged = copy.deepcopy(left)
        for key, value in right.items():
            child_path = f"{path}.{key}"
            if key in merged:
                merged[key] = _merge_reviewed_values(
                    merged[key],
                    value,
                    path=child_path,
                )
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(left, list) and isinstance(right, list):
        merged = copy.deepcopy(left)
        for item in right:
            identity = _list_item_identity(item)
            matching_index = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if identity is not None and _list_item_identity(existing) == identity
                ),
                None,
            )
            if matching_index is not None:
                merged[matching_index] = _merge_reviewed_values(
                    merged[matching_index],
                    item,
                    path=f"{path}[{identity}]",
                )
            elif item not in merged:
                merged.append(copy.deepcopy(item))
        return merged
    if left != right:
        raise ValueError(f"conflicting reviewed evidence at {path}")
    return copy.deepcopy(left)


def _list_item_identity(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") == _MULTI_AUDIO_REPORT_ENTRY_SCHEMA_VERSION and value.get(
        "audio_ref_id"
    ) not in (None, ""):
        return ("audio_ref_id", value["audio_ref_id"])
    for key in (
        "span_id",
        "candidate_id",
        "target_id",
        "point_id",
        "segment_id",
        "word_id",
        "event_id",
        "evidence_id",
        "speaker_id",
        "id",
        "index",
    ):
        if key in value:
            return (key, value[key])
    if "start" in value and "end" in value:
        return ("time", value["start"], value["end"])
    if "start_sec" in value and "end_sec" in value:
        return ("time_sec", value["start_sec"], value["end_sec"])
    if "start_time" in value and "end_time" in value:
        return ("time_value", value["start_time"], value["end_time"])
    return None


def _stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reviewed_leaf_paths(value: Any, *, prefix: str) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_reviewed_leaf_paths(item, prefix=f"{prefix}.{key}"))
        return paths or [prefix]
    if isinstance(value, list):
        paths: list[str] = []
        for index, item in enumerate(value):
            identity = _list_item_identity(item)
            selector = f"[{identity[0]}={identity[1:]}]" if identity is not None else f"[{index}]"
            paths.extend(_reviewed_leaf_paths(item, prefix=f"{prefix}{selector}"))
        return paths or [prefix]
    return [prefix]


def _reviewed_projection_paths(
    value: Any,
    *,
    prefix: str,
    include_container: bool = True,
) -> list[str]:
    """Index sanitized containers as well as leaves for exact evidence binding.

    Planner parameters may consume a reviewed collection such as the complete
    ``word_timestamps`` or ``selected_spans`` array. Those collection fields are
    part of the typed reviewed projection, so they are legitimate exact paths;
    indexing only their leaves incorrectly rejects an otherwise traceable binding.
    """

    paths = [prefix] if include_container or not isinstance(value, (dict, list)) else []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(
                _reviewed_projection_paths(
                    item,
                    prefix=f"{prefix}.{key}",
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            identity = _list_item_identity(item)
            selector = f"[{identity[0]}={identity[1:]}]" if identity is not None else f"[{index}]"
            paths.extend(
                _reviewed_projection_paths(
                    item,
                    prefix=f"{prefix}{selector}",
                )
            )
    return paths


__all__ = [
    "AudioEditExecutorPerceptionAdapter",
    "PerceptionTraceBuilder",
    "ReviewedWorkflowEvidenceProjector",
    "TraceableAudioEditPlannerAdapter",
    "WorkflowComponentAdapterError",
]
