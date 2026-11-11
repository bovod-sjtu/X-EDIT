"""Provider boundary for the separated semantic-memo Planner.

The Planner in this module never invokes a Reasoner and never consumes a
legacy Planner result. It receives one explicit semantic memo, reviewed
evidence, Session Graph state, and operation contracts, then returns a single
typed operational decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.capabilities.schemas import (
    OperationCapabilityCard,
    OperationParameterSpec,
    validate_operation_arguments,
)
from x_editor.core.invocation_trace import (
    ComponentInvocationStatus,
    ComponentInvocationTrace,
    ProviderInvocationRecord,
)
from x_editor.core.schemas import AudioEditPlan, PlannerCannotPlanReason, StrictModel
from x_editor.core.text_equivalence import equivalent_text_copy
from x_editor.core.step_references import (
    is_step_output_reference as _is_step_output_reference,
)
from x_editor.core.step_references import (
    iter_step_output_references as _iter_step_output_references,
)
from x_editor.perception.gateway_routing import route_perception_observation_need
from x_editor.perception.workflow_catalog import (
    workflow_perception_capability,
)
from x_editor.planner._invocation_trace import provider_records_from_client
from x_editor.planner.dependency_provenance import project_dependency_argument_provenance
from x_editor.planner.evidence_index import EvidenceIndexFormat
from x_editor.planner.argument_table import ArgumentContractFormat
from x_editor.planner.retry_context import RetryGuidanceMode
from x_editor.planner.observation_contracts import PerceptionJobShapeError
from x_editor.planner.prompt_package import (
    memo_qualified_planner_current_call_prompt,
    memo_qualified_planner_system_prompt,
)
from x_editor.planner.reasoned_workflow import (
    PLAN_STEP_TARGET_SUMMARY_MAX_LENGTH,
    PLANNER_DECISION_SUMMARY_MAX_LENGTH,
    MemoQualifiedPlannerInput,
    MemoQualifiedPlannerTurn,
    PlannerCandidateOperationContract,
    PlannerLegalArgumentContract,
    PlannerPerceptionJob,
    PlannerToolCandidateAssessment,
)
from x_editor.reasoner.evidence_obligations import pending_observations, validate_observation_dispatch
from x_editor.reasoner.contracts import (
    required_semantic_target_binding_failures,
    session_completion_proof_inventory,
)
from x_editor.runtime.audio_usage import (
    AudioUsageKind,
    audio_evidence_requirement_for_usage,
    canonical_speaker_evidence_references,
)


class _ProviderDirectArgumentSource(StrictModel):
    argument_path: str = Field(min_length=1, max_length=512)
    source_kind: Literal["user_request", "deterministic_context"]


class _ProviderEvidenceArgumentSource(StrictModel):
    argument_path: str = Field(min_length=1, max_length=512)
    source_kind: Literal["perception_evidence", "derived_from_evidence"]
    evidence_id: str = Field(min_length=1, max_length=256)
    evidence_path: str = Field(min_length=1, max_length=512)


_ProviderArgumentSource = Annotated[
    _ProviderDirectArgumentSource | _ProviderEvidenceArgumentSource,
    Field(discriminator="source_kind"),
]


class _ProviderPlanStepSemanticBinding(StrictModel):
    plan_step_id: str = Field(min_length=1, max_length=256)
    target: str = Field(
        min_length=1,
        max_length=PLAN_STEP_TARGET_SUMMARY_MAX_LENGTH,
    )
    operation: str = Field(min_length=1, max_length=256)
    argument_sources: list[_ProviderArgumentSource] = Field(  # type: ignore[valid-type]
        default_factory=list,
        max_length=128,
    )
    unresolved_requirements: list[str] = Field(default_factory=list, max_length=64)


class _ProviderAudioUsageBinding(StrictModel):
    """Semantic audio use authored by the Planner, without framework provenance."""

    plan_step_id: str = Field(min_length=1, max_length=256)
    argument_path: str = Field(min_length=1, max_length=512)
    audio_ref: str = Field(min_length=1, max_length=4096)
    usage: AudioUsageKind


class _ProviderExecutionSemantics(StrictModel):
    schema_version: Literal["x-editor.planner-execution-semantics/v1"] = (
        "x-editor.planner-execution-semantics/v1"
    )
    bindings: list[_ProviderPlanStepSemanticBinding] = Field(
        default_factory=list,
        max_length=256,
    )
    audio_usage_bindings: list[_ProviderAudioUsageBinding] = Field(
        default_factory=list,
        max_length=256,
    )


class _ProviderPlannerStepToolSelection(StrictModel):
    plan_step_id: str = Field(min_length=1, max_length=256)
    selected_operation: str = Field(min_length=1, max_length=256)
    candidates: list[PlannerToolCandidateAssessment] = Field(
        min_length=1,
        max_length=32,
    )
    selection_summary: str = Field(default="", max_length=1024)


class _ProviderPlannerToolSelection(StrictModel):
    schema_version: Literal["x-editor.planner-tool-selection/v1"] = (
        "x-editor.planner-tool-selection/v1"
    )
    steps: list[_ProviderPlannerStepToolSelection] = Field(
        min_length=1,
        max_length=256,
    )


class _ProviderAudioEditStep(StrictModel):
    """Only fields the Planner is authorized to author for one edit step."""

    id: str = Field(min_length=1, max_length=256)
    operation: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=1024)
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = Field(default=None, max_length=512)
    depends_on: list[str] = Field(default_factory=list, max_length=256)
    required: bool = True


class _ProviderExecutableAudioEditPlan(AudioEditPlan):
    input_audio: str = Field(min_length=1)
    steps: list[_ProviderAudioEditStep] = Field(min_length=1)
    missing_inputs: list[str] = Field(default_factory=list, max_length=0)


class _MemoQualifiedPerceptionRequestTurn(StrictModel):
    schema_version: Literal["x-editor.memo-qualified-planner-turn/v3"] = (
        "x-editor.memo-qualified-planner-turn/v3"
    )
    decision: Literal["request_perception"]
    selected_base_state_node_id: str | None = Field(max_length=256)
    selected_base_audio_ref: str | None = Field(max_length=256)
    perception_jobs: list[PlannerPerceptionJob] = Field(min_length=1, max_length=16)
    candidate_operations: list[str] = Field(default_factory=list, max_length=8)
    missing_information: list[str] = Field(default_factory=list, max_length=64)
    decision_summary: str = Field(
        default="",
        max_length=PLANNER_DECISION_SUMMARY_MAX_LENGTH,
    )


class _MemoQualifiedEvidenceCannotPlanTurn(StrictModel):
    schema_version: Literal["x-editor.memo-qualified-planner-turn/v3"] = (
        "x-editor.memo-qualified-planner-turn/v3"
    )
    decision: Literal["cannot_plan"]
    missing_information: list[str] = Field(min_length=1, max_length=64)
    semantic_conflict: str | None = Field(default=None, max_length=2048)
    cannot_plan_reasons: list[PlannerCannotPlanReason] = Field(
        default_factory=list,
        max_length=32,
    )
    decision_summary: str = Field(
        default="",
        max_length=PLANNER_DECISION_SUMMARY_MAX_LENGTH,
    )


class _MemoQualifiedExecutionTurn(StrictModel):
    schema_version: Literal["x-editor.memo-qualified-planner-turn/v3"] = (
        "x-editor.memo-qualified-planner-turn/v3"
    )
    decision: Literal["execute"]
    selected_base_state_node_id: str | None = Field(max_length=256)
    selected_base_audio_ref: str | None = Field(max_length=256)
    state_transition_kind: Literal[
        "edit_current",
        "branch_from_history",
        "restore_history",
    ]
    plan: _ProviderExecutableAudioEditPlan
    execution_semantics: _ProviderExecutionSemantics | None = None
    tool_selection: _ProviderPlannerToolSelection | None = None
    missing_information: list[str] = Field(default_factory=list, max_length=0)
    decision_summary: str = Field(
        default="",
        max_length=PLANNER_DECISION_SUMMARY_MAX_LENGTH,
    )


class _MemoQualifiedStateCompletionTurn(StrictModel):
    schema_version: Literal["x-editor.memo-qualified-planner-turn/v3"] = (
        "x-editor.memo-qualified-planner-turn/v3"
    )
    decision: Literal["complete_from_session_state"]
    selected_base_state_node_id: str = Field(min_length=1, max_length=256)
    selected_base_audio_ref: str = Field(min_length=1, max_length=256)
    state_completion_kind: Literal["select_existing", "restore_history"]
    changes_satisfied_by_base: list[str] = Field(min_length=1, max_length=32)
    decision_summary: str = Field(
        default="",
        max_length=PLANNER_DECISION_SUMMARY_MAX_LENGTH,
    )


_EvidencePlanningResponse = Annotated[
    _MemoQualifiedPerceptionRequestTurn
    | _MemoQualifiedExecutionTurn
    | _MemoQualifiedStateCompletionTurn
    | _MemoQualifiedEvidenceCannotPlanTurn,
    Field(discriminator="decision"),
]
_ExecutionPlanningResponse = Annotated[
    _MemoQualifiedPerceptionRequestTurn
    | _MemoQualifiedExecutionTurn
    | _MemoQualifiedStateCompletionTurn
    | _MemoQualifiedEvidenceCannotPlanTurn,
    Field(discriminator="decision"),
]


class MemoQualifiedPlannerError(RuntimeError):
    """Base failure at the separated Planner boundary."""


class MemoQualifiedPlannerContractError(MemoQualifiedPlannerError):
    """The provider output violated the supplied operational contract."""

    failure_kind = "protocol"


class OpenAICompatibleMemoQualifiedPlanner:
    """Plan from a semantic memo without re-running semantic interpretation."""

    def __init__(
        self,
        *,
        model_name: str,
        client: OpenAICompatibleChatClient | Any,
        provider_name: str | None = None,
        evidence_index_format: EvidenceIndexFormat = "expanded",
        argument_contract_format: ArgumentContractFormat = "expanded",
        retry_guidance_mode: RetryGuidanceMode = "diagnostic",
    ) -> None:
        if evidence_index_format not in {"expanded", "prefix-plus-suffix/v1"}:
            raise ValueError("Unsupported evidence index format")
        if argument_contract_format not in {"expanded", "argument-table/v1"}:
            raise ValueError("Unsupported argument contract format")
        if retry_guidance_mode not in {"diagnostic", "actionable/v1"}:
            raise ValueError("Unsupported retry guidance mode")
        self.evidence_index_format = evidence_index_format
        self.argument_contract_format = argument_contract_format
        self.retry_guidance_mode = retry_guidance_mode
        self.model_name = model_name
        self.client = client
        self.provider_name = provider_name or str(
            getattr(client, "provider_name", "openai_compatible")
        )
        self.last_invocation_trace: ComponentInvocationTrace | None = None
        self.last_prompt_package_fingerprint: str | None = None
        self.last_prompt_budget: dict[str, int | str] = {}
        self._last_provider_output: dict[str, Any] | None = None
        self._provider_invocations: list[ProviderInvocationRecord] = []
        self._parameter_selection_projection_count = 0
        self._evidence_declaration_projection_count = 0
        self._dependency_provenance_audit: dict[str, Any] = {}
        self._state_transition_projection_count = 0
        self._framework_provenance_projection_count = 0
        self._semantic_binding_label_projection_count = 0
        self._wire_canonicalization_count = 0
        self._argument_source_deduplication_count = 0
        self._tool_selection_quarantine_count = 0
        self._execution_base_projection_count = 0

    def plan(self, planner_input: MemoQualifiedPlannerInput) -> MemoQualifiedPlannerTurn:
        """Return one validated evidence or execution decision."""

        checked_input = MemoQualifiedPlannerInput.model_validate(
            planner_input.model_dump(mode="python")
        )
        self.last_invocation_trace = None
        self._provider_invocations = []
        self._last_provider_output = None
        self._contract_feedback = None
        self._parameter_selection_projection_count = 0
        self._evidence_declaration_projection_count = 0
        self._dependency_provenance_audit = {}
        self._state_transition_projection_count = 0
        self._framework_provenance_projection_count = 0
        self._semantic_binding_label_projection_count = 0
        self._wire_canonicalization_count = 0
        self._argument_source_deduplication_count = 0
        self._tool_selection_quarantine_count = 0
        self._execution_base_projection_count = 0
        system_prompt = memo_qualified_planner_system_prompt()
        provider_response_format = memo_qualified_planner_provider_response_format(
            phase=checked_input.phase,
            require_observation=bool(pending_observations(checked_input.semantic_memo) and checked_input.perception_capability_catalog),
        )
        response_schema = provider_response_format["json_schema"]["schema"]
        current_prompt = memo_qualified_planner_current_call_prompt(
            checked_input,
            output_schema=response_schema,
            evidence_index_format=self.evidence_index_format,
            argument_contract_format=self.argument_contract_format,
            retry_guidance_mode=self.retry_guidance_mode,
        )
        self.last_prompt_package_fingerprint = _prompt_fingerprint(
            system_prompt,
            current_prompt,
            response_schema=response_schema,
        )
        response_schema_payload = json.dumps(
            response_schema,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.last_prompt_budget = {
            "phase": checked_input.phase,
            "evidence_index_format": self.evidence_index_format,
            "argument_contract_format": self.argument_contract_format,
            "retry_guidance_mode": self.retry_guidance_mode,
            "system_chars": len(system_prompt),
            "current_call_chars": len(current_prompt),
            "total_chars": len(system_prompt) + len(current_prompt),
            "response_schema_chars": len(response_schema_payload),
            "response_schema_sha256": hashlib.sha256(
                response_schema_payload.encode("utf-8")
            ).hexdigest(),
            "max_completion_tokens": int(getattr(self.client, "max_tokens", 0) or 0),
        }
        _clear_client_json_invocations(self.client)
        try:
            payload = self.client.call_json(
                system_prompt,
                current_prompt,
                response_format=provider_response_format,
                repair_schema_name="MemoQualifiedPlannerTurn",
            )
        except Exception as exc:
            self._append_provider_invocations(succeeded=False, error=exc)
            self._set_invocation_trace(status="failed")
            raise MemoQualifiedPlannerError(str(exc)) from exc
        self._append_provider_invocations(succeeded=True)
        return self.validate_provider_output(checked_input, payload)

    def validate_provider_output(
        self, checked_input: MemoQualifiedPlannerInput, payload: dict[str, Any]
    ) -> MemoQualifiedPlannerTurn:
        """Apply shared PlanIR validation to an actual provider-authored decision.

        A joint reasoning/planning provider can use this boundary after its one
        model call. It receives no extra generation, repair or admission privilege.
        """
        self._last_provider_output = copy.deepcopy(payload)
        self._request_binding_audit = None
        try:
            payload, self._dependency_provenance_audit = project_dependency_argument_provenance(
                payload, validate_topology=_validate_plan_topology
            )
        except (ValidationError, ValueError) as exc:
            self._set_invocation_trace(status="failed")
            raise MemoQualifiedPlannerContractError(
                "Planner output violates the original plan dependency contract: "
                f"{exc}; provider_output={_bounded_json(payload)}"
            ) from exc
        (
            payload,
            wire_canonicalization_count,
            argument_source_deduplication_count,
            tool_selection_quarantine_count,
        ) = _canonicalize_provider_wire_payload(payload)
        self._wire_canonicalization_count = wire_canonicalization_count
        self._argument_source_deduplication_count = argument_source_deduplication_count
        self._tool_selection_quarantine_count = tool_selection_quarantine_count
        payload, framework_provenance_projection_count = _project_framework_owned_turn_provenance(
            payload,
            planner_input=checked_input,
        )
        self._framework_provenance_projection_count = framework_provenance_projection_count
        payload, execution_base_projection_count = _project_execution_base_audio_from_plan(
            payload,
            planner_input=checked_input,
        )
        self._execution_base_projection_count = execution_base_projection_count
        payload, state_transition_projection_count = _project_state_transition_from_selected_base(
            payload,
            planner_input=checked_input,
        )
        self._state_transition_projection_count = state_transition_projection_count
        (
            payload,
            audio_usage_argument_path_projection_count,
        ) = _project_audio_usage_binding_argument_paths(payload)
        self._audio_usage_argument_path_projection_count = (
            audio_usage_argument_path_projection_count
        )
        payload, semantic_binding_label_projection_count = (
            _project_semantic_binding_audit_labels_from_plan(payload)
        )
        self._semantic_binding_label_projection_count = semantic_binding_label_projection_count
        payload, projection_count = _project_parameter_selections_from_plan(
            payload,
            planner_input=checked_input,
        )
        self._parameter_selection_projection_count = projection_count
        payload, evidence_projection_count = _project_evidence_declarations_from_argument_sources(
            payload
        )
        self._evidence_declaration_projection_count = evidence_projection_count
        try:
            turn = MemoQualifiedPlannerTurn.model_validate(payload)
            from x_editor.planner.request_binding import bind_execution_request

            turn, self._request_binding_audit = bind_execution_request(turn, checked_input)
            turn = validate_memo_qualified_planner_turn(turn, planner_input=checked_input)
        except (ValidationError, ValueError) as exc:
            self._contract_feedback = copy.deepcopy(getattr(exc, "feedback", None))
            self._set_invocation_trace(status="failed")
            raise MemoQualifiedPlannerContractError(
                "Planner output violates the memo-qualified contract: "
                f"{exc}; provider_output={_bounded_json(payload)}"
            ) from exc
        self._set_invocation_trace(status="completed")
        return turn

    def _append_provider_invocations(
        self,
        *,
        succeeded: bool,
        error: Exception | None = None,
    ) -> None:
        self._provider_invocations.extend(
            provider_records_from_client(
                self.client,
                provider=self.provider_name,
                model=self.model_name,
                planner_purpose="memo_qualified_plan",
                start_index=len(self._provider_invocations) + 1,
                succeeded=succeeded,
                error=error,
            )
        )

    def _set_invocation_trace(self, *, status: ComponentInvocationStatus) -> None:
        self.last_invocation_trace = ComponentInvocationTrace.build(
            component="planner",
            component_name=type(self).__name__,
            route="provider" if self._provider_invocations else "deterministic",
            status=status,
            provider_invocations=self._provider_invocations,
            metadata={
                "model": self.model_name,
                "provider": self.provider_name,
                "planner_contract": "memo_qualified/v2",
                "request_binding": copy.deepcopy(getattr(self, "_request_binding_audit", None)),
                "contract_feedback": copy.deepcopy(getattr(self, "_contract_feedback", None)),
                "prompt_package_fingerprint": self.last_prompt_package_fingerprint,
                "prompt_budget": copy.deepcopy(self.last_prompt_budget),
                "parameter_selection_projection": {
                    "source": "authored_plan_step_values",
                    "projected_value_count": (self._parameter_selection_projection_count),
                    "semantic_inference": False,
                },
                "evidence_declaration_projection": {
                    "source": "authored_argument_sources",
                    "projected_reference_count": getattr(
                        self,
                        "_evidence_declaration_projection_count",
                        0,
                    ),
                    "semantic_inference": False,
                },
                "audio_flow_projection": {
                    "source": "disabled_original_plan_topology_is_validated",
                    "projected_value_count": 0,
                    "semantic_inference": False,
                },
                "dependency_provenance_projection": copy.deepcopy(self._dependency_provenance_audit),
                "audio_usage_argument_path_projection": {
                    "source": "authored_plan_argument_and_audio_ref",
                    "projected_value_count": getattr(
                        self,
                        "_audio_usage_argument_path_projection_count",
                        0,
                    ),
                    "semantic_inference": False,
                },
                "state_transition_projection": {
                    "source": "selected_session_base_and_plan_shape",
                    "projected_value_count": self._state_transition_projection_count,
                    "semantic_inference": False,
                },
                "framework_provenance_projection": {
                    "source": "supplied_memo_and_reviewed_evidence_indexes",
                    "projected_value_count": self._framework_provenance_projection_count,
                    "semantic_inference": False,
                },
                "semantic_binding_label_projection": {
                    "source": "matching_plan_step_description_or_operation",
                    "projected_value_count": self._semantic_binding_label_projection_count,
                    "semantic_inference": False,
                },
                "wire_canonicalization": {
                    "source": "provider_authored_source_kind",
                    "removed_non_authoritative_field_count": self._wire_canonicalization_count,
                    "deduplicated_argument_source_count": (
                        self._argument_source_deduplication_count
                    ),
                    "semantic_inference": False,
                },
                "optional_tool_selection_quarantine": {
                    "source": "authored_plan_step_ids_and_operations",
                    "dropped_ledger_count": self._tool_selection_quarantine_count,
                    "plan_modified": False,
                    "semantic_inference": False,
                },
                "execution_base_projection": {
                    "source": "authored_plan_input_and_selected_session_node_slots",
                    "projected_value_count": self._execution_base_projection_count,
                    "plan_modified": False,
                    "semantic_inference": False,
                },
                **(
                    {"provider_output_on_failure": _bounded_json(self._last_provider_output)}
                    if status == "failed" and self._last_provider_output is not None
                    else {}
                ),
            },
        )


def project_operation_capability_contracts(
    cards: Mapping[str, OperationCapabilityCard | Mapping[str, Any]],
    *,
    operations: Sequence[str] | None = None,
) -> list[PlannerCandidateOperationContract]:
    """Project cards into the exact, provider-visible Planner contract."""

    selected = sorted(set(operations if operations is not None else cards))
    projected: list[PlannerCandidateOperationContract] = []
    for operation in selected:
        raw_card = cards.get(operation)
        if raw_card is None:
            continue
        card = (
            raw_card
            if isinstance(raw_card, OperationCapabilityCard)
            else OperationCapabilityCard.model_validate(raw_card)
        )
        projected.append(
            PlannerCandidateOperationContract(
                operation=card.operation,
                semantic_effects=list(card.semantic_effects),
                supported_scopes=list(card.supported_scopes),
                requirements=card.requirements.model_dump(mode="json"),
                arguments=[
                    _project_argument(item) for item in card.parameter_contract.planner_selectable
                ],
                framework_owned_arguments=list(card.parameter_contract.framework_owned),
                parameter_rules={
                    "additional_parameters": card.parameter_contract.additional_parameters,
                    "at_least_one_of": copy.deepcopy(card.parameter_contract.at_least_one_of),
                    "mutually_exclusive": copy.deepcopy(card.parameter_contract.mutually_exclusive),
                    "conditional_rules": [
                        item.model_dump(mode="json", exclude_none=True)
                        for item in card.parameter_contract.conditional_rules
                    ],
                },
                preservation_profile={
                    item.dimension: item.model_dump(mode="json", exclude={"dimension"})
                    for item in card.preservation
                },
                quality_profile={
                    item.dimension: item.model_dump(mode="json", exclude={"dimension"})
                    for item in card.quality_characteristics
                },
                artifact_risk_profile={
                    item.risk: item.model_dump(mode="json", exclude={"risk"})
                    for item in card.artifact_risks
                },
                composition_profile=card.composition.model_dump(mode="json"),
                limitations=list(card.limitations),
            )
        )
    return projected


def bind_operation_contract_runtime_domains(
    contracts: Sequence[PlannerCandidateOperationContract],
    *,
    source_audio_refs: Sequence[str],
    reference_audio_refs: Sequence[str],
    frozen_operation_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[PlannerCandidateOperationContract]:
    """Bind context-dependent argument domains before the Planner chooses values.

    Capability cards describe stable legal rules. This projection supplies the
    exact audio-reference values available in one call, moves parameters fixed
    by the runtime profile out of the Planner-owned domain, and removes selector
    values whose conditional inputs cannot be authored in that call. It does
    not choose an operation or infer user intent.
    """

    source_refs = list(dict.fromkeys(str(item) for item in source_audio_refs if str(item)))
    reference_refs = list(dict.fromkeys(str(item) for item in reference_audio_refs if str(item)))
    bound: list[PlannerCandidateOperationContract] = []
    for contract in contracts:
        frozen_paths = {
            f"params.{key}"
            for key in (frozen_operation_params or {}).get(contract.operation, {})
        }
        arguments: list[PlannerLegalArgumentContract] = []
        for argument in contract.arguments:
            if argument.argument_path in frozen_paths:
                continue
            schema = copy.deepcopy(argument.value_schema)
            if argument.runtime_value_source == "source_audio_refs":
                schema["enum"] = copy.deepcopy(source_refs)
            elif argument.runtime_value_source == "reference_audio_refs":
                schema["enum"] = copy.deepcopy(reference_refs)
            elif argument.runtime_value_source == "source_or_reference_audio_refs":
                schema["enum"] = list(dict.fromkeys([*source_refs, *reference_refs]))
            arguments.append(argument.model_copy(update={"value_schema": schema}))
        arguments = _prune_unavailable_conditional_values(
            arguments,
            parameter_rules=contract.parameter_rules,
        )
        framework_owned = list(
            dict.fromkeys(
                [
                    *contract.framework_owned_arguments,
                    *(path for path in frozen_paths if path in {
                        item.argument_path for item in contract.arguments
                    }),
                ]
            )
        )
        bound.append(
            contract.model_copy(
                update={
                    "arguments": arguments,
                    "framework_owned_arguments": framework_owned,
                }
            )
        )
    return bound


def validate_memo_qualified_planner_turn(
    turn: MemoQualifiedPlannerTurn,
    *,
    planner_input: MemoQualifiedPlannerInput,
    capability_cards: Mapping[str, OperationCapabilityCard | Mapping[str, Any]] | None = None,
) -> MemoQualifiedPlannerTurn:
    """Cross-check one Planner decision against the exact call input."""

    checked = MemoQualifiedPlannerTurn.model_validate(turn.model_dump(mode="python"))
    if checked.semantic_memo_fingerprint != planner_input.semantic_memo.fingerprint:
        raise ValueError("semantic_memo_fingerprint does not match the supplied memo")
    validate_observation_dispatch(checked, planner_input)
    candidate_operations = {item.operation for item in planner_input.candidate_operation_contracts}
    perception_capabilities = {
        str(item.get("operation") or "").strip(): item
        for item in planner_input.perception_capability_catalog
    }
    perception_operations = set(perception_capabilities)
    bound_audio_refs = {
        *planner_input.source_audio_refs,
        *planner_input.reference_audio_refs,
    }

    if checked.decision == "complete_from_session_state":
        prose_handoff = (
            planner_input.semantic_memo.requested_change_ledger_status == "provider_omitted"
        )
        if planner_input.execution_request is not None and not prose_handoff:
            raise ValueError("state-only completion requires no remaining execution request")
        if not prose_handoff and checked.changes_satisfied_by_base != (
            planner_input.semantic_memo.changes_satisfied_by_base
        ):
            raise ValueError(
                "state-only completion must exactly copy changes satisfied by the base"
            )
        _validate_reasoner_selected_base(checked, planner_input=planner_input)
        _validate_state_completion_base(checked, planner_input=planner_input)
        if prose_handoff:
            _validate_prose_state_completion_authority(
                checked,
                planner_input=planner_input,
            )
        return checked

    if (
        planner_input.execution_request is None
        and planner_input.semantic_memo.requested_change_ledger_status == "complete"
    ):
        raise ValueError(
            "a memo with no remaining execution request requires state-only completion"
        )

    if checked.decision == "request_perception":
        if not set(checked.candidate_operations).issubset(candidate_operations):
            raise ValueError("perception request names an edit operation outside candidates")
        for job in checked.perception_jobs:
            job_audio_refs = {job.primary_audio_ref, *job.supporting_audio_refs}
            if not job_audio_refs.issubset(bound_audio_refs):
                raise ValueError(f"Perception job {job.job_id!r} contains an unbound audio ref")
            route = route_perception_observation_need(
                target_kinds=(item.kind for item in job.perception_targets),
                expected_evidence_fields=job.expected_evidence_fields,
                temporal_precision=job.temporal_precision,
                available_operations=perception_operations,
                selected_operation_hint=_perception_operation_hint(job.perception_operation),
            )
            route_parts = route.route_parts
            if not route_parts:
                continue
            shape_errors = {
                part.operation: error
                for part in route_parts
                if (error := _perception_route_audio_shape_error(
                    part.operation,
                    supporting_audio_count=len(job.supporting_audio_refs),
                )) is not None
            }
            if shape_errors:
                raise PerceptionJobShapeError(
                    job_id=job.job_id,
                    audio_refs=[job.primary_audio_ref, *job.supporting_audio_refs],
                    route_errors=shape_errors,
                )
        _validate_reasoner_selected_base(checked, planner_input=planner_input)
        _validate_perception_base(checked, planner_input=planner_input)
        return checked

    if checked.decision != "execute":
        return checked

    unresolved_targets = required_semantic_target_binding_failures(planner_input.semantic_memo)
    if unresolved_targets:
        raise ValueError(
            f"execute requires resolved semantic target bindings: {unresolved_targets}"
        )

    assert checked.plan is not None
    plan = checked.plan
    if planner_input.execution_request is None or not equivalent_text_copy(
        plan.request, planner_input.execution_request
    ):
        raise ValueError(
            "plan.request must match the Reasoner's remaining execution request "
            "apart from casing, whitespace, and canonical Unicode representation"
        )
    if not plan.steps:
        raise ValueError("execute requires at least one edit step")
    unknown_operations = sorted({step.operation for step in plan.steps} - candidate_operations)
    if unknown_operations:
        raise ValueError(f"plan operations are outside supplied candidates: {unknown_operations}")
    _validate_reasoner_selected_base(checked, planner_input=planner_input)
    _validate_plan_topology(plan)
    _validate_session_base(checked, planner_input=planner_input)

    contracts_by_operation = {
        item.operation: item for item in planner_input.candidate_operation_contracts
    }
    if checked.tool_selection is not None:
        selections = {item.plan_step_id: item for item in checked.tool_selection.steps}
        for step in plan.steps:
            selection = selections[step.id]
            unknown_candidates = sorted(
                {
                    candidate.operation
                    for candidate in selection.candidates
                    if candidate.operation not in candidate_operations
                }
            )
            if unknown_candidates:
                raise ValueError(
                    f"{step.id} compares operations outside supplied candidates: {unknown_candidates}"
                )
            contract = contracts_by_operation[step.operation]
            _validate_parameter_declarations(step, selection, contract)

    if capability_cards is not None:
        for step in plan.steps:
            raw_card = capability_cards.get(step.operation)
            if raw_card is None:
                raise ValueError(f"no authoritative capability card for {step.operation}")
            card = (
                raw_card
                if isinstance(raw_card, OperationCapabilityCard)
                else OperationCapabilityCard.model_validate(raw_card)
            )
            violations = validate_operation_arguments(
                card,
                inputs=step.inputs,
                params=step.params,
            )
            if violations:
                summary = ", ".join(
                    f"{item.code}:{item.path or '<operation>'}" for item in violations
                )
                raise ValueError(f"{step.id} violates operation parameter contract: {summary}")
    return checked


def _perception_operation_hint(operation: str) -> str | None:
    """Return the optional model-authored route hint, excluding neutral auto."""

    normalized = str(operation or "").strip()
    return None if normalized in {"", "auto"} else normalized


def _perception_route_audio_shape_error(
    operation: str,
    *,
    supporting_audio_count: int,
) -> str | None:
    capability = workflow_perception_capability(operation)
    if capability is None:
        return None
    if capability.audio_input_mode == "single_audio" and supporting_audio_count:
        return "single-audio capability cannot consume supporting audio refs"
    if capability.audio_input_mode == "paired_audio" and supporting_audio_count != 1:
        return "paired-audio capability requires exactly one supporting audio ref"
    if capability.audio_input_mode == "multi_audio" and supporting_audio_count < 1:
        return "multi-audio capability requires at least one supporting audio ref"
    return None


def memo_qualified_planner_response_format(
    *,
    phase: Literal["evidence_planning", "execution_planning"] = "execution_planning",
    candidate_operation_contracts: Sequence[PlannerCandidateOperationContract] = (),
    reviewed_evidence_ids: Sequence[str] = (),
    evidence_path_sources: Mapping[str, Sequence[str]] | None = None,
    perception_operation_ids: Sequence[str] = (),
    perception_capability_catalog: Sequence[Mapping[str, Any]] = (),
    bound_audio_refs: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "MemoQualifiedPlannerTurn",
            "schema": _memo_qualified_planner_response_schema(
                phase,
                candidate_operation_contracts=candidate_operation_contracts,
                reviewed_evidence_ids=reviewed_evidence_ids,
                evidence_path_sources=evidence_path_sources,
                perception_operation_ids=perception_operation_ids,
                perception_capability_catalog=perception_capability_catalog,
                bound_audio_refs=bound_audio_refs,
            ),
            "strict": True,
        },
    }


def memo_qualified_planner_provider_response_format(
    *,
    phase: Literal["evidence_planning", "execution_planning"] = "execution_planning",
    require_observation: bool = False,
) -> dict[str, Any]:
    """Stable provider wire envelope; Runtime owns conditional validation."""

    decisions = (["request_perception", "cannot_plan"] if require_observation else
                 ["request_perception", "execute", "complete_from_session_state", "cannot_plan"])
    string_array = {"items": {"type": "string"}, "type": "array"}
    perception_target = {
        "additionalProperties": False,
        "properties": {
            "kind": {
                "enum": [
                    "event_label",
                    "speaker_hint",
                    "quoted_text",
                    "semantic_property",
                    "audio_property",
                ],
                "type": "string",
            },
            "value": {"type": "string"},
        },
        "required": ["kind", "value"],
        "type": "object",
    }
    perception_job = {
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "job_id": {"type": "string"},
            "scope": {"enum": ["general", "targeted"], "type": "string"},
            "perception_operation": {"type": "string"},
            "primary_audio_ref": {"type": "string"},
            "supporting_audio_refs": string_array,
            "requirements": string_array,
            "expected_evidence_fields": string_array,
            "perception_targets": {"items": perception_target, "type": "array"},
            "temporal_precision": {
                "enum": ["coarse", "segment", "word"],
                "type": "string",
            },
            "target_hints": string_array,
            "evidence_questions": string_array,
            "observation_need_ids": string_array,
            "recognition_context": {"type": ["string", "null"], "maxLength": 2048},
            "required": {"type": "boolean"},
        },
        "required": ["job_id", "primary_audio_ref", "temporal_precision"],
        "type": "object",
    }
    plan_step = {
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "operation": {"type": "string"},
            "description": {"type": "string"},
            "inputs": {"type": "object"},
            "params": {"type": "object"},
            "expected_output": {},
            "depends_on": string_array,
            "required": {"type": "boolean"},
        },
        "required": ["id", "operation", "inputs", "params", "depends_on"],
        "type": "object",
    }
    output_artifact = {
        "additionalProperties": False,
        "properties": {
            "artifact_id": {"type": "string"},
            "from_step": {"type": "string"},
            "from_output": {"type": "string"},
            "role": {"type": "string"},
            "primary": {"type": "boolean"},
        },
        "required": ["artifact_id", "from_step", "role", "primary"],
        "type": "object",
    }
    plan = {
        "additionalProperties": False,
        "properties": {
            "request": {"type": "string"},
            "input_audio": {"type": "string"},
            "output_audio": {},
            "steps": {"items": plan_step, "type": "array", "minItems": 1},
            "output_artifacts": {"items": output_artifact, "type": "array"},
            "global_params": {"type": "object"},
            "metadata": {"type": "object"},
            "missing_inputs": string_array,
        },
        "required": ["request", "input_audio", "steps"],
        "type": "object",
    }
    argument_source = {
        "additionalProperties": False,
        "properties": {
            "argument_path": {"type": "string"},
            "source_kind": {
                "enum": [
                    "user_request",
                    "deterministic_context",
                    "perception_evidence",
                    "derived_from_evidence",
                ],
                "type": "string",
            },
            "evidence_id": {"type": "string"},
            "evidence_path": {"type": "string"},
        },
        "required": ["argument_path", "source_kind"],
        "type": "object",
    }
    semantic_binding = {
        "additionalProperties": False,
        "properties": {
            "plan_step_id": {"type": "string"},
            "target": {"type": "string"},
            "operation": {"type": "string"},
            "argument_sources": {"items": argument_source, "type": "array"},
            "unresolved_requirements": string_array,
        },
        "required": ["plan_step_id", "target", "operation"],
        "type": "object",
    }
    audio_usage_binding = {
        "additionalProperties": False,
        "properties": {
            "plan_step_id": {"type": "string"},
            "argument_path": {"type": "string"},
            "audio_ref": {"type": "string"},
            "usage": {
                "enum": [
                    "edit_input",
                    "reference_input",
                    "speaker_grounded_reference",
                    "assembly_member",
                    "comparison_input",
                ],
                "type": "string",
            },
        },
        "required": ["plan_step_id", "argument_path", "audio_ref", "usage"],
        "type": "object",
    }
    execution_semantics = {
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "bindings": {"items": semantic_binding, "type": "array"},
            "audio_usage_bindings": {"items": audio_usage_binding, "type": "array"},
        },
        "type": "object",
    }
    tool_candidate = {
        "additionalProperties": False,
        "properties": {
            "operation": {"type": "string"},
            "semantic_fit": {
                "enum": ["full", "partial", "none", "unknown"],
                "type": "string",
            },
            "preservation_fit": {
                "enum": ["full", "partial", "none", "unknown"],
                "type": "string",
            },
            "quality_fit": {
                "enum": ["full", "partial", "none", "unknown"],
                "type": "string",
            },
            "legal_inputs_available": {"type": "boolean"},
            "rejection_codes": string_array,
        },
        "required": [
            "operation",
            "semantic_fit",
            "preservation_fit",
            "quality_fit",
            "legal_inputs_available",
        ],
        "type": "object",
    }
    tool_selection_step = {
        "additionalProperties": False,
        "properties": {
            "plan_step_id": {"type": "string"},
            "selected_operation": {"type": "string"},
            "candidates": {"items": tool_candidate, "type": "array"},
            "selection_summary": {"type": "string"},
        },
        "required": ["plan_step_id", "selected_operation", "candidates"],
        "type": "object",
    }
    cannot_plan_reason = {
        "additionalProperties": False,
        "properties": {
            "code": {
                "enum": [
                    "missing_evidence",
                    "missing_perception_route",
                    "unsupported_capability",
                    "semantic_ambiguity",
                    "conflicting_evidence",
                    "invalid_parameter_domain",
                    "unsupported_output_topology",
                    "other",
                ],
                "type": "string",
            },
            "message": {"type": "string"},
            "target_ids": string_array,
            "operation_ids": string_array,
        },
        "required": ["code", "message"],
        "type": "object",
    }
    schema: dict[str, Any] = {
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string"},
            "decision": {"enum": decisions, "type": "string"},
            "selected_base_state_node_id": {},
            "selected_base_audio_ref": {},
            "state_transition_kind": {
                "enum": ["edit_current", "branch_from_history", "restore_history"],
                "type": "string",
            },
            "state_completion_kind": {
                "enum": ["select_existing", "restore_history"],
                "type": "string",
            },
            "changes_satisfied_by_base": {"items": {"type": "string"}, "type": "array"},
            "perception_jobs": {"items": perception_job, "type": "array"},
            "candidate_operations": {"items": {"type": "string"}, "type": "array"},
            "plan": plan,
            "execution_semantics": execution_semantics,
            "tool_selection": {
                "additionalProperties": False,
                "properties": {
                    "schema_version": {"type": "string"},
                    "steps": {"items": tool_selection_step, "type": "array"},
                },
                "required": ["steps"],
                "type": "object",
            },
            "missing_information": {"items": {"type": "string"}, "type": "array"},
            "semantic_conflict": {},
            "cannot_plan_reasons": {"items": cannot_plan_reason, "type": "array"},
            "decision_summary": {
                "maxLength": PLANNER_DECISION_SUMMARY_MAX_LENGTH,
                "type": "string",
            },
        },
        "required": ["decision"],
        "type": "object",
    }
    if require_observation:
        schema["properties"]["perception_jobs"]["minItems"] = 1
        cannot_plan_reason["properties"]["code"]["enum"] = [
            "unsupported_capability", "missing_perception_route"
        ]
        schema["properties"]["cannot_plan_reasons"]["minItems"] = 1
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "MemoQualifiedPlannerWireEnvelope",
            "schema": schema,
            "strict": True,
        },
    }


def _memo_qualified_planner_response_schema(
    phase: Literal["evidence_planning", "execution_planning"],
    *,
    candidate_operation_contracts: Sequence[PlannerCandidateOperationContract] = (),
    reviewed_evidence_ids: Sequence[str] = (),
    evidence_path_sources: Mapping[str, Sequence[str]] | None = None,
    perception_operation_ids: Sequence[str] = (),
    perception_capability_catalog: Sequence[Mapping[str, Any]] = (),
    bound_audio_refs: Sequence[str] = (),
) -> dict[str, Any]:
    model = (
        _EvidencePlanningResponse if phase == "evidence_planning" else _ExecutionPlanningResponse
    )
    schema = TypeAdapter(model).json_schema()
    executable_contracts = (
        [
            item
            for item in candidate_operation_contracts
            if not list((item.requirements or {}).get("evidence") or [])
        ]
        if phase == "evidence_planning"
        else list(candidate_operation_contracts)
    )
    if executable_contracts:
        _bind_step_schema_to_operation_contracts(
            schema,
            candidate_operation_contracts=executable_contracts,
        )
    if evidence_path_sources:
        _bind_evidence_reference_schema(
            schema,
            reviewed_evidence_ids=reviewed_evidence_ids,
            evidence_path_sources=evidence_path_sources,
        )
    _bind_perception_request_domains(
        schema,
        candidate_operation_ids=[item.operation for item in candidate_operation_contracts],
        perception_operation_ids=perception_operation_ids,
        perception_capability_catalog=perception_capability_catalog,
        bound_audio_refs=bound_audio_refs,
    )
    return schema


def _bind_perception_request_domains(
    response_schema: dict[str, Any],
    *,
    candidate_operation_ids: Sequence[str],
    perception_operation_ids: Sequence[str],
    perception_capability_catalog: Sequence[Mapping[str, Any]],
    bound_audio_refs: Sequence[str],
) -> None:
    """Expose exact runtime ids for the pre-evidence Planner decision."""

    definitions = response_schema.get("$defs")
    if not isinstance(definitions, dict):
        raise TypeError("Planner response schema is missing $defs")
    request_turn = definitions.get("_MemoQualifiedPerceptionRequestTurn")
    job = definitions.get("PlannerPerceptionJob")
    if not isinstance(request_turn, dict) or not isinstance(job, dict):
        raise TypeError("Planner response schema is missing Perception job contracts")
    turn_properties = request_turn.get("properties")
    job_properties = job.get("properties")
    if not isinstance(turn_properties, dict) or not isinstance(job_properties, dict):
        raise TypeError("Planner Perception job schema is missing properties")
    candidates = turn_properties.get("candidate_operations")
    supporting_refs = job_properties.get("supporting_audio_refs")
    if not isinstance(candidates, dict) or not isinstance(supporting_refs, dict):
        raise TypeError("Planner Perception job domains are missing")
    candidate_ids = sorted(set(candidate_operation_ids))
    perception_ids = sorted(
        {
            *perception_operation_ids,
            *(str(item.get("operation") or "") for item in perception_capability_catalog),
        }
        - {""}
    )
    admitted_audio_refs = list(dict.fromkeys(bound_audio_refs))
    candidates["items"] = {"enum": candidate_ids, "type": "string"}
    job_properties["perception_operation"] = {
        "enum": ["auto", *perception_ids],
        "type": "string",
        "default": "auto",
        "description": (
            "Optional backward-compatible hint. The Perception Gateway owns the concrete route."
        ),
    }
    job_properties["primary_audio_ref"] = {
        "enum": admitted_audio_refs,
        "type": "string",
    }
    supporting_refs["items"] = {"enum": admitted_audio_refs, "type": "string"}
    required_fields = job.setdefault("required", [])
    if not isinstance(required_fields, list):
        raise TypeError("Planner Perception job schema has invalid required fields")
    all_of = job.setdefault("allOf", [])
    if not isinstance(all_of, list):
        raise TypeError("Planner Perception job schema has invalid conditional constraints")


def _bind_evidence_reference_schema(
    response_schema: dict[str, Any],
    *,
    reviewed_evidence_ids: Sequence[str],
    evidence_path_sources: Mapping[str, Sequence[str]],
) -> None:
    """Restrict provider-authored evidence references to admitted id/path pairs."""

    admitted_ids = set(reviewed_evidence_ids)
    paths_by_id: dict[str, set[str]] = {evidence_id: set() for evidence_id in admitted_ids}
    for raw_path, raw_ids in evidence_path_sources.items():
        path = str(raw_path).strip()
        if not path:
            continue
        for raw_id in raw_ids:
            evidence_id = str(raw_id).strip()
            if evidence_id in admitted_ids:
                paths_by_id[evidence_id].add(path)
    paths_by_id = {evidence_id: paths for evidence_id, paths in paths_by_id.items() if paths}
    if not paths_by_id:
        return

    definitions = response_schema.get("$defs")
    if not isinstance(definitions, dict):
        raise TypeError("Planner response schema is missing $defs")
    argument_source = definitions.get("_ProviderEvidenceArgumentSource")
    if not isinstance(argument_source, dict):
        raise TypeError("Planner response schema is missing evidence source definition")
    definitions["_ProviderEvidenceArgumentSource"] = {
        "oneOf": [
            _evidence_pair_variant(
                argument_source,
                evidence_id=evidence_id,
                path_property="evidence_path",
                paths=paths,
            )
            for evidence_id, paths in sorted(paths_by_id.items())
        ],
        "title": "AdmittedEvidenceArgumentSource",
    }


def _evidence_pair_variant(
    base_schema: Mapping[str, Any],
    *,
    evidence_id: str,
    path_property: str,
    paths: set[str],
) -> dict[str, Any]:
    variant = copy.deepcopy(dict(base_schema))
    properties = variant.get("properties")
    if not isinstance(properties, dict):
        raise TypeError("Evidence reference schema is missing properties")
    properties["evidence_id"] = {"const": evidence_id, "type": "string"}
    properties[path_property] = {
        "enum": sorted(paths),
        "type": "string",
    }
    return variant


def _bind_step_schema_to_operation_contracts(
    response_schema: dict[str, Any],
    *,
    candidate_operation_contracts: Sequence[PlannerCandidateOperationContract],
) -> None:
    """Make each generated step obey the exact contract of its chosen operation."""

    definitions = response_schema.get("$defs")
    if not isinstance(definitions, dict):
        raise TypeError("Planner response schema is missing $defs")
    step_definition_name = "_ProviderAudioEditStep"
    base_step_schema = definitions.get(step_definition_name)
    if not isinstance(base_step_schema, dict):
        raise TypeError(f"Planner response schema is missing {step_definition_name}")

    operation_variants = [
        _operation_bound_step_variant(
            base_step_schema,
            contract=contract,
        )
        for contract in candidate_operation_contracts
    ]
    definitions[step_definition_name] = {
        "oneOf": operation_variants,
        "title": "OperationBoundAudioEditStep",
    }


def _operation_bound_step_variant(
    base_step_schema: Mapping[str, Any],
    *,
    contract: PlannerCandidateOperationContract,
) -> dict[str, Any]:
    variant = copy.deepcopy(dict(base_step_schema))
    properties = variant.get("properties")
    if not isinstance(properties, dict):
        raise TypeError("Provider AudioEditStep schema is missing properties")
    properties["operation"] = {
        "const": contract.operation,
        "type": "string",
    }
    required_step_fields = list(variant.get("required") or [])
    for namespace in ("inputs", "params"):
        arguments = [
            item for item in contract.arguments if item.argument_path.startswith(f"{namespace}.")
        ]
        properties[namespace] = _operation_argument_container_schema(
            namespace,
            arguments=arguments,
            reject_additional=(contract.parameter_rules.get("additional_parameters") == "reject"),
        )
        if any(item.required for item in arguments) and namespace not in required_step_fields:
            required_step_fields.append(namespace)
    variant["required"] = required_step_fields
    parameter_constraints = _operation_parameter_constraint_schemas(contract)
    if parameter_constraints:
        variant.setdefault("allOf", []).extend(parameter_constraints)
    variant["title"] = f"AudioEditStep[{contract.operation}]"
    return variant


def _operation_argument_container_schema(
    namespace: Literal["inputs", "params"],
    *,
    arguments: Sequence[PlannerLegalArgumentContract],
    reject_additional: bool,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for argument in arguments:
        _, separator, argument_name = argument.argument_path.partition(".")
        if separator != "." or not argument_name or "." in argument_name:
            raise ValueError(
                "Operation-bound Planner schemas currently require one-level argument "
                f"paths, got {argument.argument_path!r}"
            )
        value_schema = copy.deepcopy(argument.value_schema)
        if namespace == "inputs" and value_schema.get("type") == "string":
            value_schema = {
                "anyOf": [value_schema, _step_output_reference_schema()],
            }
        properties[argument_name] = value_schema
        if argument.required:
            required.append(argument_name)
    return {
        "additionalProperties": not reject_additional,
        "properties": properties,
        "required": required,
        "title": namespace.title(),
        "type": "object",
    }


def _operation_parameter_constraint_schemas(
    contract: PlannerCandidateOperationContract,
) -> list[dict[str, Any]]:
    """Compile Planner-authorable capability rules into the provider schema."""

    authored_paths = {item.argument_path for item in contract.arguments}
    rules = contract.parameter_rules
    constraints: list[dict[str, Any]] = []
    for raw_group in rules.get("at_least_one_of") or []:
        group = [str(item) for item in raw_group]
        if group and set(group).issubset(authored_paths):
            constraints.append({"anyOf": [_argument_presence_schema(path) for path in group]})
    for raw_group in rules.get("mutually_exclusive") or []:
        group = [str(item) for item in raw_group]
        if len(group) > 1 and set(group).issubset(authored_paths):
            constraints.append(
                {
                    "not": {
                        "anyOf": [
                            {
                                "allOf": [
                                    _argument_presence_schema(left),
                                    _argument_presence_schema(right),
                                ]
                            }
                            for left, right in combinations(group, 2)
                        ]
                    }
                }
            )
    for raw_rule in rules.get("conditional_rules") or []:
        if not isinstance(raw_rule, Mapping):
            continue
        condition = raw_rule.get("when")
        if not isinstance(condition, Mapping):
            continue
        referenced_paths = {
            str(condition.get("path") or ""),
            *(str(item) for item in raw_rule.get("require") or []),
            *(str(item) for item in raw_rule.get("forbid") or []),
        }
        if "" in referenced_paths or not referenced_paths.issubset(authored_paths):
            continue
        consequences = [
            *(_argument_presence_schema(str(path)) for path in raw_rule.get("require") or []),
            *(
                {"not": _argument_presence_schema(str(path))}
                for path in raw_rule.get("forbid") or []
            ),
        ]
        if consequences:
            constraints.append(
                {
                    "if": _argument_condition_schema(condition),
                    "then": {"allOf": consequences},
                }
            )
    return constraints


def _argument_condition_schema(condition: Mapping[str, Any]) -> dict[str, Any]:
    path = str(condition.get("path") or "")
    operator = str(condition.get("operator") or "")
    presence = _argument_presence_schema(path)
    if operator == "present":
        return presence
    if operator == "absent":
        return {"not": presence}
    value_schema = (
        {"const": copy.deepcopy(condition.get("value"))}
        if operator == "equals"
        else {"enum": copy.deepcopy(list(condition.get("values") or []))}
    )
    namespace, argument_name = _split_one_level_argument_path(path)
    return {
        "properties": {
            namespace: {
                "properties": {argument_name: value_schema},
                "required": [argument_name],
            }
        },
        "required": [namespace],
    }


def _argument_presence_schema(path: str) -> dict[str, Any]:
    namespace, argument_name = _split_one_level_argument_path(path)
    return {
        "properties": {namespace: {"required": [argument_name]}},
        "required": [namespace],
    }


def _split_one_level_argument_path(path: str) -> tuple[str, str]:
    namespace, separator, argument_name = path.partition(".")
    if (
        separator != "."
        or namespace not in {"inputs", "params"}
        or not argument_name
        or "." in argument_name
    ):
        raise ValueError(
            f"Operation-bound Planner schemas require one-level input/param paths, got {path!r}"
        )
    return namespace, argument_name


def _step_output_reference_schema() -> dict[str, Any]:
    return {
        "additionalProperties": False,
        "properties": {
            "from_step": {"minLength": 1, "type": "string"},
            "from_output": {
                "const": "output_audio_path",
                "type": "string",
            },
        },
        "required": ["from_step", "from_output"],
        "type": "object",
    }


def _project_framework_owned_turn_provenance(
    payload: Mapping[str, Any],
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> tuple[dict[str, Any], int]:
    """Inject mechanical memo and audio-evidence provenance after generation."""

    projected = copy.deepcopy(dict(payload))
    projected["semantic_memo_fingerprint"] = planner_input.semantic_memo.fingerprint
    projected_count = 1
    if (
        projected.get("decision")
        in {
            "request_perception",
            "execute",
            "complete_from_session_state",
        }
        and planner_input.semantic_memo.selected_base_state_node_id is not None
    ):
        projected["selected_base_state_node_id"] = (
            planner_input.semantic_memo.selected_base_state_node_id
        )
        projected["selected_base_audio_ref"] = planner_input.semantic_memo.selected_base_audio_ref
        projected_count += 2
    if projected.get("decision") == "execute":
        plan = projected.get("plan")
        selected_audio_ref = planner_input.semantic_memo.selected_base_audio_ref
        if isinstance(plan, dict) and selected_audio_ref is not None:
            plan["input_audio"] = selected_audio_ref
            projected_count += 1
    if projected.get("decision") != "execute":
        return projected, projected_count
    semantics = projected.get("execution_semantics")
    bindings = semantics.get("audio_usage_bindings") if isinstance(semantics, dict) else None
    if not isinstance(bindings, list):
        return projected, projected_count
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        usage = str(binding.get("usage") or "")
        requirement = audio_evidence_requirement_for_usage(usage)  # type: ignore[arg-type]
        evidence = []
        if usage == "speaker_grounded_reference":
            evidence = [
                item.model_dump(mode="json")
                for item in canonical_speaker_evidence_references(
                    audio_ref=str(binding.get("audio_ref") or ""),
                    evidence_path_sources=planner_input.evidence_path_sources,
                    evidence_path_audio_refs=planner_input.evidence_path_audio_refs,
                )
            ]
        binding["schema_version"] = "x-editor.workflow-audio-usage-binding/v1"
        binding["evidence_requirement"] = requirement
        binding["evidence"] = evidence
        projected_count += 3
    return projected, projected_count


def _canonicalize_provider_wire_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int, int]:
    """Canonicalize non-authoritative provider audit fields without editing PlanIR."""

    projected = copy.deepcopy(dict(payload))
    semantics = projected.get("execution_semantics")
    bindings = semantics.get("bindings") if isinstance(semantics, dict) else None
    removed = 0
    deduplicated = 0
    if isinstance(bindings, list):
        for binding in bindings:
            sources = binding.get("argument_sources") if isinstance(binding, dict) else None
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict) or source.get("source_kind") not in {
                    "user_request",
                    "deterministic_context",
                }:
                    continue
                for field_name in ("evidence_id", "evidence_path"):
                    if field_name in source:
                        source.pop(field_name)
                        removed += 1
            unique_sources: list[Any] = []
            for source in sources:
                if source in unique_sources:
                    deduplicated += 1
                    continue
                unique_sources.append(source)
            binding["argument_sources"] = unique_sources

    quarantined = 0
    if not _optional_tool_selection_matches_plan(projected):
        projected.pop("tool_selection", None)
        quarantined = 1
    return projected, removed, deduplicated, quarantined


def _optional_tool_selection_matches_plan(payload: Mapping[str, Any]) -> bool:
    """Return whether an optional audit ledger exactly names the authored PlanIR steps."""

    tool_selection = payload.get("tool_selection")
    if tool_selection is None:
        return True
    if payload.get("decision") != "execute" or not isinstance(tool_selection, Mapping):
        return False
    plan = payload.get("plan")
    raw_steps = plan.get("steps") if isinstance(plan, Mapping) else None
    raw_selections = tool_selection.get("steps")
    if not isinstance(raw_steps, list) or not isinstance(raw_selections, list):
        return False
    plan_steps = {
        str(step.get("id") or ""): str(step.get("operation") or "")
        for step in raw_steps
        if isinstance(step, Mapping)
        and str(step.get("id") or "")
        and str(step.get("operation") or "")
    }
    if len(plan_steps) != len(raw_steps):
        return False
    selection_steps: dict[str, str] = {}
    for selection in raw_selections:
        if not isinstance(selection, Mapping):
            return False
        step_id = str(selection.get("plan_step_id") or "")
        operation = str(selection.get("selected_operation") or "")
        if not step_id or not operation or step_id in selection_steps:
            return False
        candidates = selection.get("candidates")
        if not isinstance(candidates, list) or not any(
            isinstance(candidate, Mapping)
            and str(candidate.get("operation") or "") == operation
            for candidate in candidates
        ):
            return False
        selection_steps[step_id] = operation
    return selection_steps == plan_steps


def _project_semantic_binding_audit_labels_from_plan(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Project non-authoritative binding labels from the matching PlanIR step."""

    projected = copy.deepcopy(dict(payload))
    if projected.get("decision") != "execute":
        return projected, 0
    plan = projected.get("plan")
    semantics = projected.get("execution_semantics")
    steps = plan.get("steps") if isinstance(plan, dict) else None
    bindings = semantics.get("bindings") if isinstance(semantics, dict) else None
    if not isinstance(steps, list) or not isinstance(bindings, list):
        return projected, 0
    steps_by_id = {
        str(step.get("id") or ""): step
        for step in steps
        if isinstance(step, dict) and str(step.get("id") or "")
    }
    projected_count = 0
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        step = steps_by_id.get(str(binding.get("plan_step_id") or ""))
        if step is None:
            continue
        label = str(step.get("description") or "").strip()
        if not label:
            label = str(step.get("operation") or binding.get("operation") or "").strip()
        if not label:
            continue
        binding["objective"] = label
        projected_count += 1
    return projected, projected_count


def _project_parameter_selections_from_plan(
    payload: Mapping[str, Any],
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> tuple[dict[str, Any], int]:
    """Project the audit ledger from the Planner's single authored value source."""

    projected = copy.deepcopy(dict(payload))
    if projected.get("decision") != "execute":
        return projected, 0
    plan = projected.get("plan")
    tool_selection = projected.get("tool_selection")
    if not isinstance(plan, dict) or not isinstance(tool_selection, dict):
        return projected, 0
    raw_steps = plan.get("steps")
    raw_selections = tool_selection.get("steps")
    if not isinstance(raw_steps, list) or not isinstance(raw_selections, list):
        return projected, 0

    steps_by_id = {
        str(step.get("id") or ""): step
        for step in raw_steps
        if isinstance(step, dict) and str(step.get("id") or "")
    }
    contracts_by_operation = {
        item.operation: item for item in planner_input.candidate_operation_contracts
    }
    projected_count = 0
    for selection in raw_selections:
        if not isinstance(selection, dict):
            continue
        step = steps_by_id.get(str(selection.get("plan_step_id") or ""))
        operation = str(selection.get("selected_operation") or "")
        contract = contracts_by_operation.get(operation)
        if step is None or contract is None or step.get("operation") != operation:
            continue
        rows: list[dict[str, Any]] = []
        for argument in contract.arguments:
            found, value = _raw_step_argument_value(step, argument.argument_path)
            if not found:
                continue
            rows.append(
                {
                    "argument_path": argument.argument_path,
                    "selected_value": copy.deepcopy(value),
                    "contract_operation": operation,
                }
            )
        selection["parameter_selections"] = rows
        projected_count += len(rows)
    return projected, projected_count


def _project_evidence_declarations_from_argument_sources(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Project redundant evidence ledgers from provider-authored argument sources."""

    projected = copy.deepcopy(dict(payload))
    if projected.get("decision") != "execute":
        return projected, 0
    semantics = projected.get("execution_semantics")
    if not isinstance(semantics, dict):
        return projected, 0
    bindings = semantics.get("bindings")
    if not isinstance(bindings, list):
        return projected, 0

    projected_count = 0
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        sources = binding.get("argument_sources")
        if not isinstance(sources, list):
            continue
        pairs: list[tuple[str, str]] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            if source.get("source_kind") not in {
                "perception_evidence",
                "derived_from_evidence",
            }:
                continue
            evidence_id = str(source.get("evidence_id") or "").strip()
            evidence_path = str(source.get("evidence_path") or "").strip()
            pair = (evidence_id, evidence_path)
            if evidence_id and evidence_path and pair not in pairs:
                pairs.append(pair)
        binding["evidence_ids"] = list(dict.fromkeys(item[0] for item in pairs))
        binding["evidence_paths"] = [
            {"evidence_id": evidence_id, "field_path": evidence_path}
            for evidence_id, evidence_path in pairs
        ]
        projected_count += len(pairs)
    return projected, projected_count


def _raw_step_argument_value(
    step: Mapping[str, Any],
    argument_path: str,
) -> tuple[bool, Any]:
    namespace, separator, key = argument_path.partition(".")
    if separator != "." or not key or "." in key:
        return False, None
    container = step.get(namespace)
    if not isinstance(container, Mapping) or key not in container:
        return False, None
    return True, container[key]


def _bounded_json(value: Any, *, max_chars: int = 6000) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + "...<truncated>"


def _project_argument(spec: OperationParameterSpec) -> PlannerLegalArgumentContract:
    schema: dict[str, Any] = {
        "type": {
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "object": "object",
            "array": "array",
            "string": "string",
            "enum": "string",
        }[spec.value_type]
    }
    if spec.enum_values:
        schema["enum"] = copy.deepcopy(spec.enum_values)
        enum_types = {type(item) for item in spec.enum_values}
        if len(enum_types) == 1:
            only_type = next(iter(enum_types))
            schema["type"] = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
            }.get(only_type, schema["type"])
    if spec.minimum is not None:
        schema["minimum"] = spec.minimum
    if spec.maximum is not None:
        schema["maximum"] = spec.maximum
    if spec.min_items is not None:
        schema["minItems"] = spec.min_items
    if spec.max_items is not None:
        schema["maxItems"] = spec.max_items
    if spec.value_type == "array":
        # Capability cards currently constrain array cardinality but do not yet
        # carry an item schema. Keep that domain honest and explicit: any JSON
        # item is legal until the card contract is enriched with item typing.
        schema["items"] = {}
    if spec.default is not None:
        schema["default"] = copy.deepcopy(spec.default)
    description_parts = [spec.semantic_role]
    description_parts.extend(spec.value_guidance)
    if spec.value_examples:
        description_parts.append("Examples: " + json.dumps(spec.value_examples, ensure_ascii=False))
    return PlannerLegalArgumentContract(
        argument_path=spec.path,
        required=spec.required,
        value_schema=schema,
        semantic_role=spec.semantic_role,
        runtime_value_source=spec.runtime_value_source,
        description=" ".join(description_parts),
    )


def _prune_unavailable_conditional_values(
    arguments: list[PlannerLegalArgumentContract],
    *,
    parameter_rules: Mapping[str, Any],
) -> list[PlannerLegalArgumentContract]:
    by_path = {item.argument_path: item for item in arguments}
    removed_by_path: dict[str, set[Any]] = {}
    for raw_rule in parameter_rules.get("conditional_rules") or []:
        if not isinstance(raw_rule, Mapping):
            continue
        when = raw_rule.get("when")
        if not isinstance(when, Mapping) or when.get("operator") != "in":
            continue
        selector_path = str(when.get("path") or "")
        selector = by_path.get(selector_path)
        if selector is None or "enum" not in selector.value_schema:
            continue
        required_paths = [str(item) for item in raw_rule.get("require") or []]
        if not any(_argument_domain_is_empty(by_path.get(path)) for path in required_paths):
            continue
        removed_by_path.setdefault(selector_path, set()).update(
            item for item in when.get("values") or []
        )

    if not removed_by_path:
        return arguments
    projected: list[PlannerLegalArgumentContract] = []
    for argument in arguments:
        removed = removed_by_path.get(argument.argument_path)
        if not removed:
            projected.append(argument)
            continue
        schema = copy.deepcopy(argument.value_schema)
        schema["enum"] = [item for item in schema.get("enum") or [] if item not in removed]
        projected.append(argument.model_copy(update={"value_schema": schema}))
    return projected


def _argument_domain_is_empty(
    argument: PlannerLegalArgumentContract | None,
) -> bool:
    return argument is None or (
        "enum" in argument.value_schema and not argument.value_schema.get("enum")
    )


def _project_audio_usage_binding_argument_paths(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Refine a list-valued audio binding to its uniquely matching scalar element."""

    projected = copy.deepcopy(dict(payload))
    if projected.get("decision") != "execute":
        return projected, 0
    plan = projected.get("plan")
    semantics = projected.get("execution_semantics")
    steps = plan.get("steps") if isinstance(plan, dict) else None
    bindings = semantics.get("audio_usage_bindings") if isinstance(semantics, dict) else None
    if not isinstance(steps, list) or not isinstance(bindings, list):
        return projected, 0

    steps_by_id = {
        str(step.get("id") or ""): step
        for step in steps
        if isinstance(step, dict) and str(step.get("id") or "")
    }
    projection_count = 0
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        step = steps_by_id.get(str(binding.get("plan_step_id") or ""))
        argument_path = str(binding.get("argument_path") or "")
        audio_ref = binding.get("audio_ref")
        if step is None or not isinstance(audio_ref, str) or "[" in argument_path:
            continue
        found, argument_value = _raw_step_argument_value(step, argument_path)
        if not found or not isinstance(argument_value, list):
            continue
        matching_indices = [
            index for index, value in enumerate(argument_value) if value == audio_ref
        ]
        if len(matching_indices) != 1:
            continue
        binding["argument_path"] = f"{argument_path}[{matching_indices[0]}]"
        projection_count += 1
    return projected, projection_count


def _project_state_transition_from_selected_base(
    payload: Mapping[str, Any],
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> tuple[dict[str, Any], int]:
    """Derive the redundant transition label from the selected graph base."""

    projected = copy.deepcopy(dict(payload))
    if projected.get("decision") != "execute" or planner_input.session_state is None:
        return projected, 0
    selected_node_id = str(projected.get("selected_base_state_node_id") or "")
    if not selected_node_id:
        return projected, 0
    plan = projected.get("plan")
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if selected_node_id == planner_input.session_state.current_node_id:
        expected = "edit_current"
    elif isinstance(steps, list) and steps:
        expected = "branch_from_history"
    else:
        expected = "restore_history"
    if projected.get("state_transition_kind") == expected:
        return projected, 0
    projected["state_transition_kind"] = expected
    return projected, 1


def _project_execution_base_audio_from_plan(
    payload: Mapping[str, Any],
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> tuple[dict[str, Any], int]:
    """Bind a prose-mode multi-audio state node to the PlanIR's actual edit input."""

    projected = copy.deepcopy(dict(payload))
    state = planner_input.session_state
    memo = planner_input.semantic_memo
    if (
        projected.get("decision") != "execute"
        or state is None
        or memo.requested_change_ledger_status != "provider_omitted"
        or memo.selected_base_state_node_id is not None
    ):
        return projected, 0
    selected_node_id = str(projected.get("selected_base_state_node_id") or "")
    plan = projected.get("plan")
    plan_input = plan.get("input_audio") if isinstance(plan, Mapping) else None
    if not selected_node_id or not isinstance(plan_input, str) or not plan_input:
        return projected, 0
    matching_nodes = [
        node
        for node in state.nodes
        if node.node_id == selected_node_id
        and node.session_eligible
        and node.contains_audio_ref(plan_input)
    ]
    bound_audio_refs = {
        *planner_input.source_audio_refs,
        *planner_input.reference_audio_refs,
    }
    if len(matching_nodes) != 1 or plan_input not in bound_audio_refs:
        return projected, 0
    if projected.get("selected_base_audio_ref") == plan_input:
        return projected, 0
    projected["selected_base_audio_ref"] = plan_input
    return projected, 1


def _validate_plan_topology(plan: Any) -> None:
    step_ids = [step.id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("plan step ids must be unique")
    seen: set[str] = set()
    consumed_step_ids: set[str] = set()
    for step in plan.steps:
        unknown = set(step.depends_on) - seen
        if unknown:
            raise ValueError(f"{step.id} depends on absent or later steps: {sorted(unknown)}")
        for value in _iter_step_output_references({"inputs": step.inputs, "params": step.params}):
            producer = str(value["from_step"])
            if producer not in seen:
                raise ValueError(f"{step.id} references absent, current, or later step {producer}")
            if producer not in step.depends_on:
                raise ValueError(
                    f"{step.id} input reference {producer} must also appear in depends_on"
                )
            consumed_step_ids.add(producer)
        seen.add(step.id)
    terminal_step_ids = [step_id for step_id in step_ids if step_id not in consumed_step_ids]
    output_artifacts = list(plan.output_artifacts)
    if not output_artifacts and len(terminal_step_ids) != 1:
        raise ValueError(
            "a single-output audio plan requires exactly one assembled terminal step; "
            f"found {terminal_step_ids}. Connect consumed outputs with typed "
            '{"from_step":"producer_id","from_output":"output_audio_path"} '
            "values and depends_on. A dotted string is not an output reference. "
            "If outputs are independent user-requested deliverables, declare output_artifacts."
        )
    if not output_artifacts:
        return
    artifact_ids = [item.artifact_id for item in output_artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("plan output artifact ids must be unique")
    declared_terminal_steps = {item.from_step for item in output_artifacts}
    if declared_terminal_steps != set(terminal_step_ids):
        raise ValueError(
            "multi-artifact delivery must declare every terminal step and only terminal steps; "
            f"declared={sorted(declared_terminal_steps)}, terminal={terminal_step_ids}"
        )
    primary = [item.artifact_id for item in output_artifacts if item.primary]
    if len(primary) != 1:
        raise ValueError(
            "multi-artifact delivery requires exactly one primary artifact for Session Graph continuation"
        )


def _validate_session_base(
    turn: MemoQualifiedPlannerTurn,
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> None:
    state = planner_input.session_state
    if state is None:
        if turn.selected_base_state_node_id is not None:
            raise ValueError("execution cannot select a Session Graph base when none was supplied")
        if turn.state_transition_kind != "edit_current":
            raise ValueError("execution without Session Graph state must use edit_current")
        return
    if turn.selected_base_state_node_id is None or turn.selected_base_audio_ref is None:
        raise ValueError("execution must select one supplied Session Graph base state")
    matching = [
        node
        for node in state.nodes
        if node.node_id == turn.selected_base_state_node_id
        and node.contains_audio_ref(turn.selected_base_audio_ref)
        and node.session_eligible
    ]
    if len(matching) != 1:
        raise ValueError("selected Session Graph node/audio pair is absent or ineligible")
    bound_audio_refs = {
        *planner_input.source_audio_refs,
        *planner_input.reference_audio_refs,
    }
    if turn.selected_base_audio_ref not in bound_audio_refs:
        raise ValueError("selected Session Graph audio ref is not a bound runtime audio ref")
    assert turn.plan is not None
    if turn.plan.input_audio != turn.selected_base_audio_ref:
        raise ValueError("plan.input_audio must equal the selected Session Graph audio ref")
    selected_is_current = turn.selected_base_state_node_id == state.current_node_id
    if turn.state_transition_kind == "edit_current" and not selected_is_current:
        raise ValueError("edit_current must select the current Session Graph node")
    if (
        turn.state_transition_kind in {"branch_from_history", "restore_history"}
        and selected_is_current
    ):
        raise ValueError(f"{turn.state_transition_kind} must select an eligible historical node")


def _validate_state_completion_base(
    turn: MemoQualifiedPlannerTurn,
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> None:
    """Require an exact, eligible graph state and an honest selection transition."""

    state = planner_input.session_state
    if state is None:
        raise ValueError("state-only completion requires supplied Session Graph state")
    matching = [
        node
        for node in state.nodes
        if node.node_id == turn.selected_base_state_node_id
        and node.audio_ref == turn.selected_base_audio_ref
        and node.session_eligible
    ]
    if len(matching) != 1:
        raise ValueError(
            "state-only completion must select an eligible Session Graph primary output"
        )
    if turn.selected_base_audio_ref not in planner_input.source_audio_refs:
        raise ValueError("state-only Session Graph audio ref is not a bound source audio ref")
    selected_is_current = turn.selected_base_state_node_id == state.current_node_id
    expected_kind = "select_existing" if selected_is_current else "restore_history"
    if turn.state_completion_kind != expected_kind:
        raise ValueError(
            "state-only completion kind must describe whether the selected node is current "
            "or historical"
        )


def _validate_prose_state_completion_authority(
    turn: MemoQualifiedPlannerTurn,
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> None:
    """Require Graph-derived authority before prose Planner can claim completion."""

    matching = [
        proof
        for proof in session_completion_proof_inventory(planner_input.session_state)
        if proof.selected_node_id == turn.selected_base_state_node_id
        and proof.selected_audio_ref == turn.selected_base_audio_ref
    ]
    if not matching:
        raise ValueError(
            "prose state-only completion requires an authoritative Session Graph "
            "completion proof for the selected node/audio pair"
        )


def _validate_reasoner_selected_base(
    turn: MemoQualifiedPlannerTurn,
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> None:
    """Planner may copy the Reasoner's Graph decision but cannot reinterpret it."""

    expected = (
        planner_input.semantic_memo.selected_base_state_node_id,
        planner_input.semantic_memo.selected_base_audio_ref,
    )
    if expected == (None, None):
        return
    actual = (
        turn.selected_base_state_node_id,
        turn.selected_base_audio_ref,
    )
    if actual != expected:
        raise ValueError("Planner Session Graph base must exactly match the Reasoner handoff")


def _validate_perception_base(
    turn: MemoQualifiedPlannerTurn,
    *,
    planner_input: MemoQualifiedPlannerInput,
) -> None:
    """Bind evidence collection to the state that later execution will edit."""

    state = planner_input.session_state
    if state is None:
        if turn.selected_base_state_node_id is not None:
            raise ValueError("Perception cannot select a Session Graph base when none was supplied")
        return
    if turn.selected_base_state_node_id is None or turn.selected_base_audio_ref is None:
        raise ValueError("Perception must select one supplied Session Graph base state")
    matching = [
        node
        for node in state.nodes
        if node.node_id == turn.selected_base_state_node_id
        and node.contains_audio_ref(turn.selected_base_audio_ref)
        and node.session_eligible
    ]
    if len(matching) != 1:
        raise ValueError(
            "selected Perception Session Graph node/audio pair is absent or ineligible"
        )
    bound_audio_refs = {
        *planner_input.source_audio_refs,
        *planner_input.reference_audio_refs,
    }
    if turn.selected_base_audio_ref not in bound_audio_refs:
        raise ValueError(
            "selected Perception Session Graph audio ref is not a bound runtime audio ref"
        )
    # A gateway may observe only a reference asset, only the selected base, or
    # both. The selected Session Graph base remains frozen independently from
    # each job's explicit audio namespace.


def _validate_parameter_declarations(step: Any, selection: Any, contract: Any) -> None:
    declared = {item.argument_path: item for item in selection.parameter_selections}
    selectable = {item.argument_path: item for item in contract.arguments}
    framework_owned = set(contract.framework_owned_arguments)
    authored_paths = {
        *{f"inputs.{key}" for key in step.inputs},
        *{f"params.{key}" for key in step.params},
    }
    legal_paths = set(selectable) | framework_owned
    unknown_authored = sorted(
        path
        for path in authored_paths - legal_paths
        if not _covered_by_declared_container(path, selectable, framework_owned)
    )
    if unknown_authored:
        raise ValueError(
            f"{step.id} authors arguments outside the supplied contract: {unknown_authored}"
        )
    for path in framework_owned:
        found, _ = _argument_value(step, path)
        if found:
            raise ValueError(f"{step.id} authors framework-owned argument {path}")
    for path, argument in selectable.items():
        found, value = _argument_value(step, path)
        if argument.required and not found:
            raise ValueError(f"{step.id} is missing required argument {path}")
        if found:
            _validate_value_schema(path, value, argument.value_schema)
            declaration = declared.get(path)
            if declaration is None or declaration.selected_value != value:
                raise ValueError(f"{step.id} must declare the exact selected value for {path}")
    extra = sorted(set(declared) - set(selectable))
    if extra:
        raise ValueError(f"{step.id} declares non-selectable arguments: {extra}")


def _covered_by_declared_container(
    path: str,
    selectable: Mapping[str, Any],
    framework_owned: set[str],
) -> bool:
    for container_path, argument in selectable.items():
        schema_type = argument.value_schema.get("type")
        if schema_type in {"object", "array"} and path.startswith(f"{container_path}."):
            return True
    return any(path.startswith(f"{container_path}.") for container_path in framework_owned)


def _validate_value_schema(path: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected_type = schema.get("type")
    if (
        path.startswith("inputs.")
        and expected_type == "string"
        and _is_step_output_reference(value)
    ):
        return
    valid = {
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
    }.get(str(expected_type))
    if valid is not None and not valid(value):
        raise ValueError(f"{path} does not satisfy type={expected_type}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is outside the declared enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the declared minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} exceeds the declared maximum")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} has fewer than the declared minimum items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has more than the declared maximum items")


def _argument_value(step: Any, path: str) -> tuple[bool, Any]:
    root, separator, remainder = path.partition(".")
    if not separator or root not in {"inputs", "params"}:
        return False, None
    current: Any = step.inputs if root == "inputs" else step.params
    for part in remainder.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        return False, None
    return True, current


def _prompt_fingerprint(
    system_prompt: str,
    current_prompt: str,
    *,
    response_schema: Mapping[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "current_prompt": current_prompt,
            "response_schema": response_schema,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clear_client_json_invocations(client: Any) -> None:
    if hasattr(client, "last_json_invocations"):
        client.last_json_invocations = []


__all__ = [
    "MemoQualifiedPlannerContractError",
    "MemoQualifiedPlannerError",
    "OpenAICompatibleMemoQualifiedPlanner",
    "bind_operation_contract_runtime_domains",
    "memo_qualified_planner_provider_response_format",
    "memo_qualified_planner_response_format",
    "project_operation_capability_contracts",
    "validate_memo_qualified_planner_turn",
]
