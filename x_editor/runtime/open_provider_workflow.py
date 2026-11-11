"""Benchmark-independent Open Provider workflow runtime assembly.

This module owns the shared GLM Planner/Reasoner, Qwen3-Omni Perception, and
strict forward-execution composition. Dataset adapters may select inputs and
serialize results, but they must invoke this runtime instead of rebuilding the
workflow themselves.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_serializer, model_validator

from x_editor.capabilities.catalog import (
    PRIOR_GUIDANCE_PAYLOAD_KEY,
    attach_tool_prior_guidance,
)
from x_editor.capabilities.preflight import require_runtime_capability_profile_ready
from x_editor.capabilities.priors import (
    ExperimentPriorBinding,
    FrozenToolPriorSnapshot,
    ToolPriorSnapshot,
    load_tool_prior_snapshot,
)
from x_editor.capabilities.profiles import load_runtime_capability_profile_contract
from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    OPEN_GLM_QWEN_OMNI_MODE,
    QWEN3_OMNI_PROVIDER,
)
from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.provider_requests import ProviderSchedulingRuntime
from x_editor.core.schemas import StrictModel
from x_editor.frontend import OpenAICompatibleStrongOmniAudioFrontend
from x_editor.perception.speaker_attributed_review import (
    OpenAICompatibleSpeakerReviewFrontend,
)
from x_editor.perception.speech import SpeechPerceptionClient
from x_editor.perception.timestamped_caption import (
    OpenAICompatibleTimestampedCaptionFrontend,
)
from x_editor.perception.workflow_catalog import (
    workflow_perception_catalog_payload,
)
from x_editor.planner.evidence_index import EvidenceIndexFormat
from x_editor.planner.argument_table import ArgumentContractFormat
from x_editor.planner.retry_context import RetryGuidanceMode
from x_editor.reasoner.retry_context import ReasonerRetryContextMode
from x_editor.planner.memo_qualified import OpenAICompatibleMemoQualifiedPlanner
from x_editor.planner.openai_compatible_planner import GLMOpenAICompatiblePlanner
from x_editor.planner.prompt_package import (
    DEFAULT_PLANNER_PROMPT_PROFILE,
    PlannerPromptProfileName,
    compact_semantic_effect_capability_index,
    project_memo_qualified_candidate_operation_contracts,
)
from x_editor.reasoner.contracts import ReasonerShadowStrategy
from x_editor.reasoner.evidence_memo import OpenAICompatibleEvidenceMemoReasoner, OpenAICompatibleDraftMemoReasoner
from x_editor.reasoner.openai_compatible import (
    OpenAICompatibleMemoOnlySemanticReasoner,
    OpenAICompatibleMemoSemanticReasoner,
    OpenAICompatibleProseMemoSemanticReasoner,
    OpenAICompatibleSemanticReasoner,
)
from x_editor.reasoner.shadow import ReasonerShadowRunner
from x_editor.runtime.audio_asset_facts import measure_case_audio_facts
from x_editor.runtime.audio_plan_executor import AudioEditExecutor
from x_editor.runtime.forward_execution import ForwardExecutionPath
from x_editor.runtime.module_checkpoints import (
    CheckpointRunContext,
    ModuleCheckpointObserver,
)
from x_editor.runtime.provider_policy import (
    FrozenProviderPolicy,
    build_frozen_openai_client,
)
from x_editor.runtime.separated_semantic_workflow_adapter import (
    SeparatedSemanticWorkflowPlannerAdapter,
)
from x_editor.runtime.workflow_adapters import (
    AudioEditExecutorPerceptionAdapter,
    ReviewedWorkflowEvidenceProjector,
    TraceableAudioEditPlannerAdapter,
)
from x_editor.runtime.workflow_case import (
    RuntimeAudioBinding,
    WorkflowRuntimeCase,
    WorkflowRuntimeScenario,
)
from x_editor.runtime.workflow_orchestration import (
    WorkflowArm,
    WorkflowArmOrchestrator,
    WorkflowArmPolicy,
    WorkflowArmRequest,
    WorkflowArmResult,
    WorkflowPerceptionRequest,
)
from x_editor.runtime.workflow_protocol import (
    DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME,
    WorkflowProtocolPolicy,
    WorkflowProtocolProfileName,
    resolve_workflow_protocol_policy,
)
from x_editor.runtime.workflow_session import (
    WorkflowScenarioResult,
    arun_workflow_scenario,
)
from x_editor.tools.executor import ToolExecutor

if TYPE_CHECKING:
    from x_editor.runtime.workflow_continuation import WorkflowContinuation

RuntimeCapabilityProfileName = Literal[
    "full_catalog",
    "lexical_local_v1",
    "speech_edit_core_v1",
    "speech_edit_core_v2",
    "custom",
]
DeployableRuntimeCapabilityProfileName = Literal[
    "full_catalog",
    "lexical_local_v1",
    "speech_edit_core_v1",
    "speech_edit_core_v2",
]

DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS = (
    "perception.speech_context",
    "perception.speaker_attributed_review",
    "perception.semantic_span_review",
    "perception.audio_fallback",
    "perception.timestamped_caption",
    "perception.flexsed_events",
)

MODEL_BACKED_PERCEPTION_OPERATIONS = frozenset(
    {
        "perception.audio_fallback",
        "perception.semantic_span_review",
        "perception.speaker_attributed_review",
        "perception.timestamped_caption",
    }
)

LEXICAL_LOCAL_V1_PERCEPTION_OPERATIONS = (
    "perception.audio_fallback",
    "perception.speech_context",
    "perception.semantic_span_review",
)

LEXICAL_LOCAL_V1_EDIT_OPERATIONS = (
    "artifact_ops.extract_spans",
    "artifact_ops.remove_spans",
    "artifact_ops.mute_spans",
    "artifact_ops.beep_spans",
    "artifact_ops.silence",
    "artifact_ops.concat",
    "artifact_ops.reorder_spans",
    "artifact_ops.reconstruct",
)

# Frozen benchmark-neutral speech-editing closure.  The operations are existing
# runtime adapters; this profile only makes their already implemented contracts
# visible to Planner and keeps unrelated music/unsupported routes out.
SPEECH_EDIT_CORE_V1_EDIT_OPERATIONS = (
    "artifact_ops.beep_spans",
    "artifact_ops.concat",
    "artifact_ops.extract_spans",
    "artifact_ops.mute_spans",
    "artifact_ops.reconstruct",
    "artifact_ops.remove_spans",
    "artifact_ops.reorder_spans",
    "artifact_ops.silence",
    "artifact_ops.speaker_scoped_edit",
    "audio_effects",
    "edit_data_pipeline.audio_assembly",
    "edit_data_pipeline.boundary_smooth",
    "edit_data_pipeline.emotion_style_edit",
    "edit_data_pipeline.pause_edit",
    "edit_data_pipeline.pitch_rate_edit",
    "edit_data_pipeline.pitch_shift",
    "edit_data_pipeline.speech_rate",
    "edit_data_pipeline.text_edit",
    "edit_data_pipeline.volume_normalize",
    "emphasis_stress_edit",
    "intra_audio_voice_transfer",
    "no_reference_voice_conversion",
    "qwen3_tts",
    "seed_vc",
    "spatial_reverb",
)

_SPEECH_EDIT_CORE_V2_CONTRACT = load_runtime_capability_profile_contract("speech_edit_core_v2")
SPEECH_EDIT_CORE_V2_EDIT_OPERATIONS = (
    *SPEECH_EDIT_CORE_V1_EDIT_OPERATIONS,
    *_SPEECH_EDIT_CORE_V2_CONTRACT.operation_ids,
)

DEFAULT_DEPLOYABLE_RUNTIME_CAPABILITY_PROFILE_NAME: DeployableRuntimeCapabilityProfileName = (
    "speech_edit_core_v1"
)
DEFAULT_SEMANTIC_REASONER_INITIAL_MEMO_ENABLE_THINKING = False
DEFAULT_SEMANTIC_REASONER_INITIAL_MEMO_MAX_OUTPUT_TOKENS = 8192
DEFAULT_SEMANTIC_REASONER_EVIDENCE_REVIEW_MAX_OUTPUT_TOKENS = 4096

_PERCEPTION_PREFIXES = ("perception.", "flexsed.")
_NON_EDIT_OPERATIONS = {"export_wav"}
_REVIEWED_SPAN_REQUIRED_OPERATIONS = {
    "artifact_ops.extract_spans",
    "artifact_ops.remove_spans",
    "artifact_ops.mute_spans",
    "artifact_ops.beep_spans",
    "artifact_ops.speaker_scoped_edit",
}


class OpenProviderPerceptionRoutePolicy(StrictModel):
    """Frozen implementation route and accounting budget for one skill."""

    schema_version: Literal["x-editor.open-provider-perception-route-policy/v1"] = (
        "x-editor.open-provider-perception-route-policy/v1"
    )
    operation: str
    route_id: str
    max_provider_calls: int = Field(ge=0, le=20)
    max_service_calls: int = Field(ge=0, le=20)
    max_tool_calls: int = Field(ge=0, le=20)
    cache_mode: Literal["disabled", "refresh"] = "disabled"
    fallback_mode: Literal["disabled"] = "disabled"


class OpenProviderReasonerShadowPolicy(StrictModel):
    """Frozen, non-authoritative Reasoner observation policy."""

    schema_version: Literal["x-editor.open-provider-reasoner-shadow-policy/v1"] = (
        "x-editor.open-provider-reasoner-shadow-policy/v1"
    )
    mode: Literal["disabled", "observe"] = "disabled"
    strategy: ReasonerShadowStrategy = "direct_json"
    provider: FrozenProviderPolicy | None = None
    calls_per_planner_gateway: Literal[1, 2] = 1
    planner_input_authority: Literal["none"] = "none"
    execution_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_binding(self) -> OpenProviderReasonerShadowPolicy:
        if self.mode == "disabled" and self.provider is not None:
            raise ValueError("disabled Reasoner shadow cannot retain a provider")
        if self.mode == "observe":
            if self.provider is None:
                raise ValueError("observe Reasoner shadow requires a provider")
            if self.provider.component != "reasoner":
                raise ValueError("Reasoner shadow provider must use component=reasoner")
            expected_calls = 2 if self.strategy == "memo_then_compile" else 1
            if self.calls_per_planner_gateway != expected_calls:
                raise ValueError(
                    f"{self.strategy} requires calls_per_planner_gateway={expected_calls}"
                )
            expected_thinking = self.strategy in {
                "private_thinking_json",
                "memo_only",
            }
            configured_thinking = self.provider.chat_template_kwargs.get("enable_thinking")
            invalid_thinking = (
                configured_thinking is not True
                if expected_thinking
                else configured_thinking is True
            )
            if invalid_thinking:
                raise ValueError(
                    "Reasoner shadow provider thinking configuration is "
                    f"incompatible with {self.strategy}; expected "
                    f"enable_thinking={expected_thinking}"
                )
        return self


class OpenProviderSemanticPlanningPolicy(StrictModel):
    """Frozen Reasoner-to-Planner path and phase-specific provider budgets."""

    schema_version: Literal["x-editor.open-provider-semantic-planning-policy/v2"] = (
        "x-editor.open-provider-semantic-planning-policy/v2"
    )
    mode: Literal["legacy_combined", "separated_memo", "integrated_plan", "advisory_plan"] = "legacy_combined"
    integrated_reasoning: Literal["summary", "direct"] = "summary"
    adviser_input_scope: Literal["module_projection", "common_planner_input"] = "module_projection"
    action_representation: Literal["planir", "graph", "actions", "command"] = "planir"
    action_abstraction_version: Literal["legacy_syntax", "hierarchical_v1"] = "legacy_syntax"
    reasoner_strategy: Literal["memo_only", "memo_only_brief", "prose_memo", "evidence_memo", "draft_memo"] = "prose_memo"
    planner_evidence_index_format: EvidenceIndexFormat = "expanded"
    planner_argument_contract_format: ArgumentContractFormat = "expanded"
    planner_retry_guidance_mode: RetryGuidanceMode = "diagnostic"
    reasoner_retry_context_mode: ReasonerRetryContextMode = "prior_memo"
    reasoner: FrozenProviderPolicy | None = None
    source_observation_bootstrap: Literal[
        "disabled",
        "speech_context_per_audio",
        "speech_and_scene_per_audio",
    ] = "disabled"
    evidence_review_max_output_tokens: int | None = Field(
        default=None,
        ge=512,
        le=32768,
    )

    @model_serializer(mode="wrap")
    def serialize_context_options(self, handler):
        # Disabled experiments must not invalidate existing frozen profiles or
        # shared-first caches merely because new optional controls were added.
        payload = handler(self)
        if self.adviser_input_scope == "module_projection":
            payload.pop("adviser_input_scope", None)
        if self.action_representation == "planir":
            payload.pop("action_representation", None)
        if self.action_abstraction_version == "legacy_syntax":
            payload.pop("action_abstraction_version", None)
        if self.mode != "integrated_plan":
            payload.pop("integrated_reasoning", None)
        if self.planner_argument_contract_format == "expanded":
            payload.pop("planner_argument_contract_format", None)
        if self.planner_retry_guidance_mode == "diagnostic":
            payload.pop("planner_retry_guidance_mode", None)
        if self.reasoner_retry_context_mode == "prior_memo":
            payload.pop("reasoner_retry_context_mode", None)
        return payload

    @model_validator(mode="after")
    def validate_binding(self) -> OpenProviderSemanticPlanningPolicy:
        if self.action_representation == "command" and self.action_abstraction_version != "hierarchical_v1":
            raise ValueError("Command requires the public hierarchical_v1 tool API")
        if self.action_abstraction_version != "legacy_syntax" and self.mode != "advisory_plan":
            raise ValueError("hierarchical action abstraction requires common advisory Planner")
        if self.action_representation != "planir" and self.mode != "advisory_plan":
            raise ValueError("alternate action representation requires common advisory Planner")
        if self.adviser_input_scope != "module_projection" and self.mode != "advisory_plan":
            raise ValueError("common adviser input requires advisory Planner")
        if self.mode != "integrated_plan" and self.integrated_reasoning != "summary":
            raise ValueError("integrated_reasoning requires integrated_plan mode")
        if self.mode == "integrated_plan":
            if self.reasoner is not None or self.evidence_review_max_output_tokens is not None:
                raise ValueError("integrated planning has no separate Reasoner provider or budget")
            if self.reasoner_retry_context_mode != "prior_memo":
                raise ValueError("integrated planning cannot configure a separate Reasoner retry")
            return self
        if self.mode == "advisory_plan" and self.reasoner is None:
            if self.evidence_review_max_output_tokens is not None:
                raise ValueError("unguided Planner cannot retain an adviser output budget")
            return self
        if self.mode == "legacy_combined":
            if self.planner_evidence_index_format != "expanded":
                raise ValueError("evidence index encoding requires separated memo planning")
            if self.planner_argument_contract_format != "expanded" or self.planner_retry_guidance_mode != "diagnostic":
                raise ValueError("Planner context options require separated memo planning")
            if self.reasoner_retry_context_mode != "prior_memo":
                raise ValueError("Reasoner retry context requires separated memo planning")
            if self.reasoner is not None:
                raise ValueError("legacy combined planning cannot retain a separated Reasoner")
            if self.evidence_review_max_output_tokens is not None:
                raise ValueError("legacy combined planning cannot retain an evidence-review budget")
            if self.source_observation_bootstrap != "disabled":
                raise ValueError(
                    "legacy combined planning cannot enable semantic source observation"
                )
            return self
        if self.reasoner is None or self.reasoner.component != "reasoner":
            raise ValueError("separated memo planning requires component=reasoner")
        if self.evidence_review_max_output_tokens is None:
            raise ValueError("separated memo planning requires an evidence-review budget")
        if self.evidence_review_max_output_tokens > int(self.reasoner.max_output_tokens):
            raise ValueError(
                "evidence-review output budget cannot exceed the Reasoner output budget"
            )
        configured_thinking = self.reasoner.chat_template_kwargs.get("enable_thinking")
        if configured_thinking is not True and configured_thinking is not False:
            raise ValueError(
                "separated Reasoner must explicitly freeze enable_thinking independently "
                "of its memo strategy"
            )
        return self


class OpenProviderWorkflowProfile(StrictModel):
    """Frozen policy for the shared Open Provider workflow runtime."""

    schema_version: Literal["x-editor.open-provider-workflow-profile/v1"] = (
        "x-editor.open-provider-workflow-profile/v1"
    )
    planner: FrozenProviderPolicy
    perception: FrozenProviderPolicy
    perception_operation_providers: dict[str, FrozenProviderPolicy] = Field(
        default_factory=dict
    )
    reasoner_shadow: OpenProviderReasonerShadowPolicy = Field(
        default_factory=OpenProviderReasonerShadowPolicy
    )
    semantic_planning: OpenProviderSemanticPlanningPolicy = Field(
        default_factory=OpenProviderSemanticPlanningPolicy
    )
    runtime_capability_profile: RuntimeCapabilityProfileName = (
        DEFAULT_DEPLOYABLE_RUNTIME_CAPABILITY_PROFILE_NAME
    )
    allowed_perception_operations: list[str] = Field(min_length=1)
    w2_general_perception_operation: str = "perception.audio_fallback"
    perception_operation_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    perception_route_policies: list[OpenProviderPerceptionRoutePolicy] = Field(min_length=1)
    allowed_edit_operations: list[str] = Field(min_length=1)
    edit_operation_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_prior_snapshot: ToolPriorSnapshot | None = None
    tool_prior_binding: ExperimentPriorBinding | None = None
    evidence_required_operations: list[str] = Field(default_factory=list)
    workflow_protocol_profile: WorkflowProtocolProfileName = DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME
    planner_prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE
    max_perception_review_attempts: Literal[1] = 1
    targeted_perception_rounds: int = Field(
        default=1, ge=0, le=8,
        description="Additional perception gateway rounds after the unchanged source bootstrap; zero permits one planning decision without additional observation.",
    )
    evaluator_mode: Literal["disabled"] = "disabled"
    repair_mode: Literal["disabled"] = "disabled"
    fallback_mode: Literal["disabled"] = "disabled"
    executor_llm_mode: Literal["disabled"] = "disabled"

    @model_validator(mode="after")
    def validate_profile(self) -> OpenProviderWorkflowProfile:
        if self.planner.component != "planner":
            raise ValueError("planner policy must describe the planner component")
        if self.perception.component != "perception":
            raise ValueError("perception policy must describe the perception component")
        unknown_provider_operations = sorted(
            set(self.perception_operation_providers)
            - MODEL_BACKED_PERCEPTION_OPERATIONS
        )
        if unknown_provider_operations:
            raise ValueError(
                "perception_operation_providers may only bind model-backed "
                f"perception operations: {unknown_provider_operations}"
            )
        unavailable_provider_operations = sorted(
            set(self.perception_operation_providers)
            - set(self.allowed_perception_operations)
        )
        if unavailable_provider_operations:
            raise ValueError(
                "perception_operation_providers contain operations outside the "
                f"frozen catalog: {unavailable_provider_operations}"
            )
        invalid_provider_components = sorted(
            operation
            for operation, policy in self.perception_operation_providers.items()
            if policy.component != "perception"
        )
        if invalid_provider_components:
            raise ValueError(
                "perception operation providers must use component=perception: "
                f"{invalid_provider_components}"
            )
        if (
            self.semantic_planning.mode in {"separated_memo", "integrated_plan", "advisory_plan"}
            and self.reasoner_shadow.mode != "disabled"
        ):
            raise ValueError(
                "authoritative separated Reasoner and post-run Reasoner shadow cannot "
                "be enabled in the same profile"
            )
        if (
            self.semantic_planning.source_observation_bootstrap in {"speech_context_per_audio", "speech_and_scene_per_audio"}
            and "perception.speech_context" not in self.allowed_perception_operations
        ):
            raise ValueError("speech-context semantic bootstrap requires perception.speech_context")
        if (
            self.semantic_planning.source_observation_bootstrap == "speech_and_scene_per_audio"
            and "perception.audio_fallback" not in self.allowed_perception_operations
        ):
            raise ValueError("scene bootstrap requires perception.audio_fallback")
        if self.runtime_capability_profile == "full_catalog":
            expected_perception = set(DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS)
            expected_edits = set(default_edit_operations())
            if set(self.allowed_perception_operations) != expected_perception:
                raise ValueError("full_catalog must expose the complete perception catalog")
            if set(self.allowed_edit_operations) != expected_edits:
                raise ValueError("full_catalog must expose the complete edit catalog")
        elif self.runtime_capability_profile == "lexical_local_v1":
            if tuple(self.allowed_perception_operations) != (
                LEXICAL_LOCAL_V1_PERCEPTION_OPERATIONS
            ):
                raise ValueError(
                    "lexical_local_v1 perception catalog differs from its frozen contract"
                )
            if tuple(self.allowed_edit_operations) != (LEXICAL_LOCAL_V1_EDIT_OPERATIONS):
                raise ValueError("lexical_local_v1 edit catalog differs from its frozen contract")
        elif self.runtime_capability_profile == "speech_edit_core_v1":
            if tuple(self.allowed_perception_operations) != (
                DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS
            ):
                raise ValueError(
                    "speech_edit_core_v1 perception catalog differs from its frozen contract"
                )
            if tuple(self.allowed_edit_operations) != (SPEECH_EDIT_CORE_V1_EDIT_OPERATIONS):
                raise ValueError(
                    "speech_edit_core_v1 edit catalog differs from its frozen contract"
                )
        elif self.runtime_capability_profile == "speech_edit_core_v2":
            if tuple(self.allowed_perception_operations) != (
                DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS
            ):
                raise ValueError(
                    "speech_edit_core_v2 perception catalog differs from its frozen contract"
                )
            if tuple(self.allowed_edit_operations) != (SPEECH_EDIT_CORE_V2_EDIT_OPERATIONS):
                raise ValueError(
                    "speech_edit_core_v2 edit catalog differs from its frozen contract"
                )
        if len(self.allowed_edit_operations) != len(set(self.allowed_edit_operations)):
            raise ValueError("allowed_edit_operations must be unique")
        if len(self.allowed_perception_operations) != len(set(self.allowed_perception_operations)):
            raise ValueError("allowed_perception_operations must be unique")
        if self.w2_general_perception_operation not in set(self.allowed_perception_operations):
            raise ValueError(
                "w2_general_perception_operation must be in the frozen perception catalog"
            )
        if any(
            not operation.startswith(_PERCEPTION_PREFIXES)
            for operation in self.allowed_perception_operations
        ):
            raise ValueError(
                "allowed_perception_operations must contain only perception operations"
            )
        unknown_param_operations = sorted(
            set(self.perception_operation_params) - set(self.allowed_perception_operations)
        )
        if unknown_param_operations:
            raise ValueError(
                "perception_operation_params contain operations outside the "
                f"frozen catalog: {unknown_param_operations}"
            )
        route_operations = [policy.operation for policy in self.perception_route_policies]
        if len(route_operations) != len(set(route_operations)):
            raise ValueError("perception route operations must be unique")
        if set(route_operations) != set(self.allowed_perception_operations):
            raise ValueError("perception routes must cover the frozen perception catalog exactly")
        route_ids = [policy.route_id for policy in self.perception_route_policies]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("perception route_id values must be unique")
        if not set(self.evidence_required_operations).issubset(set(self.allowed_edit_operations)):
            raise ValueError("evidence_required_operations must be a subset of allowed edits")
        unknown_edit_param_operations = sorted(
            set(self.edit_operation_params) - set(self.allowed_edit_operations)
        )
        if unknown_edit_param_operations:
            raise ValueError(
                "edit_operation_params contain operations outside the frozen "
                f"catalog: {unknown_edit_param_operations}"
            )
        if (self.tool_prior_snapshot is None) != (self.tool_prior_binding is None):
            raise ValueError("tool_prior_snapshot and tool_prior_binding must be supplied together")
        if self.tool_prior_snapshot is not None:
            frozen_prior = FrozenToolPriorSnapshot(
                snapshot=self.tool_prior_snapshot,
                binding=self.tool_prior_binding,
            )
            prior_operations = {
                operation
                for guidance in frozen_prior.snapshot.guidance
                for route in guidance.routes
                for operation in route.operations
            }
            unknown_prior_operations = sorted(prior_operations - set(self.allowed_edit_operations))
            if unknown_prior_operations:
                raise ValueError(
                    "tool prior references operations outside the frozen edit "
                    f"catalog: {unknown_prior_operations}"
                )
        protocol = self.protocol_policy()
        expected_evidence_operations = (
            set(self.allowed_edit_operations)
            if protocol.evidence_requirement_scope == "all_edits"
            else set(default_evidence_required_operations(self.allowed_edit_operations))
        )
        if set(self.evidence_required_operations) != expected_evidence_operations:
            raise ValueError(
                "evidence_required_operations do not match the selected workflow protocol profile"
            )
        return self

    def planner_tool_prior_guidance(self) -> dict[str, Any] | None:
        """Return Planner-only guidance from an explicitly frozen snapshot."""

        if self.tool_prior_snapshot is None or self.tool_prior_binding is None:
            return None
        frozen_prior = FrozenToolPriorSnapshot(
            snapshot=self.tool_prior_snapshot,
            binding=self.tool_prior_binding,
        )
        return attach_tool_prior_guidance({}, frozen_prior=frozen_prior)[PRIOR_GUIDANCE_PAYLOAD_KEY]

    def workflow_policy(self) -> WorkflowArmPolicy:
        max_perception_provider_calls = max(
            policy.max_provider_calls for policy in self.perception_route_policies
        )
        return WorkflowArmPolicy(
            w1_planner_call_limit=1 + self.targeted_perception_rounds,
            w1_perception_call_limit=self.targeted_perception_rounds,
            component_invocation_trace_mode="required",
            evidence_projection_mode="reviewed",
            planner_provider_call_limit_per_gateway=1,
            reasoner_provider_call_limit_per_gateway=1,
            perception_provider_call_limit_per_gateway=(max_perception_provider_calls),
        )

    def perception_params(self, operation: str) -> dict[str, Any]:
        protocol = self.protocol_policy()
        params = dict(self.perception_operation_params.get(operation) or {})
        if operation == "perception.timestamped_caption":
            params.update(
                {
                    "max_review_attempts": self.max_perception_review_attempts,
                    "allow_slice_completion": False,
                    "slice_shortfall_mode": protocol.slice_shortfall_mode,
                }
            )
        elif operation == "perception.speaker_attributed_review":
            params["max_review_attempts"] = self.max_perception_review_attempts
        return params

    def perception_provider(self, operation: str) -> FrozenProviderPolicy:
        """Return the frozen StrongOmni provider for one model-backed route."""

        if operation not in MODEL_BACKED_PERCEPTION_OPERATIONS:
            raise KeyError(f"perception operation is not model-backed: {operation}")
        return self.perception_operation_providers.get(operation, self.perception)

    def perception_catalog_payload(self) -> list[dict[str, Any]]:
        return workflow_perception_catalog_payload(self.allowed_perception_operations)

    def perception_route_policy(
        self,
        operation: str,
    ) -> OpenProviderPerceptionRoutePolicy:
        for policy in self.perception_route_policies:
            if policy.operation == operation:
                return policy
        raise KeyError(f"missing perception route policy for {operation}")

    def protocol_policy(self) -> WorkflowProtocolPolicy:
        return resolve_workflow_protocol_policy(self.workflow_protocol_profile)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def runtime_capability_profile_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "schema_version": "x-editor.runtime-capability-profile/v1",
                "profile_name": self.runtime_capability_profile,
                "allowed_perception_operations": (self.allowed_perception_operations),
                "w2_general_perception_operation": (self.w2_general_perception_operation),
                "allowed_edit_operations": self.allowed_edit_operations,
                "evidence_required_operations": (self.evidence_required_operations),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpenProviderWorkflowRuntime:
    """Assembled workflow shared by interactive and benchmark callers."""

    profile: OpenProviderWorkflowProfile
    orchestrator: WorkflowArmOrchestrator
    planner_adapter: Any
    perception_adapter: AudioEditExecutorPerceptionAdapter
    forward_path: ForwardExecutionPath
    reasoner_shadow_runner: ReasonerShadowRunner | None = None
    provider_scheduling: ProviderSchedulingRuntime | None = None

    def planner_visible_capability_snapshot(self) -> dict[str, Any]:
        """Return the frozen capability contracts available to this Planner.

        The snapshot is benchmark-neutral and contains no outcome, oracle, or
        scorer data.  Evaluation uses it to judge a Planner decision against
        the same operation and Perception boundaries that were available at
        execution time, instead of inferring capability from a failure message.
        """

        candidate_contracts = getattr(
            self.planner_adapter,
            "candidate_operation_contracts",
            (),
        )
        if candidate_contracts:
            edit_contracts_by_phase = {
                phase: project_memo_qualified_candidate_operation_contracts(
                    list(candidate_contracts),
                    phase=phase,
                )
                for phase in ("evidence_planning", "execution_planning")
            }
        else:
            raw_cards = getattr(
                self.planner_adapter,
                "edit_operation_capability_cards",
                {},
            )
            edit_contracts = [
                copy.deepcopy(raw_cards[operation])
                for operation in sorted(raw_cards)
            ]
            edit_contracts_by_phase = {
                "evidence_planning": copy.deepcopy(edit_contracts),
                "execution_planning": copy.deepcopy(edit_contracts),
            }

        perception_catalog = getattr(
            self.planner_adapter,
            "perception_capability_catalog",
            self.profile.perception_catalog_payload(),
        )
        route_policies = {
            policy.operation: policy.model_dump(mode="json")
            for policy in self.profile.perception_route_policies
        }
        perception_contracts = []
        for raw in perception_catalog:
            capability = copy.deepcopy(dict(raw))
            operation = str(capability.get("operation") or "")
            if not operation:
                continue
            perception_contracts.append(
                {
                    "operation": operation,
                    "capability": capability,
                    "route_policy": copy.deepcopy(route_policies.get(operation)),
                    "frozen_params": self.profile.perception_params(operation),
                }
            )

        return {
            "schema_version": "x-editor.planner-visible-capability-snapshot/v1",
            "runtime_capability_profile": self.profile.runtime_capability_profile,
            "runtime_capability_profile_fingerprint": (
                self.profile.runtime_capability_profile_fingerprint
            ),
            "edit_operation_contracts_by_phase": edit_contracts_by_phase,
            "perception_operation_contracts": perception_contracts,
            "empirical_tool_prior_included": False,
            "benchmark_identity_included": False,
        }

    async def arun(
        self,
        *,
        arm: WorkflowArm,
        user_request: str,
        source_audio_refs: list[str],
        output_dir: str | Path,
        event_log_path: str | Path | None = None,
    ) -> WorkflowArmResult:
        return await self.arun_case(
            WorkflowRuntimeCase(
                user_request=user_request,
                source_audio_refs=source_audio_refs,
            ),
            arm=arm,
            output_dir=output_dir,
            event_log_path=event_log_path,
        )

    async def arun_case(
        self,
        case: WorkflowRuntimeCase,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        event_log_path: str | Path | None = None,
        turn_index: int | None = None,
    ) -> WorkflowArmResult:
        case = _as_opaque_runtime_case(case)
        request = WorkflowArmRequest(
            arm=arm,
            user_request=case.user_request,
            source_audio_refs=case.source_audio_refs,
            reference_audio_refs=case.reference_audio_refs,
            audio_resolution_mode="opaque_ref_v1",
            audio_bindings=[item.model_copy(deep=True) for item in case.audio_bindings],
            runtime_output_dir=str(output_dir),
            source_text=case.source_text,
            prior_turn_contexts=[item.model_copy(deep=True) for item in case.prior_turn_contexts],
            session_state=(
                case.session_state.model_copy(deep=True) if case.session_state is not None else None
            ),
            historical_reviewed_evidence=(
                case.historical_reviewed_evidence.model_copy(deep=True)
                if case.historical_reviewed_evidence is not None
                else None
            ),
            future_attempt_guidance=(
                case.future_attempt_guidance.model_copy(deep=True)
                if case.future_attempt_guidance is not None
                else None
            ),
            planning_context=copy.deepcopy(case.planning_context),
            audio_asset_facts=measure_case_audio_facts(
                case,
                gain_headroom_db=self.profile.edit_operation_params.get(
                    "artifact_ops.apply_gain", {}
                ).get("headroom_db"),
            ),
            planner_prompt_profile=self.profile.planner_prompt_profile,
            bootstrap_perception_requests=(
                self._semantic_bootstrap_perception_requests(case) if arm != "W2" else []
            ),
            initial_perception_request=(
                self._w2_general_perception_request(case) if arm == "W2" else None
            ),
            event_log_path=(str(event_log_path) if event_log_path is not None else None),
            policy=self.profile.workflow_policy(),
        )
        checkpoint_observer = ModuleCheckpointObserver(
            path=Path(output_dir) / "module_checkpoints.jsonl",
            context=CheckpointRunContext(
                workflow_run_id=request.run_id,
                assigned_arm=arm,
                runtime_case_fingerprint=case.fingerprint,
                turn_index=turn_index,
            ),
        )
        if isinstance(self.orchestrator, WorkflowArmOrchestrator):
            result = await self.orchestrator.arun(
                request,
                observer=checkpoint_observer,
            )
        else:
            result = await self.orchestrator.arun(request)
        return self._observe_reasoner_shadow(
            result,
            output_dir=output_dir,
            checkpoint_observer=checkpoint_observer,
        )

    def run(
        self,
        *,
        arm: WorkflowArm,
        user_request: str,
        source_audio_refs: list[str],
        output_dir: str | Path,
        event_log_path: str | Path | None = None,
    ) -> WorkflowArmResult:
        return self.run_case(
            WorkflowRuntimeCase(
                user_request=user_request,
                source_audio_refs=source_audio_refs,
            ),
            arm=arm,
            output_dir=output_dir,
            event_log_path=event_log_path,
        )

    def run_case(
        self,
        case: WorkflowRuntimeCase,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        event_log_path: str | Path | None = None,
        turn_index: int | None = None,
    ) -> WorkflowArmResult:
        case = _as_opaque_runtime_case(case)
        request = WorkflowArmRequest(
            arm=arm,
            user_request=case.user_request,
            source_audio_refs=case.source_audio_refs,
            reference_audio_refs=case.reference_audio_refs,
            audio_resolution_mode="opaque_ref_v1",
            audio_bindings=[item.model_copy(deep=True) for item in case.audio_bindings],
            runtime_output_dir=str(output_dir),
            source_text=case.source_text,
            prior_turn_contexts=[item.model_copy(deep=True) for item in case.prior_turn_contexts],
            session_state=(
                case.session_state.model_copy(deep=True) if case.session_state is not None else None
            ),
            historical_reviewed_evidence=(
                case.historical_reviewed_evidence.model_copy(deep=True)
                if case.historical_reviewed_evidence is not None
                else None
            ),
            future_attempt_guidance=(
                case.future_attempt_guidance.model_copy(deep=True)
                if case.future_attempt_guidance is not None
                else None
            ),
            planning_context=copy.deepcopy(case.planning_context),
            audio_asset_facts=measure_case_audio_facts(
                case,
                gain_headroom_db=self.profile.edit_operation_params.get(
                    "artifact_ops.apply_gain", {}
                ).get("headroom_db"),
            ),
            planner_prompt_profile=self.profile.planner_prompt_profile,
            bootstrap_perception_requests=(
                self._semantic_bootstrap_perception_requests(case) if arm != "W2" else []
            ),
            initial_perception_request=(
                self._w2_general_perception_request(case) if arm == "W2" else None
            ),
            event_log_path=(str(event_log_path) if event_log_path is not None else None),
            policy=self.profile.workflow_policy(),
        )
        checkpoint_observer = ModuleCheckpointObserver(
            path=Path(output_dir) / "module_checkpoints.jsonl",
            context=CheckpointRunContext(
                workflow_run_id=request.run_id,
                assigned_arm=arm,
                runtime_case_fingerprint=case.fingerprint,
                turn_index=turn_index,
            ),
        )
        if isinstance(self.orchestrator, WorkflowArmOrchestrator):
            result = self.orchestrator.run(
                request,
                observer=checkpoint_observer,
            )
        else:
            result = self.orchestrator.run(request)
        return self._observe_reasoner_shadow(
            result,
            output_dir=output_dir,
            checkpoint_observer=checkpoint_observer,
        )

    def _observe_reasoner_shadow(
        self,
        result: WorkflowArmResult,
        *,
        output_dir: str | Path,
        checkpoint_observer: ModuleCheckpointObserver,
    ) -> WorkflowArmResult:
        """Run post-terminal observation without changing authoritative fields."""

        if self.reasoner_shadow_runner is None:
            return result
        try:
            shadow = self.reasoner_shadow_runner.run(
                [item.model_copy(deep=True) for item in result.reasoner_shadow_inputs],
                workflow_run_id=result.run_id,
                output_dir=output_dir,
                observer=checkpoint_observer,
            )
        except Exception as exc:  # noqa: BLE001 - shadow failures cannot alter authority
            result.reasoner_shadow_observation_errors = [
                *result.reasoner_shadow_observation_errors,
                {
                    "component": "reasoner_shadow_runner",
                    "operation": "observe",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                },
            ]
            result.module_checkpoint_count = checkpoint_observer.checkpoint_count
            result.module_checkpoint_errors = list(checkpoint_observer.errors)
            return result
        traces = [
            item.invocation_trace for item in shadow.records if item.invocation_trace is not None
        ]
        result.reasoner_shadow_history = [item.model_copy(deep=True) for item in shadow.records]
        result.reasoner_shadow_logical_provider_call_count = sum(
            item.logical_provider_call_count for item in traces
        )
        result.reasoner_shadow_provider_call_count = sum(
            item.provider_call_count for item in traces
        )
        result.reasoner_shadow_retry_count = sum(
            item.retry_count + item.logical_retry_count for item in traces
        )
        result.reasoner_shadow_event_log_path = shadow.event_log_path
        result.reasoner_shadow_input_bundle_path = shadow.input_bundle_path
        result.reasoner_shadow_observation_errors = [
            *result.reasoner_shadow_observation_errors,
            *shadow.observation_errors,
        ]
        result.module_checkpoint_count = checkpoint_observer.checkpoint_count
        result.module_checkpoint_errors = list(checkpoint_observer.errors)
        return result

    def _w2_general_perception_request(
        self,
        case: WorkflowRuntimeCase,
    ) -> WorkflowPerceptionRequest:
        operation = self.profile.w2_general_perception_operation
        route_policy = self.profile.perception_route_policy(operation)
        audio_refs = [
            *case.source_audio_refs,
            *case.reference_audio_refs,
        ]
        return WorkflowPerceptionRequest(
            request_id="w2:general-perception",
            scope="general",
            instruction=(
                "Build a general, reviewed description of all supplied audio for "
                "planning. Review the source audio and every supplied reference "
                "independently. Preserve transcript, speaker, event, acoustic, "
                "timing, and uncertainty observations."
            ),
            operation=operation,
            inputs={"input_audio_path": case.source_audio_refs[0]},
            params=self.profile.perception_params(operation),
            expected_output="General reviewed observations for all supplied audio.",
            audio_refs=audio_refs,
            requirements=[
                "reviewed observations for the source and every supplied reference",
                "audio-ref-scoped evidence for each supplied audio",
                "explicit uncertainty for unavailable evidence",
            ],
            metadata={"perception_route_policy": route_policy.model_dump(mode="json")},
        )

    def _semantic_bootstrap_perception_requests(
        self,
        case: WorkflowRuntimeCase,
    ) -> list[WorkflowPerceptionRequest]:
        """Build explicit per-audio source observations for the semantic Reasoner."""

        mode = self.profile.semantic_planning.source_observation_bootstrap
        if mode == "disabled":
            return []
        if mode not in {"speech_context_per_audio", "speech_and_scene_per_audio"}:
            raise ValueError(f"unsupported semantic source observation mode: {mode}")

        operation = "perception.speech_context"
        route_policy = self.profile.perception_route_policy(operation)
        all_audio_refs = [*case.source_audio_refs, *case.reference_audio_refs]
        already_observed: set[str] = set()
        if case.source_text and case.source_audio_refs:
            already_observed.add(case.source_audio_refs[0])
        requests: list[WorkflowPerceptionRequest] = []
        for index, audio_ref in enumerate(all_audio_refs):
            if audio_ref in already_observed:
                continue
            requests.append(
                WorkflowPerceptionRequest(
                    request_id=f"semantic-bootstrap:speech-context:{index:04d}",
                    scope="general",
                    instruction=(
                        "Observe this audio independently before semantic reasoning. "
                        "Return its source transcript and word timing when speech is "
                        "present, and preserve explicit uncertainty when an observation "
                        "is unavailable. Do not interpret the edit request or choose a tool."
                    ),
                    operation=operation,
                    inputs={"input_audio_path": audio_ref},
                    params={
                        **self.profile.perception_params(operation),
                        "need_source_text": False,
                        "observe_source_text": True,
                    },
                    expected_output=("Audio-ref-scoped source text, word timing, and uncertainty."),
                    audio_refs=[audio_ref],
                    requirements=[
                        "audio-ref-scoped source text when speech is present",
                        "word timing when alignment is available",
                        "explicit uncertainty for unavailable observations",
                    ],
                    metadata={
                        "framework_stage": "semantic_source_observation",
                        "perception_route_policy": route_policy.model_dump(mode="json"),
                    },
                )
            )
        if mode == "speech_and_scene_per_audio":
            from x_editor.perception.scene_bootstrap import scene_bootstrap_requests
            requests.extend(scene_bootstrap_requests(case, self.profile))
        return requests

    async def arun_scenario(
        self,
        scenario: WorkflowRuntimeScenario,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        future_attempt_guidance_by_turn: dict[int, FutureAttemptGuidance] | None = None,
        continuation: WorkflowContinuation | dict | None = None,
    ) -> WorkflowScenarioResult:
        return await arun_workflow_scenario(
            self,
            scenario,
            arm=arm,
            output_dir=output_dir,
            future_attempt_guidance_by_turn=future_attempt_guidance_by_turn,
            continuation=continuation,
        )

    def run_scenario(
        self,
        scenario: WorkflowRuntimeScenario,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        future_attempt_guidance_by_turn: dict[int, FutureAttemptGuidance] | None = None,
        continuation: WorkflowContinuation | dict | None = None,
    ) -> WorkflowScenarioResult:
        import asyncio

        return asyncio.run(
            self.arun_scenario(
                scenario,
                arm=arm,
                output_dir=output_dir,
                future_attempt_guidance_by_turn=future_attempt_guidance_by_turn,
                continuation=continuation,
            )
        )


def _as_opaque_runtime_case(case: WorkflowRuntimeCase) -> WorkflowRuntimeCase:
    """Normalize legacy direct inputs before any model-facing projection."""

    if case.audio_resolution_mode == "opaque_ref_v1":
        return case.model_copy(deep=True)

    all_values = [*case.source_audio_refs, *case.reference_audio_refs]
    dangling_logical_refs = sorted(value for value in all_values if value.startswith("aud:v1:"))
    if dangling_logical_refs:
        raise ValueError(
            f"logical audio refs require private runtime bindings: {dangling_logical_refs}"
        )

    source_refs = [f"aud:v1:s{index:04d}" for index in range(len(case.source_audio_refs))]
    reference_refs = [f"aud:v1:r{index:04d}" for index in range(len(case.reference_audio_refs))]
    bindings = [
        RuntimeAudioBinding(
            audio_ref=source_refs[index],
            materialized_path=path,
            role="source",
            order=index,
            origin="scenario_asset",
        )
        for index, path in enumerate(case.source_audio_refs)
    ]
    bindings.extend(
        RuntimeAudioBinding(
            audio_ref=reference_refs[index],
            materialized_path=path,
            role="reference",
            order=len(source_refs) + index,
            origin="scenario_asset",
        )
        for index, path in enumerate(case.reference_audio_refs)
    )
    return WorkflowRuntimeCase(
        user_request=case.user_request,
        source_audio_refs=source_refs,
        reference_audio_refs=reference_refs,
        source_text=case.source_text,
        prior_turn_contexts=[item.model_copy(deep=True) for item in case.prior_turn_contexts],
        session_state=(
            case.session_state.model_copy(deep=True) if case.session_state is not None else None
        ),
        historical_reviewed_evidence=(
            case.historical_reviewed_evidence.model_copy(deep=True)
            if case.historical_reviewed_evidence is not None
            else None
        ),
        future_attempt_guidance=(
            case.future_attempt_guidance.model_copy(deep=True)
            if case.future_attempt_guidance is not None
            else None
        ),
        planning_context=copy.deepcopy(case.planning_context),
        audio_resolution_mode="opaque_ref_v1",
        audio_bindings=bindings,
    )


def default_edit_operations() -> list[str]:
    """Return the current shared Executor edit catalog without tool calls."""

    executor = AudioEditExecutor(
        enable_gemini_executor=False,
        agentic_mode=OPEN_GLM_QWEN_OMNI_MODE,
    )
    return [
        operation
        for operation in executor.registry.list_operations()
        if not operation.startswith(_PERCEPTION_PREFIXES) and operation not in _NON_EDIT_OPERATIONS
    ]


def default_evidence_required_operations(
    allowed_edit_operations: list[str],
) -> list[str]:
    """Return edits whose tool contract consumes reviewed target spans."""

    return sorted(set(allowed_edit_operations) & _REVIEWED_SPAN_REQUIRED_OPERATIONS)


def list_runtime_capability_profile_names() -> tuple[DeployableRuntimeCapabilityProfileName, ...]:
    """Return benchmark-neutral profiles accepted by public runtime CLIs."""

    return (
        "full_catalog",
        "lexical_local_v1",
        "speech_edit_core_v1",
        "speech_edit_core_v2",
    )


def build_default_open_provider_workflow_profile(
    *,
    planner_base_url: str,
    planner_model: str,
    perception_base_url: str,
    perception_model: str,
    planner_timeout_sec: float = 120.0,
    perception_timeout_sec: float = 180.0,
    semantic_planning_mode: Literal["legacy_combined", "separated_memo"] = ("legacy_combined"),
    semantic_reasoner_base_url: str | None = None,
    semantic_reasoner_model: str | None = None,
    semantic_reasoner_provider: str = GLM_PROVIDER,
    semantic_reasoner_timeout_sec: float = 120.0,
    semantic_reasoner_strategy: Literal[
        "memo_only", "memo_only_brief", "prose_memo", "evidence_memo", "draft_memo"
    ] = "prose_memo",
    semantic_reasoner_initial_memo_enable_thinking: bool = (
        DEFAULT_SEMANTIC_REASONER_INITIAL_MEMO_ENABLE_THINKING
    ),
    semantic_reasoner_initial_memo_max_output_tokens: int = (
        DEFAULT_SEMANTIC_REASONER_INITIAL_MEMO_MAX_OUTPUT_TOKENS
    ),
    semantic_reasoner_evidence_review_max_output_tokens: int = (
        DEFAULT_SEMANTIC_REASONER_EVIDENCE_REVIEW_MAX_OUTPUT_TOKENS
    ),
    semantic_source_observation_bootstrap: Literal[
        "disabled",
        "speech_context_per_audio",
        "speech_and_scene_per_audio",
    ]
    | None = None,
    reasoner_shadow_base_url: str | None = None,
    reasoner_shadow_model: str | None = None,
    reasoner_shadow_provider: str = GLM_PROVIDER,
    reasoner_shadow_timeout_sec: float = 120.0,
    reasoner_shadow_strategy: ReasonerShadowStrategy = "direct_json",
    workflow_protocol_profile: WorkflowProtocolProfileName = (
        DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME
    ),
    planner_prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
    runtime_capability_profile: DeployableRuntimeCapabilityProfileName = (
        DEFAULT_DEPLOYABLE_RUNTIME_CAPABILITY_PROFILE_NAME
    ),
    tool_prior_snapshot_path: str | Path | None = None,
) -> OpenProviderWorkflowProfile:
    """Build the default benchmark-independent local Open Provider profile."""

    if (reasoner_shadow_base_url is None) != (reasoner_shadow_model is None):
        raise ValueError(
            "Reasoner shadow requires both reasoner_shadow_base_url and reasoner_shadow_model"
        )
    if semantic_planning_mode == "separated_memo":
        semantic_reasoner_base_url = semantic_reasoner_base_url or planner_base_url
        semantic_reasoner_model = semantic_reasoner_model or planner_model
        if reasoner_shadow_base_url is not None:
            raise ValueError(
                "separated semantic planning cannot also enable post-run Reasoner shadow"
            )
        semantic_source_observation_bootstrap = (
            semantic_source_observation_bootstrap or "speech_context_per_audio"
        )
    elif semantic_reasoner_base_url is not None or semantic_reasoner_model is not None:
        raise ValueError(
            "semantic Reasoner endpoint/model require semantic_planning_mode=separated_memo"
        )
    elif semantic_source_observation_bootstrap not in {None, "disabled"}:
        raise ValueError(
            "semantic source observation requires semantic_planning_mode=separated_memo"
        )

    if runtime_capability_profile == "lexical_local_v1":
        operations = list(LEXICAL_LOCAL_V1_EDIT_OPERATIONS)
        perception_operations = list(LEXICAL_LOCAL_V1_PERCEPTION_OPERATIONS)
    elif runtime_capability_profile == "speech_edit_core_v1":
        operations = list(SPEECH_EDIT_CORE_V1_EDIT_OPERATIONS)
        perception_operations = list(DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS)
    elif runtime_capability_profile == "speech_edit_core_v2":
        operations = list(SPEECH_EDIT_CORE_V2_EDIT_OPERATIONS)
        perception_operations = list(DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS)
    else:
        operations = default_edit_operations()
        perception_operations = list(DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS)
    protocol = resolve_workflow_protocol_policy(workflow_protocol_profile)
    evidence_required_operations = (
        operations
        if protocol.evidence_requirement_scope == "all_edits"
        else default_evidence_required_operations(operations)
    )
    all_perception_params = {
        "perception.speech_context": {
            "strict": True,
            "refresh_cache": True,
        },
        "perception.speaker_attributed_review": {
            "strict": False,
            "run_specialized": True,
            "use_sedicow": True,
            "speaker_evidence_backend": "auto",
            "supplement_speaker_attributes": True,
            "refresh_cache": True,
            "use_temporal_reasoning": False,
        },
        "perception.semantic_span_review": {"strict": True},
        "perception.audio_fallback": {
            "strict": True,
            "run_specialized": False,
            "force_gemini": True,
            "refresh_cache": True,
        },
        "perception.timestamped_caption": {
            "use_flexsed_prior": False,
            "require_flexsed_prior_success": False,
            "enable_candidate_rubric": True,
            "strict": True,
        },
        "perception.flexsed_events": {"strict": True},
    }
    all_route_policies = [
        OpenProviderPerceptionRoutePolicy(
            operation="perception.speech_context",
            route_id="ts_asr_grounding",
            max_provider_calls=0,
            max_service_calls=2,
            max_tool_calls=0,
            cache_mode="refresh",
        ),
        OpenProviderPerceptionRoutePolicy(
            operation="perception.speaker_attributed_review",
            route_id="speaker_specialists_plus_omni_review",
            max_provider_calls=2,
            max_service_calls=4,
            max_tool_calls=0,
            cache_mode="refresh",
        ),
        OpenProviderPerceptionRoutePolicy(
            operation="perception.semantic_span_review",
            route_id="ts_asr_optional_omni_rubric",
            max_provider_calls=1,
            max_service_calls=2,
            max_tool_calls=0,
            cache_mode="refresh",
        ),
        OpenProviderPerceptionRoutePolicy(
            operation="perception.audio_fallback",
            route_id="qwen3_omni_direct",
            max_provider_calls=1,
            max_service_calls=0,
            max_tool_calls=0,
        ),
        OpenProviderPerceptionRoutePolicy(
            operation="perception.timestamped_caption",
            route_id="qwen3_omni_timestamped_caption",
            max_provider_calls=3,
            max_service_calls=0,
            max_tool_calls=0,
        ),
        OpenProviderPerceptionRoutePolicy(
            operation="perception.flexsed_events",
            route_id="flexsed_event_detection",
            max_provider_calls=0,
            max_service_calls=0,
            max_tool_calls=1,
        ),
    ]
    allowed_perception_set = set(perception_operations)
    edit_operation_params = {}
    if runtime_capability_profile in {"speech_edit_core_v1", "speech_edit_core_v2"}:
        edit_operation_params.update(
            {
                "emphasis_stress_edit": {"emphasis_strategy": "p1_local_resynthesis"},
                "no_reference_voice_conversion": {"voice_design_backend": "qwen3_tts"},
            }
        )
    if runtime_capability_profile == "speech_edit_core_v2":
        edit_operation_params.update(
            {
                "foreground_background_remix": {"preserve_duration": True},
                "speech_separation": {"output_format": "WAV"},
            }
        )
    tool_prior_snapshot = (
        load_tool_prior_snapshot(tool_prior_snapshot_path)
        if tool_prior_snapshot_path is not None
        else None
    )
    return OpenProviderWorkflowProfile(
        planner=FrozenProviderPolicy(
            component="planner",
            provider=GLM_PROVIDER,
            model=planner_model,
            base_url=planner_base_url,
            authentication_mode="none",
            timeout_sec=planner_timeout_sec,
            max_output_tokens=32768,
            temperature=0.0,
            seed=0,
        ),
        perception=FrozenProviderPolicy(
            component="perception",
            provider=QWEN3_OMNI_PROVIDER,
            model=perception_model,
            base_url=perception_base_url,
            authentication_mode="none",
            timeout_sec=perception_timeout_sec,
            max_output_tokens=1536,
            temperature=0.0,
        ),
        reasoner_shadow=(
            OpenProviderReasonerShadowPolicy(
                mode="observe",
                strategy=reasoner_shadow_strategy,
                provider=FrozenProviderPolicy(
                    component="reasoner",
                    provider=reasoner_shadow_provider,
                    model=str(reasoner_shadow_model),
                    base_url=str(reasoner_shadow_base_url),
                    authentication_mode="none",
                    timeout_sec=reasoner_shadow_timeout_sec,
                    max_output_tokens=(32768 if reasoner_shadow_strategy == "memo_only" else 8192),
                    temperature=0.0,
                    seed=0,
                    chat_template_kwargs={
                        "enable_thinking": reasoner_shadow_strategy
                        in {"private_thinking_json", "memo_only"}
                    },
                ),
                calls_per_planner_gateway=(
                    2 if reasoner_shadow_strategy == "memo_then_compile" else 1
                ),
            )
            if reasoner_shadow_base_url is not None
            else OpenProviderReasonerShadowPolicy()
        ),
        semantic_planning=(
            OpenProviderSemanticPlanningPolicy(
                mode="separated_memo",
                reasoner_strategy=semantic_reasoner_strategy,
                source_observation_bootstrap=str(semantic_source_observation_bootstrap),
                evidence_review_max_output_tokens=(
                    semantic_reasoner_evidence_review_max_output_tokens
                ),
                reasoner=FrozenProviderPolicy(
                    component="reasoner",
                    provider=semantic_reasoner_provider,
                    model=str(semantic_reasoner_model),
                    base_url=str(semantic_reasoner_base_url),
                    authentication_mode="none",
                    timeout_sec=semantic_reasoner_timeout_sec,
                    max_output_tokens=semantic_reasoner_initial_memo_max_output_tokens,
                    temperature=0.0,
                    seed=0,
                    chat_template_kwargs={
                        "enable_thinking": semantic_reasoner_initial_memo_enable_thinking
                    },
                ),
            )
            if semantic_planning_mode == "separated_memo"
            else OpenProviderSemanticPlanningPolicy()
        ),
        runtime_capability_profile=runtime_capability_profile,
        allowed_perception_operations=perception_operations,
        perception_operation_params={
            operation: params
            for operation, params in all_perception_params.items()
            if operation in allowed_perception_set
        },
        perception_route_policies=[
            policy for policy in all_route_policies if policy.operation in allowed_perception_set
        ],
        allowed_edit_operations=operations,
        edit_operation_params=edit_operation_params,
        tool_prior_snapshot=tool_prior_snapshot,
        tool_prior_binding=(
            tool_prior_snapshot.experiment_binding() if tool_prior_snapshot is not None else None
        ),
        evidence_required_operations=evidence_required_operations,
        workflow_protocol_profile=protocol.profile_name,
        planner_prompt_profile=planner_prompt_profile,
    )


def build_open_provider_workflow_runtime(
    profile: OpenProviderWorkflowProfile,
    *,
    output_dir: str | Path,
    planner_client: Any | None = None,
    perception_client: Any | None = None,
    semantic_reasoner_client: Any | None = None,
    reasoner_shadow_client: Any | None = None,
    reasoner_shadow_backend: Any | None = None,
    perception_tool_executor: ToolExecutor | Any | None = None,
    planner_backend: Any | None = None,
    perception_frontends: dict[str, Any] | None = None,
    speech_perception_client: SpeechPerceptionClient | None = None,
    provider_scheduling: ProviderSchedulingRuntime | None = None,
) -> OpenProviderWorkflowRuntime:
    """Assemble the shared workflow without invoking providers or tools."""

    output_root = Path(output_dir)
    protocol_policy = profile.protocol_policy()
    if planner_backend is None:
        planner_client = _resolve_scheduled_provider_client(
            policy=profile.planner,
            client=planner_client,
            provider_scheduling=provider_scheduling,
        )
    elif provider_scheduling is not None:
        _configure_planner_backend_scheduling(planner_backend, provider_scheduling)
    if (profile.semantic_planning.mode == "separated_memo" or
            (profile.semantic_planning.mode == "advisory_plan" and profile.semantic_planning.reasoner is not None)):
        if planner_backend is not None:
            raise ValueError("separated memo planning requires the memo-qualified Planner provider")
        assert profile.semantic_planning.reasoner is not None
        semantic_reasoner_client = _resolve_scheduled_provider_client(
            policy=profile.semantic_planning.reasoner,
            client=semantic_reasoner_client,
            provider_scheduling=provider_scheduling,
        )
    elif semantic_reasoner_client is not None:
        raise ValueError("semantic_reasoner_client requires separated_memo planning mode")
    if profile.reasoner_shadow.mode == "disabled" and (
        reasoner_shadow_client is not None or reasoner_shadow_backend is not None
    ):
        raise ValueError("Reasoner shadow backend/client requires an observe-mode profile")
    if profile.reasoner_shadow.mode == "observe" and reasoner_shadow_backend is None:
        assert profile.reasoner_shadow.provider is not None
        reasoner_shadow_client = _resolve_scheduled_provider_client(
            policy=profile.reasoner_shadow.provider,
            client=reasoner_shadow_client,
            provider_scheduling=provider_scheduling,
        )
    if perception_frontends is None:
        client_by_policy: dict[str, Any] = {}
        default_policy_key = _provider_policy_key(profile.perception)
        if perception_client is not None:
            client_by_policy[default_policy_key] = perception_client
        perception_frontends = {}
        for operation in sorted(MODEL_BACKED_PERCEPTION_OPERATIONS & set(profile.allowed_perception_operations)):
            policy = profile.perception_provider(operation)
            policy_key = _provider_policy_key(policy)
            client = client_by_policy.get(policy_key)
            if client is None:
                client = build_frozen_openai_client(policy)
                client_by_policy[policy_key] = client
            perception_frontends[operation] = _build_perception_frontend(
                operation=operation,
                policy=policy,
                client=client,
            )
    else:
        perception_frontends = dict(perception_frontends)
    perception_executor = AudioEditExecutor(
        tool_executor=perception_tool_executor,
        output_dir=output_root / "perception",
        enable_gemini_executor=False,
        agentic_mode=OPEN_GLM_QWEN_OMNI_MODE,
        perception_frontends=perception_frontends,
        speech_perception_client=speech_perception_client,
    )
    forward_executor = AudioEditExecutor(
        output_dir=output_root / "forward",
        enable_gemini_executor=False,
        agentic_mode=OPEN_GLM_QWEN_OMNI_MODE,
        # Planner-authored perception prerequisites execute inside the forward
        # plan. They must use the same frozen provider frontends as the
        # dedicated perception gateway, rather than falling back to ambient
        # provider environment variables.
        perception_frontends=perception_frontends,
        speech_perception_client=speech_perception_client,
    )
    if profile.runtime_capability_profile == "speech_edit_core_v2":
        require_runtime_capability_profile_ready(
            "speech_edit_core_v2",
            registered_operations=forward_executor.registry.list_operations(),
        )
    available_operations = [
        *[
            perception_executor.registry.describe_operation(operation)
            for operation in profile.allowed_perception_operations
        ],
        *[
            forward_executor.registry.describe_operation(operation)
            for operation in profile.allowed_edit_operations
        ],
    ]
    edit_capability_cards = {
        operation: item["operation_capability"]
        for item in available_operations
        if (operation := str(item.get("operation") or "")) in set(profile.allowed_edit_operations)
        and isinstance(item.get("operation_capability"), dict)
    }
    if profile.semantic_planning.mode == "separated_memo":
        assert profile.semantic_planning.reasoner is not None
        reasoner_class = {
            "prose_memo": OpenAICompatibleProseMemoSemanticReasoner,
            "evidence_memo": OpenAICompatibleEvidenceMemoReasoner,
                "draft_memo": OpenAICompatibleDraftMemoReasoner,
        }.get(profile.semantic_planning.reasoner_strategy, OpenAICompatibleMemoOnlySemanticReasoner)
        reasoner_kwargs: dict[str, Any] = {
            "model_name": profile.semantic_planning.reasoner.model,
            "client": semantic_reasoner_client,
            "provider_name": profile.semantic_planning.reasoner.provider,
            "evidence_review_max_tokens": (
                profile.semantic_planning.evidence_review_max_output_tokens
            ),
        }
        if reasoner_class is OpenAICompatibleMemoOnlySemanticReasoner:
            reasoner_kwargs["strategy"] = profile.semantic_planning.reasoner_strategy
            if profile.semantic_planning.reasoner_retry_context_mode != "prior_memo":
                raise ValueError("Reasoner retry context requires a prose/evidence/draft memo strategy")
        else:
            reasoner_kwargs["retry_context_mode"] = profile.semantic_planning.reasoner_retry_context_mode
        reasoner = reasoner_class(**reasoner_kwargs)
        planner = OpenAICompatibleMemoQualifiedPlanner(
            model_name=profile.planner.model,
            client=planner_client,
            provider_name=profile.planner.provider,
            evidence_index_format=profile.semantic_planning.planner_evidence_index_format,
            argument_contract_format=profile.semantic_planning.planner_argument_contract_format,
            retry_guidance_mode=profile.semantic_planning.planner_retry_guidance_mode,
        )
        planner_adapter = SeparatedSemanticWorkflowPlannerAdapter(
            reasoner=reasoner,
            planner=planner,
            operation_capability_cards=edit_capability_cards,
            perception_capability_catalog=profile.perception_catalog_payload(),
            perception_operation_params={
                operation: profile.perception_params(operation)
                for operation in profile.allowed_perception_operations
            },
            edit_operation_params=profile.edit_operation_params,
        )
    elif profile.semantic_planning.mode in {"integrated_plan", "advisory_plan"}:
        if planner_backend is not None:
            raise ValueError("integrated planning requires the frozen integrated provider")
        from x_editor.planner.integrated import IntegratedSemanticPlanner, IntegratedTurnCompiler
        from x_editor.runtime.integrated_workflow_adapter import IntegratedWorkflowPlannerAdapter
        provider_class = IntegratedSemanticPlanner
        compiler_class = IntegratedTurnCompiler
        provider_options = {"reasoning": profile.semantic_planning.integrated_reasoning}
        if profile.semantic_planning.mode == "advisory_plan":
            from x_editor.planner.advisory import AdvisorySemanticPlanner, AdvisoryTurnCompiler
            provider_class = AdvisorySemanticPlanner
            compiler_class = AdvisoryTurnCompiler
            adviser = None
            if profile.semantic_planning.reasoner is not None:
                if profile.semantic_planning.reasoner_strategy != "evidence_memo":
                    raise ValueError("advisory experiment requires the evidence-memo adviser")
                adviser = OpenAICompatibleEvidenceMemoReasoner(
                    client=semantic_reasoner_client,
                    model_name=profile.semantic_planning.reasoner.model,
                    provider_name=profile.semantic_planning.reasoner.provider,
                    evidence_review_max_tokens=profile.semantic_planning.evidence_review_max_output_tokens,
                    retry_context_mode=profile.semantic_planning.reasoner_retry_context_mode,
                )
            provider_options = {"adviser": adviser,
                                "adviser_input_scope": profile.semantic_planning.adviser_input_scope}
        integrated = provider_class(
            client=planner_client, model_name=profile.planner.model,
            provider_name=profile.planner.provider, capability_cards=edit_capability_cards,
            perception_catalog=profile.perception_catalog_payload(),
            edit_operation_params=profile.edit_operation_params,
            **provider_options,
            evidence_index_format=profile.semantic_planning.planner_evidence_index_format,
            argument_contract_format=profile.semantic_planning.planner_argument_contract_format,
        )
        if profile.semantic_planning.action_abstraction_version == "hierarchical_v1":
            from x_editor.planner.action_abstraction import AbstractionClient
            integrated.client = AbstractionClient(integrated.client,
                profile.semantic_planning.action_representation, edit_capability_cards)
        elif profile.semantic_planning.action_representation != "planir":
            from x_editor.planner.action_representation import RepresentationClient
            integrated.client = RepresentationClient(
                integrated.client, profile.semantic_planning.action_representation)
        planner_adapter = IntegratedWorkflowPlannerAdapter(
            reasoner=integrated, planner=compiler_class(integrated),
            operation_capability_cards=edit_capability_cards,
            perception_capability_catalog=profile.perception_catalog_payload(),
            perception_operation_params={operation: profile.perception_params(operation)
                                         for operation in profile.allowed_perception_operations},
            edit_operation_params=profile.edit_operation_params,
        )
    elif planner_backend is not None:
        bind_available_operations = getattr(
            planner_backend,
            "bind_available_operations",
            None,
        )
        if callable(bind_available_operations):
            bind_available_operations(available_operations)
        planner = planner_backend
        planner_adapter = TraceableAudioEditPlannerAdapter(
            planner,
            allowed_perception_operations=set(profile.allowed_perception_operations),
            allowed_edit_operations=set(profile.allowed_edit_operations),
            evidence_required_operations=set(profile.evidence_required_operations),
            perception_operation_params={
                operation: profile.perception_params(operation)
                for operation in profile.allowed_perception_operations
            },
            perception_route_policies={
                policy.operation: policy.model_dump(mode="json")
                for policy in profile.perception_route_policies
            },
            edit_operation_params=profile.edit_operation_params,
            edit_operation_capability_cards=edit_capability_cards,
            tool_prior_guidance=profile.planner_tool_prior_guidance(),
            workflow_protocol_policy=protocol_policy,
            planner_prompt_profile=profile.planner_prompt_profile,
        )
    else:
        planner = GLMOpenAICompatiblePlanner(
            model_name=profile.planner.model,
            client=planner_client,
            available_operations=available_operations,
        )
        planner_adapter = TraceableAudioEditPlannerAdapter(
            planner,
            allowed_perception_operations=set(profile.allowed_perception_operations),
            allowed_edit_operations=set(profile.allowed_edit_operations),
            evidence_required_operations=set(profile.evidence_required_operations),
            perception_operation_params={
                operation: profile.perception_params(operation)
                for operation in profile.allowed_perception_operations
            },
            perception_route_policies={
                policy.operation: policy.model_dump(mode="json")
                for policy in profile.perception_route_policies
            },
            edit_operation_params=profile.edit_operation_params,
            edit_operation_capability_cards=edit_capability_cards,
            tool_prior_guidance=profile.planner_tool_prior_guidance(),
            workflow_protocol_policy=protocol_policy,
            planner_prompt_profile=profile.planner_prompt_profile,
        )
    reasoner_shadow_runner: ReasonerShadowRunner | None = None
    if profile.reasoner_shadow.mode == "observe":
        shadow_policy = profile.reasoner_shadow.provider
        assert shadow_policy is not None
        reasoner = reasoner_shadow_backend
        if reasoner is None:
            reasoner_class = {
                "memo_then_compile": OpenAICompatibleMemoSemanticReasoner,
                "memo_only": OpenAICompatibleMemoOnlySemanticReasoner,
                "memo_only_brief": OpenAICompatibleMemoOnlySemanticReasoner,
                "prose_memo": OpenAICompatibleProseMemoSemanticReasoner,
                "evidence_memo": OpenAICompatibleEvidenceMemoReasoner,
                "draft_memo": OpenAICompatibleDraftMemoReasoner,
            }.get(profile.reasoner_shadow.strategy, OpenAICompatibleSemanticReasoner)
            reasoner_kwargs = {
                "model_name": shadow_policy.model,
                "client": reasoner_shadow_client,
                "provider_name": shadow_policy.provider,
            }
            if reasoner_class in {
                OpenAICompatibleSemanticReasoner,
                OpenAICompatibleMemoOnlySemanticReasoner,
            }:
                reasoner_kwargs["strategy"] = profile.reasoner_shadow.strategy
            reasoner = reasoner_class(**reasoner_kwargs)
        semantic_effect_ontology = [
            {
                key: copy.deepcopy(value)
                for key, value in item.items()
                if key != "supporting_operations"
            }
            for item in compact_semantic_effect_capability_index(
                available_operations,
                perception_operation_ids=set(profile.allowed_perception_operations),
            )
        ]
        reasoner_shadow_runner = ReasonerShadowRunner(
            reasoner=reasoner,
            semantic_effect_ontology=semantic_effect_ontology,
        )
    perception_adapter = AudioEditExecutorPerceptionAdapter(
        perception_executor,
        allowed_perception_operations=set(profile.allowed_perception_operations),
        deterministic_operations={
            "perception.speech_context",
            "perception.flexsed_events",
        },
        workflow_protocol_policy=protocol_policy,
    )
    forward_path = ForwardExecutionPath(forward_executor)
    orchestrator = WorkflowArmOrchestrator(
        planner=planner_adapter,
        perception=perception_adapter,
        forward_path=forward_path,
        evidence_projector=ReviewedWorkflowEvidenceProjector(),
    )
    return OpenProviderWorkflowRuntime(
        profile=profile,
        orchestrator=orchestrator,
        planner_adapter=planner_adapter,
        perception_adapter=perception_adapter,
        forward_path=forward_path,
        reasoner_shadow_runner=reasoner_shadow_runner,
        provider_scheduling=provider_scheduling,
    )


def _resolve_scheduled_provider_client(
    *,
    policy: FrozenProviderPolicy,
    client: Any | None,
    provider_scheduling: ProviderSchedulingRuntime | None,
) -> Any:
    if provider_scheduling is None:
        return client if client is not None else build_frozen_openai_client(policy)
    if policy.component not in {"planner", "reasoner"}:
        raise ValueError("provider scheduling v2 currently owns only Planner and Reasoner budgets")
    budget = provider_scheduling.config.budget(policy.component)
    if client is None:
        return build_frozen_openai_client(
            policy,
            request_scheduler=provider_scheduling.scheduler(policy.component),
            timeout_sec=budget.request_timeout_sec,
        )
    return provider_scheduling.configure_client(client, policy.component)


def _provider_policy_key(policy: FrozenProviderPolicy) -> str:
    return json.dumps(
        policy.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_perception_frontend(
    *,
    operation: str,
    policy: FrozenProviderPolicy,
    client: Any,
) -> Any:
    common = {
        "model_name": policy.model,
        "client": client,
        "provider_name": policy.provider,
    }
    if operation == "perception.timestamped_caption":
        return OpenAICompatibleTimestampedCaptionFrontend(**common)
    if operation == "perception.speaker_attributed_review":
        return OpenAICompatibleSpeakerReviewFrontend(**common)
    frontend = OpenAICompatibleStrongOmniAudioFrontend(**common)
    if operation == "perception.semantic_span_review":
        frontend.component_name = operation
    return frontend


def _configure_planner_backend_scheduling(
    planner_backend: Any,
    provider_scheduling: ProviderSchedulingRuntime,
) -> None:
    transport = getattr(planner_backend, "transport", None)
    client = getattr(transport, "client", None)
    if client is None:
        raise TypeError(
            "provider scheduling v2 cannot bind this Planner backend because it "
            "does not expose transport.client"
        )
    provider_scheduling.configure_client(client, "planner")


__all__ = [
    "DEFAULT_DEPLOYABLE_RUNTIME_CAPABILITY_PROFILE_NAME",
    "DEFAULT_OPEN_PROVIDER_PERCEPTION_OPERATIONS",
    "DEFAULT_SEMANTIC_REASONER_EVIDENCE_REVIEW_MAX_OUTPUT_TOKENS",
    "DEFAULT_SEMANTIC_REASONER_INITIAL_MEMO_ENABLE_THINKING",
    "DEFAULT_SEMANTIC_REASONER_INITIAL_MEMO_MAX_OUTPUT_TOKENS",
    "LEXICAL_LOCAL_V1_EDIT_OPERATIONS",
    "LEXICAL_LOCAL_V1_PERCEPTION_OPERATIONS",
    "MODEL_BACKED_PERCEPTION_OPERATIONS",
    "SPEECH_EDIT_CORE_V1_EDIT_OPERATIONS",
    "SPEECH_EDIT_CORE_V2_EDIT_OPERATIONS",
    "DeployableRuntimeCapabilityProfileName",
    "OpenProviderPerceptionRoutePolicy",
    "OpenProviderReasonerShadowPolicy",
    "OpenProviderSemanticPlanningPolicy",
    "OpenProviderWorkflowProfile",
    "OpenProviderWorkflowRuntime",
    "build_default_open_provider_workflow_profile",
    "build_open_provider_workflow_runtime",
    "default_edit_operations",
    "default_evidence_required_operations",
    "list_runtime_capability_profile_names",
]
