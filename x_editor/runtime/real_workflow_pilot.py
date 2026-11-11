"""Frozen configuration and local readiness checks for a real W1 pilot."""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    QWEN3_OMNI_PROVIDER,
)
from x_editor.core.schemas import StrictModel
from x_editor.runtime.forward_execution import ForwardExecutionPath
from x_editor.runtime.open_provider_workflow import (
    OpenProviderPerceptionRoutePolicy,
    OpenProviderWorkflowProfile,
    build_open_provider_workflow_runtime,
)
from x_editor.runtime.provider_policy import (
    FrozenProviderPolicy,
    build_frozen_openai_client,
)
from x_editor.runtime.workflow_adapters import (
    AudioEditExecutorPerceptionAdapter,
    TraceableAudioEditPlannerAdapter,
)
from x_editor.runtime.workflow_orchestration import (
    WorkflowArmOrchestrator,
    WorkflowArmPolicy,
    WorkflowArmRequest,
    WorkflowArmResult,
)
from x_editor.tools.executor import ToolExecutor


class OpenProviderW1PilotProfile(StrictModel):
    """One causally frozen open-provider planning-perception-execution run."""

    schema_version: Literal["x-editor.open-provider-w1-pilot/v1"] = (
        "x-editor.open-provider-w1-pilot/v1"
    )
    arm: Literal["W1"] = "W1"
    planner: FrozenProviderPolicy
    perception: FrozenProviderPolicy
    perception_operation: Literal["perception.timestamped_caption"] = (
        "perception.timestamped_caption"
    )
    source_audio_refs: list[str] = Field(min_length=1, max_length=1)
    allowed_edit_operations: list[str] = Field(min_length=1)
    use_flexsed_prior: Literal[True] = True
    require_flexsed_prior_success: Literal[True] = True
    max_perception_review_attempts: Literal[1] = 1
    max_source_duration_sec: Literal[5.0] = 5.0
    evaluator_mode: Literal["disabled"] = "disabled"
    executor_llm_mode: Literal["disabled"] = "disabled"

    @model_validator(mode="after")
    def validate_components(self) -> OpenProviderW1PilotProfile:
        if self.planner.component != "planner":
            raise ValueError("planner policy must describe the planner component")
        if self.perception.component != "perception":
            raise ValueError(
                "perception policy must describe the perception component"
            )
        if len(self.allowed_edit_operations) != len(
            set(self.allowed_edit_operations)
        ):
            raise ValueError("allowed edit operations must be unique")
        if len(self.source_audio_refs) != len(set(self.source_audio_refs)):
            raise ValueError("source_audio_refs must be unique")
        if self.perception_operation in self.allowed_edit_operations:
            raise ValueError(
                "perception operation must not be admitted as a final edit operation"
            )
        if self.allowed_edit_operations != ["artifact_ops.extract_spans"]:
            raise ValueError(
                "the first real W1 pilot freezes artifact_ops.extract_spans as its only edit"
            )
        return self

    def workflow_policy(self) -> WorkflowArmPolicy:
        return WorkflowArmPolicy(
            component_invocation_trace_mode="required",
            evidence_projection_mode="reviewed",
            planner_provider_call_limit_per_gateway=1,
            perception_provider_call_limit_per_gateway=1,
        )

    def perception_params(self) -> dict[str, Any]:
        return {
            "use_flexsed_prior": self.use_flexsed_prior,
            "require_flexsed_prior_success": (
                self.require_flexsed_prior_success
            ),
            "max_review_attempts": self.max_perception_review_attempts,
            "enable_candidate_rubric": False,
            "allow_slice_completion": False,
            "slice_shortfall_mode": "warn",
            "strict": True,
        }

    def edit_operation_params(self) -> dict[str, dict[str, Any]]:
        return {
            "artifact_ops.extract_spans": {
                "strict_explicit_spans": True,
                "crossfade_ms": 10,
            }
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WorkflowPilotReadinessFinding(StrictModel):
    code: str
    message: str


class WorkflowPilotReadinessReport(StrictModel):
    schema_version: Literal["x-editor.workflow-pilot-readiness/v1"] = (
        "x-editor.workflow-pilot-readiness/v1"
    )
    ready: bool
    external_services_probed: Literal[False] = False
    profile_fingerprint: str
    findings: list[WorkflowPilotReadinessFinding] = Field(default_factory=list)
    checked_operations: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class OpenProviderW1Pilot:
    """Assembled real-component W1 path; building it performs no service calls."""

    profile: OpenProviderW1PilotProfile
    orchestrator: WorkflowArmOrchestrator
    planner_adapter: TraceableAudioEditPlannerAdapter
    perception_adapter: AudioEditExecutorPerceptionAdapter
    forward_path: ForwardExecutionPath
    readiness: WorkflowPilotReadinessReport

    def run(
        self,
        user_request: str,
        *,
        output_dir: str | Path,
        event_log_path: str | None = None,
    ) -> WorkflowArmResult:
        if not self.readiness.ready:
            codes = ", ".join(item.code for item in self.readiness.findings)
            raise RuntimeError(f"real W1 pilot is not ready: {codes}")
        return self.orchestrator.run(
            WorkflowArmRequest(
                arm="W1",
                user_request=user_request,
                source_audio_refs=list(self.profile.source_audio_refs),
                execution_context={"output_dir": str(Path(output_dir))},
                event_log_path=event_log_path,
                policy=self.profile.workflow_policy(),
            )
        )


def build_open_provider_w1_pilot(
    profile: OpenProviderW1PilotProfile,
    *,
    output_dir: str | Path,
    planner_client: Any | None = None,
    perception_client: Any | None = None,
    perception_tool_executor: ToolExecutor | Any | None = None,
) -> OpenProviderW1Pilot:
    """Assemble the native GLM/Qwen W1 path without invoking any component."""

    runtime_profile = OpenProviderWorkflowProfile(
        planner=profile.planner,
        perception=profile.perception,
        runtime_capability_profile="custom",
        allowed_perception_operations=[profile.perception_operation],
        # This pilot can only run W1. Keep the shared runtime profile internally
        # consistent without admitting an unused W2-only operation.
        w2_general_perception_operation=profile.perception_operation,
        perception_operation_params={
            profile.perception_operation: profile.perception_params()
        },
        perception_route_policies=[
            OpenProviderPerceptionRoutePolicy(
                operation=profile.perception_operation,
                route_id="flexsed_prior_plus_qwen3_omni_review",
                max_provider_calls=1,
                max_service_calls=0,
                max_tool_calls=1,
            )
        ],
        allowed_edit_operations=list(profile.allowed_edit_operations),
        edit_operation_params=profile.edit_operation_params(),
        evidence_required_operations=list(profile.allowed_edit_operations),
    )
    runtime = build_open_provider_workflow_runtime(
        runtime_profile,
        output_dir=output_dir,
        planner_client=planner_client,
        perception_client=perception_client,
        perception_tool_executor=perception_tool_executor,
    )
    readiness = assess_open_provider_w1_readiness(
        profile,
        planner_adapter=runtime.planner_adapter,
        perception_adapter=runtime.perception_adapter,
        forward_path=runtime.forward_path,
    )
    return OpenProviderW1Pilot(
        profile=profile,
        orchestrator=runtime.orchestrator,
        planner_adapter=runtime.planner_adapter,
        perception_adapter=runtime.perception_adapter,
        forward_path=runtime.forward_path,
        readiness=readiness,
    )


def assess_open_provider_w1_readiness(
    profile: OpenProviderW1PilotProfile,
    *,
    planner_adapter: Any,
    perception_adapter: Any,
    forward_path: Any,
) -> WorkflowPilotReadinessReport:
    """Inspect assembled objects without contacting providers or services."""

    findings: list[WorkflowPilotReadinessFinding] = []
    _require(
        profile.planner.provider == GLM_PROVIDER,
        findings,
        "planner_provider_not_open_glm",
        f"Open-provider pilot requires {GLM_PROVIDER} Planner.",
    )
    _require(
        profile.perception.provider == QWEN3_OMNI_PROVIDER,
        findings,
        "perception_provider_not_qwen3_omni",
        f"Open-provider pilot requires {QWEN3_OMNI_PROVIDER} Perception.",
    )
    for audio_ref in profile.source_audio_refs:
        _require(
            Path(audio_ref).is_file(),
            findings,
            f"source_audio_missing:{audio_ref}",
            f"Source audio does not exist: {audio_ref}",
        )
        duration_sec = _pcm_wav_duration(audio_ref)
        _require(
            duration_sec is not None,
            findings,
            f"source_audio_not_readable_pcm_wav:{audio_ref}",
            f"Source audio is not a readable PCM WAV: {audio_ref}",
        )
        if duration_sec is not None:
            _require(
                duration_sec <= profile.max_source_duration_sec,
                findings,
                f"source_audio_too_long:{audio_ref}",
                "The first pilot limits source audio to 5 seconds so Perception "
                "does not spend a separate slice-proposal provider call.",
            )
    planner = getattr(planner_adapter, "planner", planner_adapter)
    _require(
        callable(getattr(planner, "plan_workflow_turn", None)),
        findings,
        "planner_reasoned_turn_missing",
        "Planner does not expose the native Planner/Reasoner workflow turn.",
    )
    _check_provider_component(
        planner,
        profile.planner,
        findings=findings,
        prefix="planner",
    )
    _require(
        set(getattr(planner_adapter, "allowed_edit_operations", set()) or set())
        == set(profile.allowed_edit_operations),
        findings,
        "planner_edit_catalog_not_frozen",
        "Planner adapter edit catalog does not match the pilot profile.",
    )
    _require(
        getattr(planner_adapter, "perception_operation_params", {}).get(
            profile.perception_operation
        )
        == profile.perception_params(),
        findings,
        "planner_perception_params_not_frozen",
        "Planner adapter Perception parameters do not match the pilot profile.",
    )
    _require(
        getattr(planner_adapter, "edit_operation_params", {})
        == profile.edit_operation_params(),
        findings,
        "planner_edit_params_not_frozen",
        "Planner adapter edit parameters do not match the pilot profile.",
    )

    allowed_perception = set(
        getattr(perception_adapter, "allowed_perception_operations", set())
    )
    _require(
        profile.perception_operation in allowed_perception,
        findings,
        "perception_operation_not_frozen",
        "The selected Perception operation is absent from the adapter catalog.",
    )
    perception_executor = getattr(perception_adapter, "executor", None)
    _require(
        perception_executor is not None,
        findings,
        "perception_executor_missing",
        "Perception adapter has no AudioEditExecutor.",
    )
    perception_tool_executor = getattr(
        perception_executor,
        "tool_executor",
        None,
    )
    _require(
        callable(getattr(perception_tool_executor, "execute", None)),
        findings,
        "flexsed_tool_executor_missing",
        "Perception Executor has no callable FlexSED tool executor.",
    )
    frontend = None
    if perception_executor is not None:
        frontend = getattr(perception_executor, "perception_frontends", {}).get(
            profile.perception_operation
        )
    _require(
        frontend is not None,
        findings,
        "perception_frontend_not_injected",
        "Timestamped-caption frontend is not explicitly injected.",
    )
    if frontend is not None:
        _check_provider_component(
            frontend,
            profile.perception,
            findings=findings,
            prefix="perception",
        )

    forward_executor = getattr(forward_path, "executor", None)
    _require(
        forward_executor is not None,
        findings,
        "forward_executor_missing",
        "ForwardExecutionPath has no Executor.",
    )
    checked_operations = [
        profile.perception_operation,
        *profile.allowed_edit_operations,
    ]
    for operation in checked_operations:
        executor = (
            perception_executor
            if operation == profile.perception_operation
            else forward_executor
        )
        registry = getattr(executor, "registry", None)
        operations = set(registry.list_operations()) if registry is not None else set()
        _require(
            operation in operations,
            findings,
            f"executor_operation_missing:{operation}",
            f"Executor registry does not contain {operation}.",
        )
    if forward_executor is not None:
        _require(
            not bool(getattr(forward_executor, "enable_gemini_executor", False)),
            findings,
            "executor_llm_enabled",
            "Forward Executor still permits LLM tool-call planning.",
        )

    return WorkflowPilotReadinessReport(
        ready=not findings,
        profile_fingerprint=profile.fingerprint,
        findings=findings,
        checked_operations=checked_operations,
    )


def _check_provider_component(
    component: Any,
    policy: FrozenProviderPolicy,
    *,
    findings: list[WorkflowPilotReadinessFinding],
    prefix: str,
) -> None:
    provider = str(getattr(component, "provider_name", "") or "")
    model = str(getattr(component, "model_name", "") or "")
    _require(
        provider == policy.provider,
        findings,
        f"{prefix}_provider_mismatch",
        f"Expected {policy.provider}, observed {provider or '<missing>'}.",
    )
    _require(
        model == policy.model,
        findings,
        f"{prefix}_model_mismatch",
        f"Expected {policy.model}, observed {model or '<missing>'}.",
    )
    client = getattr(component, "client", None)
    _require(
        client is not None,
        findings,
        f"{prefix}_client_missing",
        "Provider component has no inspectable client.",
    )
    if client is None:
        return
    _require(
        callable(getattr(client, "call_json", None)),
        findings,
        f"{prefix}_client_call_json_missing",
        "Provider client does not expose callable call_json.",
    )
    expected_require_api_key = policy.authentication_mode == "bearer"
    _require(
        getattr(client, "require_api_key", None) is expected_require_api_key,
        findings,
        f"{prefix}_authentication_mode_mismatch",
        "Provider client authentication mode does not match the pilot profile.",
    )
    api_key = str(getattr(client, "api_key", "") or "").strip()
    if expected_require_api_key:
        _require(
            bool(api_key),
            findings,
            f"{prefix}_api_key_missing",
            "Bearer authentication requires a provider API key.",
        )
    else:
        _require(
            not api_key,
            findings,
            f"{prefix}_unexpected_api_key",
            "No-auth local mode must not send a hidden API key.",
        )
    expected_base_url = str(policy.base_url or "").strip()
    observed_base_url = str(getattr(client, "base_url", "") or "").strip()
    if not expected_base_url:
        findings.append(
            WorkflowPilotReadinessFinding(
                code=f"{prefix}_base_url_missing",
                message=(
                    "Local provider base URL must be explicit in the frozen "
                    "pilot profile."
                ),
            )
        )
    else:
        _require(
            _provider_endpoint_matches(
                observed_base_url,
                expected_base_url,
            ),
            findings,
            f"{prefix}_base_url_mismatch",
            "Provider client endpoint does not match the frozen pilot profile.",
        )
    timeout = getattr(client, "timeout", getattr(client, "_timeout", None))
    _require(
        timeout is not None and float(timeout) == policy.timeout_sec,
        findings,
        f"{prefix}_timeout_mismatch",
        f"Expected timeout {policy.timeout_sec}, observed {timeout}.",
    )
    observed_max_tokens = getattr(client, "max_tokens", None)
    _require(
        observed_max_tokens == policy.max_output_tokens,
        findings,
        f"{prefix}_max_output_tokens_mismatch",
        "Provider client output-token budget does not match the frozen pilot "
        f"profile: expected {policy.max_output_tokens}, observed "
        f"{observed_max_tokens}.",
    )
    observed_temperature = getattr(client, "temperature", None)
    _require(
        observed_temperature == policy.temperature,
        findings,
        f"{prefix}_temperature_mismatch",
        "Provider client temperature does not match the frozen pilot "
        f"profile: expected {policy.temperature}, observed "
        f"{observed_temperature}.",
    )
    _require(
        getattr(client, "max_retries", getattr(client, "_max_retries", None))
        == 0,
        findings,
        f"{prefix}_transport_retry_not_frozen",
        "Provider client is not explicitly frozen to zero retries.",
    )
    for attribute, code, message in (
        (
            "allow_response_format_fallback",
            f"{prefix}_response_format_fallback_enabled",
            "Response-format fallback is not disabled.",
        ),
        (
            "allow_json_repair",
            f"{prefix}_json_repair_enabled",
            "Provider JSON repair is not disabled.",
        ),
    ):
        if hasattr(client, attribute):
            _require(
                getattr(client, attribute) is False,
                findings,
                code,
                message,
            )


def _require(
    condition: bool,
    findings: list[WorkflowPilotReadinessFinding],
    code: str,
    message: str,
) -> None:
    if not condition:
        findings.append(WorkflowPilotReadinessFinding(code=code, message=message))


def _pcm_wav_duration(audio_ref: str) -> float | None:
    try:
        with wave.open(str(audio_ref), "rb") as wav:
            frame_rate = wav.getframerate()
            if frame_rate <= 0:
                return None
            return wav.getnframes() / float(frame_rate)
    except (EOFError, OSError, wave.Error):
        return None


def _provider_endpoint_matches(observed: str, expected_base: str) -> bool:
    observed = observed.rstrip("/")
    expected_base = expected_base.rstrip("/")
    if not observed or not expected_base:
        return False
    if expected_base.endswith("/chat/completions"):
        expected_chat = expected_base
    elif expected_base.endswith("/v1"):
        expected_chat = f"{expected_base}/chat/completions"
    else:
        expected_chat = f"{expected_base}/v1/chat/completions"
    return observed in {expected_base, expected_chat}


__all__ = [
    "FrozenProviderPolicy",
    "OpenProviderW1Pilot",
    "OpenProviderW1PilotProfile",
    "WorkflowPilotReadinessFinding",
    "WorkflowPilotReadinessReport",
    "assess_open_provider_w1_readiness",
    "build_frozen_openai_client",
    "build_open_provider_w1_pilot",
]
