"""Typed provider configuration for the opt-in Evaluation-Guidance runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.evaluation.guidance_contracts import EvaluationFocus, FrozenEvaluationModel
from x_editor.evaluation.guided_runtime import EvaluationGuidedScenarioPolicy
from x_editor.evaluation.openai_compatible_guidance import (
    OpenAICompatibleEvaluationBackend,
    OpenAICompatibleGuiderBackend,
)


class GuidanceTransportConfig(FrozenEvaluationModel):
    """One local or remote OpenAI-compatible evaluation transport."""

    schema_version: Literal["x-editor.guidance-transport/v1"] = (
        "x-editor.guidance-transport/v1"
    )
    backend_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    credential_env: str | None = None
    require_api_key: bool = False
    max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    timeout_sec: float = Field(default=300.0, gt=0.0, le=1800.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_transport_retries: int = Field(default=0, ge=0, le=5)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    request_body: dict[str, Any] = Field(default_factory=dict)
    audio_content_part: str | None = None

    @model_validator(mode="after")
    def validate_authentication(self) -> GuidanceTransportConfig:
        if self.require_api_key and not self.credential_env:
            raise ValueError("API-key transport requires credential_env")
        return self

    @property
    def credential_configured(self) -> bool:
        return bool(self.credential_env and os.getenv(self.credential_env))


class EvaluationGuidedRuntimeConfig(FrozenEvaluationModel):
    """Complete frozen configuration for one feedback-loop experiment arm."""

    schema_version: Literal["x-editor.evaluation-guided-runtime-config/v1"] = (
        "x-editor.evaluation-guided-runtime-config/v1"
    )
    policy: EvaluationGuidedScenarioPolicy
    evaluator: GuidanceTransportConfig
    guider: GuidanceTransportConfig
    evaluator_supported_foci: tuple[EvaluationFocus, ...] = (
        "reasoner_semantics",
        "planner_decision",
        "plan_graph",
    )

    @model_validator(mode="after")
    def validate_bindings(self) -> EvaluationGuidedRuntimeConfig:
        if self.policy.evaluator_backend_id != self.evaluator.backend_id:
            raise ValueError("policy evaluator_backend_id does not match transport")
        if self.policy.guider_backend_id != self.guider.backend_id:
            raise ValueError("policy guider_backend_id does not match transport")
        if len(self.evaluator_supported_foci) != len(set(self.evaluator_supported_foci)):
            raise ValueError("evaluator supported foci must be unique")
        return self


def load_evaluation_guided_runtime_config(
    path: str | Path,
) -> EvaluationGuidedRuntimeConfig:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return EvaluationGuidedRuntimeConfig.model_validate(payload)


def build_openai_compatible_guidance_backends(
    config: EvaluationGuidedRuntimeConfig,
) -> tuple[OpenAICompatibleEvaluationBackend, OpenAICompatibleGuiderBackend]:
    evaluator_client = _build_client(config.evaluator, component_name="evaluator")
    guider_client = _build_client(config.guider, component_name="guider")
    return (
        OpenAICompatibleEvaluationBackend(
            backend_id=config.evaluator.backend_id,
            provider=config.evaluator.provider,
            model=config.evaluator.model,
            client=evaluator_client,
            supported_foci=frozenset(config.evaluator_supported_foci),
            temperature=config.evaluator.temperature,
        ),
        OpenAICompatibleGuiderBackend(
            backend_id=config.guider.backend_id,
            provider=config.guider.provider,
            model=config.guider.model,
            client=guider_client,
        ),
    )


def _build_client(
    config: GuidanceTransportConfig,
    *,
    component_name: str,
) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        api_key=(os.getenv(config.credential_env) if config.credential_env else None),
        base_url=config.base_url,
        model_name=config.model,
        provider_name=config.provider,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        timeout=config.timeout_sec,
        max_retries=config.max_transport_retries,
        allow_response_format_fallback=False,
        allow_json_repair=False,
        require_api_key=config.require_api_key,
        chat_template_kwargs=config.chat_template_kwargs,
        audio_content_part=config.audio_content_part,
        request_body=config.request_body,
        component_name=component_name,
    )


__all__ = [
    "EvaluationGuidedRuntimeConfig",
    "GuidanceTransportConfig",
    "build_openai_compatible_guidance_backends",
    "load_evaluation_guided_runtime_config",
]
