"""Frozen provider transport policy shared by workflow runtimes."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.core.agentic_mode import (
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
)
from x_editor.core.provider_requests import ProviderRequestScheduler
from x_editor.core.schemas import StrictModel


class FrozenProviderPolicy(StrictModel):
    """Provider controls applied before a workflow component call."""

    schema_version: Literal["x-editor.frozen-provider-policy/v1"] = (
        "x-editor.frozen-provider-policy/v1"
    )
    component: Literal["reasoner", "planner", "perception"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str | None = None
    authentication_mode: Literal["none", "bearer"] = "none"
    credential_env: str | None = None
    timeout_sec: float = Field(gt=0, le=1200)
    max_output_tokens: int | None = Field(default=None, gt=0, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    max_transport_retries: Literal[0] = 0
    response_format_fallback: Literal["disabled"] = "disabled"
    json_repair: Literal["disabled"] = "disabled"
    schema_repair: Literal["disabled"] = "disabled"
    cache_mode: Literal["disabled"] = "disabled"
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        max_length=16,
        description=(
            "Explicit provider chat-template controls frozen into the runtime "
            "profile, such as enable_thinking for a compatible local model."
        ),
    )
    audio_content_part: Literal["input_audio", "audio_url"] | None = None
    request_body: dict[str, Any] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def normalize_base_url(self) -> FrozenProviderPolicy:
        if self.base_url is not None:
            self.base_url = self.base_url.strip() or None
        if self.credential_env is not None:
            self.credential_env = self.credential_env.strip() or None
        if self.authentication_mode == "none" and self.credential_env is not None:
            raise ValueError("credential_env is forbidden when authentication is none")
        if self.max_output_tokens is None:
            self.max_output_tokens = 1536 if self.component == "perception" else 8192
        return self


def build_frozen_openai_client(
    policy: FrozenProviderPolicy,
    *,
    request_scheduler: ProviderRequestScheduler | None = None,
    timeout_sec: float | None = None,
) -> OpenAICompatibleChatClient:
    """Create the OpenAI-compatible client declared by a frozen policy."""

    resolved_timeout_sec = policy.timeout_sec if timeout_sec is None else float(timeout_sec)
    if resolved_timeout_sec <= 0:
        raise ValueError("provider request timeout must be positive")
    if request_scheduler is not None and request_scheduler.component != policy.component:
        raise ValueError(
            "provider request scheduler component does not match the frozen provider policy"
        )
    require_api_key = policy.authentication_mode == "bearer"
    api_key = None
    if require_api_key:
        api_key = (
            os.getenv(policy.credential_env, "").strip() or None
            if policy.credential_env is not None
            else resolve_openai_compatible_api_key(policy.provider)
        )
    return OpenAICompatibleChatClient(
        api_key=api_key,
        base_url=(policy.base_url or resolve_openai_compatible_base_url(policy.provider)),
        model_name=policy.model,
        provider_name=policy.provider,
        temperature=policy.temperature,
        seed=policy.seed,
        max_tokens=int(policy.max_output_tokens),
        timeout=resolved_timeout_sec,
        max_retries=0,
        retry_backoff=0,
        allow_response_format_fallback=False,
        allow_json_repair=False,
        require_api_key=require_api_key,
        chat_template_kwargs=policy.chat_template_kwargs,
        audio_content_part=policy.audio_content_part,
        request_body=policy.request_body,
        component_name=policy.component,
        request_scheduler=request_scheduler,
    )


__all__ = ["FrozenProviderPolicy", "build_frozen_openai_client"]
