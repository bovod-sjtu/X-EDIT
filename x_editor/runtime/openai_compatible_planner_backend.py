"""Provider-neutral bindings for the structured OpenAI-compatible Planner."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.planner.openai_compatible_planner import (
    OpenAICompatibleAudioEditPlanner,
)
from x_editor.runtime.component_backends import (
    BackendContractError,
    ComponentBackendRegistry,
    PlannerBackendConfig,
    ProviderTransportConfig,
)

OPENAI_COMPATIBLE_TRANSPORT_KIND = "openai-compatible"
STRUCTURED_WORKFLOW_PLANNER_BACKEND_ID = "structured-workflow-planner/v1"


class _ConfiguredOpenAICompatiblePlanner(OpenAICompatibleAudioEditPlanner):
    def __init__(self, *, provider_name: str, **kwargs: Any) -> None:
        self.provider_name = provider_name
        super().__init__(**kwargs)


@dataclass
class OpenAICompatibleProviderTransport:
    """Exact HTTP transport declared by a frozen component backend profile."""

    config: ProviderTransportConfig
    client: OpenAICompatibleChatClient = field(init=False)

    def __post_init__(self) -> None:
        if self.config.transport_kind != OPENAI_COMPATIBLE_TRANSPORT_KIND:
            raise BackendContractError(
                "OpenAI-compatible transport received a different transport_kind"
            )
        api_key: str | None = None
        require_api_key = self.config.authentication.mode == "bearer"
        if require_api_key:
            credential_env = self.config.authentication.credential_env
            api_key = os.getenv(credential_env or "", "").strip() or None
            if api_key is None:
                raise BackendContractError(
                    f"required credential environment variable {credential_env!r} is unset"
                )
        self.client = OpenAICompatibleChatClient(
            api_key=api_key,
            base_url=self.config.endpoint,
            model_name=self.config.model,
            provider_name=self.config.provider,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout_sec,
            max_retries=self.config.max_transport_retries,
            retry_backoff=0,
            allow_response_format_fallback=False,
            allow_json_repair=False,
            require_api_key=require_api_key,
            chat_template_kwargs=dict(self.config.chat_template_kwargs),
            audio_content_part=(
                self.config.request_dialect.audio_content_part
                if self.config.request_dialect is not None
                else None
            ),
            request_body=(
                self.config.request_dialect.request_body
                if self.config.request_dialect is not None
                else None
            ),
        )

    def invoke(self, request: Any) -> Any:
        """Invoke one explicit text or JSON chat request without fallback."""

        if not isinstance(request, Mapping):
            raise TypeError("OpenAI-compatible transport request must be a mapping")
        system_prompt = str(request.get("system_prompt") or "")
        user_text = str(request.get("user_text") or "")
        response_format = request.get("response_format")
        if response_format is not None and not isinstance(response_format, dict):
            raise TypeError("response_format must be a mapping or None")
        if request.get("response_kind", "json") == "text":
            return self.client.call(
                system_prompt,
                user_text,
                response_format=response_format,
            )
        return self.client.call_json(
            system_prompt,
            user_text,
            response_format=response_format,
            repair_schema_name=str(request.get("repair_schema_name") or "JSON"),
        )


@dataclass
class OpenAICompatibleStructuredPlannerBackend:
    """Bind the existing structured Planner to one exact provider transport."""

    config: PlannerBackendConfig
    transport: OpenAICompatibleProviderTransport
    _planner: OpenAICompatibleAudioEditPlanner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.backend_id != STRUCTURED_WORKFLOW_PLANNER_BACKEND_ID:
            raise BackendContractError("structured Planner adapter received a different backend_id")
        if self.config.transport_id != self.transport.config.transport_id:
            raise BackendContractError("Planner and transport identifiers do not match")
        response_format = self.transport.config.response_format
        if response_format.mode != "json_schema" or not response_format.strict_schema:
            raise BackendContractError(
                "structured Planner adapter requires strict json_schema output"
            )
        self._planner = _ConfiguredOpenAICompatiblePlanner(
            provider_name=self.transport.config.provider,
            api_key=self.transport.client.api_key,
            base_url=self.transport.config.endpoint,
            model_name=self.transport.config.model,
            temperature=self.transport.config.temperature,
            max_tokens=self.transport.config.max_output_tokens,
            timeout=self.transport.config.timeout_sec,
            client=self.transport.client,
        )

    @property
    def provider_name(self) -> str:
        return self._planner.provider_name

    @property
    def model_name(self) -> str:
        return self._planner.model_name

    @property
    def last_invocation_trace(self) -> Any:
        return self._planner.last_invocation_trace

    @last_invocation_trace.setter
    def last_invocation_trace(self, value: Any) -> None:
        self._planner.last_invocation_trace = value

    @property
    def available_operations(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._planner.available_operations)

    def bind_available_operations(
        self,
        available_operations: list[dict[str, Any]],
    ) -> None:
        if not available_operations:
            raise BackendContractError("Planner requires a non-empty operation catalog")
        self._planner.available_operations = copy.deepcopy(available_operations)

    def plan_workflow_turn(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        return self._planner.plan_workflow_turn(user_request, context)

    def plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        return self._planner.plan(user_request, context)


def register_openai_compatible_planner_backend(
    registry: ComponentBackendRegistry,
) -> None:
    """Register the exact transport and Planner factories, without aliases."""

    registry.register_transport(
        OPENAI_COMPATIBLE_TRANSPORT_KIND,
        OpenAICompatibleProviderTransport,
    )
    registry.register_planner(
        STRUCTURED_WORKFLOW_PLANNER_BACKEND_ID,
        OpenAICompatibleStructuredPlannerBackend,
    )


__all__ = [
    "OPENAI_COMPATIBLE_TRANSPORT_KIND",
    "STRUCTURED_WORKFLOW_PLANNER_BACKEND_ID",
    "OpenAICompatibleProviderTransport",
    "OpenAICompatibleStructuredPlannerBackend",
    "register_openai_compatible_planner_backend",
]
