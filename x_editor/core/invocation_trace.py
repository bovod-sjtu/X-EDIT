"""Typed accounting for component, provider, repair, and retry invocations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

ProviderInvocationStatus = Literal[
    "completed",
    "failed",
    "pre_transport_failed",
    "skipped",
]
ProviderInvocationKind = Literal[
    "primary",
    "json_repair",
    "schema_repair",
    "protocol_fallback",
    "logical_retry",
    "other",
]
ComponentInvocationStatus = Literal["completed", "failed"]
ComponentInvocationRoute = Literal[
    "deterministic",
    "deterministic_preflight",
    "provider",
    "deterministic_fallback",
    "mixed",
]
ComponentFallbackKind = Literal[
    "deterministic_fallback",
    "protocol_fallback",
    "provider_fallback",
]


class ProviderInvocationRecord(StrictModel):
    """One logical provider call and its transport-level attempt count."""

    schema_version: Literal["x-editor.provider-invocation/v1"] = (
        "x-editor.provider-invocation/v1"
    )
    invocation_index: int = Field(ge=1)
    invocation_id: str
    parent_invocation_id: str | None = None
    provider: str
    model: str | None = None
    kind: ProviderInvocationKind
    purpose: str
    status: ProviderInvocationStatus
    transport_attempt_count: int = Field(ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_attempts(self) -> ProviderInvocationRecord:
        if self.status in {"skipped", "pre_transport_failed"}:
            if self.transport_attempt_count != 0:
                raise ValueError(
                    f"{self.status} provider invocation cannot have attempts"
                )
        elif self.transport_attempt_count < 1:
            raise ValueError("attempted provider invocation requires an attempt")
        sensitive_keys = _sensitive_trace_keys(self.metrics)
        if sensitive_keys:
            raise ValueError(
                f"provider invocation metrics contain sensitive fields: {sensitive_keys}"
            )
        return self


class ComponentInvocationAdmissionPolicy(StrictModel):
    """Frozen gate applied to one Planner or Perception component call."""

    schema_version: Literal["x-editor.component-invocation-admission/v1"] = (
        "x-editor.component-invocation-admission/v1"
    )
    trace_mode: Literal["optional", "required"] = "optional"
    provider_call_limit: int = Field(default=1, ge=0, le=100)
    provider_retry_mode: Literal["disabled"] = "disabled"
    json_repair_mode: Literal["disabled"] = "disabled"
    schema_repair_mode: Literal["disabled"] = "disabled"
    fallback_mode: Literal["disabled"] = "disabled"


class ComponentFallbackRecord(StrictModel):
    """One explicit route or protocol fallback inside a component call."""

    schema_version: Literal["x-editor.component-fallback/v1"] = (
        "x-editor.component-fallback/v1"
    )
    fallback_index: int = Field(ge=1)
    kind: ComponentFallbackKind
    from_route: str
    to_route: str
    reason: str
    authorized: bool = False
    provider_invocation_id: str | None = None


class ComponentInvocationTrace(StrictModel):
    """Terminal accounting for one model-backed component gateway call."""

    schema_version: Literal["x-editor.component-invocation-trace/v1"] = (
        "x-editor.component-invocation-trace/v1"
    )
    component: Literal["reasoner", "planner", "perception"]
    component_name: str
    gateway_call_id: str | None = None
    route: ComponentInvocationRoute
    status: ComponentInvocationStatus
    provider_invocations: list[ProviderInvocationRecord] = Field(
        default_factory=list
    )
    fallback_events: list[ComponentFallbackRecord] = Field(default_factory=list)
    logical_provider_call_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    logical_retry_count: int = Field(default=0, ge=0)
    json_repair_count: int = Field(ge=0)
    schema_repair_count: int = Field(ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        component: Literal["reasoner", "planner", "perception"],
        component_name: str,
        gateway_call_id: str | None = None,
        route: ComponentInvocationRoute,
        status: ComponentInvocationStatus,
        provider_invocations: list[ProviderInvocationRecord] | None = None,
        fallback_events: list[ComponentFallbackRecord] | None = None,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ComponentInvocationTrace:
        invocations = list(provider_invocations or [])
        fallbacks = list(fallback_events or [])
        fallback_provider_ids = {
            item.provider_invocation_id
            for item in fallbacks
            if item.provider_invocation_id is not None
        }
        for invocation in invocations:
            if (
                invocation.kind != "protocol_fallback"
                or invocation.invocation_id in fallback_provider_ids
            ):
                continue
            fallbacks.append(
                ComponentFallbackRecord(
                    fallback_index=len(fallbacks) + 1,
                    kind="protocol_fallback",
                    from_route="requested_response_format",
                    to_route=_protocol_fallback_target(invocation.purpose),
                    reason=invocation.purpose,
                    authorized=False,
                    provider_invocation_id=invocation.invocation_id,
                )
            )
        if fallback_used and not any(
            item.kind in {"deterministic_fallback", "provider_fallback"}
            for item in fallbacks
        ):
            fallbacks.append(
                ComponentFallbackRecord(
                    fallback_index=len(fallbacks) + 1,
                    kind=(
                        "deterministic_fallback"
                        if route == "deterministic_fallback"
                        else "provider_fallback"
                    ),
                    from_route="provider",
                    to_route=(
                        "deterministic"
                        if route in {"deterministic_fallback", "mixed"}
                        else route
                    ),
                    reason=fallback_reason or "component fallback",
                    authorized=False,
                )
            )
        counts = _trace_counts(invocations)
        resolved_fallback_reason = fallback_reason or (
            fallbacks[0].reason if fallbacks else None
        )
        return cls(
            component=component,
            component_name=component_name,
            gateway_call_id=gateway_call_id,
            route=route,
            status=status,
            provider_invocations=invocations,
            fallback_events=fallbacks,
            fallback_used=bool(fallbacks),
            fallback_reason=resolved_fallback_reason,
            metadata=dict(metadata or {}),
            **counts,
        )

    @model_validator(mode="after")
    def validate_counts(self) -> ComponentInvocationTrace:
        invocation_indices = [
            item.invocation_index for item in self.provider_invocations
        ]
        if invocation_indices != list(range(1, len(invocation_indices) + 1)):
            raise ValueError(
                "provider invocation indices must be contiguous and start at one"
            )
        invocation_ids = [item.invocation_id for item in self.provider_invocations]
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError("provider invocation ids must be unique")
        seen_invocation_ids: set[str] = set()
        for invocation in self.provider_invocations:
            if invocation.kind == "primary" and invocation.parent_invocation_id:
                raise ValueError("primary provider invocation cannot have a parent")
            if invocation.kind in {
                "json_repair",
                "schema_repair",
                "protocol_fallback",
                "logical_retry",
            } and invocation.parent_invocation_id is None:
                raise ValueError(
                    f"{invocation.kind} provider invocation requires a parent"
                )
            if (
                invocation.parent_invocation_id is not None
                and invocation.parent_invocation_id not in seen_invocation_ids
            ):
                raise ValueError(
                    "provider invocation parent must reference an earlier invocation"
                )
            seen_invocation_ids.add(invocation.invocation_id)
        expected = _trace_counts(self.provider_invocations)
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"{field_name} must equal derived invocation count "
                    f"{expected_value}"
                )
        fallback_indices = [item.fallback_index for item in self.fallback_events]
        if fallback_indices != list(range(1, len(fallback_indices) + 1)):
            raise ValueError(
                "fallback event indices must be contiguous and start at one"
            )
        unknown_fallback_invocations = sorted(
            {
                item.provider_invocation_id
                for item in self.fallback_events
                if item.provider_invocation_id is not None
                and item.provider_invocation_id not in seen_invocation_ids
            }
        )
        if unknown_fallback_invocations:
            raise ValueError(
                "fallback events reference unknown provider invocations: "
                f"{unknown_fallback_invocations}"
            )
        protocol_invocation_ids = {
            item.invocation_id
            for item in self.provider_invocations
            if item.kind == "protocol_fallback"
        }
        recorded_protocol_ids = {
            item.provider_invocation_id
            for item in self.fallback_events
            if item.kind == "protocol_fallback"
        }
        if protocol_invocation_ids != recorded_protocol_ids:
            raise ValueError(
                "protocol fallback events must exactly match protocol fallback "
                "provider invocations"
            )
        if self.fallback_used != bool(self.fallback_events):
            raise ValueError("fallback_used must reflect fallback_events")
        if self.fallback_used and self.route not in {
            "deterministic_fallback",
            "mixed",
        }:
            raise ValueError("fallback_used requires a fallback route")
        if self.fallback_used and not self.fallback_reason:
            raise ValueError("fallback_used requires a fallback reason")
        if self.route in {"deterministic", "deterministic_preflight"}:
            if self.provider_invocations:
                raise ValueError(
                    "deterministic component routes cannot include provider calls"
                )
        if self.route == "provider" and not self.provider_invocations:
            raise ValueError("provider route requires a provider invocation")
        if self.route == "mixed" and not self.provider_invocations:
            raise ValueError("mixed route requires a provider invocation")
        attempted_invocations = [
            item
            for item in self.provider_invocations
            if item.status != "skipped"
        ]
        has_non_protocol_fallback = any(
            item.kind in {"deterministic_fallback", "provider_fallback"}
            for item in self.fallback_events
        )
        if (
            self.status == "completed"
            and self.route in {"provider", "mixed"}
            and not has_non_protocol_fallback
            and (
                not attempted_invocations
                or attempted_invocations[-1].status != "completed"
            )
        ):
            raise ValueError(
                "completed provider route requires a completed causal leaf"
            )
        sensitive_keys = _sensitive_trace_keys(self.metadata)
        if sensitive_keys:
            raise ValueError(
                f"component trace metadata contain sensitive fields: {sensitive_keys}"
            )
        return self


def _trace_counts(
    invocations: list[ProviderInvocationRecord],
) -> dict[str, int]:
    logical_calls = [item for item in invocations if item.status != "skipped"]
    attempted = [
        item
        for item in logical_calls
        if item.status != "pre_transport_failed"
    ]
    transport_retries = sum(
        max(0, item.transport_attempt_count - 1) for item in attempted
    )
    return {
        "logical_provider_call_count": len(logical_calls),
        "provider_call_count": sum(
            item.transport_attempt_count for item in attempted
        ),
        "retry_count": transport_retries,
        "logical_retry_count": sum(
            item.kind == "logical_retry" for item in logical_calls
        ),
        "json_repair_count": sum(
            item.kind == "json_repair" for item in logical_calls
        ),
        "schema_repair_count": sum(
            item.kind == "schema_repair" for item in logical_calls
        ),
    }


def component_invocation_trace_blockers(
    policy: ComponentInvocationAdmissionPolicy,
    trace: ComponentInvocationTrace | None,
    *,
    component: Literal["reasoner", "planner", "perception"],
    gateway_call_id: str | None = None,
) -> list[str]:
    if trace is None:
        if policy.trace_mode == "required":
            return [f"{component}_invocation_trace_required"]
        return []

    try:
        trace = validate_component_invocation_trace(trace)
    except Exception:
        return [f"{component}_invocation_trace_invalid"]

    blockers: list[str] = []
    if trace.component != component:
        blockers.append(f"{component}_invocation_trace_component_mismatch")
    if trace.status != "completed":
        blockers.append(f"{component}_invocation_failed")
    if policy.trace_mode == "required" and gateway_call_id is not None:
        if trace.gateway_call_id is None:
            blockers.append(f"{component}_invocation_trace_unbound")
        elif trace.gateway_call_id != gateway_call_id:
            blockers.append(f"{component}_invocation_trace_call_mismatch")
    if trace.provider_call_count > policy.provider_call_limit:
        blockers.append(
            f"{component}_provider_call_limit_exceeded:"
            f"{trace.provider_call_count}>{policy.provider_call_limit}"
        )
    if policy.provider_retry_mode == "disabled" and (
        trace.retry_count or trace.logical_retry_count
    ):
        blockers.append(f"{component}_provider_retry_forbidden")
    if policy.json_repair_mode == "disabled" and trace.json_repair_count:
        blockers.append(f"{component}_json_repair_forbidden")
    if policy.schema_repair_mode == "disabled" and trace.schema_repair_count:
        blockers.append(f"{component}_schema_repair_forbidden")
    if policy.fallback_mode == "disabled" and trace.fallback_used:
        blockers.append(f"{component}_fallback_forbidden")
    return blockers


def validate_component_invocation_trace(
    value: Any,
) -> ComponentInvocationTrace:
    """Rebuild a trace so mutated model instances cannot bypass validators."""

    if isinstance(value, ComponentInvocationTrace):
        value = value.model_dump(mode="python")
    return ComponentInvocationTrace.model_validate(value)


def _protocol_fallback_target(purpose: str) -> str:
    marker = "response_format_fallback:"
    if marker in purpose:
        return purpose.rsplit(marker, 1)[-1]
    return "alternate_protocol"


def _sensitive_trace_keys(value: Any, *, path: str = "") -> list[str]:
    if isinstance(value, dict):
        findings: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.strip().lower()
            child_path = f"{path}.{key_text}" if path else key_text
            if normalized in {
                "api_key",
                "authorization",
                "bearer_token",
                "prompt_payload",
                "raw_output",
                "raw_response",
                "raw_text",
                "request_payload",
                "response_payload",
                "secret",
                "system_prompt",
                "user_prompt",
                "user_text",
            } or any(
                part in normalized
                for part in {
                    "benchmark",
                    "ground_truth",
                    "oracle",
                    "prompt_payload",
                    "raw_output",
                    "raw_response",
                    "request_payload",
                    "response_payload",
                }
            ):
                findings.append(child_path)
            findings.extend(_sensitive_trace_keys(item, path=child_path))
        return findings
    if isinstance(value, list):
        findings: list[str] = []
        for index, item in enumerate(value):
            findings.extend(
                _sensitive_trace_keys(item, path=f"{path}[{index}]")
            )
        return findings
    return []


__all__ = [
    "ComponentInvocationRoute",
    "ComponentInvocationAdmissionPolicy",
    "ComponentInvocationStatus",
    "ComponentInvocationTrace",
    "ComponentFallbackKind",
    "ComponentFallbackRecord",
    "ProviderInvocationKind",
    "ProviderInvocationRecord",
    "ProviderInvocationStatus",
    "component_invocation_trace_blockers",
    "validate_component_invocation_trace",
]
