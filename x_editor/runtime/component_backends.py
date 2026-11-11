"""Provider-independent component backend contracts and explicit factories.

The contracts in this module deliberately stop before workflow assembly.  They
describe frozen provider bindings and construct backend instances without
selecting a model from its name, a benchmark identity, or an implicit fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator, model_validator

from x_editor.core.schemas import StrictModel

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ComponentBackendError(RuntimeError):
    """Base error for explicit component backend resolution."""


class DuplicateBackendRegistrationError(ComponentBackendError):
    """Raised when a factory identifier is registered more than once."""


class UnknownBackendError(ComponentBackendError):
    """Raised when a frozen profile references an unregistered factory."""


class BackendContractError(ComponentBackendError):
    """Raised when a factory returns an object for a different frozen config."""


class UnknownPerceptionCapabilityError(ComponentBackendError):
    """Raised when no explicitly configured backend owns a perception route."""


class _FrozenStrictModel(StrictModel):
    """Immutable base for reproducible runtime declarations."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _fingerprint(model: _FrozenStrictModel) -> str:
    payload = model.model_dump(mode="json", exclude={"fingerprint"})
    _remove_absent_v1_extensions(payload)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _remove_absent_v1_extensions(value: Any) -> None:
    """Keep fingerprints stable for profiles authored before optional v1 fields."""

    if isinstance(value, dict):
        for key in ("request_dialect", "reasoner", "evaluator"):
            if value.get(key) is None:
                value.pop(key, None)
        for item in value.values():
            _remove_absent_v1_extensions(item)
    elif isinstance(value, list):
        for item in value:
            _remove_absent_v1_extensions(item)


def _freeze_fingerprint(model: _FrozenStrictModel) -> None:
    expected = _fingerprint(model)
    observed = getattr(model, "fingerprint", "")
    if observed and observed != expected:
        raise ValueError("fingerprint does not match the frozen configuration")
    object.__setattr__(model, "fingerprint", expected)


def _require_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or re.fullmatch(_IDENTIFIER_PATTERN, normalized) is None:
        raise ValueError(
            f"{field_name} must contain only letters, digits, '.', '_', ':', '/', or '-'"
        )
    return normalized


def _normalize_capabilities(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_require_identifier(value, field_name="capability") for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("capabilities must be unique")
    return tuple(sorted(normalized))


class BackendAuthentication(_FrozenStrictModel):
    """Authentication declaration that records only a secret reference."""

    schema_version: Literal["x-editor.backend-authentication/v1"] = (
        "x-editor.backend-authentication/v1"
    )
    mode: Literal["none", "bearer"] = "none"
    credential_env: str | None = None

    @model_validator(mode="after")
    def validate_credential_reference(self) -> BackendAuthentication:
        if self.mode == "none":
            if self.credential_env is not None:
                raise ValueError("credential_env is forbidden when authentication is none")
            return self
        if self.credential_env is None or not self.credential_env.strip():
            raise ValueError("bearer authentication requires credential_env")
        normalized = self.credential_env.strip()
        if _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(normalized) is None:
            raise ValueError("credential_env must be a valid environment variable name")
        object.__setattr__(self, "credential_env", normalized)
        return self


class BackendResponseFormat(_FrozenStrictModel):
    """Frozen response-format mode; unsupported formats are not retried."""

    schema_version: Literal["x-editor.backend-response-format/v1"] = (
        "x-editor.backend-response-format/v1"
    )
    mode: Literal["json_schema", "json_object", "text"]
    strict_schema: bool = False
    fallback_mode: Literal["disabled"] = "disabled"

    @model_validator(mode="after")
    def validate_strict_schema_mode(self) -> BackendResponseFormat:
        if self.strict_schema and self.mode != "json_schema":
            raise ValueError("strict_schema is only valid for json_schema mode")
        return self


class ProviderRequestDialect(_FrozenStrictModel):
    """Provider-specific HTTP details that do not alter module contracts."""

    schema_version: Literal["x-editor.provider-request-dialect/v1"] = (
        "x-editor.provider-request-dialect/v1"
    )
    audio_content_part: Literal["input_audio", "audio_url"] = "input_audio"
    request_body: dict[str, Any] = Field(default_factory=dict, max_length=16)

    @field_validator("request_body")
    @classmethod
    def validate_request_body(cls, values: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "seed",
            "chat_template_kwargs",
            "response_format",
            "tools",
            "tool_choice",
        }
        normalized = {str(key).strip(): item for key, item in values.items()}
        if any(not key for key in normalized):
            raise ValueError("request_body keys must not be blank")
        conflicts = sorted(set(normalized) & reserved)
        if conflicts:
            raise ValueError(
                "request_body cannot override canonical request fields: "
                + ", ".join(conflicts)
            )
        try:
            json.dumps(normalized, ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("request_body must contain JSON-serializable values") from exc
        return normalized


class ProviderTransportConfig(_FrozenStrictModel):
    """One frozen provider/model/endpoint binding used by component factories."""

    schema_version: Literal["x-editor.provider-transport-config/v1"] = (
        "x-editor.provider-transport-config/v1"
    )
    transport_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    transport_kind: str = Field(pattern=_IDENTIFIER_PATTERN)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    authentication: BackendAuthentication = Field(default_factory=BackendAuthentication)
    response_format: BackendResponseFormat
    timeout_sec: float = Field(default=120.0, gt=0, le=1200)
    max_output_tokens: int = Field(default=8192, gt=0, le=131072)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    chat_template_kwargs: dict[str, bool | str | int | float] = Field(default_factory=dict)
    request_dialect: ProviderRequestDialect | None = None
    max_transport_retries: Literal[0] = 0
    fingerprint: str = ""

    @field_validator("transport_id", "transport_kind")
    @classmethod
    def normalize_identifiers(cls, value: str, info: Any) -> str:
        return _require_identifier(value, field_name=info.field_name)

    @field_validator("provider", "model")
    @classmethod
    def normalize_non_empty_names(cls, value: str, info: Any) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be blank")
        return normalized

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an absolute http(s) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint must not embed credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain a query or fragment")
        return normalized

    @field_validator("chat_template_kwargs")
    @classmethod
    def validate_chat_template_kwargs(
        cls,
        values: dict[str, bool | str | int | float],
    ) -> dict[str, bool | str | int | float]:
        if any(not str(key).strip() for key in values):
            raise ValueError("chat_template_kwargs keys must not be blank")
        return {str(key).strip(): value for key, value in values.items()}

    @model_validator(mode="after")
    def freeze_fingerprint(self) -> ProviderTransportConfig:
        _freeze_fingerprint(self)
        return self


class PlannerBackendConfig(_FrozenStrictModel):
    """Frozen Planner adapter selection and the capabilities it promises."""

    schema_version: Literal["x-editor.planner-backend-config/v1"] = (
        "x-editor.planner-backend-config/v1"
    )
    component: Literal["planner"] = "planner"
    backend_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    transport_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    capabilities: tuple[str, ...] = Field(min_length=1)
    fingerprint: str = ""

    @field_validator("backend_id", "transport_id")
    @classmethod
    def normalize_identifiers(cls, value: str, info: Any) -> str:
        return _require_identifier(value, field_name=info.field_name)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_capabilities(values)

    @model_validator(mode="after")
    def freeze_fingerprint(self) -> PlannerBackendConfig:
        _freeze_fingerprint(self)
        return self


class ReasonerBackendConfig(_FrozenStrictModel):
    """Frozen semantic Reasoner adapter and its stable output contract."""

    schema_version: Literal["x-editor.reasoner-backend-config/v1"] = (
        "x-editor.reasoner-backend-config/v1"
    )
    component: Literal["reasoner"] = "reasoner"
    backend_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    transport_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    capabilities: tuple[str, ...] = Field(min_length=1)
    strategy: Literal["memo_only", "memo_only_brief", "prose_memo", "evidence_memo", "draft_memo"] = "memo_only"
    evidence_review_max_output_tokens: int = Field(default=4096, ge=512, le=32768)
    fingerprint: str = ""

    @field_validator("backend_id", "transport_id")
    @classmethod
    def normalize_identifiers(cls, value: str, info: Any) -> str:
        return _require_identifier(value, field_name=info.field_name)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_capabilities(values)

    @model_validator(mode="after")
    def freeze_fingerprint(self) -> ReasonerBackendConfig:
        _freeze_fingerprint(self)
        return self


class PerceptionBackendConfig(_FrozenStrictModel):
    """Frozen Perception adapter and the exact capability routes it owns."""

    schema_version: Literal["x-editor.perception-backend-config/v1"] = (
        "x-editor.perception-backend-config/v1"
    )
    component: Literal["perception"] = "perception"
    backend_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    transport_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    capabilities: tuple[str, ...] = Field(min_length=1)
    fingerprint: str = ""

    @field_validator("backend_id", "transport_id")
    @classmethod
    def normalize_identifiers(cls, value: str, info: Any) -> str:
        return _require_identifier(value, field_name=info.field_name)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_capabilities(values)

    @model_validator(mode="after")
    def freeze_fingerprint(self) -> PerceptionBackendConfig:
        _freeze_fingerprint(self)
        return self


class EvaluatorBackendConfig(_FrozenStrictModel):
    """Frozen sparse Evaluator adapter; it has no execution authority."""

    schema_version: Literal["x-editor.evaluator-backend-config/v1"] = (
        "x-editor.evaluator-backend-config/v1"
    )
    component: Literal["evaluator"] = "evaluator"
    backend_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    transport_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    capabilities: tuple[str, ...] = Field(min_length=1)
    passing_score: float = Field(default=0.75, ge=0.0, le=1.0)
    max_rubrics: int = Field(default=5, ge=1, le=32)
    fingerprint: str = ""

    @field_validator("backend_id", "transport_id")
    @classmethod
    def normalize_identifiers(cls, value: str, info: Any) -> str:
        return _require_identifier(value, field_name=info.field_name)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_capabilities(values)

    @model_validator(mode="after")
    def freeze_fingerprint(self) -> EvaluatorBackendConfig:
        _freeze_fingerprint(self)
        return self


class ComponentBackendProfile(_FrozenStrictModel):
    """Complete provider-independent declaration for one runtime assembly."""

    schema_version: Literal["x-editor.component-backend-profile/v1"] = (
        "x-editor.component-backend-profile/v1"
    )
    transports: tuple[ProviderTransportConfig, ...] = Field(min_length=1)
    reasoner: ReasonerBackendConfig | None = None
    planner: PlannerBackendConfig
    perception: tuple[PerceptionBackendConfig, ...] = ()
    evaluator: EvaluatorBackendConfig | None = None
    fingerprint: str = ""

    @model_validator(mode="after")
    def validate_bindings(self) -> ComponentBackendProfile:
        transport_ids = [transport.transport_id for transport in self.transports]
        if len(transport_ids) != len(set(transport_ids)):
            raise ValueError("transport_id values must be unique")
        known_transports = set(transport_ids)
        if self.planner.transport_id not in known_transports:
            raise ValueError("planner references an unknown transport_id")
        if self.reasoner is not None and self.reasoner.transport_id not in known_transports:
            raise ValueError("reasoner references an unknown transport_id")
        if self.evaluator is not None and self.evaluator.transport_id not in known_transports:
            raise ValueError("evaluator references an unknown transport_id")

        capability_owners: dict[str, str] = {}
        for backend in self.perception:
            if backend.transport_id not in known_transports:
                raise ValueError(
                    f"perception backend {backend.backend_id!r} references an unknown transport_id"
                )
            for capability in backend.capabilities:
                previous = capability_owners.get(capability)
                if previous is not None:
                    raise ValueError(
                        f"perception capability {capability!r} is owned by both "
                        f"{previous!r} and {backend.backend_id!r}"
                    )
                capability_owners[capability] = backend.backend_id

        referenced_transports = {
            self.planner.transport_id,
            *(backend.transport_id for backend in self.perception),
        }
        if self.reasoner is not None:
            referenced_transports.add(self.reasoner.transport_id)
        if self.evaluator is not None:
            referenced_transports.add(self.evaluator.transport_id)
        unused = known_transports - referenced_transports
        if unused:
            raise ValueError(f"unreferenced transports are forbidden: {sorted(unused)}")
        _freeze_fingerprint(self)
        return self

    def transport(self, transport_id: str) -> ProviderTransportConfig:
        for transport in self.transports:
            if transport.transport_id == transport_id:
                return transport
        raise KeyError(f"unknown transport_id: {transport_id}")

    def perception_backend(self, capability: str) -> PerceptionBackendConfig:
        for backend in self.perception:
            if capability in backend.capabilities:
                return backend
        raise UnknownPerceptionCapabilityError(
            f"no perception backend is configured for capability {capability!r}"
        )


@runtime_checkable
class ProviderTransport(Protocol):
    """Provider call boundary implemented by local or remote transports."""

    @property
    def config(self) -> ProviderTransportConfig:
        """Return the exact frozen transport config used by this instance."""

    def invoke(self, request: Any) -> Any:
        """Invoke the provider once; the result may be awaitable."""


@runtime_checkable
class PlannerBackend(Protocol):
    """Planner boundary shared by the direct and workflow-turn adapters."""

    @property
    def config(self) -> PlannerBackendConfig:
        """Return the exact frozen backend config used by this instance."""

    def plan_workflow_turn(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Return one planner decision; the result may be awaitable."""

    def plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Return one direct plan for workflow arms that do not use turn decisions."""


@runtime_checkable
class ReasonerBackend(Protocol):
    """Semantic Reasoner boundary used before executable planning."""

    @property
    def config(self) -> ReasonerBackendConfig:
        """Return the exact frozen backend config used by this instance."""

    def reason(self, request: Any) -> Any:
        """Return the existing typed semantic memo contract."""


@runtime_checkable
class PerceptionCapabilityBackend(Protocol):
    """One backend serving only its explicitly declared perception routes."""

    @property
    def config(self) -> PerceptionBackendConfig:
        """Return the exact frozen backend config used by this instance."""

    def perceive(self, request: Any, *, context: dict[str, Any]) -> Any:
        """Return typed perception evidence; the result may be awaitable."""


@runtime_checkable
class EvaluatorBackend(Protocol):
    """Sparse post-delivery evaluation boundary without runtime authority."""

    @property
    def config(self) -> EvaluatorBackendConfig:
        """Return the exact frozen backend config used by this instance."""

    def evaluate(self, plan: Any, context: dict[str, Any] | None = None) -> Any:
        """Return the existing EvaluationResult contract."""


ProviderTransportFactory = Callable[[ProviderTransportConfig], ProviderTransport]
ReasonerBackendFactory = Callable[[ReasonerBackendConfig, ProviderTransport], ReasonerBackend]
PlannerBackendFactory = Callable[[PlannerBackendConfig, ProviderTransport], PlannerBackend]
PerceptionBackendFactory = Callable[
    [PerceptionBackendConfig, ProviderTransport],
    PerceptionCapabilityBackend,
]
EvaluatorBackendFactory = Callable[
    [EvaluatorBackendConfig, ProviderTransport],
    EvaluatorBackend,
]


@dataclass(frozen=True)
class ResolvedComponentBackends:
    """Constructed backend set with immutable, exact capability routing."""

    profile: ComponentBackendProfile
    planner: PlannerBackend
    transports: Mapping[str, ProviderTransport]
    perception_by_capability: Mapping[str, PerceptionCapabilityBackend]
    reasoner: ReasonerBackend | None = None
    evaluator: EvaluatorBackend | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transports", MappingProxyType(dict(self.transports)))
        object.__setattr__(
            self,
            "perception_by_capability",
            MappingProxyType(dict(self.perception_by_capability)),
        )

    def perception_for(self, capability: str) -> PerceptionCapabilityBackend:
        try:
            return self.perception_by_capability[capability]
        except KeyError as exc:
            raise UnknownPerceptionCapabilityError(
                f"no resolved perception backend for capability {capability!r}"
            ) from exc


class ComponentBackendRegistry:
    """Explicit factory registry with no aliases, inference, or fallback."""

    def __init__(self) -> None:
        self._transport_factories: dict[str, ProviderTransportFactory] = {}
        self._reasoner_factories: dict[str, ReasonerBackendFactory] = {}
        self._planner_factories: dict[str, PlannerBackendFactory] = {}
        self._perception_factories: dict[str, PerceptionBackendFactory] = {}
        self._evaluator_factories: dict[str, EvaluatorBackendFactory] = {}

    def register_transport(
        self,
        transport_kind: str,
        factory: ProviderTransportFactory,
    ) -> None:
        key = _require_identifier(transport_kind, field_name="transport_kind")
        self._register(self._transport_factories, key, factory, category="transport")

    def register_planner(
        self,
        backend_id: str,
        factory: PlannerBackendFactory,
    ) -> None:
        key = _require_identifier(backend_id, field_name="backend_id")
        self._register(self._planner_factories, key, factory, category="planner")

    def register_reasoner(
        self,
        backend_id: str,
        factory: ReasonerBackendFactory,
    ) -> None:
        key = _require_identifier(backend_id, field_name="backend_id")
        self._register(self._reasoner_factories, key, factory, category="reasoner")

    def register_perception(
        self,
        backend_id: str,
        factory: PerceptionBackendFactory,
    ) -> None:
        key = _require_identifier(backend_id, field_name="backend_id")
        self._register(self._perception_factories, key, factory, category="perception")

    def register_evaluator(
        self,
        backend_id: str,
        factory: EvaluatorBackendFactory,
    ) -> None:
        key = _require_identifier(backend_id, field_name="backend_id")
        self._register(self._evaluator_factories, key, factory, category="evaluator")

    @staticmethod
    def _register(
        registry: dict[str, Any],
        key: str,
        factory: Any,
        *,
        category: str,
    ) -> None:
        if not callable(factory):
            raise TypeError(f"{category} factory must be callable")
        if key in registry:
            raise DuplicateBackendRegistrationError(
                f"{category} factory {key!r} is already registered"
            )
        registry[key] = factory

    def build_transport(self, config: ProviderTransportConfig) -> ProviderTransport:
        factory = self._transport_factories.get(config.transport_kind)
        if factory is None:
            raise UnknownBackendError(
                f"transport factory {config.transport_kind!r} is not registered"
            )
        instance = factory(config)
        self._validate_instance(
            instance,
            ProviderTransport,
            expected_config=config,
            label="transport",
        )
        return instance

    def build_planner(
        self,
        config: PlannerBackendConfig,
        transport: ProviderTransport,
    ) -> PlannerBackend:
        self._require_transport(config.transport_id, transport)
        factory = self._planner_factories.get(config.backend_id)
        if factory is None:
            raise UnknownBackendError(f"planner factory {config.backend_id!r} is not registered")
        instance = factory(config, transport)
        self._validate_instance(
            instance,
            PlannerBackend,
            expected_config=config,
            label="planner",
        )
        return instance

    def build_reasoner(
        self,
        config: ReasonerBackendConfig,
        transport: ProviderTransport,
    ) -> ReasonerBackend:
        self._require_transport(config.transport_id, transport)
        factory = self._reasoner_factories.get(config.backend_id)
        if factory is None:
            raise UnknownBackendError(f"reasoner factory {config.backend_id!r} is not registered")
        instance = factory(config, transport)
        self._validate_instance(instance, ReasonerBackend, expected_config=config, label="reasoner")
        return instance

    def build_perception(
        self,
        config: PerceptionBackendConfig,
        transport: ProviderTransport,
    ) -> PerceptionCapabilityBackend:
        self._require_transport(config.transport_id, transport)
        factory = self._perception_factories.get(config.backend_id)
        if factory is None:
            raise UnknownBackendError(f"perception factory {config.backend_id!r} is not registered")
        instance = factory(config, transport)
        self._validate_instance(
            instance,
            PerceptionCapabilityBackend,
            expected_config=config,
            label="perception",
        )
        return instance

    def build_evaluator(
        self,
        config: EvaluatorBackendConfig,
        transport: ProviderTransport,
    ) -> EvaluatorBackend:
        self._require_transport(config.transport_id, transport)
        factory = self._evaluator_factories.get(config.backend_id)
        if factory is None:
            raise UnknownBackendError(f"evaluator factory {config.backend_id!r} is not registered")
        instance = factory(config, transport)
        self._validate_instance(instance, EvaluatorBackend, expected_config=config, label="evaluator")
        return instance

    def build_profile(
        self,
        profile: ComponentBackendProfile,
    ) -> ResolvedComponentBackends:
        transports = {
            config.transport_id: self.build_transport(config) for config in profile.transports
        }
        planner = self.build_planner(
            profile.planner,
            transports[profile.planner.transport_id],
        )
        reasoner = (
            self.build_reasoner(
                profile.reasoner,
                transports[profile.reasoner.transport_id],
            )
            if profile.reasoner is not None
            else None
        )
        perception_by_capability: dict[str, PerceptionCapabilityBackend] = {}
        for config in profile.perception:
            backend = self.build_perception(config, transports[config.transport_id])
            for capability in config.capabilities:
                perception_by_capability[capability] = backend
        evaluator = (
            self.build_evaluator(
                profile.evaluator,
                transports[profile.evaluator.transport_id],
            )
            if profile.evaluator is not None
            else None
        )
        return ResolvedComponentBackends(
            profile=profile,
            planner=planner,
            transports=transports,
            perception_by_capability=perception_by_capability,
            reasoner=reasoner,
            evaluator=evaluator,
        )

    @staticmethod
    def _require_transport(
        expected_transport_id: str,
        transport: ProviderTransport,
    ) -> None:
        if not isinstance(transport, ProviderTransport):
            raise BackendContractError("factory input does not implement ProviderTransport")
        if transport.config.transport_id != expected_transport_id:
            raise BackendContractError(
                f"backend expects transport {expected_transport_id!r}, received "
                f"{transport.config.transport_id!r}"
            )

    @staticmethod
    def _validate_instance(
        instance: Any,
        protocol: type[Any],
        *,
        expected_config: Any,
        label: str,
    ) -> None:
        if not isinstance(instance, protocol):
            raise BackendContractError(
                f"{label} factory returned an object that does not implement its protocol"
            )
        if instance.config != expected_config:
            raise BackendContractError(
                f"{label} factory returned an instance for a different frozen config"
            )


__all__ = [
    "BackendAuthentication",
    "BackendContractError",
    "BackendResponseFormat",
    "ComponentBackendError",
    "ComponentBackendProfile",
    "ComponentBackendRegistry",
    "DuplicateBackendRegistrationError",
    "EvaluatorBackend",
    "EvaluatorBackendConfig",
    "EvaluatorBackendFactory",
    "PerceptionBackendConfig",
    "PerceptionBackendFactory",
    "PerceptionCapabilityBackend",
    "PlannerBackend",
    "PlannerBackendConfig",
    "PlannerBackendFactory",
    "ProviderRequestDialect",
    "ProviderTransport",
    "ProviderTransportConfig",
    "ProviderTransportFactory",
    "ReasonerBackend",
    "ReasonerBackendConfig",
    "ReasonerBackendFactory",
    "ResolvedComponentBackends",
    "UnknownBackendError",
    "UnknownPerceptionCapabilityError",
]
