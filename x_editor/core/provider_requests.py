"""Provider request telemetry and component-scoped admission controls."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

ProviderComponent = Literal["reasoner", "planner", "perception", "unspecified"]
ScheduledProviderComponent = Literal["reasoner", "planner"]
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class ProviderQueueTimeout(TimeoutError):
    """Raised when a request cannot enter its component queue budget."""

    def __init__(self, request_id: str, component: str, elapsed_sec: float) -> None:
        self.request_id = request_id
        self.component = component
        self.elapsed_sec = elapsed_sec
        super().__init__(
            f"{component} request {request_id} exceeded its client queue timeout "
            f"after {elapsed_sec:.3f}s"
        )


class ProviderLogicalDeadlineExceeded(TimeoutError):
    """Raised when one admitted logical provider call exhausts its shared budget."""

    def __init__(
        self,
        *,
        request_id: str,
        component: str,
        timeout_sec: float,
        elapsed_sec: float,
        phase: str,
        server_termination_status: Literal["not_applicable", "unconfirmed"],
    ) -> None:
        self.request_id = request_id
        self.component = component
        self.timeout_sec = timeout_sec
        self.elapsed_sec = elapsed_sec
        self.phase = phase
        self.server_termination_status = server_termination_status
        super().__init__(
            f"{component} request {request_id} exhausted its {timeout_sec:.3f}s "
            f"logical deadline during {phase} after {elapsed_sec:.3f}s; "
            f"server termination is {server_termination_status}"
        )


@dataclass(frozen=True)
class ProviderLogicalDeadline:
    """One absolute post-admission budget shared by all work in a logical call."""

    request_id: str
    component: str
    timeout_sec: float
    started_at_monotonic: float

    @classmethod
    def start(
        cls,
        *,
        request_id: str,
        component: str,
        timeout_sec: float,
        started_at_monotonic: float | None = None,
    ) -> ProviderLogicalDeadline:
        if timeout_sec <= 0:
            raise ValueError("logical provider timeout must be positive")
        return cls(
            request_id=request_id,
            component=component,
            timeout_sec=float(timeout_sec),
            started_at_monotonic=(
                time.monotonic()
                if started_at_monotonic is None
                else float(started_at_monotonic)
            ),
        )

    @property
    def elapsed_sec(self) -> float:
        return max(0.0, time.monotonic() - self.started_at_monotonic)

    @property
    def remaining_sec(self) -> float:
        return max(0.0, self.timeout_sec - self.elapsed_sec)

    def require_remaining(
        self,
        *,
        phase: str,
        server_termination_status: Literal["not_applicable", "unconfirmed"] = (
            "not_applicable"
        ),
    ) -> float:
        remaining = self.remaining_sec
        if remaining <= 0:
            raise ProviderLogicalDeadlineExceeded(
                request_id=self.request_id,
                component=self.component,
                timeout_sec=self.timeout_sec,
                elapsed_sec=self.elapsed_sec,
                phase=phase,
                server_termination_status=server_termination_status,
            )
        return remaining


class ComponentRequestBudgetV2(StrictModel):
    """Independent timeout and concurrency budget for one model component."""

    schema_version: Literal["x-editor.component-request-budget/v2"] = (
        "x-editor.component-request-budget/v2"
    )
    component: ScheduledProviderComponent
    max_concurrency: int = Field(ge=1, le=256)
    request_timeout_sec: float = Field(gt=0, le=3600)
    queue_timeout_sec: float | None = Field(default=None, gt=0, le=3600)


class ProviderSchedulingConfigV2(StrictModel):
    """Optional v2 scheduling overlay; legacy profiles remain authoritative without it."""

    schema_version: Literal["x-editor.provider-scheduling/v2"] = (
        "x-editor.provider-scheduling/v2"
    )
    planner: ComponentRequestBudgetV2
    reasoner: ComponentRequestBudgetV2

    @model_validator(mode="after")
    def validate_components(self) -> ProviderSchedulingConfigV2:
        if self.planner.component != "planner":
            raise ValueError("planner budget must use component=planner")
        if self.reasoner.component != "reasoner":
            raise ValueError("reasoner budget must use component=reasoner")
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def budget(self, component: ScheduledProviderComponent) -> ComponentRequestBudgetV2:
        return self.planner if component == "planner" else self.reasoner


def load_provider_scheduling_config(path: str | Path) -> ProviderSchedulingConfigV2:
    """Load one explicit v2 scheduling overlay from JSON."""

    resolved = Path(path).expanduser().resolve()
    return ProviderSchedulingConfigV2.model_validate_json(
        resolved.read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class ProviderAdmission:
    request_id: str
    component: str
    enqueued_at_unix_sec: float
    admitted_at_unix_sec: float
    queue_wait_sec: float
    active_requests: int
    max_concurrency: int


class ProviderRequestScheduler:
    """Thread-safe admission gate shared by clients for one component."""

    def __init__(
        self,
        *,
        component: ScheduledProviderComponent,
        max_concurrency: int,
        queue_timeout_sec: float | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if queue_timeout_sec is not None and queue_timeout_sec <= 0:
            raise ValueError("queue_timeout_sec must be positive")
        self.component = component
        self.max_concurrency = int(max_concurrency)
        self.queue_timeout_sec = queue_timeout_sec
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._state_lock = threading.Lock()
        self._active_requests = 0

    def acquire(self, request_id: str, *, enqueued_at_unix_sec: float) -> ProviderAdmission:
        started = time.monotonic()
        if self.queue_timeout_sec is None:
            acquired = self._semaphore.acquire()
        else:
            acquired = self._semaphore.acquire(timeout=self.queue_timeout_sec)
        elapsed = time.monotonic() - started
        if not acquired:
            raise ProviderQueueTimeout(request_id, self.component, elapsed)
        with self._state_lock:
            self._active_requests += 1
            active_requests = self._active_requests
        return ProviderAdmission(
            request_id=request_id,
            component=self.component,
            enqueued_at_unix_sec=enqueued_at_unix_sec,
            admitted_at_unix_sec=time.time(),
            queue_wait_sec=elapsed,
            active_requests=active_requests,
            max_concurrency=self.max_concurrency,
        )

    def release(self) -> None:
        with self._state_lock:
            if self._active_requests < 1:
                raise RuntimeError("provider scheduler released without an active request")
            self._active_requests -= 1
        self._semaphore.release()


@dataclass
class ProviderRequestTelemetry:
    """Mutable local trace for one logical provider request."""

    request_id: str
    component: ProviderComponent
    provider: str
    model: str
    timeout_sec: float
    enqueued_at_unix_sec: float = field(default_factory=time.time)
    _started_at_monotonic: float = field(default_factory=time.monotonic, repr=False)
    terminal_status: Literal["pending", "response", "timeout", "error"] = "pending"
    phases: list[dict[str, Any]] = field(default_factory=list)
    server_metrics: dict[str, Any] = field(default_factory=dict)
    queue_wait_sec: float = 0.0
    http_wall_sec: float = 0.0
    logical_request_wall_sec: float = 0.0
    timeout_kind: Literal[
        "queue_timeout",
        "http_transport_timeout",
        "logical_request_deadline",
    ] | None = None
    server_termination_status: Literal[
        "not_applicable",
        "unconfirmed",
        "confirmed",
    ] | None = None
    _admitted_at_monotonic: float | None = field(default=None, repr=False)

    @classmethod
    def start(
        cls,
        *,
        request_id: str,
        component: ProviderComponent,
        provider: str,
        model: str,
        timeout_sec: float,
    ) -> ProviderRequestTelemetry:
        telemetry = cls(
            request_id=request_id,
            component=component,
            provider=provider,
            model=model,
            timeout_sec=timeout_sec,
        )
        telemetry.record_phase("client_queue_entered")
        return telemetry

    def record_phase(self, phase: str, **details: Any) -> None:
        record: dict[str, Any] = {
            "phase": phase,
            "at_unix_sec": round(time.time(), 6),
            "elapsed_sec": round(time.monotonic() - self._started_at_monotonic, 6),
        }
        cleaned_details = {key: value for key, value in details.items() if value is not None}
        if cleaned_details:
            record["details"] = cleaned_details
        self.phases.append(record)

    def admitted(self, admission: ProviderAdmission | None) -> float:
        admitted_at_monotonic = time.monotonic()
        self._admitted_at_monotonic = admitted_at_monotonic
        details: dict[str, Any] = {}
        if admission is not None:
            self.queue_wait_sec = admission.queue_wait_sec
            details = {
                "queue_wait_sec": round(admission.queue_wait_sec, 6),
                "active_requests": admission.active_requests,
                "max_concurrency": admission.max_concurrency,
            }
        else:
            self.queue_wait_sec = max(
                0.0,
                admitted_at_monotonic - self._started_at_monotonic,
            )
            details = {"queue_wait_sec": round(self.queue_wait_sec, 6)}
        self.record_phase("client_admitted", **details)
        return admitted_at_monotonic

    def dispatched(self) -> None:
        self.record_phase("request_dispatched")

    def http_attempt_started(
        self,
        *,
        attempt_index: int,
        attempt_in_invocation: int,
        purpose: str,
        timeout_budget_sec: float,
    ) -> float:
        started_at_monotonic = time.monotonic()
        self.record_phase(
            "http_attempt_started",
            attempt_index=attempt_index,
            attempt_in_invocation=attempt_in_invocation,
            purpose=purpose,
            timeout_budget_sec=round(timeout_budget_sec, 6),
        )
        return started_at_monotonic

    def http_attempt_finished(
        self,
        *,
        started_at_monotonic: float,
        attempt_index: int,
        attempt_in_invocation: int,
        purpose: str,
        outcome: Literal["response", "timeout", "error"],
        http_status: int | None = None,
        error_type: str | None = None,
        timeout_kind: str | None = None,
        server_termination_status: str | None = None,
    ) -> float:
        elapsed_sec = max(0.0, time.monotonic() - started_at_monotonic)
        self.http_wall_sec += elapsed_sec
        self.record_phase(
            "http_attempt_finished",
            attempt_index=attempt_index,
            attempt_in_invocation=attempt_in_invocation,
            purpose=purpose,
            outcome=outcome,
            http_wall_sec=round(elapsed_sec, 6),
            http_status=http_status,
            error_type=error_type,
            timeout_kind=timeout_kind,
            server_termination_status=server_termination_status,
        )
        return elapsed_sec

    def completed(self) -> None:
        self.terminal_status = "response"
        self.server_termination_status = "not_applicable"
        self._finish_logical_wall()
        self.record_phase("response_received")

    def failed(
        self,
        *,
        timed_out: bool,
        error_type: str,
        timeout_kind: Literal[
            "queue_timeout",
            "http_transport_timeout",
            "logical_request_deadline",
        ]
        | None = None,
        server_termination_status: Literal[
            "not_applicable",
            "unconfirmed",
            "confirmed",
        ]
        | None = None,
        queue_wait_sec: float | None = None,
    ) -> None:
        self.terminal_status = "timeout" if timed_out else "error"
        self.timeout_kind = timeout_kind
        self.server_termination_status = server_termination_status
        if queue_wait_sec is not None:
            self.queue_wait_sec = max(0.0, queue_wait_sec)
        self._finish_logical_wall()
        self.record_phase(
            "request_timed_out" if timed_out else "request_failed",
            error_type=error_type,
            timeout_kind=timeout_kind,
            server_termination_status=server_termination_status,
        )

    def _finish_logical_wall(self) -> None:
        if self._admitted_at_monotonic is None:
            self.logical_request_wall_sec = 0.0
            return
        self.logical_request_wall_sec = max(
            0.0,
            time.monotonic() - self._admitted_at_monotonic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "x-editor.provider-request-telemetry/v3",
            "request_id": self.request_id,
            "component": self.component,
            "provider": self.provider,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
            "terminal_status": self.terminal_status,
            "queue_wait_sec": round(self.queue_wait_sec, 6),
            "http_wall_sec": round(self.http_wall_sec, 6),
            "logical_request_wall_sec": round(self.logical_request_wall_sec, 6),
            "timeout_kind": self.timeout_kind,
            "server_termination_status": self.server_termination_status,
            "phases": list(self.phases),
            "server_metrics": dict(self.server_metrics),
        }


class ProviderSchedulingRuntime:
    """Runtime schedulers derived from one immutable v2 scheduling config."""

    def __init__(self, config: ProviderSchedulingConfigV2) -> None:
        self.config = config
        self._schedulers = {
            component: ProviderRequestScheduler(
                component=component,
                max_concurrency=config.budget(component).max_concurrency,
                queue_timeout_sec=config.budget(component).queue_timeout_sec,
            )
            for component in ("planner", "reasoner")
        }

    def scheduler(self, component: ScheduledProviderComponent) -> ProviderRequestScheduler:
        return self._schedulers[component]

    def configure_client(self, client: Any, component: ScheduledProviderComponent) -> Any:
        """Bind an OpenAI-compatible client without changing its provider/model identity."""

        required = ("timeout", "component_name", "request_scheduler")
        missing = [name for name in required if not hasattr(client, name)]
        if missing:
            raise TypeError(
                "provider scheduling requires an OpenAI-compatible client with "
                f"attributes {required}; missing {missing}"
            )
        budget = self.config.budget(component)
        client.timeout = budget.request_timeout_sec
        client.component_name = component
        client.request_scheduler = self.scheduler(component)
        return client


def new_provider_request_id(component: str) -> str:
    normalized = component.strip().lower() or "unspecified"
    return f"xed-{normalized}-{uuid.uuid4().hex}"


def normalize_provider_request_id(request_id: str | None, *, component: str) -> str:
    value = (request_id or "").strip() or new_provider_request_id(component)
    if _REQUEST_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "request_id must be 1-128 characters using letters, digits, '.', '_', ':', '/', or '-'"
        )
    return value


__all__ = [
    "ComponentRequestBudgetV2",
    "ProviderAdmission",
    "ProviderLogicalDeadline",
    "ProviderLogicalDeadlineExceeded",
    "ProviderQueueTimeout",
    "ProviderRequestScheduler",
    "ProviderRequestTelemetry",
    "ProviderSchedulingConfigV2",
    "ProviderSchedulingRuntime",
    "load_provider_scheduling_config",
    "new_provider_request_id",
    "normalize_provider_request_id",
]
