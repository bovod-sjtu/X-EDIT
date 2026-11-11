"""Native provider-call accounting for Perception frontends."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Literal

from x_editor.core.invocation_trace import (
    ComponentFallbackRecord,
    ComponentInvocationTrace,
    ProviderInvocationKind,
    ProviderInvocationRecord,
)

PerceptionCallPurpose = str

_SAFE_METRIC_KEYS = {
    "attempts",
    "audio_count",
    "http_status",
    "request_elapsed_sec",
    "response_format_fallback",
    "status",
}
_SUCCESS_STATUSES = {"completed", "ok", "success"}


class PerceptionInvocationTraceRecorder:
    """Accumulate immutable client snapshots for one Perception operation."""

    def __init__(self, frontend: Any, client: Any) -> None:
        self.frontend = frontend
        self.client = client
        self.component_name = _safe_label(
            getattr(frontend, "component_name", None)
            or "perception.timestamped_caption"
        )
        self.provider = _safe_label(
            getattr(frontend, "provider_name", None)
            or getattr(client, "provider_name", None)
            or type(client).__name__
        )
        self.model = _optional_safe_label(
            getattr(frontend, "model_name", None)
            or getattr(client, "model_name", None)
        )
        self.records: list[ProviderInvocationRecord] = []
        self.fallbacks: list[ComponentFallbackRecord] = []
        self.purpose_counts: Counter[str] = Counter()
        self.purpose_failures: Counter[str] = Counter()
        self.last_purpose_failed: dict[str, bool] = {}
        self.last_review_invocation_id: str | None = None

    def capture(
        self,
        purpose: PerceptionCallPurpose,
        *,
        succeeded: bool,
        error: Exception | None = None,
    ) -> ComponentInvocationTrace:
        """Snapshot the client's transient accounting before another call occurs."""

        raw_invocations = _client_invocation_snapshot(self.client)
        semantic_kind: ProviderInvocationKind = _semantic_kind(purpose)
        semantic_parent = (
            self.last_review_invocation_id
            if semantic_kind == "logical_retry"
            else None
        )
        current_parent = semantic_parent
        call_records: list[ProviderInvocationRecord] = []
        for raw_index, raw in enumerate(raw_invocations):
            native_purpose = str(raw.get("purpose") or "primary")
            native_kind = _native_kind(native_purpose)
            kind = semantic_kind if raw_index == 0 and native_kind == "primary" else native_kind
            if kind == "primary" and purpose != "timestamped_caption.review":
                kind = "other"
            invocation_index = len(self.records) + 1
            invocation_id = f"{self.component_name}:{invocation_index}"
            parent_invocation_id = (
                current_parent
                if kind
                in {
                    "json_repair",
                    "schema_repair",
                    "protocol_fallback",
                    "logical_retry",
                }
                else None
            )
            metrics = _safe_metrics(raw.get("metrics"))
            attempt_count = _attempt_count(raw, self.client)
            record_succeeded = _record_succeeded(
                raw,
                metrics=metrics,
                call_succeeded=succeeded,
                is_last=raw_index == len(raw_invocations) - 1,
            )
            status = "completed" if record_succeeded else "failed"
            if attempt_count == 0:
                status = "pre_transport_failed"
            record = ProviderInvocationRecord(
                invocation_index=invocation_index,
                invocation_id=invocation_id,
                parent_invocation_id=parent_invocation_id,
                provider=self.provider,
                model=self.model,
                kind=kind,
                purpose=_combined_purpose(purpose, native_purpose),
                status=status,
                transport_attempt_count=attempt_count,
                metrics=metrics,
                error_type=(
                    (
                        _safe_label(type(error).__name__)
                        if error is not None
                        else _optional_safe_label(metrics.get("error_type"))
                    )
                    if not record_succeeded
                    else None
                ),
                error_message=(
                    f"provider call failed during {purpose}"
                    if not record_succeeded
                    else None
                ),
            )
            self.records.append(record)
            call_records.append(record)
            current_parent = invocation_id

        self.purpose_counts[purpose] += 1
        call_failed = not call_records or call_records[-1].status != "completed"
        self.last_purpose_failed[purpose] = call_failed
        if call_failed:
            self.purpose_failures[purpose] += 1
        if purpose in {
            "timestamped_caption.review",
            "timestamped_caption.review_after_verification",
        } and call_records:
            self.last_review_invocation_id = call_records[-1].invocation_id
        return self.publish(status="failed")

    def add_deterministic_fallback(
        self,
        *,
        from_route: str,
        to_route: str,
        reason: str,
        provider_invocation_id: str | None = None,
    ) -> ComponentInvocationTrace:
        self.fallbacks.append(
            ComponentFallbackRecord(
                fallback_index=len(self.fallbacks) + 1,
                kind="deterministic_fallback",
                from_route=_safe_label(from_route),
                to_route=_safe_label(to_route),
                reason=_safe_label(reason),
                authorized=False,
                provider_invocation_id=provider_invocation_id,
            )
        )
        return self.publish(status="failed")

    def last_call_failed(self, purpose: PerceptionCallPurpose) -> bool:
        return self.last_purpose_failed.get(purpose, False)

    def latest_invocation_id(self) -> str | None:
        return self.records[-1].invocation_id if self.records else None

    def publish(self, *, status: Literal["completed", "failed"]) -> ComponentInvocationTrace:
        route = "mixed" if self.fallbacks or _has_protocol_fallback(self.records) else "provider"
        trace = ComponentInvocationTrace.build(
            component="perception",
            component_name=self.component_name,
            route=route,
            status=status,
            provider_invocations=list(self.records),
            fallback_events=list(self.fallbacks),
            metadata={
                "operation": self.component_name,
                "provider": self.provider,
                "model": self.model,
                "call_purpose_counts": dict(sorted(self.purpose_counts.items())),
                "call_purpose_failure_counts": dict(
                    sorted(self.purpose_failures.items())
                ),
            },
        )
        self.frontend.last_invocation_trace = trace
        return trace


def capture_frontend_provider_call(
    frontend: Any,
    purpose: PerceptionCallPurpose,
    *,
    succeeded: bool,
    error: Exception | None = None,
) -> ComponentInvocationTrace | None:
    recorder = _recorder(frontend, create=True)
    if recorder is None:
        return None
    return recorder.capture(purpose, succeeded=succeeded, error=error)


def reset_frontend_invocation_trace(frontend: Any) -> None:
    if hasattr(frontend, "_perception_invocation_trace_recorder"):
        delattr(frontend, "_perception_invocation_trace_recorder")
    if hasattr(frontend, "last_invocation_trace"):
        frontend.last_invocation_trace = None
    if isinstance(getattr(frontend, "telemetry", None), dict):
        frontend.telemetry = {}


def record_frontend_deterministic_fallback(
    frontend: Any,
    *,
    from_route: str,
    to_route: str,
    reason: str,
) -> ComponentInvocationTrace | None:
    recorder = _recorder(frontend, create=False)
    if recorder is None:
        return None
    return recorder.add_deterministic_fallback(
        from_route=from_route,
        to_route=to_route,
        reason=reason,
        provider_invocation_id=recorder.latest_invocation_id(),
    )


def frontend_provider_call_failed(
    frontend: Any,
    purpose: PerceptionCallPurpose,
) -> bool:
    recorder = _recorder(frontend, create=False)
    return recorder.last_call_failed(purpose) if recorder is not None else False


def finalize_frontend_invocation_trace(
    frontend: Any,
    *,
    status: Literal["completed", "failed"],
) -> ComponentInvocationTrace | None:
    recorder = _recorder(frontend, create=False)
    if recorder is None or not recorder.records:
        return None
    return recorder.publish(status=status)


def _recorder(frontend: Any, *, create: bool) -> PerceptionInvocationTraceRecorder | None:
    existing = getattr(frontend, "_perception_invocation_trace_recorder", None)
    if isinstance(existing, PerceptionInvocationTraceRecorder):
        return existing
    if not create:
        return None
    client = getattr(frontend, "client", None)
    if client is None or not any(
        hasattr(client, field)
        for field in (
            "last_json_invocations",
            "last_call_metrics",
            "last_transport_attempt_count",
        )
    ):
        return None
    recorder = PerceptionInvocationTraceRecorder(frontend, client)
    frontend._perception_invocation_trace_recorder = recorder
    return recorder


def _client_invocation_snapshot(client: Any) -> list[dict[str, Any]]:
    raw_invocations = getattr(client, "last_json_invocations", None)
    if isinstance(raw_invocations, list) and raw_invocations:
        return [deepcopy(item) if isinstance(item, dict) else {} for item in raw_invocations]
    return [
        {
            "purpose": "primary",
            "transport_attempt_count": getattr(client, "last_transport_attempt_count", 0),
            "status": (getattr(client, "last_call_metrics", None) or {}).get("status"),
            "metrics": deepcopy(getattr(client, "last_call_metrics", None) or {}),
        }
    ]


def _safe_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in _SAFE_METRIC_KEYS:
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            safe[key] = item
    error_type = value.get("error_type")
    if error_type:
        safe["error_type"] = _safe_label(error_type)
    return safe


def _attempt_count(raw: dict[str, Any], client: Any) -> int:
    value = raw.get("transport_attempt_count")
    if value is None:
        value = getattr(client, "last_transport_attempt_count", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _record_succeeded(
    raw: dict[str, Any],
    *,
    metrics: dict[str, Any],
    call_succeeded: bool,
    is_last: bool,
) -> bool:
    if not call_succeeded and is_last:
        return False
    raw_status = str(raw.get("status") or metrics.get("status") or "").lower()
    if raw_status:
        return raw_status in _SUCCESS_STATUSES
    return call_succeeded or not is_last


def _semantic_kind(purpose: PerceptionCallPurpose) -> ProviderInvocationKind:
    if purpose in {
        "timestamped_caption.review",
        "speaker_attributed_review.review",
        "audio_fallback.semantic_review",
        "semantic_span_review.candidate_rubric",
    }:
        return "primary"
    if purpose == "timestamped_caption.review_after_verification":
        return "logical_retry"
    return "other"


def _native_kind(purpose: str) -> ProviderInvocationKind:
    normalized = purpose.lower()
    if "response_format_fallback" in normalized:
        return "protocol_fallback"
    if normalized == "schema_repair" or normalized.endswith("_schema_repair"):
        return "schema_repair"
    if normalized == "json_repair" or normalized.endswith("_json_repair"):
        return "json_repair"
    if "retry" in normalized and "repair" not in normalized:
        return "logical_retry"
    if normalized == "primary":
        return "primary"
    return "other"


def _combined_purpose(purpose: PerceptionCallPurpose, native_purpose: str) -> str:
    if native_purpose == "primary":
        return purpose
    return f"{purpose}.{_safe_label(native_purpose)}"


def _has_protocol_fallback(records: list[ProviderInvocationRecord]) -> bool:
    return any(record.kind == "protocol_fallback" for record in records)


def _safe_label(value: Any) -> str:
    label = str(value or "unknown").strip()
    label = label.replace("\n", " ").replace("\r", " ")
    return label[:160] or "unknown"


def _optional_safe_label(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return _safe_label(value)


__all__ = [
    "PerceptionCallPurpose",
    "capture_frontend_provider_call",
    "finalize_frontend_invocation_trace",
    "frontend_provider_call_failed",
    "record_frontend_deterministic_fallback",
    "reset_frontend_invocation_trace",
]
