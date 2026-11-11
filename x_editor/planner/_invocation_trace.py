"""Internal helpers for exposing planner provider work to workflow adapters."""

from __future__ import annotations

from typing import Any

from x_editor.core.invocation_trace import ProviderInvocationRecord


def provider_records_from_client(
    client: Any,
    *,
    provider: str,
    model: str | None,
    planner_purpose: str,
    start_index: int,
    succeeded: bool,
    error: Exception | None = None,
) -> list[ProviderInvocationRecord]:
    raw_invocations = getattr(client, "last_json_invocations", None)
    if not isinstance(raw_invocations, list) or not raw_invocations:
        raw_invocations = [
            {
                "purpose": "primary",
                "transport_attempt_count": getattr(
                    client,
                    "last_transport_attempt_count",
                    1,
                ),
                "metrics": getattr(client, "last_call_metrics", None) or {},
            }
        ]

    records: list[ProviderInvocationRecord] = []
    current_semantic_parent = (
        f"{provider}:{start_index - 1}"
        if planner_purpose != "primary" and start_index > 1
        else None
    )
    for offset, raw in enumerate(raw_invocations):
        raw = raw if isinstance(raw, dict) else {}
        raw_purpose = str(raw.get("purpose") or "primary")
        purpose = _combined_purpose(planner_purpose, raw_purpose)
        invocation_id = f"{provider}:{start_index + offset}"
        kind = _provider_invocation_kind(purpose)
        parent_invocation_id = (
            current_semantic_parent
            if kind in {
                "json_repair",
                "schema_repair",
                "protocol_fallback",
                "logical_retry",
            }
            else None
        )
        metrics = dict(raw.get("metrics") or {})
        attempts = raw.get("transport_attempt_count")
        try:
            attempt_count = int(attempts)
        except (TypeError, ValueError):
            attempt_count = 1
        attempt_count = max(0, attempt_count)
        is_last = offset == len(raw_invocations) - 1
        raw_status = str(raw.get("status") or metrics.get("status") or "")
        record_succeeded = raw_status in {"ok", "completed", "success"}
        if not raw_status:
            record_succeeded = succeeded or not is_last
        status = "completed" if record_succeeded else "failed"
        if attempt_count == 0:
            status = "pre_transport_failed"
        records.append(
            ProviderInvocationRecord(
                invocation_index=start_index + offset,
                invocation_id=invocation_id,
                parent_invocation_id=parent_invocation_id,
                provider=provider,
                model=model,
                kind=kind,
                purpose=purpose,
                status=status,
                transport_attempt_count=attempt_count,
                metrics=metrics,
                error_type=(
                    type(error).__name__
                    if error is not None and not record_succeeded
                    else None
                ),
                error_message=(
                    str(error)
                    if error is not None and not record_succeeded
                    else None
                ),
            )
        )
        current_semantic_parent = invocation_id
    return records


def _combined_purpose(planner_purpose: str, raw_purpose: str) -> str:
    if planner_purpose == "primary":
        return raw_purpose
    if raw_purpose == "primary":
        return planner_purpose
    return f"{planner_purpose}_{raw_purpose}"


def _provider_invocation_kind(purpose: str) -> str:
    normalized = purpose.lower()
    if "response_format_fallback" in normalized:
        return "protocol_fallback"
    if normalized == "schema_repair":
        return "schema_repair"
    if normalized == "json_repair" or normalized.endswith("_json_repair"):
        return "json_repair"
    if "retry" in normalized and "repair" not in normalized:
        return "logical_retry"
    if normalized == "primary":
        return "primary"
    return "other"


__all__ = ["provider_records_from_client"]
