"""Separate logical judge work from observed provider transport attempts.

Unknown transport usage is not zero. A transport cap can be enforced only when
an adapter declares a finite per-invocation bound and reports actual usage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar

_R = TypeVar("_R")
EvaluationCallRole = Literal["evaluator", "guider"]


class EvaluationTransportBudgetError(RuntimeError):
    """The next provider invocation cannot fit a provable transport budget."""


@dataclass
class _RoleUsage:
    logical_calls: int = 0
    backend_invocations: int = 0
    failed_backend_invocations: int = 0
    known_http_attempts: int = 0
    unobserved_transport_invocations: int = 0


class EvaluationCallAccounting:
    """One append-only invocation history for an evaluation/guidance path."""

    def __init__(
        self, *, evaluator_http_limit: int | None = None, guider_http_limit: int | None = None
    ) -> None:
        self._limits = {"evaluator": evaluator_http_limit, "guider": guider_http_limit}
        if any(value is not None and value < 0 for value in self._limits.values()):
            raise ValueError("HTTP attempt limits must be nonnegative")
        self._usage = {role: _RoleUsage() for role in self._limits}
        self.invocations: list[dict[str, Any]] = []

    def start_logical_call(self, role: EvaluationCallRole) -> None:
        self._usage[role].logical_calls += 1

    def invoke(self, role: EvaluationCallRole, backend: Any, call: Callable[[], _R]) -> _R:
        usage = self._usage[role]
        limit = self._limits[role]
        bound = getattr(backend, "maximum_transport_attempts_per_invocation", None)
        bound = bound if _nonnegative_int(bound) else None
        if limit is not None:
            if usage.unobserved_transport_invocations or not _nonnegative_int(bound):
                raise EvaluationTransportBudgetError(
                    "transport usage or next-call bound is unknown"
                )
            if usage.known_http_attempts + bound > limit:
                raise EvaluationTransportBudgetError("next invocation exceeds the transport budget")
        usage.backend_invocations += 1
        failed = True
        try:
            result = call()
            failed = False
            return result
        finally:
            observed = getattr(backend, "last_transport_attempt_count", None)
            observed = observed if _nonnegative_int(observed) else None
            violated_bound = limit is not None and (observed is None or observed > bound)
            usage.failed_backend_invocations += int(failed or violated_bound)
            if observed is None:
                usage.unobserved_transport_invocations += 1
            else:
                usage.known_http_attempts += observed
            self.invocations.append(
                {
                    "role": role,
                    "backend_id": backend.backend_id,
                    "failed": failed,
                    "http_attempts": observed,
                    "declared_http_bound": bound,
                    "transport_contract_violated": violated_bound,
                }
            )
            if violated_bound:
                raise EvaluationTransportBudgetError(
                    "backend did not honor its transport accounting contract"
                )

    def snapshot(self) -> dict[str, Any]:
        return {
            role: {
                **asdict(usage),
                "http_attempt_limit": self._limits[role],
                "actual_http_attempts": (
                    usage.known_http_attempts
                    if not usage.unobserved_transport_invocations
                    else None
                ),
                "accounting_complete": not usage.unobserved_transport_invocations,
            }
            for role, usage in self._usage.items()
        }


def client_transport_bound(client: Any) -> int | None:
    """A declared bound; arbitrary clients without this contract remain unknown."""
    retries = getattr(client, "max_retries", None)
    if (
        not _nonnegative_int(retries)
        or getattr(client, "allow_json_repair", None) is not False
        or getattr(client, "allow_response_format_fallback", None) is not False
    ):
        return None
    return retries + 1


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def summarize_call_accounting(snapshots: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Aggregate observations without treating old or partial records as zero."""
    summary = {}
    for role in ("evaluator", "guider"):
        usages = [snapshot.get(role) if snapshot else None for snapshot in snapshots]
        observed = [usage for usage in usages if isinstance(usage, dict)]
        missing = len(usages) - len(observed)
        known_attempts = sum(usage["known_http_attempts"] for usage in observed)
        unknown = sum(usage["unobserved_transport_invocations"] for usage in observed)
        complete = missing == 0 and unknown == 0
        summary[role] = {
            "known_http_attempts": known_attempts,
            "actual_http_attempts": known_attempts if complete else None,
            "unobserved_transport_invocations": unknown,
            "assignments_without_accounting": missing,
            "accounting_complete": complete,
        }
    return summary
