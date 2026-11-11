"""Pure reporting helpers for post-delivery realized-flow diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def build_post_delivery_flow_report(
    *,
    expected_assignments: int,
    delivered_assignments: int,
    results: Iterable[Mapping[str, Any]],
    requests: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize reachability, measurement support, and observed utility.

    Scorer outcomes remain diagnostic. The result is not a routing prior and is
    deliberately named observed flow rather than module capacity.
    """

    rows = [dict(row) for row in results]
    for row in rows:
        if "capacity_passed" in row and row["capacity_passed"] is None:
            # A diagnostic score is not proof of a capacity pass.
            if row.get("status") != "not_applicable":
                row["status"] = "unknown"
    request_rows = list(requests or [])
    requests_by_fingerprint = {
        str(row.get("request_fingerprint") or ""): row
        for row in request_rows
        if str(row.get("request_fingerprint") or "")
    }
    applicable = [row for row in rows if row.get("status") != "not_applicable"]
    measured = [row for row in applicable if row.get("status") in {"pass", "fail"}]
    utilities = [float(row["mean_score"]) / 3.0 for row in measured]
    conditional_utility = sum(utilities) / len(utilities) if utilities else None
    execution_reachability = (
        delivered_assignments / expected_assignments if expected_assignments else None
    )
    measurable_coverage = len(measured) / len(applicable) if applicable else None
    lower_observed_flow = (
        execution_reachability * measurable_coverage * conditional_utility
        if execution_reachability is not None
        and measurable_coverage is not None
        and conditional_utility is not None
        else None
    )
    upper_observed_flow = (
        execution_reachability
        * (measurable_coverage * conditional_utility + (1.0 - measurable_coverage))
        if execution_reachability is not None
        and measurable_coverage is not None
        and conditional_utility is not None
        else None
    )
    disagreements = [row for row in measured if _has_vote_disagreement(row)]
    attempts = [
        attempt
        for row in rows
        for attempt in row.get("attempts") or []
        if isinstance(attempt, Mapping)
    ]

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("assignment_fingerprint") or "unknown")].append(row)
    assignment_reports = [
        _assignment_report(
            assignment_fingerprint,
            assignment_rows,
            requests_by_fingerprint=requests_by_fingerprint,
        )
        for assignment_fingerprint, assignment_rows in sorted(grouped.items())
    ]

    preflight_statuses = Counter(
        str(row.get("preflight_status") or "unknown") for row in request_rows
    )
    preflight_reasons = Counter(
        str(row.get("preflight_reason"))
        for row in request_rows
        if row.get("preflight_reason")
    )
    scope_sources = Counter(
        str(row.get("rubric_scope_source") or "unknown") for row in request_rows
    )
    mapping_sources = Counter(
        str(mapping.get("mapping_source") or "unknown")
        for row in request_rows
        for mapping in row.get("interval_mappings") or []
        if isinstance(mapping, Mapping)
    )
    scorer_recovered_mapping_count = sum(
        count
        for source, count in mapping_sources.items()
        if source.startswith("scorer_independent_")
    )
    ready_requests = sum(row.get("preflight_status") == "ready" for row in request_rows)
    zero_attempt_preflight_abstentions = sum(
        row.get("status") == "abstain"
        and not row.get("attempts")
        and requests_by_fingerprint.get(str(row.get("request_fingerprint") or ""), {}).get(
            "preflight_status"
        )
        != "ready"
        for row in rows
    )

    return {
        "schema_version": "x-editor.post-delivery-chain-flow-report/v3",
        "scientific_boundary": {
            "diagnostic_only": True,
            "is_capacity_estimate": False,
            "may_route_tools": False,
            "may_trigger_execution": False,
        },
        "execution": {
            "expected_assignments": expected_assignments,
            "reached_deliveries": delivered_assignments,
            "execution_reachability": execution_reachability,
        },
        "measurement": {
            "applicable_rubrics": len(applicable),
            "measured_rubrics": len(measured),
            "measurable_coverage": measurable_coverage,
            "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        },
        "interval_projection": {
            "request_count": len(request_rows),
            "ready_request_count": ready_requests,
            "ready_request_rate": (
                ready_requests / len(request_rows) if request_rows else None
            ),
            "preflight_status_counts": dict(sorted(preflight_statuses.items())),
            "preflight_reason_counts": dict(sorted(preflight_reasons.items())),
            "rubric_scope_source_counts": dict(sorted(scope_sources.items())),
            "mapping_source_counts": dict(sorted(mapping_sources.items())),
            "runtime_or_frozen_mapping_count": (
                sum(mapping_sources.values()) - scorer_recovered_mapping_count
            ),
            "scorer_recovered_mapping_count": scorer_recovered_mapping_count,
            "zero_attempt_preflight_abstentions": zero_attempt_preflight_abstentions,
            "may_change_execution": False,
        },
        "utility": {
            "conditional_mean_utility": conditional_utility,
            "normalization": "aggregated_mean_score_divided_by_3",
            "unmeasured_rubrics_imputed": False,
        },
        "uncertainty": {
            "rubrics_with_vote_disagreement": len(disagreements),
            "vote_disagreement_rate": (len(disagreements) / len(measured) if measured else None),
            "rubric_abstentions": sum(row.get("status") == "abstain" for row in rows),
            "provider_abstention_attempts": sum(
                attempt.get("status") == "abstain" for attempt in attempts
            ),
            "provider_attempts": len(attempts),
        },
        "chain_flow": {
            "observed_lower_bound": lower_observed_flow,
            "optimistic_upper_bound": upper_observed_flow,
            "formula": "reachability * (coverage * utility + unknown_share * assumption)",
            "lower_unknown_assumption": 0.0,
            "upper_unknown_assumption": 1.0,
        },
        "assignments": assignment_reports,
    }


def _assignment_report(
    assignment_fingerprint: str,
    rows: list[Mapping[str, Any]],
    *,
    requests_by_fingerprint: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    applicable = [row for row in rows if row.get("status") != "not_applicable"]
    measured = [row for row in applicable if row.get("status") in {"pass", "fail"}]
    utilities = [float(row["mean_score"]) / 3.0 for row in measured]
    assignment_requests = [
        request
        for row in rows
        if (
            request := requests_by_fingerprint.get(
                str(row.get("request_fingerprint") or "")
            )
        )
        is not None
    ]
    return {
        "schema_version": "x-editor.post-delivery-assignment-flow/v2",
        "assignment_fingerprint": assignment_fingerprint,
        "assigned_arm": str(rows[0].get("assigned_arm") or "") if rows else "",
        "execution_reached": True,
        "applicable_rubrics": len(applicable),
        "measured_rubrics": len(measured),
        "measurable_coverage": len(measured) / len(applicable) if applicable else None,
        "conditional_mean_utility": (sum(utilities) / len(utilities) if utilities else None),
        "rubrics_with_vote_disagreement": sum(_has_vote_disagreement(row) for row in measured),
        "abstained_rubrics": sum(row.get("status") == "abstain" for row in rows),
        "interval_projection": {
            "ready_rubrics": sum(
                request.get("preflight_status") == "ready"
                for request in assignment_requests
            ),
            "preflight_reason_counts": dict(
                sorted(
                    Counter(
                        str(request.get("preflight_reason"))
                        for request in assignment_requests
                        if request.get("preflight_reason")
                    ).items()
                )
            ),
            "rubric_scope_sources": sorted(
                {
                    str(request.get("rubric_scope_source") or "unknown")
                    for request in assignment_requests
                }
            ),
        },
    }


def _has_vote_disagreement(row: Mapping[str, Any]) -> bool:
    votes = [int(score) >= 2 for score in row.get("valid_scores") or []]
    return bool(votes) and any(votes) and not all(votes)


__all__ = ["build_post_delivery_flow_report"]
