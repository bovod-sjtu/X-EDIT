"""Permissive triage for specialist perception candidates.

The triage layer decides how much additional review a candidate deserves.  It
does not decide whether the workflow as a whole may continue.  Invalid
candidates are excluded individually; uncertain candidates remain explicit and
can be escalated to StrongOmni or returned to the Reasoner as missing evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class EvidenceIssueSeverity(StrEnum):
    """Effect of one evidence issue on a candidate, not on the whole workflow."""

    ADVISORY = "advisory"
    NEEDS_REVIEW = "needs_review"
    HARD_ERROR = "hard_error"


CandidateDisposition = Literal[
    "accept_prior",
    "review_global",
    "review_local",
    "exclude_candidate",
]


@dataclass(frozen=True)
class EvidenceTriagePolicy:
    """Provider-neutral review policy.

    The numeric floor only controls escalation cost.  Falling below it never
    rejects the workflow or turns a proposal into a negative observation.
    """

    local_review_confidence_floor: float = 0.45
    unknown_confidence_needs_global_review: bool = True


@dataclass(frozen=True)
class EvidenceTriageIssue:
    code: str
    severity: EvidenceIssueSeverity
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class CandidateTriageDecision:
    """Auditable review decision for one specialist or Omni candidate."""

    candidate_id: str | None
    disposition: CandidateDisposition
    review_priority: float
    review_reasons: tuple[str, ...] = ()
    issues: tuple[EvidenceTriageIssue, ...] = ()
    confidence: float | None = None
    candidate: dict[str, Any] = field(default_factory=dict)

    @property
    def should_review_globally(self) -> bool:
        return self.disposition == "review_global"

    @property
    def should_slice_locally(self) -> bool:
        return self.disposition == "review_local"

    @property
    def usable_as_prior(self) -> bool:
        return self.disposition != "exclude_candidate"

    @property
    def blocks_workflow(self) -> bool:
        # Candidate triage never owns workflow admission.
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        payload["should_review_globally"] = self.should_review_globally
        payload["should_slice_locally"] = self.should_slice_locally
        payload["usable_as_prior"] = self.usable_as_prior
        payload["blocks_workflow"] = self.blocks_workflow
        return payload


@dataclass(frozen=True)
class EvidenceTriageBatch:
    decisions: tuple[CandidateTriageDecision, ...]

    @property
    def accepted_prior_ids(self) -> tuple[str, ...]:
        return _decision_ids(self.decisions, "accept_prior")

    @property
    def global_review_ids(self) -> tuple[str, ...]:
        return _decision_ids(self.decisions, "review_global")

    @property
    def local_review_ids(self) -> tuple[str, ...]:
        return _decision_ids(self.decisions, "review_local")

    @property
    def excluded_ids(self) -> tuple[str, ...]:
        return _decision_ids(self.decisions, "exclude_candidate")

    @property
    def has_usable_candidate(self) -> bool:
        return any(decision.usable_as_prior for decision in self.decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "x-editor.perception-evidence-triage/v1",
            "decisions": [decision.to_dict() for decision in self.decisions],
            "accepted_prior_ids": list(self.accepted_prior_ids),
            "global_review_ids": list(self.global_review_ids),
            "local_review_ids": list(self.local_review_ids),
            "excluded_ids": list(self.excluded_ids),
            "has_usable_candidate": self.has_usable_candidate,
            "blocks_workflow": False,
        }


def triage_candidate_evidence(
    candidate: Mapping[str, Any],
    *,
    require_local_boundaries: bool = False,
    execution_critical: bool = False,
    policy: EvidenceTriagePolicy | None = None,
) -> CandidateTriageDecision:
    """Choose the least expensive adequate review path for one candidate.

    Structural invalidity excludes only this candidate.  Ambiguity, missing
    confidence, disagreement, and weak boundaries request review without
    becoming protocol failures.
    """

    resolved_policy = policy or EvidenceTriagePolicy()
    copied = dict(candidate)
    candidate_id = _optional_text(
        copied.get("candidate_id")
        or copied.get("event_id")
        or copied.get("segment_id")
        or copied.get("span_id")
    )
    issues: list[EvidenceTriageIssue] = []
    reasons: list[str] = []
    priority = 0.0

    _start, _end, interval_state = _candidate_interval(copied)
    if interval_state == "invalid":
        issues.append(
            EvidenceTriageIssue(
                code="invalid_candidate_interval",
                severity=EvidenceIssueSeverity.HARD_ERROR,
                message="Candidate end must be greater than start and remain non-negative.",
            )
        )
        return CandidateTriageDecision(
            candidate_id=candidate_id,
            disposition="exclude_candidate",
            review_priority=1.0,
            review_reasons=("invalid_candidate_interval",),
            issues=tuple(issues),
            confidence=_candidate_confidence(copied),
            candidate=copied,
        )

    provider_health = _normalized_state(
        copied.get("provider_health") or copied.get("provider_status")
    )
    if provider_health in {"failed", "error", "unavailable"}:
        issues.append(
            EvidenceTriageIssue(
                code="candidate_provider_failed",
                severity=EvidenceIssueSeverity.HARD_ERROR,
                message="The producing specialist did not complete successfully.",
            )
        )
        return CandidateTriageDecision(
            candidate_id=candidate_id,
            disposition="exclude_candidate",
            review_priority=1.0,
            review_reasons=("candidate_provider_failed",),
            issues=tuple(issues),
            confidence=_candidate_confidence(copied),
            candidate=copied,
        )

    target_match = _normalized_state(
        copied.get("target_match") or copied.get("matches") or copied.get("answer")
    )
    adjudication = _normalized_state(
        copied.get("strong_omni_status")
        or copied.get("adjudication")
        or copied.get("review_status")
    )
    if target_match in {"no", "false", "rejected"} or adjudication in {
        "no",
        "false",
        "rejected",
    }:
        return CandidateTriageDecision(
            candidate_id=candidate_id,
            disposition="exclude_candidate",
            review_priority=0.0,
            review_reasons=("reviewed_not_target",),
            issues=(),
            confidence=_candidate_confidence(copied),
            candidate=copied,
        )

    confidence = _candidate_confidence(copied)
    boundary_quality = _normalized_state(copied.get("boundary_quality"))
    agreement = _normalized_state(
        copied.get("cross_tool_agreement") or copied.get("corroboration_status")
    )

    local_reasons: list[str] = []
    if agreement in {"conflict", "disagree", "inconsistent"}:
        local_reasons.append("cross_tool_conflict")
        priority = max(priority, 0.95)
    if boundary_quality in {"uncertain", "unstable", "coarse", "conflict"}:
        local_reasons.append("boundary_uncertain")
        priority = max(priority, 0.9)
    if confidence is not None and confidence < resolved_policy.local_review_confidence_floor:
        local_reasons.append("low_confidence")
        priority = max(priority, 0.85)
    if target_match in {"uncertain", "unknown"} or adjudication in {
        "uncertain",
        "unknown",
    }:
        local_reasons.append("review_outcome_uncertain")
        priority = max(priority, 0.9)

    if local_reasons:
        if interval_state == "valid":
            return CandidateTriageDecision(
                candidate_id=candidate_id,
                disposition="review_local",
                review_priority=priority,
                review_reasons=tuple(dict.fromkeys(local_reasons)),
                issues=tuple(issues),
                confidence=confidence,
                candidate=copied,
            )
        issues.append(
            EvidenceTriageIssue(
                code="local_review_has_no_interval",
                severity=EvidenceIssueSeverity.NEEDS_REVIEW,
                message="The candidate needs review but has no valid interval to slice.",
            )
        )
        reasons.extend(local_reasons)
        reasons.append("global_review_required_without_interval")
        priority = max(priority, 0.9)

    explicitly_needs_review = any(
        _truthy(copied.get(field))
        for field in ("needs_review", "requires_model_review", "requires_review")
    )
    if explicitly_needs_review:
        reasons.append("specialist_requested_review")
        priority = max(priority, 0.65)
    if confidence is None and resolved_policy.unknown_confidence_needs_global_review:
        reasons.append("confidence_unreported")
        priority = max(priority, 0.55)
    if provider_health in {"degraded", "partial"}:
        reasons.append("provider_degraded")
        priority = max(priority, 0.7)

    if require_local_boundaries and execution_critical and interval_state != "valid":
        reasons.append("execution_critical_boundary_missing")
        priority = max(priority, 0.95)
    elif (
        require_local_boundaries
        and execution_critical
        and boundary_quality not in {"stable", "reviewed", "precise"}
        and adjudication not in {"accepted", "confirmed", "yes", "true"}
    ):
        reasons.append("execution_critical_boundary_unconfirmed")
        priority = max(priority, 0.8)

    if reasons:
        issues.append(
            EvidenceTriageIssue(
                code="candidate_needs_review",
                severity=EvidenceIssueSeverity.NEEDS_REVIEW,
                message="Candidate remains usable as a prior while additional review is requested.",
            )
        )
        return CandidateTriageDecision(
            candidate_id=candidate_id,
            disposition="review_global",
            review_priority=priority,
            review_reasons=tuple(dict.fromkeys(reasons)),
            issues=tuple(issues),
            confidence=confidence,
            candidate=copied,
        )

    if candidate_id is None:
        issues.append(
            EvidenceTriageIssue(
                code="candidate_id_missing",
                severity=EvidenceIssueSeverity.ADVISORY,
                message="A stable candidate id should be materialized before evidence binding.",
            )
        )
    return CandidateTriageDecision(
        candidate_id=candidate_id,
        disposition="accept_prior",
        review_priority=0.0,
        review_reasons=(),
        issues=tuple(issues),
        confidence=confidence,
        candidate=copied,
    )


def triage_candidate_evidence_batch(
    candidates: Sequence[Mapping[str, Any]],
    *,
    require_local_boundaries: bool = False,
    execution_critical: bool = False,
    policy: EvidenceTriagePolicy | None = None,
) -> EvidenceTriageBatch:
    return EvidenceTriageBatch(
        decisions=tuple(
            triage_candidate_evidence(
                candidate,
                require_local_boundaries=require_local_boundaries,
                execution_critical=execution_critical,
                policy=policy,
            )
            for candidate in candidates
        )
    )


def _decision_ids(
    decisions: Sequence[CandidateTriageDecision],
    disposition: CandidateDisposition,
) -> tuple[str, ...]:
    return tuple(
        decision.candidate_id
        for decision in decisions
        if decision.disposition == disposition and decision.candidate_id is not None
    )


def _candidate_interval(candidate: Mapping[str, Any]) -> tuple[float | None, float | None, str]:
    start = _optional_float(
        candidate.get("start_sec", candidate.get("start", candidate.get("onset")))
    )
    end = _optional_float(
        candidate.get("end_sec", candidate.get("end", candidate.get("offset")))
    )
    if start is None and end is None:
        return None, None, "missing"
    if start is None or end is None or start < 0.0 or end <= start:
        return start, end, "invalid"
    return start, end, "valid"


def _candidate_confidence(candidate: Mapping[str, Any]) -> float | None:
    for field_name in (
        "calibrated_confidence",
        "confidence",
        "peak_score",
        "mean_score",
        "score",
    ):
        raw = candidate.get(field_name)
        if isinstance(raw, str):
            label = raw.strip().lower()
            if label == "high":
                return 0.9
            if label == "medium":
                return 0.6
            if label == "low":
                return 0.25
        number = _optional_float(raw)
        if number is not None and 0.0 <= number <= 1.0:
            return number
    return None


def _normalized_state(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalized_state(value) in {"1", "true", "yes", "required"}


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
