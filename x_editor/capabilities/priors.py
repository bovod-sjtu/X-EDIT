"""Versioned development-prior contracts for capability route selection.

Tool priors are offline development evidence. Loading a snapshot does not make
it active: callers must bind its fingerprint to an experiment profile and
explicitly freeze it before exposing route-selection guidance to a Planner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SNAPSHOT_PROFILE_KEY = "tool_prior_snapshot"


class ToolPriorError(ValueError):
    """Base error for invalid or unsafe tool-prior use."""


class SnapshotHashMismatchError(ToolPriorError):
    """Raised when snapshot content no longer matches its declared hash."""


class SnapshotFreezeError(ToolPriorError):
    """Raised when a snapshot was not explicitly bound before use."""


class StrictFrozenModel(BaseModel):
    """Immutable contract object with no undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolPriorSource(StrictFrozenModel):
    capacity_report_path: str
    capacity_report_sha256: str = Field(pattern=SHA256_PATTERN)
    capacity_report_schema_version: Literal["x-editor.tool-prior-capacity-report/v1"]
    release_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_role: Literal["offline_prior_estimator"]
    experiment_kind: Literal["offline_evaluator_tool_prior_calibration"]
    data_role: Literal["development_calibration"]
    eligible_as_final_outcome: Literal[False] = False
    heldout_or_test_outcomes_used: Literal[False] = False
    claim_boundary: str


class ToolPriorUsagePolicy(StrictFrozenModel):
    guidance_scope: Literal["route_selection_guidance_only"]
    requires_explicit_validation: Literal[True] = True
    requires_explicit_freeze: Literal[True] = True
    requires_experiment_profile_fingerprint: Literal[True] = True
    automatic_catalog_update: Literal[False] = False
    automatic_planner_update: Literal[False] = False
    automatic_runtime_update: Literal[False] = False
    can_gate_runtime: Literal[False] = False
    prohibited_uses: tuple[
        Literal[
            "derive_or_update_from_heldout_or_test_outcomes",
            "report_as_heldout_or_final_outcome",
            "per_case_route_adaptation",
            "automatic_policy_mutation",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_prohibited_uses(self) -> ToolPriorUsagePolicy:
        required = {
            "derive_or_update_from_heldout_or_test_outcomes",
            "report_as_heldout_or_final_outcome",
            "per_case_route_adaptation",
            "automatic_policy_mutation",
        }
        if set(self.prohibited_uses) != required:
            raise ValueError("tool-prior usage policy must preserve every prohibited use")
        return self


class RouteEstimate(StrictFrozenModel):
    route_id: str
    operations: tuple[str, ...]
    empirical_flow_capacity_itt: float = Field(ge=0.0, le=1.0)
    frozen_sample_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_operations(self) -> RouteEstimate:
        if not self.operations or any(not item.strip() for item in self.operations):
            raise ValueError("each route requires at least one operation")
        return self


class PairwisePriorEvidence(StrictFrozenModel):
    route_a_id: str
    route_b_id: str
    mean_capacity_delta_a_minus_b: float
    bootstrap_95pct_ci: tuple[float, float]
    paired_sample_count: int = Field(gt=0)
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)
    conclusion: Literal["preferred_better", "not_distinguished"]

    @model_validator(mode="after")
    def validate_pairwise_evidence(self) -> PairwisePriorEvidence:
        low, high = self.bootstrap_95pct_ci
        if low > high:
            raise ValueError("pairwise confidence interval must be ordered")
        if self.wins + self.ties + self.losses != self.paired_sample_count:
            raise ValueError("pairwise win/tie/loss counts must equal paired_sample_count")
        if self.route_a_id == self.route_b_id:
            raise ValueError("pairwise routes must be distinct")
        if self.conclusion == "preferred_better" and low <= 0.0:
            raise ValueError("preferred_better requires a confidence interval above zero")
        if self.conclusion == "not_distinguished" and not (low <= 0.0 <= high):
            raise ValueError("not_distinguished requires a confidence interval crossing zero")
        return self


class RouteSelectionGuidance(StrictFrozenModel):
    guidance_id: str
    comparison_group: str
    task_scope: str
    decision: Literal["prefer", "no_preference"]
    ordering_semantics: Literal["explicit_preference_only", "none"]
    routes: tuple[RouteEstimate, ...]
    preferred_route_id: str | None = None
    pairwise_evidence: tuple[PairwisePriorEvidence, ...]
    decision_reason: str

    @model_validator(mode="after")
    def validate_decision(self) -> RouteSelectionGuidance:
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) < 2 or len(route_ids) != len(set(route_ids)):
            raise ValueError("guidance requires at least two uniquely identified routes")
        route_id_set = set(route_ids)
        for evidence in self.pairwise_evidence:
            if {
                evidence.route_a_id,
                evidence.route_b_id,
            } - route_id_set:
                raise ValueError("pairwise evidence references an undeclared route")

        if self.decision == "prefer":
            if self.ordering_semantics != "explicit_preference_only":
                raise ValueError("prefer guidance requires explicit_preference_only ordering")
            if self.preferred_route_id not in route_id_set:
                raise ValueError("prefer guidance requires a declared preferred route")
            if not self.pairwise_evidence or any(
                item.conclusion != "preferred_better" or item.route_a_id != self.preferred_route_id
                for item in self.pairwise_evidence
            ):
                raise ValueError("prefer guidance requires significant supporting comparisons")
        else:
            if self.ordering_semantics != "none":
                raise ValueError("no_preference guidance cannot define ordering semantics")
            if self.preferred_route_id is not None:
                raise ValueError("no_preference guidance cannot name a preferred route")
            if not self.pairwise_evidence or any(
                item.conclusion != "not_distinguished" for item in self.pairwise_evidence
            ):
                raise ValueError("no_preference requires non-distinguished comparisons")
        return self


class ExperimentPriorBinding(StrictFrozenModel):
    schema_version: Literal["x-editor.tool-prior-experiment-binding/v1"] = (
        "x-editor.tool-prior-experiment-binding/v1"
    )
    snapshot_id: str
    snapshot_version: str
    snapshot_fingerprint: str = Field(pattern=SHA256_PATTERN)
    source_data_role: Literal["development_calibration"]
    evaluator_role: Literal["offline_prior_estimator"]
    usage: Literal["route_selection_guidance_only"]
    freeze_status: Literal["explicitly_frozen"] = "explicitly_frozen"
    eligible_as_final_outcome: Literal[False] = False


class ToolPriorSnapshot(StrictFrozenModel):
    schema_version: Literal["x-editor.tool-prior-snapshot/v1"] = "x-editor.tool-prior-snapshot/v1"
    snapshot_id: str
    snapshot_version: str
    prior_snapshot_status: Literal["development_only_requires_explicit_freeze"]
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    source: ToolPriorSource
    usage_policy: ToolPriorUsagePolicy
    guidance: tuple[RouteSelectionGuidance, ...]

    @model_validator(mode="after")
    def validate_guidance(self) -> ToolPriorSnapshot:
        guidance_ids = [item.guidance_id for item in self.guidance]
        groups = [item.comparison_group for item in self.guidance]
        if not self.guidance:
            raise ValueError("tool-prior snapshot requires route guidance")
        if len(guidance_ids) != len(set(guidance_ids)):
            raise ValueError("tool-prior guidance ids must be unique")
        if len(groups) != len(set(groups)):
            raise ValueError("comparison groups must be unique within a snapshot")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the canonical hash, excluding the self-declared hash field."""

        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        return _content_sha256(payload)

    def validate_for_route_selection(self) -> None:
        """Explicitly validate development-only role and content identity."""

        if self.fingerprint != self.content_sha256:
            raise SnapshotHashMismatchError(
                "tool-prior snapshot content hash differs from content_sha256"
            )
        if self.source.data_role != "development_calibration":
            raise ToolPriorError("tool prior must be derived from development calibration")
        if self.source.eligible_as_final_outcome:
            raise ToolPriorError("tool prior cannot be a held-out or final outcome")
        if self.source.heldout_or_test_outcomes_used:
            raise ToolPriorError("tool prior cannot use held-out or test outcomes")

    def experiment_binding(self) -> ExperimentPriorBinding:
        """Build the immutable profile receipt for this snapshot."""

        self.validate_for_route_selection()
        return ExperimentPriorBinding(
            snapshot_id=self.snapshot_id,
            snapshot_version=self.snapshot_version,
            snapshot_fingerprint=self.fingerprint,
            source_data_role=self.source.data_role,
            evaluator_role=self.source.evaluator_role,
            usage=self.usage_policy.guidance_scope,
        )


class FrozenToolPriorSnapshot(StrictFrozenModel):
    """Snapshot proven to be bound to one experiment profile."""

    snapshot: ToolPriorSnapshot
    binding: ExperimentPriorBinding

    @model_validator(mode="after")
    def validate_binding(self) -> FrozenToolPriorSnapshot:
        expected = self.snapshot.experiment_binding()
        if self.binding != expected:
            raise ValueError("frozen binding does not match the tool-prior snapshot")
        return self


def load_tool_prior_snapshot(path: str | Path) -> ToolPriorSnapshot:
    """Load and hash-check one snapshot without activating it."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolPriorError(f"invalid tool-prior JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ToolPriorError(f"tool-prior snapshot must be a JSON object: {source}")
    snapshot = ToolPriorSnapshot.model_validate(payload)
    snapshot.validate_for_route_selection()
    return snapshot


def bind_tool_prior_to_experiment_profile(
    profile: Mapping[str, Any], snapshot: ToolPriorSnapshot
) -> dict[str, Any]:
    """Return a copied profile containing this snapshot's freeze receipt."""

    binding = snapshot.experiment_binding().model_dump(mode="json")
    bound = deepcopy(dict(profile))
    existing = bound.get(SNAPSHOT_PROFILE_KEY)
    if existing is not None and existing != binding:
        raise SnapshotFreezeError("experiment profile already binds a different tool prior")
    bound[SNAPSHOT_PROFILE_KEY] = binding
    return bound


def freeze_tool_prior_snapshot(
    snapshot: ToolPriorSnapshot, experiment_profile: Mapping[str, Any]
) -> FrozenToolPriorSnapshot:
    """Freeze a snapshot only after its exact fingerprint is in the profile."""

    snapshot.validate_for_route_selection()
    raw_binding = experiment_profile.get(SNAPSHOT_PROFILE_KEY)
    if raw_binding is None:
        raise SnapshotFreezeError(
            f"experiment profile must contain {SNAPSHOT_PROFILE_KEY!r} before freeze"
        )
    try:
        binding = ExperimentPriorBinding.model_validate(raw_binding)
    except (TypeError, ValueError) as exc:
        raise SnapshotFreezeError(
            "experiment profile contains an invalid tool-prior binding"
        ) from exc
    expected = snapshot.experiment_binding()
    if binding != expected:
        raise SnapshotFreezeError(
            "experiment profile tool-prior fingerprint does not match snapshot"
        )
    return FrozenToolPriorSnapshot(snapshot=snapshot, binding=binding)


def _content_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SNAPSHOT_PROFILE_KEY",
    "ExperimentPriorBinding",
    "FrozenToolPriorSnapshot",
    "PairwisePriorEvidence",
    "RouteEstimate",
    "RouteSelectionGuidance",
    "SnapshotFreezeError",
    "SnapshotHashMismatchError",
    "ToolPriorError",
    "ToolPriorSnapshot",
    "bind_tool_prior_to_experiment_profile",
    "freeze_tool_prior_snapshot",
    "load_tool_prior_snapshot",
]
