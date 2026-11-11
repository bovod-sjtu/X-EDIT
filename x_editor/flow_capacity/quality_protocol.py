"""Quality-only module probes, frozen denominators, and paired loss accounting.

This sidecar does not dispatch a workflow or choose an artifact. A qualified
input estimates attainable performance under a declared intervention, not a
mathematical upper bound. Resource receipts belong to the run ledger, not utility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator

from x_editor.evaluation.guidance_contracts import (
    AtomicEvaluationRubric, EvaluationArtifactRef, EvaluationEvidenceItem,
    FrozenEvaluationModel, canonical_evaluation_fingerprint,
)
from x_editor.flow_capacity.projection import FORBIDDEN_STANDARD_KEYS, find_forbidden_key_paths

CapacityModule = Literal["reasoner", "planner", "perception", "executor"]
InputRegime = Literal["standard", "natural"]
ExperimentArm = Literal["single", "multi_try", "guided"]


class ModuleProbe(FrozenEvaluationModel):
    """One pre-output assignment; subject and judge views are separate exports.

    Provenance guards catch structural leakage. A named qualification review is
    still necessary: arbitrary prose cannot be certified target-free by schema.
    Level is a task stratum, never another objective added to the denominator.
    """

    schema_version: Literal["x-editor.module-quality-probe/v1"] = "x-editor.module-quality-probe/v1"
    probe_id: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    split: Literal["development", "validation"]
    level: int = Field(ge=1, le=4)
    module: CapacityModule
    public_instruction: str = Field(min_length=1)
    input_evidence: tuple[EvaluationEvidenceItem, ...] = ()
    input_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    qualified_payload: dict[str, Any]
    qualification: Literal["pending_review", "qualified"] = "pending_review"
    qualification_ref: str | None = None
    rubrics: tuple[AtomicEvaluationRubric, ...] = Field(min_length=1)
    critical_rubric_ids: tuple[str, ...] = Field(min_length=1)
    implementation_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_boundary(self) -> ModuleProbe:
        own_outputs = {
            "reasoner": {"reasoner_memo", "planner_decision", "plan_ir"},
            "planner": {"planner_decision", "plan_ir"},
            "perception": {"reviewed_perception", "reasoner_memo", "plan_ir", "planner_decision"},
            "executor": set(),  # A legal plan is an Executor input, not its output.
        }[self.module]
        if any(e.kind in own_outputs or e.visibility != "judge" for e in self.input_evidence):
            raise ValueError("input evidence crosses the target module boundary")
        # Reuse the existing recursive private-answer guard; no phrase routing.
        EvaluationEvidenceItem(evidence_id="payload_check", kind="other", label="input",
                               payload=self.qualified_payload)
        private_paths = find_forbidden_key_paths(self.qualified_payload, forbidden_keys=FORBIDDEN_STANDARD_KEYS)
        private_paths += [path for e in self.input_evidence
                          for path in find_forbidden_key_paths(e.payload, forbidden_keys=FORBIDDEN_STANDARD_KEYS)]
        if private_paths:
            raise ValueError(f"private fields in subject input: {private_paths}")
        if self.qualification == "qualified" and not self.qualification_ref:
            raise ValueError("qualified inputs require an auditable review reference")
        ids = [r.rubric_id for r in self.rubrics]
        if len(ids) != len(set(ids)) or not set(self.critical_rubric_ids).issubset(ids):
            raise ValueError("rubric IDs must be unique and critical IDs must exist")
        if any(r.target_module != self.module for r in self.rubrics):
            raise ValueError("rubric target must match the measured module")
        return self

    def subject_view(self) -> dict[str, Any]:
        """No case identity, split, qualification, rubric, or private labels."""
        return deepcopy({
            "public_instruction": self.public_instruction,
            "input_evidence": [e.model_dump(mode="json") for e in self.input_evidence],
            "input_artifacts": [a.model_dump(mode="json") for a in self.input_artifacts],
            "payload": self.qualified_payload,
        })

    @property
    def fingerprint(self) -> str:
        return canonical_evaluation_fingerprint(self)


class QualityObservation(FrozenEvaluationModel):
    """Immutable outcome of a predeclared attempt; absent grades remain unknown."""

    probe_id: str
    probe_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    regime: InputRegime
    arm: ExperimentArm = "single"
    attempt: int = Field(default=0, ge=0)
    state: Literal["completed", "failed", "blocked_upstream", "not_run"]
    checkpoint_ref: str = Field(min_length=1)
    # Grade 0..3 is ordinal utility, not a calibrated success probability.
    grades: dict[str, StrictInt | None] = Field(default_factory=dict)
    judge_result_ref: str | None = None

    @model_validator(mode="after")
    def validate_grades(self) -> QualityObservation:
        if any(v is not None and (isinstance(v, bool) or v not in range(4)) for v in self.grades.values()):
            raise ValueError("grades must be integers in [0, 3] or unknown")
        if self.grades and not self.judge_result_ref:
            raise ValueError("grades require a measurement receipt")
        if self.grades and self.state != "completed":
            raise ValueError("non-completed attempts cannot carry output quality grades")
        return self


def quality_interval(probe: ModuleProbe, observation: QualityObservation | None,
                     *, regime: InputRegime) -> tuple[float, float]:
    """Critical-capped utility with tight unknown bounds, never success imputation."""
    if observation is not None:
        if observation.probe_id != probe.probe_id or observation.probe_fingerprint != probe.fingerprint:
            raise ValueError("observation does not match the frozen probe")
        if observation.regime != regime:
            raise ValueError("input regime mismatch")
        if set(observation.grades) - {r.rubric_id for r in probe.rubrics}:
            raise ValueError("unknown rubric in observation")
    if regime == "standard" and probe.qualification != "qualified":
        return (0.0, 1.0)
    if observation is None or observation.state == "not_run":
        return (0.0, 1.0)
    if observation.state == "failed" or (regime == "natural" and observation.state == "blocked_upstream"):
        return (0.0, 0.0)
    if observation.state == "blocked_upstream":
        return (0.0, 1.0)
    bounds = []
    for missing in (0.0, 1.0):
        values = {r.rubric_id: missing if observation.grades.get(r.rubric_id) is None
                  else observation.grades[r.rubric_id] / 3 for r in probe.rubrics}
        mean = sum(r.weight * values[r.rubric_id] for r in probe.rubrics) / sum(r.weight for r in probe.rubrics)
        bounds.append(min(mean, *(values[k] for k in probe.critical_rubric_ids)))
    return bounds[0], bounds[1]


def summarize_quality(probes: Sequence[ModuleProbe], observations: Sequence[QualityObservation],
                      *, regime: InputRegime, arm: ExperimentArm = "single", attempt: int = 0) -> dict[str, Any]:
    """Macro-average over the COMPLETE preregistered assignment set, including missing rows."""
    if len({p.module for p in probes}) > 1 or len({p.split for p in probes}) > 1:
        raise ValueError("summarize each module and development/validation split separately")
    by_id = {p.probe_id: p for p in probes}
    if len(by_id) != len(probes):
        raise ValueError("duplicate probe in denominator")
    selected: dict[str, QualityObservation] = {}
    for observation in observations:
        if observation.probe_id not in by_id:
            raise ValueError("observation is outside the frozen denominator")
        if (observation.regime, observation.arm, observation.attempt) == (regime, arm, attempt):
            if observation.probe_id in selected:
                raise ValueError("duplicate observation; retries need distinct attempt indices")
            selected[observation.probe_id] = observation
    intervals = [quality_interval(p, selected.get(p.probe_id), regime=regime) for p in probes]
    n = len(intervals)
    return {
        "assigned": n, "observed": len(selected), "input_regime": regime,
        "arm": arm, "attempt": attempt,
        "identified_interval": [sum(x[i] for x in intervals) / n for i in (0, 1)] if n else None,
        "fully_identified": sum(lo == hi for lo, hi in intervals),
        "strict_quality_successes": sum(lo == 1 for lo, _ in intervals),
        "is_theoretical_upper_bound": False,
        "resources_in_utility": False,
    }


def paired_quality_gate(probes: Sequence[ModuleProbe], baseline: Mapping[str, QualityObservation],
                        candidate: Mapping[str, QualityObservation], *, regime: InputRegime) -> dict[str, Any]:
    """A gain never cancels another objective's loss; uncertainty cannot pass a gate."""
    losses, gains, unknown = [], [], []
    if not probes:
        return {"status": "insufficient_evidence", "losses": [], "gains": [], "unknown": ["empty_denominator"], "assigned": 0}
    for p in probes:
        before, after = baseline.get(p.probe_id), candidate.get(p.probe_id)
        b, c = quality_interval(p, before, regime=regime), quality_interval(p, after, regime=regime)
        if b[0] != b[1] or c[0] != c[1]:
            unknown.append(p.probe_id)
        if c[1] < b[0]:
            losses.append(p.probe_id)
        elif c[0] > b[1]:
            gains.append(p.probe_id)
        if before and after:
            for r in p.rubrics:
                bg, cg = before.grades.get(r.rubric_id), after.grades.get(r.rubric_id)
                if bg is not None and cg is not None and cg < bg:
                    losses.append(f"{p.probe_id}:{r.rubric_id}")
                elif bg is None or cg is None:
                    unknown.append(f"{p.probe_id}:{r.rubric_id}")
    return {"status": "regression" if losses else "insufficient_evidence" if unknown else "no_observed_regression",
            "losses": losses, "gains": gains, "unknown": unknown,
            "assigned": len(probes)}
