"""Small public contracts for explicit iterative evaluation experiments.

The controller owns retries and selection. These observations cannot admit a
plan, overwrite an attempt, or substitute a private reference for the user goal.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from x_editor.core.future_attempt_guidance import ModuleIterationFeedback
from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel
from x_editor.evaluation.minimal_atomic_review import MinimalAtomicAnswer
from x_editor.evaluation.guidance_state import GuidanceStateUpdate
from x_editor.evaluation.edit_region_scope import (
    PRIMARY_FOCI, RegionCheck, RegionScope, RegionExemption, RegionalObservation,
    RegionCoverageJudgment, validate_region_design,
)


class RefinementPolicy(FrozenEvaluationModel):
    protocol: Literal["x-editor.public-evidence-refinement/v1"] = (
        "x-editor.public-evidence-refinement/v1"
    )
    mode: Literal["ordinary_retry", "targeted_feedback", "critic_feedback", "runtime_error_feedback"] = "targeted_feedback"
    max_retries: int = Field(default=2, ge=0, le=10, strict=True)
    skip_unused_feedback: bool = False
    selection: Literal["delivery_then_conservative_utility", "one_artifact_max_flow", "mc_conservative_utility"] = (
        "delivery_then_conservative_utility"
    )
    stop_when_all_supported: bool = False
    stop_without_actionable_guidance: bool = False
    retain_runtime_error_feedback: bool = False
    nondelivery_review: Literal["diagnose", "raw_errors"] = "diagnose"
    execution_reuse: Literal["disabled", "failed_prefix"] = "disabled"
    independent_final_assessment_required: Literal[True] = True
    selection_assessment_timing: Literal["online", "after_generation"] = "online"
    selection_assessment_reuse: Literal["none", "exact_online"] = "none"
    selection_mc_repeats: int = Field(default=3, ge=1, le=10, strict=True,
                                      exclude_if=lambda v: v == 3)
    selection_mc_seed: int = Field(default=0, ge=0, strict=True, exclude_if=lambda v: v == 0)

    @model_validator(mode="after")
    def selection_cannot_control_generation(self):
        if self.selection == 'mc_conservative_utility' and (
                self.selection_assessment_timing != 'after_generation' or
                self.selection_assessment_reuse != 'none'):
            raise ValueError('MC selection requires fresh deferred assessments, without online score reuse')
        if self.selection_assessment_reuse != 'none' and self.selection_assessment_timing != 'after_generation':
            raise ValueError('online assessment reuse requires deferred selection')
        if self.execution_reuse != "disabled" and self.max_retries > 2:
            raise ValueError("Restricted execution reuse is bounded to three total attempts")
        if self.nondelivery_review == "raw_errors" and self.mode != "targeted_feedback":
            raise ValueError("delivery-conditioned review requires targeted feedback")
        if self.retain_runtime_error_feedback and self.mode != "targeted_feedback":
            raise ValueError("runtime-error retention augments targeted feedback only")
        if self.selection_assessment_timing == "after_generation" and self.stop_when_all_supported:
            raise ValueError("deferred selection cannot supply the generation stopping rule")
        return self


class FeedbackTurn(FrozenEvaluationModel):
    turn_index: int = Field(ge=0, strict=True)
    modules: tuple[ModuleIterationFeedback, ...] = Field(min_length=1, max_length=3)
    cited_question_ids: tuple[str, ...] = Field(min_length=1)
    repair_evidence_ids: tuple[str, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def unique_modules(self):
        names = [m.module for m in self.modules]
        if len(names) != len(set(names)):
            raise ValueError("duplicate module feedback")
        return self


class ObservationCheck(FrozenEvaluationModel):
    """A proposed check of a reported event, not a verified event annotation."""

    turn_index: int = Field(ge=0, strict=True)
    audio_ref: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(gt=0, allow_inf_nan=False)
    event_claim: str = Field(min_length=1, max_length=800)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("observation interval must have positive extent")
        return self


class FeedbackDraft(FrozenEvaluationModel):
    guidance_state_updates: tuple[GuidanceStateUpdate, ...] = Field(default=(), max_length=96)
    turns: tuple[FeedbackTurn, ...] = Field(default=(), max_length=32)
    observation_checks: tuple[ObservationCheck, ...] = Field(default=(), max_length=8)
    unresolved_review_conflicts: tuple[str, ...] = Field(default=(), max_length=16)


class FeedbackScopeReview(FrozenEvaluationModel):
    turn_index: int = Field(ge=0, strict=True)
    grounded_in_visible_input: bool
    no_replacement_payload: bool
    no_unsupported_requirement: bool
    preserves_user_constraints: bool
    new_observations_accounted_for: bool | None = None
    respects_module_phase: bool | None = None
    guidance_state_accounted_for: bool | None = None
    issues: tuple[str, ...] = Field(default=(), max_length=12)

    @property
    def usable(self):
        return (
            all(
                (
                    self.grounded_in_visible_input,
                    self.no_replacement_payload,
                    self.no_unsupported_requirement,
                    self.preserves_user_constraints,
                )
            )
            and not self.issues
            and self.new_observations_accounted_for is not False
            and self.respects_module_phase is not False
            and self.guidance_state_accounted_for is not False
        )


class UtilityCriterion(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1, max_length=128)
    dimension: Literal["goal", "preservation", "perceptual_quality"]
    question: str = Field(min_length=4, max_length=1400)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class PublicUtilityDesign(FrozenEvaluationModel):
    """Authored from public turns before candidate audio is opened."""

    final_state: str = Field(min_length=4, max_length=2400)
    criteria: tuple[UtilityCriterion, ...] = Field(min_length=3, max_length=24)

    @model_validator(mode="after")
    def full_dimensions(self):
        names = [c.criterion_id for c in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("duplicate utility criteria")
        if {c.dimension for c in self.criteria} != {"goal", "preservation", "perceptual_quality"}:
            raise ValueError("utility requires goals, preservation and perceptual quality")
        return self


class RegionalCriterion(UtilityCriterion):
    region: RegionScope


class RegionUtilityDesign(PublicUtilityDesign):
    """Opt-in scope protocol; old designs retain their exact serialized form."""

    protocol: Literal["x-editor.edit-regions/v2"] = "x-editor.edit-regions/v2"
    criteria: tuple[RegionalCriterion, ...] = Field(min_length=3, max_length=24)
    region_exemptions: tuple[RegionExemption, ...] = ()

    @property
    def region_checks(self):
        return tuple(RegionCheck(criterion_id=c.criterion_id, **c.region.model_dump()) for c in self.criteria)

    @model_validator(mode="after")
    def full_dimensions(self):
        ids = [c.criterion_id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate utility criteria")
        validate_region_design(self)
        return self


class RetentionCoverage(FrozenEvaluationModel):
    subject: Literal["content", "attributes"]
    applicability: Literal["required", "not_applicable"]
    criterion_ids: tuple[str, ...] = ()
    explanation: str = Field(min_length=12, max_length=1200)


class RuntimeRegionUtilityDesign(RegionUtilityDesign):
    """Explicit retention decisions, still authored without seeing a candidate."""

    protocol: Literal["x-editor.runtime-regions/v1"] = "x-editor.runtime-regions/v1"
    retention_coverage: tuple[RetentionCoverage, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def separate_retention(self):
        if {r.subject for r in self.retention_coverage} != {"content", "attributes"}:
            raise ValueError("content and attribute retention each need an explicit decision")
        criteria = {c.criterion_id:c for c in self.criteria}
        used = set()
        for row in self.retention_coverage:
            if row.applicability == "not_applicable":
                if row.criterion_ids:
                    raise ValueError("an exempt retention subject cannot claim checked criteria")
                continue
            if not row.criterion_ids or set(row.criterion_ids) & used:
                raise ValueError("content and attribute retention need separate questions")
            for identity in row.criterion_ids:
                if identity not in criteria or criteria[identity].region.focus != "preserved_region":
                    observed = criteria[identity].region.focus if identity in criteria else "missing criterion"
                    raise ValueError(
                        f"retention must cite an actual preservation criterion: retention_coverage[{row.subject}] "
                        f"cites {identity!r}, whose region.focus is {observed!r}; required focus='preserved_region'. "
                        "Design an instruction-supported preservation comparison and cite it; "
                        "do not merely relabel an edited-region goal or exempt a required condition."
                    )
            used.update(row.criterion_ids)
        return self


def parse_utility_design(payload):
    if payload.get("protocol") == "x-editor.base-terminal-fallback/v1":
        from x_editor.evaluation.base_terminal_design import BaseTerminalDesign
        return BaseTerminalDesign.model_validate(payload)
    model = {"x-editor.edit-regions/v2":RegionUtilityDesign,
             "x-editor.runtime-regions/v1":RuntimeRegionUtilityDesign}.get(payload.get("protocol"), PublicUtilityDesign)
    return model.model_validate(payload)


class SourceListeningNotes(FrozenEvaluationModel):
    heard_speech: str = Field(max_length=4000)
    heard_sounds: str = Field(max_length=2000)
    uncertainties: tuple[str, ...] = ()


class GoalInterpretationReview(FrozenEvaluationModel):
    interpretation_supported: bool
    requirements_preserved: bool
    unexplained_assumptions: tuple[str, ...] = ()
    explanation: str = Field(min_length=8, max_length=2400)


class RegionGoalInterpretationReview(GoalInterpretationReview):
    coverage_checks: tuple[RegionCoverageJudgment, ...] = Field(min_length=3, max_length=3)


class CriterionEntailment(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1)
    supported: bool
    requirement_basis: Literal[
        "explicit_instruction", "default_preservation", "perceptual_quality", "unsupported"
    ]
    explanation: str = Field(min_length=8, max_length=1800)


class UtilityAnswer(MinimalAtomicAnswer):
    """An audio grade may be a half-step after two integer polarity judgments."""

    grade: float | None = Field(ge=0, le=3, allow_inf_nan=False)

    @field_validator('grade', mode='before')
    @classmethod
    def equivalent_integer_grade(cls, value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int,float)) or value*2 not in range(7):
            raise ValueError('audio grade must be a finite integer or half-grade from 0 to 3')
        return value


class UtilityAssessment(FrozenEvaluationModel):
    design_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answers: tuple[UtilityAnswer, ...] = ()
    errors: tuple[str, ...] = ()
    assessor_role: Literal["online_selection_proxy", "independent_outcome"]
    regional_observations: tuple[RegionalObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value,
    )
    scope_review: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)


def utility_bounds(design: PublicUtilityDesign | None, review: UtilityAssessment | None):
    """Dimension means, capped by every preservation condition. Unknown is [0,1].

    These are identification bounds under recorded judgments, not confidence
    intervals and not a calibrated probability of human satisfaction.
    """
    from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint

    if design is None:
        if review is not None:
            raise ValueError("an audio assessment requires a bound public design")
        return {"lower": 0.0, "upper": 1.0, "dimensions": {}, "preservation": {},
                "known_criteria": 0, "assigned_criteria": 0, "design_pending": True}
    answers = {}
    if review is not None:
        if review.design_fingerprint != canonical_evaluation_fingerprint(design):
            raise ValueError("utility review belongs to a different public design")
        answers = {a.question_id: a.grade for a in review.answers}
        if len(answers) != len(review.answers) or set(answers) - {
            c.criterion_id for c in design.criteria
        }:
            raise ValueError("utility review repeats or invents a criterion")
    dimensions, preservation = {}, {}
    for dimension in ("goal", "preservation", "perceptual_quality"):
        criteria = [c for c in design.criteria if c.dimension == dimension]
        if not criteria:  # Explicitly scoped designs may exempt an inapplicable dimension.
            continue
        denominator = sum(c.weight for c in criteria)
        lower = (
            sum(c.weight * (answers.get(c.criterion_id) or 0) / 3 for c in criteria) / denominator
        )
        upper = (
            sum(
                c.weight
                * (1 if answers.get(c.criterion_id) is None else answers[c.criterion_id] / 3)
                for c in criteria
            )
            / denominator
        )
        dimensions[dimension] = (lower, upper)
        if dimension == "preservation":
            preservation = {c.criterion_id: answers.get(c.criterion_id) for c in criteria}
    preservation_lower = min(((v or 0) / 3 for v in preservation.values()), default=1.0)
    preservation_upper = min((1 if v is None else v / 3 for v in preservation.values()), default=1.0)
    result = {
        "lower": min(*(v[0] for v in dimensions.values()), preservation_lower),
        "upper": min(*(v[1] for v in dimensions.values()), preservation_upper),
        "dimensions": dimensions,
        "preservation": preservation,
        "known_criteria": sum(v is not None for v in answers.values()),
        "assigned_criteria": len(design.criteria),
    }
    if isinstance(design, RegionUtilityDesign):
        critical = {c.criterion_id: answers.get(c.criterion_id) for c in design.region_checks
                    if c.focus in PRIMARY_FOCI}
        result["critical_constraints"] = critical
        result["lower"] = min(result["lower"], *((v or 0) / 3 for v in critical.values()))
        result["upper"] = min(result["upper"], *(1 if v is None else v / 3 for v in critical.values()))
        result["region_exemptions"] = [e.model_dump(mode="json") for e in design.region_exemptions]
    return result
