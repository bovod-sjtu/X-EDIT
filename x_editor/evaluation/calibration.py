"""Independent synthetic calibration for semantic Evaluator backends.

The suite contains no benchmark rows or expected benchmark answers.  It checks
whether a judge can distinguish a correct module output, a single semantic
defect, a wrong decline, a malformed PlanIR, and genuinely missing evidence.
Fixture runs validate the harness only; they are never reported as live judge
calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import (
    EvaluationEvidenceItem,
    EvaluationJudgePolicy,
    EvaluationSamplingPolicy,
    FrozenEvaluationModel,
    SemanticPathEvaluationTask,
)
from x_editor.evaluation.guidance_path import (
    EvaluationGuidancePath,
    freeze_evaluation_selection,
)
from x_editor.evaluation.guidance_prompts import build_default_semantic_rubrics

CalibrationCategory = Literal[
    "correct_output",
    "single_semantic_defect",
    "wrong_decline",
    "bad_plan_ir_mutation",
    "missing_evidence_abstention",
]
CalibrationRunMode = Literal["fixture", "live"]
CalibrationSplit = Literal["development", "validation"]


class EvaluationCalibrationCase(FrozenEvaluationModel):
    case_id: str = Field(min_length=1)
    category: CalibrationCategory
    focus: Literal["reasoner_semantics", "planner_decision", "plan_graph"]
    public_instruction: str = Field(min_length=1)
    subject_output: EvaluationEvidenceItem
    context_evidence: tuple[EvaluationEvidenceItem, ...] = ()
    candidate_operation_cards: tuple[EvaluationEvidenceItem, ...] = ()
    expected_weak_rubric_ids: tuple[str, ...] = ()
    expected_abstain_rubric_ids: tuple[str, ...] = ()
    expected_guidance_actionable: bool
    task_group_id: str | None = None
    source_group_id: str | None = None
    split: CalibrationSplit | None = None

    @model_validator(mode="after")
    def validate_expectations(self) -> EvaluationCalibrationCase:
        grouping = (self.task_group_id, self.source_group_id, self.split)
        if any(value is not None for value in grouping) and not all(grouping):
            raise ValueError("grouped calibration requires task, source, and split")
        if self.category == "correct_output" and (
            self.expected_weak_rubric_ids
            or self.expected_abstain_rubric_ids
            or self.expected_guidance_actionable
        ):
            raise ValueError("correct calibration output cannot expect a weakness")
        if self.category == "missing_evidence_abstention" and not self.expected_abstain_rubric_ids:
            raise ValueError("missing-evidence calibration requires an abstention target")
        if self.category in {"single_semantic_defect", "wrong_decline", "bad_plan_ir_mutation"} and not self.expected_weak_rubric_ids:
            raise ValueError("defective calibration output requires a weakness target")
        valid_rubrics = {item.rubric_id for item in build_default_semantic_rubrics(self.focus)}
        if not {*self.expected_weak_rubric_ids, *self.expected_abstain_rubric_ids} <= valid_rubrics:
            raise ValueError("calibration expectations name a rubric outside the selected focus")
        return self


class EvaluationCalibrationGates(FrozenEvaluationModel):
    defect_detection_rate: float = Field(default=0.9, ge=0.0, le=1.0)
    correct_output_false_positive_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    missing_evidence_abstention_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    wrong_decline_detection_rate: float = Field(default=0.9, ge=0.0, le=1.0)


class EvaluationCalibrationSuite(FrozenEvaluationModel):
    schema_version: Literal["x-editor.evaluation-calibration-suite/v1"] = (
        "x-editor.evaluation-calibration-suite/v1"
    )
    suite_id: str = Field(min_length=1)
    benchmark_cases_consumed: Literal[False] = False
    cases: tuple[EvaluationCalibrationCase, ...] = Field(min_length=5)
    gates: EvaluationCalibrationGates = Field(default_factory=EvaluationCalibrationGates)

    @model_validator(mode="after")
    def validate_coverage(self) -> EvaluationCalibrationSuite:
        categories = {item.category for item in self.cases}
        required = set(CalibrationCategory.__args__)
        if categories != required:
            raise ValueError(f"calibration suite category coverage mismatch: {categories}")
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("calibration case IDs must be unique")
        for field in ("task_group_id", "source_group_id"):
            splits: dict[str, str] = {}
            for case in self.cases:
                group = getattr(case, field)
                if group is None:
                    continue
                if group in splits and splits[group] != case.split:
                    raise ValueError(f"calibration {field} leaks across splits: {group}")
                splits[group] = case.split
        return self


class EvaluationCalibrationCaseResult(FrozenEvaluationModel):
    case_id: str
    category: CalibrationCategory
    actionable_rubric_ids: tuple[str, ...]
    abstained_rubric_ids: tuple[str, ...]
    expected_weak_detected: bool
    expected_abstention_detected: bool
    guidance_actionability_matches: bool
    task_group_id: str | None = None
    source_group_id: str | None = None
    split: CalibrationSplit | None = None
    evaluation_completed: bool = True
    failed_slot_count: int = 0
    minimum_evidence_abstained_rubric_ids: tuple[str, ...] = ()
    known_rubric_count: int = 0
    rubric_count: int = 0


class CalibrationRateEstimate(FrozenEvaluationModel):
    """Descriptive rate; correlated rows do not receive a binomial interval."""

    numerator: int
    denominator: int
    task_group_count: int | None
    source_group_count: int | None
    wilson_95: tuple[float, float] | None
    interval_note: str


class EvaluationCalibrationReport(FrozenEvaluationModel):
    schema_version: Literal["x-editor.evaluation-calibration-report/v1"] = (
        "x-editor.evaluation-calibration-report/v1"
    )
    suite_id: str
    run_mode: CalibrationRunMode
    case_results: tuple[EvaluationCalibrationCaseResult, ...]
    metrics: dict[str, float]
    gate_results: dict[str, bool]
    all_gates_passed: bool
    # Compatibility field: run_mode is caller-declared, not backend attestation.
    live_calibration_claim_allowed: bool
    fixture_validation_only: bool
    claim_scope: Literal["caller_declared_mode_and_suite_engineering_gates"] = (
        "caller_declared_mode_and_suite_engineering_gates"
    )
    metric_estimates: dict[str, CalibrationRateEstimate] = Field(default_factory=dict)
    split_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    split_metric_estimates: dict[str, dict[str, CalibrationRateEstimate]] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()


def load_evaluation_calibration_suite(
    path: str | Path,
) -> EvaluationCalibrationSuite:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return EvaluationCalibrationSuite.model_validate(payload)


def run_evaluation_calibration(
    suite: EvaluationCalibrationSuite,
    *,
    path: EvaluationGuidancePath,
    backend_id: str,
    judge_policy: EvaluationJudgePolicy,
    run_mode: CalibrationRunMode,
    split: CalibrationSplit | None = None,
) -> EvaluationCalibrationReport:
    cases = tuple(case for case in suite.cases if split is None or case.split == split)
    if not cases:
        raise ValueError("calibration selection contains no cases")
    if {case.category for case in cases} != set(CalibrationCategory.__args__):
        raise ValueError("calibration split must cover every category")
    case_results: list[EvaluationCalibrationCaseResult] = []
    for case in cases:
        task = _calibration_task(
            suite,
            case,
            backend_id=backend_id,
            judge_policy=judge_policy,
        )
        outcome = path.evaluate(task)
        slots = outcome.result.slot_results if outcome.result else ()
        completed = (
            outcome.result is not None
            and sum(item.status == "completed" for item in slots)
            >= judge_policy.minimum_valid_slots
        )
        if outcome.result is None:
            actionable: tuple[str, ...] = ()
            abstained: tuple[str, ...] = ()
        else:
            actionable = tuple(
                item.rubric_id
                for item in outcome.result.rubric_results
                if item.guidance_actionable
            )
            abstained = tuple(
                item.rubric_id
                for item in outcome.result.rubric_results
                if item.status == "unknown"
            )
        expected_weak = set(case.expected_weak_rubric_ids)
        expected_abstain = set(case.expected_abstain_rubric_ids)
        case_results.append(
            EvaluationCalibrationCaseResult(
                case_id=case.case_id,
                category=case.category,
                actionable_rubric_ids=actionable,
                abstained_rubric_ids=abstained,
                expected_weak_detected=completed and expected_weak.issubset(actionable),
                expected_abstention_detected=(
                    completed and expected_abstain.issubset(abstained)
                ),
                guidance_actionability_matches=(
                    completed and bool(actionable) == case.expected_guidance_actionable
                ),
                task_group_id=case.task_group_id,
                source_group_id=case.source_group_id,
                split=case.split,
                evaluation_completed=completed,
                failed_slot_count=sum(item.status != "completed" for item in slots),
                minimum_evidence_abstained_rubric_ids=tuple(
                    item.rubric_id for item in outcome.result.rubric_results
                    if item.status == "unknown" and item.evidence_status != "sufficient"
                ) if outcome.result else (),
                known_rubric_count=sum(
                    item.status != "unknown" for item in outcome.result.rubric_results
                ) if outcome.result else 0,
                rubric_count=len(task.rubrics),
            )
        )

    metrics = _calibration_metrics(tuple(case_results))
    gate_results = {
        "all_evaluations_completed": all(item.evaluation_completed for item in case_results),
        "correct_outputs_have_known_judgements": metrics["correct_output_known_rate"] == 1.0,
        "defect_detection_rate": (
            metrics["defect_detection_rate"] >= suite.gates.defect_detection_rate
        ),
        "correct_output_false_positive_rate": (
            metrics["correct_output_false_positive_rate"]
            <= suite.gates.correct_output_false_positive_rate
        ),
        "missing_evidence_abstention_rate": (
            metrics["missing_evidence_abstention_rate"]
            >= suite.gates.missing_evidence_abstention_rate
        ),
        "wrong_decline_detection_rate": (
            metrics["wrong_decline_detection_rate"]
            >= suite.gates.wrong_decline_detection_rate
        ),
    }
    passed = all(gate_results.values())
    return EvaluationCalibrationReport(
        suite_id=suite.suite_id,
        run_mode=run_mode,
        case_results=tuple(case_results),
        metrics=metrics,
        gate_results=gate_results,
        all_gates_passed=passed,
        live_calibration_claim_allowed=run_mode == "live" and passed,
        fixture_validation_only=run_mode == "fixture",
        metric_estimates=_calibration_estimates(tuple(case_results)),
        split_metrics={
            name: _calibration_metrics(tuple(item for item in case_results if item.split == name))
            for name in sorted({item.split for item in case_results if item.split})
        },
        split_metric_estimates={
            name: _calibration_estimates(tuple(item for item in case_results if item.split == name))
            for name in sorted({item.split for item in case_results if item.split})
        },
        category_counts={
            category: sum(item.category == category for item in case_results)
            for category in CalibrationCategory.__args__
        },
        limitations=(
            "Run mode is caller-declared; passing suite gates does not establish independent judge reliability or authorize runtime feedback.",
            "Synthetic task groups are authored engineering probes, not a random population sample.",
            "Minimum-evidence enforcement can create abstention; its success is not raw judge calibration.",
            "Unknown judgements and provider failures remain in the selected denominator.",
            "Fixture mode measures harness correctness only; no model or audio quality claim is permitted.",
        ),
    )


def _calibration_task(
    suite: EvaluationCalibrationSuite,
    case: EvaluationCalibrationCase,
    *,
    backend_id: str,
    judge_policy: EvaluationJudgePolicy,
) -> SemanticPathEvaluationTask:
    selection_policy = EvaluationSamplingPolicy(
        policy_id=f"{suite.suite_id}:all",
        mode="all_steps",
    )
    selection = freeze_evaluation_selection(
        selection_policy,
        unit_id=case.case_id,
        focus=case.focus,
    )
    return SemanticPathEvaluationTask(
        task_id=f"calibration:{case.case_id}",
        run_id=f"calibration:{suite.suite_id}",
        checkpoint_id=f"calibration-checkpoint:{case.case_id}",
        input_snapshot_ref=f"calibration-input:{case.case_id}",
        focus=case.focus,
        evaluator_role="offline_prior_estimator",
        evaluation_use="calibration",
        backend_id=backend_id,
        public_instruction=case.public_instruction,
        subject_output=case.subject_output,
        context_evidence=case.context_evidence,
        candidate_operation_cards=case.candidate_operation_cards,
        rubrics=build_default_semantic_rubrics(case.focus),
        judge_policy=judge_policy,
        selection=selection,
    )


def _calibration_metrics(
    results: tuple[EvaluationCalibrationCaseResult, ...],
) -> dict[str, float]:
    return {
        name: _rate(sum(successes), len(rows))
        for name, (rows, successes) in _metric_observations(results).items()
    }


def _metric_observations(results: tuple[EvaluationCalibrationCaseResult, ...]) -> dict:
    defect_rows = [
        item
        for item in results
        if item.category in {
            "single_semantic_defect",
            "wrong_decline",
            "bad_plan_ir_mutation",
        }
    ]
    correct_rows = [item for item in results if item.category == "correct_output"]
    abstain_rows = [
        item for item in results if item.category == "missing_evidence_abstention"
    ]
    wrong_decline_rows = [item for item in results if item.category == "wrong_decline"]
    return {
        "defect_detection_rate": (defect_rows, [item.expected_weak_detected for item in defect_rows]),
        "correct_output_false_positive_rate": (
            correct_rows, [bool(item.actionable_rubric_ids) for item in correct_rows]
        ),
        "missing_evidence_abstention_rate": (
            abstain_rows, [item.expected_abstention_detected for item in abstain_rows]
        ),
        "wrong_decline_detection_rate": (
            wrong_decline_rows, [item.expected_weak_detected for item in wrong_decline_rows]
        ),
        "evaluation_completed_rate": (list(results), [item.evaluation_completed for item in results]),
        "correct_output_known_rate": (
            correct_rows,
            [item.evaluation_completed and item.known_rubric_count == item.rubric_count
             for item in correct_rows],
        ),
    }


def _calibration_estimates(
    results: tuple[EvaluationCalibrationCaseResult, ...],
) -> dict[str, CalibrationRateEstimate]:
    estimates = {}
    for name, (rows, successes) in _metric_observations(results).items():
        count = len(rows)
        groups = {item.task_group_id for item in rows}
        sources = {item.source_group_id for item in rows}
        grouped = None not in groups and None not in sources
        independent_rows = grouped and len(groups) == count and len(sources) == count
        numerator = sum(successes)
        interval = None
        if count and independent_rows:
            z = 1.959963984540054
            rate = numerator / count
            scale = 1 + z * z / count
            center = (rate + z * z / (2 * count)) / scale
            radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2)) / scale
            interval = (max(0.0, center - radius), min(1.0, center + radius))
        estimates[name] = CalibrationRateEstimate(
            numerator=numerator, denominator=count,
            task_group_count=len(groups) if grouped else None,
            source_group_count=len(sources) if grouped else None,
            wilson_95=interval,
            interval_note=(
                "Descriptive Wilson interval under independent-group Bernoulli assumptions; not population coverage."
                if interval is not None else
                "No binomial interval: repeated groups or missing grouping metadata."
            ),
        )
    return estimates


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "EvaluationCalibrationCase",
    "EvaluationCalibrationCaseResult",
    "EvaluationCalibrationGates",
    "EvaluationCalibrationReport",
    "EvaluationCalibrationSuite",
    "load_evaluation_calibration_suite",
    "run_evaluation_calibration",
]
