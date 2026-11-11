"""Output-blind, task-specific evidence questions for semantic module review.

This module owns no execution authority. A rubric author receives public intent
only; each independent judge later sees one frozen question and the actual
checkpoint. Missing upstream evidence can coexist with a correct request to
observe it. Unknown grades remain in the denominator as an interval.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import (
    FrozenEvaluationModel,
    canonical_evaluation_fingerprint,
)

PROTOCOL_ID = "x-editor.atomic-module-evidence-review/v3"
Module = Literal["reasoner", "planner"]
PlanPart = Literal[
    "decision",
    "base_asset_version",
    "operation",
    "parameters",
    "evidence_binding",
    "dependencies",
    "preservation",
    "output_selection",
]


class PublicModuleReviewContext(FrozenEvaluationModel):
    """No candidate, historical score, private design, or golden plan fields."""

    turns: tuple[dict[str, Any], ...] = Field(min_length=1)
    target_turn_index: int = Field(ge=0)
    audio_inventory: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_prefix(self):
        if len(self.turns) != self.target_turn_index + 1:
            raise ValueError("author context must end at the evaluated turn")
        allowed_turn = {
            "turn_index",
            "user_request",
            "source_text",
            "source_asset_refs",
            "reference_asset_refs",
            "use_previous_output",
        }
        allowed_audio = {"asset_ref", "role", "content_sha256", "order"}
        if any(set(turn) - allowed_turn for turn in self.turns):
            raise ValueError("non-public or unsupported author turn field")
        if any(set(asset) - allowed_audio for asset in self.audio_inventory):
            raise ValueError("author inventory must not contain recipes or hidden targets")
        if any(turn.get("turn_index") != i for i, turn in enumerate(self.turns)):
            raise ValueError("public turns must have contiguous, explicit indices")
        return self

    @property
    def fingerprint(self):
        return canonical_evaluation_fingerprint(self.model_dump(mode="json"))


class NecessaryEvidence(FrozenEvaluationModel):
    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    fact_needed: str = Field(min_length=4, max_length=1200)
    why_needed: str = Field(min_length=4, max_length=1200)
    basis_turn_indices: tuple[int, ...] = Field(min_length=1)
    sufficient_evidence: str = Field(min_length=4, max_length=1200)
    if_missing: str = Field(min_length=4, max_length=1200)


class PlanPartQuestion(FrozenEvaluationModel):
    check_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    part: PlanPart
    question: str = Field(min_length=8, max_length=1600)
    success_condition: str = Field(min_length=8, max_length=1600)
    requirement_ids: tuple[str, ...] = ()


class CaseEvidenceDesign(FrozenEvaluationModel):
    """A coverage hypothesis, never a certificate of complete goal coverage."""

    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    intended_state: str = Field(min_length=4, max_length=2000)
    necessary_evidence: tuple[NecessaryEvidence, ...] = Field(min_length=1, max_length=32)
    planner_checks: tuple[PlanPartQuestion, ...] = Field(min_length=1, max_length=32)
    non_applicable_plan_parts: dict[PlanPart, str] = Field(default_factory=dict)
    unresolved_goal_questions: tuple[str, ...] = ()
    coverage_certified: Literal[False] = False
    authored_without_candidate: Literal[True] = True

    @model_validator(mode="after")
    def validate_ids(self):
        needs = [item.requirement_id for item in self.necessary_evidence]
        checks = [item.check_id for item in self.planner_checks]
        if len(needs) != len(set(needs)) or len(checks) != len(set(checks)):
            raise ValueError("atomic question identities must be unique")
        if any(set(check.requirement_ids) - set(needs) for check in self.planner_checks):
            raise ValueError("planner question refers to an unauthored evidence requirement")
        # These are framework dimensions, not fixed generic scoring questions.
        # Each still needs an independently authored, task-specific criterion.
        required = set(PlanPart.__args__)
        checked_parts = {check.part for check in self.planner_checks}
        omitted_parts = set(self.non_applicable_plan_parts)
        if checked_parts & omitted_parts:
            raise ValueError("a PlanIR part cannot be both scored and declared inapplicable")
        if "decision" in omitted_parts or any(
            not reason.strip() for reason in self.non_applicable_plan_parts.values()
        ):
            raise ValueError("decision always needs review; other exclusions need explicit reasons")
        if checked_parts | omitted_parts != required:
            missing = sorted(required - checked_parts - omitted_parts)
            raise ValueError(
                "design must address each PlanIR part, conditionally when appropriate; "
                f"missing_parts={missing}"
            )
        return self


class AtomicModuleQuestion(FrozenEvaluationModel):
    question_id: str
    module: Module
    kind: Literal["evidence_handling", "plan_part"]
    question: str
    success_condition: str
    requirement_ids: tuple[str, ...]
    part: PlanPart | None = None


class ReviewCitation(FrozenEvaluationModel):
    source: Literal["public_context", "module_input", "module_output", "contract_audit"]
    pointer: str = Field(pattern=r"^(|/.*)$")
    excerpt: str = Field(min_length=1, max_length=1800)


class AtomicModuleAnswer(FrozenEvaluationModel):
    question_id: str
    input_evidence_status: Literal[
        "sufficient", "missing", "conflicting", "not_required", "unknown"
    ]
    available_evidence: str
    missing_or_conflicting_evidence: tuple[str, ...]
    expected_module_behavior: str
    observed_module_behavior: str
    citations: tuple[ReviewCitation, ...] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=2500)
    verdict: Literal["met", "partial", "violated", "unknown"]
    grade: int | None = Field(default=None, ge=0, le=3)

    @model_validator(mode="after")
    def validate_verdict(self):
        allowed = {"met": {3}, "partial": {1, 2}, "violated": {0}, "unknown": {None}}
        if self.grade not in allowed[self.verdict]:
            raise ValueError(
                "grade and declared verdict disagree; original response must be retained"
            )
        return self


def bind_design(context: PublicModuleReviewContext, payload: dict) -> CaseEvidenceDesign:
    data = dict(payload)
    data.setdefault("input_fingerprint", context.fingerprint)
    design = CaseEvidenceDesign.model_validate(data)
    if design.input_fingerprint != context.fingerprint:
        raise ValueError("rubrics belong to another public context")
    indices = set(range(len(context.turns)))
    if any(set(need.basis_turn_indices) - indices for need in design.necessary_evidence):
        raise ValueError("rubric uses future or nonexistent conversation turns")
    return design


def bind_authored_design(context: PublicModuleReviewContext, payload: dict) -> CaseEvidenceDesign:
    """Bind framework-owned identity for a newly returned, recorded model draft.

    The author is not asked to calculate a fingerprint. Stored designs still use
    strict bind_design, so loading a design from another context remains invalid.
    """
    data = dict(payload)
    data.pop("input_fingerprint", None)
    return bind_design(context, data)


def module_questions(
    design: CaseEvidenceDesign, module: Module
) -> tuple[AtomicModuleQuestion, ...]:
    if module not in ("reasoner", "planner"):
        raise ValueError("only Reasoner and Planner questions are supported")
    questions = []
    owner = (
        "Reasoner的解释、证据判断及补感知需求"
        if module == "reasoner"
        else "Planner的决策、证据引用及实际参数绑定"
    )
    for need in design.necessary_evidence:
        questions.append(
            AtomicModuleQuestion(
                question_id=f"{module}_evidence_{need.requirement_id}",
                module=module,
                kind="evidence_handling",
                requirement_ids=(need.requirement_id,),
                question=f"本任务需要确认：{need.fact_needed}。实际输入是否足够？{owner}是否据此作出了正确处理？",
                success_condition=(
                    f"必要性：{need.why_needed}。充分证据：{need.sufficient_evidence}。"
                    f"缺失时：{need.if_missing}。只要求当前模块承担自身职责；"
                    "正确识别缺口并针对性请求感知可以满分，不把缺证据直接当目标不存在。"
                ),
            )
        )
    if module == "planner":
        questions.extend(
            AtomicModuleQuestion(
                question_id=f"planner_part_{check.check_id}",
                module=module,
                kind="plan_part",
                question=check.question,
                success_condition=check.success_condition,
                requirement_ids=check.requirement_ids,
                part=check.part,
            )
            for check in design.planner_checks
        )
    return tuple(questions)


def resolve_pointer(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    current = value
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def validate_answer(
    question: AtomicModuleQuestion, payload: dict, sources: dict
) -> AtomicModuleAnswer:
    answer = AtomicModuleAnswer.model_validate(payload)
    if answer.question_id != question.question_id:
        raise ValueError("judge answered a different atomic question")
    for citation in answer.citations:
        try:
            value = resolve_pointer(sources[citation.source], citation.pointer)
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise ValueError("judge cited an unavailable evidence path") from error
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if citation.excerpt not in text and not structured_citation_matches(
            value, citation.excerpt
        ):
            raise ValueError("judge evidence excerpt does not occur at its declared path")
    return answer


def structured_citation_matches(value: Any, excerpt: str) -> bool:
    """Compare complete quoted JSON fields at the declared pointer.

    Object order and JSON whitespace do not change evidence identity. We never
    search for a better pointer, change string values, or repair model grades.
    """
    if not isinstance(value, (dict, list)):
        return False

    def included(quoted, actual):
        if isinstance(quoted, dict):
            return (
                bool(quoted)
                and isinstance(actual, dict)
                and all(
                    key in actual and included(item, actual[key]) for key, item in quoted.items()
                )
            )
        if isinstance(quoted, list):
            return (
                isinstance(actual, list)
                and len(quoted) == len(actual)
                and all(included(left, right) for left, right in zip(quoted, actual, strict=True))
            )
        if isinstance(quoted, bool) or isinstance(actual, bool):
            return type(quoted) is type(actual) and quoted == actual
        return quoted == actual

    candidates = [excerpt]
    if isinstance(value, dict) and not excerpt.lstrip().startswith("{"):
        candidates.append("{" + excerpt + "}")
    for candidate in candidates:
        try:
            quoted = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(quoted, type(value)) and included(quoted, value):
            return True
    return False


def normalized_atomic_summary(questions, answers):
    """Do not let omitted, invalid, or unknown questions inflate the mean."""
    by_id = {answer.question_id: answer for answer in answers}
    if len(by_id) != len(answers):
        raise ValueError("duplicate answers cannot add scoring weight")
    expected = {question.question_id for question in questions}
    if set(by_id) - expected:
        raise ValueError("unassigned answers cannot enter this module score")
    known = [
        by_id[q.question_id].grade
        for q in questions
        if q.question_id in by_id and by_id[q.question_id].grade is not None
    ]
    count = len(questions)
    if not count:
        raise ValueError("a module needs at least one assigned question")
    unknown = count - len(known)
    lower = sum(known) / (3 * count)
    upper = (sum(known) + 3 * unknown) / (3 * count)
    return {
        "assigned_questions": count,
        "graded_questions": len(known),
        "unknown_questions": unknown,
        "normalized_score": lower if not unknown else None,
        "normalized_lower_bound": lower,
        "normalized_upper_bound": upper,
        "known_question_mean": sum(known) / (3 * len(known)) if known else None,
        "violated_questions": sum(grade == 0 for grade in known),
        "partial_questions": sum(grade in (1, 2) for grade in known),
        "all_required_conditions_met": all(grade == 3 for grade in known) if not unknown else None,
        "capacity_qualified": False,
    }
