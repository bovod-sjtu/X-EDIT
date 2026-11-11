"""Deterministic shadow recommendations over existing repair evidence."""

from __future__ import annotations

from typing import Any

from x_editor.workflow_control.schemas import (
    RepairAction,
    RepairComparison,
    RepairComparisonRelation,
    RepairDecision,
    RepairFailureClass,
    RepairOutcome,
    RepairScope,
    RepairSignal,
    RepairSignalSource,
)


class RepairController:
    """Classify repair evidence without mutating legacy execution behavior."""

    policy_version = "x-editor.repair-controller/deterministic-v1"

    def recommend(
        self,
        signal: RepairSignal,
        *,
        legacy_action: RepairAction,
        budget_delta: dict[str, float] | None = None,
        shadow_only: bool = True,
    ) -> RepairDecision:
        action, scope = self._select_action(signal)
        return RepairDecision(
            stage=signal.source,
            action=action,
            scope=scope,
            plain_summary=_recommendation_summary(action, signal),
            subject_refs=signal.subject_refs,
            reason_codes=signal.reason_codes or [signal.failure_class],
            evidence_refs=signal.evidence_refs,
            budget_delta=_recommended_budget(action, signal.source),
            shadow_only=shadow_only,
            legacy_action_observed=legacy_action,
            metadata={
                "policy_version": self.policy_version,
                "legacy_budget_delta": dict(budget_delta or {}),
                "signal": signal.model_dump(mode="json"),
            },
        )

    def compare(
        self,
        signal: RepairSignal,
        decision: RepairDecision,
        *,
        legacy_action: RepairAction,
        legacy_scope: RepairScope,
        outcome: RepairOutcome,
        budget_delta: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RepairComparison:
        return RepairComparison(
            signal_id=signal.signal_id,
            decision_id=decision.decision_id,
            legacy_action=legacy_action,
            legacy_scope=legacy_scope,
            relation=_comparison_relation(
                decision.action,
                decision.scope,
                legacy_action,
                legacy_scope,
            ),
            outcome=outcome,
            budget_delta=dict(budget_delta or {}),
            metadata={
                "policy_version": self.policy_version,
                **dict(metadata or {}),
            },
        )

    def signal_from_tool_contract(
        self,
        violations: list[dict[str, Any]],
        *,
        source: RepairSignalSource = "tool_contract",
    ) -> RepairSignal:
        failure_class = _classify_tool_contract(violations)
        subjects = _dedupe(
            str(item.get("step_id"))
            for item in violations
            if item.get("step_id")
        )
        reason_codes = _dedupe(
            _reason_code(str(item.get("reason") or "tool contract violation"))
            for item in violations
        )
        suggested_operations = [
            str(operation)
            for item in violations
            for operation in item.get("suggested_operations") or []
            if operation
        ]
        return RepairSignal(
            source=source,
            failure_class=failure_class,
            plain_summary="A planned tool call does not satisfy its runtime contract.",
            subject_refs=subjects,
            reason_codes=reason_codes or ["tool_contract_violation"],
            evidence_refs=[
                f"tool_contract:{item.get('step_id') or 'unknown'}:{item.get('stage') or 'unknown'}"
                for item in violations
            ],
            local_repair_available=failure_class
            in {"argument_contract", "evidence_missing"},
            global_semantic_change_required=failure_class
            == "objective_or_capability",
            candidate_actions=_tool_contract_candidates(
                failure_class, bool(suggested_operations)
            ),
            metadata={"violations": violations},
        )

    def signal_from_runtime_verification(
        self,
        defects: list[dict[str, Any]],
        repair_plans: list[dict[str, Any]],
    ) -> RepairSignal:
        components = _dedupe(
            str(item.get("component")) for item in defects if item.get("component")
        )
        subjects = _dedupe(
            str(item.get("step_id")) for item in defects if item.get("step_id")
        )
        micro_actions = _repair_actions(repair_plans, "micro_actions")
        major_actions = _repair_actions(repair_plans, "major_actions")
        failure_class = _classify_verifier_components(components)
        local_available = len(subjects) <= 1 and bool(micro_actions)
        candidates = _runtime_candidates(
            failure_class,
            local_available=local_available,
            has_major_actions=bool(major_actions),
        )
        return RepairSignal(
            source="runtime_verification",
            failure_class=failure_class,
            plain_summary="Intermediate verification found a localized execution defect.",
            subject_refs=subjects,
            reason_codes=[f"verifier:{component}" for component in components]
            or ["runtime_verification_defect"],
            evidence_refs=[
                f"verification:{item.get('step_id') or 'unknown'}:{item.get('component') or 'unknown'}"
                for item in defects
            ],
            local_repair_available=local_available,
            candidate_actions=candidates,
            metadata={
                "defects": defects,
                "micro_actions": micro_actions,
                "major_actions": major_actions,
            },
        )

    def signal_from_evaluation(self, evaluation: dict[str, Any]) -> RepairSignal:
        subjects = _dedupe(str(item) for item in evaluation.get("failed_steps") or [])
        objective_failure = not subjects or any(
            subject.startswith("objective:") for subject in subjects
        )
        failure_class: RepairFailureClass = (
            "objective_or_capability" if objective_failure else "artifact_quality"
        )
        return RepairSignal(
            source="evaluator",
            failure_class=failure_class,
            plain_summary="Final evaluation reports a fixable failure.",
            subject_refs=subjects,
            reason_codes=["final_evaluation_fixable_failure"],
            evidence_refs=[f"evaluation:{subject}" for subject in subjects],
            local_repair_available=bool(subjects) and not objective_failure,
            global_semantic_change_required=objective_failure,
            candidate_actions=[
                "replan_global" if objective_failure else "replan_suffix"
            ],
            metadata={"evaluation": evaluation},
        )

    def signal_from_bounded_attempt(self, attempt: dict[str, Any]) -> RepairSignal:
        action = attempt.get("action") if isinstance(attempt.get("action"), dict) else {}
        component = str(action.get("component") or "unknown")
        strategy = str(action.get("strategy") or "unknown")
        step_id = _step_id_from_action(action)
        failure_class = _classify_verifier_components([component])
        return RepairSignal(
            source="step_local",
            failure_class=failure_class,
            plain_summary="The executor attempted a bounded repair of the current step.",
            subject_refs=[step_id] if step_id else [],
            reason_codes=[f"bounded_repair:{strategy}"],
            evidence_refs=[f"bounded_repair:{action.get('action_id') or strategy}"],
            local_repair_available=True,
            candidate_actions=["repair_step_arguments"],
            metadata={"attempt": attempt},
        )

    def signal_from_local_tool_contract_attempt(
        self, attempt: dict[str, Any]
    ) -> RepairSignal:
        verifier_error = attempt.get("verifier_error")
        violations = (
            verifier_error.get("violations")
            if isinstance(verifier_error, dict)
            else []
        )
        signal = self.signal_from_tool_contract(
            [item for item in violations or [] if isinstance(item, dict)],
            source="step_local",
        )
        return signal.model_copy(
            update={
                "plain_summary": "The executor attempted to repair one tool call before invocation.",
                "subject_refs": [str(attempt["step_id"])]
                if attempt.get("step_id")
                else signal.subject_refs,
                "metadata": {"attempt": attempt},
            }
        )

    def signal_from_runtime_local_invocation(
        self, invocation: dict[str, Any]
    ) -> RepairSignal:
        step_id = str(invocation.get("step_id") or "")
        strategy = str(invocation.get("runtime_repair") or "runtime_local_repair")
        return RepairSignal(
            source="step_local",
            failure_class="artifact_quality",
            plain_summary="The executor retried the current edit through a local repair path.",
            subject_refs=[step_id] if step_id else [],
            reason_codes=[f"runtime_local_repair:{strategy}"],
            evidence_refs=[f"tool_invocation:{step_id or 'unknown'}:{strategy}"],
            local_repair_available=True,
            candidate_actions=["rerun_step"],
            metadata={"invocation": invocation},
        )

    def _select_action(self, signal: RepairSignal) -> tuple[RepairAction, RepairScope]:
        if not signal.retryable:
            return "fail_structured", "run"
        if signal.global_semantic_change_required:
            return "replan_global", "plan"
        if signal.failure_class == "transient_call":
            return "retry_same_call", "call"
        if signal.failure_class == "argument_contract":
            return "repair_step_arguments", "argument"
        if signal.failure_class == "evidence_missing":
            return "reacquire_evidence", "step"
        if signal.failure_class == "alignment_or_span":
            if signal.local_repair_available:
                return "repair_step_arguments", "span"
            return "reacquire_evidence", "step"
        if signal.failure_class == "artifact_quality":
            if signal.local_repair_available:
                return "rerun_step", "step"
            return "replan_suffix", "suffix"
        if signal.failure_class == "dependency_or_route":
            return "replan_suffix", "suffix"
        if signal.local_repair_available:
            return "rerun_step", "step"
        return "replan_suffix", "suffix"


def _classify_tool_contract(
    violations: list[dict[str, Any]],
) -> RepairFailureClass:
    text = " ".join(
        str(value).lower()
        for item in violations
        if isinstance(item, dict)
        for value in [
            item.get("reason"),
            *(item.get("tool_requirements") or []),
            *(item.get("retry_guidance") or []),
        ]
        if value
    )
    suggestions = any(item.get("suggested_operations") for item in violations)
    if "unsupported" in text:
        return "objective_or_capability"
    evidence_markers = (
        "input audio",
        "audio path",
        "source_text",
        "source text",
        "reference_audio",
        "reference audio",
        "word_timestamps",
        "transcript",
    )
    if any(marker in text for marker in evidence_markers):
        return "evidence_missing"
    if suggestions:
        return "dependency_or_route"
    if violations:
        return "argument_contract"
    return "unknown"


def _classify_verifier_components(components: list[str]) -> RepairFailureClass:
    lowered = [component.lower() for component in components]
    if any("asr" in item or "source_text" in item for item in lowered):
        return "evidence_missing"
    if any("align" in item or "span" in item for item in lowered):
        return "alignment_or_span"
    if lowered:
        return "artifact_quality"
    return "unknown"


def _tool_contract_candidates(
    failure_class: RepairFailureClass, suggested_operations: bool
) -> list[RepairAction]:
    if failure_class == "argument_contract":
        return ["repair_step_arguments", "replan_suffix"]
    if failure_class == "evidence_missing":
        return ["reacquire_evidence", "replan_suffix"]
    if suggested_operations or failure_class == "dependency_or_route":
        return ["replan_suffix", "replan_global"]
    if failure_class == "objective_or_capability":
        return ["replan_global", "fail_structured"]
    return ["replan_suffix"]


def _runtime_candidates(
    failure_class: RepairFailureClass,
    *,
    local_available: bool,
    has_major_actions: bool,
) -> list[RepairAction]:
    if failure_class == "evidence_missing":
        actions: list[RepairAction] = ["reacquire_evidence"]
    elif failure_class == "alignment_or_span" and local_available:
        actions = ["repair_step_arguments"]
    elif local_available:
        actions = ["rerun_step"]
    else:
        actions = ["replan_suffix"]
    if has_major_actions and "replan_suffix" not in actions:
        actions.append("replan_suffix")
    return actions


def _repair_actions(
    repair_plans: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    return [
        action
        for repair_plan in repair_plans
        if isinstance(repair_plan, dict)
        for action in repair_plan.get(key) or []
        if isinstance(action, dict)
    ]


def _comparison_relation(
    controller_action: RepairAction,
    controller_scope: RepairScope,
    legacy_action: RepairAction,
    legacy_scope: RepairScope,
) -> RepairComparisonRelation:
    if controller_action == legacy_action and controller_scope == legacy_scope:
        return "exact"
    scope_rank = {
        "argument": 1,
        "call": 1,
        "span": 1,
        "artifact": 2,
        "step": 2,
        "suffix": 3,
        "plan": 4,
        "objective": 4,
        "run": 4,
    }
    controller_rank = scope_rank[controller_scope]
    legacy_rank = scope_rank[legacy_scope]
    if controller_rank < legacy_rank:
        return "controller_narrower"
    if controller_rank > legacy_rank:
        return "controller_broader"
    return "different"


def _recommendation_summary(action: RepairAction, signal: RepairSignal) -> str:
    return (
        f"Shadow policy recommends {action} for {signal.failure_class}; "
        "legacy execution remains authoritative."
    )


def _recommended_budget(
    action: RepairAction, source: str
) -> dict[str, float]:
    if action == "retry_same_call":
        return {"tool_calls": 1.0}
    if action == "repair_step_arguments":
        return {"executor_step_calls": 1.0}
    if action == "reacquire_evidence":
        return {"perception_calls": 1.0}
    if action == "rerun_step":
        return {"executor_step_calls": 1.0, "verifier_calls": 1.0}
    if action == "replan_suffix":
        budget = {"planner_calls": 1.0, "executor_runs": 1.0}
        if source in {"runtime_verification", "evaluator"}:
            budget["evaluator_calls"] = 1.0
        return budget
    if action == "replan_global":
        budget = {"planner_calls": 1.0, "executor_runs": 1.0}
        if source in {"runtime_verification", "evaluator"}:
            budget["evaluator_calls"] = 1.0
        return budget
    return {}


def _step_id_from_action(action: dict[str, Any]) -> str | None:
    action_id = str(action.get("action_id") or "")
    if not action_id or ":" not in action_id:
        return None
    return action_id.split(":", 1)[0]


def _reason_code(reason: str) -> str:
    normalized = "_".join(reason.lower().split())
    return normalized[:120] or "tool_contract_violation"


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
