"""Opt-in Evaluation-Guidance wrapper around the normal scenario runtime.

The wrapper never edits an attempt in place.  It runs the ordinary UniRuntime,
evaluates immutable failure evidence, and may open one complete future attempt
under a preregistered policy.  The second attempt is the declared terminal
attempt; no hidden best-of selection is allowed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.evaluation.call_accounting import (
    EvaluationCallAccounting, summarize_call_accounting,
)
from x_editor.evaluation.completed_review import validate_completed_review
from x_editor.evaluation.guidance_contracts import (
    AtomicEvaluationRubric,
    EvaluationEvidenceItem,
    EvaluationJudgePolicy,
    EvaluationResultV2,
    EvaluationSamplingPolicy,
    FrozenEvaluationModel,
    GuidancePacket,
    GuidedAttemptDecision,
    GuidedAttemptPolicy,
    SemanticPathEvaluationTask,
    EvaluationTask,
    canonical_evaluation_fingerprint,
)
from x_editor.evaluation.guidance_path import (
    EvaluationBackend,
    EvaluationGuidanceLedger,
    EvaluationGuidancePath,
    GuidedAttemptController,
    GuiderBackend,
    build_forensic_selection,
)
from x_editor.evaluation.guidance_prompts import build_default_semantic_rubrics
from x_editor.runtime.module_checkpoints import ModuleCheckpoint
from x_editor.runtime.workflow_case import WorkflowRuntimeScenario
from x_editor.runtime.workflow_orchestration import WorkflowArm
from x_editor.runtime.workflow_session import WorkflowScenarioResult


class GuidanceCapableScenarioRuntime(Protocol):
    async def arun_scenario(
        self,
        scenario: WorkflowRuntimeScenario,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        future_attempt_guidance_by_turn: dict[int, FutureAttemptGuidance] | None = None,
    ) -> WorkflowScenarioResult: ...


class EvaluationGuidedScenarioPolicy(FrozenEvaluationModel):
    """Frozen controller policy for one development feedback-loop arm."""

    schema_version: Literal["x-editor.evaluation-guided-scenario-policy/v1"] = (
        "x-editor.evaluation-guided-scenario-policy/v1"
    )
    policy_id: str = Field(min_length=1)
    mode: Literal["disabled", "audit", "enforced"] = "disabled"
    evaluator_backend_id: str = Field(min_length=1)
    guider_backend_id: str = Field(min_length=1)
    atomic_pass_threshold: float = Field(default=2.0 / 3.0, gt=0.0, le=1.0)
    minimum_known_weight_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    max_semantic_focus_evaluations: int = Field(default=2, ge=1, le=3)
    judge_policy: EvaluationJudgePolicy = Field(default_factory=EvaluationJudgePolicy)
    attempt_policy: GuidedAttemptPolicy
    eligible_failure_stages: tuple[str, ...] = (
        "reasoner",
        "planner",
        "perception",
        "workflow_protocol",
        "plan_admission",
    )
    scorer_feedback_consumed: Literal[False] = False
    expected_answer_consumed: Literal[False] = False
    benchmark_identity_visible: Literal[False] = False
    review_completed_outputs: bool = False

    @model_validator(mode="after")
    def validate_mode(self) -> EvaluationGuidedScenarioPolicy:
        if self.mode != self.attempt_policy.mode:
            raise ValueError("scenario and guided-attempt policy modes must match")
        if self.mode == "enforced" and not 1 <= self.attempt_policy.max_new_attempts <= 2:
            raise ValueError("the guided runtime supports one or two bounded new attempts")
        if len(self.eligible_failure_stages) != len(set(self.eligible_failure_stages)):
            raise ValueError("eligible failure stages must be unique")
        return self

    @property
    def policy_fingerprint(self) -> str:
        return canonical_evaluation_fingerprint(self)


class EvaluationGuidedScenarioRuntime:
    """Run immutable normal attempts under a bounded read-only guidance loop."""

    def __init__(
        self,
        *,
        base_runtime_factory: Callable[[Path], GuidanceCapableScenarioRuntime],
        evaluator_backend: EvaluationBackend,
        guider_backend: GuiderBackend,
        policy: EvaluationGuidedScenarioPolicy,
        completed_task_builder: Callable[..., tuple[int, list[EvaluationTask]]] | None = None,
    ) -> None:
        self.base_runtime_factory = base_runtime_factory
        self.evaluator_backend = evaluator_backend
        self.guider_backend = guider_backend
        self.policy = policy
        self.completed_task_builder = completed_task_builder
        if policy.review_completed_outputs and completed_task_builder is None:
            raise ValueError("completed-output review requires an explicit task builder")

    async def arun_scenario(
        self,
        scenario: WorkflowRuntimeScenario,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        initial_result: WorkflowScenarioResult | None = None,
        initial_attempt_dir: str | Path | None = None,
    ) -> WorkflowScenarioResult:
        if (initial_result is None) != (initial_attempt_dir is None):
            raise ValueError("shared initial attempt requires both its result and evidence directory")
        if initial_result is not None:
            if self.policy.mode == "disabled":
                raise ValueError("disabled wrapper cannot consume an experimental shared initial attempt")
            frozen_path = Path(initial_attempt_dir).resolve() / "scenario_result.json"
            frozen = WorkflowScenarioResult.model_validate_json(frozen_path.read_text())
            if (frozen != initial_result or frozen.scenario_fingerprint != scenario.fingerprint
                    or frozen.assigned_arm != arm):
                raise ValueError("shared initial attempt does not match the scenario and stored receipt")
        if self.policy.mode == "disabled":
            base_runtime = self.base_runtime_factory(Path(output_dir).expanduser().resolve())
            return await base_runtime.arun_scenario(
                scenario,
                arm=arm,
                output_dir=output_dir,
            )

        root = Path(output_dir).expanduser().resolve()
        guidance_root = root / "evaluation_guidance"
        attempts_root = root / "attempts"
        guidance_root.mkdir(parents=True, exist_ok=True)
        attempts_root.mkdir(parents=True, exist_ok=True)

        attempt0_id = _attempt_id(scenario, arm=arm, attempt_index=0)
        attempt0_dir = Path(initial_attempt_dir).resolve() if initial_attempt_dir else attempts_root / "0000"
        attempt0_runtime = self.base_runtime_factory(attempt0_dir)
        planner_capability_snapshot_error: dict[str, str] | None = None
        try:
            planner_capability_snapshot = _planner_capability_snapshot(
                attempt0_runtime
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics cannot block attempt 0.
            planner_capability_snapshot = None
            planner_capability_snapshot_error = {
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2048],
            }
        if initial_result is None:
            initial = await attempt0_runtime.arun_scenario(scenario, arm=arm, output_dir=attempt0_dir)
            _write_json(attempt0_dir / "scenario_result.json", initial.model_dump(mode="json"))
        else:
            initial = initial_result

        loop: dict[str, Any] = {
            "schema_version": "x-editor.evaluation-guided-scenario-result/v1",
            "scenario_fingerprint": scenario.fingerprint,
            "assigned_arm": arm,
            "policy": self.policy.model_dump(mode="json"),
            "policy_fingerprint": self.policy.policy_fingerprint,
            "attempts": [_attempt_summary(attempt0_id, 0, initial, guidance=None)],
            "evaluation_results": [],
            "evaluation_task_count": 0,
            "planner_capability_snapshot_fingerprint": (
                canonical_evaluation_fingerprint(planner_capability_snapshot)
                if planner_capability_snapshot is not None
                else None
            ),
            "planner_capability_snapshot_error": planner_capability_snapshot_error,
            "guidance_packet": None,
            "decision": None,
            "initial_status": initial.status,
            "terminal_status": initial.status,
            "terminal_attempt_index": 0,
            "initial_delivery": initial.status == "completed",
            "terminal_delivery": initial.status == "completed",
            "recovered_by_feedback_edge": False,
            "evaluator_logical_call_count": 0,
            "guider_logical_call_count": 0,
            "scorer_feedback_consumed": False,
            "expected_answer_consumed": False,
            "hidden_best_attempt_selection": False,
        }
        _write_json(guidance_root / "loop_result.json", loop)

        ledger = EvaluationGuidanceLedger(
            guidance_root / "events.jsonl",
            run_id=attempt0_id,
        )
        path = EvaluationGuidancePath(
            ledger=ledger,
            accounting=EvaluationCallAccounting(
                evaluator_http_limit=self.policy.attempt_policy.max_evaluator_http_attempts,
                guider_http_limit=self.policy.attempt_policy.max_guider_http_attempts,
            ),
        )
        path.register_evaluator(self.evaluator_backend)
        path.register_guider(self.guider_backend)
        loop.update(
            {
                "guidance_packets": [],
                "decisions": [],
                "failure_fingerprints": [],
                "guidance_fingerprints": [],
                "terminal_selection": "last_executed_attempt",
            }
        )
        current = initial
        current_index = 0
        current_id = attempt0_id
        current_dir = attempt0_dir
        failure_history: list[str] = []
        guidance_history: list[str] = []
        evaluator_calls = 0

        while True:
            if current.status == "completed" and not self.policy.review_completed_outputs:
                reason = "initial_attempt_completed" if current_index == 0 else "attempt_completed"
                decision = _no_attempt_decision(self.policy.attempt_policy, reason=reason)
                loop["decision"] = decision.model_dump(mode="json")
                loop["decisions"].append(loop["decision"])
                break
            if current.status != "completed" and current.failure_stage not in set(self.policy.eligible_failure_stages):
                decision = _no_attempt_decision(
                    self.policy.attempt_policy,
                    reason="failure_stage_not_feedback_eligible",
                )
                loop["decision"] = decision.model_dump(mode="json")
                loop["decisions"].append(loop["decision"])
                break
            if current_index >= self.policy.attempt_policy.max_new_attempts:
                decision = _no_attempt_decision(
                    self.policy.attempt_policy,
                    reason="terminal_attempt_limit_reached",
                )
                loop["decision"] = decision.model_dump(mode="json")
                loop["decisions"].append(loop["decision"])
                break

            failure_fingerprint = _controller_failure_fingerprint(
                current,
                attempt_dir=current_dir,
            )
            loop["failure_fingerprints"].append(failure_fingerprint)
            remaining_calls = self.policy.attempt_policy.max_evaluator_calls - evaluator_calls
            if remaining_calls <= 0:
                decision = _no_attempt_decision(
                    self.policy.attempt_policy,
                    reason="evaluator_budget_exhausted",
                )
                loop["decision"] = decision.model_dump(mode="json")
                loop["decisions"].append(loop["decision"])
                break

            feedback_turn = current.failure_turn_index
            if current.status == "completed":
                calls_before_review = path.accounting.snapshot()["evaluator"]["logical_calls"]
                try:
                    feedback_turn, tasks = self.completed_task_builder(
                        scenario=scenario, result=current, attempt_dir=current_dir, attempt_id=current_id)
                    if feedback_turn not in {turn.turn_index for turn in scenario.turns}:
                        raise ValueError("quality feedback targets a nonexistent turn")
                    validate_completed_review(tasks, current.final_output_path)
                    weak_results, evaluation_results, calls_used, task_count = await self._evaluate_tasks(
                        path, tasks, evaluator_call_budget_remaining=remaining_calls)
                except Exception as exc:
                    loop.setdefault("quality_review_errors", []).append({"type": type(exc).__name__, "message": str(exc)[:2048]})
                    weak_results, evaluation_results, calls_used, task_count = [], [], 0, 0
                finally:
                    calls_used = path.accounting.snapshot()["evaluator"]["logical_calls"] - calls_before_review
            else:
                weak_results, evaluation_results, calls_used, task_count = await self._evaluate_failure(
                    path,
                    scenario=scenario,
                    attempt_id=current_id,
                    attempt_dir=current_dir,
                    result=current,
                    planner_capability_snapshot=planner_capability_snapshot,
                    evaluator_call_budget_remaining=remaining_calls,
                )
            evaluator_calls += calls_used
            loop["evaluation_results"].extend(
                item.model_dump(mode="json") for item in evaluation_results
            )
            loop["evaluation_task_count"] += task_count
            loop["evaluator_logical_call_count"] = evaluator_calls

            packet: GuidancePacket | None = None
            if weak_results:
                loop["guider_logical_call_count"] += 1
                target_modules = _guidance_targets(weak_results)
                try:
                    packet = await asyncio.to_thread(
                        path.guide,
                        backend_id=self.policy.guider_backend_id,
                        results=tuple(weak_results),
                        target_modules=target_modules,
                    )
                except Exception as exc:  # noqa: BLE001 - guidance cannot alter an attempt.
                    loop.setdefault("guidance_errors", []).append(
                        {
                            "attempt_index": current_index,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:2048],
                        }
                    )
                else:
                    loop["guidance_packet"] = packet.model_dump(mode="json")
                    loop["guidance_packets"].append(loop["guidance_packet"])
                    loop["guidance_fingerprints"].append(packet.guidance_fingerprint)

            if task_count == 0:
                decision = _no_attempt_decision(
                    self.policy.attempt_policy,
                    reason="no_evaluable_module_output",
                )
            elif not evaluation_results:
                decision = _no_attempt_decision(
                    self.policy.attempt_policy,
                    reason="evaluation_produced_no_result",
                )
            else:
                decision = GuidedAttemptController().decide(
                    self.policy.attempt_policy,
                    (packet,) if packet is not None else (),
                    completed_new_attempts=current_index,
                    evaluator_calls_used=evaluator_calls,
                    failure_fingerprint=failure_fingerprint,
                    prior_failure_fingerprints=tuple(failure_history),
                    guidance_fingerprint=(
                        packet.guidance_fingerprint if packet is not None else None
                    ),
                    prior_guidance_fingerprints=tuple(guidance_history),
                    modifiable_owner_available=bool(
                        packet
                        and set(packet.target_modules).intersection(
                            {"reasoner", "planner", "perception"}
                        )
                    ),
                    new_evidence_or_issue_available=bool(
                        packet
                        and packet.failed_or_weak_rubric_ids
                        and packet.questions_to_reconsider
                    ),
                )
            loop["decision"] = decision.model_dump(mode="json")
            loop["decisions"].append(loop["decision"])
            loop["provider_call_accounting"] = path.accounting.snapshot()
            loop["provider_invocations"] = path.accounting.invocations
            _write_json(guidance_root / "loop_result.json", loop)
            if not decision.new_attempt_authorized or packet is None:
                break

            failure_turn = feedback_turn
            if failure_turn is None:
                loop["decision_override"] = "missing_failure_turn_index"
                break
            failure_history.append(failure_fingerprint)
            guidance_history.append(packet.guidance_fingerprint)
            future_guidance = future_attempt_guidance_from_packet(
                packet,
                source_attempt_id=current_id,
                failure_stage="artifact_quality" if current.status == "completed" else current.failure_stage,
                failure_codes=("measured_output_quality_defect",) if current.status == "completed" else tuple(current.failure_codes),
            )
            next_index = current_index + 1
            next_id = _attempt_id(scenario, arm=arm, attempt_index=next_index)
            next_dir = attempts_root / f"{next_index:04d}"
            next_runtime = self.base_runtime_factory(next_dir)
            current = await next_runtime.arun_scenario(
                scenario,
                arm=arm,
                output_dir=next_dir,
                future_attempt_guidance_by_turn={failure_turn: future_guidance},
            )
            _write_json(next_dir / "scenario_result.json", current.model_dump(mode="json"))
            loop["attempts"].append(
                _attempt_summary(next_id, next_index, current, guidance=future_guidance)
            )
            current_index = next_index
            current_id = next_id
            current_dir = next_dir
            loop["terminal_status"] = current.status
            loop["terminal_attempt_index"] = current_index
            loop["terminal_delivery"] = current.status == "completed"
            loop["recovered_by_feedback_edge"] = (
                initial.status != "completed" and current.status == "completed"
            )
            _write_json(guidance_root / "loop_result.json", loop)

        loop["terminal_status"] = current.status
        loop["terminal_attempt_index"] = current_index
        loop["terminal_delivery"] = current.status == "completed"
        loop["recovered_by_feedback_edge"] = (
            initial.status != "completed" and current.status == "completed"
        )
        loop["provider_call_accounting"] = path.accounting.snapshot()
        loop["provider_invocations"] = path.accounting.invocations
        _write_json(guidance_root / "loop_result.json", loop)
        return current

    async def _evaluate_failure(
        self,
        path: EvaluationGuidancePath,
        *,
        scenario: WorkflowRuntimeScenario,
        attempt_id: str,
        attempt_dir: Path,
        result: WorkflowScenarioResult,
        planner_capability_snapshot: dict[str, Any] | None,
        evaluator_call_budget_remaining: int,
    ) -> tuple[list[EvaluationResultV2], list[EvaluationResultV2], int, int]:
        tasks = _build_failure_tasks(
            self.policy,
            scenario=scenario,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            result=result,
            planner_capability_snapshot=planner_capability_snapshot,
        )
        return await self._evaluate_tasks(path, tasks, evaluator_call_budget_remaining=evaluator_call_budget_remaining)

    async def _evaluate_tasks(self, path: EvaluationGuidancePath, tasks: list[EvaluationTask],
                              *, evaluator_call_budget_remaining: int
                              ) -> tuple[list[EvaluationResultV2], list[EvaluationResultV2], int, int]:
        all_results: list[EvaluationResultV2] = []
        evaluator_calls = 0
        for task in tasks[: self.policy.max_semantic_focus_evaluations]:
            required_calls = len(task.judge_policy.slot_ids)
            if evaluator_calls + required_calls > evaluator_call_budget_remaining:
                break
            before = path.accounting.snapshot()["evaluator"]["logical_calls"]
            outcome = await asyncio.to_thread(path.evaluate, task)
            evaluator_calls += path.accounting.snapshot()["evaluator"]["logical_calls"] - before
            if outcome.result is None:
                continue
            all_results.append(outcome.result)
            if _weak_result(outcome.result, policy=self.policy):
                # Upstream semantic weakness is handled before judging downstream
                # graph construction.  A high-quality Reasoner result permits the
                # next task (normally PlanIR) to be judged independently.
                return [outcome.result], all_results, evaluator_calls, len(tasks)
        weak = [item for item in all_results if _weak_result(item, policy=self.policy)]
        return weak, all_results, evaluator_calls, len(tasks)


def future_attempt_guidance_from_packet(
    packet: GuidancePacket,
    *,
    source_attempt_id: str,
    failure_stage: str,
    failure_codes: tuple[str, ...],
) -> FutureAttemptGuidance:
    """Project Guider output into the narrower, trusted runtime contract."""

    targets = tuple(
        module
        for module in packet.target_modules
        if module in {"reasoner", "planner", "perception"}
    )
    if "perception" in targets and "planner" not in targets:
        # Perception routes are authored from Planner observation needs in v1.
        targets = (*targets, "planner")
    targets = tuple(dict.fromkeys(targets))
    if not targets:
        raise ValueError("guidance does not target a module admitted by the future attempt")
    return FutureAttemptGuidance(
        guidance_id=packet.guidance_id,
        source_attempt_id=source_attempt_id,
        source_evaluation_result_ids=packet.source_result_ids,
        target_modules=targets,
        weak_rubric_ids=packet.failed_or_weak_rubric_ids,
        diagnostic_evidence_refs=packet.evidence_refs,
        questions_to_reconsider=packet.questions_to_reconsider,
        constraints_to_preserve=packet.constraints_to_preserve,
        source_failure_stage=failure_stage,
        source_failure_codes=_stable_failure_code_ids(failure_codes),
    )


def _stable_failure_code_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    """Keep diagnostic categories in prompts while full details stay in evidence.

    Some legacy runtime surfaces append a human explanation after ``code:``.
    That explanation is not a stable identifier and can exceed the bounded
    future-guidance contract.  The immutable source result remains referenced by
    ID, so the new attempt only needs the category token here.
    """

    result: list[str] = []
    for value in values:
        raw = str(value).strip()
        category = raw.split(":", 1)[0].strip()
        if (
            category
            and len(category) <= 128
            and all(character.isalnum() or character in "._-" for character in category)
        ):
            code_id = category
        else:
            digest = canonical_evaluation_fingerprint({"failure_code": raw})[:24]
            code_id = f"failure_code_{digest}"
        if code_id not in result:
            result.append(code_id)
    return tuple(result)


def summarize_evaluation_guidance_run(run_root: str | Path) -> dict[str, Any]:
    """Aggregate loop sidecars without changing benchmark-run evidence."""

    root = Path(run_root).expanduser().resolve()
    paths = sorted(root.glob("samples/*/*/evaluation_guidance/loop_result.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    initial_deliveries = sum(bool(item.get("initial_delivery")) for item in rows)
    terminal_deliveries = sum(bool(item.get("terminal_delivery")) for item in rows)
    initial_failures = len(rows) - initial_deliveries
    recovered = sum(bool(item.get("recovered_by_feedback_edge")) for item in rows)
    feedback_attempts = sum(int(item.get("terminal_attempt_index") or 0) for item in rows)
    evaluated_failures = sum(
        not bool(item.get("initial_delivery"))
        and int(item.get("evaluation_task_count") or 0) > 0
        for item in rows
    )
    decision_reason_counts: dict[str, int] = {}
    for item in rows:
        decision = item.get("decision")
        reasons = decision.get("reason_codes") if isinstance(decision, dict) else ()
        for reason in reasons or ():
            key = str(reason)
            decision_reason_counts[key] = decision_reason_counts.get(key, 0) + 1
    initial_failure_stage_counts: dict[str, int] = {}
    for item in rows:
        if bool(item.get("initial_delivery")):
            continue
        attempts = item.get("attempts")
        first = attempts[0] if isinstance(attempts, list) and attempts else {}
        stage = str(first.get("failure_stage") or "unknown")
        initial_failure_stage_counts[stage] = initial_failure_stage_counts.get(stage, 0) + 1
    summary = {
        "schema_version": "x-editor.evaluation-guided-run-summary/v1",
        "assignment_count": len(rows),
        "initial_forward_deliveries": initial_deliveries,
        "initial_forward_delivery_rate": (
            initial_deliveries / len(rows) if rows else None
        ),
        "initial_failures": initial_failures,
        "evaluated_initial_failures": evaluated_failures,
        "initial_failure_stage_counts": dict(sorted(initial_failure_stage_counts.items())),
        "decision_reason_counts": dict(sorted(decision_reason_counts.items())),
        "no_evaluable_module_output_failures": decision_reason_counts.get(
            "no_evaluable_module_output",
            0,
        ),
        "feedback_attempts_opened": feedback_attempts,
        "feedback_recoveries": recovered,
        "feedback_recovery_rate_over_initial_failures": (
            recovered / initial_failures if initial_failures else None
        ),
        "feedback_augmented_deliveries": terminal_deliveries,
        "feedback_augmented_delivery_rate": (
            terminal_deliveries / len(rows) if rows else None
        ),
        "feedback_recovery_rate_over_opened_attempts": (
            recovered / feedback_attempts if feedback_attempts else None
        ),
        "evaluator_logical_call_count": sum(
            int(item.get("evaluator_logical_call_count") or 0) for item in rows
        ),
        "guider_logical_call_count": sum(
            int(item.get("guider_logical_call_count") or 0) for item in rows
        ),
        "provider_call_accounting": summarize_call_accounting(
            [item.get("provider_call_accounting") for item in rows]
        ),
        "normal_forward_flow_and_feedback_recovery_reported_separately": True,
        "capacity_claim_eligible": False,
        "development_gate_only": True,
        "scorer_feedback_consumed": False,
        "hidden_best_attempt_selection": False,
        "loop_result_paths": [str(path.relative_to(root)) for path in paths],
    }
    _write_json(root / "evaluation_guided_summary.json", summary)
    return summary


def _build_failure_tasks(
    policy: EvaluationGuidedScenarioPolicy,
    *,
    scenario: WorkflowRuntimeScenario,
    attempt_id: str,
    attempt_dir: Path,
    result: WorkflowScenarioResult,
    planner_capability_snapshot: dict[str, Any] | None = None,
) -> list[SemanticPathEvaluationTask]:
    turn_index = result.failure_turn_index
    if turn_index is None or turn_index >= len(result.turn_results):
        return []
    turn_result = result.turn_results[turn_index].result
    public_turn = scenario.turns[turn_index]
    checkpoints = _load_turn_checkpoints(attempt_dir, turn_index=turn_index)
    context = _judge_context_evidence(
        scenario=scenario,
        turn_index=turn_index,
        result=result,
    )
    tasks: list[SemanticPathEvaluationTask] = []

    reasoner_checkpoint = _latest_checkpoint(checkpoints, module="reasoner")
    reasoner_subject = _reasoner_subject(turn_result, reasoner_checkpoint)
    if reasoner_subject is not None:
        tasks.append(
            _semantic_task(
                policy,
                attempt_id=attempt_id,
                focus="reasoner_semantics",
                public_instruction=public_turn.user_request,
                subject=reasoner_subject,
                context=context,
                checkpoint=reasoner_checkpoint,
                failure_ref=f"{result.failure_stage}:{turn_index}",
                candidate_operation_cards=(),
            )
        )

    planner_checkpoint = _latest_planner_checkpoint(checkpoints)
    planner_subject = _planner_subject(turn_result, planner_checkpoint)
    if planner_subject is not None:
        planner_focus, planner_evidence = planner_subject
        planner_phase = _planner_checkpoint_phase(
            planner_checkpoint,
            focus=planner_focus,
        )
        planner_capability_cards = _planner_candidate_capability_evidence(
            planner_capability_snapshot,
            phase=planner_phase,
        )
        tasks.append(
            _semantic_task(
                policy,
                attempt_id=attempt_id,
                focus=planner_focus,
                public_instruction=public_turn.user_request,
                subject=planner_evidence,
                context=(
                    *context,
                    *((reasoner_subject,) if reasoner_subject is not None else ()),
                ),
                checkpoint=planner_checkpoint,
                failure_ref=f"{result.failure_stage}:{turn_index}",
                candidate_operation_cards=planner_capability_cards,
            )
        )
    return tasks


def _semantic_task(
    policy: EvaluationGuidedScenarioPolicy,
    *,
    attempt_id: str,
    focus: Literal["reasoner_semantics", "planner_decision", "plan_graph"],
    public_instruction: str,
    subject: EvaluationEvidenceItem,
    context: tuple[EvaluationEvidenceItem, ...],
    checkpoint: ModuleCheckpoint | None,
    failure_ref: str,
    candidate_operation_cards: tuple[EvaluationEvidenceItem, ...] = (),
) -> SemanticPathEvaluationTask:
    seed = {
        "attempt_id": attempt_id,
        "focus": focus,
        "subject": subject.content_sha256,
        "failure_ref": failure_ref,
        "candidate_operation_cards": [
            item.content_sha256 for item in candidate_operation_cards
        ],
    }
    task_id = "guided_eval_" + canonical_evaluation_fingerprint(seed)[:24]
    selection_policy = EvaluationSamplingPolicy(
        policy_id=f"{policy.policy_id}:failure_forensics",
        mode="failure_forensics",
    )
    selection = build_forensic_selection(
        selection_policy,
        unit_id=task_id,
        focus=focus,
        failure_ref=failure_ref,
    )
    checkpoint_id = checkpoint.checkpoint_id if checkpoint is not None else task_id
    return SemanticPathEvaluationTask(
        task_id=task_id,
        run_id=attempt_id,
        checkpoint_id=checkpoint_id,
        input_snapshot_ref=f"input:{checkpoint_id}",
        focus=focus,
        evaluator_role="runtime_feedback_critic",
        evaluation_use="guided_attempt_diagnostic",
        backend_id=policy.evaluator_backend_id,
        public_instruction=public_instruction,
        subject_output=subject,
        context_evidence=context,
        candidate_operation_cards=candidate_operation_cards,
        rubrics=_semantic_rubrics(
            focus,
            candidate_operation_cards_present=bool(candidate_operation_cards),
        ),
        judge_policy=policy.judge_policy,
        selection=selection,
    )


def _semantic_rubrics(
    focus: Literal["reasoner_semantics", "planner_decision", "plan_graph"],
    *,
    candidate_operation_cards_present: bool,
) -> tuple[AtomicEvaluationRubric, ...]:
    rubrics = build_default_semantic_rubrics(focus)
    if candidate_operation_cards_present or focus == "reasoner_semantics":
        return rubrics

    # A generic wrapped runtime may not expose its Planner-visible capability
    # snapshot.  Keep semantic/structural review available, but do not ask the
    # model to infer operation support or parameter legality from the terminal
    # failure text itself.
    capability_dependent_ids = {
        "planner_decision.capability_boundary",
        "plan_graph.parameter_legality",
    }
    return tuple(
        rubric for rubric in rubrics if rubric.rubric_id not in capability_dependent_ids
    )


def _planner_capability_snapshot(
    runtime: GuidanceCapableScenarioRuntime,
) -> dict[str, Any] | None:
    snapshot_builder = getattr(runtime, "planner_visible_capability_snapshot", None)
    if not callable(snapshot_builder):
        return None
    snapshot = snapshot_builder()
    if not isinstance(snapshot, dict):
        raise TypeError("planner capability snapshot must be a mapping")
    return snapshot


def _planner_candidate_capability_evidence(
    snapshot: dict[str, Any] | None,
    *,
    phase: Literal["evidence_planning", "execution_planning"],
) -> tuple[EvaluationEvidenceItem, ...]:
    if snapshot is None:
        return ()

    profile_name = snapshot.get("runtime_capability_profile")
    profile_fingerprint = snapshot.get("runtime_capability_profile_fingerprint")
    evidence: list[EvaluationEvidenceItem] = []
    seen_operations: set[tuple[str, str]] = set()
    by_phase = snapshot.get("edit_operation_contracts_by_phase")
    if by_phase is not None and not isinstance(by_phase, dict):
        raise TypeError("edit_operation_contracts_by_phase must be a mapping")
    edit_contracts = by_phase.get(phase, ()) if isinstance(by_phase, dict) else ()
    for capability_kind, raw_contracts in (
        ("edit", edit_contracts),
        ("perception", snapshot.get("perception_operation_contracts") or ()),
    ):
        if not isinstance(raw_contracts, (list, tuple)):
            raise TypeError("planner capability snapshot contract groups must be lists")
        for raw in raw_contracts:
            if not isinstance(raw, dict):
                raise TypeError("planner capability contracts must be mappings")
            operation = str(raw.get("operation") or "").strip()
            if not operation:
                raise ValueError("planner capability contract is missing operation")
            key = (capability_kind, operation)
            if key in seen_operations:
                raise ValueError(f"duplicate Planner capability contract: {key}")
            seen_operations.add(key)
            evidence.append(
                EvaluationEvidenceItem(
                    evidence_id=f"candidate:{capability_kind}:{operation}",
                    kind="candidate_operation_card",
                    label=(
                        "Frozen Planner-visible edit operation contract"
                        if capability_kind == "edit"
                        else "Frozen Planner-visible Perception operation contract"
                    ),
                    payload={
                        "capability_kind": capability_kind,
                        "runtime_capability_profile": profile_name,
                        "runtime_capability_profile_fingerprint": profile_fingerprint,
                        "planner_phase": phase,
                        "contract": raw,
                    },
                )
            )
    return tuple(evidence)


def _planner_checkpoint_phase(
    checkpoint: ModuleCheckpoint | None,
    *,
    focus: Literal["planner_decision", "plan_graph"],
) -> Literal["evidence_planning", "execution_planning"]:
    if checkpoint is not None:
        record = checkpoint.output.get("record")
        trace = record.get("invocation_trace") if isinstance(record, dict) else None
        metadata = trace.get("metadata") if isinstance(trace, dict) else None
        budget = metadata.get("prompt_budget") if isinstance(metadata, dict) else None
        phase = budget.get("phase") if isinstance(budget, dict) else None
        if phase in {"evidence_planning", "execution_planning"}:
            return phase
    return "execution_planning" if focus == "plan_graph" else "evidence_planning"


def _judge_context_evidence(
    *,
    scenario: WorkflowRuntimeScenario,
    turn_index: int,
    result: WorkflowScenarioResult,
) -> tuple[EvaluationEvidenceItem, ...]:
    turn = scenario.turns[turn_index]
    runtime_result = result.turn_results[turn_index].result
    evidence: list[EvaluationEvidenceItem] = []
    visible_scenario = scenario.model_visible_projection()
    visible_turns = visible_scenario.get("turns")
    visible_turn = (
        visible_turns[turn_index]
        if isinstance(visible_turns, list) and turn_index < len(visible_turns)
        else None
    )
    evidence.append(
        EvaluationEvidenceItem(
            evidence_id=f"runtime-input:{turn_index}",
            kind="other",
            label="Benchmark-neutral runtime-visible audio and turn input",
            payload={
                "audio_assets": visible_scenario.get("audio_assets"),
                "turn": visible_turn,
            },
        )
    )
    if turn.source_text:
        evidence.append(
            EvaluationEvidenceItem(
                evidence_id=f"source-text:{turn_index}",
                kind="public_instruction",
                label="Caller-visible source text",
                payload={"source_text": turn.source_text},
            )
        )
    if runtime_result.evidence_projection is not None:
        evidence.append(
            EvaluationEvidenceItem(
                evidence_id=f"reviewed-evidence:{turn_index}",
                kind="reviewed_perception",
                label="Reviewed Perception projection",
                payload=runtime_result.evidence_projection.planner_context,
            )
        )
    if result.final_state_projection is not None:
        evidence.append(
            EvaluationEvidenceItem(
                evidence_id=f"session-state:{turn_index}",
                kind="session_state",
                label="Authoritative final Session Graph projection",
                payload=result.final_state_projection.model_dump(mode="json"),
            )
        )
    return tuple(evidence)


def _reasoner_subject(
    runtime_result: Any,
    checkpoint: ModuleCheckpoint | None,
) -> EvaluationEvidenceItem | None:
    if checkpoint is not None:
        record = checkpoint.output.get("record")
        if isinstance(record, dict) and isinstance(record.get("memo"), dict):
            return EvaluationEvidenceItem(
                evidence_id=f"reasoner:{checkpoint.checkpoint_id}",
                kind="reasoner_memo",
                label="Immutable Reasoner memo checkpoint",
                payload=record["memo"],
            )
        # A terminal failed checkpoint without a visible memo is a transport or
        # generation failure, not a semantic output that an Evaluator can judge.
        return None
    if runtime_result.reasoner_history:
        record = runtime_result.reasoner_history[-1]
        if record.status == "completed" and isinstance(record.memo, dict):
            return EvaluationEvidenceItem(
                evidence_id=f"reasoner:{record.request_fingerprint}",
                kind="reasoner_memo",
                label="Reasoner memo record",
                payload=record.memo,
            )
    return None


def _planner_subject(
    runtime_result: Any,
    checkpoint: ModuleCheckpoint | None,
) -> tuple[
    Literal["planner_decision", "plan_graph"],
    EvaluationEvidenceItem,
] | None:
    if checkpoint is not None:
        record = checkpoint.output.get("record")
        plan = (
            record.get("plan_snapshot")
            if isinstance(record, dict)
            else checkpoint.output.get("plan")
        )
        if isinstance(plan, dict):
            return (
                "plan_graph",
                EvaluationEvidenceItem(
                    evidence_id=f"planner:{checkpoint.checkpoint_id}",
                    kind="plan_ir",
                    label="Immutable executable PlanIR checkpoint",
                    payload=plan,
                ),
            )
        if isinstance(record, dict):
            decision = record.get("decision")
            if record.get("status") == "completed" and isinstance(decision, str):
                return (
                    "planner_decision",
                    EvaluationEvidenceItem(
                        evidence_id=f"planner:{checkpoint.checkpoint_id}",
                        kind="planner_decision",
                        label="Immutable Planner decision or rejection checkpoint",
                        payload=checkpoint.output,
                    ),
                )
        rejected_output = _find_nested_value(
            checkpoint.output,
            key="provider_output_on_failure",
        )
        if rejected_output is not None:
            decoded = _decode_rejected_provider_output(rejected_output)
            focus: Literal["planner_decision", "plan_graph"] = (
                "plan_graph" if _looks_like_plan_ir(decoded) else "planner_decision"
            )
            return (
                focus,
                EvaluationEvidenceItem(
                    evidence_id=f"planner-rejected:{checkpoint.checkpoint_id}",
                    kind=("plan_ir" if focus == "plan_graph" else "planner_decision"),
                    label="Rejected, unvalidated Planner provider output",
                    payload={
                        "validation_state": "rejected_unvalidated",
                        "provider_output": decoded,
                    },
                ),
            )
        # A null failure placeholder is not a semantic Planner output.
        return None
    if runtime_result.planner_history:
        record = runtime_result.planner_history[-1]
        if record.plan_snapshot is not None:
            return (
                "plan_graph",
                EvaluationEvidenceItem(
                    evidence_id=f"planner:{record.input_fingerprint}",
                    kind="plan_ir",
                    label="Executable PlanIR record",
                    payload=record.plan_snapshot.model_dump(mode="json"),
                ),
            )
        if record.status == "completed" and record.decision is not None:
            return (
                "planner_decision",
                EvaluationEvidenceItem(
                    evidence_id=f"planner:{record.input_fingerprint}",
                    kind="planner_decision",
                    label="Planner terminal decision record",
                    payload=record.model_dump(mode="json"),
                ),
            )
    return None


def _find_nested_value(value: Any, *, key: str) -> Any | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_nested_value(item, key=key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_nested_value(item, key=key)
            if found is not None:
                return found
    return None


def _decode_rejected_provider_output(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"unparsed_text": value}


def _looks_like_plan_ir(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("steps"), list):
        return True
    plan = value.get("plan")
    return isinstance(plan, dict) and isinstance(plan.get("steps"), list)


def _load_turn_checkpoints(attempt_dir: Path, *, turn_index: int) -> list[ModuleCheckpoint]:
    path = attempt_dir / "turns" / f"{turn_index:04d}" / "execution" / "module_checkpoints.jsonl"
    if not path.is_file():
        return []
    checkpoints: list[ModuleCheckpoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            checkpoints.append(ModuleCheckpoint.model_validate_json(line))
    return checkpoints


def _latest_checkpoint(
    checkpoints: list[ModuleCheckpoint],
    *,
    module: str,
) -> ModuleCheckpoint | None:
    return next((item for item in reversed(checkpoints) if item.module == module), None)


def _latest_planner_checkpoint(checkpoints: list[ModuleCheckpoint]) -> ModuleCheckpoint | None:
    return next(
        (
            item
            for item in reversed(checkpoints)
            if item.module in {"planner", "planner_reasoner"}
        ),
        None,
    )


def _weak_result(
    result: EvaluationResultV2,
    *,
    policy: EvaluationGuidedScenarioPolicy,
) -> bool:
    if result.known_weight_fraction < policy.minimum_known_weight_fraction:
        return False
    actionable_ids = {
        rubric.rubric_id
        for rubric in result.rubric_results
        if rubric.status == "graded" and rubric.guidance_actionable
    }
    if not actionable_ids:
        return False

    # A new attempt is justified only when the evaluation identifies a defect
    # that a normal forward rerun can change.  A truthful ``cannot_plan`` may
    # still use an imprecise explanation; retrying merely to improve that prose
    # inflates feedback traffic without increasing executable flow.
    if result.focus == "planner_decision":
        diagnostic_only = {
            "planner_decision.blocking_reason",
            "planner_decision.capability_boundary",
        }
        return bool(actionable_ids - diagnostic_only)
    return True


def _guidance_targets(
    results: list[EvaluationResultV2],
) -> tuple[Literal["reasoner", "planner"], ...]:
    targets: list[Literal["reasoner", "planner"]] = []
    for result in results:
        target: Literal["reasoner", "planner"] = (
            "reasoner" if result.focus == "reasoner_semantics" else "planner"
        )
        if target not in targets:
            targets.append(target)
    return tuple(targets)


def _controller_failure_fingerprint(
    result: WorkflowScenarioResult,
    *,
    attempt_dir: Path,
) -> str:
    """Fingerprint a failure boundary without benchmark or sample identity."""

    turn_index = result.failure_turn_index
    checkpoints = (
        _load_turn_checkpoints(attempt_dir, turn_index=turn_index)
        if turn_index is not None
        else []
    )
    terminal = checkpoints[-1] if checkpoints else None
    payload = {
        "module": result.failure_stage,
        "stable_codes": _stable_failure_code_ids(tuple(result.failure_codes)),
        "terminal_error_type": result.terminal_error_type,
        "checkpoint": (
            {
                "module": terminal.module,
                "boundary_kind": terminal.boundary_kind,
                "evaluation_role": terminal.evaluation_role,
                "status": terminal.status,
                "diagnostics": _stable_checkpoint_diagnostics(terminal.output),
            }
            if terminal is not None
            else None
        ),
    }
    return canonical_evaluation_fingerprint(payload)


def _stable_checkpoint_diagnostics(value: Any) -> tuple[str, ...]:
    tokens: list[str] = []
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key).strip().casefold()
            if normalized_key.endswith(("code", "codes", "type", "kind")):
                values = item if isinstance(item, (list, tuple)) else (item,)
                for candidate in values:
                    token = " ".join(str(candidate).casefold().split())[:256]
                    if token and token not in tokens:
                        tokens.append(token)
            elif isinstance(item, (dict, list, tuple)):
                for token in _stable_checkpoint_diagnostics(item):
                    if token not in tokens:
                        tokens.append(token)
    elif isinstance(value, (list, tuple)):
        for item in value:
            for token in _stable_checkpoint_diagnostics(item):
                if token not in tokens:
                    tokens.append(token)
    return tuple(tokens)


def _attempt_id(
    scenario: WorkflowRuntimeScenario,
    *,
    arm: WorkflowArm,
    attempt_index: int,
) -> str:
    return f"{scenario.fingerprint}:{arm}:attempt:{attempt_index}"


def _attempt_summary(
    attempt_id: str,
    attempt_index: int,
    result: WorkflowScenarioResult,
    *,
    guidance: FutureAttemptGuidance | None,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "status": result.status,
        "failure_turn_index": result.failure_turn_index,
        "failure_stage": result.failure_stage,
        "failure_codes": list(result.failure_codes),
        "final_output_path": result.final_output_path,
        "guidance_fingerprint": (
            guidance.guidance_fingerprint if guidance is not None else None
        ),
    }


def _no_attempt_decision(
    policy: GuidedAttemptPolicy,
    *,
    reason: str,
) -> GuidedAttemptDecision:
    return GuidedAttemptDecision(
        decision_id="guided_decision_"
        + canonical_evaluation_fingerprint(
            {"policy": policy.model_dump(mode="json"), "reason": reason}
        )[:24],
        policy_id=policy.policy_id,
        action="no_new_attempt",
        reason_codes=(reason,),
        new_attempt_authorized=False,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EvaluationGuidedScenarioPolicy",
    "EvaluationGuidedScenarioRuntime",
    "future_attempt_guidance_from_packet",
    "summarize_evaluation_guidance_run",
]
