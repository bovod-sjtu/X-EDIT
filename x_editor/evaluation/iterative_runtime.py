"""Explicit multi-try controller around the one ordinary scenario runtime.

Unlike the historical last-attempt protocol, this opt-in protocol retains all
attempts and records delivery-first selection. The critic remains advisory;
the ordinary runtime still owns plan admission, verification and each delivery.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.evaluation.assessment_cache import validate_cached_assessment
from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint
from x_editor.evaluation.refinement_flow_selection import observed_constraints, retention_constraints
from x_editor.evaluation.refinement_contracts import (
    PublicUtilityDesign,
    RefinementPolicy,
    UtilityAssessment,
    utility_bounds,
)
from x_editor.runtime.workflow_session import WorkflowScenarioResult


def write_record(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delivered_artifact(result, scenario):
    """Delivery eligibility, deliberately not a semantic quality certificate."""
    if result.scenario_fingerprint != scenario.fingerprint or result.assigned_arm != "W1":
        raise ValueError("attempt belongs to a different public scenario or workflow")
    if result.status != "completed" or not result.final_output_path:
        return None
    if [t.turn_index for t in result.turn_results] != list(range(len(scenario.turns))):
        return None
    if any(t.result.status != "completed" for t in result.turn_results):
        return None
    import soundfile as sf

    try:
        info = sf.info(result.final_output_path)
        if info.frames <= 0 or info.samplerate <= 0:
            return None
        return {
            "path": result.final_output_path,
            "sha256": sha256(result.final_output_path),
            "frames": info.frames,
            "samplerate": info.samplerate,
        }
    except (OSError, RuntimeError):
        return None


@dataclass
class IterationReview:
    guidance_by_turn: dict[int, FutureAttemptGuidance] = field(default_factory=dict)
    utility: UtilityAssessment | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    stop_reason: str | None = None


def select_attempt(attempts):
    """Maximize recorded conservative utility, with known preservation guards.

    Missing judgments cannot improve utility. An exact tie keeps the earliest
    artifact. Module grades and call counts never enter the selection value.
    """
    chosen = None
    decisions = []
    for attempt in attempts:
        if not attempt["delivered"]:
            decisions.append(
                {"attempt_index": attempt["attempt_index"], "decision": "not_delivered"}
            )
            continue
        if chosen is None:
            chosen = attempt
            decisions.append(
                {"attempt_index": attempt["attempt_index"], "decision": "first_delivery"}
            )
            continue
        old, new = chosen["utility"], attempt["utility"]
        losses = [
            key
            for key, value in retention_constraints(old).items()
            if value is not None
            and (observed_constraints(new).get(key) is None or observed_constraints(new)[key] < value)
        ]
        improves = new["lower"] > old["lower"]
        decision = "higher_supported_utility" if improves and not losses else "retain_incumbent"
        decisions.append(
            {
                "attempt_index": attempt["attempt_index"],
                "decision": decision,
                "preservation_losses_or_unknowns": losses,
            }
        )
        if improves and not losses:
            chosen = attempt
    return chosen, decisions


class IterativeScenarioRuntime:
    """Injectable evaluator permits identical actor, budgets and selector controls."""

    def __init__(self, *, factory: Callable, reviewer: Callable, policy: RefinementPolicy,
                 component_reuse=None, selection_reviewer=None, utility_design_provider=None,
                 defer_design_until_selection=False):
        self.factory, self.reviewer, self.policy = factory, reviewer, policy
        self.component_reuse = component_reuse
        self.selection_reviewer = selection_reviewer
        self.utility_design_provider = utility_design_provider
        self.defer_design_until_selection = defer_design_until_selection
        if policy.execution_reuse != "disabled" and component_reuse is not None:
            raise ValueError("Execution reuse must not also change module inference reuse")
        if policy.nondelivery_review == "raw_errors":
            if not getattr(reviewer, "decision_first", False):
                raise ValueError("delivery-conditioned recovery requires a decision-first reviewer")
            if component_reuse is not None:
                raise ValueError("delivery-conditioned recovery reruns the full Actor without component reuse")
        if policy.selection_assessment_timing == "after_generation" and selection_reviewer is None:
            raise ValueError("deferred selection requires its separate reviewer")
        if policy.selection == 'mc_conservative_utility' and not callable(getattr(selection_reviewer, 'review_mc', None)):
            raise ValueError('MC selection requires a reviewer with frozen-evidence sampling support')
        if policy.selection_assessment_reuse == 'exact_online':
            source = getattr(reviewer, 'selection_proxy_configuration', None)
            target = getattr(selection_reviewer, 'selection_proxy_configuration', None)
            if (not callable(source) or not callable(target) or source() != target()
                    or not getattr(selection_reviewer, 'decision_first', False)):
                raise ValueError('selection reuse needs identical declared proxies and a current decision gate')

    async def arun_scenario(
        self,
        scenario,
        *,
        output_dir,
        utility_design: PublicUtilityDesign | None,
        initial_result=None,
        initial_attempt_dir=None,
        resume_checkpoint=None,
    ):
        if (initial_result is None) != (initial_attempt_dir is None):
            raise ValueError("shared initial attempt requires its immutable evidence directory")
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        async def resolve_design(label):
            nonlocal utility_design
            if utility_design is not None or self.utility_design_provider is None:
                return
            try:
                # No candidate, trajectory or selected condition reaches authoring.
                utility_design = await asyncio.to_thread(self.utility_design_provider)
                receipt = {'design': utility_design.model_dump(mode='json') if utility_design else None,
                           'candidate_visible_to_author': False}
            except Exception as exc:
                receipt = {'error': f'{type(exc).__name__}: {exc}', 'quality_status': 'unknown'}
            write_record(root / 'design_resolution' / f'{label}.json', receipt)
            for reviewer in (self.reviewer, self.selection_reviewer):
                if reviewer is not None and hasattr(reviewer, 'utility_design'):
                    reviewer.utility_design = utility_design
        sources = {a.path: sha256(a.path) for a in scenario.audio_assets}
        if any(a.content_sha256 != sources[a.path] for a in scenario.audio_assets):
            raise ValueError("scenario source identity changed")
        write_record(
            root / "protocol.json",
            {
                "policy": self.policy.model_dump(mode="json"),
                "scenario_fingerprint": scenario.fingerprint,
                "source_audio_sha256": sources,
                "utility_design": utility_design.model_dump(mode="json") if utility_design else None,
                "utility_design_fingerprint": canonical_evaluation_fingerprint(utility_design) if utility_design else None,
                "shared_initial_attempt": str(initial_attempt_dir) if initial_attempt_dir else None,
                "initial_plus_retries": self.policy.max_retries + 1,
                "inner_runtime_modified": False,
                "execution_reuse": self.policy.execution_reuse,
                "equal_total_compute_claimed": False,
            },
        )
        attempts, history, guidance, outputs, audio_cache = [], [], {}, {}, {}
        immutable = dict(sources)
        start_index = 0
        resumed_review = None
        if resume_checkpoint is not None:
            from x_editor.revision.interrupted_recovery import load_prefix
            saved, outputs, guidance = load_prefix(resume_checkpoint, scenario, self.policy)
            if (initial_result is None or 0 not in outputs or
                    canonical_evaluation_fingerprint(outputs[0].model_dump(mode='json')) !=
                    canonical_evaluation_fingerprint(initial_result.model_dump(mode='json'))):
                raise ValueError('resumed prefix must retain the same initial execution')
            attempts, history, start_index = saved['attempts'], saved['history'], saved['next_attempt']
            immutable.update(saved['preserved_files'])
            immutable.update({r['artifact']['path']: r['artifact']['sha256'] for r in attempts if r.get('artifact')})
            write_record(root/'interrupted_prefix.json', saved)
            if 'resume_review_index' in saved:
                start_index = saved['resume_review_index']
                if start_index != len(attempts)-1:
                    raise ValueError('only the last spent attempt can resume its review')
                resumed_review = attempts[-1]
                attempts, history = attempts[:-1], history[:-1]
                from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
                guidance = {int(k):FutureAttemptGuidance.model_validate(v)
                            for k,v in resumed_review['guidance_consumed'].items()}
        for index in range(start_index, self.policy.max_retries + 1):
            destination = root / "attempts" / f"{index:04d}"
            attempt_id = canonical_evaluation_fingerprint([scenario.fingerprint, str(root), index])
            result, review, error = None, IterationReview(), None
            execution_resume = None
            actor_started = time.monotonic()
            try:
                if resumed_review is not None and index == start_index:
                    # An interrupted review is not another Actor opportunity.
                    # Even a started call without a result keeps its spent slot.
                    result = outputs.get(index)
                    directory = Path(resumed_review['result_path'] or destination)
                    error = resumed_review.get('error')
                    attempt_id = resumed_review['attempt_id']
                    write_record(root/'review_continuation.json', {
                        'attempt_index':index, 'new_actor_calls':0,
                        'original_attempt':resumed_review,
                        'resumed_result_available':result is not None})
                elif index == 0 and initial_result is not None:
                    result = initial_result
                    directory = Path(initial_attempt_dir)
                    # An imported result must exist verbatim in its own receipt.
                    receipts = [directory / "scenario_result.json", directory / "result.json"]
                    matching = []
                    for path in receipts:
                        if path.is_file():
                            raw = json.loads(path.read_text())
                            payload = raw.get("result", raw)
                            WorkflowScenarioResult.model_validate(payload)
                            # Transient/excluded observer fields do not survive
                            # serialization and are not part of this receipt.
                            if canonical_evaluation_fingerprint(
                                payload
                            ) == canonical_evaluation_fingerprint(result.model_dump(mode="json")):
                                matching.append(path)
                            elif path.name == 'result.json' and (directory/'completion.json').is_file():
                                from x_editor.evaluation.committed_attempt import project_committed_attempt
                                completion_path = directory/'completion.json'
                                completion = json.loads(completion_path.read_text())
                                if (completion.get('state') == 'complete'
                                        and completion.get('assigned_arm') == 'W1'
                                        and completion.get('result_sha256') == sha256(path)
                                        and Path(completion.get('result_path', '')).parts[-1:] == ('result.json',)):
                                    projected = project_committed_attempt(raw, completion)
                                    if canonical_evaluation_fingerprint(projected.model_dump(mode='json')) == canonical_evaluation_fingerprint(result.model_dump(mode='json')):
                                        matching.extend([path, completion_path])
                    if not matching:
                        raise ValueError("shared initial lacks its matching original receipt")
                    immutable.update({str(p): sha256(p) for p in matching})
                else:
                    directory = destination
                    if self.component_reuse is not None:
                        self.component_reuse.prepare(guidance)
                    if index > 0 and self.policy.execution_reuse == "failed_prefix":
                        from x_editor.evaluation.execution_resume import prepare_execution_resume
                        execution_resume = prepare_execution_resume(
                            outputs.get(index - 1), scenario_fingerprint=scenario.fingerprint,
                            previous_attempt=index, max_total_attempts=self.policy.max_retries + 1,
                            previous_directory=(attempts[-1].get('result_path') if attempts else None),
                            source_assets=scenario.audio_assets,
                        )
                        write_record(directory / "execution_resume_offer.json", execution_resume.receipt)
                    runtime = (self.factory(directory, execution_resume=execution_resume)
                               if execution_resume is not None else self.factory(directory))
                    result = await runtime.arun_scenario(
                        scenario,
                        arm="W1",
                        output_dir=directory,
                        future_attempt_guidance_by_turn=guidance or None,
                    )
                    write_record(directory / "scenario_result.json", result.model_dump(mode="json"))
                artifact = delivered_artifact(result, scenario) if result is not None else None
                if result is not None:
                    outputs[index] = result
                if artifact:
                    immutable[artifact["path"]] = artifact["sha256"]
            except Exception as exc:  # noqa: BLE001 -- preserve failed attempt without fallback delivery
                error = {"type": type(exc).__name__, "message": str(exc)[:2400]}
                artifact = None
            raw_nondelivery = artifact is None and self.policy.nondelivery_review == "raw_errors"
            if raw_nondelivery:
                # A failed execution already supplies an actionable observation.
                # Do not make an LLM diagnosis or a missing audio questionnaire a
                # prerequisite for forwarding the exact current runtime errors.
                actor_seconds, review_seconds = time.monotonic() - actor_started, 0.0
                write_record(root / "reviews" / f"{index:04d}" / "route.json", {
                    "route": "raw_errors_without_judges", "delivered": False,
                    "semantic_calls": 0, "audio_calls": 0,
                    "structured_result_available": result is not None,
                })
            elif result is not None and error is None:
                actor_seconds = time.monotonic() - actor_started
                review_started = time.monotonic()
                sequential_review = getattr(self.reviewer, 'decision_first', False)
                if artifact is not None and not self.defer_design_until_selection and not sequential_review:
                    await resolve_design(f'{index:04d}')
                try:
                    review = await self.reviewer(
                        scenario=scenario,
                        result=result,
                        directory=directory,
                        output=root / "reviews" / f"{index:04d}",
                        attempt_id=attempt_id,
                        history=history,
                        utility_design=utility_design,
                        artifact=artifact,
                        cached_utility=audio_cache.get(artifact["sha256"]) if artifact else None,
                        **({'resolve_utility_design': lambda: resolve_design(f'{index:04d}')}
                           if sequential_review else {}),
                        **({"build_feedback": self.policy.mode != "ordinary_retry"
                             and index < self.policy.max_retries}
                           if self.policy.skip_unused_feedback else {}),
                    )
                    if review.utility:
                        if not artifact or review.utility.artifact_sha256 != artifact["sha256"]:
                            raise ValueError("review utility belongs to a different artifact")
                        if review.utility.assessor_role != "online_selection_proxy":
                            raise ValueError("independent final outcomes cannot guide selection")
                        utility_bounds(utility_design, review.utility)
                        validate_cached_assessment(audio_cache.get(artifact['sha256']), review.utility)
                        audio_cache[artifact["sha256"]] = review.utility
                except Exception as exc:  # noqa: BLE001 -- preserve failed attempt without fallback delivery
                    review = IterationReview(errors=[f"{type(exc).__name__}: {exc}"])
                review_seconds = time.monotonic() - review_started
            else:
                actor_seconds, review_seconds = time.monotonic() - actor_started, 0.0
            if ((self.policy.retain_runtime_error_feedback or raw_nondelivery) and result is not None
                    and artifact is None and index < self.policy.max_retries):
                from x_editor.evaluation.runtime_error_feedback import retain_runtime_errors

                review.guidance_by_turn, error_receipt = retain_runtime_errors(
                    result, attempt_id=attempt_id, reviewed_guidance=review.guidance_by_turn,
                )
                error_receipt["review_stop_reason"] = review.stop_reason
                write_record(root / "runtime_error_feedback" / f"{index:04d}.json", error_receipt)
                if error_receipt["dispositions"] and review.guidance_by_turn:
                    # Reviewer uncertainty cannot erase a recorded execution
                    # failure. This opens a fresh attempt; admission is unchanged.
                    review.stop_reason = None
            if raw_nondelivery and not review.guidance_by_turn and index < self.policy.max_retries:
                from x_editor.evaluation.runtime_error_feedback import runtime_error_guidance

                # Some failures occur outside a turn, or claim completion without
                # producing a readable complete delivery. Preserve this observed
                # failure too; never manufacture a semantic/acoustic diagnosis.
                turn_index = result.turn_results[-1].turn_index if result and result.turn_results else 0
                facts = {
                    "turn_index": turn_index,
                    "failure_stage": "scenario_runtime" if error else "delivery_validation",
                    "failure_codes": ["scenario_runtime_exception" if error else "no_eligible_delivery"],
                    "terminal_error_message": error if error else {
                        "message": "The completed attempt has no eligible, readable final audio for all requested turns.",
                        "reported_output": result.final_output_path if result else None,
                    },
                    "failure_location_known": False,
                }
                review.guidance_by_turn[turn_index] = runtime_error_guidance(facts, attempt_id=attempt_id)
                review.stop_reason = None
                write_record(root / "runtime_error_feedback" / f"{index:04d}_delivery.json", facts)
            if (resumed_review is not None and index == start_index and result is None
                    and guidance and index < self.policy.max_retries):
                # Worker termination did not resolve the preceding diagnosis.
                # Preserve the actual packet used by the interrupted Actor;
                # do not replace it with a supposed new semantic/audio failure.
                review.guidance_by_turn = guidance
                review.stop_reason = None
                write_record(root/'interruption_guidance_retained.json', {
                    'spent_attempt_index':index, 'restored_consumed_guidance':True,
                    'new_failure_diagnosed':False, 'budget_refunded':False})
            bounds = utility_bounds(utility_design, review.utility)
            if artifact is None:
                bounds.update(lower=0.0, upper=0.0)
            row = {
                "attempt_index": index,
                "attempt_id": attempt_id,
                "status": result.status if result is not None else "failed",
                "error": error,
                "delivered": artifact is not None,
                "artifact": artifact,
                "utility": bounds,
                "guidance_consumed": {
                    str(k): v.model_dump(mode="json") for k, v in guidance.items()
                },
                "review_errors": review.errors,
                "review_stop_reason": review.stop_reason,
                "execution_resume": execution_resume.receipt if execution_resume else None,
                "result_path": str(directory) if result is not None else None,
                "timing": {"actor_or_import_wall_sec": actor_seconds,
                           "review_wall_sec": review_seconds,
                           "imported_initial": index == 0 and initial_result is not None},
            }
            attempts.append(row)
            write_record(root / "attempt_receipts" / f"{index:04d}.json", row)
            history.append(
                {
                    "attempt_index": index,
                    "review": review.memory,
                    "delivered": row["delivered"],
                    "utility": bounds,
                }
            )
            guidance = review.guidance_by_turn if self.policy.mode != "ordinary_retry" else {}
            for path, digest in immutable.items():
                if sha256(path) != digest:
                    raise ValueError("an earlier source, result or artifact was modified")
            if self.policy.stop_when_all_supported and bounds["lower"] == 1.0:
                break
            if self.policy.stop_without_actionable_guidance and (
                review.stop_reason is not None or not guidance
            ):
                write_record(root / "recovery_stop.json", {
                    "attempt_index": index,
                    "reason": review.stop_reason or "no_qualified_actionable_guidance",
                    "quality_certified": False,
                    "unused_attempt_budget": self.policy.max_retries - index,
                })
                break
        if self.policy.selection_assessment_timing == "after_generation":
            # Generation has ended. New selection observations cannot affect
            # guidance or candidates. Explicit exact-online reuse selects from
            # already observed proxy scores; independent final scores never enter.
            selection_cache = {}
            sequential_selection = getattr(self.selection_reviewer, 'decision_first', False)
            if any(row['artifact'] for row in attempts) and not sequential_selection:
                await resolve_design('selection')
            for row in attempts:
                artifact = row["artifact"]
                if artifact is None:
                    continue
                started = time.monotonic()
                try:
                    cached = selection_cache.get(artifact['sha256'])
                    if cached is None and self.policy.selection_assessment_reuse == 'exact_online':
                        cached = audio_cache.get(artifact['sha256'])
                        if cached is not None:
                            write_record(root/'selection_reuse'/f"{row['attempt_index']:04d}.json", {
                                'source': 'same_episode_online_proxy',
                                'artifact_sha256': artifact['sha256'],
                                'requires_current_semantic_gate': True,
                                'requires_rubric_and_evidence_identity': True,
                                'independent_final_assessment_reused': False,
                            })
                    review_inputs = dict(
                        scenario=scenario, result=outputs[row["attempt_index"]],
                        directory=Path(row["result_path"]),
                        output=root / "selection_reviews" / f"{row['attempt_index']:04d}",
                        attempt_id=row["attempt_id"], history=[],
                        utility_design=utility_design, artifact=artifact,
                        cached_utility=cached,
                        build_feedback=False,
                        **({'resolve_utility_design': lambda: resolve_design(f'selection-{row["attempt_index"]:04d}')}
                           if sequential_selection else {}),
                    )
                    if self.policy.selection == 'mc_conservative_utility':
                        reviews = await self.selection_reviewer.review_mc(
                            repeats=self.policy.selection_mc_repeats, seed=self.policy.selection_mc_seed,
                            **{**review_inputs, 'cached_utility': None})
                        if len(reviews) != self.policy.selection_mc_repeats:
                            raise ValueError('MC reviewer omitted scheduled draws')
                        for sampled in reviews:
                            if sampled.utility is not None and (
                                    sampled.utility.artifact_sha256 != artifact['sha256'] or
                                    sampled.utility.assessor_role != 'online_selection_proxy'):
                                raise ValueError('MC assessment has wrong artifact identity or role')
                        row['mc_utilities'] = [utility_bounds(utility_design, r.utility) for r in reviews]
                        row['mc_review_errors'] = [r.errors for r in reviews]
                        review = reviews[0]
                    else:
                        review = await self.selection_reviewer(**review_inputs)
                    if review.utility is not None:
                        if (review.utility.artifact_sha256 != artifact["sha256"]
                                or review.utility.assessor_role != "online_selection_proxy"):
                            raise ValueError("invalid deferred selection assessment identity or role")
                        if self.policy.selection != 'mc_conservative_utility':
                            validate_cached_assessment(selection_cache.get(artifact['sha256']), review.utility)
                        utility_bounds(utility_design, review.utility)
                        selection_cache[artifact["sha256"]] = review.utility
                except Exception as exc:  # noqa: BLE001 -- keep the candidate and the assessor failure
                    review = IterationReview(errors=[f"{type(exc).__name__}: {exc}"])
                    if self.policy.selection == 'mc_conservative_utility':
                        row['mc_utilities'] = [utility_bounds(utility_design, None)
                                              for _ in range(self.policy.selection_mc_repeats)]
                        row['mc_review_errors'] = [review.errors for _ in row['mc_utilities']]
                row["generation_utility"] = row["utility"]
                row["utility"] = utility_bounds(utility_design, review.utility)
                if self.policy.selection == 'mc_conservative_utility':
                    from x_editor.evaluation.mc_selection import summarize_draws
                    row['mc_summary'] = summarize_draws(row['mc_utilities'])
                    row['utility_role'] = 'first_mc_draw_diagnostic_only; selection uses mc_summary'
                row["selection_review_errors"] = review.errors
                row["timing"]["selection_review_wall_sec"] = time.monotonic() - started
                write_record(root / "selection_receipts" / f"{row['attempt_index']:04d}.json", row)
            for path, digest in immutable.items():
                if sha256(path) != digest:
                    raise ValueError("selection modified original evidence")
        if self.policy.selection == 'mc_conservative_utility':
            from x_editor.evaluation.mc_selection import select_mc_attempt
            chosen, decisions = select_mc_attempt(attempts, repeats=self.policy.selection_mc_repeats)
        elif self.policy.selection == "one_artifact_max_flow":
            from x_editor.evaluation.refinement_flow_selection import select_flow_attempt

            chosen, decisions = select_flow_attempt(attempts)
        else:
            chosen, decisions = select_attempt(attempts)
        selected = chosen["attempt_index"] if chosen else None
        receipt = {
            "policy": self.policy.model_dump(mode="json"),
            "attempts": attempts,
            "selected_attempt_index": selected,
            "selection_decisions": decisions,
            "selected_artifact": chosen["artifact"] if chosen else None,
            "selected_result": outputs[selected].model_dump(mode="json")
            if selected is not None
            else None,
            "independent_quality_gain": None,
            "original_evidence_unchanged": True,
        }
        if self.policy.selection == 'mc_conservative_utility':
            receipt['mc_configuration'] = {
                'repeats': self.policy.selection_mc_repeats, 'seed': self.policy.selection_mc_seed,
                'unit': 'one numeric positive/reverse pair per question per draw',
                'target': 'mean recorded conservative artifact utility',
                'action_value_learning': False, 'extra_actor_attempts': 0,
            }
        write_record(root / "selected_attempt.json", receipt)
        return receipt
