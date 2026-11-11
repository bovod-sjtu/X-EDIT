"""Production review support extracted without changing function bodies."""

from contextlib import nullcontext


from types import SimpleNamespace


def total_attempts(spec):
    total = spec.get("max_attempts")
    if type(total) is not int or not 1 <= total <= 11:
        raise ValueError("max_attempts must explicitly count the initial execution")
    if "max_retries" in spec and spec["max_retries"] != total - 1:
        raise ValueError("max_retries conflicts with max_attempts")
    return total


def refinement_policy(spec, arm):
    """Build the same controller contract during preflight and live execution."""
    from x_editor.revision.refinement_branch_ablation import BRANCH_ARMS
    from x_editor.evaluation.refinement_contracts import RefinementPolicy

    if spec.get("nondelivery_review") == "raw_errors" and spec.get("review_strategy") != "decision_first":
        raise ValueError("delivery-conditioned recovery requires decision-first review after delivery")
    return RefinementPolicy(
        mode="targeted_feedback" if arm in BRANCH_ARMS else arm,
        max_retries=total_attempts(spec) - 1,
        skip_unused_feedback=True,
        selection=spec.get('selection', 'one_artifact_max_flow'),
        selection_mc_repeats=spec.get('selection_mc_repeats', 3),
        selection_mc_seed=spec.get('selection_mc_seed', 0),
        retain_runtime_error_feedback=spec.get("retain_runtime_error_feedback", False),
        nondelivery_review=spec.get("nondelivery_review", "diagnose"),
        execution_reuse=spec.get("execution_reuse", "disabled"),
        stop_without_actionable_guidance=(spec.get("selective_recovery", False)
                                         or spec.get("stop_without_actionable_guidance", False)),
        stop_when_all_supported=spec.get("stop_when_all_supported", False),
        selection_assessment_timing=spec.get("selection_assessment_timing", "online"),
        selection_assessment_reuse=spec.get("selection_assessment_reuse", "none"),
    )


def judge_args(spec, limiter=None):
    strategy = spec.get("review_strategy", "defer_unused")
    if strategy not in {"full", "defer_unused", "outcome_first", "decision_first"}:
        raise ValueError("unsupported review strategy")
    minimum = spec.get('decision_minimum_grade', 2)
    if type(minimum) is not int or minimum not in (1, 2, 3):
        raise ValueError('decision minimum grade must be an integer from 1 to 3')
    rubric_protocol = spec.get("terminal_rubric_protocol", "legacy")
    if rubric_protocol not in {"legacy", "edit_regions", "runtime_regions"}:
        raise ValueError("unsupported terminal rubric protocol")
    context_mode = spec.get("evidence_context_mode", "disabled")
    if context_mode not in {"disabled", "raw_history", "explicit_state"}:
        raise ValueError("unsupported evidence context experiment")
    revisions = spec.get("focus_revision_attempts", 0)
    if type(revisions) is not int or revisions not in {0, 1}:
        raise ValueError("focus revision is an explicit zero-or-one additional critic call")
    contract_revisions = spec.get("evaluation_contract_revisions", 0)
    if type(contract_revisions) is not int or contract_revisions not in {0, 1}:
        raise ValueError("evaluation contract revisions must be zero or one")
    answer_revisions = spec.get("scope_answer_revision_attempts", 0)
    if type(answer_revisions) is not int or answer_revisions not in {0, 1}:
        raise ValueError("scope answer revisions must be zero or one")
    signal_measurements = spec.get("scope_signal_measurements", False)
    if type(signal_measurements) is not bool:
        raise ValueError("scope signal measurements must be explicitly boolean")
    purpose = spec.get('execution_purpose_review', False)
    reuse_scope = spec.get('reuse_scope_preparation', False)
    if type(reuse_scope) is not bool:
        raise ValueError('scope preparation reuse must be explicitly boolean')
    factor_evidence = spec.get('factor_review_evidence', False)
    compact_scopes = spec.get('compact_scope_preparation', False)
    if type(factor_evidence) is not bool or type(compact_scopes) is not bool:
        raise ValueError('review transport choices must be explicitly boolean')
    premise = spec.get('decision_premise_review', False)
    if type(premise) is not bool or (premise and strategy != 'decision_first'):
        raise ValueError('decision premise qualification requires decision-first review')
    if type(purpose) is not bool:
        raise ValueError('execution purpose review must be boolean')
    if purpose and (strategy != 'decision_first' or spec.get('audio_judgment_protocol') != 'numeric_polarity'):
        raise ValueError('purpose sequence requires decision-first numeric polarity review')
    speech_observer = spec.get("scope_speech_observer")
    judgment_protocol = spec.get('audio_judgment_protocol', 'legacy')
    if judgment_protocol not in {'legacy', 'evidence_ordinal', 'numeric_polarity'}:
        raise ValueError('unsupported audio judgment protocol')
    if speech_observer is not None:
        from x_editor.evaluation.scope_speech_observation import SpeechObservationConfig
        speech_observer = SpeechObservationConfig.model_validate(speech_observer).model_dump(mode="json")
    return SimpleNamespace(
        **spec["semantic_judge"], dry_run=False,
        concurrency=spec.get("judge_concurrency", 2),
        semantic_limiter=limiter,
        review_strategy=strategy,
        decision_minimum_grade=minimum,
        decision_premise_review=premise,
        strict_feedback_isolation=spec.get("strict_feedback_isolation", False),
        evidence_context_mode=context_mode,
        focus_revision_attempts=revisions,
        evaluation_contract_revisions=contract_revisions,
        bounded_uncertainty_recheck=spec.get("bounded_uncertainty_recheck", spec.get("selective_recovery", False)),
        goal_resolution_rounds=spec.get("goal_resolution_rounds", 0),
        terminal_rubric_protocol=rubric_protocol,
        task_focused_goals=spec.get('task_focused_goals', False),
        stage_aware_recovery=spec.get('stage_aware_recovery', False),
        error_first_guidance=spec.get('error_first_guidance', False) or strategy == 'decision_first',
        scope_max_rounds=spec.get("scope_max_rounds", 2),
        scope_max_observations=spec.get("scope_max_observations", 2),
        scope_max_probes=spec.get("scope_max_probes", 18),
        scope_speech_observer=speech_observer,
        scope_answer_revision_attempts=answer_revisions,
        scope_signal_measurements=signal_measurements,
        audio_judgment_protocol=judgment_protocol,
        execution_purpose_review=spec.get('execution_purpose_review', False),
        reuse_scope_preparation=reuse_scope,
        factor_review_evidence=factor_evidence,
        compact_scope_preparation=compact_scopes,
    )


def limited(call, limiter, *args, **kwargs):
    with limiter if limiter is not None else nullcontext():
        return call(*args, **kwargs)
