"""Adapt frozen module probes to the existing read-only Evaluation path."""

from __future__ import annotations

from typing import Any

from x_editor.evaluation.guidance_contracts import (
    ArtifactQualityEvaluationTask, AtomicEvaluationRubric, EvaluationArtifactRef,
    EvaluationEffectAtom, EvaluationEvidenceItem, EvaluationJudgePolicy,
    EvaluationSamplingPolicy, RubricEvidenceRequirement, SemanticPathEvaluationTask,
    EvaluationResultV2, canonical_evaluation_fingerprint,
)
from x_editor.evaluation.guidance_path import freeze_evaluation_selection
from x_editor.flow_capacity.quality_protocol import CapacityModule, ModuleProbe, QualityObservation


def module_rubrics(module: CapacityModule) -> tuple[AtomicEvaluationRubric, ...]:
    """General leaf dimensions; task level changes difficulty, not grade meaning."""
    questions = {
        "reasoner": (
            ("goal", "Does the interpretation preserve all requested changes without adding an unauthorized change?"),
            ("scope", "Are asset roles, selected history version, temporal scope and preservation constraints faithful?"),
            ("uncertainty", "Are unresolved facts distinguished from known facts without unnecessary refusal?")),
        "planner": (
            ("goal", "Would this decision or legal plan achieve every requested change with the available capabilities?"),
            ("scope", "Do bindings, dependencies, base version and preservation constraints remain correct?"),
            ("uncertainty", "Are required observations and unmet preconditions explicit, with no invented evidence?")),
        "perception": (
            ("goal", "Do the reported observations accurately cover the requested evidence in the attached audio?"),
            ("scope", "Are observations grounded to the correct asset, version, speaker or event and temporal region?"),
            ("uncertainty", "Does confidence reflect audible or measured evidence, including ambiguity and missing information?")),
        "executor": (
            ("goal", "Does the actual output achieve every requested audible change at the requested strength?"),
            ("scope", "Does it preserve protected content, order, identity, channels and timing where required?"),
            ("quality", "Is the result free of unintended distortion, seams, duplication, omissions or quality loss?")),
    }[module]
    minimum = (RubricEvidenceRequirement(requirement_id="output", acceptable_sources=("subject_output",)),)
    if module == "perception":
        minimum += (RubricEvidenceRequirement(requirement_id="input_audio", acceptable_sources=("audio_regions",)),)
    return tuple(AtomicEvaluationRubric(
        rubric_id=key, dimension=key, question=question,
        observable="Judge meaning and supported evidence; accept alternative valid representations. "
                   "Use unknown for inaudible, unmeasured or contradictory claims. A valid schema is not quality.",
        target_module=module, minimum_evidence=minimum,
    ) for key, question in questions)


def build_module_evaluation_task(
    probe: ModuleProbe, *, subject_output: Any, run_id: str, checkpoint_ref: str,
    backend_id: str, regime: str = "standard", attempt: int = 0, arm: str = "single",
    output_artifact: EvaluationArtifactRef | None = None,
    natural_context: tuple[EvaluationEvidenceItem, ...] | None = None,
    natural_artifacts: tuple[EvaluationArtifactRef, ...] | None = None,
    judge_policy: EvaluationJudgePolicy | None = None,
) -> SemanticPathEvaluationTask | ArtifactQualityEvaluationTask:
    """The natural arm must supply actual upstream evidence, never oracle priors.

    Rubrics and selection derive only from the frozen probe. A heldout outcome
    remains read-only and must not be sent back to a runtime critic.
    """
    if regime not in {"standard", "natural"}:
        raise ValueError("unknown input regime")
    if arm not in {"single", "multi_try", "guided"} or attempt < 0:
        raise ValueError("unknown experiment arm or invalid attempt")
    if regime == "natural" and natural_context is None:
        raise ValueError("natural evaluation requires explicit actual upstream context")
    if regime == "natural" and probe.module in {"perception", "executor"} and natural_artifacts is None:
        raise ValueError("natural audio evaluation requires actual version-bound input artifacts")
    context = probe.input_evidence if regime == "standard" else natural_context
    artifacts = probe.input_artifacts if regime == "standard" else natural_artifacts or ()
    if regime == "standard" and probe.qualified_payload and probe.module != "executor":
        context += (EvaluationEvidenceItem(evidence_id="module_input_payload", kind="other",
                    label="Actual target module input", payload=probe.qualified_payload),)
    focus, kind = {
        "reasoner": ("reasoner_semantics", "reasoner_memo"),
        "planner": ("planner_decision", "planner_decision"),
        "perception": ("perception_evidence", "reviewed_perception"),
        "executor": ("final_delivery", "other"),
    }[probe.module]
    unit_id = f"{probe.probe_id}:{regime}:{arm}:{attempt}"
    selection = freeze_evaluation_selection(
        EvaluationSamplingPolicy(policy_id="module-quality-all-v1", mode="all_steps"),
        unit_id=unit_id, focus=focus,
    )
    common = dict(
        task_id=f"{run_id}:{unit_id}", run_id=run_id, checkpoint_id=checkpoint_ref,
        input_snapshot_ref=f"probe:{probe.fingerprint}/input:" + canonical_evaluation_fingerprint({
            "probe": probe.fingerprint, "regime": regime,
            "context": [e.model_dump(mode="json") for e in context],
            "artifacts": [a.model_dump(mode="json") for a in artifacts],
        }), focus=focus, backend_id=backend_id,
        public_instruction=probe.public_instruction, rubrics=probe.rubrics,
        evaluator_role="heldout_final_judge" if probe.split == "validation" else "post_delivery_development_diagnostic",
        evaluation_use="heldout_outcome" if probe.split == "validation" else
                       "capacity_measurement" if regime == "standard" else "realized_flow_diagnostic",
        selection=selection, judge_policy=judge_policy or EvaluationJudgePolicy(),
    )
    if probe.module == "executor":
        if output_artifact is None:
            raise ValueError("Executor quality requires an actual bound output artifact")
        return ArtifactQualityEvaluationTask(
            **common, visibility="blind_quality", input_artifacts=artifacts,
            output_artifact=output_artifact, effect_contract_ref=probe.fingerprint,
            intended_effects=(EvaluationEffectAtom(effect_id="user_goal", requirement=probe.public_instruction,
                                                   effect_kind="requested_edit"),),
        )
    return SemanticPathEvaluationTask(
        **common, subject_output=EvaluationEvidenceItem(evidence_id="module_output", kind=kind,
            label="Measured module output; representation is not prescribed", payload=subject_output),
        context_evidence=context,
        audio_artifacts=artifacts if probe.module == "perception" else (),
    )


def observation_from_evaluation(
    probe: ModuleProbe, task: SemanticPathEvaluationTask | ArtifactQualityEvaluationTask,
    result: EvaluationResultV2, *, checkpoint_ref: str, regime: str, arm: str = "single", attempt: int = 0,
) -> QualityObservation:
    """Import an actual judge receipt; unresolved vote disagreements stay unknown."""
    if result.task_id != task.task_id or result.run_id != task.run_id or result.selection_id != task.selection.selection_id:
        raise ValueError("judge result is not bound to the measured task")
    if (task.selection.unit_id != f"{probe.probe_id}:{regime}:{arm}:{attempt}" or task.rubrics != probe.rubrics
            or not task.input_snapshot_ref.startswith(f"probe:{probe.fingerprint}/input:")
            or task.checkpoint_id != checkpoint_ref or result.focus != task.focus):
        raise ValueError("task does not match the frozen assignment")
    by_id = {r.rubric_id: r for r in result.rubric_results}
    if set(by_id) != {r.rubric_id for r in probe.rubrics}:
        raise ValueError("result rubric set mismatch")
    grades = {}
    for rubric in probe.rubrics:
        aggregate = by_id[rubric.rubric_id]
        votes = [j.grade for slot in result.slot_results if slot.status == "completed"
                 for j in slot.judgements if j.rubric_id == rubric.rubric_id]
        grades[rubric.rubric_id] = (int(aggregate.grade) if aggregate.status == "graded" and votes
                                  and aggregate.evidence_status == "sufficient"
                                  and aggregate.capacity_passed is not None
                                  and aggregate.grade is not None and aggregate.grade in (0, 1, 2, 3)
                                  and None not in votes and len(set(votes)) == 1 else None)
    return QualityObservation(probe_id=probe.probe_id, probe_fingerprint=probe.fingerprint,
        regime=regime, arm=arm, attempt=attempt, state="completed", checkpoint_ref=checkpoint_ref,
        grades=grades, judge_result_ref=result.result_id)
