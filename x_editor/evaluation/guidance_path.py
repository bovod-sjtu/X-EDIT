"""Read-only orchestration for Evaluation, Guidance, and future attempt policy."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import Field

from x_editor.evaluation.call_accounting import EvaluationCallAccounting
from x_editor.evaluation.guidance_contracts import (
    ArtifactQualityEvaluationTask,
    AtomicRubricAggregate,
    AtomicRubricJudgement,
    EvaluationArtifactRef,
    EvaluationAudioRegion,
    EvaluationEvidenceItem,
    EvaluationFocus,
    EvaluationResultV2,
    EvaluationSamplingPolicy,
    EvaluationTask,
    FrozenEvaluationModel,
    FrozenEvaluationSelection,
    GuidancePacket,
    GuidanceTargetModule,
    GuidedAttemptDecision,
    GuidedAttemptPolicy,
    JudgeSlotResult,
    StepEvaluationSpec,
    canonical_evaluation_fingerprint,
)
from x_editor.evaluation.fact_constraints import check_fact_constraints


class EvaluationGuidanceError(RuntimeError):
    """Base error for the isolated Evaluation-Guidance path."""


class DuplicateEvaluationBackendError(EvaluationGuidanceError):
    """Raised when one backend ID is registered twice."""


class UnknownEvaluationBackendError(EvaluationGuidanceError):
    """Raised when a task references an unregistered backend."""


class UnsupportedEvaluationFocusError(EvaluationGuidanceError):
    """Raised when an explicit backend cannot evaluate the requested focus."""


class EvaluationGuidanceEvent(FrozenEvaluationModel):
    """Append-only sidecar event with no execution authority."""

    schema_version: str = "x-editor.evaluation-guidance-event/v1"
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    timestamp: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    subject_refs: tuple[str, ...] = ()
    payload: dict[str, Any]
    behavior_changed: Literal[False] = False
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False


class EvaluationPathOutcome(FrozenEvaluationModel):
    """One explicit evaluation invocation result or selection skip."""

    status: str
    task_id: str
    selected: bool
    result: EvaluationResultV2 | None = None
    errors: tuple[str, ...] = ()


@runtime_checkable
class EvaluationBackend(Protocol):
    """Provider-agnostic backend for text or multimodal evaluation tasks."""

    @property
    def backend_id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def supported_foci(self) -> frozenset[EvaluationFocus]: ...

    def evaluate_slot(self, task: EvaluationTask, *, slot_id: str) -> JudgeSlotResult: ...


@runtime_checkable
class GuiderBackend(Protocol):
    """Backend that turns Evaluator findings into a constrained GuidancePacket."""

    @property
    def backend_id(self) -> str: ...

    def guide(
        self,
        results: tuple[EvaluationResultV2, ...],
        *,
        target_modules: tuple[GuidanceTargetModule, ...],
    ) -> GuidancePacket: ...


class EvaluationGuidanceLedger:
    """Append-only sidecar persistence isolated from runtime event authority."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.run_id = run_id
        self.events: list[EvaluationGuidanceEvent] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._existing_count = self._validate_existing()

    def append(
        self,
        *,
        kind: str,
        subject_refs: tuple[str, ...] = (),
        payload: dict[str, Any],
    ) -> EvaluationGuidanceEvent:
        event = EvaluationGuidanceEvent(
            event_id=f"eval_event_{uuid4().hex}",
            run_id=self.run_id,
            sequence=self._existing_count + len(self.events) + 1,
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            kind=kind,
            subject_refs=subject_refs,
            payload=payload,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
            handle.flush()
        self.events.append(event)
        return event

    def _validate_existing(self) -> int:
        if not self.path.is_file():
            return 0
        count = 0
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            event = EvaluationGuidanceEvent.model_validate_json(line)
            count += 1
            if event.run_id != self.run_id:
                raise ValueError(f"evaluation sidecar run mismatch at line {line_number}")
            if event.sequence != count:
                raise ValueError(
                    f"non-contiguous evaluation sidecar sequence at line {line_number}"
                )
        return count


class EvaluationGuidancePath:
    """Explicit offline/parallel evaluator dispatcher.

    This class is not imported by UniRuntime. A caller must explicitly submit
    immutable tasks after their selections have been frozen. Backend failures
    become sidecar facts and cannot change the measured execution result.
    """

    def __init__(self, *, ledger: EvaluationGuidanceLedger | None = None,
                 accounting: EvaluationCallAccounting | None = None) -> None:
        self.ledger = ledger
        self.accounting = accounting or EvaluationCallAccounting()
        self._evaluation_backends: dict[str, EvaluationBackend] = {}
        self._guider_backends: dict[str, GuiderBackend] = {}

    def register_evaluator(self, backend: EvaluationBackend) -> None:
        if backend.backend_id in self._evaluation_backends:
            raise DuplicateEvaluationBackendError(
                f"Evaluator backend {backend.backend_id!r} is already registered"
            )
        self._evaluation_backends[backend.backend_id] = backend

    def register_guider(self, backend: GuiderBackend) -> None:
        if backend.backend_id in self._guider_backends:
            raise DuplicateEvaluationBackendError(
                f"Guider backend {backend.backend_id!r} is already registered"
            )
        self._guider_backends[backend.backend_id] = backend

    def evaluate(self, task: EvaluationTask) -> EvaluationPathOutcome:
        self._record(
            kind="evaluation_task_registered",
            subject_refs=(task.task_id, task.checkpoint_id),
            payload={"task": task.model_dump(mode="json")},
        )
        if not task.selection.selected:
            outcome = EvaluationPathOutcome(
                status="skipped_by_frozen_selection",
                task_id=task.task_id,
                selected=False,
            )
            self._record(
                kind="evaluation_task_skipped",
                subject_refs=(task.task_id, task.selection.selection_id),
                payload=outcome.model_dump(mode="json"),
            )
            return outcome

        backend = self._evaluation_backends.get(task.backend_id)
        if backend is None:
            error = UnknownEvaluationBackendError(
                f"Evaluator backend {task.backend_id!r} is not registered"
            )
            return self._failed_outcome(task, error)
        if task.focus not in backend.supported_foci:
            error = UnsupportedEvaluationFocusError(
                f"Evaluator backend {task.backend_id!r} does not support {task.focus!r}"
            )
            return self._failed_outcome(task, error)

        slot_results = tuple(
            self._evaluate_slot_with_fixed_retry(backend, task, slot_id=slot_id)
            for slot_id in task.judge_policy.slot_ids
        )
        result = aggregate_evaluation_result(task, slot_results)
        self._record(
            kind="evaluation_completed",
            subject_refs=(task.task_id, result.result_id),
            payload={"result": result.model_dump(mode="json")},
        )
        return EvaluationPathOutcome(
            status="completed",
            task_id=task.task_id,
            selected=True,
            result=result,
        )

    def guide(
        self,
        *,
        backend_id: str,
        results: tuple[EvaluationResultV2, ...],
        target_modules: tuple[GuidanceTargetModule, ...],
    ) -> GuidancePacket:
        backend = self._guider_backends.get(backend_id)
        if backend is None:
            raise UnknownEvaluationBackendError(f"Guider backend {backend_id!r} is not registered")
        self.accounting.start_logical_call("guider")
        packet = self.accounting.invoke(
            "guider", backend, lambda: backend.guide(results, target_modules=target_modules)
        )
        result_ids = {item.result_id for item in results}
        if not set(packet.source_result_ids).issubset(result_ids):
            raise ValueError("Guider referenced an Evaluation result outside the supplied set")
        actionable_ids = {
            rubric.rubric_id
            for result in results
            for rubric in result.rubric_results
            if rubric.guidance_actionable
        }
        if not packet.failed_or_weak_rubric_ids:
            raise ValueError("Guider must identify at least one actionable weak rubric")
        if not set(packet.failed_or_weak_rubric_ids).issubset(actionable_ids):
            raise ValueError("Guider referenced a non-actionable or unknown rubric")
        cited_evidence = {
            evidence_ref
            for result in results
            for slot in result.slot_results
            for judgement in slot.judgements
            for evidence_ref in judgement.evidence_refs
        }
        if not set(packet.evidence_refs).issubset(cited_evidence):
            raise ValueError("Guider referenced evidence not cited by the Evaluator")
        self._record(
            kind="guidance_packet_created",
            subject_refs=(packet.guidance_id, *packet.source_result_ids),
            payload={"guidance": packet.model_dump(mode="json")},
        )
        return packet

    def _evaluate_slot_with_fixed_retry(
        self,
        backend: EvaluationBackend,
        task: EvaluationTask,
        *,
        slot_id: str,
    ) -> JudgeSlotResult:
        attempts = (
            task.judge_policy.maximum_transport_attempts_per_slot
            if task.judge_policy.retry_invalid_transport_within_slot
            else 1
        )
        errors: list[Exception] = []
        self.accounting.start_logical_call("evaluator")
        for _ in range(attempts):
            try:
                candidate = self.accounting.invoke(
                    "evaluator", backend, lambda: backend.evaluate_slot(task, slot_id=slot_id)
                )
                return _validate_slot_shape(candidate, task, backend, slot_id)
            except Exception as exc:  # noqa: BLE001 - isolate arbitrary provider faults.
                errors.append(exc)
        error = errors[-1]
        request_fingerprint = canonical_evaluation_fingerprint(
            {"task_id": task.task_id, "slot_id": slot_id, "backend_id": backend.backend_id}
        )
        return JudgeSlotResult(
            slot_id=slot_id,
            backend_id=backend.backend_id,
            provider=backend.provider,
            model=backend.model,
            status="provider_failure",
            request_fingerprint=request_fingerprint,
            error_type=type(error).__name__,
            error_message=str(error)[:2048],
        )

    def _failed_outcome(
        self,
        task: EvaluationTask,
        error: Exception,
    ) -> EvaluationPathOutcome:
        outcome = EvaluationPathOutcome(
            status="configuration_failure",
            task_id=task.task_id,
            selected=True,
            errors=(f"{type(error).__name__}: {error}",),
        )
        self._record(
            kind="evaluation_configuration_failed",
            subject_refs=(task.task_id,),
            payload=outcome.model_dump(mode="json"),
        )
        return outcome

    def _record(
        self,
        *,
        kind: str,
        subject_refs: tuple[str, ...],
        payload: dict[str, Any],
    ) -> None:
        if self.ledger is not None:
            self.ledger.append(
                kind=kind,
                subject_refs=subject_refs,
                payload=payload,
            )


class GuidedAttemptController:
    """Policy-only controller; it never invokes Reasoner, Planner, or Runtime."""

    def decide(
        self,
        policy: GuidedAttemptPolicy,
        guidance_packets: tuple[GuidancePacket, ...],
        *,
        completed_new_attempts: int,
        evaluator_calls_used: int,
        failure_fingerprint: str | None = None,
        prior_failure_fingerprints: tuple[str, ...] = (),
        guidance_fingerprint: str | None = None,
        prior_guidance_fingerprints: tuple[str, ...] = (),
        modifiable_owner_available: bool = True,
        new_evidence_or_issue_available: bool = True,
    ) -> GuidedAttemptDecision:
        decision_seed = {
            "policy": policy.model_dump(mode="json"),
            "guidance_ids": [item.guidance_id for item in guidance_packets],
            "completed_new_attempts": completed_new_attempts,
            "evaluator_calls_used": evaluator_calls_used,
            "failure_fingerprint": failure_fingerprint,
            "prior_failure_fingerprints": prior_failure_fingerprints,
            "guidance_fingerprint": guidance_fingerprint,
            "prior_guidance_fingerprints": prior_guidance_fingerprints,
            "modifiable_owner_available": modifiable_owner_available,
            "new_evidence_or_issue_available": new_evidence_or_issue_available,
        }
        decision_id = "guided_decision_" + canonical_evaluation_fingerprint(decision_seed)[:24]
        common = {
            "decision_id": decision_id,
            "policy_id": policy.policy_id,
            "guidance_ids": tuple(item.guidance_id for item in guidance_packets),
        }
        if policy.mode == "disabled":
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("guided_attempts_disabled",),
                new_attempt_authorized=False,
            )
        if completed_new_attempts >= policy.max_new_attempts:
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("new_attempt_budget_exhausted",),
                new_attempt_authorized=False,
            )
        if evaluator_calls_used >= policy.max_evaluator_calls:
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("evaluator_budget_exhausted",),
                new_attempt_authorized=False,
            )
        if (
            failure_fingerprint
            and policy.stop_on_repeated_failure_fingerprint
            and failure_fingerprint in prior_failure_fingerprints
        ):
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("repeated_failure_fingerprint",),
                new_attempt_authorized=False,
            )
        if not guidance_packets:
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("no_guidance_packet",),
                new_attempt_authorized=False,
            )
        if not modifiable_owner_available:
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("no_modifiable_guidance_owner",),
                new_attempt_authorized=False,
            )
        if not new_evidence_or_issue_available:
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("no_new_evidence_or_actionable_issue",),
                new_attempt_authorized=False,
            )
        if (
            guidance_fingerprint
            and policy.stop_on_repeated_guidance_fingerprint
            and guidance_fingerprint in prior_guidance_fingerprints
        ):
            return GuidedAttemptDecision(
                **common,
                action="no_new_attempt",
                reason_codes=("repeated_guidance_fingerprint",),
                new_attempt_authorized=False,
            )
        if policy.mode == "audit":
            return GuidedAttemptDecision(
                **common,
                action="shadow_recommendation",
                reason_codes=("guidance_available_within_shadow_budget",),
                new_attempt_authorized=False,
            )
        return GuidedAttemptDecision(
            **common,
            action="open_new_attempt",
            reason_codes=("preregistered_guidance_arm_within_budget",),
            new_attempt_authorized=True,
            proposed_attempt_index=completed_new_attempts + 1,
        )


def freeze_evaluation_selection(
    policy: EvaluationSamplingPolicy,
    *,
    unit_id: str,
    focus: EvaluationFocus,
) -> FrozenEvaluationSelection:
    """Select a unit without reading module output or execution outcome."""

    if policy.mode == "failure_forensics":
        raise ValueError("failure_forensics is outcome-triggered; use build_forensic_selection")
    selection_hash = canonical_evaluation_fingerprint(
        {
            "policy_fingerprint": policy.policy_fingerprint,
            "unit_id": unit_id,
            "focus": focus,
        }
    )
    if policy.mode == "off":
        selected = False
        reason = "evaluation_mode_off"
    elif policy.mode == "final_only":
        selected = focus == "final_delivery"
        reason = "final_delivery_selected" if selected else "non_final_unit_excluded"
    elif policy.mode == "all_steps":
        selected = True
        reason = "all_units_selected"
    elif focus == "final_delivery" and policy.include_final_delivery:
        selected = True
        reason = "final_delivery_always_selected"
    else:
        draw = int(selection_hash[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        selected = draw < policy.sample_rate
        reason = "deterministic_sparse_sample" if selected else "sparse_sample_excluded"
    selection_id = (
        "eval_selection_"
        + canonical_evaluation_fingerprint(
            {"selection_hash": selection_hash, "selected": selected, "reason": reason}
        )[:24]
    )
    return FrozenEvaluationSelection(
        selection_id=selection_id,
        policy_id=policy.policy_id,
        policy_fingerprint=policy.policy_fingerprint,
        unit_id=unit_id,
        focus=focus,
        selected=selected,
        selection_reason=reason,
        selected_before_output=True,
        outcome_independent=True,
        capacity_eligible=selected,
        forensic_only=False,
        selection_hash=selection_hash,
    )


def build_forensic_selection(
    policy: EvaluationSamplingPolicy,
    *,
    unit_id: str,
    focus: EvaluationFocus,
    failure_ref: str,
) -> FrozenEvaluationSelection:
    """Record outcome-triggered diagnostics that are excluded from capacity."""

    if policy.mode != "failure_forensics":
        raise ValueError("forensic selection requires failure_forensics mode")
    selection_hash = canonical_evaluation_fingerprint(
        {
            "policy_fingerprint": policy.policy_fingerprint,
            "unit_id": unit_id,
            "focus": focus,
            "failure_ref": failure_ref,
        }
    )
    return FrozenEvaluationSelection(
        selection_id="eval_forensic_" + selection_hash[:24],
        policy_id=policy.policy_id,
        policy_fingerprint=policy.policy_fingerprint,
        unit_id=unit_id,
        focus=focus,
        selected=True,
        selection_reason=f"failure_forensics:{failure_ref}",
        selected_before_output=False,
        outcome_independent=False,
        capacity_eligible=False,
        forensic_only=True,
        selection_hash=selection_hash,
    )


def bind_step_evaluation_output(
    spec: StepEvaluationSpec,
    *,
    output_artifact: EvaluationArtifactRef,
    regions: tuple[EvaluationAudioRegion, ...] = (),
    forensic_evidence: tuple[EvaluationEvidenceItem, ...] = (),
) -> ArtifactQualityEvaluationTask:
    """Mechanically bind a completed artifact to a pre-output step spec.

    The operation does not create or change rubric atoms, effects, selection,
    backend, or judge policy. Timeline lineage may supply the concrete output
    regions after execution, but it cannot change what will be judged.
    """

    return ArtifactQualityEvaluationTask(
        task_id=spec.task_id,
        run_id=spec.run_id,
        checkpoint_id=spec.checkpoint_id,
        input_snapshot_ref=spec.input_snapshot_ref,
        focus="intermediate_step",
        evaluator_role=spec.evaluator_role,
        evaluation_use=spec.evaluation_use,
        backend_id=spec.backend_id,
        visibility="blind_quality",
        public_instruction=spec.public_instruction,
        input_artifacts=spec.input_artifacts,
        output_artifact=output_artifact,
        reference_artifacts=spec.reference_artifacts,
        producer_node_id=spec.producer_node_id,
        effect_contract_ref=spec.effect_contract_ref,
        intended_effects=spec.intended_effects,
        preservation_requirements=spec.preservation_requirements,
        regions=regions,
        rubrics=spec.rubrics,
        forensic_evidence=forensic_evidence,
        context_padding_sec=spec.context_padding_sec,
        judge_policy=spec.judge_policy,
        selection=spec.selection,
    )


def _fact_evidence(task: EvaluationTask) -> dict[str, EvaluationEvidenceItem]:
    if isinstance(task, ArtifactQualityEvaluationTask):
        items = task.measurement_evidence
    else:
        items = (task.subject_output, *task.context_evidence, *task.candidate_operation_cards)
    by_id = {}
    conflicting_ids = set()
    for item in items:
        previous = by_id.get(item.evidence_id)
        if previous is not None and previous != item:
            conflicting_ids.add(item.evidence_id)
        by_id[item.evidence_id] = item
    return {key: value for key, value in by_id.items() if key not in conflicting_ids}


def aggregate_evaluation_result(
    task: EvaluationTask,
    slot_results: tuple[JudgeSlotResult, ...],
) -> EvaluationResultV2:
    """Aggregate fixed judge slots while preserving N/A and unknown mass."""

    expected_slots = tuple(task.judge_policy.slot_ids)
    observed_slots = tuple(item.slot_id for item in slot_results)
    if observed_slots != expected_slots:
        raise ValueError("judge slot results must match frozen slot order exactly")
    aggregates: list[AtomicRubricAggregate] = []
    known_weight = 0.0
    unknown_weight = 0.0
    known_weighted_score = 0.0
    fact_evidence = _fact_evidence(task)

    for rubric in task.rubrics:
        evidence_status = _rubric_evidence_status(task, rubric)
        fact_checks = check_fact_constraints(rubric.fact_constraints, fact_evidence)
        if any(item["status"] == "unknown" for item in fact_checks) and evidence_status != "conflicting":
            evidence_status = "missing"
        violations = [item for item in fact_checks if item["status"] == "violated"]
        judgements = _effective_rubric_judgements(slot_results, rubric.rubric_id)
        grades = [
            int(item.grade)
            for item in judgements
            if item.applicability == "applicable" and not item.abstain and item.grade is not None
        ]
        not_applicable_votes = sum(item.applicability == "not_applicable" for item in judgements)
        complete_grades = _complete_rubric_grades(
            judgements, expected_count=len(expected_slots), evidence_status=evidence_status
        )
        model_grade = float(statistics.median(grades)) if grades else None
        audit = {"model_grade": model_grade, "fact_checks": fact_checks}
        if not_applicable_votes == len(expected_slots) and evidence_status == "sufficient" and not violations:
            aggregate = AtomicRubricAggregate(
                **audit,
                rubric_id=rubric.rubric_id,
                weight=rubric.weight,
                status="not_applicable",
                valid_slot_count=not_applicable_votes,
                evidence_status=evidence_status,
            )
        elif evidence_status == "sufficient" and not_applicable_votes == 0 and len(grades) >= task.judge_policy.minimum_valid_slots:
            grade = model_grade
            if violations:
                grade = min(grade, *(item["violation_grade_ceiling"] for item in violations))
            normalized = grade / 3.0
            aggregate = AtomicRubricAggregate(
                **audit,
                grade_source="bounded_by_measured_constraint" if violations else "model",
                rubric_id=rubric.rubric_id,
                weight=rubric.weight,
                status="graded",
                grade=grade,
                normalized_score=normalized,
                valid_slot_count=len(grades),
                slot_grades=tuple(grades),
                disagreement=float(max(grades) - min(grades)),
                capacity_passed=False if violations else _consensus_membership(
                    complete_grades, rubric.capacity_pass_grades
                ),
                guidance_actionable=(
                    complete_grades is not None
                    and ((bool(violations) and grade in rubric.guidance_actionable_grades)
                         or (not violations and len(set(complete_grades)) == 1
                             and complete_grades[0] in rubric.guidance_actionable_grades))
                ),
                evidence_status=evidence_status,
            )
            known_weight += rubric.weight
            known_weighted_score += rubric.weight * normalized
        else:
            aggregate = AtomicRubricAggregate(
                **audit,
                grade_source="unknown_evidence" if evidence_status != "sufficient" else "model",
                rubric_id=rubric.rubric_id,
                weight=rubric.weight,
                status="unknown",
                valid_slot_count=len(grades),
                slot_grades=tuple(grades),
                disagreement=(float(max(grades) - min(grades)) if len(grades) > 1 else None),
                evidence_status=evidence_status,
            )
            unknown_weight += rubric.weight
        aggregates.append(aggregate)

    applicable_weight = known_weight + unknown_weight
    if applicable_weight == 0.0:
        score_status = "not_applicable"
        known_score = None
        lower_bound = 0.0
        upper_bound = 0.0
        known_fraction = 1.0
    elif known_weight == 0.0:
        score_status = "unknown"
        known_score = None
        lower_bound = 0.0
        upper_bound = 1.0
        known_fraction = 0.0
    else:
        score_status = "scored"
        known_score = known_weighted_score / known_weight
        lower_bound = known_weighted_score / applicable_weight
        upper_bound = (known_weighted_score + unknown_weight) / applicable_weight
        known_fraction = known_weight / applicable_weight

    result_payload = {
        "task_id": task.task_id,
        "selection_id": task.selection.selection_id,
        "slots": [item.model_dump(mode="json") for item in slot_results],
        "rubrics": [item.model_dump(mode="json") for item in aggregates],
    }
    completed_slots = [item for item in slot_results if item.status == "completed"]
    effective_groups = {item.independence_group for item in completed_slots}
    return EvaluationResultV2(
        result_id="evaluation_result_" + canonical_evaluation_fingerprint(result_payload)[:24],
        task_id=task.task_id,
        run_id=task.run_id,
        focus=task.focus,
        evaluator_role=task.evaluator_role,
        evaluation_use=task.evaluation_use,
        selection_id=task.selection.selection_id,
        slot_results=slot_results,
        rubric_results=tuple(aggregates),
        score_status=score_status,
        known_score=known_score,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        known_weight_fraction=known_fraction,
        effective_judge_count=len(effective_groups),
        correlated_slot_count=max(0, len(completed_slots) - len(effective_groups)),
        independent_vote_claimed=bool(completed_slots)
        and all(item.independence_claim == "independent" for item in completed_slots),
        capacity_eligible=(
            task.selection.capacity_eligible and task.evaluation_use == "capacity_measurement"
        ),
    )


def _validate_slot_shape(
    candidate: JudgeSlotResult,
    task: EvaluationTask,
    backend: EvaluationBackend,
    slot_id: str,
) -> JudgeSlotResult:
    if candidate.slot_id != slot_id:
        raise ValueError("backend returned a different judge slot ID")
    if candidate.backend_id != backend.backend_id:
        raise ValueError("backend result identity does not match registered backend")
    if candidate.status != "completed":
        return candidate
    expected_rubrics = tuple(item.rubric_id for item in task.rubrics)
    observed_rubrics = tuple(item.rubric_id for item in candidate.judgements)
    if observed_rubrics != expected_rubrics:
        raise ValueError("judge output must contain every frozen rubric exactly once and in order")
    allowed_refs = _judge_visible_evidence_refs(task)
    for judgement in candidate.judgements:
        unknown_refs = set(judgement.evidence_refs) - allowed_refs
        if unknown_refs:
            raise ValueError(f"judge cited unknown evidence refs: {sorted(unknown_refs)}")
    return _enforce_minimum_evidence(candidate, task)


def _enforce_minimum_evidence(
    slot: JudgeSlotResult,
    task: EvaluationTask,
) -> JudgeSlotResult:
    rubrics = {item.rubric_id: item for item in task.rubrics}
    judgements: list[AtomicRubricJudgement] = []
    for judgement in slot.judgements:
        status = _rubric_evidence_status(task, rubrics[judgement.rubric_id])
        if status == "sufficient":
            judgements.append(judgement)
            continue
        judgements.append(
            AtomicRubricJudgement(
                rubric_id=judgement.rubric_id,
                applicability="unknown",
                abstain=True,
                reason=f"minimum evidence is {status}; semantic grading is not permitted",
            )
        )
    return slot.model_copy(update={"judgements": tuple(judgements)})


def _rubric_evidence_status(task: EvaluationTask, rubric: Any) -> str:
    status = "sufficient"
    for requirement in rubric.minimum_evidence:
        matching_states: list[str] = []
        for source in requirement.acceptable_sources:
            if source == "public_instruction":
                matching_states.append("available" if task.public_instruction.strip() else "missing")
            elif source == "subject_output":
                if isinstance(task, ArtifactQualityEvaluationTask):
                    matching_states.append("available")
                else:
                    matching_states.append(task.subject_output.evidence_state)
            elif source == "context_evidence" and not isinstance(
                task, ArtifactQualityEvaluationTask
            ):
                matching_states.extend(
                    item.evidence_state
                    for item in task.context_evidence
                    if not requirement.acceptable_kinds
                    or item.kind in requirement.acceptable_kinds
                )
            elif source == "candidate_operation_cards" and not isinstance(
                task, ArtifactQualityEvaluationTask
            ):
                matching_states.extend(
                    item.evidence_state
                    for item in task.candidate_operation_cards
                    if not requirement.acceptable_kinds
                    or item.kind in requirement.acceptable_kinds
                )
            elif source == "audio_regions" and isinstance(
                task, ArtifactQualityEvaluationTask
            ):
                matching_states.extend("available" for _ in task.regions)
            elif source == "audio_regions" and task.focus == "perception_evidence":
                matching_states.extend("available" for _ in task.audio_artifacts)
            elif source == "measurement_evidence" and isinstance(task, ArtifactQualityEvaluationTask):
                matching_states.extend(e.evidence_state for e in task.measurement_evidence)
        if any(item == "conflicting" for item in matching_states):
            return "conflicting"
        available = sum(item == "available" for item in matching_states)
        if available < requirement.minimum_count:
            return "missing"
    return status


def _complete_rubric_grades(
    judgements: list[AtomicRubricJudgement],
    *,
    expected_count: int,
    evidence_status: str,
) -> tuple[int, ...] | None:
    """Only a full set of applicable, non-abstaining votes supports consensus."""
    if evidence_status != "sufficient" or len(judgements) != expected_count:
        return None
    if any(
        item.applicability != "applicable" or item.abstain or item.grade is None
        for item in judgements
    ):
        return None
    return tuple(item.grade for item in judgements if item.grade is not None)


def _consensus_membership(
    grades: tuple[int, ...] | None, accepted_grades: tuple[int, ...],
) -> bool | None:
    if not grades:
        return None
    decisions = {grade in accepted_grades for grade in grades}
    return next(iter(decisions)) if len(decisions) == 1 else None


def _effective_rubric_judgements(
    slot_results: tuple[JudgeSlotResult, ...],
    rubric_id: str,
) -> list[AtomicRubricJudgement]:
    judgements: list[AtomicRubricJudgement] = []
    for slot in slot_results:
        if slot.status != "completed":
            continue
        judgement = next(
            (item for item in slot.judgements if item.rubric_id == rubric_id),
            None,
        )
        if judgement is not None:
            judgements.append(judgement)
    # Correlated samples remain observable votes, but the aggregate separately
    # reports their effective judge groups and never calls them independent.
    return judgements


def _judge_visible_evidence_refs(task: EvaluationTask) -> set[str]:
    if isinstance(task, ArtifactQualityEvaluationTask):
        return {
            *(item.artifact_ref_id for item in task.input_artifacts),
            task.output_artifact.artifact_ref_id,
            *(item.artifact_ref_id for item in task.reference_artifacts),
            *(item.region_id for item in task.regions),
            *(item.clip_id for item in task.materialized_clips),
            *(item.evidence_id for item in task.measurement_evidence),
            *(
                item.evidence_id
                for item in task.forensic_evidence
                if item.visibility in {"judge", "forensic_only"}
            ),
        }
    return {
        "public_instruction",
        task.subject_output.evidence_id,
        *(item.artifact_ref_id for item in task.audio_artifacts),
        *(
            item.evidence_id
            for item in (*task.context_evidence, *task.candidate_operation_cards)
            if item.visibility == "judge"
        ),
    }


__all__ = [
    "DuplicateEvaluationBackendError",
    "EvaluationBackend",
    "EvaluationGuidanceEvent",
    "EvaluationGuidanceLedger",
    "EvaluationGuidancePath",
    "EvaluationPathOutcome",
    "GuidedAttemptController",
    "GuiderBackend",
    "UnknownEvaluationBackendError",
    "UnsupportedEvaluationFocusError",
    "aggregate_evaluation_result",
    "bind_step_evaluation_output",
    "build_forensic_selection",
    "freeze_evaluation_selection",
]
