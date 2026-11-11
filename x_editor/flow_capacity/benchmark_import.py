"""Import benchmark workflow evidence into the existing capacity fact ledger.

The importer is intentionally read-only and non-scoring.  Runtime facts remain
authoritative for forward-flow decisions; optional evaluator evidence can only
add ``MetricObservation`` rows after exact projection and artifact linkage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

from .identity import stable_id
from .schemas import (
    AttemptFact,
    FailureUnknownFact,
    MetricObservation,
    ModuleAssessmentRecord,
    ModuleId,
)


ModuleDisposition = Literal["reached", "accepted", "rejected", "failed"]


class ModuleExecutionObservation(StrictModel):
    """Module-bound index over one existing capacity ``AttemptFact``."""

    schema_version: Literal["x-editor.benchmark-module-observation/v1"] = (
        "x-editor.benchmark-module-observation/v1"
    )
    observation_id: str
    assessment_record_id: str
    module_id: ModuleId
    boundary_id: str
    turn_index: int = Field(ge=0)
    stage: str
    disposition: ModuleDisposition
    reason_codes: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    reflow_from_attempt_id: str | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    attempt: AttemptFact

    @model_validator(mode="after")
    def validate_attempt_link(self) -> ModuleExecutionObservation:
        if self.reflow_from_attempt_id != self.attempt.parent_attempt_id:
            raise ValueError("reflow provenance must match the attempt parent")
        if self.disposition == "accepted" and self.attempt.execution_state != "succeeded":
            raise ValueError("accepted module observation requires a succeeded attempt")
        if (
            self.disposition in {"rejected", "failed"}
            and self.attempt.execution_state == "succeeded"
        ):
            raise ValueError("rejected or failed module observation cannot be succeeded")
        return self


class TrustedEvaluatorEvidence(StrictModel):
    """Explicit trust and immutable-output link around a metric observation."""

    schema_version: Literal["x-editor.trusted-evaluator-evidence/v1"] = (
        "x-editor.trusted-evaluator-evidence/v1"
    )
    trusted: bool = False
    runtime_projection_fingerprint: str
    artifact_ref_id: str
    observation: MetricObservation

    @model_validator(mode="after")
    def validate_trust_claim(self) -> TrustedEvaluatorEvidence:
        if self.artifact_ref_id not in self.observation.evidence_refs:
            raise ValueError(
                "evaluator observation must cite its linked artifact as evidence"
            )
        if self.trusted and self.observation.truth_basis not in {
            "deterministic_contract",
            "hidden_gt",
            "human",
        }:
            raise ValueError(
                "trusted evaluator evidence requires an independently trusted truth basis"
            )
        return self


class BenchmarkCapacityImport(StrictModel):
    """One non-authoritative import batch for the existing capacity ledger."""

    schema_version: Literal["x-editor.benchmark-capacity-import/v1"] = (
        "x-editor.benchmark-capacity-import/v1"
    )
    runtime_projection_fingerprint: str
    sidecar_record_key: str
    assigned_arm: str
    module_observations: list[ModuleExecutionObservation] = Field(default_factory=list)
    metric_observations: list[MetricObservation] = Field(default_factory=list)
    failures_unknowns: list[FailureUnknownFact] = Field(default_factory=list)

    @property
    def attempts(self) -> tuple[AttemptFact, ...]:
        return tuple(item.attempt for item in self.module_observations)

    @model_validator(mode="after")
    def validate_observation_ids(self) -> BenchmarkCapacityImport:
        ids = [item.observation_id for item in self.module_observations]
        ids.extend(item.observation_id for item in self.metric_observations)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate observation_id in benchmark capacity import")
        return self


@dataclass(frozen=True)
class _ImportIdentity:
    projection_fingerprint: str
    sidecar_record_key: str
    assigned_arm: str
    outer_status: str
    scenario: dict[str, Any]


@dataclass(frozen=True)
class _RawModuleFact:
    module_id: ModuleId
    stage: str
    run_id: str
    attempt_id: str
    turn_index: int
    disposition: ModuleDisposition
    selected_for_forward_flow: bool = False
    reason_codes: tuple[str, ...] = ()
    retry_count: int = 0
    reflow_from_attempt_id: str | None = None
    latency_seconds: float | None = None
    cost: tuple[tuple[str, float], ...] = ()
    artifact_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()


def import_benchmark_evidence(
    payload: Any,
    *,
    assessments: Sequence[ModuleAssessmentRecord],
    evaluator_evidence: Sequence[TrustedEvaluatorEvidence | Mapping[str, Any]] = (),
) -> BenchmarkCapacityImport:
    """Project one arm/scenario result without deriving quality or capacity.

    ``payload`` may be either a persisted ``x-editor.benchmark-arm-result/v1``
    row or its ``x-editor.workflow-scenario-result/v1`` result payload.  The
    supplied assessments are the frozen development-set boundaries to which
    imported facts are attached.
    """

    assessment_rows = tuple(assessments)
    identity = _import_identity(payload, assessment_rows)
    assessment_by_id = _validate_assessments(assessment_rows, identity)
    assessments_by_module: dict[str, list[ModuleAssessmentRecord]] = {}
    for assessment in assessment_rows:
        assessments_by_module.setdefault(assessment.boundary.module_id, []).append(
            assessment
        )

    raw_facts = _scenario_module_facts(identity.scenario)
    if identity.outer_status == "artifact_missing":
        raw_facts.append(_artifact_missing_fact(identity.scenario))
    module_observations: list[ModuleExecutionObservation] = []
    failures_unknowns: list[FailureUnknownFact] = []
    for raw in raw_facts:
        for assessment in assessments_by_module.get(raw.module_id, []):
            observation = _module_observation(raw, assessment)
            module_observations.append(observation)
            if raw.disposition in {"rejected", "failed"}:
                failures_unknowns.extend(
                    _module_failure_facts(observation, raw.reason_codes)
                )

    _reject_duplicate_ids(
        [item.observation_id for item in module_observations],
        label="module observation",
    )
    evaluator_rows = [
        item
        if isinstance(item, TrustedEvaluatorEvidence)
        else TrustedEvaluatorEvidence.model_validate(item)
        for item in evaluator_evidence
    ]
    _reject_duplicate_ids(
        [item.observation.observation_id for item in evaluator_rows],
        label="evaluator observation",
    )
    metric_observations = _import_evaluator_observations(
        evaluator_rows,
        identity=identity,
        assessment_by_id=assessment_by_id,
        module_observations=module_observations,
    )
    measured_objectives = {
        (
            item.context.assessment_record_id,
            item.context.objective_instance_id,
        )
        for item in metric_observations
    }
    observed_attempts = {
        item.assessment_record_id: item.attempt.attempt_id
        for item in module_observations
    }
    for assessment in assessment_rows:
        for objective_id in assessment.objective_refs:
            if (assessment.assessment_record_id, objective_id) in measured_objectives:
                continue
            failures_unknowns.append(
                FailureUnknownFact(
                    fact_id=stable_id(
                        "missing_evaluation",
                        {
                            "projection": identity.projection_fingerprint,
                            "arm": identity.assigned_arm,
                            "assessment": assessment.assessment_record_id,
                            "objective": objective_id,
                        },
                    ),
                    run_id=_scenario_run_id(identity.scenario),
                    assessment_record_id=assessment.assessment_record_id,
                    objective_instance_id=objective_id,
                    attempt_id=observed_attempts.get(
                        assessment.assessment_record_id
                    ),
                    category="unknown",
                    reason_code="evaluator_evidence_missing",
                    failure_owner="evaluator",
                )
            )

    return BenchmarkCapacityImport(
        runtime_projection_fingerprint=identity.projection_fingerprint,
        sidecar_record_key=identity.sidecar_record_key,
        assigned_arm=identity.assigned_arm,
        module_observations=sorted(
            module_observations,
            key=lambda item: (
                item.turn_index,
                item.module_id,
                item.attempt.attempt_id,
                item.assessment_record_id,
            ),
        ),
        metric_observations=sorted(
            metric_observations,
            key=lambda item: item.observation_id,
        ),
        failures_unknowns=sorted(
            failures_unknowns,
            key=lambda item: item.fact_id,
        ),
    )


def _import_identity(
    payload: Any,
    assessments: Sequence[ModuleAssessmentRecord],
) -> _ImportIdentity:
    row = _as_dict(payload)
    schema_version = str(row.get("schema_version") or "")
    if schema_version == "x-editor.benchmark-arm-result/v1":
        outer_status = _required_string(row, "status")
        if outer_status == "projection_failed":
            raise ValueError(
                "projection_failed has no runtime projection or module assessment evidence"
            )
        if outer_status in {"configuration_failed", "runtime_exception"}:
            raise ValueError(
                f"{outer_status} has a runtime projection but no workflow scenario "
                "or module trace to import"
            )
        if outer_status not in {
            "completed",
            "blocked",
            "failed",
            "artifact_missing",
        }:
            raise ValueError(f"unsupported benchmark arm result status: {outer_status}")
        scenario = _required_mapping(row, "result")
        if scenario.get("schema_version") != "x-editor.workflow-scenario-result/v1":
            raise ValueError("benchmark arm result contains an unsupported result payload")
        projection = _required_string(row, "runtime_projection_fingerprint")
        if projection != _required_string(scenario, "scenario_fingerprint"):
            raise ValueError("benchmark and workflow runtime projection fingerprints differ")
        assigned_arm = _required_string(row, "assigned_arm")
        if assigned_arm != _required_string(scenario, "assigned_arm"):
            raise ValueError("benchmark and workflow assigned arms differ")
        nested_status = _required_string(scenario, "status")
        if outer_status == "artifact_missing" and nested_status != "completed":
            raise ValueError(
                "artifact_missing requires a nested completed workflow result"
            )
        if outer_status != "artifact_missing" and outer_status != nested_status:
            raise ValueError("benchmark and workflow execution statuses differ")
        audit_identity = _required_mapping(row, "audit_identity")
        sidecar_record_key = _required_string(audit_identity, "record_key")
    elif schema_version == "x-editor.workflow-scenario-result/v1":
        scenario = row
        projection = _required_string(row, "scenario_fingerprint")
        assigned_arm = _required_string(row, "assigned_arm")
        outer_status = _required_string(row, "status")
        sidecar_keys = {
            item.identity.sidecar_record_key for item in assessments
        }
        if len(sidecar_keys) != 1:
            raise ValueError(
                "workflow result import requires one shared assessment sidecar identity"
            )
        sidecar_record_key = next(iter(sidecar_keys))
    else:
        raise ValueError(f"unsupported benchmark evidence schema: {schema_version or '<missing>'}")
    return _ImportIdentity(
        projection_fingerprint=projection,
        sidecar_record_key=sidecar_record_key,
        assigned_arm=assigned_arm,
        outer_status=outer_status,
        scenario=scenario,
    )


def _validate_assessments(
    assessments: Sequence[ModuleAssessmentRecord],
    identity: _ImportIdentity,
) -> dict[str, ModuleAssessmentRecord]:
    if not assessments:
        raise ValueError("benchmark evidence import requires module assessments")
    by_id: dict[str, ModuleAssessmentRecord] = {}
    for assessment in assessments:
        if assessment.assessment_record_id in by_id:
            raise ValueError("duplicate assessment_record_id")
        if assessment.identity.case_fingerprint != identity.projection_fingerprint:
            raise ValueError("assessment runtime projection fingerprint mismatch")
        if assessment.identity.sidecar_record_key != identity.sidecar_record_key:
            raise ValueError("assessment sidecar identity mismatch")
        by_id[assessment.assessment_record_id] = assessment
    return by_id


def _scenario_module_facts(scenario: Mapping[str, Any]) -> list[_RawModuleFact]:
    facts: list[_RawModuleFact] = []
    turns = scenario.get("turn_results") or []
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("workflow scenario turn_results must be a sequence")
    for ordinal, raw_turn in enumerate(turns):
        turn = _as_dict(raw_turn)
        turn_index = int(turn.get("turn_index", ordinal))
        arm_result = _required_mapping(turn, "result")
        if arm_result.get("schema_version") not in {
            None,
            "x-editor.workflow-arm-result/v1",
        }:
            raise ValueError("unsupported workflow arm result payload")
        if _required_string(arm_result, "assigned_arm") != _required_string(
            scenario, "assigned_arm"
        ):
            raise ValueError("turn result assigned arm differs from scenario")
        facts.extend(_arm_module_facts(arm_result, turn_index=turn_index))
    return facts


def _artifact_missing_fact(scenario: Mapping[str, Any]) -> _RawModuleFact:
    turns = scenario.get("turn_results") or []
    last_turn = _as_dict(turns[-1]) if turns else {}
    turn_index = int(last_turn.get("turn_index", 0))
    arm_result = _mapping_or_empty(last_turn.get("result"))
    forward = _mapping_or_empty(arm_result.get("forward_result"))
    run_id = (
        _optional_string(arm_result.get("run_id"))
        or _scenario_run_id(scenario)
    )
    return _raw_fact(
        module_id="artifact_store",
        stage="artifact_delivery",
        run_id=run_id,
        attempt_id="artifact_delivery:runner",
        turn_index=turn_index,
        disposition="failed",
        selected_for_forward_flow=False,
        reason_codes=["completed_scenario_missing_final_artifact"],
        artifact_refs=_artifact_refs(forward),
        evidence_refs=_evidence_refs(forward),
        events=[_as_dict(item) for item in (arm_result.get("events") or [])],
    )


def _arm_module_facts(
    arm_result: Mapping[str, Any],
    *,
    turn_index: int,
) -> list[_RawModuleFact]:
    run_id = _required_string(arm_result, "run_id")
    event_rows = [_as_dict(item) for item in (arm_result.get("events") or [])]
    facts: list[_RawModuleFact] = []

    planner_history = [_as_dict(item) for item in (arm_result.get("planner_history") or [])]
    previous_planner: str | None = None
    final_plan = _mapping_or_empty(arm_result.get("final_plan"))
    final_plan_fingerprint = _optional_string(
        _mapping_or_empty(_last(planner_history)).get("plan_fingerprint")
    ) if final_plan else None
    for index, record in enumerate(planner_history, start=1):
        call_id = _optional_string(record.get("call_id")) or f"planner:{index}"
        status = str(record.get("status") or "failed")
        trace = _mapping_or_empty(record.get("invocation_trace"))
        evidence_refs = _strings(record.get("evidence_ids"))
        plan_fingerprint = _optional_string(record.get("plan_fingerprint"))
        artifact_refs = [f"plan:{plan_fingerprint}"] if plan_fingerprint else []
        reasons = _record_reason_codes(record)
        if status != "completed" and not reasons:
            reasons = ["planner_call_failed"]
        facts.append(
            _raw_fact(
                module_id="planner",
                stage="planner",
                run_id=run_id,
                attempt_id=call_id,
                turn_index=turn_index,
                disposition="accepted" if status == "completed" else "failed",
                selected_for_forward_flow=(
                    status == "completed"
                    and plan_fingerprint is not None
                    and plan_fingerprint == final_plan_fingerprint
                ),
                reason_codes=reasons,
                retry_count=_trace_retry_count(trace),
                reflow_from_attempt_id=previous_planner,
                cost=_trace_cost(trace),
                artifact_refs=artifact_refs,
                evidence_refs=evidence_refs,
                events=event_rows,
                subjects=artifact_refs,
            )
        )
        previous_planner = call_id

    perception_history = [
        _as_dict(item) for item in (arm_result.get("perception_history") or [])
    ]
    previous_perception: str | None = None
    projected_evidence = set(
        _strings(_mapping_or_empty(arm_result.get("evidence_projection")).get("evidence_ids"))
    )
    for index, record in enumerate(perception_history, start=1):
        call_id = _optional_string(record.get("call_id")) or f"perception:{index}"
        status = str(record.get("status") or "failed")
        trace = _mapping_or_empty(record.get("invocation_trace"))
        evidence = _mapping_or_empty(record.get("evidence"))
        evidence_id = _optional_string(evidence.get("evidence_id"))
        evidence_refs = _strings(record.get("evidence_ids_after"))
        if evidence_id and evidence_id not in evidence_refs:
            evidence_refs.append(evidence_id)
        reasons = _record_reason_codes(record)
        if status != "completed" and not reasons:
            reasons = ["perception_call_failed"]
        facts.append(
            _raw_fact(
                module_id="perception",
                stage="perception",
                run_id=run_id,
                attempt_id=call_id,
                turn_index=turn_index,
                disposition="accepted" if status == "completed" else "failed",
                selected_for_forward_flow=bool(
                    status == "completed"
                    and evidence_id
                    and evidence_id in projected_evidence
                ),
                reason_codes=reasons,
                retry_count=_trace_retry_count(trace),
                reflow_from_attempt_id=previous_perception,
                cost=_trace_cost(trace),
                evidence_refs=evidence_refs,
                events=event_rows,
                subjects=evidence_refs,
            )
        )
        previous_perception = call_id

    forward = _mapping_or_empty(arm_result.get("forward_result"))
    if forward:
        facts.extend(
            _forward_module_facts(
                forward,
                turn_index=turn_index,
                parent_run_id=run_id,
                event_rows=event_rows,
            )
        )
    else:
        facts.extend(
            _terminal_failure_fact(
                arm_result,
                turn_index=turn_index,
                run_id=run_id,
                existing=facts,
                event_rows=event_rows,
            )
        )
    return facts


def _forward_module_facts(
    forward: Mapping[str, Any],
    *,
    turn_index: int,
    parent_run_id: str,
    event_rows: Sequence[Mapping[str, Any]],
) -> list[_RawModuleFact]:
    run_id = _optional_string(forward.get("run_id")) or f"{parent_run_id}:forward"
    failure_stage = str(forward.get("failure_stage") or "none")
    failure_codes = _strings(forward.get("failure_codes"))
    facts: list[_RawModuleFact] = []
    admission = _mapping_or_empty(forward.get("admission"))
    if admission:
        admitted = bool(admission.get("would_admit")) and failure_stage != "plan_admission"
        reasons = [
            str(item.get("code"))
            for item in (_mapping_items(admission.get("findings")))
            if item.get("code")
        ]
        if not admitted:
            reasons.extend(failure_codes or ["plan_admission_rejected"])
        facts.append(
            _raw_fact(
                module_id="verifier",
                stage="plan_admission",
                run_id=run_id,
                attempt_id="plan_admission:0",
                turn_index=turn_index,
                disposition="accepted" if admitted else "rejected",
                selected_for_forward_flow=admitted,
                reason_codes=reasons,
                artifact_refs=[
                    f"plan:{admission['plan_fingerprint']}"
                ] if admission.get("plan_fingerprint") else [],
                events=event_rows,
            )
        )

    executed_plan = _mapping_or_empty(forward.get("executed_plan"))
    steps = _mapping_items(executed_plan.get("steps"))
    executor_run_count = int(forward.get("executor_run_count") or 0)
    artifact_refs = _artifact_refs(executed_plan)
    evidence_refs = _evidence_refs(executed_plan)
    if executor_run_count:
        summary = _mapping_or_empty(forward.get("execution_summary"))
        executor_ok = bool(summary.get("completed")) and failure_stage != "executor"
        facts.append(
            _raw_fact(
                module_id="executor",
                stage="executor",
                run_id=run_id,
                attempt_id="execution:0",
                turn_index=turn_index,
                disposition="accepted" if executor_ok else "failed",
                selected_for_forward_flow=executor_ok,
                reason_codes=([] if executor_ok else failure_codes or ["executor_failed"]),
                cost={"tool_calls": float(forward.get("tool_call_count") or 0)},
                artifact_refs=artifact_refs,
                evidence_refs=evidence_refs,
                events=event_rows,
            )
        )
    for step in steps:
        operation = str(step.get("operation") or "")
        if operation.startswith(("perception.", "flexsed.")):
            continue
        step_id = _required_string(step, "id")
        status = str(step.get("status") or "pending")
        disposition: ModuleDisposition
        if status == "completed":
            disposition = "accepted"
        elif status == "skipped":
            disposition = "rejected"
        elif status == "failed":
            disposition = "failed"
        else:
            disposition = "reached"
        reasons = []
        if step.get("error"):
            reasons.append(str(step["error"]))
        if disposition in {"rejected", "failed"} and not reasons:
            reasons.append(f"tool_step_{status}")
        facts.append(
            _raw_fact(
                module_id="tool",
                stage="tool",
                run_id=run_id,
                attempt_id=f"tool:{step_id}",
                turn_index=turn_index,
                disposition=disposition,
                selected_for_forward_flow=status == "completed",
                reason_codes=reasons,
                artifact_refs=_artifact_refs(step),
                evidence_refs=_evidence_refs(step),
                events=event_rows,
                subjects=[step_id],
            )
        )

    audit = _mapping_or_empty(forward.get("artifact_audit"))
    if audit:
        accepted = _strings(audit.get("accepted_artifact_refs"))
        rejected = _strings(audit.get("rejected_artifact_refs"))
        candidate = _strings(audit.get("candidate_artifact_refs"))
        verification_ok = str(audit.get("status")) == "would_finalize"
        findings = [
            str(item.get("code"))
            for item in _mapping_items(audit.get("findings"))
            if item.get("code")
        ]
        facts.append(
            _raw_fact(
                module_id="verifier",
                stage="runtime_verification",
                run_id=run_id,
                attempt_id="runtime_verification:0",
                turn_index=turn_index,
                disposition="accepted" if verification_ok else "rejected",
                selected_for_forward_flow=verification_ok,
                reason_codes=(
                    []
                    if verification_ok
                    else findings or ["runtime_verification_rejected"]
                ),
                artifact_refs=[*accepted, *rejected, *candidate],
                events=event_rows,
            )
        )

    commit = _mapping_or_empty(forward.get("artifact_store_commit"))
    if commit:
        committed = str(commit.get("status")) == "committed"
        commit_refs = _artifact_refs(commit)
        reasons = _strings(commit.get("failure_codes"))
        facts.append(
            _raw_fact(
                module_id="artifact_store",
                stage="artifact_store",
                run_id=run_id,
                attempt_id=_optional_string(commit.get("attempt_id")) or "artifact_store:0",
                turn_index=turn_index,
                disposition="accepted" if committed else "rejected",
                selected_for_forward_flow=committed,
                reason_codes=([] if committed else reasons or ["artifact_store_blocked"]),
                artifact_refs=commit_refs,
                events=event_rows,
            )
        )

    manifest = _mapping_or_empty(forward.get("final_delivery_manifest_shadow"))
    if manifest:
        ready = str(manifest.get("status")) == "ready"
        facts.append(
            _raw_fact(
                module_id="final_assembler",
                stage="final_assembler_shadow",
                run_id=run_id,
                attempt_id="final_assembler:shadow",
                turn_index=turn_index,
                disposition="accepted" if ready else "reached",
                selected_for_forward_flow=False,
                reason_codes=[] if ready else ["final_assembler_shadow_not_ready"],
                artifact_refs=_artifact_refs(manifest),
                events=event_rows,
            )
        )
    return facts


def _terminal_failure_fact(
    arm_result: Mapping[str, Any],
    *,
    turn_index: int,
    run_id: str,
    existing: Sequence[_RawModuleFact],
    event_rows: Sequence[Mapping[str, Any]],
) -> list[_RawModuleFact]:
    stage = str(arm_result.get("failure_stage") or "")
    module_by_stage: dict[str, ModuleId] = {
        "planner": "planner",
        "perception": "perception",
        "workflow_protocol": "verifier",
        "forward_execution": "executor",
        "configuration": "executor",
    }
    module_id = module_by_stage.get(stage)
    if module_id is None or any(
        item.module_id == module_id and item.disposition in {"rejected", "failed"}
        for item in existing
    ):
        return []
    reasons = _strings(arm_result.get("failure_codes")) or [
        f"{stage or 'workflow'}_failed"
    ]
    return [
        _raw_fact(
            module_id=module_id,
            stage=stage or module_id,
            run_id=run_id,
            attempt_id=f"{module_id}:terminal",
            turn_index=turn_index,
            disposition="rejected" if stage == "workflow_protocol" else "failed",
            reason_codes=reasons,
            events=event_rows,
        )
    ]


def _raw_fact(
    *,
    module_id: ModuleId,
    stage: str,
    run_id: str,
    attempt_id: str,
    turn_index: int,
    disposition: ModuleDisposition,
    selected_for_forward_flow: bool = False,
    reason_codes: Sequence[str] = (),
    retry_count: int = 0,
    reflow_from_attempt_id: str | None = None,
    cost: Mapping[str, float] | None = None,
    artifact_refs: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    events: Sequence[Mapping[str, Any]] = (),
    subjects: Sequence[str] = (),
) -> _RawModuleFact:
    matching_events = _matching_events(
        events,
        module_id=module_id,
        attempt_id=attempt_id,
        subjects=subjects,
    )
    event_cost = _event_cost(matching_events)
    combined_cost = dict(cost or {})
    for key, value in event_cost.items():
        combined_cost[key] = combined_cost.get(key, 0.0) + value
    latency = _event_latency(matching_events)
    return _RawModuleFact(
        module_id=module_id,
        stage=stage,
        run_id=run_id,
        attempt_id=attempt_id,
        turn_index=turn_index,
        disposition=disposition,
        selected_for_forward_flow=selected_for_forward_flow,
        reason_codes=tuple(_unique_strings(reason_codes)),
        retry_count=retry_count,
        reflow_from_attempt_id=reflow_from_attempt_id,
        latency_seconds=latency,
        cost=tuple(sorted((key, float(value)) for key, value in combined_cost.items())),
        artifact_refs=tuple(_unique_strings(artifact_refs)),
        evidence_refs=tuple(_unique_strings(evidence_refs)),
        source_event_ids=tuple(
            str(item["event_id"]) for item in matching_events if item.get("event_id")
        ),
    )


def _module_observation(
    raw: _RawModuleFact,
    assessment: ModuleAssessmentRecord,
) -> ModuleExecutionObservation:
    attempt_id = f"turn:{raw.turn_index}:{raw.attempt_id}"
    parent_id = (
        f"turn:{raw.turn_index}:{raw.reflow_from_attempt_id}"
        if raw.reflow_from_attempt_id
        else None
    )
    evidence_refs = _unique_strings([*raw.artifact_refs, *raw.evidence_refs])
    attempt = AttemptFact(
        run_id=raw.run_id,
        attempt_id=attempt_id,
        attempt_kind=("feedback" if parent_id or raw.retry_count else "runtime"),
        attempt_group_id=f"{assessment.assessment_record_id}:{raw.turn_index}",
        parent_attempt_id=parent_id,
        selected_for_forward_flow=raw.selected_for_forward_flow,
        attempt_started=True,
        execution_state={
            "accepted": "succeeded",
            "reached": "not_attempted",
            "rejected": "failed",
            "failed": "failed",
        }[raw.disposition],
        cost=dict(raw.cost),
        evidence_refs=evidence_refs,
        source_event_ids=list(raw.source_event_ids),
    )
    observation_id = stable_id(
        "module_observation",
        {
            "assessment": assessment.assessment_record_id,
            "turn": raw.turn_index,
            "module": raw.module_id,
            "stage": raw.stage,
            "attempt": attempt_id,
        },
    )
    return ModuleExecutionObservation(
        observation_id=observation_id,
        assessment_record_id=assessment.assessment_record_id,
        module_id=raw.module_id,
        boundary_id=assessment.boundary.boundary_id,
        turn_index=raw.turn_index,
        stage=raw.stage,
        disposition=raw.disposition,
        reason_codes=list(raw.reason_codes),
        retry_count=raw.retry_count,
        reflow_from_attempt_id=parent_id,
        latency_seconds=raw.latency_seconds,
        artifact_refs=list(raw.artifact_refs),
        evidence_refs=list(raw.evidence_refs),
        attempt=attempt,
    )


def _module_failure_facts(
    observation: ModuleExecutionObservation,
    reason_codes: Sequence[str],
) -> list[FailureUnknownFact]:
    reasons = _unique_strings(reason_codes) or [
        f"{observation.module_id}_{observation.disposition}"
    ]
    return [
        FailureUnknownFact(
            fact_id=stable_id(
                "module_failure",
                {"observation": observation.observation_id, "reason": reason},
            ),
            run_id=observation.attempt.run_id,
            assessment_record_id=observation.assessment_record_id,
            attempt_id=observation.attempt.attempt_id,
            category="blocked" if observation.disposition == "rejected" else "failure",
            reason_code=reason,
            failure_stage=observation.stage,
            failure_owner=observation.module_id,
            evidence_refs=observation.attempt.evidence_refs,
            source_event_id=(
                observation.attempt.source_event_ids[-1]
                if observation.attempt.source_event_ids
                else None
            ),
        )
        for reason in reasons
    ]


def _import_evaluator_observations(
    evaluator_rows: Sequence[TrustedEvaluatorEvidence],
    *,
    identity: _ImportIdentity,
    assessment_by_id: Mapping[str, ModuleAssessmentRecord],
    module_observations: Sequence[ModuleExecutionObservation],
) -> list[MetricObservation]:
    by_assessment: dict[str, list[ModuleExecutionObservation]] = {}
    for item in module_observations:
        by_assessment.setdefault(item.assessment_record_id, []).append(item)
    imported: list[MetricObservation] = []
    for envelope in evaluator_rows:
        observation = envelope.observation
        context = observation.context
        assessment = assessment_by_id.get(context.assessment_record_id)
        if assessment is None:
            raise ValueError("evaluator observation references an unknown assessment")
        if envelope.runtime_projection_fingerprint != identity.projection_fingerprint:
            raise ValueError("evaluator runtime projection fingerprint mismatch")
        if context.case_fingerprint != identity.projection_fingerprint:
            raise ValueError("evaluator observation context fingerprint mismatch")
        if context.sidecar_record_key != identity.sidecar_record_key:
            raise ValueError("evaluator observation sidecar identity mismatch")
        if context.boundary_id != assessment.boundary.boundary_id:
            raise ValueError("evaluator observation boundary mismatch")
        if context.objective_instance_id not in assessment.objective_refs:
            raise ValueError("evaluator observation objective is outside the assessment")
        if context.arm_id != identity.assigned_arm:
            raise ValueError("evaluator observation assigned arm mismatch")
        linked = by_assessment.get(assessment.assessment_record_id, [])
        known_output_refs = {
            ref
            for item in linked
            for ref in [*item.artifact_refs, *item.evidence_refs]
        }
        if envelope.artifact_ref_id not in known_output_refs:
            raise ValueError("evaluator observation artifact is not an imported module output")
        linked_attempt_ids = {item.attempt.attempt_id for item in linked}
        if context.attempt_id not in linked_attempt_ids:
            raise ValueError("evaluator observation attempt does not match the module output")
        if not envelope.trusted:
            continue
        imported.append(observation.model_copy(deep=True))
    return imported


def _matching_events(
    events: Sequence[Mapping[str, Any]],
    *,
    module_id: ModuleId,
    attempt_id: str,
    subjects: Sequence[str],
) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    subject_set = set(subjects)
    for event in events:
        if _event_module(event) != module_id:
            continue
        event_attempt = _optional_string(event.get("attempt_id"))
        event_subjects = set(_strings(event.get("subject_refs")))
        if event_attempt == attempt_id or (
            subject_set and subject_set.intersection(event_subjects)
        ):
            matches.append(event)
    return sorted(matches, key=lambda item: int(item.get("sequence") or 0))


def _event_module(event: Mapping[str, Any]) -> ModuleId | None:
    stage = str(event.get("stage") or "")
    kind = str(event.get("kind") or "")
    if stage == "planner":
        return "planner"
    if stage == "perception":
        return "perception"
    if stage == "executor":
        return "tool" if "tool" in kind else "executor"
    if stage == "verifier":
        return "verifier"
    if stage == "repair":
        return "feedback_controller"
    if stage == "evaluator":
        return "evaluator"
    if stage == "finalization":
        if "artifact_store" in kind:
            return "artifact_store"
        if "delivery_manifest" in kind or "final_assembler" in kind:
            return "final_assembler"
        return "verifier"
    return None


def _event_latency(events: Sequence[Mapping[str, Any]]) -> float | None:
    timestamps: list[datetime] = []
    for event in events:
        raw = event.get("timestamp")
        if isinstance(raw, datetime):
            timestamps.append(raw)
        elif isinstance(raw, str):
            try:
                timestamps.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                continue
    if len(timestamps) < 2:
        return None
    return max(0.0, (max(timestamps) - min(timestamps)).total_seconds())


def _event_cost(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for event in events:
        payload = _mapping_or_empty(event.get("payload"))
        raw_cost = _mapping_or_empty(payload.get("cost"))
        for key, value in raw_cost.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[str(key)] = total.get(str(key), 0.0) + float(value)
    return total


def _trace_retry_count(trace: Mapping[str, Any]) -> int:
    return int(trace.get("retry_count") or 0) + int(trace.get("logical_retry_count") or 0)


def _trace_cost(trace: Mapping[str, Any]) -> dict[str, float]:
    if not trace:
        return {}
    cost = {
        "logical_provider_calls": float(trace.get("logical_provider_call_count") or 0),
        "provider_calls": float(trace.get("provider_call_count") or 0),
        "provider_retries": float(trace.get("retry_count") or 0),
        "logical_retries": float(trace.get("logical_retry_count") or 0),
        "json_repairs": float(trace.get("json_repair_count") or 0),
        "schema_repairs": float(trace.get("schema_repair_count") or 0),
    }
    for invocation in _mapping_items(trace.get("provider_invocations")):
        for key, value in _mapping_or_empty(invocation.get("metrics")).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_key = f"provider_metric.{key}"
                cost[metric_key] = cost.get(metric_key, 0.0) + float(value)
    return {key: value for key, value in cost.items() if value != 0}


def _record_reason_codes(record: Mapping[str, Any]) -> list[str]:
    reasons = _strings(record.get("reason_codes"))
    if record.get("error_type"):
        reasons.append(str(record["error_type"]))
    trace = _mapping_or_empty(record.get("invocation_trace"))
    if trace.get("fallback_used"):
        reasons.append("component_fallback_used")
    return _unique_strings(reasons)


def _artifact_refs(value: Any) -> list[str]:
    singular = {
        "artifact_id",
        "artifact_ref_id",
        "output_artifact_id",
        "final_artifact_ref_id",
        "legacy_final_artifact_ref_id",
    }
    plural = {
        "artifact_refs",
        "accepted_artifact_refs",
        "candidate_artifact_refs",
        "rejected_artifact_refs",
        "required_artifact_refs",
        "parent_artifact_ref_ids",
    }
    return _collect_named_refs(value, singular=singular, plural=plural)


def _evidence_refs(value: Any) -> list[str]:
    return _collect_named_refs(
        value,
        singular={"evidence_id", "evidence_ref_id"},
        plural={"evidence_ids", "evidence_refs", "verification_refs"},
    )


def _collect_named_refs(
    value: Any,
    *,
    singular: set[str],
    plural: set[str],
) -> list[str]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in singular and item not in (None, ""):
                refs.append(str(item))
            elif key in plural:
                refs.extend(_strings(item))
            else:
                refs.extend(_collect_named_refs(item, singular=singular, plural=plural))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            refs.extend(_collect_named_refs(item, singular=singular, plural=plural))
    return _unique_strings(refs)


def _scenario_run_id(scenario: Mapping[str, Any]) -> str:
    turns = scenario.get("turn_results") or []
    if turns:
        first = _as_dict(turns[0])
        result = _mapping_or_empty(first.get("result"))
        if result.get("run_id"):
            return str(result["run_id"])
    return stable_id(
        "benchmark_run",
        {
            "projection": scenario.get("scenario_fingerprint"),
            "arm": scenario.get("assigned_arm"),
        },
    )


def _reject_duplicate_ids(values: Sequence[str], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} id")


def _required_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(item)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if item in (None, ""):
        raise ValueError(f"{key} is required")
    return str(item)


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported benchmark evidence type: {type(value)!r}")


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _unique_strings(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in values if item not in (None, "")))


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _last(values: Sequence[Any]) -> Any:
    return values[-1] if values else None


__all__ = [
    "BenchmarkCapacityImport",
    "ModuleExecutionObservation",
    "TrustedEvaluatorEvidence",
    "import_benchmark_evidence",
]
