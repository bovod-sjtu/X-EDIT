"""Audit-only admission checks for unmodified Planner output."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

from pydantic import ValidationError

from x_editor.capabilities.schemas import (
    OperationCapabilityCard,
    validate_operation_arguments,
)
from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.core.step_references import (
    is_step_output_reference,
    matches_step_output_annotation,
)
from x_editor.runtime.audio_usage import WorkflowAudioUsageBinding
from x_editor.runtime.operation_normalizers import (
    OperationBoundaryError,
    edit_data_pipeline_label_requirements,
    validate_planner_edit_data_pipeline_contract,
)
from x_editor.workflow_control.schemas import (
    ObjectiveCoverageAudit,
    PlanAdmissionFinding,
    PlanAdmissionRepairCandidate,
    PlanAdmissionResult,
)

_INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "input_audio": (
        "input_audio",
        "input_audio_path",
        "source_audio",
        "source_audio_path",
        "audio",
        "audio_path",
    ),
    "reference_audio": (
        "reference_audio",
        "reference_audio_path",
        "reference_audio_paths",
        "reference_audio_items",
    ),
    "events": ("events", "event", "event_labels", "target_event", "target_events"),
    "source_text": ("source_text", "transcript", "aligned_text"),
    "spans": ("spans", "selected_spans", "target_spans", "segments"),
}
_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "target_language": ("target_language", "target_lang", "language"),
}
_LOCAL_SPAN_OPERATIONS = {
    "artifact_ops.extract_spans",
    "artifact_ops.remove_spans",
    "artifact_ops.mute_spans",
    "artifact_ops.beep_spans",
    "artifact_ops.speaker_scoped_edit",
}


class PlanAdmissionGate:
    """Inspect a plan before execution without mutating or blocking it."""

    def audit(
        self,
        plan: AudioEditPlan,
        *,
        available_operations: list[dict[str, Any]] | list[str],
        context: dict[str, Any] | None = None,
    ) -> PlanAdmissionResult:
        context = dict(context or {})
        findings: list[PlanAdmissionFinding] = []
        deterministic_repairs: list[PlanAdmissionRepairCandidate] = []
        requested_repairs: list[PlanAdmissionRepairCandidate] = []
        operation_specs = _operation_specs(available_operations)
        step_ids = [step.id for step in plan.steps]
        step_id_set = set(step_ids)
        step_by_id = {step.id: step for step in plan.steps}

        if not plan.steps:
            findings.append(
                PlanAdmissionFinding(
                    code="plan_has_no_steps",
                    severity="warning",
                    plain_summary=(
                        "The plan contains no executable steps, so objective coverage "
                        "cannot be demonstrated."
                    ),
                    owner="planner",
                    field="steps",
                )
            )

        for duplicate_id, count in sorted(Counter(step_ids).items()):
            if count < 2:
                continue
            findings.append(
                PlanAdmissionFinding(
                    code="duplicate_step_id",
                    severity="blocking",
                    plain_summary=(
                        f"Step id {duplicate_id!r} is used {count} times, so "
                        "dependencies are ambiguous."
                    ),
                    owner="planner",
                    step_id=duplicate_id,
                    field="steps[].id",
                    details={"count": count},
                )
            )
            requested_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="planner",
                    code="assign_unique_step_ids",
                    plain_summary=(
                        "Assign a unique id to every plan step and update its "
                        "dependency references."
                    ),
                    step_id=duplicate_id,
                )
            )

        for step in plan.steps:
            self._audit_step(
                step,
                plan=plan,
                context=context,
                operation_specs=operation_specs,
                step_id_set=step_id_set,
                step_by_id=step_by_id,
                findings=findings,
                deterministic_repairs=deterministic_repairs,
                requested_repairs=requested_repairs,
            )

        _audit_audio_usage_bindings(
            plan,
            context=context,
            step_by_id=step_by_id,
            findings=findings,
            requested_repairs=requested_repairs,
        )

        for cycle in _dependency_cycles(plan.steps):
            findings.append(
                PlanAdmissionFinding(
                    code="dependency_cycle",
                    severity="blocking",
                    plain_summary=(
                        "The plan contains a dependency cycle and cannot be "
                        "scheduled safely."
                    ),
                    owner="planner",
                    field="steps[].depends_on",
                    details={"cycle": cycle},
                )
            )
            requested_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="planner",
                    code="break_dependency_cycle",
                    plain_summary=(
                        "Rebuild the affected dependencies as an acyclic execution "
                        "order."
                    ),
                    details={"cycle": cycle},
                )
            )

        for missing_input in plan.missing_inputs:
            findings.append(
                PlanAdmissionFinding(
                    code="planner_reported_missing_input",
                    severity="blocking",
                    plain_summary=(
                        "The Planner explicitly reports that required input "
                        f"{missing_input!r} is missing."
                    ),
                    owner="caller",
                    field="missing_inputs",
                    details={"missing_input": missing_input},
                )
            )
            requested_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="caller",
                    code="provide_missing_input",
                    plain_summary=(
                        f"Provide or explicitly resolve missing input {missing_input!r} "
                        "before execution."
                    ),
                    details={"missing_input": missing_input},
                )
            )

        objective_coverage = _audit_objective_coverage(
            plan,
            step_id_set=step_id_set,
            findings=findings,
            requested_repairs=requested_repairs,
        )
        finding_counts = dict(Counter(finding.severity for finding in findings))
        original_plan_valid = finding_counts.get("blocking", 0) == 0
        return PlanAdmissionResult(
            plan_fingerprint=_plan_fingerprint(plan),
            original_plan_valid=original_plan_valid,
            would_admit=original_plan_valid,
            findings=findings,
            objective_coverage=objective_coverage,
            deterministic_repairs=deterministic_repairs,
            requested_repairs=requested_repairs,
            finding_counts=finding_counts,
            checked_step_count=len(plan.steps),
            available_operation_count=len(operation_specs),
        )


    def _audit_step(
        self,
        step: AudioEditStep,
        *,
        plan: AudioEditPlan,
        context: dict[str, Any],
        operation_specs: dict[str, dict[str, Any]],
        step_id_set: set[str],
        step_by_id: dict[str, AudioEditStep],
        findings: list[PlanAdmissionFinding],
        deterministic_repairs: list[PlanAdmissionRepairCandidate],
        requested_repairs: list[PlanAdmissionRepairCandidate],
    ) -> None:
        operation_spec = operation_specs.get(step.operation)
        if operation_spec is None:
            findings.append(
                PlanAdmissionFinding(
                    code="unknown_operation",
                    severity="blocking",
                    plain_summary=(
                        f"Step {step.id!r} selects operation {step.operation!r}, "
                        "which is not available to this Executor."
                    ),
                    owner="planner",
                    step_id=step.id,
                    operation=step.operation,
                    field="operation",
                )
            )
            requested_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="planner",
                    code="select_available_operation",
                    plain_summary=(
                        "Select an available operation without weakening the "
                        "requested edit semantics."
                    ),
                    step_id=step.id,
                    details={"unknown_operation": step.operation},
                )
            )
        else:
            _audit_operation_capability_contract(
                step,
                operation_spec=operation_spec,
                context=context,
                findings=findings,
                requested_repairs=requested_repairs,
            )
            for required_input in operation_spec.get("required_inputs") or []:
                name = str(required_input)
                if _required_input_is_explicit(
                    step,
                    name,
                    plan=plan,
                    context=context,
                ):
                    continue
                findings.append(
                    PlanAdmissionFinding(
                        code="required_input_not_explicit",
                        severity="warning",
                        plain_summary=(
                            f"Step {step.id!r} does not explicitly bind required "
                            f"input {name!r}; compatibility resolution may still "
                            "provide it at runtime."
                        ),
                        owner="planner",
                        step_id=step.id,
                        operation=step.operation,
                        field=f"required_inputs.{name}",
                    )
                )
            for required_param in operation_spec.get("required_params") or []:
                name = str(required_param)
                if _required_param_is_explicit(step, name):
                    continue
                findings.append(
                    PlanAdmissionFinding(
                        code="required_param_not_explicit",
                        severity="warning",
                        plain_summary=(
                            f"Step {step.id!r} does not explicitly bind required "
                            f"parameter {name!r}; Executor repair may need to bind it "
                            "from grounded request evidence."
                        ),
                        owner="planner",
                        step_id=step.id,
                        operation=step.operation,
                        field=f"required_params.{name}",
                    )
                )
                requested_repairs.append(
                    PlanAdmissionRepairCandidate(
                        repair_type="planner",
                        code="bind_required_param",
                        plain_summary=(
                            f"Bind required parameter {name!r} from explicit user, "
                            "step, or perception evidence."
                        ),
                        step_id=step.id,
                        details={"required_param": name},
                    )
                )
                requested_repairs.append(
                    PlanAdmissionRepairCandidate(
                        repair_type="evidence",
                        code="bind_required_input",
                        plain_summary=(
                            f"Bind required input {name!r} to a stable plan, "
                            "artifact, or context reference."
                        ),
                        step_id=step.id,
                        details={"required_input": name},
                    )
                )

        duplicate_dependencies = [
            dependency
            for dependency, count in Counter(step.depends_on).items()
            if count > 1
        ]
        if duplicate_dependencies:
            findings.append(
                PlanAdmissionFinding(
                    code="duplicate_dependency",
                    severity="warning",
                    plain_summary=(
                        f"Step {step.id!r} repeats dependency ids that can be "
                        "deduplicated without changing order."
                    ),
                    owner="planner",
                    step_id=step.id,
                    operation=step.operation,
                    field="depends_on",
                    details={"duplicate_dependencies": sorted(duplicate_dependencies)},
                )
            )
            deterministic_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="deterministic",
                    code="deduplicate_dependencies",
                    plain_summary=(
                        "Deduplicate repeated dependency ids while preserving their "
                        "first occurrence."
                    ),
                    step_id=step.id,
                    details={"duplicate_dependencies": sorted(duplicate_dependencies)},
                )
            )

        _audit_explicit_local_span_boundaries(
            step,
            findings=findings,
            requested_repairs=requested_repairs,
        )

        for dependency in step.depends_on:
            if dependency == step.id:
                findings.append(
                    PlanAdmissionFinding(
                        code="self_dependency",
                        severity="blocking",
                        plain_summary=(
                            f"Step {step.id!r} depends on itself and can never "
                            "become ready."
                        ),
                        owner="planner",
                        step_id=step.id,
                        operation=step.operation,
                        field="depends_on",
                    )
                )
            elif dependency not in step_id_set:
                findings.append(
                    PlanAdmissionFinding(
                        code="missing_dependency",
                        severity="blocking",
                        plain_summary=(
                            f"Step {step.id!r} depends on unknown step {dependency!r}."
                        ),
                        owner="planner",
                        step_id=step.id,
                        operation=step.operation,
                        field="depends_on",
                        details={"missing_dependency": dependency},
                    )
                )
                requested_repairs.append(
                    PlanAdmissionRepairCandidate(
                        repair_type="planner",
                        code="repair_missing_dependency",
                        plain_summary=(
                            "Replace the missing dependency with a valid producer "
                            "step or remove the invalid dependency."
                        ),
                        step_id=step.id,
                        details={"missing_dependency": dependency},
                    )
                )
            elif step.required and not step_by_id[dependency].required:
                findings.append(
                    PlanAdmissionFinding(
                        code="required_step_depends_on_optional_step",
                        severity="warning",
                        plain_summary=(
                            f"Required step {step.id!r} depends on optional step "
                            f"{dependency!r}; failure semantics need an explicit "
                            "policy."
                        ),
                        owner="planner",
                        step_id=step.id,
                        operation=step.operation,
                        field="depends_on",
                        details={"optional_dependency": dependency},
                    )
                )


def _audit_audio_usage_bindings(
    plan: AudioEditPlan,
    *,
    context: dict[str, Any],
    step_by_id: dict[str, AudioEditStep],
    findings: list[PlanAdmissionFinding],
    requested_repairs: list[PlanAdmissionRepairCandidate],
) -> None:
    raw_bindings = plan.metadata.get("workflow_audio_usage_bindings")
    if raw_bindings is None:
        return
    if not isinstance(raw_bindings, list):
        _append_audio_usage_finding(
            code="audio_usage_bindings_invalid",
            summary="Planner audio usage bindings must be an array.",
            field="metadata.workflow_audio_usage_bindings",
            findings=findings,
            requested_repairs=requested_repairs,
        )
        return

    known_refs = _known_runtime_audio_refs(context)
    path_sources = context.get("workflow_evidence_path_sources")
    path_sources = path_sources if isinstance(path_sources, dict) else {}
    path_audio_refs = context.get("workflow_evidence_path_audio_refs")
    path_audio_refs = path_audio_refs if isinstance(path_audio_refs, dict) else {}
    seen_arguments: set[tuple[str, str]] = set()
    for index, raw_binding in enumerate(raw_bindings):
        field = f"metadata.workflow_audio_usage_bindings.{index}"
        try:
            binding = WorkflowAudioUsageBinding.model_validate(raw_binding)
        except ValidationError as exc:
            _append_audio_usage_finding(
                code="audio_usage_binding_invalid",
                summary="Planner emitted an invalid audio usage binding.",
                field=field,
                details={"error": str(exc)},
                findings=findings,
                requested_repairs=requested_repairs,
            )
            continue
        key = (binding.plan_step_id, binding.argument_path)
        if key in seen_arguments:
            _append_audio_usage_finding(
                code="audio_usage_binding_duplicate",
                summary="One plan argument has more than one audio usage binding.",
                field=field,
                step_id=binding.plan_step_id,
                findings=findings,
                requested_repairs=requested_repairs,
            )
            continue
        seen_arguments.add(key)
        step = step_by_id.get(binding.plan_step_id)
        if step is None:
            _append_audio_usage_finding(
                code="audio_usage_step_unknown",
                summary="Audio usage binding names an unknown plan step.",
                field=field,
                step_id=binding.plan_step_id,
                findings=findings,
                requested_repairs=requested_repairs,
            )
            continue
        try:
            argument_value = _resolve_step_argument(step, binding.argument_path)
        except (KeyError, IndexError, TypeError, ValueError):
            argument_value = None
        if _matches_declared_dependency_output(
            argument_value,
            step=step,
            binding=binding,
            plan=plan,
            known_refs=known_refs,
        ):
            findings.append(
                PlanAdmissionFinding(
                    code="audio_usage_dependency_binding_redundant",
                    severity="warning",
                    plain_summary=(
                        "The audio input is already authorized by an exact from_step "
                        "dependency; its redundant external-asset binding was ignored."
                    ),
                    owner="planner",
                    step_id=step.id,
                    operation=step.operation,
                    field=binding.argument_path,
                    details={
                        "bound_audio_ref": binding.audio_ref,
                        "from_step": argument_value["from_step"],
                        "original_binding": binding.model_dump(mode="json"),
                        "authoritative_dependency": dict(argument_value),
                        "ignored_reason": "redundant_graph_annotation_matches_typed_dependency",
                    },
                )
            )
            continue
        if argument_value != binding.audio_ref:
            _append_audio_usage_finding(
                code="audio_usage_argument_mismatch",
                summary=(
                    "Audio usage binding does not match the exact plan argument "
                    "that it claims to authorize."
                ),
                field=binding.argument_path,
                step_id=step.id,
                operation=step.operation,
                details={"bound_audio_ref": binding.audio_ref},
                findings=findings,
                requested_repairs=requested_repairs,
            )
        if known_refs and binding.audio_ref not in known_refs:
            _append_audio_usage_finding(
                code="audio_usage_ref_unbound",
                summary="Audio usage binding references an unavailable runtime asset.",
                field=binding.argument_path,
                step_id=step.id,
                operation=step.operation,
                details={"audio_ref": binding.audio_ref},
                findings=findings,
                requested_repairs=requested_repairs,
            )
        if binding.evidence_requirement == "existence_decodability":
            continue
        for reference in binding.evidence:
            source_ids = path_sources.get(reference.evidence_path)
            if not isinstance(source_ids, list) or reference.evidence_id not in source_ids:
                _append_audio_usage_finding(
                    code="audio_usage_evidence_unknown",
                    summary="Audio usage binding cites a reviewed-evidence path that does not exist.",
                    field=reference.evidence_path,
                    step_id=step.id,
                    operation=step.operation,
                    findings=findings,
                    requested_repairs=requested_repairs,
                )
                continue
            if path_audio_refs.get(reference.evidence_path) != binding.audio_ref:
                _append_audio_usage_finding(
                    code="audio_usage_evidence_wrong_audio",
                    summary="Reviewed evidence belongs to a different audio asset.",
                    field=reference.evidence_path,
                    step_id=step.id,
                    operation=step.operation,
                    details={"audio_ref": binding.audio_ref},
                    findings=findings,
                    requested_repairs=requested_repairs,
                )


def _append_audio_usage_finding(
    *,
    code: str,
    summary: str,
    field: str,
    findings: list[PlanAdmissionFinding],
    requested_repairs: list[PlanAdmissionRepairCandidate],
    step_id: str | None = None,
    operation: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    findings.append(
        PlanAdmissionFinding(
            code=code,
            severity="blocking",
            plain_summary=summary,
            owner="planner",
            step_id=step_id,
            operation=operation,
            field=field,
            details=details or {},
        )
    )
    requested_repairs.append(
        PlanAdmissionRepairCandidate(
            repair_type="planner",
            code="repair_audio_usage_binding",
            plain_summary=(
                "Bind the exact plan argument to an available audio ref and, when "
                "semantic grounding is required, cite reviewed facts from that audio."
            ),
            step_id=step_id,
            details={"finding_code": code, **(details or {})},
        )
    )


def _known_runtime_audio_refs(context: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in (
        "workflow_source_audio_refs",
        "workflow_reference_audio_refs",
        "input_audios",
        "reference_audio_paths",
    ):
        value = context.get(key)
        if isinstance(value, list):
            refs.update(str(item) for item in value if str(item))
    for raw_binding in context.get("_runtime_audio_bindings") or []:
        audio_ref = (
            getattr(raw_binding, "audio_ref", None)
            if not isinstance(raw_binding, dict)
            else raw_binding.get("audio_ref")
        )
        if audio_ref:
            refs.add(str(audio_ref))
    return refs


def _resolve_step_argument(step: AudioEditStep, path: str) -> Any:
    normalized = path.replace("[", ".").replace("]", "")
    parts = normalized.split(".")
    if not parts or parts[0] not in {"inputs", "params"}:
        raise ValueError("unsupported audio argument path")
    value: Any = step.inputs if parts[0] == "inputs" else step.params
    for part in parts[1:]:
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        else:
            raise TypeError("audio argument path does not resolve")
    return value


def _matches_declared_dependency_output(
    argument_value: Any,
    *,
    step: AudioEditStep,
    binding: WorkflowAudioUsageBinding,
    plan: AudioEditPlan,
    known_refs: set[str],
) -> bool:
    if not is_step_output_reference(argument_value):
        return False
    from_step = argument_value.get("from_step")
    step_ids = [item.id for item in plan.steps]
    if (
        from_step not in step.depends_on
        or step_ids.count(from_step) != 1
        or step_ids.count(step.id) != 1
        or step_ids.index(from_step) >= step_ids.index(step.id)
    ):
        return False
    # Even a graph-shaped spelling may identify a real external asset. An
    # explicit collision is conflicting authority, never redundant annotation.
    if binding.audio_ref in known_refs:
        return False
    if not matches_step_output_annotation(binding.audio_ref, argument_value):
        return False
    return binding.evidence_requirement == "existence_decodability" and not binding.evidence

def _operation_specs(
    available_operations: list[dict[str, Any]] | list[str],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for item in available_operations:
        if isinstance(item, str):
            operation = item.strip()
            spec: dict[str, Any] = {"operation": operation}
        elif isinstance(item, dict):
            operation = str(item.get("operation") or "").strip()
            spec = dict(item)
        else:
            continue
        if operation:
            specs[operation] = spec
    return specs


def _audit_operation_capability_contract(
    step: AudioEditStep,
    *,
    operation_spec: dict[str, Any],
    context: dict[str, Any],
    findings: list[PlanAdmissionFinding],
    requested_repairs: list[PlanAdmissionRepairCandidate],
) -> None:
    raw_card = operation_spec.get("operation_capability")
    if raw_card is None:
        return
    try:
        card = OperationCapabilityCard.model_validate(raw_card)
    except ValidationError as exc:
        findings.append(
            PlanAdmissionFinding(
                code="invalid_operation_capability_contract",
                severity="blocking",
                plain_summary=(
                    f"The runtime capability contract for {step.operation!r} is invalid."
                ),
                owner="runtime",
                step_id=step.id,
                operation=step.operation,
                field="operation_capability",
                details={"error": str(exc)},
            )
        )
        return
    frozen_by_operation = (
        context.get("workflow_frozen_edit_operation_params") or {}
    )
    frozen_values = {
        f"params.{key}": value
        for key, value in (
            frozen_by_operation.get(step.operation, {})
            if isinstance(frozen_by_operation, dict)
            else {}
        ).items()
    }
    violations = validate_operation_arguments(
        card,
        inputs=step.inputs,
        params=step.params,
        framework_values=frozen_values,
        allow_framework_owned_without_value=True,
    )
    for violation in violations:
        findings.append(
            PlanAdmissionFinding(
                code=violation.code,
                severity="blocking",
                plain_summary=violation.message,
                owner="planner",
                step_id=step.id,
                operation=step.operation,
                field=violation.path,
                details=violation.details,
            )
        )
        requested_repairs.append(
            PlanAdmissionRepairCandidate(
                repair_type="planner",
                code="select_legal_operation_parameter",
                plain_summary=(
                    "Choose this argument from the selected operation's declared "
                    "legal parameter domain, or omit it when the framework owns it."
                ),
                step_id=step.id,
                details={
                    "operation": step.operation,
                    "violation_code": violation.code,
                    "field": violation.path,
                },
            )
        )

    if step.operation.startswith("edit_data_pipeline."):
        try:
            validate_planner_edit_data_pipeline_contract(
                operation=step.operation,
                params=dict(step.params),
            )
        except OperationBoundaryError as exc:
            reason = str(exc)
            findings.append(
                PlanAdmissionFinding(
                    code="operation_parameter_format_invalid",
                    severity="blocking",
                    plain_summary=reason,
                    owner="planner",
                    step_id=step.id,
                    operation=step.operation,
                    field="params.instruction_st",
                    details={
                        "legal_value_requirements": edit_data_pipeline_label_requirements(
                            step.operation
                        )
                    },
                )
            )
            requested_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="planner",
                    code="author_legal_tool_value_format",
                    plain_summary=(
                        "Regenerate only this step using the selected capability "
                        "card's exact tagged-value grammar; do not put natural-language "
                        "prose in a service DSL field."
                    ),
                    step_id=step.id,
                    details={
                        "operation": step.operation,
                        "field": "params.instruction_st",
                        "reason": reason,
                    },
                )
            )


def _audit_explicit_local_span_boundaries(
    step: AudioEditStep,
    *,
    findings: list[PlanAdmissionFinding],
    requested_repairs: list[PlanAdmissionRepairCandidate],
) -> None:
    if step.operation not in _LOCAL_SPAN_OPERATIONS:
        return
    container_name: str | None = None
    spans: Any = None
    for candidate in ("selected_spans", "spans"):
        if candidate in step.params:
            container_name = f"params.{candidate}"
            spans = step.params[candidate]
            break
        if candidate in step.inputs:
            container_name = f"inputs.{candidate}"
            spans = step.inputs[candidate]
            break
    if spans is None:
        return
    errors: list[dict[str, Any]] = []
    if not isinstance(spans, list) or not spans:
        errors.append({"field": container_name, "reason": "must_be_non_empty_array"})
    else:
        for index, span in enumerate(spans):
            field = f"{container_name}.{index}"
            if not isinstance(span, dict):
                errors.append({"field": field, "reason": "must_be_object"})
                continue
            try:
                start_sec = float(span["start_sec"])
                end_sec = float(span["end_sec"])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    {"field": field, "reason": "start_end_must_be_numeric"}
                )
                continue
            if not math.isfinite(start_sec) or not math.isfinite(end_sec):
                errors.append(
                    {"field": field, "reason": "start_end_must_be_finite"}
                )
            elif start_sec < 0 or end_sec <= start_sec:
                errors.append(
                    {
                        "field": field,
                        "reason": "invalid_time_order",
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    }
                )
    if not errors:
        return
    findings.append(
        PlanAdmissionFinding(
            code="local_span_boundary_invalid",
            severity="blocking",
            plain_summary=(
                f"Step {step.id!r} contains local span boundaries that are not "
                "finite, ordered numeric timestamps."
            ),
            owner="planner",
            step_id=step.id,
            operation=step.operation,
            field=container_name,
            details={"errors": errors},
        )
    )
    requested_repairs.append(
        PlanAdmissionRepairCandidate(
            repair_type="planner",
            code="bind_numeric_local_span_boundaries",
            plain_summary=(
                "Bind each local start/end timestamp to reviewed evidence or "
                "return a non-executable plan; placeholders cannot enter Executor."
            ),
            step_id=step.id,
            details={"errors": errors},
        )
    )


def _required_input_is_explicit(
    step: AudioEditStep,
    required_input: str,
    *,
    plan: AudioEditPlan,
    context: dict[str, Any],
) -> bool:
    aliases = _INPUT_ALIASES.get(required_input, (required_input,))
    sources = [step.inputs, step.params, plan.global_params, context]
    for source in sources:
        for alias in aliases:
            value = source.get(alias)
            if not _present(value):
                continue
            if (
                required_input == "input_audio"
                and isinstance(value, str)
                and value.strip().lower() == "current"
            ):
                return bool(
                    plan.input_audio
                    or context.get("input_audio")
                    or context.get("input_audios")
                    or step.depends_on
                )
            return True
    if required_input == "input_audio":
        return bool(
            plan.input_audio
            or context.get("input_audio")
            or context.get("input_audios")
            or step.depends_on
        )
    return False


def _required_param_is_explicit(step: AudioEditStep, required_param: str) -> bool:
    aliases = _PARAM_ALIASES.get(required_param, (required_param,))
    return any(_present(step.params.get(alias)) for alias in aliases)


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _dependency_cycles(steps: list[AudioEditStep]) -> list[list[str]]:
    graph = {
        step.id: [dependency for dependency in step.depends_on if dependency != step.id]
        for step in steps
    }
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[list[str]] = []
    seen_cycles: set[tuple[str, ...]] = set()

    def visit(step_id: str) -> None:
        current_state = state.get(step_id, 0)
        if current_state == 2:
            return
        if current_state == 1:
            start = stack.index(step_id)
            cycle = [*stack[start:], step_id]
            key = _canonical_cycle(cycle)
            if key not in seen_cycles:
                seen_cycles.add(key)
                cycles.append(cycle)
            return
        state[step_id] = 1
        stack.append(step_id)
        for dependency in graph.get(step_id, []):
            if dependency in graph:
                visit(dependency)
        stack.pop()
        state[step_id] = 2

    for step_id in graph:
        visit(step_id)
    return cycles


def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
    nodes = cycle[:-1]
    if not nodes:
        return tuple(cycle)
    rotations = [tuple(nodes[index:] + nodes[:index]) for index in range(len(nodes))]
    return min(rotations)


def _audit_objective_coverage(
    plan: AudioEditPlan,
    *,
    step_id_set: set[str],
    findings: list[PlanAdmissionFinding],
    requested_repairs: list[PlanAdmissionRepairCandidate],
) -> list[ObjectiveCoverageAudit]:
    objectives = [
        item
        for item in plan.metadata.get("high_level_objectives") or []
        if isinstance(item, dict)
    ]
    coverage_rows = [
        item
        for item in plan.metadata.get("objective_coverage") or []
        if isinstance(item, dict)
    ]
    coverage_by_id = {
        str(item.get("objective_id") or item.get("id") or ""): item
        for item in coverage_rows
        if item.get("objective_id") or item.get("id")
    }
    if not objectives:
        findings.append(
            PlanAdmissionFinding(
                code="objective_decomposition_not_declared",
                severity="warning",
                plain_summary=(
                    "The plan does not declare high-level objectives, so admission "
                    "cannot verify semantic coverage."
                ),
                owner="planner",
                field="metadata.high_level_objectives",
            )
        )
        return []

    audits: list[ObjectiveCoverageAudit] = []
    for objective in objectives:
        objective_id = str(
            objective.get("id") or objective.get("objective_id") or ""
        ).strip()
        if not objective_id:
            objective_id = "unidentified_objective"
        required = bool(objective.get("required", True))
        coverage = coverage_by_id.get(objective_id)
        if coverage is None:
            audits.append(
                ObjectiveCoverageAudit(
                    objective_id=objective_id,
                    required=required,
                )
            )
            severity = "blocking" if required else "warning"
            findings.append(
                PlanAdmissionFinding(
                    code="objective_coverage_missing",
                    severity=severity,
                    plain_summary=(
                        f"Objective {objective_id!r} has no declared step coverage "
                        "or explicit unsupported status."
                    ),
                    owner="planner",
                    field="metadata.objective_coverage",
                    details={"objective_id": objective_id},
                )
            )
            requested_repairs.append(
                PlanAdmissionRepairCandidate(
                    repair_type="planner",
                    code="map_objective_to_steps",
                    plain_summary=(
                        f"Map objective {objective_id!r} to executable steps or "
                        "mark it explicitly unsupported."
                    ),
                    details={"objective_id": objective_id},
                )
            )
            continue

        status = str(coverage.get("status") or "").strip()
        covered_by_steps = [
            str(item) for item in coverage.get("covered_by_steps") or []
        ]
        valid_step_ids = [
            step_id for step_id in covered_by_steps if step_id in step_id_set
        ]
        unknown_step_ids = [
            step_id for step_id in covered_by_steps if step_id not in step_id_set
        ]
        executable = status == "covered_by_constraint" or (
            status == "planned" and bool(valid_step_ids) and not unknown_step_ids
        )
        audits.append(
            ObjectiveCoverageAudit(
                objective_id=objective_id,
                required=required,
                declared_status=status or None,
                covered_by_steps=covered_by_steps,
                valid_step_ids=valid_step_ids,
                unknown_step_ids=unknown_step_ids,
                executable_coverage=executable,
            )
        )
        if unknown_step_ids:
            findings.append(
                PlanAdmissionFinding(
                    code="objective_coverage_unknown_step",
                    severity="blocking" if required else "warning",
                    plain_summary=(
                        f"Objective {objective_id!r} references plan steps that do "
                        "not exist."
                    ),
                    owner="planner",
                    field="metadata.objective_coverage",
                    details={
                        "objective_id": objective_id,
                        "unknown_step_ids": unknown_step_ids,
                    },
                )
            )
        if required and not executable:
            code = (
                "required_objective_unsupported"
                if status == "unsupported"
                else "required_objective_not_executable"
            )
            findings.append(
                PlanAdmissionFinding(
                    code=code,
                    severity="blocking",
                    plain_summary=(
                        f"Required objective {objective_id!r} does not have "
                        "executable coverage in the current plan."
                    ),
                    owner="planner",
                    field="metadata.objective_coverage",
                    details={"objective_id": objective_id, "status": status},
                )
            )
    return audits


def _plan_fingerprint(plan: AudioEditPlan) -> str:
    payload = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
