"""Planner-output validation and plan-level normalization.

This layer runs before ``AudioEditPlan`` construction. It keeps model output
inside the planner contract, removes bulky context blobs from executable
params, and records deterministic corrections for later audit.
"""

from __future__ import annotations

import json
from typing import Any

from x_editor.core.schemas import AudioEditPlan, AudioEditStep


_ROOT_FIELDS = set(AudioEditPlan.model_fields)
_STEP_FIELDS = set(AudioEditStep.model_fields)
_VALID_STEP_STATUS = {"pending", "running", "success", "failed", "skipped"}
_VALID_COVERAGE_STATUS = {"planned", "covered_by_constraint", "unsupported", "uncovered"}
_COVERAGE_STATUS_ALIASES = {"covered": "planned"}

_BULKY_CONTEXT_KEYS = {
    "available_operations",
    "audio_captions",
    "diarization",
    "failed_tool_trace",
    "manual",
    "operation_manual",
    "operation_manuals",
    "perception_attempts",
    "perception_gap_fill_guidance",
    "perception_report",
    "planner_perception_guidance",
    "routing_policy_candidates",
    "span_candidates",
    "span_decisions",
    "span_selection_analysis",
    "speaker_segments",
    "tool_manual",
    "tool_manuals",
    "word_timestamps",
}
_SPAN_CONTAINER_KEYS = {"selected_spans", "spans"}
_EXCLUDED_SPAN_DECISIONS = {"exclude", "preserve_context", "reference_span", "needs_more_perception"}
_MAX_CONTEXT_STRING_LEN = 20000


class PlanValidationError(RuntimeError):
    """Raised when planner output cannot be safely normalized."""


def normalize_planner_output(
    data: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
    planner_model: str | None = None,
    phase: str = "initial",
) -> dict[str, Any]:
    """Normalize Gemini planner JSON into the stable ``AudioEditPlan`` shape.

    The function mutates and returns ``data`` so callers can keep existing
    references, but all edits are recorded under
    ``metadata.plan_normalizer_corrections``.
    """

    if not isinstance(data, dict):
        raise PlanValidationError(f"planner output must be a JSON object, got {type(data).__name__}")

    available = operation_ids(operations)
    metadata = _ensure_metadata(data)
    corrections = metadata.setdefault("plan_normalizer_corrections", [])
    if not isinstance(corrections, list):
        corrections = []
        metadata["plan_normalizer_corrections"] = corrections

    _drop_unknown_root_fields(data, corrections, phase=phase)
    data.setdefault("request", user_request)
    data.setdefault("input_audio", context.get("input_audio"))
    data.setdefault("output_audio", context.get("output_audio"))
    if "global_params" not in data:
        data["global_params"] = {}
    elif not isinstance(data.get("global_params"), dict):
        data["global_params"] = {}
        _record(corrections, phase, "global_params", "reset_non_dict_global_params")
    data.setdefault("missing_inputs", [])
    data["missing_inputs"] = _normalize_missing_inputs(data.get("missing_inputs"))
    if planner_model:
        metadata.setdefault("planner_model", planner_model)

    steps = data.get("steps")
    if not isinstance(steps, list):
        data["steps"] = []
        _record(corrections, phase, "steps", "reset_non_list_steps")
        steps = data["steps"]

    normalized_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            step = _unsupported_step(
                index=index,
                reason=f"planner step {index + 1} was not a JSON object",
                available=available,
            )
            _record(corrections, phase, step["id"], "replaced_non_dict_step_with_unsupported")
        else:
            step = raw_step
        _normalize_step_shape(
            step,
            index=index,
            available=available,
            context=context,
            corrections=corrections,
            phase=phase,
        )
        normalized_steps.append(step)
    data["steps"] = normalized_steps

    step_ids = {str(step.get("id")) for step in normalized_steps}
    _normalize_objective_metadata(metadata, step_ids=step_ids, corrections=corrections, phase=phase)
    _validate_operation_graph_metadata(data, step_ids=step_ids, corrections=corrections, phase=phase)
    return data


def operation_ids(operations: list[dict[str, Any]]) -> set[str]:
    """Return operation ids accepted by the current planner catalog."""

    ids: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        value = operation.get("operation") or operation.get("tool_id")
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def _ensure_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata
    return metadata


def _drop_unknown_root_fields(data: dict[str, Any], corrections: list[dict[str, Any]], *, phase: str) -> None:
    for key in list(data):
        if key not in _ROOT_FIELDS:
            data.pop(key, None)
            _record(corrections, phase, key, "dropped_unknown_plan_field")


def _normalize_missing_inputs(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in value]


def _normalize_step_shape(
    step: dict[str, Any],
    *,
    index: int,
    available: set[str],
    context: dict[str, Any],
    corrections: list[dict[str, Any]],
    phase: str,
) -> None:
    step_id = str(step.get("id") or f"step_{index + 1}")
    step["id"] = step_id
    for key in list(step):
        if key not in _STEP_FIELDS:
            step.pop(key, None)
            _record(corrections, phase, step_id, "dropped_unknown_step_field", field=key)

    operation = step.get("operation")
    if not isinstance(operation, str) or not operation.strip():
        if available and "unsupported" not in available:
            raise PlanValidationError(
                "planner step is missing operation, and Available Operations does not include 'unsupported'"
            )
        _rewrite_step_to_unsupported(step, "planner step is missing operation", available)
        _record(corrections, phase, step_id, "missing_operation_rewritten_to_unsupported")
    else:
        step["operation"] = operation.strip()
        if available and step["operation"] not in available:
            if "unsupported" not in available:
                raise PlanValidationError(
                    f"planner selected unavailable operation {step['operation']!r}, "
                    "and Available Operations does not include 'unsupported'"
                )
            _rewrite_step_to_unsupported(
                step,
                f"planner selected unavailable operation {step['operation']!r}",
                available,
            )
            _record(
                corrections,
                phase,
                step_id,
                "unavailable_operation_rewritten_to_unsupported",
                operation=operation.strip(),
            )

    if not isinstance(step.get("description"), str):
        step["description"] = str(step.get("description") or "")
        _record(corrections, phase, step_id, "normalized_non_string_description")
    if not isinstance(step.get("inputs"), dict):
        step["inputs"] = {}
        _record(corrections, phase, step_id, "reset_non_dict_inputs")
    if not isinstance(step.get("params"), dict):
        step["params"] = {}
        _record(corrections, phase, step_id, "reset_non_dict_params")
    if not isinstance(step.get("depends_on"), list):
        step["depends_on"] = []
        _record(corrections, phase, step_id, "reset_non_list_depends_on")
    if step.get("status") not in _VALID_STEP_STATUS:
        step["status"] = "pending"
        _record(corrections, phase, step_id, "reset_invalid_status")
    if "required" in step and not isinstance(step.get("required"), bool):
        step["required"] = bool(step.get("required"))
        _record(corrections, phase, step_id, "normalized_required_bool")

    _normalize_misplaced_speaker_scoped_arguments(
        step,
        corrections=corrections,
        phase=phase,
    )
    _sanitize_params(
        step["params"],
        operation=str(step.get("operation") or ""),
        path=f"steps.{step_id}.params",
        corrections=corrections,
        phase=phase,
    )
    _sanitize_selected_spans(step["params"], path=f"steps.{step_id}.params", corrections=corrections, phase=phase)
    _validate_session_aliases(step, context=context, corrections=corrections, phase=phase)


def _normalize_misplaced_speaker_scoped_arguments(
    step: dict[str, Any],
    *,
    corrections: list[dict[str, Any]],
    phase: str,
) -> None:
    if step.get("operation") != "artifact_ops.speaker_scoped_edit":
        return
    inputs = step["inputs"]
    params = step["params"]
    step_id = str(step.get("id") or "")

    nested_operation = inputs.pop("edit_op", None)
    if nested_operation is not None and not params.get("edit_op"):
        params["edit_op"] = nested_operation
        _record(
            corrections,
            phase,
            step_id,
            "moved_misplaced_edit_op_from_inputs_to_params",
        )
    nested_params = inputs.pop("params", None)
    if isinstance(nested_params, dict) and not isinstance(
        params.get("edit_op_params"), dict
    ):
        params["edit_op_params"] = nested_params
        _record(
            corrections,
            phase,
            step_id,
            "moved_misplaced_edit_op_params_from_inputs_to_params",
        )

    for key in _SPAN_CONTAINER_KEYS:
        raw_spans = inputs.pop(key, None)
        if raw_spans is None:
            continue
        if isinstance(raw_spans, list) and raw_spans and all(
            isinstance(span, dict)
            and span.get("start_sec") is not None
            and span.get("end_sec") is not None
            for span in raw_spans
        ):
            params.setdefault(key, raw_spans)
            reason = "moved_misplaced_span_values_from_inputs_to_params"
        else:
            reason = "dropped_evidence_references_misplaced_as_input_span_values"
        _record(corrections, phase, step_id, reason, field=key)


def _rewrite_step_to_unsupported(step: dict[str, Any], reason: str, available: set[str]) -> None:
    step["operation"] = "unsupported"
    step["description"] = step.get("description") or reason
    step["params"] = {"reason": reason}
    if not isinstance(step.get("inputs"), dict):
        step["inputs"] = {}


def _unsupported_step(*, index: int, reason: str, available: set[str]) -> dict[str, Any]:
    if available and "unsupported" not in available:
        raise PlanValidationError(f"{reason}; Available Operations does not include 'unsupported'")
    step = {
        "id": f"step_{index + 1}",
        "operation": "unsupported",
        "description": reason,
        "inputs": {},
        "params": {"reason": reason},
        "depends_on": [],
        "status": "pending",
    }
    return step


def _sanitize_params(
    value: Any,
    *,
    operation: str,
    path: str,
    corrections: list[dict[str, Any]],
    phase: str,
) -> None:
    if not isinstance(value, dict):
        return
    for key in list(value):
        item = value[key]
        key_text = str(key)
        if key_text in _BULKY_CONTEXT_KEYS:
            value.pop(key, None)
            _record(corrections, phase, path, "dropped_bulky_context_param", field=key_text)
            continue
        if _is_oversized_context_string(key_text, item, operation=operation):
            value.pop(key, None)
            _record(corrections, phase, path, "dropped_oversized_context_string_param", field=key_text)
            continue
        if isinstance(item, dict):
            _sanitize_params(item, operation=_nested_operation(operation, value), path=f"{path}.{key_text}", corrections=corrections, phase=phase)
        elif isinstance(item, list):
            for list_index, element in enumerate(item):
                if isinstance(element, dict):
                    _sanitize_params(
                        element,
                        operation=_nested_operation(operation, value),
                        path=f"{path}.{key_text}[{list_index}]",
                        corrections=corrections,
                        phase=phase,
                    )


def _nested_operation(parent_operation: str, params: dict[str, Any]) -> str:
    nested = params.get("edit_op") or params.get("edit_operation")
    if isinstance(nested, str) and nested:
        return nested
    return parent_operation


def _is_oversized_context_string(key: str, value: Any, *, operation: str) -> bool:
    if not isinstance(value, str) or len(value) <= _MAX_CONTEXT_STRING_LEN:
        return False
    executable_text_keys = {"instruction_st", "target_text", "source_text", "instruction"}
    if key in executable_text_keys and operation.startswith("edit_data_pipeline."):
        return False
    if key in executable_text_keys and operation in {"foreground_background_remix", "artifact_ops.speaker_scoped_edit"}:
        return False
    return True


def _sanitize_selected_spans(value: Any, *, path: str, corrections: list[dict[str, Any]], phase: str) -> None:
    if not isinstance(value, dict):
        return
    for key in list(value):
        item = value[key]
        child_path = f"{path}.{key}"
        if key in _SPAN_CONTAINER_KEYS and isinstance(item, list):
            kept = []
            removed = 0
            for span in item:
                if isinstance(span, dict) and _span_is_evidence_only(span):
                    removed += 1
                    continue
                kept.append(span)
            if removed:
                value[key] = kept
                _record(corrections, phase, child_path, "dropped_evidence_only_spans", count=removed)
        elif isinstance(item, dict):
            _sanitize_selected_spans(item, path=child_path, corrections=corrections, phase=phase)
        elif isinstance(item, list):
            for index, element in enumerate(item):
                if isinstance(element, dict):
                    _sanitize_selected_spans(element, path=f"{child_path}[{index}]", corrections=corrections, phase=phase)


def _span_is_evidence_only(span: dict[str, Any]) -> bool:
    evidence = " ".join(
        str(span.get(key) or "")
        for key in (
            "decision",
            "role",
            "span_role",
            "span_type",
            "purpose",
            "status",
            "source",
        )
    ).casefold()
    return any(marker in evidence for marker in _EXCLUDED_SPAN_DECISIONS)


def _validate_session_aliases(
    step: dict[str, Any],
    *,
    context: dict[str, Any],
    corrections: list[dict[str, Any]],
    phase: str,
) -> None:
    if not context.get("edit_session_graph_context") and not context.get("edit_session_audio_aliases"):
        return
    known_aliases = _known_session_aliases(context)
    for container_name in ("inputs", "params"):
        container = step.get(container_name)
        if not isinstance(container, dict):
            continue
        for path, value in _iter_string_values(container, prefix=container_name):
            if not _is_session_alias_candidate(value, known_aliases):
                continue
            if not _valid_session_alias(value, known_aliases):
                _record(
                    corrections,
                    phase,
                    str(step.get("id") or ""),
                    "invalid_session_alias_reference",
                    field=path,
                    value=value,
                )
            elif known_aliases and _alias_requires_known_target(value) and value not in known_aliases:
                _record(
                    corrections,
                    phase,
                    str(step.get("id") or ""),
                    "unknown_session_alias_reference",
                    field=path,
                    value=value,
                )


def _known_session_aliases(context: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for source in (
        context.get("edit_session_audio_aliases"),
        _mapping(context.get("edit_session_graph_context")).get("audio_aliases"),
    ):
        if isinstance(source, dict):
            aliases.update(str(key) for key in source if key not in (None, ""))
    normalized_refs = _mapping(_mapping(context.get("edit_session_graph_context")).get("normalized_refs"))
    for item in normalized_refs.get("aliases") or []:
        if isinstance(item, dict) and item.get("alias"):
            aliases.add(str(item["alias"]))
    return aliases


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _iter_string_values(value: Any, *, prefix: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            rows.extend(_iter_string_values(item, prefix=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_iter_string_values(item, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str):
        rows.append((prefix, value.strip()))
    return rows


def _is_session_alias_candidate(value: str, known_aliases: set[str]) -> bool:
    if not value:
        return False
    if value in known_aliases:
        return True
    if value in {"session_current", "current_session_audio", "session_initial", "session_original"}:
        return True
    return value.startswith(
        (
            "session_",
            "session_node:",
            "node:",
            "session_round:",
            "round_",
        )
    )


def _valid_session_alias(value: str, known_aliases: set[str] | None = None) -> bool:
    if known_aliases and value in known_aliases:
        return True
    if value in {"session_current", "current_session_audio", "session_initial", "session_original"}:
        return True
    if value.startswith(("session_node:", "node:", "session_round:", "session_audio:")):
        return ":" in value and bool(value.split(":", 1)[1])
    if value.startswith(("session_reference:", "session_reference_node:", "reference_node:")):
        return ":" in value and bool(value.split(":", 1)[1])
    if value.startswith("round_") and value.endswith("_result"):
        return True
    return False


def _alias_requires_known_target(value: str) -> bool:
    if value in {"session_current", "current_session_audio", "session_initial", "session_original"}:
        return True
    return value.startswith(
        (
            "session_node:",
            "node:",
            "session_round:",
            "round_",
            "session_audio:",
            "session_reference:",
            "session_reference_node:",
            "reference_node:",
        )
    )


def _normalize_objective_metadata(
    metadata: dict[str, Any],
    *,
    step_ids: set[str],
    corrections: list[dict[str, Any]],
    phase: str,
) -> None:
    objectives = metadata.get("high_level_objectives")
    if objectives is not None and not isinstance(objectives, list):
        metadata["high_level_objectives"] = []
        _record(corrections, phase, "metadata.high_level_objectives", "reset_non_list_objectives")
        objectives = []
    objective_ids = {
        str(item.get("id"))
        for item in objectives or []
        if isinstance(item, dict) and item.get("id") is not None
    }

    coverage = metadata.get("objective_coverage")
    if coverage is not None and not isinstance(coverage, list):
        metadata["objective_coverage"] = []
        _record(corrections, phase, "metadata.objective_coverage", "reset_non_list_objective_coverage")
        return
    if not isinstance(coverage, list):
        return
    for item in coverage:
        if not isinstance(item, dict):
            continue
        raw_status = str(item.get("status") or "")
        normalized_status = _COVERAGE_STATUS_ALIASES.get(raw_status)
        if normalized_status is not None:
            item["status"] = normalized_status
            _record(
                corrections,
                phase,
                "metadata.objective_coverage",
                "normalized_coverage_status_alias",
                from_status=raw_status,
                status=normalized_status,
            )
        elif raw_status not in _VALID_COVERAGE_STATUS:
            item["status"] = "uncovered"
            _record(corrections, phase, "metadata.objective_coverage", "reset_invalid_coverage_status")
        covered = item.get("covered_by_steps")
        if isinstance(covered, list):
            filtered = [str(step_id) for step_id in covered if str(step_id) in step_ids]
            if filtered != covered:
                item["covered_by_steps"] = filtered
                _record(corrections, phase, "metadata.objective_coverage", "dropped_unknown_covered_step")
        objective_id = str(item.get("objective_id") or "")
        if objective_ids and objective_id not in objective_ids:
            item["status"] = "uncovered"
            item["covered_by_steps"] = []
            _record(corrections, phase, "metadata.objective_coverage", "unknown_objective_marked_uncovered")


def _validate_operation_graph_metadata(
    data: dict[str, Any],
    *,
    step_ids: set[str],
    corrections: list[dict[str, Any]],
    phase: str,
) -> None:
    metadata = _ensure_metadata(data)
    raw_graph = metadata.get("operation_graph_plan")
    if raw_graph is None:
        return
    if not isinstance(raw_graph, dict):
        _record(corrections, phase, "metadata.operation_graph_plan", "invalid_operation_graph_plan")
        return
    try:
        from x_editor.session.operation_graph import build_planned_operation_graph, validate_operation_graph

        graph = build_planned_operation_graph(data)
        report = validate_operation_graph(graph, plan=data)
        metadata["operation_graph_plan_validation"] = report.model_dump(exclude_none=True)
        for issue in report.issues:
            if issue.severity not in {"error", "warning"}:
                continue
            _record(
                corrections,
                phase,
                issue.location or issue.ref_id or "metadata.operation_graph_plan",
                f"operation_graph_{issue.code}",
                severity=issue.severity,
                ref_id=issue.ref_id,
                **({"step_ids": sorted(step_ids)} if issue.code in {"node_unknown_step", "operation_unknown_step"} else {}),
            )
    except Exception as exc:
        _record(
            corrections,
            phase,
            "metadata.operation_graph_plan",
            "operation_graph_validation_error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _record(
    corrections: list[dict[str, Any]],
    phase: str,
    target: str,
    reason: str,
    **extra: Any,
) -> None:
    item = {"phase": phase, "target": target, "reason": reason}
    item.update(extra)
    corrections.append(item)
