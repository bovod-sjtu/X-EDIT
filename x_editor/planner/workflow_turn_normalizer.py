"""Conservative, auditable shape normalization for PlannerReasonerTurn payloads."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Literal

from x_editor.core.schemas import StrictModel

_DECISIONS = {"request_perception", "execute", "cannot_plan"}
_REQUIREMENT_PRIORITIES = {"must", "should", "prefer"}
_NON_EVIDENCE_SOURCE_KINDS = {"user_request", "deterministic_context"}
_PLAN_NARRATIVE_FIELDS = {"steps_summary"}
_SEMANTIC_BINDING_FIELDS = {
    "plan_step_id",
    "objective",
    "target",
    "evidence_ids",
    "evidence_paths",
    "operation",
    "argument_sources",
    "unresolved_requirements",
}


class WorkflowTurnNormalization(StrictModel):
    """One deterministic field alias or unambiguous structure projection."""

    schema_version: Literal["x-editor.workflow-turn-normalization/v1"] = (
        "x-editor.workflow-turn-normalization/v1"
    )
    rule_id: str
    action: Literal[
        "unwrap_envelope",
        "project_envelope",
        "rename_field",
        "drop_redundant_alias",
        "drop_incompatible_fields",
        "drop_list_items",
        "merge_list_items",
        "deduplicate_list_items",
        "inject_authoritative_value",
        "flatten_mapping",
        "project_list_items",
        "project_sibling_value",
        "replace_literal",
        "wrap_list",
    ]
    source_path: str
    target_path: str
    item_count: int | None = None

    def audit_code(self) -> str:
        """Return the stable compact representation stored in invocation traces."""

        suffix = f":items={self.item_count}" if self.item_count is not None else ""
        return (
            f"workflow_turn_shape:{self.rule_id}:{self.action}:"
            f"{self.source_path}->{self.target_path}{suffix}"
        )


def normalize_workflow_turn_shape(
    payload: Any,
    *,
    authoritative_request: str | None = None,
) -> tuple[Any, list[WorkflowTurnNormalization]]:
    """Normalize only registered aliases and lossless structural projections.

    Missing decisions, targets, evidence, operations, and parameter values are never
    synthesized. Conflicts and malformed values remain in the payload so strict model
    validation rejects them.
    """

    if not isinstance(payload, Mapping):
        return payload, []
    normalized: dict[str, Any] = copy.deepcopy(dict(payload))
    audit: list[WorkflowTurnNormalization] = []

    normalized = _unwrap_known_envelope(normalized, audit)
    normalized = _project_explicit_decision_envelope(normalized, audit)
    _rename_aliases(
        normalized,
        path="",
        aliases={
            "decision_type": "decision",
            "action": "decision",
            "summary": "decision_summary",
            "perception_query": "perception_intent",
            "perception_request": "perception_intent",
            "audio_edit_plan": "plan",
            "semantics": "execution_semantics",
            "tool_selections": "tool_selection",
        },
        audit=audit,
    )

    edit_intent = normalized.get("edit_intent")
    if isinstance(edit_intent, dict):
        _replace_known_literal(
            edit_intent,
            path="edit_intent",
            field_name="schema_version",
            aliases={"x-editor.planner-current-task/v1": ("x-editor.planner-edit-intent/v1")},
            audit=audit,
        )
        _rename_aliases(
            edit_intent,
            path="edit_intent",
            aliases={
                "scopes": "target_scopes",
                "objectives": "edit_objectives",
                "preservation": "preservation_requirements",
                "quality": "quality_requirements",
                "evidence_requirements": "required_evidence",
                "operation_candidates": "candidate_operations",
                "ambiguities": "uncertainties",
            },
            audit=audit,
        )
        _project_string_list_items(
            edit_intent,
            field_name="edit_objectives",
            path="edit_intent",
            text_aliases=("objective", "description", "text"),
            audit=audit,
        )
        for field_name in ("preservation_requirements", "quality_requirements"):
            _normalize_requirement_aliases(
                edit_intent.get(field_name),
                path=f"edit_intent.{field_name}",
                audit=audit,
            )
        for field_name in (
            "effect_kinds",
            "target_scopes",
            "edit_objectives",
            "required_evidence",
            "candidate_operations",
            "uncertainties",
        ):
            _deduplicate_exact_list_values(
                edit_intent,
                field_name=field_name,
                path="edit_intent",
                audit=audit,
            )

    perception_intent = normalized.get("perception_intent")
    if isinstance(perception_intent, dict):
        _replace_known_literal(
            perception_intent,
            path="perception_intent",
            field_name="schema_version",
            aliases={"x-editor.planner-current-task/v1": ("x-editor.planner-perception-intent/v1")},
            audit=audit,
        )
        _rename_aliases(
            perception_intent,
            path="perception_intent",
            aliases={
                "operation": "perception_operation",
                "objective": "edit_objective",
                "targets": "perception_targets",
                "questions": "evidence_questions",
                "evidence_fields": "expected_evidence_fields",
                "precision": "temporal_precision",
                "operation_candidates": "candidate_operations",
                "audios": "audio_refs",
                "primary_audio": "primary_audio_ref",
            },
            audit=audit,
        )
        for index, target in enumerate(perception_intent.get("perception_targets") or []):
            if not isinstance(target, dict):
                continue
            _rename_aliases(
                target,
                path=f"perception_intent.perception_targets[{index}]",
                aliases={"target_kind": "kind", "target_value": "value"},
                audit=audit,
            )

    flattened_param_keys_by_step_id: dict[str, set[str]] = {}
    flattened_input_keys_by_step_id: dict[str, set[str]] = {}
    moved_input_keys_by_step_id: dict[str, set[str]] = {}
    plan = normalized.get("plan")
    if isinstance(plan, dict):
        _rename_aliases(
            plan,
            path="plan",
            aliases={"edit_steps": "steps"},
            audit=audit,
        )
        _drop_known_plan_narrative_fields(plan, audit=audit)
        moved_input_keys_by_step_id = _move_cross_container_input_addresses(
            plan,
            audit=audit,
        )
        _canonicalize_plan_step_prefixed_keys(plan, audit=audit)
        flattened_param_keys_by_step_id = _flatten_plan_step_params(
            plan,
            audit=audit,
        )
        flattened_input_keys_by_step_id = _flatten_plan_step_inputs(
            plan,
            audit=audit,
        )
        _canonicalize_step_output_references(plan, audit=audit)
        _bind_single_dependency_audio_inputs(plan, audit=audit)
        if (
            not str(plan.get("request") or "").strip()
            and isinstance(authoritative_request, str)
            and authoritative_request.strip()
        ):
            plan["request"] = authoritative_request
            audit.append(
                WorkflowTurnNormalization(
                    rule_id="inject_authoritative_plan_request",
                    action="inject_authoritative_value",
                    source_path="$runtime.user_request",
                    target_path="plan.request",
                )
            )

    execution_semantics = normalized.get("execution_semantics")
    if isinstance(execution_semantics, list):
        normalized["execution_semantics"] = {"bindings": execution_semantics}
        audit.append(
            WorkflowTurnNormalization(
                rule_id="execution_semantics_list_to_bindings",
                action="wrap_list",
                source_path="execution_semantics",
                target_path="execution_semantics.bindings",
                item_count=len(execution_semantics),
            )
        )
        execution_semantics = normalized["execution_semantics"]
    if isinstance(execution_semantics, dict):
        _rename_aliases(
            execution_semantics,
            path="execution_semantics",
            aliases={"step_bindings": "bindings"},
            audit=audit,
        )
        _replace_known_literal(
            execution_semantics,
            path="execution_semantics",
            field_name="schema_version",
            aliases={
                "x-editor.execution-semantics/v1": ("x-editor.planner-execution-semantics/v1")
            },
            audit=audit,
        )
        for index, binding in enumerate(execution_semantics.get("bindings") or []):
            if not isinstance(binding, dict):
                continue
            _project_binding_objective(
                binding,
                plan=plan,
                path=f"execution_semantics.bindings[{index}]",
                audit=audit,
            )
            _project_binding_target_from_plan_step(
                binding,
                plan=plan,
                path=f"execution_semantics.bindings[{index}]",
                audit=audit,
            )
            _project_unique_evidence_path_strings(
                binding,
                path=f"execution_semantics.bindings[{index}]",
                audit=audit,
            )
            _canonicalize_plan_qualified_argument_paths(
                binding.get("argument_sources"),
                plan=plan,
                plan_step_id=binding.get("plan_step_id"),
                path=f"execution_semantics.bindings[{index}].argument_sources",
                audit=audit,
            )
            _sync_flattened_argument_paths(
                binding.get("argument_sources"),
                plan_step_id=binding.get("plan_step_id"),
                nested_root="inputs",
                flattened_keys_by_step_id=flattened_input_keys_by_step_id,
                path=f"execution_semantics.bindings[{index}].argument_sources",
                audit=audit,
            )
            _sync_moved_input_argument_paths(
                binding.get("argument_sources"),
                plan_step_id=binding.get("plan_step_id"),
                moved_input_keys_by_step_id=moved_input_keys_by_step_id,
                path=f"execution_semantics.bindings[{index}].argument_sources",
                audit=audit,
            )
            _project_binding_evidence_from_argument_sources(
                binding,
                path=f"execution_semantics.bindings[{index}]",
                audit=audit,
            )
            _canonicalize_framework_audio_source_kinds(
                binding.get("argument_sources"),
                path=f"execution_semantics.bindings[{index}].argument_sources",
                audit=audit,
            )
            _drop_non_evidence_source_references(
                binding.get("argument_sources"),
                path=f"execution_semantics.bindings[{index}].argument_sources",
                audit=audit,
            )
            _deduplicate_argument_sources(
                binding,
                path=f"execution_semantics.bindings[{index}]",
                audit=audit,
            )
        _drop_empty_non_evidence_bindings(execution_semantics, audit=audit)
        _merge_nonconflicting_semantic_bindings(execution_semantics, audit=audit)

    tool_selection = normalized.get("tool_selection")
    if isinstance(tool_selection, list):
        normalized["tool_selection"] = {"steps": tool_selection}
        audit.append(
            WorkflowTurnNormalization(
                rule_id="tool_selection_list_to_steps",
                action="wrap_list",
                source_path="tool_selection",
                target_path="tool_selection.steps",
                item_count=len(tool_selection),
            )
        )
        tool_selection = normalized["tool_selection"]
    if isinstance(tool_selection, dict):
        _replace_known_literal(
            tool_selection,
            path="tool_selection",
            field_name="schema_version",
            aliases={"x-editor.tool-selection/v1": "x-editor.planner-tool-selection/v1"},
            audit=audit,
        )
        _project_unique_tool_selection_step_ids(
            tool_selection,
            plan=plan,
            audit=audit,
        )
        for index, selection in enumerate(tool_selection.get("steps") or []):
            if not isinstance(selection, dict):
                continue
            _canonicalize_plan_qualified_argument_paths(
                selection.get("parameter_selections"),
                plan=plan,
                plan_step_id=selection.get("plan_step_id"),
                path=f"tool_selection.steps[{index}].parameter_selections",
                audit=audit,
            )
            _sync_flattened_argument_paths(
                selection.get("parameter_selections"),
                plan_step_id=selection.get("plan_step_id"),
                nested_root="inputs",
                flattened_keys_by_step_id=flattened_input_keys_by_step_id,
                path=f"tool_selection.steps[{index}].parameter_selections",
                audit=audit,
            )
            _sync_moved_input_argument_paths(
                selection.get("parameter_selections"),
                plan_step_id=selection.get("plan_step_id"),
                moved_input_keys_by_step_id=moved_input_keys_by_step_id,
                path=f"tool_selection.steps[{index}].parameter_selections",
                audit=audit,
            )
        _sync_flattened_parameter_selection_paths(
            tool_selection,
            flattened_param_keys_by_step_id=flattened_param_keys_by_step_id,
            audit=audit,
        )
    return normalized, audit


_PLAN_QUALIFIED_ARGUMENT_PATH = re.compile(
    r"^plan\.steps\[(?P<index>\d+)\]\.(?P<root>inputs|params)\.(?P<suffix>.+)$"
)


def _canonicalize_plan_qualified_argument_paths(
    items: Any,
    *,
    plan: dict[str, Any] | None,
    plan_step_id: Any,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if not isinstance(items, list) or not isinstance(plan, dict):
        return
    steps = plan.get("steps")
    if not isinstance(steps, list) or not isinstance(plan_step_id, str):
        return
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        argument_path = item.get("argument_path")
        if not isinstance(argument_path, str):
            continue
        match = _PLAN_QUALIFIED_ARGUMENT_PATH.fullmatch(argument_path)
        if match is None:
            continue
        step_index = int(match.group("index"))
        if step_index >= len(steps):
            continue
        step = steps[step_index]
        if not isinstance(step, dict) or step.get("id") != plan_step_id:
            continue
        canonical = f"{match.group('root')}.{match.group('suffix')}"
        item["argument_path"] = canonical
        audit.append(
            WorkflowTurnNormalization(
                rule_id="strip_matching_plan_step_argument_prefix",
                action="rename_field",
                source_path=f"{path}[{index}].argument_path={argument_path}",
                target_path=f"{path}[{index}].argument_path={canonical}",
            )
        )


def _project_binding_evidence_from_argument_sources(
    binding: dict[str, Any],
    *,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if binding.get("evidence_ids") != [] or binding.get("evidence_paths") != []:
        return
    sources = binding.get("argument_sources")
    if not isinstance(sources, list):
        return
    evidence_ids: list[str] = []
    evidence_paths: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, dict) or source.get("source_kind") not in {
            "perception_evidence",
            "derived_from_evidence",
        }:
            continue
        evidence_id = source.get("evidence_id")
        evidence_path = source.get("evidence_path")
        if not isinstance(evidence_id, str) or not isinstance(evidence_path, str):
            continue
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
        item = {"evidence_id": evidence_id, "field_path": evidence_path}
        if item not in evidence_paths:
            evidence_paths.append(item)
    if not evidence_ids:
        return
    binding["evidence_ids"] = evidence_ids
    binding["evidence_paths"] = evidence_paths
    audit.append(
        WorkflowTurnNormalization(
            rule_id="project_binding_evidence_from_argument_sources",
            action="project_sibling_value",
            source_path=f"{path}.argument_sources",
            target_path=f"{path}.evidence_ids+evidence_paths",
            item_count=len(evidence_paths),
        )
    )


def _flatten_plan_step_params(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> dict[str, set[str]]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return {}
    step_id_counts: dict[str, int] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id")
        if isinstance(step_id, str) and step_id:
            step_id_counts[step_id] = step_id_counts.get(step_id, 0) + 1

    flattened_by_step_id: dict[str, set[str]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        nested = params.get("params")
        if not isinstance(nested, dict):
            continue
        if "params" in nested or any(not isinstance(key, str) for key in nested):
            continue
        outer_keys = set(params) - {"params"}
        nested_keys = set(nested)
        if outer_keys & nested_keys:
            continue
        step["params"] = {
            **{key: copy.deepcopy(value) for key, value in params.items() if key != "params"},
            **copy.deepcopy(nested),
        }
        audit.append(
            WorkflowTurnNormalization(
                rule_id="flatten_single_nested_plan_step_params",
                action="flatten_mapping",
                source_path=f"plan.steps[{index}].params.params",
                target_path=f"plan.steps[{index}].params",
                item_count=len(nested),
            )
        )
        step_id = step.get("id")
        if isinstance(step_id, str) and step_id and step_id_counts.get(step_id) == 1:
            flattened_by_step_id[step_id] = nested_keys
    return flattened_by_step_id


def _canonicalize_plan_step_prefixed_keys(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> None:
    """Interpret exact capability addresses as container-relative field names."""

    steps = plan.get("steps")
    if not isinstance(steps, list):
        return
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        for container_name in ("inputs", "params"):
            container = step.get(container_name)
            if not isinstance(container, dict):
                continue
            prefix = f"{container_name}."
            renames: list[tuple[str, str]] = []
            for key in container:
                if not isinstance(key, str) or not key.startswith(prefix):
                    continue
                canonical_key = key[len(prefix) :]
                if not canonical_key or "." in canonical_key or canonical_key in container:
                    continue
                renames.append((key, canonical_key))
            for source_key, target_key in renames:
                container[target_key] = container.pop(source_key)
                audit.append(
                    WorkflowTurnNormalization(
                        rule_id="strip_container_prefix_from_plan_step_key",
                        action="rename_field",
                        source_path=(f"plan.steps[{step_index}].{container_name}.{source_key}"),
                        target_path=(f"plan.steps[{step_index}].{container_name}.{target_key}"),
                    )
                )


def _move_cross_container_input_addresses(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> dict[str, set[str]]:
    """Move exact ``params.inputs`` addresses into the inputs container."""

    moved_by_step_id: dict[str, set[str]] = {}
    for step_index, step in enumerate(plan.get("steps") or []):
        if not isinstance(step, dict):
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        inputs = step.get("inputs")
        if inputs is None:
            inputs = {}
            step["inputs"] = inputs
        if not isinstance(inputs, dict):
            continue
        candidates: list[tuple[str, str, Any]] = []
        nested_inputs = params.get("inputs")
        if isinstance(nested_inputs, dict):
            for key, value in nested_inputs.items():
                if isinstance(key, str) and key and "." not in key:
                    candidates.append((f"inputs.{key}", key, value))
        for key, value in params.items():
            if not isinstance(key, str) or not key.startswith("inputs."):
                continue
            target_key = key[len("inputs.") :]
            if target_key and "." not in target_key:
                candidates.append((key, target_key, value))
        moved: set[str] = set()
        for source_key, target_key, value in candidates:
            if target_key in inputs and inputs[target_key] != value:
                continue
            inputs.setdefault(target_key, copy.deepcopy(value))
            moved.add(target_key)
            if source_key.startswith("inputs.") and source_key in params:
                params.pop(source_key)
            audit.append(
                WorkflowTurnNormalization(
                    rule_id="move_cross_container_input_address",
                    action="rename_field",
                    source_path=f"plan.steps[{step_index}].params.{source_key}",
                    target_path=f"plan.steps[{step_index}].inputs.{target_key}",
                )
            )
        if isinstance(nested_inputs, dict) and set(nested_inputs).issubset(moved):
            params.pop("inputs", None)
        step_id = step.get("id")
        if moved and isinstance(step_id, str) and step_id:
            moved_by_step_id[step_id] = moved
    return moved_by_step_id


def _sync_moved_input_argument_paths(
    values: Any,
    *,
    plan_step_id: Any,
    moved_input_keys_by_step_id: dict[str, set[str]],
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if not isinstance(values, list) or not isinstance(plan_step_id, str):
        return
    moved_keys = moved_input_keys_by_step_id.get(plan_step_id, set())
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        argument_path = value.get("argument_path")
        if not isinstance(argument_path, str):
            continue
        prefix = "params.inputs."
        if not argument_path.startswith(prefix):
            continue
        input_key = argument_path[len(prefix) :]
        if input_key not in moved_keys:
            continue
        canonical_path = f"inputs.{input_key}"
        value["argument_path"] = canonical_path
        field_path = f"{path}[{index}].argument_path"
        audit.append(
            WorkflowTurnNormalization(
                rule_id="sync_moved_input_argument_path",
                action="rename_field",
                source_path=f"{field_path}:{argument_path}",
                target_path=f"{field_path}:{canonical_path}",
            )
        )


def _flatten_plan_step_inputs(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> dict[str, set[str]]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return {}
    step_id_counts: dict[str, int] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id")
        if isinstance(step_id, str) and step_id:
            step_id_counts[step_id] = step_id_counts.get(step_id, 0) + 1

    flattened_by_step_id: dict[str, set[str]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        inputs = step.get("inputs")
        if not isinstance(inputs, dict):
            continue
        nested = inputs.get("inputs")
        if not isinstance(nested, dict):
            continue
        if "inputs" in nested or any(not isinstance(key, str) for key in nested):
            continue
        outer_keys = set(inputs) - {"inputs"}
        nested_keys = set(nested)
        if outer_keys & nested_keys:
            continue
        step["inputs"] = {
            **{key: copy.deepcopy(value) for key, value in inputs.items() if key != "inputs"},
            **copy.deepcopy(nested),
        }
        audit.append(
            WorkflowTurnNormalization(
                rule_id="flatten_single_nested_plan_step_inputs",
                action="flatten_mapping",
                source_path=f"plan.steps[{index}].inputs.inputs",
                target_path=f"plan.steps[{index}].inputs",
                item_count=len(nested),
            )
        )
        step_id = step.get("id")
        if isinstance(step_id, str) and step_id and step_id_counts.get(step_id) == 1:
            flattened_by_step_id[step_id] = nested_keys
    return flattened_by_step_id


def _sync_flattened_argument_paths(
    values: Any,
    *,
    plan_step_id: Any,
    nested_root: str,
    flattened_keys_by_step_id: dict[str, set[str]],
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if not isinstance(values, list) or not isinstance(plan_step_id, str):
        return
    flattened_keys = flattened_keys_by_step_id.get(plan_step_id)
    if flattened_keys is None:
        return
    prefix = f"{nested_root}.{nested_root}."
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        argument_path = value.get("argument_path")
        if not isinstance(argument_path, str) or not argument_path.startswith(prefix):
            continue
        nested_key = argument_path[len(prefix) :]
        if nested_key not in flattened_keys:
            continue
        canonical_path = f"{nested_root}.{nested_key}"
        value["argument_path"] = canonical_path
        field_path = f"{path}[{index}].argument_path"
        audit.append(
            WorkflowTurnNormalization(
                rule_id="sync_flattened_argument_path",
                action="rename_field",
                source_path=f"{field_path}:{argument_path}",
                target_path=f"{field_path}:{canonical_path}",
            )
        )


def _sync_flattened_parameter_selection_paths(
    tool_selection: dict[str, Any],
    *,
    flattened_param_keys_by_step_id: dict[str, set[str]],
    audit: list[WorkflowTurnNormalization],
) -> None:
    selections = tool_selection.get("steps")
    if not isinstance(selections, list):
        return
    prefix = "params.params."
    for selection_index, selection in enumerate(selections):
        if not isinstance(selection, dict):
            continue
        step_id = selection.get("plan_step_id")
        if not isinstance(step_id, str):
            continue
        flattened_keys = flattened_param_keys_by_step_id.get(step_id)
        if flattened_keys is None:
            continue
        parameter_selections = selection.get("parameter_selections")
        if not isinstance(parameter_selections, list):
            continue
        for parameter_index, parameter in enumerate(parameter_selections):
            if not isinstance(parameter, dict):
                continue
            argument_path = parameter.get("argument_path")
            if not isinstance(argument_path, str) or not argument_path.startswith(prefix):
                continue
            nested_key = argument_path[len(prefix) :]
            if nested_key not in flattened_keys:
                continue
            canonical_path = f"params.{nested_key}"
            parameter["argument_path"] = canonical_path
            field_path = (
                f"tool_selection.steps[{selection_index}]."
                f"parameter_selections[{parameter_index}].argument_path"
            )
            audit.append(
                WorkflowTurnNormalization(
                    rule_id="sync_flattened_parameter_selection_path",
                    action="rename_field",
                    source_path=f"{field_path}:{argument_path}",
                    target_path=f"{field_path}:{canonical_path}",
                )
            )


def _drop_known_plan_narrative_fields(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> None:
    for field_name in sorted(_PLAN_NARRATIVE_FIELDS):
        if field_name not in plan:
            continue
        plan.pop(field_name)
        audit.append(
            WorkflowTurnNormalization(
                rule_id=f"drop_plan_{field_name}",
                action="drop_incompatible_fields",
                source_path=f"plan.{field_name}",
                target_path="plan",
            )
        )


def _project_unique_tool_selection_step_ids(
    tool_selection: dict[str, Any],
    *,
    plan: Any,
    audit: list[WorkflowTurnNormalization],
) -> None:
    selections = tool_selection.get("steps")
    if not isinstance(selections, list) or not isinstance(plan, dict):
        return
    plan_steps = plan.get("steps")
    if not isinstance(plan_steps, list):
        return
    claimed_step_ids = {
        item.get("plan_step_id")
        for item in selections
        if isinstance(item, dict)
        and isinstance(item.get("plan_step_id"), str)
        and item["plan_step_id"]
    }
    for index, selection in enumerate(selections):
        if not isinstance(selection, dict) or "plan_step_id" in selection:
            continue
        operation = selection.get("selected_operation")
        if not isinstance(operation, str) or not operation:
            continue
        candidates = [
            step
            for step in plan_steps
            if isinstance(step, dict)
            and isinstance(step.get("id"), str)
            and step["id"]
            and step.get("operation") == operation
            and step["id"] not in claimed_step_ids
        ]
        if len(candidates) != 1:
            continue
        step_id = candidates[0]["id"]
        selection["plan_step_id"] = step_id
        claimed_step_ids.add(step_id)
        audit.append(
            WorkflowTurnNormalization(
                rule_id="project_tool_selection_plan_step_id_by_operation",
                action="project_sibling_value",
                source_path=f"plan.steps[id={step_id}].operation",
                target_path=f"tool_selection.steps[{index}].plan_step_id",
            )
        )


def _canonicalize_framework_audio_source_kinds(
    value: Any,
    *,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        if item.get("source_kind") != "workflow_source_audio_refs":
            continue
        item["source_kind"] = "deterministic_context"
        source_path = f"{path}[{index}].source_kind"
        audit.append(
            WorkflowTurnNormalization(
                rule_id="canonicalize_workflow_source_audio_refs",
                action="replace_literal",
                source_path=source_path,
                target_path=source_path,
            )
        )


def _deduplicate_argument_sources(
    binding: dict[str, Any],
    *,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    sources = binding.get("argument_sources")
    if not isinstance(sources, list):
        return
    unique: list[Any] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    removed_count = len(sources) - len(unique)
    if not removed_count:
        return
    binding["argument_sources"] = unique
    source_path = f"{path}.argument_sources"
    audit.append(
        WorkflowTurnNormalization(
            rule_id="deduplicate_argument_sources",
            action="deduplicate_list_items",
            source_path=source_path,
            target_path=source_path,
            item_count=removed_count,
        )
    )


def _deduplicate_exact_list_values(
    container: dict[str, Any],
    *,
    field_name: str,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    values = container.get(field_name)
    if not isinstance(values, list):
        return
    deduplicated: list[Any] = []
    for value in values:
        if value not in deduplicated:
            deduplicated.append(value)
    removed_count = len(values) - len(deduplicated)
    if not removed_count:
        return
    container[field_name] = deduplicated
    source_path = f"{path}.{field_name}"
    audit.append(
        WorkflowTurnNormalization(
            rule_id="deduplicate_exact_list_values",
            action="deduplicate_list_items",
            source_path=source_path,
            target_path=source_path,
            item_count=removed_count,
        )
    )


def _canonicalize_step_output_references(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> None:
    """Complete the one registered shorthand for an earlier audio output."""

    for step_index, step in enumerate(plan.get("steps") or []):
        if not isinstance(step, dict):
            continue
        inputs = step.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, value in inputs.items():
            if not isinstance(value, dict) or set(value) != {"from_step"}:
                continue
            producer = value.get("from_step")
            if not isinstance(producer, str) or not producer.strip():
                continue
            value["from_output"] = "output_audio_path"
            audit.append(
                WorkflowTurnNormalization(
                    rule_id="complete_step_output_reference",
                    action="inject_authoritative_value",
                    source_path="$contract.default_audio_step_output",
                    target_path=(f"plan.steps[{step_index}].inputs.{input_name}.from_output"),
                )
            )


def _bind_single_dependency_audio_inputs(
    plan: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> None:
    """Bind a declared single audio dependency when the model reused the root source."""

    steps = [step for step in plan.get("steps") or [] if isinstance(step, dict)]
    step_order = {
        str(step.get("id")): index
        for index, step in enumerate(steps)
        if str(step.get("id") or "")
    }
    root_sources = {
        str(plan.get("input_audio"))
        if isinstance(plan.get("input_audio"), str) and plan.get("input_audio")
        else ""
    }
    for step in steps:
        if step.get("depends_on"):
            continue
        inputs = step.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for field_name in ("input_audio", "input_audio_path"):
            value = inputs.get(field_name)
            if isinstance(value, str) and value:
                root_sources.add(value)
    root_sources.discard("")

    for step_index, step in enumerate(steps):
        dependencies = step.get("depends_on")
        if not isinstance(dependencies, list) or len(dependencies) != 1:
            continue
        producer = dependencies[0]
        if (
            not isinstance(producer, str)
            or producer not in step_order
            or step_order[producer] >= step_index
        ):
            continue
        inputs = step.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for field_name in ("input_audio", "input_audio_path"):
            value = inputs.get(field_name)
            if not isinstance(value, str) or value not in root_sources:
                continue
            inputs[field_name] = {
                "from_step": producer,
                "from_output": "output_audio_path",
            }
            audit.append(
                WorkflowTurnNormalization(
                    rule_id="bind_single_dependency_audio_input",
                    action="inject_authoritative_value",
                    source_path=f"plan.steps[{step_index}].depends_on[0]",
                    target_path=f"plan.steps[{step_index}].inputs.{field_name}",
                )
            )


def _drop_empty_non_evidence_bindings(
    execution_semantics: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> None:
    bindings = execution_semantics.get("bindings")
    if not isinstance(bindings, list):
        return
    retained: list[Any] = []
    removed_indexes: list[int] = []
    for index, binding in enumerate(bindings):
        if _is_empty_non_evidence_binding(binding):
            removed_indexes.append(index)
        else:
            retained.append(binding)
    if not removed_indexes:
        return
    execution_semantics["bindings"] = retained
    audit.append(
        WorkflowTurnNormalization(
            rule_id="drop_empty_non_evidence_bindings",
            action="drop_list_items",
            source_path=(
                "execution_semantics.bindings["
                + ",".join(str(index) for index in removed_indexes)
                + "]"
            ),
            target_path="execution_semantics.bindings",
            item_count=len(removed_indexes),
        )
    )


def _is_empty_non_evidence_binding(value: Any) -> bool:
    if not isinstance(value, dict) or not set(value) <= _SEMANTIC_BINDING_FIELDS:
        return False
    if value.get("evidence_ids") != [] or value.get("evidence_paths") != []:
        return False
    unresolved = value.get("unresolved_requirements", [])
    if unresolved != []:
        return False
    sources = value.get("argument_sources", [])
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        return False
    source_kinds = [item.get("source_kind") for item in sources]
    return all(
        isinstance(source_kind, str) and source_kind in _NON_EVIDENCE_SOURCE_KINDS
        for source_kind in source_kinds
    )


def _merge_nonconflicting_semantic_bindings(
    execution_semantics: dict[str, Any],
    *,
    audit: list[WorkflowTurnNormalization],
) -> None:
    bindings = execution_semantics.get("bindings")
    if not isinstance(bindings, list):
        return
    grouped_indexes: dict[str, list[int]] = {}
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            continue
        step_id = binding.get("plan_step_id")
        if isinstance(step_id, str) and step_id:
            grouped_indexes.setdefault(step_id, []).append(index)

    merged_by_first_index: dict[int, dict[str, Any]] = {}
    removed_indexes: set[int] = set()
    for step_id, indexes in grouped_indexes.items():
        if len(indexes) < 2:
            continue
        group = [bindings[index] for index in indexes]
        merged = _merge_binding_group(group)
        if merged is None:
            continue
        first_index = indexes[0]
        merged_by_first_index[first_index] = merged
        removed_indexes.update(indexes[1:])
        audit.append(
            WorkflowTurnNormalization(
                rule_id="merge_nonconflicting_semantic_bindings",
                action="merge_list_items",
                source_path=(f"execution_semantics.bindings[plan_step_id={step_id}]"),
                target_path=f"execution_semantics.bindings[{first_index}]",
                item_count=len(indexes),
            )
        )
    if not removed_indexes:
        return
    execution_semantics["bindings"] = [
        merged_by_first_index.get(index, binding)
        for index, binding in enumerate(bindings)
        if index not in removed_indexes
    ]


def _merge_binding_group(group: list[Any]) -> dict[str, Any] | None:
    if any(
        not isinstance(binding, dict) or not set(binding) <= _SEMANTIC_BINDING_FIELDS
        for binding in group
    ):
        return None
    identity_fields = ("plan_step_id", "operation")
    merged: dict[str, Any] = {}
    for field_name in identity_fields:
        declared = [binding[field_name] for binding in group if field_name in binding]
        unique: list[Any] = []
        for value in declared:
            if value not in unique:
                unique.append(value)
        if len(unique) > 1:
            return None
        if unique:
            merged[field_name] = copy.deepcopy(unique[0])

    for field_name in ("objective", "target"):
        declared = [binding[field_name] for binding in group if field_name in binding]
        unique_text: list[str] = []
        for value in declared:
            if not isinstance(value, str) or not value.strip():
                return None
            normalized = value.strip()
            if normalized not in unique_text:
                unique_text.append(normalized)
        if unique_text:
            merged[field_name] = " | ".join(unique_text)

    for field_name in (
        "evidence_ids",
        "evidence_paths",
        "argument_sources",
        "unresolved_requirements",
    ):
        declared_lists = [binding[field_name] for binding in group if field_name in binding]
        if any(not isinstance(value, list) for value in declared_lists):
            return None
        if not declared_lists:
            continue
        combined: list[Any] = []
        for values in declared_lists:
            for value in values:
                if value not in combined:
                    combined.append(copy.deepcopy(value))
        merged[field_name] = combined
    return merged


def _project_binding_target_from_plan_step(
    binding: dict[str, Any],
    *,
    plan: Any,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if str(binding.get("target") or "").strip() or not isinstance(plan, dict):
        return
    step_id = str(binding.get("plan_step_id") or "").strip()
    if not step_id:
        return
    candidates = [
        step
        for step in (plan.get("steps") or [])
        if isinstance(step, dict) and str(step.get("id") or "").strip() == step_id
    ]
    if len(candidates) != 1:
        return
    description = candidates[0].get("description")
    if not isinstance(description, str) or not description.strip():
        return
    binding["target"] = description.strip()
    audit.append(
        WorkflowTurnNormalization(
            rule_id="project_binding_target_from_plan_step_description",
            action="project_sibling_value",
            source_path=f"plan.steps[id={step_id}].description",
            target_path=f"{path}.target",
        )
    )


def _project_binding_objective(
    binding: dict[str, Any],
    *,
    plan: Any,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if "objective" in binding:
        return
    target = binding.get("target")
    if isinstance(target, str) and target.strip():
        binding["objective"] = target.strip()
        audit.append(
            WorkflowTurnNormalization(
                rule_id="project_binding_objective_from_target",
                action="project_sibling_value",
                source_path=f"{path}.target",
                target_path=f"{path}.objective",
            )
        )
        return
    step_id = str(binding.get("plan_step_id") or "").strip()
    if not step_id or not isinstance(plan, dict):
        return
    candidates = [
        step
        for step in (plan.get("steps") or [])
        if isinstance(step, dict) and str(step.get("id") or "").strip() == step_id
    ]
    if len(candidates) != 1:
        return
    description = candidates[0].get("description")
    if not isinstance(description, str) or not description.strip():
        return
    binding["objective"] = description.strip()
    audit.append(
        WorkflowTurnNormalization(
            rule_id="project_binding_objective_from_plan_step_description",
            action="project_sibling_value",
            source_path=f"plan.steps[id={step_id}].description",
            target_path=f"{path}.objective",
        )
    )


def _replace_known_literal(
    container: dict[str, Any],
    *,
    path: str,
    field_name: str,
    aliases: dict[str, str],
    audit: list[WorkflowTurnNormalization],
) -> None:
    value = container.get(field_name)
    canonical = aliases.get(value) if isinstance(value, str) else None
    if canonical is None:
        return
    container[field_name] = canonical
    field_path = f"{path}.{field_name}"
    audit.append(
        WorkflowTurnNormalization(
            rule_id=f"canonicalize_{path.replace('.', '_')}_{field_name}",
            action="replace_literal",
            source_path=field_path,
            target_path=field_path,
        )
    )


def _project_unique_evidence_path_strings(
    binding: dict[str, Any],
    *,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    evidence_paths = binding.get("evidence_paths")
    evidence_ids = binding.get("evidence_ids")
    if not isinstance(evidence_paths, list) or not any(
        isinstance(item, str) for item in evidence_paths
    ):
        return
    if not isinstance(evidence_ids, list):
        return
    unique_ids = list(
        dict.fromkeys(
            item.strip() for item in evidence_ids if isinstance(item, str) and item.strip()
        )
    )
    if len(unique_ids) != 1 or any(not isinstance(item, (str, Mapping)) for item in evidence_paths):
        return
    evidence_id = unique_ids[0]
    projected: list[Any] = []
    projected_count = 0
    for item in evidence_paths:
        if isinstance(item, str):
            field_path = item.strip()
            if not field_path:
                return
            projected.append({"evidence_id": evidence_id, "field_path": field_path})
            projected_count += 1
        else:
            projected.append(copy.deepcopy(dict(item)))
    binding["evidence_paths"] = projected
    field_path = f"{path}.evidence_paths"
    audit.append(
        WorkflowTurnNormalization(
            rule_id="project_unique_evidence_path_strings",
            action="project_list_items",
            source_path=f"{field_path}[]",
            target_path=f"{field_path}[]",
            item_count=projected_count,
        )
    )


def _drop_non_evidence_source_references(
    value: Any,
    *,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict) or item.get("source_kind") not in {
            "user_request",
            "deterministic_context",
        }:
            continue
        removed = [
            field_name
            for field_name in ("evidence_id", "evidence_path")
            if item.get(field_name) is not None
        ]
        if not removed:
            continue
        for field_name in removed:
            item.pop(field_name, None)
        audit.append(
            WorkflowTurnNormalization(
                rule_id="drop_non_evidence_source_references",
                action="drop_incompatible_fields",
                source_path=f"{path}[{index}].{'+'.join(removed)}",
                target_path=f"{path}[{index}]",
            )
        )


def _unwrap_known_envelope(
    payload: dict[str, Any],
    audit: list[WorkflowTurnNormalization],
) -> dict[str, Any]:
    for field_name in ("planner_reasoner_turn", "workflow_turn"):
        value = payload.get(field_name)
        if set(payload) != {field_name} or not isinstance(value, Mapping):
            continue
        audit.append(
            WorkflowTurnNormalization(
                rule_id=f"unwrap_{field_name}",
                action="unwrap_envelope",
                source_path=field_name,
                target_path="$",
            )
        )
        return copy.deepcopy(dict(value))
    return payload


def _project_explicit_decision_envelope(
    payload: dict[str, Any],
    audit: list[WorkflowTurnNormalization],
) -> dict[str, Any]:
    value = payload.get("decision")
    if not isinstance(value, Mapping):
        return payload
    nested = copy.deepcopy(dict(value))
    discriminant_fields = [
        field_name
        for field_name in ("decision", "decision_type", "action", "type")
        if field_name in nested
    ]
    discriminant_values = [nested[field_name] for field_name in discriminant_fields]
    if (
        not discriminant_values
        or any(not isinstance(item, str) for item in discriminant_values)
        or len(set(discriminant_values)) != 1
        or discriminant_values[0] not in _DECISIONS
    ):
        return payload
    decision = discriminant_values[0]
    for field_name in discriminant_fields:
        nested.pop(field_name)
    outer = {key: copy.deepcopy(item) for key, item in payload.items() if key != "decision"}
    if any(key in outer and outer[key] != item for key, item in nested.items()):
        return payload
    outer.update(nested)
    outer["decision"] = decision
    audit.append(
        WorkflowTurnNormalization(
            rule_id="project_explicit_decision_envelope",
            action="project_envelope",
            source_path="decision",
            target_path="$",
        )
    )
    return outer


def _rename_aliases(
    container: dict[str, Any],
    *,
    path: str,
    aliases: dict[str, str],
    audit: list[WorkflowTurnNormalization],
) -> None:
    for alias, canonical in aliases.items():
        if alias not in container:
            continue
        source_path = f"{path}.{alias}" if path else alias
        target_path = f"{path}.{canonical}" if path else canonical
        if canonical not in container:
            container[canonical] = container.pop(alias)
            audit.append(
                WorkflowTurnNormalization(
                    rule_id=f"alias_{alias}_to_{canonical}",
                    action="rename_field",
                    source_path=source_path,
                    target_path=target_path,
                )
            )
            continue
        if container[canonical] == container[alias]:
            container.pop(alias)
            audit.append(
                WorkflowTurnNormalization(
                    rule_id=f"drop_redundant_{alias}_alias",
                    action="drop_redundant_alias",
                    source_path=source_path,
                    target_path=target_path,
                )
            )


def _project_string_list_items(
    container: dict[str, Any],
    *,
    field_name: str,
    path: str,
    text_aliases: tuple[str, ...],
    audit: list[WorkflowTurnNormalization],
) -> None:
    value = container.get(field_name)
    if not isinstance(value, list) or not any(isinstance(item, Mapping) for item in value):
        return
    projected: list[Any] = []
    projected_count = 0
    for item in value:
        if isinstance(item, str):
            projected.append(item)
            continue
        if not isinstance(item, Mapping):
            return
        candidates = [
            item[alias]
            for alias in text_aliases
            if isinstance(item.get(alias), str) and item[alias].strip()
        ]
        if len(candidates) != 1:
            return
        projected.append(candidates[0])
        projected_count += 1
    container[field_name] = projected
    field_path = f"{path}.{field_name}"
    audit.append(
        WorkflowTurnNormalization(
            rule_id=f"project_{field_name}_objects_to_strings",
            action="project_list_items",
            source_path=f"{field_path}[]",
            target_path=f"{field_path}[]",
            item_count=projected_count,
        )
    )


def _normalize_requirement_aliases(
    value: Any,
    *,
    path: str,
    audit: list[WorkflowTurnNormalization],
) -> None:
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        item_path = f"{path}[{index}]"
        _rename_aliases(
            item,
            path=item_path,
            aliases={"requirement": "description", "text": "description"},
            audit=audit,
        )
        if "description" not in item:
            dimension = item.get("dimension")
            if isinstance(dimension, str) and dimension.strip():
                item["description"] = dimension
                audit.append(
                    WorkflowTurnNormalization(
                        rule_id="project_requirement_description_from_dimension",
                        action="project_sibling_value",
                        source_path=f"{item_path}.dimension",
                        target_path=f"{item_path}.description",
                    )
                )
        _replace_known_literal(
            item,
            path=item_path,
            field_name="priority",
            aliases={"strong": "must"},
            audit=audit,
        )
        level = item.get("level")
        if level in _REQUIREMENT_PRIORITIES:
            _rename_aliases(
                item,
                path=item_path,
                aliases={"level": "priority"},
                audit=audit,
            )


__all__ = [
    "WorkflowTurnNormalization",
    "normalize_workflow_turn_shape",
]
