"""Framework-owned provenance for explicitly declared, valid PlanIR dependencies.

The graph is the authority for an earlier output pointer. This module never
repairs a graph or supplies missing observation evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, get_args

from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.core.step_references import is_step_output_reference
from x_editor.planner.reasoned_workflow import ArgumentSourceKind


def project_dependency_argument_provenance(
    payload: Mapping[str, Any], *, validate_topology: Callable[[AudioEditPlan], None]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the original graph, then project only pure dependency sources.

    ``validate_topology`` is the existing complete plan topology validator. It
    runs before any projection, including the caller's other compatibility
    projections. Returned audit rows retain superseded model declarations.
    """
    projected = deepcopy(dict(payload))
    audit: dict[str, Any] = {
        "source": "validated_plan_dependencies", "plan_modified": False,
        "projected_parameter_count": 0, "replaced_sources": [],
    }
    if projected.get("decision") != "execute":
        return projected, audit
    plan = AudioEditPlan.model_validate(projected.get("plan"))
    _validate_reference_shapes(plan)
    validate_topology(plan)
    before = _fingerprint(projected["plan"])
    audit["plan_sha256_before"] = before
    steps = {step.id: step for step in plan.steps}
    semantics = projected.get("execution_semantics")
    bindings = semantics.get("bindings") if isinstance(semantics, dict) else None
    for binding in bindings if isinstance(bindings, list) else []:
        if not isinstance(binding, dict):
            continue
        step_id = binding.get("plan_step_id")
        step = steps.get(step_id) if isinstance(step_id, str) else None
        if step is None or binding.get("operation") != step.operation:
            continue
        sources = binding.get("argument_sources", [])
        if not isinstance(sources, list):
            continue
        for namespace in ("inputs", "params"):
            for key, value in getattr(step, namespace).items():
                if not _pure_dependency_tree(value):
                    continue
                path = f"{namespace}.{key}"
                replaced, retained = [], []
                for source in sources:
                    if _is_owned_source(source, step, path):
                        replaced.append(source)
                    else:
                        retained.append(source)
                generated = {"argument_path": path, "source_kind": "deterministic_context"}
                sources = retained + [generated]
                audit["replaced_sources"].append({
                    "plan_step_id": step.id, "argument_path": path,
                    "original_sources": deepcopy(replaced), "effective_source": generated,
                })
                audit["projected_parameter_count"] += 1
        binding["argument_sources"] = sources
    audit["plan_sha256_after"] = _fingerprint(projected["plan"])
    if audit["plan_sha256_after"] != before:
        raise ValueError("dependency provenance must not change the executable plan")
    return projected, audit


def _validate_reference_shapes(plan: AudioEditPlan) -> None:
    """Do not let malformed reserved reference objects evade graph validation."""
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if "from_step" in value or "from_output" in value:
                if not is_step_output_reference(value):
                    raise ValueError(
                        f"invalid step output reference at {path}: require exact from_step/from_output "
                        f"with a nonempty from_step and from_output='output_audio_path'; "
                        f"received keys {sorted(str(k) for k in value)}, "
                        f"from_output={value.get('from_output')!r}"
                    )
            else:
                for key, nested in value.items():
                    visit(nested, f'{path}.{key}')
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f'{path}[{index}]')

    for step in plan.steps:
        visit(step.inputs, f'steps[{step.id}].inputs')
        visit(step.params, f'steps[{step.id}].params')


def _pure_dependency_tree(value: Any) -> bool:
    if is_step_output_reference(value):
        return True
    if isinstance(value, list):
        return bool(value) and all(_pure_dependency_tree(item) for item in value)
    if isinstance(value, Mapping):
        return bool(value) and all(_pure_dependency_tree(item) for item in value.values())
    return False


def _is_owned_source(source: Any, step: AudioEditStep, root_path: str) -> bool:
    """Only recognized source rows over actual dependency values are redundant."""
    if not isinstance(source, dict) or set(source) - {
        "argument_path", "source_kind", "evidence_id", "evidence_path"
    }:
        return False
    if source.get("source_kind") not in get_args(ArgumentSourceKind):
        return False
    raw_path = source.get("argument_path")
    if not isinstance(raw_path, str):
        return False
    path = re.sub(r"\[(0|[1-9][0-9]*)\]", r".\1", raw_path.strip())
    if path != root_path and not path.startswith(root_path + "."):
        return False
    parts = path.split(".")
    value: Any = {"inputs": step.inputs, "params": step.params}
    try:
        for part in parts:
            if isinstance(value, list) and part.isdigit():
                value = value[int(part)]
            elif isinstance(value, Mapping):
                value = value[part]
            else:
                return False
    except (KeyError, IndexError):
        return False
    return _pure_dependency_tree(value)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
