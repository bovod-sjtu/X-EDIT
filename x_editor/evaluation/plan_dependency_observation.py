"""Read literal PlanIR dependencies without interpreting or repairing strings."""

from collections.abc import Mapping
from copy import deepcopy

from x_editor.core.step_references import iter_step_output_references


def observe_plan_dependencies(output):
    plan = output.get("plan") if isinstance(output, Mapping) else None
    steps = plan.get("steps") if isinstance(plan, Mapping) else None
    if not isinstance(steps, list) or not steps or any(
        not isinstance(s, Mapping) or not isinstance(s.get("id"), str) for s in steps
    ):
        return None
    rows, consumed = [], set()
    for step in steps:
        refs = list(iter_step_output_references({"inputs": step.get("inputs"),
                                                "params": step.get("params")}))
        consumed.update(r["from_step"] for r in refs)
        rows.append({"step_id": step["id"], "operation": step.get("operation"),
                     "declared_order_dependencies": deepcopy(step.get("depends_on", [])),
                     "literal_inputs": deepcopy(step.get("inputs", {})),
                     "recognized_output_references": deepcopy(refs)})
    return {
        "authority": "read_only_literal_plan_observation_not_admission_or_quality",
        "steps": rows,
        "derived_terminal_step_ids": [s["id"] for s in steps if s["id"] not in consumed],
        "declared_output_artifacts": deepcopy(plan.get("output_artifacts", [])),
        "contract": (
            "Output consumption is established by exact from_step/from_output objects. "
            "depends_on declares order, not consumption. A string is not a step-output pointer. "
            "Terminal/intermediate roles are derived, not user-settable flags. "
            "The original validator still decides whether this plan is admissible."
        ),
    }
