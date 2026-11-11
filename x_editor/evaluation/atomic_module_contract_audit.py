"""Read-only format/contract checks, separate from semantic scores.

A provider response rejected before canonicalization is not a canonical PlanIR.
Its recorded rejection remains evidence; framework-owned fields are not invented
to make a replay pass. These checks neither execute tools nor certify quality.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def audit_module_contract(module, module_input, module_output, *, accepted, receipt):
    result = {
        "accepted_in_recorded_run": accepted,
        "recorded_status": receipt.get("status"),
        "recorded_error": receipt.get("error"),
        "tools_invoked": False,
        "tool_execution_verified": False,
        "semantic_correctness_certified": False,
        "checks": [],
    }
    if module == "planner":
        from x_editor.evaluation.plan_dependency_observation import observe_plan_dependencies

        result["literal_plan_dependencies"] = observe_plan_dependencies(module_output)
    if module_output is None:
        result["checks"].append({"check": "canonical_schema", "status": "unavailable"})
        return result
    if not accepted:
        result["checks"].append(
            {
                "check": "canonical_schema",
                "status": "not_applicable_to_raw_provider_response",
                "reason": "Recorded wire response was never accepted/canonicalized; see recorded_error.",
            }
        )
        return result
    if module == "reasoner":
        from x_editor.reasoner.contracts import SemanticReasoningMemo

        validator = SemanticReasoningMemo.model_validate
        source_paths = [Path(__file__).parents[1] / "reasoner/contracts.py"]
    elif module == "planner":
        from x_editor.planner.memo_qualified import validate_memo_qualified_planner_turn
        from x_editor.planner.reasoned_workflow import (
            MemoQualifiedPlannerInput,
            MemoQualifiedPlannerTurn,
        )

        validator = MemoQualifiedPlannerTurn.model_validate
        source_paths = [
            Path(__file__).parents[1] / "planner" / name
            for name in ("reasoned_workflow.py", "memo_qualified.py")
        ]
    else:
        raise ValueError("only Reasoner and Planner module contracts are supported")
    result["validator_sources"] = {
        str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
    }
    try:
        parsed = validator(module_output)
    except ValueError as error:
        result["checks"].append(
            {"check": "canonical_schema", "status": "failed", "error": str(error)}
        )
        return result
    result["checks"].append({"check": "canonical_schema", "status": "passed"})
    if module == "planner":
        try:
            planner_input = MemoQualifiedPlannerInput.model_validate(module_input)
        except ValueError as error:
            result["checks"].append(
                {"check": "input_schema", "status": "failed", "error": str(error)}
            )
            return result
        try:
            validate_memo_qualified_planner_turn(parsed, planner_input=planner_input)
        except ValueError as error:
            result["checks"].append(
                {"check": "plan_input_contract", "status": "failed", "error": str(error)}
            )
        else:
            result["checks"].append({"check": "plan_input_contract", "status": "passed"})
        result["conversion_scope"] = (
            "Canonical schema and input-bound operation/argument/dependency contracts only; "
            "tool invocation and resulting audio remain untested by this audit."
        )
        if parsed.plan is not None:
            from x_editor.runtime.execution_graph import compile_execution_graph

            try:
                graph = compile_execution_graph(parsed.plan, graph_id="atomic_review_read_only")
            except ValueError as error:
                result["checks"].append(
                    {
                        "check": "execution_graph_compilation",
                        "status": "failed",
                        "error": str(error),
                    }
                )
            else:
                result["checks"].append(
                    {"check": "execution_graph_compilation", "status": "passed"}
                )
                result["compiled_execution_graph"] = graph.model_dump(mode="json")
            graph_source = Path(__file__).parents[1] / "runtime/execution_graph.py"
            result["validator_sources"]["x_editor/runtime/execution_graph.py"] = hashlib.sha256(
                graph_source.read_bytes()
            ).hexdigest()
    return result
