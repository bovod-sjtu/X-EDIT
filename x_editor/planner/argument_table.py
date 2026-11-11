"""Lossless, opt-in column names shared across Planner argument contracts.

Only the presentation changes. Every operation, argument owner, required flag,
schema, description and limitation remains available; runtime validation still
uses the original contracts. No task-dependent shortlist or defaults are added.
"""

from collections import Counter
from copy import deepcopy
import json
from typing import Any, Literal

ArgumentContractFormat = Literal["expanded", "argument-table/v1"]
_CATALOGS = ("candidate_operation_contracts", "direct_execution_operation_contracts")


def _arguments(payload):
    for catalog in _CATALOGS:
        for operation in payload.get(catalog, []):
            for arguments in operation.get("step_argument_contract", {}).values():
                for name, specification in arguments.items():
                    yield arguments, name, specification


def factor_argument_contracts(payload: dict[str, Any]) -> dict[str, Any]:
    """Factor repeated field names when the whole encoded payload is shorter."""
    if "argument_contract_encoding" in payload:
        raise ValueError("Argument contracts are already encoded")
    result = deepcopy(payload)
    rows = list(_arguments(result))
    if not rows:
        return result
    if any(not isinstance(spec, dict) for _, _, spec in rows):
        raise ValueError("Expanded argument specifications must be objects")
    # Other shapes remain explicit objects: absent fields are never changed to
    # null/default, and future schema extensions survive the projection.
    columns = Counter(tuple(sorted(spec)) for _, _, spec in rows).most_common(1)[0][0]
    for container, name, spec in rows:
        if tuple(sorted(spec)) == columns:
            container[name] = [spec[column] for column in columns]
    result["argument_contract_encoding"] = {
        "encoding": "argument-table/v1",
        "columns": list(columns),
        "decode": (
            "Within step_argument_contract, an array is one argument specification: "
            "zip columns with its values. An object is already expanded. Resolve the "
            "required flag, value_schema and other fields before composing a call. "
            "Argument names and inputs/params ownership are unchanged. This encoding "
            "describes tool contracts only; emit ordinary literal PlanIR argument values, "
            "never these specification arrays. No fields or defaults are inferred."
        ),
    }
    size = lambda value: len(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return result if size(result) < size(payload) else deepcopy(payload)


def expand_argument_contracts(payload: dict[str, Any]) -> dict[str, Any]:
    """Exact audit inverse; never applied to model-authored executable arguments."""
    result = deepcopy(payload)
    encoding = result.pop("argument_contract_encoding", None)
    if encoding is None:
        return result
    columns = encoding.get("columns", [])
    if (encoding.get("encoding") != "argument-table/v1" or not columns
            or any(not isinstance(column, str) for column in columns)
            or len(columns) != len(set(columns))):
        raise ValueError("Invalid argument contract encoding")
    for container, name, spec in _arguments(result):
        if isinstance(spec, list):
            if len(spec) != len(columns):
                raise ValueError("Argument row does not match its columns")
            container[name] = dict(zip(columns, spec, strict=True))
        elif not isinstance(spec, dict):
            raise ValueError("Invalid argument specification")
    return result
