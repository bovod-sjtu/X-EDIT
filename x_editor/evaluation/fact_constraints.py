"""Auditable bounds from declared measurement requirements, never from prose.

These predicates check judge-visible facts. They neither infer the user's intent
nor supply a replacement model answer. Missing or malformed facts stay unknown.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FactConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    path: tuple[str, ...] = Field(min_length=1)
    operator: Literal["present", "finite_number", "equal", "at_most", "at_least"]
    expected: bool | int | float | str | None = None
    expected_path: tuple[str, ...] = ()
    allow_empty: bool = False
    purpose: Literal["obligation", "evidence_consistency"] = "obligation"
    violation_grade_ceiling: int = Field(default=1, ge=0, le=2)

    @model_validator(mode="after")
    def validate_operands(self):
        comparison = self.operator in {"equal", "at_most", "at_least"}
        if comparison and (self.expected is None) == (not self.expected_path):
            raise ValueError("comparison needs exactly one literal or evidence operand")
        if not comparison and (self.expected is not None or self.expected_path):
            raise ValueError("presence/type checks do not accept comparison operands")
        if isinstance(self.expected, float) and not math.isfinite(self.expected):
            raise ValueError("constraint operand must be finite")
        if "*" in self.expected_path:
            raise ValueError("comparison operand must resolve to one value")
        return self


def _values(value: Any, path: tuple[str, ...]) -> list[Any]:
    if not path:
        return [value]
    head, *tail = path
    if head == "*":
        if not isinstance(value, list):
            raise TypeError("wildcard requires an array")
        return [item for child in value for item in _values(child, tuple(tail))]
    if not isinstance(value, Mapping):
        raise TypeError("field path requires an object")
    return _values(value[head], tuple(tail))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_fact_constraints(
    constraints: tuple[FactConstraint, ...], evidence: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Check immutable evidence objects exposing payload/evidence_state/visibility."""
    checks = []
    for constraint in constraints:
        row = {"constraint_id": constraint.constraint_id,
               "purpose": constraint.purpose,
               "evidence_id": constraint.evidence_id, "path": list(constraint.path),
               "status": "unknown", "violation_grade_ceiling": constraint.violation_grade_ceiling}
        item = evidence.get(constraint.evidence_id)
        if item is None or item.evidence_state != "available" or item.visibility != "judge":
            row["reason"] = "required_evidence_unavailable"
            checks.append(row)
            continue
        try:
            values = _values(item.payload, constraint.path)
            expected = (_values(item.payload, constraint.expected_path)[0]
                        if constraint.expected_path else constraint.expected)
        except (KeyError, TypeError, IndexError):
            row["reason"] = "required_fact_missing_or_wrong_shape"
            checks.append(row)
            continue
        row.update(observed_values=values, expected=expected)
        if not values and not constraint.allow_empty:
            row["reason"] = "required_fact_array_empty"
        elif any(v is None for v in values):
            row["reason"] = "required_fact_null"
        elif constraint.expected_path and expected is None:
            row["reason"] = "required_comparison_operand_null"
        elif constraint.operator == "equal" and not (
            isinstance(expected, (str, bool)) or _number(expected)
        ):
            row["reason"] = "required_comparison_operand_invalid"
        elif constraint.operator == "equal" and any(
            (isinstance(expected, bool) and not isinstance(v, bool))
            or (_number(expected) and not _number(v))
            or (isinstance(expected, str) and not isinstance(v, str))
            for v in values
        ):
            row["reason"] = "required_fact_type_mismatch"
        elif constraint.operator in {"finite_number", "at_most", "at_least"} and (
            not all(_number(v) for v in values)
            or (constraint.operator != "finite_number" and not _number(expected))
        ):
            row["reason"] = "required_numeric_fact_invalid"
        else:
            if constraint.operator in {"present", "finite_number"}:
                satisfied = True
            elif constraint.operator == "equal":
                # Booleans cannot masquerade as numeric measurements.
                satisfied = all(v == expected and isinstance(v, bool) == isinstance(expected, bool)
                                for v in values)
            elif constraint.operator == "at_most":
                satisfied = all(v <= expected for v in values)
            else:
                satisfied = all(v >= expected for v in values)
            inconsistent = not satisfied and constraint.purpose == "evidence_consistency"
            row.update(status="satisfied" if satisfied else "unknown" if inconsistent else "violated",
                       reason="evidence_binding_inconsistent" if inconsistent else "declared_fact_constraint_checked")
        checks.append(row)
    return tuple(checks)
