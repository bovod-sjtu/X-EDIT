"""Scientific and executable contracts for one runtime operation.

Capability cards separate three facts that used to be mixed together in prose:
what an operation can do, what quality trade-offs it declares, and which exact
arguments the Planner is allowed to author. The same parameter contract is used
for prompt context and plan admission.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from x_editor.core.schemas import StrictModel

CapabilityLevel = Literal["high", "medium", "low", "unknown"]
PreservationLevel = Literal["strong", "moderate", "weak", "unknown"]
EvidenceKind = Literal[
    "declared_design_fact",
    "measured_dev_prior",
    "runtime_handler",
    "service_schema",
    "test",
    "unknown",
]
ParameterValueType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "enum",
    "object",
    "array",
]


class OperationRequirements(StrictModel):
    inputs: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class OperationPreservationCharacteristic(StrictModel):
    dimension: str = Field(min_length=1, max_length=128)
    level: PreservationLevel
    evidence_kind: EvidenceKind
    rationale: str = Field(min_length=1, max_length=1024)


class OperationQualityCharacteristic(StrictModel):
    dimension: str = Field(min_length=1, max_length=128)
    level: CapabilityLevel
    evidence_kind: EvidenceKind
    rationale: str = Field(min_length=1, max_length=1024)


class OperationArtifactRisk(StrictModel):
    risk: str = Field(min_length=1, max_length=128)
    level: CapabilityLevel
    evidence_kind: EvidenceKind
    rationale: str = Field(min_length=1, max_length=1024)


class OperationCompositionProfile(StrictModel):
    supports_batch_targets: bool | Literal["unknown"] = "unknown"
    cumulative_degradation_risk: CapabilityLevel = "unknown"


class OperationParameterSpec(StrictModel):
    """One exact plan argument that a Planner may select."""

    path: str = Field(min_length=1, max_length=256)
    value_type: ParameterValueType
    required: bool = False
    enum_values: list[Any] = Field(default_factory=list, max_length=128)
    minimum: float | None = None
    maximum: float | None = None
    min_items: int | None = Field(default=None, ge=0)
    max_items: int | None = Field(default=None, ge=0)
    default: Any | None = None
    semantic_role: str = Field(min_length=1, max_length=512)
    runtime_value_source: Literal[
        "source_audio_refs",
        "reference_audio_refs",
        "source_or_reference_audio_refs",
    ] | None = None
    value_guidance: list[str] = Field(default_factory=list, max_length=16)
    value_examples: list[Any] = Field(default_factory=list, max_length=8)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith(("inputs.", "params.")):
            raise ValueError("parameter path must start with inputs. or params.")
        if normalized.endswith(".") or ".." in normalized:
            raise ValueError("parameter path must be a dotted field path")
        return normalized

    @model_validator(mode="after")
    def validate_domain(self) -> OperationParameterSpec:
        if self.value_type == "enum" and not self.enum_values:
            raise ValueError("enum parameters require enum_values")
        if self.value_type != "enum" and self.enum_values:
            raise ValueError("enum_values are only valid for enum parameters")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if (self.minimum is not None or self.maximum is not None) and self.value_type not in {
            "number",
            "integer",
        }:
            raise ValueError("numeric bounds require number or integer value_type")
        if (
            self.min_items is not None or self.max_items is not None
        ) and self.value_type != "array":
            raise ValueError("item cardinality bounds require array value_type")
        if (
            self.min_items is not None
            and self.max_items is not None
            and self.min_items > self.max_items
        ):
            raise ValueError("min_items cannot exceed max_items")
        return self


class OperationParameterCondition(StrictModel):
    path: str = Field(min_length=1, max_length=256)
    operator: Literal["equals", "in", "present", "absent"]
    value: Any | None = None
    values: list[Any] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def validate_operator_payload(self) -> OperationParameterCondition:
        if self.operator == "equals" and self.value is None:
            raise ValueError("equals conditions require value")
        if self.operator == "in" and not self.values:
            raise ValueError("in conditions require values")
        return self


class OperationParameterConditionalRule(StrictModel):
    when: OperationParameterCondition
    require: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)


class OperationParameterContract(StrictModel):
    """Authoring boundary between Planner-selected and framework-owned fields."""

    additional_parameters: Literal["reject", "allow"] = "reject"
    planner_selectable: list[OperationParameterSpec] = Field(default_factory=list)
    framework_owned: list[str] = Field(default_factory=list)
    at_least_one_of: list[list[str]] = Field(default_factory=list)
    mutually_exclusive: list[list[str]] = Field(default_factory=list)
    conditional_rules: list[OperationParameterConditionalRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_paths(self) -> OperationParameterContract:
        selectable = [item.path for item in self.planner_selectable]
        if len(selectable) != len(set(selectable)):
            raise ValueError("planner_selectable paths must be unique")
        if len(self.framework_owned) != len(set(self.framework_owned)):
            raise ValueError("framework_owned paths must be unique")
        overlap = set(selectable).intersection(self.framework_owned)
        if overlap:
            raise ValueError(
                "parameter paths cannot be both planner-selectable and framework-owned: "
                f"{sorted(overlap)}"
            )
        known = set(selectable).union(self.framework_owned)
        for field_name, groups in (
            ("at_least_one_of", self.at_least_one_of),
            ("mutually_exclusive", self.mutually_exclusive),
        ):
            for group in groups:
                if len(group) < 2 or len(group) != len(set(group)):
                    raise ValueError(f"{field_name} groups require at least two unique paths")
                unknown = set(group) - known
                if unknown:
                    raise ValueError(
                        f"{field_name} references undeclared parameter paths: {sorted(unknown)}"
                    )
        for rule in self.conditional_rules:
            referenced = {rule.when.path, *rule.require, *rule.forbid}
            unknown = referenced - known
            if unknown:
                raise ValueError(
                    f"conditional rules reference undeclared parameter paths: {sorted(unknown)}"
                )
        return self


class OperationCapabilityProvenance(StrictModel):
    source_path: str = Field(min_length=1, max_length=1024)
    source_kind: Literal["manual", "service_schema", "runtime_handler", "test"]


class OperationCapabilityCard(StrictModel):
    """One operation's semantic, quality, and exact parameter-selection contract."""

    schema_version: Literal["x-editor.operation-capability-card/v1"] = (
        "x-editor.operation-capability-card/v1"
    )
    operation: str = Field(min_length=1, max_length=256)
    semantic_effects: list[str] = Field(min_length=1, max_length=64)
    supported_scopes: list[
        Literal[
            "global",
            "speaker",
            "segment",
            "lexical_boundary",
            "standalone_asset",
            "timeline",
        ]
    ] = Field(min_length=1)
    requirements: OperationRequirements = Field(default_factory=OperationRequirements)
    preservation: list[OperationPreservationCharacteristic] = Field(default_factory=list)
    quality_characteristics: list[OperationQualityCharacteristic] = Field(default_factory=list)
    artifact_risks: list[OperationArtifactRisk] = Field(default_factory=list)
    composition: OperationCompositionProfile = Field(default_factory=OperationCompositionProfile)
    parameter_contract: OperationParameterContract
    limitations: list[str] = Field(default_factory=list)
    provenance: list[OperationCapabilityProvenance] = Field(min_length=1)

    @field_validator(
        "semantic_effects",
        "requirements",
        "limitations",
        mode="before",
    )
    @classmethod
    def preserve_structured_values(cls, value: Any) -> Any:
        return value

    def planner_payload(self) -> dict[str, Any]:
        """Return only facts needed for semantic and legal route selection."""

        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "semantic_effects": list(self.semantic_effects),
            "supported_scopes": list(self.supported_scopes),
            "requirements": self.requirements.model_dump(mode="json"),
            "preservation": [item.model_dump(mode="json") for item in self.preservation],
            "quality_characteristics": [
                item.model_dump(mode="json") for item in self.quality_characteristics
            ],
            "artifact_risks": [item.model_dump(mode="json") for item in self.artifact_risks],
            "composition": self.composition.model_dump(mode="json"),
            "parameter_contract": self.parameter_contract.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
            ),
            "limitations": list(self.limitations),
            "provenance": [item.model_dump(mode="json") for item in self.provenance],
        }


class CapabilityContractViolation(StrictModel):
    code: str
    path: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


def validate_operation_arguments(
    card: OperationCapabilityCard,
    *,
    inputs: Mapping[str, Any],
    params: Mapping[str, Any],
    framework_values: Mapping[str, Any] | None = None,
    allow_framework_owned_without_value: bool = False,
) -> list[CapabilityContractViolation]:
    """Validate a plan step against the exact parameter domain in its card."""

    values: dict[str, Any] = {
        **{f"inputs.{key}": value for key, value in inputs.items()},
        **{f"params.{key}": value for key, value in params.items()},
    }
    contract = card.parameter_contract
    specs = {item.path: item for item in contract.planner_selectable}
    framework_paths = set(contract.framework_owned)
    framework_values = dict(framework_values or {})
    authoritative_values = {**values, **framework_values}
    violations: list[CapabilityContractViolation] = []

    for path, value in values.items():
        if path in framework_values:
            if framework_values[path] != value:
                violations.append(
                    CapabilityContractViolation(
                        code="frozen_operation_parameter_mismatch",
                        path=path,
                        message=f"{path} differs from the frozen runtime value.",
                        details={"expected": framework_values[path], "actual": value},
                    )
                )
            continue
        spec = specs.get(path)
        if spec is not None:
            violations.extend(_validate_value(spec, value))
            continue
        if path in framework_paths:
            if path not in framework_values:
                if allow_framework_owned_without_value:
                    continue
                violations.append(
                    CapabilityContractViolation(
                        code="framework_owned_parameter_authored_by_planner",
                        path=path,
                        message=(
                            f"{path} is framework-owned and cannot be selected by the Planner."
                        ),
                    )
                )
            continue
        if _covered_by_container(path, specs) or _covered_by_container(
            path, {key: None for key in framework_paths}
        ):
            continue
        if contract.additional_parameters == "reject":
            violations.append(
                CapabilityContractViolation(
                    code="undeclared_operation_parameter",
                    path=path,
                    message=(f"{path} is not declared in the operation's legal parameter set."),
                )
            )

    for spec in specs.values():
        if spec.required and spec.path not in values:
            violations.append(
                CapabilityContractViolation(
                    code="required_operation_parameter_missing",
                    path=spec.path,
                    message=f"Required operation parameter {spec.path} is missing.",
                )
            )

    for rule in contract.conditional_rules:
        if not _condition_matches(rule.when, authoritative_values):
            continue
        for path in rule.require:
            if path not in authoritative_values:
                violations.append(
                    CapabilityContractViolation(
                        code="conditional_operation_parameter_missing",
                        path=path,
                        message=f"Conditional operation parameter {path} is required.",
                    )
                )
        for path in rule.forbid:
            if path in authoritative_values:
                violations.append(
                    CapabilityContractViolation(
                        code="conditional_operation_parameter_forbidden",
                        path=path,
                        message=f"Conditional operation parameter {path} is forbidden.",
                    )
                )
    for group in contract.at_least_one_of:
        if not any(path in authoritative_values for path in group):
            violations.append(
                CapabilityContractViolation(
                    code="operation_parameter_choice_missing",
                    message="At least one parameter from the declared choice group is required.",
                    details={"paths": group},
                )
            )
    for group in contract.mutually_exclusive:
        present = [path for path in group if path in authoritative_values]
        if len(present) > 1:
            violations.append(
                CapabilityContractViolation(
                    code="operation_parameters_mutually_exclusive",
                    message="Mutually exclusive operation parameters were supplied together.",
                    details={"paths": group, "present": present},
                )
            )
    return violations


def _validate_value(
    spec: OperationParameterSpec,
    value: Any,
) -> list[CapabilityContractViolation]:
    if (
        spec.path.startswith("inputs.")
        and spec.value_type == "string"
        and _is_step_output_reference(value)
    ):
        return []
    valid_type = {
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "enum": lambda item: item in spec.enum_values,
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
    }[spec.value_type]
    if not valid_type(value):
        return [
            CapabilityContractViolation(
                code="operation_parameter_type_invalid",
                path=spec.path,
                message=(f"{spec.path} must satisfy value_type={spec.value_type}."),
                details={"actual_type": type(value).__name__},
            )
        ]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return [
                CapabilityContractViolation(
                    code="operation_parameter_not_finite",
                    path=spec.path,
                    message=f"{spec.path} must be a finite numeric value.",
                )
            ]
        if spec.minimum is not None and value < spec.minimum:
            return [
                CapabilityContractViolation(
                    code="operation_parameter_below_minimum",
                    path=spec.path,
                    message=f"{spec.path} is below its legal minimum.",
                    details={"minimum": spec.minimum, "actual": value},
                )
            ]
        if spec.maximum is not None and value > spec.maximum:
            return [
                CapabilityContractViolation(
                    code="operation_parameter_above_maximum",
                    path=spec.path,
                    message=f"{spec.path} exceeds its legal maximum.",
                    details={"maximum": spec.maximum, "actual": value},
                )
            ]
    if isinstance(value, list):
        if spec.min_items is not None and len(value) < spec.min_items:
            return [
                CapabilityContractViolation(
                    code="operation_parameter_below_min_items",
                    path=spec.path,
                    message=f"{spec.path} has fewer items than its legal minimum.",
                    details={"min_items": spec.min_items, "actual_items": len(value)},
                )
            ]
        if spec.max_items is not None and len(value) > spec.max_items:
            return [
                CapabilityContractViolation(
                    code="operation_parameter_above_max_items",
                    path=spec.path,
                    message=f"{spec.path} has more items than its legal maximum.",
                    details={"max_items": spec.max_items, "actual_items": len(value)},
                )
            ]
    return []


def _is_step_output_reference(value: Any) -> bool:
    """Return whether a plan input legally references an earlier step artifact."""

    if not isinstance(value, Mapping):
        return False
    step_id = value.get("from_step")
    output_key = value.get("from_output") or value.get("result_key")
    return bool(
        isinstance(step_id, str)
        and step_id.strip()
        and isinstance(output_key, str)
        and output_key.strip()
    )


def _covered_by_container(path: str, specs: Mapping[str, Any]) -> bool:
    for declared_path, spec in specs.items():
        if not path.startswith(f"{declared_path}."):
            continue
        if spec is None or getattr(spec, "value_type", None) in {"object", "array"}:
            return True
    return False


def _condition_matches(
    condition: OperationParameterCondition,
    values: Mapping[str, Any],
) -> bool:
    if condition.operator == "present":
        return condition.path in values
    if condition.operator == "absent":
        return condition.path not in values
    if condition.path not in values:
        return False
    if condition.operator == "equals":
        return values[condition.path] == condition.value
    return values[condition.path] in condition.values
