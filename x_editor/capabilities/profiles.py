"""Versioned runtime capability-profile contracts.

Profiles in this module describe only newly admitted operations. A profile may
extend a frozen predecessor without copying or mutating that predecessor's
operation list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from x_editor.core.schemas import StrictModel


class CapabilityProfileCanaryProbe(StrictModel):
    method: Literal["GET", "POST"] = "POST"
    relative_path: str
    request_json: dict[str, Any]
    required_response_json_paths: list[str] = Field(default_factory=list)


class CapabilityProfileServiceRequirement(StrictModel):
    service_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    capability_id: str
    endpoint_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    canary: CapabilityProfileCanaryProbe

    @field_validator("capability_id")
    @classmethod
    def validate_capability_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("x-editor/tool/") or not normalized.endswith("/v1"):
            raise ValueError("tool capability ids must be versioned x-editor/tool contracts")
        return normalized


class CapabilityProfileOperationAdmission(StrictModel):
    operation: str
    parameter_contract: Literal["exact"] = "exact"
    admitted_semantic_effects: list[str] = Field(min_length=1)
    verifier_kind: Literal["operation_principle", "delivery_contract"]
    verifier_check_ids: list[str] = Field(min_length=1)
    required_service_ids: list[str] = Field(default_factory=list)


class RuntimeCapabilityProfileContract(StrictModel):
    schema_version: Literal["x-editor.runtime-capability-profile-contract/v1"] = (
        "x-editor.runtime-capability-profile-contract/v1"
    )
    profile_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    extends: str
    operations: list[CapabilityProfileOperationAdmission] = Field(min_length=1)
    services: list[CapabilityProfileServiceRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> RuntimeCapabilityProfileContract:
        operation_ids = [item.operation for item in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("profile operation admissions must be unique")
        service_ids = [item.service_id for item in self.services]
        if len(service_ids) != len(set(service_ids)):
            raise ValueError("profile service requirements must be unique")
        known_services = set(service_ids)
        for item in self.operations:
            unknown = set(item.required_service_ids) - known_services
            if unknown:
                raise ValueError(
                    f"operation {item.operation!r} references unknown services: {sorted(unknown)}"
                )
        return self

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(item.operation for item in self.operations)

    @property
    def service_capability_map(self) -> dict[str, tuple[str, ...]]:
        return {
            item.service_id: (item.capability_id,)
            for item in self.services
        }


@lru_cache(maxsize=8)
def load_runtime_capability_profile_contract(
    profile_name: str,
) -> RuntimeCapabilityProfileContract:
    """Load and repository-validate one packaged capability profile."""

    if not profile_name or any(token in profile_name for token in ("/", "\\", "..")):
        raise ValueError("capability profile name must be a simple identifier")
    path = Path(__file__).with_name("profiles") / f"{profile_name}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown runtime capability profile contract: {profile_name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = RuntimeCapabilityProfileContract.model_validate(raw)
    if contract.profile_name != profile_name:
        raise ValueError("capability profile filename and profile_name differ")
    _validate_against_repository_contracts(contract)
    return contract


def _validate_against_repository_contracts(
    contract: RuntimeCapabilityProfileContract,
) -> None:
    from x_editor.capabilities.catalog import load_operation_capability_catalog
    from x_editor.workflow_control.artifact_acceptance import (
        DELIVERY_CONTRACT_CHECK_IDS,
        DELIVERY_CONTRACT_OPERATIONS,
        OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS,
    )

    catalog = load_operation_capability_catalog()
    service_ids = {item.service_id for item in contract.services}
    for admission in contract.operations:
        card = catalog.get(admission.operation)
        if card is None:
            raise ValueError(f"profile operation has no typed card: {admission.operation}")
        if card.parameter_contract.additional_parameters != "reject":
            raise ValueError(
                f"profile operation does not reject undeclared parameters: {admission.operation}"
            )
        opaque_objects = [
            item.path
            for item in card.parameter_contract.planner_selectable
            if item.value_type == "object"
        ]
        if opaque_objects:
            raise ValueError(
                f"exact profile operation exposes untyped object parameters: "
                f"{admission.operation}: {opaque_objects}"
            )
        unknown_effects = set(admission.admitted_semantic_effects) - set(
            card.semantic_effects
        )
        if unknown_effects:
            raise ValueError(
                f"profile admits effects absent from {admission.operation} card: "
                f"{sorted(unknown_effects)}"
            )
        if admission.verifier_kind == "operation_principle":
            expected_checks = OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS.get(
                admission.operation
            )
        else:
            expected_checks = (
                DELIVERY_CONTRACT_CHECK_IDS
                if admission.operation in DELIVERY_CONTRACT_OPERATIONS
                else None
            )
        if tuple(admission.verifier_check_ids) != tuple(expected_checks or ()):
            raise ValueError(
                f"profile verifier contract differs from runtime authority for "
                f"{admission.operation}"
            )
        if not set(admission.required_service_ids) <= service_ids:
            raise ValueError(f"profile service closure is incomplete for {admission.operation}")


__all__ = [
    "CapabilityProfileCanaryProbe",
    "CapabilityProfileOperationAdmission",
    "CapabilityProfileServiceRequirement",
    "RuntimeCapabilityProfileContract",
    "load_runtime_capability_profile_contract",
]
