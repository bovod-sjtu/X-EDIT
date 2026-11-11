"""Static admission preflight for versioned runtime capability profiles."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from x_editor.capabilities.catalog import (
    OperationCapabilityCatalog,
    load_operation_capability_catalog,
)
from x_editor.capabilities.profiles import load_runtime_capability_profile_contract
from x_editor.core.schemas import StrictModel


class CapabilityOperationPreflight(StrictModel):
    operation: str
    handler_registered: bool
    card_available: bool
    exact_parameter_contract: bool
    verifier_contract_available: bool
    service_preflight_declared: bool
    ready: bool
    issues: list[str] = Field(default_factory=list)


class CapabilityProfilePreflightReport(StrictModel):
    schema_version: str = "x-editor.capability-profile-preflight/v1"
    profile_name: str
    ready: bool
    operations: list[CapabilityOperationPreflight]


def preflight_runtime_capability_profile(
    profile_name: str,
    *,
    registered_operations: Iterable[str],
    catalog: OperationCapabilityCatalog | None = None,
) -> CapabilityProfilePreflightReport:
    """Check handler/card/parameter/verifier/service closure without running tools."""

    contract = load_runtime_capability_profile_contract(profile_name)
    catalog = catalog or load_operation_capability_catalog()
    registered = set(registered_operations)
    declared_services = {item.service_id for item in contract.services}
    rows: list[CapabilityOperationPreflight] = []
    for admission in contract.operations:
        card = catalog.get(admission.operation)
        handler_registered = admission.operation in registered
        card_available = card is not None
        exact_parameter_contract = bool(
            card
            and card.parameter_contract.additional_parameters == "reject"
            and not any(
                item.value_type == "object"
                for item in card.parameter_contract.planner_selectable
            )
        )
        verifier_contract_available = bool(admission.verifier_check_ids)
        service_preflight_declared = set(admission.required_service_ids) <= declared_services
        issues: list[str] = []
        if not handler_registered:
            issues.append("runtime_handler_missing")
        if not card_available:
            issues.append("typed_capability_card_missing")
        elif not exact_parameter_contract:
            issues.append("parameter_contract_not_exact")
        if not verifier_contract_available:
            issues.append("verifier_contract_missing")
        if not service_preflight_declared:
            issues.append("service_preflight_missing")
        rows.append(
            CapabilityOperationPreflight(
                operation=admission.operation,
                handler_registered=handler_registered,
                card_available=card_available,
                exact_parameter_contract=exact_parameter_contract,
                verifier_contract_available=verifier_contract_available,
                service_preflight_declared=service_preflight_declared,
                ready=not issues,
                issues=issues,
            )
        )
    return CapabilityProfilePreflightReport(
        profile_name=profile_name,
        ready=all(item.ready for item in rows),
        operations=rows,
    )


def require_runtime_capability_profile_ready(
    profile_name: str,
    *,
    registered_operations: Iterable[str],
) -> CapabilityProfilePreflightReport:
    report = preflight_runtime_capability_profile(
        profile_name,
        registered_operations=registered_operations,
    )
    if not report.ready:
        failures = {
            item.operation: item.issues
            for item in report.operations
            if not item.ready
        }
        raise ValueError(f"runtime capability profile preflight failed: {failures}")
    return report


__all__ = [
    "CapabilityOperationPreflight",
    "CapabilityProfilePreflightReport",
    "preflight_runtime_capability_profile",
    "require_runtime_capability_profile_ready",
]
