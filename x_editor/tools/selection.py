"""Explicit deployment selection shared by discovery, planning and execution."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Literal
import warnings

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


@lru_cache(maxsize=1)
def selection_contract() -> dict:
    contract = yaml.safe_load(Path(__file__).with_suffix(".yaml").read_text())
    known = set(contract["tools"])
    if set(contract["services"]) != known:
        raise ValueError("Every selectable tool must declare its deployment services")
    for operation, dependencies in contract["operations"].items():
        if not set(dependencies) <= known:
            raise ValueError(f"Unknown deployment dependencies for {operation}")
    return contract


class ToolSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["x-edit.tools/v1"] = "x-edit.tools/v1"
    edit_backend: Literal["dots_tts_edit", "edit_data_pipeline"] = "dots_tts_edit"
    enabled_tools: tuple[str, ...]

    @field_validator("enabled_tools")
    @classmethod
    def validate_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(value) - set(selection_contract()["tools"])
        if unknown:
            raise ValueError(f"Unknown enabled_tools: {sorted(unknown)}")
        if len(value) != len(set(value)):
            raise ValueError("enabled_tools must be unique")
        return value

    def operations(self) -> set[str]:
        enabled = set(self.enabled_tools)
        return {operation for operation, required in selection_contract()["operations"].items()
                if set(required) <= enabled}

    def catalog_ids(self) -> set[str]:
        contract = selection_contract()
        catalogs = {catalog for tool in self.enabled_tools for catalog in contract["tools"][tool]}
        catalogs.update(tool for tool, operation in contract["composite_cards"].items()
                        if operation in self.operations())
        return catalogs

    def deployment_services(self) -> list[str]:
        contract = selection_contract()
        services = {"planner"}
        for tool in self.enabled_tools:
            services.update(contract["services"][tool])
        if "edit_data_pipeline" in self.enabled_tools:
            services.update(contract["edit_services"][self.edit_backend])
        return sorted(services)

    def allows_call(self, tool_id: str, arguments: dict) -> bool:
        if tool_id not in self.catalog_ids():
            return False
        if tool_id == "edit_data_pipeline":
            return f"edit_data_pipeline.{arguments.get('operation', '')}" in self.operations()
        return True

    def project_profile(self, profile):
        allowed = self.operations()
        payload = profile.model_dump(mode="json")
        payload["runtime_capability_profile"] = "custom"
        payload["allowed_edit_operations"] = [operation for operation in profile.allowed_edit_operations
                                              if operation in allowed]
        payload["allowed_perception_operations"] = [operation for operation in profile.allowed_perception_operations
                                                    if operation in allowed]
        for field in ("edit_operation_params", "perception_operation_params", "perception_operation_providers"):
            payload[field] = {operation: value for operation, value in payload[field].items()
                              if operation in allowed}
        payload["evidence_required_operations"] = [operation for operation in profile.evidence_required_operations
                                                   if operation in allowed]
        payload["perception_route_policies"] = [policy for policy in payload["perception_route_policies"]
                                                 if policy["operation"] in allowed]
        if profile.tool_prior_snapshot is not None:
            prior_operations = {
                operation
                for guidance in profile.tool_prior_snapshot.guidance
                for route in guidance.routes
                for operation in route.operations
            }
            missing = prior_operations - set(payload["allowed_edit_operations"])
            if missing:
                warnings.warn(
                    "Frozen tool-prior guidance is disabled because its comparison requires "
                    f"unselected operations: {sorted(missing)}. Supply a separately validated "
                    "and frozen snapshot for this tool selection, or explicitly omit the prior.",
                    UserWarning,
                    stacklevel=2,
                )
                payload["tool_prior_snapshot"] = None
                payload["tool_prior_binding"] = None
        return type(profile).model_validate(payload)


_SELECTION: ContextVar[ToolSelection | None] = ContextVar("x_edit_tool_selection", default=None)


def current_tool_selection() -> ToolSelection | None:
    return _SELECTION.get()


@contextmanager
def tool_selection_context(selection: ToolSelection | None):
    token = _SELECTION.set(selection)
    try:
        yield
    finally:
        _SELECTION.reset(token)
