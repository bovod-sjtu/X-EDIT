"""Versioned policy profiles for workflow protocol admission.

This module only defines policy data. Planner, perception, and admission
components consume the resolved policy without duplicating workflow control.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator

from x_editor.core.schemas import StrictModel

WorkflowProtocolProfileName: TypeAlias = Literal[
    "audit_strict",
    "typed_balanced",
    "flow_safe",
]
EventLabelRequirement: TypeAlias = Literal["conditional"]
SliceShortfallMode: TypeAlias = Literal["block", "warn"]
ReviewableUncertaintyMode: TypeAlias = Literal["block", "admit"]
FrameworkOwnedPlanFieldsMode: TypeAlias = Literal[
    "block",
    "strip_and_audit",
]
BindingNormalizationMode: TypeAlias = Literal[
    "explicit_only",
    "canonical_alias",
    "exact_unique",
]
PerceptionRouteNormalizationMode: TypeAlias = Literal[
    "explicit_only",
    "exact_unique",
]
IrrelevantBindingMode: TypeAlias = Literal["validate", "drop"]
EvidenceRequirementScope: TypeAlias = Literal[
    "all_edits",
    "reviewed_span_ops",
]
WorkflowProtocolHardInvariant: TypeAlias = Literal[
    "unknown_operation",
    "audio_ref_outside_workflow_inputs",
    "unknown_evidence_id_or_path",
    "direct_evidence_value_mismatch",
    "ambiguous_alias_or_value_inference",
    "unresolved_required_semantic_params",
    "local_destructive_edit_without_grounded_boundaries",
]

WORKFLOW_PROTOCOL_PROFILE_NAMES: tuple[WorkflowProtocolProfileName, ...] = (
    "audit_strict",
    "typed_balanced",
    "flow_safe",
)
DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME: WorkflowProtocolProfileName = "flow_safe"
WORKFLOW_PROTOCOL_HARD_INVARIANTS: tuple[
    WorkflowProtocolHardInvariant, ...
] = (
    "unknown_operation",
    "audio_ref_outside_workflow_inputs",
    "unknown_evidence_id_or_path",
    "direct_evidence_value_mismatch",
    "ambiguous_alias_or_value_inference",
    "unresolved_required_semantic_params",
    "local_destructive_edit_without_grounded_boundaries",
)


class WorkflowProtocolPolicy(StrictModel):
    """Typed policy knobs shared by workflow protocol components."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    schema_version: Literal["x-editor.workflow-protocol-policy/v2"] = (
        "x-editor.workflow-protocol-policy/v2"
    )
    profile_name: WorkflowProtocolProfileName = (
        DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME
    )
    event_label_requirement: EventLabelRequirement = "conditional"
    slice_shortfall_mode: SliceShortfallMode = "warn"
    reviewable_uncertainty_mode: ReviewableUncertaintyMode = "admit"
    framework_owned_plan_fields_mode: FrameworkOwnedPlanFieldsMode = (
        "strip_and_audit"
    )
    perception_route_normalization_mode: PerceptionRouteNormalizationMode = (
        "exact_unique"
    )
    binding_normalization_mode: BindingNormalizationMode = "exact_unique"
    irrelevant_binding_mode: IrrelevantBindingMode = "drop"
    evidence_requirement_scope: EvidenceRequirementScope = "reviewed_span_ops"
    hard_invariants: tuple[WorkflowProtocolHardInvariant, ...] = Field(
        default=WORKFLOW_PROTOCOL_HARD_INVARIANTS,
        min_length=1,
    )
    description: str = Field(
        default=(
            "Maximize safe workflow flow by dropping bindings for steps that "
            "cannot consume reviewed span evidence, while retaining shared "
            "hard validation for the executable plan and evidence-driving "
            "bindings."
        ),
        min_length=1,
    )

    @field_validator("hard_invariants")
    @classmethod
    def validate_hard_invariants(
        cls,
        value: tuple[WorkflowProtocolHardInvariant, ...],
    ) -> tuple[WorkflowProtocolHardInvariant, ...]:
        """Keep hard safety boundaries independent from profile selection."""

        if value != WORKFLOW_PROTOCOL_HARD_INVARIANTS:
            raise ValueError(
                "hard_invariants must match the shared workflow invariant set"
            )
        return value


_WORKFLOW_PROTOCOL_PROFILES: Mapping[
    WorkflowProtocolProfileName,
    WorkflowProtocolPolicy,
] = MappingProxyType(
    {
        "audit_strict": WorkflowProtocolPolicy(
            profile_name="audit_strict",
            slice_shortfall_mode="block",
            reviewable_uncertainty_mode="block",
            framework_owned_plan_fields_mode="block",
            perception_route_normalization_mode="explicit_only",
            binding_normalization_mode="explicit_only",
            irrelevant_binding_mode="validate",
            evidence_requirement_scope="all_edits",
            description=(
                "Require explicit, fully auditable declarations and block "
                "reviewable uncertainty before execution."
            ),
        ),
        "typed_balanced": WorkflowProtocolPolicy(
            profile_name="typed_balanced",
            slice_shortfall_mode="warn",
            reviewable_uncertainty_mode="admit",
            framework_owned_plan_fields_mode="strip_and_audit",
            perception_route_normalization_mode="exact_unique",
            binding_normalization_mode="canonical_alias",
            irrelevant_binding_mode="validate",
            evidence_requirement_scope="reviewed_span_ops",
            description=(
                "Admit typed uncertainty and canonical aliases while retaining "
                "validation of every declared evidence binding."
            ),
        ),
        "flow_safe": WorkflowProtocolPolicy(
            profile_name="flow_safe",
            slice_shortfall_mode="warn",
            reviewable_uncertainty_mode="admit",
            framework_owned_plan_fields_mode="strip_and_audit",
            perception_route_normalization_mode="exact_unique",
            binding_normalization_mode="exact_unique",
            irrelevant_binding_mode="drop",
            evidence_requirement_scope="reviewed_span_ops",
            description=(
                "Maximize safe workflow flow by dropping bindings for steps "
                "that cannot consume reviewed span evidence, while retaining "
                "shared hard validation for the executable plan and "
                "evidence-driving bindings."
            ),
        ),
    }
)


def list_workflow_protocol_profile_names() -> tuple[
    WorkflowProtocolProfileName, ...
]:
    """Return profile names in their stable review order."""

    return WORKFLOW_PROTOCOL_PROFILE_NAMES


def resolve_workflow_protocol_policy(
    name_or_policy: (
        str | Mapping[str, object] | WorkflowProtocolPolicy | None
    ) = None,
) -> WorkflowProtocolPolicy:
    """Resolve a named profile or return an already validated policy object."""

    if name_or_policy is None:
        return _WORKFLOW_PROTOCOL_PROFILES[
            DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME
        ]
    if isinstance(name_or_policy, WorkflowProtocolPolicy):
        return name_or_policy
    if isinstance(name_or_policy, Mapping):
        return WorkflowProtocolPolicy.model_validate(dict(name_or_policy))
    if not isinstance(name_or_policy, str):
        raise TypeError(
            "workflow protocol policy must be a profile name, mapping, or "
            "WorkflowProtocolPolicy"
        )
    try:
        return _WORKFLOW_PROTOCOL_PROFILES[name_or_policy]
    except KeyError as exc:
        available = ", ".join(WORKFLOW_PROTOCOL_PROFILE_NAMES)
        raise ValueError(
            f"unknown workflow protocol profile {name_or_policy!r}; "
            f"expected one of: {available}"
        ) from exc


__all__ = [
    "DEFAULT_WORKFLOW_PROTOCOL_PROFILE_NAME",
    "WORKFLOW_PROTOCOL_HARD_INVARIANTS",
    "WORKFLOW_PROTOCOL_PROFILE_NAMES",
    "WorkflowProtocolPolicy",
    "WorkflowProtocolProfileName",
    "list_workflow_protocol_profile_names",
    "resolve_workflow_protocol_policy",
]
