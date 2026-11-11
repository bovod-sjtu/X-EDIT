"""Catalog feedback for a bounded Planner correction, never audio evidence."""

from typing import Any, Literal

from pydantic import Field

from x_editor.core.schemas import StrictModel


class PerceptionRouteAlternative(StrictModel):
    operation: str = Field(min_length=1)
    expected_evidence_fields: list[str] = Field(default_factory=list)
    target_kinds: list[str] = Field(default_factory=list)


class PerceptionCapabilityFeedback(StrictModel):
    schema_version: Literal["x-editor.perception-capability-feedback/v1"] = (
        "x-editor.perception-capability-feedback/v1"
    )
    code: Literal["ambiguous_observation_decomposition", "observation_target_exceeds_output_contract"]
    evidence_role: Literal["capability_catalog_only"] = "capability_catalog_only"
    requested_job: dict[str, Any]
    alternatives: list[list[PerceptionRouteAlternative]] = Field(min_length=1)
    instruction: str = (
        "The catalog can collect this evidence, but the combined observation job has "
        "multiple valid assignments of targets to operations. Split it into explicit "
        "jobs using the listed alternatives and the original observation objective. "
        "Keep the selected audio, user goal and required evidence unchanged. This is "
        "catalog feedback, not an observation or evidence that a target is absent."
    )
