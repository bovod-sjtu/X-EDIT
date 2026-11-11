"""Contracts for pure text/symbolic semantic transforms.

These contracts are intentionally independent from runtime tool payloads. A
semantic transform describes what text-level change was inferred from the user
request and evidence; runtime projection is responsible for turning it into a
specific tool contract such as ``edit_data_pipeline.text_edit``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from x_editor.core.schemas import StrictModel


SemanticTransformType = Literal[
    "text_replacement",
    "arithmetic_transform",
    "coreference_restore",
    "entity_extraction",
    "symbolic_transform",
    "natural_language_transform",
    "unsupported",
]

OccurrencePolicy = Literal["first", "all", "explicit", "unsupported"]


class SemanticTransform(StrictModel):
    """Auditable pure semantic transform before tool-contract projection."""

    transform_id: str
    type: SemanticTransformType
    source_units: list[str] = Field(default_factory=list)
    target_units: list[str] = Field(default_factory=list)
    occurrence_policy: OccurrencePolicy = "first"
    confidence: float = 0.0
    rationale: str = ""
    required_evidence: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    projection_hints: dict[str, Any] = Field(default_factory=dict)
    unsupported_reason: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.type != "unsupported" and not self.unsupported_reason
