"""Constrained diagnostic guidance for a new, independent workflow attempt.

The contract carries questions, invariants and optional summaries of previously
supported or unresolved work. It supplies no private answer or replacement plan.
It is authored outside the forward path and becomes visible only when an
experiment controller explicitly opens a future attempt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from x_editor.core.schemas import StrictModel

FutureGuidanceTargetModule = Literal["reasoner", "planner", "perception"]
GuidanceText = Annotated[str, Field(min_length=1, max_length=2048)]
GuidanceRef = Annotated[str, Field(min_length=1, max_length=256)]


class ModuleIterationFeedback(StrictModel):
    """Episode-local review, never a substitute for fresh source evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    module: FutureGuidanceTargetModule
    supported_work: tuple[GuidanceText, ...] = Field(default=(), max_length=16)
    unresolved_work: tuple[GuidanceText, ...] = Field(default=(), max_length=16)
    next_checks: tuple[GuidanceText, ...] = Field(default=(), max_length=16)
    upstream_changes: tuple[GuidanceText, ...] = Field(default=(), max_length=16)
    preservation_requirements: tuple[GuidanceText, ...] = Field(default=(), max_length=16)
    authority: Literal["reviewed_hypotheses_rebind_to_current_evidence"] = (
        "reviewed_hypotheses_rebind_to_current_evidence"
    )


class FutureAttemptGuidance(StrictModel):
    """Questions and constraints admitted to one future workflow attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["x-editor.future-attempt-guidance/v1"] = (
        "x-editor.future-attempt-guidance/v1"
    )
    guidance_id: GuidanceRef
    source_attempt_id: GuidanceRef
    source_evaluation_result_ids: tuple[GuidanceRef, ...] = Field(min_length=1, max_length=64)
    target_modules: tuple[FutureGuidanceTargetModule, ...] = Field(
        min_length=1,
        max_length=3,
    )
    weak_rubric_ids: tuple[GuidanceRef, ...] = Field(default=(), max_length=64)
    diagnostic_evidence_refs: tuple[GuidanceRef, ...] = Field(default=(), max_length=128)
    questions_to_reconsider: tuple[GuidanceText, ...] = Field(min_length=1, max_length=32)
    constraints_to_preserve: tuple[GuidanceText, ...] = Field(default=(), max_length=32)
    source_failure_stage: GuidanceRef
    source_failure_codes: tuple[GuidanceRef, ...] = Field(default=(), max_length=64)
    module_feedback: tuple[ModuleIterationFeedback, ...] = Field(default=(), max_length=3)
    evidence_authority: Literal["diagnostic_only"] = "diagnostic_only"
    applies_to_new_attempt_only: Literal[True] = True
    contains_expected_answer: Literal[False] = False
    contains_direct_replacement_output: Literal[False] = False
    can_gate_runtime: Literal[False] = False
    can_override_verifier: Literal[False] = False
    guidance_fingerprint: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")

    @field_validator(
        "source_evaluation_result_ids",
        "target_modules",
        "weak_rubric_ids",
        "diagnostic_evidence_refs",
        "questions_to_reconsider",
        "constraints_to_preserve",
        "source_failure_codes",
        mode="before",
    )
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        if value is None:
            return ()
        return tuple(str(item).strip() for item in value if str(item).strip())

    @model_validator(mode="after")
    def freeze_guidance(self) -> FutureAttemptGuidance:
        modules = [item.module for item in self.module_feedback]
        if len(modules) != len(set(modules)) or set(modules) - set(self.target_modules):
            raise ValueError("module feedback must be unique and addressed to a target")
        for label, values in (
            ("source evaluation result", self.source_evaluation_result_ids),
            ("target module", self.target_modules),
            ("weak rubric", self.weak_rubric_ids),
            ("diagnostic evidence", self.diagnostic_evidence_refs),
            ("question", self.questions_to_reconsider),
            ("constraint", self.constraints_to_preserve),
            ("source failure code", self.source_failure_codes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} values must be unique")
        total_prompt_chars = sum(
            len(item) for item in (*self.questions_to_reconsider, *self.constraints_to_preserve)
        )
        total_prompt_chars += sum(
            len(text)
            for packet in self.module_feedback
            for key in (
                "supported_work",
                "unresolved_work",
                "next_checks",
                "upstream_changes",
                "preservation_requirements",
            )
            for text in getattr(packet, key)
        )
        if total_prompt_chars > 32768:
            raise ValueError("future-attempt guidance exceeds the prompt budget")
        excluded = {"guidance_fingerprint"}
        if not self.module_feedback:
            excluded.add("module_feedback")  # Preserve all existing v1 fingerprints.
        expected = _fingerprint(self.model_dump(mode="json", exclude=excluded))
        if self.guidance_fingerprint and self.guidance_fingerprint != expected:
            raise ValueError("guidance_fingerprint does not match guidance content")
        object.__setattr__(self, "guidance_fingerprint", expected)
        return self

    def prompt_projection(self, *, module: FutureGuidanceTargetModule) -> dict[str, Any] | None:
        """Return the bounded, non-authoritative projection for one module."""

        if module not in self.target_modules:
            return None
        projection = {
            "guidance_id": self.guidance_id,
            "source_attempt_id": self.source_attempt_id,
            "source_evaluation_result_ids": list(self.source_evaluation_result_ids),
            "weak_rubric_ids": list(self.weak_rubric_ids),
            "diagnostic_evidence_refs": list(self.diagnostic_evidence_refs),
            "questions_to_reconsider": list(self.questions_to_reconsider),
            "constraints_to_preserve": list(self.constraints_to_preserve),
            "source_failure_stage": self.source_failure_stage,
            "source_failure_codes": list(self.source_failure_codes),
            "evidence_authority": self.evidence_authority,
            "guidance_fingerprint": self.guidance_fingerprint,
        }
        own = [
            packet.model_dump(mode="json")
            for packet in self.module_feedback
            if packet.module == module
        ]
        if own:
            projection["iteration_feedback"] = own[0]
            projection["iteration_feedback_policy"] = (
                "Preserve supported work when it still follows from the current input. "
                "Resolve the listed gaps; do not change correct work just to be different. "
                "Prior findings describe another attempt, not authoritative current facts. "
                "Bind every retained fact and interval to this attempt's actual asset/version. "
                "Explain whether new targeted observations resolve or contradict old claims. "
                "Request the missing observation through the normal capability contracts."
            )
        return projection


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["FutureAttemptGuidance", "FutureGuidanceTargetModule", "ModuleIterationFeedback"]
