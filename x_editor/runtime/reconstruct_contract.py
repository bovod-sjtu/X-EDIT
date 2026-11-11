"""Shared typed values for artifact timeline reconstruction."""

from __future__ import annotations

from typing import Literal

ReconstructionMode = Literal[
    "timeline_edits",
    "span_replacement",
    "full_timeline_passthrough",
    "placeholder_compatibility",
]
EditedArtifactScope = Literal["local_regions", "full_timeline", "none"]

RECONSTRUCTION_MODES: set[str] = {
    "timeline_edits",
    "span_replacement",
    "full_timeline_passthrough",
    "placeholder_compatibility",
}
EDITED_ARTIFACT_SCOPES: set[str] = {"local_regions", "full_timeline", "none"}


def normalized_reconstruction_mode(value: object) -> str | None:
    return _normalized_choice(value, RECONSTRUCTION_MODES)


def normalized_edited_artifact_scope(value: object) -> str | None:
    aliases = {
        "full": "full_timeline",
        "whole_timeline": "full_timeline",
        "local": "local_regions",
        "local_region": "local_regions",
        "local_span": "local_regions",
        "local_spans": "local_regions",
    }
    text = _normalized_text(value)
    if text in aliases:
        text = aliases[text]
    return text if text in EDITED_ARTIFACT_SCOPES else None


def _normalized_choice(value: object, allowed: set[str]) -> str | None:
    text = _normalized_text(value)
    return text if text in allowed else None


def _normalized_text(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")
