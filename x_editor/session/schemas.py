"""Prototype schemas for edit-session records, refs, and event logs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "x-editor.normalized_session_record.v1"
VALIDATION_SCHEMA_VERSION = "x-editor.session_validation_report.v1"
EVENT_LOG_SCHEMA_VERSION = "x-editor.session_event_log.v1"


class SessionModel(BaseModel):
    """Base model for sidecar session contracts."""

    model_config = ConfigDict(extra="forbid")


class ArtifactRef(SessionModel):
    """Stable reference to an audio artifact or graph audio asset."""

    ref_id: str
    path: str
    role: str
    kind: str
    node_id: str | None = None
    asset_id: str | None = None
    audio_id: str | None = None
    round_index: int | None = None
    step_id: str | None = None
    operation: str | None = None
    operation_graph_node_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
    provenance: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpanRef(SessionModel):
    """Stable reference to a selected timeline span."""

    span_id: str
    audio_ref_id: str | None = None
    coordinate_space: str = "audio_timeline_seconds"
    start_sec: float | None = None
    end_sec: float | None = None
    speaker_id: str | None = None
    text: str | None = None
    role: str | None = None
    source: str | None = None
    round_index: int | None = None
    step_id: str | None = None
    operation: str | None = None
    operation_graph_node_id: str | None = None
    provenance: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObjectiveRef(SessionModel):
    """Planner objective with normalized coverage links."""

    objective_id: str
    objective: str
    type: str | None = None
    required: bool = True
    source_instruction_fragment: str | None = None
    round_index: int | None = None
    covered_by_steps: list[str] = Field(default_factory=list)
    status: str | None = None
    coverage_reason: str | None = None
    artifact_ref_ids: list[str] = Field(default_factory=list)
    span_ref_ids: list[str] = Field(default_factory=list)
    covered_by_operation_node_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceTextRef(SessionModel):
    """Source text snapshot tied to an audio ref and provenance."""

    ref_id: str
    text_hash: str
    text_preview: str
    audio_ref_id: str | None = None
    source_audio_path: str | None = None
    source: str
    provenance: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionEvent(SessionModel):
    """Append-only event-log projection for replay and debugging."""

    event_id: str
    event_type: str
    timestamp: str | None = None
    round_index: int | None = None
    node_id: str | None = None
    edge_id: str | None = None
    step_id: str | None = None
    actor: str | None = None
    summary: str | None = None
    refs: dict[str, list[str]] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionEventLog(SessionModel):
    """Append-only event log for replay/debug projections."""

    schema_version: str = EVENT_LOG_SCHEMA_VERSION
    event_log_id: str
    session_id: str | None = None
    round_index: int | None = None
    events: list[SessionEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(SessionModel):
    """One consistency issue in a normalized session record."""

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    location: str | None = None
    ref_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(SessionModel):
    """Offline validation report for session graph/data projections."""

    schema_version: str = VALIDATION_SCHEMA_VERSION
    valid: bool
    issue_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)


class NormalizedSessionRecord(SessionModel):
    """Normalized sidecar record projected from graph, plan, and evidence."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    current_node_id: str | None = None
    current_audio_ref_id: str | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    span_refs: list[SpanRef] = Field(default_factory=list)
    objective_refs: list[ObjectiveRef] = Field(default_factory=list)
    source_text_refs: list[SourceTextRef] = Field(default_factory=list)
    events: list[SessionEvent] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
