"""Resolve committed timelines by their input asset, independently of scheduling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from x_editor.runtime.audio_references import canonical_audio_ref_for_runtime_path
from x_editor.runtime.timeline_ledger import TimelineLedger
from x_editor.runtime.verification import VerificationArtifact

_LEDGER_KEY = "audio_timeline_ledger"


def _same_path(left: str | None, right: str | None) -> bool:
    return bool(
        left and right
        and Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    )


def timeline_for_audio_path(state: dict[str, Any], path: str) -> TimelineLedger | None:
    """Read accepted output lineage, or the identity of a registered initial asset.

    A revision number alone is not an asset identity: independent branches can
    both be at revision one. Candidate/rejected outputs never enter step_outputs.
    The current-ledger fallback preserves direct temporal-lineage callers that
    have no executor artifact registry.
    """
    for output in reversed(list((state.get("step_outputs") or {}).values())):
        if not isinstance(output, dict) or not _same_path(
            output.get("output_audio_path"), path
        ):
            continue
        payload = output.get(_LEDGER_KEY)
        return _parse_ledger(payload)
    if _same_path(path, state.get("current_audio_path")):
        payload = state.get(_LEDGER_KEY)
        if payload is not None:
            return _parse_ledger(payload)
    initial = _same_path(path, state.get("input_audio")) or any(
        isinstance(artifact, VerificationArtifact)
        and artifact.status == "available"
        and artifact.timeline_revision == 0
        and _same_path(artifact.path, path)
        for artifact in (state.get("runtime_artifact_revisions") or {}).values()
    )
    if not initial:
        return None
    try:
        info = sf.info(path)
    except (OSError, RuntimeError):
        return None
    if info.frames <= 0 or info.samplerate <= 0:
        return None
    return TimelineLedger.identity(
        base_duration_sec=info.frames / info.samplerate,
        base_audio_ref=canonical_audio_ref_for_runtime_path(path, state) or path,
    )


def evidence_timeline(
    state: dict[str, Any],
    *,
    observed_artifact_id: str | None,
    observed_revision: int | None,
    input_artifact: VerificationArtifact,
) -> TimelineLedger | None:
    """Only an identified base asset can supply coordinates to its own ledger."""
    observed = (state.get("runtime_artifact_revisions") or {}).get(observed_artifact_id)
    if (
        observed_revision != 0
        or not isinstance(observed, VerificationArtifact)
        or observed.timeline_revision != observed_revision
        or not input_artifact.path
    ):
        return None
    ledger = timeline_for_audio_path(state, input_artifact.path)
    if ledger is None or ledger.current_timebase.revision != input_artifact.timeline_revision:
        return None
    if not ledger.base_audio_ref or not (
        ledger.base_audio_ref == observed.audio_ref_id
        or _same_path(ledger.base_audio_ref, observed.path)
    ):
        return None
    return ledger


def _parse_ledger(payload: Any) -> TimelineLedger | None:
    if isinstance(payload, TimelineLedger):
        return payload
    if isinstance(payload, dict):
        return TimelineLedger.model_validate(payload)
    return None


def crosses_unreviewed_content_transition(
    state: dict[str, Any], *, observed_artifact_id: str | None,
    input_artifact: VerificationArtifact,
) -> bool:
    """A time-axis receipt cannot establish that old lexical facts still hold."""
    boundaries = state.get("runtime_semantic_evidence_boundaries") or {}
    artifacts = state.get("runtime_artifact_revisions") or {}
    pending = [input_artifact.artifact_id]
    visited: set[str] = set()
    while pending:
        identifier = pending.pop()
        if identifier == observed_artifact_id or identifier in visited:
            continue
        visited.add(identifier)
        if identifier in boundaries:
            return True
        artifact = artifacts.get(identifier)
        if isinstance(artifact, VerificationArtifact):
            pending.extend(artifact.parent_artifact_ids)
    return False
