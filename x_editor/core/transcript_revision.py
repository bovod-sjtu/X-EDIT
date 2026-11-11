"""Provenance for an explicit re-recognition of an immutable audio asset.

An observation revision replaces a transcript, never edits the recording. Its
parent references make this different from arbitrarily preferring the latest
report. Raw observations remain available even after a revision is selected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

REVISION_OPERATION = "perception.speech_context"
REVISION_SCHEMA = "x-editor.transcript-observation-revision/v1"


def transcript_digest(text: str) -> str:
    """Bind exact observed wording; this is not a lexical similarity rule."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bind_transcript_revision(
    report: dict[str, Any],
    *,
    context: Mapping[str, Any],
    audio_ref: str,
    request_id: str,
) -> dict[str, Any] | None:
    """Build provenance after a successful, explicitly contextual ASR call.

    The caller verifies the operation and its service receipt. This function
    only binds the prior currently projected for this same audio to the new
    complete transcript. It never derives wording from a request or a hint.
    """
    previous = context.get("perception_report") or {}
    prefix = "perception_report"
    if not isinstance(previous, Mapping):
        return None
    if previous.get("audio_reports"):
        matches = [
            (index, item.get("report"))
            for index, item in enumerate(previous["audio_reports"])
            if isinstance(item, Mapping) and item.get("audio_ref_id") == audio_ref
        ]
        if len(matches) != 1 or not isinstance(matches[0][1], Mapping):
            return None
        index, previous = matches[0]
        prefix = f"perception_report.audio_reports[{index}].report"
    if previous.get("audio_ref_id") not in {None, audio_ref}:
        return None
    prior_text = previous.get("source_text")
    current_text = report.get("source_text")
    if not isinstance(prior_text, str) or not isinstance(current_text, str):
        return None
    parents = (context.get("workflow_evidence_path_sources") or {}).get(
        f"{prefix}.source_text", []
    )
    known = set(context.get("workflow_perception_evidence_ids") or [])
    if not parents or not set(parents).issubset(known):
        return None
    return {
        "schema_version": REVISION_SCHEMA,
        "audio_ref_id": audio_ref,
        "observation_request_id": request_id,
        "parent_evidence_ids": list(dict.fromkeys(parents)),
        "parent_text_sha256": transcript_digest(prior_text),
        "observed_text_sha256": transcript_digest(current_text),
        "mode": "contextual_recognition",
    }


def resolve_transcript_revisions(
    sources: Sequence[Any],
) -> tuple[set[str], dict[str, str], set[str]]:
    """Return superseded observations, rejected revisions and stale text claims.

    Revisions must point backward to existing observations of the same asset
    version and exact prior wording. Sibling revisions are both retained so
    conflicting new observations still produce the ordinary conflict result.
    """
    by_id = {source.evidence_id: source for source in sources}
    retired: set[str] = set()
    rejected: dict[str, str] = {}
    valid: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def check(source: Any) -> bool:
        key = source.evidence_id
        if key in valid:
            return True
        if key in rejected:
            return False
        row = source.report.get("transcript_revision")
        if row is None:
            return True
        reason = _revision_rejection(source, row, by_id)
        if key in visiting:
            reason = "Transcript revisions cannot contain a cycle."
        if reason:
            rejected[key] = reason
            return False
        visiting.add(key)
        parents = set(row["parent_evidence_ids"])
        if any(not check(by_id[parent]) for parent in parents):
            rejected[key] = "Transcript revision has an unadmitted parent revision."
        visiting.remove(key)
        if key in rejected:
            return False
        valid[key] = parents
        retired.update(parents)
        return True

    for source in sources:
        check(source)
    # A text-conditioned candidate made from replaced wording cannot select
    # words on the new alignment. Independent sound/speaker observations stay.
    active_by_audio: dict[str, set[str]] = {}
    for key in valid:
        if key not in retired:
            active_by_audio.setdefault(by_id[key].audio_ref_id, set()).add(
                transcript_digest(by_id[key].report["source_text"])
            )
    revised_wording = {
        audio: next(iter(values)) for audio, values in active_by_audio.items()
        if len(values) == 1
    }
    stale = {
        source.evidence_id for source in sources
        if source.operation == "perception.semantic_span_review"
        and isinstance(source.report.get("source_text"), str)
        and source.audio_ref_id in revised_wording
        and transcript_digest(source.report["source_text"]) != revised_wording[source.audio_ref_id]
    }
    return retired, rejected, stale


def _revision_rejection(source: Any, row: Any, by_id: Mapping[str, Any]) -> str | None:
    if not isinstance(row, Mapping) or row.get("schema_version") != REVISION_SCHEMA:
        return "Transcript revision has an invalid schema."
    if source.operation != REVISION_OPERATION or row.get("mode") != "contextual_recognition":
        return "Transcript revision requires the contextual speech observation contract."
    if row.get("audio_ref_id") != source.audio_ref_id:
        return "Transcript revision belongs to another audio asset."
    if row.get("observation_request_id") != source.request_id:
        return "Transcript revision is not bound to this observation request."
    wording = source.report.get("source_text")
    if not isinstance(wording, str) or row.get("observed_text_sha256") != transcript_digest(wording):
        return "Transcript revision does not bind its observed wording."
    parents = row.get("parent_evidence_ids")
    if not isinstance(parents, list) or not parents or any(
        not isinstance(parent, str) or parent not in by_id or parent == source.evidence_id
        for parent in parents
    ):
        return "Transcript revision lacks valid parent observations."
    for parent_id in parents:
        parent = by_id[parent_id]
        prior_text = parent.report.get("source_text")
        if parent.audio_ref_id != source.audio_ref_id or not isinstance(prior_text, str):
            return "Transcript revision parent belongs to a different audio scope."
        if row.get("parent_text_sha256") != transcript_digest(prior_text):
            return "Transcript revision refers to different prior wording."
        before, after = parent.report.get("timebase") or {}, source.report.get("timebase") or {}
        for field in ("observed_artifact_id", "observed_audio_path", "observed_timeline_revision"):
            if (field in before or field in after) and before.get(field) != after.get(field):
                return "Transcript revision and parent refer to different asset versions."
    return None
