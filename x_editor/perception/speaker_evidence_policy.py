"""Explicit specialist-prior visibility and unreviewed speaker evidence projection.

These policies remove information or observation calls. They never supply a
reference speaker count, invent a transcript, or certify a skipped audio review.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from x_editor.perception.speech import SpeechPerceptionError


BLIND_SPEAKER_PROMPT = """Listen to the attached complete audio. Transcribe each
audible speaker's words with start/end seconds. No specialist transcript, speaker
count, identity or timing prior is supplied. Assign anonymous labels of the form
speaker N consistently within this recording; numbers do not denote user roles.
Allow overlapping speakers. People named in speech are not necessarily speakers.
Return only transcript_lines matching the schema. Do not invent speech for
silence or resolve identity from the instruction alone."""


def blind_label_map(data):
    """A namespace inferred by the audio model, never from an oracle count."""
    labels = {}
    for line in data.get("transcript_lines", []):
        label = line.get("speaker_label")
        if not isinstance(label, str) or not re.fullmatch(r"speaker [0-9]+", label):
            raise SpeechPerceptionError("invalid anonymous speaker label")
        labels[label] = "speaker_" + label.split()[1]
    if not labels:
        raise SpeechPerceptionError("audio-only speaker observation returned no speakers")
    return labels


def specialist_only_result(evidence, context):
    """Publish measured specialist spans without pretending Omni checked them."""
    report = deepcopy(evidence.get("base_perception") or {})
    # Use real specialist spans, not the single-speaker fallback synthesized by
    # the review prompt assembler when only an unsegmented ASR sentence exists.
    segments = deepcopy(evidence.get("speaker_segments") or [])
    # The native report and active timeline share an explicit namespace. Never
    # merge unmatched numeric IDs from an auxiliary provider as extra speakers.
    attributed = []
    for result in evidence.get("sedicow") or []:
        for item in result.get("segments") or []:
            attributed.append({
                "speaker_id": item.get("speaker_id", result.get("speaker_id")),
                "start_sec": item.get("start_sec", item.get("start")),
                "end_sec": item.get("end_sec", item.get("end")),
                "transcript": item.get("transcript") or item.get("text") or "",
                "source": "sedicow_asr",
            })
    if attributed:
        if evidence.get("speaker_id_namespace") != "sedicow_asr":
            raise SpeechPerceptionError("SE-DiCoW report and active speaker timeline lack a shared namespace")
        covered = {item["speaker_id"] for item in attributed}
        remaining = [s for s in segments if s.get("speaker_id") not in covered]
        segments = attributed + remaining
        if remaining:
            report.setdefault("uncertainty", []).append(
                "SE-DiCoW did not return attributed transcripts for every measured "
                "speaker. Unmatched diarization spans are retained as separate evidence."
            )
    if not segments:
        raise SpeechPerceptionError("specialist-only speaker evidence has no measured spans")
    duration = context["duration_sec"]
    boundary_projections = []
    for segment in segments:
        start, end = segment.get("start_sec"), segment.get("end_sec")
        if (not isinstance(start, (float, int)) or isinstance(start, bool)
                or not isinstance(end, (float, int)) or isinstance(end, bool)
                or not math.isfinite(start) or not math.isfinite(end)
                or not start < end or not segment.get("speaker_id")):
            raise SpeechPerceptionError("invalid specialist speaker span")
        bounded_start, bounded_end = max(0., start), min(duration, end)
        if bounded_start >= bounded_end:
            raise SpeechPerceptionError("specialist speaker span is outside the recording")
        if (bounded_start, bounded_end) != (start, end):
            # Model frame boundaries may straddle a recording edge. Retain the
            # raw report for audit and publish only its observable intersection;
            # this is geometry, not evidence of identity or target fulfillment.
            boundary_projections.append({"speaker_id": segment["speaker_id"],
                "reported": [start, end], "observable": [bounded_start, bounded_end]})
            segment.update(start_sec=bounded_start, end_sec=bounded_end)
    segments.sort(key=lambda s: (s["start_sec"], s["end_sec"], str(s["speaker_id"])))
    report["speaker_segments"] = segments
    report["segments"] = deepcopy(segments)
    report["speakers"] = [{"speaker_id": speaker} for speaker in
                          sorted({str(s["speaker_id"]) for s in segments})]
    report["speaker_attributed_review"] = {
        "source": "specialist_observation", "omni_review_performed": False,
        "semantic_role_verified": False,
    }
    if boundary_projections:
        report["speaker_boundary_projections"] = boundary_projections
        report.setdefault("uncertainty", []).append(
            "Specialist spans crossed the recording boundary; only their intersection "
            "with the actual recording is observable. Raw endpoints are retained."
        )
    return {
        "perception_report": report, "source_text": report.get("source_text"),
        "speaker_attributed_review": {
            "status": "specialist_only", "omni_review_performed": False,
            "evidence": evidence, "raw_outputs": [], "review_attempts": [],
            "quality_check_history": [], "slice_audio_calls": [],
            "verification": {"passed": True, "scope": "specialist_span_geometry_only"},
        },
    }
