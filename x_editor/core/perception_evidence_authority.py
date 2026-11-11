"""Typed authority and mechanical reconciliation for reviewed Perception evidence.

The authority decision is based on the kind of observation an operation is
contracted to produce. Provider names, benchmark metadata, user-specific text,
and evaluator data are intentionally outside this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

from x_editor.core.transcript_revision import resolve_transcript_revisions


class ObservationKind(StrEnum):
    """Independent fact classes that may have different authoritative sources."""

    LEXICAL_CONTENT = "lexical_content"
    LEXICAL_RESOLUTION = "lexical_resolution"
    WORD_TIMING = "word_timing"
    SPEAKER_IDENTITY = "speaker_identity"
    SEGMENT_OWNERSHIP = "segment_ownership"
    SEMANTIC_JUDGMENT = "semantic_judgment"
    AUDIO_EVENT = "audio_event"
    PROSODY = "prosody"
    ACOUSTIC_PROPERTY = "acoustic_property"
    TIMELINE_SEGMENT = "timeline_segment"
    GENERAL = "general"


class AuthorityLevel(IntEnum):
    """Strength of an evidence source's contract for one observation kind."""

    INCIDENTAL = 10
    DECLARED = 20
    PRIMARY = 30


_REPORT_FIELD_KINDS: dict[str, ObservationKind] = {
    "source_text": ObservationKind.LEXICAL_CONTENT,
    "transcript_revision": ObservationKind.LEXICAL_CONTENT,
    "source_text_source": ObservationKind.LEXICAL_CONTENT,
    "source_text_audio_index": ObservationKind.LEXICAL_CONTENT,
    "source_text_audio_path": ObservationKind.LEXICAL_CONTENT,
    "language": ObservationKind.LEXICAL_CONTENT,
    "word_timestamps": ObservationKind.WORD_TIMING,
    "words": ObservationKind.WORD_TIMING,
    "speakers": ObservationKind.SPEAKER_IDENTITY,
    "speaker_profiles": ObservationKind.SPEAKER_IDENTITY,
    "selected_speaker_ids": ObservationKind.SPEAKER_IDENTITY,
    "speaker_segments": ObservationKind.SEGMENT_OWNERSHIP,
    "speaker_review_summary": ObservationKind.SEGMENT_OWNERSHIP,
    "candidate_segments": ObservationKind.SEMANTIC_JUDGMENT,
    "candidate_rubric_judgements": ObservationKind.SEMANTIC_JUDGMENT,
    "selected_spans": ObservationKind.SEMANTIC_JUDGMENT,
    "target_bindings": ObservationKind.SEMANTIC_JUDGMENT,
    "source_text_resolutions": ObservationKind.LEXICAL_RESOLUTION,
    "events": ObservationKind.AUDIO_EVENT,
    "event_segments": ObservationKind.AUDIO_EVENT,
    "audio_caption_segments": ObservationKind.AUDIO_EVENT,
    "notable_events": ObservationKind.AUDIO_EVENT,
    "prosody_segments": ObservationKind.PROSODY,
    "acoustic_segments": ObservationKind.ACOUSTIC_PROPERTY,
    "audio_quality": ObservationKind.ACOUSTIC_PROPERTY,
    "background": ObservationKind.ACOUSTIC_PROPERTY,
    "segments": ObservationKind.TIMELINE_SEGMENT,
    "time_points": ObservationKind.TIMELINE_SEGMENT,
}

_PRIMARY_KINDS_BY_OPERATION: dict[str, frozenset[ObservationKind]] = {
    "perception.speech_context": frozenset(
        {ObservationKind.LEXICAL_CONTENT, ObservationKind.WORD_TIMING}
    ),
    "perception.speaker_attributed_review": frozenset(
        {ObservationKind.SPEAKER_IDENTITY, ObservationKind.SEGMENT_OWNERSHIP}
    ),
    "perception.semantic_span_review": frozenset(
        {ObservationKind.SEMANTIC_JUDGMENT, ObservationKind.LEXICAL_RESOLUTION}
    ),
    "perception.timestamped_caption": frozenset(
        {ObservationKind.AUDIO_EVENT, ObservationKind.TIMELINE_SEGMENT}
    ),
    "perception.flexsed_events": frozenset({ObservationKind.AUDIO_EVENT}),
    "flexsed.detect_events": frozenset({ObservationKind.AUDIO_EVENT}),
    "perception.audio_fallback": frozenset({ObservationKind.ACOUSTIC_PROPERTY}),
}

_STRUCTURAL_REPORT_FIELDS = frozenset(
    {
        "audio_index",
        "audio_ref_id",
        "duration",
        "duration_sec",
        "role",
        "sample_rate_hz",
        "channels",
        "schema_version",
        "timebase",
    }
)

_ACCUMULATED_REPORT_FIELDS = frozenset(
    {
        "evidence_sources",
        "uncertainties",
        "uncertainty",
    }
)

_ALWAYS_PROJECTED_FIELDS = _STRUCTURAL_REPORT_FIELDS | _ACCUMULATED_REPORT_FIELDS


@dataclass(frozen=True)
class ReviewedEvidenceSource:
    """One immutable sanitized source report plus its typed authority claims."""

    evidence_id: str
    request_id: str
    source: str
    operation: str | None
    audio_ref_id: str | None
    report: dict[str, Any]
    authority_by_kind: dict[ObservationKind, AuthorityLevel]

    def audit_record(self) -> dict[str, Any]:
        return {
            "schema_version": "x-editor.reviewed-evidence-source/v1",
            "evidence_id": self.evidence_id,
            "request_id": self.request_id,
            "source": self.source,
            "operation": self.operation,
            "audio_ref_id": self.audio_ref_id,
            "observation_kinds": sorted(kind.value for kind in self.authority_by_kind),
            "authority_by_observation_kind": {
                kind.value: _authority_level_name(level)
                for kind, level in sorted(
                    self.authority_by_kind.items(), key=lambda item: item[0].value
                )
            },
            "report": copy.deepcopy(self.report),
        }


@dataclass(frozen=True)
class EvidenceAuthorityAssignment:
    """Resolved authority for one audio and one independent observation kind."""

    audio_ref_id: str | None
    observation_kind: ObservationKind
    status: str
    authoritative_evidence_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    candidate_levels: tuple[tuple[str, AuthorityLevel], ...]

    def audit_record(self) -> dict[str, Any]:
        return {
            "schema_version": "x-editor.evidence-authority-assignment/v1",
            "audio_ref_id": self.audio_ref_id,
            "observation_kind": self.observation_kind.value,
            "status": self.status,
            "authoritative_evidence_ids": list(self.authoritative_evidence_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "candidate_authority": {
                evidence_id: _authority_level_name(level)
                for evidence_id, level in self.candidate_levels
            },
        }


def build_reviewed_evidence_source(
    *,
    evidence_id: str,
    request_id: str,
    source: str,
    operation: str | None,
    expected_evidence_fields: Iterable[str],
    audio_ref_id: str | None,
    report: Mapping[str, Any],
) -> ReviewedEvidenceSource:
    """Bind one sanitized report to observation kinds without provider heuristics."""

    copied_report = copy.deepcopy(dict(report))
    expected_fields = tuple(str(field) for field in expected_evidence_fields)
    actual_kinds = {
        observation_kind_for_report_field(field)
        for field in copied_report
        if field not in _ALWAYS_PROJECTED_FIELDS
    }
    actual_kinds.discard(ObservationKind.GENERAL)
    declared_kinds = {
        observation_kind_for_report_field(_field_root(field)) for field in expected_fields
    }
    declared_kinds.discard(ObservationKind.GENERAL)
    primary_kinds = _PRIMARY_KINDS_BY_OPERATION.get(str(operation or ""), frozenset())

    authority_by_kind: dict[ObservationKind, AuthorityLevel] = {}
    for kind in actual_kinds | declared_kinds:
        if kind in primary_kinds:
            authority_by_kind[kind] = AuthorityLevel.PRIMARY
        elif kind in declared_kinds or not expected_fields:
            authority_by_kind[kind] = AuthorityLevel.DECLARED
        else:
            authority_by_kind[kind] = AuthorityLevel.INCIDENTAL
    return ReviewedEvidenceSource(
        evidence_id=evidence_id,
        request_id=request_id,
        source=source,
        operation=operation,
        audio_ref_id=audio_ref_id,
        report=copied_report,
        authority_by_kind=authority_by_kind,
    )


def resolve_evidence_authority(
    sources: Sequence[ReviewedEvidenceSource],
) -> tuple[list[EvidenceAuthorityAssignment], list[dict[str, Any]]]:
    """Choose canonical sources per observation kind and expose hard conflicts."""

    retired, rejected_revisions, stale_text_claims = resolve_transcript_revisions(sources)
    grouped: dict[
        tuple[str | None, ObservationKind], list[tuple[ReviewedEvidenceSource, AuthorityLevel]]
    ] = {}
    for source in sources:
        for kind, level in source.authority_by_kind.items():
            if (
                kind is ObservationKind.LEXICAL_RESOLUTION
                and not source.report.get("source_text_resolutions")
            ):
                continue
            grouped.setdefault((source.audio_ref_id, kind), []).append((source, level))

    assignments: list[EvidenceAuthorityAssignment] = []
    conflicts: list[dict[str, Any]] = []
    for source in sources:
        if source.evidence_id in rejected_revisions:
            conflicts.append(_conflict(
                code="unadmitted_transcript_revision",
                observation_kind=ObservationKind.LEXICAL_CONTENT,
                audio_ref_id=source.audio_ref_id,
                evidence_ids=[source.evidence_id], scope_ref=None,
                message=rejected_revisions[source.evidence_id], blocking=False, details={},
            ))
    for (audio_ref_id, kind), candidates in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), item[0][1].value)
    ):
        eligible = [
            (source, level) for source, level in candidates
            if not (
                kind in {ObservationKind.LEXICAL_CONTENT, ObservationKind.WORD_TIMING}
                and (source.evidence_id in retired or source.evidence_id in rejected_revisions)
            ) and not (
                source.evidence_id in stale_text_claims and kind in {
                    ObservationKind.SEMANTIC_JUDGMENT, ObservationKind.LEXICAL_RESOLUTION,
                    ObservationKind.WORD_TIMING,
                }
            )
        ]
        if not eligible:
            assignments.append(EvidenceAuthorityAssignment(
                audio_ref_id=audio_ref_id, observation_kind=kind, status="canonical",
                authoritative_evidence_ids=(),
                supporting_evidence_ids=tuple(source.evidence_id for source, _ in candidates),
                candidate_levels=tuple((source.evidence_id, level) for source, level in candidates),
            ))
            continue
        max_level = max(level for _source, level in eligible)
        strongest = [source for source, level in eligible if level == max_level]
        status = "canonical"
        authoritative = tuple(source.evidence_id for source in strongest)

        if kind is ObservationKind.LEXICAL_RESOLUTION:
            admitted = []
            for source, _level in eligible:
                reason = _lexical_resolution_rejection(source, sources)
                if reason:
                    conflicts.append(_conflict(
                        code="unadmitted_lexical_resolution",
                        observation_kind=kind,
                        audio_ref_id=audio_ref_id,
                        evidence_ids=[source.evidence_id],
                        scope_ref=None,
                        message=reason,
                        blocking=False,
                        details={},
                    ))
                else:
                    admitted.append(source)
            signatures = {
                tuple(sorted(
                    lexical_comparison_tokens(row["resolved_source_text"])
                    for row in source.report["source_text_resolutions"]
                ))
                for source in admitted
            }
            if not admitted:
                status, authoritative = "conflict", ()
            elif len(signatures) > 1:
                status, authoritative = "conflict", ()
                conflicts.append(_conflict(
                    code="conflicting_lexical_resolution_authority",
                    observation_kind=kind,
                    audio_ref_id=audio_ref_id,
                    evidence_ids=[source.evidence_id for source in admitted],
                    scope_ref=None,
                    message="Reviewed lexical resolutions disagree; no resolved wording was admitted.",
                    blocking=True,
                    details={"source_count": len(admitted)},
                ))
            else:
                authoritative = (admitted[0].evidence_id,)
        elif kind in {ObservationKind.LEXICAL_CONTENT, ObservationKind.WORD_TIMING}:
            consistency = _canonical_consistency_values(kind, strongest)
            if len(consistency) > 1:
                status = "conflict"
                authoritative = ()
                conflicts.append(
                    _conflict(
                        code=f"conflicting_{kind.value}_authority",
                        observation_kind=kind,
                        audio_ref_id=audio_ref_id,
                        evidence_ids=[source.evidence_id for source in strongest],
                        scope_ref=None,
                        message=(
                            "Equally authoritative reviewed sources disagree on "
                            f"{kind.value}; no canonical value was admitted."
                        ),
                        blocking=True,
                        details={"source_count": len(strongest)},
                    )
                )
            elif len(strongest) > 1:
                # Equivalent sources remain in the raw audit sidecar. Selecting
                # one original value avoids inventing a merged wording/timeline.
                authoritative = (strongest[0].evidence_id,)
        elif kind is ObservationKind.SEGMENT_OWNERSHIP and len(strongest) > 1:
            signatures = [
                _canonical_segment_ownership_signature(source.report)
                for source in strongest
            ]
            if all(signatures) and len(set(signatures)) == 1:
                # Repeated jobs may reuse the same diarization result. Keep every
                # raw source for audit, but publish one copy so local IDs from
                # separate evidence namespaces cannot create a false overlap.
                authoritative = (strongest[0].evidence_id,)
        elif kind is ObservationKind.ACOUSTIC_PROPERTY and len(strongest) > 1:
            signatures = [
                _canonical_acoustic_property_signature(source.report) for source in strongest
            ]
            if all(signature is not None for signature in signatures) and len(set(signatures)) == 1:
                # Repeated task-conditioned observations often paraphrase the
                # same background. Publish one canonical narrative when their
                # machine-comparable acoustic facts agree; retain every raw
                # report in the audit sidecar.
                authoritative = (strongest[0].evidence_id,)

        authoritative_set = set(authoritative)
        assignments.append(
            EvidenceAuthorityAssignment(
                audio_ref_id=audio_ref_id,
                observation_kind=kind,
                status=status,
                authoritative_evidence_ids=authoritative,
                supporting_evidence_ids=tuple(
                    source.evidence_id
                    for source, _level in candidates
                    if source.evidence_id not in authoritative_set
                ),
                candidate_levels=tuple((source.evidence_id, level) for source, level in candidates),
            )
        )
    return assignments, conflicts


def _lexical_resolution_rejection(
    source: ReviewedEvidenceSource,
    sources: Sequence[ReviewedEvidenceSource],
) -> str | None:
    """Check provenance of an existing reviewed conclusion, never invent wording.

    The semantic-span operation already admits its narrow audio-corroborated
    omission. This boundary only keeps that conclusion separate from immutable
    ASR wording and verifies its asset, canonical prior, and local citations.
    """
    if source.operation != "perception.semantic_span_review":
        return "Lexical resolution requires the owning reviewed-audio operation."
    lexical_sources = [
        item for item in sources
        if item.audio_ref_id == source.audio_ref_id
        and ObservationKind.LEXICAL_CONTENT in item.authority_by_kind
        and item.report.get("source_text")
    ]
    if not lexical_sources:
        return "Lexical resolution has no canonical wording to resolve."
    strongest_level = max(
        item.authority_by_kind[ObservationKind.LEXICAL_CONTENT]
        for item in lexical_sources
    )
    canonical = [
        item for item in lexical_sources
        if item.authority_by_kind[ObservationKind.LEXICAL_CONTENT] == strongest_level
    ]
    wording = _canonical_consistency_values(ObservationKind.LEXICAL_CONTENT, canonical)
    if len(wording) != 1:
        return "Lexical resolution has no consistent canonical wording."
    rows = source.report.get("source_text_resolutions")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return "Lexical resolution must contain one complete reviewed conclusion."
    row = rows[0]
    if (
        row.get("status") != "corroborated"
        or not str(row.get("resolved_source_text") or "").strip()
    ):
        return "Lexical resolution has not been corroborated."
    if row.get("audio_ref_id") != source.audio_ref_id:
        return "Lexical resolution is not bound to this source audio."
    if lexical_comparison_tokens(row.get("canonical_source_text")) not in wording:
        return "Lexical resolution refers to different canonical wording."
    timebase = source.report.get("timebase") or {}
    for item in canonical:
        prior_timebase = item.report.get("timebase") or {}
        for key in ("observed_artifact_id", "observed_timeline_revision", "observed_audio_path"):
            if key in timebase or key in prior_timebase:
                if timebase.get(key) != prior_timebase.get(key):
                    return "Lexical resolution and canonical wording have different audio versions."

    spans = {
        str(item.get("span_id")): item
        for item in source.report.get("selected_spans") or []
        if isinstance(item, dict)
    }
    targets = {
        str(item.get("target_id")): item
        for item in source.report.get("target_bindings") or []
        if isinstance(item, dict)
    }
    candidates = {
        str(item.get("candidate_id")): item
        for item in source.report.get("candidate_segments") or []
        if isinstance(item, dict)
    }
    span_ids = row.get("span_ids")
    target_ids = row.get("target_ids")
    refs = row.get("source_refs")
    if not span_ids or not target_ids or not refs:
        return "Lexical resolution lacks source, span, or target citations."
    if not set(span_ids).issubset(spans) or not set(target_ids).issubset(targets):
        return "Lexical resolution cites an unknown local span or target."
    known_refs = set(spans) | set(targets) | set(candidates)
    for index, word in enumerate(source.report.get("word_timestamps") or []):
        known_refs.add(f"word_timestamp:{index}")
        if isinstance(word, dict) and word.get("word_id"):
            known_refs.add(str(word["word_id"]))
    if not set(refs).issubset(known_refs):
        return "Lexical resolution cites unknown observation evidence."
    judgements = source.report.get("candidate_rubric_judgements") or []
    for target_id in target_ids:
        if not any(
            isinstance(item, dict)
            and item.get("claim_target_id") == target_id
            and item.get("candidate_id") in candidates
            and item.get("candidate_id") in refs
            and item.get("claim_explicit") is True
            and item.get("claim_kind") == "exact_phrase"
            and item.get("status") == "admitted_with_asr_disagreement"
            and item.get("corroboration_status") == "admitted_with_single_asr_omission"
            for item in judgements
        ):
            return "Lexical resolution lacks its admitted independent audio claim."
    return None


def scope_report_to_authority(
    source: ReviewedEvidenceSource,
    assignments: Sequence[EvidenceAuthorityAssignment],
    *,
    bound_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep only fields this source may publish as canonical Planner evidence."""

    assignment_by_kind = {
        item.observation_kind: item
        for item in assignments
        if item.audio_ref_id == source.audio_ref_id
    }
    scoped: dict[str, Any] = {}
    report = source.report if bound_report is None else bound_report
    for field, value in report.items():
        if field in _ALWAYS_PROJECTED_FIELDS:
            scoped[field] = copy.deepcopy(value)
            continue
        kind = observation_kind_for_report_field(field)
        if kind is ObservationKind.GENERAL:
            scoped[field] = copy.deepcopy(value)
            continue
        assignment = assignment_by_kind.get(kind)
        if assignment is None or source.evidence_id in assignment.authoritative_evidence_ids:
            scoped[field] = copy.deepcopy(value)
    return scoped


def competing_observations(
    sources: Sequence[ReviewedEvidenceSource],
    assignments: Sequence[EvidenceAuthorityAssignment],
) -> list[dict[str, Any]]:
    """Retain suppressed source claims for review without granting authority.

    Values are already sanitized source-report fields. Their local identifiers
    remain scoped to the evidence/audio pair. The JSON value stays opaque to
    candidate-inventory traversal, even if a nested claim names a canonical
    collection. These records must not enter executable
    field-path inventories or replace the canonical report.
    """

    assignment_by_scope = {
        (item.audio_ref_id, item.observation_kind): item for item in assignments
    }
    source_by_scope = {
        (source.evidence_id, source.audio_ref_id): source for source in sources
    }
    observations: list[dict[str, Any]] = []
    for source in sources:
        for field, value in source.report.items():
            kind = observation_kind_for_report_field(field)
            assignment = assignment_by_scope.get((source.audio_ref_id, kind))
            if (
                field in _ALWAYS_PROJECTED_FIELDS
                or kind is ObservationKind.GENERAL
                or assignment is None
                or source.evidence_id in assignment.authoritative_evidence_ids
            ):
                continue
            canonical_sources = [
                source_by_scope[(evidence_id, source.audio_ref_id)]
                for evidence_id in assignment.authoritative_evidence_ids
            ]
            if any(
                field in canonical.report and canonical.report[field] == value
                for canonical in canonical_sources
            ):
                continue
            observations.append(
                {
                    "evidence_id": source.evidence_id,
                    "audio_ref_id": source.audio_ref_id,
                    "operation": source.operation,
                    "observation_kind": kind.value,
                    "field": field,
                    "authority": "non_authoritative",
                    "usage": "Compare source claims and request review; not an executable value.",
                    "value_namespace": "source_report_local",
                    "canonical_evidence_ids": list(assignment.authoritative_evidence_ids),
                    "reported_value_json": json.dumps(
                        value, ensure_ascii=False, sort_keys=True, default=str
                    ),
                }
            )
    return observations


_ACOUSTIC_ANNOTATION_FIELDS = frozenset(
    {
        "audio_index",
        "audio_ref_id",
        "background_description",
        "confidence",
        "description",
        "source",
        "source_kind",
        "summary",
    }
)


def _canonical_acoustic_property_signature(report: Mapping[str, Any]) -> str | None:
    """Fingerprint comparable acoustic facts without model-authored prose."""

    comparable: dict[str, Any] = {}
    for field, value in report.items():
        if observation_kind_for_report_field(field) is not ObservationKind.ACOUSTIC_PROPERTY:
            continue
        stripped = _strip_acoustic_annotations(value)
        if stripped not in ({}, []):
            comparable[field] = stripped
    if not comparable:
        return None
    return json.dumps(
        comparable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _strip_acoustic_annotations(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_acoustic_annotations(item)
            for key, item in value.items()
            if str(key) not in _ACOUSTIC_ANNOTATION_FIELDS
        }
    if isinstance(value, list):
        return [_strip_acoustic_annotations(item) for item in value]
    return copy.deepcopy(value)


def reconcile_canonical_reviewed_report(
    report: Mapping[str, Any],
    *,
    authority_evidence_ids: Mapping[str | ObservationKind, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mechanically join lexical, speaker, and semantic observations.

    The function copies admitted word tokens verbatim. It never repairs source
    wording from a target-conditioned judgement and never derives a target
    answer. Raw source reports are expected to live in the projection sidecar.
    """

    canonical = copy.deepcopy(dict(report))
    conflicts: list[dict[str, Any]] = []
    words = _canonical_words(canonical.get("word_timestamps"))
    if words:
        _attach_speaker_ownership(canonical, words, conflicts, authority_evidence_ids)
        _project_canonical_span_wording(canonical, words, conflicts, authority_evidence_ids)
    _validate_exact_phrase_claims(canonical, words, conflicts, authority_evidence_ids)
    _append_conflict_uncertainties(canonical, conflicts)
    return canonical, conflicts


def append_conflict_uncertainties(
    report: Mapping[str, Any],
    conflicts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a report with typed conflicts projected as Planner-safe uncertainty."""

    copied = copy.deepcopy(dict(report))
    _append_conflict_uncertainties(copied, conflicts)
    return copied


def observation_kind_for_report_field(field: str) -> ObservationKind:
    return _REPORT_FIELD_KINDS.get(_field_root(field), ObservationKind.GENERAL)


def lexical_comparison_tokens(text: Any) -> tuple[str, ...]:
    """Return comparison-only tokens without rewriting the admitted text.

    Internal hyphens and apostrophes stay part of a token, so a normalizer
    cannot turn one lexical form into a different phrase. CJK characters are
    compared individually because word boundaries are not encoded by spaces.
    """

    value = str(text or "")
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current).casefold())
            current.clear()

    for index, character in enumerate(value):
        if _is_cjk(character):
            flush()
            tokens.append(character)
            continue
        if character.isalnum():
            current.append(character)
            continue
        if character in {"-", "'", "\u2019"}:
            previous_is_alnum = bool(current)
            next_is_alnum = index + 1 < len(value) and value[index + 1].isalnum()
            if previous_is_alnum and next_is_alnum:
                current.append(character)
                continue
        flush()
    flush()
    return tuple(tokens)


def _authority_level_name(level: AuthorityLevel) -> str:
    return {
        AuthorityLevel.INCIDENTAL: "incidental",
        AuthorityLevel.DECLARED: "declared",
        AuthorityLevel.PRIMARY: "primary",
    }[level]


def _field_root(field: str) -> str:
    return str(field).split(".", 1)[0].split("[", 1)[0]


def _canonical_consistency_values(
    kind: ObservationKind,
    sources: Sequence[ReviewedEvidenceSource],
) -> set[tuple[Any, ...]]:
    values: set[tuple[Any, ...]] = set()
    for source in sources:
        if kind is ObservationKind.LEXICAL_CONTENT:
            if source.report.get("source_text") not in (None, ""):
                values.add(lexical_comparison_tokens(source.report["source_text"]))
            continue
        words = _canonical_words(source.report.get("word_timestamps"))
        if words:
            values.add(
                tuple(
                    (
                        lexical_comparison_tokens(item["text"]),
                        round(float(item["start_sec"]), 4),
                        round(float(item["end_sec"]), 4),
                    )
                    for item in words
                )
            )
    return values


def _canonical_segment_ownership_signature(
    report: Mapping[str, Any],
) -> tuple[tuple[str, float, float], ...]:
    segments = report.get("speaker_segments")
    if not isinstance(segments, list):
        return ()
    signature: list[tuple[str, float, float]] = []
    for item in segments:
        if not _valid_interval(item) or item.get("speaker_id") in (None, ""):
            continue
        signature.append(
            (
                str(item["speaker_id"]),
                round(float(item["start_sec"]), 4),
                round(float(item["end_sec"]), 4),
            )
        )
    return tuple(sorted(signature))


def _canonical_words(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    words: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text") or item.get("word") or "")
        if not text or start_sec < 0 or end_sec <= start_sec:
            continue
        copied = item
        copied["start_sec"] = start_sec
        copied["end_sec"] = end_sec
        copied["text"] = text
        words.append(copied)
    words.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    return words


def _attach_speaker_ownership(
    report: dict[str, Any],
    words: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    authority_evidence_ids: Mapping[str | ObservationKind, Sequence[str]] | None,
) -> None:
    segments = report.get("speaker_segments")
    if not isinstance(segments, list):
        return
    valid_segments = [item for item in segments if _valid_interval(item)]
    for word in words:
        midpoint = (float(word["start_sec"]) + float(word["end_sec"])) / 2.0
        owners = [
            item
            for item in valid_segments
            if float(item["start_sec"]) <= midpoint < float(item["end_sec"])
            and item.get("speaker_id") not in (None, "")
        ]
        owner_ids = {str(item["speaker_id"]) for item in owners}
        previous = word.get("speaker_id")
        if previous not in (None, ""):
            # A direct word attribution is more specific than interval coverage.
            # Concurrent segments cannot revoke it or make two word events one.
            if owner_ids and str(previous) not in owner_ids:
                conflicts.append(
                    _conflict(
                        code="speaker_word_ownership_conflict",
                        observation_kind=ObservationKind.SEGMENT_OWNERSHIP,
                        audio_ref_id=_audio_ref(report),
                        evidence_ids=_combined_authority_ids(
                            authority_evidence_ids,
                            ObservationKind.WORD_TIMING,
                            ObservationKind.SEGMENT_OWNERSHIP,
                        ),
                        scope_ref=str(word.get("word_id") or "") or None,
                        message=(
                            "The existing word owner is absent from the covering speaker "
                            "segments. Attribution was retained for review, not reassigned."
                        ),
                        blocking=True,
                        details={
                            "reported_speaker_id": previous,
                            "covering_speaker_ids": sorted(owner_ids),
                        },
                    )
                )
        elif len(owner_ids) == 1:
            word["speaker_id"] = next(iter(owner_ids))
        elif len(owner_ids) > 1:
            conflicts.append(
                _conflict(
                    code="overlapping_speech_word_owner_unknown",
                    observation_kind=ObservationKind.SEGMENT_OWNERSHIP,
                    audio_ref_id=_audio_ref(report),
                    evidence_ids=_authority_ids(
                        authority_evidence_ids,
                        ObservationKind.SEGMENT_OWNERSHIP,
                    ),
                    scope_ref=str(word.get("word_id") or "") or None,
                    message=(
                        "Speaker intervals overlap, but the word has no direct speaker "
                        "attribution. Concurrent speech is possible; review ownership "
                        "only if the requested edit needs it. Interval overlap alone "
                        "does not establish a contradictory observation."
                    ),
                    blocking=False,
                    details={"covering_speaker_ids": sorted(owner_ids)},
                )
            )

    for segment in valid_segments:
        in_interval = _words_for_interval(words, segment)
        owner = segment.get("speaker_id")
        if owner in (None, ""):
            continue
        selected = [
            word for word in in_interval if str(word.get("speaker_id") or "") == str(owner)
        ]
        # Unattributed words in a mixture are not each speaker's words. Preserve
        # the reported segment wording instead of replacing it with incomplete
        # or duplicated global ASR text. Raw reports remain in the audit sidecar.
        if not selected or any(word.get("speaker_id") in (None, "") for word in in_interval):
            continue
        canonical_text = _render_word_tokens(selected)
        reported = str(segment.get("transcript") or segment.get("text") or "")
        wording_conflicts = bool(
            reported
            and lexical_comparison_tokens(reported)
            != lexical_comparison_tokens(canonical_text)
        )
        if wording_conflicts:
            conflicts.append(
                _conflict(
                    code="speaker_segment_lexical_conflict",
                    observation_kind=ObservationKind.LEXICAL_CONTENT,
                    audio_ref_id=_audio_ref(report),
                    evidence_ids=_combined_authority_ids(
                        authority_evidence_ids,
                        ObservationKind.LEXICAL_CONTENT,
                        ObservationKind.SEGMENT_OWNERSHIP,
                    ),
                    scope_ref=str(segment.get("segment_id") or "") or None,
                    message=(
                        "Speaker review wording disagreed with canonical lexical tokens; "
                        "speaker ownership was retained and wording was projected from ASR."
                    ),
                    blocking=False,
                    details={
                        "reported_text": reported,
                        "canonical_tokens": [item["text"] for item in selected],
                    },
                )
            )
        if wording_conflicts:
            segment.setdefault("reported_transcript", reported)
        segment["transcript"] = canonical_text
        segment.pop("text", None)
        word_ids = [str(item["word_id"]) for item in selected if item.get("word_id")]
        if word_ids:
            segment["word_ids"] = word_ids
            segment["source_refs"] = list(
                dict.fromkeys([*(segment.get("source_refs") or []), *word_ids])
            )


def _project_canonical_span_wording(
    report: dict[str, Any],
    words: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    authority_evidence_ids: Mapping[str | ObservationKind, Sequence[str]] | None,
) -> None:
    for field in ("candidate_segments", "selected_spans"):
        values = report.get(field)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict) or not _valid_interval(item):
                continue
            selected = _words_for_interval(words, item)
            if not selected:
                continue
            canonical_text = _render_word_tokens(selected)
            reported = str(item.get("text") or item.get("transcript") or "")
            if reported and lexical_comparison_tokens(reported) != lexical_comparison_tokens(
                canonical_text
            ):
                conflicts.append(
                    _conflict(
                        code="semantic_span_lexical_conflict",
                        observation_kind=ObservationKind.LEXICAL_CONTENT,
                        audio_ref_id=_audio_ref(report),
                        evidence_ids=_combined_authority_ids(
                            authority_evidence_ids,
                            ObservationKind.LEXICAL_CONTENT,
                            ObservationKind.SEMANTIC_JUDGMENT,
                        ),
                        scope_ref=str(item.get("candidate_id") or item.get("span_id") or "")
                        or None,
                        message=(
                            "Target-conditioned span wording disagreed with canonical "
                            "lexical tokens and was replaced only in the canonical projection."
                        ),
                        blocking=False,
                        details={
                            "reported_text": reported,
                            "canonical_tokens": [word["text"] for word in selected],
                        },
                    )
                )
            if reported:
                item.setdefault("reported_text", reported)
            item["text"] = canonical_text
            item.pop("transcript", None)
            word_ids = [str(word["word_id"]) for word in selected if word.get("word_id")]
            if word_ids:
                item["word_ids"] = word_ids


def _validate_exact_phrase_claims(
    report: dict[str, Any],
    words: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    authority_evidence_ids: Mapping[str | ObservationKind, Sequence[str]] | None,
) -> None:
    judgements = report.get("candidate_rubric_judgements")
    candidates = report.get("candidate_segments")
    if not isinstance(judgements, list) or not isinstance(candidates, list):
        return
    candidate_by_id = {
        str(item["candidate_id"]): item
        for item in candidates
        if isinstance(item, dict) and item.get("candidate_id")
    }
    conflicted_candidates: set[str] = set()
    for judgement in judgements:
        if not isinstance(judgement, dict):
            continue
        if str(judgement.get("claim_kind") or "") != "exact_phrase":
            continue
        if not bool(judgement.get("matches")):
            continue
        candidate_id = str(judgement.get("candidate_id") or "")
        claimed_text = judgement.get("claimed_text")
        candidate = candidate_by_id.get(candidate_id)
        canonical_words = (
            _words_for_interval(words, candidate)
            if candidate is not None and words and _valid_interval(candidate)
            else []
        )
        canonical_tokens = tuple(
            token for word in canonical_words for token in lexical_comparison_tokens(word["text"])
        )
        claimed_tokens = lexical_comparison_tokens(claimed_text)
        if claimed_tokens and _contains_token_sequence(canonical_tokens, claimed_tokens):
            judgement.setdefault("reported_matches", True)
            judgement["status"] = "admitted"
            continue

        if _is_framework_corroborated_single_asr_omission(judgement, claimed_tokens):
            judgement.setdefault("reported_matches", True)
            judgement["status"] = "admitted_with_asr_disagreement"
            conflicts.append(
                _conflict(
                    code="exact_phrase_audio_asr_disagreement",
                    observation_kind=ObservationKind.SEMANTIC_JUDGMENT,
                    audio_ref_id=_audio_ref(report),
                    evidence_ids=_combined_authority_ids(
                        authority_evidence_ids,
                        ObservationKind.LEXICAL_CONTENT,
                        ObservationKind.SEMANTIC_JUDGMENT,
                    ),
                    scope_ref=candidate_id or None,
                    message=(
                        "Independent reviewed-audio evidence explicitly confirmed the "
                        "target phrase while canonical ASR omitted one token; the unique "
                        "word-timestamp alignment was admitted and the disagreement remains "
                        "visible."
                    ),
                    blocking=False,
                    details={
                        "claimed_text": claimed_text,
                        "claimed_tokens": list(claimed_tokens),
                        "canonical_tokens": list(canonical_tokens),
                        "asr_omitted_lexeme": judgement.get("asr_omitted_lexeme"),
                        "corroboration_method": judgement.get("corroboration_method"),
                    },
                )
            )
            continue

        code = (
            "exact_phrase_claim_missing_text"
            if not claimed_tokens
            else "exact_phrase_token_mismatch"
            if canonical_tokens
            else "exact_phrase_canonical_tokens_missing"
        )
        judgement.setdefault("reported_matches", True)
        judgement["matches"] = False
        judgement["answer"] = "conflict"
        judgement["status"] = "conflicted"
        judgement["conflict_code"] = code
        conflicted_candidates.add(candidate_id)
        conflicts.append(
            _conflict(
                code=code,
                observation_kind=ObservationKind.SEMANTIC_JUDGMENT,
                audio_ref_id=_audio_ref(report),
                evidence_ids=_combined_authority_ids(
                    authority_evidence_ids,
                    ObservationKind.LEXICAL_CONTENT,
                    ObservationKind.SEMANTIC_JUDGMENT,
                ),
                scope_ref=candidate_id or None,
                message=(
                    "An exact-phrase judgement is not self-consistent with the "
                    "admitted canonical word sequence; the positive claim was not admitted."
                ),
                blocking=True,
                details={
                    "claimed_text": claimed_text,
                    "claimed_tokens": list(claimed_tokens),
                    "canonical_tokens": list(canonical_tokens),
                },
            )
        )

    if not conflicted_candidates:
        return
    spans = report.get("selected_spans")
    if isinstance(spans, list):
        report["selected_spans"] = [
            item
            for item in spans
            if not (
                isinstance(item, dict)
                and conflicted_candidates.intersection(
                    str(ref) for ref in item.get("source_refs") or []
                )
            )
        ]
    admitted_span_ids = {
        str(item["span_id"])
        for item in report.get("selected_spans") or []
        if isinstance(item, dict) and item.get("span_id")
    }
    bindings = report.get("target_bindings")
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            resolved = [
                str(span_id)
                for span_id in binding.get("resolved_span_ids") or []
                if str(span_id) in admitted_span_ids
            ]
            if resolved == list(binding.get("resolved_span_ids") or []):
                continue
            binding["resolved_span_ids"] = resolved
            binding["source_refs"] = [
                str(ref)
                for ref in binding.get("source_refs") or []
                if str(ref) not in conflicted_candidates
            ]
            if not resolved:
                binding["status"] = "unresolved"
                binding["confidence"] = None
                binding["reason"] = "exact phrase evidence conflicted with canonical lexical tokens"


def _is_framework_corroborated_single_asr_omission(
    judgement: dict[str, Any],
    claimed_tokens: tuple[str, ...],
) -> bool:
    if not claimed_tokens or judgement.get("claim_explicit") is not True:
        return False
    if judgement.get("corroboration_status") != "admitted_with_single_asr_omission":
        return False
    if (
        judgement.get("corroboration_method")
        != "explicit_audio_exact_phrase+unique_single_asr_omission"
    ):
        return False
    try:
        confidence = float(judgement["confidence"])
        edit_distance = int(judgement["asr_lexical_edit_distance"])
    except (KeyError, TypeError, ValueError):
        return False
    return confidence >= 0.8 and edit_distance == 1


def _append_conflict_uncertainties(
    report: dict[str, Any],
    conflicts: Sequence[Mapping[str, Any]],
) -> None:
    if not conflicts:
        return
    uncertainties = report.setdefault("uncertainties", [])
    if not isinstance(uncertainties, list):
        uncertainties = []
        report["uncertainties"] = uncertainties
    existing = {
        (str(item.get("code")), str(item.get("scope_ref")))
        for item in uncertainties
        if isinstance(item, dict)
    }
    for conflict in conflicts:
        key = (str(conflict.get("code")), str(conflict.get("scope_ref")))
        if key in existing:
            continue
        uncertainties.append(
            {
                "audio_ref_id": conflict.get("audio_ref_id"),
                "code": conflict.get("code"),
                "message": conflict.get("message"),
                "scope_ref": conflict.get("scope_ref"),
                "severity": "error" if conflict.get("blocking") else "warning",
                "blocking": bool(conflict.get("blocking")),
                "missing_evidence": [],
            }
        )
        existing.add(key)


def _valid_interval(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        return float(value["end_sec"]) > float(value["start_sec"]) >= 0.0
    except (KeyError, TypeError, ValueError):
        return False


def _words_for_interval(
    words: Sequence[dict[str, Any]],
    interval: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if interval is None or not _valid_interval(interval):
        return []
    start_sec = float(interval["start_sec"])
    end_sec = float(interval["end_sec"])
    return [
        word
        for word in words
        if float(word["end_sec"]) > start_sec and float(word["start_sec"]) < end_sec
    ]


def _render_word_tokens(words: Sequence[Mapping[str, Any]]) -> str:
    # Every emitted token is copied verbatim and remains addressable by word_id.
    return " ".join(str(item["text"]) for item in words).strip()


def _contains_token_sequence(
    canonical: Sequence[str],
    claimed: Sequence[str],
) -> bool:
    if not claimed or len(claimed) > len(canonical):
        return False
    return any(
        tuple(canonical[index : index + len(claimed)]) == tuple(claimed)
        for index in range(len(canonical) - len(claimed) + 1)
    )


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _audio_ref(report: Mapping[str, Any]) -> str | None:
    value = report.get("audio_ref_id")
    return str(value) if value not in (None, "") else None


def _authority_ids(
    mapping: Mapping[str | ObservationKind, Sequence[str]] | None,
    kind: ObservationKind,
) -> list[str]:
    if not mapping:
        return []
    values = mapping.get(kind) or mapping.get(kind.value) or []
    return list(dict.fromkeys(str(item) for item in values if str(item)))


def _combined_authority_ids(
    mapping: Mapping[str | ObservationKind, Sequence[str]] | None,
    *kinds: ObservationKind,
) -> list[str]:
    return list(
        dict.fromkeys(
            evidence_id for kind in kinds for evidence_id in _authority_ids(mapping, kind)
        )
    )


def _conflict(
    *,
    code: str,
    observation_kind: ObservationKind,
    audio_ref_id: str | None,
    evidence_ids: Sequence[str],
    scope_ref: str | None,
    message: str,
    blocking: bool,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "code": code,
        "observation_kind": observation_kind.value,
        "audio_ref_id": audio_ref_id,
        "evidence_ids": list(evidence_ids),
        "scope_ref": scope_ref,
        "details": details,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": "x-editor.evidence-conflict/v1",
        "conflict_id": f"evidence_conflict:{digest}",
        **identity,
        "message": message,
        "blocking": blocking,
    }


__all__ = [
    "AuthorityLevel",
    "EvidenceAuthorityAssignment",
    "ObservationKind",
    "ReviewedEvidenceSource",
    "append_conflict_uncertainties",
    "build_reviewed_evidence_source",
    "lexical_comparison_tokens",
    "observation_kind_for_report_field",
    "reconcile_canonical_reviewed_report",
    "resolve_evidence_authority",
    "scope_report_to_authority",
]
