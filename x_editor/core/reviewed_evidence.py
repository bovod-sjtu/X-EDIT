"""Shared sanitization for evidence admitted to strict workflow contexts."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, unquote

from x_editor.core.speaker_identifiers import allocate_speaker_codes, is_identity_label

REVIEWED_PERCEPTION_SCHEMA_VERSION = "x-editor.reviewed-perception/v2"

PROJECTED_SCALAR_FIELDS = frozenset(
    {
        "source_text",
        "source_text_audio_index",
        "source_text_audio_path",
        "source_text_source",
        "span_selection_analysis",
    }
)

REVIEWED_REPORT_ROOT_FIELDS = frozenset(
    {
        "audio_caption_segments",
        "audio_index",
        "audio_ref_id",
        "audio_quality",
        "background",
        "candidate_rubric_judgements",
        "candidate_segments",
        "channels",
        "duration",
        "duration_sec",
        "evidence_sources",
        "event_segments",
        "events",
        "language",
        "notable_events",
        "prosody_segments",
        "acoustic_segments",
        "role",
        "sample_rate_hz",
        "schema_version",
        "segments",
        "selected_speaker_refs",
        "selected_speaker_ids",
        "selected_spans",
        "source_text_resolutions",
        "transcript_revision",
        "source_text",
        "source_text_audio_index",
        "source_text_audio_path",
        "source_text_source",
        "span_selection_analysis",
        "speaker_profiles",
        "speaker_review_summary",
        "speaker_segments",
        "speakers",
        "target_bindings",
        "time_points",
        "timebase",
        "uncertainty",
        "uncertainties",
        "word_timestamps",
        "words",
    }
)

_COLLECTION_FIELD_ALLOWLISTS: dict[str, frozenset[str]] = {
    "word_timestamps": frozenset(
        {
            "word_id",
            "audio_ref_id",
            "audio_index",
            "text",
            "word",
            "normalized_text",
            "start_sec",
            "end_sec",
            "start",
            "end",
            "char_start",
            "char_end",
            "speaker_id",
            "confidence",
            "is_punctuation",
            "source",
            "source_kind",
        }
    ),
    "segments": frozenset(
        {
            "segment_id",
            "audio_ref_id",
            "audio_index",
            "start_sec",
            "end_sec",
            "start",
            "end",
            "transcript",
            "text",
            "speaker_id",
            "speaker_label",
            "sentence_type",
            "turn_index",
            "confidence",
            "description",
            "source",
            "source_kind",
        }
    ),
    "speaker_segments": frozenset(
        {
            "segment_id",
            "audio_ref_id",
            "audio_index",
            "speaker_id",
            "speaker_label",
            "start_sec",
            "end_sec",
            "start",
            "end",
            "transcript",
            "text",
            "reported_transcript",
            "word_ids",
            "source_refs",
            "overlap_speaker_ids",
            "description",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "speakers": frozenset(
        {
            "speaker_id",
            "audio_ref_id",
            "audio_index",
            "speaker_label",
            "segment_ids",
            "order",
            "description",
            "perceived_gender",
            "gender",
            "age_band",
            "role",
            "language",
            "accent",
            "emotion",
            "speaking_style",
            "vocal_texture",
            "foreground_status",
            "distance",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "audio_caption_segments": frozenset(
        {
            "segment_id",
            "audio_ref_id",
            "audio_index",
            "start_sec",
            "end_sec",
            "caption",
            "events",
            "event_type",
            "label",
            "speaker_id",
            "evidence_source",
            "evidence_slice_ids",
            "detector_refs",
            "flexsed_event_refs",
            "confidence",
            "model_certainty",
            "sed_support_prior",
            "needs_review",
            "evidence_warnings",
            "target_match",
            "source",
            "source_kind",
        }
    ),
    "events": frozenset(
        {
            "event_id",
            "audio_ref_id",
            "audio_index",
            "event_type",
            "label",
            "caption",
            "start_sec",
            "end_sec",
            "speaker_id",
            "evidence_slice_ids",
            "detector_refs",
            "confidence",
            "needs_review",
            "source",
            "source_kind",
        }
    ),
    "candidate_segments": frozenset(
        {
            "candidate_id",
            "candidate_kind",
            "audio_ref_id",
            "audio_index",
            "start_sec",
            "end_sec",
            "speaker_id",
            "text",
            "transcript",
            "reported_text",
            "word_ids",
            "description",
            "detected_properties",
            "source_refs",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "candidate_rubric_judgements": frozenset(
        {
            "candidate_id",
            "answer",
            "matches",
            "match_declared",
            "reported_matches",
            "claim_explicit",
            "claim_kind",
            "reported_claim_kind",
            "claim_target_id",
            "claimed_text",
            "reported_claimed_text",
            "status",
            "conflict_code",
            "start_sec",
            "end_sec",
            "confidence",
            "reason",
            "detected_properties",
            "evidence_slice_ids",
            "source_refs",
            "corroboration_status",
            "corroboration_method",
            "asr_lexical_edit_distance",
            "asr_omitted_lexeme",
        }
    ),
    "selected_spans": frozenset(
        {
            "span_id",
            "audio_ref_id",
            "audio_index",
            "start_sec",
            "end_sec",
            "speaker_id",
            "text",
            "reported_text",
            "word_ids",
            "label",
            "semantic_role",
            "target_id",
            "source_candidate_refs",
            "source_refs",
            "confidence",
            "reason",
            "source",
            "source_kind",
        }
    ),
    "selected_speaker_refs": frozenset(
        {
            "audio_ref_id",
            "audio_index",
            "speaker_id",
        }
    ),
    "target_bindings": frozenset(
        {
            "target_id",
            "target_kind",
            "target_value",
            "status",
            "resolved_candidate_refs",
            "resolved_span_ids",
            "resolved_speaker_refs",
            "resolved_speaker_ids",
            "resolved_event_ids",
            "resolved_acoustic_ids",
            "audio_ref_ids",
            "confidence",
            "reason",
            "source_refs",
        }
    ),
    "source_text_resolutions": frozenset(
        {
            "resolution_id",
            "audio_ref_id",
            "audio_index",
            "status",
            "canonical_source_text",
            "resolved_source_text",
            "target_ids",
            "span_ids",
            "source_refs",
            "confidence",
            "reason",
            "source",
            "source_kind",
        }
    ),
    "time_points": frozenset(
        {
            "point_id",
            "audio_ref_id",
            "audio_index",
            "time_sec",
            "char_index",
            "left_context",
            "right_context",
            "boundary_type",
            "current_gap_sec",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "prosody_segments": frozenset(
        {
            "segment_id",
            "audio_ref_id",
            "audio_index",
            "start_sec",
            "end_sec",
            "pitch",
            "pitch_contour",
            "speech_rate",
            "stress",
            "prominence",
            "intensity",
            "emotion",
            "style",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "acoustic_segments": frozenset(
        {
            "segment_id",
            "audio_ref_id",
            "audio_index",
            "scope",
            "start_sec",
            "end_sec",
            "noise_type",
            "snr_db",
            "reverb_type",
            "rt60_sec",
            "room_type",
            "clarity",
            "intelligibility",
            "rms_dbfs",
            "peak",
            "clipping_ratio",
            "voiced_ratio",
            "silence_ratio",
            "bandwidth",
            "channel_character",
            "background_description",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "uncertainties": frozenset(
        {
            "audio_ref_id",
            "audio_index",
            "code",
            "message",
            "scope_ref",
            "severity",
            "blocking",
            "missing_evidence",
        }
    ),
    "evidence_sources": frozenset(
        {
            "evidence_id",
            "operation",
            "route_id",
            "source_kind",
            "provider",
            "model",
            "evidence_refs",
        }
    ),
}

_COLLECTION_ALIASES = {
    "words": "word_timestamps",
    "speaker_profiles": "speakers",
    "event_segments": "events",
    "uncertainty": "uncertainties",
}

_MAPPING_FIELD_ALLOWLISTS: dict[str, frozenset[str]] = {
    "transcript_revision": frozenset({
        "schema_version", "audio_ref_id", "observation_request_id",
        "parent_evidence_ids", "parent_text_sha256", "observed_text_sha256", "mode",
    }),
    "audio_quality": frozenset(
        {
            "scope",
            "audio_ref_id",
            "audio_index",
            "noise_type",
            "snr_db",
            "reverb_type",
            "rt60_sec",
            "room_type",
            "clarity",
            "intelligibility",
            "rms_dbfs",
            "peak",
            "clipping_ratio",
            "voiced_ratio",
            "silence_ratio",
            "bandwidth",
            "channel_character",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "background": frozenset(
        {
            "summary",
            "description",
            "audio_ref_id",
            "audio_index",
            "noise_type",
            "snr_db",
            "reverb_type",
            "rt60_sec",
            "room_type",
            "foreground_status",
            "confidence",
            "source",
            "source_kind",
        }
    ),
    "speaker_review_summary": frozenset(
        {
            "source",
            "source_kind",
            "speech_overlap_class",
            "turn_speaker_switch_count",
        }
    ),
    "timebase": frozenset(
        {
            "schema_version",
            "observed_artifact_id",
            "observed_audio_path",
            "observed_timeline_revision",
            "audio_ref_id",
            "audio_index",
        }
    ),
}

_ID_FIELDS = {
    "audio_caption_segments": ("segment_id", "caption_segment"),
    "candidate_segments": ("candidate_id", "candidate"),
    "events": ("event_id", "event"),
    "segments": ("segment_id", "segment"),
    "selected_spans": ("span_id", "span"),
    "speaker_segments": ("segment_id", "speaker_segment"),
    "speakers": ("speaker_id", "speaker"),
    "target_bindings": ("target_id", "target"),
    "source_text_resolutions": ("resolution_id", "source_text_resolution"),
    "time_points": ("point_id", "point"),
    "word_timestamps": ("word_id", "word"),
    "prosody_segments": ("segment_id", "prosody_segment"),
    "acoustic_segments": ("segment_id", "acoustic_segment"),
}

_LOCAL_ID_DEFINITIONS: dict[str, tuple[str, str]] = {
    "audio_caption_segments": ("segment_id", "segment"),
    "candidate_segments": ("candidate_id", "candidate"),
    "events": ("event_id", "event"),
    "segments": ("segment_id", "segment"),
    "selected_spans": ("span_id", "span"),
    "speaker_segments": ("segment_id", "segment"),
    "speakers": ("speaker_id", "speaker"),
    "target_bindings": ("target_id", "target"),
    "time_points": ("point_id", "point"),
    "word_timestamps": ("word_id", "word"),
    "prosody_segments": ("segment_id", "segment"),
    "acoustic_segments": ("segment_id", "segment"),
}
_LOCAL_ID_REFERENCE_KINDS = {
    "candidate_id": "candidate",
    "event_id": "event",
    "point_id": "point",
    "segment_id": "segment",
    "span_id": "span",
    "speaker_id": "speaker",
    "target_id": "target",
    "word_id": "word",
}
_LOCAL_ID_LIST_REFERENCE_KINDS = {
    "overlap_speaker_ids": "speaker",
    "resolved_acoustic_ids": "segment",
    "resolved_event_ids": "event",
    "resolved_span_ids": "span",
    "resolved_speaker_ids": "speaker",
    "segment_ids": "segment",
    "selected_speaker_ids": "speaker",
    "span_ids": "span",
    "target_ids": "target",
    "word_ids": "word",
}
_POLYMORPHIC_LOCAL_REFERENCE_FIELDS = frozenset({"scope_ref", "source_refs"})
# Explanations are observations, not containers of executable ID definitions.
# A provider may encode an explanation as a structured value; preserve it, but
# never interpret keys inside it as framework references.
_OPAQUE_LOCAL_ID_FIELDS = frozenset({"detected_properties", "reason"})
_LOCAL_ID_KINDS = frozenset(_LOCAL_ID_REFERENCE_KINDS.values())
_LOCAL_ID_NAMESPACE_PREFIX = "ev_"
_UNBOUND_AUDIO_NAMESPACE = "<unbound-audio>"

_STRING_LIST_FIELDS = frozenset({"notable_events", "selected_speaker_ids"})

_SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "authorization",
        "client_metrics",
        "attempt",
        "cache",
        "benchmark",
        "error",
        "fallback",
        "frontend_telemetry",
        "ground_truth",
        "oracle",
        "prompt",
        "raw_output",
        "repair",
        "request_payload",
        "response_payload",
        "secret",
        "service",
        "telemetry",
        "token",
    }
)

_RESULT_CONTROL_FIELDS = frozenset(
    {
        "component_invocation_trace",
        "component_invocation_trace_invalid",
        "missing",
        "needs_review",
        "operation",
        "satisfied",
        "speaker_attributed_review",
        "timestamped_caption_review",
        "uncertainties",
        "warnings",
    }
)


def sanitize_reviewed_value(value: Any) -> Any:
    if isinstance(value, dict):
        if len(value) > 1000:
            raise ValueError("reviewed evidence object exceeds 1000 fields")
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            if key_text not in PROJECTED_SCALAR_FIELDS and any(
                part in normalized for part in _SENSITIVE_KEY_PARTS
            ):
                continue
            sanitized[key_text] = sanitize_reviewed_value(item)
        _validate_time_bounds(sanitized)
        return sanitized
    if isinstance(value, list):
        if len(value) > 10000:
            raise ValueError("reviewed evidence list exceeds 10000 items")
        return [sanitize_reviewed_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("reviewed evidence contains a non-finite number")
    return copy.deepcopy(value)


def sanitize_reviewed_perception_report(
    report: dict[str, Any],
    *,
    audio_refs: Iterable[str] = (),
) -> dict[str, Any]:
    """Project a perception report into the typed Planner-facing v2 contract."""

    normalized = _canonicalize_global_background_region(
        _normalize_reviewed_report_aliases(report)
    )
    refs = list(dict.fromkeys(str(item) for item in audio_refs if str(item)))
    default_audio_ref_id, default_audio_index = _report_audio_namespace(
        normalized,
        audio_refs=refs,
    )
    reviewed: dict[str, Any] = {}
    for raw_key, value in normalized.items():
        key = str(raw_key)
        if key not in REVIEWED_REPORT_ROOT_FIELDS or key == "schema_version":
            continue
        if key in _STRING_LIST_FIELDS:
            strings = _sanitize_string_list(value)
            if strings:
                reviewed[key] = strings
            continue
        if key in _COLLECTION_FIELD_ALLOWLISTS:
            collection = _sanitize_typed_collection(
                key,
                value,
                audio_refs=refs,
                default_audio_ref_id=default_audio_ref_id,
                default_audio_index=default_audio_index,
            )
            if collection:
                reviewed[key] = collection
            continue
        if key in _MAPPING_FIELD_ALLOWLISTS:
            mapping = _sanitize_typed_mapping(
                key,
                value,
                audio_refs=refs,
                default_audio_ref_id=default_audio_ref_id,
                default_audio_index=default_audio_index,
            )
            if mapping:
                reviewed[key] = mapping
            continue
        reviewed[key] = sanitize_reviewed_value(value)

    if not reviewed:
        return {}
    reviewed["schema_version"] = REVIEWED_PERCEPTION_SCHEMA_VERSION
    if default_audio_ref_id is not None:
        _attach_item_audio_namespace(
            reviewed,
            audio_refs=refs,
            default_audio_ref_id=default_audio_ref_id,
            default_audio_index=default_audio_index,
        )
    return sanitize_reviewed_value(reviewed)


def _canonicalize_global_background_region(report: dict[str, Any]) -> dict[str, Any]:
    """Expose a measured whole-audio background observation as an acoustic candidate."""

    normalized = copy.deepcopy(report)
    if normalized.get("acoustic_segments"):
        return normalized
    background = normalized.get("background")
    if not isinstance(background, dict):
        return normalized
    duration = normalized.get("duration_sec", normalized.get("duration"))
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return normalized
    description = background.get("description") or background.get("summary")
    acoustic_fields = {
        key: copy.deepcopy(background[key])
        for key in (
            "noise_type",
            "snr_db",
            "reverb_type",
            "rt60_sec",
            "room_type",
            "confidence",
            "source",
            "source_kind",
        )
        if background.get(key) is not None
    }
    if description is None and not acoustic_fields:
        return normalized
    normalized["acoustic_segments"] = [
        {
            "segment_id": "acoustic_background_global",
            "scope": "whole_audio",
            "start_sec": 0.0,
            "end_sec": float(duration),
            "background_description": description,
            **acoustic_fields,
        }
    ]
    return normalized


def namespace_reviewed_local_ids(
    report: dict[str, Any],
    *,
    evidence_id: str,
    audio_ref_id: str | None,
) -> dict[str, Any]:
    """Globalize one reviewed report's local object IDs and reference closure.

    Perception frontends may independently emit identifiers such as
    ``candidate_000``.  Those identifiers are local to one evidence job and one
    audio.  This two-pass projection first builds a typed symbol table, then
    rewrites definitions and references before reports are merged.
    """

    reviewed = copy.deepcopy(report)
    definitions = _reviewed_local_id_definitions(reviewed)
    if not definitions and not _contains_reviewed_local_id_references(reviewed):
        return reviewed
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("reviewed local IDs require a non-empty evidence_id")
    if audio_ref_id is None:
        audio_ref_id = _UNBOUND_AUDIO_NAMESPACE
    elif not isinstance(audio_ref_id, str) or not audio_ref_id.strip():
        raise ValueError("reviewed local IDs require an audio_ref_id namespace")

    namespace = _reviewed_local_id_namespace(evidence_id, audio_ref_id)
    speaker_locals: dict[str, str] = {}
    reserved_speaker_codes: set[str] = set()
    for kind, local_id, _path in definitions:
        if kind != "speaker" or not isinstance(local_id, str):
            continue
        parsed = _parse_reviewed_global_id(local_id)
        if parsed is not None and len(parsed[2]) == 3 and all("a" <= c <= "z" for c in parsed[2]):
            reserved_speaker_codes.add(parsed[2])
        else:
            speaker_locals[local_id] = parsed[2] if parsed is not None else local_id
    allocated = allocate_speaker_codes(
        speaker_locals.values(), namespace=namespace, reserved=reserved_speaker_codes
    )
    speaker_codes = {local_id: allocated[backend_id] for local_id, backend_id in speaker_locals.items()}
    symbols: dict[str, dict[str, str]] = {kind: {} for kind in _LOCAL_ID_KINDS}
    global_owners: dict[str, tuple[str, str]] = {}
    for kind, local_id, path in definitions:
        _register_reviewed_local_id(
            symbols,
            global_owners,
            namespace=namespace,
            kind=kind,
            local_id=local_id,
            path=path,
            projected_local_id=speaker_codes.get(local_id) if isinstance(local_id, str) else None,
        )

    return _rewrite_reviewed_local_ids(
        reviewed,
        symbols=symbols,
        namespace=namespace,
        path="perception_report",
    )


def _reviewed_local_id_definitions(
    report: dict[str, Any],
) -> list[tuple[str, Any, str]]:
    definitions: list[tuple[str, Any, str]] = []
    for collection_name, (id_field, kind) in _LOCAL_ID_DEFINITIONS.items():
        collection = report.get(collection_name)
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                continue
            path = f"perception_report.{collection_name}[{index}]"
            if item.get(id_field) not in (None, ""):
                definitions.append((kind, item[id_field], f"{path}.{id_field}"))
            # A speaker label attached to an admitted observation is itself a
            # scoped speaker declaration even when no profile object was made.
            if collection_name != "speakers" and item.get("speaker_id") not in (
                None,
                "",
            ):
                definitions.append(("speaker", item["speaker_id"], f"{path}.speaker_id"))
    return definitions


def _contains_reviewed_local_id_references(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_reviewed_local_id_references(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, child in value.items():
        if key in _OPAQUE_LOCAL_ID_FIELDS:
            continue
        if key in _LOCAL_ID_REFERENCE_KINDS and child not in (None, ""):
            return True
        if key in _LOCAL_ID_LIST_REFERENCE_KINDS and child not in (None, []):
            return True
        if key in _POLYMORPHIC_LOCAL_REFERENCE_FIELDS and child not in (
            None,
            "",
            [],
        ):
            return True
        if _contains_reviewed_local_id_references(child):
            return True
    return False


def _reviewed_local_id_namespace(evidence_id: str, audio_ref_id: str) -> str:
    digest = hashlib.sha256(f"{evidence_id}\0{audio_ref_id}".encode()).hexdigest()[:24]
    return f"{_LOCAL_ID_NAMESPACE_PREFIX}{digest}"


def _register_reviewed_local_id(
    symbols: dict[str, dict[str, str]],
    global_owners: dict[str, tuple[str, str]],
    *,
    namespace: str,
    kind: str,
    local_id: Any,
    path: str,
    projected_local_id: str | None = None,
) -> None:
    if not isinstance(local_id, str) or not local_id:
        raise TypeError(f"reviewed local ID at {path} must be a non-empty string")
    qualified = _parse_reviewed_global_id(local_id)
    if qualified is not None:
        qualified_namespace, qualified_kind, decoded_local_id = qualified
        if qualified_namespace != namespace or qualified_kind != kind:
            raise ValueError(f"reviewed local ID definition at {path} uses a foreign namespace")
        canonical_local_id = decoded_local_id
        global_id = (
            _format_reviewed_global_id(namespace, kind, projected_local_id)
            if kind == "speaker" and projected_local_id is not None
            else local_id
        )
    else:
        canonical_local_id = local_id
        global_id = _format_reviewed_global_id(
            namespace, kind, projected_local_id if kind == "speaker" and projected_local_id else local_id
        )

    existing = symbols[kind].get(canonical_local_id)
    if existing is not None and existing != global_id:
        raise ValueError(f"ambiguous reviewed {kind} ID at {path}")
    owner = global_owners.get(global_id)
    if owner is not None and owner != (kind, canonical_local_id):
        raise ValueError(f"reviewed local ID namespace collision at {path}")
    symbols[kind][canonical_local_id] = global_id
    symbols[kind][local_id] = global_id
    symbols[kind][global_id] = global_id
    global_owners[global_id] = (kind, canonical_local_id)


def _format_reviewed_global_id(namespace: str, kind: str, local_id: str) -> str:
    return f"{namespace}:{kind}:{quote(local_id, safe='-._~')}"


def _parse_reviewed_global_id(value: str) -> tuple[str, str, str] | None:
    parts = value.split(":", 2)
    if len(parts) != 3:
        return None
    namespace, kind, encoded_local_id = parts
    if (
        not namespace.startswith(_LOCAL_ID_NAMESPACE_PREFIX)
        or len(namespace) != len(_LOCAL_ID_NAMESPACE_PREFIX) + 24
        or any(character not in "0123456789abcdef" for character in namespace[3:])
        or kind not in _LOCAL_ID_KINDS
        or not encoded_local_id
    ):
        return None
    return namespace, kind, unquote(encoded_local_id)


def _rewrite_reviewed_local_ids(
    value: Any,
    *,
    symbols: dict[str, dict[str, str]],
    namespace: str,
    path: str,
    definition_field: str | None = None,
    definition_kind: str | None = None,
) -> Any:
    if isinstance(value, list):
        return [
            _rewrite_reviewed_local_ids(
                item,
                symbols=symbols,
                namespace=namespace,
                path=f"{path}[{index}]",
                definition_field=definition_field,
                definition_kind=definition_kind,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    rewritten: dict[str, Any] = {}
    for key, child in value.items():
        child_path = f"{path}.{key}"
        if key == "speaker_label" and isinstance(child, str):
            local_speaker = value.get("speaker_id")
            parsed_speaker = _parse_reviewed_global_id(local_speaker) if isinstance(local_speaker, str) else None
            label_id = parsed_speaker[2] if parsed_speaker is not None else local_speaker
            if isinstance(label_id, str) and is_identity_label(child, label_id):
                projected_speaker = symbols["speaker"].get(local_speaker)
                if projected_speaker is not None:
                    rewritten[key] = projected_speaker.rsplit(":", 1)[-1]
                    continue
        if key in _OPAQUE_LOCAL_ID_FIELDS:
            rewritten[key] = copy.deepcopy(child)
            continue
        if key in _LOCAL_ID_DEFINITIONS and isinstance(child, list):
            item_definition_field, item_definition_kind = _LOCAL_ID_DEFINITIONS[key]
            rewritten[key] = _rewrite_reviewed_local_ids(
                child,
                symbols=symbols,
                namespace=namespace,
                path=child_path,
                definition_field=item_definition_field,
                definition_kind=item_definition_kind,
            )
            continue
        if key == definition_field and definition_kind is not None:
            rewritten[key] = _rewrite_reviewed_definition_id(
                child,
                symbols=symbols,
                namespace=namespace,
                kind=definition_kind,
                path=child_path,
            )
            continue
        if key in _LOCAL_ID_REFERENCE_KINDS and child not in (None, ""):
            rewritten[key] = _rewrite_reviewed_typed_reference(
                child,
                symbols=symbols,
                kind=_LOCAL_ID_REFERENCE_KINDS[key],
                path=child_path,
            )
            continue
        if key in _LOCAL_ID_LIST_REFERENCE_KINDS:
            rewritten[key] = _rewrite_reviewed_typed_reference_list(
                child,
                symbols=symbols,
                kind=_LOCAL_ID_LIST_REFERENCE_KINDS[key],
                path=child_path,
            )
            continue
        if key in _POLYMORPHIC_LOCAL_REFERENCE_FIELDS:
            rewritten[key] = _rewrite_reviewed_polymorphic_references(
                child,
                symbols=symbols,
                path=child_path,
            )
            continue
        rewritten[key] = _rewrite_reviewed_local_ids(
            child,
            symbols=symbols,
            namespace=namespace,
            path=child_path,
        )
    return rewritten


def _rewrite_reviewed_definition_id(
    value: Any,
    *,
    symbols: dict[str, dict[str, str]],
    namespace: str,
    kind: str,
    path: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"reviewed local ID at {path} must be a non-empty string")
    qualified = _parse_reviewed_global_id(value)
    if qualified is not None:
        qualified_namespace, qualified_kind, _local_id = qualified
        if qualified_namespace != namespace or qualified_kind != kind:
            raise ValueError(f"reviewed local ID definition at {path} uses a foreign namespace")
    try:
        return symbols[kind][value]
    except KeyError as exc:
        raise ValueError(f"unregistered reviewed {kind} definition at {path}") from exc


def _rewrite_reviewed_typed_reference(
    value: Any,
    *,
    symbols: dict[str, dict[str, str]],
    kind: str,
    path: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"reviewed {kind} reference at {path} must be a non-empty string")
    qualified = _parse_reviewed_global_id(value)
    if qualified is not None and qualified[1] != kind:
        raise ValueError(f"reviewed {kind} reference at {path} has the wrong kind")
    try:
        return symbols[kind][value]
    except KeyError as exc:
        raise ValueError(f"dangling reviewed {kind} reference at {path}: {value!r}") from exc


def _rewrite_reviewed_typed_reference_list(
    value: Any,
    *,
    symbols: dict[str, dict[str, str]],
    kind: str,
    path: str,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"reviewed {kind} references at {path} must be a list")
    return [
        _rewrite_reviewed_typed_reference(
            item,
            symbols=symbols,
            kind=kind,
            path=f"{path}[{index}]",
        )
        for index, item in enumerate(value)
    ]


def _rewrite_reviewed_polymorphic_references(
    value: Any,
    *,
    symbols: dict[str, dict[str, str]],
    path: str,
) -> Any:
    if isinstance(value, list):
        return [
            _rewrite_reviewed_polymorphic_reference(
                item,
                symbols=symbols,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return _rewrite_reviewed_polymorphic_reference(
        value,
        symbols=symbols,
        path=path,
    )


def _rewrite_reviewed_polymorphic_reference(
    value: Any,
    *,
    symbols: dict[str, dict[str, str]],
    path: str,
) -> Any:
    if not isinstance(value, str) or not value:
        return copy.deepcopy(value)
    qualified = _parse_reviewed_global_id(value)
    if qualified is not None:
        return symbols[qualified[1]].get(value, value)
    matches = {
        (kind, kind_symbols[value])
        for kind, kind_symbols in symbols.items()
        if value in kind_symbols
    }
    if len(matches) > 1:
        kinds = sorted(kind for kind, _global_id in matches)
        raise ValueError(
            f"ambiguous reviewed polymorphic reference at {path}: {value!r} matches {kinds}"
        )
    if matches:
        return next(iter(matches))[1]
    return value


def sanitize_reviewed_perception_facts(
    facts: dict[str, Any],
    *,
    audio_refs: Iterable[str] = (),
    validate_audio_bindings: bool = True,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    report = _report_with_result_fields(facts)
    if isinstance(report, dict):
        reviewed_report = sanitize_reviewed_perception_report(
            report,
            audio_refs=audio_refs,
        )
        if reviewed_report:
            sanitized["perception_report"] = reviewed_report
    for field in PROJECTED_SCALAR_FIELDS:
        if facts.get(field) is not None:
            sanitized[field] = sanitize_reviewed_value(facts[field])
    if validate_audio_bindings:
        validate_reviewed_audio_bindings(sanitized, audio_refs=audio_refs)
    return sanitized


def rebase_reviewed_audio_namespace(
    value: Any,
    *,
    audio_ref: str,
) -> Any:
    """Rebase one audio-scoped evidence snapshot to its local index namespace.

    ``audio_ref`` remains the authoritative identity. Only index fields whose
    enclosing observation is explicitly bound to that ref are changed. This is
    used when a report is extracted from a multi-audio projection and stored as
    a single-audio Session Graph snapshot.
    """

    normalized_ref = str(audio_ref).strip()
    if not normalized_ref:
        raise ValueError("reviewed audio namespace requires a non-empty audio_ref")
    rebased = copy.deepcopy(value)

    def _visit(item: Any, *, inherited_ref: str | None = None) -> None:
        if isinstance(item, list):
            for child in item:
                _visit(child, inherited_ref=inherited_ref)
            return
        if not isinstance(item, dict):
            return

        explicit_ref = item.get("audio_ref_id")
        effective_ref = (
            str(explicit_ref) if isinstance(explicit_ref, str) and explicit_ref else inherited_ref
        )
        if effective_ref == normalized_ref and "audio_index" in item:
            item["audio_index"] = 0

        source_text_ref = item.get("source_text_audio_path")
        if source_text_ref == normalized_ref and "source_text_audio_index" in item:
            item["source_text_audio_index"] = 0

        for child in item.values():
            _visit(child, inherited_ref=effective_ref)

    _visit(rebased)
    return rebased


def _report_with_result_fields(facts: dict[str, Any]) -> dict[str, Any] | None:
    raw_report = facts.get("perception_report")
    report = copy.deepcopy(raw_report) if isinstance(raw_report, dict) else {}
    for field in REVIEWED_REPORT_ROOT_FIELDS:
        if field in {"schema_version", "duration"} or field not in facts:
            continue
        value = facts[field]
        if field in report and report[field] != value:
            raise ValueError(f"reviewed perception result conflicts with perception_report.{field}")
        report[field] = copy.deepcopy(value)
    return report or None


def _normalize_reviewed_report_aliases(report: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(report)
    if "duration_sec" not in normalized and "duration" in normalized:
        normalized["duration_sec"] = normalized["duration"]
    normalized.pop("duration", None)

    for alias, canonical in _COLLECTION_ALIASES.items():
        alias_value = normalized.pop(alias, None)
        if alias_value is None:
            continue
        if canonical == "uncertainties" and not isinstance(alias_value, list):
            alias_value = [alias_value]
        existing = normalized.get(canonical)
        if existing is None:
            normalized[canonical] = alias_value
        elif isinstance(existing, list) and isinstance(alias_value, list):
            normalized[canonical] = [*existing, *alias_value]

    semantic_review = normalized.get("semantic_span_review")
    if isinstance(semantic_review, dict):
        _copy_review_field_if_missing(
            normalized,
            semantic_review,
            source_field="selected_spans",
        )
        _copy_review_field_if_missing(
            normalized,
            semantic_review,
            source_field="candidate_segments",
        )
        _copy_review_field_if_missing(
            normalized,
            semantic_review,
            source_field="candidate_rubric_judgements",
        )

    speaker_review = normalized.get("speaker_attributed_review")
    if isinstance(speaker_review, dict):
        normalized.setdefault(
            "speaker_review_summary",
            {
                key: copy.deepcopy(speaker_review[key])
                for key in _MAPPING_FIELD_ALLOWLISTS["speaker_review_summary"]
                if key in speaker_review
            },
        )
    return normalized


def _copy_review_field_if_missing(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    source_field: str,
) -> None:
    if source_field not in target and source.get(source_field) is not None:
        target[source_field] = copy.deepcopy(source[source_field])


def _sanitize_typed_collection(
    field: str,
    value: Any,
    *,
    audio_refs: list[str],
    default_audio_ref_id: str | None = None,
    default_audio_index: int | None = None,
) -> list[Any]:
    if field == "uncertainties" and value not in (None, "") and not isinstance(value, list):
        value = [value]
    if not isinstance(value, list):
        return []
    allowed = _COLLECTION_FIELD_ALLOWLISTS[field]
    id_spec = _ID_FIELDS.get(field)
    output: list[Any] = []
    seen: dict[tuple[Any, ...], Any] = {}
    seen_positions: dict[tuple[Any, ...], int] = {}
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, dict):
            if field in {"uncertainties"}:
                output.append(
                    {
                        "code": "perception_uncertainty",
                        "message": str(raw_item),
                        "severity": "warning",
                        "blocking": False,
                    }
                )
            elif field == "events" and str(raw_item).strip():
                output.append(
                    {
                        "event_id": f"event_{index:03d}",
                        "label": str(raw_item).strip(),
                    }
                )
            continue
        item = _canonicalize_observation_fields(raw_item)
        item = {
            str(key): copy.deepcopy(child) for key, child in item.items() if str(key) in allowed
        }
        if "audio_ref_id" in allowed:
            _attach_item_audio_namespace(
                item,
                audio_refs=audio_refs,
                default_audio_ref_id=default_audio_ref_id,
                default_audio_index=default_audio_index,
            )
        identity = _observation_identity(field, item)
        if identity is not None:
            comparison_value = sanitize_reviewed_value(item)
            if identity in seen:
                if field == "speakers" and all(
                    seen[identity][key] == comparison_value[key]
                    for key in seen[identity].keys() & comparison_value.keys()
                ):
                    # Profiles supplement an existing measured identity. A
                    # missing field is not a contradictory observation; never
                    # overwrite an existing disagreeing attribute or namespace.
                    merged = {**seen[identity], **comparison_value}
                    seen[identity] = merged
                    output[seen_positions[identity]] = merged
                    continue
                if seen[identity] != comparison_value:
                    raise ValueError(
                        f"duplicate reviewed {field} identity {identity!r} has conflicting content"
                    )
                continue
            seen[identity] = comparison_value
            seen_positions[identity] = len(output)
        if id_spec is not None:
            id_field, prefix = id_spec
            item.setdefault(id_field, f"{prefix}_{index:03d}")
        if item:
            output.append(sanitize_reviewed_value(item))
    return output


def _sanitize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        )
    )


def _observation_identity(
    collection: str,
    item: dict[str, Any],
) -> tuple[Any, ...] | None:
    namespace = (item.get("audio_ref_id"), item.get("audio_index"))
    for key in (
        "span_id",
        "candidate_id",
        "target_id",
        "point_id",
        "segment_id",
        "word_id",
        "event_id",
        "evidence_id",
    ):
        if item.get(key) not in (None, ""):
            return (*namespace, key, item[key])
    # A speaker profile is unique per speaker, but observations attributed to a
    # speaker are not. One speaker may own any number of words, events, or
    # timeline segments.
    if collection in {"speakers", "selected_speaker_refs"} and item.get(
        "speaker_id"
    ) not in (None, ""):
        return (*namespace, "speaker_id", item["speaker_id"])
    if item.get("start_sec") is not None and item.get("end_sec") is not None:
        return (
            *namespace,
            "timeline",
            item["start_sec"],
            item["end_sec"],
            item.get("text") or item.get("transcript") or item.get("label"),
            item.get("speaker_id"),
        )
    return None


def _sanitize_typed_mapping(
    field: str,
    value: Any,
    *,
    audio_refs: list[str],
    default_audio_ref_id: str | None = None,
    default_audio_index: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = _MAPPING_FIELD_ALLOWLISTS[field]
    item = _canonicalize_observation_fields(value)
    item = {str(key): copy.deepcopy(child) for key, child in item.items() if str(key) in allowed}
    if "audio_ref_id" in allowed:
        _attach_item_audio_namespace(
            item,
            audio_refs=audio_refs,
            default_audio_ref_id=default_audio_ref_id,
            default_audio_index=default_audio_index,
        )
    return sanitize_reviewed_value(item)


def _canonicalize_observation_fields(value: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(value)
    if "start_sec" not in item and "start" in item:
        item["start_sec"] = item.pop("start")
    else:
        item.pop("start", None)
    if "end_sec" not in item and "end" in item:
        item["end_sec"] = item.pop("end")
    else:
        item.pop("end", None)
    if "text" not in item and "word" in item:
        item["text"] = item.pop("word")
    else:
        item.pop("word", None)
    if "candidate_id" not in item and "segment_id" in item:
        item["candidate_id"] = item["segment_id"]
    return item


def _attach_item_audio_namespace(
    item: dict[str, Any],
    *,
    audio_refs: list[str],
    default_audio_ref_id: str | None = None,
    default_audio_index: int | None = None,
) -> None:
    if not audio_refs:
        return
    raw_index = item.get("audio_index")
    raw_ref = item.get("audio_ref_id")
    if raw_ref is None and raw_index is None:
        if default_audio_ref_id is not None:
            item["audio_ref_id"] = default_audio_ref_id
            item["audio_index"] = default_audio_index
            return
        if len(audio_refs) == 1:
            item["audio_ref_id"] = audio_refs[0]
            item["audio_index"] = 0
            return
        raise ValueError(
            "reviewed observation requires audio_ref_id or audio_index for multi-audio evidence"
        )
    if raw_index is not None:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("reviewed observation audio_index must be an integer")
        if raw_index < 0 or raw_index >= len(audio_refs):
            raise ValueError("reviewed observation audio_index is outside the evidence namespace")
        if raw_ref is None:
            item["audio_ref_id"] = audio_refs[raw_index]
        elif str(raw_ref) != audio_refs[raw_index]:
            raise ValueError("reviewed observation audio ref and index disagree")
    elif raw_ref is not None:
        if str(raw_ref) not in audio_refs:
            raise ValueError("reviewed observation audio_ref_id is outside the evidence namespace")
        item["audio_index"] = audio_refs.index(str(raw_ref))
    if default_audio_ref_id is not None and str(item.get("audio_ref_id")) != default_audio_ref_id:
        raise ValueError("reviewed observation conflicts with the report audio namespace")
def _report_audio_namespace(
    report: dict[str, Any],
    *,
    audio_refs: list[str],
) -> tuple[str | None, int | None]:
    """Resolve an explicit report-level namespace for child observations."""

    if not audio_refs:
        return None, None
    if report.get("audio_ref_id") is None and report.get("audio_index") is None:
        if len(audio_refs) == 1:
            return audio_refs[0], 0
        return None, None
    binding = {
        "audio_ref_id": report.get("audio_ref_id"),
        "audio_index": report.get("audio_index"),
    }
    _attach_item_audio_namespace(binding, audio_refs=audio_refs)
    return str(binding["audio_ref_id"]), int(binding["audio_index"])


def sanitize_reviewed_perception_result(
    result: dict[str, Any],
    *,
    audio_refs: Iterable[str] = (),
) -> dict[str, Any]:
    sanitized = sanitize_reviewed_perception_facts(
        result,
        audio_refs=audio_refs,
    )
    report = sanitized.get("perception_report")
    if isinstance(report, dict):
        for field in REVIEWED_REPORT_ROOT_FIELDS:
            if field in report:
                sanitized[field] = copy.deepcopy(report[field])
    for field in _RESULT_CONTROL_FIELDS:
        if field not in result:
            continue
        if field == "component_invocation_trace":
            sanitized[field] = copy.deepcopy(result[field])
        else:
            sanitized[field] = sanitize_reviewed_value(result[field])
    validate_reviewed_audio_bindings(sanitized, audio_refs=audio_refs)
    return sanitized


def validate_reviewed_audio_bindings(
    value: dict[str, Any],
    *,
    audio_refs: Iterable[str],
) -> None:
    refs = list(dict.fromkeys(str(item) for item in audio_refs if str(item)))
    report = value.get("perception_report")
    mappings = [value]
    if isinstance(report, dict):
        mappings.append(report)

    paths = {
        str(mapping["source_text_audio_path"])
        for mapping in mappings
        if mapping.get("source_text_audio_path") not in (None, "")
    }
    indices = {
        mapping["source_text_audio_index"]
        for mapping in mappings
        if mapping.get("source_text_audio_index") is not None
    }
    if len(paths) > 1 or len(indices) > 1:
        raise ValueError("reviewed source-text audio bindings conflict")
    path = next(iter(paths), None)
    index = next(iter(indices), None)
    if path is not None and (not refs or path not in refs):
        raise ValueError("reviewed source_text_audio_path is outside the evidence namespace")
    if index is not None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("reviewed source_text_audio_index must be an integer")
        if not refs or index < 0 or index >= len(refs):
            raise ValueError("reviewed source_text_audio_index is outside the evidence namespace")
        if path is not None and refs[index] != path:
            raise ValueError("reviewed source-text audio path and index disagree")


def _validate_time_bounds(value: dict[str, Any]) -> None:
    for start_key, end_key in (
        ("start", "end"),
        ("start_sec", "end_sec"),
        ("start_time", "end_time"),
    ):
        if start_key not in value or end_key not in value:
            continue
        start = value[start_key]
        end = value[end_key]
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise TypeError("reviewed time bounds must be numeric")
        if start < 0 or end <= start:
            raise ValueError("reviewed time bounds require 0 <= start < end")
    for duration_key in ("duration", "duration_sec"):
        if duration_key not in value:
            continue
        duration = value[duration_key]
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("reviewed duration must be a non-negative number")


__all__ = [
    "PROJECTED_SCALAR_FIELDS",
    "REVIEWED_PERCEPTION_SCHEMA_VERSION",
    "REVIEWED_REPORT_ROOT_FIELDS",
    "namespace_reviewed_local_ids",
    "rebase_reviewed_audio_namespace",
    "sanitize_reviewed_perception_facts",
    "sanitize_reviewed_perception_report",
    "sanitize_reviewed_perception_result",
    "sanitize_reviewed_value",
    "validate_reviewed_audio_bindings",
]
