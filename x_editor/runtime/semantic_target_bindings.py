"""Mechanical projection from scoped Reasoner refs to executable evidence fields."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from x_editor.reasoner.contracts import (
    SemanticCandidateRef,
    SemanticReasoningMemo,
    SemanticTargetCandidateKind,
)


@dataclass(frozen=True)
class MaterializedSemanticEvidence:
    reviewed_evidence: dict[str, Any]
    evidence_path_sources: dict[str, list[str]]
    evidence_path_audio_refs: dict[str, str]


@dataclass(frozen=True)
class _CandidateRecord:
    audio_ref_id: str
    candidate_id: str
    kind: Literal["span", "speaker", "audio_event", "acoustic_region"]
    value: dict[str, Any]
    path: str


_CandidateLookup = dict[
    SemanticTargetCandidateKind,
    dict[tuple[str, str], list[_CandidateRecord]],
]

_CANDIDATE_COLLECTIONS: dict[
    str,
    tuple[SemanticTargetCandidateKind, str],
] = {
    "candidate_segments": ("span", "candidate_id"),
    "segments": ("span", "segment_id"),
    "word_timestamps": ("span", "word_id"),
    "speaker_segments": ("span", "segment_id"),
    "event_segments": ("audio_event", "segment_id"),
    "prosody_segments": ("span", "segment_id"),
    "acoustic_segments": ("acoustic_region", "segment_id"),
    "audio_caption_segments": ("audio_event", "segment_id"),
    "selected_spans": ("span", "span_id"),
    "events": ("audio_event", "event_id"),
}
_SPEAKER_COLLECTIONS = {"speakers", "speaker_profiles", "speaker_segments"}
_SPAN_COPY_FIELDS = (
    "audio_index",
    "start_sec",
    "end_sec",
    "speaker_id",
    "text",
    "reported_text",
    "word_ids",
    "label",
    "caption",
    "description",
    "event_type",
    "detected_properties",
    "source_refs",
    "source",
    "source_kind",
)


def materialize_semantic_target_bindings(
    reviewed_evidence: Mapping[str, Any],
    *,
    memo: SemanticReasoningMemo,
    evidence_path_sources: Mapping[str, list[str]] | None = None,
    evidence_path_audio_refs: Mapping[str, str] | None = None,
) -> MaterializedSemanticEvidence:
    """Dereference admitted audio-scoped candidates without interpreting semantics."""

    projected = copy.deepcopy(dict(reviewed_evidence))
    path_sources = {
        str(path): list(dict.fromkeys(str(item) for item in source_ids))
        for path, source_ids in (evidence_path_sources or {}).items()
    }
    path_audio_refs = {
        str(path): str(audio_ref) for path, audio_ref in (evidence_path_audio_refs or {}).items()
    }
    if memo.phase != "evidence_review" or not memo.target_selectors:
        return MaterializedSemanticEvidence(projected, path_sources, path_audio_refs)

    report = projected.get("perception_report")
    if not isinstance(report, dict):
        raise TypeError("semantic target materialization requires perception_report")
    candidates = _candidate_records(report, path_audio_refs=path_audio_refs)
    bindings = {item.target_id: item for item in memo.target_bindings}

    selected_spans: list[dict[str, Any]] = []
    selected_span_keys: dict[tuple[str, str, float, float, str | None], int] = {}
    selected_speaker_refs: list[dict[str, str]] = []
    target_bindings: list[dict[str, Any]] = []

    for selector in memo.target_selectors:
        binding = bindings.get(selector.target_id)
        resolved_span_ids: list[str] = []
        resolved_speaker_refs: list[dict[str, str]] = []
        resolved_event_ids: list[str] = []
        resolved_acoustic_ids: list[str] = []
        if binding is not None and binding.status == "resolved":
            for candidate_ref in binding.candidate_refs:
                records = _records_for_ref(
                    candidates,
                    kind=binding.candidate_kind,
                    candidate_ref=candidate_ref,
                )
                if binding.candidate_kind in {
                    "span",
                    "audio_event",
                    "acoustic_region",
                }:
                    span_record = _one_span_candidate(candidate_ref, records)
                    if _valid_interval(span_record.value):
                        span_id = _append_selected_span(
                            selected_spans,
                            selected_span_keys=selected_span_keys,
                            record=span_record,
                            candidate_ref=candidate_ref,
                            target_id=selector.target_id,
                            confidence=binding.confidence,
                            reason=binding.reason,
                        )
                        resolved_span_ids.append(span_id)
                    speaker_id = _optional_text(span_record.value.get("speaker_id"))
                    if speaker_id:
                        resolved_speaker_refs.append(
                            _speaker_ref(candidate_ref.audio_ref_id, speaker_id)
                        )
                    if binding.candidate_kind == "audio_event":
                        resolved_event_ids.append(candidate_ref.candidate_id)
                    elif binding.candidate_kind == "acoustic_region":
                        resolved_acoustic_ids.append(candidate_ref.candidate_id)
                else:
                    speaker_ref = _speaker_ref(
                        candidate_ref.audio_ref_id,
                        candidate_ref.candidate_id,
                    )
                    _append_unique_mapping(selected_speaker_refs, speaker_ref)
                    _append_unique_mapping(resolved_speaker_refs, speaker_ref)
                    for record in records:
                        if not _valid_interval(record.value):
                            continue
                        span_id = _append_selected_span(
                            selected_spans,
                            selected_span_keys=selected_span_keys,
                            record=record,
                            candidate_ref=candidate_ref,
                            target_id=selector.target_id,
                            confidence=binding.confidence,
                            reason=binding.reason,
                        )
                        resolved_span_ids.append(span_id)

        resolved_candidate_refs = (
            [item.model_dump(mode="json") for item in binding.candidate_refs]
            if binding is not None
            else []
        )
        target_bindings.append(
            {
                "target_id": selector.target_id,
                "target_kind": selector.candidate_kind,
                "target_value": selector.description,
                "status": binding.status if binding is not None else "unresolved",
                "resolved_candidate_refs": resolved_candidate_refs,
                "resolved_span_ids": list(dict.fromkeys(resolved_span_ids)),
                "resolved_speaker_refs": _dedupe_mappings(resolved_speaker_refs),
                "resolved_speaker_ids": _unscoped_speaker_ids(resolved_speaker_refs),
                "resolved_event_ids": list(dict.fromkeys(resolved_event_ids)),
                "resolved_acoustic_ids": list(dict.fromkeys(resolved_acoustic_ids)),
                "audio_ref_ids": list(
                    dict.fromkeys(item["audio_ref_id"] for item in resolved_candidate_refs)
                ),
                "confidence": binding.confidence if binding is not None else None,
                "reason": (
                    binding.reason
                    if binding is not None
                    else "Reasoner did not return a binding for this target selector"
                ),
                "source_refs": list(
                    dict.fromkeys(item["candidate_id"] for item in resolved_candidate_refs)
                ),
            }
        )

    report["selected_spans"] = selected_spans
    report["selected_speaker_refs"] = _dedupe_mappings(selected_speaker_refs)
    report["selected_speaker_ids"] = _unscoped_speaker_ids(selected_speaker_refs)
    report["target_bindings"] = target_bindings
    _project_materialized_provenance(
        report=report,
        candidates=candidates,
        memo=memo,
        path_sources=path_sources,
        path_audio_refs=path_audio_refs,
    )
    return MaterializedSemanticEvidence(projected, path_sources, path_audio_refs)


def _candidate_records(
    report: Mapping[str, Any],
    *,
    path_audio_refs: Mapping[str, str],
) -> _CandidateLookup:
    records: _CandidateLookup = {
        "span": {},
        "speaker": {},
        "audio_event": {},
        "acoustic_region": {},
    }

    def append(
        kind: SemanticTargetCandidateKind,
        candidate_id: Any,
        value: Any,
        *,
        path: str,
        inherited_audio_ref_id: str | None,
    ) -> None:
        candidate_text = _optional_text(candidate_id)
        if candidate_text is None or not isinstance(value, Mapping):
            return
        audio_ref_id = _resolved_candidate_audio_ref(
            value,
            path=path,
            inherited_audio_ref_id=inherited_audio_ref_id,
            path_audio_refs=path_audio_refs,
        )
        if audio_ref_id is None:
            return
        copied = copy.deepcopy(dict(value))
        copied["audio_ref_id"] = audio_ref_id
        record = _CandidateRecord(
            audio_ref_id=audio_ref_id,
            candidate_id=candidate_text,
            kind=kind,
            value=copied,
            path=path,
        )
        current = records[kind].setdefault((audio_ref_id, candidate_text), [])
        identity = _record_measurement_identity(record)
        for index, existing in enumerate(current):
            if _record_measurement_identity(existing) != identity:
                continue
            if _record_preference(record) > _record_preference(existing):
                current[index] = record
            return
        current.append(record)

    def visit(
        value: Any,
        *,
        path: str,
        parent_key: str | None = None,
        inherited_audio_ref_id: str | None = None,
    ) -> None:
        if isinstance(value, Mapping):
            own_audio_ref_id = _optional_text(value.get("audio_ref_id"))
            if (
                own_audio_ref_id is not None
                and inherited_audio_ref_id is not None
                and own_audio_ref_id != inherited_audio_ref_id
            ):
                raise ValueError(
                    f"candidate namespace conflict at {path}: "
                    f"{own_audio_ref_id!r} != {inherited_audio_ref_id!r}"
                )
            effective_audio_ref_id = own_audio_ref_id or inherited_audio_ref_id
            if parent_key in _CANDIDATE_COLLECTIONS:
                configured_kind, candidate_id_field = _CANDIDATE_COLLECTIONS[parent_key]
                declared_kind = value.get("candidate_kind")
                candidate_kind = (
                    declared_kind
                    if parent_key == "candidate_segments" and declared_kind in records
                    else configured_kind
                )
                append(
                    candidate_kind,
                    value.get(candidate_id_field),
                    value,
                    path=path,
                    inherited_audio_ref_id=effective_audio_ref_id,
                )
            if parent_key in _SPEAKER_COLLECTIONS or parent_key == "candidate_segments":
                append(
                    "speaker",
                    value.get("speaker_id"),
                    value,
                    path=path,
                    inherited_audio_ref_id=effective_audio_ref_id,
                )
            for key, child in value.items():
                visit(
                    child,
                    path=f"{path}.{key}",
                    parent_key=str(key),
                    inherited_audio_ref_id=effective_audio_ref_id,
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(
                    child,
                    path=f"{path}{_projection_list_selector(child, index)}",
                    parent_key=parent_key,
                    inherited_audio_ref_id=inherited_audio_ref_id,
                )

    visit(report, path="perception_report")
    return records


def _records_for_ref(
    candidates: _CandidateLookup,
    *,
    kind: SemanticTargetCandidateKind,
    candidate_ref: SemanticCandidateRef,
) -> list[_CandidateRecord]:
    records = candidates[kind].get(
        (candidate_ref.audio_ref_id, candidate_ref.candidate_id),
        [],
    )
    if not records:
        raise ValueError(
            "resolved semantic target has no admitted scoped candidate: "
            f"audio_ref_id={candidate_ref.audio_ref_id!r}, "
            f"kind={kind!r}, candidate_id={candidate_ref.candidate_id!r}"
        )
    return records


def _one_span_candidate(
    candidate_ref: SemanticCandidateRef,
    records: list[_CandidateRecord],
) -> _CandidateRecord:
    valid = [item for item in records if _valid_interval(item.value)]
    unique = {
        (
            item.audio_ref_id,
            float(item.value["start_sec"]),
            float(item.value["end_sec"]),
        )
        for item in valid
    }
    if len(unique) != 1:
        raise ValueError(
            "scoped span candidate does not identify one measured interval: "
            f"audio_ref_id={candidate_ref.audio_ref_id!r}, "
            f"candidate_id={candidate_ref.candidate_id!r}"
        )
    return max(valid, key=_record_preference)


def _append_selected_span(
    selected_spans: list[dict[str, Any]],
    *,
    selected_span_keys: dict[
        tuple[str, str, float, float, str | None],
        int,
    ],
    record: _CandidateRecord,
    candidate_ref: SemanticCandidateRef,
    target_id: str,
    confidence: float | None,
    reason: str,
) -> str:
    start_sec = float(record.value["start_sec"])
    end_sec = float(record.value["end_sec"])
    speaker_id = _optional_text(record.value.get("speaker_id"))
    key = (target_id, record.audio_ref_id, start_sec, end_sec, speaker_id)
    source_candidate_ref = candidate_ref.model_dump(mode="json")
    existing_index = selected_span_keys.get(key)
    if existing_index is not None:
        existing = selected_spans[existing_index]
        existing["source_candidate_refs"] = _dedupe_mappings(
            [*(existing.get("source_candidate_refs") or []), source_candidate_ref]
        )
        existing["source_refs"] = list(
            dict.fromkeys(
                [
                    *(existing.get("source_refs") or []),
                    candidate_ref.candidate_id,
                ]
            )
        )
        if not existing.get("speaker_id") and record.value.get("speaker_id"):
            existing["speaker_id"] = record.value["speaker_id"]
        return str(existing["span_id"])

    span_id = f"reasoner_span_{len(selected_spans):03d}"
    value = record.value
    span = {
        field: copy.deepcopy(value[field])
        for field in _SPAN_COPY_FIELDS
        if field in value and value[field] is not None
    }
    if "text" not in span and isinstance(value.get("transcript"), str):
        span["text"] = value["transcript"]
    span.update(
        {
            "span_id": span_id,
            "audio_ref_id": record.audio_ref_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "semantic_role": "edit_target",
            "target_id": target_id,
            "source_candidate_refs": [source_candidate_ref],
            "source_refs": list(
                dict.fromkeys([candidate_ref.candidate_id, *(span.get("source_refs") or [])])
            ),
            "confidence": confidence,
            "reason": reason,
            "source": "reasoner_candidate_binding",
            "source_kind": "semantic_binding",
        }
    )
    selected_span_keys[key] = len(selected_spans)
    selected_spans.append(span)
    return span_id


def _project_materialized_provenance(
    *,
    report: dict[str, Any],
    candidates: _CandidateLookup,
    memo: SemanticReasoningMemo,
    path_sources: dict[str, list[str]],
    path_audio_refs: dict[str, str],
) -> None:
    bindings = {item.target_id: item for item in memo.target_bindings}
    for target_index, selector in enumerate(memo.target_selectors):
        binding = bindings.get(selector.target_id)
        if binding is None:
            continue
        candidate_records = [
            record
            for candidate_ref in binding.candidate_refs
            for record in candidates[binding.candidate_kind].get(
                (candidate_ref.audio_ref_id, candidate_ref.candidate_id),
                [],
            )
        ]
        source_ids = _sources_for_records(candidate_records, path_sources)
        audio_ref = _one_record_audio_ref(candidate_records)
        binding_prefix = f"perception_report.target_bindings.{target_index}"
        _assign_projection_paths(
            report["target_bindings"][target_index],
            prefix=binding_prefix,
            source_ids=source_ids,
            audio_ref=audio_ref,
            path_sources=path_sources,
            path_audio_refs=path_audio_refs,
        )
        for ref_index, candidate_ref in enumerate(binding.candidate_refs):
            records = candidates[binding.candidate_kind].get(
                (candidate_ref.audio_ref_id, candidate_ref.candidate_id),
                [],
            )
            _assign_projection_paths(
                candidate_ref.model_dump(mode="json"),
                prefix=f"{binding_prefix}.resolved_candidate_refs.{ref_index}",
                source_ids=_sources_for_records(records, path_sources),
                audio_ref=candidate_ref.audio_ref_id,
                path_sources=path_sources,
                path_audio_refs=path_audio_refs,
            )

    for span_index, span in enumerate(report["selected_spans"]):
        candidate_records = [
            record
            for raw_ref in span.get("source_candidate_refs") or []
            if isinstance(raw_ref, Mapping)
            for kind in (
                "span",
                "speaker",
                "audio_event",
                "acoustic_region",
            )
            for record in candidates[kind].get(
                (
                    str(raw_ref.get("audio_ref_id") or ""),
                    str(raw_ref.get("candidate_id") or ""),
                ),
                [],
            )
            if _valid_interval(record.value)
        ]
        _assign_projection_paths(
            span,
            prefix=f"perception_report.selected_spans.{span_index}",
            source_ids=_sources_for_records(candidate_records, path_sources),
            audio_ref=str(span["audio_ref_id"]),
            path_sources=path_sources,
            path_audio_refs=path_audio_refs,
        )

    for speaker_index, speaker_ref in enumerate(report["selected_speaker_refs"]):
        records = candidates["speaker"].get(
            (speaker_ref["audio_ref_id"], speaker_ref["speaker_id"]),
            [],
        )
        _assign_projection_paths(
            speaker_ref,
            prefix=f"perception_report.selected_speaker_refs.{speaker_index}",
            source_ids=_sources_for_records(records, path_sources),
            audio_ref=speaker_ref["audio_ref_id"],
            path_sources=path_sources,
            path_audio_refs=path_audio_refs,
        )


def _assign_projection_paths(
    value: Any,
    *,
    prefix: str,
    source_ids: list[str],
    audio_ref: str | None,
    path_sources: dict[str, list[str]],
    path_audio_refs: dict[str, str],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assign_projection_paths(
                child,
                prefix=f"{prefix}.{key}",
                source_ids=source_ids,
                audio_ref=audio_ref,
                path_sources=path_sources,
                path_audio_refs=path_audio_refs,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assign_projection_paths(
                child,
                prefix=f"{prefix}.{index}",
                source_ids=source_ids,
                audio_ref=audio_ref,
                path_sources=path_sources,
                path_audio_refs=path_audio_refs,
            )
        return
    if source_ids:
        path_sources[prefix] = list(source_ids)
    if audio_ref is not None:
        path_audio_refs[prefix] = audio_ref


def _resolved_candidate_audio_ref(
    value: Mapping[str, Any],
    *,
    path: str,
    inherited_audio_ref_id: str | None,
    path_audio_refs: Mapping[str, str],
) -> str | None:
    internal = _optional_text(value.get("audio_ref_id"))
    recovered = {
        str(audio_ref)
        for source_path, audio_ref in path_audio_refs.items()
        if _paths_overlap(path, str(source_path)) and str(audio_ref)
    }
    if inherited_audio_ref_id:
        recovered.add(inherited_audio_ref_id)
    if internal is not None:
        if recovered and recovered != {internal}:
            raise ValueError(f"candidate audio_ref_id conflicts with reviewed provenance at {path}")
        return internal
    return next(iter(recovered)) if len(recovered) == 1 else None


def _paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or right.startswith((left + ".", left + "["))
        or left.startswith((right + ".", right + "["))
    )


def _sources_for_records(
    records: list[_CandidateRecord],
    path_sources: Mapping[str, list[str]],
) -> list[str]:
    return list(
        dict.fromkeys(
            source_id
            for record in records
            for path, source_ids in path_sources.items()
            if _paths_overlap(record.path, str(path))
            for source_id in source_ids
        )
    )


def _one_record_audio_ref(records: list[_CandidateRecord]) -> str | None:
    refs = list(dict.fromkeys(record.audio_ref_id for record in records))
    return refs[0] if len(refs) == 1 else None


def _projection_list_selector(value: Any, index: int) -> str:
    identity = _projection_list_identity(value)
    return f"[{identity[0]}={identity[1:]}]" if identity is not None else f"[{index}]"


def _projection_list_identity(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("audio_ref_id") not in (None, "") and "report" in value and "status" in value:
        return ("audio_ref_id", value["audio_ref_id"])
    for key in (
        "span_id",
        "candidate_id",
        "target_id",
        "point_id",
        "segment_id",
        "word_id",
        "event_id",
        "evidence_id",
        "speaker_id",
        "id",
        "index",
    ):
        if key in value:
            return (key, value[key])
    if "start" in value and "end" in value:
        return ("time", value["start"], value["end"])
    if "start_sec" in value and "end_sec" in value:
        return ("time_sec", value["start_sec"], value["end_sec"])
    return None


def _record_measurement_identity(record: _CandidateRecord) -> tuple[Any, ...]:
    if _valid_interval(record.value):
        return (
            record.kind,
            record.audio_ref_id,
            float(record.value["start_sec"]),
            float(record.value["end_sec"]),
            _optional_text(record.value.get("speaker_id")),
        )
    return (record.kind, record.audio_ref_id, record.candidate_id)


def _record_preference(record: _CandidateRecord) -> tuple[int, int, str]:
    populated = sum(value not in (None, "", [], {}) for value in record.value.values())
    return (populated, -len(record.path), record.path)


def _speaker_ref(audio_ref_id: str, speaker_id: str) -> dict[str, str]:
    return {"audio_ref_id": audio_ref_id, "speaker_id": speaker_id}


def _append_unique_mapping(
    values: list[dict[str, str]],
    item: dict[str, str],
) -> None:
    if item not in values:
        values.append(item)


def _dedupe_mappings(values: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for item in values:
        _append_unique_mapping(output, item)
    return output


def _unscoped_speaker_ids(speaker_refs: list[dict[str, str]]) -> list[str]:
    scoped = _dedupe_mappings(speaker_refs)
    audio_refs = {item["audio_ref_id"] for item in scoped}
    if len(audio_refs) > 1:
        return []
    return list(dict.fromkeys(item["speaker_id"] for item in scoped))


def _valid_interval(value: Mapping[str, Any]) -> bool:
    try:
        start = float(value["start_sec"])
        end = float(value["end_sec"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(start) and math.isfinite(end) and end > start


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


__all__ = ["MaterializedSemanticEvidence", "materialize_semantic_target_bindings"]
