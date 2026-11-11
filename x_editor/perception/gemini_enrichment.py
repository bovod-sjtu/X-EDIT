"""Gemini 2.5 Pro audio-perception enrichment.

This module augments deterministic ASR/alignment/diarization output with
instruction-relevant audio observations from Gemini. Deterministic timestamps
remain preferred for cutting; Gemini is mainly used for source-text fallback and
speaker profile hints such as male/female/child/adult.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from x_editor.core.config import resolve_gemini_api_key, runtime_artifact_path
from x_editor.frontend import GeminiAudioFrontend


PROFILE_TERMS = [
    "male",
    "female",
    "man",
    "woman",
    "boy",
    "girl",
    "little girl",
    "little boy",
    "child",
    "kid",
    "adult",
    "elderly",
    "男",
    "女",
    "男声",
    "女声",
    "男说话",
    "女说话",
    "小女孩",
    "小男孩",
    "儿童",
    "小孩",
    "成人",
    "老人",
]

SPEAKER_TERMS = [
    "speaker",
    "voice",
    "说话人",
    "人声",
    "声线",
    "音色",
    "target speaker",
    "first speaker",
    "second speaker",
]


def needs_gemini_audio_perception(
    user_prompt: str,
    perception: dict[str, Any] | None,
    *,
    need_words: bool = False,
    need_speakers: bool = False,
) -> bool:
    """Return true when Gemini can supply missing perception context."""

    if not _enabled():
        return False
    if not resolve_gemini_api_key():
        return False

    perception = perception if isinstance(perception, dict) else {}
    text = user_prompt.lower()
    has_source_text = bool(str(perception.get("source_text") or "").strip())
    has_words = bool(perception.get("word_timestamps"))
    has_speaker_segments = bool(perception.get("speaker_segments") or perception.get("segments"))
    has_profiles = _has_speaker_profiles(perception)

    if need_words and (not has_source_text or not has_words):
        return True
    if need_speakers and not has_speaker_segments:
        return True
    if _contains(text, PROFILE_TERMS) and not has_profiles:
        return True
    if _contains(text, SPEAKER_TERMS) and need_speakers and not has_profiles:
        return True
    return False


def enrich_perception_with_gemini(
    *,
    user_prompt: str,
    audio_path: str | Path,
    perception: dict[str, Any] | None = None,
    reference_audio_paths: list[str] | None = None,
    base_audio_ref: str | None = None,
    reference_audio_refs: list[str] | None = None,
    output_dir: str | Path | None = None,
    frontend: GeminiAudioFrontend | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Merge Gemini audio observations into an existing perception report."""

    base = dict(perception or {})
    if not force and not needs_gemini_audio_perception(user_prompt, base, need_words=True, need_speakers=True):
        return base
    if not _enabled() and not force:
        return base

    audio = Path(audio_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Gemini perception audio does not exist: {audio}")

    try:
        gemini = _load_or_call_gemini(
            user_prompt=user_prompt,
            audio_path=audio,
            reference_audio_paths=reference_audio_paths or [],
            base_audio_ref=base_audio_ref,
            reference_audio_refs=reference_audio_refs,
            output_dir=output_dir,
            frontend=frontend,
        )
    except Exception as exc:
        merged = dict(base)
        merged.setdefault("errors", []).append({"stage": "gemini_perception", "message": str(exc)})
        return merged

    return merge_gemini_perception(base, gemini)


def merge_gemini_perception(
    base: dict[str, Any],
    gemini: dict[str, Any],
    *,
    perception_key: str = "gemini_perception",
    source_label: str = "gemini_2_5pro",
    service_key: str = "gemini_audio_perception",
    service_model: str = "gemini-2.5-pro",
) -> dict[str, Any]:
    """Merge fallback model output while preserving deterministic timestamps first."""

    merged = dict(base)
    merged[perception_key] = gemini
    merged.setdefault("services", {})
    if isinstance(merged["services"], dict):
        merged["services"][service_key] = service_model

    if not str(merged.get("source_text") or "").strip() and gemini.get("source_text"):
        merged["source_text"] = gemini["source_text"]
        merged["source_text_source"] = source_label
        for binding_field in (
            "source_text_audio_path",
            "source_text_audio_index",
        ):
            if gemini.get(binding_field) is not None:
                merged[binding_field] = gemini[binding_field]

    for key in ["audio_quality", "background"]:
        if gemini.get(key) and not merged.get(key):
            merged[key] = gemini[key]
    _extend_list_field(merged, "notable_events", gemini.get("notable_events"))
    _extend_list_field(merged, "selected_spans", gemini.get("selected_spans"))
    _extend_list_field(merged, "uncertainty", gemini.get("uncertainty"))

    merged = _merge_speaker_profiles(merged, gemini, source_label=source_label)
    if not (merged.get("speaker_segments") or merged.get("segments")):
        fallback_segments = _gemini_segments_as_speaker_segments(gemini, source_label=source_label)
        if fallback_segments:
            merged["speaker_segments"] = fallback_segments
            namespaces = {
                (segment.get("audio_ref_id"), segment.get("audio_index"))
                for segment in fallback_segments
            }
            for audio_ref_id, audio_index in sorted(
                namespaces,
                key=lambda item: (str(item[0] or ""), item[1] if item[1] is not None else -1),
            ):
                uncertainty = {
                    "code": "approximate_speaker_segments",
                    "message": (
                        f"speaker_segments came from {source_label} audio "
                        "perception and may be approximate"
                    ),
                    "severity": "warning",
                    "blocking": False,
                }
                if audio_ref_id is not None:
                    uncertainty["audio_ref_id"] = audio_ref_id
                if audio_index is not None:
                    uncertainty["audio_index"] = audio_index
                merged.setdefault("uncertainty", []).append(uncertainty)
    return merged


def _load_or_call_gemini(
    *,
    user_prompt: str,
    audio_path: Path,
    reference_audio_paths: list[str],
    base_audio_ref: str | None,
    reference_audio_refs: list[str] | None,
    output_dir: str | Path | None,
    frontend: GeminiAudioFrontend | None,
) -> dict[str, Any]:
    cache_path = _cache_path(
        user_prompt=user_prompt,
        audio_path=audio_path,
        reference_audio_paths=reference_audio_paths,
        base_audio_ref=base_audio_ref,
        reference_audio_refs=reference_audio_refs,
        output_dir=output_dir,
    )
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    report = (frontend or GeminiAudioFrontend()).perceive_sync(
        user_prompt=user_prompt,
        base_audio_path=str(audio_path),
        reference_audio_paths=reference_audio_paths,
        base_audio_ref=base_audio_ref,
        reference_audio_refs=reference_audio_refs,
    )
    data = report.model_dump()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _cache_path(
    *,
    user_prompt: str,
    audio_path: Path,
    reference_audio_paths: list[str],
    base_audio_ref: str | None,
    reference_audio_refs: list[str] | None,
    output_dir: str | Path | None,
) -> Path | None:
    if os.getenv("X_EDITOR_GEMINI_PERCEPTION_CACHE", "1").lower() in {"0", "false", "no"}:
        return None
    root = Path(
        output_dir
        or os.getenv("X_EDITOR_PERCEPTION_CACHE_DIR")
        or runtime_artifact_path("perception_cache")
    ) / "gemini_perception_cache"
    material = {
        "prompt": user_prompt,
        "audio": _file_fingerprint(audio_path),
        "refs": [_file_fingerprint(Path(path).expanduser().resolve()) for path in reference_audio_paths],
        "logical_base_ref": base_audio_ref,
        "logical_reference_refs": reference_audio_refs,
    }
    key = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return root / f"{key}.json"


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _merge_speaker_profiles(base: dict[str, Any], gemini: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    deterministic_segments = _normalized_segments(base.get("speaker_segments") or base.get("segments") or [])
    gemini_segments = _normalized_segments(gemini.get("segments") or [])
    gemini_speakers = _speaker_list_as_dict(gemini.get("speakers") or [])
    if not deterministic_segments:
        return _merge_speaker_lists(base, gemini, source_label=source_label)

    profile_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for segment in deterministic_segments:
        best = _best_overlap_segment(segment, gemini_segments)
        if best:
            profile_by_speaker.setdefault(segment["speaker_id"], []).append(best)

    if not profile_by_speaker:
        return _merge_speaker_lists(base, gemini, source_label=source_label)

    speakers = _speaker_list_as_dict(base.get("speakers") or [])
    for speaker_id, matches in profile_by_speaker.items():
        target = speakers.setdefault(speaker_id, {"speaker_id": speaker_id})
        descriptions = []
        genders = []
        confidences = []
        for item in matches:
            gemini_profile = gemini_speakers.get(str(item.get("speaker_id") or ""))
            descriptions.extend(
                str(value)
                for value in [
                    gemini_profile.get("description") if gemini_profile else None,
                    gemini_profile.get("role") if gemini_profile else None,
                    item.get("description"),
                    item.get("transcript"),
                ]
                if value
            )
            if item.get("perceived_gender"):
                genders.append(str(item["perceived_gender"]))
            elif gemini_profile and gemini_profile.get("perceived_gender"):
                genders.append(str(gemini_profile["perceived_gender"]))
            if item.get("confidence"):
                confidences.append(str(item["confidence"]))
            elif gemini_profile and gemini_profile.get("confidence"):
                confidences.append(str(gemini_profile["confidence"]))
        if descriptions:
            merged_descriptions = []
            if target.get("description"):
                merged_descriptions.append(str(target["description"]))
            merged_descriptions.extend(descriptions)
            target["description"] = " | ".join(dict.fromkeys(merged_descriptions))
        if genders and not target.get("perceived_gender"):
            target["perceived_gender"] = _majority(genders)
        target.setdefault("confidence", _lowest_confidence(confidences) if confidences else "medium")
        target["profile_source"] = f"{source_label}_overlap"
        target["gemini_matched_segments"] = [
            {key: item.get(key) for key in ["segment_id", "speaker_id", "start_sec", "end_sec", "description", "perceived_gender", "confidence"]}
            for item in matches
        ]

    base["speakers"] = list(speakers.values())
    _enrich_segments_with_profiles(base, profile_by_speaker, source_label=source_label)
    return base


def _merge_speaker_lists(base: dict[str, Any], gemini: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    speakers = _speaker_list_as_dict(base.get("speakers") or [])
    for item in gemini.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        speaker_id = str(item.get("speaker_id") or f"gemini_speaker_{len(speakers)}")
        if speaker_id in speakers:
            continue
        copied = dict(item)
        copied["speaker_id"] = speaker_id
        copied["profile_source"] = source_label
        speakers[speaker_id] = copied
    base["speakers"] = list(speakers.values())
    return base


def _enrich_segments_with_profiles(
    base: dict[str, Any],
    profile_by_speaker: dict[str, list[dict[str, Any]]],
    *,
    source_label: str,
) -> None:
    raw_segments = base.get("speaker_segments") or base.get("segments") or []
    if not isinstance(raw_segments, list):
        return
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        speaker_id = str(segment.get("speaker_id") or segment.get("speaker") or segment.get("label") or "")
        matches = profile_by_speaker.get(speaker_id) or []
        if not matches:
            continue
        best = matches[0]
        segment.setdefault("description", best.get("description"))
        segment.setdefault("perceived_gender", best.get("perceived_gender"))
        segment.setdefault("profile_source", f"{source_label}_overlap")


def _gemini_segments_as_speaker_segments(gemini: dict[str, Any], *, source_label: str) -> list[dict[str, Any]]:
    segments = []
    for item in gemini.get("segments") or []:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        segment = {
            "speaker_id": str(item.get("speaker_id") or "gemini_speaker"),
            "start_sec": start_f,
            "end_sec": end_f,
            "transcript": item.get("transcript"),
            "description": item.get("description"),
            "perceived_gender": item.get("perceived_gender"),
            "confidence": item.get("confidence", "low"),
            "source": source_label,
        }
        for namespace_field in ("audio_ref_id", "audio_index"):
            if item.get(namespace_field) is not None:
                segment[namespace_field] = item[namespace_field]
        segments.append(segment)
    return segments


def _normalized_segments(items: list[Any]) -> list[dict[str, Any]]:
    segments = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id") or item.get("speaker") or item.get("label") or f"speaker_{index}"
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        copied = dict(item)
        copied.update({"speaker_id": str(speaker_id), "start_sec": start_f, "end_sec": end_f})
        segments.append(copied)
    return sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"]))


def _best_overlap_segment(segment: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    best_overlap = 0.0
    for candidate in candidates:
        overlap = _overlap_seconds(
            float(segment["start_sec"]),
            float(segment["end_sec"]),
            float(candidate["start_sec"]),
            float(candidate["end_sec"]),
        )
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best if best_overlap > 0 else None


def _speaker_list_as_dict(items: list[Any]) -> dict[str, dict[str, Any]]:
    speakers = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id") or item.get("id")
        if speaker_id is None:
            continue
        speakers[str(speaker_id)] = dict(item)
    return speakers


def _has_speaker_profiles(perception: dict[str, Any]) -> bool:
    for collection in [perception.get("speakers"), perception.get("speaker_segments"), perception.get("segments")]:
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            if item.get("description") or item.get("perceived_gender") or item.get("gender") or item.get("age"):
                return True
    return False


def _extend_list_field(target: dict[str, Any], key: str, values: Any) -> None:
    if not values:
        return
    if not isinstance(values, list):
        values = [values]
    target.setdefault(key, [])
    if not isinstance(target[key], list):
        target[key] = [target[key]]
    for value in values:
        if value not in target[key]:
            target[key].append(value)


def _majority(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _lowest_confidence(values: list[str]) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(values, key=lambda value: order.get(value, 1), reverse=True)[0]


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _enabled() -> bool:
    return os.getenv("X_EDITOR_ENABLE_GEMINI_PERCEPTION_FALLBACK", "1").lower() not in {"0", "false", "no"}
