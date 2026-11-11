"""Intra-audio speaker timbre transfer runtime.

This module handles edits such as "make the second speaker sound like the
first speaker" when both source and reference voices are inside the same input
audio. It uses diarization for speaker time spans, extracts internal reference
clips, runs SeedVC on target-speaker chunks, then places the converted chunks
back onto the original timeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydub import AudioSegment

from x_editor.perception import SpeechPerceptionClient
from x_editor.tools.catalog.seed_vc.client import SeedVCClient
from x_editor.tools.mcp.schemas import ToolResult


ToolGateway = Callable[[str, dict[str, Any]], dict[str, Any]]


ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "第五": 5,
    "第四": 4,
    "第三": 3,
    "第二": 2,
    "第一": 1,
    "第5": 5,
    "第4": 4,
    "第3": 3,
    "第2": 2,
    "第1": 1,
    "五号": 5,
    "四号": 4,
    "三号": 3,
    "二号": 2,
    "一号": 1,
}


@dataclass(frozen=True)
class SpeakerSegment:
    speaker_id: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class SpeakerMapping:
    target_speaker_id: str
    reference_speaker_id: str
    reason: str


def execute_intra_audio_voice_transfer(
    *,
    state: dict[str, Any],
    output_path: str,
    output_dir: str,
    instruction: str | None = None,
    mapping_params: dict[str, Any] | None = None,
    tool_gateway: ToolGateway | None = None,
) -> ToolResult:
    """Run diarization -> internal reference extraction -> SeedVC -> stitch."""

    input_audio_path = state.get("base_audio_path")
    if not isinstance(input_audio_path, str) or not input_audio_path:
        return _failure("intra_audio_voice_transfer requires state.base_audio_path")
    input_audio = Path(input_audio_path)
    if not input_audio.exists():
        return _failure(f"input audio does not exist: {input_audio}")

    prompt = instruction or str(state.get("user_prompt") or "")
    work_dir = Path(output_dir) / "intra_audio_voice_transfer" / Path(output_path).stem
    segments_dir = work_dir / "segments"
    refs_dir = work_dir / "references"
    converted_dir = work_dir / "converted"
    for directory in [segments_dir, refs_dir, converted_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        original = AudioSegment.from_file(input_audio)
        duration_sec = len(original) / 1000.0
        perception, perception_source = _ensure_diarization(state, input_audio)
        word_timestamps = _normalized_words(perception.get("word_timestamps") or [])
        speaker_segments = _normalized_speaker_segments(perception)
        if not speaker_segments:
            return _failure(
                "intra_audio_voice_transfer requires diarization speaker_segments; "
                "pyannote did not return usable speaker turns."
            )
        speaker_segments = _clip_segments(speaker_segments, duration_sec)
        speaker_segments = _snap_segments_to_word_boundaries(speaker_segments, word_timestamps, duration_sec)
        speaker_segments = _merge_adjacent_segments(speaker_segments, max_gap_sec=0.12)
        speaker_segments = _split_long_segments(
            speaker_segments,
            word_timestamps,
            max_segment_sec=_max_segment_seconds(),
            duration_sec=duration_sec,
        )
        speaker_order = _speaker_order(speaker_segments)
        if len(speaker_order) < 2:
            return _failure(
                "intra_audio_voice_transfer needs at least two diarized speakers; "
                f"found {len(speaker_order)}."
            )

        profiles = _speaker_profiles(perception)
        mappings = _resolve_structured_mappings(
            mapping_params or {},
            speaker_order,
            profiles,
            speaker_segments=speaker_segments,
        )
        if not mappings:
            mappings = _resolve_mappings(prompt, speaker_order, speaker_segments, profiles)
        if not mappings:
            return _failure(
                "Could not resolve an internal speaker transfer mapping from the instruction. "
                "Use an explicit mapping such as 'second speaker to match first speaker'."
            )

        warnings: list[str] = []
        ref_paths: dict[str, str] = {}
        for ref_speaker_id in sorted({item.reference_speaker_id for item in mappings}):
            ref_path = refs_dir / f"reference_{_safe_name(ref_speaker_id)}.wav"
            ref_segments = [seg for seg in speaker_segments if seg.speaker_id == ref_speaker_id]
            ref_result = _write_reference_audio(original, ref_segments, ref_path)
            if ref_result["duration_sec"] < _min_reference_seconds():
                return _failure(
                    f"Reference speaker {ref_speaker_id} has only "
                    f"{ref_result['duration_sec']:.2f}s usable audio; SeedVC needs a longer internal reference."
            )
            ref_paths[ref_speaker_id] = str(ref_path)

        mapping_by_target = {item.target_speaker_id: item for item in mappings}
        conversion_segments = _conversion_segments_from_explicit_target_spans(
            mapping_params or {},
            speaker_segments,
            mappings,
        ) or speaker_segments
        edited = original
        converted_records: list[dict[str, Any]] = []
        min_target_sec = _min_target_seconds(prompt, mapping_params or {})
        seed_client = (
            None
            if tool_gateway is not None
            else SeedVCClient(
                os.getenv("SEED_VC_SERVICE_URL", "http://127.0.0.1:8003"),
                timeout=float(os.getenv("X_EDITOR_SEED_VC_TIMEOUT", "300")),
            )
        )

        for index, segment in enumerate(conversion_segments):
            mapping = mapping_by_target.get(segment.speaker_id)
            if mapping is None:
                continue
            if segment.duration_sec < min_target_sec:
                warnings.append(
                    f"Skipped short target segment {index} for speaker {segment.speaker_id}: "
                    f"{segment.duration_sec:.2f}s."
                )
                continue

            source_path = segments_dir / f"target_{index:03d}_{_safe_name(segment.speaker_id)}.wav"
            converted_path = converted_dir / f"converted_{index:03d}_{_safe_name(segment.speaker_id)}_to_{_safe_name(mapping.reference_speaker_id)}.wav"
            _slice_audio(original, segment.start_sec, segment.end_sec).export(source_path, format="wav")
            seed_vc_args = {
                "source_audio_path": str(source_path),
                "reference_audio_path": ref_paths[mapping.reference_speaker_id],
                "output_path": str(converted_path),
                "diffusion_steps": int(
                    os.getenv("X_EDITOR_INTRA_VOICE_SEEDVC_STEPS", "30")
                ),
                "length_adjust": float(
                    os.getenv("X_EDITOR_INTRA_VOICE_SEEDVC_LENGTH_ADJUST", "1.0")
                ),
                "inference_cfg_rate": float(
                    os.getenv("X_EDITOR_INTRA_VOICE_SEEDVC_CFG", "0.7")
                ),
            }
            if tool_gateway is not None:
                service_result = tool_gateway("seed_vc", seed_vc_args)
                if not isinstance(service_result, dict):
                    raise RuntimeError(
                        "intra_audio_voice_transfer seed_vc gateway returned a non-object result"
                    )
            else:
                assert seed_client is not None
                service_result = seed_client.convert(**seed_vc_args)
            converted_audio = _fit_audio_duration(
                AudioSegment.from_file(converted_path),
                int(round(segment.duration_sec * 1000)),
            )
            fitted_path = converted_dir / f"converted_{index:03d}_fitted.wav"
            converted_audio.export(fitted_path, format="wav")
            edited = _replace_range(edited, converted_audio, segment.start_sec, segment.end_sec)
            converted_records.append(
                {
                    "index": index,
                    "target_speaker_id": segment.speaker_id,
                    "reference_speaker_id": mapping.reference_speaker_id,
                    "start_sec": round(segment.start_sec, 6),
                    "end_sec": round(segment.end_sec, 6),
                    "duration_sec": round(segment.duration_sec, 6),
                    "source_segment_path": str(source_path),
                    "converted_segment_path": str(converted_path),
                    "fitted_segment_path": str(fitted_path),
                    "seed_vc_output": service_result,
                }
            )

        if not converted_records:
            return _failure("No target speaker segments were long enough to convert.")

        final_output = Path(output_path)
        final_output.parent.mkdir(parents=True, exist_ok=True)
        edited.export(final_output, format="wav")
        source_copy = work_dir / "source.wav"
        if not source_copy.exists():
            shutil.copy2(input_audio, source_copy)

        speaker_segments_path = work_dir / "speaker_segments.json"
        speaker_segments_path.write_text(
            json.dumps([_segment_dict(seg) for seg in speaker_segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest = {
            "tool": "intra_audio_voice_transfer",
            "instruction": prompt,
            "input_audio_path": str(input_audio),
            "source_audio_copy": str(source_copy),
            "output_audio_path": str(final_output),
            "perception_source": perception_source,
            "duration_sec": duration_sec,
            "speaker_order": speaker_order,
            "speaker_profiles": profiles,
            "speaker_segments_path": str(speaker_segments_path),
            "mappings": [item.__dict__ for item in mappings],
            "explicit_target_segments": (
                [_segment_dict(seg) for seg in conversion_segments]
                if conversion_segments is not speaker_segments
                else []
            ),
            "reference_audio_paths": ref_paths,
            "converted_segments": converted_records,
            "warnings": warnings,
            "services": {
                "diarization_model": _diarization_model(),
                "seed_vc_url": os.getenv("SEED_VC_SERVICE_URL", "http://127.0.0.1:8003"),
                "seed_vc_execution": (
                    "tool_gateway" if tool_gateway is not None else "direct_client"
                ),
            },
            "constraints": {
                "timeline_duration_preserved": True,
                "converted_chunks_fit_original_duration": True,
                "word_boundary_snapping": bool(word_timestamps),
                "max_segment_sec": _max_segment_seconds(),
            },
        }
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return ToolResult(
            tool_name="intra_audio_voice_transfer.execute",
            success=True,
            output={
                "output_audio_path": str(final_output),
                "manifest_path": str(manifest_path),
                "speaker_segments_path": str(speaker_segments_path),
                "source_audio_copy": str(source_copy),
                "reference_audio_paths": ref_paths,
                "converted_segment_count": len(converted_records),
                "mappings": [item.__dict__ for item in mappings],
                "warnings": warnings,
            },
        )
    except Exception as exc:
        return _failure(f"intra_audio_voice_transfer failed: {type(exc).__name__}: {exc}")


def _ensure_diarization(state: dict[str, Any], input_audio: Path) -> tuple[dict[str, Any], str]:
    perception = _perception_from_state(state)
    if _normalized_speaker_segments(perception):
        return perception, "state.perception_report"

    client = SpeechPerceptionClient(
        diarization_model=_diarization_model(),
        cache_dir=os.getenv("X_EDITOR_PERCEPTION_CACHE_DIR") or (Path(os.getenv("TMPDIR", "/tmp")) / "x_editor" / "perception_cache"),
    )
    result = client.perceive(
        input_audio,
        use_asr=False,
        use_alignment=False,
        use_diarization=True,
        refresh_cache=False,
        strict_diarization=True,
    )
    return result, "runtime.pyannote"


def _perception_from_state(state: dict[str, Any]) -> dict[str, Any]:
    perception = state.get("perception_report") or state.get("audio_perception") or {}
    return perception if isinstance(perception, dict) else {}


def _normalized_speaker_segments(perception: dict[str, Any]) -> list[SpeakerSegment]:
    raw = perception.get("speaker_segments") or perception.get("segments") or []
    base_segments = _speaker_segments_from_items(raw)
    gemini = perception.get("gemini_perception")
    gemini_segments = (
        _normalized_gemini_segments(gemini.get("segments") or [])
        if isinstance(gemini, dict)
        else []
    )
    if len(_speaker_order(base_segments)) < 2 and len(_speaker_order(gemini_segments)) >= 2:
        return gemini_segments
    return base_segments


def _speaker_segments_from_items(items: list[Any]) -> list[SpeakerSegment]:
    segments: list[SpeakerSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if speaker_id is None or start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        segments.append(SpeakerSegment(str(speaker_id), start_f, end_f))
    return sorted(segments, key=lambda item: (item.start_sec, item.end_sec, item.speaker_id))


def _normalized_words(items: list[Any]) -> list[dict[str, Any]]:
    words = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("word") or item.get("token")
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if text is None or start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        words.append({"text": str(text), "start_sec": start_f, "end_sec": end_f})
    return sorted(words, key=lambda item: (item["start_sec"], item["end_sec"]))


def _clip_segments(segments: list[SpeakerSegment], duration_sec: float) -> list[SpeakerSegment]:
    clipped = []
    for seg in segments:
        start = max(0.0, min(seg.start_sec, duration_sec))
        end = max(0.0, min(seg.end_sec, duration_sec))
        if end > start:
            clipped.append(SpeakerSegment(seg.speaker_id, start, end))
    return clipped


def _snap_segments_to_word_boundaries(
    segments: list[SpeakerSegment],
    words: list[dict[str, Any]],
    duration_sec: float,
) -> list[SpeakerSegment]:
    if not words:
        return segments
    snapped = []
    for seg in segments:
        segment_words = [
            word
            for word in words
            if _overlap_seconds(seg.start_sec, seg.end_sec, word["start_sec"], word["end_sec"]) > 0
        ]
        if not segment_words:
            snapped.append(seg)
            continue
        start = max(0.0, min(float(segment_words[0]["start_sec"]), duration_sec))
        end = max(0.0, min(float(segment_words[-1]["end_sec"]), duration_sec))
        max_reasonable_expansion = max(0.75, seg.duration_sec * 0.5)
        expansion = (seg.start_sec - start) + (end - seg.end_sec)
        if expansion > max_reasonable_expansion:
            snapped.append(seg)
            continue
        if end > start:
            snapped.append(SpeakerSegment(seg.speaker_id, start, end))
    return snapped


def _merge_adjacent_segments(segments: list[SpeakerSegment], *, max_gap_sec: float) -> list[SpeakerSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker_id == prev.speaker_id and seg.start_sec - prev.end_sec <= max_gap_sec:
            merged[-1] = SpeakerSegment(prev.speaker_id, prev.start_sec, max(prev.end_sec, seg.end_sec))
        else:
            merged.append(seg)
    return merged


def _split_long_segments(
    segments: list[SpeakerSegment],
    words: list[dict[str, Any]],
    *,
    max_segment_sec: float,
    duration_sec: float,
) -> list[SpeakerSegment]:
    output: list[SpeakerSegment] = []
    for seg in segments:
        if seg.duration_sec < max_segment_sec:
            output.append(seg)
            continue
        start = seg.start_sec
        while start < seg.end_sec:
            limit = min(seg.end_sec, start + max_segment_sec)
            word_ends = [
                float(word["end_sec"])
                for word in words
                if start + 0.25 < float(word["end_sec"]) <= limit
            ]
            end = max(word_ends) if word_ends else limit
            end = min(max(end, start + 0.25), duration_sec, seg.end_sec)
            output.append(SpeakerSegment(seg.speaker_id, start, end))
            start = end
    return output


def _speaker_order(segments: list[SpeakerSegment]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for seg in sorted(segments, key=lambda item: item.start_sec):
        if seg.speaker_id in seen:
            continue
        seen.add(seg.speaker_id)
        order.append(seg.speaker_id)
    return order


def _speaker_profiles(perception: dict[str, Any]) -> dict[str, str]:
    profiles: dict[str, list[str]] = {}
    for collection_name in ["speakers", "segments", "speaker_segments"]:
        raw = perception.get(collection_name) or []
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict) or not item.get("speaker_id"):
                continue
            pieces = [
                item.get("speaker_id"),
                item.get("description"),
                item.get("perceived_gender"),
                item.get("transcript"),
            ]
            text = " ".join(str(piece) for piece in pieces if piece)
            if text:
                profiles.setdefault(str(item["speaker_id"]), []).append(text)
    _add_gemini_overlap_profiles(perception, profiles)
    return {speaker_id: " | ".join(values).lower() for speaker_id, values in profiles.items()}


def _add_gemini_overlap_profiles(perception: dict[str, Any], profiles: dict[str, list[str]]) -> None:
    gemini = perception.get("gemini_perception")
    if not isinstance(gemini, dict):
        return
    gemini_speaker_profiles: dict[str, str] = {}
    for item in gemini.get("speakers") or []:
        if not isinstance(item, dict) or not item.get("speaker_id"):
            continue
        pieces = [
            item.get("speaker_id"),
            item.get("description"),
            item.get("perceived_gender"),
            item.get("confidence"),
        ]
        text = " ".join(str(piece) for piece in pieces if piece)
        if text:
            gemini_speaker_profiles[str(item["speaker_id"])] = text
    if not gemini_speaker_profiles:
        return

    base_segments = _normalized_speaker_segments(perception)
    gemini_segments = _normalized_gemini_segments(gemini.get("segments") or [])
    if not base_segments or not gemini_segments:
        return

    overlap: dict[str, dict[str, float]] = {}
    for base in base_segments:
        for gemini_seg in gemini_segments:
            seconds = _overlap_seconds(base.start_sec, base.end_sec, gemini_seg.start_sec, gemini_seg.end_sec)
            if seconds <= 0:
                continue
            overlap.setdefault(base.speaker_id, {}).setdefault(gemini_seg.speaker_id, 0.0)
            overlap[base.speaker_id][gemini_seg.speaker_id] += seconds
    for speaker_id, scores in overlap.items():
        if not scores:
            continue
        gemini_id, seconds = max(scores.items(), key=lambda item: item[1])
        if seconds < 0.15:
            continue
        profile = gemini_speaker_profiles.get(gemini_id)
        if profile:
            profiles.setdefault(speaker_id, []).append(profile)


def _normalized_gemini_segments(items: list[Any]) -> list[SpeakerSegment]:
    segments: list[SpeakerSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if speaker_id is None or start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f > start_f:
            segments.append(SpeakerSegment(str(speaker_id), start_f, end_f))
    return sorted(segments, key=lambda item: (item.start_sec, item.end_sec, item.speaker_id))


def _resolve_mappings(
    instruction: str,
    speaker_order: list[str],
    segments: list[SpeakerSegment],
    profiles: dict[str, str],
) -> list[SpeakerMapping]:
    text = instruction.lower()
    explicit = _resolve_explicit_ordinal_mapping(text, speaker_order)
    if explicit:
        return explicit

    attr_mapping = _resolve_attribute_mapping(text, speaker_order, profiles)
    if attr_mapping:
        return attr_mapping

    if _contains(text, ["swap", "exchange", "互换", "交换", "彼此"]) and len(speaker_order) == 2:
        return [
            SpeakerMapping(speaker_order[0], speaker_order[1], "two-speaker swap"),
            SpeakerMapping(speaker_order[1], speaker_order[0], "two-speaker swap"),
        ]

    if _contains(text, ["all other speakers", "every other speaker", "所有其他说话人", "其他说话人"]):
        ref_index = _first_ordinal(text)
        if ref_index and 1 <= ref_index <= len(speaker_order):
            ref = speaker_order[ref_index - 1]
            return [
                SpeakerMapping(target, ref, "all other speakers to ordinal reference")
                for target in speaker_order
                if target != ref
            ]

    # A conservative fallback for terse two-speaker requests like
    # "make speaker 2 sound like speaker 1" when the regex sees only numbers.
    mentions = _ordinal_mentions(text)
    if len(mentions) >= 2 and len(speaker_order) >= max(mentions[0], mentions[1]):
        return [
            SpeakerMapping(
                speaker_order[mentions[0] - 1],
                speaker_order[mentions[1] - 1],
                "ordinal mention order",
            )
        ]

    return []


def _resolve_structured_mappings(
    params: dict[str, Any],
    speaker_order: list[str],
    profiles: dict[str, str] | None = None,
    speaker_segments: list[SpeakerSegment] | None = None,
) -> list[SpeakerMapping]:
    """Resolve planner-provided speaker ids before falling back to text parsing.

    Planner-facing terms use ``source_speaker_id(s)`` for the speaker(s) to
    convert and ``target_speaker_id`` for the in-audio reference voice. The
    internal dataclass keeps the historical name ``target_speaker_id`` for the
    converted speaker, so this adapter is intentionally explicit.
    """

    if not params or not speaker_order:
        return []

    pairs: list[tuple[Any, Any, str]] = []
    raw_mappings = (
        params.get("speaker_mappings")
        or params.get("speaker_map")
        or params.get("speaker_mapping")
        or params.get("mappings")
    )
    if isinstance(raw_mappings, list):
        for item in raw_mappings:
            if not isinstance(item, dict):
                continue
            reference = (
                item.get("reference_speaker_id")
                or item.get("target_speaker_id")
                or item.get("target_voice_speaker_id")
                or item.get("reference")
                or item.get("target")
                or item.get("reference_selector")
                or item.get("target_selector")
            )
            sources = (
                item.get("source_speaker_ids")
                or item.get("source_speaker_id")
                or item.get("source")
                or item.get("source_selector")
                or item.get("source_speaker")
            )
            for source in _as_list(sources):
                pairs.append((source, reference, "structured mapping list"))

    reference = (
        params.get("reference_speaker_id")
        or params.get("target_speaker_id")
        or params.get("target_voice_speaker_id")
        or params.get("reference_speaker")
        or params.get("target_speaker")
        or params.get("target_voice_speaker")
        or params.get("reference_speaker_selector")
        or params.get("target_speaker_selector")
        or params.get("target_voice_speaker_selector")
        or params.get("reference_selector")
        or params.get("target_selector")
    )
    sources = (
        params.get("source_speaker_ids")
        or params.get("source_speaker_id")
        or params.get("source_speakers")
        or params.get("source_speaker")
        or params.get("source_speaker_selector")
        or params.get("source_selector")
        or params.get("speaker_selector")
    )
    for source in _as_list(sources):
        pairs.append((source, reference, "structured speaker ids"))

    if not pairs and reference is not None:
        for source in _source_speakers_from_explicit_target_spans(params, speaker_segments or []):
            pairs.append((source, reference, "structured target span overlap"))

    if not pairs:
        return []

    numeric_values = [_numeric_speaker_value(value) for pair in pairs for value in pair[:2]]
    zero_based = any(value == 0 for value in numeric_values if value is not None)
    mappings: list[SpeakerMapping] = []
    seen: set[tuple[str, str]] = set()
    for source, reference, reason in pairs:
        source_id = _resolve_structured_speaker_id(source, speaker_order, zero_based=zero_based, profiles=profiles)
        reference_id = _resolve_structured_speaker_id(reference, speaker_order, zero_based=zero_based, profiles=profiles)
        if not source_id or not reference_id or source_id == reference_id:
            continue
        key = (source_id, reference_id)
        if key in seen:
            continue
        seen.add(key)
        mappings.append(SpeakerMapping(source_id, reference_id, reason))
    return mappings


def _source_speakers_from_explicit_target_spans(
    params: dict[str, Any],
    speaker_segments: list[SpeakerSegment],
) -> list[str]:
    if not speaker_segments:
        return []
    source_ids: list[str] = []
    for start_sec, end_sec in _target_spans_from_params(params):
        speaker_id = _speaker_for_span(start_sec, end_sec, speaker_segments)
        if speaker_id and speaker_id not in source_ids:
            source_ids.append(speaker_id)
    return source_ids


def _conversion_segments_from_explicit_target_spans(
    params: dict[str, Any],
    speaker_segments: list[SpeakerSegment],
    mappings: list[SpeakerMapping],
) -> list[SpeakerSegment]:
    if not speaker_segments or not mappings:
        return []
    allowed_targets = {mapping.target_speaker_id for mapping in mappings}
    segments: list[SpeakerSegment] = []
    for start_sec, end_sec in _target_spans_from_params(params):
        speaker_id = _speaker_for_span(start_sec, end_sec, speaker_segments)
        if not speaker_id and len(allowed_targets) == 1:
            speaker_id = next(iter(allowed_targets))
        if not speaker_id or speaker_id not in allowed_targets:
            continue
        if end_sec <= start_sec:
            continue
        segments.append(SpeakerSegment(speaker_id, start_sec, end_sec))
    return segments


def _target_spans_from_params(params: dict[str, Any]) -> list[tuple[float, float]]:
    raw_values: list[Any] = []
    for key in ("target_span", "target_spans", "source_span", "source_spans", "selected_span", "selected_spans"):
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            raw_values.extend(value)
        else:
            raw_values.append(value)

    spans: list[tuple[float, float]] = []
    for value in raw_values:
        span = _coerce_span(value)
        if span:
            spans.append(span)
    return spans


def _coerce_span(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        start = value.get("start_sec")
        end = value.get("end_sec")
        if start is None:
            start = value.get("start")
        if end is None:
            end = value.get("end")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = value[0], value[1]
    else:
        return None
    try:
        start_sec = float(start)
        end_sec = float(end)
    except (TypeError, ValueError):
        return None
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


def _speaker_for_span(
    start_sec: float,
    end_sec: float,
    speaker_segments: list[SpeakerSegment],
) -> str | None:
    best: tuple[float, float, str] | None = None
    midpoint = (start_sec + end_sec) / 2.0
    for index, segment in enumerate(speaker_segments):
        overlap = _overlap_seconds(start_sec, end_sec, segment.start_sec, segment.end_sec)
        midpoint_bonus = 0.001 if segment.start_sec <= midpoint <= segment.end_sec else 0.0
        score = overlap + midpoint_bonus
        if score <= 0:
            continue
        candidate = (score, -float(index), segment.speaker_id)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _resolve_structured_speaker_id(
    value: Any,
    speaker_order: list[str],
    *,
    zero_based: bool,
    profiles: dict[str, str] | None = None,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("speaker_id", "target_speaker_id", "source_speaker_id", "id"):
            if value.get(key) is not None:
                resolved = _resolve_structured_speaker_id(
                    value.get(key),
                    speaker_order,
                    zero_based=zero_based,
                    profiles=profiles,
                )
                if resolved:
                    return resolved
        gender = _explicit_gender_from_selector_mapping(value)
        if gender:
            matches = [
                speaker_id
                for speaker_id in speaker_order
                if _profile_has_explicit_gender((profiles or {}).get(speaker_id, ""), gender)
            ]
            if len(matches) == 1:
                return matches[0]
        selector = _profile_selector_from_mapping(value)
        if selector:
            return _resolve_profile_selector(selector, speaker_order, profiles or {})
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw in speaker_order:
        return raw
    lowered = raw.lower()
    by_lower = {speaker_id.lower(): speaker_id for speaker_id in speaker_order}
    if lowered in by_lower:
        return by_lower[lowered]

    numeric = _numeric_speaker_value(raw)
    if numeric is None:
        return _resolve_profile_selector(raw, speaker_order, profiles or {})
    candidates = [
        f"SPEAKER_{numeric:02d}",
        f"speaker_{numeric:02d}",
        f"speaker_{numeric}",
        str(numeric),
    ]
    for candidate in candidates:
        if candidate in speaker_order:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]

    index = numeric if zero_based else numeric - 1
    if 0 <= index < len(speaker_order):
        return speaker_order[index]
    return _resolve_profile_selector(raw, speaker_order, profiles or {})


def _profile_selector_from_mapping(value: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "perceived_gender",
        "gender",
        "description",
        "speaker_description",
        "role",
        "age",
        "selector",
    ):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            pieces.append(item.strip())
    if not pieces:
        return ""
    selector = " ".join(pieces)
    if any(piece.lower() in {"male", "man", "female", "woman"} for piece in pieces):
        selector = f"{selector} speaker"
    return selector


def _explicit_gender_from_selector_mapping(value: dict[str, Any]) -> str | None:
    for key in ("perceived_gender", "gender"):
        item = value.get(key)
        if not isinstance(item, str):
            continue
        lowered = item.lower()
        if lowered in {"female", "woman", "women", "girl", "女", "女性"}:
            return "female"
        if lowered in {"male", "man", "men", "boy", "男", "男性"}:
            return "male"
    return None


def _profile_has_explicit_gender(profile: str, gender: str) -> bool:
    text = profile.lower()
    if gender == "female":
        return bool(re.search(r"\b(female|woman|women|girl)\b", text)) or "女" in text
    if gender == "male":
        return bool(re.search(r"\b(male|man|men|boy)\b", text)) or "男" in text
    return False


def _resolve_profile_selector(selector: str, speaker_order: list[str], profiles: dict[str, str]) -> str | None:
    if not selector or not profiles:
        return None
    scored = []
    for speaker_id in speaker_order:
        score = _profile_selector_score(selector, profiles.get(speaker_id, ""))
        if score > 0:
            scored.append((score, speaker_id))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], speaker_order.index(item[1])))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _profile_selector_score(selector: str, profile: str) -> int:
    selector_text = selector.lower()
    profile_text = profile.lower()
    if not profile_text:
        return 0
    groups: list[tuple[list[str], list[str], int]] = []
    if _contains(selector_text, ["baby", "infant", "toddler", "child", "kid", "children", "boy", "girl", "小孩", "儿童", "婴儿"]):
        groups.append(
            (
                [
                    "baby",
                    "infant",
                    "toddler",
                    "child",
                    "kid",
                    "children",
                    "young child",
                    "little boy",
                    "little girl",
                    "boy",
                    "girl",
                    "high-pitched",
                    "childlike",
                    "小孩",
                    "儿童",
                    "婴儿",
                ],
                ["adult", "grown", "mother", "father", "middle-aged", "older"],
                3,
            )
        )
    if _contains(selector_text, ["adult", "grown", "mother", "father", "man", "woman", "成年人", "成人"]):
        groups.append(
            (
                ["adult", "grown", "mother", "father", "man", "woman", "middle-aged", "older", "成年人", "成人"],
                ["baby", "infant", "toddler", "child", "kid", "children", "little boy", "little girl", "high-pitched", "childlike"],
                3,
            )
        )
    if _contains(selector_text, ["male", "man", "boy", "男"]):
        groups.append((["male", "man", "boy", "男"], ["female", "woman", "girl", "女"], 2))
    if _contains(selector_text, ["female", "woman", "girl", "女"]):
        groups.append((["female", "woman", "girl", "女"], ["male", "man", "boy", "男"], 2))
    if not groups:
        return 0
    score = 0
    for positive_terms, negative_terms, weight in groups:
        score += weight * sum(1 for term in positive_terms if term in profile_text)
        score -= (weight + 1) * sum(1 for term in negative_terms if term in profile_text)
    return score


def _numeric_speaker_value(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.search(r"speaker[_\s-]?0*([0-9]+)\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _resolve_explicit_ordinal_mapping(text: str, speaker_order: list[str]) -> list[SpeakerMapping]:
    patterns = [
        r"(first|1st|one|second|2nd|two|third|3rd|three|fourth|4th|four)\s+speaker[^.。;；,，]{0,120}?(?:match|like|as|into|to)\s+the\s+(first|1st|one|second|2nd|two|third|3rd|three|fourth|4th|four)\s+speaker",
        r"(第一|第二|第三|第四|第1|第2|第3|第4|一号|二号|三号|四号)[^，。；;]{0,80}?(?:改成|变成|转换成|迁移到|匹配|像)(第一|第二|第三|第四|第1|第2|第3|第4|一号|二号|三号|四号)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        target = ORDINAL_WORDS.get(match.group(1), None)
        ref = ORDINAL_WORDS.get(match.group(2), None)
        if target and ref and 1 <= target <= len(speaker_order) and 1 <= ref <= len(speaker_order) and target != ref:
            return [
                SpeakerMapping(
                    speaker_order[target - 1],
                    speaker_order[ref - 1],
                    "explicit ordinal mapping",
                )
            ]
    return []


def _resolve_attribute_mapping(text: str, speaker_order: list[str], profiles: dict[str, str]) -> list[SpeakerMapping]:
    if not profiles:
        return []
    child_terms = ["child", "kid", "children", "baby", "infant", "toddler", "boy", "girl", "young boy", "young girl", "high-pitched", "儿童", "童声", "小孩", "婴儿"]
    adult_terms = ["adult", "grown", "middle-aged", "older", "mother", "father", "成年人", "成人"]
    swap_pairs = [
        ("child", child_terms, "adult", adult_terms),
        ("male", ["male", "man", "男"], "female", ["female", "woman", "女"]),
    ]
    for left_label, left_terms, right_label, right_terms in swap_pairs:
        mapping = _bidirectional_attribute_swap_mapping(
            text,
            left_label,
            left_terms,
            right_label,
            right_terms,
            speaker_order,
            profiles,
        )
        if mapping:
            return mapping

    directional_pairs = [
        ("child", child_terms, "adult", adult_terms),
        ("adult", adult_terms, "child", child_terms),
        ("male", ["male", "man", "男"], "female", ["female", "woman", "女"]),
        ("female", ["female", "woman", "女"], "male", ["male", "man", "男"]),
    ]
    for target_label, target_terms, ref_label, ref_terms in directional_pairs:
        if not _directional_attribute_request(text, target_terms, ref_terms):
            continue
        mapping = _attribute_mapping_for_terms(target_label, target_terms, ref_label, ref_terms, speaker_order, profiles)
        if mapping:
            return mapping

    attr_pairs = [
        (
            "adult",
            adult_terms,
            "child",
            child_terms,
        ),
        ("male", ["male", "man", "男"], "female", ["female", "woman", "女"]),
        ("female", ["female", "woman", "女"], "male", ["male", "man", "男"]),
    ]
    for target_label, target_terms, ref_label, ref_terms in attr_pairs:
        if not _contains(text, target_terms) or not _contains(text, ref_terms):
            continue
        mapping = _attribute_mapping_for_terms(target_label, target_terms, ref_label, ref_terms, speaker_order, profiles)
        if mapping:
            return mapping
    return []


def _bidirectional_attribute_swap_mapping(
    text: str,
    left_label: str,
    left_terms: list[str],
    right_label: str,
    right_terms: list[str],
    speaker_order: list[str],
    profiles: dict[str, str],
) -> list[SpeakerMapping]:
    if not _contains(text, ["swap", "exchange", "互换", "交换", "彼此"]):
        return []
    if not _contains(text, left_terms) or not _contains(text, right_terms):
        return []
    left = [
        speaker_id
        for speaker_id in speaker_order
        if _profile_matches_attribute(left_label, profiles.get(speaker_id, ""), left_terms)
        and not _profile_matches_attribute(right_label, profiles.get(speaker_id, ""), right_terms)
    ]
    right = [
        speaker_id
        for speaker_id in speaker_order
        if _profile_matches_attribute(right_label, profiles.get(speaker_id, ""), right_terms)
        and not _profile_matches_attribute(left_label, profiles.get(speaker_id, ""), left_terms)
    ]
    if len(left) != 1 or len(right) != 1:
        return []
    return [
        SpeakerMapping(left[0], right[0], f"bidirectional {left_label}/{right_label} profile swap"),
        SpeakerMapping(right[0], left[0], f"bidirectional {left_label}/{right_label} profile swap"),
    ]


def _directional_attribute_request(text: str, source_terms: list[str], reference_terms: list[str]) -> bool:
    source_pos = _first_term_position(text, source_terms)
    ref_pos = _first_term_position(text, reference_terms)
    if source_pos is None or ref_pos is None or source_pos >= ref_pos:
        return False
    between = text[source_pos : min(len(text), ref_pos + 40)]
    return _contains(
        between,
        ["sound like", "sounds like", "like", "match", "with", "to", "into", "as", "变成", "转换成", "像"],
    )


def _first_term_position(text: str, terms: list[str]) -> int | None:
    positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
    return min(positions) if positions else None


def _attribute_mapping_for_terms(
    target_label: str,
    target_terms: list[str],
    ref_label: str,
    ref_terms: list[str],
    speaker_order: list[str],
    profiles: dict[str, str],
) -> list[SpeakerMapping]:
    reference_candidates = [
        speaker_id
        for speaker_id in speaker_order
        if _profile_matches_attribute(ref_label, profiles.get(speaker_id, ""), ref_terms)
    ]
    if not reference_candidates:
        return []
    ref = reference_candidates[0]
    targets = [
        speaker_id
        for speaker_id in speaker_order
        if speaker_id != ref and _profile_matches_attribute(target_label, profiles.get(speaker_id, ""), target_terms)
    ]
    if not targets and target_label == "adult" and ref_label == "child":
        targets = [speaker_id for speaker_id in speaker_order if speaker_id != ref]
    if not targets:
        return []
    return [
        SpeakerMapping(target, ref, f"{target_label} speakers to {ref_label} speaker")
        for target in targets
    ]


def _profile_matches_attribute(label: str, profile: str, terms: list[str]) -> bool:
    text = profile.lower()
    if label == "male":
        return bool(re.search(r"\b(?:male|man|men|boy)\b", text) or "男" in text) and not bool(
            re.search(r"\b(?:female|woman|women|girl)\b", text) or "女" in text
        )
    if label == "female":
        return bool(re.search(r"\b(?:female|woman|women|girl)\b", text) or "女" in text) and not bool(
            re.search(r"\b(?:male|man|men|boy)\b", text) or "男" in text
        )
    if label == "child":
        return _contains(text, terms) and not _contains(text, ["adult", "grown", "middle-aged", "older", "mother", "father", "成年人", "成人"])
    if label == "adult":
        return _contains(text, terms) and not _contains(text, ["baby", "infant", "toddler", "childlike", "儿童", "童声", "小孩", "婴儿"])
    return _contains(text, terms)


def _ordinal_mentions(text: str) -> list[int]:
    mentions: list[tuple[int, int]] = []
    for token, value in ORDINAL_WORDS.items():
        for match in re.finditer(re.escape(token), text):
            mentions.append((match.start(), value))
    for match in re.finditer(r"\bspeaker[_\s-]?([0-9]+)\b", text):
        value = int(match.group(1))
        if value >= 1:
            mentions.append((match.start(), value))
    return [value for _, value in sorted(mentions)]


def _first_ordinal(text: str) -> int | None:
    mentions = _ordinal_mentions(text)
    return mentions[0] if mentions else None


def _write_reference_audio(original: AudioSegment, segments: list[SpeakerSegment], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pieces = AudioSegment.empty()
    silence = AudioSegment.silent(duration=120, frame_rate=original.frame_rate)
    max_ms = int(round(_max_reference_seconds() * 1000))
    used_segments = []
    for seg in sorted(segments, key=lambda item: item.duration_sec, reverse=True):
        if seg.duration_sec < 0.2:
            continue
        piece = _slice_audio(original, seg.start_sec, seg.end_sec)
        remaining = max_ms - len(pieces)
        if remaining <= 0:
            break
        if len(piece) > remaining:
            piece = piece[:remaining]
        pieces += piece + silence
        used_segments.append(_segment_dict(seg))
        if len(pieces) >= max_ms:
            break
    if len(pieces) == 0:
        pieces = AudioSegment.silent(duration=1, frame_rate=original.frame_rate)
    pieces.export(output_path, format="wav")
    return {
        "path": str(output_path),
        "duration_sec": len(pieces) / 1000.0,
        "segments": used_segments,
    }


def _replace_range(audio: AudioSegment, replacement: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))
    target_ms = max(0, end_ms - start_ms)
    return audio[:start_ms] + _fit_audio_duration(replacement, target_ms) + audio[end_ms:]


def _fit_audio_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    target_ms = max(0, target_ms)
    if len(audio) > target_ms:
        return audio[:target_ms]
    if len(audio) < target_ms:
        return audio + AudioSegment.silent(duration=target_ms - len(audio), frame_rate=audio.frame_rate)
    return audio


def _slice_audio(audio: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    return audio[int(round(start_sec * 1000)) : int(round(end_sec * 1000))]


def _segment_dict(seg: SpeakerSegment) -> dict[str, Any]:
    return {
        "speaker_id": seg.speaker_id,
        "start_sec": round(seg.start_sec, 6),
        "end_sec": round(seg.end_sec, 6),
        "duration_sec": round(seg.duration_sec, 6),
    }


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "speaker"


def _diarization_model() -> str:
    return os.getenv("X_EDITOR_DIARIZATION_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md-v2")


def _max_segment_seconds() -> float:
    return min(float(os.getenv("X_EDITOR_INTRA_VOICE_MAX_SEGMENT_SEC", "29.5")), 29.5)


def _min_target_seconds(instruction: str | None = None, mapping_params: dict[str, Any] | None = None) -> float:
    value = float(os.getenv("X_EDITOR_INTRA_VOICE_MIN_TARGET_SEC", "0.35"))
    evidence = " ".join(
        str(item)
        for item in [
            instruction,
            (mapping_params or {}).get("mapping_mode"),
            (mapping_params or {}).get("instruction"),
        ]
        if item
    ).lower()
    if _contains(
        evidence,
        [
            "gender_timbre_swap",
            "word",
            "words",
            "number",
            "numbers",
            "count",
            "counting",
            "content",
            "men and women",
            "male and female",
        ],
    ):
        return min(value, 0.2)
    return value


def _min_reference_seconds() -> float:
    return float(os.getenv("X_EDITOR_INTRA_VOICE_MIN_REFERENCE_SEC", "0.8"))


def _max_reference_seconds() -> float:
    return float(os.getenv("X_EDITOR_INTRA_VOICE_MAX_REFERENCE_SEC", "12.0"))


def _failure(message: str) -> ToolResult:
    return ToolResult(
        tool_name="intra_audio_voice_transfer.execute",
        success=False,
        error_message=message,
    )
