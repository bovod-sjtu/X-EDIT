"""Segmented execution for long edit-data-pipeline calls.

Long service-layer edits are split on word boundaries before invoking the
edit-data-pipeline service. When diarization is available, segment boundaries
also respect speaker turns so each edited chunk contains one speaker only.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from html import escape, unescape
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from pydub.silence import detect_silence

from x_editor.perception.speech import SpeechPerceptionClient, SpeechPerceptionError
from x_editor.planner.structured_edit import build_edit_data_pipeline_arguments
from x_editor.tools.mcp.schemas import ToolResult
from x_editor.tools.edit_backend import uses_dots_model

SEGMENTABLE_OPERATIONS = {
    "text_edit",
    "pitch_rate_edit",
    "emotion_style_edit",
    "volume_normalize",
    "speech_rate",
}


@dataclass
class EditChunk:
    index: int
    start_sec: float
    end_sec: float
    text: str
    speaker_id: str | None
    words: list[dict[str, Any]]
    reason: str
    editable: bool = True

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class TextEditPhraseRequest:
    kind: str
    source: str
    target: str


async def maybe_execute_segmented_edit_pipeline(
    *,
    tool_executor: Any,
    arguments: dict[str, Any],
    state: dict[str, Any],
    output_dir: str,
    force_local_text_edit: bool = False,
) -> ToolResult | None:
    """Execute a long edit-data-pipeline call in word-boundary chunks.

    Returns ``None`` when the call should use the normal one-shot execution
    path. Returns a ``ToolResult`` for segmented execution or for a segmentation
    precondition failure.
    """

    operation = str(arguments.get("operation") or "").strip()
    if uses_dots_model(operation):
        return None
    if operation not in SEGMENTABLE_OPERATIONS:
        return None

    input_audio = arguments.get("input_audio_path")
    output_path = arguments.get("output_path")
    if not isinstance(input_audio, str) or not isinstance(output_path, str):
        return None

    audio = AudioSegment.from_file(input_audio)
    duration_sec = len(audio) / 1000.0
    max_segment_sec = _max_segment_seconds()
    target_limited_segment_sec = _text_edit_target_segment_seconds(arguments, duration_sec=duration_sec)
    target_limit_trigger = target_limited_segment_sec is not None
    preemptive_target_trigger = _should_preemptively_segment_text_edit(arguments, duration_sec=duration_sec)
    phrase_request = _single_text_edit_phrase_request(arguments)
    if phrase_request is not None and (
        force_local_text_edit
        or _should_execute_phrase_local_text_edit(
            arguments,
            duration_sec=duration_sec,
            phrase_request=phrase_request,
            target_limit_trigger=target_limit_trigger,
        )
    ):
        if force_local_text_edit or _phrase_local_alignment_ready(
            state=state,
            input_audio=input_audio,
            duration_sec=duration_sec,
            phrase_request=phrase_request,
        ):
            return await _execute_phrase_local_text_edit(
                tool_executor=tool_executor,
                arguments=arguments,
                state=state,
                output_dir=output_dir,
                audio=audio,
                duration_sec=duration_sec,
                phrase_request=phrase_request,
                forced=force_local_text_edit,
            )
    if force_local_text_edit and operation == "text_edit":
        return _failure(
            "text_edit local repair requires a single <sub> or <del> instruction_st with an "
            "alignable source phrase."
        )
    if target_limited_segment_sec is not None:
        max_segment_sec = min(max_segment_sec, target_limited_segment_sec)
    if preemptive_target_trigger:
        preemptive_segment_sec = _preemptive_text_edit_segment_seconds()
        if preemptive_segment_sec > 0.0:
            max_segment_sec = min(max_segment_sec, preemptive_segment_sec)
    if duration_sec < min(30.0, max_segment_sec) and not target_limit_trigger and not preemptive_target_trigger:
        return None

    perception = _perception_from_state(state)
    word_timestamps = _normalized_words(perception.get("word_timestamps") or [])
    alignment_refresh_reason: str | None = None
    if _alignment_refresh_required_for_input(
        word_timestamps,
        perception=perception,
        input_audio_path=input_audio,
        input_duration_sec=duration_sec,
    ):
        refreshed = _refresh_alignment_for_segmentation(
            input_audio_path=input_audio,
            source_text=str(arguments.get("source_text") or state.get("source_text") or ""),
            language=arguments.get("language") or perception.get("language"),
            output_dir=output_dir,
            input_duration_sec=duration_sec,
        )
        if isinstance(refreshed, ToolResult):
            return refreshed
        perception = {**perception, **refreshed}
        if "speaker_segments" not in refreshed and "segments" not in refreshed:
            perception.pop("speaker_segments", None)
            perception.pop("segments", None)
        word_timestamps = _normalized_words(refreshed.get("word_timestamps") or [])
        alignment_refresh_reason = "complex_inherited_alignment_requires_realign"
    else:
        word_timestamps = _inherit_or_scale_words_for_input(
            word_timestamps,
            perception=perception,
            input_audio_path=input_audio,
            input_duration_sec=duration_sec,
        )
    if not word_timestamps:
        if target_limit_trigger or preemptive_target_trigger:
            refreshed = _refresh_alignment_for_segmentation(
                input_audio_path=input_audio,
                source_text=str(arguments.get("source_text") or state.get("source_text") or ""),
                language=arguments.get("language") or perception.get("language"),
                output_dir=output_dir,
                input_duration_sec=duration_sec,
            )
            if isinstance(refreshed, ToolResult):
                return refreshed
            word_timestamps = _normalized_words(refreshed.get("word_timestamps") or [])
            perception = {**perception, **refreshed}
            word_timestamps = _inherit_or_scale_words_for_input(
                word_timestamps,
                perception=perception,
                input_audio_path=input_audio,
                input_duration_sec=duration_sec,
            )
        if not word_timestamps:
            if target_limit_trigger or preemptive_target_trigger:
                return _failure(
                    "requires_alignment_for_segmentation: edit_data_pipeline text_edit may exceed the F5 target "
                    "timeline limit, but no word_timestamps are available for safe segmentation."
                )
            return _failure(
                "edit_data_pipeline segmented execution requires word_timestamps for audio >= 30s; "
                "run perception with forced alignment enabled."
            )

    speaker_segments = _normalized_speaker_segments(perception)
    speaker_segments = _inherit_or_scale_speaker_segments_for_input(
        speaker_segments,
        perception=perception,
        input_audio_path=input_audio,
        input_duration_sec=duration_sec,
    )
    prompt = str(arguments.get("instruction") or state.get("user_prompt") or "")
    require_speaker = _require_speaker_segments(operation=operation, prompt=prompt, perception=perception)
    if require_speaker and not speaker_segments:
        return _failure(
            "edit_data_pipeline segmented execution requires speaker_segments for audio >= 30s; "
            "run perception with diarization enabled so each chunk can be single-speaker."
        )
    silence_spans = _detect_silence_spans(audio)
    pause_spans = _word_gap_spans(word_timestamps, duration_sec=duration_sec)

    chunks = _build_chunks(
        word_timestamps,
        speaker_segments,
        duration_sec=duration_sec,
        max_segment_sec=max_segment_sec,
    )
    problem = _validate_chunks(chunks, max_segment_sec=max_segment_sec, require_speaker=require_speaker)
    if problem:
        return _failure(problem)

    work_dir = Path(output_dir) / "segmented_edit" / Path(output_path).stem
    work_dir.mkdir(parents=True, exist_ok=True)
    edited_audio = AudioSegment.empty()
    segment_results: list[dict[str, Any]] = []
    segment_output_durations_sec: list[float] = []

    for chunk in chunks:
        chunk_input = work_dir / f"segment_{chunk.index:03d}_input.wav"
        chunk_output = work_dir / f"segment_{chunk.index:03d}_output.wav"
        _slice_audio(audio, chunk.start_sec, chunk.end_sec).export(chunk_input, format="wav")

        if not chunk.editable or not chunk.text:
            shutil.copy2(chunk_input, chunk_output)
            segment_results.append(_segment_record(chunk, chunk_input, chunk_output, copied=True))
            output_audio = AudioSegment.from_file(chunk_output)
            segment_output_durations_sec.append(len(output_audio) / 1000.0)
            edited_audio += output_audio
            continue

        segment_args = _tagged_text_edit_segment_arguments(
            arguments,
            chunk=chunk,
            input_audio_path=str(chunk_input),
            output_path=str(chunk_output),
            output_dir=str(work_dir / f"segment_{chunk.index:03d}_service_artifacts"),
        )
        if segment_args is None:
            if operation == "text_edit" and not _text_edit_applies(prompt, chunk.text, chunk.index):
                shutil.copy2(chunk_input, chunk_output)
                segment_results.append(_segment_record(chunk, chunk_input, chunk_output, copied=True))
                output_audio = AudioSegment.from_file(chunk_output)
                segment_output_durations_sec.append(len(output_audio) / 1000.0)
                edited_audio += output_audio
                continue

            segment_args = build_edit_data_pipeline_arguments(
                operation,
                prompt,
                chunk.text,
                input_audio_path=str(chunk_input),
                output_path=str(chunk_output),
                output_dir=str(work_dir / f"segment_{chunk.index:03d}_service_artifacts"),
            )
        if segment_args is None:
            shutil.copy2(chunk_input, chunk_output)
            segment_results.append(_segment_record(chunk, chunk_input, chunk_output, copied=True))
            output_audio = AudioSegment.from_file(chunk_output)
            segment_output_durations_sec.append(len(output_audio) / 1000.0)
            edited_audio += output_audio
            continue
        _copy_segment_overrides(arguments, segment_args)

        result = await tool_executor.execute("edit_data_pipeline", segment_args)
        record = _segment_record(chunk, chunk_input, chunk_output, copied=False, result=result.output)
        if not result.success:
            return ToolResult(
                tool_name="edit_data_pipeline.apply",
                success=False,
                error_message=f"segmented edit failed at segment {chunk.index}: {result.error_message}",
                output={
                    "segmented": True,
                    "failed_segment": record,
                    "segments": segment_results,
                },
            )
        segment_results.append(record)
        output_audio = AudioSegment.from_file(chunk_output)
        segment_output_durations_sec.append(len(output_audio) / 1000.0)
        edited_audio += output_audio

    boundary_smoothing_records: list[dict[str, Any]] = []
    if operation == "text_edit" and len(chunks) > 1:
        active_boundaries: list[dict[str, Any]] = []
        draft_time_sec = 0.0
        for index, chunk in enumerate(chunks[:-1]):
            draft_time_sec += segment_output_durations_sec[index]
            original_boundary_sec = chunk.end_sec
            if _is_direct_concat_boundary(
                original_boundary_sec,
                duration_sec=duration_sec,
                silence_spans=silence_spans,
                pause_spans=pause_spans,
            ):
                continue
            active_boundaries.append(
                {
                    "draft_time_sec": draft_time_sec,
                    "original_time_sec": original_boundary_sec,
                    "reason": "non_silence_active_split",
                }
            )
        if active_boundaries:
            edited_audio, boundary_smoothing_records = await _smooth_active_boundaries(
                tool_executor=tool_executor,
                draft_audio=edited_audio,
                boundaries=active_boundaries,
                work_dir=work_dir,
                base_output_stem="segmented_concat",
                text=_render_words(word_timestamps),
            )

    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    edited_audio.export(final_output, format="wav")

    manifest_path = work_dir / "segment_manifest.json"
    manifest = {
        "operation": operation,
        "input_audio_path": input_audio,
        "output_audio_path": str(final_output),
        "duration_seconds": duration_sec,
        "segmentation_trigger": (
            "target_f5_limit"
            if target_limit_trigger
            else "preemptive_text_edit_target_limit"
            if preemptive_target_trigger
            else "input_duration"
        ),
        "max_segment_seconds": max_segment_sec,
        "configured_max_segment_seconds": _max_segment_seconds(),
        "segments": segment_results,
        "speaker_segments_required": require_speaker,
        "boundary_mode": "active_split_boundary_smooth" if boundary_smoothing_records else "direct_concat",
        "boundary_smoothing": boundary_smoothing_records,
    }
    if alignment_refresh_reason:
        manifest["alignment_refresh_reason"] = alignment_refresh_reason
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output = {
        "output_audio_path": str(final_output),
        "operation": operation,
        "segmented": True,
        "segmentation_trigger": (
            "target_f5_limit"
            if target_limit_trigger
            else "preemptive_text_edit_target_limit"
            if preemptive_target_trigger
            else "input_duration"
        ),
        "segment_count": len(chunks),
        "max_segment_seconds": max_segment_sec,
        "configured_max_segment_seconds": _max_segment_seconds(),
        "segment_manifest_path": str(manifest_path),
        "segments": segment_results,
        "boundary_mode": manifest["boundary_mode"],
        "boundary_smoothing": boundary_smoothing_records,
    }
    if alignment_refresh_reason:
        output["alignment_refresh_reason"] = alignment_refresh_reason
    return ToolResult(tool_name="edit_data_pipeline.apply", success=True, output=output)


async def _execute_phrase_local_text_edit(
    *,
    tool_executor: Any,
    arguments: dict[str, Any],
    state: dict[str, Any],
    output_dir: str,
    audio: AudioSegment,
    duration_sec: float,
    phrase_request: TextEditPhraseRequest,
    forced: bool,
) -> ToolResult:
    perception = _perception_from_state(state)
    word_timestamps = _normalized_words(perception.get("word_timestamps") or [])
    if not word_timestamps:
        return _failure(
            "text_edit local repair requires word_timestamps or full aligned transcript; "
            "refusing to send a short phrase source_text with full audio to F5."
        )

    match_span = _find_phrase_word_span(word_timestamps, phrase_request.source)
    if match_span is None:
        return _failure(
            "text_edit local repair could not locate the source phrase in word_timestamps: "
            f"{phrase_request.source!r}"
        )

    silence_spans = _detect_silence_spans(audio)
    pause_spans = _word_gap_spans(word_timestamps, duration_sec=duration_sec)
    window = _build_phrase_edit_window(
        word_timestamps,
        match_span=match_span,
        duration_sec=duration_sec,
        silence_spans=silence_spans,
        pause_spans=pause_spans,
    )
    if window["end_sec"] - window["start_sec"] >= 30.0:
        return _failure(
            "text_edit local repair selected an edit window >= 30s; refusing to call F5 with an unsafe segment."
        )

    output_path = str(arguments.get("output_path") or "")
    if not output_path:
        return _failure("text_edit local repair requires output_path")
    work_dir = Path(output_dir) / "segmented_edit" / Path(output_path).stem / "phrase_local"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_input = work_dir / "local_input.wav"
    local_output = work_dir / "local_output.wav"
    _slice_audio(audio, window["start_sec"], window["end_sec"]).export(local_input, format="wav")

    window_words = _words_overlapping_window(word_timestamps, window["start_sec"], window["end_sec"])
    window_text = _render_words(window_words)
    if not window_text:
        return _failure("text_edit local repair selected a window with no aligned transcript words")
    instruction_st = str(arguments.get("instruction_st") or "")
    window_instruction = _project_text_edit_tags_to_segment(instruction_st, window_text)
    aligned_source_phrase = phrase_request.source
    if window_instruction is None:
        aligned_source_phrase = _render_words(
            word_timestamps[match_span[0] : match_span[1]]
        )
        aligned_request = TextEditPhraseRequest(
            kind=phrase_request.kind,
            source=aligned_source_phrase or phrase_request.source,
            target=phrase_request.target,
        )
        window_instruction = _project_single_text_edit_phrase_to_segment(
            aligned_request,
            window_text,
        )
    if window_instruction is None:
        return _failure(
            "text_edit local repair could not project instruction_st onto the local transcript window"
        )

    local_args = dict(arguments)
    local_args.update(
        {
            "operation": "text_edit",
            "input_audio_path": str(local_input),
            "output_path": str(local_output),
            "output_dir": str(work_dir / "service_artifacts"),
            "source_text": window_text,
            "instruction_st": window_instruction,
            "target_text": _render_text_edit_target(window_instruction),
        }
    )
    local_args.setdefault("synthesis_mode", "per_op")
    local_args.setdefault("max_f5_edit_seconds", 32.0)

    result = await tool_executor.execute("edit_data_pipeline", local_args)
    segment_record = {
        "index": 0,
        "start_sec": round(window["start_sec"], 6),
        "end_sec": round(window["end_sec"], 6),
        "duration_sec": round(window["end_sec"] - window["start_sec"], 6),
        "text": window_text,
        "reason": "phrase_local_text_edit",
        "editable": True,
        "copied": False,
        "input_audio_path": str(local_input),
        "output_audio_path": str(local_output),
        "tool_output": result.output or {},
    }
    if not result.success:
        return ToolResult(
            tool_name="edit_data_pipeline.apply",
            success=False,
            error_message=f"local phrase text_edit failed: {result.error_message}",
            output={
                "segmented": True,
                "local_text_edit": True,
                "failed_segment": segment_record,
                "segments": [segment_record],
            },
        )
    if not local_output.exists():
        return _failure("local phrase text_edit succeeded without producing output audio")

    edited_window = AudioSegment.from_file(local_output)
    prefix = _slice_audio(audio, 0.0, window["start_sec"])
    suffix = _slice_audio(audio, window["end_sec"], duration_sec)
    draft_audio = prefix + edited_window + suffix

    boundary_smoothing_records: list[dict[str, Any]] = []
    active_boundaries: list[dict[str, Any]] = []
    if not _is_direct_concat_boundary(
        window["start_sec"],
        duration_sec=duration_sec,
        silence_spans=silence_spans,
        pause_spans=pause_spans,
    ):
        active_boundaries.append(
            {
                "draft_time_sec": window["start_sec"],
                "original_time_sec": window["start_sec"],
                "reason": "phrase_local_window_start",
            }
        )
    edited_window_duration_sec = len(edited_window) / 1000.0
    draft_end_boundary = window["start_sec"] + edited_window_duration_sec
    if not _is_direct_concat_boundary(
        window["end_sec"],
        duration_sec=duration_sec,
        silence_spans=silence_spans,
        pause_spans=pause_spans,
    ):
        active_boundaries.append(
            {
                "draft_time_sec": draft_end_boundary,
                "original_time_sec": window["end_sec"],
                "reason": "phrase_local_window_end",
            }
        )
    if active_boundaries:
        draft_audio, boundary_smoothing_records = await _smooth_active_boundaries(
            tool_executor=tool_executor,
            draft_audio=draft_audio,
            boundaries=active_boundaries,
            work_dir=work_dir,
            base_output_stem="phrase_local_concat",
            text=window_text,
        )

    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    draft_audio.export(final_output, format="wav")

    boundary_mode = "active_split_boundary_smooth" if boundary_smoothing_records else "silence_direct_concat"
    manifest_path = work_dir / "segment_manifest.json"
    manifest = {
        "operation": "text_edit",
        "input_audio_path": str(arguments.get("input_audio_path") or ""),
        "output_audio_path": str(final_output),
        "duration_seconds": duration_sec,
        "segmentation_trigger": "runtime_f5_limit_repair" if forced else "phrase_local_text_edit",
        "local_text_edit": True,
        "source_phrase": phrase_request.source,
        "aligned_source_phrase": aligned_source_phrase,
        "target_phrase": phrase_request.target,
        "boundary_mode": boundary_mode,
        "window": window,
        "segments": [segment_record],
        "boundary_smoothing": boundary_smoothing_records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ToolResult(
        tool_name="edit_data_pipeline.apply",
        success=True,
        output={
            "output_audio_path": str(final_output),
            "operation": "text_edit",
            "segmented": True,
            "local_text_edit": True,
            "segmentation_trigger": manifest["segmentation_trigger"],
            "segment_count": 1,
            "segment_manifest_path": str(manifest_path),
            "segments": [segment_record],
            "boundary_mode": boundary_mode,
            "boundary_smoothing": boundary_smoothing_records,
        },
    )


def _single_text_edit_phrase_request(arguments: dict[str, Any]) -> TextEditPhraseRequest | None:
    if str(arguments.get("operation") or "") != "text_edit":
        return None
    instruction_st = arguments.get("instruction_st")
    if not isinstance(instruction_st, str):
        return None
    matches = list(_TEXT_EDIT_TAG_RE.finditer(instruction_st))
    if len(matches) != 1:
        return None
    match = matches[0]
    if match.group("sub_attrs") is not None:
        target = _sub_target(match.group("sub_attrs") or "")
        source = unescape(match.group("sub_body") or "")
        if not target or not source.strip():
            return None
        return TextEditPhraseRequest(kind="sub", source=source.strip(), target=target.strip())
    if match.group("del_body") is not None:
        source = unescape(match.group("del_body") or "")
        if not source.strip():
            return None
        return TextEditPhraseRequest(kind="del", source=source.strip(), target="")
    return None


def _should_execute_phrase_local_text_edit(
    arguments: dict[str, Any],
    *,
    duration_sec: float,
    phrase_request: TextEditPhraseRequest,
    target_limit_trigger: bool,
) -> bool:
    source_text = str(arguments.get("source_text") or "").strip()
    source_matches_phrase = bool(source_text) and _match_text_key(source_text) == _match_text_key(phrase_request.source)
    if source_matches_phrase:
        # This is a whole-utterance substitution, not a local phrase repair.
        # The one-shot service already has the exact transcript and does not
        # need an additional ASR surface-form projection.
        return False
    if target_limit_trigger:
        return True
    if duration_sec <= 10.0:
        return False
    phrase_units = _duration_units(phrase_request.source)
    instruction_source = _render_text_edit_source(str(arguments.get("instruction_st") or "")).strip()
    instruction_is_phrase_only = _match_text_key(instruction_source) == _match_text_key(phrase_request.source)
    instruction_has_single_local_phrase = _contains_match_units(instruction_source, phrase_request.source)
    return phrase_units <= 8.0 and (instruction_is_phrase_only or instruction_has_single_local_phrase)


def _phrase_local_alignment_ready(
    *,
    state: dict[str, Any],
    input_audio: str,
    duration_sec: float,
    phrase_request: TextEditPhraseRequest,
) -> bool:
    perception = _perception_from_state(state)
    word_timestamps = _normalized_words(perception.get("word_timestamps") or [])
    if not word_timestamps:
        return False
    if _alignment_refresh_required_for_input(
        word_timestamps,
        perception=perception,
        input_audio_path=input_audio,
        input_duration_sec=duration_sec,
    ):
        return False
    return _find_phrase_word_span(word_timestamps, phrase_request.source) is not None


def _find_phrase_word_span(words: list[dict[str, Any]], phrase: str) -> tuple[int, int] | None:
    phrase_units = _match_units(phrase)
    if not phrase_units:
        return None
    indexed_units: list[tuple[int, str]] = []
    for index, word in enumerate(words):
        units = _match_units(str(word.get("text") or ""))
        if not units:
            continue
        for unit in units:
            indexed_units.append((index, unit))
    if not indexed_units:
        return None
    needle_len = len(phrase_units)
    for offset in range(0, len(indexed_units) - needle_len + 1):
        if [unit for _, unit in indexed_units[offset : offset + needle_len]] != phrase_units:
            continue
        start_index = indexed_units[offset][0]
        end_index = indexed_units[offset + needle_len - 1][0] + 1
        return start_index, end_index
    return None


def _match_units(text: str) -> list[str]:
    text = unescape(str(text or "")).lower()
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        units = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[a-z0-9]+", text)
    else:
        units = re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)?", text)
    return [re.sub(r"['’]", "", unit) for unit in units if re.sub(r"['’]", "", unit)]


def _match_text_key(text: str) -> str:
    return " ".join(_match_units(text))


def _contains_match_units(text: str, phrase: str) -> bool:
    haystack = _match_units(text)
    needle = _match_units(phrase)
    if not haystack or not needle or len(needle) > len(haystack):
        return False
    return any(haystack[offset : offset + len(needle)] == needle for offset in range(0, len(haystack) - len(needle) + 1))


def _build_phrase_edit_window(
    words: list[dict[str, Any]],
    *,
    match_span: tuple[int, int],
    duration_sec: float,
    silence_spans: list[tuple[float, float]],
    pause_spans: list[tuple[float, float]],
) -> dict[str, Any]:
    match_start, match_end = match_span
    span_start = max(0.0, float(words[match_start]["start_sec"]))
    span_end = min(duration_sec, float(words[match_end - 1]["end_sec"]))
    max_window_sec = min(28.5, _max_segment_seconds())
    context_sec = min(6.0, max(1.0, (max_window_sec - max(0.1, span_end - span_start)) / 2.0))
    desired_start = max(0.0, span_start - context_sec)
    desired_end = min(duration_sec, span_end + context_sec)
    if desired_end - desired_start > max_window_sec:
        center = (span_start + span_end) / 2.0
        desired_start = max(0.0, center - max_window_sec / 2.0)
        desired_end = min(duration_sec, desired_start + max_window_sec)
        desired_start = max(0.0, desired_end - max_window_sec)

    pause_points = _pause_boundary_points([*silence_spans, *pause_spans])
    low_start = max(0.0, span_end - max_window_sec + 0.05)
    high_end = min(duration_sec, span_start + max_window_sec - 0.05)
    start_sec, start_mode = _select_pause_or_active_boundary(
        desired=desired_start,
        low=low_start,
        high=max(low_start, span_start - 0.01),
        pause_points=pause_points,
        fallback=desired_start,
        edge=0.0,
    )
    end_sec, end_mode = _select_pause_or_active_boundary(
        desired=desired_end,
        low=min(high_end, span_end + 0.01),
        high=high_end,
        pause_points=pause_points,
        fallback=desired_end,
        edge=duration_sec,
    )
    if end_sec <= span_end:
        end_sec = min(duration_sec, max(span_end + 0.05, desired_end))
        end_mode = "active"
    if start_sec >= span_start:
        start_sec = max(0.0, min(span_start - 0.05, desired_start))
        start_mode = "active"
    if end_sec - start_sec >= 30.0:
        start_sec = max(0.0, min(start_sec, span_start - 0.05))
        end_sec = min(duration_sec, start_sec + max_window_sec)
        if end_sec <= span_end:
            end_sec = min(duration_sec, span_end + 0.05)
            start_sec = max(0.0, end_sec - max_window_sec)
    return {
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "phrase_start_sec": round(span_start, 6),
        "phrase_end_sec": round(span_end, 6),
        "start_boundary": start_mode,
        "end_boundary": end_mode,
    }


def _select_pause_or_active_boundary(
    *,
    desired: float,
    low: float,
    high: float,
    pause_points: list[float],
    fallback: float,
    edge: float,
) -> tuple[float, str]:
    if abs(fallback - edge) <= 0.001:
        return edge, "edge"
    candidates = [point for point in pause_points if low <= point <= high]
    if candidates:
        return min(candidates, key=lambda point: abs(point - desired)), "pause"
    return min(max(fallback, low), high), "active"


def _pause_boundary_points(spans: list[tuple[float, float]]) -> list[float]:
    points = []
    for start, end in spans:
        if end <= start:
            continue
        points.append((start + end) / 2.0)
    return sorted(set(round(point, 6) for point in points))


def _words_overlapping_window(
    words: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    selected = []
    for word in words:
        if float(word["end_sec"]) <= start_sec or float(word["start_sec"]) >= end_sec:
            continue
        selected.append(word)
    return selected


def _project_single_text_edit_phrase_to_segment(
    phrase_request: TextEditPhraseRequest,
    segment_text: str,
) -> str | None:
    source = phrase_request.source
    match = re.search(re.escape(source), segment_text, flags=re.IGNORECASE)
    if not match:
        return None
    actual_source = match.group(0)
    if phrase_request.kind == "sub":
        tag = f'<sub targ="{escape(phrase_request.target, quote=True)}">{escape(actual_source, quote=False)}</sub>'
    elif phrase_request.kind == "del":
        tag = f"<del>{escape(actual_source, quote=False)}</del>"
    else:
        return None
    return (
        escape(segment_text[: match.start()], quote=False)
        + tag
        + escape(segment_text[match.end() :], quote=False)
    )


def _detect_silence_spans(audio: AudioSegment) -> list[tuple[float, float]]:
    if len(audio) <= 0:
        return []
    dbfs = audio.dBFS
    silence_thresh = -50.0 if not math.isfinite(dbfs) else min(-35.0, dbfs - 16.0)
    spans_ms = detect_silence(audio, min_silence_len=160, silence_thresh=silence_thresh)
    return [(start / 1000.0, end / 1000.0) for start, end in spans_ms if end > start]


def _word_gap_spans(words: list[dict[str, Any]], *, duration_sec: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    if not words:
        return spans
    first_start = max(0.0, float(words[0]["start_sec"]))
    if first_start >= 0.18:
        spans.append((0.0, first_start))
    for previous, current in zip(words, words[1:]):
        start = float(previous["end_sec"])
        end = float(current["start_sec"])
        if end - start >= 0.18:
            spans.append((start, end))
    last_end = min(duration_sec, float(words[-1]["end_sec"]))
    if duration_sec - last_end >= 0.18:
        spans.append((last_end, duration_sec))
    return spans


def _is_direct_concat_boundary(
    boundary_sec: float,
    *,
    duration_sec: float,
    silence_spans: list[tuple[float, float]],
    pause_spans: list[tuple[float, float]],
) -> bool:
    if boundary_sec <= 0.001 or boundary_sec >= duration_sec - 0.001:
        return True
    return _time_in_spans(boundary_sec, silence_spans, tolerance_sec=0.04) or _time_in_spans(
        boundary_sec,
        pause_spans,
        tolerance_sec=0.04,
    )


def _time_in_spans(
    time_sec: float,
    spans: list[tuple[float, float]],
    *,
    tolerance_sec: float,
) -> bool:
    for start, end in spans:
        if start - tolerance_sec <= time_sec <= end + tolerance_sec:
            return True
    return False


async def _smooth_active_boundaries(
    *,
    tool_executor: Any,
    draft_audio: AudioSegment,
    boundaries: list[dict[str, Any]],
    work_dir: Path,
    base_output_stem: str,
    text: str,
) -> tuple[AudioSegment, list[dict[str, Any]]]:
    if not boundaries:
        return draft_audio, []
    records: list[dict[str, Any]] = []
    current_audio = draft_audio
    for index, boundary in enumerate(sorted(boundaries, key=lambda item: float(item["draft_time_sec"]), reverse=True)):
        duration_sec = current_audio.frame_count() / current_audio.frame_rate
        boundary_sec = max(0.0, min(duration_sec, float(boundary["draft_time_sec"])))
        window_start = max(0.0, boundary_sec - 10.0)
        window_end = min(duration_sec, boundary_sec + 10.0)
        if window_end - window_start > 28.5:
            overflow = (window_end - window_start) - 28.5
            window_start += overflow / 2.0
            window_end -= overflow / 2.0
        if window_end <= window_start or window_end - window_start >= 30.0:
            records.append(
                {
                    **boundary,
                    "status": "skipped",
                    "reason": "smoothing_window_invalid_or_exceeds_f5_limit",
                    "window_start_sec": round(window_start, 6),
                    "window_end_sec": round(window_end, 6),
                }
            )
            continue
        local_boundary_sec = boundary_sec - window_start
        window_input = work_dir / f"{base_output_stem}_smooth_{index:03d}_input.wav"
        window_output = work_dir / f"{base_output_stem}_smooth_{index:03d}_output.wav"
        # Use the same frame coordinates for extraction and replacement. Pydub's
        # millisecond length can round away a short, unedited audio tail.
        start_frame = round(window_start * current_audio.frame_rate)
        end_frame = round(window_end * current_audio.frame_rate)
        current_audio.get_sample_slice(start_frame, end_frame).export(window_input, format="wav")
        args = {
            "operation": "boundary_smooth",
            "input_audio_path": str(window_input),
            "output_path": str(window_output),
            "output_dir": str(work_dir / f"{base_output_stem}_smooth_{index:03d}_artifacts"),
            "text": text,
            "points": [
                {
                    "mode": "smooth_only",
                    "position_sec": round(local_boundary_sec, 6),
                }
            ],
        }
        result = await tool_executor.execute("edit_data_pipeline", args)
        record = {
            **boundary,
            "window_start_sec": round(window_start, 6),
            "window_end_sec": round(window_end, 6),
            "window_duration_sec": round(window_end - window_start, 6),
            "local_boundary_time_sec": round(local_boundary_sec, 6),
            "input_audio_path": str(window_input),
            "output_audio_path": str(window_output),
            "tool_success": bool(result.success),
            "tool_error": result.error_message,
        }
        if not result.success:
            record["status"] = "failed"
            records.append(record)
            continue
        if not window_output.exists():
            record["status"] = "failed"
            record["reason"] = "boundary_smooth_missing_output"
            records.append(record)
            continue
        smoothed = AudioSegment.from_file(window_output)
        duration_drift_sec = abs(
            smoothed.frame_count() / smoothed.frame_rate
            - (end_frame - start_frame) / current_audio.frame_rate
        )
        record["duration_drift_sec"] = round(duration_drift_sec, 6)
        if duration_drift_sec > 0.02:
            record["status"] = "rejected"
            record["reason"] = "boundary_smooth_duration_drift_exceeded_20ms"
            records.append(record)
            continue
        current_audio = (
            current_audio.get_sample_slice(end_sample=start_frame)
            + smoothed
            + current_audio.get_sample_slice(start_sample=end_frame)
        )
        record["status"] = "applied"
        records.append(record)
    return current_audio, list(reversed(records))


def _max_segment_seconds() -> float:
    value = float(os.getenv("X_EDITOR_EDIT_SEGMENT_MAX_SEC", "29.5"))
    return min(value, 29.5)


def _preemptive_text_edit_segment_seconds() -> float:
    value = float(os.getenv("X_EDITOR_TEXT_EDIT_PREEMPTIVE_SEGMENT_SEC", "18.0"))
    return max(0.0, value)


def _should_preemptively_segment_text_edit(arguments: dict[str, Any], *, duration_sec: float) -> bool:
    if str(arguments.get("operation") or "") != "text_edit":
        return False
    if duration_sec < _preemptive_text_edit_segment_seconds():
        return False
    instruction_st = arguments.get("instruction_st")
    if not isinstance(instruction_st, str) or not _TEXT_EDIT_TAG_RE.search(instruction_st):
        return False
    source_text = str(arguments.get("source_text") or _render_text_edit_source(instruction_st) or "")
    if not source_text.strip():
        return False
    if duration_sec < 30.0 and _is_whole_utterance_text_edit(arguments):
        return False
    tag_count = len(list(_TEXT_EDIT_TAG_RE.finditer(instruction_st)))
    if tag_count > 1:
        return True
    source_units = _duration_units(source_text)
    target_units = _duration_units(_render_text_edit_target(instruction_st))
    return source_units > 0.0 and target_units >= source_units


def _text_edit_target_segment_seconds(arguments: dict[str, Any], *, duration_sec: float) -> float | None:
    if str(arguments.get("operation") or "") != "text_edit":
        return None
    instruction_st = arguments.get("instruction_st")
    if not isinstance(instruction_st, str) or not _TEXT_EDIT_TAG_RE.search(instruction_st):
        return None
    if duration_sec < 30.0 and _is_whole_utterance_text_edit(arguments):
        return None
    try:
        limit_sec = float(arguments.get("max_f5_edit_seconds") or 0.0)
    except (TypeError, ValueError):
        limit_sec = 0.0
    if limit_sec <= 0.0:
        return None
    source_text = str(arguments.get("source_text") or _render_text_edit_source(instruction_st) or "")
    target_text = _render_text_edit_target(instruction_st)
    source_units = _duration_units(source_text)
    target_units = _duration_units(target_text)
    if source_units <= 0.0 or target_units <= source_units:
        return None
    source_chars = _nonspace_len(source_text)
    target_chars = _nonspace_len(target_text)
    unit_ratio = target_units / source_units
    char_ratio = target_chars / source_chars if source_chars > 0 and target_chars > source_chars else 1.0
    growth_ratio = max(unit_ratio, char_ratio)
    estimated_target_sec = duration_sec * growth_ratio
    if estimated_target_sec < limit_sec * 0.95:
        return None
    return max(8.0, min(_max_segment_seconds(), limit_sec / growth_ratio * 0.75))


def _is_whole_utterance_text_edit(arguments: dict[str, Any]) -> bool:
    request = _single_text_edit_phrase_request(arguments)
    if request is None or request.kind != "sub":
        return False
    source_text = str(arguments.get("source_text") or "").strip()
    instruction_st = str(arguments.get("instruction_st") or "")
    rendered_source = _render_text_edit_source(instruction_st).strip()
    if not source_text or not rendered_source:
        return False
    source_key = _match_text_key(source_text)
    return (
        source_key == _match_text_key(request.source)
        and source_key == _match_text_key(rendered_source)
    )


def _inherit_or_scale_words_for_input(
    words: list[dict[str, Any]],
    *,
    perception: dict[str, Any],
    input_audio_path: str,
    input_duration_sec: float,
) -> list[dict[str, Any]]:
    if not words:
        return []
    ratio = _alignment_scale_ratio(
        perception=perception,
        input_audio_path=input_audio_path,
        input_duration_sec=input_duration_sec,
        fallback_end=max(float(item["end_sec"]) for item in words),
    )
    if ratio is None:
        return words
    return [
        {
            **item,
            "start_sec": max(0.0, float(item["start_sec"]) * ratio),
            "end_sec": min(input_duration_sec, max(0.0, float(item["end_sec"]) * ratio)),
            "alignment_source": "scaled_inherited_word_timestamps",
            "alignment_scale": ratio,
        }
        for item in words
    ]


def _inherit_or_scale_speaker_segments_for_input(
    segments: list[dict[str, Any]],
    *,
    perception: dict[str, Any],
    input_audio_path: str,
    input_duration_sec: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    ratio = _alignment_scale_ratio(
        perception=perception,
        input_audio_path=input_audio_path,
        input_duration_sec=input_duration_sec,
        fallback_end=max(float(item["end_sec"]) for item in segments),
    )
    if ratio is None:
        return segments
    return [
        {
            **item,
            "start_sec": max(0.0, float(item["start_sec"]) * ratio),
            "end_sec": min(input_duration_sec, max(0.0, float(item["end_sec"]) * ratio)),
            "alignment_source": "scaled_inherited_speaker_segments",
            "alignment_scale": ratio,
        }
        for item in segments
    ]


def _alignment_refresh_required_for_input(
    items: list[dict[str, Any]],
    *,
    perception: dict[str, Any],
    input_audio_path: str,
    input_duration_sec: float,
) -> bool:
    if not items:
        return False
    source_audio = str(perception.get("audio_path") or perception.get("source_audio_path") or "")
    if not source_audio:
        return False
    if Path(source_audio).expanduser().resolve() == Path(input_audio_path).expanduser().resolve():
        return False
    source_duration = _safe_float(
        perception.get("duration_sec")
        or perception.get("duration_seconds")
        or perception.get("audio_duration_sec")
    )
    if source_duration is None:
        source_duration = _audio_duration_seconds(source_audio)
    if source_duration is None:
        source_duration = max(float(item["end_sec"]) for item in items)
    if not source_duration or source_duration <= 0.0:
        return False
    ratio = input_duration_sec / source_duration
    if abs(ratio - 1.0) < 0.015:
        return False
    return not 0.5 <= ratio <= 2.0


def _alignment_scale_ratio(
    *,
    perception: dict[str, Any],
    input_audio_path: str,
    input_duration_sec: float,
    fallback_end: float,
) -> float | None:
    source_audio = str(perception.get("audio_path") or perception.get("source_audio_path") or "")
    if source_audio and Path(source_audio).expanduser().resolve() == Path(input_audio_path).expanduser().resolve():
        return None
    source_duration = _safe_float(
        perception.get("duration_sec")
        or perception.get("duration_seconds")
        or perception.get("audio_duration_sec")
    )
    if source_duration is None and source_audio:
        source_duration = _audio_duration_seconds(source_audio)
    if source_duration is None:
        source_duration = fallback_end
    if not source_duration or source_duration <= 0.0:
        return None
    ratio = input_duration_sec / source_duration
    if abs(ratio - 1.0) < 0.015:
        return None
    if 0.5 <= ratio <= 2.0:
        return ratio
    return None


def _refresh_alignment_for_segmentation(
    *,
    input_audio_path: str,
    source_text: str,
    language: Any,
    output_dir: str,
    input_duration_sec: float | None = None,
) -> dict[str, Any] | ToolResult:
    if not source_text.strip():
        return _failure(
            "requires_alignment_for_segmentation: source_text is required before calling the aligner."
        )
    try:
        client = SpeechPerceptionClient(cache_dir=Path(output_dir) / "segmented_edit_perception_cache")
        if _source_text_likely_partial_for_segmentation(source_text, input_duration_sec):
            return client.perceive(
                input_audio_path,
                language=str(language) if language else None,
                source_text_hint=source_text,
                use_asr=True,
                use_alignment=True,
                use_diarization=False,
                strict_asr=True,
                strict_alignment=True,
            )
        return client.perceive(
            input_audio_path,
            language=str(language) if language else None,
            source_text_hint=source_text,
            use_asr=False,
            use_alignment=True,
            use_diarization=False,
            strict_alignment=True,
        )
    except SpeechPerceptionError as exc:
        return _failure(
            "requires_alignment_for_segmentation: aligner refresh failed or service is unavailable: "
            f"{exc}"
        )


def _audio_duration_seconds(path: str) -> float | None:
    try:
        return len(AudioSegment.from_file(path)) / 1000.0
    except Exception:
        return None


def _source_text_likely_partial_for_segmentation(source_text: str, input_duration_sec: float | None) -> bool:
    """Return true when source_text is too short to be a full segmentation transcript."""

    if input_duration_sec is None:
        return False
    duration = _safe_float(input_duration_sec)
    if duration is None or duration < max(8.0, _preemptive_text_edit_segment_seconds()):
        return False
    units = _duration_units(source_text)
    if units <= 0.0:
        return False
    # Typical read speech has far more than one word/character per second.
    # A transcript much sparser than the audio is likely a local edit phrase,
    # not a safe full-clip alignment contract for F5 segmentation.
    return units < max(8.0, duration * 0.75)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_text_edit_source(instruction_st: str) -> str:
    def repl(match: re.Match[str]) -> str:
        if match.group("sub_attrs") is not None:
            return match.group("sub_body") or ""
        if match.group("ins_body") is not None:
            return ""
        if match.group("del_body") is not None:
            return match.group("del_body") or ""
        return match.group(0)

    return unescape(_TEXT_EDIT_TAG_RE.sub(repl, instruction_st))


def _duration_units(text: str) -> float:
    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text))
    other_chars = len(re.findall(r"[^\sA-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff.,!?;:，。！？；：、'’\"“”‘’()\\[\\]{}<>/\\\\|-]", text))
    return float(cjk_chars + latin_words + other_chars)


def _nonspace_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _require_speaker_segments(*, operation: str, prompt: str, perception: dict[str, Any] | None = None) -> bool:
    raw = os.getenv("X_EDITOR_SEGMENT_REQUIRE_SPEAKER")
    if raw is not None:
        return raw.lower() not in {"0", "false", "no"}
    return _mentions_speaker_scope(prompt) or _perception_has_multispeaker_evidence(perception or {})


def _mentions_speaker_scope(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return bool(
        re.search(
            r"\b(?:speaker|role|character|person|man|woman|male|female|boy|girl|child|kid|adult|mother|father)\b",
            lowered,
        )
        or re.search(r"(说话人|角色|男声|女声|男的|女的|小孩|儿童|成人|母亲|父亲)", lowered)
    )


def _perception_has_multispeaker_evidence(perception: dict[str, Any]) -> bool:
    speaker_ids: set[str] = set()
    for key in ("speaker_segments", "segments", "speakers"):
        raw = perception.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                speaker_id = item.get("speaker_id") or item.get("speaker")
            else:
                speaker_id = item
            if speaker_id is not None and str(speaker_id).strip():
                speaker_ids.add(str(speaker_id))
    diarization = perception.get("diarization")
    if isinstance(diarization, dict):
        raw_speakers = diarization.get("speakers")
        if isinstance(raw_speakers, list):
            speaker_ids.update(str(item) for item in raw_speakers if str(item).strip())
        raw_segments = diarization.get("segments")
        if isinstance(raw_segments, list):
            for item in raw_segments:
                if isinstance(item, dict) and item.get("speaker_id") is not None:
                    speaker_ids.add(str(item["speaker_id"]))
    return len(speaker_ids) > 1


def _perception_from_state(state: dict[str, Any]) -> dict[str, Any]:
    perception = state.get("perception_report") or state.get("audio_perception") or {}
    return perception if isinstance(perception, dict) else {}


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
        start_f = float(start)
        end_f = float(end)
        if end_f < start_f:
            continue
        if end_f == start_f:
            end_f = start_f + 0.01
        words.append(
            {
                "text": str(text),
                "start_sec": start_f,
                "end_sec": end_f,
                "is_punctuation": bool(item.get("is_punctuation", False)),
            }
        )
    return sorted(words, key=lambda item: (item["start_sec"], item["end_sec"]))


def _normalized_speaker_segments(perception: dict[str, Any]) -> list[dict[str, Any]]:
    raw = perception.get("speaker_segments") or perception.get("segments") or []
    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id")
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if speaker_id is None or start is None or end is None:
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        segments.append({"speaker_id": str(speaker_id), "start_sec": start_f, "end_sec": end_f})
    return sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"]))


def _build_chunks(
    words: list[dict[str, Any]],
    speaker_segments: list[dict[str, Any]],
    *,
    duration_sec: float,
    max_segment_sec: float,
) -> list[EditChunk]:
    chunks: list[EditChunk] = []
    current: list[dict[str, Any]] = []
    current_start = 0.0
    pending_sentence_boundary = False
    reason = "start"

    for word in words:
        word = dict(word)
        word["speaker_id"] = _speaker_for_word(word, speaker_segments)
        if current:
            prev = current[-1]
            speaker_changed = (
                prev.get("speaker_id") is not None
                and word.get("speaker_id") is not None
                and prev.get("speaker_id") != word.get("speaker_id")
            )
            too_long = word["end_sec"] - current_start >= max(0.5, max_segment_sec - 0.25)
            if pending_sentence_boundary or speaker_changed or too_long:
                boundary = _word_boundary_time(prev, word)
                if too_long:
                    boundary = min(boundary, current_start + max(0.01, max_segment_sec - 0.05))
                    boundary = max(boundary, float(prev["end_sec"]))
                chunks.append(_make_edit_chunk(len(chunks), current_start, boundary, current, reason))
                current_start = boundary
                current = []
                reason = "sentence" if pending_sentence_boundary else "speaker_change" if speaker_changed else "max_length"
                pending_sentence_boundary = False
        current.append(word)
        pending_sentence_boundary = _is_sentence_end(word)

    if current:
        chunks.append(_make_edit_chunk(len(chunks), current_start, duration_sec, current, reason))

    return _split_oversized_silence(chunks, max_segment_sec=max_segment_sec)


def _split_oversized_silence(chunks: list[EditChunk], *, max_segment_sec: float) -> list[EditChunk]:
    output: list[EditChunk] = []
    for chunk in chunks:
        if chunk.duration_sec <= max_segment_sec + 0.05:
            chunk.index = len(output)
            output.append(chunk)
            continue
        if not chunk.words:
            start = chunk.start_sec
            while start < chunk.end_sec:
                end = min(chunk.end_sec, start + max_segment_sec)
                output.append(EditChunk(len(output), start, end, "", None, [], "silence", editable=False))
                start = end
            continue
        last_word_end = max(word["end_sec"] for word in chunk.words)
        edited_end = min(chunk.end_sec, max(last_word_end, chunk.start_sec + 0.01))
        edited = EditChunk(
            len(output),
            chunk.start_sec,
            edited_end,
            chunk.text,
            chunk.speaker_id,
            chunk.words,
            chunk.reason,
            editable=True,
        )
        output.append(edited)
        start = edited_end
        while start < chunk.end_sec:
            end = min(chunk.end_sec, start + max_segment_sec)
            output.append(EditChunk(len(output), start, end, "", None, [], "tail_silence", editable=False))
            start = end
    return output


def _make_edit_chunk(
    index: int,
    start_sec: float,
    end_sec: float,
    words: list[dict[str, Any]],
    reason: str,
) -> EditChunk:
    speaker_ids = {word.get("speaker_id") for word in words if word.get("speaker_id") is not None}
    return EditChunk(
        index=index,
        start_sec=max(0.0, start_sec),
        end_sec=max(start_sec, end_sec),
        text=_render_words(words),
        speaker_id=next(iter(speaker_ids)) if len(speaker_ids) == 1 else None,
        words=words,
        reason=reason,
        editable=True,
    )


def _speaker_for_word(word: dict[str, Any], speaker_segments: list[dict[str, Any]]) -> str | None:
    if not speaker_segments:
        return None
    midpoint = (word["start_sec"] + word["end_sec"]) / 2.0
    for segment in speaker_segments:
        if segment["start_sec"] <= midpoint <= segment["end_sec"]:
            return segment["speaker_id"]
    best_segment = None
    best_overlap = 0.0
    for segment in speaker_segments:
        overlap = max(0.0, min(word["end_sec"], segment["end_sec"]) - max(word["start_sec"], segment["start_sec"]))
        if overlap > best_overlap:
            best_segment = segment
            best_overlap = overlap
    return best_segment["speaker_id"] if best_segment and best_overlap > 0 else None


def _word_boundary_time(prev_word: dict[str, Any], next_word: dict[str, Any]) -> float:
    prev_end = float(prev_word["end_sec"])
    next_start = float(next_word["start_sec"])
    if next_start > prev_end:
        return (prev_end + next_start) / 2.0
    return prev_end


def _is_sentence_end(word: dict[str, Any]) -> bool:
    text = str(word.get("text") or "")
    return bool(word.get("is_punctuation")) and text in {".", "!", "?", "。", "！", "？"} or bool(
        re.search(r"[.!?。！？]$", text)
    )


def _render_words(words: list[dict[str, Any]]) -> str:
    tokens = [str(word.get("text") or "") for word in words if str(word.get("text") or "")]
    if _looks_like_cjk_character_stream(tokens):
        return "".join(tokens)
    pieces: list[str] = []
    for text in tokens:
        if not pieces:
            pieces.append(text)
        elif re.fullmatch(r"[,.!?;:，。！？；：、]+", text):
            pieces[-1] += text
        else:
            pieces.append(text)
    return " ".join(pieces)


def _looks_like_cjk_character_stream(tokens: list[str]) -> bool:
    content = [token for token in tokens if token.strip()]
    if not content:
        return False
    cjk_count = sum(1 for token in content if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", token))
    return cjk_count >= max(1, int(len(content) * 0.6))


def _validate_chunks(chunks: list[EditChunk], *, max_segment_sec: float, require_speaker: bool) -> str | None:
    for chunk in chunks:
        if chunk.duration_sec >= 30.0:
            return f"segment {chunk.index} is {chunk.duration_sec:.3f}s; expected every segment to be < 30s"
        if chunk.duration_sec > max_segment_sec + 0.05:
            return f"segment {chunk.index} exceeds configured max segment length {max_segment_sec:.3f}s"
        speaker_ids = {word.get("speaker_id") for word in chunk.words if word.get("speaker_id") is not None}
        if require_speaker and chunk.editable and chunk.words and len(speaker_ids) != 1:
            return f"segment {chunk.index} is not single-speaker; speaker_ids={sorted(speaker_ids)}"
    return None


def _slice_audio(audio: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    return audio[int(round(start_sec * 1000)) : int(round(end_sec * 1000))]


def _text_edit_applies(prompt: str, segment_text: str, segment_index: int) -> bool:
    quoted = _quoted_strings(prompt)
    lowered = prompt.lower()
    if len(quoted) >= 2 and _contains(lowered, ["replace", "change", "modify", "relace", "改成", "替换成", "换成"]):
        return quoted[0].lower() in segment_text.lower()
    if _contains(lowered, ["replace the first sentence", "replace first sentence"]):
        return segment_index == 0
    if _contains(lowered, ["delete the first sentence", "remove the first sentence"]):
        return segment_index == 0
    return True


def _copy_segment_overrides(source: dict[str, Any], target: dict[str, Any]) -> None:
    protected = {"input_audio_path", "output_path", "source_text", "instruction_st", "target_text", "text", "speaker_audio_path"}
    for key, value in source.items():
        if key in protected or key not in target:
            continue
        target[key] = value


_TEXT_EDIT_TAG_RE = re.compile(
    r"<sub\s+(?P<sub_attrs>[^>]*)>(?P<sub_body>.*?)</sub>"
    r"|<ins\b[^>]*>(?P<ins_body>.*?)</ins>"
    r"|<del\b[^>]*>(?P<del_body>.*?)</del>",
    flags=re.DOTALL,
)


def _tagged_text_edit_segment_arguments(
    arguments: dict[str, Any],
    *,
    chunk: EditChunk,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
) -> dict[str, Any] | None:
    if str(arguments.get("operation") or "") != "text_edit":
        return None
    instruction_st = arguments.get("instruction_st")
    if not isinstance(instruction_st, str) or not _TEXT_EDIT_TAG_RE.search(instruction_st):
        return None
    segment_instruction = _project_text_edit_tags_to_segment(instruction_st, chunk.text)
    if segment_instruction is None:
        return None
    segment_args = dict(arguments)
    segment_args.update(
        {
            "operation": "text_edit",
            "input_audio_path": input_audio_path,
            "output_path": output_path,
            "output_dir": output_dir,
            "source_text": chunk.text,
            "instruction_st": segment_instruction,
            "target_text": _render_text_edit_target(segment_instruction),
        }
    )
    segment_args.setdefault("synthesis_mode", "per_op")
    return segment_args


def _project_text_edit_tags_to_segment(instruction_st: str, segment_text: str) -> str | None:
    if not segment_text:
        return None
    parts: list[str] = []
    cursor = 0
    applied = False
    for match in _TEXT_EDIT_TAG_RE.finditer(instruction_st):
        if match.group("sub_attrs") is not None:
            source = unescape(match.group("sub_body") or "")
            target = _sub_target(match.group("sub_attrs") or "")
            if target is None:
                continue
            tag = f'<sub targ="{escape(target, quote=True)}">{escape(source, quote=False)}</sub>'
        elif match.group("del_body") is not None:
            source = unescape(match.group("del_body") or "")
            tag = f"<del>{escape(source, quote=False)}</del>"
        else:
            # Insertions do not have an unambiguous audio timestamp in a chunked
            # projection, so leave them to the structured fallback path.
            continue
        if not source:
            continue
        position = segment_text.find(source, cursor)
        if position < 0:
            continue
        parts.append(escape(segment_text[cursor:position], quote=False))
        parts.append(tag)
        cursor = position + len(source)
        applied = True
    if not applied:
        return None
    parts.append(escape(segment_text[cursor:], quote=False))
    return "".join(parts)


def _sub_target(attrs: str) -> str | None:
    match = re.search(r'targ="([^"]*)"', attrs) or re.search(r"targ='([^']*)'", attrs)
    return unescape(match.group(1)) if match else None


def _render_text_edit_target(instruction_st: str) -> str:
    def repl(match: re.Match[str]) -> str:
        if match.group("sub_attrs") is not None:
            return escape(_sub_target(match.group("sub_attrs") or "") or "", quote=False)
        if match.group("ins_body") is not None:
            return match.group("ins_body") or ""
        if match.group("del_body") is not None:
            return ""
        return match.group(0)

    return unescape(_TEXT_EDIT_TAG_RE.sub(repl, instruction_st))


def _segment_record(
    chunk: EditChunk,
    input_path: Path,
    output_path: Path,
    *,
    copied: bool,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "index": chunk.index,
        "start_sec": round(chunk.start_sec, 6),
        "end_sec": round(chunk.end_sec, 6),
        "duration_sec": round(chunk.duration_sec, 6),
        "speaker_id": chunk.speaker_id,
        "text": chunk.text,
        "reason": chunk.reason,
        "editable": chunk.editable,
        "copied": copied,
        "input_audio_path": str(input_path),
        "output_audio_path": str(output_path),
        "tool_output": result or {},
    }


def _quoted_strings(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", text):
        values.append(next(group for group in match.groups() if group is not None))
    return values


def _contains(text: str, needles: list[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _failure(message: str) -> ToolResult:
    return ToolResult(tool_name="edit_data_pipeline.apply", success=False, error_message=message)
