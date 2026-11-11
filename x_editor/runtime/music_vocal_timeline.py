"""Timeline reconstruction for singing/rap speech edits.

This module composes existing X-Editor perception/artifact utilities with the
ACE-Step music-vocal tool. It does not implement music generation locally: every
selected speech span is delegated to ``ace_step_music`` and then fitted back to
the original timeline.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydub import AudioSegment

from x_editor.runtime.artifact_ops import (
    TimeSpan,
    requires_perception_segment_selector,
    select_timeline_spans_with_perception_fallback,
)


ExecuteTool = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]

MAX_ACE_STEP_DURATION_SEC = 600.0
MIN_ACE_STEP_DURATION_SEC = 10.0


async def execute_music_vocal_timeline_edit(
    *,
    execute_tool: ExecuteTool,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    instruction: str,
    source_text: str | None = None,
    perception_report: dict[str, Any] | None = None,
    reference_audio_paths: list[str] | None = None,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn selected speech spans into singing/rap and splice them back.

    The executor resolves target spans from explicit params, word timestamps,
    speaker diarization, or a conservative whole-audio fallback for requests
    that address "this clip/segment". Each selected span is used as both source
    audio and the default timbre/style reference for ACE-Step.
    """

    params = dict(params or {})
    state = dict(state or {})
    input_audio = Path(input_audio_path)
    if not input_audio.exists():
        raise FileNotFoundError(f"music vocal timeline input audio does not exist: {input_audio}")

    original = AudioSegment.from_file(input_audio)
    duration_sec = len(original) / 1000.0
    work_dir = Path(output_dir)
    segments_dir = work_dir / "segments"
    generated_dir = work_dir / "generated"
    fitted_dir = work_dir / "fitted"
    for directory in [segments_dir, generated_dir, fitted_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    perception = perception_report if isinstance(perception_report, dict) else {}
    spans, span_selection_evidence = _resolve_music_vocal_spans(
        instruction=instruction,
        perception=perception,
        input_audio_path=str(input_audio),
        output_dir=str(work_dir),
        duration_sec=duration_sec,
        source_text=source_text,
        params=params,
        agentic_mode=params.get("_agentic_mode") or params.get("agentic_mode") or (state.get("context") or {}).get("agentic_mode"),
    )
    if not spans:
        raise RuntimeError(
            "music_vocal_timeline_edit could not resolve the speaker or speech segment to convert. "
            "Provide selected_spans or run perception with ASR word_timestamps and speaker_segments."
        )

    output_timeline = original
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    vocal_style = str(params.get("vocal_style") or _music_vocal_style(instruction))
    reference_strategy = str(params.get("reference_strategy") or "selected_segment")
    external_refs = [str(path) for path in (reference_audio_paths or []) if path]

    for index, span in enumerate(spans):
        segment_input = segments_dir / f"span_{index:03d}_source.wav"
        _slice_audio(original, span.start_sec, span.end_sec).export(segment_input, format="wav")

        raw_output = generated_dir / f"span_{index:03d}_ace_step_raw.wav"
        fitted_output = fitted_dir / f"span_{index:03d}_fitted.wav"
        lyrics = _lyrics_for_span(span, spans=spans, source_text=source_text, perception=perception)
        if not lyrics and not _allow_missing_lyrics(params):
            raise RuntimeError(
                f"music_vocal_timeline_edit span {index} has no transcript/lyrics. "
                "Run ASR+alignment first, pass source_text, or set allow_missing_lyrics=true for best-effort generation."
            )
        if not lyrics:
            warnings.append(f"span {index} sent to ACE-Step without lyrics; content preservation is not guaranteed")

        reference_audio = _reference_audio_for_span(
            segment_input=str(segment_input),
            external_refs=external_refs,
            strategy=reference_strategy,
        )
        ace_args = _ace_step_arguments(
            instruction=instruction,
            span=span,
            segment_input=str(segment_input),
            reference_audio=reference_audio,
            output_path=str(raw_output),
            lyrics=lyrics,
            vocal_style=vocal_style,
            params=params,
        )
        result = await execute_tool("ace_step_music", ace_args, state)
        generated_path = _extract_output_audio_path(result) or str(raw_output)
        if not Path(generated_path).exists():
            raise RuntimeError(f"ACE-Step did not create an output file for span {index}: {generated_path}")

        target_ms = int(round(span.duration_sec * 1000))
        replacement = AudioSegment.from_file(generated_path)
        replacement = _match_loudness(_fit_audio_duration(replacement, target_ms), _slice_audio(original, span.start_sec, span.end_sec))
        fade_ms = min(int(params.get("fade_ms", params.get("crossfade_ms", 20))), max(0, target_ms // 4))
        if fade_ms > 0:
            replacement = replacement.fade_in(fade_ms).fade_out(fade_ms)
        replacement.export(fitted_output, format="wav")
        output_timeline = _replace_range(output_timeline, replacement, span.start_sec, span.end_sec)

        records.append(
            {
                **span.to_dict(),
                "input_segment_path": str(segment_input),
                "reference_audio_path": reference_audio,
                "lyrics": lyrics,
                "ace_step_arguments": _redact_ace_args(ace_args),
                "ace_step_output_path": generated_path,
                "fitted_segment_path": str(fitted_output),
                "ace_step_result": result,
            }
        )

    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output_timeline.export(final_output, format="wav")
    manifest = {
        "operation": "music_vocal_timeline_edit",
        "instruction": instruction,
        "input_audio_path": str(input_audio),
        "output_audio_path": str(final_output),
        "vocal_style": vocal_style,
        "selected_spans": [span.to_dict() for span in spans],
        "segments": records,
        "warnings": warnings,
        "perception_segment_selection": span_selection_evidence,
        "constraints": {
            "timeline_duration_preserved": True,
            "segment_source_audio_used_as_default_reference": reference_strategy == "selected_segment",
            "ace_step_min_duration_sec": MIN_ACE_STEP_DURATION_SEC,
            "ace_step_max_duration_sec": MAX_ACE_STEP_DURATION_SEC,
            "speaker_identity": "best_effort",
            "lyrics_content": "requires transcript/lyrics for content preservation",
        },
        "perception_services": perception.get("services", {}),
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_audio_path": str(final_output),
        "operation": "music_vocal_timeline_edit",
        "manifest_path": str(manifest_path),
        "selected_spans": manifest["selected_spans"],
        "perception_segment_selection": span_selection_evidence,
        "segments": records,
        "warnings": warnings,
        "vocal_style": vocal_style,
    }


def _resolve_music_vocal_spans(
    *,
    instruction: str,
    perception: dict[str, Any],
    input_audio_path: str,
    output_dir: str,
    duration_sec: float,
    source_text: str | None,
    params: dict[str, Any],
    agentic_mode: str | None = None,
) -> tuple[list[TimeSpan], dict[str, Any] | None]:
    explicit = _spans_from_params(params.get("selected_spans") or params.get("spans"))
    if explicit:
        return _clip_merge_spans(explicit, duration_sec), None

    selector_action = "speaker_scoped_edit"
    spans, selection_evidence = select_timeline_spans_with_perception_fallback(
        instruction,
        perception,
        input_audio_path=input_audio_path,
        output_dir=output_dir,
        duration_sec=duration_sec,
        action=selector_action,
        agentic_mode=agentic_mode,
    )
    if spans:
        return (
            _split_for_ace_step(_clip_merge_spans(spans, duration_sec), _normalized_words(perception), source_text),
            selection_evidence,
        )
    if requires_perception_segment_selector(instruction, selector_action):
        return [], selection_evidence

    words = _normalized_words(perception)
    if words and _whole_audio_requested(instruction):
        return _split_for_ace_step([_span_from_words(words, duration_sec)], words, source_text), None

    if _whole_audio_requested(instruction) or _single_speaker_hint(instruction, perception):
        text = source_text or str(perception.get("source_text") or "")
        return (
            _split_for_ace_step(
                [TimeSpan(0.0, duration_sec, "whole_audio", None, text, "whole_audio_fallback")],
                words,
                source_text,
            ),
            None,
        )

    return [], selection_evidence


def _ace_step_arguments(
    *,
    instruction: str,
    span: TimeSpan,
    segment_input: str,
    reference_audio: str | None,
    output_path: str,
    lyrics: str,
    vocal_style: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    mode = str(params.get("mode") or "cover")
    duration_seconds = _generation_duration_seconds(span.duration_sec, params)
    args: dict[str, Any] = {
        "instruction": instruction,
        "lyrics": lyrics,
        "music_prompt": str(params.get("music_prompt") or _music_prompt(instruction, vocal_style)),
        "source_audio_path": segment_input,
        "reference_audio_path": reference_audio,
        "output_path": output_path,
        "mode": mode,
        "vocal_style": vocal_style,
        "vocal_language": str(params.get("vocal_language") or _language_hint(lyrics or instruction)),
        "duration_seconds": duration_seconds,
        "audio_format": str(params.get("audio_format") or "wav"),
        "audio_cover_strength": float(params.get("audio_cover_strength", 0.7 if vocal_style == "rap" else 0.8)),
        "inference_steps": int(params.get("inference_steps", 8)),
        "guidance_scale": float(params.get("guidance_scale", 7.0)),
        "thinking": bool(params.get("thinking", mode == "text2music")),
        "use_format": bool(params.get("use_format", mode == "text2music")),
    }
    if mode == "repaint":
        args.setdefault("repainting_start", 0.0)
        args.setdefault("repainting_end", span.duration_sec)
    for key in [
        "seed",
        "use_random_seed",
        "bpm",
        "key_scale",
        "time_signature",
        "model",
        "poll_interval_seconds",
        "timeout_seconds",
    ]:
        if key in params:
            args[key] = params[key]
    return {key: value for key, value in args.items() if value not in (None, "")}


def _generation_duration_seconds(span_duration_sec: float, params: dict[str, Any]) -> float:
    requested = params.get("duration_seconds")
    if requested is not None:
        value = float(requested)
    else:
        value = max(MIN_ACE_STEP_DURATION_SEC, math.ceil(span_duration_sec))
    return max(MIN_ACE_STEP_DURATION_SEC, min(MAX_ACE_STEP_DURATION_SEC, value))


def _lyrics_for_span(
    span: TimeSpan,
    *,
    spans: list[TimeSpan],
    source_text: str | None,
    perception: dict[str, Any],
) -> str:
    if span.text.strip():
        return span.text.strip()
    words = _normalized_words(perception)
    selected_text = _words_text_for_span(words, span)
    if selected_text:
        return selected_text
    if len(spans) == 1:
        return str(source_text or perception.get("source_text") or "").strip()
    return ""


def _reference_audio_for_span(*, segment_input: str, external_refs: list[str], strategy: str) -> str | None:
    if strategy == "external_first" and external_refs:
        return external_refs[0]
    return segment_input


def _spans_from_params(value: Any) -> list[TimeSpan]:
    if not isinstance(value, list):
        return []
    spans: list[TimeSpan] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None or end is None:
            continue
        spans.append(
            TimeSpan(
                float(start),
                float(end),
                str(item.get("label") or "explicit_selection"),
                str(item["speaker_id"]) if item.get("speaker_id") is not None else None,
                str(item.get("text") or ""),
                str(item.get("source") or "params.selected_spans"),
            )
        )
    return spans


def _normalized_words(perception: dict[str, Any]) -> list[dict[str, Any]]:
    raw = perception.get("word_timestamps") or []
    words: list[dict[str, Any]] = []
    for item in raw:
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
        words.append(
            {
                "text": str(text),
                "start_sec": start_f,
                "end_sec": end_f,
                "speaker_id": item.get("speaker_id"),
                "is_punctuation": bool(item.get("is_punctuation", False)),
            }
        )
    return sorted(words, key=lambda item: (item["start_sec"], item["end_sec"]))


def _span_from_words(words: list[dict[str, Any]], duration_sec: float) -> TimeSpan:
    return TimeSpan(
        max(0.0, float(words[0]["start_sec"])),
        min(duration_sec, float(words[-1]["end_sec"])),
        "word_timestamps",
        _single_speaker_id(words),
        _render_words(words),
        "word_timestamps",
    )


def _split_for_ace_step(spans: list[TimeSpan], words: list[dict[str, Any]], source_text: str | None) -> list[TimeSpan]:
    result: list[TimeSpan] = []
    for span in spans:
        if span.duration_sec <= MAX_ACE_STEP_DURATION_SEC:
            result.append(span)
            continue
        start = span.start_sec
        while start < span.end_sec:
            limit = min(span.end_sec, start + MAX_ACE_STEP_DURATION_SEC)
            word_ends = [
                float(word["end_sec"])
                for word in words
                if start < float(word["end_sec"]) <= limit
                and _overlap_seconds(start, limit, float(word["start_sec"]), float(word["end_sec"])) > 0
            ]
            end = max(word_ends) if word_ends else limit
            if end <= start:
                end = limit
            split_words = [
                word
                for word in words
                if _overlap_seconds(start, end, float(word["start_sec"]), float(word["end_sec"])) > 0
            ]
            result.append(
                TimeSpan(
                    start,
                    end,
                    f"{span.label}:ace_split",
                    span.speaker_id,
                    _render_words(split_words) or span.text or (source_text or ""),
                    span.source,
                )
            )
            start = end
    return result


def _clip_merge_spans(spans: list[TimeSpan], duration_sec: float, *, merge_gap_sec: float = 0.03) -> list[TimeSpan]:
    clipped = [span.clipped(duration_sec) for span in spans]
    valid = sorted((span for span in clipped if span is not None), key=lambda item: (item.start_sec, item.end_sec))
    if not valid:
        return []
    merged = [valid[0]]
    for span in valid[1:]:
        previous = merged[-1]
        same_speaker = previous.speaker_id == span.speaker_id
        if span.start_sec <= previous.end_sec + merge_gap_sec and (same_speaker or not previous.speaker_id or not span.speaker_id):
            merged[-1] = TimeSpan(
                previous.start_sec,
                max(previous.end_sec, span.end_sec),
                previous.label or span.label,
                previous.speaker_id or span.speaker_id,
                " ".join(part for part in [previous.text, span.text] if part).strip(),
                previous.source or span.source,
            )
        else:
            merged.append(span)
    return merged


def _whole_audio_requested(instruction: str) -> bool:
    text = instruction.lower()
    return _contains(
        text,
        [
            "this audio",
            "this clip",
            "this segment",
            "whole audio",
            "entire audio",
            "the sentence",
            "the line",
            "这段",
            "整段",
            "这句话",
            "这句",
            "这条",
        ],
    )


def _single_speaker_hint(instruction: str, perception: dict[str, Any]) -> bool:
    if not _contains(instruction.lower(), ["speaker", "voice", "speech", "说话人", "人声", "男声", "女声"]):
        return False
    segments = perception.get("speaker_segments") or perception.get("segments") or []
    speaker_ids = {
        str(item.get("speaker_id") or item.get("speaker") or item.get("label"))
        for item in segments
        if isinstance(item, dict) and (item.get("speaker_id") or item.get("speaker") or item.get("label"))
    }
    return len(speaker_ids) <= 1


def _music_vocal_style(instruction: str) -> str:
    text = instruction.lower()
    if _contains(text, ["说唱", "嘻哈", "hip hop", "hip-hop"]) or re.search(r"\b(rap|rapping|rapper)\b", text):
        return "rap"
    if _contains(text, ["唱", "歌唱", "演唱", "sing", "singing", "sung", "song"]):
        return "singing"
    return "music_vocal"


def _music_prompt(instruction: str, vocal_style: str) -> str:
    text = instruction.lower()
    if vocal_style == "rap":
        return "hip hop beat with confident rap vocals, clear rhythm, studio mix"
    if _contains(text, ["摇滚", "rock"]):
        return "rock song with expressive lead singing vocal, clear lyrics, studio mix"
    if _contains(text, ["民谣", "folk"]):
        return "folk song with warm lead singing vocal, acoustic arrangement, clear lyrics"
    if _contains(text, ["合唱", "对唱", "duet"]):
        return "duet or chorus style with clear lead vocals and separated parts, studio mix"
    return "short pop song with expressive lead singing vocal, clear lyrics, studio mix"


def _language_hint(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _allow_missing_lyrics(params: dict[str, Any]) -> bool:
    value = params.get("allow_missing_lyrics", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _slice_audio(audio: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    return audio[int(round(start_sec * 1000)) : int(round(end_sec * 1000))]


def _replace_range(audio: AudioSegment, replacement: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))
    return audio[:start_ms] + replacement + audio[end_ms:]


def _fit_audio_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    target_ms = max(0, target_ms)
    if len(audio) > target_ms:
        return audio[:target_ms]
    if len(audio) < target_ms:
        return audio + AudioSegment.silent(duration=target_ms - len(audio), frame_rate=audio.frame_rate)
    return audio


def _match_loudness(replacement: AudioSegment, original: AudioSegment) -> AudioSegment:
    if original.dBFS == float("-inf") or replacement.dBFS == float("-inf"):
        return replacement
    gain = original.dBFS - replacement.dBFS
    if abs(gain) < 0.1:
        return replacement
    return replacement.apply_gain(gain)


def _words_text_for_span(words: list[dict[str, Any]], span: TimeSpan) -> str:
    selected = [
        word
        for word in words
        if _overlap_seconds(span.start_sec, span.end_sec, float(word["start_sec"]), float(word["end_sec"])) > 0
    ]
    return _render_words(selected)


def _render_words(words: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for word in words:
        text = str(word.get("text") or "")
        if not text:
            continue
        if not pieces:
            pieces.append(text)
        elif re.fullmatch(r"[,.!?;:，。！？；：、]+", text):
            pieces[-1] += text
        else:
            pieces.append(text)
    return " ".join(pieces)


def _single_speaker_id(words: list[dict[str, Any]]) -> str | None:
    speakers = {str(word["speaker_id"]) for word in words if word.get("speaker_id") is not None}
    return next(iter(speakers)) if len(speakers) == 1 else None


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _extract_output_audio_path(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in ("output_audio_path", "output_path", "generated_audio_path", "target_audio_path"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _redact_ace_args(args: dict[str, Any]) -> dict[str, Any]:
    copied = dict(args)
    for key in ("api_key", "authorization", "token"):
        if key in copied:
            copied[key] = "<redacted>"
    return copied


def copy_input_as_music_vocal_stub(input_audio_path: str, output_path: str) -> dict[str, Any]:
    """Test helper for dry environments; not used by production execution."""

    src = Path(input_audio_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"output_audio_path": str(dst), "backend_used": "stub_copy"}
