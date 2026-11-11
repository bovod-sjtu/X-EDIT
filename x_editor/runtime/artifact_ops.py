"""Deterministic audio artifact and timeline operations.

This module is an adapter around existing artifacts and services. It does not
implement model-level audio editing; it only selects spans, cuts/stitches audio,
and, for speaker-scoped local edits, delegates each selected segment to the
existing edit-data-pipeline service before putting it back on the timeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from pydub.generators import Sine
from pydub.silence import detect_silence

from x_editor.agents.base import GeminiClient
from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    OPEN_GLM_QWEN_OMNI_MODE,
    QWEN3_OMNI_PROVIDER,
    perception_provider_name,
    provider_service_hint,
    resolve_agentic_mode,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.config import resolve_gemini_api_key
from x_editor.perception import (
    PerceptionRequirements,
    build_audio_perception_fallback,
    missing_requirements,
    needs_gemini_audio_perception,
    needs_qwen3_omni_audio_perception,
)
from x_editor.planner.structured_edit import (
    build_edit_data_pipeline_arguments,
    build_structured_edit_sequence,
)
from x_editor.runtime.audio_segment_io import load_audio_segment, replace_fixed_audio_interval
from x_editor.runtime.audio_frame_geometry import (
    AudioFrameGeometry,
    append_audio_frames,
    fit_audio_frames,
    replace_audio_frames,
    slice_audio_frames,
)
from x_editor.perception.observation_scope import observation_scope
from x_editor.runtime.audio_effects import (
    apply_audio_effect,
    canonical_audio_effect_preset,
    target_vocal_effort_effect_preset,
)
from x_editor.runtime.operation_normalizers import (
    build_edit_data_pipeline_tool_arguments,
    canonicalize_volume_normalize_method,
)
from x_editor.runtime.reconstruct_contract import (
    normalized_edited_artifact_scope,
    normalized_reconstruction_mode,
)
from x_editor.runtime.reorder_contract import (
    normalize_reorder_indices,
    reorder_indices_for_speaker_order,
)
from x_editor.tools.catalog.clearer_voice_studio.client import ClearerVoiceClient
from x_editor.tools.catalog.edit_data_pipeline.client import EditDataPipelineClient

ToolGateway = Callable[[str, dict[str, Any]], dict[str, Any]]


def _call_required_tool_gateway(
    gateway: ToolGateway | None,
    tool_id: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if gateway is None:
        raise RuntimeError(
            f"speaker_scoped_edit {tool_id} requires the AudioEditExecutor tool gateway"
        )
    result = gateway(tool_id, dict(arguments))
    if not isinstance(result, dict):
        raise RuntimeError(f"speaker_scoped_edit {tool_id} gateway returned a non-object result")
    return result


MAX_SEGMENT_SEC = 29.5
MIN_TAGGED_EDIT_SEGMENT_SEC = 0.25
PROFANITY_TERMS = {
    "ass",
    "asshole",
    "bastard",
    "bitch",
    "bullshit",
    "crap",
    "cunt",
    "damn",
    "dick",
    "fag",
    "fuck",
    "fucked",
    "fucker",
    "fucking",
    "hell",
    "piss",
    "shit",
    "shitty",
    "slut",
    "whore",
}

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

PERCEPTION_SEGMENT_SELECTOR_PROPERTY_TERMS = [
    "accent",
    "accented",
    "dialect",
    "language",
    "english",
    "japanese",
    "chinese",
    "mandarin",
    "cantonese",
    "korean",
    "french",
    "spanish",
    "german",
    "russian",
    "indian",
    "thai",
    "vietnamese",
    "whisper",
    "shout",
    "singing",
    "rap",
    "baby",
    "infant",
    "toddler",
    "child",
    "kid",
    "girl",
    "boy",
    "woman",
    "man",
    "female",
    "male",
    "elderly",
    "oldest",
    "older",
    "younger",
    "youngest",
    "senior",
    "aged",
    "middle-aged",
    "middle aged",
    "teen",
    "teenage",
    "young",
    "adult",
    "customer",
    "employee",
    "police officer",
    "officer",
    "police",
    "cop",
    "policeman",
    "policewoman",
    "narrator",
    "robber",
    "victim",
    "emotion",
    "emotional",
    "angry",
    "angriest",
    "angrier",
    "sad",
    "saddest",
    "happy",
    "happiest",
    "surprise",
    "surprised",
    "excited",
    "calm",
    "calmest",
    "frightened",
    "scared",
    "depressed",
    "laugh",
    "cry",
    "crying",
    "scream",
    "reverb",
    "echo",
    "noisy",
    "speech with",
    "speaker with",
    "segments with",
    "portions with",
    "marked by",
    "characterized by",
    "respondent",
    "interviewer",
    "answer",
    "answered",
    "question",
    "correct",
    "incorrect",
    "reply",
    "response",
    "asked",
    "volume",
    "loudness",
    "intensity",
    "whistle-level",
    "whistle level",
    "high-pitched",
    "sharp",
    "loud",
    "quiet",
    "soft",
    "underwater",
    "under water",
    "gurgling",
    "bubbling",
    "telephone",
    "radio",
    "filtered",
    "strong wind",
    "windy",
]


@dataclass(frozen=True)
class TimeSpan:
    """A concrete region on an audio timeline."""

    start_sec: float
    end_sec: float
    label: str = ""
    speaker_id: str | None = None
    text: str = ""
    source: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def clipped(self, duration_sec: float) -> "TimeSpan | None":
        start = max(0.0, min(float(self.start_sec), duration_sec))
        end = max(0.0, min(float(self.end_sec), duration_sec))
        if end <= start:
            return None
        return TimeSpan(start, end, self.label, self.speaker_id, self.text, self.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 6),
            "end_sec": round(self.end_sec, 6),
            "duration_sec": round(self.duration_sec, 6),
            "label": self.label,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "source": self.source,
        }


@dataclass(frozen=True)
class SegmentCandidate:
    """One aligned candidate span for perception-model property selection."""

    segment_id: str
    span: TimeSpan
    transcript: str = ""
    description: str = ""
    source: str = ""
    target_id: str | None = None
    target_kind: str | None = None
    target_value: str | None = None
    authoritative_bounds: bool = False
    source_refs: tuple[str, ...] = ()


def execute_artifact_operation(
    *,
    action: Any,
    artifact_paths: dict[str, str],
    state: dict[str, Any],
    base_audio_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Execute one ``artifact_ops`` graph action."""

    input_audio_path = _resolve_artifact_input(action, artifact_paths, base_audio_path)
    output_path = str(Path(output_dir) / f"{action.step_id}.wav")
    operation = _infer_graph_operation(action, state)
    instruction = str(state.get("user_prompt") or action.objective or "")
    return execute_artifact_step(
        operation=operation,
        instruction=instruction,
        input_audio_path=input_audio_path,
        output_path=output_path,
        output_dir=output_dir,
        perception_report=_perception_from_state(state),
        reference_audio_paths=state.get("reference_audio_paths") or [],
        source_text=state.get("source_text"),
        artifact_paths=artifact_paths,
        edited_audio_path=_edited_artifact_path(action, artifact_paths),
        params={
            "action_objective": action.objective,
            "output_artifact_role": action.output_artifact_role,
            "_agentic_mode": (state.get("context") or {}).get("agentic_mode")
            or state.get("agentic_mode"),
        },
    )


def execute_artifact_step(
    *,
    operation: str,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    perception_report: dict[str, Any] | None = None,
    reference_audio_paths: list[str] | None = None,
    source_text: str | None = None,
    artifact_paths: dict[str, str] | None = None,
    edited_audio_path: str | None = None,
    params: dict[str, Any] | None = None,
    tool_gateway: ToolGateway | None = None,
) -> dict[str, Any]:
    """Execute one structured artifact operation.

    ``operation`` is intentionally small and deterministic:
    ``extract_spans``, ``remove_spans``, ``mute_spans``, ``beep_spans``, ``silence``,
    ``concat``, ``reorder_spans``, ``reconstruct`` or ``speaker_scoped_edit``.
    """

    params = dict(params or {})
    agentic_mode = _artifact_agentic_mode(params)
    op = _normalize_operation(operation)
    input_audio = Path(input_audio_path)
    if not input_audio.exists():
        raise FileNotFoundError(f"artifact_ops input audio does not exist: {input_audio}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(output_dir) / "artifact_ops" / out.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    if op == "apply_gain":
        from x_editor.runtime.relative_gain import apply_relative_gain

        result = apply_relative_gain(
            input_audio_path=input_audio_path, output_audio_path=output_path, params=params,
        )
        return _write_manifest(work_dir, {
            "operation": op, "input_audio_path": input_audio_path,
            "output_audio_path": output_path,
            "selected_spans": result["selected_spans"], "result": result,
        })

    if op == "silence":
        duration_sec = _duration_seconds_from_params(params, instruction, default=1.0)
        result = synthesize_silence(str(out), duration_sec, reference_audio_path=str(input_audio))
        manifest = {
            "operation": op,
            "instruction": instruction,
            "input_audio_path": str(input_audio),
            "output_audio_path": str(out),
            "duration_sec": duration_sec,
            "result": result,
        }
        return _write_manifest(work_dir, manifest)

    if op == "concat":
        sources = _concat_sources(
            instruction=instruction,
            input_audio_path=str(input_audio),
            reference_audio_paths=reference_audio_paths or [],
            artifact_paths=artifact_paths or {},
            params=params,
        )
        pause_duration_sec = _pause_duration_seconds(params, instruction)
        result = concat_audios(
            sources,
            str(out),
            crossfade_ms=int(params.get("crossfade_ms", 0)),
            pause_duration_sec=pause_duration_sec,
        )
        manifest = {
            "operation": op,
            "instruction": instruction,
            "input_audio_path": str(input_audio),
            "output_audio_path": str(out),
            "sources": sources,
            "pause_duration_sec": pause_duration_sec,
            "result": result,
        }
        return _write_manifest(work_dir, manifest)

    if op == "reconstruct":
        return _execute_reconstruct(
            instruction=instruction,
            input_audio_path=str(input_audio),
            edited_audio_path=edited_audio_path,
            output_path=str(out),
            output_dir=str(work_dir),
            perception_report=perception_report or {},
            params=params,
        )

    if op == "reorder_spans":
        return _execute_reorder_spans(
            instruction=instruction,
            input_audio_path=str(input_audio),
            output_path=str(out),
            output_dir=str(work_dir),
            perception_report=perception_report or {},
            params=params,
        )

    if op == "speaker_scoped_edit":
        return execute_speaker_scoped_edit(
            instruction=instruction,
            input_audio_path=str(input_audio),
            output_path=str(out),
            output_dir=str(work_dir),
            perception_report=perception_report or {},
            source_text=source_text,
            reference_audio_paths=reference_audio_paths or [],
            params=params,
            tool_gateway=tool_gateway,
        )

    audio = load_audio_segment(input_audio)
    geometry = AudioFrameGeometry.from_segment(audio)
    duration_sec = geometry.duration_sec
    selection_instruction = _speaker_scope_selection_instruction(instruction, params)
    strict_explicit_spans = bool(params.pop("strict_explicit_spans", False))
    explicit_spans = params.get("selected_spans") or params.get("spans")
    has_explicit_spans = isinstance(explicit_spans, list) and bool(explicit_spans)
    if strict_explicit_spans and not has_explicit_spans:
        raise RuntimeError(f"artifact_ops.{op} strict execution requires explicit selected_spans")
    perception = (
        dict(perception_report or {})
        if strict_explicit_spans
        else _ensure_perception(
            perception_report or {},
            audio_path=str(input_audio),
            output_dir=output_dir,
            source_text=source_text,
            need_words=not has_explicit_spans,
            need_speakers=(
                not has_explicit_spans
                and (
                    _instruction_mentions_speaker(instruction)
                    or _instruction_mentions_speaker(selection_instruction)
                )
            ),
            instruction=selection_instruction,
            reference_audio_paths=reference_audio_paths or [],
            agentic_mode=agentic_mode,
        )
    )
    selection_evidence = None
    endpoint_adjustments = []
    selected_spans_source = "selector"
    if isinstance(explicit_spans, list) and explicit_spans:
        for item in explicit_spans:
            if isinstance(item, dict) and (
                item.get("start_sec", item.get("start")) is None
                or item.get("end_sec", item.get("end")) is None
            ):
                raise ValueError("explicit selected_spans require both start and end coordinates")
        spans = _spans_from_dicts(explicit_spans)
        if len(spans) != len(explicit_spans):
            raise ValueError("explicit selected_spans require numeric start/end coordinates")
        for index, span in enumerate(spans):
            geometry.interval(span.start_sec, span.end_sec)
            canonical_end = geometry.canonical_end_sec(span.end_sec)
            if canonical_end != span.end_sec:
                endpoint_adjustments.append(dict(
                    span_index=index, requested_end_sec=span.end_sec,
                    canonical_end_sec=canonical_end, frame_count=geometry.frame_count,
                    sample_rate=geometry.sample_rate, policy="decimal_rounded_eof_v1",
                ))
        spans = _clip_merge_spans(
            spans, duration_sec,
            merge_gap_sec=0.0 if op in {
                "extract_spans", "remove_spans", "mute_spans", "beep_spans"
            } else 0.03,
        )
        selected_spans_source = "params.selected_spans"
        refined_spans = (
            []
            if strict_explicit_spans
            else _refine_planner_spans_with_grounded_local_target(
                instruction,
                perception,
                spans,
                duration_sec,
            )
        )
        if refined_spans:
            spans = refined_spans
            selected_spans_source = "selector_refined_params"
    else:
        spans = _spans_from_speaker_scope_params(params, perception, duration_sec)
    if not spans:
        spans, selection_evidence = select_timeline_spans_with_perception_fallback(
            selection_instruction,
            perception,
            input_audio_path=str(input_audio),
            output_dir=str(work_dir),
            duration_sec=duration_sec,
            action=op,
            agentic_mode=agentic_mode,
        )
    if (
        not spans
        and op in {"remove_spans", "mute_spans"}
        and _wants_audio_silence_spans(selection_instruction, action=op)
    ):
        spans = _audio_silence_spans(str(input_audio), duration_sec=duration_sec, params=params)
        if spans:
            selected_spans_source = "audio_silence_detection"
            selection_evidence = {
                "status": "completed",
                "source": "pydub.detect_silence",
                "reason": "instruction requests timeline removal/muting of silent audio regions",
                "span_count": len(spans),
            }
    if not spans:
        if _should_use_whole_audio_as_reference_voice_fallback(instruction, params):
            spans = [
                TimeSpan(
                    0.0,
                    duration_sec,
                    "whole_reference_voice_fallback",
                    source="reference_voice_fallback",
                )
            ]
        else:
            selector_reason = ""
            if selection_evidence:
                selector_reason = f"; perception segment selector: {selection_evidence.get('reason') or selection_evidence.get('status')}"
            raise RuntimeError(
                f"artifact_ops could not resolve timeline spans for instruction: {instruction}{selector_reason}"
            )

    reference_voice_fallback = any(span.source == "reference_voice_fallback" for span in spans)
    if reference_voice_fallback and op != "extract_spans":
        raise RuntimeError(
            f"artifact_ops could not resolve timeline spans for instruction: {instruction}"
        )

    span_padding_sec = float(params.pop("span_padding_sec", 0.0) or 0.0)
    if span_padding_sec < 0:
        raise ValueError("span_padding_sec must be non-negative")
    if span_padding_sec > 0 and op in {
        "remove_spans",
        "mute_spans",
        "beep_spans",
    }:
        spans = _clip_merge_spans(
            [
                TimeSpan(
                    max(0.0, span.start_sec - span_padding_sec),
                    min(duration_sec, span.end_sec + span_padding_sec),
                    span.label,
                    span.speaker_id,
                    span.text,
                    (f"{span.source}+span_padding_sec" if span.source else "span_padding_sec"),
                )
                for span in spans
            ],
            duration_sec,
            merge_gap_sec=0.0 if has_explicit_spans else 0.03,
        )
        selected_spans_source = f"{selected_spans_source}+span_padding_sec"

    if op == "extract_spans":
        result = extract_spans(
            str(input_audio),
            spans,
            str(out),
            crossfade_ms=_artifact_crossfade_ms(params, instruction, spans, default=10),
        )
    elif op == "remove_spans":
        result = remove_spans(
            str(input_audio),
            spans,
            str(out),
            crossfade_ms=_artifact_crossfade_ms(params, instruction, spans, default=10),
        )
    elif op == "mute_spans":
        result = mute_spans(str(input_audio), spans, str(out))
    elif op == "beep_spans":
        result = beep_spans(
            str(input_audio),
            spans,
            str(out),
            frequency_hz=float(params.get("frequency_hz", 1000.0)),
        )
    else:
        raise ValueError(f"Unsupported artifact operation: {operation}")

    manifest = {
        "operation": op,
        "instruction": instruction,
        "input_audio_path": str(input_audio),
        "output_audio_path": str(out),
        "selected_spans": [span.to_dict() for span in spans],
        "selected_spans_source": "reference_voice_fallback"
        if reference_voice_fallback
        else selected_spans_source,
        "span_padding_sec": span_padding_sec,
        "endpoint_adjustments": endpoint_adjustments,
        "result": result,
        "perception_services": perception.get("services", {}),
    }
    if reference_voice_fallback:
        manifest["reference_voice_fallback"] = {
            "reason": "No timed reference-speaker span was available; using the whole reference audio as VC enrollment.",
            "selection_instruction": params.get("selection_instruction")
            or params.get("selector")
            or instruction,
        }
    if selection_evidence:
        manifest["perception_segment_selection"] = selection_evidence
    return _write_manifest(work_dir, manifest)


def _artifact_crossfade_ms(
    params: dict[str, Any], instruction: str, spans: list[TimeSpan], *, default: int
) -> int:
    if params.get("crossfade_ms") is not None:
        return max(0, int(params.get("crossfade_ms") or 0))
    text = str(instruction or "").lower()
    if len(spans) > 1 and any(
        marker in text
        for marker in (
            "seamlessly join",
            "seamless join",
            "join these",
            "smoothly join",
            "natural transition",
            "without hard cuts",
            "无缝",
            "自然拼接",
        )
    ):
        return 40
    return default


def _refine_planner_spans_with_grounded_local_target(
    instruction: str,
    perception: dict[str, Any],
    planner_spans: list[TimeSpan],
    duration_sec: float,
) -> list[TimeSpan]:
    """Replace planner spans when natural-language local grounding disagrees."""

    if not planner_spans or _explicit_time_spans(instruction):
        return []
    words = _normalized_words(perception.get("word_timestamps") or [])
    if _local_unit_kind_from_instruction(instruction) != "question":
        return []
    if _local_unit_ordinal_from_instruction(instruction) is None:
        return []
    grounded = select_timeline_spans(
        instruction, perception, duration_sec=duration_sec, action="extract_spans"
    )
    if not grounded or _spans_cover_same_targets(planner_spans, grounded):
        return []
    if _planner_spans_already_match_grounded_text(planner_spans, grounded, words):
        return []
    return grounded


def _planner_spans_already_match_grounded_text(
    planner_spans: list[TimeSpan],
    grounded_spans: list[TimeSpan],
    words: list[dict[str, Any]],
) -> bool:
    if not words or not grounded_spans:
        return False
    for planner_span in planner_spans:
        window_text = _normalize_text_for_contains(_words_text_for_span(words, planner_span))
        if not window_text:
            continue
        for grounded in grounded_spans:
            target_text = _normalize_text_for_contains(grounded.text)
            if target_text and (target_text in window_text or window_text in target_text):
                return True
    return False


def _question_spans_from_source_text_words(
    instruction: str,
    perception: dict[str, Any],
    words: list[dict[str, Any]],
    duration_sec: float,
) -> list[TimeSpan]:
    ordinal = _local_unit_ordinal_from_instruction(instruction)
    if ordinal is None or _local_unit_kind_from_instruction(instruction) != "question" or not words:
        return []
    source_text = str(perception.get("source_text") or "").strip()
    if not source_text:
        return []
    questions = [
        sentence
        for sentence, _, _ in _sentence_pieces_from_text(source_text)
        if _text_looks_like_question(sentence)
    ]
    if not questions:
        return []
    if ordinal == -1:
        question = questions[-1]
    elif 1 <= ordinal <= len(questions):
        question = questions[ordinal - 1]
    else:
        return []
    matches = _phrase_occurrences_in_words(question, words)
    if not matches:
        return []
    selected = matches[-1] if ordinal == -1 else matches[0]
    span = TimeSpan(
        float(selected[0]["start_sec"]),
        float(selected[-1]["end_sec"]),
        "question_segment",
        _single_speaker_id(selected),
        _render_words(selected),
        "source_text+word_timestamps_question",
    ).clipped(duration_sec)
    return [span] if span else []


def _spans_cover_same_targets(left: list[TimeSpan], right: list[TimeSpan]) -> bool:
    left_valid = [span for span in left if span.end_sec > span.start_sec]
    right_valid = [span for span in right if span.end_sec > span.start_sec]
    if len(left_valid) != len(right_valid):
        return False
    for left_span, right_span in zip(
        sorted(left_valid, key=lambda span: (span.start_sec, span.end_sec)),
        sorted(right_valid, key=lambda span: (span.start_sec, span.end_sec)),
    ):
        if _span_iou(left_span, right_span) < 0.7:
            return False
    return True


def _span_iou(left: TimeSpan, right: TimeSpan) -> float:
    intersection = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    union = max(left.end_sec, right.end_sec) - min(left.start_sec, right.start_sec)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _should_use_whole_audio_as_reference_voice_fallback(
    instruction: str, params: dict[str, Any]
) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            params.get("selection_instruction"),
            params.get("selector"),
            params.get("target_speaker_description"),
            params.get("speaker_description"),
        ]
        if value
    ).lower()
    wants_reference = _contains(
        evidence,
        [
            "reference audio",
            "reference voice",
            "voice clone reference",
            "clone reference",
            "enrollment",
            "audio2",
            "audio 2",
            "参考音频",
            "参考声音",
        ],
    )
    wants_voice = _contains(
        evidence,
        [
            "voice",
            "speaker",
            "female",
            "male",
            "woman",
            "man",
            "声音",
            "说话人",
            "女声",
            "男声",
        ],
    )
    return wants_reference and wants_voice


def _wants_audio_silence_spans(instruction: str, *, action: str | None = None) -> bool:
    text = str(instruction or "").lower()
    has_silence_target = _contains(
        text,
        [
            "silence",
            "silent segment",
            "silent segments",
            "silent part",
            "silent parts",
            "静音",
            "无声",
        ],
    )
    if not has_silence_target:
        return False
    if action in {"remove_spans", "mute_spans"}:
        return True
    return _contains(
        text,
        ["remove", "delete", "cut", "discard", "mute", "去掉", "删除", "移除", "静音"],
    )


def _audio_silence_spans(
    input_audio_path: str, *, duration_sec: float, params: dict[str, Any]
) -> list[TimeSpan]:
    audio = load_audio_segment(input_audio_path)
    min_silence_len = int(params.get("min_silence_ms") or params.get("min_silence_len") or 180)
    if min_silence_len < 40:
        min_silence_len = 40
    if params.get("silence_thresh_dbfs") is not None:
        silence_thresh = float(params["silence_thresh_dbfs"])
    else:
        dbfs = audio.dBFS
        silence_thresh = -45.0 if dbfs == float("-inf") else min(-35.0, float(dbfs) - 12.0)
    keep_safety_ms = int(params.get("silence_edge_keep_ms") or 20)
    raw_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10,
    )
    spans = []
    for start_ms, end_ms in raw_ranges:
        start = max(0, int(start_ms) + keep_safety_ms)
        end = min(len(audio), int(end_ms) - keep_safety_ms)
        if end - start < max(40, min_silence_len // 3):
            continue
        spans.append(
            TimeSpan(
                start / 1000.0,
                end / 1000.0,
                "silence",
                None,
                "",
                "audio_silence_detection",
            )
        )
    return _clip_merge_spans(spans, duration_sec, merge_gap_sec=0.05)


def select_timeline_spans(
    instruction: str,
    perception: dict[str, Any],
    *,
    duration_sec: float,
    action: str = "extract_spans",
) -> list[TimeSpan]:
    """Resolve a natural-language target into concrete timeline spans."""

    text = instruction.lower()
    words = _normalized_words(perception.get("word_timestamps") or [])
    speaker_segments = _normalized_speaker_segments(perception)

    explicit = _explicit_time_spans(instruction)
    if explicit:
        return _clip_merge_spans(explicit, duration_sec)

    if _wants_profanity_mask(text):
        profanity = _profanity_spans(words, speaker_segments, duration_sec)
        if profanity:
            return profanity

    if _contains(text, ["pause", "silence", "停顿", "静音"]) and _contains(
        text, ["remove", "delete", "去掉", "删除", "移除"]
    ):
        gaps = _pause_gap_spans(words, duration_sec=duration_sec)
        if gaps:
            return gaps

    if _wants_filler_word_removal(text):
        filler_spans = _filler_word_spans(words, duration_sec)
        if filler_spans:
            return filler_spans

    question_word_spans = _question_spans_from_source_text_words(
        instruction, perception, words, duration_sec
    )
    if question_word_spans:
        return question_word_spans

    phrase_spans = _phrase_spans(instruction, words)
    if phrase_spans:
        return _clip_merge_spans(phrase_spans, duration_sec)
    phrase_spans = _phrase_spans_from_segments(
        instruction,
        speaker_segments,
        duration_sec,
        prefer_subspan=action == "extract_spans",
    )
    if phrase_spans:
        return _clip_merge_spans(phrase_spans, duration_sec)

    sentence_spans = _sentence_spans_from_instruction(instruction, words, duration_sec)
    if sentence_spans:
        return sentence_spans
    sentence_spans = _sentence_spans_from_segments(instruction, speaker_segments, duration_sec)
    if sentence_spans:
        return sentence_spans

    semantic_spans = _semantic_segment_spans(instruction, perception)
    if semantic_spans:
        return _clip_merge_spans(semantic_spans, duration_sec)

    target_speakers = resolve_target_speakers(instruction, perception)
    if not target_speakers:
        local_spans = _speaker_local_ordinal_spans(
            instruction,
            speaker_segments,
            words,
            duration_sec,
        )
        if local_spans:
            return _clip_merge_spans(local_spans, duration_sec)
    if target_speakers:
        local_spans = _speaker_local_ordinal_spans(
            instruction,
            speaker_segments,
            words,
            duration_sec,
            target_speakers=target_speakers,
        )
        if local_spans:
            return _clip_merge_spans(local_spans, duration_sec)
        spans = [seg for seg in speaker_segments if seg.speaker_id in target_speakers]
        if words:
            spans = _snap_spans_to_word_boundaries(spans, words, duration_sec)
        if not (
            _should_defer_to_perception_segment_selector(instruction, action)
            and (
                _is_broad_single_speaker_selection(instruction, perception, target_speakers)
                or _is_broad_semantic_speaker_property_selection(
                    instruction, perception, target_speakers
                )
            )
        ):
            return _clip_merge_spans(spans, duration_sec)

    if _should_defer_to_perception_segment_selector(instruction, action):
        return []

    if _contains(
        text,
        [
            "all speech",
            "speech portions",
            "speech portion",
            "speech segment",
            "speech segments",
            "speech region",
            "speech regions",
            "spoken portions",
            "spoken portion",
            "spoken narration segments",
            "spoken narration segment",
            "spoken narration",
            "spoken content",
            "spoken voice",
            "narration segments",
            "narration segment",
            "narration",
            "speaking parts",
            "speaking voice",
            "voice from the audio",
            "pure speech",
            "speech content",
            "所有语音",
            "全部语音",
            "人声部分",
        ],
    ):
        if speaker_segments:
            spans = speaker_segments
            if words:
                spans = _snap_spans_to_word_boundaries(spans, words, duration_sec)
            return _clip_merge_spans(spans, duration_sec)
        if words:
            return _word_groups_as_spans(words, duration_sec)

    if _contains(
        text,
        [
            "first segment",
            "second segment",
            "last segment",
            "第一个片段",
            "第二个片段",
            "最后一个片段",
        ],
    ):
        spans = _segment_spans_by_order(instruction, speaker_segments, words, duration_sec)
        if spans:
            return spans

    if len(speaker_segments) == 1 and _contains(
        text, ["speaker", "voice", "speech", "说话人", "人声"]
    ):
        spans = speaker_segments
        if words:
            spans = _snap_spans_to_word_boundaries(spans, words, duration_sec)
        return _clip_merge_spans(spans, duration_sec)

    if (
        action == "extract_spans"
        and words
        and _contains(text, ["extract", "isolate", "keep only", "提取", "只保留", "仅保留"])
        and not _should_use_perception_segment_selector(instruction, action)
    ):
        return _word_groups_as_spans(words, duration_sec)

    return []


def requires_perception_segment_selector(instruction: str, action: str = "extract_spans") -> bool:
    """Return true when broad fallbacks are unsafe without model confirmation."""

    return _should_defer_to_perception_segment_selector(instruction, action)


def select_timeline_spans_with_perception_fallback(
    instruction: str,
    perception: dict[str, Any],
    *,
    input_audio_path: str,
    output_dir: str,
    duration_sec: float,
    action: str = "extract_spans",
    agentic_mode: str | None = None,
    selector_client: OpenAICompatibleChatClient | None = None,
    input_audio_ref: str | None = None,
    reference_audio_paths: list[str] | None = None,
    reference_audio_refs: list[str] | None = None,
    semantic_targets: list[dict[str, str]] | None = None,
    force_model_review_on_empty: bool = False,
    deterministic_targets_complete: bool | None = None,
    aligned_candidate_segments: list[dict[str, Any]] | None = None,
    include_default_candidates: bool = True,
    authoritative_candidate_bounds: bool = False,
) -> tuple[list[TimeSpan], dict[str, Any] | None]:
    """Resolve target spans, escalating property/role/entity selection to audio perception.

    The deterministic resolver handles exact timestamps, word/phrase matches,
    speaker ids, and already-grounded profiles first.  If that evidence is not
    enough for a property-scoped instruction, this helper exports aligned
    candidate clips and asks the perception fallback model to judge each clip.
    """

    spans = select_timeline_spans(
        instruction,
        perception,
        duration_sec=duration_sec,
        action=action,
    )
    if (
        spans
        and not reference_audio_paths
        and deterministic_targets_complete is not False
    ):
        return _clip_merge_spans(spans, duration_sec), None
    spans, selection_evidence = _select_spans_with_perception_model(
        instruction=instruction,
        perception=perception,
        input_audio_path=input_audio_path,
        output_dir=output_dir,
        duration_sec=duration_sec,
        action=action,
        agentic_mode=agentic_mode,
        selector_client=selector_client,
        input_audio_ref=input_audio_ref,
        reference_audio_paths=reference_audio_paths,
        reference_audio_refs=reference_audio_refs,
        semantic_targets=semantic_targets,
        force_model_review=(
            force_model_review_on_empty or deterministic_targets_complete is False
        ),
        aligned_candidate_segments=aligned_candidate_segments,
        include_default_candidates=include_default_candidates,
        authoritative_candidate_bounds=authoritative_candidate_bounds,
    )
    if authoritative_candidate_bounds:
        return _clip_spans_preserving_boundaries(spans, duration_sec), selection_evidence
    return _clip_merge_spans(spans, duration_sec), selection_evidence


def resolve_target_speakers(instruction: str, perception: dict[str, Any]) -> list[str]:
    """Resolve speaker ids from ordinal or profile hints in an instruction."""

    segments = _normalized_speaker_segments(perception)
    order = _speaker_order(segments, perception)
    if not order:
        return []

    text = instruction.lower()
    explicit_ids = _explicit_speaker_ids(instruction, order)
    if explicit_ids:
        return explicit_ids

    ordinal = _first_speaker_ordinal(text)
    if ordinal and 1 <= ordinal <= len(order):
        return [order[ordinal - 1]]

    profiles = _speaker_profiles(perception)
    target_names = _target_name_mentions(instruction)
    if target_names:
        matched = [
            speaker_id
            for speaker_id in order
            if any(
                _speaker_profile_matches_name(profiles.get(speaker_id, ""), name)
                for name in target_names
            )
        ]
        if matched:
            return matched

    attr_kind = _target_attribute_kind(text)
    if attr_kind:
        explicit_matches = {
            speaker_id: _explicit_speaker_attribute_matches(perception, speaker_id, attr_kind)
            for speaker_id in order
        }
        if any(value is True for value in explicit_matches.values()):
            return [speaker_id for speaker_id, matched in explicit_matches.items() if matched]
        descriptive_profiles = _speaker_descriptive_profiles(perception)
        descriptive_matches = [
            speaker_id
            for speaker_id in order
            if _speaker_profile_matches(descriptive_profiles.get(speaker_id, ""), attr_kind)
        ]
        if descriptive_matches:
            return descriptive_matches
        matched = [
            speaker_id
            for speaker_id in order
            if _speaker_profile_matches(profiles.get(speaker_id, ""), attr_kind)
        ]
        if matched:
            return matched
        if len(order) == 1:
            return order
        return []

    if len(order) == 1 and _contains(text, ["speaker", "voice", "speech", "说话人", "人声"]):
        return order

    targeted_by_profile = _speaker_ids_from_generic_target_profile(
        instruction,
        perception,
        order=order,
        profiles=profiles,
    )
    if targeted_by_profile:
        return targeted_by_profile

    return []


def _speaker_ids_match_profile_name(
    instruction: str,
    speaker_ids: list[str],
    perception: dict[str, Any],
) -> bool:
    names = _target_name_mentions(instruction)
    if not names or not speaker_ids:
        return False
    profiles = _speaker_profiles(perception)
    descriptive_profiles = _speaker_descriptive_profiles(perception)
    matched: set[str] = set()
    for speaker_id in speaker_ids:
        profile = " ".join(
            value
            for value in [
                profiles.get(str(speaker_id), ""),
                descriptive_profiles.get(str(speaker_id), ""),
            ]
            if value
        )
        if any(_speaker_profile_matches_name(profile, name) for name in names):
            matched.add(str(speaker_id))
    return bool(matched) and matched == {str(speaker_id) for speaker_id in speaker_ids}


def _speaker_ids_from_generic_target_profile(
    instruction: str,
    perception: dict[str, Any],
    *,
    order: list[str] | None = None,
    profiles: dict[str, str] | None = None,
) -> list[str]:
    if not _is_generic_person_voice_selector(instruction):
        return []
    speaker_order = order or _speaker_order(_normalized_speaker_segments(perception), perception)
    if not speaker_order:
        return []
    profile_map = profiles or _speaker_profiles(perception)
    descriptive_profiles = _speaker_descriptive_profiles(perception)
    scored: list[tuple[int, str]] = []
    for speaker_id in speaker_order:
        profile = " ".join(
            value
            for value in [
                profile_map.get(str(speaker_id), ""),
                descriptive_profiles.get(str(speaker_id), ""),
            ]
            if value
        )
        score = _generic_target_profile_score(profile)
        if score > 0:
            scored.append((score, str(speaker_id)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_speaker = scored[0]
    if best_score < 2:
        return []
    if len(scored) > 1 and scored[1][0] == best_score:
        return []
    return [best_speaker]


def _is_generic_person_voice_selector(instruction: str) -> bool:
    text = str(instruction or "").lower()
    if not text:
        return False
    if (
        _target_name_mentions(instruction)
        or _first_speaker_ordinal(text)
        or _target_attribute_kind(text)
    ):
        return False
    return (
        "person" in text and _contains(text, ["voice", "speech", "speaking", "speaker"])
    ) or _contains(
        text,
        [
            "the speaker's voice",
            "speaker's voice",
            "lead speaker",
            "foreground speaker",
            "main speaker",
            "primary speaker",
            "human voice",
            "voice of the person",
            "person's voice",
            "person voice",
            "这个人的声音",
            "那个人的声音",
            "人的声音",
            "主说话人",
            "前景说话人",
        ],
    )


def _generic_target_profile_score(profile: str) -> int:
    text = str(profile or "").lower()
    if not text:
        return 0
    score = 0
    if re.search(r"\buser request\b.{0,120}\b(?:target|targets|targeted|targeting)\b", text):
        score = max(score, 5)
    if re.search(
        r"\b(?:request|instruction)\b.{0,100}\b(?:likely|probably|appears to)\s+targets?\b",
        text,
    ):
        score = max(score, 4)
    if _contains(text, ["target speaker", "requested speaker", "speaker requested by the user"]):
        score = max(score, 4)
    if _contains(text, ["foreground speaker", "lead speaker", "main speaker", "primary speaker"]):
        score = max(score, 3)
    if _contains(text, ["foreground voice", "lead voice", "main voice", "primary voice"]):
        score = max(score, 2)
    if _contains(text, ["background speaker", "background voice", "other speaker"]):
        score -= 1
    return score


def _filter_speaker_ids_by_selection_audio_scope(
    speaker_ids: list[str],
    *,
    selection_instruction: str,
    grounding_instruction: str,
    perception: dict[str, Any],
) -> list[str]:
    if not speaker_ids:
        return speaker_ids
    attr_kind = _target_attribute_kind(selection_instruction.lower())
    if not attr_kind:
        return speaker_ids
    scope = _attribute_audio_scope(grounding_instruction, attr_kind)
    if not scope:
        return speaker_ids
    scoped = [
        speaker_id
        for speaker_id in speaker_ids
        if _speaker_audio_scope_matches(perception, speaker_id, scope)
    ]
    return scoped or speaker_ids


def _attribute_audio_scope(instruction: str, attr_kind: str) -> str | None:
    text = re.sub(r"\baudio\s*([12])\b", r"audio\1", str(instruction or "").lower())
    attr_pattern = _attribute_scope_pattern(attr_kind)
    if not attr_pattern:
        return None
    patterns = [
        rf"{attr_pattern}.{{0,80}}?\b(?:speech|voice|speaker|timbre|words|utterance|utterances)?\b.{{0,40}}?\b(?:in|from|of)\s+audio([12])\b",
        rf"\baudio([12])\b.{{0,80}}?{attr_pattern}.{{0,40}}?\b(?:speech|voice|speaker|timbre|words|utterance|utterances)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"audio{match.group(1)}"
    return None


def _attribute_scope_pattern(attr_kind: str) -> str | None:
    patterns = {
        "female": r"\b(?:female|woman|girl|women|girls)\b",
        "male": r"\b(?:male|man|boy|men|boys)\b",
        "little_girl": r"\b(?:little girl|girl|female child)\b",
        "little_boy": r"\b(?:little boy|boy|male child)\b",
        "baby": r"\b(?:baby|infant|toddler)\b",
        "child": r"\b(?:child|kid|children|baby|infant|toddler|girl|boy)\b",
        "adult": r"\b(?:adult|grown[- ]?up)\b",
    }
    return patterns.get(attr_kind)


def _speaker_audio_scope_matches(perception: dict[str, Any], speaker_id: str, scope: str) -> bool:
    scope = scope.lower().strip()
    if scope not in {"audio1", "audio2"}:
        return False
    positive = (
        [
            "audio1",
            "audio 1",
            "input1",
            "input 1",
            "base audio",
            "source audio",
            "current audio",
        ]
        if scope == "audio1"
        else [
            "audio2",
            "audio 2",
            "input2",
            "input 2",
            "reference audio",
            "second audio",
        ]
    )
    negative = (
        ["audio2", "audio 2", "input2", "input 2", "reference audio", "second audio"]
        if scope == "audio1"
        else [
            "audio1",
            "audio 1",
            "input1",
            "input 1",
            "base audio",
            "source audio",
            "current audio",
        ]
    )
    saw_negative = False
    for _collection_name, raw, _weight in _speaker_profile_collections(perception):
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            item_speaker = (
                item.get("speaker_id") or item.get("speaker") or item.get("label") or item.get("id")
            )
            if str(item_speaker) != str(speaker_id):
                continue
            values = [
                item.get("description"),
                item.get("source"),
                item.get("audio_id"),
                item.get("audio"),
                item.get("input_audio"),
                item.get("role"),
            ]
            text = " ".join(str(value).lower() for value in values if value is not None)
            if any(marker in text for marker in positive):
                return True
            if any(marker in text for marker in negative):
                saw_negative = True
    return False if saw_negative else False


def extract_spans(
    input_audio_path: str,
    spans: list[TimeSpan],
    output_path: str,
    *,
    crossfade_ms: int = 10,
) -> dict[str, Any]:
    audio = load_audio_segment(input_audio_path)
    geometry = AudioFrameGeometry.from_segment(audio)
    for span in spans:
        geometry.interval(span.start_sec, span.end_sec)
    selected = _clip_merge_spans(spans, geometry.duration_sec, merge_gap_sec=0.0)
    if not selected:
        raise RuntimeError("extract_spans requires at least one valid span")
    output = AudioSegment.empty()
    for span in selected:
        piece = slice_audio_frames(audio, span.start_sec, span.end_sec)
        if not output.raw_data or crossfade_ms <= 0:
            output += piece
        else:
            output = append_audio_frames(output, piece, crossfade_ms=crossfade_ms)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    return {
        "output_audio_path": output_path,
        "operation": "extract_spans",
        "span_count": len(selected),
        "selected_duration_sec": round(sum(span.duration_sec for span in selected), 6),
        "output_duration_sec": AudioFrameGeometry.from_segment(output).duration_sec,
        "crossfade_ms": crossfade_ms,
    }


def remove_spans(
    input_audio_path: str,
    spans: list[TimeSpan],
    output_path: str,
    *,
    crossfade_ms: int = 10,
) -> dict[str, Any]:
    audio = load_audio_segment(input_audio_path)
    geometry = AudioFrameGeometry.from_segment(audio)
    for span in spans:
        geometry.interval(span.start_sec, span.end_sec)
    selected = _clip_merge_spans(spans, geometry.duration_sec, merge_gap_sec=0.0)
    if not selected:
        raise RuntimeError("remove_spans requires at least one valid span")
    pieces: list[AudioSegment] = []
    cursor_frame = 0
    for span in selected:
        first, last = geometry.interval(span.start_sec, span.end_sec)
        if first > cursor_frame:
            pieces.append(audio.get_sample_slice(start_sample=cursor_frame, end_sample=first))
        cursor_frame = max(cursor_frame, last)
    if cursor_frame < geometry.frame_count:
        pieces.append(audio.get_sample_slice(start_sample=cursor_frame))

    output = audio.get_sample_slice(end_sample=0)
    for piece in pieces:
        if not output.raw_data or crossfade_ms <= 0:
            output += piece
        else:
            output = append_audio_frames(output, piece, crossfade_ms=crossfade_ms)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    return {
        "output_audio_path": output_path,
        "operation": "remove_spans",
        "span_count": len(selected),
        "removed_duration_sec": round(sum(span.duration_sec for span in selected), 6),
        "output_duration_sec": AudioFrameGeometry.from_segment(output).duration_sec,
        "crossfade_ms": crossfade_ms,
    }


def mute_spans(input_audio_path: str, spans: list[TimeSpan], output_path: str) -> dict[str, Any]:
    audio = load_audio_segment(input_audio_path)
    geometry = AudioFrameGeometry.from_segment(audio)
    for span in spans:
        geometry.interval(span.start_sec, span.end_sec)
    selected = _clip_merge_spans(spans, geometry.duration_sec, merge_gap_sec=0.0)
    if not selected:
        raise RuntimeError("mute_spans requires at least one valid span")
    output = audio
    for span in selected:
        first, last = geometry.interval(span.start_sec, span.end_sec)
        silence = audio._spawn(bytes((last - first) * audio.frame_width))
        output = replace_audio_frames(
            output, silence, span.start_sec, span.end_sec
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    return {
        "output_audio_path": output_path,
        "operation": "mute_spans",
        "span_count": len(selected),
        "timeline_duration_preserved": AudioFrameGeometry.from_segment(output) == geometry,
        "output_duration_sec": AudioFrameGeometry.from_segment(output).duration_sec,
    }


def beep_spans(
    input_audio_path: str,
    spans: list[TimeSpan],
    output_path: str,
    *,
    frequency_hz: float = 1000.0,
) -> dict[str, Any]:
    audio = load_audio_segment(input_audio_path)
    geometry = AudioFrameGeometry.from_segment(audio)
    for span in spans:
        geometry.interval(span.start_sec, span.end_sec)
    selected = _clip_merge_spans(spans, geometry.duration_sec, merge_gap_sec=0.0)
    if not selected:
        raise RuntimeError("beep_spans requires at least one valid span")
    output = audio
    for span in selected:
        first, last = geometry.interval(span.start_sec, span.end_sec)
        duration_ms = (last - first) * 1000 / geometry.sample_rate
        beep = (
            Sine(frequency_hz)
            .to_audio_segment(duration=duration_ms)
            .apply_gain(-6.0)
            .set_frame_rate(audio.frame_rate)
            .set_channels(audio.channels)
            .set_sample_width(audio.sample_width)
        )
        fade_ms = min(5, duration_ms // 2)
        # A sub-two-millisecond tone has no full-millisecond fade. Pydub's
        # zero-duration fade is undefined, so retain its generated samples.
        if fade_ms > 0:
            beep = beep.fade_in(fade_ms).fade_out(fade_ms)
        output = replace_audio_frames(
            output, fit_audio_frames(beep, last - first), span.start_sec, span.end_sec
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    return {
        "output_audio_path": output_path,
        "operation": "beep_spans",
        "span_count": len(selected),
        "timeline_duration_preserved": AudioFrameGeometry.from_segment(output) == geometry,
        "frequency_hz": frequency_hz,
        "output_duration_sec": AudioFrameGeometry.from_segment(output).duration_sec,
    }


def synthesize_silence(
    output_path: str,
    duration_sec: float,
    *,
    reference_audio_path: str | None = None,
) -> dict[str, Any]:
    duration_ms = max(0, int(round(float(duration_sec) * 1000)))
    frame_rate = 16000
    channels = 1
    sample_width = 2
    if reference_audio_path and Path(reference_audio_path).exists():
        reference = load_audio_segment(reference_audio_path)
        frame_rate = reference.frame_rate
        channels = reference.channels
        sample_width = reference.sample_width
    silence = (
        AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)
        .set_channels(channels)
        .set_sample_width(sample_width)
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    silence.export(output_path, format="wav")
    return {
        "output_audio_path": output_path,
        "operation": "silence",
        "duration_sec": round(duration_ms / 1000.0, 6),
    }


def concat_audios(
    audio_paths: list[str],
    output_path: str,
    *,
    crossfade_ms: int = 0,
    pause_duration_sec: float = 0.0,
) -> dict[str, Any]:
    if not audio_paths:
        raise RuntimeError("concat requires at least one audio path")
    output = AudioSegment.empty()
    reference_format: AudioSegment | None = None
    sources = []
    pause_ms = max(0, int(round(float(pause_duration_sec or 0.0) * 1000)))
    inserted_pause_count = 0
    for index, path_text in enumerate(audio_paths):
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"concat source audio does not exist: {path}")
        audio = load_audio_segment(path)
        if reference_format is None:
            reference_format = audio
        else:
            audio = _match_audio_format(audio, reference_format)
        sources.append({"path": str(path), "duration_sec": round(len(audio) / 1000.0, 6)})
        if index > 0 and pause_ms > 0:
            pause = (
                AudioSegment.silent(duration=pause_ms, frame_rate=audio.frame_rate)
                .set_channels(audio.channels)
                .set_sample_width(audio.sample_width)
            )
            output += pause
            inserted_pause_count += 1
        if len(output) == 0 or crossfade_ms <= 0:
            output += audio
        else:
            output = output.append(audio, crossfade=min(crossfade_ms, len(output), len(audio)))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    return {
        "output_audio_path": output_path,
        "operation": "concat",
        "source_count": len(audio_paths),
        "sources": sources,
        "pause_duration_sec": round(pause_ms / 1000.0, 6),
        "inserted_pause_count": inserted_pause_count,
        "output_duration_sec": round(len(output) / 1000.0, 6),
    }


def execute_speaker_scoped_edit(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    perception_report: dict[str, Any],
    source_text: str | None,
    reference_audio_paths: list[str] | None = None,
    params: dict[str, Any] | None = None,
    tool_gateway: ToolGateway | None = None,
) -> dict[str, Any]:
    """Edit selected speaker turns via edit-data-pipeline and stitch them back."""

    params = dict(params or {})
    grounding_instruction = _speaker_scope_grounding_instruction(instruction, params)
    input_audio = Path(input_audio_path)
    original = load_audio_segment(input_audio)
    duration_sec = len(original) / 1000.0
    perception = _ensure_perception(
        perception_report,
        audio_path=str(input_audio),
        output_dir=output_dir,
        source_text=source_text,
        need_words=True,
        need_speakers=True,
        instruction=grounding_instruction,
        reference_audio_paths=reference_audio_paths or [],
    )
    words = _normalized_words(perception.get("word_timestamps") or [])
    selection_instruction = _speaker_scope_selection_instruction(grounding_instruction, params)
    selection_evidence: dict[str, Any] | None = None
    selector_required = requires_perception_segment_selector(
        selection_instruction, "speaker_scoped_edit"
    )
    explicit_speaker_selector = _has_explicit_speaker_selector(params)
    target_speakers = _target_speakers_from_params(params, perception)
    broad_voice_all_speakers = _speaker_scope_defaults_to_all_speakers_for_broad_voice_edit(
        selection_instruction,
        grounding_instruction,
        params,
    )
    all_speakers_requested = (
        _speaker_scope_requests_all_speakers(selection_instruction, grounding_instruction)
        or broad_voice_all_speakers
    )
    if all_speakers_requested and not explicit_speaker_selector:
        requested = _speaker_order(_normalized_speaker_segments(perception), perception)
        if requested:
            target_speakers = requested
            selector_required = False
    if not target_speakers:
        target_speakers = resolve_target_speakers(selection_instruction, perception)
        target_speakers = _filter_speaker_ids_by_selection_audio_scope(
            target_speakers,
            selection_instruction=selection_instruction,
            grounding_instruction=grounding_instruction,
            perception=perception,
        )
    if not target_speakers:
        target_speakers = resolve_target_speakers(grounding_instruction, perception)
    if target_speakers and (
        _speaker_ids_match_profile_name(selection_instruction, target_speakers, perception)
        or _speaker_ids_match_profile_name(grounding_instruction, target_speakers, perception)
    ):
        selector_required = False
    if not target_speakers and all_speakers_requested:
        target_speakers = _speaker_order(_normalized_speaker_segments(perception), perception)
        if target_speakers:
            selector_required = False
    if not target_speakers and _speaker_scope_defaults_to_all_speakers_for_broad_voice_edit(
        selection_instruction,
        grounding_instruction,
        params,
    ):
        target_speakers = _speaker_order(_normalized_speaker_segments(perception), perception)
        if target_speakers:
            selector_required = False
    if explicit_speaker_selector and target_speakers:
        selector_required = False
    if (
        target_speakers
        and isinstance(params.get("selector"), str)
        and str(params.get("selector") or "").strip()
    ):
        selector_required = False
    spans = _filter_spans_by_target_speakers(
        _spans_from_speaker_scope_params(params, perception, duration_sec),
        target_speakers,
    )
    if not spans:
        spans = _speaker_scoped_local_target_spans(
            selection_instruction,
            perception,
            duration_sec,
            target_speakers=target_speakers,
        )
    if not spans and selection_instruction != grounding_instruction:
        spans = _speaker_scoped_local_target_spans(
            grounding_instruction,
            perception,
            duration_sec,
            target_speakers=target_speakers,
        )
    if not spans:
        spans = _filter_spans_by_target_speakers(
            select_timeline_spans(
                selection_instruction,
                perception,
                duration_sec=duration_sec,
                action="speaker_scoped_edit",
            ),
            target_speakers,
        )
    if not spans and selection_instruction != grounding_instruction:
        spans = _filter_spans_by_target_speakers(
            select_timeline_spans(
                grounding_instruction,
                perception,
                duration_sec=duration_sec,
                action="speaker_scoped_edit",
            ),
            target_speakers,
        )
    if (
        not spans
        and not target_speakers
        and _speaker_scope_defaults_to_all_speakers_for_style(
            selection_instruction,
            grounding_instruction,
            params,
        )
    ):
        target_speakers = _speaker_order(_normalized_speaker_segments(perception), perception)
        if target_speakers:
            selector_required = False
    if not spans:
        spans, selection_evidence = select_timeline_spans_with_perception_fallback(
            selection_instruction,
            perception,
            input_audio_path=str(input_audio),
            output_dir=output_dir,
            duration_sec=duration_sec,
            action="speaker_scoped_edit",
        )
        spans = _filter_spans_by_target_speakers(spans, target_speakers)
    if not target_speakers and spans:
        target_speakers = _speaker_ids_from_spans(spans) or ["semantic_timeline_span"]
    if not target_speakers:
        raise RuntimeError(
            "speaker_scoped_edit requires a resolvable target speaker or semantic target spans. "
            "Use ordinal wording such as 'first speaker', provide speaker profile metadata, "
            "or provide/select semantic spans such as internal monologues."
        )

    if not spans and selector_required:
        selector_reason = "no selected spans"
        if selection_evidence:
            selector_reason = str(
                selection_evidence.get("reason")
                or selection_evidence.get("status")
                or selector_reason
            )
        raise RuntimeError(
            "speaker_scoped_edit could not confirm target spans with the perception segment selector: "
            f"{selector_reason}"
        )

    if not spans:
        spans = _speaker_scoped_local_target_spans(
            grounding_instruction,
            perception,
            duration_sec,
            target_speakers=target_speakers,
        )
    if not spans:
        spans = [
            seg
            for seg in _normalized_speaker_segments(perception)
            if seg.speaker_id in target_speakers
        ]
    spans = _enrich_spans_from_overlapping_speaker_segments(
        spans,
        perception,
        target_speakers=target_speakers,
    )
    if words:
        spans = _snap_spans_to_word_boundaries(spans, words, duration_sec)
    spans = _split_long_spans(
        _clip_merge_spans(spans, duration_sec), words, max_segment_sec=MAX_SEGMENT_SEC
    )
    if not spans:
        raise RuntimeError(
            "speaker_scoped_edit resolved target speaker but found no editable spans"
        )

    direct_operation, direct_params = _speaker_scoped_direct_operation(
        params,
        grounding_instruction,
        reference_audio_paths or [],
    )
    operations = (
        [direct_operation]
        if direct_operation
        else _speaker_scoped_operations(
            grounding_instruction, source_text or perception.get("source_text") or ""
        )
    )
    operations = _prefer_segment_speech_rate_for_scoped_rate_edits(
        operations, grounding_instruction, direct_params
    )
    if not operations:
        raise RuntimeError(
            "speaker_scoped_edit found no supported local edit operation in the instruction"
        )
    spans = _refine_phrase_segment_spans_for_tagged_edit(
        spans,
        selection_instruction=selection_instruction,
        grounding_instruction=grounding_instruction,
        operations=operations,
        duration_sec=duration_sec,
    )
    if not spans:
        raise RuntimeError(
            "speaker_scoped_edit resolved target speaker but found no editable spans"
        )
    event_removal_spans = _speaker_scoped_event_removal_spans(params, duration_sec)

    segments_dir = Path(output_dir) / "segments"
    edited_dir = Path(output_dir) / "edited"
    service_dir = Path(output_dir) / "service_artifacts"
    for directory in [segments_dir, edited_dir, service_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    edited_timeline = original
    client = EditDataPipelineClient() if tool_gateway is None else None
    clearer_voice_client: ClearerVoiceClient | None = None
    no_ref_voice_references: dict[str, dict[str, Any]] = {}
    segment_records: list[dict[str, Any]] = []
    event_removal_records: list[dict[str, Any]] = []
    preserve_timeline = bool(params.get("preserve_timeline", True))
    if not preserve_timeline:
        raise RuntimeError(
            "speaker_scoped_edit currently requires preserve_timeline=True; "
            "duration-shifting multi-span reconstruction needs audio_assembly."
        )

    for index, span in enumerate(spans):
        segment_input = segments_dir / f"speaker_span_{index:03d}_input.wav"
        current_path = edited_dir / f"speaker_span_{index:03d}_stage0.wav"
        _slice_audio(original, span.start_sec, span.end_sec).export(segment_input, format="wav")
        shutil.copy2(segment_input, current_path)
        span_text = span.text or _words_text_for_span(words, span)
        current_text = (
            span_text
            or _direct_source_text_for_span(span=span, all_spans=spans, direct_params=direct_params)
            or _whole_span_source_text(
                span, duration_sec, source_text or perception.get("source_text")
            )
        )
        if not current_text and any(
            op
            not in {
                "volume_normalize",
                "seed_vc",
                "speech_enhancement",
                "audio_effects",
                "no_reference_voice_conversion",
                "mute_spans",
            }
            for op in operations
        ):
            if _should_mute_nonverbal_span(span, grounding_instruction, perception):
                current_path = _write_silence_like(
                    current_path,
                    edited_dir / f"speaker_span_{index:03d}_stage1_mute_nonverbal.wav",
                    duration_sec=span.duration_sec,
                )
                replacement = load_audio_segment(current_path)
                fitted_path = edited_dir / f"speaker_span_{index:03d}_fitted.wav"
                replacement = _fit_audio_duration(replacement, int(round(span.duration_sec * 1000)))
                replacement.export(fitted_path, format="wav")
                edited_timeline = _replace_range(
                    edited_timeline, replacement, span.start_sec, span.end_sec
                )
                segment_records.append(
                    {
                        **span.to_dict(),
                        "input_segment_path": str(segment_input),
                        "edited_segment_path": str(current_path),
                        "fitted_segment_path": str(fitted_path),
                        "stage_records": [
                            {
                                "operation": "mute_nonverbal",
                                "reason": "requested non-verbal laughter removal and span has no transcript",
                                "arguments": {"duration_sec": round(span.duration_sec, 6)},
                                "output": {"output_audio_path": str(current_path)},
                            }
                        ],
                    }
                )
                continue
            if _should_skip_untranscribed_nonverbal_span(span, perception):
                segment_records.append(
                    {
                        **span.to_dict(),
                        "input_segment_path": str(segment_input),
                        "edited_segment_path": str(current_path),
                        "fitted_segment_path": str(current_path),
                        "stage_records": [
                            {
                                "operation": "skip_untranscribed_nonverbal",
                                "reason": "span has nonverbal/crying/screaming evidence but no transcript for tagged edit",
                                "arguments": {"duration_sec": round(span.duration_sec, 6)},
                                "output": {"output_audio_path": str(current_path)},
                            }
                        ],
                    }
                )
                continue
            if set(operations) <= {"text_edit"}:
                segment_records.append(
                    {
                        **span.to_dict(),
                        "input_segment_path": str(segment_input),
                        "edited_segment_path": str(current_path),
                        "fitted_segment_path": str(current_path),
                        "stage_records": [
                            {
                                "operation": "text_edit",
                                "skipped": True,
                                "reason": "span lacks transcript text required for tagged text_edit; preserved original span",
                                "arguments": {
                                    "duration_sec": round(span.duration_sec, 6),
                                    "has_text": False,
                                },
                            }
                        ],
                    }
                )
                continue
            if set(operations) <= {
                "text_edit",
                "pitch_rate_edit",
                "emotion_style_edit",
                "pause_edit",
            }:
                segment_records.append(
                    {
                        **span.to_dict(),
                        "input_segment_path": str(segment_input),
                        "edited_segment_path": str(current_path),
                        "fitted_segment_path": str(current_path),
                        "stage_records": [
                            {
                                "operation": ",".join(operations),
                                "skipped": True,
                                "reason": "span lacks transcript text required for tagged edit-data-pipeline synthesis; preserved original span",
                                "arguments": {
                                    "duration_sec": round(span.duration_sec, 6),
                                    "has_text": False,
                                },
                            }
                        ],
                    }
                )
                continue
            raise RuntimeError(
                f"speaker_scoped_edit span {index} needs transcript text for operations {operations}; "
                "run perception with ASR+alignment."
            )

        stage_records = []
        for stage_index, operation in enumerate(operations, start=1):
            stage_output = (
                edited_dir / f"speaker_span_{index:03d}_stage{stage_index}_{operation}.wav"
            )
            if operation == "mute_spans":
                current_path = _write_silence_like(
                    current_path,
                    stage_output,
                    duration_sec=span.duration_sec,
                )
                stage_records.append(
                    {
                        "operation": operation,
                        "reason": "requested speaker-scoped speech removal/muting",
                        "arguments": {"duration_sec": round(span.duration_sec, 6)},
                        "output": {"output_audio_path": str(current_path)},
                    }
                )
                continue
            if operation in {
                "text_edit",
                "pitch_rate_edit",
                "emotion_style_edit",
                "pause_edit",
            } and (not current_text or span.duration_sec < MIN_TAGGED_EDIT_SEGMENT_SEC):
                if _should_mute_nonverbal_span(span, grounding_instruction, perception):
                    current_path = _write_silence_like(
                        current_path,
                        stage_output,
                        duration_sec=span.duration_sec,
                    )
                    stage_records.append(
                        {
                            "operation": "mute_nonverbal",
                            "reason": "requested non-verbal laughter removal",
                            "arguments": {"duration_sec": round(span.duration_sec, 6)},
                            "output": {"output_audio_path": str(current_path)},
                        }
                    )
                    break
                stage_records.append(
                    {
                        "operation": operation,
                        "skipped": True,
                        "reason": "span is too short or lacks transcript for tagged edit-data-pipeline synthesis",
                        "arguments": {
                            "duration_sec": round(span.duration_sec, 6),
                            "has_text": bool(current_text),
                        },
                    }
                )
                continue
            if operation == "seed_vc":
                reference_audio = _speaker_scoped_reference_audio(
                    params,
                    direct_params,
                    reference_audio_paths or [],
                )
                seed_vc_args = {
                    "source_audio_path": str(current_path),
                    "reference_audio_path": reference_audio,
                    "output_path": str(stage_output),
                    "diffusion_steps": int(
                        direct_params.get("diffusion_steps", params.get("diffusion_steps", 30))
                    ),
                    "length_adjust": float(
                        direct_params.get("length_adjust", params.get("length_adjust", 1.0))
                    ),
                    "inference_cfg_rate": float(
                        direct_params.get(
                            "inference_cfg_rate", params.get("inference_cfg_rate", 0.7)
                        )
                    ),
                }
                result = _call_required_tool_gateway(tool_gateway, "seed_vc", seed_vc_args)
                current_path = Path(result["output_audio_path"])
                stage_records.append(
                    {
                        "operation": operation,
                        "arguments": _redact_large_args(seed_vc_args),
                        "output": result,
                    }
                )
                continue

            if operation == "no_reference_voice_conversion":
                voice_prompt = _speaker_scoped_no_ref_voice_prompt(
                    grounding_instruction, direct_params, params
                )
                reference_key = voice_prompt
                reference_record = no_ref_voice_references.get(reference_key)
                if reference_record is None:
                    reference_path = (
                        service_dir
                        / f"no_ref_voice_reference_{len(no_ref_voice_references):03d}.wav"
                    )
                    reference_text = _speaker_scoped_no_ref_reference_text(direct_params, params)
                    reference_args = {
                        "text": reference_text,
                        "output_path": str(reference_path),
                        "mode": "custom_voice",
                        "instruct": voice_prompt,
                        "language": str(
                            direct_params.get("language") or params.get("language") or "Auto"
                        ),
                        "voice_design_backend": str(
                            direct_params.get("voice_design_backend")
                            or params.get("voice_design_backend")
                            or os.getenv("X_EDITOR_VOICE_DESIGN_BACKEND")
                            or "qwen3_tts"
                        ),
                    }
                    reference_result = _call_required_tool_gateway(
                        tool_gateway, "qwen3_tts", reference_args
                    )
                    reference_record = {
                        "reference_audio_path": str(
                            reference_result.get("output_audio_path") or reference_path
                        ),
                        "reference_text": reference_text,
                        "voice_prompt": voice_prompt,
                        "reference_generation": reference_result,
                    }
                    no_ref_voice_references[reference_key] = reference_record
                seed_vc_args = {
                    "source_audio_path": str(current_path),
                    "reference_audio_path": reference_record["reference_audio_path"],
                    "output_path": str(stage_output),
                    "diffusion_steps": int(
                        direct_params.get("diffusion_steps", params.get("diffusion_steps", 30))
                    ),
                    "length_adjust": float(
                        direct_params.get("length_adjust", params.get("length_adjust", 1.0))
                    ),
                    "inference_cfg_rate": float(
                        direct_params.get(
                            "inference_cfg_rate", params.get("inference_cfg_rate", 0.7)
                        )
                    ),
                }
                result = _call_required_tool_gateway(tool_gateway, "seed_vc", seed_vc_args)
                current_path = Path(result["output_audio_path"])
                stage_records.append(
                    {
                        "operation": operation,
                        "arguments": {
                            "source_audio_path": str(segment_input),
                            "reference_audio_path": reference_record["reference_audio_path"],
                            "output_path": str(stage_output),
                            "voice_prompt": voice_prompt,
                        },
                        "output": {
                            "output_audio_path": str(current_path),
                            "target_voice_reference_audio_path": reference_record[
                                "reference_audio_path"
                            ],
                            "target_voice_reference_text": reference_record["reference_text"],
                            "voice_design_prompt": voice_prompt,
                            "reference_generation": reference_record["reference_generation"],
                            "seed_vc_conversion": result,
                        },
                    }
                )
                continue

            if operation == "speech_enhancement":
                enhancement_args = {
                    "input_audio_path": str(current_path),
                    "output_path": str(stage_output),
                    "mode": str(direct_params.get("mode") or params.get("mode") or "general"),
                    "preserve_background": bool(
                        direct_params.get(
                            "preserve_background",
                            params.get("preserve_background", False),
                        )
                    ),
                    "enhancement_goal": str(
                        direct_params.get("enhancement_goal")
                        or params.get("enhancement_goal")
                        or grounding_instruction
                    ),
                }
                if tool_gateway is not None:
                    result = tool_gateway("speech_enhancement", enhancement_args)
                else:
                    if clearer_voice_client is None:
                        clearer_voice_client = ClearerVoiceClient(
                            os.getenv("CLEARERVOICE_SERVICE_URL", "http://127.0.0.1:8027"),
                            timeout=float(os.getenv("X_EDITOR_CLEARERVOICE_TIMEOUT", "300")),
                        )
                    result = clearer_voice_client.enhance(**enhancement_args)
                current_path = Path(result["output_audio_path"])
                stage_records.append(
                    {
                        "operation": operation,
                        "arguments": _redact_large_args(enhancement_args),
                        "output": result,
                    }
                )
                continue

            if operation == "audio_effects":
                effect_instruction = str(
                    direct_params.get("instruction")
                    or direct_params.get("edit_instruction")
                    or params.get("instruction")
                    or grounding_instruction
                )
                result = apply_audio_effect(
                    input_audio_path=str(current_path),
                    output_path=str(stage_output),
                    instruction=effect_instruction,
                    preset=str(direct_params.get("preset") or params.get("preset") or "") or None,
                    effect=str(
                        direct_params.get("effect")
                        or direct_params.get("effect_name")
                        or params.get("effect")
                        or params.get("effect_name")
                        or ""
                    )
                    or None,
                    strength=float(direct_params.get("strength", params.get("strength", 0.85))),
                    reference_audio_path=direct_params.get("reference_audio_path")
                    or params.get("reference_audio_path"),
                    preserve_duration=True,
                    output_dir=str(service_dir / f"span_{index:03d}_{operation}"),
                )
                current_path = Path(result["output_audio_path"])
                stage_records.append(
                    {
                        "operation": operation,
                        "arguments": {
                            "input_audio_path": str(segment_input),
                            "output_path": str(stage_output),
                            "instruction": effect_instruction,
                            "preset": result.get("preset"),
                            "strength": result.get("strength"),
                        },
                        "output": result,
                    }
                )
                continue

            if operation == "speech_rate":
                args = _speaker_scoped_speech_rate_args(
                    instruction=str(
                        direct_params.get("instruction")
                        or direct_params.get("edit_instruction")
                        or instruction
                    ),
                    input_audio_path=str(current_path),
                    output_path=str(stage_output),
                    current_text=current_text,
                    params=params,
                    direct_params=direct_params,
                )
            elif operation == "volume_normalize":
                args = _speaker_scoped_volume_normalize_args(
                    instruction=str(
                        direct_params.get("instruction")
                        or direct_params.get("edit_instruction")
                        or instruction
                    ),
                    input_audio_path=str(current_path),
                    output_path=str(stage_output),
                    params=params,
                    direct_params=direct_params,
                )
            else:
                nested_instruction = str(
                    direct_params.get("instruction")
                    or direct_params.get("edit_instruction")
                    or instruction
                )
                nested_params = _speaker_scoped_segment_edit_params(direct_params, current_text)
                args = None
                if (
                    operation
                    in {
                        "text_edit",
                        "pitch_rate_edit",
                        "emotion_style_edit",
                        "pause_edit",
                    }
                    and nested_params
                ):
                    try:
                        args = build_edit_data_pipeline_tool_arguments(
                            operation=operation,
                            instruction=nested_instruction,
                            source_text=current_text,
                            input_audio_path=str(current_path),
                            output_path=str(stage_output),
                            output_dir=str(service_dir / f"span_{index:03d}_{operation}"),
                            language=_language_hint(current_text),
                            params=nested_params,
                        )
                    except Exception:
                        args = None
                if args is None:
                    args = build_edit_data_pipeline_arguments(
                        operation,
                        nested_instruction,
                        current_text,
                        input_audio_path=str(current_path),
                        output_path=str(stage_output),
                        output_dir=str(service_dir / f"span_{index:03d}_{operation}"),
                        language=_language_hint(current_text),
                    )
            if args is None and operation == "pitch_rate_edit":
                pitch_instruction = str(
                    direct_params.get("instruction")
                    or direct_params.get("edit_instruction")
                    or instruction
                )
                args = _speaker_scoped_pitch_rate_args_from_pitch_or_rate_request(
                    instruction=pitch_instruction,
                    input_audio_path=str(current_path),
                    output_path=str(stage_output),
                    current_text=current_text,
                    span=span,
                    direct_params=direct_params,
                    output_dir=str(service_dir / f"span_{index:03d}_{operation}"),
                )
                if args is None:
                    args = _speaker_scoped_pitch_rate_args_from_intonation_request(
                        instruction=pitch_instruction,
                        input_audio_path=str(current_path),
                        output_path=str(stage_output),
                        current_text=current_text,
                        span=span,
                        direct_params=direct_params,
                        output_dir=str(service_dir / f"span_{index:03d}_{operation}"),
                    )
            if args is None and operation == "text_edit":
                target_text = _speaker_scoped_text_edit_target_text(
                    instruction=str(
                        direct_params.get("instruction")
                        or direct_params.get("edit_instruction")
                        or instruction
                    ),
                    current_text=current_text,
                    direct_params=direct_params,
                    span=span,
                )
            else:
                target_text = None
            if args is None and operation == "text_edit" and target_text:
                text_instruction = str(
                    direct_params.get("instruction")
                    or direct_params.get("edit_instruction")
                    or instruction
                )
                args = build_edit_data_pipeline_tool_arguments(
                    operation="text_edit",
                    instruction=text_instruction,
                    source_text=current_text,
                    input_audio_path=str(current_path),
                    output_path=str(stage_output),
                    output_dir=str(service_dir / f"span_{index:03d}_{operation}"),
                    language=_language_hint(current_text),
                    params={
                        **direct_params,
                        "source_text": current_text,
                        "target_text": target_text,
                        "instruction_st": _tagged_text_substitution(
                            current_text,
                            target_text,
                        ),
                        "evaluate": False,
                    },
                )
            if args is None:
                raise RuntimeError(
                    f"speaker_scoped_edit could not build edit-data-pipeline arguments for operation={operation}"
                )
            try:
                result = (
                    tool_gateway("edit_data_pipeline", args)
                    if tool_gateway is not None
                    else client.apply(args)  # type: ignore[union-attr]
                )
            except Exception as exc:
                retry_result = _speaker_scoped_retry_without_prepend_context_flags(
                    client=client,
                    tool_gateway=tool_gateway,
                    operation=operation,
                    error=exc,
                    args=args,
                )
                if retry_result is not None:
                    current_path = Path(retry_result["output_audio_path"])
                    current_text = str(
                        retry_result.get("target_text")
                        or retry_result.get("generated_text")
                        or current_text
                    )
                    stage_records.append(
                        {
                            "operation": operation,
                            "arguments": _redact_large_args(args),
                            "retry": True,
                            "retry_reason": str(exc),
                            "retry_repair": "dropped_prepend_context_optional_flags",
                            "output": retry_result,
                        }
                    )
                    continue
                fallback_result = _speaker_scoped_pitch_rate_failure_fallback(
                    operation=operation,
                    error=exc,
                    args=args,
                    input_audio_path=str(current_path),
                    output_path=str(stage_output),
                    span=span,
                    current_text=current_text,
                    instruction=str(
                        direct_params.get("instruction")
                        or direct_params.get("edit_instruction")
                        or instruction
                    ),
                    direct_params=direct_params,
                )
                if fallback_result is None:
                    fallback_result = _speaker_scoped_text_edit_failure_fallback(
                        operation=operation,
                        error=exc,
                        args=args,
                        input_audio_path=str(current_path),
                        output_path=str(stage_output),
                        current_text=current_text,
                        instruction=str(
                            direct_params.get("instruction")
                            or direct_params.get("edit_instruction")
                            or instruction
                        ),
                    )
                if fallback_result is None:
                    raise
                current_path = Path(fallback_result["output_audio_path"])
                stage_records.append(
                    {
                        "operation": operation,
                        "arguments": _redact_large_args(args),
                        "fallback": True,
                        "fallback_reason": str(exc),
                        "output": fallback_result,
                    }
                )
                continue
            current_path = Path(result["output_audio_path"])
            current_text = str(
                result.get("target_text") or result.get("generated_text") or current_text
            )
            stage_records.append(
                {
                    "operation": operation,
                    "arguments": _redact_large_args(args),
                    "output": result,
                }
            )

        replacement = load_audio_segment(current_path)
        fitted_path = edited_dir / f"speaker_span_{index:03d}_fitted.wav"
        if preserve_timeline:
            replacement = _fit_audio_duration(replacement, int(round(span.duration_sec * 1000)))
        replacement.export(fitted_path, format="wav")
        edited_timeline = _replace_range(edited_timeline, replacement, span.start_sec, span.end_sec)
        segment_records.append(
            {
                **span.to_dict(),
                "input_segment_path": str(segment_input),
                "edited_segment_path": str(current_path),
                "fitted_segment_path": str(fitted_path),
                "stage_records": stage_records,
            }
        )

    if event_removal_spans:
        edited_timeline, event_removal_records = _apply_speaker_scoped_event_removals(
            edited_timeline,
            event_removal_spans=event_removal_spans,
            speaker_edit_spans=spans,
            words=words,
            original=original,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    edited_timeline.export(output, format="wav")
    manifest = {
        "operation": "speaker_scoped_edit",
        "instruction": instruction,
        "input_audio_path": str(input_audio),
        "output_audio_path": str(output),
        "target_speakers": target_speakers,
        "operations": operations,
        "selected_spans": [span.to_dict() for span in spans],
        "segments": segment_records,
        "constraints": {
            "timeline_duration_preserved": preserve_timeline,
            "max_segment_sec": MAX_SEGMENT_SEC,
            "word_boundary_snapping": bool(words),
            "single_speaker_segments": True,
        },
        "perception_services": perception.get("services", {}),
    }
    if event_removal_records:
        manifest["event_removal_segments"] = event_removal_records
        manifest["event_removal_policy"] = params.get("event_removal_policy")
    if selection_evidence:
        manifest["perception_segment_selection"] = selection_evidence
    return _write_manifest(Path(output_dir), manifest)


def reconstruct_audio_placeholder(
    base_audio_path: str,
    edited_audio_path: str,
    output_path: str,
) -> dict[str, str]:
    """Compatibility fallback for older reconstruction callers."""

    src = Path(edited_audio_path) if edited_audio_path else Path()
    copied_role = "edited_artifact"
    if not edited_audio_path or not src.is_file():
        src = Path(base_audio_path)
        copied_role = "base_audio"
    if not src.is_file():
        raise FileNotFoundError(
            f"No source audio for reconstruction: {edited_audio_path} or {base_audio_path}"
        )
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "output_audio_path": str(dst),
        "copied_audio_path": str(src),
        "copied_artifact_role": copied_role,
        "note": f"compatibility reconstruction copied {copied_role} without applying timeline edits",
    }


def _copy_reconstruct_artifact(
    source_path: str,
    output_path: str,
    *,
    copied_role: str,
) -> dict[str, str]:
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Reconstruction source audio does not exist: {source}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != output.resolve():
        shutil.copy2(source, output)
    return {
        "output_audio_path": str(output),
        "copied_audio_path": str(source),
        "copied_artifact_role": copied_role,
        "note": f"reconstruction copied the declared {copied_role}",
    }


def _require_reconstruct_contract(
    *,
    declared_mode: str | None,
    declared_scope: str | None,
    expected_mode: str,
    expected_scope: str,
) -> None:
    if declared_mode is not None and declared_mode != expected_mode:
        raise RuntimeError(
            "artifact_ops.reconstruct reconstruction_mode conflicts with its inputs: "
            f"declared {declared_mode}, inferred {expected_mode}"
        )
    if declared_scope is not None and declared_scope != expected_scope:
        raise RuntimeError(
            "artifact_ops.reconstruct edited_artifact_scope conflicts with its inputs: "
            f"declared {declared_scope}, inferred {expected_scope}"
        )


def _execute_reconstruct(
    *,
    instruction: str,
    input_audio_path: str,
    edited_audio_path: str | None,
    output_path: str,
    output_dir: str,
    perception_report: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    raw_declared_mode = params.get("reconstruction_mode")
    declared_mode = normalized_reconstruction_mode(raw_declared_mode)
    if raw_declared_mode not in (None, "") and declared_mode is None:
        raise RuntimeError("artifact_ops.reconstruct received an invalid reconstruction_mode")
    raw_declared_scope = params.get("edited_artifact_scope")
    declared_scope = normalized_edited_artifact_scope(raw_declared_scope)
    if raw_declared_scope not in (None, "") and declared_scope is None:
        raise RuntimeError("artifact_ops.reconstruct received an invalid edited_artifact_scope")
    edits = params.get("edits")
    if isinstance(edits, list) and edits:
        _require_reconstruct_contract(
            declared_mode=declared_mode,
            declared_scope=declared_scope,
            expected_mode="timeline_edits",
            expected_scope="local_regions",
        )
        return _execute_reconstruct_edits(
            instruction=instruction,
            input_audio_path=input_audio_path,
            edits=edits,
            edit_specs=params.get("instruction_st"),
            output_path=output_path,
            output_dir=output_dir,
            perception_report=perception_report,
        )

    edited_span_paths = params.get("edited_spans_audio_paths") or params.get("edited_spans_audio")
    span_specs = params.get("spans") or params.get("selected_spans")
    if isinstance(edited_span_paths, (list, tuple)) and edited_span_paths and span_specs:
        _require_reconstruct_contract(
            declared_mode=declared_mode,
            declared_scope=declared_scope,
            expected_mode="span_replacement",
            expected_scope="local_regions",
        )
        return _execute_reconstruct_span_replacements(
            instruction=instruction,
            input_audio_path=input_audio_path,
            replacement_audio_paths=[str(item) for item in edited_span_paths],
            span_specs=span_specs,
            output_path=output_path,
            output_dir=output_dir,
            perception_report=perception_report,
        )

    edited = Path(edited_audio_path) if edited_audio_path else None
    if not edited or not edited.exists():
        _require_reconstruct_contract(
            declared_mode=declared_mode,
            declared_scope=declared_scope,
            expected_mode="placeholder_compatibility",
            expected_scope="none",
        )
        result = reconstruct_audio_placeholder(
            input_audio_path, edited_audio_path or "", output_path
        )
        manifest = {
            "operation": "reconstruct",
            "instruction": instruction,
            "input_audio_path": input_audio_path,
            "edited_audio_path": edited_audio_path,
            "output_audio_path": output_path,
            "result": result,
            "mode": "placeholder_compatibility",
            "reconstruction_mode": "placeholder_compatibility",
            "edited_artifact_scope": "none",
            "acceptance_eligible": False,
            "degraded_constraints": [
                "no edited artifact was available; copied the base audio without applying reconstruction"
            ],
        }
        return _write_manifest(Path(output_dir), manifest)

    spans = params.get("selected_spans") or params.get("spans")
    if not isinstance(spans, list) or not spans:
        if declared_scope != "full_timeline":
            raise RuntimeError(
                "artifact_ops.reconstruct edited audio without spans is ambiguous; "
                "declare edited_artifact_scope=full_timeline for passthrough"
            )
        _require_reconstruct_contract(
            declared_mode=declared_mode,
            declared_scope=declared_scope,
            expected_mode="full_timeline_passthrough",
            expected_scope="full_timeline",
        )
        result = _copy_reconstruct_artifact(
            str(edited),
            output_path,
            copied_role="full_timeline_edited_artifact",
        )
        manifest = {
            "operation": "reconstruct",
            "instruction": instruction,
            "input_audio_path": input_audio_path,
            "edited_audio_path": str(edited),
            "output_audio_path": output_path,
            "result": result,
            "mode": "full_timeline_passthrough",
            "reconstruction_mode": "full_timeline_passthrough",
            "edited_artifact_scope": "full_timeline",
            "full_timeline_source_path": str(edited),
            "acceptance_eligible": True,
        }
        return _write_manifest(Path(output_dir), manifest)

    _require_reconstruct_contract(
        declared_mode=declared_mode,
        declared_scope=declared_scope,
        expected_mode="span_replacement",
        expected_scope="local_regions",
    )
    parsed_spans = _spans_from_dicts(spans)
    audio = load_audio_segment(input_audio_path)
    replacement = load_audio_segment(edited)
    edited_audio = audio
    if len(parsed_spans) == 1:
        span = parsed_spans[0]
        edited_audio = _replace_range(
            edited_audio,
            _fit_audio_duration(replacement, int(round(span.duration_sec * 1000))),
            span.start_sec,
            span.end_sec,
        )
    else:
        raise RuntimeError(
            "reconstruct currently supports explicit replacement for one span or copy fallback"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    edited_audio.export(output_path, format="wav")
    manifest = {
        "operation": "reconstruct",
        "instruction": instruction,
        "input_audio_path": input_audio_path,
        "edited_audio_path": str(edited),
        "output_audio_path": output_path,
        "selected_spans": [span.to_dict() for span in parsed_spans],
        "mode": "span_replacement",
        "reconstruction_mode": "span_replacement",
        "edited_artifact_scope": "local_regions",
        "replacement_records": [
            {
                **parsed_spans[0].to_dict(),
                "replacement_audio_path": str(edited),
            }
        ],
        "acceptance_eligible": True,
        "perception_services": perception_report.get("services", {}),
    }
    return _write_manifest(Path(output_dir), manifest)


def _execute_reorder_spans(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    perception_report: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    base = load_audio_segment(input_audio_path)
    duration_sec = len(base) / 1000.0
    spans = _reorder_target_spans(params, perception_report, duration_sec)
    if len(spans) < 2:
        raise RuntimeError("reorder_spans requires at least two target spans")
    spans = sorted(spans, key=lambda span: (span.start_sec, span.end_sec))
    _validate_reorder_spans(spans)
    raw_order = params.get("order")
    raw_speaker_order = params.get("target_speaker_order")
    if raw_order is not None and raw_speaker_order is not None:
        raise RuntimeError("reorder_spans accepts either order or target_speaker_order, not both")
    if raw_order is None and raw_speaker_order is None:
        reorder_indices = list(reversed(range(len(spans))))
        order_source = "default_reverse"
    elif raw_order is not None:
        try:
            reorder_indices = normalize_reorder_indices(
                raw_order,
                span_count=len(spans),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"reorder_spans received invalid explicit order: {exc}") from exc
        order_source = "explicit_order"
    else:
        try:
            reorder_indices = reorder_indices_for_speaker_order(
                raw_speaker_order,
                span_speaker_ids=[span.speaker_id for span in spans],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"reorder_spans received invalid explicit speaker order: {exc}"
            ) from exc
        order_source = "explicit_speaker_order"

    output = (
        AudioSegment.silent(duration=0, frame_rate=base.frame_rate)
        .set_channels(base.channels)
        .set_sample_width(base.sample_width)
    )
    preserve_duration = bool(params.get("preserve_duration", True))
    cursor_ms = 0
    reorder_records = []
    for destination_index, source_index in enumerate(reorder_indices):
        destination_span = spans[destination_index]
        source_span = spans[source_index]
        destination_start_ms = int(round(destination_span.start_sec * 1000))
        destination_end_ms = int(round(destination_span.end_sec * 1000))
        source_start_ms = int(round(source_span.start_sec * 1000))
        source_end_ms = int(round(source_span.end_sec * 1000))
        output += base[cursor_ms:destination_start_ms]
        replacement = base[source_start_ms:source_end_ms]
        if preserve_duration:
            replacement = _fit_audio_duration(
                replacement,
                destination_end_ms - destination_start_ms,
            )
        output_start_sec = len(output) / 1000.0
        output += replacement
        output_end_sec = len(output) / 1000.0
        reorder_records.append(
            {
                "destination_index": destination_index,
                "source_index": source_index,
                "destination_span": destination_span.to_dict(),
                "source_span": source_span.to_dict(),
                "output_span": {
                    "start_sec": round(output_start_sec, 6),
                    "end_sec": round(output_end_sec, 6),
                },
                "fit_to_slot_duration": preserve_duration,
            }
        )
        cursor_ms = destination_end_ms
    output += base[cursor_ms:]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    manifest = {
        "operation": "reorder_spans",
        "instruction": instruction,
        "input_audio_path": input_audio_path,
        "output_audio_path": output_path,
        "selected_spans": [span.to_dict() for span in spans],
        "selected_spans_source": "params.selected_spans"
        if params.get("selected_spans") or params.get("spans")
        else "speaker_segments",
        "reorder_indices": reorder_indices,
        "reorder_records": reorder_records,
        "order_source": order_source,
        "mode": "reverse"
        if reorder_indices == list(reversed(range(len(spans))))
        else "custom_order",
        "input_duration_sec": round(duration_sec, 6),
        "output_duration_sec": round(len(output) / 1000.0, 6),
        "timeline_duration_preserved": preserve_duration,
        "perception_services": perception_report.get("services", {}),
    }
    return _write_manifest(Path(output_dir), manifest)


def _validate_reorder_spans(spans: list[TimeSpan]) -> None:
    previous_end = 0.0
    for span in spans:
        if span.end_sec <= span.start_sec:
            raise RuntimeError("reorder_spans target spans must have positive duration")
        if span.start_sec < previous_end - 1e-9:
            raise RuntimeError("reorder_spans target spans must not overlap")
        previous_end = span.end_sec


def _reorder_target_spans(
    params: dict[str, Any], perception: dict[str, Any], duration_sec: float
) -> list[TimeSpan]:
    explicit = params.get("selected_spans") or params.get("spans") or params.get("segments")
    if isinstance(explicit, list) and explicit:
        spans = []
        for span in _spans_from_dicts(explicit):
            clipped = span.clipped(duration_sec)
            if clipped:
                spans.append(clipped)
        return spans
    speaker_segments = _normalized_speaker_segments(perception)
    if not speaker_segments:
        return []
    sorted_segments = sorted(speaker_segments, key=lambda item: (item.start_sec, item.end_sec))
    seen: set[str] = set()
    selected: list[TimeSpan] = []
    fallback_turns: list[TimeSpan] = []
    for span in sorted_segments:
        clipped = span.clipped(duration_sec)
        if clipped:
            fallback_turns.append(clipped)
        speaker = span.speaker_id or f"span_{len(selected)}"
        if speaker in seen:
            continue
        if clipped:
            selected.append(clipped)
            seen.add(speaker)
    if len(selected) < 2 and len(fallback_turns) >= 2:
        return fallback_turns
    return selected


def _execute_reconstruct_span_replacements(
    *,
    instruction: str,
    input_audio_path: str,
    replacement_audio_paths: list[str],
    span_specs: Any,
    output_path: str,
    output_dir: str,
    perception_report: dict[str, Any],
) -> dict[str, Any]:
    base = load_audio_segment(input_audio_path)
    duration_sec = base.frame_count() / base.frame_rate
    spans = _reconstruct_spans_from_specs(span_specs, perception_report, duration_sec)
    if not spans:
        raise RuntimeError("reconstruct could not resolve replacement spans from params.spans")
    replacements = [Path(path) for path in replacement_audio_paths if path]
    for path in replacements:
        if not path.is_file():
            raise FileNotFoundError(f"reconstruct replacement audio does not exist: {path}")
    if not replacements:
        raise RuntimeError("reconstruct requires edited_spans_audio_paths for replacement spans")

    edited_audio = base
    replacement_records = []
    if len(replacements) == 1:
        replacement = load_audio_segment(replacements[0])
        total_target_ms = sum(max(1, int(round(span.duration_sec * 1000))) for span in spans)
        replacement_cursor = 0
        for index, span in enumerate(spans):
            target_ms = max(1, int(round(span.duration_sec * 1000)))
            if index == len(spans) - 1:
                chunk = replacement[replacement_cursor:]
            else:
                chunk_end = min(
                    len(replacement),
                    replacement_cursor + int(round(len(replacement) * target_ms / total_target_ms)),
                )
                chunk = replacement[replacement_cursor:chunk_end]
                replacement_cursor = chunk_end
            edited_audio = replace_fixed_audio_interval(
                edited_audio, chunk, start_sec=span.start_sec, end_sec=span.end_sec
            )
            replacement_records.append(
                {**span.to_dict(), "replacement_audio_path": str(replacements[0])}
            )
    elif len(replacements) == len(spans):
        for span, path in zip(spans, replacements):
            edited_audio = replace_fixed_audio_interval(
                edited_audio, load_audio_segment(path),
                start_sec=span.start_sec, end_sec=span.end_sec,
            )
            replacement_records.append({**span.to_dict(), "replacement_audio_path": str(path)})
    else:
        raise RuntimeError("reconstruct requires one replacement audio or one replacement per span")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    edited_audio.export(output_path, format="wav")
    manifest = {
        "operation": "reconstruct",
        "instruction": instruction,
        "input_audio_path": input_audio_path,
        "output_audio_path": output_path,
        "selected_spans": [span.to_dict() for span in spans],
        "replacement_records": replacement_records,
        "mode": "span_replacement",
        "reconstruction_mode": "span_replacement",
        "edited_artifact_scope": "local_regions",
        "acceptance_eligible": True,
        "perception_services": perception_report.get("services", {}),
    }
    return _write_manifest(Path(output_dir), manifest)


def _execute_reconstruct_edits(
    *,
    instruction: str,
    input_audio_path: str,
    edits: list[Any],
    edit_specs: Any,
    output_path: str,
    output_dir: str,
    perception_report: dict[str, Any],
) -> dict[str, Any]:
    base = load_audio_segment(input_audio_path)
    geometry = AudioFrameGeometry.from_segment(base)
    duration_sec = geometry.duration_sec
    parsed = []
    specs = edit_specs if isinstance(edit_specs, list) else []
    for index, item in enumerate(edits):
        if not isinstance(item, dict):
            continue
        spec = _reconstruct_edit_spec_for_index(specs, index)
        replacement_path = (
            item.get("input_audio_path")
            or item.get("audio_path")
            or item.get("path")
            or item.get("replacement_audio_path")
            or item.get("replacement_audio")
            or item.get("edited_audio_path")
            or item.get("edited_audio")
            or spec.get("input_audio_path")
            or spec.get("audio_path")
            or spec.get("path")
            or spec.get("replacement_audio_path")
            or spec.get("replacement_audio")
            or spec.get("edited_audio_path")
            or spec.get("edited_audio")
        )
        if not isinstance(replacement_path, str) or not replacement_path:
            continue
        replacement = Path(replacement_path)
        if not replacement.is_file():
            raise FileNotFoundError(
                f"reconstruct edit replacement audio does not exist: {replacement_path}"
            )
        raw_mode = item.get("mode") or spec.get("mode") or spec.get("operation")
        start_value = item.get("start_sec", spec.get("start_sec", 0.0))
        end_value = item.get("end_sec", spec.get("end_sec", start_value))
        replacement_duration_sec = _audio_duration_sec(str(replacement))
        start = _reconstruct_timeline_position(
            start_value,
            duration_sec=duration_sec,
            replacement_duration_sec=replacement_duration_sec,
            default=0.0,
        )
        end = _reconstruct_timeline_position(
            end_value,
            duration_sec=duration_sec,
            replacement_duration_sec=replacement_duration_sec,
            default=start,
        )
        start = max(0.0, min(start, duration_sec))
        end = max(start, min(end, duration_sec))
        mode = _reconstruct_timeline_edit_mode(raw_mode, start=start, end=end)
        parsed.append((start, end, replacement, index, mode))
    if not parsed:
        raise RuntimeError("reconstruct edits did not include any valid replacement audio")
    _validate_reconstruct_timeline_edit_geometry(parsed)

    output = base
    replacement_items = [item for item in parsed if item[4] not in {"overlay", "mix"}]
    overlay_items = [item for item in parsed if item[4] in {"overlay", "mix"}]
    if replacement_items:
        output = (
            AudioSegment.silent(duration=0, frame_rate=base.frame_rate)
            .set_channels(base.channels)
            .set_sample_width(base.sample_width)
        )
        cursor_frame = 0
        edit_records = []
        for start, end, replacement_path, index, mode in sorted(
            replacement_items, key=lambda item: (item[0], item[1], item[3])
        ):
            start_frame = max(cursor_frame, round(start * geometry.sample_rate))
            end_frame = max(start_frame, round(end * geometry.sample_rate))
            if cursor_frame < start_frame:
                output += base.get_sample_slice(start_sample=cursor_frame, end_sample=start_frame)
            replacement = _match_audio_format(load_audio_segment(replacement_path), base)
            target_frames = end_frame - start_frame
            fit_to_span = target_frames > 0 and mode == "replace"
            if fit_to_span:
                replacement = fit_audio_frames(replacement, target_frames)
            output += replacement
            cursor_frame = end_frame
            edit_records.append(
                {
                    "index": index,
                    "mode": mode,
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                    "replacement_audio_path": str(replacement_path),
                    "replacement_duration_sec": AudioFrameGeometry.from_segment(replacement).duration_sec,
                    "fit_to_span_duration": fit_to_span,
                }
            )
        if cursor_frame < geometry.frame_count:
            output += base.get_sample_slice(start_sample=cursor_frame)
    else:
        edit_records = []

    for start, end, replacement_path, index, mode in sorted(
        overlay_items, key=lambda item: (item[0], item[1], item[3])
    ):
        start_ms = int(round(start * 1000))
        replacement = _match_audio_format(load_audio_segment(replacement_path), output)
        output = output.overlay(replacement, position=start_ms)
        edit_records.append(
            {
                "index": index,
                "mode": mode,
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "replacement_audio_path": str(replacement_path),
                "replacement_duration_sec": round(len(replacement) / 1000.0, 6),
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")
    manifest = {
        "operation": "reconstruct",
        "instruction": instruction,
        "input_audio_path": input_audio_path,
        "output_audio_path": output_path,
        "edits": edit_records,
        "mode": "timeline_edits",
        "reconstruction_mode": "timeline_edits",
        "edited_artifact_scope": "local_regions",
        "acceptance_eligible": True,
        "input_duration_sec": duration_sec,
        "output_duration_sec": AudioFrameGeometry.from_segment(output).duration_sec,
        "perception_services": perception_report.get("services", {}),
    }
    return _write_manifest(Path(output_dir), manifest)


def _reconstruct_timeline_edit_mode(
    value: Any,
    *,
    start: float,
    end: float,
) -> str:
    explicit = str(value or "").strip().lower().replace("-", "_")
    if not explicit:
        return "insert" if abs(end - start) <= 1e-9 else "replace"
    aliases = {"mix": "overlay", "insertion": "insert"}
    mode = aliases.get(explicit, explicit)
    if mode not in {"insert", "overlay", "replace", "splice"}:
        raise RuntimeError("reconstruct timeline edit mode must be insert, overlay/mix, replace, or splice")
    if mode == "insert" and abs(end - start) > 1e-9:
        raise RuntimeError("reconstruct insert edits require start_sec == end_sec")
    if mode in {"replace", "splice"} and end <= start:
        raise RuntimeError("reconstruct replace/splice edits require end_sec > start_sec")
    return mode


def _validate_reconstruct_timeline_edit_geometry(
    edits: list[tuple[float, float, Path, int, str]],
) -> None:
    previous_end = 0.0
    for start, end, _path, _index, mode in sorted(
        edits,
        key=lambda item: (item[0], item[1], item[3]),
    ):
        if mode == "overlay":
            continue
        if start < previous_end - 1e-9:
            raise RuntimeError("reconstruct replacement edit spans must not overlap")
        previous_end = max(previous_end, end)


def _reconstruct_edit_spec_for_index(specs: list[Any], index: int) -> dict[str, Any]:
    for item in specs:
        if (
            isinstance(item, dict)
            and int(item.get("edit_index", item.get("index", -1)) or -1) == index
        ):
            return item
    if index < len(specs) and isinstance(specs[index], dict):
        return specs[index]
    return {}


def _reconstruct_timeline_position(
    value: Any,
    *,
    duration_sec: float,
    replacement_duration_sec: float,
    default: float,
) -> float:
    if isinstance(value, str):
        text = " ".join(value.strip().lower().split())
        if text in {"midpoint", "middle", "center", "centre"}:
            return max(0.0, (duration_sec - replacement_duration_sec) / 2.0)
        if text in {"end", "tail"}:
            return max(0.0, duration_sec - replacement_duration_sec)
        if text in {"start", "beginning"}:
            return 0.0
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _audio_duration_sec(audio_path: str) -> float:
    try:
        return AudioFrameGeometry.from_segment(load_audio_segment(audio_path)).duration_sec
    except Exception:
        return 0.0


def _match_audio_format(audio: AudioSegment, reference: AudioSegment) -> AudioSegment:
    if audio.frame_rate != reference.frame_rate:
        audio = audio.set_frame_rate(reference.frame_rate)
    if audio.sample_width != reference.sample_width:
        audio = audio.set_sample_width(reference.sample_width)
    if audio.channels != reference.channels:
        if audio.channels != 1 and reference.channels != 1:
            audio = audio.set_channels(1)
        audio = audio.set_channels(reference.channels)
    return audio


def _reconstruct_spans_from_specs(
    span_specs: Any, perception: dict[str, Any], duration_sec: float
) -> list[TimeSpan]:
    if not isinstance(span_specs, list):
        return []
    spans: list[TimeSpan] = []
    speaker_segments = _normalized_speaker_segments(perception)
    words = _normalized_words(perception.get("word_timestamps") or [])
    for item in span_specs:
        if not isinstance(item, dict):
            continue
        if "start_sec" in item or "start" in item or "end_sec" in item or "end" in item:
            start_value = item.get("start_sec", item.get("start", 0.0))
            end_value = item.get("end_sec", item.get("end", duration_sec))
            raw_span = TimeSpan(
                float(start_value),
                float(end_value),
                str(item.get("label") or "param_span"),
                str(item.get("speaker_id")) if item.get("speaker_id") is not None else None,
                str(item.get("text") or ""),
                "params.spans",
            )
            span = raw_span.clipped(duration_sec) or _tail_align_transcript_span(
                raw_span, duration_sec
            )
            if span:
                spans.append(span)
            continue
        speaker_id = item.get("speaker_id")
        if speaker_id is not None:
            spans.extend(seg for seg in speaker_segments if seg.speaker_id == str(speaker_id))
    if words:
        spans = _snap_spans_to_word_boundaries(spans, words, duration_sec)
    return _clip_merge_spans_with_tail_alignment(spans, duration_sec)


def _write_manifest(work_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path = manifest.get("output_audio_path") or (manifest.get("result") or {}).get(
        "output_audio_path"
    )
    result = dict(manifest.get("result") or {})
    if output_path:
        result["output_audio_path"] = str(output_path)
    result.update(
        {
            "operation": manifest.get("operation"),
            "manifest_path": str(manifest_path),
            "selected_spans": manifest.get("selected_spans", []),
        }
    )
    if "target_speakers" in manifest:
        result["target_speakers"] = manifest["target_speakers"]
    if "segments" in manifest:
        result["segments"] = manifest["segments"]
    if "operations" in manifest:
        result["operations"] = manifest["operations"]
    if "edits" in manifest:
        result["edits"] = manifest["edits"]
    if "reorder_indices" in manifest:
        result["reorder_indices"] = manifest["reorder_indices"]
    if "reorder_records" in manifest:
        result["reorder_records"] = manifest["reorder_records"]
    if "order_source" in manifest:
        result["order_source"] = manifest["order_source"]
    if "timeline_duration_preserved" in manifest:
        result["timeline_duration_preserved"] = manifest["timeline_duration_preserved"]
    if "input_duration_sec" in manifest:
        result["input_duration_sec"] = manifest["input_duration_sec"]
    if "output_duration_sec" in manifest:
        result["output_duration_sec"] = manifest["output_duration_sec"]
    if "perception_segment_selection" in manifest:
        result["perception_segment_selection"] = manifest["perception_segment_selection"]
    if "selected_spans_source" in manifest:
        result["selected_spans_source"] = manifest["selected_spans_source"]
    if "span_padding_sec" in manifest:
        result["span_padding_sec"] = manifest["span_padding_sec"]
    if "endpoint_adjustments" in manifest:
        result["endpoint_adjustments"] = manifest["endpoint_adjustments"]
    if "reference_voice_fallback" in manifest:
        result["reference_voice_fallback"] = manifest["reference_voice_fallback"]
    for key in (
        "mode",
        "reconstruction_mode",
        "edited_artifact_scope",
        "acceptance_eligible",
        "degraded_constraints",
        "full_timeline_source_path",
    ):
        if key in manifest:
            result[key] = manifest[key]
    return result


def _normalize_operation(operation: str) -> str:
    text = str(operation or "").strip().lower()
    aliases = {
        "extract": "extract_spans",
        "selection": "extract_spans",
        "select": "extract_spans",
        "reorder": "reorder_spans",
        "reorder_segments": "reorder_spans",
        "swap_order": "reorder_spans",
        "swap_segments": "reorder_spans",
        "remove": "remove_spans",
        "delete": "remove_spans",
        "cut": "remove_spans",
        "generate_silence": "silence",
        "synthetic_silence": "silence",
        "silence_clip": "silence",
        "silent_clip": "silence",
        "mute": "mute_spans",
        "beep": "beep_spans",
        "censor": "beep_spans",
        "bleep": "beep_spans",
        "assembly": "concat",
        "append": "concat",
        "prepend": "concat",
        "concatenate": "concat",
        "speaker_scoped_local_edit": "speaker_scoped_edit",
        "speaker-scoped-edit": "speaker_scoped_edit",
        "timeline_reconstruct": "reconstruct",
    }
    return aliases.get(text, text)


def _infer_graph_operation(action: Any, state: dict[str, Any]) -> str:
    role = str(getattr(action, "output_artifact_role", "") or "").lower()
    objective = str(getattr(action, "objective", "") or "").lower()
    prompt = str(state.get("user_prompt") or "").lower()
    text = f"{role} {objective} {prompt}"
    if "speaker_scoped" in text or "speaker-scoped" in text:
        return "speaker_scoped_edit"
    if _contains(
        text,
        [
            "concatenate",
            "concat",
            "append",
            "prepend",
            "merge audio",
            "stitch",
            "拼接",
            "连接",
            "合并",
        ],
    ):
        return "concat"
    if _contains(text, ["extract", "isolate", "keep only", "只保留", "仅保留", "提取", "抽取"]):
        return "extract_spans"
    if _contains(text, ["mute", "silence the", "静音"]):
        return "mute_spans"
    if _contains(
        text,
        [
            "beep",
            "bleep",
            "curse word",
            "curse words",
            "profanity",
            "censor",
            "swear word",
            "swear words",
        ],
    ):
        return "beep_spans"
    if _contains(text, ["remove", "delete", "cut out", "去掉", "删除", "移除"]):
        return "remove_spans"
    if _contains(text, ["reconstruct", "replace only", "重建"]):
        return "reconstruct"
    return "reconstruct"


def _resolve_artifact_input(
    action: Any, artifact_paths: dict[str, str], base_audio_path: str
) -> str:
    refs = list(getattr(action, "input_artifacts", []) or [])
    if refs:
        first_ref = refs[0]
        return artifact_paths.get(
            first_ref, first_ref if Path(str(first_ref)).exists() else base_audio_path
        )
    return base_audio_path


def _edited_artifact_path(action: Any, artifact_paths: dict[str, str]) -> str | None:
    refs = list(getattr(action, "input_artifacts", []) or [])
    if len(refs) < 2:
        return None
    ref = refs[-1]
    return artifact_paths.get(ref, ref)


def _perception_from_state(state: dict[str, Any]) -> dict[str, Any]:
    perception = state.get("perception_report") or state.get("audio_perception") or {}
    return perception if isinstance(perception, dict) else {}


def _ensure_perception(
    perception: dict[str, Any],
    *,
    audio_path: str,
    output_dir: str,
    source_text: str | None,
    need_words: bool,
    need_speakers: bool,
    instruction: str,
    reference_audio_paths: list[str],
    agentic_mode: str | None = None,
) -> dict[str, Any]:
    resolved_mode = resolve_agentic_mode(agentic_mode)
    perception = _promote_model_timeline_segments(perception)
    needs_profiles = (
        needs_qwen3_omni_audio_perception(
            instruction,
            perception,
            need_words=False,
            need_speakers=need_speakers,
        )
        if perception_provider_name(resolved_mode) == QWEN3_OMNI_PROVIDER
        else needs_gemini_audio_perception(
            instruction,
            perception,
            need_words=False,
            need_speakers=need_speakers,
        )
    )
    requirements = PerceptionRequirements(
        source_text=need_words,
        word_timestamps=need_words,
        speaker_segments=need_speakers,
        speaker_profiles=needs_profiles,
        segment_descriptions=need_speakers,
    )
    if not missing_requirements(perception, requirements):
        return perception

    result = build_audio_perception_fallback(agentic_mode=resolved_mode).perceive(
        user_prompt=instruction,
        audio_path=audio_path,
        base_perception=perception,
        source_text=source_text,
        reference_audio_paths=reference_audio_paths,
        output_dir=output_dir,
        requirements=requirements,
        run_specialized=True,
        strict=False,
    )
    return result.perception


def _promote_model_timeline_segments(perception: dict[str, Any]) -> dict[str, Any]:
    """Use fallback-model timeline segments as first-class perception evidence.

    Audio fallback stores rich segment descriptions under provider-specific
    keys such as ``gemini_perception.segments`` or
    ``qwen3_omni_perception.segments``. Artifact operations can consume those
    directly, and treating them as missing evidence would trigger an unnecessary
    second fallback call that may overwrite deterministic/test-provided spans.
    """

    if perception.get("segments") or perception.get("speaker_segments"):
        return perception
    segments: list[Any] = []
    for key in ("qwen3_omni_perception", "gemini_perception"):
        model_report = perception.get(key)
        if not isinstance(model_report, dict):
            continue
        raw_segments = model_report.get("segments")
        if isinstance(raw_segments, list) and raw_segments:
            segments = raw_segments
            break
    if not segments:
        return perception
    promoted = [item for item in segments if isinstance(item, dict)]
    if not promoted:
        return perception
    updated = dict(perception)
    updated["segments"] = promoted
    return updated


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


def _normalized_speaker_segments(perception: dict[str, Any]) -> list[TimeSpan]:
    raw = perception.get("speaker_segments") or perception.get("segments") or []
    source = "speaker_segments" if perception.get("speaker_segments") else "segments"
    if not raw and isinstance(perception.get("gemini_perception"), dict):
        gemini = perception["gemini_perception"]
        raw = gemini.get("speaker_segments") or gemini.get("segments") or []
        source = (
            "gemini_perception.speaker_segments"
            if gemini.get("speaker_segments")
            else "gemini_perception.segments"
        )
    spans = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id") or item.get("speaker") or item.get("label")
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
        text = str(item.get("transcript") or item.get("text") or "")
        spans.append(TimeSpan(start_f, end_f, "speaker_segment", str(speaker_id), text, source))
    return sorted(spans, key=lambda item: (item.start_sec, item.end_sec, item.speaker_id or ""))


def _speaker_order(segments: list[TimeSpan], perception: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for seg in segments:
        if seg.speaker_id and seg.speaker_id not in seen:
            seen.add(seg.speaker_id)
            order.append(seg.speaker_id)
    if order:
        return order
    speakers = perception.get("speakers") or []
    if isinstance(speakers, list):
        for item in speakers:
            speaker_id = item.get("speaker_id") if isinstance(item, dict) else item
            if speaker_id is not None and str(speaker_id) not in seen:
                seen.add(str(speaker_id))
                order.append(str(speaker_id))
    return order


def _speaker_profiles(perception: dict[str, Any]) -> dict[str, str]:
    profiles: dict[str, list[str]] = {}
    for collection_name, raw, _weight in _speaker_profile_collections(perception):
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            speaker_id = (
                item.get("speaker_id") or item.get("speaker") or item.get("label") or item.get("id")
            )
            if speaker_id is None:
                continue
            pieces = [
                speaker_id,
                item.get("description"),
                item.get("perceived_gender"),
                item.get("gender"),
                item.get("age"),
                item.get("role"),
                item.get("transcript"),
                item.get("text"),
            ]
            value = " ".join(str(piece) for piece in pieces if piece)
            if value:
                profiles.setdefault(str(speaker_id), []).append(value)
    return {speaker_id: " | ".join(values).lower() for speaker_id, values in profiles.items()}


def _speaker_descriptive_profiles(perception: dict[str, Any]) -> dict[str, str]:
    profiles: dict[str, list[str]] = {}
    raw = perception.get("speakers") or []
    if isinstance(raw, list):
        _collect_speaker_description_profiles(raw, profiles)
    gemini = perception.get("gemini_perception")
    if isinstance(gemini, dict):
        raw = gemini.get("speakers") or []
        if isinstance(raw, list):
            _collect_speaker_description_profiles(raw, profiles)
    return {speaker_id: " | ".join(values).lower() for speaker_id, values in profiles.items()}


def _collect_speaker_description_profiles(raw: list[Any], profiles: dict[str, list[str]]) -> None:
    for item in raw:
        if not isinstance(item, dict):
            continue
        speaker_id = (
            item.get("speaker_id") or item.get("speaker") or item.get("label") or item.get("id")
        )
        if speaker_id is None:
            continue
        pieces = [
            speaker_id,
            item.get("description"),
            item.get("perceived_gender"),
            item.get("gender"),
            item.get("age"),
            item.get("role"),
        ]
        value = " ".join(str(piece) for piece in pieces if piece)
        if value:
            profiles.setdefault(str(speaker_id), []).append(value)


def _speaker_profile_collections(
    perception: dict[str, Any],
) -> list[tuple[str, list[Any], int]]:
    collections: list[tuple[str, list[Any], int]] = []
    for collection_name in ["speakers", "segments", "speaker_segments"]:
        raw = perception.get(collection_name) or []
        if isinstance(raw, list):
            collections.append((collection_name, raw, 1))
    gemini = perception.get("gemini_perception")
    if isinstance(gemini, dict):
        for collection_name in ["speakers", "segments", "speaker_segments"]:
            raw = gemini.get(collection_name) or []
            if isinstance(raw, list):
                collections.append((f"gemini_perception.{collection_name}", raw, 3))
    return collections


def _explicit_speaker_attribute_matches(
    perception: dict[str, Any],
    speaker_id: str,
    attr_kind: str,
) -> bool | None:
    if attr_kind not in {"female", "male"}:
        return None
    votes = {"female": 0, "male": 0}
    for _collection_name, raw, weight in _speaker_profile_collections(perception):
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            item_speaker = (
                item.get("speaker_id") or item.get("speaker") or item.get("label") or item.get("id")
            )
            if str(item_speaker) != str(speaker_id):
                continue
            for key in ("perceived_gender", "gender"):
                value = item.get(key)
                if not isinstance(value, str):
                    continue
                gender = _explicit_gender_label(value)
                if gender:
                    votes[gender] += weight
    if votes["female"] == 0 and votes["male"] == 0:
        return None
    other = "male" if attr_kind == "female" else "female"
    if votes[attr_kind] == votes[other]:
        return None
    return votes[attr_kind] > votes[other]


def _explicit_gender_label(value: str) -> str | None:
    lowered = value.lower()
    if re.search(r"\b(female|woman|girl)\b", lowered) or _contains(
        lowered, ["女性", "女声", "女说话"]
    ):
        return "female"
    if re.search(r"\b(male|man|boy)\b", lowered) or _contains(lowered, ["男性", "男声", "男说话"]):
        return "male"
    return None


def _target_name_mentions(instruction: str) -> list[str]:
    names: list[str] = []
    patterns = [
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)['’]s\s+(?:speech|voice|words|utterance|utterances|line|lines|answer|response)\b",
        r"\b(?:speech|voice|words|utterance|utterances|line|lines|answer|response)\s+(?:of|from|by)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b",
        r"\b(?:speaker|person|character|role)\s+(?:named|called)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, instruction):
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            value = _strip_leading_instruction_words_from_name(value)
            if value and value.lower() not in {"speaker", "audio"} and value not in names:
                names.append(value)
    return names


def _strip_leading_instruction_words_from_name(value: str) -> str:
    stop_words = {
        "extract",
        "isolate",
        "remove",
        "delete",
        "mute",
        "change",
        "make",
        "convert",
        "replace",
        "identify",
        "find",
        "keep",
        "lower",
        "raise",
        "increase",
        "decrease",
    }
    words = value.split()
    while len(words) > 1 and words[0].lower() in stop_words:
        words = words[1:]
    return " ".join(words)


def _speaker_profile_matches_name(profile: str, name: str) -> bool:
    if not profile or not name:
        return False
    normalized_profile = re.sub(r"[^a-z0-9]+", " ", profile.lower()).strip()
    normalized_name = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    if not normalized_name:
        return False
    return bool(
        re.search(
            rf"\b(?:identified as|named|called|role|character)\s+(?:the\s+)?{re.escape(normalized_name)}\b",
            normalized_profile,
        )
        or re.search(
            rf"\b{re.escape(normalized_name)}\s+(?:speaker|voice|character|role)\b",
            normalized_profile,
        )
    )


def _explicit_speaker_ids(instruction: str, speaker_order: list[str]) -> list[str]:
    ids = []
    for speaker_id in speaker_order:
        if re.search(rf"\b{re.escape(speaker_id.lower())}\b", instruction.lower()):
            ids.append(speaker_id)
    if ids:
        return ids
    for match in re.finditer(r"\bspeaker[_\s-]?([0-9]+)\b", instruction.lower()):
        ordinal = int(match.group(1))
        if 1 <= ordinal <= len(speaker_order):
            ids.append(speaker_order[ordinal - 1])
    return ids


def _first_speaker_ordinal(text: str) -> int | None:
    if not _contains(text, ["speaker", "说话人", "voice", "声"]):
        return None
    mentions: list[tuple[int, int]] = []
    for token, value in ORDINAL_WORDS.items():
        if token in {"one", "two", "three", "four"}:
            patterns = [
                rf"\b(?:speaker|voice|person)\s+(?:number\s+)?{re.escape(token)}\b",
                rf"\bnumber\s+{re.escape(token)}\s+(?:speaker|voice|person)\b",
                rf"\b{re.escape(token)}\s+(?:speaker|voice|person)\b",
            ]
            matches = [match for pattern in patterns for match in re.finditer(pattern, text)]
        else:
            matches = list(re.finditer(re.escape(token), text))
        for match in matches:
            mentions.append((match.start(), value))
    if not mentions:
        return None
    return sorted(mentions)[0][1]


def _target_attribute_kind(text: str) -> str | None:
    if _contains(text, ["customer", "client", "顾客", "客户"]):
        return "customer"
    if _contains(text, ["employee", "shop owner", "owner", "clerk", "staff", "老板", "店员"]):
        return "employee"
    if _contains(
        text,
        [
            "police officer",
            "officer",
            "police",
            "cop",
            "policeman",
            "policewoman",
            "警察",
            "警官",
        ],
    ):
        return "police_officer"
    if _contains(text, ["robber", "thief", "劫匪", "小偷"]):
        return "robber"
    if _contains(text, ["victim", "受害者"]):
        return "victim"
    if _contains(text, ["narrator", "narration", "旁白", "讲述者"]):
        return "narrator"
    if _contains(
        text,
        [
            "singing voice",
            "singing vocal",
            "singing vocals",
            "singer",
            "vocals",
            "歌声",
            "演唱声",
            "唱歌声",
        ],
    ):
        return "singing"
    if _contains(text, ["whisper", "whispering", "whispered", "耳语", "低语", "悄声"]):
        return "whispering"
    if _contains(text, ["little girl", "girl's voice", "female child", "小女孩", "女孩"]):
        return "little_girl"
    if _contains(text, ["little boy", "boy's voice", "male child", "小男孩", "男孩"]):
        return "little_boy"
    if _contains(text, ["baby", "infant", "toddler", "婴儿", "宝宝", "幼儿"]):
        return "baby"
    if _contains(text, ["child", "kid", "children", "儿童", "小孩", "童声"]):
        return "child"
    if _contains(text, ["adult", "成年人", "成人"]):
        return "adult"
    if _contains(text, ["female", "woman", "girl", "女说话", "女声", "女性"]):
        return "female"
    if _contains(text, ["male", "man", "boy", "男说话", "男声", "男性"]):
        return "male"
    return None


def _speaker_profile_matches(profile: str, attr_kind: str) -> bool:
    if not profile:
        return False
    if attr_kind == "little_girl":
        return _contains(profile, ["little girl", "girl", "女孩", "小女孩"]) or (
            _contains(profile, ["female", "woman", "女", "女性"])
            and _contains(profile, ["child", "kid", "children", "儿童", "小孩"])
        )
    if attr_kind == "little_boy":
        return _contains(profile, ["little boy", "boy", "男孩", "小男孩"]) or (
            _contains(profile, ["male", "man", "男", "男性"])
            and _contains(profile, ["child", "kid", "children", "儿童", "小孩"])
        )
    if attr_kind == "baby":
        return _contains(profile, ["baby", "infant", "toddler", "婴儿", "宝宝", "幼儿"])
    if attr_kind == "child":
        return _contains(
            profile,
            [
                "child",
                "kid",
                "children",
                "baby",
                "infant",
                "toddler",
                "儿童",
                "小孩",
                "童声",
                "婴儿",
                "宝宝",
                "幼儿",
                "girl",
                "boy",
                "女孩",
                "男孩",
            ],
        )
    if attr_kind == "adult":
        return _contains(profile, ["adult", "成年人", "成人"])
    if attr_kind == "female":
        mentions = _gender_mentions(profile)
        return "female" in mentions and "male" not in mentions
    if attr_kind == "male":
        mentions = _gender_mentions(profile)
        return "male" in mentions and "female" not in mentions
    if attr_kind == "customer":
        return _contains(
            profile,
            ["customer", "client", "ordering food", "ordering", "顾客", "客户", "点单"],
        )
    if attr_kind == "employee":
        return _contains(
            profile,
            [
                "employee",
                "shop owner",
                "owner",
                "clerk",
                "staff",
                "server",
                "老板",
                "店员",
                "服务员",
            ],
        )
    if attr_kind == "police_officer":
        return _contains(
            profile,
            [
                "police officer",
                "officer",
                "police",
                "cop",
                "policeman",
                "policewoman",
                "警察",
                "警官",
            ],
        )
    if attr_kind == "robber":
        return _contains(profile, ["robber", "thief", "劫匪", "小偷"])
    if attr_kind == "victim":
        return _contains(profile, ["victim", "受害者"])
    if attr_kind == "narrator":
        return _contains(profile, ["narrator", "narration", "旁白", "讲述"])
    if attr_kind == "singing":
        return _contains(
            profile,
            ["singing", "singer", "song", "vocal", "vocals", "歌声", "演唱", "唱歌"],
        )
    if attr_kind == "whispering":
        return _contains(
            profile,
            [
                "whisper",
                "whispering",
                "whispered",
                "raspy whisper",
                "耳语",
                "低语",
                "悄声",
            ],
        )
    return False


def _gender_mentions(text: str) -> set[str]:
    lowered = text.lower()
    result: set[str] = set()
    if re.search(r"\b(female|woman|girl)\b", lowered) or _contains(
        lowered, ["女性", "女声", "女说话"]
    ):
        result.add("female")
    if re.search(r"\b(male|man|boy)\b", lowered) or _contains(lowered, ["男性", "男声", "男说话"]):
        result.add("male")
    return result


def _explicit_time_spans(instruction: str) -> list[TimeSpan]:
    spans = []
    for match in re.finditer(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|second|seconds|秒)\s*(?:-|to|到|至)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|second|seconds|秒)?",
        instruction,
        flags=re.IGNORECASE,
    ):
        start = float(match.group(1))
        end = float(match.group(2))
        spans.append(TimeSpan(start, end, "explicit_time", source="instruction"))
    return spans


def _phrase_spans(instruction: str, words: list[dict[str, Any]]) -> list[TimeSpan]:
    if not words:
        return []
    phrases = _phrase_candidates_from_instruction(instruction)
    if not phrases:
        return []
    all_occurrences = _wants_all_phrase_occurrences(instruction)
    normalized_words = [_normalize_token(word["text"]) for word in words]
    spans = []
    for phrase in phrases:
        phrase_tokens = [
            _normalize_token(token) for token in re.findall(r"[\w'-]+|[\u4e00-\u9fff]+", phrase)
        ]
        phrase_tokens = [token for token in phrase_tokens if token]
        if not phrase_tokens:
            continue
        matched = False
        for index in range(0, len(words) - len(phrase_tokens) + 1):
            if normalized_words[index : index + len(phrase_tokens)] == phrase_tokens:
                selected = words[index : index + len(phrase_tokens)]
                spans.append(
                    TimeSpan(
                        float(selected[0]["start_sec"]),
                        float(selected[-1]["end_sec"]),
                        f"phrase:{phrase}",
                        _single_speaker_id(selected),
                        _render_words(selected),
                        "word_timestamps",
                    )
                )
                matched = True
                if not all_occurrences:
                    break
        if not matched:
            compact_span = _compact_phrase_span(phrase, words, normalized_words)
            if compact_span:
                spans.append(compact_span)
    return spans


def _phrase_occurrences_in_words(
    phrase: str, words: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    phrase_tokens = [
        _normalize_token(token) for token in re.findall(r"[\w'-]+|[\u4e00-\u9fff]+", phrase)
    ]
    phrase_tokens = [token for token in phrase_tokens if token]
    if not phrase_tokens or len(words) < len(phrase_tokens):
        return []
    normalized_words = [_normalize_token(str(word.get("text") or "")) for word in words]
    matches: list[list[dict[str, Any]]] = []
    for index in range(0, len(words) - len(phrase_tokens) + 1):
        if normalized_words[index : index + len(phrase_tokens)] == phrase_tokens:
            matches.append(words[index : index + len(phrase_tokens)])
    return matches


def _compact_phrase_span(
    phrase: str,
    words: list[dict[str, Any]],
    normalized_words: list[str],
) -> TimeSpan | None:
    normalized_phrase = _normalize_text_for_contains(phrase)
    if not normalized_phrase:
        return None
    max_window = max(1, min(len(words), len(normalized_phrase) + 6))
    for start in range(len(words)):
        compact = ""
        selected: list[dict[str, Any]] = []
        for end in range(start, min(len(words), start + max_window)):
            token = normalized_words[end]
            if not token:
                continue
            compact += token
            selected.append(words[end])
            if compact == normalized_phrase:
                return TimeSpan(
                    float(selected[0]["start_sec"]),
                    float(selected[-1]["end_sec"]),
                    f"phrase:{phrase}",
                    _single_speaker_id(selected),
                    _render_words(selected),
                    "word_timestamps+compact_phrase",
                )
            if not normalized_phrase.startswith(compact):
                break
    return None


def _phrase_spans_from_segments(
    instruction: str,
    segments: list[TimeSpan],
    duration_sec: float | None = None,
    *,
    prefer_subspan: bool = False,
) -> list[TimeSpan]:
    phrases = _phrase_candidates_from_instruction(instruction)
    if not phrases or not segments:
        return []
    all_occurrences = _wants_all_phrase_occurrences(instruction)
    spans = []
    for phrase in phrases:
        normalized_phrase = _normalize_text_for_contains(phrase)
        if not normalized_phrase:
            continue
        for segment in segments:
            if normalized_phrase in _normalize_text_for_contains(segment.text):
                if not _segment_matches_phrase_conditions(instruction, phrase, segment):
                    continue
                fallback_span = TimeSpan(
                    segment.start_sec,
                    segment.end_sec,
                    f"phrase_segment:{phrase}",
                    segment.speaker_id,
                    segment.text,
                    f"{segment.source}+segment_transcript",
                )
                if duration_sec is not None and prefer_subspan:
                    span = _phrase_subspan_from_segment(
                        fallback_span, [phrase], instruction, duration_sec
                    )
                    if span is None:
                        span = (
                            _tail_align_transcript_span(fallback_span, duration_sec)
                            or fallback_span
                        )
                elif duration_sec is not None:
                    span = _tail_align_transcript_span(fallback_span, duration_sec) or fallback_span
                else:
                    span = fallback_span
                spans.append(span)
                if not all_occurrences:
                    break
    return spans


def _phrase_candidates_from_instruction(instruction: str) -> list[str]:
    phrases = _quoted_strings(instruction)
    if not phrases:
        plain = _plain_phrase_candidate_from_instruction(instruction)
        phrases = [plain] if plain else []
    return _dedupe_phrase_candidates(_expand_phrase_transcript_variants(phrases))


def _plain_phrase_candidate_from_instruction(instruction: str) -> str | None:
    candidate = re.sub(r"\s+", " ", instruction).strip(" \t\r\n'\"“”.,;:!?")
    if not candidate:
        return None
    if _contains_cjk(candidate):
        return candidate if 4 <= len(candidate) <= 80 else None

    tokens = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", candidate)
    if len(tokens) < 4 or len(tokens) > 18:
        return None
    lowered = candidate.lower()
    if re.search(
        r"\b(?:change|convert|turn|make|modify|edit|replace|remove|delete|extract|isolate|keep|mute|mask|translate|"
        r"separate|enhance|denoise|add|insert|overlay|mix|splice|reconstruct)\b",
        lowered,
    ):
        return None
    if _contains(
        lowered,
        [
            " final sentence",
            " last sentence",
            " first sentence",
            " second sentence",
            " target speaker",
            " source speaker",
            " spoken voice",
            " speech segment",
            " speaker's voice",
        ],
    ):
        return None
    return candidate


def _expand_phrase_transcript_variants(phrases: list[str]) -> list[str]:
    variants: list[str] = []
    for phrase in phrases:
        variants.append(phrase)
        variants.append(
            re.sub(
                r"\b([A-Za-z][A-Za-z0-9-]*)\s+has\s+been\b",
                r"\1's been",
                phrase,
                flags=re.IGNORECASE,
            )
        )
        variants.append(
            re.sub(
                r"\b([A-Za-z][A-Za-z0-9-]*)'s\s+been\b",
                r"\1 has been",
                phrase,
                flags=re.IGNORECASE,
            )
        )
        variants.append(
            re.sub(
                r"\b([A-Za-z][A-Za-z0-9-]*)\s+is\s+not\b",
                r"\1's not",
                phrase,
                flags=re.IGNORECASE,
            )
        )
        variants.append(
            re.sub(
                r"\b([A-Za-z][A-Za-z0-9-]*)'s\s+not\b",
                r"\1 is not",
                phrase,
                flags=re.IGNORECASE,
            )
        )
    return variants


def _dedupe_phrase_candidates(phrases: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for phrase in phrases:
        candidate = re.sub(r"\s+", " ", phrase).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _wants_all_phrase_occurrences(instruction: str) -> bool:
    text = instruction.lower()
    return bool(re.search(r"\b(?:all|every|each)\b", text)) or _contains(
        text,
        [
            "all instances",
            "every instance",
            "each instance",
            "all occurrences",
            "every occurrence",
            "each occurrence",
            "all times",
            "every time",
            "所有",
            "全部",
            "每个",
            "每次",
        ],
    )


def _segment_matches_phrase_conditions(instruction: str, phrase: str, segment: TimeSpan) -> bool:
    """Filter phrase matches when the instruction also names a source-side property."""

    text = instruction.lower()
    if _wants_source_questioning_tone(text):
        evidence = " ".join(
            part for part in [segment.text, segment.label, segment.source] if part
        ).lower()
        if _phrase_has_trailing_punctuation(segment.text, phrase, {"?", "？"}):
            return True
        return _contains(
            evidence,
            [
                "questioning",
                "question-like",
                "question like",
                "interrogative",
                "rising intonation",
                "rising tone",
                "asking a question",
                "像提问",
                "疑问",
                "升调",
            ],
        )
    return True


def _wants_source_questioning_tone(text: str) -> bool:
    has_questioning = _contains(
        text,
        [
            "questioning tone",
            "questioning intonation",
            "interrogative intonation",
            "interrogative tone",
            "rising intonation",
            "rising tone",
            "question-like",
            "疑问语气",
            "疑问语调",
            "升调",
        ],
    )
    if not has_questioning:
        return False
    return _contains(
        text,
        ["spoken with", "from", "change", "convert", "turn", "变成", "改成", "转换"],
    )


def _phrase_has_trailing_punctuation(text: str, phrase: str, marks: set[str]) -> bool:
    match = _normalized_phrase_match(text, phrase)
    if not match:
        return False
    _, end = match
    suffix = text[end:]
    match_suffix = re.match(r"\s*([^\w\s\u4e00-\u9fff])", suffix)
    return bool(match_suffix and match_suffix.group(1) in marks)


def _refine_phrase_segment_spans_for_tagged_edit(
    spans: list[TimeSpan],
    *,
    selection_instruction: str,
    grounding_instruction: str,
    operations: list[str],
    duration_sec: float,
) -> list[TimeSpan]:
    if not spans or not ({"pitch_rate_edit", "emotion_style_edit"} & set(operations)):
        return spans
    instruction = selection_instruction if selection_instruction else grounding_instruction
    phrases = _quoted_strings(instruction) or _quoted_strings(grounding_instruction)
    if not phrases:
        return spans
    if not _should_refine_phrase_segment_spans(instruction, grounding_instruction, operations):
        return spans

    refined: list[TimeSpan] = []
    changed = False
    for span in spans:
        phrase_span = _phrase_subspan_from_segment(span, phrases, instruction, duration_sec)
        if phrase_span is None:
            refined.append(span)
            continue
        refined.append(phrase_span)
        changed = True
    if not changed:
        return spans
    return _clip_merge_spans(refined, duration_sec, merge_gap_sec=0.0)


def _should_refine_phrase_segment_spans(
    selection_instruction: str,
    grounding_instruction: str,
    operations: list[str],
) -> bool:
    evidence = f"{selection_instruction} {grounding_instruction}".lower()
    if {"pitch_rate_edit", "emotion_style_edit"} & set(operations) and _contains(
        evidence,
        [
            "questioning tone",
            "questioning intonation",
            "declarative tone",
            "interrogative",
            "rising intonation",
            "falling intonation",
            "emphasis",
            "emphasize",
            "stress",
            "pitch",
            "tone",
            "intonation",
            "语气",
            "语调",
            "重音",
            "强调",
        ],
    ):
        return True
    return _contains(evidence, ["phrase", "word", "term", "短语", "词"])


def _phrase_subspan_from_segment(
    span: TimeSpan,
    phrases: list[str],
    instruction: str,
    duration_sec: float,
) -> TimeSpan | None:
    if not span.text.strip() or span.duration_sec <= 0:
        return None
    for phrase in phrases:
        match = _normalized_phrase_match(span.text, phrase)
        if not match:
            continue
        start_char, end_char = match
        trailing_marks = {".", "!", "?", "。", "！", "？"}
        if _wants_source_questioning_tone(instruction.lower()):
            trailing_marks = {"?", "？"}
        end_char = _extend_span_text_over_trailing_marks(span.text, end_char, trailing_marks)
        selected_text = span.text[start_char:end_char].strip()
        if not selected_text:
            selected_text = phrase
        text_len = max(1, len(span.text))
        start = span.start_sec + span.duration_sec * (start_char / text_len)
        end = span.start_sec + span.duration_sec * (end_char / text_len)
        pad = min(0.06, max(0.02, span.duration_sec * 0.04))
        start = max(span.start_sec, start - pad)
        end = min(span.end_sec, end + pad)
        min_duration = min(span.duration_sec, max(MIN_TAGGED_EDIT_SEGMENT_SEC + 0.05, 0.32))
        if end - start < min_duration:
            center = (start + end) / 2.0
            start = max(span.start_sec, center - min_duration / 2.0)
            end = min(span.end_sec, start + min_duration)
            start = max(span.start_sec, end - min_duration)
        if end <= start:
            return None
        return TimeSpan(
            start,
            min(end, duration_sec),
            f"{span.label}:phrase_subspan",
            span.speaker_id,
            selected_text,
            f"{span.source}+phrase_subspan_estimate",
        ).clipped(duration_sec)
    return None


def _normalized_phrase_match(text: str, phrase: str) -> tuple[int, int] | None:
    norm_text, mapping = _normalize_text_with_char_map(text)
    norm_phrase, _ = _normalize_text_with_char_map(phrase)
    if not norm_text or not norm_phrase:
        return None
    start_norm = norm_text.find(norm_phrase)
    if start_norm < 0:
        return None
    end_norm = start_norm + len(norm_phrase) - 1
    return mapping[start_norm], mapping[end_norm] + 1


def _normalize_text_with_char_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(text.lower()):
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            chars.append(char)
            mapping.append(index)
    return "".join(chars), mapping


def _extend_span_text_over_trailing_marks(text: str, end_char: int, marks: set[str]) -> int:
    index = end_char
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] in marks:
        return index + 1
    return end_char


def _tail_align_transcript_span(span: TimeSpan, duration_sec: float) -> TimeSpan | None:
    """Keep transcript-derived final spans usable when perception timecodes drift past audio end."""

    clipped = span.clipped(duration_sec)
    if span.end_sec <= duration_sec:
        return clipped
    if duration_sec <= 0 or span.duration_sec <= 0:
        return clipped
    drift_sec = span.end_sec - duration_sec
    if drift_sec > max(3.0, duration_sec * 0.5):
        return clipped
    estimated_duration = min(span.duration_sec, duration_sec)
    end = duration_sec
    start = max(0.0, end - estimated_duration)
    if end <= start:
        return clipped
    source = span.source
    if "tail_aligned" not in source:
        source = f"{source}+tail_aligned" if source else "tail_aligned"
    return TimeSpan(start, end, span.label, span.speaker_id, span.text, source)


def _semantic_segment_spans(
    instruction: str,
    perception: dict[str, Any],
    *,
    target_speakers: list[str] | None = None,
) -> list[TimeSpan]:
    selector_text = _semantic_segment_selector_text(instruction)
    text = selector_text.lower()
    groups: list[list[str]] = []
    wants_chinese_utterances = _contains(
        text,
        [
            "chinese utterance",
            "chinese utterances",
            "chinese speech",
            "chinese part",
            "chinese parts",
            "mandarin utterance",
            "mandarin utterances",
            "中文",
            "汉语",
            "普通话",
        ],
    )
    if _contains(text, ["internal monologue", "internal monologues", "monologue", "monologues"]):
        groups.append(["internal monologue", "monologue"])
    if _contains(text, ["reverb", "echo", "echoey", "echoing"]):
        groups.append(["reverb", "echo", "echoey", "echoing"])
    if _contains(text, ["whisper", "whispered", "whispering", "低语", "耳语"]):
        groups.append(["whisper", "whispered", "whispering", "低语", "耳语"])
    if _contains(
        text,
        [
            "singing voice",
            "singing vocal",
            "singing vocals",
            "singer",
            "vocals",
            "歌声",
            "演唱声",
            "唱歌声",
        ],
    ):
        groups.append(["singing", "singer", "song", "vocal", "vocals", "歌声", "演唱", "唱歌"])
    if _contains(text, ["rap", "rapped", "rapping", "说唱"]):
        groups.append(["rap", "rapped", "rapping", "说唱"])
    if _contains(text, ["intercom", "walkie-talkie", "walkie talkie"]):
        groups.append(["intercom", "walkie-talkie", "walkie talkie"])
    if _contains(text, ["underwater", "under water", "gurgling", "bubbling"]):
        groups.append(["underwater", "under water", "gurgling", "bubbling"])
    if _contains(text, ["telephone", "phone", "radio", "filtered"]):
        groups.append(["telephone", "phone", "radio", "filtered"])
    if _contains(text, ["strong wind", "windy", "wind"]):
        groups.append(["strong wind", "windy", "wind"])
    if _contains(text, ["counting numbers", "counting", "counts numbers"]):
        groups.append(["counting", "number", "numbers"])
    if _contains(text, ["far away", "far-away", "distant", "from afar"]):
        groups.append(["far away", "far-away", "distant", "background"])
    if _contains(
        text,
        [
            "whistle-level",
            "whistle level",
            "whistle",
            "high-pitched",
            "high pitched",
            "sharp",
            "intensity",
        ],
    ):
        groups.append(
            [
                "whistle-level",
                "whistle level",
                "whistle",
                "high-pitched",
                "high pitched",
                "sharp",
                "intensity",
            ]
        )
    if _contains(text, ["loudness", "volume", "loud", "quiet", "soft"]):
        groups.append(
            [
                "loudness",
                "volume",
                "loud",
                "quiet",
                "soft",
                "louder",
                "quieter",
                "softly",
            ]
        )
    if _contains(
        text,
        [
            "spanish part",
            "spanish parts",
            "spanish speech",
            "spanish segment",
            "spanish segments",
        ],
    ):
        groups.append(["spanish", "español"])
    if _contains(
        text,
        [
            "happy",
            "happier",
            "happiest",
            "joyful",
            "cheerful",
            "excited",
            "excitement",
            "glad",
            "delighted",
            "开心",
            "高兴",
            "兴奋",
            "快乐",
        ],
    ):
        groups.append(
            [
                "happy",
                "happier",
                "happiest",
                "joyful",
                "cheerful",
                "excited",
                "excitement",
                "glad",
                "delighted",
                "开心",
                "高兴",
                "兴奋",
                "快乐",
                "太好了",
            ]
        )
    if _contains(
        text,
        [
            "sad",
            "sadder",
            "saddest",
            "depressed",
            "sorrowful",
            "tearful",
            "crying",
            "unhappy",
            "难过",
            "悲伤",
            "沮丧",
            "哭",
        ],
    ):
        groups.append(
            [
                "sad",
                "sadder",
                "saddest",
                "depressed",
                "sorrowful",
                "tearful",
                "crying",
                "unhappy",
                "难过",
                "悲伤",
                "沮丧",
                "哭",
            ]
        )
    if _contains(
        text,
        [
            "angry",
            "anger",
            "angrier",
            "angriest",
            "mad",
            "furious",
            "irritated",
            "annoyed",
            "生气",
            "愤怒",
            "恼火",
        ],
    ):
        groups.append(
            [
                "angry",
                "anger",
                "angrier",
                "angriest",
                "mad",
                "furious",
                "irritated",
                "annoyed",
                "生气",
                "愤怒",
                "恼火",
            ]
        )
    if _contains(
        text,
        [
            "scared",
            "frightened",
            "fearful",
            "afraid",
            "terrified",
            "nervous",
            "害怕",
            "恐惧",
            "紧张",
        ],
    ):
        groups.append(
            [
                "scared",
                "frightened",
                "fearful",
                "afraid",
                "terrified",
                "nervous",
                "害怕",
                "恐惧",
                "紧张",
            ]
        )
    if _contains(
        text,
        [
            "calm",
            "calmer",
            "calmest",
            "neutral",
            "soothing",
            "relaxed",
            "平静",
            "冷静",
            "舒缓",
        ],
    ):
        groups.append(
            [
                "calm",
                "calmer",
                "calmest",
                "neutral",
                "soothing",
                "relaxed",
                "平静",
                "冷静",
                "舒缓",
            ]
        )
    if not groups and not wants_chinese_utterances:
        return []

    spans: list[TimeSpan] = []
    target = {str(speaker_id) for speaker_id in target_speakers or []}
    for item in _raw_timeline_segments(perception):
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        speaker_id = item.get("speaker_id") or item.get("speaker") or item.get("label")
        if target and (speaker_id is None or str(speaker_id) not in target):
            continue
        evidence = " ".join(
            str(item.get(key) or "")
            for key in ["description", "transcript", "text", "label", "event", "type"]
        ).lower()
        if wants_chinese_utterances and not (
            _contains_cjk(str(item.get("transcript") or item.get("text") or ""))
            or _contains(evidence, ["chinese", "mandarin", "中文", "汉语", "普通话"])
        ):
            continue
        if all(any(token in evidence for token in group) for group in groups):
            spans.append(
                TimeSpan(
                    start_f,
                    end_f,
                    "semantic_segment",
                    str(speaker_id) if speaker_id is not None else None,
                    str(
                        item.get("transcript") or item.get("text") or item.get("description") or ""
                    ),
                    "semantic_segment_descriptions",
                )
            )
    if wants_chinese_utterances and any(_contains_cjk(span.text) for span in spans):
        spans = [span for span in spans if _contains_cjk(span.text)]
    return spans


def _semantic_segment_selector_text(instruction: str) -> str:
    """Keep source-side properties for edits like "turn the happy voice into sad"."""

    text = instruction.strip()
    patterns = [
        r"\b(?:turn|change|convert|transform)\s+(.+?)\s+(?:into|to)\b",
        r"\b(?:make)\s+(.+?)\s+(?:sound|become)\b",
        r"\bfrom\s+(.+?)\s+(?:to|into)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ,.;:")
            if candidate:
                return candidate
    return text


def _select_spans_with_perception_model(
    *,
    instruction: str,
    perception: dict[str, Any],
    input_audio_path: str,
    output_dir: str,
    duration_sec: float,
    action: str,
    agentic_mode: str | None = None,
    selector_client: OpenAICompatibleChatClient | None = None,
    input_audio_ref: str | None = None,
    reference_audio_paths: list[str] | None = None,
    reference_audio_refs: list[str] | None = None,
    semantic_targets: list[dict[str, str]] | None = None,
    force_model_review: bool = False,
    aligned_candidate_segments: list[dict[str, Any]] | None = None,
    include_default_candidates: bool = True,
    authoritative_candidate_bounds: bool = False,
) -> tuple[list[TimeSpan], dict[str, Any]]:
    """Ask the audio perception model to choose aligned segments matching a request.

    This is the generic fallback for requests like "extract speech with X
    property": deterministic alignment provides candidate spans, and Gemini
    judges whether each candidate satisfies the requested audio/semantic
    property.  The returned spans are still deterministic timeline cuts.
    """

    if os.getenv("X_EDITOR_ENABLE_PERCEPTION_SEGMENT_SELECTOR", "1").lower() in {
        "0",
        "false",
        "no",
    }:
        return [], {
            "status": "skipped",
            "reason": "disabled",
            "model_call_attempted": False,
        }
    if (
        not reference_audio_paths
        and not force_model_review
        and not _should_use_perception_segment_selector(instruction, action)
    ):
        return [], {
            "status": "skipped",
            "reason": "instruction does not require model-scored segment selection",
            "model_call_attempted": False,
        }
    resolved_mode = resolve_agentic_mode(agentic_mode)
    provider_label = _segment_selector_provider_label(resolved_mode)
    service_unavailable_reason = (
        None
        if selector_client is not None
        else _segment_selector_service_unavailable_reason(resolved_mode)
    )
    if service_unavailable_reason:
        return [], {
            "status": "unavailable",
            "reason": service_unavailable_reason,
            "provider": provider_label,
            "model_call_attempted": False,
        }

    aligned_candidates = _provided_segment_candidates(
        aligned_candidate_segments,
        duration_sec=duration_sec,
    )
    default_candidates = (
        _segment_property_candidates(perception, duration_sec)
        if include_default_candidates or not aligned_candidates
        else []
    )
    candidates = _dedupe_segment_candidates(
        [*aligned_candidates, *default_candidates],
        duration_sec,
    )
    if authoritative_candidate_bounds:
        candidates = [
            replace(candidate, authoritative_bounds=True) for candidate in candidates
        ]
    if force_model_review and include_default_candidates:
        coverage = observation_scope(
            ({"start_sec": c.span.start_sec, "end_sec": c.span.end_sec} for c in candidates),
            duration_sec,
        )
        candidates.extend(
            SegmentCandidate(
                segment_id=f"unobserved_interval_{index:03d}",
                span=TimeSpan(gap["start_sec"], gap["end_sec"]),
                description="Previously unobserved source interval; locate any matching subpart.",
                source="source_audio_geometry",
                authoritative_bounds=False,
            )
            for index, gap in enumerate(coverage["unobserved_intervals"])
        )
    if not candidates:
        return [], {
            "status": "skipped",
            "reason": "no aligned candidate segments",
            "model_call_attempted": False,
        }

    max_candidates = int(os.getenv("X_EDITOR_PERCEPTION_SEGMENT_SELECTOR_MAX_CANDIDATES", "24"))
    candidate_limit = max(1, max_candidates)
    if len(aligned_candidates) > candidate_limit:
        return [], {
            "status": "failed",
            "reason": (
                "aligned typed candidates exceed the configured candidate rubric budget"
            ),
            "candidate_count": len(candidates),
            "aligned_candidate_count": len(aligned_candidates),
            "candidate_limit": candidate_limit,
            "provider": provider_label,
            "model_call_attempted": False,
        }
    candidates = candidates[:candidate_limit]
    work_dir = Path(output_dir) / "perception_segment_selector"
    work_dir.mkdir(parents=True, exist_ok=True)

    model_call_attempted = False
    try:
        audio = load_audio_segment(input_audio_path)
        audio_paths = _export_segment_candidate_clips(audio, candidates, work_dir)
        model_call_attempted = True
        response = _call_segment_selector_model(
            instruction=instruction,
            action=action,
            candidates=candidates,
            audio_paths=audio_paths,
            agentic_mode=resolved_mode,
            selector_client=selector_client,
            input_audio_ref=input_audio_ref,
            reference_audio_paths=reference_audio_paths,
            reference_audio_refs=reference_audio_refs,
            semantic_targets=semantic_targets,
        )
    except Exception as exc:
        return [], {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "candidate_count": len(candidates),
            "provider": provider_label,
            "model_call_attempted": model_call_attempted,
        }

    by_id = {candidate.segment_id: candidate for candidate in candidates}
    selected, selected_ids, judgements = _selected_spans_from_model_response(
        response, by_id, duration_sec
    )
    selected = (
        _clip_spans_preserving_boundaries(selected, duration_sec)
        if authoritative_candidate_bounds
        else _clip_merge_spans(selected, duration_sec)
    )
    evidence = {
        "status": "completed",
        "provider": provider_label,
        "agentic_mode": resolved_mode,
        "instruction": instruction,
        "action": action,
        "candidate_count": len(candidates),
        "model_call_attempted": True,
        "observation_scope": observation_scope(
            ({"start_sec": c.span.start_sec, "end_sec": c.span.end_sec} for c in candidates),
            duration_sec,
        ),
        "candidate_segments": [
            {
                "candidate_id": candidate.segment_id,
                "audio_ref_id": str(input_audio_ref or input_audio_path),
                "start_sec": round(candidate.span.start_sec, 6),
                "end_sec": round(candidate.span.end_sec, 6),
                "speaker_id": candidate.span.speaker_id,
                "text": candidate.transcript or candidate.span.text,
                "description": candidate.description,
                "source_refs": list(candidate.source_refs)
                or ([candidate.source] if candidate.source else []),
                "source": "perception_segment_selector",
                **(
                    {
                        "target_id": candidate.target_id,
                        "target_kind": candidate.target_kind,
                        "target_value": candidate.target_value,
                    }
                    if candidate.target_id
                    else {}
                ),
                "bounds_authority": (
                    "aligner" if candidate.authoritative_bounds else "model_refinable"
                ),
            }
            for candidate in candidates
        ],
        "selected_segment_ids": selected_ids,
        "selected_count": len(selected),
        "judgements": judgements,
    }
    if isinstance(response.get("needs_more_evidence"), bool):
        evidence["needs_more_evidence"] = response["needs_more_evidence"]
    if not selected:
        evidence["reason"] = "perception model selected no candidate segments"
    return selected, evidence


def _should_use_perception_segment_selector(instruction: str, action: str) -> bool:
    text = instruction.lower()
    if action not in {
        "extract_spans",
        "remove_spans",
        "mute_spans",
        "beep_spans",
        "speaker_scoped_edit",
    }:
        return False
    if _target_name_mentions(instruction):
        return True
    if _quoted_strings(instruction) or _contains(
        text, ["phrase", "entity", "proper noun", "name", "named", "term"]
    ):
        return True
    if _contains(text, PERCEPTION_SEGMENT_SELECTOR_PROPERTY_TERMS):
        return True
    return _contains(
        text, ["extract", "isolate", "keep only", "remove", "mute", "delete", "only"]
    ) and _contains(
        text,
        [
            "with",
            "where",
            "who is",
            "who sounds",
            "that is",
            "that are",
            "containing",
            "marked by",
            "characterized by",
        ],
    )


def _should_defer_to_perception_segment_selector(instruction: str, action: str) -> bool:
    """Avoid broad timeline fallbacks for property/entity-scoped requests.

    Exact deterministic evidence is consumed before this point.  When no exact
    phrase, sentence, semantic segment, or profanity/pause/filler match exists,
    a property-scoped request should be decided per aligned candidate segment
    by the audio perception selector instead of falling through to "all speech"
    or "the only speaker".
    """

    text = instruction.lower()
    if not _should_use_perception_segment_selector(instruction, action):
        return False
    if _target_name_mentions(instruction):
        return True
    if _quoted_strings(instruction) or _contains(
        text, ["phrase", "entity", "proper noun", "name", "named", "term"]
    ):
        return True
    return _contains(text, PERCEPTION_SEGMENT_SELECTOR_PROPERTY_TERMS)


def _is_broad_single_speaker_selection(
    instruction: str,
    perception: dict[str, Any],
    target_speakers: list[str],
) -> bool:
    """Detect the unsafe "only one speaker, so select all speech" fallback."""

    segments = _normalized_speaker_segments(perception)
    order = _speaker_order(segments, perception)
    if len(order) != 1 or target_speakers != [order[0]]:
        return False
    text = instruction.lower()
    if _explicit_speaker_ids(instruction, order) or _first_speaker_ordinal(text):
        return False
    attr_kind = _target_attribute_kind(text)
    if attr_kind:
        explicit_match = _explicit_speaker_attribute_matches(perception, order[0], attr_kind)
        if explicit_match is True:
            return False
        if _speaker_profile_matches(_speaker_profiles(perception).get(order[0], ""), attr_kind):
            return False
    return True


def _is_broad_semantic_speaker_property_selection(
    instruction: str,
    perception: dict[str, Any],
    target_speakers: list[str],
) -> bool:
    segments = _normalized_speaker_segments(perception)
    order = _speaker_order(segments, perception)
    if len(target_speakers) <= 1 or len(target_speakers) < len(order):
        return False
    text = instruction.lower()
    attr_kind = _target_attribute_kind(text)
    if attr_kind not in {
        "male",
        "female",
        "adult",
        "child",
        "little_girl",
        "little_boy",
        "baby",
    }:
        return False
    return _contains(
        text,
        [
            "who ",
            "that ",
            "which ",
            "mentions",
            "mentioned",
            "says",
            "said",
            "asks",
            "asked",
            "identified as",
            "known as",
            "having",
            "lost",
            "with ",
        ],
    )


def _provided_segment_candidates(
    raw_candidates: list[dict[str, Any]] | None,
    *,
    duration_sec: float,
) -> list[SegmentCandidate]:
    candidates: list[SegmentCandidate] = []
    seen_ids: set[str] = set()
    for raw in raw_candidates or []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen_ids:
            continue
        try:
            start_sec = float(raw["start_sec"])
            end_sec = float(raw["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        span = TimeSpan(
            start_sec,
            end_sec,
            str(raw.get("label") or "aligned_candidate"),
            str(raw.get("speaker_id")) if raw.get("speaker_id") is not None else None,
            str(raw.get("text") or ""),
            str(raw.get("source") or "aligned_candidate_contract"),
        ).clipped(duration_sec)
        if span is None or span.duration_sec < 0.01:
            continue
        seen_ids.add(candidate_id)
        candidates.append(
            SegmentCandidate(
                candidate_id,
                span,
                transcript=str(raw.get("text") or ""),
                description=str(raw.get("description") or ""),
                source=str(raw.get("source") or "aligned_candidate_contract"),
                target_id=(str(raw["target_id"]) if raw.get("target_id") else None),
                target_kind=(
                    str(raw["target_kind"]) if raw.get("target_kind") else None
                ),
                target_value=(
                    str(raw["target_value"]) if raw.get("target_value") else None
                ),
                authoritative_bounds=bool(raw.get("authoritative_bounds", True)),
                source_refs=tuple(str(item) for item in raw.get("source_refs") or []),
            )
        )
    return candidates


def _segment_property_candidates(
    perception: dict[str, Any], duration_sec: float
) -> list[SegmentCandidate]:
    raw_spans = _raw_timeline_segment_candidates(perception, duration_sec)
    words = _normalized_words(perception.get("word_timestamps") or [])
    word_spans: list[TimeSpan] = []
    if words:
        word_spans = _word_sentence_spans(words, duration_sec) or _word_groups_as_spans(
            words, duration_sec
        )
    speaker_spans = _normalized_speaker_segments(perception)

    if raw_spans:
        candidates = raw_spans
    elif word_spans:
        candidates = [
            SegmentCandidate(
                f"candidate_{index:03d}",
                span,
                transcript=span.text,
                source=span.source,
            )
            for index, span in enumerate(word_spans)
        ]
    else:
        candidates = [
            SegmentCandidate(
                f"candidate_{index:03d}",
                span,
                transcript=span.text,
                source=span.source,
            )
            for index, span in enumerate(speaker_spans)
        ]

    return _dedupe_segment_candidates(candidates, duration_sec)


def _raw_timeline_segment_candidates(
    perception: dict[str, Any], duration_sec: float
) -> list[SegmentCandidate]:
    candidates: list[SegmentCandidate] = []
    index = 0
    for source, raw in _raw_timeline_segment_collections(perception):
        for item in raw:
            start = item.get("start_sec", item.get("start"))
            end = item.get("end_sec", item.get("end"))
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                continue
            span = TimeSpan(
                start_f,
                end_f,
                "perception_candidate",
                str(item.get("speaker_id") or item.get("speaker") or item.get("label") or "")
                or None,
                str(item.get("transcript") or item.get("text") or ""),
                source,
            ).clipped(duration_sec)
            if not span or span.duration_sec < 0.08:
                continue
            candidates.append(
                SegmentCandidate(
                    f"candidate_{index:03d}",
                    span,
                    transcript=str(item.get("transcript") or item.get("text") or ""),
                    description=str(
                        item.get("description") or item.get("event") or item.get("type") or ""
                    ),
                    source=source,
                )
            )
            index += 1
    return candidates


def _raw_timeline_segment_collections(
    perception: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    collections: list[tuple[str, list[dict[str, Any]]]] = []
    for key in ("speaker_segments", "segments"):
        raw = perception.get(key)
        if isinstance(raw, list):
            collections.append((key, [item for item in raw if isinstance(item, dict)]))
    gemini = perception.get("gemini_perception")
    if isinstance(gemini, dict):
        for key in ("speaker_segments", "segments"):
            raw = gemini.get(key)
            if isinstance(raw, list):
                collections.append(
                    (
                        f"gemini_perception.{key}",
                        [item for item in raw if isinstance(item, dict)],
                    )
                )
    return collections


def _dedupe_segment_candidates(
    candidates: list[SegmentCandidate], duration_sec: float
) -> list[SegmentCandidate]:
    output: list[SegmentCandidate] = []
    seen: set[tuple[int, int, str, str]] = set()
    seen_ids: set[str] = set()
    for candidate in candidates:
        clipped = candidate.span.clipped(duration_sec)
        if not clipped:
            continue
        key = (
            int(round(clipped.start_sec * 100)),
            int(round(clipped.end_sec * 100)),
            candidate.transcript.strip().lower(),
            str(candidate.target_id or ""),
        )
        if key in seen or candidate.segment_id in seen_ids:
            continue
        seen.add(key)
        seen_ids.add(candidate.segment_id)
        output.append(
            SegmentCandidate(
                candidate.segment_id,
                clipped,
                transcript=candidate.transcript,
                description=candidate.description,
                source=candidate.source,
                target_id=candidate.target_id,
                target_kind=candidate.target_kind,
                target_value=candidate.target_value,
                authoritative_bounds=candidate.authoritative_bounds,
                source_refs=candidate.source_refs,
            )
        )
    return output


def _export_segment_candidate_clips(
    audio: AudioSegment,
    candidates: list[SegmentCandidate],
    work_dir: Path,
) -> list[str]:
    paths: list[str] = []
    for candidate in candidates:
        path = work_dir / f"{candidate.segment_id}.wav"
        _slice_audio(audio, candidate.span.start_sec, candidate.span.end_sec).export(
            path, format="wav"
        )
        paths.append(str(path))
    return paths


def _call_segment_selector_model(
    *,
    instruction: str,
    action: str,
    candidates: list[SegmentCandidate],
    audio_paths: list[str],
    agentic_mode: str | None = None,
    selector_client: OpenAICompatibleChatClient | None = None,
    input_audio_ref: str | None = None,
    reference_audio_paths: list[str] | None = None,
    reference_audio_refs: list[str] | None = None,
    semantic_targets: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    resolved_mode = resolve_agentic_mode(agentic_mode)
    system_prompt = (
        "You are X-Editor's audio segment property selector. "
        "You receive aligned candidate audio clips and must judge which clips satisfy the user's target condition. "
        "Use the audio first, then the provided transcript/description metadata. "
        "A rejected clip establishes nothing about unattached source intervals. "
        "Uncertain rejection is not evidence of absence; set needs_more_evidence=true. "
        "Return only JSON."
    )
    refs = list(reference_audio_paths or [])
    logical_reference_refs = list(reference_audio_refs or [])
    if logical_reference_refs and len(logical_reference_refs) != len(refs):
        raise ValueError(
            "reference_audio_refs must match reference_audio_paths for semantic review"
        )
    source_ref = str(input_audio_ref or "source_audio")
    target_contracts = [
        {
            "target_id": str(item.get("target_id") or f"semantic_target_{index:03d}"),
            "kind": str(item.get("kind") or ""),
            "value": str(item.get("value") or ""),
        }
        for index, item in enumerate(semantic_targets or [])
        if isinstance(item, dict) and str(item.get("kind") or "") and str(item.get("value") or "")
    ]
    all_bounds_authoritative = bool(candidates) and all(
        candidate.authoritative_bounds for candidate in candidates
    )
    decision_rule = (
        "Select only candidate_id values whose attached clips satisfy the requested target. "
        "Candidate start/end bounds come from the aligner and are authoritative: do not "
        "propose, adjust, or return timestamps. A typed candidate already owns its target_id; "
        "do not rewrite that binding. Return one judgement per candidate_id and use "
        "matches=false when uncertain."
        if all_bounds_authoritative
        else (
            "For each candidate clip, set matches=true only if that clip itself should be "
            "edited/extracted for the requested target property. If only a phrase/entity/"
            "subpart inside the clip matches, include start_sec and end_sec on the original "
            "audio timeline when you can estimate them. If the decision asserts a verbatim "
            "phrase, use claim_kind=exact_phrase, bind the matching semantic_target_contract "
            "target_id when one exists, and repeat that contract value in claimed_text. For "
            "non-lexical judgements use the corresponding property claim kind. If uncertain, "
            "prefer matches=false and explain briefly."
        )
    )
    judgement_schema: dict[str, Any] = {
        "segment_id": "candidate_000",
        "matches": True,
        "confidence": 0.0,
        "reason": "short audio-grounded reason",
        "detected_properties": ["optional labels"],
    }
    if not all_bounds_authoritative:
        judgement_schema.update(
            {
                "claim_kind": (
                    "exact_phrase|semantic_property|speaker_property|audio_property|other"
                ),
                "claim_target_id": "semantic_target_000 or null",
                "claimed_text": "required only for an exact_phrase claim",
                "start_sec": 0.0,
                "end_sec": 1.0,
            }
        )
    payload = {
        "user_instruction": instruction,
        "operation": action,
        "decision_rule": decision_rule,
        "candidate_bounds_authority": (
            "aligner" if all_bounds_authoritative else "model_refinable"
        ),
        "audio_order": [candidate.segment_id for candidate in candidates],
        "semantic_target_contracts": target_contracts,
        "audio_inventory": [
            {
                "attachment_index": index,
                "kind": "source_candidate",
                "audio_ref_id": source_ref,
                "candidate_id": candidate.segment_id,
                **({"target_id": candidate.target_id} if candidate.target_id else {}),
            }
            for index, candidate in enumerate(candidates)
        ]
        + [
            {
                "attachment_index": len(candidates) + index,
                "kind": "reference_context",
                "audio_ref_id": (
                    logical_reference_refs[index]
                    if logical_reference_refs
                    else f"reference_context_{index}"
                ),
            }
            for index in range(len(refs))
        ],
        "candidates": [
            {
                "segment_id": candidate.segment_id,
                "start_sec": round(candidate.span.start_sec, 6),
                "end_sec": round(candidate.span.end_sec, 6),
                "speaker_id": candidate.span.speaker_id,
                "transcript": candidate.transcript or candidate.span.text,
                "description": candidate.description,
                "source": candidate.source or candidate.span.source,
                **(
                    {
                        "target_id": candidate.target_id,
                        "target_kind": candidate.target_kind,
                        "target_value": candidate.target_value,
                    }
                    if candidate.target_id
                    else {}
                ),
                "bounds_authority": (
                    "aligner" if candidate.authoritative_bounds else "model_refinable"
                ),
            }
            for candidate in candidates
        ],
        "required_json_schema": {
            "selected_segment_ids": ["candidate_000"],
            "judgements": [judgement_schema],
            "needs_more_evidence": False,
        },
    }
    timeout = float(os.getenv("X_EDITOR_PERCEPTION_SEGMENT_SELECTOR_TIMEOUT", "90"))
    user_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if perception_provider_name(resolved_mode) == QWEN3_OMNI_PROVIDER:
        client = selector_client or OpenAICompatibleChatClient(
            api_key=resolve_openai_compatible_api_key(QWEN3_OMNI_PROVIDER),
            base_url=resolve_openai_compatible_base_url(QWEN3_OMNI_PROVIDER),
            model_name=resolve_openai_compatible_model(QWEN3_OMNI_PROVIDER),
            provider_name=QWEN3_OMNI_PROVIDER,
            temperature=0.01,
            max_tokens=int(os.getenv("X_EDITOR_QWEN3_OMNI_SEGMENT_SELECTOR_MAX_TOKENS", "2048")),
            timeout=timeout,
            require_api_key=False,
        )
        return client.call_json(
            _qwen3_omni_segment_selector_system_prompt(system_prompt),
            user_text,
            audio_paths=[*audio_paths, *refs],
            response_format={"type": "json_object"},
            repair_schema_name="PerceptionSegmentSelection",
        )
    client = GeminiClient(timeout=timeout)
    return client.call_json(
        system_prompt,
        user_text,
        audio_paths=[*audio_paths, *refs],
    )


def _artifact_agentic_mode(params: dict[str, Any]) -> str | None:
    return (
        params.get("_agentic_mode")
        or params.get("agentic_mode")
        or params.get("perception_agentic_mode")
    )


def _segment_selector_provider_label(agentic_mode: str | None) -> str:
    return (
        "qwen3_omni_audio"
        if perception_provider_name(agentic_mode) == QWEN3_OMNI_PROVIDER
        else "gemini_2_5_pro_audio"
    )


def _segment_selector_service_unavailable_reason(
    agentic_mode: str | None,
) -> str | None:
    if perception_provider_name(agentic_mode) == QWEN3_OMNI_PROVIDER:
        if resolve_openai_compatible_base_url(QWEN3_OMNI_PROVIDER):
            return None
        return f"Qwen3-Omni segment selector service unavailable: {provider_service_hint(QWEN3_OMNI_PROVIDER)}"
    if resolve_gemini_api_key():
        return None
    return "Gemini key could not be resolved"


def _qwen3_omni_segment_selector_system_prompt(system_prompt: str) -> str:
    return (
        system_prompt.rstrip()
        + "\n\nProvider note:\n"
        + "- You are Qwen3-Omni running in X-Editor's open agentic mode.\n"
        + "- Judge each candidate clip directly from the attached audio and metadata.\n"
        + "- Do not mention Gemini or invent tool calls; return only the JSON object.\n"
    )


def _selected_spans_from_model_response(
    response: dict[str, Any],
    candidates: dict[str, SegmentCandidate],
    duration_sec: float,
) -> tuple[list[TimeSpan], list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    selected_spans: list[TimeSpan] = []
    segments_with_explicit_span: set[str] = set()
    raw_ids = response.get("selected_segment_ids") or response.get("selected_segments") or []
    if isinstance(raw_ids, list):
        for item in raw_ids:
            if isinstance(item, str) and item in candidates:
                selected.append(item)
            elif isinstance(item, dict) and str(item.get("segment_id") or "") in candidates:
                selected.append(str(item["segment_id"]))

    raw_spans = response.get("selected_spans") or []
    if isinstance(raw_spans, list):
        for item in raw_spans:
            if not isinstance(item, dict):
                continue
            segment_id = item.get("segment_id") or item.get("id")
            if segment_id is None or str(segment_id) not in candidates:
                continue
            span = _model_judgement_span(candidates[str(segment_id)], item, duration_sec)
            if span:
                selected_spans.append(span)
                selected.append(str(segment_id))
                segments_with_explicit_span.add(str(segment_id))

    judgements: list[dict[str, Any]] = []
    raw_judgements = response.get("judgements") or response.get("judgments") or []
    if isinstance(raw_judgements, list):
        for item in raw_judgements:
            if not isinstance(item, dict):
                continue
            segment_id = item.get("segment_id") or item.get("id")
            if segment_id is None or str(segment_id) not in candidates:
                continue
            candidate = candidates[str(segment_id)]
            match_declared = any(
                field in item for field in ("matches", "match", "selected")
            )
            compact = {
                "segment_id": str(segment_id),
                "matches": _truthy_match(
                    item.get("matches", item.get("match", item.get("selected")))
                ),
                "match_declared": match_declared,
                "confidence": item.get("confidence"),
                "reason": str(item.get("reason") or item.get("rationale") or ""),
                "detected_properties": item.get("detected_properties") or item.get("labels") or [],
            }
            if candidate.target_id:
                compact["claim_target_id"] = candidate.target_id
                if candidate.target_kind == "quoted_text":
                    compact["claim_kind"] = "exact_phrase"
                    compact["claimed_text"] = candidate.target_value or candidate.transcript
                compact["claim_binding_source"] = "typed_aligned_candidate"
            else:
                claim_kind = str(item.get("claim_kind") or item.get("match_kind") or "")
                if claim_kind in {
                    "exact_phrase",
                    "semantic_property",
                    "speaker_property",
                    "audio_property",
                    "other",
                }:
                    compact["claim_kind"] = claim_kind
                claim_target_id = item.get("claim_target_id")
                if isinstance(claim_target_id, str) and claim_target_id:
                    compact["claim_target_id"] = claim_target_id
                claimed_text = item.get("claimed_text") or item.get("claimed_phrase")
                if isinstance(claimed_text, str) and claimed_text:
                    compact["claimed_text"] = claimed_text
            span = _model_judgement_span(candidate, item, duration_sec)
            if span and not candidate.authoritative_bounds:
                compact["start_sec"] = round(span.start_sec, 6)
                compact["end_sec"] = round(span.end_sec, 6)
            judgements.append(compact)
            if compact["matches"] and str(segment_id) not in selected:
                selected.append(str(segment_id))
            if compact["matches"] and span:
                selected_spans.append(span)
                segments_with_explicit_span.add(str(segment_id))
    selected = _dedupe_strings(selected)
    selected_span_keys = {
        (round(span.start_sec, 3), round(span.end_sec, 3), span.speaker_id)
        for span in selected_spans
    }
    for segment_id in selected:
        if segment_id in segments_with_explicit_span:
            continue
        candidate = candidates.get(segment_id)
        if not candidate:
            continue
        key = (
            round(candidate.span.start_sec, 3),
            round(candidate.span.end_sec, 3),
            candidate.span.speaker_id,
        )
        if key not in selected_span_keys:
            selected_spans.append(candidate.span)
            selected_span_keys.add(key)
    return selected_spans, selected, judgements


def _model_judgement_span(
    candidate: SegmentCandidate | None,
    item: dict[str, Any],
    duration_sec: float,
) -> TimeSpan | None:
    if candidate is None:
        return None
    if candidate.authoritative_bounds:
        return candidate.span.clipped(duration_sec)
    start_value = item.get("start_sec", item.get("start"))
    end_value = item.get("end_sec", item.get("end"))
    if start_value is None or end_value is None:
        return None
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    base = candidate.span
    if 0.0 <= start < end <= base.duration_sec + 0.25 and start < max(0.0, base.start_sec - 0.25):
        start += base.start_sec
        end += base.start_sec
    start = max(base.start_sec, min(start, base.end_sec))
    end = min(base.end_sec, max(end, start))
    if end <= start:
        return None
    return TimeSpan(
        start,
        end,
        f"{base.label}:model_selected_subspan",
        base.speaker_id,
        base.text,
        f"{base.source}+perception_segment_selector",
    ).clipped(duration_sec)


def _truthy_match(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) >= 0.5
    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "y",
            "1",
            "match",
            "matches",
            "selected",
        }
    return False


def _dedupe_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _speaker_scoped_text_edit_target_text(
    *,
    instruction: str,
    current_text: str,
    direct_params: dict[str, Any],
    span: TimeSpan | None = None,
) -> str | None:
    target = _segment_target_text_for_span(span, direct_params)
    if target:
        return target
    target = direct_params.get("target_text")
    if isinstance(target, str) and target.strip():
        return target.strip()
    target = _target_text_from_tagged_substitution(direct_params.get("instruction_st"))
    if target:
        return target
    nested_instruction = str(direct_params.get("instruction_st") or instruction or "")
    agentic_mode = _artifact_agentic_mode(direct_params)
    if direct_params.get("_requires_target_text_generation"):
        return _generate_speaker_scoped_target_text(
            nested_instruction, current_text, agentic_mode=agentic_mode
        )
    if not _should_generate_speaker_scoped_target_text(nested_instruction, current_text):
        return None
    return _generate_speaker_scoped_target_text(
        nested_instruction, current_text, agentic_mode=agentic_mode
    )


def _segment_target_text_for_span(
    span: TimeSpan | None, direct_params: dict[str, Any]
) -> str | None:
    if span is None:
        return None
    segment_id = _segment_id_from_span(span)
    if not segment_id:
        return None
    for item in _segment_edit_specs(direct_params):
        if not isinstance(item, dict):
            continue
        item_id = item.get("segment_id") or item.get("id")
        if str(item_id) != segment_id:
            continue
        target = item.get("target_text") or item.get("replacement_text") or item.get("text")
        if isinstance(target, str) and target.strip():
            return target.strip()
    return None


def _segment_id_from_span(span: TimeSpan) -> str | None:
    for value in (span.label, span.source):
        match = re.search(r"\bsegment_id:([^|\s]+)", str(value or ""))
        if match:
            return match.group(1)
    return None


def _target_text_from_tagged_substitution(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    match = re.search(r"<sub\s+[^>]*targ=\"([^\"]*)\"", value) or re.search(
        r"<sub\s+[^>]*targ='([^']*)'", value
    )
    if not match:
        return None
    target = (
        match.group(1)
        .replace("&quot;", '"')
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )
    return target.strip() or None


def _should_generate_speaker_scoped_target_text(instruction: str, current_text: str) -> bool:
    if not current_text or not current_text.strip():
        return False
    lowered = instruction.lower()
    if "translate" in lowered and not _contains(
        lowered,
        [
            "fluent",
            "native",
            "natural",
            "grammatical",
            "grammar",
            "coherent",
            "standard",
        ],
    ):
        return False
    return _contains(
        lowered,
        [
            "fluent",
            "native",
            "natural",
            "grammatical",
            "grammar",
            "coherent",
            "standard",
            "word-by-word",
            "unnatural",
            "correct any",
            "make it sound",
            "通顺",
            "自然",
            "标准",
            "语法",
        ],
    )


def _generate_speaker_scoped_target_text(
    instruction: str,
    current_text: str,
    *,
    agentic_mode: str | None = None,
) -> str | None:
    try:
        timeout = float(os.getenv("X_EDITOR_SPEAKER_TEXT_REWRITE_TIMEOUT", "60"))
    except ValueError:
        timeout = 60.0
    system_prompt = (
        "You rewrite one short speech transcript segment for X-Editor text_edit. "
        "Return only JSON. Preserve language unless the instruction explicitly asks otherwise. "
        "Do not add commentary, speaker labels, timestamps, or quotes."
    )
    payload = {
        "instruction": instruction,
        "source_text": current_text,
        "required_json_schema": {
            "target_text": "rewritten transcript only",
            "reason": "brief reason",
        },
    }
    try:
        if resolve_agentic_mode(agentic_mode) == OPEN_GLM_QWEN_OMNI_MODE:
            response = OpenAICompatibleChatClient(
                api_key=resolve_openai_compatible_api_key(GLM_PROVIDER),
                base_url=resolve_openai_compatible_base_url(GLM_PROVIDER),
                model_name=resolve_openai_compatible_model(GLM_PROVIDER),
                provider_name=GLM_PROVIDER,
                timeout=timeout,
                max_tokens=2048,
            ).call_json(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                response_format={"type": "json_object"},
                repair_schema_name="SpeakerScopedTargetText",
            )
        else:
            response = GeminiClient(timeout=timeout).call_json(
                system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)
            )
    except Exception:
        return None
    if not isinstance(response, dict):
        return None
    target = response.get("target_text")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _target_language_from_translation_instruction(instruction: str) -> str | None:
    text = (instruction or "").lower()
    language_markers = [
        ("english", "English"),
        ("英语", "English"),
        ("mandarin", "Mandarin Chinese"),
        ("chinese", "Mandarin Chinese"),
        ("中文", "Mandarin Chinese"),
        ("spanish", "Spanish"),
        ("西班牙语", "Spanish"),
        ("japanese", "Japanese"),
        ("日语", "Japanese"),
        ("korean", "Korean"),
        ("韩语", "Korean"),
        ("french", "French"),
        ("法语", "French"),
        ("german", "German"),
        ("德语", "German"),
        ("italian", "Italian"),
        ("portuguese", "Portuguese"),
        ("russian", "Russian"),
    ]
    for marker, language in language_markers:
        if marker in text:
            return language
    return None


def _translation_instruction_for_target(target_language: Any, fallback_instruction: str) -> str:
    language = str(target_language or "").strip()
    if language:
        return f"Translate this speech transcript into {language} while preserving the intended meaning."
    return str(
        fallback_instruction
        or "Translate this speech transcript into the requested target language."
    )


def _tagged_text_substitution(source_text: str, target_text: str) -> str:
    return f'<sub targ="{_xml_attr_value(target_text)}">{_xml_text_value(source_text)}</sub>'


def _xml_attr_value(text: str) -> str:
    return _xml_text_value(text).replace('"', "&quot;")


def _xml_text_value(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wants_profanity_mask(text: str) -> bool:
    return _contains(
        text,
        [
            "beep",
            "bleep",
            "curse word",
            "curse words",
            "profanity",
            "profane",
            "swear word",
            "swear words",
            "cuss word",
            "cuss words",
            "censor",
            "脏话",
            "粗口",
            "消音",
        ],
    )


def _wants_filler_word_removal(text: str) -> bool:
    return _contains(
        text,
        [
            "filler word",
            "filler words",
            "disfluency",
            "disfluencies",
            "um",
            "uh",
            "口头禅",
            "语气词",
            "填充词",
        ],
    ) and _contains(text, ["remove", "delete", "cut", "去掉", "删除", "移除"])


def _filler_word_spans(words: list[dict[str, Any]], duration_sec: float) -> list[TimeSpan]:
    spans: list[TimeSpan] = []
    filler_tokens = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "ahh", "like"}
    previous_token = ""
    for word in words:
        token = _normalize_token(str(word.get("text") or ""))
        if not token:
            continue
        if token in filler_tokens or token == previous_token:
            spans.append(_padded_word_span(word, "filler_or_repeated_word", duration_sec))
        previous_token = token
    return _clip_merge_spans(spans, duration_sec)


def _padded_word_span(word: dict[str, Any], label: str, duration_sec: float) -> TimeSpan:
    start = float(word["start_sec"])
    end = float(word["end_sec"])
    if end <= start:
        end = start + 0.12
    start = max(0.0, start - 0.03)
    end = min(duration_sec, end + 0.03)
    return TimeSpan(
        start,
        end,
        label,
        str(word["speaker_id"]) if word.get("speaker_id") is not None else None,
        str(word.get("text") or ""),
        "word_timestamps",
    )


def _profanity_spans(
    words: list[dict[str, Any]], segments: list[TimeSpan], duration_sec: float
) -> list[TimeSpan]:
    spans: list[TimeSpan] = []
    for word in words:
        token = _normalize_profanity_token(str(word.get("text") or ""))
        if _is_profanity_token(token):
            spans.append(
                TimeSpan(
                    max(0.0, float(word["start_sec"]) - 0.025),
                    min(duration_sec, float(word["end_sec"]) + 0.025),
                    "profanity_word",
                    str(word["speaker_id"]) if word.get("speaker_id") is not None else None,
                    str(word.get("text") or ""),
                    "word_timestamps",
                )
            )
    if spans:
        return _clip_merge_spans(spans, duration_sec)

    for segment in segments:
        if any(
            _is_profanity_token(_normalize_profanity_token(token))
            for token in re.findall(r"[\w*#@!'-]+", segment.text)
        ):
            spans.append(
                TimeSpan(
                    segment.start_sec,
                    segment.end_sec,
                    "profanity_segment",
                    segment.speaker_id,
                    segment.text,
                    f"{segment.source}+profanity_transcript",
                )
            )
    return _clip_merge_spans(spans, duration_sec)


def _sentence_spans_from_instruction(
    instruction: str,
    words: list[dict[str, Any]],
    duration_sec: float,
) -> list[TimeSpan]:
    if not words:
        return []
    ordinal = _sentence_ordinal_from_instruction(instruction)
    if ordinal is None:
        return []
    sentences = _word_sentence_spans(words, duration_sec)
    if not sentences:
        return []
    if ordinal == -1:
        return [sentences[-1]]
    if 1 <= ordinal <= len(sentences):
        return [sentences[ordinal - 1]]
    return []


def _sentence_spans_from_segments(
    instruction: str,
    segments: list[TimeSpan],
    duration_sec: float,
) -> list[TimeSpan]:
    ordinal = _sentence_ordinal_from_instruction(instruction)
    if ordinal is None or not segments:
        return []
    sentence_spans: list[TimeSpan] = []
    for segment in segments:
        pieces = _sentence_pieces_from_text(segment.text)
        if not pieces:
            clipped = segment.clipped(duration_sec)
            if clipped:
                sentence_spans.append(clipped)
            continue
        text_len = max(1, len(segment.text))
        for sentence_text, start_char, end_char in pieces:
            start = segment.start_sec + segment.duration_sec * (start_char / text_len)
            end = segment.start_sec + segment.duration_sec * (end_char / text_len)
            if end > duration_sec and segment.end_sec > duration_sec:
                estimated_duration = max(0.1, end - start)
                end = duration_sec
                start = max(0.0, end - estimated_duration)
            span = TimeSpan(
                start,
                end,
                "sentence_segment",
                segment.speaker_id,
                sentence_text,
                f"{segment.source}+segment_transcript_sentence",
            ).clipped(duration_sec)
            if span:
                sentence_spans.append(span)
    if not sentence_spans:
        return []
    if ordinal == -1:
        return [sentence_spans[-1]]
    if 1 <= ordinal <= len(sentence_spans):
        return [sentence_spans[ordinal - 1]]
    return []


def _speaker_local_ordinal_spans(
    instruction: str,
    speaker_segments: list[TimeSpan],
    words: list[dict[str, Any]],
    duration_sec: float,
    *,
    target_speakers: list[str] | None = None,
) -> list[TimeSpan]:
    """Select the Nth sentence/turn inside the already-resolved speaker scope."""

    ordinal = _local_unit_ordinal_from_instruction(instruction)
    if ordinal is None:
        return []
    if _local_unit_ordinal_is_indirect_reference(instruction):
        return []

    scoped_segments = speaker_segments
    if target_speakers:
        target = {str(speaker_id) for speaker_id in target_speakers}
        scoped_segments = [segment for segment in scoped_segments if segment.speaker_id in target]

    unit_kind = _local_unit_kind_from_instruction(instruction)
    if unit_kind == "question" and scoped_segments:
        question_spans = _question_spans_from_segments(scoped_segments, duration_sec)
        if question_spans:
            if ordinal == -1:
                return [question_spans[-1]]
            if 1 <= ordinal <= len(question_spans):
                return [question_spans[ordinal - 1]]

    if scoped_segments and _sentence_ordinal_from_instruction(instruction) is not None:
        sentence_spans = _sentence_spans_from_segments(instruction, scoped_segments, duration_sec)
        if sentence_spans:
            return sentence_spans

    candidates = [segment for segment in scoped_segments if segment.clipped(duration_sec)]
    if not candidates and words and target_speakers:
        target = {str(speaker_id) for speaker_id in target_speakers}
        scoped_words = [word for word in words if str(word.get("speaker_id")) in target]
        candidates = _word_sentence_spans(scoped_words, duration_sec) or _word_groups_as_spans(
            scoped_words, duration_sec
        )

    if not candidates:
        return []
    if ordinal == -1:
        return [candidates[-1]]
    if 1 <= ordinal <= len(candidates):
        return [candidates[ordinal - 1]]
    return []


def _local_unit_ordinal_from_instruction(instruction: str) -> int | None:
    text = instruction.lower()
    unit = r"(?:sentence|line|utterance|turn|segment|reply|response|answer|question|statement)"
    for token, value in ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(token)}\s+{unit}s?\b", text):
            return value
        if re.search(rf"\b{unit}\s+(?:number\s+)?{re.escape(token)}\b", text):
            return value
    if re.search(rf"\b(?:last|final)\s+{unit}s?\b", text):
        return -1
    if re.search(rf"\b(?:last|final)\b.{{0,80}}\b{unit}s?\b", text):
        return -1
    if _contains(text, ["第一句", "第一个句子", "第一段话", "第一条发言"]):
        return 1
    if _contains(text, ["第二句", "第二个句子", "第二段话", "第二条发言"]):
        return 2
    if _contains(text, ["第三句", "第三个句子", "第三段话", "第三条发言"]):
        return 3
    if _contains(text, ["最后一句", "最后一个句子", "最后一段话", "最后一条发言"]):
        return -1
    return None


def _local_unit_kind_from_instruction(instruction: str) -> str | None:
    text = instruction.lower()
    if _local_unit_ordinal_is_indirect_reference(instruction):
        return None
    if _contains(text, ["question", "questions", "问句", "问题", "提问"]):
        return "question"
    if _contains(
        text,
        [
            "answer",
            "answers",
            "reply",
            "replies",
            "response",
            "responses",
            "回答",
            "答复",
        ],
    ):
        return "answer"
    if _contains(text, ["sentence", "sentences", "句子", "一句"]):
        return "sentence"
    if _contains(
        text,
        [
            "line",
            "lines",
            "utterance",
            "utterances",
            "turn",
            "turns",
            "statement",
            "statements",
            "话",
            "台词",
        ],
    ):
        return "utterance"
    return None


def _local_unit_ordinal_is_indirect_reference(instruction: str) -> bool:
    text = instruction.lower()
    return bool(
        re.search(
            r"\b(?:answer|answers|response|responses|reply|replies)\b.{0,60}\b(?:to|for|after|following)\b.{0,100}\bquestion",
            text,
        )
    )


def _question_spans_from_segments(segments: list[TimeSpan], duration_sec: float) -> list[TimeSpan]:
    spans: list[TimeSpan] = []
    for segment in segments:
        pieces = _sentence_pieces_from_text(segment.text)
        if pieces:
            text_len = max(1, len(segment.text))
            for sentence_text, start_char, end_char in pieces:
                if not _text_looks_like_question(sentence_text):
                    continue
                start = segment.start_sec + segment.duration_sec * (start_char / text_len)
                end = segment.start_sec + segment.duration_sec * (end_char / text_len)
                span = TimeSpan(
                    start,
                    end,
                    "question_segment",
                    segment.speaker_id,
                    sentence_text,
                    f"{segment.source}+question_transcript_sentence",
                ).clipped(duration_sec)
                if span:
                    spans.append(span)
            continue
        if _span_evidence_looks_like_question(segment):
            clipped = segment.clipped(duration_sec)
            if clipped:
                spans.append(clipped)
    return spans


def _span_evidence_looks_like_question(span: TimeSpan) -> bool:
    evidence = " ".join(part for part in [span.text, span.label, span.source] if part)
    return _text_looks_like_question(evidence)


def _text_looks_like_question(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if stripped.endswith(("?", "？")):
        return True
    if re.match(
        r"^(?:who|what|where|when|why|how|which|whose|whom|do|does|did|is|are|am|was|were|can|could|would|should|will|may|might|shall|have|has|had)\b",
        lowered,
    ):
        return True
    return _contains(
        lowered,
        [
            "asking a question",
            "asks a question",
            "question in the audio",
            "final question",
            "last question",
            "question-like",
            "questioning",
            "提问",
            "问句",
        ],
    )


def _sentence_ordinal_from_instruction(instruction: str) -> int | None:
    text = instruction.lower()
    if _contains(text, ["first sentence", "第一句", "第一个句子"]):
        return 1
    if _contains(text, ["second sentence", "第二句", "第二个句子"]):
        return 2
    if _contains(text, ["last sentence", "最后一句", "最后一个句子"]):
        return -1
    return None


def _sentence_pieces_from_text(text: str) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []
    pieces: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^.!?。！？]+[.!?。！？]?", text):
        raw = match.group(0)
        sentence = raw.strip()
        if not sentence:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = match.end() - (len(raw) - len(raw.rstrip()))
        pieces.append((sentence, start, end))
    return pieces


def _word_sentence_spans(words: list[dict[str, Any]], duration_sec: float) -> list[TimeSpan]:
    sentences = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        if _is_sentence_end(word):
            sentences.append(_span_from_words(current, "sentence"))
            current = []
    if current:
        sentences.append(_span_from_words(current, "sentence"))
    return _clip_merge_spans(sentences, duration_sec, merge_gap_sec=0.0)


def _segment_spans_by_order(
    instruction: str,
    speaker_segments: list[TimeSpan],
    words: list[dict[str, Any]],
    duration_sec: float,
) -> list[TimeSpan]:
    text = instruction.lower()
    candidates = speaker_segments or _word_sentence_spans(words, duration_sec)
    if not candidates:
        return []
    if _contains(text, ["first segment", "第一个片段"]):
        return [candidates[0]]
    if _contains(text, ["second segment", "第二个片段"]) and len(candidates) >= 2:
        return [candidates[1]]
    if _contains(text, ["last segment", "最后一个片段"]):
        return [candidates[-1]]
    return []


def _pause_gap_spans(words: list[dict[str, Any]], *, duration_sec: float) -> list[TimeSpan]:
    if len(words) < 2:
        return []
    spans = []
    min_gap = 0.3
    for left, right in zip(words, words[1:]):
        start = float(left["end_sec"])
        end = float(right["start_sec"])
        if end - start >= min_gap:
            spans.append(TimeSpan(start, end, "pause_gap", source="word_timestamps"))
    return _clip_merge_spans(spans, duration_sec)


def _word_groups_as_spans(words: list[dict[str, Any]], duration_sec: float) -> list[TimeSpan]:
    if not words:
        return []
    groups: list[list[dict[str, Any]]] = []
    current = [words[0]]
    for word in words[1:]:
        if float(word["start_sec"]) - float(current[-1]["end_sec"]) > 0.5:
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return _clip_merge_spans(
        [_span_from_words(group, "speech_group") for group in groups], duration_sec
    )


def _snap_spans_to_word_boundaries(
    spans: list[TimeSpan],
    words: list[dict[str, Any]],
    duration_sec: float,
) -> list[TimeSpan]:
    snapped = []
    for span in spans:
        selected_words = [
            word
            for word in words
            if _overlap_seconds(
                span.start_sec,
                span.end_sec,
                float(word["start_sec"]),
                float(word["end_sec"]),
            )
            > 0
        ]
        if not selected_words:
            clipped = span.clipped(duration_sec)
            if clipped:
                snapped.append(clipped)
            continue
        snapped_start = max(0.0, float(selected_words[0]["start_sec"]))
        snapped_end = min(duration_sec, float(selected_words[-1]["end_sec"]))
        snapped_text = _render_words(selected_words)
        if _contains_cjk(span.text) and not _contains_cjk(snapped_text):
            clipped = span.clipped(duration_sec)
            if clipped:
                snapped.append(clipped)
            continue
        max_reasonable_expansion = max(0.75, span.duration_sec * 0.5)
        expansion = (span.start_sec - snapped_start) + (snapped_end - span.end_sec)
        if expansion > max_reasonable_expansion:
            clipped = span.clipped(duration_sec)
            if clipped:
                snapped.append(clipped)
            continue
        snapped.append(
            TimeSpan(
                snapped_start,
                snapped_end,
                span.label or "word_snapped",
                span.speaker_id,
                snapped_text,
                f"{span.source}+word_timestamps",
            )
        )
    return snapped


def _enrich_spans_from_overlapping_speaker_segments(
    spans: list[TimeSpan],
    perception: dict[str, Any],
    *,
    target_speakers: list[str] | None = None,
) -> list[TimeSpan]:
    if not spans:
        return spans
    speaker_segments = _normalized_speaker_segments(perception)
    if not speaker_segments:
        return spans
    target = {str(speaker_id) for speaker_id in target_speakers or []}
    enriched: list[TimeSpan] = []
    for span in spans:
        if span.text.strip() and span.speaker_id:
            enriched.append(span)
            continue
        overlaps: list[TimeSpan] = []
        for segment in speaker_segments:
            if target and str(segment.speaker_id) not in target:
                continue
            overlap = _overlap_seconds(
                span.start_sec, span.end_sec, segment.start_sec, segment.end_sec
            )
            if overlap <= 0.05:
                continue
            overlaps.append(segment)
        if not overlaps:
            enriched.append(span)
            continue
        text = span.text.strip() or _merge_span_texts([segment.text for segment in overlaps])
        speaker_ids = {segment.speaker_id for segment in overlaps if segment.speaker_id}
        speaker_id = span.speaker_id
        if not speaker_id and len(speaker_ids) == 1:
            speaker_id = next(iter(speaker_ids))
        source = span.source
        if "speaker_segment_enriched" not in source:
            source = f"{source}+speaker_segment_enriched" if source else "speaker_segment_enriched"
        enriched.append(
            TimeSpan(span.start_sec, span.end_sec, span.label, speaker_id, text, source)
        )
    return enriched


def _merge_span_texts(values: list[str]) -> str:
    text = ""
    for value in values:
        text = _merge_span_text(text, value)
    return text


def _split_long_spans(
    spans: list[TimeSpan],
    words: list[dict[str, Any]],
    *,
    max_segment_sec: float,
) -> list[TimeSpan]:
    output: list[TimeSpan] = []
    for span in spans:
        if span.duration_sec < 30.0 and span.duration_sec <= max_segment_sec + 0.05:
            output.append(span)
            continue
        start = span.start_sec
        while start < span.end_sec:
            limit = min(span.end_sec, start + max_segment_sec)
            word_ends = [
                float(word["end_sec"])
                for word in words
                if start + 0.2 < float(word["end_sec"]) <= limit
                and _overlap_seconds(start, limit, float(word["start_sec"]), float(word["end_sec"]))
                > 0
            ]
            end = max(word_ends) if word_ends else limit
            end = max(start + 0.1, min(end, span.end_sec))
            split_words = [
                word
                for word in words
                if _overlap_seconds(start, end, float(word["start_sec"]), float(word["end_sec"]))
                > 0
            ]
            output.append(
                TimeSpan(
                    start,
                    end,
                    f"{span.label}:split",
                    span.speaker_id,
                    _render_words(split_words) or span.text,
                    span.source,
                )
            )
            start = end
    return output


def _clip_merge_spans(
    spans: list[TimeSpan], duration_sec: float, *, merge_gap_sec: float = 0.03
) -> list[TimeSpan]:
    clipped = [span.clipped(duration_sec) for span in spans]
    valid = sorted(
        (span for span in clipped if span is not None),
        key=lambda item: (item.start_sec, item.end_sec),
    )
    if not valid:
        return []
    merged = [valid[0]]
    for span in valid[1:]:
        prev = merged[-1]
        same_speaker = prev.speaker_id == span.speaker_id
        if span.start_sec <= prev.end_sec + merge_gap_sec and (
            same_speaker or not prev.speaker_id or not span.speaker_id
        ):
            merged[-1] = TimeSpan(
                prev.start_sec,
                max(prev.end_sec, span.end_sec),
                prev.label or span.label,
                prev.speaker_id or span.speaker_id,
                _merge_span_text(prev.text, span.text),
                prev.source or span.source,
            )
        else:
            merged.append(span)
    return merged


def _clip_spans_preserving_boundaries(
    spans: list[TimeSpan], duration_sec: float
) -> list[TimeSpan]:
    """Clip and order authoritative candidates without merging their identities."""

    clipped = [span.clipped(duration_sec) for span in spans]
    return sorted(
        (span for span in clipped if span is not None),
        key=lambda item: (item.start_sec, item.end_sec),
    )


def _clip_merge_spans_with_tail_alignment(
    spans: list[TimeSpan],
    duration_sec: float,
    *,
    merge_gap_sec: float = 0.03,
) -> list[TimeSpan]:
    aligned: list[TimeSpan] = []
    for span in spans:
        clipped = span.clipped(duration_sec)
        if clipped is None:
            clipped = _tail_align_transcript_span(span, duration_sec)
        if clipped is not None:
            aligned.append(clipped)
    return _clip_merge_spans(aligned, duration_sec, merge_gap_sec=merge_gap_sec)


def _merge_span_text(first: str, second: str) -> str:
    left = (first or "").strip()
    right = (second or "").strip()
    if not left:
        return right
    if not right or right == left:
        return left
    return f"{left} {right}"


def _spans_from_dicts(items: list[Any]) -> list[TimeSpan]:
    spans = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
            label = str(item[2]) if len(item) >= 3 and item[2] is not None else ""
            try:
                spans.append(TimeSpan(float(start), float(end), label, source="params"))
            except (TypeError, ValueError):
                continue
            continue
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None and end is None:
            continue
        if start is None:
            start = 0.0
        if end is None:
            end = float("inf")
        spans.append(
            TimeSpan(
                float(start),
                float(end),
                str(item.get("label") or ""),
                str(item["speaker_id"]) if item.get("speaker_id") is not None else None,
                str(item.get("text") or ""),
                str(item.get("source") or "params"),
            )
        )
    return spans


def _speaker_scoped_operations(instruction: str, source_text: str) -> list[str]:
    lowered = instruction.lower()
    if any(
        marker in lowered
        for marker in [
            "remove reverb",
            "remove the reverb",
            "remove echo",
            "remove the echo",
            "dereverb",
            "de-reverb",
            "normal speech",
            "more natural",
        ]
    ):
        return ["speech_enhancement"]
    sequence = build_structured_edit_sequence(
        instruction, source_text or "<source_text_from_perception>"
    )
    operations = [item.operation for item in sequence if item.supported]
    return [
        op
        for op in operations
        if op
        in {
            "text_edit",
            "pitch_rate_edit",
            "emotion_style_edit",
            "pause_edit",
            "volume_normalize",
            "speech_enhancement",
        }
    ]


def _speaker_scope_grounding_instruction(instruction: str, params: dict[str, Any]) -> str:
    pieces = [instruction]
    for key in (
        "target_speaker_description",
        "speaker_description",
        "target_description",
        "speaker_selector",
        "target_speaker",
        "target_voice",
        "selector_instruction",
        "selection_instruction",
        "select_instruction",
        "span_instruction",
    ):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value)
    selector = params.get("selector")
    if isinstance(selector, dict):
        for key in (
            "description",
            "speaker_description",
            "target_speaker_description",
            "role",
            "gender",
            "age",
        ):
            value = selector.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value)
    elif isinstance(selector, str) and selector.strip():
        pieces.append(selector)
    return " ".join(piece for piece in pieces if piece).strip()


def _target_speakers_from_params(params: dict[str, Any], perception: dict[str, Any]) -> list[str]:
    order = _speaker_order(_normalized_speaker_segments(perception), perception)
    if not order:
        return []

    candidates: list[Any] = []
    for key in ("speaker_id", "target_speaker_id", "speaker_ids", "target_speaker_ids"):
        value = params.get(key)
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
        elif value is not None:
            candidates.append(value)

    selector = params.get("selector")
    if isinstance(selector, dict):
        for key in (
            "speaker_id",
            "target_speaker_id",
            "speaker_ids",
            "target_speaker_ids",
        ):
            value = selector.get(key)
            if isinstance(value, (list, tuple)):
                candidates.extend(value)
            elif value is not None:
                candidates.append(value)

    result: list[str] = []
    for candidate in candidates:
        if str(candidate).strip().lower() in {
            "all",
            "all_speakers",
            "all speakers",
            "*",
        }:
            for speaker_id in order:
                if speaker_id not in result:
                    result.append(speaker_id)
            continue
        speaker_id = _speaker_id_from_candidate(candidate, order)
        if speaker_id and speaker_id not in result:
            result.append(speaker_id)
    return result


def _speaker_scope_requests_all_speakers(*instructions: str) -> bool:
    text = " ".join(instruction for instruction in instructions if instruction).lower()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:all|every|each)\s+(?:speaker|speakers|voice|voices|person|people)\b",
            text,
        )
        or re.search(r"\b(?:speaker|speakers|voice|voices)\s+(?:all|each|every)\b", text)
        or _contains(
            text,
            [
                "all speakers",
                "every speaker",
                "each speaker",
                "all voices",
                "every voice",
                "each voice",
                "全部说话人",
                "所有说话人",
                "每个说话人",
                "全部人声",
                "所有人声",
                "每个人声",
                "全部声音",
                "所有声音",
                "每个声音",
            ],
        )
    )


def _speaker_scope_defaults_to_all_speakers_for_broad_voice_edit(
    selection_instruction: str,
    grounding_instruction: str,
    params: dict[str, Any],
) -> bool:
    if _has_explicit_speaker_selector(params):
        return False
    evidence = " ".join(
        str(value)
        for value in [
            selection_instruction,
            grounding_instruction,
            params.get("edit_op"),
            params.get("edit_operation"),
            params.get("operation"),
            params.get("selector"),
        ]
        if value
    ).lower()
    if not evidence:
        return False
    scoped_terms = [
        "first speaker",
        "second speaker",
        "third speaker",
        "female speaker",
        "male speaker",
        "child speaker",
        "woman",
        "man",
        "girl",
        "boy",
        "customer",
        "employee",
        "narrator",
        "speaker_",
        "internal monologue",
    ]
    if _contains(evidence, scoped_terms) or _target_name_mentions(evidence):
        return False
    generic_voice = bool(
        re.search(r"\b(?:the\s+)?person'?s\s+(?:voice|speech)\b", evidence)
        or re.search(r"\b(?:the\s+)?human\s+(?:voice|speech)\b", evidence)
        or re.search(r"\b(?:the\s+)?(?:voice|speech)\b", selection_instruction.lower())
    )
    if not generic_voice:
        return False
    return _contains(
        evidence,
        [
            "volume_normalize",
            "volume",
            "loudness",
            "louder",
            "quieter",
            "reduce",
            "lower",
            "increase",
            "decrease",
            "gain",
            "db",
        ],
    )


def _speaker_scope_defaults_to_all_speakers_for_style(
    selection_instruction: str,
    grounding_instruction: str,
    params: dict[str, Any],
) -> bool:
    if _has_explicit_speaker_selector(params):
        return False
    evidence = " ".join(
        str(value)
        for value in [
            selection_instruction,
            grounding_instruction,
            params.get("edit_op"),
            params.get("edit_operation"),
            params.get("operation"),
        ]
        if value
    ).lower()
    if not evidence:
        return False
    scoped_terms = [
        "first speaker",
        "second speaker",
        "third speaker",
        "speaker_",
        "female speaker",
        "male speaker",
        "child speaker",
        "woman",
        "man",
        "girl",
        "boy",
        "customer",
        "employee",
        "speaker's first",
        "speaker's last",
        "internal monologue",
    ]
    if _contains(evidence, scoped_terms):
        return False
    style_terms = [
        "emotion_style_edit",
        "style",
        "theatrical",
        "dramatic",
        "pitch variation",
        "intonation",
        "tone",
        "语气",
        "语调",
        "风格",
    ]
    return _contains(evidence, style_terms)


def _has_explicit_speaker_selector(params: dict[str, Any]) -> bool:
    for key in ("speaker_id", "target_speaker_id", "speaker_ids", "target_speaker_ids"):
        if params.get(key) is not None:
            return True
    selector = params.get("selector")
    if isinstance(selector, dict):
        return any(
            selector.get(key) is not None
            for key in (
                "speaker_id",
                "target_speaker_id",
                "speaker_ids",
                "target_speaker_ids",
            )
        )
    return False


def _speaker_id_from_candidate(candidate: Any, speaker_order: list[str]) -> str | None:
    if candidate is None:
        return None
    text = str(candidate).strip()
    if not text:
        return None
    if text in speaker_order:
        return text
    lowered = text.lower()
    for speaker_id in speaker_order:
        if lowered == speaker_id.lower():
            return speaker_id
    if re.fullmatch(r"[0-9]+", lowered):
        ordinal = int(lowered)
        if 1 <= ordinal <= len(speaker_order):
            return speaker_order[ordinal - 1]
        zero_based = int(lowered)
        if 0 <= zero_based < len(speaker_order):
            return speaker_order[zero_based]
    match = re.search(r"speaker[_\s-]?([0-9]+)", lowered)
    if match:
        ordinal = int(match.group(1))
        if 1 <= ordinal <= len(speaker_order):
            return speaker_order[ordinal - 1]
    return None


def _spans_from_speaker_scope_params(
    params: dict[str, Any],
    perception: dict[str, Any],
    duration_sec: float,
) -> list[TimeSpan]:
    edit_spans = _spans_from_segment_edit_specs(params, perception, duration_sec)
    if edit_spans:
        return edit_spans
    for key in (
        "spans",
        "selected_spans",
        "time_spans",
        "segments",
        "segments_to_edit",
    ):
        spans = params.get(key)
        if isinstance(spans, list) and spans:
            return _reconstruct_spans_from_specs(spans, perception, duration_sec)
    selector = params.get("selector")
    if isinstance(selector, dict):
        for key in (
            "spans",
            "selected_spans",
            "time_spans",
            "segments",
            "segments_to_edit",
        ):
            spans = selector.get(key)
            if isinstance(spans, list) and spans:
                return _reconstruct_spans_from_specs(spans, perception, duration_sec)
    speaker_segments = _normalized_speaker_segments(perception)
    speaker_order = _speaker_order(speaker_segments, perception)
    selector_values: list[Any] = []
    for key in (
        "speaker_id",
        "target_speaker_id",
        "source_speaker_id",
        "speaker",
        "target_speaker",
        "selector",
    ):
        selector_values.append(params.get(key))
    if isinstance(selector, dict):
        selector_values.extend(
            selector.get(key)
            for key in (
                "speaker_id",
                "target_speaker_id",
                "source_speaker_id",
                "speaker",
                "target_speaker",
                "id",
            )
        )
    for value in selector_values:
        speaker_id = _speaker_id_from_candidate(value, speaker_order)
        if not speaker_id:
            continue
        spans = [segment for segment in speaker_segments if segment.speaker_id == speaker_id]
        words = _normalized_words(perception.get("word_timestamps") or [])
        if words:
            spans = _snap_spans_to_word_boundaries(spans, words, duration_sec)
        return _clip_merge_spans_with_tail_alignment(spans, duration_sec)
    return []


def _spans_from_segment_edit_specs(
    params: dict[str, Any],
    perception: dict[str, Any],
    duration_sec: float,
) -> list[TimeSpan]:
    edits = _segment_edit_specs(params)
    if not edits:
        return []
    by_id = _perception_segments_by_id(perception)
    spans: list[TimeSpan] = []
    for item in edits:
        if not isinstance(item, dict):
            continue
        segment_id = item.get("segment_id") or item.get("id")
        if segment_id is None:
            continue
        raw = by_id.get(str(segment_id))
        if not raw:
            continue
        start = raw.get("start_sec", raw.get("start"))
        end = raw.get("end_sec", raw.get("end"))
        if start is None or end is None:
            continue
        speaker = raw.get("speaker_id") or raw.get("speaker") or raw.get("label")
        try:
            span = TimeSpan(
                float(start),
                float(end),
                f"segment_id:{segment_id}",
                str(speaker) if speaker is not None else None,
                str(raw.get("transcript") or raw.get("text") or ""),
                "edit_op_params.edits",
            ).clipped(duration_sec)
        except (TypeError, ValueError):
            span = None
        if span:
            spans.append(span)
    return _clip_merge_spans_with_tail_alignment(spans, duration_sec, merge_gap_sec=0.0)


def _segment_edit_specs(params: dict[str, Any]) -> list[Any]:
    specs: list[Any] = []
    for source in (
        params,
        params.get("selector") if isinstance(params.get("selector"), dict) else None,
    ):
        if not isinstance(source, dict):
            continue
        direct = source.get("edits")
        if isinstance(direct, list):
            specs.extend(direct)
        for nested_key in ("edit_op_params", "edit_params", "sub_operation_params"):
            nested = source.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get("edits"), list):
                specs.extend(nested["edits"])
    return specs


def _perception_segments_by_id(perception: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for collection in ("segments", "speaker_segments"):
        raw = perception.get(collection) or []
        if isinstance(raw, list):
            _collect_segments_by_id(raw, result)
    gemini = perception.get("gemini_perception")
    if isinstance(gemini, dict):
        for collection in ("segments", "speaker_segments"):
            raw = gemini.get(collection) or []
            if isinstance(raw, list):
                _collect_segments_by_id(raw, result)
    return result


def _collect_segments_by_id(items: list[Any], result: dict[str, dict[str, Any]]) -> None:
    for item in items:
        if not isinstance(item, dict):
            continue
        segment_id = item.get("segment_id") or item.get("id")
        if segment_id is not None:
            result.setdefault(str(segment_id), item)


def _speaker_scoped_local_target_spans(
    instruction: str,
    perception: dict[str, Any],
    duration_sec: float,
    *,
    target_speakers: list[str] | None = None,
) -> list[TimeSpan]:
    words = _normalized_words(perception.get("word_timestamps") or [])
    speaker_segments = _normalized_speaker_segments(perception)
    if target_speakers:
        speaker_segments = [
            segment for segment in speaker_segments if segment.speaker_id in target_speakers
        ]
    spans: list[TimeSpan] = []
    explicit = _explicit_time_spans(instruction)
    if explicit:
        spans.extend(explicit)
    if not spans and _wants_entire_dialogue_selection(instruction):
        spans.extend(speaker_segments)
    if not spans:
        spans.extend(
            _speaker_local_ordinal_spans(
                instruction,
                speaker_segments,
                words,
                duration_sec,
                target_speakers=target_speakers,
            )
        )
    if not spans:
        spans.extend(_audio_half_spans(instruction, speaker_segments, duration_sec))
    if not spans:
        spans.extend(_sentence_half_spans(instruction, speaker_segments, words, duration_sec))
    if not spans:
        spans.extend(_phrase_spans(instruction, words))
        spans = _filter_spans_by_target_speakers(spans, target_speakers or [])
    if not spans:
        spans.extend(_phrase_spans_from_segments(instruction, speaker_segments, duration_sec))
    if not spans:
        spans.extend(_sentence_spans_from_instruction(instruction, words, duration_sec))
        spans = _filter_spans_by_target_speakers(spans, target_speakers or [])
    if not spans:
        spans.extend(
            _semantic_segment_spans(
                instruction,
                perception,
                target_speakers=target_speakers,
            )
        )
    return _clip_merge_spans(spans, duration_sec)


def _audio_half_spans(
    instruction: str,
    speaker_segments: list[TimeSpan],
    duration_sec: float,
) -> list[TimeSpan]:
    text = instruction.lower()
    wants_audio_scope = _contains(
        text,
        [
            "audio",
            "clip",
            "recording",
            "timeline",
            "dialogue",
            "conversation",
            "音频",
            "整段",
        ],
    )
    wants_first = _contains(text, ["first half", "front half", "beginning half", "前半", "上半"])
    wants_second = _contains(
        text, ["second half", "last half", "latter half", "final half", "后半", "下半"]
    )
    if not wants_audio_scope or not (wants_first or wants_second) or not speaker_segments:
        return []
    midpoint = duration_sec / 2.0
    selected: list[TimeSpan] = []
    for segment in speaker_segments:
        center = segment.start_sec + segment.duration_sec / 2.0
        if (wants_first and center <= midpoint) or (wants_second and center >= midpoint):
            clipped = segment.clipped(duration_sec)
            if clipped:
                selected.append(clipped)
    return selected


def _wants_entire_dialogue_selection(instruction: str) -> bool:
    text = instruction.lower()
    if _quoted_strings(instruction) and _contains(
        text, ["instance", "occurrence", "phrase", "word", "term"]
    ):
        return False
    if re.search(r"\b(?:this|the)\s+(?:dialogue|conversation)\b", text):
        return True
    whole_markers = [
        "entire",
        "whole",
        "all",
        "full",
        "complete",
        "全部",
        "整个",
        "整段",
        "所有",
    ]
    dialogue_markers = [
        "dialogue",
        "conversation",
        "speech",
        "speaking",
        "spoken",
        "utterance",
        "utterances",
        "audio",
        "语音",
        "对话",
        "讲话",
        "说话",
    ]
    return _contains(text, whole_markers) and _contains(text, dialogue_markers)


def _sentence_half_spans(
    instruction: str,
    speaker_segments: list[TimeSpan],
    words: list[dict[str, Any]],
    duration_sec: float,
) -> list[TimeSpan]:
    text = instruction.lower()
    wants_sentence = _contains(text, ["sentence", "utterance", "line", "句子", "话", "台词"])
    wants_first = _contains(text, ["first half", "front half", "beginning half", "前半", "上半"])
    wants_second = _contains(
        text, ["second half", "last half", "latter half", "final half", "后半", "下半"]
    )
    if not wants_sentence or not (wants_first or wants_second):
        return []
    described = [
        segment
        for segment in speaker_segments
        if segment.text.strip()
        or _contains(segment.label.lower(), ["segment", "sentence", "utterance"])
    ]
    if len(described) >= 2:
        return [described[0] if wants_first else described[-1]]
    if speaker_segments:
        segment = speaker_segments[0]
        midpoint = segment.start_sec + segment.duration_sec / 2.0
        if wants_first:
            clipped = TimeSpan(
                segment.start_sec,
                midpoint,
                f"{segment.label}:first_half",
                segment.speaker_id,
                _first_half_text(segment.text),
                f"{segment.source}+sentence_half",
            ).clipped(duration_sec)
            return [clipped] if clipped else []
        clipped = TimeSpan(
            midpoint,
            segment.end_sec,
            f"{segment.label}:second_half",
            segment.speaker_id,
            _second_half_text(segment.text),
            f"{segment.source}+sentence_half",
        ).clipped(duration_sec)
        return [clipped] if clipped else []
    word_spans = _word_sentence_spans(words, duration_sec)
    if not word_spans:
        return []
    span = word_spans[0]
    midpoint = span.start_sec + span.duration_sec / 2.0
    if wants_first:
        clipped = TimeSpan(
            span.start_sec,
            midpoint,
            "sentence_first_half",
            span.speaker_id,
            _first_half_text(span.text),
            "word_sentence_half",
        ).clipped(duration_sec)
        return [clipped] if clipped else []
    clipped = TimeSpan(
        midpoint,
        span.end_sec,
        "sentence_second_half",
        span.speaker_id,
        _second_half_text(span.text),
        "word_sentence_half",
    ).clipped(duration_sec)
    return [clipped] if clipped else []


def _first_half_text(text: str) -> str:
    if not text.strip():
        return ""
    tokens = text.split()
    if len(tokens) <= 1:
        return text.strip()
    return " ".join(tokens[: max(1, len(tokens) // 2)]).strip()


def _second_half_text(text: str) -> str:
    if not text.strip():
        return ""
    tokens = text.split()
    if len(tokens) <= 1:
        return text.strip()
    return " ".join(tokens[max(1, len(tokens) // 2) :]).strip()


def _speaker_scoped_segment_edit_params(
    params: dict[str, Any], current_text: str
) -> dict[str, Any]:
    edit_params = dict(params)
    selection_text = str(current_text or edit_params.get("source_text") or "").strip()
    if selection_text:
        edit_params.setdefault("source_text", selection_text)
    instruction_st = edit_params.get("instruction_st")
    if isinstance(instruction_st, str) and selection_text:
        replacement = _xml_text_value(selection_text)
        edit_params["instruction_st"] = (
            instruction_st.replace("{selection_text}", replacement)
            .replace("{current_text}", replacement)
            .replace("{source_text}", replacement)
        )
    return edit_params


def _speaker_scoped_direct_operation(
    params: dict[str, Any],
    instruction: str,
    reference_audio_paths: list[str],
) -> tuple[str | None, dict[str, Any]]:
    operation, operation_params = _extract_direct_operation(params.get("edit_op"))
    if operation is None:
        operation, operation_params = _extract_direct_operation(params.get("edit_operation"))
    if operation is None:
        operation, operation_params = _extract_direct_operation(params.get("sub_operation"))
    if operation is None:
        operation, operation_params = _extract_direct_operation(params.get("operation"))

    if operation:
        normalized = operation.removeprefix("edit_data_pipeline.").replace("-", "_")
        for nested_key in ("edit_params", "edit_op_params", "sub_operation_params"):
            nested_params = params.get(nested_key)
            if isinstance(nested_params, dict):
                operation_params.update(nested_params)
        if normalized in {
            "text_edit",
            "pitch_rate_edit",
            "emotion_style_edit",
            "pause_edit",
        }:
            natural_instruction = operation_params.get("instruction_st")
            if (
                isinstance(natural_instruction, str)
                and natural_instruction.strip()
                and "<" not in natural_instruction
                and not operation_params.get("instruction")
                and not operation_params.get("edit_instruction")
            ):
                operation_params["instruction"] = natural_instruction.strip()
                operation_params.pop("instruction_st", None)
        if normalized == "emotion_style_edit":
            _fill_direct_emotion_instruction(operation_params)
        if normalized in {
            "seed_vc",
            "voice_conversion",
            "voice_clone",
            "timbre_transfer",
            "timber_transfer",
        }:
            return "seed_vc", operation_params
        if normalized in {
            "speech_enhancement",
            "enhancement",
            "clearervoice",
            "clearer_voice",
            "denoise",
            "dereverb",
            "clarity",
        }:
            if normalized in {"denoise", "dereverb", "clarity"}:
                operation_params.setdefault("mode", normalized)
            return "speech_enhancement", operation_params
        if normalized == "text_edit" and _speaker_scoped_quality_cleanup_request(
            instruction, operation_params
        ):
            operation_params.setdefault("mode", "general")
            operation_params.setdefault(
                "enhancement_goal",
                str(
                    operation_params.get("instruction")
                    or operation_params.get("edit_instruction")
                    or instruction
                ),
            )
            return "speech_enhancement", operation_params
        if normalized == "text_edit" and _speaker_scoped_text_edit_is_tone_style_only(
            instruction,
            operation_params,
        ):
            operation_params.pop("instruction_st", None)
            operation_params.pop("target_text", None)
            operation_params.setdefault(
                "instruction", str(operation_params.get("instruction") or instruction)
            )
            return "emotion_style_edit", operation_params
        if normalized in {
            "audio_effects",
            "audio_effect",
            "acoustic_effect",
            "channel_effect",
            "audio_fx",
            "effects",
        }:
            operation_params.setdefault(
                "instruction", str(operation_params.get("instruction") or instruction)
            )
            return "audio_effects", operation_params
        if normalized == "emotion_style_edit":
            vocal_effort_preset = target_vocal_effort_effect_preset(
                _speaker_scoped_operation_evidence(instruction, operation_params)
            )
            if vocal_effort_preset:
                operation_params.setdefault("preset", vocal_effort_preset)
                operation_params.setdefault("strength", 0.72)
                operation_params.setdefault(
                    "instruction",
                    str(operation_params.get("instruction") or instruction),
                )
                return "audio_effects", operation_params
        if normalized == "emotion_style_edit" and _emotion_style_request_is_reverb_cleanup(
            instruction,
            operation_params,
        ):
            operation_params.setdefault("mode", "dereverb")
            operation_params.setdefault(
                "enhancement_goal",
                str(operation_params.get("instruction") or instruction),
            )
            return "speech_enhancement", operation_params
        if normalized == "emotion_style_edit" and _emotion_style_direct_op_requests_emphasis(
            instruction,
            operation_params,
        ):
            operation_params.pop("instruction_st", None)
            return "pitch_rate_edit", operation_params
        normalized_artifact = normalized.removeprefix("artifact_ops.")
        if normalized_artifact in {
            "mute",
            "silence",
            "mute_spans",
            "remove",
            "delete",
            "remove_speech",
            "remove_spans",
        }:
            return "mute_spans", operation_params
        if normalized in {"volume", "volume_gain", "gain", "attenuate", "weaken"}:
            return "volume_normalize", operation_params
        if normalized in {
            "speech_rate",
            "speaking_rate",
            "rate",
            "speed",
            "speed_change",
        }:
            return "speech_rate", operation_params
        if normalized in {
            "pitch_shift",
            "pitch",
            "pitch_change",
            "pitch_adjust",
            "lower_pitch",
            "raise_pitch",
        }:
            _ensure_speaker_scoped_pitch_instruction(operation_params, instruction)
            return "pitch_rate_edit", operation_params
        if normalized in {
            "seamless_expressive_s2st",
            "seamless_s2st",
            "speech_to_speech_translation",
            "s2st",
            "translate",
            "translation",
        }:
            target_language = (
                operation_params.get("target_language")
                or operation_params.get("target_lang")
                or operation_params.get("language")
                or _target_language_from_translation_instruction(instruction)
            )
            operation_params["_requires_target_text_generation"] = True
            if target_language:
                operation_params["target_language"] = target_language
            operation_params.setdefault(
                "instruction",
                _translation_instruction_for_target(target_language, instruction),
            )
            return "text_edit", operation_params
        if normalized in {
            "no_reference_voice_conversion",
            "voice_design",
            "target_voice_design",
        }:
            operation_params.setdefault(
                "instruction", str(operation_params.get("instruction") or instruction)
            )
            if _speaker_scoped_no_reference_voice_design_is_cleanup(instruction, operation_params):
                operation_params.setdefault("mode", "general")
                operation_params.setdefault(
                    "enhancement_goal",
                    str(operation_params.get("instruction") or instruction),
                )
                return "speech_enhancement", operation_params
            return "no_reference_voice_conversion", operation_params
        if normalized == "pitch_rate_edit" and _pitch_rate_operation_is_affective_style(
            instruction,
            operation_params,
        ):
            operation_params.pop("instruction_st", None)
            return "emotion_style_edit", operation_params
        if (
            normalized == "pitch_rate_edit"
            and _pitch_rate_request_is_question_to_declarative(
                instruction, operation_params, {}, ""
            )
            and not _speaker_scoped_has_concrete_pitch_or_rate_request(
                operation_params, instruction
            )
        ):
            operation_params.pop("instruction_st", None)
            operation_params.setdefault(
                "instruction", str(operation_params.get("instruction") or instruction)
            )
            return "emotion_style_edit", operation_params
        if normalized == "pitch_rate_edit" and _speaker_scoped_rate_only_request(
            instruction, operation_params
        ):
            return "speech_rate", operation_params
        if normalized == "pitch_rate_edit" and _speaker_scoped_has_pitch_shift_params(
            operation_params, instruction
        ):
            _ensure_speaker_scoped_pitch_instruction(operation_params, instruction)
        if normalized in {
            "text_edit",
            "pitch_rate_edit",
            "emotion_style_edit",
            "pause_edit",
            "volume_normalize",
        }:
            return normalized, operation_params

    preset = canonical_audio_effect_preset(
        " ".join(
            str(value)
            for value in [
                instruction,
                params.get("preset"),
                params.get("effect"),
                params.get("effect_name"),
            ]
            if value
        )
    )
    if preset:
        operation_params.setdefault("preset", preset)
        operation_params.setdefault(
            "instruction", str(operation_params.get("instruction") or instruction)
        )
        return "audio_effects", operation_params

    if (reference_audio_paths or _candidate_reference_audio_values(params)) and _contains(
        instruction.lower(),
        [
            "timbre",
            "timber",
            "voice conversion",
            "sound like",
            "match the voice",
            "音色",
            "声线",
        ],
    ):
        return "seed_vc", operation_params
    return None, {}


def _speaker_scoped_retry_without_prepend_context_flags(
    *,
    client: EditDataPipelineClient | None,
    tool_gateway: ToolGateway | None,
    operation: str,
    error: Exception,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    if operation != "text_edit":
        return None
    if "prepend context is required but disabled" not in str(error).lower():
        return None
    retry_args = dict(args)
    removed = []
    for key in (
        "disable_prepend_context",
        "fail_if_cannot_prepend_context",
        "min_unmasked_context_sec",
        "min_unmasked_context_ratio",
    ):
        if key in retry_args:
            retry_args.pop(key, None)
            removed.append(key)
    if not removed:
        return None
    if tool_gateway is not None:
        result = tool_gateway("edit_data_pipeline", retry_args)
    elif client is not None:
        result = client.apply(retry_args)
    else:  # pragma: no cover - guarded by execute_speaker_scoped_edit setup
        raise RuntimeError("edit-data-pipeline retry has no service gateway")
    result.setdefault("request_payload_repair", {})
    result["request_payload_repair"].update(
        {
            "reason": "service rejected optional prepend-context fields",
            "dropped_fields": sorted(removed),
        }
    )
    return result


def _speaker_scoped_text_edit_is_tone_style_only(
    instruction: str, operation_params: dict[str, Any]
) -> bool:
    if operation_params.get("target_text"):
        return False
    instruction_st = operation_params.get("instruction_st")
    if isinstance(instruction_st, str) and re.search(
        r"<\s*(?:sub|ins|del)\b", instruction_st, flags=re.IGNORECASE
    ):
        return False
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            operation_params.get("instruction"),
            operation_params.get("edit_instruction"),
            instruction_st,
            operation_params.get("style"),
            operation_params.get("emotion_style"),
        ]
        if value
    ).lower()
    if _pitch_rate_request_is_question_to_declarative(instruction, operation_params, {}, ""):
        return True
    has_style = _contains(
        evidence,
        [
            "tone",
            "style",
            "emotion",
            "questioning",
            "declarative",
            "interrogative",
            "rising intonation",
            "falling intonation",
            "语气",
            "语调",
            "情绪",
            "风格",
        ],
    )
    has_lexical_edit = _contains(
        evidence,
        [
            "replace",
            "delete",
            "remove",
            "insert",
            "swap",
            "substitute",
            "change the word",
            "change words",
            "替换",
            "删除",
            "插入",
            "交换",
        ],
    )
    return has_style and not has_lexical_edit


def _speaker_scoped_has_concrete_pitch_or_rate_request(
    params: dict[str, Any], instruction: str
) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("instruction_st"),
        ]
        if value
    ).lower()
    return bool(
        re.search(r"<(?:pitch|rate|speed),", evidence)
        or any(
            params.get(key) is not None
            for key in (
                "semitones",
                "pitch_semitones",
                "pitch_shift_semitones",
                "rate",
                "factor",
                "speed",
            )
        )
        or re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?\s*(?:semitones?|x|倍)", evidence)
    )


def _fill_direct_emotion_instruction(operation_params: dict[str, Any]) -> None:
    if (
        operation_params.get("instruction")
        or operation_params.get("edit_instruction")
        or operation_params.get("instruction_st")
    ):
        return
    emotion = _canonical_direct_emotion(
        operation_params.get("emotion")
        or operation_params.get("emotion_style")
        or operation_params.get("target_emotion")
        or operation_params.get("style")
    )
    if not emotion:
        return
    if emotion == "none":
        operation_params["instruction"] = "Keep the selected speech emotionally neutral."
        return
    operation_params["instruction"] = f"Make the selected speech sound {emotion}."


def _canonical_direct_emotion(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if not text:
        return None
    aliases = {
        "angry": "angry",
        "anger": "angry",
        "mad": "angry",
        "aggressive": "angry",
        "happy": "happy",
        "happiness": "happy",
        "joyful": "happy",
        "sad": "sad",
        "sadness": "sad",
        "afraid": "afraid",
        "fear": "afraid",
        "fearful": "afraid",
        "scared": "afraid",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "melancholic": "melancholic",
        "melancholy": "melancholic",
        "surprised": "surprised",
        "surprise": "surprised",
        "calm": "calm",
        "soft": "calm",
        "softly": "calm",
        "gentle": "calm",
        "neutral": "calm",
        "none": "none",
    }
    return aliases.get(text)


def _ensure_speaker_scoped_pitch_instruction(
    operation_params: dict[str, Any], instruction: str
) -> None:
    if any(
        operation_params.get(key) for key in ("instruction", "edit_instruction", "instruction_st")
    ):
        return
    semitones = _speaker_scoped_pitch_semitones(instruction, operation_params)
    if semitones is None:
        semitones = 4.0
    operation_params.setdefault("semitones", semitones)
    operation_params["instruction"] = f"shift pitch by {semitones:g} semitones"


def _speaker_scoped_has_pitch_shift_params(
    operation_params: dict[str, Any], instruction: str
) -> bool:
    if any(
        operation_params.get(key) is not None
        for key in ("semitones", "pitch_semitones", "pitch_shift_semitones")
    ):
        return True
    evidence = " ".join(
        str(value)
        for value in [
            operation_params.get("instruction"),
            operation_params.get("edit_instruction"),
            operation_params.get("instruction_st"),
            instruction,
        ]
        if value
    ).lower()
    return _contains(
        evidence,
        [
            "pitch",
            "semitone",
            "deeper",
            "lower voice",
            "higher voice",
            "降调",
            "升调",
            "更低",
            "更高",
        ],
    )


def _speaker_scoped_operation_evidence(instruction: str, operation_params: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in [
            instruction,
            operation_params.get("instruction"),
            operation_params.get("edit_instruction"),
            operation_params.get("style"),
            operation_params.get("emotion_style"),
            operation_params.get("instruction_st"),
        ]
        if value
    )


def _speaker_scoped_no_reference_voice_design_is_cleanup(
    instruction: str, operation_params: dict[str, Any]
) -> bool:
    return _speaker_scoped_quality_cleanup_request(instruction, operation_params)


def _speaker_scoped_quality_cleanup_request(
    instruction: str, operation_params: dict[str, Any]
) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            operation_params.get("instruction"),
            operation_params.get("edit_instruction"),
            operation_params.get("voice_description"),
            operation_params.get("target_voice_description"),
            operation_params.get("style"),
        ]
        if value
    ).lower()
    artifact = _contains(
        evidence,
        [
            "electronic",
            "robotic",
            "robot voice",
            "synthetic",
            "processed",
            "mechanical",
            "metallic",
        ],
    )
    cleanup_goal = _contains(
        evidence,
        [
            "natural",
            "normal voice",
            "normal speech",
            "clean",
            "clear",
            "human voice",
            "reduce",
            "remove",
            "less",
            "decrease",
            "weaken",
        ],
    )
    return artifact and cleanup_goal


def _speaker_scoped_no_ref_voice_prompt(
    instruction: str,
    direct_params: dict[str, Any],
    params: dict[str, Any],
) -> str:
    prompt = " ".join(
        str(value)
        for value in [
            direct_params.get("voice_prompt"),
            direct_params.get("instruct"),
            direct_params.get("target_voice_description"),
            direct_params.get("voice_description"),
            direct_params.get("target_voice"),
            params.get("voice_prompt"),
            params.get("target_voice_description"),
            instruction,
        ]
        if value
    ).strip()
    lowered = prompt.lower()
    if _contains(lowered, ["female", "woman voice", "girl voice", "女声", "女性"]):
        return "Adult female speaking voice, natural conversational speech, clear human timbre, same language as input."
    if _contains(lowered, ["male", "man voice", "boy voice", "男声", "男性"]):
        return "Adult male speaking voice, natural conversational speech, clear human timbre, same language as input."
    return (
        prompt
        or "Natural conversational speaking voice, clear human timbre, same language as input."
    )


def _speaker_scoped_no_ref_reference_text(
    direct_params: dict[str, Any], params: dict[str, Any]
) -> str:
    value = (
        direct_params.get("reference_text")
        or direct_params.get("target_voice_reference_text")
        or params.get("reference_text")
        or params.get("target_voice_reference_text")
    )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "This is a clean target voice reference for the edited speech."


def _emotion_style_direct_op_requests_emphasis(
    instruction: str, operation_params: dict[str, Any]
) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            operation_params.get("instruction"),
            operation_params.get("edit_instruction"),
            operation_params.get("style"),
            operation_params.get("emotion_style"),
        ]
        if value
    ).lower()
    return _contains(
        evidence,
        ["emphasis", "emphasize", "emphasise", "stress", "stressed", "重音", "强调"],
    )


def _pitch_rate_operation_is_affective_style(instruction: str, params: dict[str, Any]) -> bool:
    param_evidence = " ".join(
        str(value)
        for value in [
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("instruction_st"),
            params.get("style"),
            params.get("emotion_style"),
        ]
        if value
    ).lower()
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            param_evidence,
        ]
        if value
    ).lower()
    has_concrete_pitch_or_rate = bool(
        re.search(r"<(?:pitch|rate|speed),", evidence)
        or any(params.get(key) is not None for key in ("semitones", "rate", "factor", "speed"))
        or re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?\s*(?:semitones?|x|倍)", evidence)
    )
    if has_concrete_pitch_or_rate:
        return False
    affective_style_terms = [
        "theatrical",
        "dramatic",
        "expressive",
        "emotional style",
        "acting style",
        "lively",
        "energetic",
        "vibrant",
        "cheerful",
        "excited",
        "joyful",
        "happy",
        "animated",
        "spirited",
        "enthusiastic",
        "活泼",
        "生动",
        "有活力",
        "开心",
        "高兴",
        "兴奋",
        "快乐",
    ]
    has_style = _contains(evidence, affective_style_terms)
    if not has_style:
        return False
    if param_evidence and _contains(param_evidence, affective_style_terms):
        param_has_plain_pitch_or_rate = any(
            token in param_evidence
            for token in [
                "semitone",
                "rate",
                "speed",
                "faster",
                "slower",
                "tempo",
                "音高",
                "语速",
                "加速",
                "减速",
            ]
        )
        if not param_has_plain_pitch_or_rate:
            return True
    has_plain_pitch_or_rate_request = bool(
        any(
            token in evidence
            for token in [
                "semitone",
                "rate",
                "speed",
                "faster",
                "slower",
                "tempo",
                "音高",
                "语速",
                "加速",
                "减速",
            ]
        )
    )
    return not has_plain_pitch_or_rate_request or any(
        token in evidence
        for token in [
            "pitch contour",
            "pitch contours",
            "pitch variation",
            "pitch variations",
        ]
    )


def _prefer_segment_speech_rate_for_scoped_rate_edits(
    operations: list[str],
    instruction: str,
    direct_params: dict[str, Any],
) -> list[str]:
    if not operations:
        return operations
    if not _speaker_scoped_rate_only_request(instruction, direct_params):
        return operations
    return [
        "speech_rate" if operation == "pitch_rate_edit" else operation for operation in operations
    ]


def _speaker_scoped_rate_only_request(instruction: str, direct_params: dict[str, Any]) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            direct_params.get("instruction"),
            direct_params.get("edit_instruction"),
            direct_params.get("instruction_st"),
            direct_params.get("rate"),
            direct_params.get("factor"),
            direct_params.get("speed"),
        ]
        if value is not None
    ).lower()
    has_rate = bool(
        re.search(r"<(?:rate|speed),\s*factor=", evidence)
        or any(
            token in evidence
            for token in [
                "speech rate",
                "speaking rate",
                "speed up",
                "slow down",
                "faster",
                "slower",
                "rate",
                "speed",
                "语速",
                "加速",
                "减速",
                "放慢",
            ]
        )
        or direct_params.get("rate") is not None
        or direct_params.get("factor") is not None
        or direct_params.get("speed") is not None
    )
    has_pitch = bool(
        "<pitch" in evidence
        or direct_params.get("semitones") is not None
        or any(token in evidence for token in ["pitch", "semitone", "音高", "升调", "降调"])
    )
    return has_rate and not has_pitch


def _speaker_scoped_speech_rate_args(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    current_text: str,
    params: dict[str, Any],
    direct_params: dict[str, Any],
) -> dict[str, Any]:
    method = str(
        direct_params.get("method")
        or direct_params.get("rate_method")
        or params.get("method")
        or params.get("rate_method")
        or "wsola+indextts2"
    )
    args: dict[str, Any] = {
        "operation": "speech_rate",
        "input_audio_path": input_audio_path,
        "output_path": output_path,
        "instruction": instruction,
        "rate": _speaker_scoped_rate_factor(instruction, direct_params),
        "method": method,
        "speaker_audio_path": input_audio_path,
    }
    if current_text:
        args["text"] = current_text
        args["source_text"] = current_text
    return args


def _speaker_scoped_rate_factor(instruction: str, direct_params: dict[str, Any]) -> float:
    for key in ("rate", "factor", "speed"):
        value = direct_params.get(key)
        if value is None:
            continue
        try:
            return min(3.0, max(0.3, float(value)))
        except (TypeError, ValueError):
            pass
    instruction_st = str(direct_params.get("instruction_st") or "")
    match = re.search(r"<(?:rate|speed),\s*factor=([0-9]+(?:\.[0-9]+)?)", instruction_st)
    if match:
        return min(3.0, max(0.3, float(match.group(1))))
    text = instruction.lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:x|倍)", text)
    if match:
        return min(3.0, max(0.3, float(match.group(1))))
    if _contains(text, ["slow", "slower", "slow down", "放慢", "减速", "慢"]):
        return 0.8
    if _contains(text, ["fast", "faster", "speed up", "increase", "加速", "快"]):
        return 1.2
    return 1.0


def _speaker_scoped_pitch_rate_args_from_pitch_or_rate_request(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    current_text: str,
    span: TimeSpan,
    direct_params: dict[str, Any],
    output_dir: str,
) -> dict[str, Any] | None:
    text = current_text.strip()
    if not text:
        return None
    semitones = _speaker_scoped_pitch_semitones(instruction, direct_params)
    rate = _speaker_scoped_explicit_rate_factor(instruction, direct_params)
    if semitones is None and rate is None:
        return None
    instruction_st = _speaker_scoped_pitch_rate_instruction_st(
        text,
        semitones=semitones,
        rate=rate,
    )
    return {
        "operation": "pitch_rate_edit",
        "input_audio_path": input_audio_path,
        "output_path": output_path,
        "output_dir": output_dir,
        "instruction": instruction,
        "source_text": text,
        "instruction_st": instruction_st,
        "language": _language_hint(text),
        "evaluate": False,
        "pitch_method": str(direct_params.get("pitch_method") or "psola+indextts2_cco"),
        "smooth_policy": str(direct_params.get("smooth_policy") or "never"),
        "span_start_sec": round(span.start_sec, 6),
        "span_end_sec": round(span.end_sec, 6),
    }


def _speaker_scoped_pitch_rate_instruction_st(
    text: str,
    *,
    semitones: float | None,
    rate: float | None,
) -> str:
    escaped = _xml_text_value(text)
    if semitones is not None:
        escaped = f"<pitch, semitones={semitones:g}>{escaped}</pitch>"
    if rate is not None:
        escaped = f"<rate, factor={rate:g}>{escaped}</rate>"
    return escaped


def _speaker_scoped_explicit_rate_factor(
    instruction: str,
    direct_params: dict[str, Any],
) -> float | None:
    text = instruction.lower()
    has_rate_request = any(
        direct_params.get(key) is not None for key in ("rate", "factor", "speed")
    ) or _contains(
        text,
        [
            "slow",
            "slower",
            "slow down",
            "fast",
            "faster",
            "speed up",
            "speaking rate",
            "speech rate",
            "语速",
            "放慢",
            "加速",
        ],
    )
    if not has_rate_request:
        return None
    return _speaker_scoped_rate_factor(instruction, direct_params)


def _speaker_scoped_pitch_semitones(
    instruction: str,
    direct_params: dict[str, Any],
) -> float | None:
    for key in ("semitones", "pitch_shift_semitones", "pitch_semitones"):
        value = direct_params.get(key)
        if value is None:
            continue
        try:
            return max(-12.0, min(12.0, float(value)))
        except (TypeError, ValueError):
            pass
    instruction_st = str(direct_params.get("instruction_st") or "")
    match = re.search(r"<pitch,\s*semitones=([+-]?[0-9]+(?:\.[0-9]+)?)", instruction_st)
    if match:
        return max(-12.0, min(12.0, float(match.group(1))))
    text = instruction.lower()
    if _contains(
        text,
        [
            "deeper",
            "deepen",
            "lower pitch",
            "lower voice",
            "lower the pitch",
            "sound deeper",
            "低沉",
            "降低音高",
            "低音",
        ],
    ):
        return -3.0
    if _contains(
        text,
        [
            "higher pitch",
            "higher voice",
            "raise pitch",
            "raise the pitch",
            "sound higher",
            "升高音高",
            "高音",
        ],
    ):
        return 3.0
    return None


def _speaker_scoped_pitch_rate_args_from_intonation_request(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    current_text: str,
    span: TimeSpan,
    direct_params: dict[str, Any],
    output_dir: str,
) -> dict[str, Any] | None:
    if not current_text.strip():
        return None
    if not _pitch_rate_request_is_question_to_declarative(
        instruction, direct_params, {}, current_text
    ):
        return None
    return {
        "operation": "pitch_rate_edit",
        "input_audio_path": input_audio_path,
        "output_path": output_path,
        "output_dir": output_dir,
        "instruction": instruction,
        "source_text": current_text,
        "instruction_st": _declarative_tone_pitch_instruction(current_text),
        "language": _language_hint(current_text),
        "evaluate": False,
        "pitch_method": str(direct_params.get("pitch_method") or "psola+indextts2_cco"),
        "smooth_policy": str(direct_params.get("smooth_policy") or "never"),
        "span_start_sec": round(span.start_sec, 6),
        "span_end_sec": round(span.end_sec, 6),
    }


def _declarative_tone_pitch_instruction(current_text: str) -> str:
    text = current_text.strip()
    match = re.match(r"^(?P<body>.*?)(?P<punct>[?？]+)$", text)
    if match and match.group("body").strip():
        body = match.group("body").rstrip()
        suffix = text[len(body) :]
        return f"<pitch, semitones=-2>{body}</pitch>{suffix}"
    return f"<pitch, semitones=-2>{text}</pitch>"


def _speaker_scoped_pitch_rate_failure_fallback(
    *,
    operation: str,
    error: Exception,
    args: dict[str, Any],
    input_audio_path: str,
    output_path: str,
    span: TimeSpan,
    current_text: str,
    instruction: str,
    direct_params: dict[str, Any],
) -> dict[str, Any] | None:
    if operation != "pitch_rate_edit":
        return None
    error_text = f"{type(error).__name__}: {error}".lower()
    recoverable_pitch_failure = any(
        marker in error_text
        for marker in (
            "empty edit region",
            "minimum pitch",
            "pitch analysis not performed",
            "not converted to manipulation",
        )
    )
    if not recoverable_pitch_failure:
        return None
    if not _pitch_rate_request_is_question_to_declarative(
        instruction, direct_params, args, current_text
    ):
        return None
    if span.duration_sec > 1.25 and "?" not in current_text and "？" not in current_text:
        return None
    return _apply_declarative_tone_pitch_fallback(
        input_audio_path=input_audio_path,
        output_path=output_path,
        current_text=current_text,
        span=span,
        error_text=str(error),
    )


def _speaker_scoped_text_edit_failure_fallback(
    *,
    operation: str,
    error: Exception,
    args: dict[str, Any],
    input_audio_path: str,
    output_path: str,
    current_text: str,
    instruction: str,
) -> dict[str, Any] | None:
    # Text replacement inside existing/source audio is an edit-data-pipeline
    # boundary. Do not regenerate failed lexical edits with Qwen3-TTS; callers
    # should retry with smaller word-boundary segments or surface the original
    # edit-data-pipeline failure.
    return None


def _pitch_rate_request_is_question_to_declarative(
    instruction: str,
    direct_params: dict[str, Any],
    args: dict[str, Any],
    current_text: str,
) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            direct_params.get("instruction"),
            direct_params.get("edit_instruction"),
            direct_params.get("instruction_st"),
            args.get("instruction"),
            args.get("instruction_st"),
            current_text,
        ]
        if value
    ).lower()
    has_question_source = _contains(
        evidence,
        [
            "questioning",
            "interrogative",
            "rising intonation",
            "rising tone",
            "question-like",
            "friends?",
            "?",
            "？",
        ],
    )
    has_declarative_target = _contains(
        evidence,
        [
            "declarative",
            "falling intonation",
            "falling tone",
            "statement tone",
            "semitones=-",
            "contour='(0,0),(100,-",
            'contour="(0,0),(100,-',
        ],
    )
    return has_question_source and has_declarative_target


def _apply_declarative_tone_pitch_fallback(
    *,
    input_audio_path: str,
    output_path: str,
    current_text: str,
    span: TimeSpan,
    error_text: str,
) -> dict[str, Any]:
    audio = load_audio_segment(input_audio_path)
    duration_ms = len(audio)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if duration_ms < 80:
        audio.export(output, format="wav")
    else:
        split_ms = int(max(30, min(duration_ms - 20, duration_ms * 0.52)))
        head = audio[:split_ms]
        tail = audio[split_ms:]
        shifted_tail = _pitch_shift_audio_segment(tail, semitones=-2.2)
        crossfade_ms = int(max(5, min(24, len(head), len(shifted_tail), duration_ms * 0.08)))
        processed = head.append(shifted_tail, crossfade=crossfade_ms)
        processed = _fit_audio_duration(processed, duration_ms)
        processed = processed.fade_out(min(45, max(10, duration_ms // 5)))
        processed.export(output, format="wav")
    return {
        "operation": "pitch_rate_edit",
        "backend": "local_declarative_tone_pitch_fallback",
        "output_audio_path": str(output),
        "input_audio_path": input_audio_path,
        "source_text": current_text,
        "selected_span": span.to_dict(),
        "fallback_reason": error_text,
        "timeline_duration_preserved": True,
    }


def _pitch_shift_audio_segment(audio: AudioSegment, *, semitones: float) -> AudioSegment:
    if len(audio) <= 0:
        return audio
    factor = 2 ** (float(semitones) / 12.0)
    shifted_rate = max(1000, int(audio.frame_rate * factor))
    shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": shifted_rate})
    return shifted.set_frame_rate(audio.frame_rate)


def _speaker_scoped_volume_normalize_args(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    params: dict[str, Any],
    direct_params: dict[str, Any],
) -> dict[str, Any]:
    operation_params: dict[str, Any] = {}
    for source in (params, direct_params):
        for key in ("target_db", "method", "headroom_db", "service_url", "trace_id"):
            if source.get(key) is not None:
                operation_params[key] = source[key]
    operation_params.setdefault("target_db", _speaker_scoped_target_db(instruction))
    operation_params["method"] = canonicalize_volume_normalize_method(
        operation_params.get("method"),
        strict=True,
    )
    return {
        "operation": "volume_normalize",
        "input_audio_path": input_audio_path,
        "output_path": output_path,
        "instruction": instruction,
        **operation_params,
    }


def _speaker_scoped_target_db(instruction: str) -> float:
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:db|dB|分贝)", instruction)
    if match:
        return min(0.0, max(-60.0, float(match.group(1))))
    lowered = instruction.lower()
    if _contains(
        lowered,
        [
            "mute",
            "silence",
            "remove the voice",
            "完全静音",
            "静音",
        ],
    ):
        return -60.0
    if _contains(
        lowered,
        [
            "weaken",
            "attenuate",
            "suppress",
            "turn down",
            "lower",
            "reduce",
            "decrease",
            "quieter",
            "softer",
            "weaker",
            "压低",
            "减弱",
            "降低",
        ],
    ):
        return -25.0
    if _contains(
        lowered,
        [
            "boost",
            "amplify",
            "increase",
            "raise",
            "louder",
            "strengthen",
            "增强",
            "提高",
        ],
    ):
        return -18.0
    return -20.0


def _extract_direct_operation(value: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(value, dict):
        operation = value.get("operation") or value.get("tool") or value.get("tool_id")
        nested_params = value.get("params") if isinstance(value.get("params"), dict) else {}
        return (str(operation).strip().lower() if operation else None), dict(nested_params)
    if isinstance(value, str) and value.strip():
        return value.strip().lower(), {}
    return None, {}


def _speaker_scope_selection_instruction(instruction: str, params: dict[str, Any]) -> str:
    for key in (
        "select_instruction",
        "selection_instruction",
        "selector_instruction",
        "span_instruction",
    ):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    selector = params.get("selector")
    if isinstance(selector, dict):
        for key in (
            "select_instruction",
            "selection_instruction",
            "selector_instruction",
            "span_instruction",
            "description",
        ):
            value = selector.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return instruction


def _speaker_ids_from_spans(spans: list[TimeSpan]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for span in spans:
        speaker_id = span.speaker_id
        if speaker_id and speaker_id not in seen:
            seen.add(speaker_id)
            result.append(speaker_id)
    return result


def _filter_spans_by_target_speakers(
    spans: list[TimeSpan], target_speakers: list[str] | None
) -> list[TimeSpan]:
    if not spans or not target_speakers:
        return spans
    target = {str(speaker_id) for speaker_id in target_speakers}
    filtered = [span for span in spans if not span.speaker_id or str(span.speaker_id) in target]
    return filtered


def _emotion_style_request_is_reverb_cleanup(instruction: str, params: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for value in [
            instruction,
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("enhancement_goal"),
        ]
        if value
    ).lower()
    return _contains(text, ["reverb", "echo", "echoey", "dereverb"]) and _contains(
        text,
        ["remove", "clean", "normal speech", "natural speech", "without"],
    )


def _speaker_scoped_reference_audio(
    params: dict[str, Any],
    direct_params: dict[str, Any],
    reference_audio_paths: list[str],
) -> str:
    candidates = _candidate_reference_audio_values(
        direct_params
    ) + _candidate_reference_audio_values(params)
    candidates.extend(reference_audio_paths)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            return str(path)
    if candidates:
        raise FileNotFoundError(
            f"speaker_scoped_edit seed_vc reference audio does not exist: {candidates[0]}"
        )
    raise RuntimeError("speaker_scoped_edit seed_vc requires reference_audio_path")


def _candidate_reference_audio_values(params: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in (
        "reference_audio_path",
        "reference_audio",
        "ref_audio_path",
        "speaker_audio_path",
    ):
        value = params.get(key)
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
        elif value:
            candidates.append(value)
    for nested_key in ("edit_op", "edit_params", "params"):
        nested = params.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend(_candidate_reference_audio_values(nested))
    return candidates


def _concat_sources(
    *,
    instruction: str,
    input_audio_path: str,
    reference_audio_paths: list[str],
    artifact_paths: dict[str, str],
    params: dict[str, Any],
) -> list[str]:
    explicit = (
        params.get("audio_paths")
        or params.get("audio_path_list")
        or params.get("input_audios")
        or params.get("input_audio_paths")
        or params.get("sources")
    )
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit]
    paths = [input_audio_path] + [str(path) for path in reference_audio_paths]
    if len(paths) > 1:
        text = instruction.lower()
        if _looks_like_generated_audio_artifact(
            input_audio_path
        ) and not _concat_instruction_requests_known_inputs(text):
            return [input_audio_path]
        if _contains(
            text,
            [
                "second before first",
                "second audio before first",
                "第二个放在第一个前",
                "第二段放前",
            ],
        ):
            return [paths[1], paths[0], *paths[2:]]
        return paths
    artifact_values = [path for key, path in sorted(artifact_paths.items()) if key != "input_audio"]
    if artifact_values:
        return [input_audio_path, *artifact_values]
    return [input_audio_path]


def _concat_instruction_requests_known_inputs(text: str) -> bool:
    return bool(
        re.search(r"\b(?:audio|input)\s*[0-9]+\b", text)
        or _contains(
            text,
            [
                "first audio",
                "second audio",
                "third audio",
                "first segment",
                "second segment",
                "reference audio",
                "source audio",
                "original audio",
            ],
        )
    )


def _looks_like_generated_audio_artifact(path: str) -> bool:
    path_text = str(path or "")
    name = Path(path_text).name.lower()
    if name.startswith("step_") or "_step_" in name:
        return True
    return any(
        marker in path_text
        for marker in (
            "/generation/samples/",
            "/runtime_validation_runs/",
            "/artifact_ops/",
        )
    )


def _pause_duration_seconds(params: dict[str, Any], instruction: str) -> float:
    # The request may describe silence already materialized by another plan step.
    # Concat owns only its explicit inter-source pause, whose contract default is zero.
    return _duration_seconds_from_params(
        params,
        "",
        keys=(
            "pause_duration_sec",
            "silence_duration_sec",
            "gap_duration_sec",
            "insert_pause_sec",
            "pause_seconds",
        ),
        default=0.0,
    )


def _duration_seconds_from_params(
    params: dict[str, Any],
    instruction: str,
    *,
    keys: tuple[str, ...] = (
        "duration_sec",
        "duration_seconds",
        "seconds",
        "silence_duration_sec",
    ),
    default: float,
) -> float:
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    text = str(instruction or "").lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*-?\s*(?:seconds?|secs?|sec|s|秒)", text)
    if match and _contains(text, ["pause", "silence", "silent", "gap", "停顿", "静音", "空白"]):
        return max(0.0, float(match.group(1)))
    return max(0.0, float(default))


def _slice_audio(audio: AudioSegment, start_sec: float, end_sec: float) -> AudioSegment:
    return audio[int(round(start_sec * 1000)) : int(round(end_sec * 1000))]


def _replace_range(
    audio: AudioSegment, replacement: AudioSegment, start_sec: float, end_sec: float
) -> AudioSegment:
    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))
    replacement = _match_audio_format(replacement, audio)
    return audio[:start_ms] + replacement + audio[end_ms:]


def _fit_audio_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    target_ms = max(0, target_ms)
    if len(audio) > target_ms:
        return audio[:target_ms]
    if len(audio) < target_ms:
        silence = (
            AudioSegment.silent(duration=target_ms - len(audio), frame_rate=audio.frame_rate)
            .set_channels(audio.channels)
            .set_sample_width(audio.sample_width)
        )
        return audio + silence
    return audio


def _match_audio_format(audio: AudioSegment, reference: AudioSegment) -> AudioSegment:
    matched = audio.set_frame_rate(reference.frame_rate).set_sample_width(reference.sample_width)
    if matched.channels == reference.channels:
        return matched
    if matched.channels == 1 or reference.channels == 1:
        return matched.set_channels(reference.channels)
    return matched.set_channels(1).set_channels(reference.channels)


def _span_from_words(words: list[dict[str, Any]], label: str) -> TimeSpan:
    return TimeSpan(
        float(words[0]["start_sec"]),
        float(words[-1]["end_sec"]),
        label,
        _single_speaker_id(words),
        _render_words(words),
        "word_timestamps",
    )


def _words_text_for_span(words: list[dict[str, Any]], span: TimeSpan) -> str:
    selected = [
        word
        for word in words
        if _overlap_seconds(
            span.start_sec,
            span.end_sec,
            float(word["start_sec"]),
            float(word["end_sec"]),
        )
        > 0
    ]
    return _render_words(selected)


def _whole_span_source_text(span: TimeSpan, duration_sec: float, source_text: Any) -> str:
    if not isinstance(source_text, str) or not source_text.strip():
        return ""
    if span.start_sec <= 0.05 and span.end_sec >= max(0.0, duration_sec - 0.05):
        return source_text
    return ""


def _direct_source_text_for_span(
    *,
    span: TimeSpan,
    all_spans: list[TimeSpan],
    direct_params: dict[str, Any],
) -> str:
    for key in (
        "span_source_text",
        "segment_source_text",
        "segment_text",
        "source_text",
        "text",
    ):
        value = direct_params.get(key)
        if isinstance(value, str) and value.strip():
            if key in {"source_text", "text"} and len(all_spans) != 1:
                continue
            return value.strip()
    if isinstance(direct_params.get("span_texts"), list):
        index = next((idx for idx, item in enumerate(all_spans) if item is span), -1)
        if 0 <= index < len(direct_params["span_texts"]):
            value = direct_params["span_texts"][index]
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _should_mute_nonverbal_span(
    span: TimeSpan, instruction: str, perception: dict[str, Any]
) -> bool:
    if not _wants_laughter_removal(instruction):
        return False
    if _span_has_laughter_evidence(span, perception):
        return True
    return span.duration_sec < MIN_TAGGED_EDIT_SEGMENT_SEC and not span.text.strip()


def _speaker_scoped_event_removal_spans(
    params: dict[str, Any], duration_sec: float
) -> list[TimeSpan]:
    raw = params.get("event_removal_spans") or params.get("remove_event_spans")
    if not isinstance(raw, list) or not raw:
        return []
    spans: list[TimeSpan] = []
    for item in raw:
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start"))
            end = item.get("end_sec", item.get("end"))
            if start is None or end is None:
                continue
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                continue
            label = str(
                item.get("label") or item.get("event") or item.get("caption") or "event_removal"
            )
            source = str(item.get("source") or item.get("evidence_source") or "event_grounding")
            spans.append(TimeSpan(start_f, end_f, label=label, source=source))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                spans.append(
                    TimeSpan(
                        float(item[0]),
                        float(item[1]),
                        label=str(item[2]) if len(item) > 2 else "event_removal",
                        source="event_grounding",
                    )
                )
            except (TypeError, ValueError):
                continue
    return _clip_merge_spans(spans, duration_sec, merge_gap_sec=0.12)


def _apply_speaker_scoped_event_removals(
    edited_timeline: AudioSegment,
    *,
    event_removal_spans: list[TimeSpan],
    speaker_edit_spans: list[TimeSpan],
    words: list[dict[str, Any]],
    original: AudioSegment,
) -> tuple[AudioSegment, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for span in event_removal_spans:
        covered_overlap = sum(
            _overlap_seconds(span.start_sec, span.end_sec, edited.start_sec, edited.end_sec)
            for edited in speaker_edit_spans
        )
        if covered_overlap >= max(0.08, span.duration_sec * 0.35):
            records.append(
                {
                    **span.to_dict(),
                    "operation": "event_removal",
                    "status": "skipped_covered_by_speaker_resynthesis",
                    "covered_overlap_sec": round(covered_overlap, 6),
                }
            )
            continue
        word_overlap = _event_span_word_overlap_sec(span, words)
        if word_overlap >= max(0.04, span.duration_sec * 0.2):
            records.append(
                {
                    **span.to_dict(),
                    "operation": "event_removal",
                    "status": "skipped_word_overlap",
                    "word_overlap_sec": round(word_overlap, 6),
                }
            )
            continue
        silence = (
            AudioSegment.silent(
                duration=max(1, int(round(span.duration_sec * 1000))),
                frame_rate=original.frame_rate,
            )
            .set_channels(original.channels)
            .set_sample_width(original.sample_width)
        )
        edited_timeline = _replace_range(edited_timeline, silence, span.start_sec, span.end_sec)
        records.append(
            {
                **span.to_dict(),
                "operation": "event_removal",
                "status": "muted_grounded_non_speech_event",
                "replacement": "silence",
            }
        )
    return edited_timeline, records


def _event_span_word_overlap_sec(span: TimeSpan, words: list[dict[str, Any]]) -> float:
    overlap = 0.0
    for word in words:
        try:
            start = float(word.get("start_sec", 0.0))
            end = float(word.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        overlap += _overlap_seconds(span.start_sec, span.end_sec, start, end)
    return overlap


def _should_skip_untranscribed_nonverbal_span(span: TimeSpan, perception: dict[str, Any]) -> bool:
    if span.text.strip():
        return False
    if _span_has_nonverbal_evidence(span, perception):
        return True
    return span.duration_sec < MIN_TAGGED_EDIT_SEGMENT_SEC


def _wants_laughter_removal(instruction: str) -> bool:
    text = instruction.lower()
    has_remove = any(
        token in text
        for token in ["remove", "delete", "drop", "get rid of", "去掉", "删除", "移除"]
    )
    has_laughter = any(
        token in text for token in ["laugh", "laughter", "giggle", "chuckle", "笑声", "发笑"]
    )
    return has_remove and has_laughter


def _span_has_laughter_evidence(span: TimeSpan, perception: dict[str, Any]) -> bool:
    evidence = " ".join(part for part in [span.label, span.text, span.source] if part)
    for item in _raw_timeline_segments(perception):
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if _overlap_seconds(span.start_sec, span.end_sec, start_f, end_f) <= 0:
            continue
        for key in ("description", "transcript", "text", "label"):
            value = item.get(key)
            if isinstance(value, str):
                evidence += " " + value
    lowered = evidence.lower()
    return any(
        token in lowered for token in ["laugh", "laughter", "giggle", "chuckle", "笑声", "发笑"]
    )


def _span_has_nonverbal_evidence(span: TimeSpan, perception: dict[str, Any]) -> bool:
    evidence = " ".join(part for part in [span.label, span.text, span.source] if part)
    for item in _raw_timeline_segments(perception):
        try:
            start = float(item.get("start_sec", item.get("start", 0.0)))
            end = float(item.get("end_sec", item.get("end", 0.0)))
        except (TypeError, ValueError):
            continue
        if _overlap_seconds(span.start_sec, span.end_sec, start, end) <= 0:
            continue
        evidence += " " + " ".join(
            str(item.get(key) or "")
            for key in ["label", "text", "transcript", "description", "event", "type"]
        )
    lowered = evidence.lower()
    return any(
        token in lowered
        for token in [
            "cry",
            "crying",
            "scream",
            "screaming",
            "sob",
            "sobbing",
            "wail",
            "laugh",
            "laughter",
            "giggle",
            "cough",
            "non-speech",
            "nonverbal",
            "non-verbal",
            "哭",
            "哭声",
            "尖叫",
            "喊叫",
            "笑声",
            "咳嗽",
        ]
    )


def _raw_timeline_segments(perception: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for key in ("speaker_segments", "segments"):
        raw = perception.get(key)
        if isinstance(raw, list):
            segments.extend(item for item in raw if isinstance(item, dict))
    gemini = perception.get("gemini_perception")
    if isinstance(gemini, dict):
        for key in ("speaker_segments", "segments"):
            raw = gemini.get(key)
            if isinstance(raw, list):
                segments.extend(item for item in raw if isinstance(item, dict))
    return segments


def _write_silence_like(template_path: Path, output_path: Path, *, duration_sec: float) -> Path:
    duration_ms = max(1, int(round(duration_sec * 1000)))
    try:
        template = load_audio_segment(template_path)
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=template.frame_rate)
        silence = silence.set_channels(template.channels).set_sample_width(template.sample_width)
    except Exception:
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=48000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silence.export(output_path, format="wav")
    return output_path


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


def _is_sentence_end(word: dict[str, Any]) -> bool:
    text = str(word.get("text") or "")
    return (
        bool(word.get("is_punctuation"))
        and text in {".", "!", "?", "。", "！", "？"}
        or bool(re.search(r"[.!?。！？]$", text))
    )


def _quoted_strings(text: str) -> list[str]:
    values: list[str] = []
    single_quoted = r"(?<!\w)'((?:[^']|(?<=\w)'(?=\w))*)'(?!\w)"
    for match in re.finditer(rf"“([^”]+)”|{single_quoted}|\"([^\"]+)\"", text):
        values.append(next(group for group in match.groups() if group is not None))
    return values


def _normalize_token(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff'-]+", "", text.lower())


def _normalize_profanity_token(text: str) -> str:
    token = text.lower().replace("*", "").replace("#", "").replace("@", "").replace("!", "")
    return re.sub(r"[^a-z'-]+", "", token)


def _is_profanity_token(token: str) -> bool:
    if token in PROFANITY_TERMS:
        return True
    return any(
        token.startswith(term) and len(token) <= len(term) + 3 for term in {"fuck", "shit", "bitch"}
    )


def _normalize_text_for_contains(text: str) -> str:
    return re.sub(r"\s+", "", _normalize_token(text))


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _instruction_mentions_speaker(instruction: str) -> bool:
    if _target_name_mentions(instruction):
        return True
    return _contains(
        instruction,
        [
            "speaker",
            "voice",
            "person",
            "说话人",
            "女声",
            "男声",
            "little girl",
            "little boy",
            "child",
            "adult",
        ],
    )


def _language_hint(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _redact_large_args(args: dict[str, Any]) -> dict[str, Any]:
    compact = dict(args)
    for key in ["source_text", "instruction_st", "target_text"]:
        value = compact.get(key)
        if isinstance(value, str) and len(value) > 240:
            compact[key] = value[:240] + "...<truncated>"
    return compact
