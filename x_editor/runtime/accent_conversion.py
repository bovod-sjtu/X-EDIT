"""Accent/dialect conversion orchestration.

This runtime composes existing model tools instead of introducing a new model:

1. Resolve the requested accent/dialect from the user instruction.
2. Select a reference clip from the accent library.
3. Slice a target-speaker reference from the input audio.
4. Use SeedVC to convert the accent reference into the target speaker timbre.
5. Use Qwen3-TTS voice-clone mode with the converted reference to synthesize
   the selected speaker content.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydub import AudioSegment

from x_editor.accent_library import AccentDialectLibrary
from x_editor.accent_library.labels import best_label_for_request, canonical_label, extract_mentions
from x_editor.accent_library.schemas import AccentLibraryEntry, AccentMatch
from x_editor.perception import PerceptionRequirements, build_audio_perception_fallback


ExecuteTool = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
MAX_ACCENT_TTS_SPAN_SEC = 8.0
MAX_ACCENT_TTS_CHARS = 28
ACCENT_WORD_GROUP_GAP_SEC = 0.65
MIN_ACCENT_TTS_SPAN_SEC = 1.2
ACCENT_SHORT_SPAN_MERGE_GAP_SEC = 1.5


class AccentConversionError(RuntimeError):
    """Raised for user-actionable accent conversion failures."""


@dataclass(frozen=True)
class TimeSpan:
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
class AccentConversionPlan:
    target_accent: str
    match: AccentMatch
    target_speaker_reference_path: str
    converted_accent_reference_path: str
    spans: list[TimeSpan]
    target_text: str
    reference_audio_path: str | None = None
    target_accent_evidence: dict[str, Any] | None = None
    span_selection_evidence: dict[str, Any] | None = None


async def execute_accent_conversion(
    *,
    execute_tool: ExecuteTool,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    instruction: str,
    source_text: str | None,
    perception_report: dict[str, Any] | None,
    target_accent: str | None = None,
    reference_audio_path: str | None = None,
    reference_perception_report: dict[str, Any] | None = None,
    agentic_mode: str | None = None,
) -> dict[str, Any]:
    """Run accent conversion with library retrieval, SeedVC, and Qwen3-TTS."""

    input_audio = Path(input_audio_path)
    if not input_audio.exists():
        raise FileNotFoundError(f"accent_conversion input audio does not exist: {input_audio}")

    output_audio = Path(output_path).expanduser().resolve()
    work_dir = Path(output_dir).expanduser().resolve() / "accent_conversion" / output_audio.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    plan = _build_accent_conversion_plan(
        instruction=instruction,
        input_audio_path=str(input_audio),
        output_path=str(output_audio),
        work_dir=work_dir,
        source_text=source_text,
        perception_report=perception_report or {},
        target_accent=target_accent,
        reference_audio_path=reference_audio_path,
        reference_perception_report=reference_perception_report,
        agentic_mode=agentic_mode,
    )

    seed_output = await execute_tool(
        "seed_vc",
        {
            "source_audio_path": plan.match.entry.reference_audio_path,
            "reference_audio_path": plan.target_speaker_reference_path,
            "output_path": plan.converted_accent_reference_path,
            "diffusion_steps": 30,
            "length_adjust": 1.0,
            "inference_cfg_rate": 0.7,
        },
    )
    converted_reference = _output_audio_path(seed_output) or plan.converted_accent_reference_path

    synthesized_segments: list[tuple[TimeSpan, str]] = []
    tts_retry_events: list[dict[str, Any]] = []
    tts_deadline = _accent_tts_deadline()
    if len(plan.spans) <= 1:
        text = plan.target_text
        segment_output_path = str(work_dir / "accent_tts.wav")
        synthesized, retry_events = await _synthesize_qwen3_tts_clone(
            execute_tool=execute_tool,
            args=_qwen3_tts_clone_args(
                text=text,
                output_path=segment_output_path,
                converted_reference=converted_reference,
                accent_reference_text=plan.match.entry.transcript,
                target_accent=plan.target_accent,
            ),
            output_path=segment_output_path,
            retry_label="accent_tts",
            deadline=tts_deadline,
        )
        tts_retry_events.extend(retry_events)
        if _is_full_span(plan.spans, input_audio_path):
            output_audio.parent.mkdir(parents=True, exist_ok=True)
            output_audio.write_bytes(Path(synthesized).read_bytes())
        else:
            span = plan.spans[0]
            fitted = str(work_dir / "accent_tts_fit.wav")
            _fit_audio_to_duration(synthesized, fitted, span.duration_sec)
            _replace_spans(input_audio_path, [(span, fitted)], str(output_audio))
        synthesized_segments.append((plan.spans[0], synthesized))
    else:
        for index, span in enumerate(plan.spans):
            text = _span_text(span) or plan.target_text
            segment_output_path = str(work_dir / f"accent_tts_{index:02d}.wav")
            synthesized, retry_events = await _synthesize_qwen3_tts_clone(
                execute_tool=execute_tool,
                args=_qwen3_tts_clone_args(
                    text=text,
                    output_path=segment_output_path,
                    converted_reference=converted_reference,
                    accent_reference_text=plan.match.entry.transcript,
                    target_accent=plan.target_accent,
            ),
                output_path=segment_output_path,
                retry_label=f"accent_tts_{index:02d}",
                deadline=tts_deadline,
            )
            tts_retry_events.extend(retry_events)
            fitted = str(work_dir / f"accent_tts_{index:02d}_fit.wav")
            _fit_audio_to_duration(synthesized, fitted, span.duration_sec)
            synthesized_segments.append((span, fitted))
        _replace_spans(input_audio_path, synthesized_segments, str(output_audio))

    manifest = {
        "operation": "accent_conversion",
        "instruction": instruction,
        "input_audio_path": input_audio_path,
        "output_audio_path": str(output_audio),
        "target_accent": plan.target_accent,
        "accent_reference": {
            "entry_id": plan.match.entry.id,
            "canonical_accent": plan.match.entry.canonical_accent,
            "reference_audio_path": plan.match.entry.reference_audio_path,
            "transcript": plan.match.entry.transcript,
            "score": plan.match.score,
            "reasons": plan.match.reasons,
        },
        "target_speaker_reference_path": plan.target_speaker_reference_path,
        "converted_accent_reference_path": converted_reference,
        "reference_audio_path": plan.reference_audio_path,
        "target_accent_evidence": plan.target_accent_evidence,
        "perception_segment_selection": plan.span_selection_evidence,
        "target_text": plan.target_text,
        "selected_spans": [span.to_dict() for span in plan.spans],
        "synthesized_segments": [
            {"span": span.to_dict(), "audio_path": path}
            for span, path in synthesized_segments
        ],
        "tts_retry_events": tts_retry_events,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def resolve_requested_accent(instruction: str) -> str | None:
    """Return the best target accent/dialect label requested by text."""

    text = instruction.strip()
    lowered = text.lower()
    target_fragments: list[str] = []
    for pattern in [
        r"\bfrom\b.+?\bto\b\s+(.+)$",
        r"\bto\b\s+(.+)$",
        r"\binto\b\s+(.+)$",
        r"\bwith\b\s+(?:a|an|the)?\s*(.+?accent.+)$",
        r"\bhave\b\s+(?:a|an|the)?\s*(.+?accent.+)$",
        r"\bapply\b\s+(?:a|an|the)?\s*(.+?accent.+)$",
        r"变成(.+)$",
        r"转换成(.+)$",
        r"转换为(.+)$",
        r"改成(.+)$",
        r"改为(.+)$",
        r"转成(.+)$",
        r"转为(.+)$",
        r"译成(.+)$",
    ]:
        match = re.search(pattern, lowered if pattern.startswith(r"\b") else text, re.IGNORECASE)
        if match:
            target_fragments.append(match.group(1))

    for fragment in target_fragments:
        mentions = extract_mentions(fragment)
        if mentions:
            return _prefer_specific_label(mentions)

    explicit = _special_target_label(text)
    if explicit:
        return explicit
    return best_label_for_request(text)


def accent_conversion_failure_reason(instruction: str) -> str | None:
    """Return a routing-time failure reason, or None if a reference exists."""

    target = resolve_requested_accent(instruction)
    if target is None:
        return "No accent or dialect target could be resolved from the instruction."
    try:
        match = select_accent_reference(target)
    except AccentConversionError as exc:
        return str(exc)
    if match is None:
        return f"No usable accent-library reference found for {target}."
    return None


def requests_reference_accent_match(instruction: str) -> bool:
    """Return true when the target accent/dialect should be inferred from a reference audio."""

    return _requests_reference_accent_match(instruction)


def select_accent_reference(
    target_accent: str,
    *,
    gender: str | None = None,
    age_group: str | None = None,
    language: str | None = None,
) -> AccentMatch | None:
    library = AccentDialectLibrary.load()
    if not library.entries:
        raise AccentConversionError(
            f"Accent library is empty or missing at {library.path}. "
            "Build a user-owned library with python -m x_editor.accent_library build, "
            "then set X_EDITOR_ACCENT_LIBRARY_PATH. See x_editor/accent_library/README.md."
        )

    canonical = canonical_label(target_accent) or target_accent
    try:
        matches = library.find(
            accent=canonical,
            gender=gender,
            age_group=age_group,
            language=language,
            top_k=10,
        )
    except ValueError as exc:
        raise AccentConversionError(str(exc)) from exc

    exact = [match for match in matches if match.entry.canonical_accent == canonical]
    if exact:
        return exact[0]

    if _allow_compatible_reference(canonical) and matches:
        return matches[0]

    available = ", ".join(sorted(library.summary().get("labels", {}).keys()))
    raise AccentConversionError(
        f"No exact usable accent-library reference found for {canonical}. "
        f"Available labels: {available or 'none'}."
    )


def _build_accent_conversion_plan(
    *,
    instruction: str,
    input_audio_path: str,
    output_path: str,
    work_dir: Path,
    source_text: str | None,
    perception_report: dict[str, Any],
    target_accent: str | None = None,
    reference_audio_path: str | None = None,
    reference_perception_report: dict[str, Any] | None = None,
    agentic_mode: str | None = None,
) -> AccentConversionPlan:
    explicit_target_accent = (target_accent or "").strip()
    target_accent = _resolve_explicit_or_instruction_target(explicit_target_accent, instruction)
    target_accent_evidence: dict[str, Any] | None = None
    if explicit_target_accent:
        target_accent_evidence = {"source": "explicit_param", "target_accent": explicit_target_accent}
    reference_match: AccentMatch | None = None
    if not target_accent and reference_audio_path and _requests_reference_accent_match(instruction):
        target_accent, target_accent_evidence = _infer_reference_accent(
            instruction=instruction,
            reference_audio_path=reference_audio_path,
            work_dir=work_dir,
            reference_perception_report=reference_perception_report,
            agentic_mode=agentic_mode,
        )
        if not target_accent:
            reference_match = _reference_audio_accent_match(
                reference_audio_path=reference_audio_path,
                reference_perception_report=reference_perception_report,
                evidence=target_accent_evidence,
            )
            target_accent = reference_match.entry.canonical_accent
    if not target_accent:
        raise AccentConversionError(
            "Could not resolve target accent/dialect from instruction. "
            "Use a specific target such as Indian English, British English, Cantonese, or Sichuan Dialect."
        )

    spans, span_selection_evidence = _target_speaker_spans(
        instruction,
        perception_report,
        input_audio_path,
        work_dir=work_dir,
    )
    target_text = _target_text_for_accent_conversion(instruction, source_text, spans)
    if not target_text:
        raise AccentConversionError(
            "Could not determine what text the target speaker should say. "
            "ASR/source_text or selected speaker word timestamps are required."
        )

    gender = _target_gender(instruction, perception_report, spans)
    match = reference_match or select_accent_reference(
        target_accent,
        gender=gender,
        language=_language_for_text(target_text),
    )
    if match is None:
        raise AccentConversionError(f"No usable accent-library reference found for {target_accent}.")

    target_ref = str(work_dir / "target_speaker_reference.wav")
    _slice_reference_audio(input_audio_path, spans, target_ref)
    converted_ref = str(work_dir / "accent_reference_target_timbre.wav")
    return AccentConversionPlan(
        target_accent=canonical_label(target_accent) or target_accent,
        match=match,
        target_speaker_reference_path=target_ref,
        converted_accent_reference_path=converted_ref,
        spans=spans,
        target_text=target_text,
        reference_audio_path=reference_audio_path,
        target_accent_evidence=target_accent_evidence,
        span_selection_evidence=span_selection_evidence,
    )


def _resolve_explicit_or_instruction_target(explicit_target_accent: str, instruction: str) -> str | None:
    instruction_target = resolve_requested_accent(instruction)
    if not explicit_target_accent:
        return instruction_target
    if canonical_label(explicit_target_accent):
        return explicit_target_accent
    return instruction_target or explicit_target_accent


def _requests_reference_accent_match(instruction: str) -> bool:
    text = instruction.lower()
    return any(
        term in text
        for term in [
            "same accent as",
            "same dialect as",
            "accent back",
            "dialect back",
            "back to the original accent",
            "back to original accent",
            "back to her original accent",
            "back to his original accent",
            "back to their original accent",
            "restore the original accent",
            "restore original accent",
            "restore her accent",
            "restore his accent",
            "restore their accent",
            "revert the accent",
            "revert accent",
            "revert to the original accent",
            "revert to original accent",
            "previous accent",
            "original accent",
            "original dialect",
            "match the accent",
            "match the dialect",
            "matching accent",
            "matching dialect",
            "reference accent",
            "reference dialect",
            "accent of audio",
            "dialect of audio",
            "accent of the reference",
            "dialect of the reference",
            "accent from the reference",
            "dialect from the reference",
            "accent from reference",
            "dialect from reference",
            "accent from the reference audio",
            "dialect from the reference audio",
            "参考音频的口音",
            "参考音频的方言",
            "和参考音频一样",
            "和参考音频相同",
            "和音频2一样",
            "和音频 2 一样",
            "与音频2一样",
            "与音频 2 一样",
            "恢复原来的口音",
            "恢复原口音",
            "恢复原来的方言",
            "变回原来的口音",
            "变回原口音",
            "改回原来的口音",
            "改回原口音",
            "口音改回去",
            "方言改回去",
            "相同口音",
            "相同方言",
            "同样的口音",
            "同样的方言",
        ]
    )


def _infer_reference_accent(
    *,
    instruction: str,
    reference_audio_path: str,
    work_dir: Path,
    reference_perception_report: dict[str, Any] | None,
    agentic_mode: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    evidence: dict[str, Any] = {
        "strategy": "reference_audio_accent_match",
        "reference_audio_path": reference_audio_path,
    }

    if reference_perception_report:
        label, detail = _accent_from_perception_text(reference_perception_report)
        if label:
            evidence.update({"source": "provided_reference_perception", **detail})
            return label, evidence

    reference = Path(reference_audio_path).expanduser().resolve()
    if not reference.exists():
        evidence["error"] = f"reference audio does not exist: {reference}"
        return None, evidence

    try:
        result = build_audio_perception_fallback(agentic_mode=agentic_mode).perceive(
            user_prompt=(
                "Identify the language, accent, or dialect heard in this reference audio. "
                "Use specific names such as Minnan/Hokkien, Cantonese, Wu/Shanghainese, "
                "Sichuan, Dongbei, Beijing, British English, Indian English, or other "
                "canonical accent/dialect labels when audible."
            ),
            audio_path=reference,
            base_perception={},
            source_text=None,
            reference_audio_paths=[],
            output_dir=work_dir / "reference_accent_perception",
            requirements=PerceptionRequirements(
                source_text=True,
                speaker_profiles=True,
                segment_descriptions=True,
            ),
            run_specialized=True,
            allow_model_fallback=True,
            force_model_fallback=True,
            refresh_cache=False,
            strict=False,
        )
        evidence["perception_attempts"] = result.attempts
        label, detail = _accent_from_perception_text(result.perception)
        if label:
            evidence.update({"source": "reference_audio_perception", **detail})
            return label, evidence
        evidence["perception_missing_label"] = True
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"

    return None, evidence


def _reference_audio_accent_match(
    *,
    reference_audio_path: str,
    reference_perception_report: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
) -> AccentMatch:
    transcript = None
    if reference_perception_report:
        transcript = (
            reference_perception_report.get("source_text")
            or reference_perception_report.get("transcript")
            or reference_perception_report.get("text")
        )
    entry = AccentLibraryEntry(
        id="provided_reference_accent",
        canonical_accent="Reference Accent",
        reference_audio_path=reference_audio_path,
        transcript=str(transcript).strip() if transcript else None,
        provenance={
            "source": "provided_reference_audio",
            "reason": "reference accent requested but no canonical accent label could be resolved",
            "evidence": evidence or {},
        },
    )
    return AccentMatch(entry=entry, score=0.5, reasons=["provided reference audio used as accent reference"])


def _accent_from_perception_text(perception: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    text = _collect_perception_text(perception)
    if not text:
        return None, {"matched_text": ""}

    mentions = extract_mentions(text)
    if mentions:
        label = _prefer_specific_label(mentions)
        return label, {
            "method": "canonical_alias",
            "matched_labels": mentions,
            "matched_text": _truncate_evidence_text(text),
        }

    lowered = text.lower()
    for label, cues in _REFERENCE_DIALECT_CUES.items():
        hits = [cue for cue in cues if cue in text or cue.lower() in lowered]
        if hits:
            return label, {
                "method": "dialect_lexical_cues",
                "matched_cues": hits[:8],
                "matched_text": _truncate_evidence_text(text),
            }
    return None, {"matched_text": _truncate_evidence_text(text)}


def _collect_perception_text(value: Any) -> str:
    parts: list[str] = []

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, str):
            if key and any(skip in key for skip in ["path", "url", "id"]):
                return
            stripped = item.strip()
            if stripped:
                parts.append(stripped)
            return
        if isinstance(item, dict):
            for sub_key, sub_value in item.items():
                visit(sub_value, str(sub_key).lower())
            return
        if isinstance(item, list):
            for sub_value in item:
                visit(sub_value, key)

    visit(value)
    return " ".join(parts)


def _truncate_evidence_text(text: str, max_chars: int = 600) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


_REFERENCE_DIALECT_CUES: dict[str, tuple[str, ...]] = {
    "Minnan Dialect": (
        "阿嬷",
        "阿嬤",
        "无法度",
        "沒法度",
        "没法度",
        "代志",
        "代誌",
        "歹势",
        "歹勢",
        "沒要緊",
        "没要紧",
        "bô",
        "huat",
        "hokkien",
        "minnan",
        "southern min",
    ),
    "Cantonese": ("唔", "冇", "咩", "係", "喺", "嘅", "廣東話", "广东话", "粤语", "cantonese"),
    "Sichuan Dialect": ("啥子", "巴适", "安逸", "莫得", "要得", "sichuan"),
    "Dongbei Dialect": ("咋整", "嘎哈", "埋汰", "贼拉", "dongbei", "northeastern"),
    "Wu Chinese": ("侬", "阿拉", "勿", "伐", "覅", "辰光", "蛮好", "wuyu", "wu chinese"),
    "Shanghai Dialect": ("上海话", "shanghainese", "shanghai dialect"),
    "Suzhou Dialect": ("苏州话", "suzhou dialect", "suzhou wu"),
    "Ningbo Dialect": ("宁波话", "ningbo dialect", "ningbo wu"),
    "Beijing Dialect": ("儿化", "胡同", "beijing dialect"),
}


def _target_speaker_spans(
    instruction: str,
    perception_report: dict[str, Any],
    input_audio_path: str,
) -> tuple[list[TimeSpan], dict[str, Any] | None]:
    return _target_speaker_spans(instruction, perception_report, input_audio_path, work_dir=None)


def _target_speaker_spans(
    instruction: str,
    perception_report: dict[str, Any],
    input_audio_path: str,
    *,
    work_dir: Path | None,
) -> tuple[list[TimeSpan], dict[str, Any] | None]:
    from x_editor.runtime.artifact_ops import (
        resolve_target_speakers,
        select_timeline_spans_with_perception_fallback,
    )

    duration = _audio_duration_sec(input_audio_path)
    speaker_ids = resolve_target_speakers(instruction, perception_report)
    raw_segments = _speaker_segments(perception_report)
    words = _word_timestamps(perception_report)

    if speaker_ids:
        spans = [
            TimeSpan(
                float(item["start_sec"]),
                float(item["end_sec"]),
                label="target_speaker",
                speaker_id=str(item.get("speaker_id")),
                text=_text_for_span(item, words),
                source="speaker_segments",
            )
            for item in raw_segments
            if str(item.get("speaker_id")) in speaker_ids
            and item.get("start_sec") is not None
            and item.get("end_sec") is not None
            and float(item["end_sec"]) > float(item["start_sec"])
        ]
        clipped = _clip_merge_spans(spans, duration)
        if clipped:
            return clipped, None
        raise AccentConversionError(
            f"Resolved target speaker(s) {speaker_ids}, but no valid speaker timeline spans were found."
        )

    selector_action = "speaker_scoped_edit"
    if _requires_source_property_span_selection(instruction):
        spans, selection_evidence = select_timeline_spans_with_perception_fallback(
            instruction,
            perception_report,
            input_audio_path=input_audio_path,
            output_dir=str(work_dir or Path(input_audio_path).resolve().parent),
            duration_sec=duration,
            action=selector_action,
        )
        if spans:
            return [
                TimeSpan(
                    span.start_sec,
                    span.end_sec,
                    span.label,
                    span.speaker_id,
                    span.text,
                    span.source,
                )
                for span in spans
            ], selection_evidence
        reason = "no selected source property spans"
        if selection_evidence:
            reason = str(selection_evidence.get("reason") or selection_evidence.get("status") or reason)
        raise AccentConversionError(
            "Could not confirm which source audio segments should receive accent conversion: "
            f"{reason}"
        )

    if _requires_explicit_speaker(instruction) and raw_segments:
        raise AccentConversionError(
            "Instruction asks for a speaker-scoped accent conversion, but diarization/profile metadata "
            "could not identify the requested speaker."
        )

    if _prefers_whole_speech_accent_span(instruction):
        whole_span = _whole_speech_span(words, raw_segments, duration)
        if whole_span is not None and not _should_chunk_whole_speech_span_for_tts(whole_span):
            return [whole_span], {
                "strategy": "whole_speech_single_tts_span",
                "reason": "unscoped accent/dialect conversion should avoid multi-segment clone calls",
            }

    word_groups = _word_group_spans(words, duration)
    if word_groups:
        evidence = None
        if _prefers_whole_speech_accent_span(instruction):
            evidence = {
                "strategy": "whole_speech_chunked_tts_spans",
                "reason": "whole-speech accent conversion exceeded the single Qwen3-TTS clone chunk budget",
                "chunk_count": len(word_groups),
            }
        return word_groups, evidence

    if _prefers_whole_speech_accent_span(instruction):
        whole_span = _whole_speech_span(words, raw_segments, duration)
        if whole_span is not None:
            return [whole_span], {
                "strategy": "whole_speech_single_tts_span",
                "reason": "word-level chunking was unavailable; using one whole-speech clone call",
            }

    return [TimeSpan(0.0, duration, label="full_audio", text="", source="full_audio")], None


def _word_group_spans(words: list[dict[str, Any]], duration_sec: float) -> list[TimeSpan]:
    if not words:
        return []
    groups: list[TimeSpan] = []
    current_start: float | None = None
    current_end = 0.0
    current_words: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_words
        if current_start is None or not current_words:
            return
        start = max(0.0, current_start - 0.05)
        end = min(duration_sec, current_end + 0.08)
        text = _join_word_text(current_words)
        if end - start >= 0.2 and text:
            groups.append(TimeSpan(start, end, label="word_group", text=text, source="word_timestamps"))
        current_start = None
        current_end = 0.0
        current_words = []

    for item in words:
        text = str(item.get("text") or item.get("word") or item.get("token") or "").strip()
        if not text:
            continue
        try:
            start = float(item.get("start_sec", item.get("start")))
            end = float(item.get("end_sec", item.get("end")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if current_start is None:
            current_start = start
            current_end = end
            current_words = [text]
            continue
        candidate_text = _join_word_text([*current_words, text])
        should_split = (
            start - current_end >= ACCENT_WORD_GROUP_GAP_SEC
            or end - current_start > MAX_ACCENT_TTS_SPAN_SEC
            or len(candidate_text) > MAX_ACCENT_TTS_CHARS
        )
        if should_split:
            flush()
            current_start = start
            current_end = end
            current_words = [text]
        else:
            current_end = max(current_end, end)
            current_words.append(text)
    flush()
    groups = _merge_short_word_groups(groups, duration_sec)
    groups = _limit_word_groups(groups, duration_sec)
    return groups if len(groups) > 1 else []


def _merge_short_word_groups(groups: list[TimeSpan], duration_sec: float) -> list[TimeSpan]:
    if len(groups) <= 1:
        return groups
    min_span = _env_float("X_EDITOR_ACCENT_MIN_TTS_SPAN_SEC", MIN_ACCENT_TTS_SPAN_SEC)
    max_gap = _env_float("X_EDITOR_ACCENT_SHORT_SPAN_MERGE_GAP_SEC", ACCENT_SHORT_SPAN_MERGE_GAP_SEC)
    merged = list(groups)
    index = 0
    while index < len(merged):
        span = merged[index]
        if span.duration_sec >= min_span:
            index += 1
            continue
        prev_gap = span.start_sec - merged[index - 1].end_sec if index > 0 else None
        next_gap = merged[index + 1].start_sec - span.end_sec if index + 1 < len(merged) else None
        candidates: list[tuple[float, int]] = []
        if prev_gap is not None and prev_gap <= max_gap:
            candidates.append((prev_gap, index - 1))
        if next_gap is not None and next_gap <= max_gap:
            candidates.append((next_gap, index + 1))
        if not candidates:
            index += 1
            continue
        _, target_index = min(candidates, key=lambda item: item[0])
        if target_index < index:
            merged[target_index] = _merge_two_spans(merged[target_index], span, duration_sec)
            del merged[index]
            index = max(0, target_index)
        else:
            merged[index] = _merge_two_spans(span, merged[target_index], duration_sec)
            del merged[target_index]
    return merged


def _limit_word_groups(groups: list[TimeSpan], duration_sec: float) -> list[TimeSpan]:
    max_groups = _env_int("X_EDITOR_ACCENT_MAX_TTS_SEGMENTS", 6)
    if max_groups <= 0 or len(groups) <= max_groups:
        return groups
    merged = list(groups)
    while len(merged) > max_groups:
        gap, index = min(
            (
                (merged[i + 1].start_sec - merged[i].end_sec, i)
                for i in range(len(merged) - 1)
            ),
            key=lambda item: item[0],
        )
        if gap > _env_float("X_EDITOR_ACCENT_MAX_FORCE_MERGE_GAP_SEC", 4.0):
            break
        merged[index] = _merge_two_spans(merged[index], merged[index + 1], duration_sec)
        del merged[index + 1]
    return merged


def _merge_two_spans(left: TimeSpan, right: TimeSpan, duration_sec: float) -> TimeSpan:
    first, second = sorted([left, right], key=lambda item: item.start_sec)
    return TimeSpan(
        start_sec=max(0.0, min(first.start_sec, second.start_sec)),
        end_sec=min(duration_sec, max(first.end_sec, second.end_sec)),
        label=first.label,
        speaker_id=first.speaker_id if first.speaker_id == second.speaker_id else None,
        text=_join_word_text([first.text, second.text]),
        source=first.source,
    )


def _join_word_text(parts: list[str]) -> str:
    compact = "".join(parts)
    cjk_count = sum(1 for char in compact if "\u4e00" <= char <= "\u9fff")
    if cjk_count >= max(1, len(compact) // 2):
        return compact
    return " ".join(parts)


def _prefers_whole_speech_accent_span(instruction: str) -> bool:
    text = instruction.lower()
    local_terms = ["only", "just", "segment", "segments", "part", "parts", "portion", "portions", "局部", "片段", "部分"]
    if any(term in text for term in local_terms):
        return False
    if _requires_explicit_speaker(instruction) or _requires_source_property_span_selection(instruction):
        return False
    return any(
        term in text
        for term in [
            "this speech",
            "the speech",
            "speech in the audio",
            "this audio",
            "the audio",
            "entire",
            "whole",
            "整段",
            "全部",
        ]
    )


def _should_chunk_whole_speech_span_for_tts(span: TimeSpan) -> bool:
    text = _span_text(span)
    max_chars = _env_int("X_EDITOR_ACCENT_SINGLE_TTS_MAX_CHARS", MAX_ACCENT_TTS_CHARS)
    max_sec = _env_float("X_EDITOR_ACCENT_SINGLE_TTS_MAX_SEC", MAX_ACCENT_TTS_SPAN_SEC)
    return span.duration_sec > max_sec or (bool(text) and len(text) > max_chars)


def _whole_speech_span(
    words: list[dict[str, Any]],
    raw_segments: list[dict[str, Any]],
    duration_sec: float,
) -> TimeSpan | None:
    valid_words: list[tuple[float, float, str, str | None]] = []
    for item in words:
        text = str(item.get("text") or item.get("word") or item.get("token") or "").strip()
        if not text:
            continue
        try:
            start = float(item.get("start_sec", item.get("start")))
            end = float(item.get("end_sec", item.get("end")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        speaker_id = str(item.get("speaker_id")) if item.get("speaker_id") is not None else None
        valid_words.append((start, end, text, speaker_id))
    if valid_words:
        speaker_ids = {speaker_id for _, _, _, speaker_id in valid_words if speaker_id}
        return TimeSpan(
            start_sec=max(0.0, min(item[0] for item in valid_words) - 0.05),
            end_sec=min(duration_sec, max(item[1] for item in valid_words) + 0.08),
            label="whole_speech",
            speaker_id=next(iter(speaker_ids)) if len(speaker_ids) == 1 else None,
            text=_join_word_text([item[2] for item in valid_words]),
            source="word_timestamps",
        )

    valid_segments: list[TimeSpan] = []
    for item in raw_segments:
        try:
            start = float(item.get("start_sec", item.get("start")))
            end = float(item.get("end_sec", item.get("end")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        text = str(item.get("text") or item.get("transcript") or item.get("utterance") or "").strip()
        speaker_id = str(item.get("speaker_id")) if item.get("speaker_id") is not None else None
        valid_segments.append(TimeSpan(start, end, "whole_speech", speaker_id, text, "speaker_segments"))
    if valid_segments:
        speaker_ids = {span.speaker_id for span in valid_segments if span.speaker_id}
        return TimeSpan(
            start_sec=max(0.0, min(span.start_sec for span in valid_segments)),
            end_sec=min(duration_sec, max(span.end_sec for span in valid_segments)),
            label="whole_speech",
            speaker_id=next(iter(speaker_ids)) if len(speaker_ids) == 1 else None,
            text=_join_word_text([span.text for span in valid_segments]),
            source="speaker_segments",
        )

    if duration_sec > 0:
        return TimeSpan(0.0, duration_sec, label="full_audio", text="", source="full_audio")
    return None


def _slice_reference_audio(input_audio_path: str, spans: list[TimeSpan], output_path: str) -> None:
    longest = max(spans, key=lambda item: item.duration_sec)
    start = max(0.0, longest.start_sec)
    end = min(longest.end_sec, start + 10.0)
    if end - start < 0.5:
        start = 0.0
        end = min(_audio_duration_sec(input_audio_path), 10.0)
    _slice_audio(input_audio_path, output_path, start, end)


def _qwen3_tts_clone_args(
    *,
    text: str,
    output_path: str,
    converted_reference: str,
    accent_reference_text: str | None,
    target_accent: str,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "text": text,
        "output_path": output_path,
        "mode": "voice_clone",
        "language": "Auto",
        "ref_audio_path": converted_reference,
        "ref_text": accent_reference_text,
        "x_vector_only_mode": not bool(accent_reference_text),
        "instruct": (
            f"Speak exactly the provided text with {target_accent} pronunciation/accent. "
            "Do not add, omit, translate, or paraphrase words."
        ),
    }
    return args


async def _synthesize_qwen3_tts_clone(
    *,
    execute_tool: ExecuteTool,
    args: dict[str, Any],
    output_path: str,
    retry_label: str,
    deadline: float | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    retry_events: list[dict[str, Any]] = []
    max_retries = _env_int("X_EDITOR_ACCENT_TTS_RETRIES", 1)
    retry_delay = max(0.0, float(os.getenv("X_EDITOR_ACCENT_TTS_RETRY_DELAY_SEC", "2.0")))
    base_output = Path(output_path)
    current_args = dict(args)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt:
            retry_output = base_output.with_name(f"{base_output.stem}_retry{attempt}{base_output.suffix}")
            current_args["output_path"] = str(retry_output)
            if retry_delay:
                await asyncio.sleep(retry_delay)
        try:
            timeout = _accent_tts_call_timeout(deadline)
            current_args["timeout_seconds"] = timeout
            tts_output = await asyncio.wait_for(
                execute_tool("qwen3_tts", current_args),
                timeout=_accent_tts_mcp_call_timeout(timeout, deadline),
            )
            synthesized = _output_audio_path(tts_output) or str(current_args["output_path"])
            return synthesized, retry_events
        except Exception as exc:
            last_error = exc
            candidate = str(current_args.get("output_path") or "")
            if _usable_audio_file(candidate):
                retry_events.append(
                    {
                        "segment": retry_label,
                        "attempt": attempt,
                        "status": "recovered_output_file_after_exception",
                        "output_path": candidate,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return candidate, retry_events
            retry_events.append(
                {
                    "segment": retry_label,
                    "attempt": attempt,
                    "status": "failed",
                    "output_path": candidate,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if attempt >= max_retries:
                if _accent_tts_custom_fallback_enabled():
                    fallback = await _synthesize_qwen3_tts_custom_fallback(
                        execute_tool=execute_tool,
                        clone_args=current_args,
                        base_output=base_output,
                        retry_label=retry_label,
                        deadline=deadline,
                        clone_error=exc,
                    )
                    if fallback is not None:
                        synthesized, event = fallback
                        retry_events.append(event)
                        return synthesized, retry_events
                raise

    assert last_error is not None
    raise last_error


async def _synthesize_qwen3_tts_custom_fallback(
    *,
    execute_tool: ExecuteTool,
    clone_args: dict[str, Any],
    base_output: Path,
    retry_label: str,
    deadline: float | None,
    clone_error: Exception,
) -> tuple[str, dict[str, Any]] | None:
    fallback_output = base_output.with_name(f"{base_output.stem}_custom_fallback{base_output.suffix}")
    fallback_args = {
        "text": clone_args.get("text"),
        "output_path": str(fallback_output),
        "mode": "custom_voice",
        "speaker": os.getenv("X_EDITOR_ACCENT_TTS_CUSTOM_SPEAKER", "vivian"),
        "language": clone_args.get("language", "Auto"),
        "instruct": clone_args.get("instruct"),
        "timeout_seconds": _accent_tts_custom_fallback_timeout(deadline),
    }
    try:
        fallback_timeout = float(fallback_args["timeout_seconds"])
        tts_output = await asyncio.wait_for(
            execute_tool("qwen3_tts", fallback_args),
            timeout=_accent_tts_mcp_call_timeout(fallback_timeout, deadline),
        )
    except Exception:
        return None
    synthesized = _output_audio_path(tts_output) or str(fallback_output)
    if not _usable_audio_file(synthesized):
        return None
    return synthesized, {
        "segment": retry_label,
        "attempt": "custom_voice_fallback",
        "status": "fallback_custom_voice_after_clone_failure",
        "output_path": synthesized,
        "error": f"{type(clone_error).__name__}: {clone_error}",
        "boundary": (
            "Qwen3-TTS voice_clone did not complete; custom_voice fallback preserves text/accent instruction "
            "but does not guarantee original speaker timbre."
        ),
    }


def _accent_tts_custom_fallback_enabled() -> bool:
    value = os.getenv("X_EDITOR_ACCENT_TTS_CUSTOM_FALLBACK", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _accent_tts_custom_fallback_timeout(deadline: float | None) -> float:
    raw = os.getenv("X_EDITOR_ACCENT_TTS_CUSTOM_TIMEOUT")
    if raw is not None:
        timeout = max(0.1, float(raw))
    else:
        timeout = min(120.0, _accent_tts_tool_timeout_sec())
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0.1
    return max(0.1, min(timeout, remaining))


def _accent_tts_tool_timeout_sec() -> float:
    raw = os.getenv("X_EDITOR_ACCENT_TTS_TOOL_TIMEOUT")
    if raw is not None:
        return max(0.1, float(raw))
    base = float(os.getenv("X_EDITOR_QWEN3_TTS_TIMEOUT", "300")) + 15.0
    cap = _env_float("X_EDITOR_ACCENT_TTS_DEFAULT_CALL_TIMEOUT_CAP", 60.0)
    return max(1.0, min(base, cap))


def _accent_tts_deadline() -> float | None:
    raw = os.getenv("X_EDITOR_ACCENT_TTS_TOTAL_TIMEOUT")
    if raw is None:
        budget = 900.0
    else:
        budget = float(raw)
    if budget <= 0:
        return None
    return time.monotonic() + max(1.0, budget)


def _accent_tts_call_timeout(deadline: float | None) -> float:
    timeout = _accent_tts_tool_timeout_sec()
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AccentConversionError("Accent conversion exceeded the Qwen3-TTS total time budget.")
    return max(0.1, min(timeout, remaining))


def _accent_tts_mcp_grace_sec() -> float:
    return max(0.0, _env_float("X_EDITOR_ACCENT_TTS_MCP_GRACE_SEC", 5.0))


def _accent_tts_mcp_call_timeout(request_timeout: float, deadline: float | None) -> float:
    retry_count = _env_int("X_EDITOR_QWEN3_TTS_RETRIES", 1)
    backoff = max(0.0, _env_float("X_EDITOR_QWEN3_TTS_RETRY_BACKOFF", 1.0))
    backoff_budget = sum(backoff * (2**attempt) for attempt in range(retry_count))
    timeout = max(0.1, request_timeout) * (retry_count + 1) + backoff_budget + _accent_tts_mcp_grace_sec()
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.1
        timeout = min(timeout, remaining + _accent_tts_mcp_grace_sec())
    return max(0.1, timeout)


def _usable_audio_file(path: str | None) -> bool:
    if not path:
        return False
    audio = Path(path)
    try:
        return audio.exists() and audio.stat().st_size > 1024
    except OSError:
        return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _target_text_for_accent_conversion(
    instruction: str,
    source_text: str | None,
    spans: list[TimeSpan],
) -> str:
    quoted = _quoted_strings(instruction)
    if quoted and re.search(r"\b(say|saying|speak|generate|create|synthesize)\b", instruction.lower()):
        return quoted[-1]
    span_texts = [_span_text(span) for span in spans]
    joined = " ".join(text for text in span_texts if text).strip()
    if joined:
        return joined
    return (source_text or "").strip()


def _target_gender(
    instruction: str,
    perception_report: dict[str, Any],
    spans: list[TimeSpan],
) -> str | None:
    lowered = instruction.lower()
    if any(value in lowered for value in ["female", "woman", "girl", "女"]):
        return "female"
    if any(value in lowered for value in ["male", "man", "boy", "男"]):
        return "male"
    profiles = _speaker_profiles(perception_report)
    for span in spans:
        if span.speaker_id and span.speaker_id in profiles:
            profile = profiles[span.speaker_id].lower()
            if "female" in profile or "woman" in profile:
                return "female"
            if "male" in profile or "man" in profile:
                return "male"
    return None


def _speaker_segments(perception_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = perception_report.get("speaker_segments") or perception_report.get("segments") or []
    segments: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if start is None or end is None:
            continue
        speaker_id = item.get("speaker_id", item.get("speaker"))
        segments.append({**item, "start_sec": float(start), "end_sec": float(end), "speaker_id": speaker_id})
    return segments


def _word_timestamps(perception_report: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for item in perception_report.get("word_timestamps") or []:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        text = item.get("text") or item.get("word") or item.get("char")
        if start is None or end is None or not text:
            continue
        words.append({**item, "start_sec": float(start), "end_sec": float(end), "text": str(text)})
    return words


def _text_for_span(segment: dict[str, Any], words: list[dict[str, Any]]) -> str:
    explicit = segment.get("text") or segment.get("transcript")
    if explicit:
        return str(explicit).strip()
    start = float(segment["start_sec"])
    end = float(segment["end_sec"])
    speaker_id = str(segment.get("speaker_id") or "")
    selected = []
    for word in words:
        word_speaker = str(word.get("speaker_id") or word.get("speaker") or "")
        mid = (float(word["start_sec"]) + float(word["end_sec"])) / 2.0
        if start <= mid <= end and (not speaker_id or not word_speaker or word_speaker == speaker_id):
            selected.append(str(word["text"]))
    return _join_words(selected)


def _speaker_profiles(perception_report: dict[str, Any]) -> dict[str, str]:
    profiles: dict[str, str] = {}
    for item in perception_report.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id") or item.get("speaker")
        if speaker_id is None:
            continue
        profiles[str(speaker_id)] = " ".join(
            str(item.get(key) or "")
            for key in ["perceived_gender", "description", "gender"]
        )
    return profiles


def _clip_merge_spans(spans: list[TimeSpan], duration: float) -> list[TimeSpan]:
    clipped = [span.clipped(duration) for span in spans]
    valid = sorted((span for span in clipped if span is not None), key=lambda item: item.start_sec)
    if not valid:
        return []
    merged: list[TimeSpan] = [valid[0]]
    for span in valid[1:]:
        last = merged[-1]
        if span.start_sec <= last.end_sec + 0.08 and span.speaker_id == last.speaker_id:
            merged[-1] = TimeSpan(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, span.end_sec),
                label=last.label,
                speaker_id=last.speaker_id,
                text=_join_words([last.text, span.text]),
                source=last.source,
            )
        else:
            merged.append(span)
    return merged


def _replace_spans(input_audio_path: str, replacements: list[tuple[TimeSpan, str]], output_path: str) -> None:
    source = AudioSegment.from_file(input_audio_path)
    output = AudioSegment.empty()
    cursor_ms = 0
    for span, replacement_path in sorted(replacements, key=lambda item: item[0].start_sec):
        start_ms = max(0, int(round(span.start_sec * 1000)))
        end_ms = min(len(source), int(round(span.end_sec * 1000)))
        if start_ms > cursor_ms:
            output += source[cursor_ms:start_ms]
        replacement = AudioSegment.from_file(replacement_path)
        target_ms = max(0, end_ms - start_ms)
        if len(replacement) < target_ms:
            replacement += AudioSegment.silent(duration=target_ms - len(replacement), frame_rate=replacement.frame_rate)
        output += replacement[:target_ms]
        cursor_ms = end_ms
    if cursor_ms < len(source):
        output += source[cursor_ms:]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.export(output_path, format="wav")


def _fit_audio_to_duration(input_audio_path: str, output_audio_path: str, duration_sec: float) -> None:
    audio = AudioSegment.from_file(input_audio_path)
    target_ms = max(1, int(round(duration_sec * 1000)))
    if len(audio) < target_ms:
        audio += AudioSegment.silent(duration=target_ms - len(audio), frame_rate=audio.frame_rate)
    Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
    audio[:target_ms].export(output_audio_path, format="wav")


def _slice_audio(input_audio_path: str, output_audio_path: str, start_sec: float, end_sec: float) -> None:
    Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            input_audio_path,
            "-t",
            f"{max(0.001, end_sec - start_sec):.3f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_audio_path,
        ],
        check=True,
    )


def _audio_duration_sec(audio_path: str) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _output_audio_path(output: dict[str, Any]) -> str | None:
    for key in ["output_audio_path", "target_audio_path", "output_path", "generated_audio_path"]:
        value = output.get(key)
        if isinstance(value, str):
            return value
    return None


def _is_full_span(spans: list[TimeSpan], input_audio_path: str) -> bool:
    if len(spans) != 1:
        return False
    duration = _audio_duration_sec(input_audio_path)
    span = spans[0]
    return span.start_sec <= 0.05 and span.end_sec >= duration - 0.05


def _span_text(span: TimeSpan) -> str:
    return (span.text or "").strip()


def _join_words(words: list[str]) -> str:
    text = " ".join(word.strip() for word in words if word and word.strip())
    text = re.sub(r"\s+([,.!?;:，。！？；：])", r"\1", text)
    return text.strip()


def _quoted_strings(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", text):
        values.append(next(group for group in match.groups() if group is not None))
    return values


def _special_target_label(text: str) -> str | None:
    lowered = text.lower()
    if "standard english" in lowered or "native english" in lowered or "local american" in lowered:
        return "General American English"
    if "native american accent" in lowered:
        return "General American English"
    if "native chinese speaker" in lowered or "chinese speaker" in lowered:
        return "Chinese-accented English"
    if "standard mandarin" in lowered or "putonghua" in lowered or "普通话" in text:
        return "Standard Mandarin"
    return None


def _prefer_specific_label(labels: list[str]) -> str:
    for label in reversed(labels):
        if label != "Wu Chinese":
            return label
    return labels[-1]


def _allow_compatible_reference(canonical: str) -> bool:
    return canonical in {
        "General American English",
        "Southern American English",
        "British English",
        "Australian English",
        "New Zealand English",
        "Chinese-accented English",
        "Korean-accented English",
        "Japanese-accented English",
        "Thai-accented English",
        "Standard Mandarin",
        "Taiwanese Mandarin",
        "Native Korean",
        "Kansai Japanese",
        "Beijing Dialect",
        "Beijing Guanhua",
    }


def _language_for_text(text: str) -> str | None:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None


def _requires_explicit_speaker(instruction: str) -> bool:
    lowered = instruction.lower()
    return any(
        value in lowered
        for value in [
            "female speaker",
            "male speaker",
            "woman",
            "man's",
            "woman's",
            "first speaker",
            "second speaker",
            "target speaker",
            "女",
            "男",
            "第一说话人",
            "第二说话人",
        ]
    )


def _requires_source_property_span_selection(instruction: str) -> bool:
    """Detect accent-conversion requests that should edit only property-matched spans."""

    text = instruction.lower()
    scope_terms = [
        "only",
        "just",
        "segment",
        "segments",
        "part",
        "parts",
        "portion",
        "portions",
        "where",
        "that are",
        "that is",
        "speech with",
        "speaker with",
        "speaking with",
        "accented",
        "dialectal",
        "局部",
        "片段",
        "部分",
        "带有",
        "具有",
    ]
    property_terms = [
        "accented",
        "accent",
        "dialect",
        "non-native",
        "foreign",
        "japanese",
        "korean",
        "indian",
        "thai",
        "british",
        "american",
        "southern",
        "cantonese",
        "mandarin",
        "sichuan",
        "ningbo",
        "minnan",
        "shanghai",
        "口音",
        "方言",
        "日式",
        "韩式",
        "印度",
        "泰式",
        "英式",
        "美式",
        "粤语",
        "四川",
        "宁波",
        "闽南",
        "上海",
    ]
    if not any(term in text for term in scope_terms):
        return False
    if not any(term in text for term in property_terms):
        return False
    local_terms = ["only", "just", "segment", "segments", "part", "parts", "portion", "portions", "局部", "片段", "部分"]
    if any(term in text for term in ["this speech", "this audio", "entire", "whole", "整段", "全部"]) and not any(
        term in text for term in local_terms
    ):
        return False
    return True


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
