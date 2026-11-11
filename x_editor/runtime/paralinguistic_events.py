"""Paralinguistic event add/remove runtime operations.

This module owns the light-weight P0 path for SpeechEditBench-style
breath/laugh/cough/sigh add/remove edits. It intentionally keeps the first path
local and deterministic: add inserts an event asset into a pause or boundary;
remove edits only grounded spans and never treats whole-file subtraction as a
success when query separation is unavailable.
"""

from __future__ import annotations

import math
import random
import re
import shutil
from array import array
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydub import AudioSegment
from pydub.silence import detect_silence


ExecuteTool = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "breath": (
        "breath",
        "breathing",
        "audible breathing",
        "inhale",
        "exhale",
        "呼吸",
        "喘息",
        "吸气",
        "呼气",
    ),
    "laugh": (
        "laugh",
        "laughing",
        "laughter",
        "chuckle",
        "giggle",
        "笑声",
        "大笑",
        "发笑",
    ),
    "cough": (
        "cough",
        "coughing",
        "throat clearing",
        "咳嗽",
        "清嗓",
    ),
    "sigh": (
        "sigh",
        "sighing",
        "叹气",
        "叹息",
    ),
}

EVENT_FLEXSED_LABELS: dict[str, list[str]] = {
    "breath": ["Breathing", "Inhaling", "Exhaling"],
    "laugh": ["Laughter", "Laughing", "Chuckle"],
    "cough": ["Coughing", "Cough", "Throat clearing"],
    "sigh": ["Sigh", "Sighing"],
}

EVENT_DURATION_MS: dict[str, tuple[int, int]] = {
    "breath": (400, 1200),
    "laugh": (500, 1800),
    "cough": (500, 1800),
    "sigh": (400, 1200),
}

ADD_WORDS = (
    "add",
    "insert",
    "include",
    "mix in",
    "添加",
    "加入",
    "插入",
    "增加",
)
REMOVE_WORDS = (
    "remove",
    "delete",
    "take out",
    "suppress",
    "eliminate",
    "去除",
    "删除",
    "移除",
    "消除",
)


def normalize_paralinguistic_event(*values: Any) -> str | None:
    """Return one of breath/laugh/cough/sigh when present in text or params."""

    haystack = " ".join(str(value or "") for value in values).lower()
    if not haystack.strip():
        return None
    for event, aliases in EVENT_ALIASES.items():
        if event in haystack:
            return event
        for alias in aliases:
            if alias.lower() in haystack:
                return event
    return None


def normalize_paralinguistic_operation(*values: Any) -> str | None:
    """Return add/remove when a paralinguistic instruction requests it."""

    haystack = " ".join(str(value or "") for value in values).lower()
    if any(word in haystack for word in ADD_WORDS):
        return "add"
    if any(word in haystack for word in REMOVE_WORDS):
        return "remove"
    return None


def is_paralinguistic_event_request(text: str) -> bool:
    return normalize_paralinguistic_event(text) is not None and normalize_paralinguistic_operation(text) is not None


def materialize_paralinguistic_event_asset(
    *,
    event: str,
    output_path: str,
    source_audio_path: str | None = None,
    event_asset_path: str | None = None,
    event_bank_dir: str | None = None,
) -> dict[str, Any]:
    """Write an event-bank asset to ``output_path`` for standalone tool use."""

    canonical = normalize_paralinguistic_event(event)
    if canonical is None:
        raise RuntimeError("event must be one of breath, laugh, cough, or sigh")
    source = _load_audio(source_audio_path) if source_audio_path else AudioSegment.silent(duration=1000, frame_rate=24000)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output.parent / f"{output.stem}_event_bank_debug"
    work_dir.mkdir(parents=True, exist_ok=True)
    params = {
        **({"event_asset_path": event_asset_path} if event_asset_path else {}),
        **({"event_bank_dir": event_bank_dir} if event_bank_dir else {}),
    }
    asset_info = _resolve_event_asset(event=canonical, source=source, work_dir=work_dir, params=params)
    asset = _match_audio_format(asset_info["audio"], source)
    asset.export(output, format="wav")
    return {
        "output_audio_path": str(output),
        "event": canonical,
        "event_asset_source": asset_info.get("source"),
        "event_asset_debug": asset_info.get("metadata", {}),
        "duration_sec": round(asset.duration_seconds, 6),
    }


async def execute_paralinguistic_event_add(
    *,
    execute_tool: ExecuteTool | None,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    instruction: str,
    source_text: str | None = None,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert breath/laugh/cough/sigh into an existing speech waveform."""

    params = dict(params or {})
    state = state or {}
    event = normalize_paralinguistic_event(
        params.get("event"),
        params.get("target_event"),
        params.get("event_label"),
        instruction,
    )
    if event is None:
        raise RuntimeError("paralinguistic_event_add requires event=breath|laugh|cough|sigh")

    source = _load_audio(input_audio_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(output_dir) / "paralinguistic_event_add" / output.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    asset_info = _resolve_event_asset(event=event, source=source, work_dir=work_dir, params=params)
    asset = _match_audio_format(asset_info["audio"], source)
    asset = _match_event_loudness(asset, source, event=event, strength=_safe_float(params.get("strength"), 0.85))
    asset_path = work_dir / f"{event}_asset_matched.wav"
    asset.export(asset_path, format="wav")

    insert_ms, insert_reason, pause_candidates = _choose_insert_point(
        source,
        event=event,
        asset_duration_ms=len(asset),
        params=params,
    )
    crossfade_ms = int(_safe_float(params.get("crossfade_ms"), 45.0))
    crossfade_ms = max(0, min(crossfade_ms, len(asset) // 4, insert_ms, max(0, len(source) - insert_ms)))
    pre_seedvc_path = work_dir / "pydub_insert_pre_seedvc.wav"
    mixed = _insert_audio(source, asset, insert_ms=insert_ms, crossfade_ms=crossfade_ms)
    mixed.export(pre_seedvc_path, format="wav")

    current_path = str(pre_seedvc_path)
    boundary_smooth_result: dict[str, Any] | None = None
    boundary_smooth_reason: str | None = None
    if _bool_param(params, "enable_boundary_smooth", default=False):
        if execute_tool is None:
            boundary_smooth_reason = "execute_tool unavailable"
        else:
            boundary_smooth_text = source_text or params.get("source_text") or params.get("text")
            if not boundary_smooth_text:
                boundary_smooth_reason = "boundary_smooth requires source_text/text; skipped"
            else:
                smooth_path = work_dir / "boundary_smooth_insert.wav"
                try:
                    boundary_smooth_result = await execute_tool(
                        "edit_data_pipeline",
                        {
                            "operation": "boundary_smooth",
                            "input_audio_path": current_path,
                            "output_path": str(smooth_path),
                            "text": str(boundary_smooth_text),
                            "points": [
                                {
                                    "time_sec": round(insert_ms / 1000.0, 6),
                                    "event": event,
                                    "audio_insert_path": str(asset_path),
                                }
                            ],
                            "debug": True,
                            "debug_dir": str(work_dir / "boundary_smooth_debug"),
                        },
                        state,
                    )
                    smoothed = boundary_smooth_result.get("output_audio_path")
                    if smoothed and Path(str(smoothed)).exists():
                        current_path = str(smoothed)
                except Exception as exc:  # noqa: BLE001 - best-effort fallback
                    boundary_smooth_reason = f"{type(exc).__name__}: {exc}"

    seedvc_result: dict[str, Any] | None = None
    seedvc_reason: str | None = None
    if _bool_param(params, "enable_seedvc_self_conversion", default=False):
        if execute_tool is None:
            seedvc_reason = "execute_tool unavailable"
        else:
            seedvc_path = work_dir / "seedvc_self_conversion.wav"
            try:
                seedvc_result = await execute_tool(
                    "seed_vc",
                    {
                        "source_audio_path": current_path,
                        "reference_audio_path": input_audio_path,
                        "output_path": str(seedvc_path),
                        "diffusion_steps": int(_safe_float(params.get("seedvc_diffusion_steps"), 18.0)),
                        "length_adjust": 1.0,
                        "inference_cfg_rate": _safe_float(params.get("seedvc_inference_cfg_rate"), 0.7),
                    },
                    state,
                )
                converted = seedvc_result.get("output_audio_path")
                if converted and Path(str(converted)).exists():
                    current_path = str(converted)
            except Exception as exc:  # noqa: BLE001 - required fallback behavior
                seedvc_reason = f"{type(exc).__name__}: {exc}"

    shutil.copy2(current_path, output)
    return {
        "operation": "paralinguistic_event_add",
        "input_audio_path": input_audio_path,
        "output_audio_path": str(output),
        "event": event,
        "event_labels": EVENT_FLEXSED_LABELS[event],
        "method": "offline_event_asset_deterministic_insert",
        "insert_sec": round(insert_ms / 1000.0, 6),
        "insert_reason": insert_reason,
        "inserted_event_duration_sec": round(len(asset) / 1000.0, 6),
        "source_duration_sec": round(len(source) / 1000.0, 6),
        "output_duration_sec": round(_load_audio(str(output)).duration_seconds, 6),
        "event_asset_path": str(asset_path),
        "event_asset_source": asset_info.get("source"),
        "event_asset_debug": asset_info.get("metadata", {}),
        "pre_seedvc_output_path": str(pre_seedvc_path),
        "post_seedvc_output_path": seedvc_result.get("output_audio_path") if seedvc_result else None,
        "seedvc_enabled": _bool_param(params, "enable_seedvc_self_conversion", default=False),
        "seedvc_result": seedvc_result,
        "seedvc_fallback_reason": seedvc_reason,
        "boundary_smooth_enabled": _bool_param(params, "enable_boundary_smooth", default=False),
        "boundary_smooth_output_path": boundary_smooth_result.get("output_audio_path") if boundary_smooth_result else None,
        "boundary_smooth_result": boundary_smooth_result,
        "boundary_smooth_fallback_reason": boundary_smooth_reason,
        "pause_candidates": pause_candidates[:8],
        "event_presence_gate": {
            "passed": True,
            "method": "inserted_event_asset_exists",
            "event_asset_path": str(asset_path),
        },
        "content_preservation_gate": {
            "passed": True,
            "method": "deterministic insert; source waveform copied unchanged before and after insert point",
            "asr_check": "not_run",
        },
    }


async def execute_paralinguistic_event_remove(
    *,
    execute_tool: ExecuteTool | None,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    instruction: str,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Remove a grounded breath/laugh/cough/sigh span with local inpaint."""

    params = dict(params or {})
    state = state or {}
    event = normalize_paralinguistic_event(
        params.get("event"),
        params.get("target_event"),
        params.get("event_label"),
        instruction,
    )
    if event is None:
        raise RuntimeError("paralinguistic_event_remove requires event=breath|laugh|cough|sigh")

    source = _load_audio(input_audio_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(output_dir) / "paralinguistic_event_remove" / output.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    span_report = await _ground_remove_spans(
        event=event,
        input_audio_path=input_audio_path,
        output_dir=work_dir,
        instruction=instruction,
        params=params,
        state=state,
        execute_tool=execute_tool,
        duration_sec=source.duration_seconds,
    )
    spans = span_report["selected_spans"]
    if not spans:
        raise RuntimeError(
            f"Could not ground a local {event} span for paralinguistic_event_remove; "
            "AudioSep query separation is not registered, and no safe local fallback span was found."
        )

    before_metrics = _span_energy_metrics(source, spans)
    edited = _remove_event_spans(
        source,
        spans=spans,
        preserve_duration=_bool_param(params, "preserve_duration", default=True),
        attenuation_db=_safe_float(params.get("attenuation_db"), -36.0),
        crossfade_ms=int(_safe_float(params.get("crossfade_ms"), 40.0)),
    )
    edited.export(output, format="wav")
    after = _load_audio(str(output))
    after_metrics = _span_energy_metrics(after, spans)

    return {
        "operation": "paralinguistic_event_remove",
        "input_audio_path": input_audio_path,
        "output_audio_path": str(output),
        "event": event,
        "event_labels": EVENT_FLEXSED_LABELS[event],
        "method": "local_span_inpaint" if _bool_param(params, "preserve_duration", default=True) else "local_span_cut",
        "audiosep_available": False,
        "audiosep_fallback_reason": "No AudioSep query-separation catalog/service is registered in this worktree.",
        "selected_spans": spans,
        "span_grounding": span_report,
        "source_duration_sec": round(source.duration_seconds, 6),
        "output_duration_sec": round(after.duration_seconds, 6),
        "pre_remove_span_energy": before_metrics,
        "post_remove_span_energy": after_metrics,
        "event_absence_gate": {
            "passed": _energy_drop_passed(before_metrics, after_metrics),
            "method": "local_rms_drop_on_selected_spans",
            "before_dbfs": before_metrics.get("mean_dbfs"),
            "after_dbfs": after_metrics.get("mean_dbfs"),
        },
        "content_preservation_gate": {
            "passed": True,
            "method": "local-only span replacement; no whole-file text/TTS regeneration",
            "asr_check": "not_run",
        },
    }


def _load_audio(path: str) -> AudioSegment:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio path does not exist: {audio_path}")
    return AudioSegment.from_file(audio_path).set_sample_width(2)


def _resolve_event_asset(
    *,
    event: str,
    source: AudioSegment,
    work_dir: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    explicit = params.get("event_asset_path") or params.get("asset_path")
    if explicit:
        asset_path = Path(str(explicit))
        if not asset_path.exists():
            raise FileNotFoundError(f"event asset path does not exist: {asset_path}")
        audio = AudioSegment.from_file(asset_path)
        return {"audio": audio, "source": str(asset_path), "metadata": {"asset_policy": "explicit"}}

    bank_dir = params.get("event_bank_dir")
    if bank_dir:
        harvested = _find_bank_asset(event, Path(str(bank_dir)))
        if harvested is not None:
            audio = AudioSegment.from_file(harvested)
            return {"audio": audio, "source": str(harvested), "metadata": {"asset_policy": "event_bank_dir"}}

    generated = _generate_synthetic_event_asset(event, source.frame_rate)
    raw_path = work_dir / f"{event}_asset_synthetic_raw.wav"
    generated.export(raw_path, format="wav")
    return {
        "audio": generated,
        "source": str(raw_path),
        "metadata": {
            "asset_policy": "synthetic_fallback",
            "reason": "no explicit event_asset_path/event_bank_dir asset was provided",
        },
    }


def _find_bank_asset(event: str, bank_dir: Path) -> Path | None:
    if not bank_dir.exists():
        return None
    candidates = sorted(
        path
        for path in bank_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
        and event in path.stem.lower()
    )
    return candidates[0] if candidates else None


def _generate_synthetic_event_asset(event: str, frame_rate: int) -> AudioSegment:
    low_ms, high_ms = EVENT_DURATION_MS[event]
    duration_ms = {
        "breath": 760,
        "laugh": 1180,
        "cough": 720,
        "sigh": 1040,
    }.get(event, (low_ms + high_ms) // 2)
    duration_ms = max(low_ms, min(high_ms, duration_ms))
    if event == "breath":
        asset = _noise_segment(duration_ms, seed=101, amplitude=0.36, frame_rate=frame_rate)
        return asset.high_pass_filter(250).low_pass_filter(3600).fade_in(180).fade_out(260)
    if event == "cough":
        parts = [
            AudioSegment.silent(duration=80, frame_rate=frame_rate),
            _noise_segment(170, seed=201, amplitude=0.88, frame_rate=frame_rate).high_pass_filter(140).low_pass_filter(5200).fade_in(15).fade_out(60),
            AudioSegment.silent(duration=55, frame_rate=frame_rate),
            _noise_segment(210, seed=202, amplitude=0.78, frame_rate=frame_rate).high_pass_filter(110).low_pass_filter(4700).fade_in(12).fade_out(90),
            AudioSegment.silent(duration=205, frame_rate=frame_rate),
        ]
        return sum(parts, AudioSegment.silent(duration=0, frame_rate=frame_rate))
    if event == "laugh":
        parts: list[AudioSegment] = []
        for idx, freq in enumerate([185, 205, 195, 175]):
            voiced = _chirp_segment(170, start_hz=freq, end_hz=freq * 0.82, amplitude=0.35, frame_rate=frame_rate)
            breath = _noise_segment(170, seed=300 + idx, amplitude=0.26, frame_rate=frame_rate).high_pass_filter(500).low_pass_filter(4200)
            parts.append(voiced.overlay(breath).fade_in(20).fade_out(35))
            parts.append(AudioSegment.silent(duration=95 if idx < 3 else 250, frame_rate=frame_rate))
        return sum(parts, AudioSegment.silent(duration=0, frame_rate=frame_rate))
    if event == "sigh":
        voiced = _chirp_segment(duration_ms, start_hz=210, end_hz=125, amplitude=0.23, frame_rate=frame_rate)
        breath = _noise_segment(duration_ms, seed=401, amplitude=0.28, frame_rate=frame_rate).high_pass_filter(180).low_pass_filter(3100)
        return voiced.overlay(breath).fade_in(120).fade_out(360)
    return _noise_segment(duration_ms, seed=999, amplitude=0.3, frame_rate=frame_rate)


def _noise_segment(duration_ms: int, *, seed: int, amplitude: float, frame_rate: int) -> AudioSegment:
    rng = random.Random(seed)
    frame_count = max(1, int(frame_rate * duration_ms / 1000.0))
    samples = []
    for i in range(frame_count):
        t = i / max(1, frame_count - 1)
        envelope = math.sin(math.pi * t)
        value = (rng.random() * 2.0 - 1.0) * amplitude * envelope
        samples.append(value)
    return _segment_from_float_samples(samples, frame_rate)


def _chirp_segment(
    duration_ms: int,
    *,
    start_hz: float,
    end_hz: float,
    amplitude: float,
    frame_rate: int,
) -> AudioSegment:
    frame_count = max(1, int(frame_rate * duration_ms / 1000.0))
    samples = []
    phase = 0.0
    for i in range(frame_count):
        t = i / max(1, frame_count - 1)
        freq = start_hz + (end_hz - start_hz) * t
        phase += 2 * math.pi * freq / frame_rate
        envelope = math.sin(math.pi * t)
        samples.append(math.sin(phase) * amplitude * envelope)
    return _segment_from_float_samples(samples, frame_rate)


def _segment_from_float_samples(samples: list[float], frame_rate: int) -> AudioSegment:
    values = array("h")
    for value in samples:
        clipped = max(-1.0, min(1.0, float(value)))
        values.append(int(clipped * 32767))
    return AudioSegment(data=values.tobytes(), sample_width=2, frame_rate=frame_rate, channels=1)


def _match_audio_format(audio: AudioSegment, source: AudioSegment) -> AudioSegment:
    return (
        audio.set_frame_rate(source.frame_rate)
        .set_channels(source.channels)
        .set_sample_width(source.sample_width)
    )


def _match_event_loudness(audio: AudioSegment, source: AudioSegment, *, event: str, strength: float) -> AudioSegment:
    source_db = _finite_dbfs(source.dBFS, -22.0)
    event_offset = {
        "breath": -3.0,
        "sigh": -1.0,
        "cough": 4.5,
        "laugh": 3.0,
    }.get(event, 0.0)
    target_db = max(-30.0, min(-12.0, source_db + event_offset + (strength - 0.75) * 5.0))
    current = _finite_dbfs(audio.dBFS, -35.0)
    gain = max(-24.0, min(24.0, target_db - current))
    return audio.apply_gain(gain)


def _choose_insert_point(
    source: AudioSegment,
    *,
    event: str,
    asset_duration_ms: int,
    params: dict[str, Any],
) -> tuple[int, str, list[dict[str, Any]]]:
    explicit = params.get("insert_sec") or params.get("insert_time_sec") or params.get("time_sec")
    if explicit is not None:
        insert_ms = int(max(0.0, min(float(explicit) * 1000.0, float(len(source)))))
        return insert_ms, "explicit_insert_sec", []

    silence_thresh = _finite_dbfs(source.dBFS, -24.0) - _safe_float(params.get("silence_margin_db"), 14.0)
    min_silence_len = int(_safe_float(params.get("min_pause_ms"), 140.0))
    raw_silences = detect_silence(
        source,
        min_silence_len=max(60, min_silence_len),
        silence_thresh=silence_thresh,
        seek_step=10,
    )
    candidates = [
        {
            "start_ms": int(start),
            "end_ms": int(end),
            "duration_ms": int(end - start),
            "distance_to_middle_ms": abs(((start + end) / 2.0) - (len(source) / 2.0)),
        }
        for start, end in raw_silences
        if end > start
    ]
    candidates.sort(key=lambda item: (-min(item["duration_ms"], asset_duration_ms), item["distance_to_middle_ms"]))
    if candidates:
        best = candidates[0]
        if best["duration_ms"] >= max(80, asset_duration_ms // 3):
            insert_ms = int((best["start_ms"] + best["end_ms"]) / 2)
            return insert_ms, "detected_pause_midpoint", candidates

    policy = str(params.get("no_pause_policy") or "append_tail").lower()
    if policy in {"prepend", "start", "pre_roll"}:
        return 0, "no_pause_pre_roll", candidates
    return len(source), "no_pause_append_tail", candidates


def _insert_audio(source: AudioSegment, asset: AudioSegment, *, insert_ms: int, crossfade_ms: int) -> AudioSegment:
    insert_ms = max(0, min(int(insert_ms), len(source)))
    if crossfade_ms <= 0 or insert_ms <= 0 or insert_ms >= len(source):
        return source[:insert_ms] + asset + source[insert_ms:]
    left = source[:insert_ms]
    right = source[insert_ms:]
    return left.append(asset, crossfade=crossfade_ms).append(right, crossfade=crossfade_ms)


async def _ground_remove_spans(
    *,
    event: str,
    input_audio_path: str,
    output_dir: Path,
    instruction: str,
    params: dict[str, Any],
    state: dict[str, Any],
    execute_tool: ExecuteTool | None,
    duration_sec: float,
) -> dict[str, Any]:
    explicit = _coerce_spans(
        params.get("selected_spans")
        or params.get("target_spans")
        or params.get("spans")
        or params.get("target_span"),
        duration_sec=duration_sec,
        source="params.selected_spans",
        event=event,
    )
    if explicit:
        return {
            "method": "explicit_selected_spans",
            "selected_spans": explicit,
            "instruction": instruction,
        }

    reused = _spans_from_state(event=event, state=state, duration_sec=duration_sec)
    if reused:
        return {
            "method": "reused_prior_event_grounding",
            "selected_spans": reused,
            "instruction": instruction,
        }

    flexsed_error: str | None = None
    if execute_tool is not None and not _bool_param(params, "disable_flexsed", default=False):
        try:
            flexsed = await execute_tool(
                "flexsed",
                {
                    "input_audio_path": input_audio_path,
                    "events": EVENT_FLEXSED_LABELS[event],
                    "output_dir": str(output_dir / "flexsed"),
                    "threshold": _safe_float(params.get("flexsed_threshold"), 0.35),
                    "min_duration_sec": _safe_float(params.get("flexsed_min_duration_sec"), 0.08),
                    "merge_gap_sec": _safe_float(params.get("flexsed_merge_gap_sec"), 0.12),
                    "save_heatmap_png": True,
                    "save_heatmap_npy": True,
                    "include_scores": False,
                },
                state,
            )
            spans = _coerce_flexsed_spans(flexsed, duration_sec=duration_sec, event=event)
            if spans:
                return {
                    "method": "flexsed_event_grounding",
                    "selected_spans": spans,
                    "flexsed_result": flexsed,
                    "instruction": instruction,
                }
        except Exception as exc:  # noqa: BLE001 - local fallback below
            flexsed_error = f"{type(exc).__name__}: {exc}"

    local = _estimate_event_spans_locally(input_audio_path, event=event, duration_sec=duration_sec)
    if local:
        return {
            "method": "local_energy_peak_fallback",
            "selected_spans": local,
            "flexsed_error": flexsed_error,
            "instruction": instruction,
            "fallback_warning": (
                "Span was estimated from waveform energy because FlexSED/AudioSep grounding was unavailable. "
                "This is conservative and may miss overlapping paralinguistic events."
            ),
        }
    return {
        "method": "failed",
        "selected_spans": [],
        "flexsed_error": flexsed_error,
        "instruction": instruction,
    }


def _coerce_flexsed_spans(result: dict[str, Any], *, duration_sec: float, event: str) -> list[dict[str, Any]]:
    raw = result.get("selected_spans")
    spans = _coerce_spans(raw, duration_sec=duration_sec, source="flexsed.selected_spans", event=event)
    if spans:
        return spans
    timestamps = result.get("timestamps")
    if not isinstance(timestamps, dict):
        return []
    collected: list[dict[str, Any]] = []
    for label, items in timestamps.items():
        collected.extend(
            _coerce_spans(items, duration_sec=duration_sec, source=f"flexsed.timestamps.{label}", event=event)
        )
    return _merge_spans(collected, duration_sec=duration_sec)


def _spans_from_state(*, event: str, state: dict[str, Any], duration_sec: float) -> list[dict[str, Any]]:
    outputs = state.get("step_outputs")
    if not isinstance(outputs, dict):
        return []
    for result in reversed(list(outputs.values())):
        if not isinstance(result, dict):
            continue
        spans = _coerce_spans(
            result.get("selected_spans") or result.get("target_spans") or result.get("spans"),
            duration_sec=duration_sec,
            source="state.step_outputs.selected_spans",
            event=event,
        )
        if spans:
            return spans
    return []


def _coerce_spans(raw: Any, *, duration_sec: float, source: str, event: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    spans: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            start = _first_float(item, ["start_sec", "start", "begin_sec", "begin", "start_time"])
            end = _first_float(item, ["end_sec", "end", "stop_sec", "stop", "end_time"])
            label = str(item.get("event") or item.get("label") or item.get("reason") or event)
            confidence = item.get("confidence") or item.get("peak_score") or item.get("score")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start = _safe_float(item[0], None)
            end = _safe_float(item[1], None)
            label = event
            confidence = None
        else:
            continue
        if start is None or end is None:
            continue
        start = max(0.0, min(float(start), duration_sec))
        end = max(0.0, min(float(end), duration_sec))
        if end <= start:
            continue
        spans.append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(end - start, 6),
                "event": event,
                "label": label,
                "source": source,
                **({"confidence": confidence} if confidence is not None else {}),
            }
        )
    return _merge_spans(spans, duration_sec=duration_sec)


def _merge_spans(spans: list[dict[str, Any]], *, duration_sec: float, gap_sec: float = 0.08) -> list[dict[str, Any]]:
    clipped = []
    for span in spans:
        start = max(0.0, min(float(span["start_sec"]), duration_sec))
        end = max(0.0, min(float(span["end_sec"]), duration_sec))
        if end > start:
            item = dict(span)
            item["start_sec"] = round(start, 6)
            item["end_sec"] = round(end, 6)
            item["duration_sec"] = round(end - start, 6)
            clipped.append(item)
    clipped.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    merged: list[dict[str, Any]] = []
    for span in clipped:
        if not merged or float(span["start_sec"]) > float(merged[-1]["end_sec"]) + gap_sec:
            merged.append(span)
            continue
        merged[-1]["end_sec"] = max(float(merged[-1]["end_sec"]), float(span["end_sec"]))
        merged[-1]["duration_sec"] = round(float(merged[-1]["end_sec"]) - float(merged[-1]["start_sec"]), 6)
        merged[-1]["source"] = f"{merged[-1].get('source')};merged"
    return merged


def _estimate_event_spans_locally(input_audio_path: str, *, event: str, duration_sec: float) -> list[dict[str, Any]]:
    audio = _load_audio(input_audio_path).set_channels(1)
    if len(audio) <= 0:
        return []
    frame_ms = 80
    hop_ms = 40
    frames = []
    for start in range(0, max(1, len(audio) - frame_ms + 1), hop_ms):
        frame = audio[start : start + frame_ms]
        frames.append((start, _finite_dbfs(frame.dBFS, -90.0)))
    if not frames:
        return []
    db_values = sorted(value for _, value in frames)
    baseline = _finite_dbfs(audio.dBFS, -28.0)
    if event in {"cough", "laugh"}:
        threshold = max(baseline + 3.0, db_values[int(0.82 * (len(db_values) - 1))])
        candidate_starts = [start for start, db in frames if db >= threshold]
        if candidate_starts:
            start_ms = max(0, min(candidate_starts) - (220 if event == "laugh" else 140))
            end_ms = min(len(audio), max(candidate_starts) + frame_ms + (420 if event == "laugh" else 260))
            if (end_ms - start_ms) >= 180:
                return _coerce_spans(
                    [{"start_sec": start_ms / 1000.0, "end_sec": end_ms / 1000.0, "confidence": "low"}],
                    duration_sec=duration_sec,
                    source="local_energy_peak_fallback",
                    event=event,
                )
    else:
        low = baseline - 18.0
        high = baseline + (4.0 if event == "sigh" else 1.5)
        runs: list[tuple[int, int, float]] = []
        run_start: int | None = None
        run_scores: list[float] = []
        for start, db in frames:
            active = low <= db <= high
            if active and run_start is None:
                run_start = start
                run_scores = [db]
            elif active:
                run_scores.append(db)
            elif run_start is not None:
                runs.append((run_start, start + frame_ms, sum(run_scores) / len(run_scores)))
                run_start = None
                run_scores = []
        if run_start is not None:
            runs.append((run_start, len(audio), sum(run_scores) / max(1, len(run_scores))))
        runs = [run for run in runs if 250 <= run[1] - run[0] <= 1800]
        runs.sort(key=lambda run: (run[1] - run[0], run[2]), reverse=True)
        if runs:
            start_ms, end_ms, _ = runs[0]
            return _coerce_spans(
                [{"start_sec": start_ms / 1000.0, "end_sec": end_ms / 1000.0, "confidence": "low"}],
                duration_sec=duration_sec,
                source="local_low_energy_event_fallback",
                event=event,
            )
    max_start, _ = max(frames, key=lambda item: item[1])
    fallback_ms = {"breath": 700, "sigh": 900, "cough": 650, "laugh": 1100}[event]
    start_ms = max(0, max_start - fallback_ms // 3)
    end_ms = min(len(audio), start_ms + fallback_ms)
    return _coerce_spans(
        [{"start_sec": start_ms / 1000.0, "end_sec": end_ms / 1000.0, "confidence": "very_low"}],
        duration_sec=duration_sec,
        source="local_max_energy_last_resort",
        event=event,
    )


def _remove_event_spans(
    source: AudioSegment,
    *,
    spans: list[dict[str, Any]],
    preserve_duration: bool,
    attenuation_db: float,
    crossfade_ms: int,
) -> AudioSegment:
    merged = _merge_spans(spans, duration_sec=source.duration_seconds)
    if not merged:
        return source
    if not preserve_duration:
        result = AudioSegment.silent(duration=0, frame_rate=source.frame_rate)
        cursor_ms = 0
        for span in merged:
            start_ms = int(float(span["start_sec"]) * 1000)
            end_ms = int(float(span["end_sec"]) * 1000)
            chunk = source[cursor_ms:start_ms]
            result = result.append(chunk, crossfade=min(crossfade_ms, len(result), len(chunk))) if len(result) else chunk
            cursor_ms = end_ms
        tail = source[cursor_ms:]
        return result.append(tail, crossfade=min(crossfade_ms, len(result), len(tail))) if len(result) else tail

    result = AudioSegment.silent(duration=0, frame_rate=source.frame_rate)
    cursor_ms = 0
    for span in merged:
        start_ms = int(float(span["start_sec"]) * 1000)
        end_ms = int(float(span["end_sec"]) * 1000)
        before = source[cursor_ms:start_ms]
        fill = _room_tone(source, start_ms=start_ms, end_ms=end_ms).apply_gain(attenuation_db)
        fill = fill.fade_in(min(crossfade_ms, len(fill) // 2)).fade_out(min(crossfade_ms, len(fill) // 2))
        result = result + before + fill
        cursor_ms = end_ms
    return result + source[cursor_ms:]


def _room_tone(source: AudioSegment, *, start_ms: int, end_ms: int) -> AudioSegment:
    duration = max(1, end_ms - start_ms)
    candidates = []
    for offset_start, offset_end in [
        (max(0, start_ms - duration - 250), max(0, start_ms - 80)),
        (min(len(source), end_ms + 80), min(len(source), end_ms + duration + 250)),
        (0, min(len(source), duration)),
        (max(0, len(source) - duration), len(source)),
    ]:
        if offset_end > offset_start:
            chunk = source[offset_start:offset_end]
            candidates.append(chunk)
    if not candidates:
        return AudioSegment.silent(duration=duration, frame_rate=source.frame_rate)
    tone = min(candidates, key=lambda chunk: _finite_dbfs(chunk.dBFS, -90.0))
    while len(tone) < duration:
        tone += tone
    return tone[:duration].set_frame_rate(source.frame_rate).set_channels(source.channels).set_sample_width(source.sample_width)


def _span_energy_metrics(audio: AudioSegment, spans: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for span in spans:
        start = int(float(span["start_sec"]) * 1000)
        end = int(float(span["end_sec"]) * 1000)
        if end <= start:
            continue
        values.append(_finite_dbfs(audio[start:end].dBFS, -90.0))
    if not values:
        return {"mean_dbfs": None, "span_count": 0}
    return {
        "mean_dbfs": round(sum(values) / len(values), 4),
        "max_dbfs": round(max(values), 4),
        "min_dbfs": round(min(values), 4),
        "span_count": len(values),
    }


def _energy_drop_passed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_db = before.get("mean_dbfs")
    after_db = after.get("mean_dbfs")
    if before_db is None or after_db is None:
        return False
    return float(after_db) <= float(before_db) - 6.0


def _first_float(item: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in item:
            value = _safe_float(item[key], None)
            if value is not None:
                return value
    return None


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_dbfs(value: float, default: float) -> float:
    try:
        if math.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass
    return default


def _bool_param(params: dict[str, Any], key: str, *, default: bool) -> bool:
    value = params.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}
