"""Timestamp utilities for FlexSED frame-level event scores."""

from __future__ import annotations

from typing import Any

import numpy as np


FRAME_RATE_HZ = 25.0
FRAME_STEP_SEC = 1.0 / FRAME_RATE_HZ


def extract_event_timestamps(
    scores: np.ndarray,
    events: list[str],
    *,
    threshold: float = 0.5,
    min_duration_sec: float = 0.08,
    merge_gap_sec: float = 0.08,
    frame_rate_hz: float = FRAME_RATE_HZ,
    audio_duration_sec: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convert FlexSED frame probabilities into merged timestamp spans."""

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a 2-D [events, frames] matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(events):
        raise ValueError(f"events length {len(events)} does not match score rows {matrix.shape[0]}")
    min_frames = max(1, int(np.ceil(min_duration_sec * frame_rate_hz)))
    merge_gap_frames = int(np.ceil(merge_gap_sec * frame_rate_hz))
    output: dict[str, list[dict[str, Any]]] = {}
    for row_index, event in enumerate(events):
        row = matrix[row_index]
        active = np.flatnonzero(row >= threshold)
        raw_ranges = _contiguous_ranges(active)
        merged_ranges = _merge_ranges(raw_ranges, max_gap_frames=merge_gap_frames)
        spans: list[dict[str, Any]] = []
        for start_frame, end_frame_exclusive in merged_ranges:
            if end_frame_exclusive - start_frame < min_frames:
                continue
            start_sec = start_frame / frame_rate_hz
            end_sec = end_frame_exclusive / frame_rate_hz
            if audio_duration_sec is not None:
                start_sec = max(0.0, min(start_sec, audio_duration_sec))
                end_sec = max(start_sec, min(end_sec, audio_duration_sec))
            window = row[start_frame:end_frame_exclusive]
            spans.append(
                {
                    "event": event,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(max(0.0, end_sec - start_sec), 3),
                    "frame_start": int(start_frame),
                    "frame_end": int(end_frame_exclusive - 1),
                    "threshold": threshold,
                    "peak_score": round(float(np.max(window)), 6) if window.size else 0.0,
                    "mean_score": round(float(np.mean(window)), 6) if window.size else 0.0,
                }
            )
        output[event] = spans
    return output


def propose_event_peak_windows(
    scores: np.ndarray,
    events: list[str],
    *,
    frame_rate_hz: float = FRAME_RATE_HZ,
    audio_duration_sec: float | None = None,
    max_candidates_per_event: int = 4,
    min_peak_separation_sec: float = 0.5,
    context_sec: float = 0.2,
) -> list[dict[str, Any]]:
    """Expose low-confidence score peaks as review proposals, not detections."""

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a 2-D [events, frames] matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(events):
        raise ValueError(f"events length {len(events)} does not match score rows {matrix.shape[0]}")
    separation_frames = max(1, int(np.ceil(min_peak_separation_sec * frame_rate_hz)))
    context_frames = max(0, int(np.ceil(context_sec * frame_rate_hz)))
    proposals: list[dict[str, Any]] = []
    for row_index, event in enumerate(events):
        row = np.nan_to_num(matrix[row_index], nan=0.0, posinf=0.0, neginf=0.0)
        if row.size == 0 or float(np.max(row)) <= 0.0:
            continue
        median = float(np.median(row))
        mad = float(np.median(np.abs(row - median)))
        proposal_floor = max(float(np.quantile(row, 0.75)), median + 1.5 * mad)
        local_peaks = [
            index
            for index, score in enumerate(row)
            if float(score) >= proposal_floor
            and (index == 0 or score >= row[index - 1])
            and (index == row.size - 1 or score >= row[index + 1])
        ]
        if not local_peaks:
            local_peaks = [int(np.argmax(row))]
        selected: list[int] = []
        for peak_index in sorted(local_peaks, key=lambda index: float(row[index]), reverse=True):
            if any(abs(peak_index - existing) < separation_frames for existing in selected):
                continue
            selected.append(peak_index)
            if len(selected) >= max_candidates_per_event:
                break
        for peak_index in selected:
            peak_score = float(row[peak_index])
            support_floor = max(median + mad, peak_score * 0.5)
            start_frame = peak_index
            end_frame = peak_index + 1
            while start_frame > 0 and float(row[start_frame - 1]) >= support_floor:
                start_frame -= 1
            while end_frame < row.size and float(row[end_frame]) >= support_floor:
                end_frame += 1
            start_frame = max(0, start_frame - context_frames)
            end_frame = min(row.size, end_frame + context_frames)
            start_sec = start_frame / frame_rate_hz
            end_sec = end_frame / frame_rate_hz
            if audio_duration_sec is not None:
                start_sec = max(0.0, min(start_sec, audio_duration_sec))
                end_sec = max(start_sec, min(end_sec, audio_duration_sec))
            proposals.append(
                {
                    "candidate_id": f"flexsed_peak_{row_index:02d}_{peak_index:05d}",
                    "event": event,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "peak_time_sec": round(peak_index / frame_rate_hz, 3),
                    "peak_score": round(peak_score, 6),
                    "proposal_floor": round(proposal_floor, 6),
                    "support_kind": "soft_peak_proposal",
                    "requires_model_review": True,
                }
            )
    proposals.sort(key=lambda item: (str(item["event"]), -float(item["peak_score"])))
    return proposals


def summarize_window_events(
    scores: np.ndarray,
    events: list[str],
    *,
    start_sec: float,
    end_sec: float,
    p: int | float = 3,
    frame_rate_hz: float = FRAME_RATE_HZ,
    audio_duration_sec: float | None = None,
    score_mode: str = "max",
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Summarize the strongest candidate events inside one time window.

    ``p`` accepts two top-p conventions:
    - integer or float > 1: return the top ``int(p)`` events.
    - 0 < float <= 1: return the nucleus set whose normalized cumulative
      window score reaches ``p``.
    """

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a 2-D [events, frames] matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(events):
        raise ValueError(f"events length {len(events)} does not match score rows {matrix.shape[0]}")
    if start_sec < 0:
        start_sec = 0.0
    if audio_duration_sec is not None:
        end_sec = min(float(end_sec), float(audio_duration_sec))
    if end_sec <= start_sec:
        raise ValueError(f"end_sec must be greater than start_sec, got start={start_sec}, end={end_sec}")

    frame_count = matrix.shape[1]
    start_frame = max(0, min(frame_count, int(np.floor(start_sec * frame_rate_hz))))
    if start_frame >= frame_count:
        raise ValueError(
            f"start_sec {start_sec} is outside the available score range "
            f"(duration {frame_count / frame_rate_hz:.3f}s)"
        )
    end_frame = max(start_frame + 1, min(frame_count, int(np.ceil(end_sec * frame_rate_hz))))
    window = matrix[:, start_frame:end_frame]
    mode = score_mode.strip().lower()
    if mode == "mean":
        event_scores = np.mean(window, axis=1)
    elif mode in {"p95", "percentile95", "95p"}:
        event_scores = np.percentile(window, 95, axis=1)
    elif mode == "max":
        event_scores = np.max(window, axis=1)
    else:
        raise ValueError(f"unsupported score_mode: {score_mode}")

    ranked = [
        {"event": event, "score": float(score), "rank": index + 1}
        for index, (event, score) in enumerate(
            sorted(zip(events, event_scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        )
        if float(score) >= min_score
    ]
    selected = _select_top_p(ranked, p)
    actual_start_sec = start_frame / frame_rate_hz
    actual_end_sec = end_frame / frame_rate_hz
    if audio_duration_sec is not None:
        actual_end_sec = min(actual_end_sec, audio_duration_sec)
    line = f"[{_format_time(actual_start_sec)}-{_format_time(actual_end_sec)}] " + ", ".join(
        item["event"] for item in selected
    )
    return {
        "start_sec": round(actual_start_sec, 3),
        "end_sec": round(actual_end_sec, 3),
        "frame_start": int(start_frame),
        "frame_end": int(end_frame - 1),
        "p": p,
        "top_p_mode": "top_k" if float(p) > 1 else "nucleus",
        "score_mode": mode,
        "min_score": min_score,
        "events": selected,
        "ranked_events": ranked,
        "line": line.strip(),
    }


def _select_top_p(ranked: list[dict[str, Any]], p: int | float) -> list[dict[str, Any]]:
    if not ranked:
        return []
    value = float(p)
    if value <= 0:
        raise ValueError(f"p must be positive, got {p}")
    if value > 1:
        count = max(1, int(value))
        return ranked[:count]
    total = sum(max(0.0, float(item["score"])) for item in ranked)
    if total <= 0:
        return ranked[:1]
    selected: list[dict[str, Any]] = []
    cumulative = 0.0
    for item in ranked:
        selected.append(item)
        cumulative += max(0.0, float(item["score"])) / total
        if cumulative >= value:
            break
    return selected


def _format_time(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _contiguous_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        ranges.append((start, prev + 1))
        start = current
        prev = current
    ranges.append((start, prev + 1))
    return ranges


def _merge_ranges(ranges: list[tuple[int, int]], *, max_gap_frames: int) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap <= max_gap_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged
