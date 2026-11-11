"""Deterministic relative gain on a whole waveform or explicit sample intervals.

No normalization, limiting, selector inference, resampling or boundary fades are
performed. Float64 WAV output preserves decoded non-target samples exactly.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class RelativeGainBoundaryError(ValueError):
    """The requested gain cannot be applied without changing its contract."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelativeGainBoundaryError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise RelativeGainBoundaryError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise RelativeGainBoundaryError(f"{name} must be a finite number")
    return result


def validate_gain_params(params: dict[str, Any]) -> tuple[float, float]:
    """Validate the exact public contract without reading audio or inferring scope."""
    allowed = {"gain_db", "selected_spans", "headroom_db", "_agentic_mode", "output_path", "output_dir"}
    unknown = set(params) - allowed
    if unknown:
        raise RelativeGainBoundaryError(f"apply_gain does not accept parameters: {sorted(unknown)}")
    gain = _number(params.get("gain_db"), "gain_db")
    ceiling = _number(params.get("headroom_db", 0.0), "headroom_db")
    if ceiling > 0:
        raise RelativeGainBoundaryError("headroom_db is a peak ceiling in dBFS and must be <= 0")
    try:
        factor = 10.0 ** (gain / 20.0)
        peak_limit = 10.0 ** (ceiling / 20.0)
    except OverflowError as exc:
        raise RelativeGainBoundaryError("gain_db exceeds representable linear gain") from exc
    if not math.isfinite(factor) or factor <= 0 or peak_limit <= 0:
        raise RelativeGainBoundaryError("gain_db/headroom_db must have representable positive linear values")
    if "selected_spans" in params:
        spans = params["selected_spans"]
        if not isinstance(spans, list) or not spans:
            raise RelativeGainBoundaryError("selected_spans must be a non-empty list; omit it for whole audio")
        for span in spans:
            if not isinstance(span, dict):
                raise RelativeGainBoundaryError("each selected span requires start_sec and end_sec")
            start = _number(span.get("start_sec"), "start_sec")
            end = _number(span.get("end_sec"), "end_sec")
            if start < 0 or end <= start:
                raise RelativeGainBoundaryError("selected spans require 0 <= start_sec < end_sec")
    return factor, peak_limit


def gain_frame_intervals(
    params: dict[str, Any], *, frames: int, sample_rate: int,
) -> list[tuple[int, int]]:
    """Round boundaries to nearest frames; union overlapping spans, applying once."""
    validate_gain_params(params)
    if frames <= 0 or sample_rate <= 0:
        raise RelativeGainBoundaryError("apply_gain requires non-empty audio")
    if "selected_spans" not in params:
        return [(0, frames)]
    duration = frames / sample_rate
    intervals = []
    for span in params["selected_spans"]:
        start, end = float(span["start_sec"]), float(span["end_sec"])
        if end > duration:
            raise RelativeGainBoundaryError("selected span exceeds input audio duration; spans are never clipped")
        lo, hi = round(start * sample_rate), round(end * sample_rate)
        if hi <= lo:
            raise RelativeGainBoundaryError("selected span contains no frames after nearest-frame rounding")
        intervals.append((lo, hi))
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def apply_relative_gain(
    *, input_audio_path: str, output_audio_path: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Apply the exact gain or fail before producing an output artifact."""
    factor, peak_limit = validate_gain_params(params)
    source, destination = Path(input_audio_path), Path(output_audio_path)
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise RelativeGainBoundaryError("apply_gain must not overwrite its input")
    samples, sample_rate = sf.read(source, dtype="float64", always_2d=True)
    if not np.isfinite(samples).all():
        raise RelativeGainBoundaryError("input audio contains non-finite samples")
    intervals = gain_frame_intervals(params, frames=len(samples), sample_rate=sample_rate)
    selected_peak = max(float(np.max(np.abs(samples[lo:hi]))) for lo, hi in intervals)
    resulting_peak = selected_peak * factor
    if not math.isfinite(resulting_peak) or resulting_peak > peak_limit:
        raise RelativeGainBoundaryError(
            f"insufficient_headroom: requested gain_db={params['gain_db']} produces "
            f"selected peak {resulting_peak:.9g} above {peak_limit:.9g}; no limiting or gain reduction is allowed"
        )
    output = samples.copy()
    for lo, hi in intervals:
        output[lo:hi] *= factor
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, sample_rate, format="WAV", subtype="DOUBLE")
    spans = [{"start_sec": lo / sample_rate, "end_sec": hi / sample_rate} for lo, hi in intervals]
    return {
        "output_audio_path": str(destination),
        "input_audio_path": str(source),
        "selected_spans": spans if "selected_spans" in params else [],
        "source_spans": spans if "selected_spans" in params else [],
        "target_spans": spans if "selected_spans" in params else [],
        "timeline_duration_preserved": True,
        "identity_outside_regions": True,
        "relative_gain_contract": {
            "input_audio_path": str(source),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "gain_db": float(params["gain_db"]),
            "linear_gain": factor,
            "headroom_db": float(params.get("headroom_db", 0.0)),
            "frame_intervals": [list(item) for item in intervals],
            "frame_count": len(samples),
            "sample_rate": sample_rate,
            "channels": samples.shape[1],
            "output_subtype": "DOUBLE",
            "selected_input_peak": selected_peak,
            "selected_output_peak": resulting_peak,
            "boundary_rounding": "nearest_frame_ties_to_even",
            "overlap_policy": "union_apply_once",
        },
    }
