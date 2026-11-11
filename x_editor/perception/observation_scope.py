"""Measured coverage of clips sent to an observer, in source coordinates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def observation_scope(intervals: Iterable[Mapping[str, Any]], duration_sec: float) -> dict[str, Any]:
    """Compute coverage, never infer silence or content in the omitted regions."""
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("observation duration must be finite and positive")
    valid = []
    for item in intervals:
        try:
            start, end = float(item["start_sec"]), float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (start, end))):
            continue
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            valid.append((start, end))
    merged: list[dict[str, float]] = []
    for start, end in sorted(valid):
        if merged and start <= merged[-1]["end_sec"] + 1e-6:
            merged[-1]["end_sec"] = max(end, merged[-1]["end_sec"])
        else:
            merged.append({"start_sec": start, "end_sec": end})
    gaps = []
    cursor = 0.0
    for item in merged:
        if item["start_sec"] > cursor + 1e-6:
            gaps.append({"start_sec": cursor, "end_sec": item["start_sec"]})
        cursor = item["end_sec"]
    if cursor < duration - 1e-6:
        gaps.append({"start_sec": cursor, "end_sec": duration})
    return {"duration_sec": duration, "observed_intervals": merged,
            "unobserved_intervals": gaps, "covers_entire_asset": not gaps}
