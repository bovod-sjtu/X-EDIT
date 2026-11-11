"""Shared contract helpers for deterministic timeline span reordering."""

from __future__ import annotations

from typing import Any


def normalize_reorder_indices(value: Any, *, span_count: int | None = None) -> list[int]:
    """Return one complete index permutation or raise for ambiguous input."""

    if not isinstance(value, list):
        raise TypeError("order must be a list of integer span indices")
    expected_count = len(value) if span_count is None else span_count
    if len(value) != expected_count:
        raise ValueError(
            f"order must contain exactly {expected_count} indices, got {len(value)}"
        )
    indices = [_normalize_reorder_index(item) for item in value]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            "order must contain every span index exactly once without duplicates"
        )
    return indices


def normalize_reorder_speaker_order(value: Any) -> list[str]:
    """Return unique, non-empty speaker IDs in the explicitly requested order."""

    if not isinstance(value, list):
        raise TypeError("target_speaker_order must be a list of speaker IDs")
    speaker_ids = [str(item).strip() for item in value]
    if not speaker_ids or any(not speaker_id for speaker_id in speaker_ids):
        raise ValueError("target_speaker_order must contain non-empty speaker IDs")
    if len(set(speaker_ids)) != len(speaker_ids):
        raise ValueError("target_speaker_order must not contain duplicate speakers")
    return speaker_ids


def reorder_indices_for_speaker_order(
    value: Any,
    *,
    span_speaker_ids: list[str | None],
) -> list[int]:
    requested = normalize_reorder_speaker_order(value)
    if any(speaker_id is None or not str(speaker_id).strip() for speaker_id in span_speaker_ids):
        raise ValueError("selected spans do not all have speaker IDs")
    available = [str(speaker_id).strip() for speaker_id in span_speaker_ids]
    if len(set(available)) != len(available):
        raise ValueError("selected spans do not map one-to-one to unique speakers")
    if len(requested) != len(available) or set(requested) != set(available):
        raise ValueError(
            "target_speaker_order must contain every selected speaker exactly once"
        )
    by_speaker = {speaker_id: index for index, speaker_id in enumerate(available)}
    return [by_speaker[speaker_id] for speaker_id in requested]


def _normalize_reorder_index(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("order indices must be integers, not booleans")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    raise TypeError(f"order index is not an integer: {value!r}")
