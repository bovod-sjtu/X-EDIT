"""Conservative scorer-only localization from independent word alignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from x_editor.evaluation.wording_evidence import (
    IndependentWordingObservation,
    wording_tokens,
)

DEFAULT_MINIMUM_ALIGNMENT_COVERAGE = 0.9
DEFAULT_MINIMUM_TOKEN_COUNT = 3
DEFAULT_MAXIMUM_BOUNDARY_PADDING_SEC = 0.3
DEFAULT_MAXIMUM_WORD_OVERLAP_SEC = 0.02
DEFAULT_BOUNDED_MINIMUM_TOKEN_COUNT = 5
DEFAULT_BOUNDED_MAXIMUM_EDIT_FRACTION = 0.2


@dataclass(frozen=True)
class ScorerInterval:
    interval_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class ScorerRecoveredInterval:
    interval_id: str
    source_start_sec: float
    source_end_sec: float
    output_start_sec: float
    output_end_sec: float
    token_count: int
    localization_confidence: float
    method: Literal["exact_unique_sequence", "bounded_unique_sequence"]


@dataclass(frozen=True)
class IndependentIntervalAlignment:
    status: str
    reason: str
    mappings: tuple[ScorerRecoveredInterval, ...] = ()


@dataclass(frozen=True)
class _WordUnit:
    tokens: tuple[str, ...]
    start_sec: float
    end_sec: float


def recover_local_interval_alignment(
    *,
    source: IndependentWordingObservation,
    output: IndependentWordingObservation,
    intervals: Sequence[ScorerInterval],
    minimum_alignment_coverage: float = DEFAULT_MINIMUM_ALIGNMENT_COVERAGE,
    minimum_token_count: int = DEFAULT_MINIMUM_TOKEN_COUNT,
    maximum_boundary_padding_sec: float = DEFAULT_MAXIMUM_BOUNDARY_PADDING_SEC,
    maximum_word_overlap_sec: float = DEFAULT_MAXIMUM_WORD_OVERLAP_SEC,
) -> IndependentIntervalAlignment:
    """Map frozen local intervals only when lexical localization is unambiguous.

    Alignment coverage is used only to localize audio crops. It is deliberately
    not treated as evidence that either ASR transcript is semantically correct.
    """

    if not intervals:
        return _unavailable("no frozen local intervals were provided")
    if source.alignment_text_coverage is None or output.alignment_text_coverage is None:
        return _unavailable("independent full-audio alignment coverage is unavailable")
    if (
        source.alignment_text_coverage < minimum_alignment_coverage
        or output.alignment_text_coverage < minimum_alignment_coverage
    ):
        return _unavailable(
            "independent full-audio alignment coverage is below the localization threshold"
        )

    source_units = _word_units(source, maximum_word_overlap_sec=maximum_word_overlap_sec)
    output_units = _word_units(output, maximum_word_overlap_sec=maximum_word_overlap_sec)
    if source_units is None or output_units is None:
        return _unavailable("independent word timestamps are absent or non-monotonic")
    source_tokens, _ = _flatten_units(source_units)
    output_tokens, output_token_units = _flatten_units(output_units)

    requested = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    if any(
        left.end_sec > right.start_sec
        for left, right in pairwise(requested)
    ):
        return _unavailable("frozen source intervals overlap")

    recovered: list[ScorerRecoveredInterval] = []
    for interval in requested:
        selected_units = [
            unit
            for unit in source_units
            if interval.start_sec
            <= (unit.start_sec + unit.end_sec) / 2.0
            <= interval.end_sec
        ]
        if not selected_units:
            return _unavailable(
                f"{interval.interval_id}: no aligned source words fall inside the interval"
            )
        phrase = tuple(token for unit in selected_units for token in unit.tokens)
        if len(phrase) < minimum_token_count:
            return _unavailable(
                f"{interval.interval_id}: fewer than {minimum_token_count} lexical tokens "
                "cannot establish a robust unique localization"
            )
        boundary_padding = abs(selected_units[0].start_sec - interval.start_sec) + abs(
            selected_units[-1].end_sec - interval.end_sec
        )
        if boundary_padding > maximum_boundary_padding_sec:
            return _unavailable(
                f"{interval.interval_id}: aligned words do not bind both interval boundaries "
                "within the frozen padding limit"
            )

        source_matches = _exact_occurrences(source_tokens, phrase)
        output_matches = _exact_occurrences(output_tokens, phrase)
        if len(source_matches) != 1:
            return _unavailable(
                f"{interval.interval_id}: lexical sequence is not unique in source audio"
            )
        if len(output_matches) == 1:
            output_start_index = output_matches[0]
            output_end_index = output_start_index + len(phrase) - 1
            method: Literal["exact_unique_sequence", "bounded_unique_sequence"] = (
                "exact_unique_sequence"
            )
        else:
            bounded_matches = _bounded_occurrences(output_tokens, phrase)
            if len(bounded_matches) != 1:
                return _unavailable(
                    f"{interval.interval_id}: no unique exact or bounded lexical "
                    "localization exists in output audio"
                )
            output_start_index, output_end_exclusive = bounded_matches[0]
            output_end_index = output_end_exclusive - 1
            method = "bounded_unique_sequence"
        output_start_unit = output_token_units[output_start_index]
        output_end_unit = output_token_units[output_end_index]
        recovered.append(
            ScorerRecoveredInterval(
                interval_id=interval.interval_id,
                source_start_sec=interval.start_sec,
                source_end_sec=interval.end_sec,
                output_start_sec=output_start_unit.start_sec,
                output_end_sec=output_end_unit.end_sec,
                token_count=len(phrase),
                localization_confidence=min(
                    source.alignment_text_coverage,
                    output.alignment_text_coverage,
                ),
                method=method,
            )
        )

    if any(
        left.output_end_sec > right.output_start_sec
        for left, right in pairwise(recovered)
    ):
        return _unavailable("recovered output intervals overlap or change order")
    return IndependentIntervalAlignment(
        status="ready",
        reason=(
            "unique monotonic lexical sequences independently localized every frozen "
            "source interval in the delivered output"
        ),
        mappings=tuple(recovered),
    )


def _word_units(
    observation: IndependentWordingObservation,
    *,
    maximum_word_overlap_sec: float,
) -> list[_WordUnit] | None:
    units: list[_WordUnit] = []
    previous_end = 0.0
    for word in observation.words:
        tokens = tuple(wording_tokens(word.normalized_text or word.text))
        if not tokens:
            continue
        if word.start_sec < previous_end - maximum_word_overlap_sec:
            return None
        units.append(
            _WordUnit(
                tokens=tokens,
                start_sec=word.start_sec,
                end_sec=word.end_sec,
            )
        )
        previous_end = word.end_sec
    return units or None


def _flatten_units(
    units: Sequence[_WordUnit],
) -> tuple[tuple[str, ...], tuple[_WordUnit, ...]]:
    tokens: list[str] = []
    token_units: list[_WordUnit] = []
    for unit in units:
        tokens.extend(unit.tokens)
        token_units.extend([unit] * len(unit.tokens))
    return tuple(tokens), tuple(token_units)


def _exact_occurrences(haystack: Sequence[str], needle: Sequence[str]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if tuple(haystack[index : index + len(needle)]) == tuple(needle)
    ]


def _bounded_occurrences(
    haystack: Sequence[str],
    needle: Sequence[str],
    *,
    minimum_token_count: int = DEFAULT_BOUNDED_MINIMUM_TOKEN_COUNT,
    maximum_edit_fraction: float = DEFAULT_BOUNDED_MAXIMUM_EDIT_FRACTION,
) -> list[tuple[int, int]]:
    if len(needle) < minimum_token_count:
        return []
    maximum_edits = max(1, int(len(needle) * maximum_edit_fraction))
    minimum_length = max(2, len(needle) - maximum_edits)
    maximum_length = len(needle) + maximum_edits
    candidates: list[tuple[int, int]] = []
    for start, token in enumerate(haystack):
        if token != needle[0]:
            continue
        for length in range(minimum_length, maximum_length + 1):
            end = start + length
            if end > len(haystack) or haystack[end - 1] != needle[-1]:
                continue
            if _token_edit_distance(needle, haystack[start:end]) <= maximum_edits:
                candidates.append((start, end))
    return candidates


def _token_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _unavailable(reason: str) -> IndependentIntervalAlignment:
    return IndependentIntervalAlignment(status="unavailable", reason=reason)


__all__ = [
    "DEFAULT_BOUNDED_MAXIMUM_EDIT_FRACTION",
    "DEFAULT_BOUNDED_MINIMUM_TOKEN_COUNT",
    "DEFAULT_MAXIMUM_BOUNDARY_PADDING_SEC",
    "DEFAULT_MAXIMUM_WORD_OVERLAP_SEC",
    "DEFAULT_MINIMUM_ALIGNMENT_COVERAGE",
    "DEFAULT_MINIMUM_TOKEN_COUNT",
    "IndependentIntervalAlignment",
    "ScorerInterval",
    "ScorerRecoveredInterval",
    "recover_local_interval_alignment",
]
