"""Project semantic transforms into executable runtime contracts."""

from __future__ import annotations

import re
from html import escape
from typing import Any

from x_editor.semantic.schemas import SemanticTransform


class SemanticProjectionError(RuntimeError):
    """Raised when a semantic transform cannot be safely projected."""


def project_semantic_transform_for_text_edit(
    transform: SemanticTransform | dict[str, Any],
    source_text: str,
) -> dict[str, Any]:
    """Return ``source_text``, ``instruction_st``, and ``target_text`` params.

    The function handles only text-like semantic transforms. It does not invoke
    audio services and does not decide which runtime operation to use.
    """

    item = transform if isinstance(transform, SemanticTransform) else SemanticTransform(**transform)
    if item.type not in {
        "text_replacement",
        "arithmetic_transform",
        "coreference_restore",
        "symbolic_transform",
        "natural_language_transform",
    }:
        raise SemanticProjectionError(f"semantic transform {item.type!r} cannot be projected to text_edit")
    if not source_text.strip():
        raise SemanticProjectionError("text_edit semantic projection requires source_text")
    if not item.source_units or not item.target_units:
        raise SemanticProjectionError("text_edit semantic projection requires source_units and target_units")

    pairs = _replacement_pairs(item)
    instruction_st, target_text, replacements = _replacement_instruction(
        source_text,
        pairs,
        all_occurrences=item.occurrence_policy in {"all", "explicit"},
    )
    if replacements <= 0:
        raise SemanticProjectionError(
            "semantic transform source_units could not be found in source_text for text_edit projection"
        )
    return {
        "source_text": source_text,
        "instruction_st": instruction_st,
        "target_text": target_text,
        "semantic_transform": item.model_dump(),
        "semantic_transform_id": item.transform_id,
    }


def _replacement_pairs(item: SemanticTransform) -> list[tuple[list[str], str]]:
    paired = item.evidence.get("paired_replacements") if isinstance(item.evidence, dict) else None
    if isinstance(paired, list):
        pairs: list[tuple[list[str], str]] = []
        for entry in paired:
            if not isinstance(entry, dict):
                continue
            sources = entry.get("source_units")
            if isinstance(sources, str):
                source_units = [sources]
            elif isinstance(sources, list):
                source_units = [str(value) for value in sources if str(value)]
            else:
                source = entry.get("source")
                source_units = [str(source)] if source else []
            target = entry.get("target")
            if source_units and target is not None:
                pairs.append((source_units, str(target)))
        if pairs:
            return pairs
    if len(item.source_units) == len(item.target_units) and len(item.source_units) > 1:
        return [([source], target) for source, target in zip(item.source_units, item.target_units)]
    return [(list(item.source_units), item.target_units[0])]


def _replacement_instruction(
    source_text: str,
    pairs: list[tuple[list[str], str]],
    *,
    all_occurrences: bool,
) -> tuple[str, str, int]:
    spans = _find_replacement_spans(source_text, pairs, all_occurrences=all_occurrences)
    if not spans:
        return source_text, source_text, 0
    instruction_parts: list[str] = []
    target_parts: list[str] = []
    cursor = 0
    for start, end, target_unit in spans:
        instruction_parts.append(_xml_text(source_text[cursor:start]))
        if target_unit:
            instruction_parts.append(f'<sub targ="{_xml_attr(target_unit)}">{_xml_text(source_text[start:end])}</sub>')
        else:
            instruction_parts.append(f"<del>{_xml_text(source_text[start:end])}</del>")
        target_parts.append(source_text[cursor:start])
        target_parts.append(target_unit)
        cursor = end
    instruction_parts.append(_xml_text(source_text[cursor:]))
    target_parts.append(source_text[cursor:])
    return "".join(instruction_parts), "".join(target_parts), len(spans)


def _find_replacement_spans(
    source_text: str,
    pairs: list[tuple[list[str], str]],
    *,
    all_occurrences: bool,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for source_units, target_unit in pairs:
        found_for_pair = False
        for unit in sorted(source_units, key=len, reverse=True):
            for match in _iter_unit_matches(source_text, unit):
                span = (match.start(), match.end())
                if any(_overlaps(span, (existing[0], existing[1])) for existing in spans):
                    continue
                spans.append((span[0], span[1], target_unit))
                found_for_pair = True
                if not all_occurrences:
                    break
            if found_for_pair and not all_occurrences:
                break
    spans.sort(key=lambda item: (item[0], item[1]))
    return spans


def _iter_unit_matches(source_text: str, unit: str):
    if any(ord(ch) > 127 for ch in unit):
        yield from re.finditer(re.escape(unit), source_text)
        return
    compact_seat = re.fullmatch(r"(\d{1,3})([A-Za-z])", unit.strip())
    if compact_seat:
        pattern = rf"(?<![A-Za-z0-9]){compact_seat.group(1)}\s*{compact_seat.group(2)}(?![A-Za-z0-9])"
        yield from re.finditer(pattern, source_text, flags=re.IGNORECASE)
        return
    yield from re.finditer(rf"(?<![A-Za-z0-9]){re.escape(unit)}(?![A-Za-z0-9])", source_text, flags=re.IGNORECASE)


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _xml_text(value: str) -> str:
    return escape(value, quote=False)


def _xml_attr(value: str) -> str:
    return escape(value, quote=True)
