"""Resolve pure natural-language/symbolic edits into semantic transforms."""

from __future__ import annotations

import re
from typing import Any, Iterable

from x_editor.semantic.schemas import SemanticTransform


class SemanticTransformResolver:
    """Small deterministic resolver for auditable semantic transforms.

    This layer covers objective, reusable text/symbolic cases that do not need
    audio perception: explicit text replacements, edit-history restores,
    sequence arithmetic, and alias evidence supplied by metadata/perception.
    Broader language reasoning can be added behind the same output contract.
    """

    def resolve(
        self,
        user_request: str,
        *,
        source_text: str | None = None,
        session_history: list[dict[str, Any]] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> list[SemanticTransform]:
        request = str(user_request or "").strip()
        if not request:
            return [_unsupported("empty_request", "User request is empty.")]

        history = session_history or []
        evidence = evidence or {}
        resolvers = (
            self._resolve_sequence_arithmetic,
            self._resolve_history_restore,
            self._resolve_deletion,
            self._resolve_quantity_replacement,
            self._resolve_explicit_replacement,
        )
        for resolver in resolvers:
            transform = resolver(request, source_text=source_text, session_history=history, evidence=evidence)
            if transform is not None:
                return transform if isinstance(transform, list) else [transform]
        return [
            _unsupported(
                "unsupported_semantic_transform",
                "No supported pure semantic transform could be resolved from the request and evidence.",
            )
        ]

    def _resolve_explicit_replacement(
        self,
        request: str,
        *,
        source_text: str | None,
        session_history: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> SemanticTransform | list[SemanticTransform] | None:
        pairs = _replacement_pairs(request)
        if not pairs:
            return None
        paired_payload = []
        source_units: list[str] = []
        target_units: list[str] = []
        for old_text, new_text in pairs:
            old_units = _alias_units(old_text, evidence, source_text=source_text) or [old_text]
            new_units = _target_alias_units(new_text, evidence) or [new_text]
            target_unit = new_units[0]
            paired_payload.append(
                {
                    "source": old_text,
                    "target": target_unit,
                    "source_units": old_units,
                    "target_units": new_units,
                }
            )
            source_units.extend(old_units)
            target_units.append(target_unit)
        occurrence_policy = "all" if _requests_all_occurrences(request) else "first"
        return SemanticTransform(
            transform_id="semantic_text_replacement_1",
            type="text_replacement",
            source_units=source_units,
            target_units=target_units,
            occurrence_policy=occurrence_policy,
            confidence=0.86 if any(item["source_units"] != [item["source"]] for item in paired_payload) else 0.78,
            rationale="Resolved an explicit source-to-target text replacement from the user request.",
            required_evidence=["source_text"],
            evidence={
                "paired_replacements": paired_payload,
                "alias_evidence_used": any(item["source_units"] != [item["source"]] for item in paired_payload),
            },
            projection_hints={
                "recommended_operation": "edit_data_pipeline.text_edit",
                "contract": "text_replacement",
            },
        )

    def _resolve_deletion(
        self,
        request: str,
        *,
        source_text: str | None,
        session_history: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> SemanticTransform | None:
        match = re.search(
            r"\b(?:remove|delete)\s+(?:all\s+occurrences\s+of\s+|every\s+instance\s+of\s+|the\s+word\s+|"
            r"the\s+phrase\s+)?(?P<target>\"[^\"]+\"|'[^']+'|“[^”]+”|[^.,]+)",
            request,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        target_text = _strip_phrase(match.group("target"))
        if not target_text:
            return None
        source_units = _alias_units(target_text, evidence, source_text=source_text) or [target_text]
        return SemanticTransform(
            transform_id="semantic_text_deletion_1",
            type="text_replacement",
            source_units=source_units,
            target_units=[""],
            occurrence_policy="all" if _requests_all_occurrences(request) else "first",
            confidence=0.82,
            rationale="Resolved a deletion request as a text replacement with an empty target.",
            required_evidence=["source_text"],
            evidence={
                "requested_source_text": target_text,
                "paired_replacements": [
                    {
                        "source": target_text,
                        "target": "",
                        "source_units": source_units,
                        "target_units": [""],
                    }
                ],
            },
            projection_hints={"recommended_operation": "edit_data_pipeline.text_edit", "contract": "text_deletion"},
        )

    def _resolve_quantity_replacement(
        self,
        request: str,
        *,
        source_text: str | None,
        session_history: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> SemanticTransform | None:
        match = re.search(
            r"\b(?:change|replace)\s+the\s+number\s+of\s+(?P<noun>.+?)\s+to\s+(?P<number>[A-Za-z0-9-]+)\b",
            request,
            flags=re.IGNORECASE,
        )
        if not match or not source_text:
            return None
        noun = _strip_phrase(match.group("noun"))
        target_number = _strip_phrase(match.group("number"))
        if not noun or not target_number:
            return None
        source_phrase = _source_quantity_phrase(source_text, noun)
        if not source_phrase:
            return None
        target_phrase = f"{target_number} {noun}"
        return SemanticTransform(
            transform_id="semantic_quantity_replacement_1",
            type="natural_language_transform",
            source_units=[source_phrase],
            target_units=[target_phrase],
            occurrence_policy="first",
            confidence=0.84,
            rationale="Resolved a quantity phrase by binding the requested noun phrase to its current spoken count.",
            required_evidence=["source_text"],
            evidence={
                "quantity_noun": noun,
                "requested_target_number": target_number,
                "paired_replacements": [
                    {
                        "source": source_phrase,
                        "target": target_phrase,
                        "source_units": [source_phrase],
                        "target_units": [target_phrase],
                    }
                ],
            },
            projection_hints={"recommended_operation": "edit_data_pipeline.text_edit", "contract": "quantity_phrase"},
        )

    def _resolve_history_restore(
        self,
        request: str,
        *,
        source_text: str | None,
        session_history: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> SemanticTransform | None:
        lowered = request.lower()
        wants_original = any(
            phrase in lowered
            for phrase in (
                "back to the original",
                "back to original",
                "recover the number",
                "recover it",
                "restore",
                "change it back",
                "change back",
                "just been removed",
                "previously removed",
            )
        )
        if not wants_original:
            return None
        prior = _latest_history_transform(session_history)
        if not prior:
            return None
        source_units = _string_list(prior.get("target_units"))
        target_units = _string_list(prior.get("source_units"))
        if "removed" in lowered:
            source_units = _candidate_names_for_restore(source_text, exclude=target_units) or source_units
        if not source_units or not target_units:
            return None
        return SemanticTransform(
            transform_id="semantic_coreference_restore_1",
            type="coreference_restore",
            source_units=source_units,
            target_units=target_units,
            occurrence_policy=(
                str(prior.get("occurrence_policy") or "all")
                if prior.get("occurrence_policy") in {"first", "all", "explicit"}
                else "all"
            ),
            confidence=0.82,
            rationale="Resolved a restore/coreference request using the previous round's semantic transform.",
            required_evidence=["session_history"],
            evidence={"history_transform_id": prior.get("transform_id"), "request_kind": "restore_previous_edit"},
            projection_hints={"recommended_operation": "edit_data_pipeline.text_edit"},
        )

    def _resolve_sequence_arithmetic(
        self,
        request: str,
        *,
        source_text: str | None,
        session_history: list[dict[str, Any]],
        evidence: dict[str, Any],
    ) -> SemanticTransform | None:
        lowered = request.lower()
        if "seat" not in lowered and "row" not in lowered:
            return None
        delta_match = re.search(r"(?:by|增加|加)\s*(\d+)\s*rows?", lowered)
        if not delta_match:
            return None
        delta = int(delta_match.group(1))
        if any(word in lowered for word in ("backward", "increase", "behind", "后")):
            signed_delta = delta
        elif any(word in lowered for word in ("forward", "decrease", "front", "前")):
            signed_delta = -delta
        else:
            signed_delta = delta

        base = _latest_seat_code(session_history) or _first_seat_code(source_text)
        if not base:
            return _unsupported(
                "missing_seat_code",
                "Seat sequence arithmetic requires a prior modified seat code or a seat code in source_text.",
            )
        number, suffix = base
        target = f"{number + signed_delta}{suffix}"
        compact_source = f"{number}{suffix}"
        spaced_source = f"{number} {suffix}"
        return SemanticTransform(
            transform_id="semantic_arithmetic_transform_1",
            type="arithmetic_transform",
            source_units=[compact_source, spaced_source],
            target_units=[target],
            occurrence_policy="explicit",
            confidence=0.9,
            rationale=(
                "Resolved seat row arithmetic by applying the requested row delta "
                "to the previously modified seat code."
            ),
            required_evidence=["session_history_or_source_text"],
            evidence={"base_sequence": compact_source, "delta": signed_delta, "dimension": "seat_row"},
            projection_hints={"recommended_operation": "edit_data_pipeline.text_edit", "sequence_kind": "seat_row"},
        )


def resolve_semantic_transforms(
    user_request: str,
    *,
    source_text: str | None = None,
    session_history: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> list[SemanticTransform]:
    return SemanticTransformResolver().resolve(
        user_request,
        source_text=source_text,
        session_history=session_history,
        evidence=evidence,
    )


def _replacement_pairs(request: str) -> list[tuple[str, str]]:
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', request)
    quoted_values = [next(part for part in match if part) for match in quoted]
    lowered = request.lower()
    if len(quoted_values) >= 2 and any(word in lowered for word in ("replace", "change", "substitute")):
        return [
            (_strip_phrase(quoted_values[index]), _strip_phrase(quoted_values[index + 1]))
            for index in range(0, len(quoted_values) - 1, 2)
        ]
    patterns = [
        (
            r"replace\s+(?:every\s+instance\s+of\s+|all\s+instances\s+of\s+|the\s+word\s+|"
            r"the\s+phrase\s+|the\s+product\s+name\s+)?(?P<old>.+?)\s+with\s+(?P<new>.+?)(?:[.,]|$)"
        ),
        r"change\s+(?P<old>.+?)\s+to\s+(?P<new>.+?)(?:[.,]|$)",
    ]
    for pattern in patterns:
        pairs = []
        for match in re.finditer(pattern, request, flags=re.IGNORECASE):
            old_text = _strip_phrase(match.group("old"))
            new_text = _strip_phrase(match.group("new"))
            if old_text and new_text:
                pairs.append((old_text, new_text))
        if pairs:
            return pairs
    return []


def _requests_all_occurrences(request: str) -> bool:
    lowered = re.sub(r"\s+", " ", request.lower())
    return any(
        phrase in lowered
        for phrase in (
            "every instance",
            "all instance",
            "all the instance",
            "all occurrences",
            "all the occurrences",
            "each instance",
            "所有",
            "全部",
            "每个",
        )
    )


def _alias_units(
    text: str,
    evidence: dict[str, Any],
    *,
    source_text: str | None,
    prefer_alias: bool = True,
) -> list[str]:
    aliases = evidence.get("entity_aliases") if isinstance(evidence, dict) else None
    values: list[str] = []
    if isinstance(aliases, dict):
        raw = aliases.get(text) or aliases.get(text.lower())
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
    values.append(text)
    values.extend(_orthographic_variants(text))
    values.extend(_seat_code_source_variants(text))
    if prefer_alias and source_text:
        in_source = [item for item in values if _contains_unit(source_text, item)]
        if in_source:
            return _dedupe(in_source)
    return _dedupe(values)


def _target_alias_units(text: str, evidence: dict[str, Any]) -> list[str]:
    aliases = evidence.get("entity_aliases") if isinstance(evidence, dict) else None
    if isinstance(aliases, dict):
        raw = aliases.get(text) or aliases.get(text.lower())
        if isinstance(raw, str):
            return _dedupe([raw])
        if isinstance(raw, list):
            values = [str(item) for item in raw if item]
            if values:
                return _dedupe(values)
    return _dedupe([text, *_orthographic_variants(text)])


def _orthographic_variants(text: str) -> list[str]:
    lowered = text.lower()
    if lowered == "aluminum":
        return ["aluminium"]
    if lowered == "aluminium":
        return ["aluminum"]
    return []


def _seat_code_source_variants(text: str) -> list[str]:
    seat = _parse_seat_code(text)
    if not seat:
        return []
    number, suffix = seat
    number_words = _number_words(number)
    variants = [f"{number} {suffix}", f"{number}{suffix}"]
    if number_words:
        variants.extend([f"{number_words} {suffix}", f"{number_words}{suffix}"])
    return variants


def _number_words(value: int) -> str | None:
    ones = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
    }
    tens = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
    if value in ones:
        return ones[value]
    if value in tens:
        return tens[value]
    if 20 < value < 100:
        base = value // 10 * 10
        rest = value % 10
        return f"{tens[base]}-{ones[rest]}"
    return None


def _source_quantity_phrase(source_text: str, noun: str) -> str | None:
    noun_pattern = re.escape(noun)
    pattern = rf"\b(?P<number>[A-Za-z0-9-]+)\s+{noun_pattern}\b"
    match = re.search(pattern, source_text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _latest_history_transform(session_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(session_history):
        if not isinstance(item, dict):
            continue
        if item.get("type") in {
            "text_replacement",
            "arithmetic_transform",
            "coreference_restore",
            "natural_language_transform",
            "symbolic_transform",
        }:
            return item
    return None


def _latest_seat_code(session_history: list[dict[str, Any]]) -> tuple[int, str] | None:
    for item in reversed(session_history):
        if not isinstance(item, dict):
            continue
        candidates = _string_list(item.get("target_units")) + _string_list(item.get("source_units"))
        for candidate in candidates:
            seat = _parse_seat_code(candidate)
            if seat:
                return seat
    return None


def _first_seat_code(source_text: str | None) -> tuple[int, str] | None:
    if not source_text:
        return None
    for match in re.finditer(r"\b(\d{1,3})\s*([A-Z])\b", source_text):
        return int(match.group(1)), match.group(2)
    return None


def _parse_seat_code(value: str) -> tuple[int, str] | None:
    match = re.search(r"\b(\d{1,3})\s*([A-Z])\b", value)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _candidate_names_for_restore(source_text: str | None, *, exclude: Iterable[str]) -> list[str]:
    if not source_text:
        return []
    excluded = {item.lower() for item in exclude}
    names = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", source_text):
        value = match.group(0)
        if value.lower() not in excluded and value.lower() not in {"do", "well", "who", "um"}:
            names.append(value)
    return _dedupe(names)


def _contains_unit(source_text: str, unit: str) -> bool:
    if not unit:
        return False
    if any(ord(ch) > 127 for ch in unit):
        return unit in source_text
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(unit)}(?![A-Za-z0-9])", source_text, flags=re.IGNORECASE) is not None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _strip_phrase(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"^(?:the\s+word|the\s+phrase|the\s+product\s+name|every\s+instance\s+of|all\s+instances\s+of)\s+",
        "",
        text,
        flags=re.I,
    )
    return text.strip(" \"'“”.,")


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _unsupported(transform_id: str, reason: str) -> SemanticTransform:
    return SemanticTransform(
        transform_id=transform_id,
        type="unsupported",
        occurrence_policy="unsupported",
        confidence=0.0,
        rationale=reason,
        unsupported_reason=reason,
    )
