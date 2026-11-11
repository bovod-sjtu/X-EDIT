"""Validate explicit choices for the categorical emotion synthesis service.

Descriptions are metadata, never a transport-layer instruction to infer a
category. The Planner owns that semantic choice using the exposed capability.
"""

from __future__ import annotations

import math
import re
from html import unescape

EMOTION_CHOICES = frozenset(
    {"afraid", "angry", "calm", "disgusted", "happy", "melancholic", "sad", "surprised"}
)


def explicit_emotion_attributes(attrs: str) -> tuple[str, str | None, str | None]:
    """Read a category and intensity, without interpreting descriptive prose."""
    values = {
        match.group(1).lower(): unescape(match.group(3))
        for match in re.finditer(
            r"\b([a-z_]+)\s*=\s*(['\"])(.*?)\2", attrs, re.IGNORECASE | re.DOTALL
        )
    }
    category = (values.get("type") or values.get("type_ref") or "").casefold()
    if category not in EMOTION_CHOICES:
        raise ValueError(
            "emotion_style_edit requires an explicit supported emotion type; "
            f"choose from {', '.join(sorted(EMOTION_CHOICES))}. "
            "Free-form descriptions do not select a category."
        )
    level, alpha = values.get("level"), values.get("alpha")
    if level is not None and level not in {"1", "2", "3"}:
        raise ValueError("emotion_style_edit level must be 1, 2, or 3")
    if alpha is not None:
        try:
            number = float(alpha)
        except ValueError as exc:
            raise ValueError("emotion_style_edit alpha must be in [0, 1]") from exc
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("emotion_style_edit alpha must be in [0, 1]")
    if level is None and alpha is None:
        raise ValueError("emotion_style_edit requires explicit level or alpha")
    if level is not None and alpha is not None:
        raise ValueError("emotion_style_edit accepts one intensity representation: level or alpha")
    return category, level, alpha


def validate_explicit_emotion_choices(instruction: str) -> None:
    """Validate semantic choices; the operation validator owns full tag grammar."""
    if re.search(r"<desc\b|<none\b", instruction, re.IGNORECASE):
        raise ValueError(
            "emotion_style_edit requires an explicit supported category and intensity; "
            "leave unchanged words untagged"
        )
    for match in re.finditer(r"<emo\b([^>]*)>", instruction, re.IGNORECASE | re.DOTALL):
        explicit_emotion_attributes(match.group(1))
