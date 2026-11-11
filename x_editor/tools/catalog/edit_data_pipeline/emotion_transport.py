"""Keep whitespace separators out of standalone emotion synthesis segments."""

from __future__ import annotations

import re

from x_editor.core.tagged_text import render_tagged_source


_EMOTION_SEGMENT = re.compile(r"(<emo\b[^>]*>.*?</emo>)", re.IGNORECASE | re.DOTALL)


def bind_emotion_separator_whitespace(instruction: str) -> str:
    parts = _EMOTION_SEGMENT.split(instruction)
    for index in range(1, len(parts), 2):
        opening_end = parts[index].index(">") + 1
        if not parts[index][opening_end:-6].strip():
            raise ValueError("Emotion edit spans must contain non-whitespace source text")
    for index in range(0, len(parts), 2):
        separator = parts[index]
        if not separator or separator.strip():
            continue
        if index > 0:
            parts[index - 1] = parts[index - 1][:-6] + separator + parts[index - 1][-6:]
        elif index + 1 < len(parts):
            opening_end = parts[index + 1].index(">") + 1
            parts[index + 1] = parts[index + 1][:opening_end] + separator + parts[index + 1][opening_end:]
        else:
            continue
        parts[index] = ""
    result = "".join(parts)
    if render_tagged_source(result, decode_entities=False) != render_tagged_source(instruction, decode_entities=False):
        raise ValueError("Emotion transport must preserve the exact rendered source")
    return result
