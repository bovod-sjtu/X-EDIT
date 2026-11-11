"""Parser utilities for MOSS-Transcribe-Diarize compact transcripts."""

from __future__ import annotations

import re
from typing import Any


def parse_moss_transcript(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"\[(?P<start>\d+(?:\.\d+)?)\]\s*\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
        flags=re.IGNORECASE | re.DOTALL,
    )
    segments = []
    for match in pattern.finditer(str(text or "")):
        body = re.sub(r"\s+", " ", match.group("text")).strip()
        if not body:
            continue
        segments.append(
            {
                "speaker_id": _normalize_speaker_id(match.group("speaker")),
                "start_sec": round(float(match.group("start")), 6),
                "end_sec": round(float(match.group("end")), 6),
                "text": body,
                "source": "moss_compact_text",
            }
        )
    return sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"]))


def _normalize_speaker_id(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text or "0"
