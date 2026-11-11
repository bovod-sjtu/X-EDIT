"""Explicit per-run selection, with a legacy environment fallback for direct callers."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

TAGGED_OPERATIONS = frozenset({"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"})
BACKENDS = ("dots_tts_edit", "edit_data_pipeline")
DEFAULT_DOTS_URL = "http://127.0.0.1:8049"


def selected_edit_backend() -> str:
    from x_editor.tools.selection import current_tool_selection

    selection = current_tool_selection()
    if selection is not None:
        return selection.edit_backend
    value = os.getenv("X_EDITOR_EDIT_BACKEND", "dots_tts_edit")
    if value not in BACKENDS:
        raise ValueError(f"X_EDITOR_EDIT_BACKEND must be one of {BACKENDS}, got {value!r}")
    return value


def uses_dots_model(operation: str) -> bool:
    return operation in TAGGED_OPERATIONS and selected_edit_backend() == "dots_tts_edit"


def dots_service_url(override: str | None = None) -> str:
    value = override or os.getenv("DOTS_TTS_EDIT_URL", DEFAULT_DOTS_URL)
    parsed = urlsplit(value)
    parsed.port
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("DOTS_TTS_EDIT_URL must be an HTTP(S) endpoint without credentials, query or fragment")
    return value.rstrip("/")
