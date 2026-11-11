"""Deterministic identities for immutable capacity releases and reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data without machine-specific formatting."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, value: Any, *, length: int = 24) -> str:
    return f"{prefix}_{content_sha256(value)[:length]}"


def visible_case_fingerprint(
    *,
    request: str,
    source_text: str | None,
    visible_context: Mapping[str, Any],
    session_history: Sequence[Mapping[str, Any]],
    audio_assets: Sequence[Mapping[str, Any]],
) -> str:
    """Hash prompt-visible semantics while excluding local paths and row ids."""

    payload = {
        "request": request,
        "source_text": source_text,
        "visible_context": dict(visible_context),
        "session_history": [dict(item) for item in session_history],
        "audio_assets": [
            {
                "role": item.get("role"),
                "order": item.get("order"),
                "content_sha256": item.get("content_sha256") or "unhashed",
            }
            for item in audio_assets
        ],
    }
    return content_sha256(payload)


def exact_sidecar_record_key(value: Any) -> str:
    """Hash the exact offline source row; the result is never prompt-visible."""

    return content_sha256(value)
