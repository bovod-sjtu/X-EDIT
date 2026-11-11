"""Bound time-axis receipts for source-index-preserving signal processing.

Only a producer with an explicit sample-index contract may call the declaration
function. Equal duration is a validation check, never the basis for inferring
alignment. The receipt says nothing about preserved words, speakers or quality.
This module also works in the standalone separator service environment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import soundfile as sf

SCHEMA = "x-editor.source-aligned-time-axis/v1"


def _identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("time-axis identity requires nonempty decoded audio")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "frame_count": info.frames, "sample_rate_hz": info.samplerate,
    }


def _same_duration(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    tolerance = max(1 / left["sample_rate_hz"], 1 / right["sample_rate_hz"]) + 1e-9
    return abs(left["frame_count"] / left["sample_rate_hz"]
               - right["frame_count"] / right["sample_rate_hz"]) <= tolerance


def declare_source_aligned_audio(
    input_path: str | Path, output_paths: Sequence[str | Path], *, basis: str,
) -> dict[str, Any] | None:
    """Bind a producer's declared t_out=t_in mapping to actual audio bytes."""
    if not basis.strip() or not output_paths:
        return None
    try:
        source = _identity(input_path)
        outputs = [_identity(path) for path in dict.fromkeys(output_paths)]
    except (OSError, ValueError, RuntimeError):
        return None
    if not all(_same_duration(source, output) for output in outputs):
        return None
    return {
        "schema_version": SCHEMA, "input": source, "outputs": outputs,
        "mapping": {"scale": 1.0, "offset_sec": 0.0},
        "basis": basis, "semantic_evidence_preserved": False,
    }


def source_aligned_audio_covers(
    receipt: Any, input_path: str | Path, output_paths: Sequence[str | Path],
) -> bool:
    """Verify the receipt's identity/version and geometry, without inferring it."""
    if not isinstance(receipt, Mapping) or not output_paths:
        return False
    if (receipt.get("schema_version") != SCHEMA
            or receipt.get("mapping") != {"scale": 1.0, "offset_sec": 0.0}
            or not receipt.get("basis")
            or receipt.get("semantic_evidence_preserved") is not False):
        return False
    try:
        source = _identity(input_path)
        if receipt.get("input") != source:
            return False
        for path in output_paths:
            observed = _identity(path)
            if observed not in receipt.get("outputs", []) or not _same_duration(source, observed):
                return False
    except (OSError, ValueError, RuntimeError, TypeError, KeyError):
        return False
    return True
