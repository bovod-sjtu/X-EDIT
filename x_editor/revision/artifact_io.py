"""Artifact identity and append-only recording, shared by runtime reviewers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import soundfile as sf
from x_editor.evaluation.guidance_contracts import EvaluationArtifactRef

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)

def audio_ref(path: Path, identity: str, *, role: str = "input", revision: int = 0) -> EvaluationArtifactRef:
    info = sf.info(path)
    return EvaluationArtifactRef(artifact_ref_id=identity, role=role, path=str(path.resolve()),
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        duration_sec=info.frames / info.samplerate, timeline_revision=revision)
