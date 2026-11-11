"""Qwen3-Omni audio-perception enrichment for open agentic mode."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from x_editor.agents.openai_compatible import OpenAICompatibleServiceUnavailable
from x_editor.core.agentic_mode import (
    QWEN3_OMNI_PROVIDER,
    provider_service_hint,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.config import runtime_artifact_path
from x_editor.perception.gemini_enrichment import (
    PROFILE_TERMS,
    SPEAKER_TERMS,
    merge_gemini_perception,
)

if TYPE_CHECKING:
    from x_editor.frontend.qwen3_omni_audio_frontend import Qwen3OmniAudioFrontend


def needs_qwen3_omni_audio_perception(
    user_prompt: str,
    perception: dict[str, Any] | None,
    *,
    need_words: bool = False,
    need_speakers: bool = False,
) -> bool:
    if not _enabled():
        return False
    perception = perception if isinstance(perception, dict) else {}
    text = user_prompt.lower()
    has_source_text = bool(str(perception.get("source_text") or "").strip())
    has_words = bool(perception.get("word_timestamps"))
    has_speaker_segments = bool(perception.get("speaker_segments") or perception.get("segments"))
    has_profiles = _has_speaker_profiles(perception)

    if need_words and (not has_source_text or not has_words):
        return True
    if need_speakers and not has_speaker_segments:
        return True
    if _contains(text, PROFILE_TERMS) and not has_profiles:
        return True
    if _contains(text, SPEAKER_TERMS) and need_speakers and not has_profiles:
        return True
    return False


def enrich_perception_with_qwen3_omni(
    *,
    user_prompt: str,
    audio_path: str | Path,
    perception: dict[str, Any] | None = None,
    reference_audio_paths: list[str] | None = None,
    base_audio_ref: str | None = None,
    reference_audio_refs: list[str] | None = None,
    output_dir: str | Path | None = None,
    frontend: Qwen3OmniAudioFrontend | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Merge Qwen3-Omni semantic observations into an existing perception report."""

    base = dict(perception or {})
    if not force and not needs_qwen3_omni_audio_perception(user_prompt, base, need_words=True, need_speakers=True):
        return base
    if not _enabled() and not force:
        return base
    if frontend is None and not resolve_openai_compatible_base_url(
        QWEN3_OMNI_PROVIDER
    ):
        raise RuntimeError(f"Qwen3-Omni perception fallback service unavailable: {provider_service_hint(QWEN3_OMNI_PROVIDER)}")

    audio = Path(audio_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Qwen3-Omni perception audio does not exist: {audio}")

    try:
        qwen = _load_or_call_qwen3_omni(
            user_prompt=user_prompt,
            audio_path=audio,
            reference_audio_paths=reference_audio_paths or [],
            base_audio_ref=base_audio_ref,
            reference_audio_refs=reference_audio_refs,
            output_dir=output_dir,
            frontend=frontend,
        )
    except OpenAICompatibleServiceUnavailable:
        raise
    except Exception as exc:
        merged = dict(base)
        merged.setdefault("errors", []).append({"stage": "qwen3_omni_perception", "message": str(exc)})
        return merged

    return merge_qwen3_omni_perception(base, qwen)


def merge_qwen3_omni_perception(base: dict[str, Any], qwen: dict[str, Any]) -> dict[str, Any]:
    return merge_gemini_perception(
        base,
        qwen,
        perception_key="qwen3_omni_perception",
        source_label="qwen3_omni",
        service_key="qwen3_omni_audio_perception",
        service_model=resolve_openai_compatible_model(QWEN3_OMNI_PROVIDER),
    )


def _load_or_call_qwen3_omni(
    *,
    user_prompt: str,
    audio_path: Path,
    reference_audio_paths: list[str],
    base_audio_ref: str | None,
    reference_audio_refs: list[str] | None,
    output_dir: str | Path | None,
    frontend: Qwen3OmniAudioFrontend | None,
) -> dict[str, Any]:
    cache_path = _cache_path(
        user_prompt=user_prompt,
        audio_path=audio_path,
        reference_audio_paths=reference_audio_paths,
        base_audio_ref=base_audio_ref,
        reference_audio_refs=reference_audio_refs,
        output_dir=output_dir,
    )
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if frontend is None:
        from x_editor.frontend.qwen3_omni_audio_frontend import (
            Qwen3OmniAudioFrontend,
        )

        frontend = Qwen3OmniAudioFrontend()
    report = frontend.perceive_sync(
        user_prompt=user_prompt,
        base_audio_path=str(audio_path),
        reference_audio_paths=reference_audio_paths,
        base_audio_ref=base_audio_ref,
        reference_audio_refs=reference_audio_refs,
    )
    data = report.model_dump(exclude_none=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _cache_path(
    *,
    user_prompt: str,
    audio_path: Path,
    reference_audio_paths: list[str],
    base_audio_ref: str | None,
    reference_audio_refs: list[str] | None,
    output_dir: str | Path | None,
) -> Path | None:
    if os.getenv("X_EDITOR_QWEN3_OMNI_PERCEPTION_CACHE", "1").lower() in {"0", "false", "no"}:
        return None
    root = Path(
        output_dir
        or os.getenv("X_EDITOR_PERCEPTION_CACHE_DIR")
        or runtime_artifact_path("perception_cache")
    ) / "qwen3_omni_perception_cache"
    material = {
        "prompt": user_prompt,
        "audio": _file_fingerprint(audio_path),
        "refs": [_file_fingerprint(Path(path).expanduser().resolve()) for path in reference_audio_paths],
        "logical_base_ref": base_audio_ref,
        "logical_reference_refs": reference_audio_refs,
    }
    key = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return root / f"{key}.json"


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _has_speaker_profiles(perception: dict[str, Any]) -> bool:
    for collection in [perception.get("speakers"), perception.get("speaker_segments"), perception.get("segments")]:
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            if item.get("description") or item.get("perceived_gender") or item.get("gender") or item.get("age"):
                return True
    return False


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _enabled() -> bool:
    return os.getenv("X_EDITOR_ENABLE_QWEN3_OMNI_PERCEPTION_FALLBACK", "1").lower() not in {"0", "false", "no"}
