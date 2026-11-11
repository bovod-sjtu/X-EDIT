"""Gemini 2.5 Pro audio perception frontend."""

from __future__ import annotations

import os
import re
from pathlib import Path

from x_editor.agents.base import GeminiClient
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key
from x_editor.core.schemas import PerceptionReport, SpeakerSegment


_CANONICAL_AUDIO_REF = re.compile(r"aud:v1:[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class GeminiAudioFrontend:
    """Audio frontend that grounds the workflow in real audio perception."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 40960,
        timeout: float = 90.0,
        client: GeminiClient | None = None,
    ) -> None:
        config = AudioEditConfig()
        timeout = float(os.getenv("X_EDITOR_GEMINI_AUDIO_TIMEOUT", str(timeout)))
        self.client = client or GeminiClient(
            api_key=resolve_gemini_api_key(api_key) or config.omni_api_key,
            base_url=base_url or config.omni_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        prompt_dir = Path(__file__).parent.parent / "prompts"
        self.system_prompt = (prompt_dir / "tool_perception_system.md").read_text(encoding="utf-8")
        self.user_prompt = (prompt_dir / "tool_perception_user.md").read_text(encoding="utf-8")

    async def perceive(
        self,
        user_prompt: str,
        base_audio_path: str,
        reference_audio_paths: list[str] | None = None,
        *,
        base_audio_ref: str | None = None,
        reference_audio_refs: list[str] | None = None,
    ) -> PerceptionReport:
        return self.perceive_sync(
            user_prompt,
            base_audio_path,
            reference_audio_paths,
            base_audio_ref=base_audio_ref,
            reference_audio_refs=reference_audio_refs,
        )

    def perceive_sync(
        self,
        user_prompt: str,
        base_audio_path: str,
        reference_audio_paths: list[str] | None = None,
        *,
        base_audio_ref: str | None = None,
        reference_audio_refs: list[str] | None = None,
    ) -> PerceptionReport:
        refs = reference_audio_paths or []
        logical_base_ref, logical_reference_refs = _logical_audio_refs(
            base_audio_ref=base_audio_ref,
            reference_audio_refs=reference_audio_refs,
            reference_count=len(refs),
        )
        physical_paths = [base_audio_path, *refs]
        logical_refs = [logical_base_ref, *logical_reference_refs]
        user_text = self.user_prompt.format(
            user_prompt=_redact_attached_audio_locations(
                user_prompt,
                physical_paths=physical_paths,
                logical_refs=logical_refs,
            ),
            audio_inventory=_audio_inventory(
                base_audio_ref=logical_base_ref,
                reference_audio_refs=logical_reference_refs,
            ),
        )
        result = self.client.call_json(
            system_prompt=self.system_prompt,
            user_text=user_text,
            audio_paths=physical_paths,
        )
        segments = []
        for idx, item in enumerate(result.get("segments", []) or []):
            if isinstance(item, dict):
                segments.append(_speaker_segment_from_raw(item, idx))
        return PerceptionReport(
            source_text=result.get("source_text"),
            speakers=result.get("speakers", []) or [],
            segments=segments,
            audio_quality=result.get("audio_quality", {}) or {},
            background=result.get("background", {}) or {},
            notable_events=_string_list(result.get("notable_events", []) or []),
            selected_spans=_selected_spans(result.get("selected_spans", []) or []),
            uncertainty=_string_list(result.get("uncertainty", []) or []),
        )


def _speaker_segment_from_raw(item: dict[str, object], index: int) -> SpeakerSegment:
    """Normalize Gemini output to the strict internal speaker-segment schema."""

    return SpeakerSegment(
        segment_id=str(item.get("segment_id") or f"seg_{index}"),
        speaker_id=str(item.get("speaker_id") or item.get("speaker") or "unknown"),
        start_sec=_optional_float(item.get("start_sec", item.get("start"))),
        end_sec=_optional_float(item.get("end_sec", item.get("end"))),
        transcript=_optional_str(item.get("transcript") or item.get("text")),
        perceived_gender=_optional_str(item.get("perceived_gender") or item.get("gender")),
        description=_optional_str(item.get("description")),
        confidence=str(item.get("confidence") or "medium"),
    )


def _logical_audio_refs(
    *,
    base_audio_ref: str | None,
    reference_audio_refs: list[str] | None,
    reference_count: int,
) -> tuple[str, list[str]]:
    if reference_audio_refs is not None and len(reference_audio_refs) != reference_count:
        raise ValueError(
            "reference_audio_refs must have the same length as reference_audio_paths"
        )
    base_ref = _canonical_audio_ref(base_audio_ref or "aud:v1:s0000")
    reference_refs = [
        _canonical_audio_ref(value)
        for value in (
            reference_audio_refs
            if reference_audio_refs is not None
            else [f"aud:v1:r{index:04d}" for index in range(reference_count)]
        )
    ]
    all_refs = [base_ref, *reference_refs]
    if len(set(all_refs)) != len(all_refs):
        raise ValueError("logical audio refs must be unique")
    return base_ref, reference_refs


def _canonical_audio_ref(value: str) -> str:
    logical_ref = str(value).strip()
    if not _CANONICAL_AUDIO_REF.fullmatch(logical_ref):
        raise ValueError(
            "logical audio refs must use canonical aud:v1:<opaque-id> syntax"
        )
    return logical_ref


def _audio_inventory(
    *,
    base_audio_ref: str,
    reference_audio_refs: list[str],
) -> str:
    rows = [
        f"- audio_index=0; audio_ref_id={base_audio_ref}; role=source; role_order=0"
    ]
    rows.extend(
        f"- audio_index={index + 1}; audio_ref_id={audio_ref}; "
        f"role=reference; role_order={index}"
        for index, audio_ref in enumerate(reference_audio_refs)
    )
    return "\n".join(rows)


def _redact_attached_audio_locations(
    user_prompt: str,
    *,
    physical_paths: list[str],
    logical_refs: list[str],
) -> str:
    text = str(user_prompt)
    replacements: dict[str, str] = {}
    private_directories: set[str] = set()
    for raw_path, logical_ref in zip(physical_paths, logical_refs):
        path = Path(raw_path).expanduser()
        for value in {str(raw_path), str(path), str(path.resolve(strict=False))}:
            if value:
                replacements[value] = logical_ref
        if path.name:
            replacements[path.name] = logical_ref
        for parent in {str(path.parent), str(path.resolve(strict=False).parent)}:
            if parent not in {"", ".", "/"}:
                private_directories.add(parent)

    for value, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = text.replace(value, replacement)
    for value in sorted(private_directories, key=len, reverse=True):
        text = text.replace(value, "[private-audio-directory]")
    return text


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(items: object) -> list[str]:
    if not items:
        return []
    if not isinstance(items, list):
        items = [items]
    strings: list[str] = []
    for item in items:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            desc = item.get("description") or item.get("detail") or item.get("text") or item.get("event_id") or item.get("scope")
            strings.append(str(desc) if desc is not None else str(item))
        else:
            strings.append(str(item))
    return strings


def _selected_spans(items: object) -> list[dict[str, object]]:
    if not isinstance(items, list):
        return []
    spans: list[dict[str, object]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        start = _optional_float(item.get("start_sec", item.get("start")))
        end = _optional_float(item.get("end_sec", item.get("end")))
        if start is None or end is None or end <= start:
            continue
        span = {
            "span_id": str(item.get("span_id") or f"gemini_span_{index:03d}"),
            "start_sec": start,
            "end_sec": end,
            "speaker_id": _optional_str(item.get("speaker_id")),
            "text": _optional_str(item.get("text") or item.get("transcript")),
            "label": _optional_str(item.get("label")),
            "semantic_role": _optional_str(item.get("semantic_role"))
            or "edit_target",
            "confidence": str(item.get("confidence") or "medium"),
            "reason": _optional_str(item.get("reason")),
            "source_refs": [str(value) for value in item.get("source_refs", [])]
            if isinstance(item.get("source_refs"), list)
            else [],
        }
        audio_ref_id = _optional_str(item.get("audio_ref_id") or item.get("audio_ref"))
        if audio_ref_id:
            span["audio_ref_id"] = audio_ref_id
        audio_index = _optional_int(item.get("audio_index"))
        if audio_index is not None:
            span["audio_index"] = audio_index
        spans.append(span)
    return spans


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
