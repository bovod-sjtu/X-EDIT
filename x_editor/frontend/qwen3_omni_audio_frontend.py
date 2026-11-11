"""StrongOmni audio perception frontends via OpenAI-compatible chat."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.core.agentic_mode import (
    QWEN3_OMNI_PROVIDER,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.schemas import PerceptionReport, SpeakerSegment
from x_editor.perception.invocation_trace import (
    capture_frontend_provider_call,
    finalize_frontend_invocation_trace,
    reset_frontend_invocation_trace,
)


_CANONICAL_AUDIO_REF = re.compile(r"aud:v1:[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class StrongOmniPreparedRequest:
    """Exact provider-neutral text and audio ordering sent to a StrongOmni model."""

    system_prompt: str
    user_text: str
    audio_paths: tuple[str, ...]
    logical_audio_refs: tuple[str, ...]


class OpenAICompatibleStrongOmniAudioFrontend:
    """Provider-neutral frontend for semantic audio perception review."""

    provider_name = QWEN3_OMNI_PROVIDER
    component_name = "perception.audio_fallback"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int | None = None,
        timeout: float = 90.0,
        client: OpenAICompatibleChatClient | None = None,
        provider_name: str | None = None,
        provider_display_name: str | None = None,
    ) -> None:
        self.provider_name = provider_name or self.provider_name
        display_name = provider_display_name or (
            "Qwen3-Omni"
            if self.provider_name == QWEN3_OMNI_PROVIDER
            else self.provider_name
        )
        self.model_name = resolve_openai_compatible_model(self.provider_name, model_name)
        self.last_invocation_trace = None
        self.client = client or OpenAICompatibleChatClient(
            api_key=resolve_openai_compatible_api_key(self.provider_name, api_key),
            base_url=resolve_openai_compatible_base_url(self.provider_name, base_url),
            model_name=self.model_name,
            provider_name=self.provider_name,
            temperature=temperature,
            max_tokens=_resolve_strong_omni_max_tokens(max_tokens),
            timeout=timeout,
        )
        prompt_dir = Path(__file__).parent.parent / "prompts"
        self.system_prompt = (
            (prompt_dir / "tool_perception_system.md").read_text(encoding="utf-8")
            + f"\n\nYou are {display_name} used as an open semantic audio review fallback. "
            "Keep specialist ASR/alignment as separate observations. For a direct "
            "listening question, report what the attached audio actually contains, "
            "including speech or sounds omitted by ASR. Candidate words in the "
            "question are hypotheses, not a transcript to copy. Report your audible "
            "wording and approximate intervals with uncertainty; do not claim "
            "specialist word-level timing accuracy. Preserve simultaneous voices "
            "instead of forcing their intervals to be disjoint."
        )
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
        reset_frontend_invocation_trace(self)
        prepared = self.prepare_request(
            user_prompt,
            base_audio_path,
            reference_audio_paths,
            base_audio_ref=base_audio_ref,
            reference_audio_refs=reference_audio_refs,
        )
        try:
            result = self.client.call_json(
                prepared.system_prompt,
                prepared.user_text,
                audio_paths=list(prepared.audio_paths),
                response_format={"type": "json_object"},
                repair_schema_name="PerceptionReport",
            )
        except Exception as exc:
            capture_frontend_provider_call(
                self,
                "audio_fallback.semantic_review",
                succeeded=False,
                error=exc,
            )
            finalize_frontend_invocation_trace(self, status="failed")
            raise
        capture_frontend_provider_call(
            self,
            "audio_fallback.semantic_review",
            succeeded=True,
        )
        finalize_frontend_invocation_trace(self, status="completed")
        segments = []
        for idx, item in enumerate(result.get("segments", []) or []):
            if isinstance(item, dict):
                segments.append(_speaker_segment_from_raw(item, idx))
        return PerceptionReport(
            source_text=result.get("source_text"),
            source_text_audio_path=(
                prepared.logical_audio_refs[0]
                if result.get("source_text") is not None
                else None
            ),
            source_text_audio_index=(
                0 if result.get("source_text") is not None else None
            ),
            speakers=result.get("speakers", []) or [],
            segments=segments,
            audio_quality=result.get("audio_quality", {}) or {},
            background=result.get("background", {}) or {},
            notable_events=_string_list(result.get("notable_events", []) or []),
            selected_spans=_selected_spans(result.get("selected_spans", []) or []),
            uncertainty=_string_list(result.get("uncertainty", []) or []),
        )

    def prepare_request(
        self,
        user_prompt: str,
        base_audio_path: str,
        reference_audio_paths: list[str] | None = None,
        *,
        base_audio_ref: str | None = None,
        reference_audio_refs: list[str] | None = None,
    ) -> StrongOmniPreparedRequest:
        """Build the exact request without invoking a provider.

        Comparison and audit code uses this method to prove that different
        provider transports receive identical text, audio order, and logical
        references. Physical paths remain runtime-only bindings.
        """

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
        return StrongOmniPreparedRequest(
            system_prompt=self.system_prompt,
            user_text=user_text,
            audio_paths=tuple(physical_paths),
            logical_audio_refs=tuple(logical_refs),
        )


class Qwen3OmniAudioFrontend(OpenAICompatibleStrongOmniAudioFrontend):
    """Backward-compatible Qwen3-Omni frontend name."""


def _speaker_segment_from_raw(item: dict[str, object], index: int) -> SpeakerSegment:
    return SpeakerSegment(
        segment_id=str(item.get("segment_id") or f"omni_seg_{index}"),
        speaker_id=str(item.get("speaker_id") or item.get("speaker") or "unknown"),
        audio_ref_id=_optional_str(item.get("audio_ref_id") or item.get("audio_ref")),
        audio_index=_optional_int(item.get("audio_index")),
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
            "span_id": str(item.get("span_id") or f"omni_span_{index:03d}"),
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


def _resolve_strong_omni_max_tokens(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    for env_name in (
        "X_EDITOR_STRONG_OMNI_MAX_TOKENS",
        "X_EDITOR_QWEN3_OMNI_MAX_TOKENS",
    ):
        raw = os.getenv(env_name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                continue
    return 2048
