"""Fallback-first audio perception orchestration.

The primary perception path remains specialized services such as ASR,
alignment, and diarization.  A mode-selected audio model is treated as the
framework-level fallback provider when those services are unavailable or do not
expose the speaker/role/phrase evidence needed by an edit request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from x_editor.core.agentic_mode import OPEN_GLM_QWEN_OMNI_MODE, resolve_agentic_mode
from x_editor.core.config import runtime_artifact_path
from x_editor.perception.gemini_enrichment import (
    enrich_perception_with_gemini,
    needs_gemini_audio_perception,
)
from x_editor.perception.qwen3_omni_enrichment import (
    enrich_perception_with_qwen3_omni,
    needs_qwen3_omni_audio_perception,
)
from x_editor.perception.speech import SpeechPerceptionClient


class PerceptionFallbackError(RuntimeError):
    """Raised only after specialized perception and fallback perception fail."""


@dataclass(frozen=True)
class PerceptionRequirements:
    """Minimum evidence a downstream operation needs from perception."""

    source_text: bool = False
    observe_source_text: bool = False
    word_timestamps: bool = False
    exact_word_timestamps: bool = False
    speaker_segments: bool = False
    speaker_profiles: bool = False
    segment_descriptions: bool = False


@dataclass
class FallbackPerceptionResult:
    """Structured result from the fallback perception provider."""

    perception: dict[str, Any]
    satisfied: bool
    missing: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)


class GeminiAudioPerceptionFallback:
    """Compatibility class for framework-level audio perception fallback.

    The default provider is Gemini 2.5 Pro.  Open mode should construct this
    through ``build_audio_perception_fallback`` so the provider becomes
    Qwen3-Omni without changing the downstream evidence contract.
    """

    def __init__(
        self,
        *,
        speech_client: SpeechPerceptionClient | None = None,
        provider: str = "gemini",
    ) -> None:
        self.speech_client = speech_client
        self.provider = _normalize_fallback_provider(provider)

    def perceive(
        self,
        *,
        user_prompt: str,
        audio_path: str | Path,
        base_perception: dict[str, Any] | None = None,
        source_text: str | None = None,
        recognition_context: str | None = None,
        reference_audio_paths: list[str] | None = None,
        base_audio_ref: str | None = None,
        reference_audio_refs: list[str] | None = None,
        alternate_audio_paths: list[str | Path] | None = None,
        output_dir: str | Path | None = None,
        requirements: PerceptionRequirements | None = None,
        language_hints: list[str | None] | None = None,
        run_specialized: bool = True,
        allow_gemini: bool = True,
        force_gemini: bool = False,
        allow_model_fallback: bool | None = None,
        force_model_fallback: bool | None = None,
        refresh_cache: bool = False,
        strict: bool = False,
        frontend: Any | None = None,
    ) -> FallbackPerceptionResult:
        """Return perception that satisfies requirements when possible.

        Specialized services are tried first when ``run_specialized`` is true.
        The selected fallback provider is invoked when the specialized result is
        insufficient, when the instruction needs profile/role/phrase grounding,
        or when forced.  A strict call raises only after the fallback result is
        still insufficient.
        """

        if recognition_context is not None:
            if not isinstance(recognition_context, str):
                raise TypeError("recognition_context must be a string")
            recognition_context = recognition_context.strip() or None
        if recognition_context and not run_specialized:
            raise ValueError("recognition_context is an ASR input; use user_prompt for direct model listening")
        if allow_model_fallback is not None:
            allow_gemini = allow_model_fallback
        if force_model_fallback is not None:
            force_gemini = force_model_fallback
        requirements = requirements or PerceptionRequirements(source_text=True)
        perception = dict(base_perception or {})
        attempts: list[dict[str, Any]] = list(perception.get("perception_attempts") or [])
        audio = Path(audio_path).expanduser().resolve()
        refs = list(reference_audio_paths or [])

        if run_specialized:
            try:
                client = self.speech_client or SpeechPerceptionClient(
                    cache_dir=Path(output_dir) / "perception_cache" if output_dir else runtime_artifact_path("perception_cache")
                )
                specialized, specialized_attempts = _perceive_specialized_with_repair(
                    client,
                    audio,
                    alternate_audio_paths=list(alternate_audio_paths or []),
                    source_text=source_text or perception.get("source_text") or None,
                    requirements=requirements,
                    language_hints=list(language_hints or []),
                    refresh_cache=refresh_cache,
                    **({"recognition_context": recognition_context} if recognition_context else {}),
                )
                attempts.extend(specialized_attempts)
                if recognition_context:
                    # A requested re-recognition is a new observation, not a
                    # request to align or fill gaps in the old transcript.
                    prior_speech = {
                        key: perception[key]
                        for key in ("source_text", "word_timestamps", "source_text_source")
                        if key in perception
                    }
                    perception = specialized
                    if prior_speech:
                        perception["prior_speech_observation"] = prior_speech
                else:
                    perception = _merge_non_empty(perception, specialized)
            except Exception as exc:
                if recognition_context:
                    # Do not present an old transcript or an unrequested model
                    # fallback as a successful contextual recognition.
                    raise
                failure = {
                    "provider": "specialized_speech_perception",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if hasattr(exc, "to_dict"):
                    failure["error_details"] = exc.to_dict()
                attempts.append(
                    failure
                )
                perception.setdefault("errors", []).append(
                    {"stage": "specialized_perception", "message": f"{type(exc).__name__}: {exc}"}
                )

        missing_before = missing_requirements(perception, requirements)
        provider_config = _fallback_provider_config(self.provider)
        should_call_fallback = (
            allow_gemini
            and (
                force_gemini
                or bool(missing_before)
                or provider_config["needs"](
                    user_prompt,
                    perception,
                    need_words=requirements.word_timestamps,
                    need_speakers=requirements.speaker_segments,
                )
            )
        )
        if should_call_fallback:
            before_errors = len(perception.get("errors") or [])
            fallback_call_failed = False
            try:
                perception = provider_config["enrich"](
                    user_prompt=user_prompt,
                    audio_path=audio,
                    perception=perception,
                    reference_audio_paths=refs,
                    base_audio_ref=base_audio_ref,
                    reference_audio_refs=reference_audio_refs,
                    output_dir=output_dir,
                    frontend=frontend,
                    force=True,
                )
            except Exception as exc:
                error = {"stage": provider_config["stage"], "message": f"{type(exc).__name__}: {exc}"}
                perception.setdefault("errors", []).append(error)
                attempts.append(
                    {
                        "provider": provider_config["attempt_provider"],
                        "status": "failed",
                        "errors": [error],
                        "used_for": missing_before or ["instruction_relevant_grounding"],
                    }
                )
                fallback_call_failed = True
                new_errors = []
            else:
                new_errors = list(perception.get("errors") or [])[before_errors:]
            if fallback_call_failed:
                pass
            elif new_errors and not _has_provider_perception(perception, provider_config):
                attempts.append(
                    {
                        "provider": provider_config["attempt_provider"],
                        "status": "failed",
                        "errors": new_errors,
                    }
                )
            else:
                attempts.append(
                    {
                        "provider": provider_config["attempt_provider"],
                        "status": "completed",
                        "used_for": missing_before or ["instruction_relevant_grounding"],
                    }
                )
        elif not allow_gemini and missing_before:
            attempts.append(
                {
                    "provider": provider_config["attempt_provider"],
                    "status": "skipped",
                    "reason": "disabled",
                    "would_have_used_for": missing_before,
                }
            )

        missing_after = missing_requirements(perception, requirements)
        if (
            requirements.observe_source_text
            and not requirements.source_text
            and not _has_source_text(perception)
        ):
            perception.setdefault("uncertainties", []).append(
                {
                    "audio_ref_id": base_audio_ref,
                    "code": "source_text_not_observed",
                    "message": "No reliable speech transcript was observed for this audio.",
                    "severity": "info",
                    "blocking": False,
                    "missing_evidence": ["source_text"],
                }
            )
        satisfied = not missing_after
        if missing_after and any(
            item.get("status") in {"completed", "insufficient"} and not item.get("errors")
            for item in attempts
        ):
            perception.setdefault("uncertainties", []).append({
                "audio_ref_id": base_audio_ref or str(audio),
                "code": "observation_incomplete",
                "message": "The completed observation did not supply: "
                           + ", ".join(missing_after)
                           + ". This is missing evidence, not a finding of absence.",
                "severity": "advisory", "blocking": False,
                "missing_evidence": list(missing_after),
            })
        perception["perception_attempts"] = attempts
        perception["perception_fallback"] = {
            "provider": provider_config["attempt_provider"],
            "status": "satisfied" if satisfied else "insufficient",
            "requirements": requirements.__dict__,
            "missing_before_fallback": missing_before,
            "missing_after_fallback": missing_after,
        }
        if strict and not satisfied:
            if not allow_gemini:
                failure_path = (
                    "specialist speech perception "
                    "(Qwen3-ASR/forced alignment; model fallback disabled)"
                )
            else:
                failure_path = f"{provider_config['attempt_provider']} fallback"
            raise PerceptionFallbackError(
                "Perception requirements remain unsatisfied after "
                f"{failure_path}: " + ", ".join(missing_after)
            )
        return FallbackPerceptionResult(
            perception=perception,
            satisfied=satisfied,
            missing=missing_after,
            attempts=attempts,
        )


class Qwen3OmniAudioPerceptionFallback(GeminiAudioPerceptionFallback):
    """Open perception fallback backed by Qwen3-Omni semantic audio review."""

    def __init__(self, *, speech_client: SpeechPerceptionClient | None = None) -> None:
        super().__init__(speech_client=speech_client, provider="qwen3_omni")


def build_audio_perception_fallback(
    *,
    agentic_mode: str | None = None,
    speech_client: SpeechPerceptionClient | None = None,
) -> GeminiAudioPerceptionFallback:
    mode = resolve_agentic_mode(agentic_mode)
    if mode == OPEN_GLM_QWEN_OMNI_MODE:
        return Qwen3OmniAudioPerceptionFallback(speech_client=speech_client)
    return GeminiAudioPerceptionFallback(speech_client=speech_client)


def _perceive_specialized_with_repair(
    client: SpeechPerceptionClient,
    primary_audio: Path,
    *,
    alternate_audio_paths: list[str | Path],
    source_text: str | None,
    requirements: PerceptionRequirements,
    language_hints: list[str | None],
    refresh_cache: bool,
    recognition_context: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Try specialized perception, including bounded ASR repair attempts.

    Multi-audio tasks sometimes put the most useful speech-bearing or
    language-reference signal in a later input. Prefer the first audio that
    yields source text, but keep every attempt in the trace.
    """

    # Recognition context is scoped to this asset. It must not silently move
    # to a reference audio merely because no target speech was recognized.
    audios = [primary_audio]
    if not recognition_context:
        audios.extend(Path(path).expanduser().resolve() for path in alternate_audio_paths)
    attempts: list[dict[str, Any]] = []
    needs_asr = bool(recognition_context) or (
        (requirements.source_text or requirements.observe_source_text
         or requirements.word_timestamps or requirements.exact_word_timestamps)
        and not source_text
    )
    if not needs_asr:
        perception = client.perceive(
            primary_audio,
            source_text_hint=source_text,
            use_asr=False,
            use_alignment=requirements.word_timestamps,
            use_diarization=requirements.speaker_segments,
            refresh_cache=refresh_cache,
            strict_asr=False,
            strict_alignment=False,
            strict_diarization=False,
        )
        attempts.append(
            {
                "provider": "specialized_speech_perception",
                "status": "completed",
                "audio_index": 0,
                "audio_path": str(primary_audio),
                "language": perception.get("language"),
                "refresh_cache": refresh_cache,
                "source_text": perception.get("source_text"),
            }
        )
        perception["source_text_audio_index"] = 0
        perception["source_text_audio_path"] = str(primary_audio)
        if str(perception.get("source_text") or "").strip():
            perception.setdefault("source_text_source", "source_text_hint" if source_text else "qwen3_asr")
        return perception, attempts

    best: tuple[int, dict[str, Any], list[str]] | None = None
    for audio_index, audio in enumerate(audios):
        tried: set[str | None] = set()
        for attempt_index, (language, attempt_refresh) in enumerate(
            ([(language_hints[0] if language_hints else None, refresh_cache)]
             if recognition_context else _specialized_attempt_plan(language_hints, refresh_cache))
        ):
            if language in tried and not attempt_refresh:
                continue
            tried.add(language)
            try:
                perception = client.perceive(
                    audio,
                    language=language,
                    source_text_hint=None,
                    use_asr=True,
                    use_alignment=requirements.word_timestamps,
                    use_diarization=requirements.speaker_segments,
                    refresh_cache=attempt_refresh,
                    strict_asr=False,
                    strict_alignment=False,
                    strict_diarization=False,
                    **({"recognition_context": recognition_context} if recognition_context else {}),
                )
                text = str(perception.get("source_text") or "").strip()
                missing = missing_requirements(perception, requirements)
                attempts.append(
                    {
                        "provider": "specialized_speech_perception",
                        "status": "completed" if not missing else "insufficient",
                        "audio_index": audio_index,
                        "audio_path": str(audio),
                        "language": language,
                        "refresh_cache": attempt_refresh,
                        "source_text": text,
                        "has_word_timestamps": _has_word_timestamps(perception),
                        "missing_evidence": missing,
                        "errors": perception.get("errors", []),
                        **({"recognition_context": recognition_context,
                            "recognition_context_role": "unverified_recognition_hint"}
                           if recognition_context else {}),
                    }
                )
                if not missing:
                    perception["source_text_audio_index"] = audio_index
                    perception["source_text_audio_path"] = str(audio)
                    if text:
                        perception.setdefault("source_text_source", "qwen3_asr")
                    return perception, attempts
                if best is None or len(missing) < len(best[2]):
                    best = (audio_index, perception, missing)
            except Exception as exc:
                if recognition_context:
                    raise
                failure = {
                    "provider": "specialized_speech_perception",
                    "status": "failed",
                    "audio_index": audio_index,
                    "audio_path": str(audio),
                    "language": language,
                    "refresh_cache": attempt_refresh,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if hasattr(exc, "to_dict"):
                    failure["error_details"] = exc.to_dict()
                attempts.append(
                    failure
                )

    if best is not None:
        audio_index, perception, _missing = best
        perception["source_text_audio_index"] = audio_index
        perception["source_text_audio_path"] = str(audios[audio_index])
        if str(perception.get("source_text") or "").strip():
            perception.setdefault("source_text_source", "qwen3_asr")
        return perception, attempts
    return {"source_text": ""}, attempts


def _specialized_attempt_plan(
    language_hints: list[str | None],
    refresh_cache: bool,
) -> list[tuple[str | None, bool]]:
    plan: list[tuple[str | None, bool]] = [(None, refresh_cache)]
    if not refresh_cache:
        plan.append((None, True))
    for language in language_hints:
        plan.append((language, True))
    deduped: list[tuple[str | None, bool]] = []
    seen: set[tuple[str | None, bool]] = set()
    for item in plan:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def missing_requirements(perception: dict[str, Any], requirements: PerceptionRequirements) -> list[str]:
    """Return requirement names still missing from a perception report."""

    missing: list[str] = []
    if requirements.source_text and not _has_source_text(perception):
        missing.append("source_text")
    if requirements.exact_word_timestamps and not _has_word_timestamps(perception):
        missing.append("exact_word_timestamps")
    elif requirements.word_timestamps and not (
        _has_word_timestamps(perception) or _has_described_segments(perception)
    ):
        missing.append("word_timestamps_or_described_segments")
    if requirements.speaker_segments and not _has_speaker_segments(perception):
        missing.append("speaker_segments")
    if requirements.speaker_profiles and not _has_speaker_profiles(perception):
        missing.append("speaker_profiles")
    if requirements.segment_descriptions and not _has_described_segments(perception):
        missing.append("segment_descriptions")
    return missing


def _normalize_fallback_provider(provider: str) -> str:
    raw = str(provider or "gemini").strip().lower().replace("-", "_")
    if raw in {"gemini", "gemini_2_5_pro", "gemini_2_5pro"}:
        return "gemini"
    if raw in {"qwen", "qwen3", "qwen3_omni", "open_omni"}:
        return "qwen3_omni"
    raise ValueError(f"Unsupported perception fallback provider: {provider!r}")


def _fallback_provider_config(provider: str) -> dict[str, Any]:
    if provider == "qwen3_omni":
        return {
            "attempt_provider": "qwen3_omni_audio_fallback",
            "stage": "qwen3_omni_audio_perception",
            "perception_key": "qwen3_omni_perception",
            "needs": needs_qwen3_omni_audio_perception,
            "enrich": enrich_perception_with_qwen3_omni,
        }
    return {
        "attempt_provider": "gemini_2_5_pro_audio_fallback",
        "stage": "gemini_audio_perception",
        "perception_key": "gemini_perception",
        "needs": needs_gemini_audio_perception,
        "enrich": enrich_perception_with_gemini,
    }


def _has_provider_perception(perception: dict[str, Any], provider_config: dict[str, Any]) -> bool:
    key = provider_config.get("perception_key")
    return isinstance(key, str) and bool(perception.get(key))


def _merge_non_empty(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if value or key not in merged:
            merged[key] = value
    return merged


def _has_source_text(perception: dict[str, Any]) -> bool:
    return bool(str(perception.get("source_text") or "").strip())


def _has_word_timestamps(perception: dict[str, Any]) -> bool:
    return bool(perception.get("word_timestamps"))


def _has_speaker_segments(perception: dict[str, Any]) -> bool:
    for key in ("speaker_segments", "segments"):
        for item in perception.get(key) or []:
            if not isinstance(item, dict):
                continue
            if item.get("speaker_id") and item.get("start_sec", item.get("start")) is not None and item.get("end_sec", item.get("end")) is not None:
                return True
    return False


def _has_speaker_profiles(perception: dict[str, Any]) -> bool:
    for key in ("speakers", "speaker_segments", "segments"):
        for item in perception.get(key) or []:
            if not isinstance(item, dict):
                continue
            if item.get("description") or item.get("perceived_gender") or item.get("gender") or item.get("age") or item.get("role"):
                return True
    return False


def _has_described_segments(perception: dict[str, Any]) -> bool:
    for key in ("speaker_segments", "segments"):
        for item in perception.get(key) or []:
            if not isinstance(item, dict):
                continue
            if (
                item.get("start_sec", item.get("start")) is not None
                and item.get("end_sec", item.get("end")) is not None
                and (item.get("transcript") or item.get("text") or item.get("description"))
            ):
                return True
    return False
