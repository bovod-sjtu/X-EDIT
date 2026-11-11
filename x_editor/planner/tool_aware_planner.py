"""Tool-aware planning interface and deterministic bootstrap planner.

This planner is intentionally conservative. It provides a runnable contract for
the new graph before Gemini-based prompts are wired in, and it refuses requests
that are outside the current tool pool instead of inventing capabilities.
"""

from __future__ import annotations

import os
import re
from typing import Any

import requests

from x_editor.core.config import runtime_artifact_path
from x_editor.runtime.accent_conversion import (
    accent_conversion_failure_reason,
    requests_reference_accent_match,
    resolve_requested_accent,
)
from x_editor.runtime.audio_effects import canonical_audio_effect_preset
from x_editor.runtime.paralinguistic_events import (
    normalize_paralinguistic_event,
    normalize_paralinguistic_operation,
)
from x_editor.core.schemas import EnrichedIntent, ToolAction, ToolCall, ToolCapability, ToolManual
from x_editor.planner.routing_policy import routing_policy_context
from x_editor.planner.structured_edit import (
    build_edit_data_pipeline_arguments,
    build_local_text_edit_repair_arguments,
    build_structured_edit_sequence,
)
from x_editor.tools.loader import get_catalog_dir
from x_editor.tools.edit_backend import dots_service_url, uses_dots_model
from x_editor.tools.service_bootstrap import _matches_when
from x_editor.tools.external_services import (
    ExternalToolSetupRequired,
    POLICIES,
    require_external_service,
    require_pipeline_service,
)

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in stripped environments
    yaml = None

LANGUAGE_NAME_TO_CODE = {
    "arabic": "arb",
    "阿拉伯语": "arb",
    "chinese": "cmn",
    "mandarin": "cmn",
    "中文": "cmn",
    "汉语": "cmn",
    "普通话": "cmn",
    "german": "deu",
    "德语": "deu",
    "english": "eng",
    "英语": "eng",
    "spanish": "spa",
    "西班牙语": "spa",
    "french": "fra",
    "法语": "fra",
    "hindi": "hin",
    "印地语": "hin",
    "indonesian": "ind",
    "印尼语": "ind",
    "italian": "ita",
    "意大利语": "ita",
    "japanese": "jpn",
    "日语": "jpn",
    "korean": "kor",
    "韩语": "kor",
    "dutch": "nld",
    "荷兰语": "nld",
    "polish": "pol",
    "波兰语": "pol",
    "portuguese": "por",
    "葡萄牙语": "por",
    "russian": "rus",
    "俄语": "rus",
    "swedish": "swe",
    "瑞典语": "swe",
    "turkish": "tur",
    "土耳其语": "tur",
    "ukrainian": "ukr",
    "乌克兰语": "ukr",
    "urdu": "urd",
    "乌尔都语": "urd",
    "vietnamese": "vie",
    "越南语": "vie",
    "arb": "arb",
    "cmn": "cmn",
    "deu": "deu",
    "eng": "eng",
    "spa": "spa",
    "fra": "fra",
    "hin": "hin",
    "ind": "ind",
    "ita": "ita",
    "jpn": "jpn",
    "kor": "kor",
    "nld": "nld",
    "pol": "pol",
    "por": "por",
    "rus": "rus",
    "swe": "swe",
    "tur": "tur",
    "ukr": "ukr",
    "urd": "urd",
    "vie": "vie",
}


_TOOL_SERVICE_URLS = {
    "seamless_expressive_s2st": ("SEAMLESS_S2ST_SERVICE_URL", "http://127.0.0.1:8039"),
    "ace_step_music": ("ACE_STEP_API_URL", "http://127.0.0.1:8044"),
    "qwen3_tts": ("QWEN3_TTS_SERVICE_URL", "http://127.0.0.1:8017"),
    "seed_vc": ("SEED_VC_SERVICE_URL", "http://127.0.0.1:8003"),
    "speech_enhancement": ("CLEARERVOICE_SERVICE_URL", "http://127.0.0.1:8027"),
    "tse": ("QUARKAUDIO_UNISE_SERVICE_URL", "http://127.0.0.1:8025"),
    "speech_separation": ("AUDIO_SEPARATOR_ROFORMER_SERVICE_URL", "http://127.0.0.1:8042"),
}

_EDIT_PIPELINE_SERVICE_URLS = {
    "text_edit": ("EDIT_PIPELINE_TEXT_TAGGED_URL", "http://127.0.0.1:8035"),
    "pitch_rate_edit": ("EDIT_PIPELINE_PITCH_RATE_URL", "http://127.0.0.1:8036"),
    "emotion_style_edit": ("EDIT_PIPELINE_EMOTION_STYLE_URL", "http://127.0.0.1:8037"),
    "pause_edit": ("EDIT_PIPELINE_PAUSE_EDIT_URL", "http://127.0.0.1:8038"),
    "volume_normalize": ("EDIT_PIPELINE_VOLUME_URL", "http://127.0.0.1:8006"),
    "speech_rate": ("EDIT_PIPELINE_SPEECH_RATE_URL", "http://127.0.0.1:8004"),
    "boundary_smooth": ("EDIT_PIPELINE_BOUNDARY_SMOOTHER_URL", "http://127.0.0.1:8018"),
    "audio_assembly": ("EDIT_PIPELINE_AUDIO_ASSEMBLY_URL", "http://127.0.0.1:8021"),
}


class ToolAwarePlanner:
    """Planner facade for intent enrichment, routing, plan creation, and tool calls."""

    def __init__(
        self,
        *,
        check_service_health: bool | None = None,
        service_health_timeout: float = 0.75,
    ) -> None:
        if check_service_health is None:
            check_service_health = os.getenv("X_EDITOR_PLANNER_CHECK_SERVICE_HEALTH", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.check_service_health = check_service_health
        self.service_health_timeout = service_health_timeout
        self._service_health_cache: dict[tuple[str, str, str], bool] = {}

    def query_routing_policy(
        self,
        intent: EnrichedIntent | str,
        capabilities: dict[str, ToolCapability] | None = None,
        *,
        reference_audio_paths: list[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Return policy-backed candidate workflows for planner/executor context."""

        return routing_policy_context(
            intent,
            capabilities=capabilities,
            reference_audio_paths=reference_audio_paths,
            limit=limit,
        )

    def enrich_intent(
        self,
        user_prompt: str,
        perception_report: dict[str, Any] | None = None,
        reference_audio_paths: list[str] | None = None,
    ) -> EnrichedIntent:
        prompt = user_prompt.strip()
        target_objects: list[dict[str, Any]] = []
        requirements: list[str] = []
        preserve: list[str] = []
        assumptions: list[str] = []

        if _mentions_female(prompt):
            target_objects.append({"type": "speaker", "hint": "female speaker"})
        if _mentions_male(prompt):
            requirements.append("target voice should sound male when requested")
        if _contains(prompt, ["清晰", "干净", "降噪", "噪声", "clear", "clean", "noise"]):
            requirements.append("speech quality should be improved")
        if _wants_speech_to_speech_translation(prompt):
            requirements.append("translate input speech into target-language speech")
        accent_target = resolve_requested_accent(prompt)
        if _wants_accent_conversion(prompt) and accent_target:
            requirements.append(f"convert target speaker pronunciation/accent to {accent_target}")
        if _wants_singing_or_rap(prompt):
            requirements.append("generate or edit the requested speaker/line as singing or rap music vocals")
        if _contains(prompt, ["不变", "保持", "preserve", "unchanged"]):
            preserve.append("preserve unspecified speakers, content, and background unless explicitly edited")
        if _wants_expressive_preservation(prompt):
            preserve.append("preserve cross-language vocal style, speech rate, and pauses as much as the selected S2ST model supports")
        if reference_audio_paths:
            assumptions.append("reference audio is intended as target speaker evidence")
        derived_speaker = _speaker_reference_candidate(perception_report or {}, prompt)
        if derived_speaker:
            target_objects.append(
                {
                    "type": "speaker",
                    "hint": derived_speaker.get("target_hint", "target speaker"),
                    "speaker_id": derived_speaker.get("speaker_id"),
                    "segment_id": derived_speaker.get("segment_id"),
                    "start_sec": derived_speaker.get("start_sec"),
                    "end_sec": derived_speaker.get("end_sec"),
                    "reference_source": "perception_segments",
                }
            )
            assumptions.append("target speaker reference can be derived from perceived diarization segments")

        return EnrichedIntent(
            original_request=user_prompt,
            clarified_goal=prompt,
            target_objects=target_objects,
            edit_requirements=requirements,
            preservation_requirements=preserve,
            assumptions=assumptions,
            uncertainty=list((perception_report or {}).get("uncertainty", [])),
        )

    def route_capabilities(
        self,
        intent: EnrichedIntent,
        capabilities: dict[str, ToolCapability],
        reference_audio_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        text = intent.clarified_goal.lower()
        refs = reference_audio_paths or []

        if _contains(
            text,
            [
                "动物声",
                "bird call",
                "空间位置",
                "运动轨迹",
                "spatial trajectory",
                "sound effect",
                "sound event",
            ],
        ):
            return {
                "planning_status": "UNSUPPORTED",
                "selected_tools": [],
                "error_message": "Request requires sound-event replacement or spatial trajectory preservation outside the current speech-editing tool pool.",
            }
        selected: list[str] = []
        if _wants_audio_effect_tool(text):
            if "audio_effects" not in capabilities:
                return _unsupported(
                    "Request requires an audio-effects tool for telephone/radio/underwater/wind/muffled/channel effects, "
                    "but `audio_effects` is not registered."
                )
            if _requests_reference_channel_match(intent.clarified_goal) and not refs:
                return _unsupported("Reference channel matching requires a concrete reference_audio_path such as audio2.")
            _append_runtime_tool(selected, "audio_effects")
            return {"planning_status": "READY", "selected_tools": selected, "error_message": None}

        if _wants_accent_conversion(intent.clarified_goal):
            if "seed_vc" not in capabilities:
                return _unsupported("Accent conversion requires SeedVC, but `seed_vc` is not registered.")
            if "qwen3_tts" not in capabilities:
                return _unsupported("Accent conversion requires zero-shot TTS, but `qwen3_tts` is not registered.")
            if not self._tool_service_available("seed_vc"):
                return _unsupported(
                    "Accent conversion requires SeedVC to convert the accent reference into the target speaker timbre, "
                    f"but SeedVC is unavailable at {_service_url('seed_vc', _TOOL_SERVICE_URLS)}."
                )
            if not self._qwen3_tts_voice_clone_available():
                return _unsupported(
                    "Accent conversion requires Qwen3-TTS voice_clone synthesis, "
                    f"but Qwen3-TTS voice_clone is unavailable at {_qwen3_tts_voice_clone_service_url()}. "
                    "The service health must report base_loaded=true and tokenizer_loaded=true."
                )
            library_reason = None
            if not (refs and requests_reference_accent_match(intent.clarified_goal)):
                library_reason = accent_conversion_failure_reason(intent.clarified_goal)
            if library_reason:
                return _unsupported(f"Accent conversion cannot be planned: {library_reason}")
            _append_runtime_tool(selected, "accent_conversion")
            return {"planning_status": "READY", "selected_tools": selected, "error_message": None}

        if _wants_speech_to_speech_translation(intent.clarified_goal):
            target_language = _target_language_code(intent.clarified_goal)
            if target_language is None:
                return {
                    "planning_status": "UNSUPPORTED",
                    "selected_tools": [],
                    "error_message": "Speech-to-speech translation requires an explicit target language, e.g. 'translate to French'.",
                }
            if "seamless_expressive_s2st" not in capabilities:
                return {
                    "planning_status": "UNSUPPORTED",
                    "selected_tools": [],
                    "error_message": "Request needs speech-to-speech translation, but seamless_expressive_s2st is not registered.",
                }
            error = self._append_checked_tool(selected, "seamless_expressive_s2st", capabilities)
            if error:
                return _unsupported(
                    error
                    + " The retry planner may use text_edit only when perception provides a reliable source transcript "
                    "and the target-language transcript can be generated explicitly."
                )
            if _contains(text, ["清晰", "干净", "降噪", "噪声", "音质", "clear", "clean", "enhance"]):
                _append_if_available(selected, "speech_enhancement", capabilities)
            return {"planning_status": "READY", "selected_tools": selected, "error_message": None}

        if _wants_singing_or_rap(intent.clarified_goal):
            if "ace_step_music" not in capabilities:
                return {
                    "planning_status": "UNSUPPORTED",
                    "selected_tools": [],
                    "error_message": "Request needs singing/rap music-vocal generation, but ace_step_music is not registered.",
                }
            _append_if_available(selected, "ace_step_music", capabilities)
            if _contains(text, ["清晰", "干净", "降噪", "噪声", "音质", "clear", "clean", "enhance"]):
                _append_if_available(selected, "speech_enhancement", capabilities)
            return {"planning_status": "READY", "selected_tools": selected, "error_message": None}

        paralinguistic_operation = normalize_paralinguistic_operation(intent.clarified_goal)
        paralinguistic_event = normalize_paralinguistic_event(intent.clarified_goal)
        if paralinguistic_operation and paralinguistic_event:
            _append_runtime_tool(selected, f"paralinguistic_event_{paralinguistic_operation}")
            return {"planning_status": "READY", "selected_tools": selected, "error_message": None}

        changes_text = _wants_text_edit(intent.clarified_goal)
        asks_emotion_style = _wants_emotion_style_edit(intent.clarified_goal)
        wants_edit_pipeline = _wants_edit_data_pipeline(intent.clarified_goal)
        wants_extraction = _wants_audio_extraction(text)
        wants_speech_separation = _wants_speech_separation(text)
        wants_assembly = _wants_audio_assembly(text)
        wants_removal = _wants_audio_removal(text)
        wants_profanity_mask = _wants_profanity_mask(text)
        wants_spatial_reverb = _wants_spatial_reverb(text)
        wants_scene_change = _wants_scene_or_background_change(text)
        wants_foreground_background_remix = _wants_foreground_background_remix(text)
        wants_quality_cleanup = _wants_quality_cleanup(text)
        wants_speaker_scope = _wants_speaker_scoped_edit(text)
        wants_intra_audio_voice_transfer = _wants_intra_audio_voice_transfer(text, refs)
        if changes_text and _wants_global_text_edit_across_speakers(text):
            wants_speaker_scope = False
        wants_no_ref_voice_synthesis = _wants_no_ref_voice_synthesis(text, refs)
        wants_source_voice_clone_tts = _wants_source_voice_clone_tts(text, refs)
        wants_speaker_param_edit = _wants_speaker_profile_param_edit(text, refs)
        wants_reference_vc = _wants_reference_voice_conversion(text, refs) or (
            bool(refs) and _wants_speaker_profile_param_edit(text, [])
        )
        wants_tse = _wants_target_speaker_extraction(text, refs)
        wants_scoped_quality_cleanup = wants_quality_cleanup and not _wants_phone_effect_cleanup(text)
        speaker_scoped_local_edit = wants_speaker_scope and any(
            [
                changes_text,
                asks_emotion_style,
                _wants_rate_edit(text),
                _wants_pitch_edit(text),
                _wants_volume_edit(text),
                _wants_pause_edit(text),
                wants_scoped_quality_cleanup,
            ]
        )
        if wants_extraction:
            speaker_scoped_local_edit = False
        if speaker_scoped_local_edit and wants_speech_separation and not wants_extraction:
            # Generic speech_separation returns source stems, not stable male/female/role speaker ids.
            # Keep speaker-scoped enhancement/volume edits on artifact_ops, which can use perception evidence.
            wants_speech_separation = False

        if changes_text:
            # Lexical edits to existing audio are a hard edit-data-pipeline
            # boundary. "same/original voice" is a preservation constraint, not
            # permission to regenerate via TTS.
            wants_no_ref_voice_synthesis = False
            wants_source_voice_clone_tts = False
        if asks_emotion_style or _wants_rate_edit(text) or _wants_pitch_edit(text) or _wants_pause_edit(text):
            # Existing-audio tone/prosody operations are handled by
            # edit-data-pipeline when representable. Do not let broad "voice"
            # wording promote them to no-reference TTS.
            wants_no_ref_voice_synthesis = False

        if wants_no_ref_voice_synthesis:
            # No-reference timbre/persona control is better represented as
            # voice-designed TTS than as pitch/volume-only approximation.
            # Qwen3-TTS can absorb simple text and emotion/style changes
            # through the synthesized `text` and natural-language `instruct`.
            wants_edit_pipeline = False

        if refs and wants_reference_vc and not any(
            [changes_text, asks_emotion_style, _wants_rate_edit(text), _wants_pitch_edit(text), _wants_volume_edit(text), _wants_pause_edit(text)]
        ):
            wants_edit_pipeline = False

        if wants_assembly:
            _append_runtime_tool(selected, "artifact_ops")

        if wants_spatial_reverb:
            error = self._append_checked_tool(selected, "spatial_reverb", capabilities, check_runtime=False)
            if error:
                return _unsupported(error)

        if wants_foreground_background_remix:
            if "speech_separation" not in capabilities:
                return _unsupported("Foreground/background remix requires `speech_separation`, but it is not present in the current catalog.")
            if not self._tool_service_available("speech_separation"):
                return _unsupported(
                    "Foreground/background remix requires BS-RoFormer speech_separation service, but it is unavailable at "
                    f"{_service_url('speech_separation', _TOOL_SERVICE_URLS)}."
                )
            if wants_edit_pipeline:
                edit_operations = _edit_pipeline_operations(intent.clarified_goal)
                unavailable = [
                    operation
                    for operation in edit_operations
                    if not self._edit_pipeline_operation_available(operation)
                ]
                if unavailable:
                    return _unsupported(
                        "Foreground/background remix requires edit-data-pipeline operation(s): "
                        + ", ".join(unavailable)
                        + "."
                    )
            if wants_quality_cleanup and (
                "speech_enhancement" not in capabilities or not self._tool_service_available("speech_enhancement")
            ):
                return _unsupported(
                    "Foreground/background cleanup requires `speech_enhancement`, but ClearerVoice is unavailable at "
                    f"{_service_url('speech_enhancement', _TOOL_SERVICE_URLS)}."
                )
            _append_runtime_tool(selected, "foreground_background_remix")
            wants_speech_separation = False
            wants_edit_pipeline = False
            wants_extraction = False
            wants_removal = False
            wants_quality_cleanup = False

        if wants_scene_change and not wants_spatial_reverb and not wants_foreground_background_remix:
            return _unsupported(
                "Request changes foreground/background or acoustic environment. "
                "No current X-Editor tool can model this scene-level speech audio edit."
            )

        if wants_intra_audio_voice_transfer:
            error = self._append_checked_tool(selected, "intra_audio_voice_transfer", capabilities, check_runtime=False)
            if error:
                return _unsupported(error)
            if not self._tool_service_available("seed_vc"):
                return _unsupported(
                    "Request transfers timbre between speakers inside the same audio, but SeedVC service is unavailable at "
                    f"{_service_url('seed_vc', _TOOL_SERVICE_URLS)}."
                )

        if wants_speech_separation and not wants_extraction:
            error = self._append_checked_tool(selected, "speech_separation", capabilities)
            if error:
                return _unsupported(error)

        if wants_removal and not wants_quality_cleanup:
            _append_runtime_tool(selected, "artifact_ops")

        if wants_profanity_mask:
            _append_runtime_tool(selected, "artifact_ops")

        if speaker_scoped_local_edit:
            edit_operations = _edit_pipeline_operations(intent.clarified_goal)
            unavailable = [
                operation
                for operation in edit_operations
                if not self._edit_pipeline_operation_available(operation)
            ]
            if unavailable:
                return _unsupported(
                    "Required edit-data-pipeline service is unavailable for speaker-scoped operation(s): "
                    + ", ".join(unavailable)
                    + "."
                )
            _append_runtime_tool(selected, "artifact_ops")
            wants_edit_pipeline = False
            wants_speaker_param_edit = False

        if wants_extraction:
            if wants_tse and refs:
                error = self._append_checked_tool(selected, "tse", capabilities)
                if error:
                    return _unsupported(error)
            elif wants_speech_separation:
                error = self._append_checked_tool(selected, "speech_separation", capabilities)
                if error:
                    return _unsupported(error)
            else:
                _append_runtime_tool(selected, "artifact_ops")

        if wants_edit_pipeline:
            edit_operations = _edit_pipeline_operations(intent.clarified_goal)
            unavailable = [
                operation
                for operation in edit_operations
                if not self._edit_pipeline_operation_available(operation)
            ]
            if unavailable:
                return _unsupported(
                    "Required edit-data-pipeline service is unavailable for operation(s): "
                    + ", ".join(unavailable)
                    + "."
                )
            error = self._append_checked_tool(selected, "edit_data_pipeline", capabilities, check_runtime=False)
            if error:
                return _unsupported(error)

        if wants_reference_vc:
            error = self._append_checked_tool(selected, "seed_vc", capabilities)
            if error:
                return _unsupported(error)
        elif wants_no_ref_voice_synthesis or wants_source_voice_clone_tts:
            voice_design_tool = _no_reference_voice_design_tool(capabilities)
            error = self._append_checked_tool(
                selected,
                voice_design_tool,
                capabilities,
                bootstrap_arguments=_qwen3_voice_design_backend_argument() if voice_design_tool == "qwen3_tts" else None,
            )
            if error:
                return _unsupported(error)
        elif wants_speaker_param_edit and "edit_data_pipeline" not in selected:
            edit_operations = _edit_pipeline_operations(intent.clarified_goal)
            unavailable = [
                operation
                for operation in edit_operations
                if not self._edit_pipeline_operation_available(operation)
            ]
            if unavailable:
                return _unsupported(
                    "Required edit-data-pipeline service is unavailable for operation(s): "
                    + ", ".join(unavailable)
                    + "."
                )
            error = self._append_checked_tool(selected, "edit_data_pipeline", capabilities, check_runtime=False)
            if error:
                return _unsupported(error)

        if wants_quality_cleanup:
            error = self._append_checked_tool(selected, "speech_enhancement", capabilities)
            if error:
                if _wants_phone_effect_cleanup(intent.clarified_goal):
                    error = self._append_checked_tool(selected, "qwen3_tts", capabilities)
                    if error:
                        return _unsupported(error)
                else:
                    return _unsupported(error)

        if not selected:
            return {
                "planning_status": "UNSUPPORTED",
                "selected_tools": [],
                "error_message": "No current X-Editor tool matches the request.",
            }

        return {"planning_status": "READY", "selected_tools": selected, "error_message": None}

    def _append_checked_tool(
        self,
        selected: list[str],
        tool_id: str,
        capabilities: dict[str, ToolCapability],
        *,
        check_runtime: bool = True,
        bootstrap_arguments: dict[str, Any] | None = None,
    ) -> str | None:
        if tool_id not in capabilities:
            return f"Tool `{tool_id}` is not present in the current catalog."
        if check_runtime and not self._tool_service_available(tool_id, bootstrap_arguments=bootstrap_arguments):
            return f"Tool `{tool_id}` service is unavailable at {_service_url(tool_id, _TOOL_SERVICE_URLS)}."
        if tool_id not in selected:
            selected.append(tool_id)
        return None

    def _tool_service_available(self, tool_id: str, *, bootstrap_arguments: dict[str, Any] | None = None) -> bool:
        return self._http_service_available("tool", tool_id, _TOOL_SERVICE_URLS, bootstrap_arguments=bootstrap_arguments)

    def _qwen3_tts_voice_clone_available(self) -> bool:
        if not getattr(self, "check_service_health", True):
            return True
        url = _qwen3_tts_voice_clone_service_url()
        cache_key = ("tool", "qwen3_tts_voice_clone", url)
        cache = getattr(self, "_service_health_cache", None)
        if cache is None:
            cache = {}
            self._service_health_cache = cache
        if cache_key not in cache:
            cache[cache_key] = _probe_qwen3_tts_voice_clone_url(
                url,
                getattr(self, "service_health_timeout", 0.75),
            )
            if not cache[cache_key]:
                cache[cache_key] = _managed_service_bootstrap_possible(
                    "qwen3_tts",
                    {"mode": "voice_clone"},
                )
        return cache[cache_key]

    def _edit_pipeline_operation_available(self, operation: str) -> bool:
        return self._http_service_available(
            "edit_pipeline",
            operation,
            _EDIT_PIPELINE_SERVICE_URLS,
            tool_id="edit_data_pipeline",
            bootstrap_arguments={"operation": operation},
        )

    def _http_service_available(
        self,
        namespace: str,
        key: str,
        urls: dict[str, tuple[str, str]],
        *,
        tool_id: str | None = None,
        bootstrap_arguments: dict[str, Any] | None = None,
    ) -> bool:
        try:
            if namespace == "edit_pipeline":
                require_pipeline_service(key)
            elif key in POLICIES:
                require_external_service(key)
        except ExternalToolSetupRequired:
            return False
        if not getattr(self, "check_service_health", True):
            return True
        if key not in urls:
            return True
        url = (dots_service_url() if namespace == "edit_pipeline" and uses_dots_model(key)
               else _service_url(key, urls))
        bootstrap_key = tuple(sorted((bootstrap_arguments or {}).items()))
        cache_key = (namespace, key, url, bootstrap_key)
        cache = getattr(self, "_service_health_cache", None)
        if cache is None:
            cache = {}
            self._service_health_cache = cache
        if cache_key not in cache:
            if namespace == "tool" and key == "seamless_expressive_s2st":
                cache[cache_key] = _probe_seamless_s2st_url(url, getattr(self, "service_health_timeout", 0.75))
            else:
                cache[cache_key] = _probe_health_url(url, getattr(self, "service_health_timeout", 0.75))
            if not cache[cache_key]:
                cache[cache_key] = _managed_service_bootstrap_possible(
                    tool_id or key,
                    bootstrap_arguments or {},
                )
        return cache[cache_key]

    def create_plan(
        self,
        intent: EnrichedIntent,
        selected_tools: list[str],
        base_artifact_id: str = "input_audio",
    ) -> list[ToolAction]:
        actions: list[ToolAction] = []
        current = base_artifact_id

        if "seamless_expressive_s2st" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_seamless_expressive_s2st",
                tool_id="seamless_expressive_s2st",
                objective="translate input speech into target-language speech while preserving vocal style, speech rate, and pauses when expressive runtime is available",
                input_artifacts=[current],
                output_artifact_role="translated_speech_audio",
                validation_criteria=[
                    "output speech is in the requested target language",
                    "translated meaning matches the source speech",
                    "vocal style, speech rate, and pauses are preserved when expressive backend is used",
                    "fallback or failure reason is explicit if expressive runtime is unavailable",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "translated_speech_audio"

        if "intra_audio_voice_transfer" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_intra_audio_voice_transfer",
                tool_id="intra_audio_voice_transfer",
                objective="diarize speakers, extract internal speaker references, convert target turns with SeedVC, and reconstruct the original timeline",
                input_artifacts=[current],
                output_artifact_role="intra_audio_voice_transfer_audio",
                validation_criteria=[
                    "target speaker timbre follows the requested internal reference speaker",
                    "spoken content and global timing are preserved",
                    "unselected speaker turns remain unchanged",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "intra_audio_voice_transfer_audio"

        if "accent_conversion" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_accent_conversion",
                tool_id="accent_conversion",
                objective=(
                    "select a matching accent-library reference, convert that reference to the target speaker timbre "
                    "with SeedVC, then synthesize the target speaker content using Qwen3-TTS voice_clone"
                ),
                input_artifacts=[current],
                output_artifact_role="accent_converted_audio",
                validation_criteria=[
                    "target accent/dialect is resolved to a canonical accent-library label",
                    "selected accent reference exists and is suitable for VC/TTS reference use",
                    "target speaker reference is derived from diarization/perception or the full input audio",
                    "synthesized speech says the target speaker content with the requested accent",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "accent_converted_audio"

        if "audio_effects" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_audio_effects",
                tool_id="audio_effects",
                objective=(
                    "apply deterministic speech-preserving acoustic/channel effects such as telephone, "
                    "old radio, underwater, wind, muffled wall, or reference channel matching"
                ),
                input_artifacts=[current],
                output_artifact_role="audio_effects_audio",
                validation_criteria=[
                    "spoken words and speaker identity remain recognizable",
                    "requested acoustic/channel effect is audible",
                    "timeline duration is preserved unless explicitly changed",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "audio_effects_audio"

        if "ace_step_music" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_ace_step_music",
                tool_id="ace_step_music",
                objective=(
                    "locate the requested speaker or line spans, use each selected speech segment as "
                    "ACE-Step source/reference audio for singing/rap music-vocal generation, and splice "
                    "the fitted generated segments back into the original timeline"
                ),
                input_artifacts=[current],
                output_artifact_role="music_vocal_audio",
                validation_criteria=[
                    "output audio exists and is playable",
                    "output contains singing or rap/music-vocal delivery requested by the user",
                    "unselected speakers and non-target timeline regions remain unchanged",
                    "lyrics and musical style match the prompt as much as ACE-Step supports",
                    "speaker identity and exact timing are reported as best-effort, not guaranteed",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "music_vocal_audio"

        if "paralinguistic_event_add" in selected_tools:
            event = normalize_paralinguistic_event(intent.clarified_goal) or "<breath|laugh|cough|sigh>"
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_paralinguistic_event_add",
                tool_id="paralinguistic_event_add",
                objective=(
                    f"add the requested paralinguistic event ({event}) by inserting an event asset at "
                    "a pause or safe boundary while preserving original speech"
                ),
                input_artifacts=[current],
                output_artifact_role="paralinguistic_event_added_audio",
                validation_criteria=[
                    f"the added {event} event is audible",
                    "speech content and speaker identity are preserved",
                    "the edit does not use text_edit or whole-utterance TTS",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "paralinguistic_event_added_audio"

        if "paralinguistic_event_remove" in selected_tools:
            event = normalize_paralinguistic_event(intent.clarified_goal) or "<breath|laugh|cough|sigh>"
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_paralinguistic_event_remove",
                tool_id="paralinguistic_event_remove",
                objective=(
                    f"remove the requested paralinguistic event ({event}) by grounding local event spans "
                    "and inpainting or ducking only those spans"
                ),
                input_artifacts=[current],
                output_artifact_role="paralinguistic_event_removed_audio",
                validation_criteria=[
                    f"the target {event} event is absent or strongly suppressed",
                    "speech content and speaker identity are preserved outside local event spans",
                    "the edit does not use text_edit or whole-utterance TTS",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "paralinguistic_event_removed_audio"

        if "tse" in selected_tools:
            actions.append(ToolAction(
                step_id="s1_tse",
                tool_id="tse",
                objective="extract the target speaker so later edits affect only that speaker",
                input_artifacts=[current],
                output_artifact_role="target_speaker_audio",
                validation_criteria=[
                    "target output contains the intended speaker",
                    "non-target speaker leakage is minimal",
                ],
            ))
            current = "target_speaker_audio"

        if "speech_separation" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_speech_separation",
                tool_id="speech_separation",
                objective="separate overlapping or mixed multi-speaker speech/source stems using audio-separator RoFormer backend",
                input_artifacts=[current],
                output_artifact_role="separated_speech_audio",
                validation_criteria=[
                    "output stems reduce speaker/source overlap",
                    "produced stems are inspected before assuming exact speaker identity",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "separated_speech_audio"

        if "foreground_background_remix" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_foreground_background_remix",
                tool_id="foreground_background_remix",
                objective=(
                    "separate foreground speech/vocals from background, edit only the selected foreground stem, "
                    "and remix it with the preserved background at the original timeline length"
                ),
                input_artifacts=[current],
                output_artifact_role="foreground_background_remix_audio",
                validation_criteria=[
                    "background timing and speed remain unchanged",
                    "requested foreground speech edit is audible",
                    "final audio duration stays aligned with the original unless explicitly requested otherwise",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "foreground_background_remix_audio"

        if "artifact_ops" in selected_tools:
            output_role = _artifact_operation_output_role(intent.clarified_goal)
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_artifact_ops",
                tool_id="artifact_ops",
                objective=_artifact_operation_objective(intent.clarified_goal),
                input_artifacts=[current],
                output_artifact_role=output_role,
                validation_criteria=_artifact_operation_validation(intent.clarified_goal),
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = output_role

        if "qwen3_tts" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_qwen3_tts",
                tool_id="qwen3_tts",
                objective="regenerate requested speech content or requested textual voice style using Qwen3-TTS or its selected voice-design backend",
                input_artifacts=[current],
                output_artifact_role="synthesized_audio",
                validation_criteria=[
                    "generated speech says the requested content",
                    "style and perceived voice match the enriched intent",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "synthesized_audio"


        if "seed_vc" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_seed_vc",
                tool_id="seed_vc",
                objective="convert speech to match the provided reference speaker while preserving content",
                input_artifacts=[current],
                output_artifact_role="converted_audio",
                validation_criteria=[
                    "converted speech resembles the reference speaker",
                    "spoken content is preserved",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "converted_audio"

        if "qwen3_tts" in selected_tools or (
            "tse" in selected_tools and any(t in selected_tools for t in ["qwen3_tts", "seed_vc"])
        ):
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_artifact_ops",
                tool_id="artifact_ops",
                objective="reconstruct the original scene by replacing only the edited regions",
                input_artifacts=[base_artifact_id, current],
                output_artifact_role="reconstructed_audio",
                validation_criteria=[
                    "unmodified regions remain unchanged",
                    "edited regions align with the original timing as much as possible",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "reconstructed_audio"

        if "edit_data_pipeline" in selected_tools:
            edit_operations = _edit_pipeline_operations(intent.clarified_goal)
            for operation_index, operation in enumerate(edit_operations):
                is_last_edit_operation = operation_index == len(edit_operations) - 1
                operation_label = _edit_operation_label(operation)
                output_role = (
                    "edit_pipeline_audio"
                    if is_last_edit_operation
                    else f"{operation_label}_edited_audio"
                )
                actions.append(ToolAction(
                    step_id=f"s{len(actions)+1}_edit_data_pipeline_{operation_label}",
                    tool_id="edit_data_pipeline",
                    objective=_edit_operation_objective(operation),
                    input_artifacts=[current],
                    output_artifact_role=output_role,
                    validation_criteria=_edit_operation_validation(operation),
                    depends_on=[actions[-1].step_id] if actions else [],
                ))
                current = output_role

        if "speech_enhancement" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_speech_enhancement",
                tool_id="speech_enhancement",
                objective="improve final speech clarity without changing requested voice/content edits",
                input_artifacts=[current],
                output_artifact_role="final_audio",
                validation_criteria=[
                    "speech is clearer than before enhancement",
                    "no obvious distortion or clipping is introduced",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "final_audio"

        if "spatial_reverb" in selected_tools:
            actions.append(ToolAction(
                step_id=f"s{len(actions)+1}_spatial_reverb",
                tool_id="spatial_reverb",
                objective="apply static room impulse response reverb matching the requested acoustic space",
                input_artifacts=[current],
                output_artifact_role="spatial_reverb_audio",
                validation_criteria=[
                    "speech content and speaker identity remain unchanged",
                    "perceived room size and reverberation match the request",
                    "no clipping or excessive intelligibility loss is introduced",
                ],
                depends_on=[actions[-1].step_id] if actions else [],
            ))
            current = "spatial_reverb_audio"

        return actions

    def create_tool_calls(
        self,
        actions: list[ToolAction],
        manuals: dict[str, ToolManual],
        state: dict[str, Any],
        artifact_paths: dict[str, str] | None = None,
    ) -> list[ToolCall]:
        calls: list[ToolCall] = []
        output_dir = state.get("output_dir") or str(runtime_artifact_path())
        base_audio = state.get("base_audio_path")
        refs = state.get("reference_audio_paths") or []
        prompt = state.get("user_prompt", "")
        source_text = state.get("source_text")
        perception_report = state.get("perception_report") or {}
        for action in actions:
            manual = manuals.get(action.tool_id)
            if manual is None:
                continue
            args = self._arguments_for_action(
                action,
                base_audio,
                refs,
                prompt,
                output_dir,
                artifact_paths or {},
                source_text,
                perception_report,
            )
            calls.append(ToolCall(
                step_id=action.step_id,
                tool_id=action.tool_id,
                mcp_tool_name=manual.mcp_tool_name,
                arguments=args,
            ))
        return calls

    def repair_tool_call(
        self,
        action: ToolAction,
        failed_arguments: dict[str, Any],
        state: dict[str, Any],
        artifact_paths: dict[str, str],
        error_message: str,
    ) -> dict[str, Any] | None:
        """Build a single error-informed retry call for known repairable failures."""

        if action.tool_id != "edit_data_pipeline":
            return None
        operation = str(failed_arguments.get("operation") or "")
        prompt = state.get("user_prompt", "")
        source_text = state.get("source_text")
        output_dir = state.get("output_dir") or str(runtime_artifact_path())
        output_path = _repair_output_path(
            failed_arguments.get("output_path") or f"{output_dir}/{action.step_id}.wav"
        )
        input_audio = failed_arguments.get("input_audio_path") or _artifact_ref(
            action.input_artifacts[0],
            state.get("base_audio_path"),
            artifact_paths,
        )

        if operation == "text_edit" and _is_text_edit_repairable(error_message):
            repaired = None
            if _needs_local_text_repair(error_message):
                repaired = build_local_text_edit_repair_arguments(
                    prompt,
                    source_text,
                    input_audio_path=input_audio,
                    output_path=output_path,
                    output_dir=output_dir,
                )
            if repaired is None:
                repaired = build_edit_data_pipeline_arguments(
                    "text_edit",
                    prompt,
                    source_text,
                    input_audio_path=input_audio,
                    output_path=output_path,
                    output_dir=output_dir,
                )
            if repaired is None:
                return None
            repaired["disable_prepend_context"] = True
            repaired["fail_if_cannot_prepend_context"] = False
            repaired["min_unmasked_context_sec"] = 0.0
            repaired["min_unmasked_context_ratio"] = 0.0
            repaired["synthesis_mode"] = "per_op"
            repaired["duration_mode"] = "estimate"
            repaired["expand_radius"] = 0.12
            repaired["merge_gap"] = 0.0
            repaired["debug"] = True
            return repaired

        if operation == "pitch_rate_edit" and _is_pitch_rate_repairable(error_message):
            repaired = build_edit_data_pipeline_arguments(
                "pitch_rate_edit",
                prompt,
                source_text,
                input_audio_path=state.get("base_audio_path") or input_audio,
                output_path=output_path,
                output_dir=output_dir,
            )
            if repaired is None:
                return None
            repaired["aligner_backend"] = "qwen3"
            return repaired

        return None

    def _arguments_for_action(
        self,
        action: ToolAction,
        base_audio: str,
        refs: list[str],
        prompt: str,
        output_dir: str,
        artifact_paths: dict[str, str] | None = None,
        source_text: str | None = None,
        perception_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_paths = artifact_paths or {}
        if action.tool_id == "seamless_expressive_s2st":
            expressive_requested = _wants_expressive_preservation(prompt)
            return {
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "target_language": _target_language_code(prompt) or "<missing_target_language>",
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "source_language": _source_language_code(prompt),
                "mode": "expressive" if expressive_requested else "m4t_v2",
                "preserve_vocal_style": _preserve_vocal_style(prompt),
                "preserve_speech_rate": _preserve_speech_rate(prompt),
                "preserve_pauses": _preserve_pauses(prompt),
                "allow_fallback_to_m4t": not expressive_requested,
                "duration_factor": 1.0,
            }
        if action.tool_id == "ace_step_music":
            lyrics = _music_lyrics(prompt, source_text)
            mode = _ace_step_mode(prompt, has_source=bool(base_audio))
            return {
                "instruction": prompt,
                "lyrics": lyrics,
                "music_prompt": _music_prompt(prompt),
                "source_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths) if mode in {"cover", "repaint"} else None,
                "reference_audio_path": refs[0] if refs else None,
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "mode": mode,
                "vocal_style": _music_vocal_style(prompt),
                "vocal_language": _language_hint(lyrics or prompt) if lyrics or prompt else "unknown",
                "duration_seconds": _music_duration_seconds(prompt),
                "audio_format": "wav",
                "repainting_start": 0.0,
                "repainting_end": -1.0,
                "audio_cover_strength": 0.7 if _music_vocal_style(prompt) == "rap" else 0.8,
                "inference_steps": 8,
                "thinking": mode == "text2music",
                "use_format": True,
            }
        if action.tool_id == "tse":
            reference_audio_path = refs[0] if refs else "<missing_reference_audio>"
            derived_reference = None
            if not refs:
                derived_reference = _speaker_reference_candidate(perception_report or {}, prompt)
                if derived_reference:
                    reference_audio_path = f"{output_dir}/{action.step_id}_reference.wav"
            return {
                "mixture_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "reference_audio_path": reference_audio_path,
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "enroll_duration": 5.0,
                **({"_x_editor_derived_reference": derived_reference} if derived_reference else {}),
            }
        if action.tool_id == "speech_separation":
            return {
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "output_dir": f"{output_dir}/{action.step_id}_stems",
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "model_family": "bs_roformer",
                "output_format": "WAV",
                "stem_policy": "vocals",
            }
        if action.tool_id == "qwen3_tts":
            if _wants_source_voice_clone_tts(prompt, []):
                return {
                    "text": _target_text(prompt, source_text),
                    "output_path": f"{output_dir}/{action.step_id}.wav",
                    "mode": "voice_clone",
                    "ref_audio_path": base_audio,
                    "language": "Auto",
                }
            return {
                "text": _target_text(prompt, source_text),
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "mode": "custom_voice",
                "speaker": _qwen3_tts_speaker(prompt),
                "instruct": _qwen3_tts_voice_prompt(prompt),
                "language": "Auto",
                **_qwen3_voice_design_backend_argument(),
            }
        if action.tool_id == "seed_vc":
            return {
                "source_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "reference_audio_path": refs[0] if refs else "<missing_reference_audio>",
                "output_path": f"{output_dir}/{action.step_id}.wav",
            }
        if action.tool_id == "intra_audio_voice_transfer":
            return {
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "instruction": prompt,
                "source_text": source_text or "",
                "perception_report": {},
            }
        if action.tool_id == "speech_enhancement":
            return {
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "mode": "clarity",
                "enhancement_goal": prompt,
            }
        if action.tool_id == "spatial_reverb":
            params = _spatial_reverb_parameters(prompt)
            return {
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "instruction": prompt,
                "output_dir": f"{output_dir}/{action.step_id}_artifacts",
                **params,
            }
        if action.tool_id == "audio_effects":
            preset = canonical_audio_effect_preset(prompt) or "telephone"
            args = {
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "instruction": prompt,
                "preset": preset,
                "strength": _audio_effect_strength(prompt),
                "preserve_duration": True,
                "output_dir": f"{output_dir}/{action.step_id}_artifacts",
            }
            if preset == "match_channel_reference":
                args["reference_audio_path"] = refs[0] if refs else "<missing_reference_audio>"
            return args
        if action.tool_id == "edit_data_pipeline":
            operation = _edit_operation_for_action(action, prompt)
            structured_args = build_edit_data_pipeline_arguments(
                operation,
                prompt,
                source_text,
                input_audio_path=_artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                output_path=f"{output_dir}/{action.step_id}.wav",
                output_dir=output_dir,
            )
            if structured_args is not None:
                return structured_args
            args: dict[str, Any] = {
                "operation": operation,
                "input_audio_path": _artifact_ref(action.input_artifacts[0], base_audio, artifact_paths),
                "output_path": f"{output_dir}/{action.step_id}.wav",
                "instruction": prompt,
            }
            if args["operation"] == "speech_rate":
                args.update({
                    "rate": _speech_rate_factor(prompt),
                    "method": "wsola+indextts2",
                    "speaker_audio_path": args["input_audio_path"],
                })
            elif args["operation"] == "text_edit":
                source = source_text or "<source_text_from_perception>"
                instruction_st, target = _text_edit_instruction_st(prompt, source)
                args.update({
                    "source_text": source,
                    "instruction_st": instruction_st,
                    "language": _language_hint(source),
                    "evaluate": False,
                    "target_text": target,
                })
            elif args["operation"] == "pitch_rate_edit":
                source = _source_text_for_operation(args["operation"], prompt, source_text)
                wants_pitch = _wants_pitch_edit(prompt)
                if wants_pitch:
                    semitones = _pitch_semitones(prompt)
                    instruction_st = f'<pitch, semitones={semitones:g}>{_xml_text(source)}</pitch>'
                else:
                    factor = _speech_rate_factor(prompt)
                    instruction_st = f'<rate, factor={factor:g}>{_xml_text(source)}</rate>'
                args.update({
                    "source_text": source,
                    "instruction_st": instruction_st,
                    "language": _language_hint(source),
                    "evaluate": False,
                    "pitch_method": "psola+indextts2_cco",
                    "smooth_policy": "never" if wants_pitch else "auto",
                })
            elif args["operation"] == "emotion_style_edit":
                source = _source_text_for_operation(args["operation"], prompt, source_text)
                emotion = _emotion_tag(prompt)
                args.update({
                    "source_text": source,
                    "instruction_st": f'<{emotion}, level=2>{_xml_text(source)}</{emotion}>',
                    "language": _language_hint(source),
                    "evaluate": False,
                    "persist": True,
                    "output_dir": f"{output_dir}/{action.step_id}_service_artifacts",
                })
            elif args["operation"] == "volume_normalize":
                args.update({
                    "target_db": _target_db(prompt),
                    "method": "rms",
                })
            elif args["operation"] == "pause_edit":
                source = _source_text_for_operation(args["operation"], prompt, source_text)
                args.update({
                    "source_text": source,
                    "instruction_st": _pause_instruction_st(prompt, source),
                    "language": _language_hint(source),
                    "evaluate": False,
                    "persist": True,
                    "output_dir": f"{output_dir}/{action.step_id}_service_artifacts",
                    "profile_id": 1,
                    "stage2_method": "method_one",
                    "expand_radius": 0.16,
                })
            elif args["operation"] == "pitch_shift":
                args.update({
                    "semitones": _pitch_semitones(prompt),
                    "method": "psola",
                })
            if source_text and args["operation"] in {"speech_rate", "pitch_shift"}:
                args["text"] = source_text
            return args
        return {}


def _contains(text: str, needles: list[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _mentions_phrase(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    for needle in needles:
        value = needle.lower()
        if re.search(r"[\u4e00-\u9fff]", value):
            if value in lowered:
                return True
        elif re.search(rf"(?<![a-z]){re.escape(value)}(?![a-z])", lowered):
            return True
    return False


def _mentions_female(text: str) -> bool:
    return _mentions_phrase(text, ["female", "woman", "women", "woman's", "girl", "female speaker", "女", "女声", "女性"])


def _mentions_male(text: str) -> bool:
    return _mentions_phrase(text, ["male", "man", "men", "man's", "boy", "male speaker", "男", "男声", "男性"])


def _intent_has_derived_speaker_reference(intent: EnrichedIntent) -> bool:
    return any(
        item.get("reference_source") == "perception_segments"
        for item in intent.target_objects
        if isinstance(item, dict)
    )


def _speaker_reference_candidate(perception_report: dict[str, Any], prompt: str) -> dict[str, Any] | None:
    segments = _perception_segments(perception_report)
    if not segments:
        return None

    lowered = prompt.lower()
    target_hints = _speaker_target_hints(lowered)
    ordinal = _speaker_ordinal(lowered)
    groups: dict[str, dict[str, Any]] = {}
    for segment in segments:
        speaker_id = str(segment.get("speaker_id") or segment.get("speaker") or "")
        if not speaker_id:
            continue
        start = segment.get("start_sec")
        end = segment.get("end_sec")
        if start is None or end is None or float(end) <= float(start):
            continue
        group = groups.setdefault(
            speaker_id,
            {
                "speaker_id": speaker_id,
                "segments": [],
                "duration": 0.0,
                "first_start": float(start),
                "score": 0,
            },
        )
        group["segments"].append(segment)
        group["duration"] += float(end) - float(start)
        group["first_start"] = min(group["first_start"], float(start))
        group["score"] += _speaker_segment_score(segment, target_hints, lowered)

    if not groups:
        return None
    candidates = sorted(groups.values(), key=lambda item: (-item["score"], item["first_start"], -item["duration"]))
    if target_hints and ordinal is not None:
        matching = [item for item in candidates if item["score"] > 0]
        if len(matching) >= ordinal:
            candidates = sorted(matching, key=lambda item: item["first_start"])
            chosen = candidates[ordinal - 1]
        elif all(item["score"] <= 0 for item in candidates) and len(candidates) >= ordinal:
            candidates = sorted(candidates, key=lambda item: item["first_start"])
            chosen = candidates[ordinal - 1]
        else:
            chosen = candidates[0]
    else:
        chosen = candidates[0]
        if target_hints and chosen["score"] <= 0:
            return None

    best_segment = max(chosen["segments"], key=lambda item: float(item["end_sec"]) - float(item["start_sec"]))
    return {
        "speaker_id": chosen["speaker_id"],
        "segment_id": best_segment.get("segment_id"),
        "start_sec": float(best_segment["start_sec"]),
        "end_sec": float(best_segment["end_sec"]),
        "target_hint": ", ".join(target_hints) if target_hints else "target speaker",
        "score": chosen["score"],
        "confidence": "medium" if chosen["score"] > 0 else "low",
        "selection_reason": "matched speaker descriptors" if chosen["score"] > 0 else "diarization ordinal fallback",
    }


def _perception_segments(perception_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = perception_report.get("segments") or perception_report.get("speaker_segments") or []
    speaker_profiles = _perception_speaker_profiles(perception_report)
    segments: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            continue
        segment = dict(item)
        segment.setdefault("segment_id", f"seg_{idx}")
        if "start" in segment and "start_sec" not in segment:
            segment["start_sec"] = segment["start"]
        if "end" in segment and "end_sec" not in segment:
            segment["end_sec"] = segment["end"]
        profile = speaker_profiles.get(str(segment.get("speaker_id") or segment.get("speaker") or ""))
        if profile:
            for key in ["perceived_gender", "description", "confidence"]:
                if profile.get(key) and not segment.get(key):
                    segment[key] = profile[key]
        segments.append(segment)
    return segments


def _perception_speaker_profiles(perception_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for item in perception_report.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        speaker_id = item.get("speaker_id") or item.get("speaker")
        if speaker_id is None:
            continue
        profiles[str(speaker_id)] = item
    return profiles


def _speaker_target_hints(lowered: str) -> list[str]:
    hints: list[str] = []
    if _mentions_female(lowered):
        hints.append("female")
    if _mentions_male(lowered):
        hints.append("male")
    mapping = [
        ("child", ["child", "kid", "boy", "girl", "baby", "toddler", "小孩", "孩子", "婴儿"]),
        ("old", ["old man", "old woman", "elderly", "老人"]),
        ("customer", ["customer", "客户", "顾客"]),
        ("robber", ["robber", "劫匪"]),
        ("victim", ["victim", "受害者"]),
        ("whisper", ["whisper", "whispered", "whispering", "耳语", "低语"]),
    ]
    for hint, needles in mapping:
        if _mentions_phrase(lowered, needles):
            hints.append(hint)
    return hints


def _speaker_ordinal(lowered: str) -> int | None:
    if any(value in lowered for value in ["second", "2nd", "第二"]):
        return 2
    if any(value in lowered for value in ["third", "3rd", "第三"]):
        return 3
    if any(value in lowered for value in ["first", "1st", "第一"]):
        return 1
    return None


def _speaker_segment_score(segment: dict[str, Any], target_hints: list[str], prompt: str) -> int:
    haystack = " ".join(
        str(segment.get(key) or "").lower()
        for key in ["perceived_gender", "description", "transcript", "speaker_id", "segment_id"]
    )
    score = 0
    for hint in target_hints:
        if hint in {"female", "male"} and hint in haystack:
            score += 4
        elif hint in haystack:
            score += 3
    if "whisper" in prompt and ("whisper" in haystack or "低语" in haystack or "耳语" in haystack):
        score += 3
    return score


def _wants_audio_effect_tool(text: str) -> bool:
    lowered = text.lower()
    if _contains(lowered, ["去混响", "移除混响", "remove reverb", "dereverb", "de-reverb"]):
        return False
    if _contains(
        lowered,
        [
            "not like it's over the phone",
            "not like it is over the phone",
            "more natural",
            "natural voice",
            "reduce the mechanical feel",
            "less mechanical",
            "remove telephone effect",
            "remove phone effect",
            "remove radio effect",
        ],
    ):
        return False
    if _wants_spatial_reverb(lowered):
        return False
    if re.search(
        r"\b(?:sound|sounds|sounding|speaking|spoken|coming|played|play)\b.{0,60}\b(?:over|through|from|on)\s+the\s+(?:phone|telephone)\b",
        lowered,
    ):
        return True
    if re.search(r"\b(?:phone|telephone)[ -]?(?:like|style|channel|effect)\b", lowered):
        return True
    return _contains(
        lowered,
        [
            "acoustic channel condition",
            "same acoustic channel",
            "same channel condition",
            "recorded in the same acoustic",
            "recorded with the same acoustic",
            "old-fashioned radio",
            "old fashioned radio",
            "radio effect",
            "phone call",
            "telephone",
            "underwater",
            "under water",
            "strong wind",
            "windy",
            "muffled",
            "heavy wall",
            "through a wall",
            "acoustic environment",
        ],
    )


def _requests_reference_channel_match(text: str) -> bool:
    lowered = text.lower()
    return _contains(
        lowered,
        [
            "acoustic channel condition",
            "same acoustic channel",
            "same channel condition",
            "recorded in the same acoustic",
            "recorded with the same acoustic",
            "match the channel",
            "match channel",
        ],
    )


def _audio_effect_strength(text: str) -> float:
    lowered = text.lower()
    if _contains(lowered, ["extremely", "very", "strong", "heavy", "severe", "明显", "很强", "非常"]):
        return 0.9
    if _contains(lowered, ["slightly", "subtle", "light", "mild", "一点", "轻微"]):
        return 0.45
    return 0.75


def _wants_speech_to_speech_translation(text: str) -> bool:
    lowered = text.lower()
    if _contains(lowered, ["text translation only", "transcript only", "字幕", "仅文本", "只要文字", "只要文本"]):
        return False
    wants_translation = _contains(
        lowered,
        [
            "翻译",
            "译成",
            "翻成",
            "转成",
            "translate",
            "translation",
            "speech-to-speech",
            "speech to speech",
            "s2st",
        ],
    )
    return wants_translation and _target_language_code(text) is not None


def _wants_accent_conversion(text: str) -> bool:
    lowered = text.lower()
    if _wants_audio_extraction(lowered):
        return False
    removes_accent = bool(
        re.search(r"\b(remove|delete|drop|lose|neutralize|get rid of)\b.{0,32}\baccent\b", lowered)
        or re.search(r"(去掉|删除|移除|消除).{0,12}口音", text)
    )
    if _wants_audio_removal(lowered) and not removes_accent:
        return False
    if _contains(lowered, ["pronunciation of", "correct the pronunciation", "swap the pronunciation", "clone the pronunciation"]):
        return False
    if _contains(lowered, ["accent unchanged", "keep her voice and accent unchanged", "keep the woman's voice and accent unchanged"]):
        return False
    has_target = resolve_requested_accent(text) is not None or _contains(
        lowered,
        ["same dialect as audio", "same accent as audio", "accent of audio", "dialect of audio"],
    )
    if not has_target:
        return False
    return _contains(
        lowered,
        [
            "accent",
            "dialect",
            "pronunciation",
            "native english",
            "native chinese",
            "standard english",
            "standard mandarin",
            "putonghua",
            "口音",
            "方言",
            "普通话",
            "北京话",
            "上海话",
            "苏州话",
            "宁波话",
            "粤语",
            "四川话",
            "闽南语",
        ],
    )


def _wants_expressive_preservation(text: str) -> bool:
    lowered = text.lower()
    if _negates_expressive_preservation(lowered):
        return False
    explicit = _contains(
        lowered,
        ["vocal style", "voice style", "speaking style", "prosody", "expressive", "same delivery"],
    )
    keep = _contains(lowered, ["保持", "保留", "尽量保持", "尽量保留", "preserve", "keep", "same"])
    expressive_dimension = _contains(
        lowered,
        [
            "说话风格",
            "风格",
            "语气",
            "音色",
            "语速",
            "停顿",
            "暂停",
            "节奏",
            "韵律",
            "vocal style",
            "voice style",
            "speaking style",
            "style",
            "speech rate",
            "speaking rate",
            "rate",
            "pause",
            "pauses",
            "rhythm",
            "delivery",
        ],
    )
    return explicit or (keep and expressive_dimension)


def _wants_singing_or_rap(text: str) -> bool:
    lowered = text.lower()
    if _contains(
        lowered,
        [
            "不要唱",
            "不用唱",
            "不唱",
            "不要说唱",
            "不用说唱",
            "not sing",
            "don't sing",
            "dont sing",
            "do not sing",
            "not rap",
            "don't rap",
            "dont rap",
            "do not rap",
        ],
    ):
        return False
    if _wants_existing_music_vocal_extraction(lowered):
        return False
    if _contains(
        lowered,
        [
            "唱歌",
            "唱出来",
            "唱一下",
            "唱一段",
            "唱成",
            "演唱",
            "歌唱",
            "合唱",
            "对唱",
            "说唱",
            "嘻哈",
            "hip hop",
            "hip-hop",
            "duet",
            "chorus",
        ],
    ):
        return True
    if re.search(r"\b(sing|singing|sung|song|rap|rapping|rapper)\b", lowered):
        return True
    return False


def _wants_existing_music_vocal_extraction(text: str) -> bool:
    extraction = _contains(
        text,
        [
            "extract",
            "isolate",
            "keep only",
            "only the",
            "remove everything else",
            "remove all else",
            "提取",
            "抽取",
            "分离",
            "只保留",
            "仅保留",
        ],
    )
    existing_vocal = _contains(
        text,
        [
            "singing voice",
            "singing vocal",
            "singing vocals",
            "singing",
            "unaccompanied singing",
            "a cappella",
            "acapella",
            "clean vocals",
            "sung voice",
            "singer",
            "vocals",
            "歌声",
            "演唱声",
            "唱歌声",
        ],
    )
    generation = bool(
        re.search(
            r"\b(make|turn|convert|transform|change)\b.+\b(to|into|as)\b.+\b(sing|singing|rap|rapping|song-like|music-vocal)\b",
            text,
        )
        or re.search(r"\bmake\b.+\b(sing|rap)\b", text)
        or _contains(text, ["改成唱", "变成唱", "唱成", "改成说唱", "变成说唱"])
    )
    return extraction and existing_vocal and not generation


def _negates_expressive_preservation(text: str) -> bool:
    return _contains(
        text,
        [
            "no need to preserve",
            "do not preserve",
            "don't preserve",
            "dont preserve",
            "without preserving",
            "无需保持",
            "无需保留",
            "不需要保持",
            "不需要保留",
            "不用保持",
            "不用保留",
            "不要保持",
            "不要保留",
        ],
    )


def _preserve_vocal_style(text: str) -> bool:
    if _negates_expressive_preservation(text.lower()):
        return False
    return _wants_expressive_preservation(text) or _contains(
        text.lower(),
        ["vocal style", "voice style", "speaking style", "说话风格", "音色", "语气", "风格"],
    )


def _preserve_speech_rate(text: str) -> bool:
    if _negates_expressive_preservation(text.lower()):
        return False
    return _wants_expressive_preservation(text) or _contains(
        text.lower(),
        ["speech rate", "speaking rate", "语速", "节奏", "rhythm", "pace"],
    )


def _preserve_pauses(text: str) -> bool:
    if _negates_expressive_preservation(text.lower()):
        return False
    return _wants_expressive_preservation(text) or _contains(
        text.lower(),
        ["pause", "pauses", "停顿", "暂停", "silence"],
    )


def _target_language_code(text: str) -> str | None:
    lowered = text.lower()
    explicit = re.search(r"(?:target(?: language)?|tgt_lang|目标语言)\s*[:=]\s*([a-zA-Z_]+)", lowered)
    if explicit:
        return LANGUAGE_NAME_TO_CODE.get(explicit.group(1).lower(), explicit.group(1))

    for name, code in sorted(LANGUAGE_NAME_TO_CODE.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(name.lower())
        if re.search(rf"(?:to|into|in)\s+{escaped}\b", lowered):
            return code
        if any(marker + name in text for marker in ["翻译成", "译成", "翻成", "转成", "变成", "到"]):
            return code

    # For prompts like "法语语音翻译" where the target is the only language
    # mentioned, use the sole language mention as a conservative fallback.
    mentioned = {
        code
        for name, code in LANGUAGE_NAME_TO_CODE.items()
        if (re.search(rf"\b{re.escape(name.lower())}\b", lowered) or name in text)
    }
    if len(mentioned) == 1:
        return next(iter(mentioned))
    return None


def _source_language_code(text: str) -> str | None:
    lowered = text.lower()
    explicit = re.search(r"(?:source(?: language)?|src_lang|源语言)\s*[:=]\s*([a-zA-Z_]+)", lowered)
    if explicit:
        return LANGUAGE_NAME_TO_CODE.get(explicit.group(1).lower(), explicit.group(1))
    for name, code in sorted(LANGUAGE_NAME_TO_CODE.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(name.lower())
        if re.search(rf"from\s+{escaped}\b", lowered):
            return code
        if any(marker + name in text for marker in ["从", "由"]):
            return code
    return None


def _contains_word(text: str, words: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word.lower())}\b", lowered) for word in words)


def _append_if_available(selected: list[str], tool_id: str, capabilities: dict[str, ToolCapability]) -> None:
    if tool_id in capabilities and tool_id not in selected:
        selected.append(tool_id)


def _append_runtime_tool(selected: list[str], tool_id: str) -> None:
    """Append a runtime utility that is not exposed as a catalog model tool."""

    if tool_id not in selected:
        selected.append(tool_id)


def _unsupported(message: str) -> dict[str, Any]:
    return {"planning_status": "UNSUPPORTED", "selected_tools": [], "error_message": message}


def _service_url(key: str, urls: dict[str, tuple[str, str]]) -> str:
    env_name, default = urls[key]
    return (os.getenv(env_name) or default).rstrip("/")


def _qwen3_tts_voice_clone_service_url() -> str:
    return (
        os.getenv("QWEN3_TTS_CLONE_SERVICE_URL")
        or os.getenv("QWEN3_TTS_VOICE_CLONE_SERVICE_URL")
        or os.getenv("QWEN3_TTS_SERVICE_URL")
        or "http://127.0.0.1:8019"
    ).rstrip("/")


def _probe_health_url(base_url: str, timeout: float) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code < 500 and response.status_code != 404


def _probe_qwen3_tts_voice_clone_url(base_url: str, timeout: float) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        if response.status_code >= 500 or response.status_code == 404:
            return False
        data = response.json()
    except (ValueError, requests.RequestException):
        return False
    return bool(data.get("base_loaded") and data.get("tokenizer_loaded"))


def _probe_seamless_s2st_url(base_url: str, timeout: float) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        if response.status_code >= 500 or response.status_code == 404:
            return False
        data = response.json()
    except (ValueError, requests.RequestException):
        return False
    availability = data.get("availability") or {}
    expressive_ready = bool(availability.get("expressivity_predict") and availability.get("expressive_model_dir_exists"))
    m4t_ready = bool(availability.get("m4t_predict"))
    return expressive_ready or m4t_ready


def _managed_service_bootstrap_possible(tool_id: str, arguments: dict[str, Any] | None = None) -> bool:
    """Return whether executor-side service bootstrap can reasonably handle this tool.

    Planner health checks run before ToolExecutor exists.  If a catalog tool
    declares a managed service with a concrete command, or with a command_env
    that is configured in the current environment, routing should not fail just
    because the HTTP endpoint is currently offline.  The executor will perform
    the authoritative bootstrap and smoke test.
    """

    if os.getenv("X_EDITOR_ENABLE_SERVICE_BOOTSTRAP", "1").lower() in {"0", "false", "no"}:
        return False
    if yaml is None:
        return False
    config_path = get_catalog_dir() / tool_id / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except OSError:
        return False
    if not isinstance(config, dict):
        return False
    for spec in config.get("managed_services", []) or []:
        if not isinstance(spec, dict) or not _managed_service_when_matches(spec.get("when"), arguments or {}):
            continue
        if spec.get("external_only"):
            continue
        if spec.get("external_tool"):
            try:
                require_external_service(spec["external_tool"])
            except ExternalToolSetupRequired:
                continue
        command = spec.get("command")
        if isinstance(command, str) and command.strip():
            return True
        if isinstance(command, list) and command:
            return True
        command_env = spec.get("command_env")
        if command_env and os.getenv(str(command_env)):
            return True
    return False


def _managed_service_when_matches(when: Any, arguments: dict[str, Any]) -> bool:
    return _matches_when(when, arguments)


def _wants_edit_data_pipeline(text: str) -> bool:
    return any(
        [
            _wants_text_edit(text),
            _wants_emotion_style_edit(text),
            _wants_rate_edit(text),
            _wants_volume_edit(text),
            _wants_pause_edit(text),
            _wants_pitch_edit(text),
            _wants_speaker_profile_param_edit(text, []),
        ]
    )


def _wants_text_edit(text: str) -> bool:
    lowered = text.lower()
    if _voice_transfer_request_without_text_edit_target(lowered):
        return False
    if _text_rewrite_pair(text):
        return True
    if _swap_text_request(text):
        return True
    if _contains(lowered, ["改字", "改文字", "改文本", "改内容", "改文案", "改台词"]):
        return True
    if _contains(
        lowered,
        [
            "replace the first sentence",
            "replace first sentence",
            "replace the last sentence",
            "modify the phrase",
            "modify phrase",
            "modify the pronunciation",
            "correct the pronunciation",
            "change the word",
            "change word",
            "replace the word",
            "replace word",
            "replace all",
            "change the number",
            "period of silence",
        ],
    ):
        return True
    if _contains(lowered, ["替换", "换成", "替换成", "重说成", "重说为"]) and not _wants_emotion_style_edit(lowered):
        return True
    if _contains(lowered, ["text edit", "rewrite text", "replace text", "change text", "change sentence", "modify text", "modify the word"]):
        return True
    if len(_quoted_strings(text)) >= 2 and _contains(lowered, ["replace", "change", "modify", "relace", "swap", "switch", "exchange"]):
        return True
    return False


def _voice_transfer_request_without_text_edit_target(text: str) -> bool:
    lowered = text.lower()
    if _contains(
        lowered,
        [
            "replace the word",
            "replace word",
            "change the word",
            "change word",
            "modify the phrase",
            "modify phrase",
            "replace text",
            "change text",
            "change sentence",
            "modify text",
            "pronunciation",
            "change the number",
        ],
    ):
        return False
    if _wants_intra_audio_voice_transfer(lowered, []):
        return True
    return _contains(
        lowered,
        ["timbre", "voice timbre", "speaker timbre", "speaker's timbre", "音色", "声纹"],
    ) and _contains(lowered, ["replace", "change", "convert", "transfer", "swap", "switch", "替换", "换成", "转换"])


def _wants_emotion_style_edit(text: str) -> bool:
    lowered = text.lower()
    if _contains(
        lowered,
        [
            "情绪",
            "语气",
            "风格",
            "开心",
            "高兴",
            "难过",
            "悲伤",
            "生气",
            "惊讶",
            "happy",
            "sad",
            "angry",
            "surprised",
            "excited",
            "expressive",
            "emotional",
            "urgent",
            "cheerful",
            "neutral",
            "calm",
            "hushed",
            "reflective",
            "introspective",
            "subdued",
            "internal monologue",
            "internal thought",
            "depressed",
            "reluctant",
            "frightened",
            "afraid",
            "fearful",
            "shout",
            "shouting",
            "yell",
            "yelling",
            "whisper",
            "whispering",
            "whispered",
            "大喊",
            "喊叫",
            "耳语",
            "低语",
        ],
    ):
        return True
    if _tone_or_style_is_only_preserved(lowered):
        return False
    return _contains_word(lowered, ["emotion", "mood", "tone", "style"])


def _wants_volume_edit(text: str) -> bool:
    return _contains(
        text.lower(),
        [
            "音量",
            "响度",
            "volume",
            "loudness",
            "normalize",
            "normalise",
            "louder",
            "quieter",
            "softer",
            "softly pronounce",
            "speak softly",
        ],
    )


def _wants_rate_edit(text: str) -> bool:
    return _contains(
        text.lower(),
        [
            "语速",
            "加速",
            "减速",
            "放慢",
            "变快",
            "变慢",
            "speech rate",
            "speaking rate",
            "speed",
            "tempo",
            "pace",
            "slow down",
            "speed up",
            "faster",
            "slower",
            "reduce the speaker’s speaking rate",
            "reduce the speaker's speaking rate",
            "increase the speaking rate",
        ],
    )


def _wants_pause_edit(text: str) -> bool:
    lowered = text.lower()
    if _contains(
        lowered,
        [
            "去掉停顿",
            "删除停顿",
            "移除停顿",
            "去除停顿",
            "remove pause",
            "delete pause",
            "remove the pause",
            "delete the pause",
            "remove pauses",
            "delete pauses",
            "remove the pauses",
            "delete the pauses",
            "remove silence",
            "delete silence",
            "remove the silence",
            "delete the silence",
        ],
    ):
        return False
    return _contains(
        lowered,
        [
            "停顿",
            "暂停",
            "停一下",
            "插入停顿",
            "加停顿",
            "pause",
            "insert silence",
            "add silence",
        ],
    )


def _wants_quality_cleanup(text: str) -> bool:
    return _contains(
        text.lower(),
        [
            "清晰",
            "干净",
            "降噪",
            "噪声",
            "音质",
            "clear",
            "clean",
            "denoise",
            "enhance",
            "more natural",
            "not like it's over the phone",
            "not like it is over the phone",
            "remove telephone effect",
            "remove phone effect",
            "less mechanical",
            "去混响",
            "移除混响",
            "remove reverb",
            "remove the reverb",
            "remove echo",
            "remove the echo",
            "remove reverb and echo",
            "remove the reverb and echo",
            "dereverb",
            "de-reverb",
        ],
    )


def _wants_phone_effect_cleanup(text: str) -> bool:
    lowered = text.lower()
    return _contains(
        lowered,
        [
            "not like it's over the phone",
            "not like it is over the phone",
            "remove telephone effect",
            "remove phone effect",
            "more natural",
            "less mechanical",
        ],
    )


def _wants_audio_extraction(text: str) -> bool:
    extraction_terms = [
        "提取",
        "抽取",
        "分离",
        "只保留",
        "仅保留",
        "截取",
        "extract",
        "isolate",
        "keep only",
        "speech portions",
        "spoken portions",
        "japanese-accented",
        "accented english",
    ]
    if _contains(text, extraction_terms):
        return True
    speaker_terms = ["第一说话人", "第二说话人", "first speaker", "second speaker", "target speaker"]
    return _contains(text, speaker_terms) and _contains(text, ["speech", "audio", "utterance", "话", "语音"])


def _wants_speech_separation(text: str) -> bool:
    lowered = text.lower()
    return _contains(
        lowered,
        [
            "多人说话分离",
            "多人语音分离",
            "说话人分离",
            "重叠说话",
            "重叠语音",
            "混合说话",
            "混合语音",
            "分离所有说话人",
            "separate speakers",
            "separate the speakers",
            "speaker separation",
            "speech separation",
            "overlapping speech",
            "overlapped speech",
            "mixed speech",
            "split speakers",
            "split the speakers",
            "separate all speakers",
        ],
    )


def _wants_audio_assembly(text: str) -> bool:
    if _wants_pause_edit(text):
        return False
    return _contains(
        text,
        [
            "拼接",
            "连接",
            "合并",
            "追加",
            "前面加",
            "后面加",
            "add the first audio",
            "before the second",
            "after the second",
            "concatenate",
            "append",
            "prepend",
            "merge audio",
            "stitch",
            "splice",
        ],
    )


def _wants_audio_removal(text: str) -> bool:
    if _wants_quality_cleanup(text) and _contains(text, ["噪声", "noise", "denoise"]):
        return False
    return _contains(
        text,
        [
            "去掉",
            "删除",
            "移除",
            "静音",
            "消除",
            "remove",
            "delete",
            "mute",
            "silence the",
            "cut out",
            "drop",
            "undo the",
        ],
    )


def _wants_profanity_mask(text: str) -> bool:
    return _contains(
        text,
        [
            "beep",
            "bleep",
            "curse word",
            "curse words",
            "profanity",
            "profane",
            "swear word",
            "swear words",
            "cuss word",
            "cuss words",
            "censor",
            "脏话",
            "粗口",
            "消音",
        ],
    )


def _wants_scene_or_background_change(text: str) -> bool:
    if _wants_quality_cleanup(text) and _contains(text, ["background noise", "背景噪声", "背景噪音"]):
        return False
    if _wants_quality_cleanup(text) and _contains(
        text.lower(),
        [
            "去混响",
            "移除混响",
            "remove reverb",
            "remove the reverb",
            "remove echo",
            "remove the echo",
            "remove reverb and echo",
            "remove the reverb and echo",
            "dereverb",
            "de-reverb",
        ],
    ):
        return False
    if _wants_spatial_reverb(text):
        return False
    return _contains(
        text,
        [
            "前景",
            "背景",
            "环境",
            "混响",
            "大厅",
            "foreground",
            "background",
            "wind effect",
            "room",
            "hall",
            "reverb",
            "reverberation",
            "echo",
            "underwater",
            "telephone effect",
            "large empty hall",
        ],
    )


def _wants_foreground_background_remix(text: str) -> bool:
    lowered = text.lower()
    if _wants_spatial_reverb(lowered):
        return False
    has_foreground_background_boundary = _contains(
        lowered,
        [
            "foreground",
            "background",
            "background music",
            "background sound",
            "music at its original speed",
            "music unchanged",
            "伴奏",
            "背景音乐",
            "背景声",
            "前景",
            "人声",
            "vocals",
            "vocal",
            "voice and only keep",
            "only keep the background",
            "keep the background",
            "keep background",
        ],
    )
    if not has_foreground_background_boundary:
        return False
    return any(
        [
            _wants_audio_removal(lowered),
            _wants_audio_extraction(lowered),
            _wants_edit_data_pipeline(lowered),
            _wants_quality_cleanup(lowered),
        ]
    )


def _wants_spatial_reverb(text: str) -> bool:
    lowered = text.lower()
    if _contains(
        lowered,
        [
            "去混响",
            "移除混响",
            "remove reverb",
            "remove the reverb",
            "remove echo",
            "remove the echo",
            "remove reverb and echo",
            "remove the reverb and echo",
            "dereverb",
            "de-reverb",
        ],
    ):
        return False
    if _contains(
        lowered,
        [
            "混响",
            "回声",
            "空间感",
            "空荡",
            "空旷",
            "大房间",
            "小房间",
            "大厅",
            "礼堂",
            "教堂",
            "浴室",
            "房间里",
            "空间里",
            "large empty room",
            "big empty room",
        ],
    ):
        return True
    return _contains_word(lowered, ["reverb", "reverberation", "echo", "room", "hall", "cathedral", "bathroom"])


def _wants_speaker_scoped_edit(text: str) -> bool:
    return _contains(
        text,
        [
            "第一说话人",
            "第二说话人",
            "目标说话人",
            "女说话",
            "男说话",
            "女声部分",
            "男声部分",
            "first speaker",
            "second speaker",
            "target speaker",
            "female speaker",
            "male speaker",
            "first man",
            "second man",
            "first woman",
            "second woman",
            "woman's",
            "man's",
            "customer",
            "customer's",
            "child's",
            "baby",
            "speaker while",
            "other speaker",
            "non-target speaker",
            "the whisper",
            "whispering speaker",
            "whispered speaker",
            "whispered voice",
            "whispering voice",
            "whispered part",
            "whispering part",
            "internal monologue",
            "internal monologues",
            "inner monologue",
            "inner monologues",
        ],
    )


def _wants_global_text_edit_across_speakers(text: str) -> bool:
    lowered = text.lower()
    if _contains(
        lowered,
        [
            "all speakers",
            "every speaker",
            "each speaker",
            "both speakers",
            "male and female speakers",
            "female and male speakers",
            "both male and female",
            "both female and male",
            "for both male and female",
            "for both female and male",
            "所有说话人",
            "全部说话人",
            "每个说话人",
        ],
    ):
        return True
    return _contains(lowered, ["for both", "for all", "for every", "for each"]) and _contains(
        lowered,
        ["speaker", "speakers", "voice", "voices"],
    )


def _wants_target_speaker_extraction(text: str, refs: list[str]) -> bool:
    if not refs:
        return False
    return _contains(
        text,
        [
            "目标说话人",
            "参考说话人",
            "target speaker",
            "reference speaker",
            "extract speaker",
            "isolate speaker",
        ],
    )


def _wants_reference_voice_conversion(text: str, refs: list[str]) -> bool:
    if not refs and not _contains(text, ["参考音频", "reference audio"]):
        return False
    return _contains(
        text,
        [
            "音色",
            "声纹",
            "说话人身份",
            "声音转换",
            "voice conversion",
            "speaker identity",
            "vocal timbre",
            "timbre",
            "sound like",
            "clone",
            "convert voice",
            "voice from",
            "with the voice",
            "using the voice",
            "voice of",
            "用声音",
            "用这个声音",
            "replace the speaker identity",
            "replace speaker identity",
        ],
    )


def _wants_intra_audio_voice_transfer(text: str, refs: list[str]) -> bool:
    if refs:
        return False
    lowered = text.lower()
    same_audio_terms = [
        "dialogue",
        "in this audio",
        "same audio",
        "inside the audio",
        "内部",
        "同一音频",
        "这段音频",
        "adult speaker",
        "child speaker",
        "adult voices",
        "child's voice",
        "all adult",
        "source speaker",
        "target speaker",
        "first speaker",
        "second speaker",
        "first speaking man",
        "second speaking man",
        "first speaking woman",
        "second speaking woman",
        "first man",
        "second man",
        "first woman",
        "second woman",
        "第一个说话人",
        "第二个说话人",
        "第一说话人",
        "第二说话人",
    ]
    if _contains(lowered, ["swap voices", "exchange voices", "voice swap", "互换声音", "交换声音", "互换音色", "交换音色"]):
        return _contains(lowered, same_audio_terms)
    return _contains(lowered, ["timbre", "音色", "声纹", "speaker timbre", "speaker's timbre", "voice timbre"]) and _contains(
        lowered,
        same_audio_terms,
    )


def _wants_no_ref_voice_synthesis(text: str, refs: list[str]) -> bool:
    """Return true for no-reference voice/timbre design requests."""

    if refs:
        return False
    if (
        _wants_audio_extraction(text)
        or _wants_audio_assembly(text)
        or _wants_audio_removal(text)
        or _wants_intra_audio_voice_transfer(text, refs)
        or _voice_is_only_preserved(text)
    ):
        return False
    has_voice_action = _contains(
        text,
        [
            "变成",
            "改成",
            "转换成",
            "换成",
            "make",
            "change",
            "convert",
            "transform",
            "turn",
            "sound like",
            "sound more like",
            "reduce",
            "make",
        ],
    ) or _contains_word(text, ["speak"])
    if not has_voice_action:
        return False
    return _contains(
        text,
        [
            "音色",
            "声线",
            "嗓音",
            "声音",
            "男声",
            "女声",
            "童声",
            "儿童声",
            "萝莉",
            "御姐",
            "老人声",
            "年轻",
            "低沉",
            "沙哑",
            "磁性",
            "timbre",
            "voice timbre",
            "voice",
            "vocal",
            "speaker identity",
            "male voice",
            "female voice",
            "child voice",
            "adult voice",
            "adult",
            "baby voice",
            "baby",
            "kid voice",
            "elderly voice",
            "old voice",
            "young voice",
            "younger",
            "deep voice",
            "raspy voice",
            "warm voice",
            "mechanical",
            "natural",
            "spoken by an adult",
        ],
    )


def _no_reference_voice_design_tool(capabilities: dict[str, ToolCapability]) -> str:
    return _preferred_voice_design_backend()


def _qwen3_voice_design_backend_argument() -> dict[str, str]:
    return {"voice_design_backend": _preferred_voice_design_backend()}


def _preferred_voice_design_backend() -> str:
    value = os.getenv("QWEN3_TTS_VOICE_DESIGN_BACKEND") or os.getenv("X_EDITOR_VOICE_DESIGN_BACKEND") or "qwen3_tts"
    value = value.strip().lower().replace("-", "_")
    if value not in {"qwen", "qwen3", "qwen3tts", "qwen3_tts", "auto"}:
        raise ValueError(f"Unsupported voice-design backend: {value!r}; install an extension separately")
    return "qwen3_tts"


def _wants_source_voice_clone_tts(text: str, refs: list[str]) -> bool:
    if refs:
        return False
    lowered = text.lower()
    if _wants_audio_extraction(lowered) or _wants_audio_removal(lowered):
        return False
    wants_generation = _contains(
        lowered,
        [
            "generate",
            "synthesize",
            "resynthesize",
            "re-synthesize",
            "tts",
            "say ",
            "saying",
            "speak ",
            "read ",
        ],
    )
    if not wants_generation:
        return False
    return _contains(
        lowered,
        [
            "preserve his original voice",
            "preserve her original voice",
            "preserve the original voice",
            "preserve the speaker's voice",
            "preserve speaker identity",
            "preserve his voice",
            "preserve her voice",
            "same voice",
            "original voice",
            "voice identity",
            "speaker identity",
            "clone the pronunciation",
            "clone pronunciation",
            "strictly clone",
        ],
    )


def _wants_speaker_profile_param_edit(text: str, refs: list[str]) -> bool:
    if refs:
        return False
    if (
        _wants_audio_extraction(text)
        or _wants_audio_assembly(text)
        or _wants_audio_removal(text)
        or _wants_intra_audio_voice_transfer(text, refs)
        or _voice_is_only_preserved(text)
    ):
        return False
    if not _contains(text, ["change", "convert", "transform", "replace", "modify", "make", "turn", "speak", "改", "变"]):
        return False
    return _contains(
        text,
        [
            "男声",
            "女声",
            "童声",
            "儿童声",
            "声音变",
            "male voice",
            "female voice",
            "child voice",
            "adult voice",
            "adult",
            "baby voice",
            "baby",
            "old voice",
            "elderly voice",
            "younger",
            "deep male",
            "gentle female",
            "deeper voice",
            "voice deeper",
            "lower voice",
            "voice lower",
            "higher voice",
            "voice higher",
            "voice to",
            "voice into",
            "voice with",
            "speaker sounds",
            "speaker sound",
        ],
    )


def _wants_textual_voice_synthesis(text: str, refs: list[str]) -> bool:
    return _wants_no_ref_voice_synthesis(text, refs)


def _voice_is_only_preserved(text: str) -> bool:
    return _contains(
        text,
        [
            "same voice",
            "original voice",
            "preserve his original voice",
            "preserve her original voice",
            "preserve the original voice",
            "preserving the original voice",
            "matching the same voice",
            "match the same voice",
            "voice unchanged",
        ],
    )


def _tone_or_style_is_only_preserved(text: str) -> bool:
    return _contains(
        text,
        [
            "same voice and tone",
            "same tone",
            "original tone",
            "matching the same voice and tone",
            "match the same voice and tone",
            "tone unchanged",
            "same style",
            "original style",
            "style unchanged",
        ],
    )


def _edit_pipeline_operations(prompt: str) -> list[str]:
    structured = build_structured_edit_sequence(prompt, "<source_text_from_perception>")
    structured_operations = [item.operation for item in structured if item.supported]
    if structured_operations:
        return structured_operations

    operations: list[str] = []
    if _wants_text_edit(prompt):
        operations.append("text_edit")
    if _wants_emotion_style_edit(prompt):
        operations.append("emotion_style_edit")
    if _wants_pause_edit(prompt):
        operations.append("pause_edit")
    if _wants_rate_edit(prompt) or _wants_pitch_edit(prompt):
        operations.append("pitch_rate_edit")
    if _wants_volume_edit(prompt):
        operations.append("volume_normalize")
    if not operations:
        operations.append(_edit_pipeline_operation(prompt))
    return operations


def _edit_operation_label(operation: str) -> str:
    return {
        "text_edit": "text",
        "emotion_style_edit": "emotion",
        "pitch_rate_edit": "pitch_rate",
        "volume_normalize": "volume",
        "speech_rate": "speech_rate",
        "pitch_shift": "pitch",
        "pause_edit": "pause",
        "boundary_smooth": "pause",
        "audio_assembly": "assembly",
    }.get(operation, operation)


def _edit_operation_for_action(action: ToolAction, prompt: str) -> str:
    step_id = action.step_id.lower()
    if step_id.endswith("_text"):
        return "text_edit"
    if step_id.endswith("_emotion"):
        return "emotion_style_edit"
    if step_id.endswith("_pitch_rate"):
        return "pitch_rate_edit"
    if step_id.endswith("_volume"):
        return "volume_normalize"
    if step_id.endswith("_pause"):
        return "pause_edit"
    return _edit_pipeline_operation(prompt)


def _edit_operation_objective(operation: str) -> str:
    return {
        "text_edit": "rewrite requested spoken text through the tagged text-edit pipeline",
        "emotion_style_edit": "apply requested emotion or speaking style to the current edited audio",
        "pitch_rate_edit": "apply requested pitch or speech-rate change through the pitch/rate pipeline",
        "pause_edit": "insert the requested pause through the pause edit pipeline",
        "volume_normalize": "normalize or adjust loudness",
    }.get(operation, "apply requested edit-data-pipeline operation")


def _edit_operation_validation(operation: str) -> list[str]:
    return {
        "text_edit": [
            "output speech says the requested target text",
            "speaker and unedited context are preserved as much as possible",
        ],
        "emotion_style_edit": [
            "output speech keeps the current text",
            "requested emotion or style is audible",
        ],
        "pitch_rate_edit": [
            "output reflects the requested pitch or speech-rate edit",
            "speech content remains recognizable",
        ],
        "volume_normalize": [
            "output loudness reflects the requested target level",
            "output avoids clipping",
        ],
        "pause_edit": [
            "output keeps the spoken text unchanged",
            "requested pause insertion is audible at the marked location",
        ],
    }.get(operation, ["output reflects the requested edit"])


def _artifact_operation_objective(prompt: str) -> str:
    text = prompt.lower()
    if _wants_audio_extraction(text):
        return "extract selected timeline spans into a new audio artifact"
    if _wants_speaker_scoped_edit(text) and any(
        [
            _wants_text_edit(prompt),
            _wants_emotion_style_edit(prompt),
            _wants_rate_edit(text),
            _wants_pitch_edit(text),
            _wants_volume_edit(text),
            _wants_pause_edit(text),
            _wants_quality_cleanup(text) and not _wants_phone_effect_cleanup(text),
        ]
    ):
        return (
            "speaker_scoped_local_edit: diarize/select target speaker turns, "
            "edit each single-speaker word-boundary segment through edit-data-pipeline, "
            "and reconstruct the original timeline"
        )
    if _wants_audio_assembly(text):
        return "concat timeline audio artifacts in the requested order"
    if _wants_profanity_mask(text):
        return "replace profanity or curse-word spans with beep tones while preserving duration"
    if _contains(text, ["mute", "silence the", "静音"]):
        return "mute selected timeline spans while preserving duration"
    if _wants_audio_removal(text):
        return "remove selected timeline spans and reconstruct the remaining audio"
    return "reconstruct or copy the current audio artifact"


def _artifact_operation_output_role(prompt: str) -> str:
    objective = _artifact_operation_objective(prompt)
    if objective.startswith("speaker_scoped_local_edit"):
        return "speaker_scoped_edit_audio"
    if objective.startswith("concat"):
        return "concatenated_audio"
    if objective.startswith("extract"):
        return "extracted_audio"
    if objective.startswith("replace profanity"):
        return "beeped_audio"
    if objective.startswith("mute"):
        return "muted_audio"
    if objective.startswith("remove"):
        return "removed_audio"
    return "artifact_audio"


def _artifact_operation_validation(prompt: str) -> list[str]:
    objective = _artifact_operation_objective(prompt)
    if objective.startswith("speaker_scoped_local_edit"):
        return [
            "only the requested speaker turns are edited",
            "each edit-data-pipeline segment is below 30 seconds and cut on word boundaries",
            "non-target speakers and unselected regions remain unchanged",
        ]
    if objective.startswith("extract"):
        return [
            "output contains the selected speech/audio spans",
            "unselected regions are absent from the output",
        ]
    if objective.startswith("remove") or objective.startswith("mute"):
        return [
            "requested regions are removed or muted",
            "remaining speech/audio is stitched without unresolved artifact references",
        ]
    if objective.startswith("replace profanity"):
        return [
            "curse/profanity words are replaced by beep tones",
            "non-target speech content and timing remain unchanged",
        ]
    if objective.startswith("concat"):
        return ["output audio contains input artifacts in the requested order"]
    return ["output audio artifact exists"]


def _edit_pipeline_operation(prompt: str) -> str:
    text = prompt.lower()
    if _wants_emotion_style_edit(text):
        return "emotion_style_edit"
    if _wants_text_edit(text):
        return "text_edit"
    if _wants_rate_edit(text):
        return "pitch_rate_edit"
    if _wants_volume_edit(text):
        return "volume_normalize"
    if _wants_pause_edit(text):
        return "pause_edit"
    if _wants_pitch_edit(text):
        return "pitch_rate_edit"
    return "pitch_rate_edit"


def _wants_pitch_edit(prompt: str) -> bool:
    return _contains(
        prompt.lower(),
        [
            "音高",
            "升调",
            "降调",
            "半音",
            "pitch",
            "semitone",
            "emphasis",
            "emphasize",
            "emphasise",
            "stress",
            "stressed",
            "intonation",
            "rising intonation",
            "falling intonation",
            "重音",
            "强调",
        ],
    )


def _speech_rate_factor(prompt: str) -> float:
    text = prompt.lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:x|倍)", text)
    if match:
        value = float(match.group(1))
        if 0.25 <= value <= 4.0:
            return value
    if _contains(text, ["慢", "减速", "放慢", "slower"]):
        return 0.8
    if _contains(text, ["快", "加速", "faster"]):
        return 1.2
    return 1.0


def _target_db(prompt: str) -> float:
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:db|dB|分贝)", prompt)
    if match:
        value = float(match.group(1))
        if -60.0 <= value <= 0.0:
            return value
    return -20.0


def _pitch_semitones(prompt: str) -> float:
    text = prompt.lower()
    match = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:半音|semitones?|st)", text)
    if match:
        value = float(match.group(1))
        if -24.0 <= value <= 24.0:
            return value
    if _contains(text, ["降低", "低", "降调", "lower", "down"]):
        return -3.0
    if _contains(text, ["升高", "高", "升调", "higher", "up"]):
        return 3.0
    return 0.0


def _emotion_tag(prompt: str) -> str:
    text = prompt.lower()
    if _contains(text, ["shout", "shouting", "yell", "yelling", "大喊", "喊叫"]):
        return "angry"
    if _contains(text, ["开心", "高兴", "happy", "excited"]):
        return "happy"
    if _contains(text, ["生气", "愤怒", "angry"]):
        return "angry"
    if _contains(text, ["悲伤", "难过", "sad"]):
        return "sad"
    if _contains(text, ["害怕", "恐惧", "afraid", "fear", "frightened", "fearful"]):
        return "afraid"
    if _contains(text, ["depressed", "reluctant", "melancholic"]):
        return "melancholic"
    if _contains(
        text,
        [
            "hushed",
            "reflective",
            "introspective",
            "subdued",
            "internal monologue",
            "inner monologue",
            "internal thought",
            "内心独白",
        ],
    ):
        return "melancholic"
    if _contains(text, ["惊讶", "surprised"]):
        return "surprised"
    if _contains(text, ["平静", "calm"]):
        return "calm"
    return "happy"


def _music_vocal_style(prompt: str) -> str:
    text = prompt.lower()
    if _contains(text, ["说唱", "嘻哈", "hip hop", "hip-hop"]) or re.search(r"\b(rap|rapping|rapper)\b", text):
        return "rap"
    if _contains(text, ["唱", "sing", "sung", "song"]):
        return "singing"
    return "music_vocal"


def _music_prompt(prompt: str) -> str:
    style = _music_vocal_style(prompt)
    if style == "rap":
        return "hip hop beat with confident rap vocals, clear rhythm, studio mix"
    if _contains(prompt.lower(), ["摇滚", "rock"]):
        return "rock song with expressive lead singing vocal, clear lyrics, studio mix"
    if _contains(prompt.lower(), ["民谣", "folk"]):
        return "folk song with warm lead singing vocal, acoustic arrangement, clear lyrics"
    return "short pop song with expressive lead singing vocal, clear lyrics, studio mix"


def _music_lyrics(prompt: str, source_text: str | None) -> str:
    lyrics = _target_text(prompt, source_text)
    if lyrics.startswith("<missing_"):
        return ""
    return lyrics


def _ace_step_mode(prompt: str, has_source: bool) -> str:
    text = prompt.lower()
    if has_source and _contains(text, ["repaint", "局部", "这一段", "片段", "选中部分", "selected region"]):
        return "repaint"
    if has_source and _contains(text, ["cover", "翻唱", "改编", "变成", "变为", "唱", "rap", "说唱"]):
        return "cover"
    return "text2music"


def _music_duration_seconds(prompt: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:秒|seconds?|sec|s)\b", prompt.lower())
    if match:
        value = float(match.group(1))
        if 10.0 <= value <= 600.0:
            return value
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:分钟|minutes?|min)\b", prompt.lower())
    if match:
        value = float(match.group(1)) * 60.0
        if 10.0 <= value <= 600.0:
            return value
    return 10.0


def _language_hint(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _xml_attr(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _target_hint(prompt: str) -> str:
    if _mentions_female(prompt):
        return "female speaker"
    if _mentions_male(prompt):
        return "male speaker"
    return "target speaker described by the user request"


def _qwen3_tts_speaker(prompt: str) -> str:
    target_text = _qwen3_tts_target_voice_text(prompt)
    if _voice_has(
        target_text,
        [
            "女声",
            "女性",
            "女人",
            "女孩",
            "female voice",
            "female voices",
            "female",
            "female speaker",
            "woman's voice",
            "woman voice",
            "woman",
            "girl's voice",
            "girl voice",
            "girl",
        ],
    ):
        return "vivian"
    if _voice_has(
        target_text,
        [
            "男声",
            "男性",
            "男人",
            "男孩",
            "低沉",
            "磁性",
            "male voice",
            "male voices",
            "male",
            "male speaker",
            "man's voice",
            "man voice",
            "man",
            "boy's voice",
            "boy voice",
            "boy's",
            "boy",
            "deep voice",
            "deep male",
        ],
    ):
        return "ryan"
    return "vivian"


def _qwen3_tts_voice_prompt(prompt: str) -> str:
    """Build an explicit voice-design prompt for Qwen3-TTS custom_voice mode."""

    text = _qwen3_tts_target_voice_text(prompt)
    gender_age = "adult neutral speaker"
    pitch_range = "natural conversational pitch range"
    timbre_texture = "natural, clean, intelligible timbre"
    emotion = "neutral and natural emotion"
    speaking_rate = "natural speaking rate"
    delivery = "fluent, stable delivery with clear pronunciation"

    if _voice_has(
        text,
        ["童声", "儿童声", "小孩", "孩子", "child voice", "kid voice", "little girl", "little boy", "girl's voice", "girl", "boy's voice", "boy voice", "boy's", "boy"],
    ) and not _negates_child_voice(text):
        if _voice_has(text, ["男孩", "little boy", "boy's voice", "boy voice", "boy's", "boy"]):
            gender_age = "child-like male speaker"
        elif _voice_has(text, ["女孩", "little girl", "girl's voice", "girl voice"]):
            gender_age = "child-like female speaker"
        else:
            gender_age = "child-like speaker"
        gender_age += "; specifically a 5- to 8-year-old prepubescent child, not a teenager and not an adult female or adult male"
        pitch_range = "very high child pitch range with lively but controlled variation, bright but not shrill"
        timbre_texture = "small vocal-tract, very light, bright child timbre; minimal chest resonance, no mature adult resonance"
        delivery = "fluent, stable delivery with clear pronunciation; playful child-like narration without screaming"
    elif _voice_has(text, ["萝莉"]):
        gender_age = "young female speaker"
        pitch_range = "higher pitch range, cute but natural"
        timbre_texture = "bright, sweet, youthful timbre"
    elif _voice_has(text, ["御姐"]):
        gender_age = "mature female speaker"
        pitch_range = "medium-low pitch range"
        timbre_texture = "smooth, confident, full female timbre"
    elif _voice_has(text, ["老人", "老年", "年老", "苍老", "elderly voice", "old voice", "older voice", "old", "elderly"]):
        if _voice_has(text, ["女声", "女性", "女人", "female voice", "female voices", "female speaker", "female", "woman's voice", "woman voice", "woman"]):
            gender_age = "elderly female speaker"
        elif _voice_has(text, ["男声", "男性", "男人", "male voice", "male voices", "male speaker", "male", "man's voice", "man voice", "man"]):
            gender_age = "elderly male speaker"
        else:
            gender_age = "elderly speaker"
        gender_age += "; specifically 80 years or older, not a middle-aged adult"
        pitch_range = "low, thin aged pitch range with slight tremor and measured movement"
        timbre_texture = (
            "aged, breathy, fragile timbre with weak breath support; "
            "mildly leaky or softened dental consonants as if some teeth are missing, while remaining intelligible"
        )
        speaking_rate = "slow, measured speaking rate"
        delivery = "careful elderly delivery with slight vocal tremor and clear enough pronunciation"
    elif _voice_has(text, ["中年", "middle-aged", "middle aged", "middle-aged voice"]):
        gender_age = "middle-aged adult speaker; specifically 40 to 55 years old"
        pitch_range = "moderate, steady adult pitch range"
        timbre_texture = "mature, stable timbre with natural resonance"
        delivery = "steady middle-aged adult delivery with clear pronunciation"
    elif _voice_has(text, ["女声", "女性", "女人", "female voice", "female voices", "female speaker", "female", "woman's voice", "woman voice", "woman"]):
        gender_age = "adult female speaker"
        pitch_range = "natural female pitch range"
        timbre_texture = "clear, smooth female timbre"
    elif _voice_has(text, ["男声", "男性", "男人", "male voice", "male voices", "male speaker", "male", "man's voice", "man voice", "man", "deep male"]):
        gender_age = "adult male speaker"
        pitch_range = "medium-low male pitch range"
        timbre_texture = "natural male timbre with chest resonance"

    if _voice_has(text, ["低沉", "更低", "低音", "deeper", "deep voice", "lower voice", "voice lower"]):
        pitch_range = "low pitch range with natural variation"
        timbre_texture = "warm, resonant, full-bodied timbre, not muffled"
    if _voice_has(text, ["更高", "尖一点", "higher voice", "voice higher"]):
        pitch_range = "higher pitch range while remaining natural and not shrill"
    if _voice_has(text, ["沙哑", "嘶哑", "raspy", "hoarse"]):
        timbre_texture = f"{timbre_texture}; slight raspy texture without hurting intelligibility"
    if _voice_has(text, ["磁性", "浑厚", "warm", "full", "rich"]):
        timbre_texture = "warm, rich, full-bodied timbre with clear articulation"
    if _voice_has(text, ["年轻", "younger", "young voice", "young"]):
        if "female speaker" in gender_age:
            gender_age = "young adult female speaker; specifically 18 to 25 years old, not a child"
        elif "male speaker" in gender_age:
            gender_age = "young adult male speaker; specifically 18 to 25 years old, not a child"
        else:
            gender_age = "young adult speaker; specifically 18 to 25 years old, not a child"
        timbre_texture = "energetic, bright young-adult timbre with clean articulation and stable breath"
    if _voice_has(text, ["温柔", "gentle", "soft"]):
        delivery = "gentle, soft delivery with clear pronunciation"
    if _voice_has(text, ["严肃", "serious"]):
        delivery = "serious and steady delivery with controlled energy"
    if _voice_has(text, ["开心", "高兴", "happy", "excited"]):
        emotion = "happy and expressive emotion"
    if _voice_has(text, ["生气", "愤怒", "angry", "anger"]):
        emotion = "angry emotion at high intensity while keeping pronunciation clear"
    if _voice_has(text, ["惊讶", "surprised"]):
        emotion = "surprised emotion at high intensity with lively pitch movement"
    if _voice_has(text, ["悲伤", "难过", "sad"]):
        emotion = "sad emotion, subdued and natural"
    if _voice_has(text, ["慢", "放慢", "slower", "slow down"]):
        speaking_rate = "slower speaking rate"
    if _voice_has(text, ["快", "加速", "faster", "speed up"]):
        speaking_rate = "faster speaking rate while keeping articulation clear"
    if _voice_has(text, ["phone", "telephone"]):
        timbre_texture = "telephone-like narrowband timbre while keeping speech intelligible"
    if _voice_has(text, ["electronic voice", "robotic voice", "natural voice", "natural"]):
        timbre_texture = "natural human voice without electronic or microphone coloration"

    return (
        "Voice design for Qwen3-TTS custom_voice. "
        f"Gender/age: {gender_age}. "
        f"Pitch/range: {pitch_range}. "
        f"Timbre texture: {timbre_texture}. "
        f"Emotion/intensity: {emotion}. "
        f"Speaking rate: {speaking_rate}. "
        f"Delivery: {delivery}. "
        "Content constraint: speak exactly the words provided in the text field; "
        "do not add, omit, translate, or paraphrase any words."
    )


def _spatial_reverb_parameters(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    if _contains(text, ["cathedral", "church", "教堂", "大教堂"]):
        return {"room_preset": "cathedral", "t60": 4.2, "dry_wet": 0.56}
    if _contains(text, ["hall", "大厅", "礼堂", "会堂", "auditorium"]):
        return {"room_preset": "hall", "t60": 2.4, "dry_wet": 0.48}
    if _contains(text, ["bathroom", "tile", "瓷砖", "浴室", "卫生间"]):
        return {"room_preset": "bathroom", "t60": 1.05, "dry_wet": 0.36}
    if _contains(text, ["large empty room", "big empty room", "empty room", "空荡", "空旷", "大房间", "很大的房间"]):
        return {"room_preset": "large_empty_room", "t60": 1.8, "dry_wet": 0.42}
    if _contains(text, ["small room", "小房间", "卧室", "bedroom"]):
        return {"room_preset": "small_room", "t60": 0.35, "dry_wet": 0.18}
    if _contains(text, ["subtle", "slight", "一点", "轻微"]):
        return {"room_preset": "medium_room", "t60": 0.55, "dry_wet": 0.16}
    return {"room_preset": "medium_room", "t60": 0.65, "dry_wet": 0.26}


def _qwen3_tts_target_voice_text(prompt: str) -> str:
    """Extract the target voice phrase instead of source attributes."""

    lowered = prompt.lower()
    markers = [
        "转换成",
        "转换为",
        "变成",
        "变为",
        "改成",
        "改为",
        "换成",
        "换为",
        "转成",
        "转为",
        " into ",
        " to ",
        " as ",
        " in ",
    ]
    best_index = -1
    best_marker = ""
    for marker in markers:
        index = lowered.rfind(marker)
        if index > best_index:
            best_index = index
            best_marker = marker
    if best_index >= 0:
        target = lowered[best_index + len(best_marker):].strip(" .。,:，;；")
        if target:
            return target
    return lowered


def _voice_has(text: str, needles: list[str]) -> bool:
    for needle in needles:
        lowered = needle.lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9' -]*", lowered):
            if re.search(rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])", text):
                return True
        elif lowered in text:
            return True
    return False


def _negates_child_voice(text: str) -> bool:
    lowered = text.lower()
    return any(
        pattern in lowered
        for pattern in [
            "不要童声",
            "不要变成童声",
            "不要童声化",
            "不是童声",
            "不是儿童",
            "不是小孩",
            "非童声",
            "非儿童",
            "not a child",
            "not child",
            "not childlike",
            "not child-like",
            "avoid child",
        ]
    )


def _target_text(prompt: str, source_text: str | None) -> str:
    rewrite = _text_rewrite_pair(prompt)
    if rewrite:
        old, new = rewrite
        if source_text and old in source_text:
            return source_text.replace(old, new, 1)
        return new
    for pattern in [r"“([^”]+)”", r"'([^']+)'", r'"([^"]+)"']:
        match = re.search(pattern, prompt)
        if match:
            return match.group(1)
    if source_text:
        return source_text
    return "<missing_source_text_from_perception>"


def _artifact_ref(artifact_id: str, base_audio: str, artifact_paths: dict[str, str] | None = None) -> str:
    if artifact_id == "input_audio":
        return base_audio
    if artifact_paths and artifact_id in artifact_paths:
        return artifact_paths[artifact_id]
    return f"<artifact:{artifact_id}>"


def _quoted_strings(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", text):
        values.append(next(group for group in match.groups() if group is not None))
    return values


def _text_rewrite_pair(prompt: str) -> tuple[str, str] | None:
    quoted = _quoted_strings(prompt)
    if len(quoted) >= 2 and _contains(
        prompt.lower(),
        ["改成", "替换成", "换成", "改为", "替换为", "换为", "replace", "change", "modify", "relace"],
    ):
        return quoted[0], quoted[1]
    lowered = prompt.lower()
    if len(quoted) == 1 and _contains(lowered, ["period of silence", "silence", "silent gap"]):
        return quoted[0], ""
    patterns = [
        r"\breplace\s+(?:all\s+)?(?:instances?\s+of\s+)?(?:the\s+word\s+)?(?P<old>[A-Za-z0-9_.+-]+)\s+with\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
        r"\bchange\s+(?:the\s+)?word\s+(?P<old>[A-Za-z0-9_.+-]+)\s+to\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
        r"\bchange\s+(?P<old>[A-Za-z0-9_.+-]+)\s+to\s+(?P<new>[A-Za-z0-9_.+-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            old = match.group("old").strip(".,;:!?")
            new = match.group("new").strip(".,;:!?")
            if old and new:
                return old, new
    return None


def _swap_text_request(prompt: str) -> bool:
    lowered = prompt.lower()
    if _contains(
        lowered,
        [
            "swap",
            "switch",
            "exchange",
            "swap order",
            "switch order",
            "change the order",
            "change order",
            "order in which",
            "交换",
            "互换",
            "调换",
            "对调",
            "顺序",
        ],
    ):
        return True
    return False


def _repair_output_path(output_path: str) -> str:
    if output_path.endswith(".wav"):
        return output_path[:-4] + "_repair.wav"
    return output_path + "_repair.wav"


def _is_text_edit_repairable(error_message: str) -> bool:
    lowered = error_message.lower()
    return _contains(
        lowered,
        [
            "text_edit service error",
            "prepend context",
            "f5 edit limit",
            "target timeline exceeds",
            "empty source span",
            "source span",
            "alignment",
        ],
    )


def _needs_local_text_repair(error_message: str) -> bool:
    lowered = error_message.lower()
    return _contains(lowered, ["target timeline exceeds", "f5 edit limit"])


def _is_pitch_rate_repairable(error_message: str) -> bool:
    lowered = error_message.lower()
    return _contains(
        lowered,
        [
            "pitch_rate_edit service error",
            "empty edit region",
            "0.000s -> 0.000s",
            "alignment",
        ],
    )


def _text_after_requested_rewrite(prompt: str, source_text: str) -> str:
    rewrite = _text_rewrite_pair(prompt)
    if not rewrite:
        return source_text
    old, new = rewrite
    if old in source_text:
        return source_text.replace(old, new, 1)
    return _target_text(prompt, source_text)


def _text_edit_instruction_st(prompt: str, source_text: str) -> tuple[str, str]:
    rewrite = _text_rewrite_pair(prompt)
    if rewrite:
        old, new = rewrite
        if old in source_text:
            tagged = source_text.replace(
                old,
                f'<sub targ="{_xml_attr(new)}">{_xml_text(old)}</sub>',
                1,
            )
            return tagged, source_text.replace(old, new, 1)
    target = _target_text(prompt, source_text)
    return f'<sub targ="{_xml_attr(target)}">{_xml_text(source_text)}</sub>', target


def _pause_instruction_st(prompt: str, source_text: str) -> str:
    level = _pause_level(prompt)
    tagged_pause = f'<pause act="ins" level="{level}"/>'
    quoted = _quoted_strings(prompt)
    for marker in quoted:
        index = source_text.find(marker)
        if index >= 0:
            return source_text[: index + len(marker)] + tagged_pause + source_text[index + len(marker) :]
    punctuation_indices = [source_text.find(value) for value in [",", "，", ";", "；"]]
    punctuation_indices = [index for index in punctuation_indices if 0 < index < len(source_text) - 1]
    if punctuation_indices:
        index = punctuation_indices[0] + 1
        return source_text[:index] + tagged_pause + source_text[index:]
    parts = source_text.split(" ", 1)
    if len(parts) == 2:
        return parts[0] + tagged_pause + " " + parts[1]
    index = max(1, len(source_text) // 2)
    return source_text[:index] + tagged_pause + source_text[index:]


def _pause_level(prompt: str) -> int:
    text = prompt.lower()
    match = re.search(r"level\s*=?\s*([1-3])|([1-3])\s*级", text)
    if match:
        return int(next(group for group in match.groups() if group))
    if _contains(text, ["长", "long"]):
        return 3
    if _contains(text, ["短", "short"]):
        return 1
    return 2


def _source_text_for_operation(operation: str, prompt: str, source_text: str | None) -> str:
    source = source_text or "<source_text_from_perception>"
    if operation != "text_edit" and _wants_text_edit(prompt):
        return _text_after_requested_rewrite(prompt, source)
    return source
