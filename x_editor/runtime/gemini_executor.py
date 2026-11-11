"""Gemini 2.5 Pro helper for strict executor tool-call normalization."""

from __future__ import annotations

import json
import os
import re
import wave
from html import unescape
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydub import AudioSegment

from x_editor.agents.base import GeminiClient
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key
from x_editor.core.schemas import AudioEditStep, ExecutorToolCall, ToolManual
from x_editor.core.text_equivalence import equivalent_transcript_copy
from x_editor.planner.tool_model_context import tool_model_capability_context
from x_editor.runtime.audio_references import resolve_audio_reference
from x_editor.runtime.executor_tool_call_planner import ExecutorToolCallPlannerError
from x_editor.runtime.operation_normalizers import canonicalize_language
from x_editor.workflow_control.policy import repair_action_is_enforced

_VOLUME_NORMALIZE_METHODS = {"rms", "peak"}


class GeminiExecutorError(ExecutorToolCallPlannerError):
    """Raised when Gemini executor output cannot be safely executed."""


class GeminiExecutorToolCallPlanner:
    """Normalize one ``AudioEditStep`` into a strict executor tool-call JSON.

    Gemini is used only for semantic normalization and parameter mapping. The
    returned call is always checked again by deterministic validators before the
    existing X-Editor runtime executes anything.
    """

    prompt_version = "x-editor.executor-tool-call/gemini-v2"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 16384,
        timeout: float = 60.0,
    ) -> None:
        config = AudioEditConfig()
        self.model_name = model_name or os.getenv("X_EDITOR_GEMINI_EXECUTOR_MODEL", "gemini-2.5-pro")
        self.api_key = resolve_gemini_api_key(api_key) or config.omni_api_key
        self.client = GeminiClient(
            api_key=self.api_key,
            base_url=base_url or config.omni_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def plan_tool_call(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        operation_spec: dict[str, Any],
        manual: ToolManual | None = None,
        verifier_error: dict[str, Any] | str | None = None,
    ) -> ExecutorToolCall:
        """Return a strict normalized call for one step."""

        if not self.api_key:
            raise GeminiExecutorError("Shared Gemini 2.5 Pro API key is required for Gemini executor normalization")

        data = self.client.call_json(
            _SYSTEM_PROMPT,
            _user_prompt(
                model_name=self.model_name,
                step=step,
                state=state,
                operation_spec=operation_spec,
                manual=manual,
                verifier_error=verifier_error,
            ),
        )
        if not isinstance(data, dict):
            raise GeminiExecutorError(f"Gemini executor expected a JSON object, got {type(data).__name__}")
        try:
            call = ExecutorToolCall(**data)
        except ValidationError as exc:
            raise GeminiExecutorError(f"Gemini executor returned invalid ExecutorToolCall schema: {exc}") from exc
        validate_executor_tool_call(
            call=call,
            step=step,
            state=state,
            operation_spec=operation_spec,
            manual=manual,
        )
        return call


def apply_executor_tool_call(
    step: AudioEditStep,
    call: ExecutorToolCall,
    state: dict[str, Any] | None = None,
) -> AudioEditStep:
    """Merge Gemini-normalized fields back into an ``AudioEditStep`` copy."""

    updated = step.model_copy(deep=True)
    updated.inputs.update(_filter_normalized_inputs(step.operation, call.normalized_inputs))
    updated.params.update(_filter_normalized_params(step.operation, call.normalized_params, state=state))
    return updated


def validate_executor_tool_call(
    *,
    call: ExecutorToolCall,
    step: AudioEditStep,
    state: dict[str, Any],
    operation_spec: dict[str, Any],
    manual: ToolManual | None = None,
) -> None:
    """Deterministically verify a Gemini-normalized executor tool call."""

    if call.step_id != step.id:
        raise GeminiExecutorError(f"Gemini executor step_id mismatch: {call.step_id!r} != {step.id!r}")
    if call.operation != step.operation:
        raise GeminiExecutorError(f"Gemini executor operation mismatch: {call.operation!r} != {step.operation!r}")
    expected_tool_id = operation_spec.get("tool_id")
    if expected_tool_id and call.tool_id and call.tool_id != expected_tool_id:
        raise GeminiExecutorError(f"Gemini executor tool_id mismatch: {call.tool_id!r} != {expected_tool_id!r}")
    if call.missing_inputs and call.allow_execute:
        raise GeminiExecutorError("Gemini executor marked missing_inputs but still allowed execution")
    if call.blocked_reason and call.allow_execute:
        raise GeminiExecutorError("Gemini executor marked blocked_reason but still allowed execution")
    if not call.allow_execute:
        return

    merged_inputs = dict(step.inputs)
    merged_inputs.update(_filter_normalized_inputs(step.operation, call.normalized_inputs))
    merged_params = dict(step.params)
    merged_params.update(_filter_normalized_params(step.operation, call.normalized_params, state=state))

    for key, value in {**merged_inputs, **merged_params}.items():
        if key in {"output_path", "debug_output_dir", "output_dir", "parquet_output_dir"}:
            continue
        if key.endswith("_audio_path") or key in {
            "audio_path",
            "source_audio_path",
            "target_audio_path",
            "reference_audio_path",
            "ref_audio_path",
            "mixture_audio_path",
        }:
            _validate_audio_path_value(key, value, state)

    operation = step.operation
    if operation.startswith("edit_data_pipeline."):
        _validate_edit_data_pipeline_call(
            operation=operation.split(".", 1)[1],
            inputs=merged_inputs,
            params=merged_params,
            arguments=call.arguments,
            state=state,
        )
    elif operation == "seed_vc":
        _require_existing_path(
            _first_value("reference_audio_path", merged_inputs, merged_params, call.arguments, state=state),
            "seed_vc reference_audio_path",
            state=state,
        )
    elif operation == "qwen3_tts":
        text = _first_value("text", merged_params, call.arguments, state=state) or state.get("source_text")
        if not isinstance(text, str) or not text.strip():
            raise GeminiExecutorError("qwen3_tts requires non-empty text")
        mode = _first_value("mode", merged_params, call.arguments, state=state) or "custom_voice"
        if mode == "custom_voice":
            instruct = _first_value("instruct", merged_params, call.arguments, state=state)
            if instruct is not None and not isinstance(instruct, str):
                raise GeminiExecutorError("qwen3_tts instruct must be a string when provided")
    elif operation == "seamless_expressive_s2st":
        target_language = _first_value(
            "target_language", merged_params, call.arguments, state=state
        )
        if not canonicalize_language(target_language):
            raise GeminiExecutorError(
                "seamless_expressive_s2st requires a valid target_language"
            )
    elif operation == "tse":
        _require_existing_path(
            _first_value("reference_audio_path", merged_inputs, merged_params, call.arguments, state=state),
            "tse reference_audio_path",
            state=state,
        )
    elif operation == "audio_effects":
        input_audio = (
            _first_value("input_audio_path", merged_inputs, merged_params, call.arguments, state=state)
            or state.get("current_audio_path")
            or state.get("input_audio")
        )
        if input_audio is not None and input_audio not in {"current", "<current>"}:
            _require_existing_path(input_audio, "audio_effects input_audio_path", state=state)
        preset = (
            merged_params.get("preset")
            or merged_params.get("effect")
            or call.arguments.get("preset")
            or call.arguments.get("effect")
        )
        if preset == "match_channel_reference":
            _require_existing_path(
                _first_value("reference_audio_path", merged_inputs, merged_params, call.arguments, state=state),
                "audio_effects reference_audio_path",
                state=state,
            )
    elif operation == "speech_separation":
        input_audio = _first_value("input_audio_path", merged_inputs, merged_params, call.arguments, state=state)
        if input_audio is not None and input_audio not in {"current", "<current>"}:
            _require_existing_path(input_audio, "speech_separation input_audio_path", state=state)
    elif operation == "foreground_background_remix":
        input_audio = (
            _first_value("input_audio_path", merged_inputs, merged_params, call.arguments, state=state)
            or state.get("current_audio_path")
            or state.get("input_audio")
        )
        if input_audio is not None and input_audio not in {"current", "<current>"}:
            _require_existing_path(input_audio, "foreground_background_remix input_audio_path", state=state)
        edit_operation = str(
            merged_params.get("edit_operation")
            or call.arguments.get("edit_operation")
            or ""
        ).removeprefix("edit_data_pipeline.")
        if edit_operation in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"}:
            source_text = (
                merged_params.get("source_text")
                or call.arguments.get("source_text")
                or state.get("source_text")
            )
            if not isinstance(source_text, str) or not source_text.strip():
                raise GeminiExecutorError(
                    f"foreground_background_remix edit_operation={edit_operation!r} requires source_text"
                )


_EDIT_PIPELINE_PROTECTED_NORMALIZED_PARAMS = {
    "operation",
    "input_audio_path",
    "audio_path",
    "source_audio_path",
    "output_path",
    "source_text",
    "instruction_st",
}


def _filter_normalized_inputs(operation: str, values: dict[str, Any]) -> dict[str, Any]:
    if not operation.startswith("edit_data_pipeline."):
        return dict(values)
    return {key: value for key, value in values.items() if key not in {"output_path", "operation"}}


def _filter_normalized_params(
    operation: str,
    values: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = (state or {}).get("context") if isinstance((state or {}).get("context"), dict) else {}
    if operation == "seamless_expressive_s2st" and repair_action_is_enforced(
        context, "repair_step_arguments"
    ):
        result = dict(values)
        target_language = (
            result.pop("target_language", None)
            or result.pop("target_lang", None)
            or result.pop("language", None)
        )
        if target_language is not None:
            code = canonicalize_language(target_language)
            if code:
                result["target_language"] = {
                    "zho": "cmn",
                    "fre": "fra",
                    "ger": "deu",
                }.get(code, code)
        return result
    if not operation.startswith("edit_data_pipeline."):
        return dict(values)
    result = {
        key: value
        for key, value in values.items()
        if key not in _EDIT_PIPELINE_PROTECTED_NORMALIZED_PARAMS
    }
    verified = _verified_edit_pipeline_instruction(operation, values, state=state)
    if verified:
        result.update(verified)
    elif _unsafe_failed_text_edit_instruction(operation, values, state=state):
        result.pop("target_text", None)
    if operation == "edit_data_pipeline.pitch_shift":
        method = result.get("method")
        if method is not None:
            result["method"] = _canonical_pitch_shift_method(method)
    return result


def _canonical_pitch_shift_method(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "psola_qwen3_codec": "psola+qwen3_codec",
        "psola_indextts2_cco": "psola+indextts2_cco",
        "psola_mimo_codec": "psola+mimo_codec",
        "psola+qwen3+codec": "psola+qwen3_codec",
        "psola+indextts2+cco": "psola+indextts2_cco",
        "psola+mimo+codec": "psola+mimo_codec",
    }
    canonical = aliases.get(text, text)
    if canonical in {"psola", "psola+qwen3_codec", "psola+indextts2_cco", "psola+mimo_codec"}:
        return canonical
    return "psola"


def _verified_edit_pipeline_instruction(
    operation: str,
    values: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_operation = operation.removeprefix("edit_data_pipeline.")
    if normalized_operation not in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"}:
        return {}
    instruction_st = values.get("instruction_st")
    if not isinstance(instruction_st, str) or not instruction_st.strip():
        return {}
    source_text = values.get("source_text") or (state or {}).get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        return {}
    if normalized_operation == "text_edit":
        rendered_source = _render_text_edit_source(instruction_st)
        if not equivalent_transcript_copy(rendered_source, source_text):
            return {}
        if not _has_text_edit_tag(instruction_st):
            return {}
        rendered_target = _render_text_edit_target(instruction_st)
        if not rendered_target.strip() or _compact_text(rendered_target) == _compact_text(source_text):
            return {}
        if _whole_source_sub_with_short_target(instruction_st, source_text, rendered_target):
            return {}
        return {
            "source_text": source_text,
            "instruction_st": instruction_st,
            "target_text": rendered_target,
        }
    stripped = _strip_instruction_tags(instruction_st)
    if not equivalent_transcript_copy(stripped, source_text):
        return {}
    return {
        "source_text": source_text,
        "instruction_st": instruction_st,
    }


def _unsafe_failed_text_edit_instruction(
    operation: str,
    values: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> bool:
    if operation != "edit_data_pipeline.text_edit":
        return False
    instruction_st = values.get("instruction_st")
    if not isinstance(instruction_st, str) or not instruction_st.strip():
        return False
    source_text = values.get("source_text") or (state or {}).get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        return False
    return _whole_source_sub_with_short_target(
        instruction_st,
        source_text,
        _render_text_edit_target(instruction_st),
    )


def _has_text_edit_tag(value: str) -> bool:
    return bool(
        re.search(r"<sub\s+[^>]*targ=", value)
        or re.search(r"<ins\b[^>]*>.*?</ins>", value, flags=re.DOTALL)
        or re.search(r"<del\b[^>]*>.*?</del>", value, flags=re.DOTALL)
    )


def _strip_instruction_tags(value: str) -> str:
    text = re.sub(r"<pause\s+[^>]*/>", "", value)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text)


def _render_text_edit_source(value: str) -> str:
    text = re.sub(r"<ins\b[^>]*>.*?</ins>", "", value, flags=re.DOTALL)
    return _strip_instruction_tags(text)


def _render_text_edit_target(value: str) -> str:
    rendered = re.sub(r"<sub\s+([^>]*)>(.*?)</sub>", _sub_target_replacement, value, flags=re.DOTALL)
    rendered = re.sub(r"<del\b[^>]*>.*?</del>", "", rendered, flags=re.DOTALL)
    return _strip_instruction_tags(rendered)


def _sub_target_replacement(match: re.Match[str]) -> str:
    attrs = match.group(1)
    target = _target_text_from_sub_attrs(attrs)
    return target if target is not None else unescape(match.group(2))


def _target_text_from_sub_attrs(attrs: str) -> str | None:
    match = re.search(r'targ="([^"]*)"', attrs) or re.search(r"targ='([^']*)'", attrs)
    return unescape(match.group(1)) if match else None


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _whole_source_sub_with_short_target(instruction_st: str, source_text: str, rendered_target: str) -> bool:
    matches = list(re.finditer(r"<sub\s+(?P<attrs>[^>]*)>(?P<body>.*?)</sub>", instruction_st, flags=re.DOTALL))
    if len(matches) != 1:
        return False
    body = _strip_instruction_tags(matches[0].group("body"))
    if _compact_text(body) != _compact_text(source_text):
        return False
    source_units = _text_units(source_text)
    target_units = _text_units(rendered_target)
    if source_units <= 1:
        return False
    return target_units <= 3 and target_units < max(4, source_units * 0.35)


def _text_units(value: str) -> int:
    cjk = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", value)
    if cjk:
        return len(cjk)
    return len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", value, flags=re.UNICODE))


def _validate_against_manual(call: ExecutorToolCall, manual: ToolManual) -> None:
    schema = manual.input_schema or {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        unknown = sorted(set(call.arguments) - set(properties))
        if unknown:
            raise GeminiExecutorError(
                f"Gemini executor produced unsupported argument(s) for {manual.tool_id}: {', '.join(unknown)}"
            )
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list):
        missing = [key for key in required if key not in call.arguments]
        if missing:
            raise GeminiExecutorError(
                f"Gemini executor arguments missing required field(s) for {manual.tool_id}: {', '.join(missing)}"
            )


def _validate_edit_data_pipeline_call(
    *,
    operation: str,
    inputs: dict[str, Any],
    params: dict[str, Any],
    arguments: dict[str, Any],
    state: dict[str, Any],
) -> None:
    effective = {}
    effective.update(inputs)
    effective.update(params)

    if effective.get("operation") not in {None, operation}:
        raise GeminiExecutorError(
            f"edit_data_pipeline operation mismatch: {effective.get('operation')!r} != {operation!r}"
        )

    input_audio = (
        effective.get("input_audio_path")
        or effective.get("audio_path")
        or state.get("current_audio_path")
        or state.get("input_audio")
    )
    if input_audio is not None and input_audio not in {"current", "<current>"}:
        input_audio = resolve_audio_reference(input_audio, state)
        _require_existing_path(input_audio, "edit_data_pipeline input_audio_path", state=state)

    if operation in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"}:
        source_text = effective.get("source_text") or state.get("source_text")
        instruction_st = effective.get("instruction_st")
        if not isinstance(source_text, str) or not source_text.strip():
            raise GeminiExecutorError(f"edit_data_pipeline.{operation} requires source_text")
        if instruction_st is not None and not isinstance(instruction_st, str):
            raise GeminiExecutorError(f"edit_data_pipeline.{operation} instruction_st must be a string")

    if operation == "text_edit":
        _validate_text_edit_length(input_audio, state)
    elif operation == "pitch_rate_edit":
        instruction_st = str(effective.get("instruction_st") or "")
        if "<pitch" in instruction_st and "<rate" not in instruction_st:
            if str(effective.get("smooth_policy") or "never") != "never":
                raise GeminiExecutorError("pitch-only edit must use smooth_policy='never' to preserve duration")
    elif operation == "speech_rate":
        method = str(effective.get("method") or "wsola+indextts2").lower()
        if method in {"indextts2", "wsola+indextts2", "auto"}:
            text = effective.get("text") or effective.get("source_text") or state.get("source_text")
            if not isinstance(text, str) or not text.strip():
                raise GeminiExecutorError(f"speech_rate method={method!r} requires text/source_text")
    elif operation == "volume_normalize":
        method = str(effective.get("method") or "rms").strip().lower().replace("_", "-")
        if method not in _VOLUME_NORMALIZE_METHODS:
            raise GeminiExecutorError(
                "volume_normalize method must be one of ['peak', 'rms']; "
                f"got {method!r}. Use edit_data_pipeline.speech_rate for WSOLA/IndexTTS2 rate methods."
            )
    elif operation == "pause_edit":
        instruction_st = str(effective.get("instruction_st") or "")
        if instruction_st and "<pause" not in instruction_st:
            raise GeminiExecutorError("pause_edit requires a <pause .../> tag in instruction_st")
    elif operation == "audio_assembly":
        if effective.get("target_text") and not effective.get("edit_tagged_text"):
            raise GeminiExecutorError(
                "audio_assembly does not accept target_text-only edits; use text_edit or provide edit_tagged_text"
            )
        if effective.get("target_audio_path") is not None:
            _require_existing_path(effective.get("target_audio_path"), "audio_assembly target_audio_path", state=state)


def _validate_text_edit_length(input_audio: Any, state: dict[str, Any]) -> None:
    if input_audio in {None, "current", "<current>"}:
        input_audio = state.get("current_audio_path") or state.get("input_audio")
    input_audio = resolve_audio_reference(input_audio, state)
    if not isinstance(input_audio, str) or not input_audio:
        raise GeminiExecutorError("text_edit requires input_audio_path")
    duration = _audio_duration_seconds(Path(input_audio))
    if duration is None:
        raise GeminiExecutorError(
            "text_edit input duration could not be verified; frontend must provide a <30s segment"
        )
    if duration < 30.0:
        return
    perception = state.get("perception_report") if isinstance(state.get("perception_report"), dict) else {}
    if not perception.get("word_timestamps"):
        raise GeminiExecutorError(
            "text_edit input is >=30s; executor requires word_timestamps for word-boundary segmentation"
        )
    if (
        _long_text_edit_requires_speaker_segments(perception, state.get("user_prompt"))
        and not perception.get("speaker_segments")
        and not perception.get("segments")
    ):
        raise GeminiExecutorError(
            "text_edit input is >=30s and appears multi-speaker or speaker/role scoped; "
            "executor requires speaker_segments so each chunk can preserve speaker boundaries"
        )


def _long_text_edit_requires_speaker_segments(perception: dict[str, Any], prompt: Any) -> bool:
    if _perception_has_multispeaker_evidence(perception):
        return True
    text = str(prompt or "").lower()
    return bool(
        re.search(
            r"\b(?:speaker|role|character|person|man|woman|male|female|boy|girl|child|kid|adult|mother|father)\b",
            text,
        )
        or re.search(r"(说话人|角色|男声|女声|男的|女的|小孩|儿童|成人|母亲|父亲)", text)
    )


def _perception_has_multispeaker_evidence(perception: dict[str, Any]) -> bool:
    speaker_ids: set[str] = set()
    for key in ("speaker_segments", "segments", "speakers"):
        raw = perception.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, dict):
                speaker_id = item.get("speaker_id") or item.get("speaker")
            else:
                speaker_id = item
            if speaker_id is not None and str(speaker_id).strip():
                speaker_ids.add(str(speaker_id))
    diarization = perception.get("diarization")
    if isinstance(diarization, dict):
        raw_speakers = diarization.get("speakers")
        if isinstance(raw_speakers, list):
            speaker_ids.update(str(item) for item in raw_speakers if str(item).strip())
        raw_segments = diarization.get("segments")
        if isinstance(raw_segments, list):
            for item in raw_segments:
                if isinstance(item, dict) and item.get("speaker_id") is not None:
                    speaker_ids.add(str(item["speaker_id"]))
    return len(speaker_ids) > 1


def _validate_audio_path_value(key: str, value: Any, state: dict[str, Any]) -> None:
    if value is None:
        return
    if isinstance(value, str) and value in {"", "current", "<current>"}:
        return
    if isinstance(value, list):
        for item in value:
            _validate_audio_path_value(key, item, state)
        return
    value = resolve_audio_reference(value, state)
    _require_existing_path(value, key, state=state)


def _require_existing_path(value: Any, label: str, *, state: dict[str, Any] | None = None) -> None:
    if state is not None:
        value = resolve_audio_reference(value, state)
    if not isinstance(value, str) or not value:
        raise GeminiExecutorError(f"{label} is required")
    path = Path(value)
    if not path.exists():
        raise GeminiExecutorError(f"{label} does not exist: {value}")


def _first_value(key: str, *dicts: dict[str, Any], state: dict[str, Any]) -> Any:
    for item in dicts:
        value = item.get(key)
        if value is not None and value != "":
            return value
    refs = state.get("reference_audio_paths") or []
    if key == "reference_audio_path" and refs:
        return refs[0]
    return None


def _audio_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            return frames / sample_rate if sample_rate else None
    except (wave.Error, OSError, EOFError):
        try:
            return len(AudioSegment.from_file(path)) / 1000.0
        except Exception:
            return None


def _user_prompt(
    *,
    model_name: str,
    step: AudioEditStep,
    state: dict[str, Any],
    operation_spec: dict[str, Any],
    manual: ToolManual | None,
    verifier_error: dict[str, Any] | str | None = None,
) -> str:
    argument_repair = repair_action_is_enforced(
        state.get("context") if isinstance(state.get("context"), dict) else {},
        "repair_step_arguments",
    ) and verifier_error is not None
    public_state = {
        "user_prompt": state.get("user_prompt"),
        "input_audio": state.get("input_audio"),
        "input_audios": state.get("input_audios"),
        "current_audio_path": state.get("current_audio_path"),
        "output_audio": state.get("output_audio"),
        "output_dir": state.get("output_dir"),
        "source_text": state.get("source_text"),
        "reference_audio_paths": state.get("reference_audio_paths"),
        "perception_report": _compact_perception(state.get("perception_report")),
        "previous_executor_normalization_error": state.get("last_executor_normalization_error"),
        "previous_executor_verification_error": state.get("last_executor_verification_error"),
    }
    if argument_repair:
        public_state["user_request"] = public_state["user_prompt"]
        public_state["argument_repair_verifier_error"] = verifier_error
    manual_json = manual.model_dump() if manual is not None else None
    repair_schema = ""
    if argument_repair:
        repair_schema = """
  "repair_resolution": "resolved",
  "argument_bindings": [
    {
      "field_path": "params.exact_field_name",
      "binding_kind": "semantic",
      "semantic_value": "meaning inferred from visible evidence",
      "proposed_value": "normalized value written to the field",
      "evidence": [{"source_ref": "user_request", "excerpt": "exact source excerpt"}],
      "alternatives": [],
      "rationale": "short field-specific rationale"
    }
  ],"""
    return f"""## Executor Model
{model_name}

## AudioEditStep JSON
{step.model_dump_json(indent=2)}

## Operation Spec JSON
{json.dumps(operation_spec, ensure_ascii=False, indent=2)}

## Tool Manual JSON
{json.dumps(manual_json, ensure_ascii=False, indent=2)}

## Tool/Model Capability Comparison
{tool_model_capability_context()}

## Runtime State JSON
{json.dumps(public_state, ensure_ascii=False, indent=2)}

Return only one JSON object matching this exact schema:
{{
  "step_id": "{step.id}",
  "operation": "{step.operation}",
  "tool_id": "tool id from Operation Spec, or null",
  "mcp_tool_name": "manual mcp_tool_name, or null",
  "arguments": {{}},
  "normalized_inputs": {{}},
  "normalized_params": {{}},
  "allow_execute": true,
  "missing_inputs": [],
  "blocked_reason": null,
{repair_schema}
  "validation_notes": []
}}
"""


def _compact_perception(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in [
        "language",
        "source_text",
        "speakers",
        "word_timestamps",
        "speaker_segments",
        "segments",
        "audio_quality",
        "background",
        "notable_events",
        "uncertainty",
        "services",
        "asr",
    ]:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, list) and len(item) > 8:
            result[key] = item[:8]
            result[f"{key}_truncated_count"] = len(item)
        else:
            result[key] = item
    return result


_SYSTEM_PROMPT = """You are the execution-layer tool-call normalizer for X-Editor.

Rules:
- Output ONLY one valid JSON object. No markdown fences or explanations.
- You do not execute audio processing.
- You must keep step_id and operation exactly equal to the provided AudioEditStep.
- Use only the tool_id and mcp_tool_name implied by the Operation Spec and Tool Manual.
- Treat arguments as trace/debug payload only unless the runtime explicitly asks for direct execution.
- Fill normalized_inputs and normalized_params in the shape consumed by AudioEditStep; the executor's
  deterministic operation normalizers will build final service payloads such as output_path and instruction_st.
- Do not invent unknown operation names, tool ids, manual fields, or hidden processors.
- If required inputs are missing, set allow_execute=false, list missing_inputs, and set blocked_reason.
- When Runtime State contains argument_repair_verifier_error, change only exact repairable_fields named there.
  Set repair_resolution to resolved only when one binding is unambiguous. For every changed field, return one
  argument_bindings item whose proposed_value exactly matches the normalized field value. Semantic bindings must
  cite an exact excerpt from user_request, step_description, perception, or source_text. If alternatives remain,
  set repair_resolution=ambiguous, list alternatives, set allow_execute=false, and do not propose a patch.
- For edit_data_pipeline service-layer operations, provide high-level semantic params when possible; do not rely on
  raw arguments to bypass the deterministic instruction_st converter.
- For edit_data_pipeline.text_edit, a verified tool instruction is:
  source_text plus instruction_st whose stripped source renders exactly to source_text, and whose target_text is
  the full sentence after applying <sub targ="...">old</sub>, <ins>...</ins>, and <del>...</del>. Do not set
  target_text to only the raw targ attribute value unless the whole utterance should truly become that short text.
- If you provide instruction_st for text_edit but cannot make it satisfy that contract, set allow_execute=false
  with a local blocked_reason instead of sending partial fields that may be misinterpreted.
- For the pipeline text_edit backend, the frontend segment must be shorter than 30 seconds unless segmentation data is available. The default dots model uses its own source-plus-output context budget.
- For translation fallback through text_edit, translate source_text into the target language yourself and provide
  target_text; never provide only target_language because the deterministic text_edit converter cannot translate.
- For pitch-only pitch_rate_edit, use instruction_st with <pitch, semitones=...>, use smooth_policy="never", and do not set rate/factor/speed.
- For edit_data_pipeline.emotion_style_edit, categorical emotion labels are fixed, but descriptive delivery
  styles are supported by the deterministic converter through <emo desc="...">. Do not block descriptive
  style requests such as public broadcast, storytelling, narration, audiobook, theatrical, reluctant,
  hushed, reflective, or conversational solely because they are not one of the fixed categorical emotions.
  Pass the natural-language instruction through and let the runtime build/validate instruction_st.
- For artifact_ops.speaker_scoped_edit with nested edit_data_pipeline.emotion_style_edit, the same descriptive
  style support applies. If the request targets all speech or the whole utterance, allow execution with the
  natural-language instruction instead of requiring a separate explicit speaker/span selector.
- For speech rate changes, prefer method="wsola+indextts2" and provide text/source_text.
- For foreground/background requests, use foreground_background_remix instead of directly editing the mixed audio.
- For perception.speech_context, keep the original instruction as the grounding prompt and allow execution when input_audio_path/current exists. This is the lightweight speech-context path for transcript-conditioned edits: Qwen3ASR first, alignment/diarization only when requested, Gemini only when explicitly allowed.
- For perception.audio_fallback, keep the original instruction as the grounding prompt and allow execution when input_audio_path/current exists. This is the compatibility alias for heavier framework-level perception fallback: specialized ASR/alignment/diarization first, Gemini 2.5 Pro audio when those are missing or insufficient.
- For perception.speaker_attributed_review, keep the original instruction as the grounding prompt. Use it for multi-speaker speech span grounding, speaker order, overlap, speaker questions/answers, or speaker-scoped utterance localization before artifact_ops.
- For perception.timestamped_caption or perception.flexsed_events, use them for general sound, environmental audio, music, acoustic effects, animal sounds, background noise, and non-speech/paralinguistic event boundaries. Do not use their coarse speech/male/female/question/overlap labels as the only evidence for exact transcript or speaker boundaries; require perception.speaker_attributed_review for those.
- If a step sequence uses both speech/speaker evidence and event/caption evidence, keep both results as perception evidence and let the subsequent artifact_ops step edit only the reconciled target spans.
- For reference clip or enrollment extraction, do not normalize the call into bare artifact_ops.extract_spans unless concrete timestamps are already present. Speech/speaker/voice reference clips require perception.speaker_attributed_review first; sound-event/acoustic reference clips require perception.timestamped_caption, with an optional concrete-event prior when useful. The following artifact_ops.extract_spans step should receive grounded selected_spans.
- For perception.gemini_audio, treat it as a compatibility alias that forces Gemini-backed fallback perception.
- For artifact_ops timeline extraction/removal/muting, do not block solely because a child/woman/boy/role/phrase detector is not a separate tool. If perception_report contains or can be enriched with source_text, speaker_segments, word_timestamps, or model segment descriptions, pass the natural-language instruction through; deterministic runtime selection will resolve spans.
- For synthetic silence/pause/gap creation, do not route through qwen3_tts. Use artifact_ops.silence or artifact_ops.concat with pause_duration_sec when those operations are listed.
- For no-reference voice design, qwen3_tts instruct must include gender/age, pitch range, timbre texture, emotion/intensity, speaking rate, and exact-word preservation.
- For reference voice conversion, prefer seed_vc and require reference_audio_path.
- For telephone/radio/underwater/wind/outdoor-background-noise/outdoor-ambience/crowd-background-noise/background-music/muffled/channel-match requests, use audio_effects when the operation is available. Outdoor ambience is represented by the strong_wind preset, crowd/people/audience background noise by the crowd_noise preset, and generic background music by the background_music preset, unless a more specific soundscape/music tool is available. Do not map these acoustic effects to speech_enhancement, spatial_reverb, seed_vc, or qwen3_tts unless the request also explicitly needs cleanup, room-only reverb, timbre transfer, or re-synthesis.
- For audio_effects.match_channel_reference, require reference_audio_path and prefer audio2/input:1 when the request context has two input audios.
- When the request context has multiple inputs, you may refer to them as input:0/input:1 (zero-based)
  or audio1/audio2 (human-readable); input:0/audio1 is the main audio and input:1/audio2 is the first reference.
"""
