"""Foreground/background separation, stem edit, and remix runtime adapter.

This adapter is deliberately thin. It does not implement a separation or speech
editing model itself; it orchestrates the existing catalog tools:

1. ``speech_separation`` for BS-RoFormer / MelBand-RoFormer stems.
2. ``edit_data_pipeline``, ``audio_effects``, or ``speech_enhancement`` for
   the selected foreground stem.
3. Local timeline-safe gain/fit/remix using pydub.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydub import AudioSegment
from x_editor.tools.catalog.audio_separator_roformer.time_axis import (
    declare_source_aligned_audio, source_aligned_audio_covers,
)
from x_editor.tools.catalog.audio_separator_roformer.stem_selection import select_stem

from x_editor.agents.base import GeminiClient
from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    OPEN_GLM_QWEN_OMNI_MODE,
    resolve_agentic_mode,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.perception import PerceptionRequirements, build_audio_perception_fallback
from x_editor.runtime.operation_normalizers import build_edit_data_pipeline_tool_arguments, canonicalize_language


ExecuteTool = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


async def execute_foreground_background_remix(
    *,
    execute_tool: ExecuteTool,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    instruction: str,
    source_text: str | None,
    params: dict[str, Any] | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run ``separate -> edit selected stem -> remix`` with existing tools."""

    params = dict(params or {})
    action = str(params.get("foreground_action") or _infer_foreground_action(instruction)).strip().lower()
    nested_operation, _ = _normalize_nested_edit_operation(
        params.get("edit_operation") if params.get("edit_operation") is not None else params.get("edit_op")
    )
    # The legacy top-level gain field is an explicit numeric operation contract,
    # independent of prose; keep it compatible without guessing a nested tool.
    explicit_gain = params.get("gain_db") is not None
    if action == "edit_foreground" and not nested_operation and not explicit_gain:
        raise ValueError(
            "foreground_background_remix edit_foreground requires an explicit edit_operation "
            "(or edit_op) and its operation-specific parameters; instruction text does not select a tool"
        )
    agentic_mode = resolve_agentic_mode(
        params.pop("_agentic_mode", None)
        or params.pop("agentic_mode", None)
        or (state.get("context") or {}).get("agentic_mode")
        or state.get("agentic_mode")
    )
    work_dir = Path(output_dir) / "foreground_background_remix"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_audio = Path(input_audio_path)
    if not input_audio.exists():
        raise FileNotFoundError(f"input audio not found: {input_audio}")

    separation_params = dict(params.pop("separation", {}) or {})
    requested_foreground_policy = str(params.get("foreground_stem_policy", "speech"))
    safe_stem_policy = _safe_separation_stem_policy(requested_foreground_policy)
    separation_result = await execute_tool(
        "speech_separation",
        {
            "input_audio_path": str(input_audio),
            "output_dir": str(work_dir / "stems"),
            "output_path": str(work_dir / "selected_foreground_raw.wav"),
            "model_family": separation_params.pop("model_family", params.pop("model_family", "bs_roformer")),
            "output_format": separation_params.pop("output_format", params.pop("output_format", "WAV")),
            "stem_policy": separation_params.pop("stem_policy", safe_stem_policy),
            **separation_params,
        },
        state,
    )
    stem_paths = _existing_paths(separation_result.get("stem_paths") or [])
    if not stem_paths:
        selected = separation_result.get("selected_stem_path") or separation_result.get("output_audio_path")
        stem_paths = _existing_paths([selected])
    if not stem_paths:
        raise RuntimeError("speech_separation did not return any usable stem_paths")

    foreground_policy = str(params.pop("foreground_stem_policy", requested_foreground_policy))
    background_policy = str(params.pop("background_stem_policy", "background"))
    foreground_path = _select_stem(stem_paths, foreground_policy, role="foreground")
    if foreground_path is None:
        raise RuntimeError("speech_separation returned stems, but no foreground stem could be selected")
    background_path = _select_stem(stem_paths, background_policy, role="background", exclude={foreground_path})
    if background_path is None:
        raise RuntimeError("speech_separation did not provide a distinct background stem")
    action = str(params.pop("foreground_action", "") or _infer_foreground_action(instruction)).strip().lower()
    if not action:
        action = "edit_foreground"

    original_audio = AudioSegment.from_file(str(input_audio))
    target_duration_ms = len(original_audio)

    foreground_after_edit = foreground_path
    edit_result: dict[str, Any] | None = None
    raw_edit_operation = params.pop("edit_operation", None)
    if raw_edit_operation is None:
        raw_edit_operation = params.pop("edit_op", None)
    edit_operation, nested_edit_params = _normalize_nested_edit_operation(raw_edit_operation)
    edit_params = dict(nested_edit_params)
    explicit_edit_params = params.pop("edit_params", None)
    if explicit_edit_params is None:
        explicit_edit_params = params.pop("edit_op_params", {})
    if isinstance(explicit_edit_params, dict):
        edit_params.update(explicit_edit_params)
    edit_operation = str(edit_operation or "").strip()
    scoped_selection = _scoped_selection_instruction(params, edit_params)
    if _should_use_scoped_artifact_edit(edit_operation, scoped_selection):
        original_edit_operation = edit_operation
        original_edit_params = dict(edit_params)
        edit_operation = "artifact_ops.speaker_scoped_edit"
        edit_params = {
            "selection_instruction": scoped_selection,
            "edit_op": original_edit_operation,
            "edit_op_params": original_edit_params,
            "preserve_timeline": True,
        }
    if edit_operation in {"", "copy", "edit_data_pipeline.volume_normalize"} and params.get("gain_db") is not None:
        edit_operation = "gain"
        edit_params.setdefault("gain_db", params.get("gain_db"))

    if action in {"remove_foreground", "remove_voice", "keep_background", "background_only"}:
        foreground_after_edit = None
    elif action in {"isolate_foreground", "foreground_only", "extract_foreground"}:
        foreground_after_edit = foreground_path
    else:
        foreground_after_edit, edit_result = await _edit_foreground_stem(
            execute_tool=execute_tool,
            foreground_path=foreground_path,
            output_path=work_dir / "foreground_edited.wav",
            instruction=instruction,
            source_text=source_text,
            edit_operation=edit_operation,
            edit_params=edit_params,
            state=state,
            agentic_mode=agentic_mode,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preserve_duration = bool(params.pop("preserve_duration", True))
    final_audio = _render_final_audio(
        original_audio=original_audio,
        foreground_path=foreground_after_edit,
        background_path=background_path,
        isolate_foreground=action in {"isolate_foreground", "foreground_only", "extract_foreground"},
        foreground_gain_db=_optional_float(params.pop("foreground_gain_db", None)),
        background_gain_db=_optional_float(params.pop("background_gain_db", None)),
        target_duration_ms=target_duration_ms,
        preserve_duration=preserve_duration,
    )
    final_audio.export(output, format="wav")

    post_enhance = bool(params.pop("post_enhance", False))
    post_enhance_result: dict[str, Any] | None = None
    if post_enhance:
        enhanced_path = work_dir / "post_enhanced.wav"
        post_enhance_result = await execute_tool(
            "speech_enhancement",
            {
                "input_audio_path": str(output),
                "output_path": str(enhanced_path),
                "mode": "clarity",
                "preserve_background": True,
                "enhancement_goal": instruction,
            },
            state,
        )
        enhanced_output = post_enhance_result.get("output_audio_path")
        if isinstance(enhanced_output, str) and Path(enhanced_output).exists():
            shutil.copy2(enhanced_output, output)

    result = {
        "output_audio_path": str(output),
        "operation": "foreground_background_remix",
        "tool_chain": _tool_chain_for_result(edit_operation, edit_result, post_enhance=post_enhance),
        "input_audio_path": str(input_audio),
        "stem_paths": [str(path) for path in stem_paths],
        "foreground_stem_path": str(foreground_path),
        "background_stem_path": str(background_path) if background_path else None,
        "foreground_action": action,
        "edit_operation": edit_operation or None,
        "edited_foreground_path": str(foreground_after_edit) if foreground_after_edit else None,
        "edit_result": edit_result,
        "post_enhance_result": post_enhance_result,
        "preserve_duration": preserve_duration,
        "duration_ms": len(AudioSegment.from_file(output)),
    }
    if (
        preserve_duration
        and edit_result is None
        and not post_enhance
        and source_aligned_audio_covers(
            separation_result.get("source_time_axis"), input_audio, stem_paths,
        )
    ):
        receipt = declare_source_aligned_audio(
            input_audio, [output],
            basis="source-aligned separation followed by zero-origin stem selection/mixing",
        )
        if receipt is not None:
            result["source_time_axis"] = receipt
    return result


async def _edit_foreground_stem(
    *,
    execute_tool: ExecuteTool,
    foreground_path: Path,
    output_path: Path,
    instruction: str,
    source_text: str | None,
    edit_operation: str,
    edit_params: dict[str, Any],
    state: dict[str, Any],
    agentic_mode: str,
) -> tuple[Path, dict[str, Any] | None]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = edit_operation.removeprefix("edit_data_pipeline.").strip()
    if normalized in {"", "none", "noop", "copy"}:
        shutil.copy2(foreground_path, output_path)
        return output_path, {"operation": "copy", "output_audio_path": str(output_path)}

    if normalized in {"speech_enhancement", "enhance", "denoise", "clearvoice"}:
        result = await execute_tool(
            "speech_enhancement",
            {
                "input_audio_path": str(foreground_path),
                "output_path": str(output_path),
                "mode": edit_params.pop("mode", "clarity"),
                "preserve_background": False,
                "enhancement_goal": instruction,
                **edit_params,
            },
            state,
        )
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {
        "audio_effects",
        "audio_effect",
        "telephone",
        "phone",
        "phone_call",
        "old_radio",
        "radio",
        "underwater",
        "strong_wind",
        "muffled_wall",
        "whisper_style",
        "shouting_style",
        "match_channel_reference",
    }:
        preset = edit_params.pop("preset", None) or edit_params.pop("effect", None)
        if not preset and normalized not in {"audio_effects", "audio_effect"}:
            preset = normalized
        result = await execute_tool(
            "audio_effects",
            {
                "input_audio_path": str(foreground_path),
                "output_path": str(output_path),
                "instruction": instruction,
                "preset": preset,
                "strength": float(edit_params.pop("strength", 0.75)),
                "preserve_duration": bool(edit_params.pop("preserve_duration", True)),
                **edit_params,
            },
            state,
        )
        result.setdefault("foreground_tool_chain", ["audio_effects"])
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {"seamless_expressive_s2st", "seamless_s2st", "s2st", "speech_to_speech_translation"}:
        target_language = (
            edit_params.pop("target_language", None)
            or edit_params.pop("target_lang", None)
            or edit_params.pop("language", None)
        )
        if not target_language:
            raise RuntimeError("foreground_background_remix seamless_expressive_s2st edit requires target_language")
        source_language = edit_params.pop("source_language", None) or edit_params.pop("source_lang", None)
        target_text = edit_params.pop("target_text", None) or edit_params.pop("text", None)
        fallback_source_text = edit_params.pop("source_text", None) or source_text or state.get("source_text")
        seamless_args = {
            "input_audio_path": str(foreground_path),
            "output_path": str(output_path),
            "target_language": str(target_language),
            "source_language": source_language,
            "mode": edit_params.pop("mode", "auto"),
            "preserve_vocal_style": bool(edit_params.pop("preserve_vocal_style", True)),
            "preserve_speech_rate": bool(edit_params.pop("preserve_speech_rate", True)),
            "preserve_pauses": bool(edit_params.pop("preserve_pauses", True)),
            "allow_fallback_to_m4t": bool(edit_params.pop("allow_fallback_to_m4t", False)),
            "duration_factor": float(edit_params.pop("duration_factor", 1.0)),
            **edit_params,
        }
        if source_language is None:
            seamless_args.pop("source_language", None)
        try:
            result = await execute_tool("seamless_expressive_s2st", seamless_args, state)
        except RuntimeError as exc:
            if not _seamless_error_allows_text_tts_fallback(exc):
                raise
            result = await _fallback_seamless_to_qwen_voice_clone(
                execute_tool=execute_tool,
                foreground_path=foreground_path,
                output_path=output_path,
                instruction=instruction,
                source_text=fallback_source_text,
                target_text=target_text,
                target_language=str(target_language),
                source_language=str(source_language) if source_language else None,
                unavailable_error=exc,
                state=state,
                agentic_mode=agentic_mode,
            )
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {"gain", "volume_gain"}:
        audio = AudioSegment.from_file(str(foreground_path))
        audio = audio + float(edit_params.get("gain_db", 0.0))
        audio.export(output_path, format="wav")
        return output_path, {"operation": "gain", "gain_db": float(edit_params.get("gain_db", 0.0)), "output_audio_path": str(output_path)}

    if normalized in {"artifact_ops.speaker_scoped_edit", "speaker_scoped_edit"}:
        from x_editor.runtime.artifact_ops import execute_artifact_step

        result = execute_artifact_step(
            operation="speaker_scoped_edit",
            instruction=instruction,
            input_audio_path=str(foreground_path),
            output_path=str(output_path),
            output_dir=str(output_path.parent / "artifact_ops"),
            perception_report=state.get("perception_report") or {},
            source_text=source_text or state.get("source_text"),
            reference_audio_paths=state.get("reference_audio_paths") or [],
            params=edit_params,
        )
        return Path(result.get("output_audio_path") or output_path), result

    artifact_operation = normalized.removeprefix("artifact_ops.").strip()
    if artifact_operation in {"extract_spans", "remove_spans", "mute_spans", "beep_spans", "reorder_spans"}:
        from x_editor.runtime.artifact_ops import execute_artifact_step

        result = execute_artifact_step(
            operation=artifact_operation,
            instruction=instruction,
            input_audio_path=str(foreground_path),
            output_path=str(output_path),
            output_dir=str(output_path.parent / "artifact_ops"),
            perception_report=state.get("perception_report") or {},
            source_text=source_text or state.get("source_text"),
            reference_audio_paths=state.get("reference_audio_paths") or [],
            params=edit_params,
        )
        result.setdefault("foreground_tool_chain", [f"artifact_ops.{artifact_operation}"])
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {"qwen3_tts", "qwen_tts", "tts", "voice_clone"}:
        text = edit_params.pop("text", None) or edit_params.pop("target_text", None) or source_text
        if not text:
            raise RuntimeError("foreground_background_remix qwen3_tts edit requires text or target_text")
        style_text = edit_params.pop("style_text", None)
        edit_params.pop("instruction_st", None)
        edit_params.pop("operation", None)
        result = await execute_tool(
            "qwen3_tts",
            {
                "text": str(text),
                "output_path": str(output_path),
                "mode": edit_params.pop("mode", "voice_clone"),
                "ref_audio_path": str(foreground_path),
                "language": edit_params.pop("language", "Auto"),
                "instruct": str(style_text or instruction),
                "x_vector_only_mode": bool(edit_params.pop("x_vector_only_mode", True)),
                **edit_params,
            },
            state,
        )
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {"accent_conversion", "dialect_transfer", "dialect_conversion", "accent_transfer"}:
        from x_editor.runtime.accent_conversion import execute_accent_conversion

        target_accent = (
            edit_params.get("target_accent")
            or edit_params.get("accent")
            or edit_params.get("dialect")
            or edit_params.get("target_dialect")
        )
        accent_instruction = str(
            edit_params.pop("instruction", None)
            or edit_params.pop("edit_instruction", None)
            or instruction
        )
        if target_accent and str(target_accent).lower() not in accent_instruction.lower():
            accent_instruction = f"{accent_instruction} Target accent/dialect: {target_accent}."
        reference_audio = (
            edit_params.pop("reference_audio_path", None)
            or edit_params.pop("reference_audio", None)
            or edit_params.pop("ref_audio_path", None)
        )
        reference_perception = edit_params.pop("reference_perception_report", None)
        accent_source_text = str(edit_params.pop("source_text", None) or source_text or state.get("source_text") or "")

        async def execute_existing_tool(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return await execute_tool(tool_id, arguments, state)

        result = await execute_accent_conversion(
            execute_tool=execute_existing_tool,
            input_audio_path=str(foreground_path),
            output_path=str(output_path),
            output_dir=str(output_path.parent / "accent_conversion"),
            instruction=accent_instruction,
            source_text=accent_source_text,
            perception_report=state.get("perception_report") or {},
            reference_audio_path=str(reference_audio) if reference_audio else None,
            reference_perception_report=reference_perception,
            agentic_mode=agentic_mode,
        )
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {"seed_vc", "voice_conversion", "voice_clone_conversion", "timbre_transfer", "timber_transfer"}:
        reference_audio = (
            edit_params.pop("reference_audio_path", None)
            or edit_params.pop("reference_audio", None)
            or edit_params.pop("ref_audio_path", None)
            or edit_params.pop("speaker_audio_path", None)
        )
        if not reference_audio:
            raise RuntimeError("foreground_background_remix seed_vc edit requires reference_audio_path")
        seed_params = {
            key: edit_params.pop(key)
            for key in list(edit_params)
            if key in {"diffusion_steps", "length_adjust", "inference_cfg_rate", "seed"}
        }
        result = await execute_tool(
            "seed_vc",
            {
                "source_audio_path": str(foreground_path),
                "reference_audio_path": str(reference_audio),
                "output_path": str(output_path),
                **seed_params,
            },
            state,
        )
        return Path(result.get("output_audio_path") or output_path), result

    if normalized in {"no_reference_voice_conversion", "voice_design", "target_voice_design", "custom_voice"}:
        reference_text = str(
            edit_params.pop("reference_text", None)
            or edit_params.pop("target_voice_reference_text", None)
            or source_text
            or "This is a target voice reference for the edited speech."
        )
        voice_prompt = str(
            edit_params.pop("voice_prompt", None)
            or edit_params.pop("instruct", None)
            or edit_params.pop("instruction", None)
            or edit_params.pop("target_voice", None)
            or edit_params.pop("voice_description", None)
            or instruction
        )
        reference_path = output_path.with_name(f"{output_path.stem}_target_voice_reference.wav")
        reference_result = await execute_tool(
            "qwen3_tts",
            {
                "text": reference_text,
                "output_path": str(reference_path),
                "mode": "custom_voice",
                "instruct": voice_prompt,
                "language": edit_params.pop("language", "Auto"),
            },
            state,
        )
        target_reference = reference_result.get("output_audio_path") or str(reference_path)
        seed_params = {
            key: edit_params.pop(key)
            for key in list(edit_params)
            if key in {"diffusion_steps", "length_adjust", "inference_cfg_rate", "seed"}
        }
        conversion_result = await execute_tool(
            "seed_vc",
            {
                "source_audio_path": str(foreground_path),
                "reference_audio_path": str(target_reference),
                "output_path": str(output_path),
                **seed_params,
            },
            state,
        )
        return Path(conversion_result.get("output_audio_path") or output_path), {
            "operation": "no_reference_voice_conversion",
            "output_audio_path": conversion_result.get("output_audio_path") or str(output_path),
            "target_voice_reference_audio_path": str(target_reference),
            "target_voice_reference_text": reference_text,
            "voice_design_prompt": voice_prompt,
            "reference_generation": reference_result,
            "seed_vc_conversion": conversion_result,
        }

    if normalized == "speech_rate":
        source_text = _ensure_nested_speech_rate_source_text(
            foreground_path=foreground_path,
            instruction=instruction,
            source_text=source_text,
            edit_params=edit_params,
            state=state,
            agentic_mode=agentic_mode,
        )

    args = _edit_data_pipeline_args(
        operation=normalized,
        instruction=instruction,
        source_text=source_text,
        input_audio_path=str(foreground_path),
        output_path=str(output_path),
        output_dir=str(output_path.parent / "edit_data_pipeline_artifacts"),
        edit_params=edit_params,
    )
    result = await execute_tool("edit_data_pipeline", args, state)
    return Path(result.get("output_audio_path") or output_path), result


def _ensure_nested_speech_rate_source_text(
    *,
    foreground_path: Path,
    instruction: str,
    source_text: str | None,
    edit_params: dict[str, Any],
    state: dict[str, Any],
    agentic_mode: str,
) -> str:
    effective = source_text
    nested_source_text = edit_params.get("source_text") or edit_params.get("text")
    if not effective and isinstance(nested_source_text, str) and nested_source_text.strip():
        effective = nested_source_text.strip()
    if isinstance(effective, str) and effective.strip():
        return effective.strip()

    step_id = str(state.get("_current_step_id") or "foreground_background_remix")
    repair_key = f"{step_id}:foreground_background_remix.speech_rate"
    language_hints = _nested_source_text_language_hints(edit_params, state)
    repair_record: dict[str, Any] = {
        "strategy": "perception_source_text_before_nested_foreground_speech_rate",
        "step_id": step_id,
        "operation": "foreground_background_remix",
        "nested_operation": "edit_data_pipeline.speech_rate",
        "input_audio_path": str(foreground_path),
        "language_hints": language_hints,
        "reason": (
            "Nested foreground speech_rate should use transcript-conditioned quality path; "
            "run Qwen3ASR/specialized speech perception on the selected foreground stem, "
            "with the configured audio fallback provider if needed."
        ),
        "agentic_mode": agentic_mode,
    }
    state.setdefault("source_text_repairs", {})[repair_key] = repair_record

    try:
        result = build_audio_perception_fallback(agentic_mode=agentic_mode).perceive(
            user_prompt=instruction or "Recover source text before foreground speech-rate editing.",
            audio_path=str(foreground_path),
            base_perception={},
            source_text=None,
            reference_audio_paths=list(state.get("reference_audio_paths") or []),
            output_dir=str(Path(str(state.get("output_dir") or foreground_path.parent)) / f"{step_id}_foreground_source_text_repair"),
            requirements=PerceptionRequirements(source_text=True),
            language_hints=language_hints,
            run_specialized=True,
            allow_model_fallback=True,
            force_model_fallback=False,
            refresh_cache=True,
            strict=True,
        )
    except Exception as exc:
        repair_record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise RuntimeError(
            "foreground_background_remix nested edit_data_pipeline.speech_rate requires source_text "
            "for wsola+indextts2; automatic ASR/model perception repair failed, so pure wsola "
            f"was not used as an implicit fallback. Error: {exc}"
        ) from exc

    perception = result.perception
    repaired_text = str(perception.get("source_text") or "").strip()
    repair_record.update(
        {
            "status": "completed" if repaired_text else "insufficient",
            "satisfied": result.satisfied,
            "missing": result.missing,
            "attempts": result.attempts,
            "source_text_length": len(repaired_text),
            "source_text_preview": repaired_text[:160],
            "source_text_audio_path": perception.get("source_text_audio_path"),
            "source_text_audio_index": perception.get("source_text_audio_index"),
        }
    )
    if not repaired_text:
        raise RuntimeError(
            "foreground_background_remix nested edit_data_pipeline.speech_rate requires source_text "
            "for wsola+indextts2; perception repair completed but did not produce transcript text."
        )

    edit_params.setdefault("source_text", repaired_text)
    state["source_text"] = repaired_text
    if perception.get("source_text_audio_path"):
        state["source_text_audio_path"] = perception.get("source_text_audio_path")
    if perception.get("source_text_audio_index") is not None:
        state["source_text_audio_index"] = perception.get("source_text_audio_index")
    state["perception_report"] = perception
    return repaired_text


async def _fallback_seamless_to_qwen_voice_clone(
    *,
    execute_tool: ExecuteTool,
    foreground_path: Path,
    output_path: Path,
    instruction: str,
    source_text: Any,
    target_text: Any,
    target_language: str,
    source_language: str | None,
    unavailable_error: Exception,
    state: dict[str, Any],
    agentic_mode: str,
) -> dict[str, Any]:
    if not isinstance(target_text, str) or not target_text.strip():
        target_text = _translate_audio_or_text_for_s2st_fallback(
            input_audio_path=str(foreground_path),
            source_text=source_text,
            target_language=target_language,
            source_language=source_language,
            request=instruction,
            agentic_mode=agentic_mode,
        )
    if not isinstance(target_text, str) or not target_text.strip():
        raise RuntimeError(
            "foreground_background_remix seamless_expressive_s2st is unavailable and fallback could not "
            "obtain target_text from planner params, source_text, or configured provider translation"
        ) from unavailable_error

    language = _language_name_from_code(target_language) or _qwen3_tts_language_from_text_and_request(
        target_text,
        instruction,
    )
    step_id = str(state.get("_current_step_id") or "foreground_background_remix")
    state.setdefault("operation_rewrites", {})[step_id] = {
        "from": "foreground_background_remix.seamless_expressive_s2st",
        "to": "foreground_background_remix.qwen3_tts_voice_clone",
        "reason": f"Seamless runtime unavailable: {type(unavailable_error).__name__}: {unavailable_error}",
        "target_language": target_language,
        "source_language": source_language,
        "fallback_target_text": target_text,
        "boundary": (
            "Qwen3-TTS voice_clone fallback preserves speaker timbre best-effort and remixes the original "
            "background, but it does not provide SeamlessExpressive guarantees for vocal style, speech rate, or pauses."
        ),
    }
    qwen_result = await execute_tool(
        "qwen3_tts",
        {
            "text": target_text,
            "output_path": str(output_path),
            "mode": "voice_clone",
            "ref_audio_path": str(foreground_path),
            "language": language,
            "instruct": (
                "Synthesize the translated speech using the foreground speech as the speaker-timbre reference. "
                "Keep the same speaker identity as much as this fallback path allows."
            ),
            "x_vector_only_mode": True,
        },
        state,
    )
    return {
        "output_audio_path": qwen_result.get("output_audio_path") or str(output_path),
        "operation": "seamless_expressive_s2st_fallback_qwen3_tts",
        "foreground_tool_chain": ["seamless_expressive_s2st", "qwen3_tts"],
        "target_text": target_text,
        "target_language": target_language,
        "source_language": source_language,
        "fallback_reason": str(unavailable_error),
        "fallback_boundary": (
            "SeamlessExpressive/M4T did not run. This is a transparent text-translation plus "
            "voice-clone TTS fallback, not a successful Seamless S2ST execution."
        ),
        "qwen3_tts_result": qwen_result,
    }


def _edit_data_pipeline_args(
    *,
    operation: str,
    instruction: str,
    source_text: str | None,
    input_audio_path: str,
    output_path: str,
    output_dir: str,
    edit_params: dict[str, Any],
) -> dict[str, Any]:
    effective_source_text = source_text
    nested_source_text = edit_params.get("source_text")
    if not effective_source_text and isinstance(nested_source_text, str) and nested_source_text.strip():
        effective_source_text = nested_source_text
    if operation in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit", "speech_rate"} and not effective_source_text:
        raise RuntimeError(
            f"foreground_background_remix edit_data_pipeline.{operation} requires source_text for the selected foreground stem"
        )
    args = build_edit_data_pipeline_tool_arguments(
        operation=operation,
        instruction=instruction,
        source_text=effective_source_text,
        input_audio_path=input_audio_path,
        output_path=output_path,
        output_dir=output_dir,
        language=edit_params.get("language"),
        params=edit_params,
    )
    args["operation"] = operation
    args["input_audio_path"] = input_audio_path
    args["output_path"] = output_path
    args.setdefault("instruction", instruction)
    if operation == "volume_normalize":
        args.setdefault("method", "rms")
        args.setdefault("target_db", -20.0)
    if operation == "pitch_rate_edit":
        args.setdefault("pitch_method", "psola+indextts2_cco")
        args.setdefault("smooth_policy", "never" if "<pitch" in str(args.get("instruction_st", "")) else "auto")
    return args


def _nested_source_text_language_hints(edit_params: dict[str, Any], state: dict[str, Any]) -> list[str | None]:
    perception = state.get("perception_report") if isinstance(state.get("perception_report"), dict) else {}
    asr = perception.get("asr") if isinstance(perception.get("asr"), dict) else {}
    values = [
        edit_params.get("expected_language"),
        edit_params.get("language"),
        state.get("language"),
        (state.get("context") or {}).get("language") if isinstance(state.get("context"), dict) else None,
        perception.get("language"),
        asr.get("language"),
    ]
    hints: list[str | None] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        hints.append(text)
    return hints


def _render_final_audio(
    *,
    original_audio: AudioSegment,
    foreground_path: Path | None,
    background_path: Path | None,
    isolate_foreground: bool,
    foreground_gain_db: float | None,
    background_gain_db: float | None,
    target_duration_ms: int,
    preserve_duration: bool,
) -> AudioSegment:
    if isolate_foreground:
        foreground = _load_or_silence(foreground_path, original_audio)
        if foreground_gain_db is not None:
            foreground += foreground_gain_db
        return _fit_duration(foreground, target_duration_ms) if preserve_duration else foreground

    background = _load_or_silence(background_path, original_audio)
    if background_gain_db is not None:
        background += background_gain_db
    if preserve_duration:
        background = _fit_duration(background, target_duration_ms)

    if foreground_path is None:
        return background

    foreground = _load_or_silence(foreground_path, original_audio)
    if foreground_gain_db is not None:
        foreground += foreground_gain_db
    if preserve_duration:
        foreground = _fit_duration(foreground, target_duration_ms)
    elif len(foreground) > len(background):
        background = _fit_duration(background, len(foreground))
    return background.overlay(foreground)


def _load_or_silence(path: Path | None, reference: AudioSegment) -> AudioSegment:
    if path is not None and path.exists():
        audio = AudioSegment.from_file(str(path))
        return _match_audio_format(audio, reference)
    return (
        AudioSegment.silent(duration=len(reference), frame_rate=reference.frame_rate)
        .set_channels(reference.channels)
        .set_sample_width(reference.sample_width)
    )


def _fit_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    if len(audio) == target_ms:
        return audio
    if len(audio) > target_ms:
        return audio[:target_ms]
    return audio + AudioSegment.silent(duration=target_ms - len(audio), frame_rate=audio.frame_rate).set_channels(audio.channels)


def _match_audio_format(audio: AudioSegment, reference: AudioSegment) -> AudioSegment:
    if audio.frame_rate != reference.frame_rate:
        audio = audio.set_frame_rate(reference.frame_rate)
    if audio.sample_width != reference.sample_width:
        audio = audio.set_sample_width(reference.sample_width)
    if audio.channels != reference.channels:
        if audio.channels != 1 and reference.channels != 1:
            audio = audio.set_channels(1)
        audio = audio.set_channels(reference.channels)
    return audio


def _normalize_nested_edit_operation(value: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(value, dict):
        operation = value.get("operation") or value.get("tool") or value.get("tool_id")
        nested_params = value.get("params") if isinstance(value.get("params"), dict) else {}
        return (str(operation).strip() if operation else None), dict(nested_params)
    if isinstance(value, str) and value.strip():
        return value.strip(), {}
    return None, {}


def _scoped_selection_instruction(params: dict[str, Any], edit_params: dict[str, Any]) -> str | None:
    for key in ("selection_instruction", "selector", "target_selector", "span_selector"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("selection_instruction", "selector", "target_selector", "span_selector"):
        value = edit_params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _should_use_scoped_artifact_edit(edit_operation: str, scoped_selection: str | None) -> bool:
    if not scoped_selection:
        return False
    normalized = edit_operation.removeprefix("edit_data_pipeline.").strip().replace("-", "_")
    if normalized in {"artifact_ops.speaker_scoped_edit", "speaker_scoped_edit"}:
        return False
    return normalized in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit", "volume_normalize"}


def _existing_paths(values: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        if path.exists() and path.is_file():
            paths.append(path.resolve())
    return paths


def _select_stem(
    stem_paths: list[Path],
    policy: str,
    *,
    role: str,
    exclude: set[Path] | None = None,
) -> Path | None:
    exclude = exclude or set()
    candidates = [path for path in stem_paths if path not in exclude]
    if not candidates:
        return None
    return select_stem(candidates, policy)


def _safe_separation_stem_policy(policy: str) -> str:
    if not policy.strip():
        raise ValueError("stem_selection_requires_candidates_and_policy")
    return policy


def _infer_foreground_action(instruction: str) -> str:
    text = instruction.lower()
    if _contains(text, ["delete all the voice", "remove all voice", "only keep the background", "keep background", "background only"]):
        return "remove_foreground"
    if _contains(text, ["extract", "isolate", "keep only"]) and _contains(text, ["voice", "speech", "speaker", "vocal"]):
        return "isolate_foreground"
    return "edit_foreground"


def _tool_chain_for(edit_operation: str, *, post_enhance: bool) -> list[str]:
    chain = ["speech_separation"]
    normalized = edit_operation.removeprefix("edit_data_pipeline.").strip()
    if normalized in {"speech_enhancement", "enhance", "denoise", "clearvoice"}:
        chain.append("speech_enhancement")
    elif normalized in {
        "audio_effects",
        "audio_effect",
        "telephone",
        "phone",
        "phone_call",
        "old_radio",
        "radio",
        "underwater",
        "strong_wind",
        "muffled_wall",
        "whisper_style",
        "shouting_style",
        "match_channel_reference",
    }:
        chain.append("audio_effects")
    elif normalized in {"seamless_expressive_s2st", "seamless_s2st", "s2st", "speech_to_speech_translation"}:
        chain.append("seamless_expressive_s2st")
    elif normalized in {"artifact_ops.speaker_scoped_edit", "speaker_scoped_edit"}:
        chain.append("artifact_ops.speaker_scoped_edit")
    elif normalized in {"qwen3_tts", "qwen_tts", "tts", "voice_clone"}:
        chain.append("qwen3_tts")
    elif normalized in {"accent_conversion", "dialect_transfer", "dialect_conversion", "accent_transfer"}:
        chain.extend(["accent_conversion", "seed_vc", "qwen3_tts"])
    elif normalized in {"seed_vc", "voice_conversion", "voice_clone_conversion", "timbre_transfer", "timber_transfer"}:
        chain.append("seed_vc")
    elif normalized in {"no_reference_voice_conversion", "voice_design", "target_voice_design", "custom_voice"}:
        chain.extend(["qwen3_tts", "seed_vc"])
    elif normalized in {"gain", "volume_gain"}:
        chain.append("local_gain")
    elif normalized and normalized not in {"copy", "none", "noop"}:
        chain.append("edit_data_pipeline")
    chain.append("remix")
    if post_enhance:
        chain.append("speech_enhancement")
    return chain


def _tool_chain_for_result(
    edit_operation: str,
    edit_result: dict[str, Any] | None,
    *,
    post_enhance: bool,
) -> list[str]:
    nested = edit_result.get("foreground_tool_chain") if isinstance(edit_result, dict) else None
    if isinstance(nested, list) and nested:
        chain = ["speech_separation"]
        chain.extend(str(item) for item in nested if item)
        chain.append("remix")
        if post_enhance:
            chain.append("speech_enhancement")
        return chain
    return _tool_chain_for(edit_operation, post_enhance=post_enhance)


def _seamless_error_allows_text_tts_fallback(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "seamless" in text
        and (
            "not available" in text
            or "service error 503" in text
            or "missing expressivity_predict" in text
            or ("missing" in text and "model" in text)
            or "gated artifacts" in text
        )
    )


def _translate_audio_or_text_for_s2st_fallback(
    *,
    input_audio_path: str,
    source_text: Any,
    target_language: str,
    source_language: str | None,
    request: str | None,
    agentic_mode: str | None = None,
) -> str | None:
    target_name = _language_name_from_code(target_language) or target_language
    source_name = _language_name_from_code(source_language) if source_language else None
    system_prompt = (
        "You are the translation fallback for an audio editing runtime. "
        "Produce a target-language transcript for speech synthesis. "
        "Translate every spoken utterance in chronological order. "
        "Do not include timestamps, speaker labels, explanations, markdown, or quotes."
    )
    payload = {
        "task": "translate_foreground_speech_for_voice_clone_tts_fallback",
        "request": request,
        "source_text_hint": source_text if isinstance(source_text, str) else None,
        "source_language": source_name or source_language,
        "target_language": target_name,
        "required_json_schema": {
            "target_text": "complete translated speech transcript only",
            "source_used": "audio | source_text_hint | both",
            "reason": "brief note",
        },
    }
    mode = resolve_agentic_mode(agentic_mode)
    audio_paths = [input_audio_path] if input_audio_path and Path(str(input_audio_path)).exists() else None
    try:
        timeout = float(os.getenv("X_EDITOR_S2ST_FALLBACK_TRANSLATION_TIMEOUT", "90"))
    except ValueError:
        timeout = 90.0
    try:
        if mode == OPEN_GLM_QWEN_OMNI_MODE:
            if not str(source_text or "").strip():
                return None
            response = OpenAICompatibleChatClient(
                api_key=resolve_openai_compatible_api_key(GLM_PROVIDER),
                base_url=resolve_openai_compatible_base_url(GLM_PROVIDER),
                model_name=resolve_openai_compatible_model(GLM_PROVIDER),
                provider_name=GLM_PROVIDER,
                timeout=timeout,
                max_tokens=4096,
            ).call_json(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                response_format={"type": "json_object"},
                repair_schema_name="ForegroundTranslationFallback",
            )
        else:
            response = GeminiClient(timeout=timeout).call_json(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, indent=2),
                audio_paths=audio_paths,
            )
    except Exception:
        return None
    target = response.get("target_text") if isinstance(response, dict) else None
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _language_name_from_code(code: Any) -> str | None:
    normalized = canonicalize_language(code) or str(code or "").strip().lower().replace("_", "-")
    mapping = {
        "en": "English",
        "eng": "English",
        "english": "English",
        "zh": "Chinese",
        "zho": "Chinese",
        "cmn": "Chinese",
        "mandarin": "Chinese",
        "chinese": "Chinese",
        "ja": "Japanese",
        "jpn": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "kor": "Korean",
        "korean": "Korean",
        "es": "Spanish",
        "spa": "Spanish",
        "spanish": "Spanish",
        "fr": "French",
        "fra": "French",
        "fre": "French",
        "french": "French",
        "de": "German",
        "deu": "German",
        "ger": "German",
        "german": "German",
        "it": "Italian",
        "ita": "Italian",
        "italian": "Italian",
        "pt": "Portuguese",
        "por": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "rus": "Russian",
        "russian": "Russian",
        "th": "Thai",
        "tha": "Thai",
        "thai": "Thai",
    }
    return mapping.get(normalized)


def _qwen3_tts_language_from_text_and_request(target_text: str, request: str | None) -> str:
    request_lower = (request or "").lower()
    if "english" in request_lower or "英语" in request_lower or "英文" in request_lower:
        return "English"
    if "chinese" in request_lower or "mandarin" in request_lower or "中文" in request_lower or "普通话" in request_lower:
        return "Chinese"
    if "japanese" in request_lower or "日语" in request_lower or "日本语" in request_lower:
        return "Japanese"
    if "korean" in request_lower or "韩语" in request_lower or "韩国语" in request_lower:
        return "Korean"
    if "spanish" in request_lower or "西班牙语" in request_lower:
        return "Spanish"
    if "french" in request_lower or "法语" in request_lower:
        return "French"
    if "german" in request_lower or "德语" in request_lower:
        return "German"
    if "italian" in request_lower or "意大利语" in request_lower:
        return "Italian"
    if "portuguese" in request_lower or "葡萄牙语" in request_lower:
        return "Portuguese"
    if "russian" in request_lower or "俄语" in request_lower:
        return "Russian"
    if any("\u4e00" <= char <= "\u9fff" for char in target_text):
        return "Chinese"
    if any("\u3040" <= char <= "\u30ff" for char in target_text):
        return "Japanese"
    if any("\uac00" <= char <= "\ud7af" for char in target_text):
        return "Korean"
    letters = [char for char in target_text if char.isalpha()]
    ascii_letters = [char for char in letters if char.isascii()]
    if letters and len(ascii_letters) / max(1, len(letters)) > 0.85:
        return "English"
    return "Auto"


def _contains(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
