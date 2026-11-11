"""Deterministic routing boundaries for planner output.

This module retains the legacy enforcement mode and also exposes a shadow mode
used by capacity experiments. Shadow mode records what the legacy semantic
rules would have changed without claiming that correction as Planner ability.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
from typing import Any

from x_editor.planner.plan_normalizer import operation_ids
from x_editor.planner.structured_edit import build_pause_edit
from x_editor.runtime.paralinguistic_events import (
    normalize_paralinguistic_event,
    normalize_paralinguistic_operation,
)

TRANSCRIPT_REQUIRED_EDIT_DATA_PIPELINE_OPERATIONS = {
    "edit_data_pipeline.text_edit",
    "edit_data_pipeline.pitch_rate_edit",
    "edit_data_pipeline.emotion_style_edit",
    "edit_data_pipeline.pause_edit",
    "edit_data_pipeline.speech_rate",
}

SPEAKER_GROUNDED_ARTIFACT_OPERATIONS = {
    "artifact_ops.extract_spans",
    "artifact_ops.remove_spans",
    "artifact_ops.mute_spans",
    "artifact_ops.beep_spans",
    "artifact_ops.speaker_scoped_edit",
}


def apply_routing_hard_boundaries(
    data: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
) -> None:
    """Apply non-negotiable cross-operation routing boundaries in-place."""

    steps = data.get("steps")
    if not isinstance(steps, list):
        return
    available = operation_ids(operations)
    metadata = data.setdefault("metadata", {})
    corrections = metadata.setdefault("routing_boundary_corrections", [])
    if not isinstance(corrections, list):
        corrections = []
        metadata["routing_boundary_corrections"] = corrections

    _apply_paralinguistic_event_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_operation_intent_conflict_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_non_lexical_replacement_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_reference_free_speaker_span_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _ensure_source_text_perception_for_text_edits(
        steps,
        data=data,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_generation_and_domain_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_emphasis_stress_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_foreground_background_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _apply_speaker_scoped_boundaries(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _promote_extracted_local_edit_before_scoped_edit_to_source_timeline(
        steps,
        context=context,
        corrections=corrections,
    )
    _promote_safe_preprocessing_before_scoped_edit_to_source_timeline(
        steps,
        corrections=corrections,
    )
    _ensure_speaker_attributed_review_for_artifact_span_grounding(
        steps,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _ensure_source_text_perception_for_text_edits(
        steps,
        data=data,
        user_request=user_request,
        context=context,
        available=available,
        corrections=corrections,
    )
    _collapse_redundant_terminal_speaker_scoped_reconstruct(
        steps,
        corrections=corrections,
    )
    _normalize_extract_then_reorder(
        steps,
        corrections=corrections,
    )


def observe_routing_hard_boundaries(
    data: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
) -> None:
    """Record legacy semantic rewrite proposals without changing Planner output."""

    candidate = copy.deepcopy(data)
    apply_routing_hard_boundaries(
        candidate,
        user_request=user_request,
        context=context,
        operations=operations,
    )
    candidate_metadata = candidate.get("metadata")
    proposals = (
        candidate_metadata.get("routing_boundary_corrections")
        if isinstance(candidate_metadata, dict)
        else None
    )
    if not isinstance(proposals, list) or not proposals:
        return
    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    proposed_steps = candidate.get("steps")
    metadata["routing_boundary_shadow"] = {
        "schema_version": "x-editor.routing-boundary-shadow/v1",
        "behavior_changed": False,
        "proposal_count": len(proposals),
        "proposals": copy.deepcopy(proposals),
        "authored_steps_fingerprint": _routing_shadow_fingerprint(
            data.get("steps")
        ),
        "proposed_steps_fingerprint": _routing_shadow_fingerprint(
            proposed_steps
        ),
    }


def _routing_shadow_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _apply_paralinguistic_event_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    event = normalize_paralinguistic_event(user_request)
    operation = normalize_paralinguistic_operation(user_request)
    if not event or operation not in {"add", "remove"}:
        return
    target_operation = f"paralinguistic_event_{operation}"
    if target_operation not in available:
        return
    dict_steps = _dict_steps(steps)
    wrong_operations = {
        "qwen3_tts",
        "edit_data_pipeline.text_edit",
        "edit_data_pipeline.pitch_rate_edit",
        "edit_data_pipeline.emotion_style_edit",
        "edit_data_pipeline.audio_assembly",
        "artifact_ops.extract_spans",
        "artifact_ops.remove_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.reconstruct",
        "artifact_ops.speaker_scoped_edit",
    }
    for step in dict_steps:
        current = str(step.get("operation") or "")
        if current == target_operation:
            _ensure_paralinguistic_event_params(step, user_request=user_request, event=event)
            continue
        if current not in wrong_operations:
            continue
        if len(dict_steps) > 1 and not _step_mentions_paralinguistic_event(step, event):
            continue
        _rewrite_step_to_paralinguistic_event(
            step,
            operation=target_operation,
            event=event,
            user_request=user_request,
            context=context,
        )
        _append_correction(
            corrections,
            step,
            current,
            target_operation,
            "paralinguistic_add_remove_requires_dedicated_event_operation",
        )


def _apply_operation_intent_conflict_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Resolve planner steps that express two executable operation intents."""

    for step in _dict_steps(steps):
        operation = str(step.get("operation") or "")
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        emotion_label = _emotion_style_label_from_contract(params)
        if (
            operation == "edit_data_pipeline.pitch_rate_edit"
            and emotion_label is not None
            and "edit_data_pipeline.emotion_style_edit" in available
        ):
            local_text = _unique_selected_span_text(context)
            if local_text:
                emotion_label = re.sub(
                    r"(<\s*emo\b[^>]*>).*?(</\s*emo\s*>)",
                    lambda match: f"{match.group(1)}{local_text}{match.group(2)}",
                    emotion_label,
                    count=1,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            step["operation"] = "edit_data_pipeline.emotion_style_edit"
            step["params"] = {
                "instruction": user_request,
                "instruction_st": emotion_label,
                "source_text": local_text or context.get("source_text"),
            }
            _append_correction(
                corrections,
                step,
                operation,
                "edit_data_pipeline.emotion_style_edit",
                "emotion_label_contract_requires_emotion_style_operation",
            )
            continue
        inner = _nested_operation_name(params.get("operation"))
        if not operation.startswith("edit_data_pipeline.") or not inner:
            continue

        inner_operation = (
            inner if inner in available else f"edit_data_pipeline.{inner}"
        )
        if inner_operation == operation:
            params.pop("operation", None)
            continue

        if (
            _requests_no_reference_voice_conversion(user_request, context)
            and "no_reference_voice_conversion" in available
            and inner in {"text_edit", "pitch_rate_edit", "emotion_style_edit"}
        ):
            params.pop("operation", None)
            _rewrite_step_to_no_reference_voice_conversion(
                step,
                user_request=user_request,
                context=context,
            )
            _append_correction(
                corrections,
                step,
                operation,
                "no_reference_voice_conversion",
                "conflicting_edit_data_pipeline_operation_replanned_as_no_reference_voice_conversion",
            )
            continue

        if operation == "edit_data_pipeline.text_edit" and _has_text_edit_instruction_tag(params):
            params.pop("operation", None)
            _drop_text_edit_conflicting_prosody_defaults(params)
            step["params"] = params
            _append_correction(
                corrections,
                step,
                operation,
                operation,
                "text_edit_label_contract_preserved_over_conflicting_inner_operation",
            )
            continue

        if (
            operation == "edit_data_pipeline.emotion_style_edit"
            and (
                _has_emotion_style_instruction_tag(params)
                or emotion_label is not None
            )
        ):
            if emotion_label is not None and not _has_emotion_style_instruction_tag(
                params
            ):
                local_text = _unique_selected_span_text(context)
                if local_text:
                    emotion_label = re.sub(
                        r"(<\s*emo\b[^>]*>).*?(</\s*emo\s*>)",
                        lambda match: (
                            f"{match.group(1)}{local_text}{match.group(2)}"
                        ),
                        emotion_label,
                        count=1,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                params["instruction_st"] = emotion_label
                params["source_text"] = local_text or context.get("source_text")
                params.setdefault("instruction", user_request)
            params.pop("operation", None)
            _drop_text_edit_conflicting_prosody_defaults(params)
            step["params"] = params
            _append_correction(
                corrections,
                step,
                operation,
                operation,
                "emotion_style_label_contract_preserved_over_conflicting_inner_operation",
            )
            continue

        if operation == "edit_data_pipeline.text_edit" and _is_existing_audio_text_replacement(user_request, context):
            params.pop("operation", None)
            _drop_text_edit_conflicting_prosody_defaults(params)
            params.setdefault("instruction", user_request)
            if context.get("source_text"):
                params.setdefault("source_text", context.get("source_text"))
            if context.get("target_text"):
                params.setdefault("target_text", context.get("target_text"))
            step["params"] = params
            _append_correction(
                corrections,
                step,
                operation,
                operation,
                "lexical_text_edit_preserved_over_conflicting_inner_operation",
            )
            continue

        if inner_operation in available:
            params.pop("operation", None)
            step["operation"] = inner_operation
            step["params"] = params
            _append_correction(
                corrections,
                step,
                operation,
                inner_operation,
                "conflicting_edit_data_pipeline_operation_canonicalized",
            )
            continue

        params.pop("operation", None)
        _rewrite_step_to_unsupported(
            step,
            (
                f"Planner produced conflicting operation intents: {operation} with "
                f"params.operation={inner!r}, but {inner_operation} is unavailable."
            ),
            available,
        )
        _append_correction(
            corrections,
            step,
            operation,
            str(step.get("operation")),
            "conflicting_edit_data_pipeline_operation_unavailable",
        )


def _apply_non_lexical_replacement_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Route replacement targets that are sounds/effects away from lexical text_edit."""

    artifact_operation = _non_lexical_replacement_artifact_operation(user_request, available)
    if artifact_operation is None:
        return
    for step in _dict_steps(steps):
        operation = str(step.get("operation") or "")
        if operation not in {"edit_data_pipeline.text_edit", "qwen3_tts"}:
            continue
        original = operation
        _rewrite_step_to_artifact_span_operation(
            step,
            operation=artifact_operation,
            user_request=user_request,
            context=context,
        )
        _append_correction(
            corrections,
            step,
            original,
            artifact_operation,
            "non_lexical_replacement_target_requires_artifact_span_operation",
        )


def _apply_reference_free_speaker_span_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Recover speaker extract/remove/mute plans that were marked unsupported/effects."""

    artifact_operation = _pure_speaker_artifact_operation(user_request, {}, available)
    if artifact_operation is None:
        return
    for step in _dict_steps(steps):
        operation = str(step.get("operation") or "")
        if operation not in {"unsupported", "audio_effects"}:
            continue
        original = operation
        _rewrite_step_to_artifact_span_operation(
            step,
            operation=artifact_operation,
            user_request=user_request,
            context=context,
        )
        _append_correction(
            corrections,
            step,
            original,
            artifact_operation,
            "reference_free_speaker_span_request_requires_artifact_span_operation",
        )


def _apply_generation_and_domain_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    if _is_existing_audio_text_replacement(user_request, context):
        if "edit_data_pipeline.text_edit" in available:
            _collapse_lexical_reorder_chain_to_text_edit(
                steps,
                user_request=user_request,
                context=context,
                corrections=corrections,
            )
        for step in _dict_steps(steps):
            if step.get("operation") != "qwen3_tts":
                continue
            original = str(step.get("operation") or "")
            if "edit_data_pipeline.text_edit" in available:
                _rewrite_step_to_edit_data_pipeline(
                    step,
                    operation="text_edit",
                    user_request=user_request,
                    context=context,
                    reason=(
                        "existing-audio lexical edits must use edit_data_pipeline.text_edit; "
                        "qwen3_tts is not allowed for text replacement"
                    ),
                )
                _append_correction(corrections, step, original, "edit_data_pipeline.text_edit", "existing_audio_text_replacement_hard_boundary")
            else:
                _rewrite_step_to_unsupported(
                    step,
                    "Existing-audio lexical replacement requires edit_data_pipeline.text_edit; qwen3_tts is not allowed.",
                    available,
                )
                _append_correction(corrections, step, original, str(step.get("operation")), "missing_edit_data_pipeline_text_edit_for_existing_audio_text_replacement")

    if _is_dialect_translation_request(user_request):
        _apply_dialect_translation_boundary(
            steps,
            user_request=user_request,
            context=context,
            available=available,
            corrections=corrections,
        )
        return

    translation_language = _requested_translation_target_language(user_request)
    seamless_unavailable = _operation_unavailable(context, "seamless_expressive_s2st")
    if translation_language and "seamless_expressive_s2st" in available and not seamless_unavailable:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation == "seamless_expressive_s2st":
                _ensure_seamless_params(step, user_request=user_request, target_language=translation_language)
                continue
            if operation not in {
                "qwen3_tts",
                "accent_conversion",
                "edit_data_pipeline.text_edit",
                "edit_data_pipeline.audio_assembly",
            }:
                continue
            if _translation_request_is_localized(user_request) and operation.startswith("edit_data_pipeline."):
                continue
            _rewrite_step_to_seamless_s2st(
                step,
                user_request=user_request,
                target_language=translation_language,
                context=context,
            )
            _append_correction(corrections, step, operation, "seamless_expressive_s2st", "speech_translation_requires_seamless_s2st")

    if _requests_accent_conversion(user_request) and "accent_conversion" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation == "accent_conversion":
                _ensure_accent_conversion_params(step, user_request=user_request, context=context)
                continue
            if operation == "artifact_ops.speaker_scoped_edit":
                continue
            if operation not in {"qwen3_tts", "seed_vc", "edit_data_pipeline.text_edit", "edit_data_pipeline.emotion_style_edit", "edit_data_pipeline.pitch_rate_edit"}:
                continue
            _rewrite_step_to_accent_conversion(step, user_request=user_request, context=context)
            _append_correction(corrections, step, operation, "accent_conversion", "accent_or_dialect_request_requires_accent_conversion")
        if not _is_dialect_translation_request(user_request):
            _collapse_redundant_accent_conversion_steps(steps, corrections=corrections)

    if _requests_no_reference_voice_conversion(user_request, context) and "no_reference_voice_conversion" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation == "no_reference_voice_conversion":
                _ensure_no_reference_voice_conversion_params(
                    step,
                    user_request=user_request,
                    context=context,
                )
                continue
            if operation == "artifact_ops.speaker_scoped_edit":
                continue
            if operation not in {
                "qwen3_tts",
                "seed_vc",
                "edit_data_pipeline.text_edit",
                "edit_data_pipeline.pitch_rate_edit",
                "edit_data_pipeline.emotion_style_edit",
            }:
                continue
            _rewrite_step_to_no_reference_voice_conversion(step, user_request=user_request, context=context)
            _append_correction(corrections, step, operation, "no_reference_voice_conversion", "existing_audio_voice_design_requires_seedvc_reference_conversion")
        _collapse_redundant_no_reference_voice_conversion_steps(
            steps,
            corrections=corrections,
        )

    if _requests_spatial_room_effect(user_request) and "spatial_reverb" in available:
        has_dedicated_spatial_step = any(
            str(item.get("operation") or "") == "spatial_reverb"
            for item in _dict_steps(steps)
        )
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation == "spatial_reverb":
                _ensure_spatial_reverb_params(step, user_request=user_request)
                continue
            if (
                operation == "edit_data_pipeline.emotion_style_edit"
                and has_dedicated_spatial_step
            ):
                continue
            if operation not in {"unsupported", "audio_effects", "edit_data_pipeline.emotion_style_edit"}:
                continue
            _rewrite_step_to_spatial_reverb(step, user_request=user_request)
            _append_correction(corrections, step, operation, "spatial_reverb", "spatial_room_effect_requires_spatial_reverb")

    if _requests_channel_or_environment_effect(user_request) and "audio_effects" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation not in {"seed_vc", "edit_data_pipeline.emotion_style_edit", "spatial_reverb"}:
                continue
            _rewrite_step_to_audio_effects(step, user_request=user_request, context=context)
            _append_correction(corrections, step, operation, "audio_effects", "channel_or_environment_effect_must_not_use_voice_conversion")

    if _is_pure_concat_request(user_request) and "artifact_ops.concat" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation == "artifact_ops.concat":
                _ensure_concat_params(step, user_request=user_request, context=context)
                continue
            if operation not in {"qwen3_tts", "edit_data_pipeline.audio_assembly", "audio_assembly", "foreground_background_remix"}:
                continue
            _rewrite_step_to_concat(step, user_request=user_request, context=context)
            _append_correction(corrections, step, operation, "artifact_ops.concat", "pure_concat_requires_artifact_concat")

    if _is_simple_overlay_request(user_request, context) and "artifact_ops.reconstruct" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation == "artifact_ops.reconstruct":
                _ensure_reconstruct_overlay_params(step, user_request=user_request, context=context)
                continue
            if operation not in {
                "qwen3_tts",
                "edit_data_pipeline.audio_assembly",
                "audio_assembly",
                "foreground_background_remix",
                "artifact_ops.concat",
                "audio_effects",
                "unsupported",
            }:
                continue
            _rewrite_step_to_reconstruct(step, user_request=user_request, context=context, mode="overlay")
            _append_correction(corrections, step, operation, "artifact_ops.reconstruct", "multi_audio_overlay_requires_artifact_reconstruct")

    if _requests_synthetic_silence(user_request):
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation != "qwen3_tts" or "artifact_ops.silence" not in available:
                continue
            _rewrite_step_to_silence(step, user_request=user_request, context=context)
            _append_correction(corrections, step, operation, "artifact_ops.silence", "synthetic_silence_must_not_use_tts")

    if "edit_data_pipeline.pause_edit" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation != "artifact_ops.silence":
                continue
            source_text = _context_source_text(context)
            instruction = " ".join(
                str(value or "").strip()
                for value in (user_request, step.get("description"))
                if str(value or "").strip()
            )
            if not source_text or not _requests_pause_inside_existing_audio(
                instruction
            ):
                continue
            structured = build_pause_edit(instruction, source_text)
            if structured is None or not structured.instruction_st:
                continue
            _rewrite_step_to_pause_edit(
                step,
                source_text=source_text,
                instruction_st=structured.instruction_st,
                arguments=structured.arguments,
            )
            _append_correction(
                corrections,
                step,
                operation,
                "edit_data_pipeline.pause_edit",
                "existing_audio_pause_must_not_be_standalone_silence",
            )

    if _is_simple_insert_request(user_request) and "artifact_ops.reconstruct" in available:
        for step in _dict_steps(steps):
            operation = str(step.get("operation") or "")
            if operation not in {"edit_data_pipeline.audio_assembly", "audio_assembly"}:
                continue
            if not _audio_assembly_insert_has_artifact(step):
                continue
            _rewrite_step_to_reconstruct(step, user_request=user_request, context=context, mode="insert")
            _append_correction(corrections, step, operation, "artifact_ops.reconstruct", "simple_insert_requires_artifact_reconstruct")


def _collapse_lexical_reorder_chain_to_text_edit(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    corrections: list[dict[str, Any]],
) -> None:
    """Use lexical substitution for an explicit quoted-term swap."""

    executable_steps = _dict_steps(steps)
    if not executable_steps or not any(
        str(step.get("operation") or "") == "artifact_ops.reorder_spans"
        for step in executable_steps
    ):
        return
    allowed_chain = {
        "artifact_ops.extract_spans",
        "artifact_ops.reorder_spans",
        "artifact_ops.reconstruct",
    }
    if any(
        str(step.get("operation") or "") not in allowed_chain
        for step in executable_steps
    ):
        return
    quoted_terms = _quoted_request_terms(user_request)
    if len(quoted_terms) != 2 or quoted_terms[0].casefold() == quoted_terms[1].casefold():
        return
    source_text = _context_source_text(context)
    if not source_text:
        return
    swapped = _swap_explicit_terms_in_text(
        source_text,
        left=quoted_terms[0],
        right=quoted_terms[1],
    )
    if swapped is None:
        return
    target_text, instruction_st = swapped

    kept = executable_steps[0]
    kept_id = str(kept.get("id") or "")
    removed_ids = [
        str(step.get("id") or "")
        for step in executable_steps[1:]
        if step in steps
    ]
    for step in executable_steps[1:]:
        if step in steps:
            steps.remove(step)
    kept["operation"] = "edit_data_pipeline.text_edit"
    kept["description"] = (
        f"Swap every explicit occurrence of {quoted_terms[0]!r} and "
        f"{quoted_terms[1]!r} in the existing speech."
    )
    input_audio = context.get("input_audio")
    if not input_audio:
        audio_paths = _context_audio_paths(context)
        input_audio = audio_paths[0] if audio_paths else "current"
    kept["inputs"] = {"input_audio": input_audio}
    kept["params"] = {
        "instruction": user_request,
        "source_text": source_text,
        "target_text": target_text,
        "instruction_st": instruction_st,
    }
    language = _context_language(context)
    if language:
        kept["params"]["language"] = language
    kept["expected_output"] = None
    kept["depends_on"] = []
    corrections.append(
        {
            "step_id": kept_id,
            "removed_step_ids": removed_ids,
            "from": "artifact_ops.extract_reorder_reconstruct",
            "to": "edit_data_pipeline.text_edit",
            "reason": "explicit_quoted_lexical_swap_requires_text_edit",
        }
    )


def _quoted_request_terms(user_request: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(
        r'“([^”]+)”|"([^\"]+)"|\'([^\']+)\'|‘([^’]+)’',
        str(user_request or ""),
    ):
        value = next((group for group in match.groups() if group), "").strip()
        if value:
            terms.append(value)
    return terms


def _context_source_text(context: dict[str, Any]) -> str:
    source_text = context.get("source_text")
    if not source_text and isinstance(context.get("perception_report"), dict):
        source_text = context["perception_report"].get("source_text")
    return str(source_text or "").strip()


def _context_language(context: dict[str, Any]) -> str | None:
    language = context.get("language")
    if not language and isinstance(context.get("perception_report"), dict):
        language = context["perception_report"].get("language")
    return str(language).strip() if language else None


def _swap_explicit_terms_in_text(
    source_text: str,
    *,
    left: str,
    right: str,
) -> tuple[str, str] | None:
    pattern = re.compile(
        rf"(?<!\w)(?:{re.escape(left)}|{re.escape(right)})(?!\w)",
        re.IGNORECASE,
    )
    seen: set[str] = set()

    def replacement_for(value: str) -> str:
        seen.add(value.casefold())
        target = right if value.casefold() == left.casefold() else left
        if value[:1].isupper():
            return target[:1].upper() + target[1:]
        if value.isupper():
            return target.upper()
        return target

    target_text = pattern.sub(lambda match: replacement_for(match.group(0)), source_text)
    if seen != {left.casefold(), right.casefold()}:
        return None

    def tagged(match: re.Match[str]) -> str:
        source = match.group(0)
        target = replacement_for(source)
        return (
            f'<sub targ="{html.escape(target, quote=True)}">'
            f"{html.escape(source)}</sub>"
        )

    instruction_st = pattern.sub(tagged, source_text)
    return target_text, instruction_st


def _apply_dialect_translation_boundary(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Force accent/dialect translation into expression normalization + accent conversion."""

    text_edit_available = "edit_data_pipeline.text_edit" in available
    accent_available = "accent_conversion" in available
    dict_steps = _dict_steps(steps)

    if text_edit_available:
        for step in dict_steps:
            operation = str(step.get("operation") or "")
            if operation in {
                "seamless_expressive_s2st",
                "qwen3_tts",
                "edit_data_pipeline.audio_assembly",
            }:
                _rewrite_step_to_dialect_expression_text_edit(
                    step,
                    user_request=user_request,
                    context=context,
                )
                _append_correction(
                    corrections,
                    step,
                    operation,
                    "edit_data_pipeline.text_edit",
                    "dialect_translation_requires_expression_normalization_then_accent_conversion",
                )

    text_steps = [
        step
        for step in _dict_steps(steps)
        if str(step.get("operation") or "") == "edit_data_pipeline.text_edit"
    ]
    accent_steps = [
        step
        for step in _dict_steps(steps)
        if str(step.get("operation") or "") == "accent_conversion"
    ]

    if text_edit_available and accent_steps and not text_steps:
        for accent_step in accent_steps:
            text_step = _new_dialect_expression_text_edit_step(
                steps,
                user_request=user_request,
                context=context,
            )
            steps.insert(steps.index(accent_step), text_step)
            _make_step_depend_on(accent_step, str(text_step["id"]))
            _ensure_accent_conversion_params(accent_step, user_request=user_request, context=context)
            corrections.append(
                {
                    "step_id": accent_step.get("id"),
                    "inserted_step_id": text_step["id"],
                    "operation": "edit_data_pipeline.text_edit",
                    "reason": "dialect_translation_requires_expression_normalization_before_accent_conversion",
                }
            )
        text_steps = [
            step
            for step in _dict_steps(steps)
            if str(step.get("operation") or "") == "edit_data_pipeline.text_edit"
        ]

    if accent_available and text_steps and not accent_steps:
        text_step = text_steps[-1]
        accent_step = _new_accent_conversion_step(
            steps,
            user_request=user_request,
            context=context,
            depends_on=str(text_step.get("id") or ""),
        )
        steps.append(accent_step)
        corrections.append(
            {
                "step_id": accent_step["id"],
                "operation": "accent_conversion",
                "reason": "dialect_translation_requires_accent_conversion_after_expression_normalization",
            }
        )
    elif accent_available:
        for accent_step in accent_steps:
            _ensure_accent_conversion_params(accent_step, user_request=user_request, context=context)

    if text_edit_available:
        for text_step in text_steps:
            _ensure_dialect_expression_text_edit_params(text_step, user_request=user_request, context=context)


def _apply_foreground_background_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    for step in _dict_steps(steps):
        operation = str(step.get("operation") or "")
        if operation == "foreground_background_remix":
            params = (
                step.get("params")
                if isinstance(step.get("params"), dict)
                else {}
            )
            action = str(
                params.get("foreground_action") or "edit_foreground"
            ).strip()
            has_nested_edit = bool(
                params.get("edit_op") or params.get("edit_operation")
            )
            if (
                action == "edit_foreground"
                and not has_nested_edit
                and _requests_foreground_isolation(user_request)
            ):
                params["foreground_action"] = "isolate_foreground"
                step["params"] = params
                _append_correction(
                    corrections,
                    step,
                    operation,
                    operation,
                    "foreground_isolation_requires_isolate_action",
                )
            continue
        if "foreground_background_remix" in available and _direct_edit_needs_foreground_background_remix(user_request, operation):
            _rewrite_step_to_foreground_background_remix(
                step,
                user_request=user_request,
                context=context,
                reason="scoped_or_background_sensitive_speech_edit_requires_foreground_background_remix",
            )
            _append_correction(corrections, step, operation, "foreground_background_remix", "scoped_or_background_sensitive_speech_edit_requires_foreground_background_remix")
            continue
        if operation == "edit_data_pipeline.volume_normalize" and "artifact_ops.speaker_scoped_edit" in available:
            if _requests_speaker_specific_volume(user_request):
                _rewrite_step_to_speaker_scoped_edit(
                    step,
                    nested_operation="volume_normalize",
                    user_request=user_request,
                    context=context,
                    reason="speaker_specific_volume_requires_speaker_scoped_wrapper",
                )
                _append_correction(corrections, step, operation, "artifact_ops.speaker_scoped_edit", "speaker_specific_volume_requires_speaker_scoped_wrapper")

    _append_missing_terminal_foreground_isolation(
        steps,
        user_request=user_request,
        available=available,
        corrections=corrections,
    )


def _append_missing_terminal_foreground_isolation(
    steps: list[Any],
    *,
    user_request: str,
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Complete an explicit extract-only foreground-isolation request."""

    if "foreground_background_remix" not in available:
        return
    if not _requests_foreground_isolation(user_request):
        return
    executable_steps = _dict_steps(steps)
    if not executable_steps or any(
        str(step.get("operation") or "") == "foreground_background_remix"
        for step in executable_steps
    ):
        return
    extract_step = executable_steps[-1]
    if str(extract_step.get("operation") or "") != "artifact_ops.extract_spans":
        return
    extract_step_id = str(extract_step.get("id") or "").strip()
    if not extract_step_id:
        return

    generated_step_id = _next_step_id(steps)
    steps.append(
        {
            "id": generated_step_id,
            "operation": "foreground_background_remix",
            "description": (
                "Isolate the extracted foreground vocal and suppress instrumental "
                "accompaniment and background noise."
            ),
            "inputs": {
                "input_audio": {
                    "from_step": extract_step_id,
                    "from_output": "output_audio_path",
                }
            },
            "params": {
                "foreground_stem_policy": "speech",
                "background_stem_policy": "background",
                "foreground_action": "isolate_foreground",
                "preserve_duration": True,
            },
            "expected_output": None,
            "depends_on": [extract_step_id],
            "status": "pending",
            "required": extract_step.get("required", True),
        }
    )
    corrections.append(
        {
            "step_id": extract_step_id,
            "generated_step_id": generated_step_id,
            "operation": "foreground_background_remix",
            "reason": "explicit_foreground_isolation_requires_background_suppression_after_span_extraction",
        }
    )


def _apply_emphasis_stress_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    if "emphasis_stress_edit" not in available:
        return
    if not _requests_emphasis_stress_edit(user_request):
        return
    for step in _dict_steps(steps):
        operation = str(step.get("operation") or "")
        params = step.setdefault("params", {})
        if not isinstance(params, dict):
            params = {}
            step["params"] = params
        nested_operation = (
            _nested_operation_name(
                params.get("edit_op") or params.get("edit_operation")
            )
            if operation == "artifact_ops.speaker_scoped_edit"
            else None
        )
        if operation != "edit_data_pipeline.pitch_rate_edit" and not (
            operation == "artifact_ops.speaker_scoped_edit"
            and nested_operation == "pitch_rate_edit"
        ):
            continue
        if str(params.get("emphasis_strategy") or "").strip().lower() in {
            "pitch_rate",
            "legacy",
            "legacy_pitch_rate",
        }:
            continue
        target_phrase = _emphasis_target_phrase(params, user_request)
        if not target_phrase:
            continue
        original_params = dict(params)
        step["operation"] = "emphasis_stress_edit"
        params.clear()
        if operation == "edit_data_pipeline.pitch_rate_edit":
            params.update(original_params)
        params.setdefault("instruction", user_request)
        params.setdefault("target_phrase", target_phrase)
        params.setdefault("source_text", context.get("source_text"))
        params.setdefault("emphasis_strategy", "p1_local_resynthesis")
        nested_params = original_params.get("edit_op_params")
        params["_pitch_rate_fallback_params"] = (
            copy.deepcopy(nested_params)
            if isinstance(nested_params, dict)
            else original_params
        )
        _append_correction(
            corrections,
            step,
            operation,
            "emphasis_stress_edit",
            (
                "scoped_emphasis_prefers_p1_local_resynthesis"
                if operation == "artifact_ops.speaker_scoped_edit"
                else "emphasis_stress_prefers_p1_local_resynthesis"
            ),
        )
    _collapse_redundant_emphasis_stress_steps(
        steps,
        user_request=user_request,
        context=context,
        corrections=corrections,
    )


def _collapse_redundant_emphasis_stress_steps(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    corrections: list[dict[str, Any]],
) -> None:
    """Use the composite emphasis operation instead of a cyclic manual expansion."""

    executable_steps = _dict_steps(steps)
    if len(executable_steps) <= 1 or any(
        str(step.get("operation") or "") != "emphasis_stress_edit"
        for step in executable_steps
    ):
        return

    request_target = _emphasis_target_phrase({}, user_request)
    declared_targets = {
        str(params.get("target_phrase") or request_target or "").strip().casefold()
        for step in executable_steps
        for params in [
            step.get("params") if isinstance(step.get("params"), dict) else {}
        ]
        if str(params.get("target_phrase") or request_target or "").strip()
    }
    if len(declared_targets) != 1:
        return
    audio_paths = _context_audio_paths(context)
    if len(audio_paths) != 1:
        return

    kept = executable_steps[0]
    kept_id = str(kept.get("id") or "")
    kept_params = kept.setdefault("params", {})
    if not isinstance(kept_params, dict):
        kept_params = {}
        kept["params"] = kept_params
    for extra in executable_steps[1:]:
        extra_params = extra.get("params") if isinstance(extra.get("params"), dict) else {}
        for key in ("target_phrase", "source_text", "language"):
            value = extra_params.get(key)
            if value not in (None, "", []) and kept_params.get(key) in (None, "", []):
                kept_params[key] = value
    kept_params.setdefault("instruction", user_request)
    kept_params.setdefault("target_phrase", request_target)
    kept_params.setdefault("source_text", context.get("source_text"))
    kept_params.setdefault("emphasis_strategy", "p1_local_resynthesis")
    kept["inputs"] = {"input_audio": audio_paths[0]}
    kept["depends_on"] = []

    for extra in executable_steps[1:]:
        extra_id = str(extra.get("id") or "")
        if extra in steps:
            steps.remove(extra)
        corrections.append(
            {
                "step_id": kept_id or extra_id,
                "removed_step_id": extra_id,
                "operation": "emphasis_stress_edit",
                "reason": "global_emphasis_operation_collapses_redundant_manual_chain",
            }
        )


def _apply_speaker_scoped_boundaries(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    _split_explicit_compound_scoped_emotion_rate_steps(
        steps,
        user_request=user_request,
        available=available,
        corrections=corrections,
    )
    executable_steps = _dict_steps(steps)
    steps_by_id = {
        str(step.get("id") or ""): step
        for step in executable_steps
        if step.get("id")
    }
    for step in executable_steps:
        operation = str(step.get("operation") or "")
        if (
            operation in TRANSCRIPT_REQUIRED_EDIT_DATA_PIPELINE_OPERATIONS
            and "artifact_ops.speaker_scoped_edit" in available
            and _context_has_selected_spans(context)
            and not _step_consumes_extracted_span_artifact(step, steps_by_id)
            and (
                operation != "edit_data_pipeline.text_edit"
                or _text_edit_requires_speaker_scoped_wrapper(user_request)
            )
            and (
                _has_local_or_event_scope(user_request.lower())
                or _has_phrase_or_perceptual_scope(user_request)
            )
        ):
            nested_operation = operation.removeprefix("edit_data_pipeline.")
            _rewrite_step_to_speaker_scoped_edit(
                step,
                nested_operation=nested_operation,
                user_request=user_request,
                context=context,
                reason="reviewed_local_scope_requires_speaker_scoped_wrapper",
            )
            _append_correction(
                corrections,
                step,
                operation,
                "artifact_ops.speaker_scoped_edit",
                "reviewed_local_scope_requires_speaker_scoped_wrapper",
            )
        if step.get("operation") != "artifact_ops.speaker_scoped_edit":
            continue
        params = step.setdefault("params", {})
        if not isinstance(params, dict):
            params = {}
            step["params"] = params

        pure_artifact_operation = _pure_speaker_artifact_operation(user_request, params, available)
        if pure_artifact_operation is not None:
            _rewrite_step_to_artifact_span_operation(
                step,
                operation=pure_artifact_operation,
                user_request=user_request,
                context=context,
            )
            _append_correction(
                corrections,
                step,
                "artifact_ops.speaker_scoped_edit",
                pure_artifact_operation,
                "bare_speaker_scoped_wrapper_rewritten_to_pure_artifact_span_operation",
            )
            continue

        nested_domain = _scoped_domain_operation(user_request, available)
        nested_edit_pipeline = _scoped_edit_data_pipeline_operation(user_request)
        if nested_domain is not None:
            existing_nested = _nested_operation_name(params.get("edit_op") or params.get("edit_operation"))
            if _has_nested_operation(params) and not (
                nested_domain == "no_reference_voice_conversion"
                and existing_nested in {
                    "text_edit",
                    "pitch_rate_edit",
                    "emotion_style_edit",
                    "qwen3_tts",
                    "seed_vc",
                }
            ):
                continue
            _attach_scoped_nested_domain_operation(
                params,
                nested_operation=nested_domain,
                user_request=user_request,
                context=context,
            )
            corrections.append(
                {
                    "step_id": step.get("id"),
                    "operation": "artifact_ops.speaker_scoped_edit",
                    "nested_edit_op": params.get("edit_op"),
                    "reason": "speaker_scoped_wrapper_requires_nested_domain_operation",
                }
            )
            continue

        nested_operation = nested_edit_pipeline
        if nested_operation is None:
            continue
        if (
            nested_operation == "pitch_rate_edit"
            and "foreground_background_remix" in available
            and _direct_edit_needs_foreground_background_remix(user_request, "edit_data_pipeline.pitch_rate_edit")
        ):
            _rewrite_scoped_step_to_foreground_background_remix(
                step,
                nested_operation=nested_operation,
                user_request=user_request,
                context=context,
                reason="time_scoped_rate_edit_requires_foreground_background_remix",
            )
            _append_correction(corrections, step, "artifact_ops.speaker_scoped_edit", "foreground_background_remix", "time_scoped_rate_edit_requires_foreground_background_remix")
            continue
        if _should_rewrite_scoped_wrapper_to_direct_edit(
            user_request,
            context=context,
            nested_operation=nested_operation,
            params=params,
            step_text=" ".join(
                str(value or "")
                for value in (
                    step.get("description"),
                    step.get("expected_output"),
                )
            ),
        ) and f"edit_data_pipeline.{nested_operation}" in available:
            _rewrite_step_to_edit_data_pipeline(
                step,
                operation=nested_operation,
                user_request=user_request,
                context=context,
                reason="unscoped edit-data-pipeline operation should not be wrapped in artifact_ops.speaker_scoped_edit",
            )
            _append_correction(corrections, step, "artifact_ops.speaker_scoped_edit", f"edit_data_pipeline.{nested_operation}", "direct_edit_data_pipeline_preference")
            continue
        edit_op = params.get("edit_op") or params.get("edit_operation")
        if _is_nested_edit_data_pipeline_op(edit_op):
            continue
        params["edit_op"] = f"edit_data_pipeline.{nested_operation}"
        edit_params = params.get("edit_op_params")
        if not isinstance(edit_params, dict):
            edit_params = {}
            params["edit_op_params"] = edit_params
        edit_params.setdefault("instruction", user_request)
        if context.get("source_text"):
            edit_params.setdefault("source_text", context.get("source_text"))
        params.setdefault("selection_instruction", _speaker_scope_hint(user_request))
        corrections.append(
            {
                "step_id": step.get("id"),
                "operation": "artifact_ops.speaker_scoped_edit",
                "nested_edit_op": params["edit_op"],
                "reason": "speaker_scoped_wrapper_requires_nested_edit_data_pipeline",
            }
        )


def _split_explicit_compound_scoped_emotion_rate_steps(
    steps: list[Any],
    *,
    user_request: str,
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    if "edit_data_pipeline.pitch_rate_edit" not in available:
        return
    existing_ids = {
        str(step.get("id") or "") for step in _dict_steps(steps) if step.get("id")
    }
    for step in list(_dict_steps(steps)):
        if str(step.get("operation") or "") != "artifact_ops.speaker_scoped_edit":
            continue
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        nested_operation = _nested_operation_name(
            params.get("edit_op") or params.get("edit_operation")
        )
        if nested_operation != "emotion_style_edit":
            continue
        nested_params = params.get("edit_op_params")
        if not isinstance(nested_params, dict):
            nested_params = params.get("edit_params")
        nested_params = (
            dict(nested_params) if isinstance(nested_params, dict) else {}
        )
        rate_values = [
            value
            for container in (params, nested_params)
            for key in (
                "rate",
                "rate_factor",
                "speech_rate_factor",
                "tempo_factor",
            )
            for value in [container.get(key)]
            if value is not None
        ]
        try:
            unique_rates = {float(value) for value in rate_values}
        except (TypeError, ValueError):
            continue
        if len(unique_rates) != 1:
            continue
        rate = unique_rates.pop()
        if not 0.3 <= rate <= 3.0 or abs(rate - 1.0) < 1e-9:
            continue

        for container in (params, nested_params):
            for key in (
                "rate",
                "rate_factor",
                "speech_rate_factor",
                "tempo_factor",
            ):
                container.pop(key, None)
        if nested_params:
            params["edit_op_params"] = nested_params
            params.pop("edit_params", None)

        source_step_id = str(step.get("id") or "")
        if not source_step_id:
            continue
        generated_step_id = f"{source_step_id}_rate"
        suffix = 2
        while generated_step_id in existing_ids:
            generated_step_id = f"{source_step_id}_rate_{suffix}"
            suffix += 1
        existing_ids.add(generated_step_id)

        rate_params = {
            "instruction": user_request,
            "rate": rate,
        }
        source_text = nested_params.get("source_text") or params.get("source_text")
        if source_text:
            rate_params["source_text"] = source_text
        scoped_params = {
            key: copy.deepcopy(value)
            for key, value in params.items()
            if key
            in {
                "selected_spans",
                "spans",
                "selected_speaker_ids",
                "speaker_ids",
                "selection_instruction",
                "preserve_timeline",
            }
        }
        scoped_params.update(
            {
                "edit_op": "edit_data_pipeline.pitch_rate_edit",
                "edit_op_params": rate_params,
            }
        )
        generated = {
            "id": generated_step_id,
            "operation": "artifact_ops.speaker_scoped_edit",
            "description": f"Apply the explicitly requested speech-rate factor {rate:g} to the same reviewed scope.",
            "inputs": {
                "input_audio": {
                    "from_step": source_step_id,
                    "from_output": "output_audio_path",
                }
            },
            "params": scoped_params,
            "expected_output": None,
            "depends_on": [source_step_id],
            "status": "pending",
            "required": step.get("required", True),
        }
        steps.insert(steps.index(step) + 1, generated)
        corrections.append(
            {
                "step_id": source_step_id,
                "generated_step_id": generated_step_id,
                "operation": "artifact_ops.speaker_scoped_edit",
                "nested_edit_op": "edit_data_pipeline.pitch_rate_edit",
                "rate": rate,
                "reason": "explicit_compound_emotion_rate_split_into_scoped_steps",
            }
        )


def _promote_safe_preprocessing_before_scoped_edit_to_source_timeline(
    steps: list[Any],
    *,
    corrections: list[dict[str, Any]],
) -> None:
    """Keep full-timeline scoped wrappers off previously extracted clips."""

    dict_steps = _dict_steps(steps)
    steps_by_id = {
        str(step.get("id") or ""): step
        for step in dict_steps
        if step.get("id")
    }
    consumers = {
        step_id: [
            step
            for step in dict_steps
            if step is not producer and _step_references_id(step, step_id)
        ]
        for step_id, producer in steps_by_id.items()
    }
    for scoped in dict_steps:
        if str(scoped.get("operation") or "") != "artifact_ops.speaker_scoped_edit":
            continue
        producer = _single_audio_input_producer(scoped, steps_by_id)
        timeline_entry = scoped
        if (
            producer is not None
            and str(producer.get("operation") or "") == "speech_enhancement"
        ):
            timeline_entry = producer
            producer = _single_audio_input_producer(producer, steps_by_id)
        if producer is None or str(producer.get("operation") or "") != "artifact_ops.extract_spans":
            continue
        extract = producer
        extract_id = str(extract.get("id") or "")
        if consumers.get(extract_id) != [timeline_entry]:
            continue
        extract_spans = _explicit_step_spans(extract)
        scoped_spans = _explicit_step_spans(scoped)
        if not _same_single_span(extract_spans, scoped_spans):
            continue
        source_audio = _step_audio_input(extract)
        if source_audio in (None, ""):
            continue
        if not _replace_step_audio_input_reference(
            timeline_entry,
            old_step_id=extract_id,
            replacement=source_audio,
        ):
            continue
        dependencies = timeline_entry.get("depends_on")
        if isinstance(dependencies, list):
            timeline_entry["depends_on"] = [
                item for item in dependencies if str(item) != extract_id
            ]
        if extract in steps:
            steps.remove(extract)
        corrections.append(
            {
                "step_id": scoped.get("id"),
                "removed_step_id": extract_id,
                "timeline_entry_step_id": timeline_entry.get("id"),
                "operation": "artifact_ops.speaker_scoped_edit",
                "reason": (
                    "full_timeline_scoped_edit_requires_source_timeline_"
                    "before_safe_preprocessing"
                ),
            }
        )


def _promote_extracted_local_edit_before_scoped_edit_to_source_timeline(
    steps: list[Any],
    *,
    context: dict[str, Any],
    corrections: list[dict[str, Any]],
) -> None:
    """Run a uniquely scoped local edit chain against the source timeline."""

    dict_steps = _dict_steps(steps)
    steps_by_id = {
        str(step.get("id") or ""): step
        for step in dict_steps
        if step.get("id")
    }
    consumers = {
        step_id: [
            step
            for step in dict_steps
            if step is not producer and _step_references_id(step, step_id)
        ]
        for step_id, producer in steps_by_id.items()
    }
    supported_local_operations = {
        "edit_data_pipeline.emotion_style_edit",
        "edit_data_pipeline.pitch_rate_edit",
        "edit_data_pipeline.text_edit",
    }
    for scoped in dict_steps:
        if str(scoped.get("operation") or "") != "artifact_ops.speaker_scoped_edit":
            continue
        local_edit = _single_audio_input_producer(scoped, steps_by_id)
        if local_edit is None:
            continue
        local_operation = str(local_edit.get("operation") or "")
        if local_operation not in supported_local_operations:
            continue
        extract = _single_audio_input_producer(local_edit, steps_by_id)
        if extract is None or str(extract.get("operation") or "") != (
            "artifact_ops.extract_spans"
        ):
            continue
        extract_id = str(extract.get("id") or "")
        local_id = str(local_edit.get("id") or "")
        if consumers.get(extract_id) != [local_edit] or consumers.get(local_id) != [scoped]:
            continue
        extract_spans = _explicit_step_spans(extract)
        scoped_spans = _explicit_step_spans(scoped) or _nested_explicit_step_spans(
            scoped
        ) or _unique_reviewed_context_span(context)
        if not _same_single_span(extract_spans, scoped_spans):
            continue
        source_audio = _step_audio_input(extract)
        if source_audio in (None, ""):
            continue

        original_params = copy.deepcopy(local_edit.get("params") or {})
        scoped_params = scoped.get("params")
        scoped_params = scoped_params if isinstance(scoped_params, dict) else {}
        scoped_params["selected_spans"] = copy.deepcopy(extract_spans)
        scoped_nested_params = scoped_params.get("edit_op_params")
        if isinstance(scoped_nested_params, dict):
            scoped_nested_params.pop("selected_spans", None)
            scoped_nested_params.pop("spans", None)
        local_edit["operation"] = "artifact_ops.speaker_scoped_edit"
        local_edit["inputs"] = {"input_audio": copy.deepcopy(source_audio)}
        local_edit["params"] = {
            "selected_spans": copy.deepcopy(extract_spans),
            "selection_instruction": str(
                local_edit.get("description") or scoped_params.get("selection_instruction") or ""
            ),
            "edit_op": local_operation,
            "edit_op_params": original_params,
            "preserve_timeline": True,
        }
        for key in ("selected_speaker_ids", "speaker_ids"):
            if scoped_params.get(key):
                local_edit["params"][key] = copy.deepcopy(scoped_params[key])
        local_edit["depends_on"] = [
            item
            for item in local_edit.get("depends_on") or []
            if str(item) != extract_id
        ]
        if extract in steps:
            steps.remove(extract)
        corrections.append(
            {
                "step_id": local_id,
                "removed_step_id": extract_id,
                "downstream_step_id": scoped.get("id"),
                "from": local_operation,
                "to": "artifact_ops.speaker_scoped_edit",
                "reason": (
                    "extracted_local_edit_chain_requires_full_source_"
                    "timeline_scoped_wrappers"
                ),
            }
        )


def _single_audio_input_producer(
    step: dict[str, Any],
    steps_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    value = _step_audio_input(step)
    if not isinstance(value, dict):
        return None
    producer_id = str(value.get("from_step") or "")
    return steps_by_id.get(producer_id)


def _step_audio_input(step: dict[str, Any]) -> Any:
    inputs = step.get("inputs")
    if not isinstance(inputs, dict):
        return None
    for key in ("input_audio", "input_audio_path"):
        if key in inputs:
            return inputs[key]
    return None


def _replace_step_audio_input_reference(
    step: dict[str, Any],
    *,
    old_step_id: str,
    replacement: Any,
) -> bool:
    inputs = step.get("inputs")
    if not isinstance(inputs, dict):
        return False
    for key in ("input_audio", "input_audio_path"):
        value = inputs.get(key)
        if isinstance(value, dict) and str(value.get("from_step") or "") == old_step_id:
            inputs[key] = copy.deepcopy(replacement)
            return True
    return False


def _step_references_id(step: dict[str, Any], step_id: str) -> bool:
    if step_id in {str(item) for item in step.get("depends_on") or []}:
        return True

    def contains(value: Any) -> bool:
        if isinstance(value, dict):
            if str(value.get("from_step") or "") == step_id:
                return True
            return any(contains(item) for item in value.values())
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    return contains(step.get("inputs"))


def _explicit_step_spans(step: dict[str, Any]) -> list[Any] | None:
    params = step.get("params")
    if not isinstance(params, dict):
        return None
    spans = params.get("selected_spans") or params.get("spans")
    return spans if isinstance(spans, list) else None


def _nested_explicit_step_spans(step: dict[str, Any]) -> list[Any] | None:
    params = step.get("params")
    if not isinstance(params, dict):
        return None
    nested = params.get("edit_op_params") or params.get("edit_params")
    if not isinstance(nested, dict):
        return None
    spans = nested.get("selected_spans") or nested.get("spans")
    return spans if isinstance(spans, list) else None


def _unique_reviewed_context_span(
    context: dict[str, Any],
) -> list[Any] | None:
    report = context.get("perception_report")
    if not isinstance(report, dict):
        return None
    for key in ("selected_spans", "speaker_segments"):
        spans = report.get(key)
        if not isinstance(spans, list) or len(spans) != 1:
            continue
        span = spans[0]
        if not isinstance(span, dict):
            continue
        try:
            if float(span["end_sec"]) <= float(span["start_sec"]):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        return spans
    return None


def _same_single_span(left: list[Any] | None, right: list[Any] | None) -> bool:
    if not left or not right or len(left) != 1 or len(right) != 1:
        return False
    if not isinstance(left[0], dict) or not isinstance(right[0], dict):
        return False
    try:
        return (
            float(left[0]["start_sec"]),
            float(left[0]["end_sec"]),
        ) == (
            float(right[0]["start_sec"]),
            float(right[0]["end_sec"]),
        )
    except (KeyError, TypeError, ValueError):
        return False


def _step_consumes_extracted_span_artifact(
    step: dict[str, Any],
    steps_by_id: dict[str, dict[str, Any]],
) -> bool:
    def _contains_extract_ref(value: Any) -> bool:
        if isinstance(value, dict):
            producer_id = str(value.get("from_step") or "")
            producer = steps_by_id.get(producer_id)
            if (
                producer is not None
                and str(producer.get("operation") or "")
                == "artifact_ops.extract_spans"
            ):
                return True
            return any(_contains_extract_ref(child) for child in value.values())
        if isinstance(value, list):
            return any(_contains_extract_ref(child) for child in value)
        return False

    return _contains_extract_ref(step.get("inputs"))


def _dict_steps(steps: list[Any]) -> list[dict[str, Any]]:
    return [step for step in steps if isinstance(step, dict)]


def _ensure_source_text_perception_for_text_edits(
    steps: list[Any],
    *,
    data: dict[str, Any],
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Transcript-conditioned edit-data-pipeline operations need source_text."""

    transcript_operations = TRANSCRIPT_REQUIRED_EDIT_DATA_PIPELINE_OPERATIONS & available
    if not transcript_operations:
        return
    if _has_source_text(context) or _plan_has_source_text(data):
        return
    transcript_steps = [
        step
        for step in _dict_steps(steps)
        if str(step.get("operation") or "") in transcript_operations
        and not _step_has_source_text(step)
    ]
    if not transcript_steps:
        return

    perception_step = _existing_source_text_perception_step(steps)
    if perception_step is None:
        perception_operation = None
        if "perception.speech_context" in available:
            perception_operation = "perception.speech_context"
        elif "perception.audio_fallback" in available:
            perception_operation = "perception.audio_fallback"
        elif "perception.gemini_audio" in available:
            perception_operation = "perception.gemini_audio"
        if perception_operation is None:
            corrections.append(
                {
                    "operations": sorted({str(step.get("operation") or "") for step in transcript_steps}),
                    "reason": "edit_data_pipeline_requires_source_text_but_no_perception_operation_available",
                    "missing": ["source_text", "perception.speech_context"],
                }
            )
            return
        perception_step = {
            "id": _next_step_id(steps),
            "operation": perception_operation,
            "description": "Transcribe the source speech before transcript-conditioned editing.",
            "inputs": {"input_audio": "current"},
            "params": {
                "instruction": user_request,
                "need_source_text": True,
                "need_words": False,
                "need_speakers": False,
                "need_speaker_profiles": False,
                "allow_gemini": False,
            },
            "depends_on": [],
            "status": "pending",
        }
        first_transcript_index = min(steps.index(step) for step in transcript_steps if step in steps)
        steps.insert(first_transcript_index, perception_step)

    perception_step_id = str(perception_step.get("id") or "")
    affected_steps = []
    for step in transcript_steps:
        depends = step.get("depends_on")
        if isinstance(depends, list) and perception_step_id in {str(item) for item in depends}:
            continue
        _make_step_depend_on(step, perception_step_id)
        affected_steps.append(step)
    if not affected_steps:
        return
    corrections.append(
        {
            "step_id": perception_step_id,
            "operation": perception_step.get("operation"),
            "affected_transcript_required_steps": [step.get("id") for step in affected_steps],
            "affected_operations": sorted({str(step.get("operation") or "") for step in affected_steps}),
            "reason": "inserted_source_text_perception_before_transcript_required_edit_without_transcript",
        }
    )


def _has_source_text(container: Any) -> bool:
    if not isinstance(container, dict):
        return False
    value = container.get("source_text")
    if isinstance(value, str) and value.strip():
        return True
    perception = container.get("perception_report")
    if isinstance(perception, dict):
        if isinstance(perception.get("source_text"), str) and perception["source_text"].strip():
            return True
        gemini = perception.get("gemini_perception")
        if isinstance(gemini, dict) and isinstance(gemini.get("source_text"), str) and gemini["source_text"].strip():
            return True
    return False


def _plan_has_source_text(data: dict[str, Any]) -> bool:
    global_params = data.get("global_params")
    return _has_source_text(global_params)


def _step_has_source_text(step: dict[str, Any]) -> bool:
    params = step.get("params")
    return _has_source_text(params)


def _scoped_wrapper_has_source_text_for_direct_rewrite(context: dict[str, Any], params: dict[str, Any]) -> bool:
    if _has_source_text(context) or _has_source_text(params):
        return True
    for key in ("edit_op_params", "edit_params"):
        nested = params.get(key)
        if _has_source_text(nested):
            return True
    return False


def _existing_source_text_perception_step(steps: list[Any]) -> dict[str, Any] | None:
    for step in _dict_steps(steps):
        operation = str(step.get("operation") or "")
        if operation not in {
            "perception.speech_context",
            "perception.audio_fallback",
            "perception.gemini_audio",
            "perception.speaker_attributed_review",
        }:
            continue
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        if params.get("need_source_text") is False:
            params["need_source_text"] = True
        return step
    return None


def _ensure_speaker_attributed_review_for_artifact_span_grounding(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    available: set[str],
    corrections: list[dict[str, Any]],
) -> None:
    """Insert strong speech/speaker grounding before speaker-defined span edits."""

    if "perception.speaker_attributed_review" not in available:
        return
    if _context_has_reviewed_speaker_attributed_evidence(context):
        return

    target_steps = [
        step
        for step in _dict_steps(steps)
        if _artifact_step_needs_speaker_attributed_review(step, user_request=user_request)
    ]
    if not target_steps:
        return

    perception_step = _existing_speaker_attributed_review_step(steps)
    if perception_step is None:
        perception_step = {
            "id": _next_step_id(steps),
            "operation": "perception.speaker_attributed_review",
            "description": "Review the source speech with speaker-attributed transcript evidence before span selection.",
            "inputs": {"input_audio": "current"},
            "params": {
                "instruction": user_request,
                "need_source_text": True,
                "need_words": True,
                "need_speakers": True,
                "need_speaker_profiles": True,
                "grounding_goal": "speech_speaker_span_selection",
            },
            "depends_on": [],
            "status": "pending",
        }
        first_target_index = min(steps.index(step) for step in target_steps if step in steps)
        steps.insert(first_target_index, perception_step)
    else:
        _ensure_speaker_attributed_review_params(perception_step, user_request=user_request)

    perception_step_id = str(perception_step.get("id") or "")
    affected_steps = []
    for step in target_steps:
        depends = step.get("depends_on")
        if isinstance(depends, list) and perception_step_id in {str(item) for item in depends}:
            continue
        _make_step_depend_on(step, perception_step_id)
        affected_steps.append(step)
    if not affected_steps:
        return
    corrections.append(
        {
            "step_id": perception_step_id,
            "operation": "perception.speaker_attributed_review",
            "affected_artifact_steps": [step.get("id") for step in affected_steps],
            "affected_operations": sorted({str(step.get("operation") or "") for step in affected_steps}),
            "reason": "inserted_speaker_attributed_review_before_speaker_span_artifact_operation",
        }
    )


def _existing_speaker_attributed_review_step(steps: list[Any]) -> dict[str, Any] | None:
    for step in _dict_steps(steps):
        if str(step.get("operation") or "") != "perception.speaker_attributed_review":
            continue
        return step
    return None


def _ensure_speaker_attributed_review_params(step: dict[str, Any], *, user_request: str) -> None:
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    params.setdefault("instruction", user_request)
    params["need_source_text"] = True
    params["need_words"] = True
    params["need_speakers"] = True
    params["need_speaker_profiles"] = True
    params.setdefault("grounding_goal", "speech_speaker_span_selection")
    step["params"] = params


def _artifact_step_needs_speaker_attributed_review(step: dict[str, Any], *, user_request: str) -> bool:
    operation = str(step.get("operation") or "")
    if operation not in SPEAKER_GROUNDED_ARTIFACT_OPERATIONS:
        return False
    if _artifact_step_has_explicit_spans(step):
        return False
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    evidence = _span_selection_evidence_text(user_request, params)
    return _span_selection_needs_speaker_attributed_review(evidence)


def _artifact_step_has_explicit_spans(step: dict[str, Any]) -> bool:
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    for key in ("selected_spans", "spans", "time_spans", "target_spans"):
        value = params.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _span_selection_evidence_text(user_request: str, params: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in [
            user_request,
            params.get("instruction"),
            params.get("selection_instruction"),
            params.get("speaker_selector"),
            params.get("selector"),
            params.get("grounding_instruction"),
            params.get("target_description"),
        ]
        if value
    )


def _span_selection_needs_speaker_attributed_review(text: str) -> bool:
    normalized = " ".join(str(text or "").casefold().split())
    if not normalized:
        return False
    if re.search(r"[\"“][^\"”]+[\"”]|'[^']+'|‘[^’]+’|「[^」]+」|『[^』]+』", normalized):
        return True
    return _contains_any(
        normalized,
        [
            "speaker",
            "speakers",
            "talker",
            "voice of",
            "spoken by",
            "speech of",
            "speech from",
            "words said by",
            "sentence spoken by",
            "phrase spoken by",
            "speaker says",
            "says",
            "said",
            "utterance",
            "question",
            "answer",
            "interviewer",
            "respondent",
            "host",
            "guest",
            "narrator",
            "caller",
            "child speaker",
            "child voice",
            "kid voice",
            "baby speaker",
            "baby voice",
            "boy voice",
            "girl voice",
            "male speaker",
            "female speaker",
            "male voice",
            "female voice",
            "first speaker",
            "second speaker",
            "final speaker",
            "last speaker",
            "first sentence",
            "second sentence",
            "final sentence",
            "last sentence",
            "quoted phrase",
            "spoken phrase",
            "spoken word",
            "transcript",
            "说话人",
            "讲话人",
            "男声",
            "女声",
            "童声",
            "儿童",
            "小孩",
            "第一个说话人",
            "第二个说话人",
            "最后一个说话人",
            "问题",
            "回答",
            "台词",
        ],
    )


def _context_has_reviewed_speaker_attributed_evidence(context: dict[str, Any]) -> bool:
    """Return true only for reviewed speaker-attributed evidence, not raw diarization priors."""

    containers: list[Any] = [context]
    perception = context.get("perception_report")
    if isinstance(perception, dict):
        containers.append(perception)
        for key in (
            "gemini_perception",
            "qwen3_omni",
            "open_omni",
            "speaker_attributed_review",
        ):
            nested = perception.get(key)
            if isinstance(nested, dict):
                containers.append(nested)
    return any(_container_has_reviewed_speaker_attributed_evidence(container) for container in containers)


def _container_has_reviewed_speaker_attributed_evidence(container: Any) -> bool:
    if not isinstance(container, dict):
        return False
    if _review_status_is_success(container.get("speaker_attributed_review")) and _container_has_speaker_span_items(container):
        return True
    for key in ("speaker_segments", "segments", "turns", "utterances"):
        raw = container.get(key)
        if not isinstance(raw, list):
            continue
        if any(_segment_has_time_anchor(item) and _segment_source_is_speaker_attributed_review(item) for item in raw):
            return True
    return False


def _review_status_is_success(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    status = str(raw.get("status") or raw.get("verification_status") or "").strip().lower()
    if status in {"success", "succeeded", "passed", "ok", "verified"}:
        return True
    verification = raw.get("verification")
    return isinstance(verification, dict) and verification.get("passed") is True


def _container_has_speaker_span_items(container: dict[str, Any]) -> bool:
    for key in ("speaker_segments", "segments", "turns", "utterances"):
        raw = container.get(key)
        if isinstance(raw, list) and any(_segment_has_time_anchor(item) for item in raw):
            return True
    return False


def _segment_has_time_anchor(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("start_sec") is not None and item.get("end_sec") is not None:
        return True
    if item.get("start") is not None and item.get("end") is not None:
        return True
    timestamps = item.get("timestamps")
    return isinstance(timestamps, dict) and timestamps.get("start_sec") is not None and timestamps.get("end_sec") is not None


def _segment_source_is_speaker_attributed_review(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    evidence = " ".join(
        str(item.get(key) or "")
        for key in ("source", "evidence_source", "profile_source", "span_source")
    ).strip().lower()
    return "speaker_attributed_review" in evidence


def _append_correction(
    corrections: list[dict[str, Any]],
    step: dict[str, Any],
    source: str,
    target: str,
    reason: str,
) -> None:
    corrections.append({"step_id": step.get("id"), "from": source, "to": target, "reason": reason})


def _ensure_paralinguistic_event_params(step: dict[str, Any], *, user_request: str, event: str) -> None:
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    params.setdefault("event", event)
    params.setdefault("instruction", user_request)
    step["params"] = params


def _rewrite_step_to_paralinguistic_event(
    step: dict[str, Any],
    *,
    operation: str,
    event: str,
    user_request: str,
    context: dict[str, Any],
) -> None:
    step["operation"] = operation
    step["description"] = (
        f"{operation} for {event}; use event insertion/local span removal, not text_edit or whole-utterance TTS"
    )
    _ensure_paralinguistic_event_params(step, user_request=user_request, event=event)


def _step_mentions_paralinguistic_event(step: dict[str, Any], event: str) -> bool:
    text_parts = [
        step.get("description"),
        step.get("operation"),
        step.get("expected_output"),
    ]
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    text_parts.extend(params.get(key) for key in ("instruction", "event", "target_event", "selection_instruction"))
    haystack = " ".join(str(item or "") for item in text_parts)
    return normalize_paralinguistic_event(haystack) == event


def _rewrite_step_to_edit_data_pipeline(
    step: dict[str, Any],
    *,
    operation: str,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    step["operation"] = f"edit_data_pipeline.{operation}"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    nested_params = original_params.get("edit_op_params")
    if not isinstance(nested_params, dict):
        nested_params = original_params.get("edit_params")
    params = dict(nested_params) if isinstance(nested_params, dict) else {}
    filtered_parent_params = {
        key: value
        for key, value in original_params.items()
        if key
        not in {
            "edit_op",
            "edit_operation",
            "edit_op_params",
            "edit_params",
            "instruct",
            "mode",
            "ref_audio_path",
            "reference_audio_path",
            "selection_instruction",
            "selector",
            "speaker",
            "target_events",
            "voice_prompt",
        }
    }
    params.update(filtered_parent_params)
    if operation == "volume_normalize":
        params = _sanitize_volume_normalize_params(params)
    if isinstance(nested_params, dict) and nested_params.get("instruction"):
        params["instruction"] = user_request
    else:
        params.setdefault("instruction", user_request)
    if context.get("source_text"):
        params.setdefault("source_text", context.get("source_text"))
    step["params"] = params


def _rewrite_step_to_dialect_expression_text_edit(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
) -> None:
    _rewrite_step_to_edit_data_pipeline(
        step,
        operation="text_edit",
        user_request=user_request,
        context=context,
        reason="Normalize localized/dialectal wording before accent or dialect conversion.",
    )
    _ensure_dialect_expression_text_edit_params(step, user_request=user_request, context=context)


def _ensure_dialect_expression_text_edit_params(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
) -> None:
    step["operation"] = "edit_data_pipeline.text_edit"
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    params.setdefault("instruction", user_request)
    params.setdefault(
        "dialect_translation_stage",
        "regional_expression_normalization",
    )
    params.setdefault(
        "normalization_goal",
        _dialect_expression_normalization_goal(user_request),
    )
    if context.get("source_text"):
        params.setdefault("source_text", context.get("source_text"))
    if context.get("target_text"):
        params.setdefault("target_text", context.get("target_text"))
    # Wording is a Planner decision; do not synthesize a target from a lexicon.
    step["params"] = params


def _collapse_redundant_accent_conversion_steps(
    steps: list[Any],
    *,
    corrections: list[dict[str, Any]],
) -> None:
    dict_steps = _dict_steps(steps)
    executable_steps = [
        step
        for step in dict_steps
        if str(step.get("operation") or "") not in {"perception.speech_context", "perception.audio_fallback"}
    ]
    accent_steps = [
        step for step in executable_steps if str(step.get("operation") or "") == "accent_conversion"
    ]
    if len(accent_steps) <= 1 or len(accent_steps) != len(executable_steps):
        return

    kept = accent_steps[0]
    kept_id = str(kept.get("id") or "")
    kept_params = kept.setdefault("params", {})
    if not isinstance(kept_params, dict):
        kept_params = {}
        kept["params"] = kept_params

    for extra in accent_steps[1:]:
        extra_id = str(extra.get("id") or "")
        extra_params = extra.get("params") if isinstance(extra.get("params"), dict) else {}
        for key, value in extra_params.items():
            if key not in kept_params or kept_params.get(key) in (None, "", []):
                kept_params[key] = value
        if extra.get("description") and not kept.get("description"):
            kept["description"] = extra.get("description")
        if extra in steps:
            steps.remove(extra)
        if extra_id and kept_id:
            _replace_dependency(steps, old_step_id=extra_id, new_step_id=kept_id)
        corrections.append(
            {
                "step_id": kept_id or extra_id,
                "removed_step_id": extra_id,
                "operation": "accent_conversion",
                "reason": "pure_accent_conversion_collapses_redundant_duplicate_steps",
            }
        )


def _collapse_redundant_no_reference_voice_conversion_steps(
    steps: list[Any],
    *,
    corrections: list[dict[str, Any]],
) -> None:
    dict_steps = _dict_steps(steps)
    executable_steps = [
        step
        for step in dict_steps
        if not str(step.get("operation") or "").startswith(("perception.", "flexsed."))
    ]
    conversion_steps = [
        step
        for step in executable_steps
        if str(step.get("operation") or "") == "no_reference_voice_conversion"
    ]
    if len(conversion_steps) <= 1 or len(conversion_steps) != len(executable_steps):
        return

    kept = conversion_steps[0]
    kept_id = str(kept.get("id") or "")
    kept_params = kept.setdefault("params", {})
    if not isinstance(kept_params, dict):
        kept_params = {}
        kept["params"] = kept_params
    for extra in conversion_steps[1:]:
        extra_id = str(extra.get("id") or "")
        extra_params = extra.get("params") if isinstance(extra.get("params"), dict) else {}
        for key, value in extra_params.items():
            if key not in kept_params or kept_params.get(key) in (None, "", []):
                kept_params[key] = value
        if extra in steps:
            steps.remove(extra)
        if extra_id and kept_id:
            _replace_dependency(steps, old_step_id=extra_id, new_step_id=kept_id)
        corrections.append(
            {
                "step_id": kept_id or extra_id,
                "removed_step_id": extra_id,
                "operation": "no_reference_voice_conversion",
                "reason": "global_voice_conversion_collapses_redundant_duplicate_steps",
            }
        )


def _collapse_redundant_terminal_speaker_scoped_reconstruct(
    steps: list[Any],
    *,
    corrections: list[dict[str, Any]],
) -> None:
    """Drop a terminal reconstruct that treats a full-timeline edit as a clip."""

    dict_steps = _dict_steps(steps)
    if len(dict_steps) < 2:
        return
    reconstruct = dict_steps[-1]
    if str(reconstruct.get("operation") or "") != "artifact_ops.reconstruct":
        return
    params = reconstruct.get("params") if isinstance(reconstruct.get("params"), dict) else {}
    inputs = reconstruct.get("inputs") if isinstance(reconstruct.get("inputs"), dict) else {}
    replacement = (
        params.get("replacement_audio")
        or params.get("replacement_audio_path")
        or inputs.get("replacement_audio")
        or inputs.get("replacement_audio_path")
    )
    if not isinstance(replacement, dict):
        return
    replacement_step_id = str(replacement.get("from_step") or "")
    if not replacement_step_id:
        return
    replacement_step = next(
        (
            step
            for step in dict_steps[:-1]
            if str(step.get("id") or "") == replacement_step_id
        ),
        None,
    )
    if replacement_step is None or str(replacement_step.get("operation") or "") != "artifact_ops.speaker_scoped_edit":
        return
    if reconstruct in steps:
        steps.remove(reconstruct)
    corrections.append(
        {
            "step_id": replacement_step_id,
            "removed_step_id": reconstruct.get("id"),
            "operation": "artifact_ops.reconstruct",
            "reason": "speaker_scoped_edit_already_returns_full_timeline",
        }
    )


def _normalize_extract_then_reorder(
    steps: list[Any],
    *,
    corrections: list[dict[str, Any]],
) -> None:
    """Reorder grounded source spans on the source timeline, not an extracted clip."""

    dict_steps = _dict_steps(steps)
    by_id = {str(step.get("id") or ""): step for step in dict_steps}
    for reorder in dict_steps:
        if str(reorder.get("operation") or "") != "artifact_ops.reorder_spans":
            continue
        dependencies = reorder.get("depends_on")
        if not isinstance(dependencies, list) or len(dependencies) != 1:
            continue
        extract_id = str(dependencies[0])
        extract = by_id.get(extract_id)
        if extract is None or str(extract.get("operation") or "") != "artifact_ops.extract_spans":
            continue
        if any(
            extract_id in (step.get("depends_on") or [])
            for step in dict_steps
            if step is not reorder
        ):
            continue
        extract_params = extract.get("params") if isinstance(extract.get("params"), dict) else {}
        spans = extract_params.get("selected_spans") or extract_params.get("spans")
        if not isinstance(spans, list) or len(spans) < 2:
            continue
        if not all(
            isinstance(span, dict)
            and span.get("start_sec") is not None
            and span.get("end_sec") is not None
            for span in spans
        ):
            continue
        ordered_spans = sorted(
            (dict(span) for span in spans),
            key=lambda span: (float(span["start_sec"]), float(span["end_sec"])),
        )
        span_index = {
            str(span.get("span_id")): index
            for index, span in enumerate(ordered_spans)
            if span.get("span_id")
        }
        reorder_params = reorder.get("params") if isinstance(reorder.get("params"), dict) else {}
        sequence = reorder_params.pop("reorder_sequence", None)
        if isinstance(sequence, list) and sequence:
            sequence_ids = [
                str(item.get("span_id") or "")
                for item in sequence
                if isinstance(item, dict)
            ]
            if len(sequence_ids) != len(ordered_spans) or any(
                span_id not in span_index for span_id in sequence_ids
            ):
                continue
            reorder_params["order"] = [span_index[span_id] for span_id in sequence_ids]
        reorder_params["selected_spans"] = ordered_spans
        reorder["params"] = reorder_params
        reorder["inputs"] = (
            dict(extract.get("inputs"))
            if isinstance(extract.get("inputs"), dict)
            else {"input_audio": "current"}
        )
        corrections.append(
            {
                "step_id": reorder.get("id"),
                "source_step_id": extract_id,
                "operation": "artifact_ops.reorder_spans",
                "reason": "reorder_grounded_spans_on_source_timeline",
            }
        )


def _replace_dependency(steps: list[Any], *, old_step_id: str, new_step_id: str) -> None:
    if not old_step_id or not new_step_id or old_step_id == new_step_id:
        return
    for step in _dict_steps(steps):
        depends_on = step.get("depends_on")
        if not isinstance(depends_on, list):
            continue
        updated: list[Any] = []
        for dependency in depends_on:
            replacement = new_step_id if str(dependency) == old_step_id else dependency
            if replacement not in updated:
                updated.append(replacement)
        step["depends_on"] = updated


def _new_dialect_expression_text_edit_step(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "id": _next_step_id(steps),
        "operation": "edit_data_pipeline.text_edit",
        "description": "Normalize localized or dialectal expressions before accent conversion.",
        "inputs": {"input_audio": "current"},
        "params": {},
        "depends_on": [],
        "status": "pending",
    }
    _ensure_dialect_expression_text_edit_params(step, user_request=user_request, context=context)
    return step


def _new_accent_conversion_step(
    steps: list[Any],
    *,
    user_request: str,
    context: dict[str, Any],
    depends_on: str,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "id": _next_step_id(steps),
        "operation": "accent_conversion",
        "description": "Convert the normalized speech to the requested accent or dialect.",
        "inputs": {"input_audio": "current"},
        "params": {},
        "depends_on": [depends_on] if depends_on else [],
        "status": "pending",
    }
    _ensure_accent_conversion_params(step, user_request=user_request, context=context)
    return step


def _ensure_accent_conversion_params(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
) -> None:
    step["operation"] = "accent_conversion"
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    reference_audio = _reference_audio_from_context(context)
    if reference_audio:
        inputs.setdefault("reference_audio_path", reference_audio)
    step["inputs"] = inputs
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    params.setdefault("instruction", user_request)
    params.setdefault("preserve_timeline", True)
    params.setdefault("dialect_translation_stage", "accent_or_dialect_conversion")
    target_accent = _requested_accent_label(user_request) or _dialect_translation_target_accent_label(user_request)
    if target_accent:
        params.setdefault("target_accent", target_accent)
    if reference_audio:
        params.setdefault("reference_audio_path", reference_audio)
    step["params"] = params


def _next_step_id(steps: list[Any]) -> str:
    existing = {str(step.get("id")) for step in _dict_steps(steps) if step.get("id")}
    index = 1
    while f"step_{index}" in existing:
        index += 1
    return f"step_{index}"


def _make_step_depend_on(step: dict[str, Any], dependency: str) -> None:
    if not dependency:
        return
    depends_on = step.get("depends_on")
    if not isinstance(depends_on, list):
        depends_on = []
    if dependency not in depends_on:
        depends_on.insert(0, dependency)
    step["depends_on"] = depends_on


def _rewrite_step_to_foreground_background_remix(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    original_operation = str(step.get("operation") or "")
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    nested_params = dict(original_params)
    nested_params.setdefault("instruction", user_request)
    if context.get("source_text"):
        nested_params.setdefault("source_text", context.get("source_text"))
    _repair_normal_speech_emotion_params(nested_params, user_request=user_request, context=context)
    step["operation"] = "foreground_background_remix"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "edit_op": original_operation,
        "edit_op_params": nested_params,
        "foreground_stem_policy": "speech",
        "background_stem_policy": "background",
        "foreground_action": "edit_foreground",
        "preserve_duration": True,
    }


def _rewrite_scoped_step_to_foreground_background_remix(
    step: dict[str, Any],
    *,
    nested_operation: str,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    edit_params = original_params.get("edit_op_params")
    if not isinstance(edit_params, dict):
        edit_params = original_params.get("edit_params")
    nested_params = dict(edit_params) if isinstance(edit_params, dict) else {}
    nested_params.setdefault("instruction", user_request)
    if context.get("source_text"):
        nested_params.setdefault("source_text", context.get("source_text"))
    step["operation"] = "foreground_background_remix"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "edit_op": f"edit_data_pipeline.{nested_operation}",
        "edit_op_params": nested_params,
        "foreground_stem_policy": "speech",
        "background_stem_policy": "background",
        "foreground_action": "edit_foreground",
        "preserve_duration": True,
    }


def _rewrite_step_to_speaker_scoped_edit(
    step: dict[str, Any],
    *,
    nested_operation: str,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    edit_params = dict(original_params)
    if nested_operation == "volume_normalize":
        edit_params = _sanitize_volume_normalize_params(edit_params)
    edit_params.setdefault("instruction", user_request)
    if context.get("source_text"):
        edit_params.setdefault("source_text", context.get("source_text"))
    step["operation"] = "artifact_ops.speaker_scoped_edit"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    step["params"] = {
        "selection_instruction": _speaker_scope_hint(user_request),
        "edit_op": f"edit_data_pipeline.{nested_operation}",
        "edit_op_params": edit_params,
        "preserve_timeline": True,
    }


def _rewrite_step_to_seamless_s2st(
    step: dict[str, Any],
    *,
    user_request: str,
    target_language: str,
    context: dict[str, Any],
) -> None:
    step["operation"] = "seamless_expressive_s2st"
    step["description"] = step.get("description") or "Translate the source speech with speech-to-speech translation."
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio_path", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "target_language": target_language,
        "source_language": _source_language_code(user_request),
        "mode": "expressive" if _wants_expressive_preservation(user_request) else "m4t_v2",
        "preserve_vocal_style": _preserve_vocal_style(user_request),
        "preserve_speech_rate": _preserve_speech_rate(user_request),
        "preserve_pauses": _preserve_pauses(user_request),
        "allow_fallback_to_m4t": not _wants_expressive_preservation(user_request),
    }
    if context.get("source_text"):
        step["params"].setdefault("source_text", context.get("source_text"))


def _ensure_seamless_params(step: dict[str, Any], *, user_request: str, target_language: str) -> None:
    params = step.setdefault("params", {})
    if not isinstance(params, dict):
        params = {}
        step["params"] = params
    params.setdefault("instruction", user_request)
    params.setdefault("target_language", target_language)
    params.setdefault("mode", "expressive" if _wants_expressive_preservation(user_request) else "m4t_v2")
    params.setdefault("preserve_vocal_style", _preserve_vocal_style(user_request))
    params.setdefault("preserve_speech_rate", _preserve_speech_rate(user_request))
    params.setdefault("preserve_pauses", _preserve_pauses(user_request))


def _rewrite_step_to_accent_conversion(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    step["operation"] = "accent_conversion"
    step["description"] = step.get("description") or "Convert the requested speech accent or dialect."
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    reference_audio = _reference_audio_from_context(context)
    if reference_audio:
        inputs.setdefault("reference_audio_path", reference_audio)
    step["inputs"] = inputs
    params = {"instruction": user_request, "preserve_timeline": True}
    target_accent = _requested_accent_label(user_request)
    if target_accent:
        params["target_accent"] = target_accent
    if reference_audio:
        params["reference_audio_path"] = reference_audio
    step["params"] = params


def _rewrite_step_to_no_reference_voice_conversion(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    step["operation"] = "no_reference_voice_conversion"
    step["description"] = step.get("description") or "Convert the existing source audio into the requested target voice."
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "target_voice_description": _target_voice_description(user_request),
        "preserve_words": True,
        "preserve_timeline": True,
    }
    if context.get("source_text"):
        step["params"]["source_text"] = context.get("source_text")


def _ensure_no_reference_voice_conversion_params(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
) -> None:
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    params.setdefault("instruction", user_request)
    params.setdefault("target_voice_description", _target_voice_description(user_request))
    params.setdefault("preserve_words", True)
    params.setdefault("preserve_timeline", True)
    if context.get("source_text"):
        params.setdefault("source_text", context.get("source_text"))
    step["params"] = params


def _rewrite_step_to_spatial_reverb(step: dict[str, Any], *, user_request: str) -> None:
    step["operation"] = "spatial_reverb"
    step["description"] = step.get("description") or "Apply the requested room or spatial reverb."
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio_path", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "room_preset": _spatial_reverb_room_preset(user_request),
    }


def _ensure_spatial_reverb_params(step: dict[str, Any], *, user_request: str) -> None:
    inputs = step.setdefault("inputs", {})
    if isinstance(inputs, dict):
        inputs.setdefault("input_audio_path", "current")
    params = step.setdefault("params", {})
    if isinstance(params, dict):
        params.setdefault("instruction", user_request)
        params.setdefault("room_preset", _spatial_reverb_room_preset(user_request))


def _rewrite_step_to_audio_effects(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    step["operation"] = "audio_effects"
    step["description"] = step.get("description") or "Apply the requested acoustic/channel effect."
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio_path", "current")
    reference_audio = _reference_audio_from_context(context)
    if reference_audio:
        inputs.setdefault("reference_audio_path", reference_audio)
    step["inputs"] = inputs
    params = {"instruction": user_request, "preset": _audio_effect_preset(user_request), "strength": 0.72}
    if reference_audio:
        params["reference_audio_path"] = reference_audio
    step["params"] = params


def _rewrite_step_to_concat(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    step["operation"] = "artifact_ops.concat"
    step["description"] = step.get("description") or "Concatenate the requested audio inputs."
    step["inputs"] = {"input_audio_paths": _context_audio_paths(context) or ["current"]}
    step["params"] = {"instruction": user_request}
    pause = _infer_pause_duration_seconds(user_request)
    if pause is not None:
        step["params"]["pause_duration_sec"] = pause


def _ensure_concat_params(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    inputs = step.setdefault("inputs", {})
    if isinstance(inputs, dict) and not inputs.get("input_audio_paths"):
        inputs["input_audio_paths"] = _context_audio_paths(context) or ["current"]
    params = step.setdefault("params", {})
    if isinstance(params, dict):
        params.setdefault("instruction", user_request)
        pause = _infer_pause_duration_seconds(user_request)
        if pause is not None:
            params.setdefault("pause_duration_sec", pause)


def _rewrite_step_to_silence(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    step["operation"] = "artifact_ops.silence"
    step["description"] = step.get("description") or "Generate deterministic silence."
    step["inputs"] = {}
    step["params"] = {"duration_sec": _infer_pause_duration_seconds(user_request) or 1.0}


def _rewrite_step_to_pause_edit(
    step: dict[str, Any],
    *,
    source_text: str,
    instruction_st: str,
    arguments: dict[str, Any],
) -> None:
    original_inputs = step.get("inputs")
    inputs = dict(original_inputs) if isinstance(original_inputs, dict) else {}
    if not any(
        key in inputs
        for key in ("input_audio", "input_audio_path", "audio", "audio_path")
    ):
        inputs["input_audio_path"] = "current"
    step["operation"] = "edit_data_pipeline.pause_edit"
    step["inputs"] = inputs
    step["params"] = {
        "source_text": source_text,
        "instruction_st": instruction_st,
        **dict(arguments),
    }


def _rewrite_step_to_reconstruct(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    mode: str = "insert",
) -> None:
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    target_audio = (
        original_params.get("target_audio_path")
        or original_params.get("insert_audio_path")
        or original_params.get("generated_audio_path")
        or original_params.get("replacement_audio_path")
        or original_params.get("overlay_audio_path")
        or original_params.get("overlay_audio")
        or original_params.get("audio_path")
        or _reference_audio_from_context(context)
    )
    insert_time = original_params.get(
        "insert_time_sec",
        original_params.get(
            "time_sec",
            original_params.get("start_sec", _infer_timeline_position(user_request, default="midpoint" if mode == "overlay" else 0.0)),
        ),
    )
    edit_mode = "mix" if mode == "overlay" else str(original_params.get("mode") or "insert")
    step["operation"] = "artifact_ops.reconstruct"
    step["description"] = step.get("description") or "Reconstruct the timeline with an external audio artifact."
    step["inputs"] = {"input_audio_path": "current"}
    step["params"] = {
        "instruction": user_request,
        "edits": [
            {
                "start_sec": insert_time,
                "end_sec": insert_time,
                "input_audio_path": target_audio,
                "mode": edit_mode,
            }
        ],
    }


def _ensure_reconstruct_overlay_params(step: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    inputs = step.setdefault("inputs", {})
    if isinstance(inputs, dict):
        inputs.setdefault("input_audio_path", "current")
    params = step.setdefault("params", {})
    if not isinstance(params, dict):
        params = {}
        step["params"] = params
    params.setdefault("instruction", user_request)
    edits = params.get("edits")
    if not isinstance(edits, list) or not edits:
        target_audio = _reference_audio_from_context(context)
        params["edits"] = [
            {
                "start_sec": _infer_timeline_position(user_request, default="midpoint"),
                "end_sec": _infer_timeline_position(user_request, default="midpoint"),
                "input_audio_path": target_audio,
                "mode": "mix",
            }
        ]
        return
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        edit.setdefault("input_audio_path", _reference_audio_from_context(context))
        position = _infer_timeline_position(user_request, default="midpoint")
        edit.setdefault("start_sec", position)
        edit.setdefault("end_sec", edit.get("start_sec"))
        edit.setdefault("mode", "mix")


def _rewrite_step_to_artifact_span_operation(
    step: dict[str, Any],
    *,
    operation: str,
    user_request: str,
    context: dict[str, Any],
) -> None:
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    selection = (
        original_params.get("selection_instruction")
        or original_params.get("speaker_selector")
        or original_params.get("selector")
        or _speaker_scope_hint(user_request)
    )
    step["operation"] = operation
    step["description"] = step.get("description") or "Apply the requested speaker/timeline span operation."
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio_path", "current")
    step["inputs"] = inputs
    params = {
        "instruction": user_request,
        "selection_instruction": selection,
    }
    if context.get("source_text"):
        params["source_text"] = context.get("source_text")
    step["params"] = params


def _rewrite_step_to_unsupported(step: dict[str, Any], reason: str, available: set[str]) -> None:
    step["operation"] = "unsupported" if "unsupported" in available else "unsupported"
    step["params"] = {"reason": reason}
    step["description"] = reason


def _attach_scoped_nested_domain_operation(
    params: dict[str, Any],
    *,
    nested_operation: str,
    user_request: str,
    context: dict[str, Any],
) -> None:
    params["edit_op"] = nested_operation
    edit_params = params.get("edit_op_params")
    if not isinstance(edit_params, dict):
        edit_params = params.get("edit_params")
    if not isinstance(edit_params, dict):
        edit_params = {}
    edit_params.setdefault("instruction", user_request)
    if nested_operation == "seamless_expressive_s2st":
        target_language = _requested_translation_target_language(user_request)
        if target_language:
            edit_params.setdefault("target_language", target_language)
    if nested_operation == "accent_conversion":
        reference_audio = _reference_audio_from_context(context)
        if reference_audio:
            edit_params.setdefault("reference_audio_path", reference_audio)
    if nested_operation == "no_reference_voice_conversion":
        edit_params.setdefault("target_voice_description", _target_voice_description(user_request))
        edit_params.setdefault("preserve_words", True)
    params["edit_op_params"] = edit_params
    params.setdefault("selection_instruction", _speaker_scope_hint(user_request))
    params.setdefault("preserve_timeline", True)


def _repair_normal_speech_emotion_params(params: dict[str, Any], *, user_request: str, context: dict[str, Any]) -> None:
    if not _requests_crying_to_normal_speech(user_request):
        return
    source_text = params.get("source_text") or context.get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        return
    params["instruction_st"] = f'<emo desc="normal speaking voice">{source_text}</emo>'
    params["emotion_condition_mode"] = "desc"


def _is_existing_audio_text_replacement(user_request: str, context: dict[str, Any]) -> bool:
    text = user_request.lower()
    has_source_audio = bool(context.get("input_audio") or context.get("input_audios"))
    if not has_source_audio:
        return False
    if _requests_question_to_declarative_tone(text):
        return False
    if _text_replacement_signal(text):
        return True
    quoted_count = len(re.findall(r"“[^”]+”|'[^']+'|\"[^\"]+\"", user_request))
    return quoted_count >= 2 and _contains_any(text, ["replace", "change", "modify", "swap", "switch", "exchange"])


def _text_replacement_signal(text: str) -> bool:
    return _contains_any(
        text,
        [
            "replace all",
            "replace every",
            "replace the word",
            "replace word",
            "replace the phrase",
            "replace phrase",
            "replace the sentence",
            "replace first sentence",
            "replace the first sentence",
            "replace the last sentence",
            "change all",
            "change every",
            "change the word",
            "change word",
            "modify the phrase",
            "modify phrase",
            "correct the pronunciation",
            "modify the pronunciation",
            "swap the word",
            "swap words",
            "switch the word",
            "switch words",
            "替换",
            "换成",
            "改字",
            "改文字",
            "改文本",
            "改内容",
            "改台词",
        ],
    )


def _requests_question_to_declarative_tone(text: str) -> bool:
    return _contains_any(
        text,
        [
            "questioning tone",
            "question tone",
            "questioning intonation",
            "interrogative tone",
            "interrogative intonation",
            "rising intonation",
            "rising tone",
        ],
    ) and _contains_any(
        text,
        [
            "declarative tone",
            "declarative",
            "statement tone",
            "matter-of-fact",
            "falling intonation",
            "falling tone",
        ],
    )


def _scoped_domain_operation(user_request: str, available: set[str]) -> str | None:
    if _is_dialect_translation_request(user_request):
        if "accent_conversion" in available:
            return "accent_conversion"
        return None
    if _requested_translation_target_language(user_request) and "seamless_expressive_s2st" in available:
        return "seamless_expressive_s2st"
    if _requests_accent_conversion(user_request) and "accent_conversion" in available:
        return "accent_conversion"
    if _requests_no_reference_voice_conversion(user_request, {}) and "no_reference_voice_conversion" in available:
        return "no_reference_voice_conversion"
    return None


def _requests_emphasis_stress_edit(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(
        token in lowered
        for token in [
            "emphasis",
            "emphasize",
            "emphasise",
            "stress",
            "add stress",
            "stress to",
            "stress the",
            "word stress",
            "speech stress",
            "stressed",
            "more prominent",
            "pronunciation of",
            "重音",
            "重读",
            "强调",
            "着重",
        ]
    )


def _emphasis_target_phrase(params: dict[str, Any], user_request: str) -> str | None:
    for key in ("target_phrase", "target_word", "phrase", "word"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().strip("\"'“”‘’")
    for key in ("edit_tagged_text", "instruction_st"):
        value = params.get(key)
        if not isinstance(value, str):
            continue
        for pattern in [
            r"<target>(?P<body>[^<>]+)</target>",
            r"<pitch\b[^>]*>(?P<body>[^<>]+)</pitch>",
            r"<rate\b[^>]*>(?P<body>[^<>]+)</rate>",
        ]:
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if match:
                phrase = match.group("body").strip()
                if phrase:
                    return phrase
    for pattern in [r'"([^"\n]+)"', r"'([^'\n]+)'", r"“([^”\n]+)”", r"‘([^’\n]+)’"]:
        match = re.search(pattern, user_request)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


def _scoped_edit_data_pipeline_operation(user_request: str) -> str | None:
    text = user_request.lower()
    if _requests_question_to_declarative_tone(text):
        return "emotion_style_edit"
    if _text_replacement_signal(text):
        return "text_edit"
    if _contains_any(text, ["emphasis", "emphasize", "emphasise", "stress", "intonation", "pitch", "faster", "slower", "speed", "speaking rate", "speech rate", "slow down", "speed up", "语速", "音高", "强调", "重音"]):
        return "pitch_rate_edit"
    if _contains_any(text, ["emotion", "emotional", "expressive", "tone", "style", "mood", "happy", "sad", "angry", "calm", "cheerful", "depressed", "melancholic", "hushed", "reflective", "情绪", "语气", "风格", "开心", "悲伤", "难过", "生气", "平静"]):
        if _contains_any(text, ["rap", "sing", "singing", "song", "accent", "dialect", "telephone", "phone call", "radio", "underwater", "wind"]):
            return None
        return "emotion_style_edit"
    if _contains_any(text, ["pause", "停顿"]):
        return "pause_edit"
    if _contains_any(text, ["volume", "loudness", "louder", "quieter", "音量", "响度"]):
        return "volume_normalize"
    return None


def _has_text_edit_instruction_tag(params: dict[str, Any]) -> bool:
    value = params.get("instruction_st") or params.get("instruction")
    if not isinstance(value, str):
        return False
    return bool(re.search(r"<\s*(?:sub|ins|del)\b", value, flags=re.IGNORECASE))


def _has_emotion_style_instruction_tag(params: dict[str, Any]) -> bool:
    value = params.get("instruction_st") or params.get("instruction")
    if not isinstance(value, str):
        return False
    return bool(
        re.search(
            r"<\s*(?:emo\b|desc\b|afraid\b|angry\b|calm\b|disgusted\b|"
            r"happy\b|melancholic\b|sad\b|surprised\b)",
            value,
            flags=re.IGNORECASE,
        )
    )


def _emotion_style_label_from_contract(params: dict[str, Any]) -> str | None:
    contract = params.get("label_contract")
    if not isinstance(contract, dict):
        return None
    labels = contract.get("supported_labels")
    if not isinstance(labels, list):
        return None
    emotion_labels = [
        str(label).strip()
        for label in labels
        if isinstance(label, str)
        and re.search(r"<\s*emo\b[^>]*>.*?</\s*emo\s*>", label, re.IGNORECASE | re.DOTALL)
    ]
    return emotion_labels[0] if len(emotion_labels) == 1 else None


def _unique_selected_span_text(context: dict[str, Any]) -> str | None:
    report = context.get("perception_report")
    spans = report.get("selected_spans") if isinstance(report, dict) else None
    if not isinstance(spans, list) or len(spans) != 1:
        return None
    span = spans[0]
    if not isinstance(span, dict):
        return None
    text = span.get("text")
    return str(text).strip() if isinstance(text, str) and text.strip() else None


def _drop_text_edit_conflicting_prosody_defaults(params: dict[str, Any]) -> None:
    for key in (
        "rate",
        "method",
        "pitch_method",
        "rate_method",
        "psola_indextts2_cco_pitch_boost",
        "smooth_policy",
        "target_db",
        "headroom_db",
    ):
        params.pop(key, None)


def _should_rewrite_scoped_wrapper_to_direct_edit(
    user_request: str,
    *,
    context: dict[str, Any],
    nested_operation: str,
    params: dict[str, Any] | None = None,
    step_text: str = "",
) -> bool:
    params = params if isinstance(params, dict) else {}
    text = " ".join(str(user_request or "").lower().split())
    if nested_operation == "volume_normalize" and _declares_global_audio_scope(
        step_text
    ):
        return True
    scoped_evidence = " ".join(
        str(value)
        for value in [
            user_request,
            params.get("selection_instruction"),
            params.get("selector"),
            params.get("grounding_instruction"),
            params.get("target_description"),
        ]
        if value
    )
    if _has_phrase_or_perceptual_scope(scoped_evidence):
        return False
    if not _request_has_speaker_or_span_scope(user_request):
        return True
    if nested_operation in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit", "speech_rate"} and not _scoped_wrapper_has_source_text_for_direct_rewrite(context, params):
        return False
    if nested_operation not in {"emotion_style_edit", "pitch_rate_edit"}:
        return False
    if not _has_generic_speaker_scope(text):
        return False
    if _has_local_or_event_scope(text):
        return False
    return _context_has_single_speaker(context)


def _declares_global_audio_scope(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return _contains_any(
        normalized,
        [
            "entire audio",
            "entire clip",
            "entire recording",
            "entire conversation",
            "whole audio",
            "whole clip",
            "whole recording",
            "whole conversation",
            "full audio",
            "full clip",
            "overall volume",
            "all speakers",
            "all participants",
            "整段",
            "全段",
            "全部说话人",
            "所有说话人",
        ],
    )


def _sanitize_volume_normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "service_url",
        "target_db",
        "method",
        "headroom_db",
        "trace_id",
        "instruction",
    }
    clean = {key: value for key, value in params.items() if key in allowed}
    method = str(clean.get("method") or "rms").strip().lower()
    clean["method"] = method if method in {"rms", "peak"} else "rms"
    return clean


def _request_has_speaker_or_span_scope(user_request: str) -> bool:
    text = user_request.lower()
    return _contains_any(
        text,
        [
            "speaker",
            "male",
            "female",
            "man",
            "woman",
            "child",
            "kid",
            "boy",
            "girl",
            "first sentence",
            "second sentence",
            "first half",
            "second half",
            "from ",
            "until",
            "segment",
            "phrase",
            "word",
            "internal monologue",
            "internal monologues",
            "说话人",
            "男",
            "女",
            "小孩",
            "儿童",
            "片段",
            "前半",
            "后半",
        ],
    )


def _text_edit_requires_speaker_scoped_wrapper(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _contains_any(
        text,
        [
            "speaker",
            "male",
            "female",
            "man's voice",
            "woman's voice",
            "child",
            "kid",
            "boy",
            "girl",
            "spoken by",
            "said by",
            "customer",
            "respondent",
            "说话人",
            "男声",
            "女声",
            "儿童",
            "小孩",
        ],
    )


def _has_phrase_or_perceptual_scope(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    if _requests_question_to_declarative_tone(normalized):
        return True
    if re.search(r"[\"“][^\"”]+[\"”]|'[^']+'|‘[^’]+’|「[^」]+」|『[^』]+』", normalized):
        return True
    return _contains_any(
        normalized,
        [
            "all instances",
            "every instance",
            "each instance",
            "instances of",
            "occurrences of",
            "spoken with",
            "spoken in",
            "said with",
            "word spoken",
            "phrase spoken",
            "questioning tone",
            "questioning intonation",
            "interrogative tone",
            "interrogative intonation",
        ],
    )


def _has_generic_speaker_scope(text: str) -> bool:
    return _contains_any(
        text,
        [
            "the speaker",
            "speaker's voice",
            "speaker’s voice",
            "male speaker",
            "female speaker",
            "man's voice",
            "man’s voice",
            "woman's voice",
            "woman’s voice",
            "male voice",
            "female voice",
            "说话人",
            "男声",
            "女声",
        ],
    )


def _has_local_or_event_scope(text: str) -> bool:
    if re.search(r"[\"“][^\"”]+[\"”]|'[^']+'|‘[^’]+’|「[^」]+」|『[^』]+』", text):
        return True
    return _contains_any(
        text,
        [
            "only",
            "except",
            "first speaker",
            "second speaker",
            "first sentence",
            "second sentence",
            "last sentence",
            "final sentence",
            "first utterance",
            "last utterance",
            "first half",
            "second half",
            "from ",
            "until",
            "between",
            "after ",
            "before ",
            "seconds",
            "twelfth",
            "segment",
            "quoted phrase",
            "the phrase",
            "spoken phrase",
            "spoken word",
            "line",
            "question",
            "answer",
            "spoken by",
            "instances",
            "all instances",
            "every instance",
            "remove",
            "delete",
            "mute",
            "extract",
            "laughter",
            "laughing",
            "cough",
            "sigh",
            "applause",
            "internal monologue",
            "internal monologues",
            "只有",
            "仅",
            "第一个",
            "第二个",
            "最后",
            "前半",
            "后半",
            "片段",
            "词",
            "短语",
            "删除",
            "移除",
            "笑声",
        ],
    )


def _context_has_single_speaker(context: dict[str, Any]) -> bool:
    perception = context.get("perception_report")
    if not isinstance(perception, dict):
        return False
    speaker_ids = _speaker_ids_from_container(perception.get("speakers"))
    if not speaker_ids:
        gemini = perception.get("gemini_perception")
        if isinstance(gemini, dict):
            speaker_ids = _speaker_ids_from_container(gemini.get("speakers"))
    if not speaker_ids:
        speaker_ids = _speaker_ids_from_container(perception.get("speaker_segments"))
    if not speaker_ids:
        gemini = perception.get("gemini_perception")
        if isinstance(gemini, dict):
            speaker_ids = _speaker_ids_from_container(gemini.get("segments"))
    return len(speaker_ids) == 1


def _context_has_selected_spans(context: dict[str, Any]) -> bool:
    perception = context.get("perception_report")
    if not isinstance(perception, dict):
        return False
    selected_spans = perception.get("selected_spans")
    if isinstance(selected_spans, list) and any(
        isinstance(item, dict)
        and item.get("start_sec") is not None
        and item.get("end_sec") is not None
        for item in selected_spans
    ):
        return True
    return _context_has_reviewed_speaker_attributed_evidence(context)


def _direct_edit_needs_foreground_background_remix(user_request: str, operation: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not operation:
        return False
    if _requests_preserve_background(text) and (
        operation.startswith("edit_data_pipeline.") or operation in {"qwen3_tts", "seed_vc"}
    ):
        return True
    if operation == "edit_data_pipeline.emotion_style_edit" and _requests_crying_to_normal_speech(text):
        return True
    if operation == "edit_data_pipeline.emotion_style_edit" and _requests_global_emotion_foreground_edit(text):
        return True
    if operation == "edit_data_pipeline.pitch_rate_edit":
        return _has_explicit_time_scope(text) and _contains_any(
            text,
            ["slow down", "speed up", "speaking rate", "speech rate", "faster", "slower", "pace", "tempo"],
        )
    return False


def _requests_preserve_background(text: str) -> bool:
    return _contains_any(
        text,
        [
            "background unchanged",
            "background music unchanged",
            "keep the background",
            "preserve the background",
            "without changing the background",
            "music unchanged",
            "背景不变",
            "保留背景",
            "保持背景",
            "背景音乐不变",
        ],
    )


def _requests_foreground_isolation(text: str) -> bool:
    normalized = " ".join(str(text or "").casefold().split())
    extraction = _contains_any(
        normalized,
        [
            "extract",
            "isolate",
            "retain only",
            "keep only",
            "only keep",
            "只保留",
            "仅保留",
            "提取",
            "分离出",
        ],
    )
    foreground = _contains_any(
        normalized,
        [
            "voice",
            "voices",
            "vocal",
            "vocals",
            "speech",
            "singing",
            "a cappella",
            "acappella",
            "人声",
            "说话声",
            "歌声",
            "清唱",
        ],
    )
    suppresses_background = "unaccompanied" in normalized or (
        _contains_any(
            normalized,
            [
                "remove",
                "suppress",
                "exclude",
                "without",
                "去除",
                "移除",
                "抑制",
                "不要",
            ],
        )
        and _contains_any(
            normalized,
            [
                "instrumental",
                "background",
                "accompaniment",
                "伴奏",
                "背景",
            ],
        )
    )
    return extraction and foreground and suppresses_background


def _requests_speaker_specific_volume(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _contains_any(text, ["volume", "loudness", "louder", "quieter", "音量", "响度"]) and _request_has_speaker_or_span_scope(text)


def _requests_crying_to_normal_speech(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(lowered, ["crying tone", "crying voice", "crying", "sobbing", "weepy", "tearful"]) and _contains_any(
        lowered,
        ["normal speaking voice", "normal voice", "natural speaking voice", "plain speaking voice", "neutral voice"],
    )


def _requests_global_emotion_foreground_edit(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(lowered, ["entire audio", "whole audio", "full audio", "entire clip", "whole clip"]) and _contains_any(
        lowered,
        ["emotion", "tone", "mood", "sad", "happy", "angry", "calm", "extremely sad", "very sad"],
    )


def _has_explicit_time_scope(text: str) -> bool:
    return bool(
        re.search(r"\bfrom\s+(?:the\s+)?(?:\d+(?:\.\d+)?|first|second|third|twelfth)\s*(?:s|sec|secs|second|seconds)\b", text)
        or re.search(r"\b(?:starting|start)\s+from\b", text)
        or _contains_any(
            text,
            [
                "until the end",
                "to the end",
                "from the beginning",
                "first half",
                "second half",
                "latter half",
                "后半",
                "前半",
                "从",
                "到结尾",
            ],
        )
    )


def _speaker_ids_from_container(raw: Any) -> set[str]:
    if not isinstance(raw, list):
        return set()
    speaker_ids: set[str] = set()
    anonymous_count = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("speaker_id") or item.get("speaker") or item.get("label")
        if value is None:
            anonymous_count += 1
            continue
        speaker_ids.add(_normalize_speaker_id(value))
    if not speaker_ids and anonymous_count == 1:
        return {"anonymous"}
    return speaker_ids


def _normalize_speaker_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(speaker|spk|person)[_\-\s]*", "", text)
    return text or "unknown"


def _speaker_scope_hint(user_request: str) -> str:
    text = user_request.lower()
    for hint in [
        "female speaker",
        "male speaker",
        "first speaker",
        "second speaker",
        "child speaker",
        "the first sentence",
        "the second half",
        "the first half",
    ]:
        if hint in text:
            return hint
    return user_request


def _pure_speaker_artifact_operation(
    user_request: str,
    params: dict[str, Any],
    available: set[str],
) -> str | None:
    if _has_nested_operation(params):
        return None
    evidence = " ".join(
        str(value)
        for value in [
            user_request,
            params.get("instruction"),
            params.get("selection_instruction"),
            params.get("speaker_selector"),
            params.get("selector"),
        ]
        if value
    ).lower()
    if not _contains_any(
        evidence,
        [
            "speaker",
            "voice",
            "person",
            "talker",
            "man",
            "woman",
            "说话人",
            "讲话人",
            "人声",
            "男声",
            "女声",
        ],
    ):
        return None
    if _contains_any(evidence, ["mute", "silence", "make silent", "静音", "消音"]):
        return "artifact_ops.mute_spans" if "artifact_ops.mute_spans" in available else None
    if _contains_any(evidence, ["remove", "delete", "cut out", "drop", "erase", "去掉", "删除", "移除"]):
        return "artifact_ops.remove_spans" if "artifact_ops.remove_spans" in available else None
    if _contains_any(
        evidence,
        [
            "extract",
            "isolate",
            "keep only",
            "only keep",
            "separate out",
            "导出",
            "提取",
            "分离",
            "只保留",
        ],
    ):
        return "artifact_ops.extract_spans" if "artifact_ops.extract_spans" in available else None
    return None


def _non_lexical_replacement_artifact_operation(user_request: str, available: set[str]) -> str | None:
    text = " ".join(str(user_request or "").lower().split())
    if not (_text_replacement_signal(text) or _sound_effect_replacement_signal(text)):
        return None
    if not _replacement_target_is_sound_or_effect(text):
        return None
    if _contains_any(text, ["beep", "bleep", "censor tone", "censor sound", "profanity", "curse word", "swear word"]):
        return "artifact_ops.beep_spans" if "artifact_ops.beep_spans" in available else None
    return None


def _sound_effect_replacement_signal(text: str) -> bool:
    return bool(
        re.search(r"\b(?:replace|substitut(?:e|ing)?|censor|mask)\b.+\b(?:with|to|into)\b", text)
        or _contains_any(text, ["替换成", "替换为", "换成", "改成", "改为"])
    )


def _replacement_target_is_sound_or_effect(text: str) -> bool:
    return _contains_any(
        text,
        [
            "with a sound",
            "with the sound",
            "with sound",
            "with a tone",
            "with the tone",
            "with tone",
            "with an effect",
            "with the effect",
            "with audio effect",
            "with noise",
            "sound effect",
            "bleeping sound",
            "beeping sound",
            "censor tone",
            "审查音",
            "消音音效",
            "哔声",
            "音效",
        ],
    )


def _has_nested_operation(params: dict[str, Any]) -> bool:
    value = params.get("edit_op") or params.get("edit_operation")
    return bool(_nested_operation_name(value))


def _nested_operation_name(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("operation")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized.removeprefix("edit_data_pipeline.").removeprefix("artifact_ops.")


def _is_nested_edit_data_pipeline_op(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("operation")
    if not isinstance(value, str):
        return False
    normalized = value.removeprefix("edit_data_pipeline.")
    return normalized in {"text_edit", "emotion_style_edit", "pitch_rate_edit", "pause_edit", "volume_normalize"}


LANGUAGE_NAME_TO_CODE = {
    "english": "eng",
    "eng": "eng",
    "en": "eng",
    "英语": "eng",
    "mandarin": "cmn",
    "mandarin chinese": "cmn",
    "chinese": "cmn",
    "中文": "cmn",
    "普通话": "cmn",
    "spanish": "spa",
    "西班牙语": "spa",
    "japanese": "jpn",
    "日语": "jpn",
    "korean": "kor",
    "韩语": "kor",
    "french": "fra",
    "法语": "fra",
    "german": "deu",
    "德语": "deu",
    "italian": "ita",
    "意大利语": "ita",
    "portuguese": "por",
    "葡萄牙语": "por",
    "russian": "rus",
    "俄语": "rus",
    "thai": "tha",
    "泰语": "tha",
}

_DIALECT_TRANSLATION_SOURCE_TERMS = [
    "taiwanese language",
    "taiwanese mandarin",
    "taiwan mandarin",
    "taiwanese slang",
    "taiwanese",
    "hokkien",
    "minnan",
    "min nan",
    "shanghainese",
    "cantonese",
    "sichuanese",
    "beijing dialect",
    "regional dialect",
    "local dialect",
    "dialectal",
    "dialect",
    "accented",
    "accent",
    "台语",
    "臺語",
    "台湾话",
    "臺灣話",
    "台湾腔",
    "臺灣腔",
    "台灣腔",
    "闽南语",
    "閩南語",
    "上海话",
    "粤语",
    "广东话",
    "四川话",
    "北京话",
    "方言",
    "口音",
    "地域表达",
    "地方表达",
]

_DIALECT_EXPRESSION_SOURCE_TERMS = [
    "taiwanese language",
    "taiwanese mandarin",
    "taiwan mandarin",
    "taiwanese slang",
    "hokkien",
    "minnan",
    "min nan",
    "shanghainese",
    "cantonese",
    "sichuanese",
    "beijing dialect",
    "regional dialect",
    "local dialect",
    "dialectal",
    "dialect",
    "台语",
    "臺語",
    "台湾话",
    "臺灣話",
    "闽南语",
    "閩南語",
    "上海话",
    "粤语",
    "广东话",
    "四川话",
    "北京话",
    "方言",
    "地域表达",
    "地方表达",
]

_DIALECT_TRANSLATION_ACTION_TERMS = [
    "translate",
    "translation",
    "convert",
    "change",
    "turn",
    "make",
    "normalize",
    "standardize",
    "de-regionalize",
    "deregionalize",
    "localize",
    "regionalize",
    "翻译",
    "译成",
    "翻成",
    "转成",
    "转换",
    "改成",
    "变成",
    "标准化",
    "去地域化",
    "地域化",
]

_DIALECT_TRANSLATION_STRONG_ACTION_TERMS = [
    "translate",
    "translation",
    "normalize",
    "standardize",
    "de-regionalize",
    "deregionalize",
    "localize",
    "regionalize",
    "翻译",
    "译成",
    "翻成",
    "标准化",
    "去地域化",
    "地域化",
]

_STANDARD_VARIETY_TARGET_TERMS = [
    "standard mandarin",
    "mandarin",
    "putonghua",
    "standard chinese",
    "native chinese",
    "standard english",
    "general american english",
    "received pronunciation",
    "普通话",
    "标准普通话",
    "标准中文",
    "中文普通话",
    "标准英语",
]


def _requested_translation_target_language(user_request: str) -> str | None:
    text = " ".join(str(user_request or "").lower().split())
    if not _contains_any(text, ["translate", "translation", "翻译", "译成", "翻成"]):
        return None
    explicit = re.search(r"(?:target(?: language)?|tgt_lang|目标语言)\s*[:=]\s*([a-zA-Z_]+)", text)
    if explicit:
        return LANGUAGE_NAME_TO_CODE.get(explicit.group(1).lower(), explicit.group(1).lower())
    for name, code in sorted(LANGUAGE_NAME_TO_CODE.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(name.lower())
        if re.search(rf"(?:to|into|in)\s+{escaped}\b", text):
            return code
        if any(marker + name in user_request for marker in ["翻译成", "译成", "翻成", "转成", "变成", "到"]):
            return code
    mentioned = {
        code
        for name, code in LANGUAGE_NAME_TO_CODE.items()
        if re.search(rf"\b{re.escape(name.lower())}\b", text) or name in user_request
    }
    if len(mentioned) == 1:
        return next(iter(mentioned))
    return None


def _source_language_code(user_request: str) -> str | None:
    text = str(user_request or "").lower()
    match = re.search(r"(?:from|source(?: language)?|src_lang|源语言)\s*[:=]?\s*([a-zA-Z_]+)", text)
    if not match:
        return None
    return LANGUAGE_NAME_TO_CODE.get(match.group(1).lower(), match.group(1).lower())


def _is_dialect_translation_request(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not _contains_any(text, _DIALECT_TRANSLATION_ACTION_TERMS):
        return False
    if not _contains_any(text, _DIALECT_TRANSLATION_SOURCE_TERMS):
        return False
    has_expression_source = _contains_any(text, _DIALECT_EXPRESSION_SOURCE_TERMS)
    has_strong_action = _contains_any(text, _DIALECT_TRANSLATION_STRONG_ACTION_TERMS)
    if not has_expression_source and not has_strong_action:
        return False
    target_language = _requested_translation_target_language(user_request)
    if target_language in {"cmn", "eng"}:
        return True
    return _contains_any(text, _STANDARD_VARIETY_TARGET_TERMS)


def _dialect_translation_target_accent_label(user_request: str) -> str | None:
    text = " ".join(str(user_request or "").lower().split())
    target_language = _requested_translation_target_language(user_request)
    if target_language == "cmn" or _contains_any(text, ["mandarin", "putonghua", "普通话", "标准中文"]):
        return "Standard Mandarin"
    if target_language == "eng" or _contains_any(text, ["standard english", "general american english"]):
        return "Standard English"
    for marker, label in [
        ("beijing dialect", "Beijing Guanhua"),
        ("北京话", "Beijing Guanhua"),
        ("台湾腔", "Taiwanese Mandarin"),
        ("臺灣腔", "Taiwanese Mandarin"),
        ("taiwanese mandarin", "Taiwanese Mandarin"),
    ]:
        if marker in text or marker in user_request:
            return label
    return None


def _dialect_expression_normalization_goal(user_request: str) -> str:
    target_accent = _dialect_translation_target_accent_label(user_request)
    if target_accent == "Standard Mandarin":
        return "Convert localized/Taiwanese/dialectal expressions into standard Mandarin wording before accent conversion."
    if target_accent == "Standard English":
        return "Convert dialectal or regional expressions into standard English wording before accent conversion."
    return "Convert regional expressions to the requested target variety before accent or dialect conversion."


def _translation_request_is_localized(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _contains_any(text, ["only", "speaker", "utterance", "line", "phrase", "sentence", "parts", "segments", "spoken by", "仅", "只", "片段", "说话人"])


def _operation_unavailable(context: dict[str, Any], operation: str) -> bool:
    unavailable = context.get("unavailable_operations") or []
    if isinstance(unavailable, list) and operation in unavailable:
        return True
    previous = context.get("previous_failure")
    if isinstance(previous, dict):
        haystack = " ".join(str(value) for value in previous.values()).lower()
        return operation.lower() in haystack and _contains_any(haystack, ["unavailable", "not available", "missing", "failed"])
    return False


def _requests_accent_conversion(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if _is_dialect_translation_request(user_request):
        return True
    if _requested_translation_target_language(user_request) and not _contains_any(text, ["accent", "dialect", "口音", "方言"]):
        return False
    if _contains_any(text, ["accent unchanged", "keep her voice and accent unchanged", "same-language speech fluent", "native chinese wording"]):
        return False
    if _contains_any(text, ["pronunciation of", "correct the pronunciation", "swap the pronunciation", "clone the pronunciation"]):
        return False
    return _contains_any(
        text,
        [
            "accent",
            "dialect",
            "standard english",
            "standard mandarin",
            "putonghua",
            "口音",
            "方言",
            "普通话",
            "北京话",
            "上海话",
            "粤语",
            "四川话",
            "闽南语",
        ],
    )


def _requested_accent_label(user_request: str) -> str | None:
    if _is_dialect_translation_request(user_request):
        return _dialect_translation_target_accent_label(user_request)
    text = str(user_request or "")
    quoted = re.findall(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", text)
    for groups in quoted:
        value = next((group for group in groups if group), "").strip()
        if value and _contains_any(value.lower(), ["accent", "dialect", "口音", "方言", "普通话"]):
            return value
    lowered = text.lower()
    for marker in ["british accent", "american accent", "standard mandarin", "putonghua", "mandarin", "taiwanese mandarin", "taiwanese", "粤语", "四川话", "上海话", "北京话", "台湾腔", "臺灣腔"]:
        if marker in lowered or marker in text:
            return marker
    return None


def _requests_no_reference_voice_conversion(user_request: str, context: dict[str, Any]) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    has_source_audio = (
        bool(
            context.get("input_audio")
            or context.get("input_audios")
            or context.get("workflow_source_audio_refs")
        )
        if context
        else True
    )
    if not has_source_audio:
        return False
    if _contains_any(text, ["accent", "dialect", "translate", "translation", "口音", "方言", "翻译"]):
        return False
    has_voice_class = _contains_any(
        text,
        [
            "child voice",
            "kid voice",
            "children voice",
            "boy voice",
            "girl voice",
            "male voice",
            "male voices",
            "female voice",
            "female voices",
            "man voice",
            "woman voice",
            "younger voice",
            "older voice",
            "young voice",
            "old voice",
            "younger speaker",
            "older speaker",
            "sounds younger",
            "sounds older",
            "sound younger",
            "sound older",
            "sound much younger",
            "sound much older",
            "much younger",
            "much older",
            "elderly",
            "aged voice",
            "elderly voice",
            "different timbre",
            "change the voice",
            "小孩音色",
            "儿童音色",
            "童声",
            "男声",
            "女声",
            "年轻声音",
            "年老声音",
        ],
    )
    preserve_words = _contains_any(
        text,
        [
            "keep the words",
            "keeping the words",
            "keeping words",
            "preserve words",
            "preserve the words",
            "preserving words",
            "preserving the words",
            "same words",
            "same wording",
            "text unchanged",
            "spoken content unchanged",
            "keep the spoken content",
            "keeping the spoken content",
            "preserve the spoken content",
            "content unchanged",
            "不改变文本",
            "保持文本",
        ],
    )
    explicit_existing_voice_conversion = _contains_any(
        text,
        ["change all", "convert all", "change the", "convert the", "make all", "turn all"],
    ) and _contains_any(text, ["voice", "voices", "speaker", "speakers"])
    return has_voice_class and (
        preserve_words
        or explicit_existing_voice_conversion
        or _contains_any(text, ["existing audio", "source audio", "this audio", "这段音频"])
    )


def _target_voice_description(user_request: str) -> str:
    text = " ".join(str(user_request or "").split())
    return text or "requested target voice"


def _requests_spatial_room_effect(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_any(
        text,
        [
            "remove reverb",
            "remove the reverb",
            "remove echo",
            "remove the echo",
            "reduce reverb",
            "reduce echo",
            "suppress reverb",
            "suppress echo",
            "eliminate reverb",
            "eliminate echo",
            "dereverb",
            "de-reverb",
            "denoise",
            "noise reduction",
            "去混响",
            "去除混响",
            "消除混响",
            "去回声",
            "去除回声",
            "消除回声",
        ],
    ):
        return False
    return _contains_any(
        text,
        [
            "sense of space",
            "spatial sense",
            "spatial feeling",
            "spacious",
            "space around",
            "room sound",
            "large room",
            "empty room",
            "hall",
            "cathedral",
            "bathroom",
            "reverb",
            "echo",
            "空间感",
            "空间效果",
            "混响",
            "回声",
            "大厅",
            "空旷",
            "空荡",
        ],
    )


def _spatial_reverb_room_preset(user_request: str) -> str:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_any(text, ["cathedral", "church", "教堂"]):
        return "cathedral"
    if _contains_any(text, ["bathroom", "bath room", "浴室", "卫生间"]):
        return "bathroom"
    if _contains_any(text, ["hall", "stage", "auditorium", "大厅", "礼堂"]):
        return "hall"
    return "large_empty_room"


def _requests_channel_or_environment_effect(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_any(text, ["voice", "speaker timbre", "speaker's timbre", "音色"]) and not _contains_any(text, ["channel", "telephone", "radio", "underwater", "wind", "muffled"]):
        return False
    return _contains_any(
        text,
        [
            "telephone",
            "phone call",
            "old-radio",
            "old radio",
            "underwater",
            "strong wind",
            "wind effect",
            "muffled",
            "heavy wall",
            "same acoustic channel",
            "channel condition",
            "recording channel",
            "电话",
            "电台",
            "水下",
            "强风",
            "隔墙",
            "闷",
        ],
    )


def _audio_effect_preset(user_request: str) -> str:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_any(text, ["telephone", "phone call", "电话"]):
        return "telephone"
    if _contains_any(text, ["old radio", "old-radio", "radio", "电台"]):
        return "old_radio"
    if _contains_any(text, ["underwater", "水下"]):
        return "underwater"
    if _contains_any(text, ["wind", "强风"]):
        return "strong_wind"
    if _contains_any(text, ["muffled", "heavy wall", "隔墙", "闷"]):
        return "muffled_wall"
    if _contains_any(text, ["acoustic channel", "channel condition", "recording channel"]):
        return "match_channel_reference"
    return "custom"


def _is_pure_concat_request(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    has_concat = _contains_any(text, ["concatenate", "concat", "join", "merge", "stitch together", "append", "拼接", "连接", "合并"])
    if not has_concat:
        return False
    return not _contains_any(
        text,
        [
            "translate",
            "replace",
            "change",
            "remove",
            "delete",
            "effect",
            "voice",
            "accent",
            "speech rate",
            "background",
            "overlay",
            "mix",
            "翻译",
            "替换",
            "删除",
            "音色",
            "口音",
        ],
    )


def _requests_synthetic_silence(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _contains_any(text, ["silence", "silent audio", "silent clip", "pause", "gap", "空白", "静音", "停顿"])


def _requests_pause_inside_existing_audio(instruction: str) -> bool:
    text = " ".join(str(instruction or "").lower().split())
    has_pause = _contains_any(
        text,
        ["pause", "silence", "short break", "brief break", "停顿", "暂停"],
    )
    has_placement = _contains_any(
        text,
        [
            "insert",
            "add",
            "after",
            "before",
            "between",
            "inside",
            "within",
            "在其后",
            "之后",
            "之前",
            "中间",
            "插入",
            "加入",
        ],
    )
    return has_pause and has_placement


def _is_simple_insert_request(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _contains_any(text, ["insert", "add", "place", "put", "before", "after", "at ", "插入", "加入", "前面", "后面"])


def _is_simple_overlay_request(user_request: str, context: dict[str, Any]) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    has_overlay_intent = _contains_any(
        text,
        [
            "overlay",
            "mix",
            "mix in",
            "superimpose",
            "layer",
            "put audio",
            "add audio",
            "叠加",
            "疊加",
            "混合",
            "混入",
        ],
    )
    if not has_overlay_intent:
        return False
    has_external_audio = len(_context_audio_paths(context)) >= 2 or bool(_reference_audio_from_context(context))
    if not has_external_audio:
        return False
    return not _contains_any(
        text,
        [
            "foreground",
            "background unchanged",
            "preserve the background",
            "speech separation",
            "前景",
            "背景不变",
            "保留背景",
        ],
    )


def _audio_assembly_insert_has_artifact(step: dict[str, Any]) -> bool:
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    return any(params.get(key) for key in ("target_audio_path", "insert_audio_path", "generated_audio_path", "replacement_audio_path"))


def _infer_timeline_position(user_request: str, *, default: Any = 0.0) -> Any:
    text = " ".join(str(user_request or "").lower().split())
    match = re.search(r"\b(?:at|from|starting at|start at)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    if _contains_any(text, ["midpoint", "middle", "center", "centre", "中点", "中間", "中间"]):
        return "midpoint"
    if _contains_any(text, ["end", "ending", "at the tail", "结尾", "末尾"]):
        return "end"
    if _contains_any(text, ["beginning", "start", "开头", "开始"]):
        return 0.0
    return default


def _infer_pause_duration_seconds(user_request: str) -> float | None:
    text = " ".join(str(user_request or "").lower().split())
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\s+(?:silence|pause|gap)", text)
    if not match:
        match = re.search(r"(?:silence|pause|gap).{0,16}?(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    if _contains_any(text, ["silence", "pause", "gap", "停顿", "静音"]):
        return 1.0
    return None


def _wants_expressive_preservation(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_any(text, ["do not preserve", "without preserving", "no need to preserve", "不要保持", "无需保留"]):
        return False
    return _contains_any(text, ["vocal style", "voice style", "speaking style", "prosody", "expressive", "same delivery", "preserve", "keep", "保持", "保留"])


def _preserve_vocal_style(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _wants_expressive_preservation(text) or _contains_any(text, ["vocal style", "voice style", "speaking style", "prosody", "风格", "语气", "音色"])


def _preserve_speech_rate(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _wants_expressive_preservation(text) or _contains_any(text, ["speech rate", "speaking rate", "pace", "tempo", "语速", "节奏"])


def _preserve_pauses(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return _wants_expressive_preservation(text) or _contains_any(text, ["pause", "pauses", "silence", "停顿", "暂停"])


def _reference_audio_from_context(context: dict[str, Any]) -> str | None:
    refs = context.get("reference_audio_paths") or []
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, str) and ref:
                return ref
    audios = context.get("input_audios") or context.get("input_audio_paths") or []
    if isinstance(audios, list) and len(audios) >= 2 and isinstance(audios[1], str):
        return audios[1]
    return None


def _context_audio_paths(context: dict[str, Any]) -> list[str]:
    for key in ("input_audios", "input_audio_paths"):
        value = context.get(key)
        if isinstance(value, list):
            paths = [str(item) for item in value if isinstance(item, str) and item]
            if paths:
                return paths
    input_audio = context.get("input_audio")
    refs = context.get("reference_audio_paths") or []
    paths = [str(input_audio)] if isinstance(input_audio, str) and input_audio else []
    if isinstance(refs, list):
        paths.extend(str(item) for item in refs if isinstance(item, str) and item)
    return paths


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)
