"""Operation-specific parameter normalization before tool execution.

This layer is intentionally narrow: it converts planner/executor semantic
params into the contracts expected by existing X-Editor tool adapters, and it
raises explicit boundary errors when a request cannot be represented safely.
It does not run audio models.
"""

from __future__ import annotations

import copy
import difflib
import re
from html import escape, unescape
from pathlib import Path
from typing import Any

from x_editor.core.schemas import AudioEditStep
from x_editor.tools.edit_backend import uses_dots_model
from x_editor.core.tagged_text import render_tagged_source
from x_editor.core.emotion_choices import validate_explicit_emotion_choices
from x_editor.core.text_equivalence import equivalent_text_copy, equivalent_transcript_copy
from x_editor.runtime.audio_effects import canonical_audio_effect_preset
from x_editor.runtime.audio_references import resolve_audio_reference
from x_editor.runtime.normalization_audit import (
    NormalizationAudit,
    build_normalization_audit,
    normalization_snapshot,
)
from x_editor.runtime.reconstruct_contract import (
    RECONSTRUCTION_MODES,
    normalized_edited_artifact_scope,
    normalized_reconstruction_mode,
)
from x_editor.runtime.reorder_contract import (
    normalize_reorder_indices,
    normalize_reorder_speaker_order,
)
from x_editor.runtime.semantic_projection import (
    SemanticProjectionError,
    project_semantic_transform_for_text_edit,
)

TAGGED_EDIT_OPERATIONS = {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit"}
ALLOWED_EMOTIONS = {"happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm", "none"}
PITCH_SHIFT_METHODS = {"psola", "psola+qwen3_codec", "psola+indextts2_cco", "psola+mimo_codec"}
VOLUME_NORMALIZE_METHODS = {"rms", "peak"}

EDIT_DATA_PIPELINE_LABEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "text_edit": {
        "field": "instruction_st",
        "supported_labels": [
            '<sub targ="new phrase">old phrase</sub>',
            "<ins>inserted phrase</ins>",
            "<del>deleted phrase</del>",
        ],
        "allowed_tags": ["sub", "ins", "del"],
        "requirements": [
            "Stripping tags as source text must exactly match source_text.",
            "Use <sub targ=\"new phrase\">old phrase</sub> for replacements.",
            "Use <ins>...</ins> only for pure insertion and <del>...</del> only for pure deletion.",
            "Do not use prosody, emotion, pitch/rate, pause, target, or arbitrary XML tags in text_edit.",
        ],
    },
    "pitch_rate_edit": {
        "field": "instruction_st",
        "supported_labels": [
            "<pitch, semitones=3>phrase</pitch>",
            "<rate, factor=1.2>phrase</rate>",
        ],
        "allowed_tags": ["pitch", "rate"],
        "requirements": [
            "Stripping tags must exactly match source_text.",
            "Pitch tags must be exactly <pitch, semitones=N>...</pitch> with integer N in [-12, 12].",
            "Rate tags must be exactly <rate, factor=F>...</rate> with F in [0.3, 3.0].",
            "Use <rate>, not <speed>; use <pitch>, not <prosody> or pitch-contour attributes.",
            "For pitch-only edits, smooth_policy must be 'never' and rate/factor/speed params must be absent.",
        ],
    },
    "emotion_style_edit": {
        "field": "instruction_st",
        "supported_labels": [
            '<emo type="happy" level="2">phrase</emo>',
        ],
        "allowed_tags": ["emo", *sorted(ALLOWED_EMOTIONS - {"none"})],
        "requirements": [
            "Stripping tags must exactly match source_text.",
            "Categorical emotion tags must be <emo type=\"emotion\" level=\"1|2|3\">...</emo>.",
            "Allowed categorical emotions are: afraid, angry, calm, disgusted, happy, melancholic, sad, surprised.",
            "Free-form descriptions cannot select a service category. The Planner must choose a supported type and intensity only when it fulfills the requested effect; otherwise report the capability limitation.",
            "Do not use emotion_style_edit for rap, singing, accent/dialect conversion, or channel effects.",
        ],
    },
    "pause_edit": {
        "field": "instruction_st",
        "supported_labels": [
            '<pause act="ins" level="1"/>',
            '<pause act="ins" level="2"/>',
            '<pause act="ins" level="3"/>',
        ],
        "allowed_tags": ["pause"],
        "requirements": [
            "Stripping pause tags must exactly match source_text.",
            "Only self-closing insertion tags are supported by the current executor path.",
            "Use <pause act=\"ins\" level=\"1|2|3\"/>. Do not use pause deletion/adjust tags here; use boundary_smooth points for low-level pause adjustment.",
        ],
    },
    "audio_assembly": {
        "field": "edit_tagged_text",
        "supported_labels": [
            "<target>replacement span transcript</target>",
        ],
        "allowed_tags": ["target"],
        "requirements": [
            "audio_assembly does not use instruction_st.",
            "edit_tagged_text may only mark target spans with <target>...</target>.",
            "Provide target_audio_path and source/target time spans when using audio_assembly.",
            "Do not send target_text-only edits to audio_assembly; use text_edit instead.",
        ],
    },
    "speech_rate": {
        "field": "params",
        "supported_labels": [],
        "allowed_tags": [],
        "requirements": [
            "speech_rate does not accept label tags; use rate/method/text parameters.",
            "For localized tagged speed edits, use pitch_rate_edit with <rate, factor=...>...</rate>.",
        ],
    },
    "volume_normalize": {
        "field": "params",
        "supported_labels": [],
        "allowed_tags": [],
        "requirements": [
            "volume_normalize does not accept label tags; use target_db/headroom_db/method parameters.",
            "For span-scoped gain changes, use artifact_ops.speaker_scoped_edit with nested volume_normalize.",
        ],
    },
    "pitch_shift": {
        "field": "params",
        "supported_labels": [],
        "allowed_tags": [],
        "requirements": [
            "pitch_shift does not accept label tags; use semitones/method parameters.",
            "pitch_shift method must be one of psola, psola+qwen3_codec, psola+indextts2_cco, or psola+mimo_codec; WSOLA methods are speech-rate methods, not pitch-shift methods.",
            "For localized tagged pitch edits, use pitch_rate_edit with <pitch, semitones=N>...</pitch>.",
        ],
    },
    "boundary_smooth": {
        "field": "points",
        "supported_labels": [],
        "allowed_tags": [],
        "requirements": [
            "boundary_smooth does not accept instruction_st labels; use explicit points with pause intervals or insert audio.",
        ],
    },
}

PROTECTED_EDIT_PIPELINE_KEYS = {
    "operation",
    "input_audio_path",
    "audio_path",
    "source_audio_path",
    "output_path",
    "source_text",
    "instruction_st",
    "target_text",
}

COMMON_EDIT_PIPELINE_OVERRIDES = {
    "seed",
    "num_steps",
    "guidance_scale",
    "use_xvector",
    "speaker_scale",
    "service_url",
    "generation_id",
    "sample_id",
    "clip_id",
    "evaluate",
    "persist",
    "output_dir",
    "language",
    "aligner_backend",
    "debug",
    "debug_dir",
    "debug_output_dir",
    "trace_id",
}

EDIT_PIPELINE_OVERRIDE_KEYS: dict[str, set[str]] = {
    "text_edit": COMMON_EDIT_PIPELINE_OVERRIDES
    | {
        "save_parquet",
        "parquet_output_dir",
        "parquet_dataset_name",
        "output_dataset_name",
        "duration_mode",
        "synthesis_mode",
        "expand_radius",
        "merge_gap",
        "nfe_step",
        "cfg_strength",
        "sway_sampling_coef",
        "source_crossfade_sec",
        "regen_crossfade_sec",
        "seed",
        "disable_prepend_context",
        "fail_if_cannot_prepend_context",
        "min_unmasked_context_sec",
        "min_unmasked_context_ratio",
        "max_f5_edit_seconds",
    },
    "pitch_rate_edit": COMMON_EDIT_PIPELINE_OVERRIDES
    | {
        "pitch_method",
        "rate_method",
        "psola_indextts2_cco_pitch_boost",
        "crossfade_ms",
        "smooth_policy",
        "smooth_margin_sec",
    },
    "emotion_style_edit": COMMON_EDIT_PIPELINE_OVERRIDES
    | {
        "instruction_nl",
        "return_synthetic_target_audio",
        "max_candidates",
        "multi_frequent_smoothing_radius_sec",
        "synthesis_mode",
        "emotion_condition_mode",
    },
    "pause_edit": COMMON_EDIT_PIPELINE_OVERRIDES
    | {
        "profile_id",
        "save_parquet",
        "parquet_output_dir",
        "parquet_dataset_name",
        "stage2_method",
        "synth_method",
        "expand_radius",
        "center_regen_second_pass",
        "center_regen_margin_sec",
    },
    "speech_rate": {
        "service_url",
        "rate",
        "method",
        "text",
        "speaker_audio_path",
        "allow_method_fallback",
        "language",
        "trace_id",
    },
    "volume_normalize": {"service_url", "target_db", "method", "headroom_db", "trace_id"},
    "pitch_shift": {"service_url", "semitones", "method", "trace_id"},
    "boundary_smooth": {"service_url", "points", "debug_dir", "trace_id"},
    "audio_assembly": {
        "service_url",
        "source_audio_path",
        "target_audio_path",
        "edit_tagged_text",
        "time_spans_from_source",
        "time_spans_from_target",
        "preserve_outer_boundaries",
        "aligner_backend",
        "smoothing",
        "debug",
        "debug_dir",
        "trace_id",
        "language",
    },
}

EDIT_PIPELINE_PARAMETER_OWNERS: dict[str, set[str]] = {
    "pitch_method": {"pitch_rate_edit", "pitch_shift"},
    "rate_method": {"pitch_rate_edit", "speech_rate"},
}

EDIT_DATA_PIPELINE_SAFE_MISROUTE_TARGETS = {
    "pitch_rate_edit",
    "emotion_style_edit",
    "pause_edit",
    "speech_rate",
    "volume_normalize",
    "pitch_shift",
    "boundary_smooth",
}


class OperationBoundaryError(RuntimeError):
    """Raised when a plan step crosses a tool capability or contract boundary."""


class NonEquivalentNormalizationError(OperationBoundaryError):
    """Raised when normalization would alter Planner-selected semantics."""


def edit_data_pipeline_label_contract(operation: str) -> dict[str, Any] | None:
    """Return the supported label contract for an edit-data-pipeline operation."""

    normalized = operation.removeprefix("edit_data_pipeline.").replace("-", "_")
    contract = EDIT_DATA_PIPELINE_LABEL_CONTRACTS.get(normalized)
    return dict(contract) if contract else None


def edit_data_pipeline_label_requirements(operation: str) -> list[str]:
    """Return human-readable executor requirements for an operation's label form."""

    contract = edit_data_pipeline_label_contract(operation)
    if not contract:
        return ["Use only documented edit-data-pipeline operation-specific parameters."]
    requirements = list(contract.get("requirements") or [])
    examples = contract.get("supported_labels") or []
    if examples:
        requirements.append(f"Supported label examples: {', '.join(str(item) for item in examples)}")
    return requirements


def _apply_semantic_transform_projection(
    operation: str,
    params: dict[str, Any],
    source_text: Any,
) -> dict[str, Any]:
    """Project explicit semantic transform params into text-edit fields."""

    if operation != "text_edit" or "semantic_transform" not in params:
        return params
    if not isinstance(source_text, str) or not source_text.strip():
        raise OperationBoundaryError("edit_data_pipeline.text_edit semantic_transform projection requires source_text")
    try:
        projected = project_semantic_transform_for_text_edit(params["semantic_transform"], source_text)
    except SemanticProjectionError as exc:
        raise OperationBoundaryError(f"edit_data_pipeline.text_edit semantic_transform projection failed: {exc}") from exc
    merged = dict(params)
    merged.update(projected)
    return merged


def validate_edit_data_pipeline_instruction_st(
    *,
    operation: str,
    source_text: str,
    instruction_st: str,
    target_text: Any = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Validate the final structured label string accepted by the service wrapper."""

    _validate_tagged_instruction(
        operation=operation.removeprefix("edit_data_pipeline.").replace("-", "_"),
        source_text=source_text,
        instruction_st=instruction_st,
        target_text=target_text,
        params=params or {},
    )


def validate_planner_edit_data_pipeline_contract(
    *,
    operation: str,
    params: dict[str, Any],
) -> None:
    """Validate Planner-authored tagged values without inventing edit semantics.

    Reversible service encodings are accepted here so PlanAdmission and the
    Executor use the same boundary. Natural-language prose, missing source
    wording, and contradictory target text remain blocking contract errors.
    """

    normalized_operation = operation.removeprefix("edit_data_pipeline.").replace(
        "-", "_"
    )
    if normalized_operation not in TAGGED_EDIT_OPERATIONS:
        return
    source_text = params.get("source_text")
    instruction_st = params.get("instruction_st")
    if not isinstance(source_text, str) or not source_text.strip():
        raise OperationBoundaryError(
            f"edit_data_pipeline.{normalized_operation} requires source_text"
        )
    if not isinstance(instruction_st, str) or not instruction_st.strip():
        raise OperationBoundaryError(
            f"edit_data_pipeline.{normalized_operation} requires a complete tagged instruction_st"
        )

    canonical_instruction = instruction_st
    if normalized_operation == "text_edit":
        canonical_instruction = _sanitize_text_edit_instruction_for_service(
            canonical_instruction
        )
        canonical_instruction = _repair_adjacent_text_edit_target_spacing(
            source_text=source_text,
            instruction_st=canonical_instruction,
            target_text=params.get("target_text"),
        )
    elif normalized_operation == "emotion_style_edit":
        canonical_instruction = _canonicalize_explicit_emotion_brackets_for_service(
            canonical_instruction
        )
        canonical_instruction = _canonicalize_explicit_emotion_instruction_for_service(
            canonical_instruction
        )

    _validate_tagged_instruction(
        operation=normalized_operation,
        source_text=source_text,
        instruction_st=canonical_instruction,
        target_text=params.get("target_text"),
        params=params,
    )


def normalize_step_for_operation(
    step: AudioEditStep,
    state: dict[str, Any],
) -> NormalizationAudit:
    """Normalize a step transactionally and reject semantic rewrites."""

    before = normalization_snapshot(
        operation=step.operation,
        inputs=copy.deepcopy(step.inputs),
        params=copy.deepcopy(step.params),
    )
    state_before = {
        key: copy.deepcopy(state[key]) if key in state else None
        for key in _NORMALIZATION_STATE_KEYS
    }
    state_keys_present = {key for key in _NORMALIZATION_STATE_KEYS if key in state}
    _normalize_step_for_operation_unchecked(step, state)
    after = normalization_snapshot(
        operation=step.operation,
        inputs=copy.deepcopy(step.inputs),
        params=copy.deepcopy(step.params),
    )
    audit = build_normalization_audit(before, after)
    state.setdefault("normalization_audits", {})[step.id] = audit.as_dict()
    if audit.semantics_preserved:
        return audit

    step.operation = before.operation
    step.inputs = copy.deepcopy(before.inputs)
    step.params = copy.deepcopy(before.params)
    for key in _NORMALIZATION_STATE_KEYS:
        if key in state_keys_present:
            state[key] = state_before[key]
        else:
            state.pop(key, None)
    state.setdefault("normalization_audits", {})[step.id] = audit.as_dict()
    changed = ", ".join(
        item.path for item in audit.transformations if not item.semantics_preserving
    )
    raise NonEquivalentNormalizationError(
        "normalization would change Planner semantics"
        f"{f' at {changed}' if changed else ''}; reject and replan instead of repairing"
    )


_NORMALIZATION_STATE_KEYS = (
    "artifact_ops_reconstruct_normalizations",
    "operation_param_normalizations",
    "operation_rewrites",
    "source_text_repairs",
)


def _normalize_step_for_operation_unchecked(
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    if step.operation.startswith("edit_data_pipeline."):
        _normalize_edit_data_pipeline_step(step, state)
    elif step.operation in {"seed_vc", "tse"}:
        _normalize_reference_audio_step(step, state)
    elif step.operation == "qwen3_tts":
        _normalize_qwen3_tts_step(step, state)
    elif step.operation == "artifact_ops.speaker_scoped_edit":
        _normalize_speaker_scoped_edit_step(step, state)
    elif step.operation == "foreground_background_remix":
        _normalize_foreground_background_remix_step(step, state)
    elif step.operation == "artifact_ops.reconstruct":
        _normalize_reconstruct_step(step, state)
    elif step.operation == "artifact_ops.reorder_spans":
        _normalize_reorder_spans_step(step)
    elif step.operation == "artifact_ops.concat":
        _normalize_concat_dependency_inputs(step, state)
    elif step.operation == "audio_effects":
        _normalize_audio_effects_step(step, state)
    elif step.operation == "spatial_reverb":
        step.params.setdefault("preserve_duration", True)


def _normalize_concat_dependency_inputs(
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    """Collapse resolved dependency outputs into concat's canonical source list."""

    primary = step.inputs.get("input_audio")
    if not isinstance(primary, list) or not primary or not all(
        isinstance(item, str) and item for item in primary
    ):
        return
    step_outputs = state.get("step_outputs")
    if not isinstance(step_outputs, dict) or not step.depends_on:
        return
    dependency_paths: list[str] = []
    for dependency_id in step.depends_on:
        result = step_outputs.get(dependency_id)
        if not isinstance(result, dict):
            return
        path = next(
            (
                result.get(key)
                for key in (
                    "output_audio_path",
                    "audio_path",
                    "output_path",
                )
                if isinstance(result.get(key), str) and result.get(key)
            ),
            None,
        )
        if path is None:
            return
        dependency_paths.append(path)
    if len(primary) != len(dependency_paths) or set(primary) != set(
        dependency_paths
    ):
        return
    for mapping in (step.inputs, step.params):
        for key in (
            "audio_paths",
            "input_audio_paths",
            "input_audios",
            "sources",
        ):
            mapping.pop(key, None)
    step.inputs["input_audio"] = primary[0]
    step.inputs["audio_paths"] = list(primary)


def _apply_voice_design_tool_preference(step: AudioEditStep, state: dict[str, Any]) -> None:
    if step.operation not in {"qwen3_tts"}:
        return
    params = dict(step.params)
    if step.operation == "qwen3_tts":
        mode = str(params.get("mode") or "custom_voice")
        ref_audio = (
            params.get("ref_audio_path")
            or params.get("voice_clone_reference_path")
            or params.get("voice_clone_reference_audio_path")
            or params.get("voice_clone_reference")
            or params.get("voice_clone_ref_path")
            or params.get("reference_audio_path")
            or step.inputs.get("ref_audio_path")
            or step.inputs.get("voice_clone_reference_path")
            or step.inputs.get("voice_clone_reference_audio_path")
            or step.inputs.get("voice_clone_reference")
            or step.inputs.get("reference_audio_path")
            or step.inputs.get("voice_clone_ref_path")
        )
        if mode == "voice_clone" or ref_audio:
            return
    preference = _voice_design_tool_preference(state)
    if preference == "qwen3_tts":
        step.operation = "qwen3_tts"
        params.setdefault("mode", "custom_voice")
        step.params = params


def _voice_design_tool_preference(state: dict[str, Any]) -> str | None:
    context = state.get("context") or {}
    value = None
    preferences = context.get("tool_preferences")
    if isinstance(preferences, dict):
        value = preferences.get("no_reference_voice_design") or preferences.get("voice_design_tool")
    value = value or context.get("voice_design_tool")
    plan = state.get("plan")
    global_params = getattr(plan, "global_params", None)
    if not value and isinstance(global_params, dict):
        plan_preferences = global_params.get("tool_preferences")
        if isinstance(plan_preferences, dict):
            value = plan_preferences.get("no_reference_voice_design") or plan_preferences.get("voice_design_tool")
        value = value or global_params.get("voice_design_tool")
    if value in (None, "", "auto"):
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text in {"qwen", "qwen3", "qwen3tts", "qwen3_tts"}:
        return "qwen3_tts"
    raise OperationBoundaryError(f"Unsupported voice-design tool preference: {value!r}")


def _normalize_audio_effects_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    params = dict(step.params)
    explicit_preset = params.get("preset") or params.get("effect") or params.get("effect_name")
    if explicit_preset in (None, ""):
        raise OperationBoundaryError(
            "audio_effects requires an explicit preset; normalization does not infer effects from user text"
        )
    preset = canonical_audio_effect_preset(str(explicit_preset))
    if not preset:
        raise OperationBoundaryError(f"audio_effects preset is unsupported: {explicit_preset!r}")
    params["preset"] = preset
    params.pop("effect", None)
    params.pop("effect_name", None)
    params.setdefault("preserve_duration", True)
    step.params = params


def _requests_spatial_reverb_effect(text: str) -> bool:
    lowered = text.lower()
    if not lowered:
        return False
    if any(
        term in lowered
        for term in (
            "remove reverb",
            "remove the reverb",
            "remove echo",
            "remove the echo",
            "dereverb",
            "de-reverb",
        )
    ):
        return False
    return any(
        term in lowered
        for term in (
            "sense of space",
            "spatial",
            "spacious",
            "roomy",
            "room ambience",
            "room ambiance",
            "reverb",
            "reverberation",
            "echo",
            "hall",
            "auditorium",
            "cathedral",
            "bathroom",
            "large room",
            "empty room",
            "空间感",
            "混响",
            "回声",
            "空荡",
            "空旷",
        )
    )


def build_edit_data_pipeline_tool_arguments(
    *,
    operation: str,
    instruction: str,
    source_text: str | None,
    input_audio_path: str,
    output_path: str,
    output_dir: str | None,
    language: str | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build the final service payload for edit-data-pipeline operations.

    Planner params are treated as authoritative semantics. This function may
    encode explicit typed fields for the service, but it never reparses the
    natural-language instruction to invent missing edit semantics.
    """

    operation = operation.removeprefix("edit_data_pipeline.")
    source = _plain_source_text(source_text or params.get("source_text"))
    params = _normalize_pitch_rate_params(
        operation,
        params,
        source_text=source if isinstance(source, str) else None,
        instruction=instruction,
    )
    params = _apply_semantic_transform_projection(operation, params, source)
    _promote_text_edit_tags_from_instruction(params, source_text=source, operation=operation)
    if operation == "emotion_style_edit" and isinstance(params.get("instruction_st"), str):
        normalized_instruction = _canonicalize_explicit_emotion_instruction_for_service(
            str(params["instruction_st"])
        )
        if normalized_instruction != params["instruction_st"]:
            params = dict(params)
            params["instruction_st"] = normalized_instruction
            params["emotion_condition_mode"] = (
                "desc" if _valid_canonical_emotion_desc_instruction(normalized_instruction) else "type_level"
            )
    language = _edit_pipeline_language(canonicalize_language(language or params.get("language")))
    if operation == "emotion_style_edit":
        language = _emotion_style_service_language(language)
    clean_params = filter_edit_pipeline_service_overrides(operation, params)
    if operation == "emotion_style_edit":
        clean_params.pop("language", None)
    if language:
        clean_params["language"] = language

    args: dict[str, Any] = {
        "operation": operation,
        "input_audio_path": input_audio_path,
        "output_path": output_path,
        "instruction": instruction,
    }

    if isinstance(source, str) and source.strip():
        args["source_text"] = source

    if operation in TAGGED_EDIT_OPERATIONS:
        if not isinstance(source, str) or not source.strip():
            raise OperationBoundaryError(f"edit_data_pipeline.{operation} requires source_text")
        _raise_for_unsupported_tagged_style(operation, instruction)
        if operation == "text_edit" and _should_use_rendered_target_from_instruction_st(source, params):
            params = dict(params)
            params["target_text"] = _render_text_edit_target(str(params["instruction_st"]))

        if _has_valid_user_supplied_instruction(operation, source, params):
            args["instruction_st"] = _sanitize_text_edit_instruction_for_service(
                str(params["instruction_st"])
            ) if operation == "text_edit" else str(params["instruction_st"])
            target_text = params.get("target_text")
            if operation == "text_edit" and not (isinstance(target_text, str) and target_text):
                target_text = _text_edit_target_text(source, params)
            if isinstance(target_text, str) and target_text:
                args["target_text"] = target_text
            args.update(clean_params)
        elif operation == "text_edit" and _can_build_target_text_edit(source, params):
            target_text = str(_text_edit_target_text(source, params))
            args["instruction_st"] = _coarse_text_edit_instruction(source, target_text)
            args["target_text"] = target_text
            args.update(clean_params)
        else:
            raise OperationBoundaryError(
                f"edit_data_pipeline.{operation} could not convert the request to a valid tagged instruction_st; "
                "expose this as a tool-boundary failure instead of forcing an invalid tool call"
            )

        args["operation"] = operation
        args["input_audio_path"] = input_audio_path
        args["output_path"] = output_path
        args["instruction"] = instruction
        args["source_text"] = source
        if operation == "pitch_rate_edit" and isinstance(args.get("instruction_st"), str):
            args["instruction_st"] = _normalize_pitch_instruction_semitones(str(args["instruction_st"]))
        if operation == "emotion_style_edit" and isinstance(args.get("instruction_st"), str):
            normalized_instruction = _canonicalize_explicit_emotion_instruction_for_service(
                str(args["instruction_st"])
            )
            if normalized_instruction != args["instruction_st"]:
                args["instruction_st"] = normalized_instruction
                args["emotion_condition_mode"] = "desc" if _valid_canonical_emotion_desc_instruction(normalized_instruction) else "type_level"
        if language:
            args["language"] = language
        _validate_tagged_instruction(
            operation=operation,
            source_text=source,
            instruction_st=str(args.get("instruction_st") or ""),
            target_text=args.get("target_text"),
            params=args,
        )
    else:
        _validate_untagged_edit_pipeline_params(operation, params)
        args.update(clean_params)
        if language:
            args["language"] = language

    _apply_edit_data_pipeline_defaults(operation, instruction, source, input_audio_path, args)
    return args


def filter_edit_pipeline_service_overrides(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """Keep only operation-specific service knobs and drop planner-only fields."""

    params = _isolate_edit_pipeline_parameter_ownership(operation, params)
    allowed = EDIT_PIPELINE_OVERRIDE_KEYS.get(operation, set())
    result: dict[str, Any] = {}
    for key, value in params.items():
        if key in PROTECTED_EDIT_PIPELINE_KEYS:
            continue
        if key in allowed:
            result[key] = _canonicalize_known_value(key, value, operation=operation)
    return result


def _isolate_edit_pipeline_parameter_ownership(
    operation: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Drop foreign operation knobs before target-operation defaults are applied.

    ``method`` is intentionally overloaded by several services. If an explicit
    operation-owned method field carries the same value, that field establishes
    the value's provenance and the ambiguous shared ``method`` must not leak into
    a different nested operation.
    """

    isolated = dict(params)
    foreign_method_values: list[Any] = []
    for key, owners in EDIT_PIPELINE_PARAMETER_OWNERS.items():
        if key not in isolated or operation in owners:
            continue
        foreign_method_values.append(isolated.pop(key))
    if "method" in isolated and any(
        isolated["method"] == value for value in foreign_method_values
    ):
        isolated.pop("method", None)
    return isolated


def canonicalize_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower().replace("_", "-")
    mapping = {
        "ar": "ara",
        "arabic": "ara",
        "cs": "ces",
        "czech": "ces",
        "da": "dan",
        "danish": "dan",
        "de": "deu",
        "german": "deu",
        "el": "ell",
        "greek": "ell",
        "en": "eng",
        "english": "eng",
        "en-us": "eng",
        "en-gb": "eng",
        "fa": "fas",
        "farsi": "fas",
        "persian": "fas",
        "filipino": "fil",
        "tl": "fil",
        "tagalog": "fil",
        "fi": "fin",
        "finnish": "fin",
        "fr": "fra",
        "french": "fra",
        "hi": "hin",
        "hindi": "hin",
        "hu": "hun",
        "hungarian": "hun",
        "id": "ind",
        "indonesian": "ind",
        "it": "ita",
        "italian": "ita",
        "ja": "jpn",
        "japanese": "jpn",
        "ja-jp": "jpn",
        "ko": "kor",
        "korean": "kor",
        "ms": "msa",
        "malay": "msa",
        "nl": "nld",
        "dutch": "nld",
        "pl": "pol",
        "polish": "pol",
        "pt": "por",
        "portuguese": "por",
        "ro": "ron",
        "romanian": "ron",
        "ru": "rus",
        "russian": "rus",
        "es": "spa",
        "spanish": "spa",
        "sv": "swe",
        "swedish": "swe",
        "th": "tha",
        "thai": "tha",
        "tr": "tur",
        "turkish": "tur",
        "vi": "vie",
        "vietnamese": "vie",
        "yue": "yue",
        "cantonese": "yue",
        "zh": "zho",
        "chinese": "zho",
        "mandarin": "zho",
        "cmn": "zho",
        "zh-cn": "zho",
        "zh-hans": "zho",
        "中文": "zho",
        "普通话": "zho",
    }
    return mapping.get(lowered, lowered)


def _edit_pipeline_language(value: str | None) -> str | None:
    if value in {"eng", "en-us", "en-gb"}:
        return "en"
    if value in {"zho", "cmn", "zh-cn", "zh-hans"}:
        return "zh"
    return value


def _emotion_style_service_language(value: str | None) -> str | None:
    if value in {"en", "eng", "english"}:
        return "en"
    if value in {"zh", "zho", "cmn", "chinese"}:
        return "zh"
    return None


def _normalize_edit_data_pipeline_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    operation = step.operation.split(".", 1)[1]
    params = dict(step.params)
    instruction = str(
        params.get("instruction")
        or state.get("user_prompt")
        or step.description
        or ""
    )
    source_text, source_text_repair = _edit_pipeline_source_text_for_step(
        operation=operation,
        params=params,
        local_source_text=_local_source_text_for_step(step, state),
        state_source_text=state.get("source_text"),
    )
    if source_text_repair:
        state.setdefault("source_text_repairs", {})[step.id] = source_text_repair
    language = _edit_pipeline_language(canonicalize_language(params.get("language") or _language_from_state(state)))
    if operation == "emotion_style_edit":
        language = _emotion_style_service_language(language)

    params.pop("operation", None)
    params.pop("language", None)
    if language:
        params["language"] = language

    if operation == "pitch_shift" and "semitones" not in params:
        raise OperationBoundaryError(
            "edit_data_pipeline.pitch_shift requires explicit semitones; "
            "normalization does not infer semantic factors from user text"
        )
    if operation == "pitch_shift":
        params["method"] = _canonicalize_pitch_shift_method(
            params.get("method"), strict=params.get("method") not in (None, "")
        )
    if operation == "volume_normalize":
        original_method = params.get("method")
        params["method"] = _canonicalize_volume_normalize_method(
            original_method,
            strict=original_method not in (None, ""),
        )
    if operation == "speech_rate":
        if params.get("rate") is None:
            raise OperationBoundaryError(
                "edit_data_pipeline.speech_rate requires explicit rate; "
                "normalization does not infer semantic factors from user text"
            )
        params.setdefault("method", "wsola+indextts2")
        if source_text and "text" not in params:
            params["text"] = source_text
    if operation == "audio_assembly":
        _validate_audio_assembly_label_contract(params)
        if params.get("target_text") and not params.get("edit_tagged_text"):
            raise OperationBoundaryError(
                "edit_data_pipeline.audio_assembly does not accept target_text alone; "
                "it requires target_audio_path plus edit_tagged_text/time spans"
            )

    if operation in TAGGED_EDIT_OPERATIONS:
        params = _apply_semantic_transform_projection(operation, params, source_text)
        params = _normalize_pitch_rate_params(operation, params, source_text=source_text, instruction=instruction)
        _promote_text_edit_tags_from_instruction(params, source_text=source_text, operation=operation)
        if operation == "emotion_style_edit" and isinstance(params.get("instruction_st"), str):
            params["instruction_st"] = _canonicalize_explicit_emotion_brackets_for_service(
                str(params["instruction_st"])
            )
        if not isinstance(source_text, str) or not source_text.strip():
            raise OperationBoundaryError(f"edit_data_pipeline.{operation} requires source_text")
        _raise_for_unsupported_tagged_style(operation, instruction)
        if operation == "text_edit" and isinstance(params.get("instruction_st"), str):
            params["instruction_st"] = _expand_unique_local_text_edit_to_context(
                source_text=source_text,
                instruction_st=str(params["instruction_st"]),
            )
            params["instruction_st"] = _repair_adjacent_text_edit_target_spacing(
                source_text=source_text,
                instruction_st=str(params["instruction_st"]),
                target_text=params.get("target_text"),
            )
            params["instruction_st"] = _canonicalize_whitespace_equivalent_text_edit_instruction(
                source_text=source_text,
                instruction_st=str(params["instruction_st"]),
                target_text=params.get("target_text"),
            )
        if operation == "text_edit" and _should_use_rendered_target_from_instruction_st(source_text, params):
            params["target_text"] = _render_text_edit_target(str(params["instruction_st"]))
        if _has_valid_user_supplied_instruction(operation, source_text, params):
            params["instruction_st"] = str(params["instruction_st"])
            if operation == "text_edit" and not params.get("target_text"):
                target_text = _text_edit_target_text(source_text, params)
                if isinstance(target_text, str) and target_text.strip():
                    params["target_text"] = target_text
        elif operation == "text_edit" and _has_partial_text_edit_instruction(
            source_text,
            params.get("instruction_st"),
        ):
            raise OperationBoundaryError(
                "edit_data_pipeline.text_edit local instruction_st could not be projected "
                "to one unique exact occurrence in source_text"
            )
        elif operation == "text_edit" and _can_build_target_text_edit(source_text, params):
            target_text = str(_text_edit_target_text(source_text, params))
            params["instruction_st"] = _coarse_text_edit_instruction(source_text, target_text)
            params["target_text"] = target_text
        else:
            raise OperationBoundaryError(
                f"edit_data_pipeline.{operation} requires instruction_st conversion; "
                "mechanical conversion failed and supplied instruction_st was missing or invalid"
            )
        if operation == "pitch_rate_edit" and isinstance(params.get("instruction_st"), str):
            params["instruction_st"] = _normalize_pitch_instruction_semitones(str(params["instruction_st"]))
        if operation == "emotion_style_edit" and isinstance(params.get("instruction_st"), str):
            normalized_instruction = _canonicalize_explicit_emotion_instruction_for_service(
                str(params["instruction_st"])
            )
            if normalized_instruction != params["instruction_st"]:
                params["instruction_st"] = normalized_instruction
                params["emotion_condition_mode"] = (
                    "desc"
                    if _valid_canonical_emotion_desc_instruction(normalized_instruction)
                    else "type_level"
                )
        params["source_text"] = source_text

    if operation not in TAGGED_EDIT_OPERATIONS:
        _validate_untagged_edit_pipeline_params(operation, params)

    params = _sanitize_edit_step_params(operation, params)
    step.params = params


def _maybe_rewrite_misrouted_edit_data_pipeline_operation(
    step: AudioEditStep,
    state: dict[str, Any],
    operation: str,
    params: dict[str, Any],
) -> str:
    if operation != "text_edit":
        return operation
    raw_nested = params.get("operation")
    if raw_nested is None:
        nested = _infer_direct_text_edit_misroute_target(step, params)
        nested_source = None
    else:
        nested = str(raw_nested).strip().removeprefix("edit_data_pipeline.").replace("-", "_")
        nested_source = str(raw_nested)
    if not nested or nested in {"", "text_edit"}:
        return operation
    if nested not in EDIT_DATA_PIPELINE_SAFE_MISROUTE_TARGETS:
        return operation

    original_operation = step.operation
    target_operation = f"edit_data_pipeline.{nested}"
    step.operation = target_operation
    record = {
        "from": original_operation,
        "to": target_operation,
        "reason": (
            "planner selected edit_data_pipeline.text_edit but the step carries operation-specific "
            "edit-data-pipeline evidence; using the operation-specific contract instead"
        ),
    }
    if nested_source is not None:
        record["nested_operation"] = nested_source
    else:
        record["inferred_operation"] = nested
    rewrites = state.setdefault("operation_rewrites", {})
    existing = rewrites.get(step.id)
    if isinstance(existing, dict):
        existing["misrouted_edit_data_pipeline_operation"] = record
        existing.setdefault("from", original_operation)
        existing["to"] = target_operation
    else:
        rewrites[step.id] = record
    return nested


def _infer_direct_text_edit_misroute_target(step: AudioEditStep, params: dict[str, Any]) -> str | None:
    """Recover planner steps that put pitch/rate knobs directly under text_edit.

    This intentionally stays conservative: lexical text edits with a target text
    or supported text-edit tags remain text_edit, even if a stray prosody field
    is present.
    """

    if _has_lexical_text_edit_evidence(params):
        return None
    if _has_direct_pitch_rate_evidence(step, params):
        return "pitch_rate_edit"
    return None


def _has_lexical_text_edit_evidence(params: dict[str, Any]) -> bool:
    target_text = params.get("target_text")
    if isinstance(target_text, str) and target_text.strip():
        return True
    instruction_st = params.get("instruction_st")
    return isinstance(instruction_st, str) and _has_text_edit_tag(instruction_st)


def _has_direct_pitch_rate_evidence(step: AudioEditStep, params: dict[str, Any]) -> bool:
    if any(
        params.get(key) is not None
        for key in (
            "rate",
            "factor",
            "speed",
            "semitones",
            "pitch_method",
            "rate_method",
            "psola_indextts2_cco_pitch_boost",
        )
    ):
        return True
    evidence = " ".join(
        str(value)
        for value in [
            step.description,
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("instruction_st"),
        ]
        if value
    ).lower()
    return bool(
        re.search(r"<(?:pitch|rate|speed),", evidence)
        or re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?\s*(?:semitones?|x|倍)", evidence)
        or _contains_any(
            evidence,
            [
                "pitch",
                "semitone",
                "higher voice",
                "lower voice",
                "raise the voice",
                "lower the voice",
                "deeper voice",
                "speaking rate",
                "speech rate",
                "speed up",
                "slow down",
                "faster",
                "slower",
                "tempo",
                "音高",
                "升调",
                "降调",
                "语速",
                "加速",
                "减速",
            ],
        )
    )


def _rewrite_scoped_style_text_edit_to_artifact_step(
    step: AudioEditStep,
    params: dict[str, Any],
    instruction: str,
    source_text: str | None,
) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            step.description,
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("selection_instruction"),
            params.get("edit_op"),
            params.get("edit_op_params"),
        ]
        if value
    )
    has_scoped_edit_shape = any(
        key in params for key in ("edit_op", "edit_instruction", "selection_instruction", "edit_op_params", "edit_params")
    )
    if not has_scoped_edit_shape:
        return False
    lowered = evidence.lower()
    if not _contains_any(lowered, ["tone", "style", "prosody", "intonation", "questioning", "declarative"]):
        return False

    edit_op = str(params.get("edit_op") or "").removeprefix("edit_data_pipeline.")
    if _requests_question_to_declarative_from_text(lowered):
        nested_op = "edit_data_pipeline.emotion_style_edit"
    elif edit_op in {"pitch_rate_edit", "emotion_style_edit", "pause_edit", "volume_normalize"}:
        nested_op = f"edit_data_pipeline.{edit_op}"
    else:
        nested_op = "edit_data_pipeline.emotion_style_edit"

    edit_op_params: dict[str, Any] = {}
    for key in ("edit_op_params", "edit_params"):
        value = params.get(key)
        if isinstance(value, dict):
            edit_op_params.update(value)
    nested_instruction = str(
        params.get("edit_instruction")
        or edit_op_params.get("instruction")
        or params.get("instruction")
        or instruction
    )
    if nested_instruction:
        edit_op_params["instruction"] = nested_instruction
    if source_text:
        edit_op_params.setdefault("source_text", source_text)
    selection_instruction = str(
        params.get("selection_instruction")
        or params.get("instruction")
        or instruction
        or step.description
        or ""
    )
    step.operation = "artifact_ops.speaker_scoped_edit"
    step.params = {
        "instruction": instruction or step.description,
        "selection_instruction": selection_instruction,
        "edit_op": nested_op,
        "edit_op_params": edit_op_params,
    }
    if source_text:
        step.params["source_text"] = source_text
    return True


def _requests_question_to_declarative_from_text(text: str) -> bool:
    return _contains_any(
        text,
        ["questioning tone", "questioning intonation", "interrogative intonation", "rising intonation", "questioning"],
    ) and _contains_any(text, ["declarative tone", "declarative", "falling intonation", "statement tone"])


def _has_concrete_pitch_or_rate_request(params: dict[str, Any], instruction: str) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("instruction_st"),
        ]
        if value
    ).lower()
    return bool(
        re.search(r"<(?:pitch|rate|speed),", evidence)
        or any(params.get(key) is not None for key in ("semitones", "rate", "factor", "speed"))
        or re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?\s*(?:semitones?|x|倍)", evidence)
    )


def _local_source_text_for_step(step: AudioEditStep, state: dict[str, Any]) -> str | None:
    """Infer short local source text when a tagged edit consumes extracted spans."""

    for output in _candidate_span_outputs_for_step(step, state):
        spans = output.get("selected_spans")
        text = _source_text_from_selected_spans(spans)
        if text:
            return text
        text = _source_text_from_perception_spans(spans, state.get("perception_report") or {})
        if text:
            return text
    return None


def _edit_pipeline_source_text_for_step(
    *,
    operation: str,
    params: dict[str, Any],
    local_source_text: Any,
    state_source_text: Any,
) -> tuple[Any, dict[str, Any] | None]:
    param_source = _plain_source_text(params.get("source_text"))
    local_source = _plain_source_text(local_source_text)
    state_source = _plain_source_text(state_source_text)
    contextual_source = local_source or state_source
    source = param_source or contextual_source

    if (
        operation == "text_edit"
        and isinstance(param_source, str)
        and isinstance(contextual_source, str)
        and _text_edit_phrase_source_can_expand_to_context(param_source, contextual_source, params)
    ):
        context_name = "local_selected_spans" if local_source else "state.source_text"
        return contextual_source, {
            "strategy": "expand_phrase_source_text_to_full_context",
            "operation": f"edit_data_pipeline.{operation}",
            "candidate_source_text": param_source[:160],
            "context_source": context_name,
            "context_source_text_preview": contextual_source[:240],
            "reason": (
                "Planner supplied a phrase-level text_edit source_text even though the current audio "
                "context has a longer transcript containing that phrase. Rebuilding the service label "
                "against the full context preserves the edit-data-pipeline text_edit contract."
            ),
        }
    return source, None


def _text_edit_phrase_source_can_expand_to_context(
    candidate_source: str,
    context_source: str,
    params: dict[str, Any],
) -> bool:
    candidate = _compact_text(candidate_source)
    context = _compact_text(context_source)
    if not candidate or not context or candidate == context:
        return False
    if candidate not in context:
        return False
    instruction_st = params.get("instruction_st")
    if isinstance(instruction_st, str) and instruction_st.strip() and _has_text_edit_tag(instruction_st):
        rendered_source = _compact_text(_render_text_edit_source(instruction_st))
        if rendered_source and rendered_source not in {candidate, context}:
            return False
        rendered_target = _compact_text(_render_text_edit_target(instruction_st))
        if rendered_target == context:
            return False
    target_text = params.get("target_text")
    if isinstance(target_text, str) and _compact_text(target_text) == context:
        return False
    return True


def _candidate_span_outputs_for_step(step: AudioEditStep, state: dict[str, Any]) -> list[dict[str, Any]]:
    step_outputs = state.get("step_outputs") or {}
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
        if isinstance(value, dict) and id(value) not in seen and isinstance(value.get("selected_spans"), list):
            candidates.append(value)
            seen.add(id(value))

    for dep_id in step.depends_on:
        add(step_outputs.get(dep_id))

    input_refs = {
        str(value)
        for value in [
            step.inputs.get("input_audio_path"),
            step.inputs.get("input_audio"),
            step.inputs.get("audio_path"),
            step.inputs.get("audio"),
        ]
        if value
    }
    if input_refs:
        for output in step_outputs.values():
            if not isinstance(output, dict):
                continue
            output_path = output.get("output_audio_path") or output.get("path")
            if isinstance(output_path, str) and output_path in input_refs:
                add(output)
    return candidates


def _source_text_from_selected_spans(spans: Any) -> str | None:
    if not isinstance(spans, list):
        return None
    pieces = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = span.get("text") or span.get("transcript")
        if isinstance(text, str) and text.strip():
            pieces.append(text.strip())
    return " ".join(pieces).strip() or None


def _source_text_from_perception_spans(spans: Any, perception: dict[str, Any]) -> str | None:
    if not isinstance(spans, list) or not isinstance(perception, dict):
        return None
    pieces = []
    segments = _perception_segments(perception)
    words = _perception_words(perception)
    for span in spans:
        if not isinstance(span, dict):
            continue
        start = _span_time(span, "start")
        end = _span_time(span, "end")
        if start is None or end is None or end <= start:
            continue
        segment_text = _best_segment_text_for_span(segments, start, end)
        if segment_text:
            pieces.append(segment_text)
            continue
        word_text = _word_text_for_span(words, start, end)
        if word_text:
            pieces.append(word_text)
    return " ".join(pieces).strip() or None


def _perception_segments(perception: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for key in ("segments", "speaker_segments"):
        value = perception.get(key)
        if isinstance(value, list):
            segments.extend(item for item in value if isinstance(item, dict))
    nested = perception.get("gemini_perception")
    if isinstance(nested, dict):
        for key in ("segments", "speaker_segments"):
            value = nested.get(key)
            if isinstance(value, list):
                segments.extend(item for item in value if isinstance(item, dict))
    return segments


def _perception_words(perception: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for key in ("word_timestamps", "words"):
        value = perception.get(key)
        if isinstance(value, list):
            words.extend(item for item in value if isinstance(item, dict))
    nested = perception.get("gemini_perception")
    if isinstance(nested, dict):
        for key in ("word_timestamps", "words"):
            value = nested.get(key)
            if isinstance(value, list):
                words.extend(item for item in value if isinstance(item, dict))
    return words


def _best_segment_text_for_span(segments: list[dict[str, Any]], start: float, end: float) -> str | None:
    best_text = None
    best_overlap = 0.0
    span_duration = max(0.001, end - start)
    for segment in segments:
        seg_start = _span_time(segment, "start")
        seg_end = _span_time(segment, "end")
        if seg_start is None or seg_end is None or seg_end <= seg_start:
            continue
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap <= best_overlap or overlap / span_duration < 0.35:
            continue
        text = segment.get("transcript") or segment.get("text")
        if isinstance(text, str) and text.strip():
            best_text = text.strip()
            best_overlap = overlap
    return best_text


def _word_text_for_span(words: list[dict[str, Any]], start: float, end: float) -> str | None:
    pieces = []
    for word in sorted(words, key=lambda item: _span_time(item, "start") or 0.0):
        word_start = _span_time(word, "start")
        word_end = _span_time(word, "end")
        if word_start is None or word_end is None:
            continue
        midpoint = (word_start + word_end) / 2.0
        if start <= midpoint <= end:
            text = word.get("text") or word.get("word")
            if isinstance(text, str) and text.strip():
                pieces.append(text.strip())
    return " ".join(pieces).strip() or None


def _span_time(mapping: dict[str, Any], edge: str) -> float | None:
    keys = (f"{edge}_sec", edge, f"{edge}_time")
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_style_only_text_edit_misroute(
    step: AudioEditStep,
    params: dict[str, Any],
    instruction: str,
    state: dict[str, Any],
) -> bool:
    if params.get("target_text"):
        return False
    instruction_st = params.get("instruction_st")
    if isinstance(instruction_st, str) and _has_text_edit_tag(instruction_st):
        return False
    if _has_text_content_edit_signal(params, instruction, step, state):
        return False
    text = _style_instruction_from_text_edit_params(step, params, instruction, state).lower()
    return _contains_any(
        text,
        [
            "internal monologue",
            "inner monologue",
            "internal thought",
            "voiceover",
            "voice-over",
            "hushed",
            "reflective",
            "introspective",
            "subdued",
            "melancholic",
            "calm tone",
            "soft tone",
            "tone",
            "mood",
            "emotion",
            "style",
            "语气",
            "情绪",
            "风格",
            "内心独白",
            "旁白",
        ],
    )


def _style_instruction_from_text_edit_params(
    step: AudioEditStep,
    params: dict[str, Any],
    instruction: str,
    state: dict[str, Any],
) -> str:
    pieces = [
        instruction,
        params.get("instruction_st") if isinstance(params.get("instruction_st"), str) else None,
        step.description,
        state.get("user_prompt"),
    ]
    return " ".join(str(piece).strip() for piece in pieces if str(piece or "").strip())


def _has_text_content_edit_signal(
    params: dict[str, Any],
    instruction: str,
    step: AudioEditStep,
    state: dict[str, Any],
) -> bool:
    if any(
        key in params
        for key in [
            "old_phrase",
            "new_phrase",
            "old_text",
            "new_text",
            "replace_with",
            "replacement_text",
            "insert_text",
            "delete_text",
        ]
    ):
        return True
    text = _style_instruction_from_text_edit_params(step, params, instruction, state).lower()
    if len(_quoted_phrases(text)) >= 2:
        return True
    return _contains_any(
        text,
        [
            "replace the word",
            "replace word",
            "replace phrase",
            "change the word",
            "change words",
            "swap the word",
            "swap words",
            "switch order",
            "switch the order",
            "substitute",
            "rewrite the text",
            "spoken text",
            "transcript",
            "替换词",
            "替换短语",
            "交换顺序",
        ],
    )


def _quoted_phrases(text: str) -> list[str]:
    return [
        match
        for groups in re.findall(r"'([^']+)'|\"([^\"]+)\"|“([^”]+)”|‘([^’]+)’", text)
        for match in groups
        if match
    ]


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_reference_audio_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    refs = state.get("reference_audio_paths") or []
    reference_audio = (
        step.params.get("reference_audio_path")
        or step.params.get("reference_audio")
        or step.params.get("ref_audio_path")
        or step.params.get("speaker_audio_path")
        or step.inputs.get("reference_audio_path")
        or step.inputs.get("reference_audio")
        or step.inputs.get("ref_audio_path")
        or step.inputs.get("speaker_audio_path")
        or (refs[0] if refs else None)
    )
    if not reference_audio:
        raise OperationBoundaryError(f"{step.operation} requires reference_audio_path")
    reference_audio = resolve_audio_reference(reference_audio, state)
    path = Path(str(reference_audio))
    if not path.exists():
        raise OperationBoundaryError(f"{step.operation} reference_audio_path does not exist: {path}")
    step.inputs["reference_audio_path"] = str(path)


def _normalize_qwen3_tts_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    params = dict(step.params)
    text = params.get("text") or state.get("source_text")
    if text:
        params["text"] = str(text)
    mode = str(params.get("mode") or "").strip()
    ref_audio = (
        params.get("ref_audio_path")
        or params.pop("voice_clone_reference_path", None)
        or params.pop("voice_clone_reference_audio_path", None)
        or params.pop("voice_clone_reference", None)
        or params.pop("voice_clone_ref_path", None)
        or params.pop("reference_audio_path", None)
        or step.inputs.get("ref_audio_path")
        or step.inputs.get("voice_clone_reference_path")
        or step.inputs.get("voice_clone_reference_audio_path")
        or step.inputs.get("voice_clone_reference")
        or step.inputs.get("reference_audio_path")
        or step.inputs.get("voice_clone_ref_path")
    )
    if mode == "voice_clone" and ref_audio:
        params["ref_audio_path"] = str(ref_audio)
    if mode == "voice_clone" and params.get("ref_audio_path") and not params.get("ref_text"):
        params["x_vector_only_mode"] = True
    step.params = params




def _normalize_speaker_scoped_edit_step(
    step: AudioEditStep,
    state: dict[str, Any],
) -> None:
    """Normalize the nested tool contract before its gateway verifier runs."""

    params = dict(step.params)
    edit_operation, edit_params = _foreground_background_nested_edit(params)
    nested_operation = edit_operation.removeprefix("edit_data_pipeline.")
    if nested_operation == "volume_normalize":
        isolated = _isolate_edit_pipeline_parameter_ownership(
            "volume_normalize", edit_params
        )
        if isolated.get("method") not in (None, ""):
            try:
                isolated["method"] = _canonicalize_volume_normalize_method(
                    isolated["method"], strict=True
                )
            except OperationBoundaryError:
                # The nested destination owns its method default. Foreign
                # shared-schema values must not cross the operation boundary.
                isolated.pop("method", None)
        clean = filter_edit_pipeline_service_overrides(
            "volume_normalize",
            isolated,
        )
        clean.setdefault("target_db", -18.0)
        # Nested parameter bags are assembled from heterogeneous tool schemas.
        # An invalid overloaded method belongs to the source schema, so the
        # destination operation owns the fallback instead of inheriting it.
        clean["method"] = _canonicalize_volume_normalize_method(
            clean.get("method"),
            strict=False,
        )
        params["edit_op"] = "edit_data_pipeline.volume_normalize"
        params["edit_op_params"] = clean
        params.pop("edit_operation", None)
        params.pop("edit_params", None)
    step.params = params


def _normalize_foreground_background_remix_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    params = dict(step.params)
    raw_action = str(params.get("foreground_action") or "").strip().casefold()
    action_aliases = {
        "edit_foreground": "edit_foreground",
        "remove_foreground": "remove_foreground",
        "remove_voice": "remove_foreground",
        "keep_background": "remove_foreground",
        "background_only": "remove_foreground",
        "isolate_foreground": "isolate_foreground",
        "foreground_only": "isolate_foreground",
        "extract_foreground": "isolate_foreground",
    }
    if raw_action in action_aliases:
        params["foreground_action"] = action_aliases[raw_action]
    edit_operation, edit_params = _foreground_background_nested_edit(params)
    source_text = params.get("source_text") or edit_params.get("source_text") or edit_params.get("text") or state.get("source_text")
    if edit_operation:
        params["edit_op"] = edit_operation
        params.pop("edit_operation", None)
        params["edit_op_params"] = edit_params
    nested_operation = edit_operation.removeprefix("edit_data_pipeline.") if edit_operation else ""
    if nested_operation in TAGGED_EDIT_OPERATIONS and source_text:
        params.setdefault("source_text", source_text)
        edit_params.setdefault("source_text", source_text)
    if nested_operation == "emotion_style_edit":
        _repair_normal_speech_nested_emotion_params(
            edit_params,
            instruction=str(params.get("instruction") or state.get("user_prompt") or step.description or ""),
            source_text=str(source_text or ""),
        )
    if nested_operation in {"speech_rate", "pitch_rate_edit"}:
        if _strengthen_time_scoped_rate_params(
            edit_params,
            instruction=str(params.get("instruction") or state.get("user_prompt") or step.description or ""),
        ):
            params["preserve_duration"] = False
    if nested_operation in TAGGED_EDIT_OPERATIONS and not (params.get("source_text") or state.get("source_text")):
        raise OperationBoundaryError(
            f"foreground_background_remix edit_operation={nested_operation!r} requires source_text for tagged edit"
        )
    step.params = params


def _foreground_background_nested_edit(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_edit = params.get("edit_op")
    if raw_edit is None:
        raw_edit = params.get("edit_operation")
    operation = ""
    nested_params: dict[str, Any] = {}
    if isinstance(raw_edit, dict):
        operation = str(raw_edit.get("operation") or raw_edit.get("tool") or raw_edit.get("tool_id") or "").strip()
        if isinstance(raw_edit.get("params"), dict):
            nested_params.update(raw_edit["params"])
    elif isinstance(raw_edit, str):
        operation = raw_edit.strip()
    explicit_params = params.get("edit_op_params")
    if explicit_params is None:
        explicit_params = params.get("edit_params")
    if isinstance(explicit_params, dict):
        nested_params.update(explicit_params)
    if operation and "." not in operation and operation in TAGGED_EDIT_OPERATIONS:
        operation = f"edit_data_pipeline.{operation}"
    return operation, nested_params


def _repair_normal_speech_nested_emotion_params(
    params: dict[str, Any],
    *,
    instruction: str,
    source_text: str,
) -> None:
    if not source_text.strip() or not _requests_crying_to_normal_speech(instruction):
        return
    params["instruction_st"] = f'<emo desc="normal speaking voice">{source_text}</emo>'
    params["emotion_condition_mode"] = "desc"


def _strengthen_time_scoped_rate_params(params: dict[str, Any], *, instruction: str) -> bool:
    lowered = str(instruction or "").lower()
    if not _contains_any(lowered, ["slow down", "slower", "reduce the speaking rate", "decrease the speaking rate"]):
        return False
    rate = params.get("rate", params.get("factor", params.get("speed")))
    try:
        value = float(rate)
    except (TypeError, ValueError):
        value = _infer_rate_factor(instruction)
    target_rate = min(value, 0.65)
    params["rate"] = target_rate
    instruction_st = params.get("instruction_st")
    if isinstance(instruction_st, str) and "<rate" in instruction_st:
        params["instruction_st"] = re.sub(
            r"(<rate,\s*factor=)([0-9]+(?:\.[0-9]+)?)(\s*>)",
            lambda match: f"{match.group(1)}{min(float(match.group(2)), 0.65):g}{match.group(3)}",
            instruction_st,
        )
    elif isinstance(params.get("source_text"), str) and str(params["source_text"]).strip():
        params["instruction_st"] = _full_rate_instruction(str(params["source_text"]), target_rate)
    return True


def _normalize_reconstruct_step(step: AudioEditStep, state: dict[str, Any]) -> None:
    params = dict(step.params)
    replacement_spans = params.get("replacement_spans")
    if isinstance(replacement_spans, list) and len(replacement_spans) == 1:
        replacement_spec = replacement_spans[0]
        replacement_audio = (
            replacement_spec.get("replacement_audio")
            or replacement_spec.get("replacement_audio_path")
            or replacement_spec.get("audio_path")
            if isinstance(replacement_spec, dict)
            else None
        )
        producer_id = _full_timeline_reconstruct_producer(
            replacement_audio,
            step=step,
            state=state,
        )
        if producer_id is not None:
            params.pop("replacement_spans", None)
            params["edited_audio_path"] = replacement_audio
            params["reconstruction_mode"] = "full_timeline_passthrough"
            params["edited_artifact_scope"] = "full_timeline"
            state.setdefault(
                "artifact_ops_reconstruct_normalizations", {}
            ).setdefault(step.id, []).append(
                {
                    "strategy": "use_full_timeline_scoped_edit_output",
                    "producer_step_id": producer_id,
                }
            )
    raw_mode = params.get("reconstruction_mode")
    mode = normalized_reconstruction_mode(raw_mode)
    if raw_mode not in (None, "") and mode is None:
        raise OperationBoundaryError(
            "artifact_ops.reconstruct reconstruction_mode must be one of: "
            f"{', '.join(sorted(RECONSTRUCTION_MODES))}"
        )
    raw_scope = params.get("edited_artifact_scope")
    scope = normalized_edited_artifact_scope(raw_scope)
    if raw_scope not in (None, "") and scope is None:
        raise OperationBoundaryError(
            "artifact_ops.reconstruct edited_artifact_scope must be one of: "
            "full_timeline, local_regions, none"
        )

    param_replacements = params.get("replacements")
    input_replacements = step.inputs.get("replacements")
    replacements = param_replacements or input_replacements
    if replacements:
        if not isinstance(replacements, list):
            raise OperationBoundaryError(
                "artifact_ops.reconstruct replacements must be a list of timeline edits"
            )
        existing_edits = params.get("edits") or step.inputs.get("edits")
        if existing_edits and existing_edits != replacements:
            raise OperationBoundaryError(
                "artifact_ops.reconstruct received conflicting edits and replacements"
            )
        params["edits"] = [
            {**item, "mode": item.get("mode", "replace")}
            if isinstance(item, dict)
            else item
            for item in replacements
        ]
        params.pop("replacements", None)
        step.inputs.pop("replacements", None)
        state.setdefault(
            "artifact_ops_reconstruct_normalizations", {}
        ).setdefault(step.id, []).append(
            {
                "strategy": "canonicalize_replacements_as_timeline_edits",
                "replacement_count": len(replacements),
            }
        )

    legacy_replacement_audio = (
        params.get("replacement_audio_path")
        or params.get("replacement_audio")
        or step.inputs.get("replacement_audio_path")
        or step.inputs.get("replacement_audio")
    )
    legacy_start = params.get("start_sec", params.get("replacement_start_sec"))
    legacy_end = params.get("end_sec", params.get("replacement_end_sec"))
    if legacy_start is None:
        legacy_start = step.inputs.get(
            "start_sec", step.inputs.get("replacement_start_sec")
        )
    if legacy_end is None:
        legacy_end = step.inputs.get(
            "end_sec", step.inputs.get("replacement_end_sec")
        )
    explicit_legacy_spans = (
        params.get("selected_spans")
        or params.get("spans")
        or step.inputs.get("selected_spans")
        or step.inputs.get("spans")
    )
    if (
        legacy_replacement_audio not in (None, "")
        and legacy_start is None
        and legacy_end is None
        and isinstance(explicit_legacy_spans, list)
        and explicit_legacy_spans
    ):
        if not params.get("edited_spans_audio_paths"):
            params["edited_spans_audio_paths"] = [legacy_replacement_audio]
        for key in ("replacement_audio_path", "replacement_audio"):
            params.pop(key, None)
            step.inputs.pop(key, None)
        state.setdefault(
            "artifact_ops_reconstruct_normalizations", {}
        ).setdefault(step.id, []).append(
            {
                "strategy": "pair_replacement_audio_with_explicit_spans",
                "replacement_count": 1,
                "span_count": len(explicit_legacy_spans),
            }
        )
        legacy_replacement_audio = None
    if any(
        value is not None
        for value in (legacy_replacement_audio, legacy_start, legacy_end)
    ):
        if (
            legacy_replacement_audio in (None, "")
            or legacy_start is None
            or legacy_end is None
        ):
            raise OperationBoundaryError(
                "artifact_ops.reconstruct legacy local replacement requires "
                "replacement_audio plus start_sec and end_sec"
            )
        had_explicit_spans = bool(
            params.get("selected_spans") or params.get("spans")
        )
        if not had_explicit_spans:
            params["selected_spans"] = [
                {"start_sec": legacy_start, "end_sec": legacy_end}
            ]
        if not params.get("edited_spans_audio_paths"):
            params["edited_spans_audio_paths"] = [legacy_replacement_audio]
        for key in (
            "replacement_audio_path",
            "replacement_audio",
            "start_sec",
            "end_sec",
            "replacement_start_sec",
            "replacement_end_sec",
        ):
            params.pop(key, None)
            step.inputs.pop(key, None)
        state.setdefault(
            "artifact_ops_reconstruct_normalizations", {}
        ).setdefault(step.id, []).append(
            {
                "strategy": "canonicalize_legacy_local_replacement",
                "replacement_count": 1,
                "span_source": (
                    "explicit_spans"
                    if had_explicit_spans
                    else "legacy_scalar_boundaries"
                ),
            }
        )

    edits = params.get("edits") or step.inputs.get("edits")
    if edits:
        if not isinstance(edits, list):
            raise OperationBoundaryError(
                "artifact_ops.reconstruct edits must be a list of timeline edit records"
            )
        params["edits"] = edits
    span_replacement_paths = (
        params.get("edited_spans_audio_paths")
        or params.get("edited_spans_audio")
        or step.inputs.get("edited_spans_audio_paths")
        or step.inputs.get("edited_spans_audio")
    )
    if span_replacement_paths:
        if not isinstance(span_replacement_paths, (list, tuple)):
            raise OperationBoundaryError(
                "artifact_ops.reconstruct edited_spans_audio_paths must be a list"
            )
        params["edited_spans_audio_paths"] = list(span_replacement_paths)
        params.pop("edited_spans_audio", None)
    replacement_spans = (
        params.get("selected_spans")
        or params.get("spans")
        or step.inputs.get("selected_spans")
        or step.inputs.get("spans")
    )
    if replacement_spans:
        if not isinstance(replacement_spans, list):
            raise OperationBoundaryError(
                "artifact_ops.reconstruct replacement spans must be a list"
            )
        params["selected_spans"] = replacement_spans
    edited_audio = (
        params.get("edited_audio_path")
        or params.get("edited_audio")
        or step.inputs.get("edited_audio_path")
        or step.inputs.get("edited_audio")
    )
    if edited_audio:
        params["edited_audio_path"] = edited_audio
        params.pop("edited_audio", None)

    inferred_mode: str
    inferred_scope: str
    if isinstance(edits, list) and edits:
        inferred_mode = "timeline_edits"
        inferred_scope = "local_regions"
    elif replacement_spans and (span_replacement_paths or edited_audio):
        inferred_mode = "span_replacement"
        inferred_scope = "local_regions"
    elif span_replacement_paths or replacement_spans:
        raise OperationBoundaryError(
            "artifact_ops.reconstruct local replacement requires both replacement "
            "audio artifacts and explicit spans"
        )
    elif edited_audio:
        if scope != "full_timeline":
            raise OperationBoundaryError(
                "artifact_ops.reconstruct received edited audio without replacement spans; "
                "declare edited_artifact_scope=full_timeline for a complete-timeline "
                "passthrough, or provide explicit spans for a local replacement"
            )
        inferred_mode = "full_timeline_passthrough"
        inferred_scope = "full_timeline"
    else:
        inferred_mode = "placeholder_compatibility"
        inferred_scope = "none"

    if mode is not None and mode != inferred_mode:
        raise OperationBoundaryError(
            "artifact_ops.reconstruct reconstruction_mode conflicts with its artifact inputs: "
            f"declared {mode}, inferred {inferred_mode}"
        )
    if scope is not None and scope != inferred_scope:
        raise OperationBoundaryError(
            "artifact_ops.reconstruct edited_artifact_scope conflicts with its artifact inputs: "
            f"declared {scope}, inferred {inferred_scope}"
        )
    params["reconstruction_mode"] = inferred_mode
    params["edited_artifact_scope"] = inferred_scope

    if inferred_mode in {"timeline_edits", "span_replacement"}:
        base_audio = (
            state.get("input_audio")
            or (state.get("artifacts") or {}).get("input_audio")
            or (state.get("artifacts") or {}).get("base")
        )
        if not base_audio:
            raise OperationBoundaryError(
                "artifact_ops.reconstruct local edits require the original/base input audio"
            )
        step.inputs["input_audio_path"] = str(base_audio)
    step.params = params


def _full_timeline_reconstruct_producer(
    replacement_audio: Any,
    *,
    step: AudioEditStep,
    state: dict[str, Any],
) -> str | None:
    if not isinstance(replacement_audio, str) or not replacement_audio:
        return None
    outputs = state.get("step_outputs")
    if not isinstance(outputs, dict):
        return None
    for producer_id in step.depends_on:
        output = outputs.get(producer_id)
        if not isinstance(output, dict):
            continue
        if output.get("operation") != "speaker_scoped_edit":
            continue
        if output.get("output_audio_path") == replacement_audio:
            return producer_id
    return None


def _normalize_reorder_spans_step(step: AudioEditStep) -> None:
    params = dict(step.params)
    params.setdefault("preserve_duration", True)
    raw_order = params.get("order")
    raw_speaker_order = params.get("target_speaker_order")
    if raw_order is not None and raw_speaker_order is not None:
        raise OperationBoundaryError(
            "artifact_ops.reorder_spans accepts either order or target_speaker_order, not both"
        )
    if raw_order is not None:
        explicit_spans = (
            params.get("selected_spans")
            or params.get("spans")
            or params.get("segments")
        )
        span_count = len(explicit_spans) if isinstance(explicit_spans, list) else None
        try:
            params["order"] = normalize_reorder_indices(
                raw_order,
                span_count=span_count,
            )
        except (TypeError, ValueError) as exc:
            raise OperationBoundaryError(
                f"artifact_ops.reorder_spans received invalid order: {exc}"
            ) from exc
    elif raw_speaker_order is not None:
        try:
            params["target_speaker_order"] = normalize_reorder_speaker_order(
                raw_speaker_order
            )
        except (TypeError, ValueError) as exc:
            raise OperationBoundaryError(
                "artifact_ops.reorder_spans received invalid target_speaker_order: "
                f"{exc}"
            ) from exc
    step.params = params


def _rewrite_audio_assembly_insert_as_reconstruct(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    action = str(params.get("operation") or params.get("action") or params.get("mode") or "").strip().lower()
    insert_time = params.get("insert_time_sec", params.get("time_sec", params.get("start_sec")))
    if action not in {"insert", "insertion", "append_at_time"} and insert_time is None:
        return False
    replacement = (
        params.get("target_audio_path")
        or params.get("target_audio")
        or params.get("insert_audio_path")
        or params.get("insert_audio")
        or step.inputs.get("reference_audio")
        or step.inputs.get("reference_audio_path")
        or step.inputs.get("target_audio")
        or step.inputs.get("target_audio_path")
        or state.get("current_audio_path")
    )
    if not replacement:
        raise OperationBoundaryError("audio_assembly insert requires a generated/target audio artifact")
    try:
        start_sec = float(insert_time)
    except (TypeError, ValueError) as exc:
        raise OperationBoundaryError("audio_assembly insert requires numeric insert_time_sec") from exc
    base_audio = (
        state.get("input_audio")
        or (state.get("artifacts") or {}).get("input_audio")
        or (state.get("artifacts") or {}).get("base")
    )
    if not base_audio:
        raise OperationBoundaryError("audio_assembly insert requires the original/base input audio")
    instruction = str(params.get("instruction") or step.description or state.get("user_prompt") or "")
    step.operation = "artifact_ops.reconstruct"
    step.inputs = {"input_audio_path": str(base_audio)}
    step.params = {
        "instruction": instruction,
        "edits": [
            {
                "start_sec": max(0.0, start_sec),
                "end_sec": max(0.0, start_sec),
                "input_audio_path": replacement,
            }
        ],
    }
    return True


def _rewrite_pure_concat_audio_assembly(
    step: AudioEditStep,
    state: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    instruction = str(params.get("instruction") or step.description or state.get("user_prompt") or "")
    lowered = instruction.lower()
    if not any(token in lowered for token in ["concatenate", "concat", "combine", "merge", "join", "拼接", "合并"]):
        return False
    if params.get("target_audio_path") or params.get("edit_tagged_text") or params.get("time_spans_from_source"):
        return False
    audio_paths = (
        params.get("audio_paths")
        or step.inputs.get("audio_paths")
        or step.inputs.get("input_audio_paths")
        or step.inputs.get("input_audios")
    )
    step.operation = "artifact_ops.concat"
    step.inputs = {"input_audio_path": step.inputs.get("input_audio_path") or "current"}
    if isinstance(audio_paths, list) and audio_paths:
        step.params = {"instruction": instruction, "audio_paths": audio_paths}
    else:
        step.params = {"instruction": instruction}
    return True


def _extract_structured_params(args: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "source_text",
        "instruction_st",
        "target_text",
        "language",
        "evaluate",
        "persist",
        "output_dir",
        "pitch_method",
        "smooth_policy",
        "return_synthetic_target_audio",
        "max_candidates",
        "profile_id",
        "stage2_method",
        "synth_method",
        "expand_radius",
    }
    return {key: value for key, value in args.items() if key in keys}


def _plain_source_text(value: Any) -> Any:
    if not isinstance(value, str) or "<" not in value or ">" not in value:
        return value
    return _compact_text(_strip_instruction_tags(value))


def _sanitize_edit_step_params(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    clean = filter_edit_pipeline_service_overrides(operation, params)
    if "instruction" in params:
        clean["instruction"] = params["instruction"]
    for key in ["source_text", "instruction_st", "target_text"]:
        if key in params:
            if key == "instruction_st" and operation == "text_edit" and isinstance(params[key], str):
                clean[key] = _sanitize_text_edit_instruction_for_service(params[key])
            else:
                clean[key] = params[key]
    return clean


def _promote_text_edit_tags_from_instruction(
    params: dict[str, Any],
    *,
    source_text: Any,
    operation: str,
) -> None:
    """Move a valid text-edit tag string from instruction to instruction_st.

    Planner/executor repair models sometimes place the edit-data-pipeline
    service label in the human-readable ``instruction`` field.  Promotion is
    allowed only for text_edit, only when ``instruction_st`` is absent, and only
    when rendering the tag script exactly matches the grounded source text.
    """

    if operation != "text_edit":
        return
    if isinstance(params.get("instruction_st"), str) and str(params.get("instruction_st")).strip():
        return
    instruction = params.get("instruction")
    if not isinstance(instruction, str) or not _has_text_edit_tag(instruction):
        return
    if not isinstance(source_text, str) or not source_text.strip():
        return
    if _compact_text(_render_text_edit_source(instruction)) != _compact_text(source_text):
        return
    rendered_target = _render_text_edit_target(instruction)
    if not rendered_target.strip() or _compact_text(rendered_target) == _compact_text(source_text):
        return
    existing_target = params.get("target_text")
    if isinstance(existing_target, str) and existing_target.strip():
        if _compact_text(existing_target) != _compact_text(rendered_target) and not _target_text_is_tag_fragment(
            existing_target,
            instruction,
        ):
            return
    params["instruction_st"] = instruction
    params["target_text"] = rendered_target


def _target_text_is_tag_fragment(target_text: str, instruction_st: str) -> bool:
    target = _compact_text(target_text)
    if not target:
        return False
    fragments: list[str] = []
    for match in _text_edit_tag_matches(instruction_st):
        if match.group("sub_attrs") is not None:
            sub_target = _sub_target_from_attrs(match.group("sub_attrs") or "")
            if sub_target:
                fragments.append(sub_target)
        elif match.group("ins_body") is not None:
            fragments.append(_strip_instruction_tags(match.group("ins_body") or ""))
    return any(_compact_text(fragment) == target for fragment in fragments if fragment)


def _expand_unique_local_text_edit_to_context(
    *,
    source_text: str,
    instruction_st: str,
) -> str:
    """Embed a complete local edit script at its unique exact source occurrence."""

    rendered_source = _render_text_edit_source(instruction_st)
    if not rendered_source or rendered_source == source_text:
        return instruction_st
    if source_text.count(rendered_source) != 1:
        return instruction_st
    start = source_text.index(rendered_source)
    end = start + len(rendered_source)
    expanded = (
        escape(source_text[:start], quote=False)
        + instruction_st
        + escape(source_text[end:], quote=False)
    )
    if _render_text_edit_source(expanded) != source_text:
        return instruction_st
    return expanded


def _has_partial_text_edit_instruction(source_text: str, instruction_st: Any) -> bool:
    if not isinstance(instruction_st, str) or not _has_text_edit_tag(instruction_st):
        return False
    rendered_source = _render_text_edit_source(instruction_st)
    return bool(rendered_source) and rendered_source != source_text


def _has_valid_user_supplied_instruction(operation: str, source_text: str, params: dict[str, Any]) -> bool:
    instruction_st = params.get("instruction_st")
    if not isinstance(instruction_st, str) or not instruction_st.strip():
        return False
    if operation == "text_edit":
        target_text = _text_edit_target_text(source_text, params)
        if isinstance(target_text, str) and target_text.strip() and not (
            _equivalent_transcript_text(_render_text_edit_source(instruction_st), source_text)
            and _equivalent_target_text(_render_text_edit_target(instruction_st), target_text)
        ):
            return False
    try:
        _validate_tagged_instruction(
            operation=operation,
            source_text=source_text,
            instruction_st=instruction_st,
            target_text=params.get("target_text"),
            params=params,
        )
    except OperationBoundaryError:
        return False
    return True


def _should_use_rendered_target_from_instruction_st(source_text: str, params: dict[str, Any]) -> bool:
    instruction_st = params.get("instruction_st")
    if not isinstance(instruction_st, str) or not instruction_st.strip() or not _has_text_edit_tag(instruction_st):
        return False
    rendered_source = _render_text_edit_source(instruction_st)
    if _compact_text(rendered_source) != _compact_text(source_text):
        return False
    rendered_target = _render_text_edit_target(instruction_st)
    target_text = params.get("target_text")
    if not isinstance(target_text, str) or not target_text.strip():
        return False
    return not _equivalent_target_text(target_text, rendered_target)


_ADJACENT_TEXT_EDIT_TAG_BOUNDARY = re.compile(
    r"</(?:sub|ins|del)>(?=<(?:sub|ins|del)\b)",
    flags=re.IGNORECASE,
)


def _repair_adjacent_text_edit_target_spacing(
    *,
    source_text: str,
    instruction_st: str,
    target_text: Any,
) -> str:
    """Encode an omitted inter-sentence space between adjacent edit tags.

    This may only add ``<ins> </ins>`` at an existing adjacent-tag boundary,
    and only when the result renders the exact authored source and target. It
    never chooses words, punctuation, edit spans, or target values.
    """

    if not isinstance(target_text, str) or not target_text:
        return instruction_st
    if _render_text_edit_source(instruction_st) != source_text:
        return instruction_st
    rendered_target = _render_text_edit_target(instruction_st)
    if rendered_target == target_text:
        return instruction_st
    if not _sentence_boundary_spacing_equivalent(rendered_target, target_text):
        return instruction_st

    boundaries = list(_ADJACENT_TEXT_EDIT_TAG_BOUNDARY.finditer(instruction_st))
    if not boundaries or len(boundaries) > 8:
        return instruction_st
    for mask in range(1, 1 << len(boundaries)):
        candidate = instruction_st
        for index in range(len(boundaries) - 1, -1, -1):
            if not mask & (1 << index):
                continue
            offset = boundaries[index].end()
            candidate = candidate[:offset] + "<ins> </ins>" + candidate[offset:]
        if (
            _render_text_edit_source(candidate) == source_text
            and _render_text_edit_target(candidate) == target_text
        ):
            return candidate
    return instruction_st


def _sentence_boundary_spacing_equivalent(left: str, right: str) -> bool:
    def canonical(value: str) -> str:
        compact = re.sub(r"\s+", " ", unescape(value)).strip()
        return re.sub(
            r"([.!?。！？][\"'”’]?)\s+(?=\S)",
            r"\1",
            compact,
        )

    return canonical(left) == canonical(right)


def _text_edit_instruction_renders_exactly(
    *,
    instruction_st: str,
    source_text: str,
    target_text: str,
) -> bool:
    return _render_text_edit_source(instruction_st) == source_text and _render_text_edit_target(instruction_st) == target_text


def _canonicalize_whitespace_equivalent_text_edit_instruction(
    *,
    source_text: str,
    instruction_st: str,
    target_text: Any,
) -> str:
    """Compile an exact tag script when the supplied script differs only in whitespace."""

    if not isinstance(target_text, str) or not target_text.strip():
        return instruction_st
    rendered_source = _render_text_edit_source(instruction_st)
    rendered_target = _render_text_edit_target(instruction_st)
    if rendered_source == source_text and rendered_target == target_text:
        return instruction_st
    if _compact_text(rendered_source) != _compact_text(source_text):
        return instruction_st
    if _compact_text(rendered_target) != _compact_text(target_text):
        return instruction_st
    candidate = _coarse_text_edit_instruction(source_text, target_text)
    if not _text_edit_instruction_renders_exactly(
        instruction_st=candidate,
        source_text=source_text,
        target_text=target_text,
    ):
        return instruction_st
    return candidate


def _should_repair_broad_text_edit_instruction(source_text: str, params: dict[str, Any]) -> bool:
    instruction_st = params.get("instruction_st")
    if not isinstance(instruction_st, str):
        return False
    target_text = _text_edit_target_text(source_text, params)
    if not isinstance(target_text, str) or not target_text.strip():
        return False
    rendered_source = _render_text_edit_source(instruction_st)
    rendered_target = _render_text_edit_target(instruction_st)
    if _compact_text(rendered_source) != _compact_text(source_text):
        return False
    if _compact_text(rendered_target) != _compact_text(target_text):
        return False
    tag_matches = list(_text_edit_tag_matches(instruction_st))
    if len(tag_matches) == 1 and tag_matches[0].group("sub_attrs") is not None:
        source_chunk = _strip_instruction_tags(tag_matches[0].group("sub_body") or "")
        target_chunk = _sub_target_from_attrs(tag_matches[0].group("sub_attrs") or "")
        if _compact_text(source_chunk) == _compact_text(source_text):
            return True
        return target_chunk is not None and _substitution_wraps_unchanged_boundary(source_chunk, target_chunk)
    for match in tag_matches:
        if match.group("sub_attrs") is None:
            continue
        source_chunk = _strip_instruction_tags(match.group("sub_body") or "")
        target_chunk = _sub_target_from_attrs(match.group("sub_attrs") or "")
        if target_chunk is not None and _substitution_wraps_unchanged_boundary(source_chunk, target_chunk):
            return True
    return False


def _can_build_target_text_edit(source_text: str, params: dict[str, Any]) -> bool:
    return isinstance(source_text, str) and bool(source_text.strip()) and _text_edit_target_text(source_text, params) is not None


def _text_edit_target_text(source_text: str, params: dict[str, Any]) -> str | None:
    target_text = params.get("target_text")
    if isinstance(target_text, str) and target_text.strip():
        if _has_text_edit_tag(target_text):
            rendered_target = _render_text_edit_target(target_text)
            rendered_source = _render_text_edit_source(target_text)
            if rendered_target.strip() and (
                not rendered_source.strip()
                or _compact_text(rendered_source) == _compact_text(source_text)
                or _compact_text(rendered_target) != _compact_text(target_text)
            ):
                return rendered_target
        return target_text
    instruction_st = params.get("instruction_st")
    if isinstance(instruction_st, str) and instruction_st.strip() and _has_text_edit_tag(instruction_st):
        rendered = _render_text_edit_target(instruction_st)
        if rendered.strip() and _compact_text(rendered) != _compact_text(source_text):
            return rendered
    return None


def _full_text_substitution_instruction(source_text: str, target_text: str) -> str:
    return f'<sub targ="{escape(target_text, quote=True)}">{escape(source_text, quote=False)}</sub>'


def _coarse_text_edit_instruction(source_text: str, target_text: str) -> str:
    """Render a coarse local edit script with few valid text-edit tags."""

    if source_text == target_text:
        return _full_text_substitution_instruction(source_text, target_text)

    source_units = _text_edit_diff_units(source_text)
    target_units = _text_edit_diff_units(target_text)
    matcher = difflib.SequenceMatcher(a=source_units, b=target_units, autojunk=False)
    parts: list[str] = []
    for op, i1, i2, j1, j2 in _coalesced_text_edit_opcodes(matcher.get_opcodes(), source_units, target_units):
        source_chunk = "".join(source_units[i1:i2])
        target_chunk = "".join(target_units[j1:j2])
        if op == "equal":
            parts.append(escape(source_chunk, quote=False))
        elif op == "delete":
            parts.append(f"<del>{escape(source_chunk, quote=False)}</del>")
        elif op == "insert":
            parts.append(f"<ins>{escape(target_chunk, quote=False)}</ins>")
        elif source_chunk and target_chunk:
            parts.append(
                f'<sub targ="{escape(target_chunk, quote=True)}">'
                f"{escape(source_chunk, quote=False)}</sub>"
            )

    instruction = "".join(parts)
    if (
        not _has_text_edit_tag(instruction)
        or _compact_text(_render_text_edit_source(instruction)) != _compact_text(source_text)
        or _compact_text(_render_text_edit_target(instruction)) != _compact_text(target_text)
    ):
        return _full_text_substitution_instruction(source_text, target_text)
    return instruction


def _coalesced_text_edit_opcodes(
    opcodes: list[tuple[str, int, int, int, int]],
    source_units: list[str],
    target_units: list[str],
) -> list[tuple[str, int, int, int, int]]:
    result: list[tuple[str, int, int, int, int]] = []
    index = 0
    while index < len(opcodes):
        op, i1, i2, j1, j2 = opcodes[index]
        if op == "equal":
            result.append((op, i1, i2, j1, j2))
            index += 1
            continue

        group_i1, group_i2 = i1, i2
        group_j1, group_j2 = j1, j2
        index += 1
        while index + 1 < len(opcodes):
            bridge = opcodes[index]
            next_opcode = opcodes[index + 1]
            if bridge[0] != "equal" or next_opcode[0] == "equal":
                break
            source_bridge = "".join(source_units[bridge[1] : bridge[2]])
            target_bridge = "".join(target_units[bridge[3] : bridge[4]])
            if source_bridge != target_bridge or not _is_text_edit_bridge(source_bridge):
                break
            group_i2 = next_opcode[2]
            group_j2 = next_opcode[4]
            index += 2

        source_empty = group_i1 == group_i2
        target_empty = group_j1 == group_j2
        if source_empty and not target_empty:
            group_op = "insert"
        elif target_empty and not source_empty:
            group_op = "delete"
        else:
            group_op = "replace"
        result.append((group_op, group_i1, group_i2, group_j1, group_j2))
    return result


def _is_text_edit_bridge(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if re.search(r"[,.;:!?，。？！；：、]", stripped):
        return False
    if re.search(r"[A-Za-z0-9]", stripped):
        return False
    return stripped in {"的", "地", "得", "了"}


def _sub_target_from_attrs(attrs: str) -> str | None:
    target_match = re.search(r'targ="([^"]*)"', attrs) or re.search(r"targ='([^']*)'", attrs)
    return unescape(target_match.group(1)) if target_match else None


def _sanitize_text_edit_instruction_for_service(instruction_st: str) -> str:
    def repl(match: re.Match[str]) -> str:
        target = _sub_target_from_attrs(match.group("sub_attrs") or "")
        if target is None:
            return match.group(0)
        return f'<sub targ="{escape(target, quote=True)}">{match.group("sub_body")}</sub>'

    return _TEXT_EDIT_TAG_RE.sub(lambda match: repl(match) if match.group("sub_attrs") is not None else match.group(0), instruction_st)


_WORDISH_TOKEN_RE = re.compile(r"\s+|[^\W_]+(?:['’][^\W_]+)*|[^\s]", flags=re.UNICODE)
_TEXT_EDIT_TAG_RE = re.compile(
    r"<sub\s+(?P<sub_attrs>[^>]*)>(?P<sub_body>.*?)</sub>"
    r"|<ins\b[^>]*>(?P<ins_body>.*?)</ins>"
    r"|<del\b[^>]*>(?P<del_body>.*?)</del>",
    flags=re.DOTALL,
)


def _text_edit_diff_units(value: str) -> list[str]:
    if _contains_han(value):
        return list(value)
    return [match.group(0) for match in _WORDISH_TOKEN_RE.finditer(value)] or [value]


def _contains_han(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", value))


def _text_edit_tag_matches(instruction_st: str) -> list[re.Match[str]]:
    return list(_TEXT_EDIT_TAG_RE.finditer(instruction_st))


def _substitution_wraps_unchanged_boundary(source_chunk: str, target_chunk: str) -> bool:
    source_units = [unit for unit in _text_edit_diff_units(source_chunk) if unit.strip()]
    target_units = [unit for unit in _text_edit_diff_units(target_chunk) if unit.strip()]
    if not source_units or not target_units:
        return False
    return source_units[0] == target_units[0] or source_units[-1] == target_units[-1]


def _full_rate_instruction(source_text: str, rate: float) -> str:
    return f'<rate, factor={rate:g}>{escape(source_text, quote=False)}</rate>'


def _full_pitch_instruction(source_text: str, semitones: Any) -> str:
    semitone_value = _integer_pitch_semitones(semitones)
    return f'<pitch, semitones={semitone_value}>{escape(source_text, quote=False)}</pitch>'


def _integer_pitch_semitones(
    value: Any,
    *,
    default: int = 8,
    strict: bool = False,
) -> int:
    try:
        semitones = float(value)
    except (TypeError, ValueError):
        if strict:
            raise OperationBoundaryError(
                f"pitch semitones must be an integer in [-12, 12]; got {value!r}"
            ) from None
        semitones = float(default)
    if strict and (not semitones.is_integer() or semitones < -12 or semitones > 12):
        raise OperationBoundaryError(
            f"pitch semitones must be an integer in [-12, 12]; got {value!r}"
        )
    return max(-12, min(12, round(semitones)))


_PITCH_TAG_SEMITONES_RE = re.compile(r"(<pitch,\s*semitones=)([+-]?[0-9]+(?:\.[0-9]+)?)(\s*>)")
_UNSUPPORTED_PITCH_WRAPPER_RE = re.compile(r"<pitch\b(?!,\s*semitones=)[^>]*>", re.IGNORECASE)
_UNSUPPORTED_RATE_WRAPPER_RE = re.compile(r"<rate\b(?!,\s*factor=)[^>]*>", re.IGNORECASE)
_UNSUPPORTED_PITCH_WRAPS_VALID_PITCH_RE = re.compile(
    r"<pitch\b(?!,\s*semitones=)[^>]*>\s*"
    r"(?P<body><pitch,\s*semitones=[+-]?[0-9]+(?:\.[0-9]+)?\s*>.*?</pitch>)"
    r"\s*</pitch>",
    re.IGNORECASE | re.DOTALL,
)
_UNSUPPORTED_RATE_WRAPS_VALID_RATE_RE = re.compile(
    r"<rate\b(?!,\s*factor=)[^>]*>\s*"
    r"(?P<body><rate,\s*factor=[0-9]+(?:\.[0-9]+)?\s*>.*?</rate>)"
    r"\s*</rate>",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_pitch_instruction_semitones(instruction_st: str) -> str:
    """Validate pitch tags without moving their span or changing their value."""

    _validate_pitch_rate_label_values(instruction_st)
    return instruction_st


def _strip_unsupported_pitch_rate_wrappers(instruction_st: str) -> str:
    normalized = instruction_st
    for _ in range(4):
        updated = _UNSUPPORTED_PITCH_WRAPS_VALID_PITCH_RE.sub(r"\g<body>", normalized)
        updated = _UNSUPPORTED_RATE_WRAPS_VALID_RATE_RE.sub(r"\g<body>", updated)
        if updated == normalized:
            return normalized
        normalized = updated
    return normalized


def _move_pitch_terminal_question_marks(instruction_st: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group("body").rstrip()
        suffix = match.group("body")[len(body) :] + match.group("marks")
        return f"{match.group('open')}{body}{match.group('close')}{suffix}"

    return re.sub(
        r"(?P<open><pitch,\s*semitones=[+-]?[0-9]+\s*>)(?P<body>[^<]*?)(?P<marks>[?？]+)(?P<close></pitch>)",
        repl,
        instruction_st,
    )


def _normalize_pitch_rate_params(
    operation: str,
    params: dict[str, Any],
    *,
    source_text: str | None = None,
    instruction: str | None = None,
) -> dict[str, Any]:
    if operation != "pitch_rate_edit":
        return dict(params)
    normalized = dict(params)
    if "semitones" in normalized:
        normalized["semitones"] = _integer_pitch_semitones(
            normalized.get("semitones"), strict=True
        )
    instruction_st = normalized.get("instruction_st")
    if isinstance(instruction_st, str) and instruction_st.strip():
        normalized["instruction_st"] = _normalize_pitch_instruction_semitones(
            instruction_st
        )
    elif "semitones" in normalized and isinstance(source_text, str) and source_text.strip():
        normalized["instruction_st"] = _full_pitch_instruction(source_text, normalized["semitones"])
        normalized.setdefault("smooth_policy", "never")
    elif any(key in normalized for key in ("rate", "factor", "speed")) and isinstance(source_text, str) and source_text.strip():
        rate = _rate_factor_from_params(normalized)
        normalized["instruction_st"] = _full_rate_instruction(source_text, rate)
    return normalized


def _pitch_request_should_override_rate(instruction: str | None, params: dict[str, Any]) -> bool:
    text = str(instruction or "").lower()
    if not text:
        return False
    has_rate_hint = any(params.get(key) is not None for key in ("rate", "factor", "speed"))
    if not has_rate_hint:
        return False
    asks_pitch = any(token in text for token in ["音高", "pitch", "semitone", "升调", "降调", "deeper", "higher"])
    forbids_rate = any(
        token in text
        for token in [
            "不要改变语速",
            "不要改语速",
            "不改变语速",
            "keep the same speed",
            "preserve speed",
            "preserve rate",
            "do not change speed",
            "do not change rate",
            "without changing speed",
            "without changing rate",
        ]
    )
    return asks_pitch and forbids_rate


def _pitch_rate_operation_is_affective_style(instruction: str, params: dict[str, Any]) -> bool:
    evidence = " ".join(
        str(value)
        for value in [
            instruction,
            params.get("instruction"),
            params.get("edit_instruction"),
            params.get("instruction_st"),
            params.get("style"),
            params.get("emotion_style"),
        ]
        if value
    ).lower()
    has_concrete_pitch_or_rate = bool(
        re.search(r"<(?:pitch|rate|speed),", evidence)
        or any(params.get(key) is not None for key in ("semitones", "rate", "factor", "speed"))
        or re.search(r"[+-]?[0-9]+(?:\.[0-9]+)?\s*(?:semitones?|x|倍)", evidence)
    )
    if has_concrete_pitch_or_rate:
        return False
    affective_style_tokens = [
        "theatrical",
        "dramatic",
        "expressive",
        "emotional style",
        "acting style",
        "lively",
        "energetic",
        "vibrant",
        "cheerful",
        "excited",
        "joyful",
        "happy",
        "animated",
        "spirited",
        "enthusiastic",
        "活泼",
        "生动",
        "有活力",
        "开心",
        "高兴",
        "兴奋",
        "快乐",
    ]
    has_style = any(token in evidence for token in affective_style_tokens)
    if not has_style:
        return False
    has_plain_pitch_or_rate_request = bool(
        any(
            token in evidence
            for token in [
                "semitone",
                "rate",
                "speed",
                "faster",
                "slower",
                "tempo",
                "音高",
                "语速",
                "加速",
                "减速",
            ]
        )
    )
    return not has_plain_pitch_or_rate_request or any(
        token in evidence for token in ["pitch contour", "pitch contours", "pitch variation", "pitch variations"]
    )


def _rate_factor_from_params(params: dict[str, Any]) -> float:
    for key in ("rate", "factor", "speed"):
        value = params.get(key)
        if value is None:
            continue
        try:
            factor = float(value)
        except (TypeError, ValueError):
            raise OperationBoundaryError(
                f"pitch_rate_edit {key} must be numeric; got {value!r}"
            ) from None
        if factor < 0.3 or factor > 3.0:
            raise OperationBoundaryError("pitch_rate_edit rate factor must be in [0.3, 3.0]")
        return factor
    raise OperationBoundaryError(
        "pitch_rate_edit requires an explicit rate/factor/speed value for a rate edit"
    )


_INSTRUCTION_TAG_NAME_RE = re.compile(r"</?\s*([A-Za-z_][A-Za-z0-9_:-]*)(?=[\s,>/])")


def _instruction_tag_names(value: str) -> list[str]:
    return [match.group(1) for match in _INSTRUCTION_TAG_NAME_RE.finditer(value)]


def _reject_unsupported_instruction_tags(operation: str, instruction_st: str, *, field: str = "instruction_st") -> None:
    contract = edit_data_pipeline_label_contract(operation)
    allowed = set(contract.get("allowed_tags") or []) if contract else set()
    if not allowed:
        if _instruction_tag_names(instruction_st):
            raise OperationBoundaryError(
                f"edit_data_pipeline.{operation} does not accept label tags in {field}; "
                f"{'; '.join(edit_data_pipeline_label_requirements(operation))}"
            )
        return
    tag_names = set(_instruction_tag_names(instruction_st))
    unsupported = sorted(tag_names - allowed)
    if unsupported:
        supported = ", ".join(str(item) for item in contract.get("supported_labels") or sorted(allowed))
        raise OperationBoundaryError(
            f"edit_data_pipeline.{operation} {field} has unsupported tag(s): {', '.join(unsupported)}. "
            f"Supported label forms: {supported}"
        )


def _validate_text_edit_label_shapes(instruction_st: str) -> None:
    for tag in re.findall(r"<sub\b[^>]*>", instruction_st):
        if not re.search(r"\btarg=(['\"]).*?\1", tag):
            raise OperationBoundaryError('text_edit <sub> tags must use targ="new phrase"')
    for tag_name in ("sub", "ins", "del"):
        opening_count = len(re.findall(rf"<{tag_name}\b[^>]*>", instruction_st))
        closing_count = len(re.findall(rf"</{tag_name}>", instruction_st))
        if opening_count != closing_count:
            raise OperationBoundaryError(f"text_edit <{tag_name}> tags must have matching closing tags")


def _validate_pitch_rate_label_values(instruction_st: str) -> None:
    for tag in re.findall(r"<pitch\b[^>]*>", instruction_st):
        match = re.fullmatch(r"<pitch,\s*semitones=([+-]?[0-9]+)\s*>", tag)
        if not match:
            raise OperationBoundaryError("pitch_rate_edit pitch tag must be exactly <pitch, semitones=N>")
        semitones = int(match.group(1))
        if semitones < -12 or semitones > 12:
            raise OperationBoundaryError("pitch_rate_edit pitch semitones must be in [-12, 12]")
    for tag in re.findall(r"<rate\b[^>]*>", instruction_st):
        match = re.fullmatch(r"<rate,\s*factor=([0-9]+(?:\.[0-9]+)?)\s*>", tag)
        if not match:
            raise OperationBoundaryError("pitch_rate_edit rate tag must be exactly <rate, factor=F>")
        factor = float(match.group(1))
        if factor < 0.3 or factor > 3.0:
            raise OperationBoundaryError("pitch_rate_edit rate factor must be in [0.3, 3.0]")


def _validate_emotion_style_label_shapes(instruction_st: str) -> None:
    try:
        validate_explicit_emotion_choices(instruction_st)
    except ValueError as exc:
        raise OperationBoundaryError(str(exc)) from exc
    for tag in re.findall(r"<desc\b[^>]*>", instruction_st):
        if not re.fullmatch(r"<desc(?:\s*,)?\s+[^>]*text=(['\"]).+?\1[^>]*>", tag, flags=re.DOTALL):
            raise OperationBoundaryError('emotion_style_edit desc tags must use text="style description"')
    desc_open_count = len(re.findall(r"<desc\b[^>]*>", instruction_st))
    desc_close_count = len(re.findall(r"</desc>", instruction_st))
    if desc_open_count != desc_close_count:
        raise OperationBoundaryError("emotion_style_edit <desc> tags must have matching </desc> closing tags")
    emo_open_count = len(re.findall(r"<emo\b[^>]*>", instruction_st))
    emo_close_count = len(re.findall(r"</emo>", instruction_st))
    if emo_open_count != emo_close_count:
        raise OperationBoundaryError("emotion_style_edit <emo> tags must have matching </emo> closing tags")


def _canonicalize_explicit_emotion_instruction_for_service(instruction_st: str) -> str:
    """Encode explicit legacy emotion tags without inferring an emotion or span."""

    if re.search(
        r"<(?:none(?:,\s*level=[1-3])?|emo\b[^>]*\btype=(['\"])none\1)",
        instruction_st,
        flags=re.IGNORECASE,
    ):
        raise OperationBoundaryError(
            "emotion_style_edit does not accept emotion 'none'; Planner must provide an explicit supported style"
        )

    desc_pattern = re.compile(
        r"<desc(?:\s*,)?\s+[^>]*text=(['\"])(?P<desc>.*?)\1[^>]*>"
        r"(?P<body>.*?)</desc>",
        flags=re.DOTALL,
    )

    def encode_desc(match: re.Match[str]) -> str:
        description = unescape(match.group("desc") or "")
        body = match.group("body") or ""
        return (
            f'<emo desc="{escape(description, quote=True)}">'
            f"{escape(unescape(body), quote=False)}</emo>"
        )

    normalized = desc_pattern.sub(encode_desc, instruction_st)
    emotions = "|".join(sorted(re.escape(item) for item in ALLOWED_EMOTIONS - {"none"}))
    legacy_level = re.compile(
        rf"<(?P<tag>{emotions}),\s*level=(?P<level>[1-3])>"
        rf"(?P<body>.*?)</(?P=tag)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized = legacy_level.sub(
        lambda match: _canonical_emotion_level_tag(
            match.group("tag").lower(),
            int(match.group("level")),
            match.group("body"),
        ),
        normalized,
    )
    unsupported_plain = re.search(
        rf"<(?P<tag>{emotions})>(?P<body>.*?)</(?P=tag)>",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if unsupported_plain:
        raise OperationBoundaryError(
            "legacy emotion tags require an explicit level; normalization does not choose emotion intensity"
        )
    return normalized


def _canonicalize_explicit_emotion_brackets_for_service(instruction_st: str) -> str:
    """Translate an explicit known emotion bracket tag without choosing intensity."""

    emotions = "|".join(
        sorted(re.escape(item) for item in ALLOWED_EMOTIONS - {"none"})
    )
    pattern = re.compile(
        rf"\[(?P<tag>{emotions})\](?P<body>.*?)\[/(?P=tag)\]",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        emotion = match.group("tag").lower()
        body = escape(unescape(match.group("body") or ""), quote=False)
        return f'<emo desc="{emotion}">{body}</emo>'

    return pattern.sub(replace, instruction_st)


def _normalize_emotion_instruction_for_service(instruction_st: str, *, instruction: str = "") -> str:
    normalized = _normalize_emotion_desc_instruction_for_service(instruction_st, instruction=instruction)
    normalized = _normalize_none_emotion_instruction_for_service(normalized)
    return _normalize_legacy_emotion_instruction_for_service(normalized)


def _project_emotion_instruction_to_full_source(
    instruction_st: str,
    *,
    source_text: str,
    instruction: str = "",
) -> str:
    """Repair local emotion tags so stripping tags still yields source_text.

    Planner output sometimes tags only a target phrase, or replaces the tagged
    phrase body with an English style word (for example "reluctant"). The edit
    service contract requires the full transcript with tags around the edited
    span, so recover the span from quoted user text or the tagged body.
    """

    if _compact_text(_strip_instruction_tags(instruction_st)) == _compact_text(source_text):
        return instruction_st

    match = re.search(
        r"(?P<open><(?:emo|desc)\b[^>]*>)(?P<body>.*?)(?P<close></(?:emo|desc)>)",
        instruction_st,
        flags=re.DOTALL,
    )
    if not match:
        return instruction_st

    candidates = _emotion_projection_span_candidates(instruction=instruction, tagged_body=match.group("body"))
    for candidate in candidates:
        span = _find_source_phrase_span(source_text, candidate, prefer_last=_prefers_last_mentioned_span(instruction))
        if span is None:
            continue
        start, end = span
        if end <= start:
            continue
        return f"{source_text[:start]}{match.group('open')}{source_text[start:end]}{match.group('close')}{source_text[end:]}"

    return instruction_st


def _emotion_projection_span_candidates(*, instruction: str, tagged_body: str) -> list[str]:
    candidates: list[str] = []
    quote_patterns = [
        r"[\"“]([^\"”]+)[\"”]",
        r"['‘]([^'’]+)['’]",
        r"「([^」]+)」",
        r"『([^』]+)』",
    ]
    for pattern in quote_patterns:
        for value in re.findall(pattern, instruction):
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

    stripped_body = _strip_instruction_tags(tagged_body).strip()
    if stripped_body:
        candidates.append(stripped_body)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = _compact_text(candidate)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        deduped.append(candidate)
    return deduped


def _prefers_last_mentioned_span(instruction: str) -> bool:
    lowered = instruction.lower()
    return any(term in lowered for term in ("final", "last", "ending", "end of", "最后", "末尾", "结尾"))


def _find_source_phrase_span(source_text: str, phrase: str, *, prefer_last: bool = False) -> tuple[int, int] | None:
    if not phrase:
        return None
    matches = [match.span() for match in re.finditer(re.escape(phrase), source_text)]
    if matches:
        return matches[-1] if prefer_last else matches[0]

    pattern = re.escape(phrase.strip())
    pattern = re.sub(r"(?:\\\s)+", r"\\s+", pattern)
    if pattern == re.escape(phrase.strip()):
        return None
    matches = [match.span() for match in re.finditer(pattern, source_text)]
    if matches:
        return matches[-1] if prefer_last else matches[0]
    return None


def _normalize_emotion_desc_instruction_for_service(instruction_st: str, *, instruction: str = "") -> str:
    normalized = _normalize_legacy_emotion_desc_instruction_for_service(instruction_st, instruction=instruction)
    return _normalize_canonical_emotion_desc_instruction_for_service(normalized, instruction=instruction)


def _normalize_legacy_emotion_desc_instruction_for_service(instruction_st: str, *, instruction: str = "") -> str:
    desc_pattern = re.compile(
        r"<desc(?:\s*,)?\s+[^>]*text=(['\"])(?P<desc>.*?)\1[^>]*>(?P<body>.*?)</desc>",
        flags=re.DOTALL,
    )
    if not desc_pattern.search(instruction_st):
        return instruction_st

    def repl(match: re.Match[str]) -> str:
        description = unescape(match.group("desc") or "")
        body = match.group("body") or ""
        emotion = _emotion_from_desc_style(f"{instruction} {description}")
        if emotion is None:
            return _canonical_emotion_desc_tag(description, body)
        tag, level = emotion
        return _canonical_emotion_level_tag(tag, level, body)

    return desc_pattern.sub(repl, instruction_st)


def _normalize_canonical_emotion_desc_instruction_for_service(instruction_st: str, *, instruction: str = "") -> str:
    desc_pattern = re.compile(
        r"<emo\b(?P<attrs>[^>]*\bdesc=(['\"])(?P<desc>.*?)\2[^>]*)>(?P<body>.*?)</emo>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not desc_pattern.search(instruction_st):
        return instruction_st

    def repl(match: re.Match[str]) -> str:
        description = unescape(match.group("desc") or "")
        body = match.group("body") or ""
        if _unsupported_emotion_desc(description):
            return match.group(0)
        emotion = _emotion_from_desc_style(f"{instruction} {description}")
        if emotion is not None:
            tag, level = emotion
            return _canonical_emotion_level_tag(tag, level, body)
        attrs = match.group("attrs") or ""
        type_ref = _attr_value(attrs, "type_ref") or _attr_value(attrs, "type") or "calm"
        if type_ref not in ALLOWED_EMOTIONS or type_ref == "none":
            type_ref = "calm"
        return _canonical_emotion_desc_tag(description, body, type_ref=type_ref)

    return desc_pattern.sub(repl, instruction_st)


def _normalize_none_emotion_instruction_for_service(instruction_st: str) -> str:
    normalized = re.sub(
        r"<none,\s*level=[1-3]>(?P<body>.*?)</none>",
        lambda match: _canonical_emotion_level_tag("calm", 1, match.group("body")),
        instruction_st,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(
        r"<emo\b(?=[^>]*\btype=(['\"])none\1)[^>]*>(?P<body>.*?)</emo>",
        lambda match: _canonical_emotion_level_tag("calm", 1, match.group("body")),
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _normalize_legacy_emotion_instruction_for_service(instruction_st: str) -> str:
    emotions = "|".join(sorted(re.escape(item) for item in ALLOWED_EMOTIONS))
    legacy_level = re.compile(
        rf"<(?P<tag>{emotions}),\s*level=(?P<level>[1-3])>(?P<body>.*?)</(?P=tag)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    normalized = legacy_level.sub(
        lambda match: _canonical_emotion_level_tag(
            "calm" if match.group("tag").lower() == "none" else match.group("tag").lower(),
            1 if match.group("tag").lower() == "none" else int(match.group("level")),
            match.group("body"),
        ),
        instruction_st,
    )
    legacy_plain = re.compile(
        rf"<(?P<tag>{emotions})>(?P<body>.*?)</(?P=tag)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return legacy_plain.sub(
        lambda match: _canonical_emotion_level_tag(
            "calm" if match.group("tag").lower() == "none" else match.group("tag").lower(),
            1 if match.group("tag").lower() == "none" else 2,
            match.group("body"),
        ),
        normalized,
    )


def _canonical_emotion_level_tag(emotion: str, level: int, body: str) -> str:
    return f'<emo type="{escape(str(emotion), quote=True)}" level="{int(level)}">{escape(unescape(body), quote=False)}</emo>'


def _canonical_emotion_desc_tag(description: str, body: str, *, type_ref: str = "calm") -> str:
    return (
        f'<emo desc="{escape(str(description), quote=True)}" type_ref="{escape(str(type_ref), quote=True)}">'
        f"{escape(unescape(body), quote=False)}</emo>"
    )


def _emotion_from_desc_style(text: str) -> tuple[str, int] | None:
    lowered = text.lower()
    if _contains_any(
        lowered,
        [
            "theatrical",
            "dramatic",
            "expressive",
            "acting style",
            "performed style",
            "performance style",
            "animated",
            "exaggerated",
            "over-the-top",
            "over the top",
            "stage-like",
            "stage like",
            "戏剧",
            "夸张",
            "表演",
        ],
    ):
        return "surprised", 3
    if _contains_any(lowered, ["more emotional", "emotional", "emotionally expressive"]):
        return "surprised", 2
    if _contains_any(
        lowered,
        [
            "lively",
            "energetic",
            "vibrant",
            "cheerful",
            "excited",
            "joyful",
            "spirited",
            "enthusiastic",
            "活泼",
            "生动",
            "有活力",
            "兴奋",
        ],
    ):
        return "happy", 3
    if _contains_any(
        lowered,
        [
            "casual",
            "everyday",
            "conversational",
            "relaxed",
            "normal speaking voice",
            "normal voice",
            "neutral voice",
            "plain speaking voice",
            "plainspoken",
            "plain spoken",
            "普通",
            "日常",
            "随意",
            "口语",
        ],
    ):
        return "calm", 1
    if _contains_any(lowered, ["angry", "furious", "aggressive", "hostile", "irritated", "生气", "愤怒"]):
        return "angry", 3
    if _contains_any(
        lowered,
        [
            "profound disappointment",
            "disappointment",
            "disappointed",
            "reluctant",
            "sad",
            "sorrowful",
            "grieving",
            "tearful",
            "悲伤",
            "难过",
            "失望",
            "不情愿",
        ],
    ):
        return "sad", 3 if _contains_any(lowered, ["profound", "extreme", "deep", "强烈"]) else 2
    if _contains_any(
        lowered,
        [
            "panicked",
            "panic",
            "terrified",
            "tremor",
            "tremors",
            "breathiness",
            "breathy",
            "afraid",
            "fearful",
            "scared",
            "nervous",
            "anxious",
            "害怕",
            "恐惧",
            "紧张",
        ],
    ):
        if _contains_any(lowered, ["panicked", "panic", "terrified", "tremor", "breath"]):
            return "afraid", 3
        return "afraid", 2
    if _contains_any(lowered, ["disgusted", "disgust", "repulsed", "厌恶", "反感"]):
        return "disgusted", 2
    if _contains_any(
        lowered,
        [
            "hushed",
            "reflective",
            "introspective",
            "subdued",
            "internal monologue",
            "inner monologue",
            "internal thought",
            "self-talk",
            "self talk",
            "low-voiced",
            "low voiced",
            "contemplative",
            "melancholic",
            "内心独白",
        ],
    ):
        return "melancholic", 2
    if _contains_any(lowered, ["calm", "soft", "gentle", "soothing", "平静", "舒缓"]):
        return "calm", 2
    if _contains_any(lowered, ["audiobook", "narrator", "narration style", "有声书", "旁白"]):
        return "calm", 2
    if _contains_any(
        lowered,
        [
            "natural",
            "non-mechanical",
            "non mechanical",
            "less mechanical",
            "mechanical feel",
            "robotic",
            "less robotic",
            "human voice",
            "more human",
            "自然",
            "机械感",
            "机器人感",
        ],
    ):
        return "calm", 1
    return None


def _validate_audio_assembly_label_contract(params: dict[str, Any]) -> None:
    instruction_st = params.get("instruction_st")
    if isinstance(instruction_st, str) and instruction_st.strip():
        raise OperationBoundaryError(
            "audio_assembly does not accept instruction_st; use edit_tagged_text with <target>...</target> spans"
        )
    edit_tagged_text = params.get("edit_tagged_text")
    if not isinstance(edit_tagged_text, str) or not edit_tagged_text.strip():
        return
    _reject_unsupported_instruction_tags("audio_assembly", edit_tagged_text, field="edit_tagged_text")
    target_open_count = len(re.findall(r"<target\b[^>]*>", edit_tagged_text))
    target_close_count = len(re.findall(r"</target>", edit_tagged_text))
    if target_open_count != target_close_count:
        raise OperationBoundaryError("audio_assembly <target> tags must have matching </target> closing tags")
    if target_open_count == 0:
        raise OperationBoundaryError("audio_assembly edit_tagged_text requires at least one <target>...</target> span")


def _validate_untagged_edit_pipeline_params(operation: str, params: dict[str, Any]) -> None:
    if operation == "audio_assembly":
        _validate_audio_assembly_label_contract(params)
        return
    if operation == "volume_normalize":
        _canonicalize_volume_normalize_method(params.get("method"), strict=True)
    if operation == "speech_rate":
        rate = params.get("rate")
        if rate is None:
            raise OperationBoundaryError("edit_data_pipeline.speech_rate requires explicit rate")
        try:
            rate_value = float(rate)
        except (TypeError, ValueError):
            raise OperationBoundaryError(
                f"edit_data_pipeline.speech_rate rate must be numeric; got {rate!r}"
            ) from None
        if rate_value <= 0:
            raise OperationBoundaryError("edit_data_pipeline.speech_rate rate must be positive")
    if operation == "pitch_shift":
        if params.get("semitones") is None:
            raise OperationBoundaryError("edit_data_pipeline.pitch_shift requires explicit semitones")
        _integer_pitch_semitones(params["semitones"], strict=True)
    instruction_st = params.get("instruction_st")
    if isinstance(instruction_st, str) and instruction_st.strip():
        raise OperationBoundaryError(
            f"edit_data_pipeline.{operation} does not accept instruction_st labels; "
            f"{'; '.join(edit_data_pipeline_label_requirements(operation))}"
        )
    for key, value in params.items():
        if not isinstance(value, str) or "<" not in value or ">" not in value:
            continue
        tags = _instruction_tag_names(value)
        if tags:
            raise OperationBoundaryError(
                f"edit_data_pipeline.{operation} parameter {key!r} contains unsupported label tag(s): "
                f"{', '.join(sorted(set(tags)))}. {'; '.join(edit_data_pipeline_label_requirements(operation))}"
            )


def _validate_tagged_instruction(
    *,
    operation: str,
    source_text: str,
    instruction_st: str,
    target_text: Any = None,
    params: dict[str, Any],
) -> None:
    if not instruction_st.strip():
        raise OperationBoundaryError(f"edit_data_pipeline.{operation} requires non-empty instruction_st")
    _reject_unsupported_instruction_tags(operation, instruction_st)

    stripped = _render_text_edit_source(instruction_st) if operation == "text_edit" else _strip_instruction_tags(instruction_st)
    if not _equivalent_transcript_text(stripped, source_text):
        raise OperationBoundaryError(
            f"edit_data_pipeline.{operation} instruction_st stripped text must match source_text"
        )

    if operation == "text_edit":
        _validate_text_edit_label_shapes(instruction_st)
        if not _has_text_edit_tag(instruction_st):
            raise OperationBoundaryError(
                "text_edit instruction_st requires at least one <sub targ=\"...\">...</sub>, <ins>...</ins>, or <del>...</del> tag"
            )
        rendered = _render_text_edit_target(instruction_st)
        if isinstance(target_text, str) and target_text and not _equivalent_target_text(rendered, target_text):
            raise OperationBoundaryError("text_edit target_text does not match rendered instruction_st target")
    elif operation == "pitch_rate_edit":
        has_pitch = "<pitch" in instruction_st
        has_rate = "<rate" in instruction_st
        if not has_pitch and not has_rate:
            raise OperationBoundaryError("pitch_rate_edit instruction_st requires <pitch,...> or <rate,...>")
        _validate_pitch_rate_label_values(instruction_st)
        if has_pitch and not has_rate:
            for key in ("rate", "factor", "speed"):
                if key in params:
                    raise OperationBoundaryError("pitch-only edit must not include rate/factor/speed parameters")
            if str(params.get("smooth_policy") or "never") != "never":
                raise OperationBoundaryError("pitch-only edit must use smooth_policy='never' to preserve duration")
    elif operation == "emotion_style_edit":
        _validate_emotion_style_label_shapes(instruction_st)
        if _valid_canonical_emotion_instruction(instruction_st):
            return
        matches = re.findall(r"<([a-z_]+),\s*level=([1-3])>", instruction_st)
        if matches:
            for emotion, _level in matches:
                if emotion not in ALLOWED_EMOTIONS:
                    raise OperationBoundaryError(
                        f"emotion_style_edit emotion {emotion!r} is unsupported; allowed={sorted(ALLOWED_EMOTIONS)}"
                    )
                if f"</{emotion}>" not in instruction_st:
                    raise OperationBoundaryError(f"emotion_style_edit missing closing tag </{emotion}>")
            return
        if _valid_emotion_desc_instruction(instruction_st):
            return
        raise OperationBoundaryError(
            "emotion_style_edit instruction_st requires <emo type=\"emotion\" level=\"1|2|3\"> or <emo desc=\"...\">"
        )
    elif operation == "pause_edit":
        if not re.search(r"<pause\s+[^>]*/>", instruction_st):
            raise OperationBoundaryError("pause_edit instruction_st requires a self-closing <pause .../> tag")
        if not re.search(r"<pause\s+[^>]*act=[\"']ins[\"'][^>]*level=[\"'][1-3][\"'][^>]*/>", instruction_st):
            raise OperationBoundaryError("pause_edit supports insertion tags only: <pause act=\"ins\" level=\"1|2|3\"/>")
        match = re.search(r"<pause\s+[^>]*/>", instruction_st)
        assert match is not None
        if not instruction_st[: match.start()].strip() or not instruction_st[
            match.end() :
        ].strip():
            raise OperationBoundaryError(
                "pause_edit insertion tag requires non-empty left and right context"
            )


def _valid_emotion_desc_instruction(instruction_st: str) -> bool:
    desc_matches = re.findall(r"<desc(?:\s*,)?\s+[^>]*text=(['\"])(.*?)\1[^>]*>", instruction_st, flags=re.DOTALL)
    if not desc_matches or "</desc>" not in instruction_st:
        return False
    return not any(_unsupported_emotion_desc(desc) for _quote, desc in desc_matches)


def _valid_canonical_emotion_instruction(instruction_st: str) -> bool:
    return bool(_canonical_emotion_level_matches(instruction_st) or _valid_canonical_emotion_desc_instruction(instruction_st))


def _canonical_emotion_level_matches(instruction_st: str) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    for tag in re.finditer(r"<emo\b(?P<attrs>[^>]*)>", instruction_st, flags=re.IGNORECASE | re.DOTALL):
        attrs = tag.group("attrs") or ""
        if re.search(r"\bdesc=(['\"]).+?\1", attrs, flags=re.DOTALL):
            continue
        type_match = re.search(r"\btype=(['\"])(?P<type>[^'\"]+)\1", attrs, flags=re.IGNORECASE)
        level_match = re.search(r"\blevel=(['\"])(?P<level>[1-3])\1", attrs, flags=re.IGNORECASE)
        alpha_match = re.search(r"\balpha=(['\"])(?P<alpha>0(?:\.\d+)?|1(?:\.0+)?)\1", attrs, flags=re.IGNORECASE)
        if not type_match:
            continue
        emotion = type_match.group("type").lower()
        if emotion not in ALLOWED_EMOTIONS or emotion == "none":
            raise OperationBoundaryError(
                f"emotion_style_edit emotion {emotion!r} is unsupported; allowed={sorted(ALLOWED_EMOTIONS - {'none'})}"
            )
        if not level_match and not alpha_match:
            raise OperationBoundaryError('emotion_style_edit <emo type="..."> tags must include level="1|2|3" or alpha="0..1"')
        matches.append((emotion, level_match.group("level") if level_match else alpha_match.group("alpha")))
    return matches


def _valid_canonical_emotion_desc_instruction(instruction_st: str) -> bool:
    desc_matches = re.findall(
        r"<emo\b(?=[^>]*\bdesc=(['\"])(.*?)\1)[^>]*>",
        instruction_st,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not desc_matches or "</emo>" not in instruction_st:
        return False
    return not any(_unsupported_emotion_desc(desc) for _quote, desc in desc_matches)


def _unsupported_emotion_desc(description: str) -> bool:
    lowered = str(description or "").lower()
    return bool(
        re.search(r"\b(?:rap|rapping|rapper|rapped)\b", lowered)
        or re.search(r"\b(?:accent|accented|dialect|dialectal|singing|sung)\b", lowered)
    )


def _attr_value(attrs: str, name: str) -> str | None:
    quoted = re.search(rf'{re.escape(name)}\s*=\s*(["\'])(.*?)\1', attrs, flags=re.DOTALL)
    if quoted:
        return unescape(quoted.group(2))
    match = re.search(rf"{re.escape(name)}\s*=\s*([^,\"'\\s>]+)", attrs)
    return unescape(match.group(1)) if match else None


def _strip_instruction_tags(value: str) -> str:
    return render_tagged_source(value)


def _has_text_edit_tag(value: str) -> bool:
    return bool(
        re.search(r"<sub\s+[^>]*targ=", value)
        or re.search(r"<ins\b[^>]*>.*?</ins>", value, flags=re.DOTALL)
        or re.search(r"<del\b[^>]*>.*?</del>", value, flags=re.DOTALL)
    )


def _render_text_edit_source(value: str) -> str:
    return render_tagged_source(value, lexical=True)


def _render_text_edit_target(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        attrs = match.group(1)
        target_match = re.search(r'targ="([^"]*)"', attrs) or re.search(r"targ='([^']*)'", attrs)
        return unescape(target_match.group(1)) if target_match else unescape(match.group(2))

    rendered = re.sub(r"<sub\s+([^>]*)>(.*?)</sub>", repl, value, flags=re.DOTALL)
    rendered = re.sub(r"<del\b[^>]*>.*?</del>", "", rendered, flags=re.DOTALL)
    return _strip_instruction_tags(rendered)


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _equivalent_transcript_text(left: str, right: str) -> bool:
    """Compare rendered spoken text without changing labels or edit extents."""

    return equivalent_transcript_copy(unescape(left), unescape(right))


def _equivalent_target_text(left: str, right: str) -> bool:
    """Keep target punctuation significant: it can specify the desired prosody."""

    return equivalent_text_copy(unescape(left), unescape(right))


def _raise_for_unsupported_tagged_style(operation: str, instruction: str) -> None:
    if operation != "emotion_style_edit":
        return
    lowered = instruction.lower()
    if re.search(r"\b(?:rap|rapping|rapper|rapped)\b", lowered):
        raise OperationBoundaryError("rap style needs a singing/rapping prosody tool, not emotion_style_edit")
    if re.search(r"\b(?:accent|accented|dialect|dialectal)\b", lowered):
        raise OperationBoundaryError("accent/dialect conversion is outside emotion_style_edit")


def _apply_edit_data_pipeline_defaults(
    operation: str,
    instruction: str,
    source_text: str | None,
    input_audio_path: str,
    args: dict[str, Any],
) -> None:
    if operation == "text_edit":
        args.setdefault("seed", 42)
        if uses_dots_model(operation):
            return
        args.setdefault("max_f5_edit_seconds", 32.0)
        tag_count = _text_edit_tag_count(str(args.get("instruction_st") or ""))
        if tag_count == 1 and _short_substitution_text_edit(str(args.get("instruction_st") or "")):
            args.setdefault("synthesis_mode", "per_op")
            args.setdefault("duration_mode", "estimate")
            args.setdefault("expand_radius", 0.12)
            args.setdefault("merge_gap", 0.0)
            args.setdefault("disable_prepend_context", True)
            args.setdefault("fail_if_cannot_prepend_context", False)
            args.setdefault("min_unmasked_context_sec", 0.0)
            args.setdefault("min_unmasked_context_ratio", 0.0)
        elif tag_count > 1 and isinstance(source_text, str) and _looks_long_for_f5_text(source_text):
            args.setdefault("synthesis_mode", "per_op")
    elif operation == "volume_normalize":
        args["method"] = _canonicalize_volume_normalize_method(
            args.get("method"), strict=args.get("method") not in (None, "")
        )
        args.setdefault("headroom_db", -1.0)
    elif operation == "speech_rate":
        if args.get("rate") is None:
            raise OperationBoundaryError(
                "edit_data_pipeline.speech_rate requires explicit rate; "
                "normalization does not infer semantic factors from user text"
            )
        args.setdefault("method", "wsola+indextts2")
        args.setdefault("speaker_audio_path", input_audio_path)
        if source_text:
            args.setdefault("text", source_text)
    elif operation == "pitch_rate_edit":
        args.setdefault("pitch_method", "psola+indextts2_cco")
        args.setdefault("smooth_policy", "never" if "<pitch" in str(args.get("instruction_st", "")) else "auto")
        args.setdefault("psola_indextts2_cco_pitch_boost", "mul:1.6")
        args.setdefault("crossfade_ms", 40.0)
        if str(args.get("smooth_policy", "")).lower() != "never":
            args.setdefault("smooth_margin_sec", 0.1)
    elif operation == "pitch_shift":
        if args.get("semitones") is None:
            raise OperationBoundaryError(
                "edit_data_pipeline.pitch_shift requires explicit semitones; "
                "normalization does not infer semantic factors from user text"
            )
        args["semitones"] = _integer_pitch_semitones(args["semitones"], strict=True)
        args["method"] = _canonicalize_pitch_shift_method(
            args.get("method"), strict=args.get("method") not in (None, "")
        )
    elif operation == "pause_edit":
        args.setdefault("persist", True)
    elif operation == "boundary_smooth" and source_text:
        args.setdefault("text", source_text)
    elif operation == "emotion_style_edit":
        args.setdefault("persist", True)
        args.setdefault("return_synthetic_target_audio", True)
        args.setdefault("max_candidates", 3)
        emotion_instruction = str(args.get("instruction_st") or "")
        if "<desc" in emotion_instruction or _valid_canonical_emotion_desc_instruction(emotion_instruction):
            args.setdefault("emotion_condition_mode", "desc")
            args.setdefault("synthesis_mode", "single")
    elif operation == "audio_assembly":
        _validate_audio_assembly_label_contract(args)
        if args.get("target_text") and not args.get("edit_tagged_text"):
            raise OperationBoundaryError(
                "audio_assembly cannot execute target_text-only edits; use text_edit or provide target_audio_path/edit_tagged_text"
            )


def _looks_long_for_f5_text(text: str) -> bool:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(text) > 220 or cjk_chars > 120


def _text_edit_tag_count(instruction_st: str) -> int:
    return len(
        re.findall(r"<sub\s+[^>]*targ=", instruction_st)
        + re.findall(r"<ins\b[^>]*>", instruction_st)
        + re.findall(r"<del\b[^>]*>", instruction_st)
    )


def _short_substitution_text_edit(instruction_st: str) -> bool:
    matches = _text_edit_tag_matches(instruction_st)
    if not matches:
        return False
    for match in matches:
        if match.group("sub_attrs") is None:
            return False
        source_chunk = _strip_instruction_tags(match.group("sub_body") or "").strip()
        target_chunk = _sub_target_from_attrs(match.group("sub_attrs") or "")
        if target_chunk is None:
            return False
        if _text_edit_content_units(source_chunk) > 3 or _text_edit_content_units(target_chunk) > 3:
            return False
    return True


def _text_edit_content_units(value: str) -> int:
    return len([unit for unit in _text_edit_diff_units(value) if unit.strip() and re.search(r"[\w\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", unit)])


def _requests_crying_to_normal_speech(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(lowered, ["crying tone", "crying voice", "crying", "sobbing", "weepy", "tearful"]) and _contains_any(
        lowered,
        ["normal speaking voice", "normal voice", "natural speaking voice", "plain speaking voice", "neutral voice"],
    )


def _canonicalize_known_value(key: str, value: Any, *, operation: str | None = None) -> Any:
    if key == "language":
        return canonicalize_language(value)
    if key == "semitones":
        return _integer_pitch_semitones(value, strict=True)
    if key == "aligner_backend":
        return _canonicalize_aligner_backend(value, operation=operation)
    if key == "smooth_policy" and value is not None:
        return str(value).strip().lower()
    if key == "method" and value is not None:
        if operation == "volume_normalize":
            return _canonicalize_volume_normalize_method(value, strict=True)
        if operation == "pitch_shift":
            return _canonicalize_pitch_shift_method(value, strict=True)
        return str(value).strip().lower()
    return value


def _canonicalize_pitch_shift_method(value: Any, *, strict: bool = False) -> str:
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
    if canonical in PITCH_SHIFT_METHODS:
        return canonical
    if strict:
        raise OperationBoundaryError(
            "edit_data_pipeline.pitch_shift method must be one of "
            f"{sorted(PITCH_SHIFT_METHODS)}; got {value!r}"
        )
    return "psola"


def canonicalize_volume_normalize_method(value: Any, *, strict: bool = False) -> str:
    """Return the edit-data-pipeline volume normalization method.

    ``volume_normalize`` only supports RMS or peak normalization. Keep this as
    the single public contract helper so wrapper tools do not hand-copy the
    service method boundary.
    """

    return _canonicalize_volume_normalize_method(value, strict=strict)


def _canonicalize_volume_normalize_method(value: Any, *, strict: bool = False) -> str:
    method = str(value or "rms").strip().lower().replace("_", "-")
    aliases = {
        "root-mean-square": "rms",
        "root mean square": "rms",
        "loudness": "rms",
        "dbfs": "rms",
        "peak-normalize": "peak",
        "peak-normalization": "peak",
        "max": "peak",
    }
    method = aliases.get(method, method)
    if method in VOLUME_NORMALIZE_METHODS:
        return method
    if strict:
        raise OperationBoundaryError(
            "edit_data_pipeline.volume_normalize method must be one of "
            f"{sorted(VOLUME_NORMALIZE_METHODS)}; got {value!r}. "
            "Speech-rate methods such as wsola+indextts2 belong to edit_data_pipeline.speech_rate."
        )
    return "rms"


def _canonicalize_aligner_backend(value: Any, *, operation: str | None = None) -> Any:
    if value is None:
        return value
    text = str(value).strip().lower().replace("-", "_")
    if not text:
        return value
    if text in {"qwen", "qwen3", "qwen3_omni", "qwen_3"}:
        return "qwen3"
    if text in {"ctc", "ctc_forced_aligner", "forced_aligner"}:
        return "ctc"
    if operation in {"pitch_rate_edit", "pause_edit", "emotion_style_edit", "audio_assembly"} and text in {"auto", "default", "legacy"}:
        return "qwen3"
    return text


def _infer_pitch_semitones(instruction: str) -> int:
    text = instruction.lower()
    match = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:半音|semitones?|st)", text)
    if match:
        return max(-12, min(12, int(round(float(match.group(1))))))
    if any(token in text for token in ["半个八度", "half octave"]):
        return -6 if any(token in text for token in ["降低", "低", "降", "lower", "down"]) else 6
    if any(token in text for token in ["降低", "低", "降调", "lower", "down", "deeper"]):
        return -8
    return 8


def _infer_rate_factor(instruction: str) -> float:
    text = instruction.lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:x|倍)", text)
    if match:
        return min(3.0, max(0.3, float(match.group(1))))
    if any(token in text for token in ["slow", "慢", "减速", "放慢", "slower"]):
        return 0.8
    if any(token in text for token in ["fast", "快", "加速", "faster", "speed up"]):
        return 1.2
    return 1.0


def _infer_target_db(instruction: str) -> float:
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:db|dB|分贝)", instruction)
    if match:
        return min(0.0, max(-60.0, float(match.group(1))))
    lowered = instruction.lower()
    if any(token in lowered for token in ["decrease", "lower", "reduce", "降低", "减小"]):
        return -24.0
    return -18.0


def _requests_synthetic_silence(instruction: str) -> bool:
    text = instruction.lower()
    return _contains_any(text, ["silence", "silent", "pause", "gap", "静音", "停顿", "空白"]) and _contains_any(
        text,
        ["generate", "create", "insert", "synthetic", "生成", "创建", "插入"],
    )


def _infer_duration_seconds(instruction: str, *, default: float) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*-?\s*(?:seconds?|secs?|sec|s|秒)", instruction.lower())
    if match:
        return max(0.0, float(match.group(1)))
    return max(0.0, float(default))


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _language_from_state(state: dict[str, Any]) -> str | None:
    perception = state.get("perception_report")
    if not isinstance(perception, dict):
        return None
    language = perception.get("language") or (perception.get("asr") or {}).get("language")
    return str(language) if language else None




def _qwen3_voice_design_prompt(*, request: Any, source_text: str, perception: dict[str, Any]) -> str:
    request_text = str(request or "")
    gender_age = "adult neutral speaker"
    pitch_range = "natural conversational pitch range"
    timbre_texture = "natural, clean, intelligible timbre"
    emotion = "neutral and natural emotion"
    speaking_rate = "natural speaking rate"
    delivery = "fluent, stable delivery with clear pronunciation"

    if _voice_has(
        request_text,
        [
            "童声",
            "儿童声",
            "儿童音色",
            "小孩",
            "孩子",
            "child voice",
            "kid voice",
            "little girl",
            "little boy",
            "girl voice",
            "boy voice",
        ],
    ) and not _negates_child_voice(request_text):
        if _voice_has(request_text, ["男孩", "little boy", "boy voice"]):
            gender_age = "child-like male speaker"
        elif _voice_has(request_text, ["女孩", "little girl", "girl voice"]):
            gender_age = "child-like female speaker"
        else:
            gender_age = "child-like speaker"
        gender_age += "; specifically a 5- to 8-year-old prepubescent child, not a teenager and not an adult female or adult male"
        pitch_range = "very high child pitch range with lively but controlled variation, bright but not shrill"
        timbre_texture = (
            "small vocal-tract, very light, bright, innocent child timbre; "
            "minimal chest resonance, no mature adult resonance"
        )
        delivery = "fluent, stable delivery with clear pronunciation; playful child-like narration without screaming"
    elif _voice_has(request_text, ["老人", "老年", "年老", "苍老", "elderly voice", "old voice", "older voice", "old man", "old woman"]):
        gender_age = "very elderly speaker; specifically 80 years or older, not a middle-aged adult"
        pitch_range = "low, thin aged pitch range with slight tremor and measured movement"
        timbre_texture = (
            "aged, breathy, fragile timbre with weak breath support; "
            "mildly leaky or softened dental consonants as if some teeth are missing, while remaining intelligible"
        )
        speaking_rate = "slow, measured speaking rate"
        delivery = "careful elderly delivery with slight vocal tremor and clear enough pronunciation"
    elif _voice_has(request_text, ["中年", "middle-aged", "middle aged", "middle-aged voice"]):
        gender_age = "middle-aged adult speaker; specifically 40 to 55 years old"
        pitch_range = "moderate, steady adult pitch range"
        timbre_texture = "mature, stable timbre with natural resonance"
        delivery = "steady middle-aged adult delivery with clear pronunciation"
    elif _voice_has(request_text, ["女声", "女性", "女人", "female voice", "female speaker", "woman voice"]):
        gender_age = "adult female speaker; specifically a 25- to 40-year-old adult, not a child"
        pitch_range = "natural female pitch range"
        timbre_texture = "clear, smooth female timbre"
    elif _voice_has(request_text, ["男声", "男性", "男人", "male voice", "male speaker", "man voice"]):
        gender_age = "adult male speaker; specifically a 25- to 45-year-old adult"
        pitch_range = "medium-low male pitch range"
        timbre_texture = "natural male timbre with chest resonance"

    if _voice_has(request_text, ["低沉", "更低", "低音", "deeper", "deep voice", "lower voice"]):
        pitch_range = "low pitch range with natural variation"
        timbre_texture = "warm, resonant, full-bodied timbre, not muffled"
    if _voice_has(request_text, ["更高", "尖一点", "higher voice", "voice higher"]):
        pitch_range = "higher pitch range while remaining natural and not shrill"
    if _voice_has(request_text, ["沙哑", "嘶哑", "raspy", "hoarse"]):
        timbre_texture = f"{timbre_texture}; slight raspy texture without hurting intelligibility"
    if _voice_has(request_text, ["磁性", "浑厚", "warm", "full", "rich"]):
        timbre_texture = "warm, rich, full-bodied timbre with clear articulation"
    if _voice_has(request_text, ["年轻", "younger", "young voice"]):
        gender_age = gender_age.replace("adult neutral speaker", "young adult speaker; specifically 18 to 25 years old, not a child")
        timbre_texture = "energetic, bright young-adult timbre with clean articulation and stable breath"
    if _voice_has(request_text, ["温柔", "gentle", "soft"]):
        delivery = "gentle, soft delivery with clear pronunciation"
    if _voice_has(request_text, ["严肃", "serious"]):
        delivery = "serious and steady delivery with controlled energy"
    if _voice_has(request_text, ["开心", "高兴", "happy", "excited"]):
        emotion = "happy and expressive emotion"
    if _voice_has(request_text, ["生气", "愤怒", "angry", "anger"]):
        emotion = "angry emotion at high intensity while keeping pronunciation clear"
    if _voice_has(request_text, ["惊讶", "surprised"]):
        emotion = "surprised emotion at high intensity with lively pitch movement"
    if _voice_has(request_text, ["悲伤", "难过", "sad"]):
        emotion = "sad emotion, subdued and natural"
    if _voice_has(request_text, ["慢", "放慢", "slower", "slow down"]):
        speaking_rate = "slower speaking rate"
    if _voice_has(request_text, ["快", "加速", "faster", "speed up"]):
        speaking_rate = "faster speaking rate while keeping articulation clear"

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


def _is_standard_qwen3_custom_voice_prompt(value: str) -> bool:
    text = value.strip()
    if not text.startswith("Voice design for Qwen3-TTS custom_voice."):
        return False
    return all(
        field in text
        for field in [
            "Gender/age:",
            "Pitch/range:",
            "Timbre texture:",
            "Emotion/intensity:",
            "Speaking rate:",
            "Delivery:",
            "Content constraint:",
        ]
    )


def _qwen3_custom_voice_speaker(request: Any) -> str:
    text = str(request or "")
    if _voice_has(text, ["男声", "男性", "男人", "男孩", "male voice", "male speaker", "man voice", "boy voice"]):
        return "ryan"
    return "vivian"


def _voice_has(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


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
