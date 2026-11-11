"""Adapters between legacy graph ToolActions and structured AudioEditPlans."""

from __future__ import annotations

from typing import Any

from x_editor.core.schemas import AudioEditPlan, AudioEditStep, ToolAction
from x_editor.runtime.accent_conversion import requests_reference_accent_match


class ToolActionAdapterError(RuntimeError):
    """Raised when a graph ToolAction cannot be represented as an AudioEditStep."""


def tool_actions_to_audio_edit_plan(
    actions: list[ToolAction | dict[str, Any]],
    *,
    state: dict[str, Any],
) -> AudioEditPlan:
    """Convert graph actions into the structured executor contract."""

    parsed = [item if isinstance(item, ToolAction) else ToolAction(**item) for item in actions]
    steps: list[AudioEditStep] = []
    for action in parsed:
        operation = operation_for_tool_action(action, state=state)
        params = _params_for_action(action, state)
        inputs = {"input_audio_path": action.input_artifacts[0]} if action.input_artifacts else {}
        if action.tool_id in {"qwen3_tts"}:
            inputs = {}
        steps.append(
            AudioEditStep(
                id=action.step_id,
                operation=operation,
                description=action.objective,
                inputs=inputs,
                params=params,
                expected_output=action.output_artifact_role,
                depends_on=list(action.depends_on),
            )
        )
    return AudioEditPlan(
        request=str(state.get("user_prompt") or ""),
        input_audio=state.get("base_audio_path"),
        output_audio=state.get("output_audio_path"),
        steps=steps,
        global_params={
            "source_text": state.get("source_text"),
            "perception_report": state.get("perception_report") or state.get("audio_perception") or {},
            "reference_audio_paths": state.get("reference_audio_paths") or [],
        },
        metadata={"source_contract": "ToolAction", "adapter": "tool_actions_to_audio_edit_plan"},
    )


def operation_for_tool_action(action: ToolAction, *, state: dict[str, Any] | None = None) -> str:
    """Map one legacy graph action to an AudioEditExecutor operation id."""

    if action.tool_id == "artifact_ops":
        return _artifact_operation(action)
    if action.tool_id == "edit_data_pipeline":
        return f"edit_data_pipeline.{_edit_data_pipeline_operation(action)}"
    if action.tool_id == "ace_step_music":
        return "ace_step_music.timeline_edit"
    direct = {
        "audio_effects",
        "accent_conversion",
        "foreground_background_remix",
        "intra_audio_voice_transfer",
        "qwen3_tts",
        "seed_vc",
        "speech_enhancement",
        "speech_separation",
        "spatial_reverb",
        "tse",
        "paralinguistic_event_add",
        "paralinguistic_event_remove",
    }
    if action.tool_id in direct:
        return action.tool_id
    raise ToolActionAdapterError(f"Cannot adapt tool action to AudioEditExecutor operation: {action.tool_id}")


def _params_for_action(action: ToolAction, state: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if action.tool_id == "artifact_ops":
        params["action_objective"] = action.objective
        params["output_artifact_role"] = action.output_artifact_role
        if state.get("user_prompt"):
            params["instruction"] = state["user_prompt"]
        if len(action.input_artifacts) > 1:
            params["edited_audio_path"] = action.input_artifacts[-1]
    if action.tool_id == "qwen3_tts" and state.get("source_text"):
        params["text"] = state["source_text"]
    if action.tool_id in {
        "foreground_background_remix",
        "intra_audio_voice_transfer",
        "ace_step_music",
        "accent_conversion",
        "audio_effects",
        "spatial_reverb",
        "speech_enhancement",
    } and state.get("user_prompt"):
        params.setdefault("instruction", state["user_prompt"])
    if action.tool_id == "audio_effects" and _requests_reference_channel_match(str(state.get("user_prompt") or "")):
        refs = state.get("reference_audio_paths") or []
        if refs:
            params.setdefault("reference_audio_path", refs[0])
    if action.tool_id == "accent_conversion" and requests_reference_accent_match(str(state.get("user_prompt") or "")):
        refs = state.get("reference_audio_paths") or []
        if refs:
            params.setdefault("reference_audio_path", refs[0])
    return params


def _artifact_operation(action: ToolAction) -> str:
    role = action.output_artifact_role
    objective = action.objective.lower()
    if role == "reconstructed_audio" or len(action.input_artifacts) > 1:
        return "artifact_ops.reconstruct"
    if role == "extracted_audio" or "extract" in objective or (
        "only" in objective and "replace" not in objective and "replacing" not in objective
    ):
        return "artifact_ops.extract_spans"
    if role == "removed_audio" or "remove" in objective or "delete" in objective:
        return "artifact_ops.remove_spans"
    if role == "muted_audio" or "mute" in objective:
        return "artifact_ops.mute_spans"
    if role == "beeped_audio" or any(term in objective for term in ["beep", "bleep", "curse word", "profanity", "censor"]):
        return "artifact_ops.beep_spans"
    if role == "concatenated_audio" or "concat" in objective or "append" in objective:
        return "artifact_ops.concat"
    if role == "speaker_scoped_edit_audio" or "speaker-scoped" in objective:
        return "artifact_ops.speaker_scoped_edit"
    if "reconstruct" in objective:
        return "artifact_ops.reconstruct"
    return "artifact_ops.extract_spans"


def _edit_data_pipeline_operation(action: ToolAction) -> str:
    haystack = f"{action.step_id} {action.objective}".lower()
    for operation in [
        "text_edit",
        "pitch_rate_edit",
        "emotion_style_edit",
        "pause_edit",
        "speech_rate",
        "volume_normalize",
        "pitch_shift",
        "boundary_smooth",
        "audio_assembly",
    ]:
        if operation in haystack:
            return operation
    raise ToolActionAdapterError(f"Cannot infer edit-data-pipeline operation for action: {action.step_id}")


def _requests_reference_channel_match(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in [
            "acoustic channel condition",
            "same acoustic channel",
            "same channel condition",
            "recorded in the same acoustic",
            "recorded with the same acoustic",
            "match channel",
        ]
    )
