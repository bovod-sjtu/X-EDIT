"""Deterministic preflight checks for runtime tool-call contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from x_editor.core.schemas import AudioEditStep
from x_editor.runtime.audio_references import resolve_audio_reference
from x_editor.runtime.operation_normalizers import (
    OperationBoundaryError,
    build_edit_data_pipeline_tool_arguments,
    canonicalize_language,
    canonicalize_volume_normalize_method,
    edit_data_pipeline_label_requirements,
)
from x_editor.workflow_control.policy import repair_action_is_enforced


@dataclass
class ToolContractViolation:
    """A blocking mismatch between a planned step and a tool input contract."""

    step_id: str
    operation: str
    tool_id: str | None
    stage: str
    reason: str
    severity: str = "blocking"
    failed_call: dict[str, Any] = field(default_factory=dict)
    tool_requirements: list[str] = field(default_factory=list)
    suggested_operations: list[str] = field(default_factory=list)
    retry_guidance: list[str] = field(default_factory=list)
    repairable_fields: list[str] = field(default_factory=list)
    requires_semantic_binding: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "tool_id": self.tool_id,
            "stage": self.stage,
            "severity": self.severity,
            "reason": self.reason,
            "failed_call": _jsonable(self.failed_call),
            "tool_requirements": list(self.tool_requirements),
            "suggested_operations": list(self.suggested_operations),
            "retry_guidance": list(self.retry_guidance),
            "repairable_fields": list(self.repairable_fields),
            "requires_semantic_binding": self.requires_semantic_binding,
        }


class ToolContractViolationError(RuntimeError):
    """Raised before a tool is invoked when preflight finds a bad call."""

    def __init__(self, violations: list[ToolContractViolation]) -> None:
        self.violations = violations
        reason = "; ".join(violation.reason for violation in violations) or "tool contract violation"
        super().__init__(reason)


def verify_step_tool_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None = None,
    output_path: str | None = None,
) -> list[ToolContractViolation]:
    """Return blocking tool contract violations without invoking the tool."""

    violations: list[ToolContractViolation] = []
    operation = step.operation

    if _requires_existing_input_audio(operation, operation_spec):
        try:
            _resolve_input_audio_for_contract(step, state)
        except Exception as exc:
            violations.append(
                build_tool_contract_violation(
                    step,
                    state,
                    operation_spec=operation_spec,
                    stage="pre_tool_invocation",
                    reason=f"{type(exc).__name__}: {exc}",
                    failed_call=_failed_call(step),
                    tool_requirements=["An existing input audio path or resolvable audio alias is required."],
                )
            )

    if operation == "artifact_ops.apply_gain":
        from x_editor.runtime.relative_gain import RelativeGainBoundaryError, validate_gain_params

        try:
            validate_gain_params(step.params)
        except RelativeGainBoundaryError as exc:
            violations.append(build_tool_contract_violation(
                step, state, operation_spec=operation_spec,
                stage="pre_tool_invocation", reason=str(exc), failed_call=_failed_call(step),
                tool_requirements=[
                    "Finite gain_db; optional nonempty explicit selected_spans; headroom_db <= 0. "
                    "Relative gain must not be replaced by an absolute target_db."
                ],
            ))

    if operation.startswith("edit_data_pipeline."):
        violation = _verify_edit_data_pipeline_contract(
            step,
            state,
            operation_spec=operation_spec,
            output_path=output_path,
        )
        if violation is not None:
            violations.append(violation)
    elif _is_speaker_scoped_edit_operation(operation):
        violation = _verify_speaker_scoped_edit_contract(
            step,
            state,
            operation_spec=operation_spec,
        )
        if violation is not None:
            violations.append(violation)
    elif operation == "foreground_background_remix":
        violation = _verify_foreground_background_remix_contract(
            step,
            state,
            operation_spec=operation_spec,
        )
        if violation is not None:
            violations.append(violation)
    elif operation == "artifact_ops.concat":
        violation = _verify_concat_contract(
            step,
            state,
            operation_spec=operation_spec,
        )
        if violation is not None:
            violations.append(violation)
    elif operation == "emphasis_stress_edit":
        violation = _verify_emphasis_stress_contract(
            step,
            state,
            operation_spec=operation_spec,
        )
        if violation is not None:
            violations.append(violation)
    elif operation == "no_reference_voice_conversion":
        violation = _verify_no_reference_voice_contract(
            step,
            state,
            operation_spec=operation_spec,
        )
        if violation is not None:
            violations.append(violation)
    elif operation in {"qwen3_tts"}:
        text = step.params.get("text") or state.get("source_text")
        if not isinstance(text, str) or not text.strip():
            violations.append(
                build_tool_contract_violation(
                    step,
                    state,
                    operation_spec=operation_spec,
                    stage="pre_tool_invocation",
                    reason=f"{operation} requires non-empty text",
                    failed_call=_failed_call(step),
                    tool_requirements=["Provide params.text, or provide source_text in runtime context."],
                    suggested_operations=["perception.speech_context", operation],
                    retry_guidance=[
                        "If the target words are not known, first obtain ASR/perception evidence, then replan synthesis.",
                    ],
                )
            )
        if operation == "qwen3_tts":
            violation = _verify_qwen3_tts_mode_contract(
                step,
                state,
                operation_spec=operation_spec,
            )
            if violation is not None:
                violations.append(violation)
    elif operation == "seamless_expressive_s2st":
        explicit_target_language = (
            step.params.get("target_language")
            or step.params.get("target_lang")
            or step.params.get("language")
        )
        context = state.get("context") if isinstance(state.get("context"), dict) else {}
        target_language = explicit_target_language
        if not repair_action_is_enforced(context, "repair_step_arguments"):
            target_language = target_language or _target_language_from_request(
                state.get("user_prompt") or step.description
            )
        if not target_language:
            violations.append(
                build_tool_contract_violation(
                    step,
                    state,
                    operation_spec=operation_spec,
                    stage="pre_tool_invocation",
                    reason="seamless_expressive_s2st requires target_language",
                    failed_call=_failed_call(step),
                    tool_requirements=["Provide target_language as a target language name or Seamless ISO-639-3 code."],
                    retry_guidance=["Extract the requested target language from the user instruction before calling Seamless."],
                    repairable_fields=["params.target_language"],
                    requires_semantic_binding=True,
                )
            )
    elif operation in {"seed_vc", "tse"}:
        reference_audio = _reference_audio_for_contract(step, state)
        if not reference_audio:
            violations.append(
                build_tool_contract_violation(
                    step,
                    state,
                    operation_spec=operation_spec,
                    stage="pre_tool_invocation",
                    reason=f"{operation} requires reference_audio_path",
                    failed_call=_failed_call(step),
                    tool_requirements=["Provide a concrete existing reference_audio_path for enrollment/voice conversion."],
                    suggested_operations=["speech_separation"] if operation == "tse" else [],
                    retry_guidance=[
                        "Use diarization/perception to locate a speaker reference segment, or choose another operation that does not require reference enrollment.",
                    ],
                )
            )
        else:
            resolved = resolve_audio_reference(reference_audio, state)
            if not isinstance(resolved, str) or not Path(resolved).exists():
                violations.append(
                    build_tool_contract_violation(
                        step,
                        state,
                        operation_spec=operation_spec,
                        stage="pre_tool_invocation",
                        reason=f"{operation} reference_audio_path does not exist: {resolved}",
                        failed_call=_failed_call(step),
                        tool_requirements=["reference_audio_path must resolve to an existing local audio file."],
                        retry_guidance=["Select a valid reference from reference_audio_paths/reference_audio_items or extract one before this step."],
                    )
                )
        if operation == "seed_vc":
            violations.extend(
                _verify_seed_vc_numeric_contract(
                    step,
                    state,
                    operation_spec=operation_spec,
                )
            )

    return violations


def build_tool_contract_violation(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None = None,
    stage: str,
    reason: str,
    failed_call: dict[str, Any] | None = None,
    tool_requirements: list[str] | None = None,
    suggested_operations: list[str] | None = None,
    retry_guidance: list[str] | None = None,
    repairable_fields: list[str] | None = None,
    requires_semantic_binding: bool = False,
) -> ToolContractViolation:
    suggestions = suggested_operations or _suggested_operations(step.operation, reason, step, state)
    requirements = tool_requirements or _tool_requirements(step.operation)
    guidance = retry_guidance or _retry_guidance(step.operation, reason, suggestions)
    return ToolContractViolation(
        step_id=step.id,
        operation=step.operation,
        tool_id=_tool_id_for_step(step, operation_spec),
        stage=stage,
        reason=reason,
        failed_call=failed_call or _failed_call(step),
        tool_requirements=requirements,
        suggested_operations=suggestions,
        retry_guidance=guidance,
        repairable_fields=list(repairable_fields or []),
        requires_semantic_binding=requires_semantic_binding,
    )


def _verify_edit_data_pipeline_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
    output_path: str | None,
) -> ToolContractViolation | None:
    params = dict(step.params)
    operation = step.operation.split(".", 1)[1]
    params.pop("operation", None)
    instruction = str(params.pop("instruction", None) or step.description or state.get("user_prompt") or "")
    source_text = params.pop("source_text", None) or state.get("source_text")
    language = _edit_pipeline_language(params.pop("language", None) or _language_from_state(state))
    try:
        input_audio_path = _resolve_input_audio_for_contract(step, state)
        build_edit_data_pipeline_tool_arguments(
            operation=operation,
            instruction=instruction,
            source_text=str(source_text) if source_text is not None else None,
            input_audio_path=input_audio_path,
            output_path=output_path or _contract_output_path(step, state),
            output_dir=state.get("output_dir"),
            language=language,
            params=params,
        )
    except OperationBoundaryError as exc:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=str(exc),
            failed_call={
                **_failed_call(step),
                "resolved_source_text": source_text,
                "resolved_language": language,
            },
        )
    except Exception as exc:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=f"{type(exc).__name__}: {exc}",
            failed_call=_failed_call(step),
        )
    return None


def _verify_speaker_scoped_edit_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> ToolContractViolation | None:
    nested_operation, nested_params = _speaker_scoped_nested_operation(step.params)
    nested_name = str(nested_operation or "").removeprefix("artifact_ops.")
    if nested_name in {"extract_spans", "concat", "reorder_spans", "reconstruct", "speaker_scoped_edit"}:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=f"speaker_scoped_edit does not support nested {nested_operation!r}",
            failed_call=_failed_call(step),
            tool_requirements=[
                "speaker_scoped_edit edits selected regions and restores them to the source timeline.",
                "Extraction and assembly are standalone operations, not nested local edits.",
            ],
            retry_guidance=[
                "Use the standalone operation with grounded spans and explicit input/output dependencies; do not change the requested output topology."
            ],
            repairable_fields=["operation", "params.edit_op", "params.edit_op_params"],
        )
    if _normalize_nested_operation_name(nested_operation) != "volume_normalize":
        return None
    try:
        canonicalize_volume_normalize_method(nested_params.get("method"), strict=True)
    except OperationBoundaryError as exc:
        repairable_field = _speaker_scoped_volume_method_field(step.params)
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=str(exc),
            failed_call={
                **_failed_call(step),
                "nested_operation": "edit_data_pipeline.volume_normalize",
                "nested_params": nested_params,
            },
            tool_requirements=[
                "artifact_ops.speaker_scoped_edit is a selector/reconstruction wrapper; its nested volume_normalize edit must satisfy the edit-data-pipeline volume contract.",
                "For edit_op=edit_data_pipeline.volume_normalize, edit_op_params.method must be either 'rms' or 'peak'.",
                "Speech-rate methods such as wsola+indextts2 belong to speech_rate or pitch_rate_edit, not volume_normalize.",
            ],
            retry_guidance=[
                "Regenerate only this failed step's nested edit_op_params; keep the speaker selector, objective, and already valid fields unchanged.",
                "If the requested edit is loudness normalization or volume attenuation, set method to 'rms' unless the request explicitly asks for peak normalization.",
                "Do not switch the operation to speech_rate or pitch_rate_edit unless the user asked to change speed, rate, or pitch.",
            ],
            repairable_fields=[repairable_field],
        )
    return None


def _verify_foreground_background_remix_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> ToolContractViolation | None:
    action = str(
        step.params.get("foreground_action") or "edit_foreground"
    ).strip()
    allowed_actions = {
        "edit_foreground",
        "remove_foreground",
        "isolate_foreground",
    }
    if action not in allowed_actions:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=(
                "foreground_background_remix has an unsupported "
                f"foreground_action: {action!r}"
            ),
            failed_call=_failed_call(step),
            tool_requirements=[
                "foreground_action must be edit_foreground, remove_foreground, or isolate_foreground."
            ],
            repairable_fields=["params.foreground_action"],
        )
    nested_operation = step.params.get("edit_op") or step.params.get(
        "edit_operation"
    )
    if (
        action == "edit_foreground"
        and not nested_operation
        and step.params.get("gain_db") is None
    ):
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=(
                "foreground_action='edit_foreground' requires an explicit nested "
                "edit_op/edit_operation or gain_db"
            ),
            failed_call=_failed_call(step),
            tool_requirements=[
                "Use isolate_foreground to output only the selected foreground stem.",
                "Use remove_foreground to output only background/residual audio.",
                "Use edit_foreground only with an explicit nested edit operation or gain_db.",
            ],
            repairable_fields=[
                "params.foreground_action",
                "params.edit_op",
            ],
        )
    return None


def _verify_concat_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> ToolContractViolation | None:
    primary = next(
        (
            mapping.get(key)
            for mapping in (step.inputs, step.params)
            for key in ("input_audio", "input_audio_path", "audio_path")
            if mapping.get(key) not in (None, "")
        ),
        None,
    )
    sources = next(
        (
            mapping.get(key)
            for mapping in (step.inputs, step.params)
            for key in (
                "audio_paths",
                "input_audio_paths",
                "input_audios",
                "sources",
            )
            if isinstance(mapping.get(key), list) and mapping.get(key)
        ),
        None,
    )
    if primary is None or not isinstance(sources, list):
        return None
    resolved_primary = resolve_audio_reference(primary, state)
    resolved_sources = [resolve_audio_reference(item, state) for item in sources]
    if resolved_primary in resolved_sources:
        return None
    return build_tool_contract_violation(
        step,
        state,
        operation_spec=operation_spec,
        stage="pre_tool_invocation",
        reason=(
            "artifact_ops.concat declares conflicting primary and multi-source "
            "audio inputs; the runtime would ignore one declaration"
        ),
        failed_call=_failed_call(step),
        tool_requirements=[
            "Declare the intended concat order in one canonical audio path list."
        ],
        retry_guidance=[
            "Do not combine input_audio with a different audio_paths/input_audio_paths list."
        ],
    )


def _verify_emphasis_stress_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> ToolContractViolation | None:
    source_text = step.params.get("source_text") or state.get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason="emphasis_stress_edit requires source_text",
            failed_call=_failed_call(step),
            tool_requirements=[
                "Provide the transcript matching input_audio before local emphasis resynthesis."
            ],
            suggested_operations=["perception.speech_context", "emphasis_stress_edit"],
        )
    instruction = str(
        step.params.get("instruction")
        or step.description
        or state.get("user_prompt")
        or ""
    )
    target_phrase = step.params.get("target_phrase") or _quoted_phrase(instruction)
    if not isinstance(target_phrase, str) or not target_phrase.strip():
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason="emphasis_stress_edit requires an explicit grounded target phrase",
            failed_call=_failed_call(step),
            tool_requirements=[
                "Provide params.target_phrase or an explicitly quoted phrase that occurs in source_text."
            ],
            repairable_fields=["params.target_phrase"],
            requires_semantic_binding=True,
        )
    if target_phrase.strip().casefold() not in source_text.casefold():
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=(
                "emphasis_stress_edit target_phrase is not grounded in source_text: "
                f"{target_phrase!r}"
            ),
            failed_call=_failed_call(step),
            tool_requirements=[
                "The exact target phrase must occur in the transcript for the input audio."
            ],
            repairable_fields=["params.target_phrase"],
            requires_semantic_binding=True,
        )
    return None


def _verify_no_reference_voice_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> ToolContractViolation | None:
    instruction = (
        step.params.get("voice_prompt")
        or step.params.get("instruction")
        or step.description
        or state.get("user_prompt")
    )
    if not isinstance(instruction, str) or not instruction.strip():
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason="no_reference_voice_conversion requires a target voice description",
            failed_call=_failed_call(step),
            tool_requirements=[
                "Provide instruction or voice_prompt describing the target voice before generating an independent reference."
            ],
            repairable_fields=["params.instruction"],
            requires_semantic_binding=True,
        )
    backend = str(step.params.get("voice_design_backend") or "auto").strip()
    if backend not in {"auto", "qwen3_tts"}:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=(
                "no_reference_voice_conversion has an unsupported "
                f"voice_design_backend: {backend!r}"
            ),
            failed_call=_failed_call(step),
            tool_requirements=[
                "voice_design_backend must be auto, qwen3_tts."
            ],
            repairable_fields=["params.voice_design_backend"],
        )
    return None


def _verify_qwen3_tts_mode_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> ToolContractViolation | None:
    mode = str(step.params.get("mode") or "custom_voice").strip()
    if mode not in {"custom_voice", "voice_clone"}:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=f"qwen3_tts has an unsupported mode: {mode!r}",
            failed_call=_failed_call(step),
            tool_requirements=["qwen3_tts mode must be custom_voice or voice_clone."],
            repairable_fields=["params.mode"],
        )
    if mode == "custom_voice" and "voice_design_backend" in step.params:
        backend = str(step.params.get("voice_design_backend") or "").strip()
        normalized_backend = backend.casefold().replace("-", "_")
        if normalized_backend not in {
            "auto",
            "qwen",
            "qwen3",
            "qwen3tts",
            "qwen3_tts",
        }:
            return build_tool_contract_violation(
                step,
                state,
                operation_spec=operation_spec,
                stage="pre_tool_invocation",
                reason=(
                    "qwen3_tts custom_voice has an unsupported "
                    f"voice_design_backend: {backend!r}"
                ),
                failed_call=_failed_call(step),
                tool_requirements=[
                    "voice_design_backend must select qwen3_tts, auto."
                ],
                repairable_fields=["params.voice_design_backend"],
            )
    if mode != "voice_clone":
        return None
    reference_audio = (
        step.params.get("ref_audio_path")
        or step.params.get("reference_audio_path")
        or step.inputs.get("reference_audio_path")
        or step.inputs.get("reference_audio")
    )
    if not reference_audio:
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason="qwen3_tts voice_clone requires reference audio",
            failed_call=_failed_call(step),
            tool_requirements=[
                "Provide ref_audio_path/reference_audio_path for voice_clone, or use custom_voice for no-reference synthesis."
            ],
            repairable_fields=["params.ref_audio_path"],
            requires_semantic_binding=True,
        )
    resolved = resolve_audio_reference(reference_audio, state)
    if not isinstance(resolved, str) or not Path(resolved).exists():
        return build_tool_contract_violation(
            step,
            state,
            operation_spec=operation_spec,
            stage="pre_tool_invocation",
            reason=f"qwen3_tts voice_clone reference audio does not exist: {resolved}",
            failed_call=_failed_call(step),
            tool_requirements=["Voice-clone reference audio must resolve to an existing file."],
        )
    return None


def _verify_seed_vc_numeric_contract(
    step: AudioEditStep,
    state: dict[str, Any],
    *,
    operation_spec: dict[str, Any] | None,
) -> list[ToolContractViolation]:
    constraints = {
        "diffusion_steps": (1.0, 100.0),
        "length_adjust": (0.5, 2.0),
        "inference_cfg_rate": (0.0, 1.0),
    }
    violations: list[ToolContractViolation] = []
    for key, (minimum, maximum) in constraints.items():
        if key not in step.params:
            continue
        raw_value = step.params.get(key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = None
        if value is not None and minimum <= value <= maximum:
            continue
        violations.append(
            build_tool_contract_violation(
                step,
                state,
                operation_spec=operation_spec,
                stage="pre_tool_invocation",
                reason=(
                    f"seed_vc {key} must be between {minimum:g} and "
                    f"{maximum:g}; got {raw_value!r}"
                ),
                failed_call=_failed_call(step),
                tool_requirements=[
                    f"Provide params.{key} in the inclusive range "
                    f"[{minimum:g}, {maximum:g}]."
                ],
                repairable_fields=[f"params.{key}"],
            )
        )
    return violations


def _quoted_phrase(value: str) -> str | None:
    for pattern in (r"'([^']+)'", r'\"([^\"]+)\"', r"“([^”]+)”", r"‘([^’]+)’"):
        match = re.search(pattern, value)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


def _speaker_scoped_nested_operation(params: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    operation: str | None = None
    operation_params: dict[str, Any] = {}
    for key in ("edit_op", "edit_operation", "sub_operation", "operation"):
        candidate_operation, candidate_params = _extract_nested_operation_value(params.get(key))
        if candidate_operation and operation is None:
            operation = candidate_operation
        operation_params.update(candidate_params)
    for key in ("edit_op_params", "edit_params", "sub_operation_params"):
        value = params.get(key)
        if isinstance(value, dict):
            operation_params.update(value)
    for key in _speaker_scoped_wrapper_nested_param_keys(operation):
        if key in params and key not in operation_params:
            operation_params[key] = params[key]
    return operation, operation_params


def _speaker_scoped_volume_method_field(params: dict[str, Any]) -> str:
    for key in ("edit_op_params", "edit_params", "sub_operation_params"):
        value = params.get(key)
        if isinstance(value, dict) and "method" in value:
            return f"params.{key}.method"
    return "params.method" if "method" in params else "params.edit_op_params.method"


def _speaker_scoped_wrapper_nested_param_keys(operation: str | None) -> set[str]:
    normalized = _normalize_nested_operation_name(operation)
    if normalized == "volume_normalize":
        return {
            "method",
            "target_db",
            "headroom_db",
            "gain_db",
            "loudness_db",
            "peak_db",
            "normalize_target_db",
            "source_text",
            "instruction_st",
            "target_text",
        }
    return set()


def _extract_nested_operation_value(value: Any) -> tuple[str | None, dict[str, Any]]:
    if value is None:
        return None, {}
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict):
        operation = (
            value.get("operation")
            or value.get("op")
            or value.get("edit_op")
            or value.get("edit_operation")
        )
        nested_params: dict[str, Any] = {}
        for key in ("params", "arguments", "edit_op_params", "edit_params"):
            child = value.get(key)
            if isinstance(child, dict):
                nested_params.update(child)
        for key, item in value.items():
            if key not in {"operation", "op", "edit_op", "edit_operation", "params", "arguments", "edit_op_params", "edit_params"}:
                nested_params.setdefault(str(key), item)
        return str(operation) if operation else None, nested_params
    return str(value), {}


def _normalize_nested_operation_name(value: str | None) -> str:
    return str(value or "").strip().lower().removeprefix("edit_data_pipeline.").replace("-", "_")


def _is_speaker_scoped_edit_operation(operation: str) -> bool:
    return operation in {"artifact_ops.speaker_scoped_edit", "speaker_scoped_edit"}


def _requires_existing_input_audio(operation: str, operation_spec: dict[str, Any] | None) -> bool:
    if operation in {"qwen3_tts", "unsupported"}:
        return False
    if operation in {"artifact_ops.silence", "artifact_ops.concat"}:
        return False
    if operation.startswith("edit_data_pipeline."):
        return True
    if isinstance(operation_spec, dict):
        required = operation_spec.get("required_inputs") or []
        if any(str(item) in {"input_audio", "input_audio_path"} for item in required):
            return True
    return operation in {
        "audio_effects",
        "seamless_expressive_s2st",
        "seed_vc",
        "tse",
        "speech_separation",
        "spatial_reverb",
        "intra_audio_voice_transfer",
        "foreground_background_remix",
        "accent_conversion",
        "no_reference_voice_conversion",
        "speech_enhancement",
        "ace_step_music.timeline_edit",
    } or operation.startswith("artifact_ops.")


def _resolve_input_audio_for_contract(step: AudioEditStep, state: dict[str, Any]) -> str:
    value = (
        step.inputs.get("input_audio_path")
        or step.inputs.get("input_audio")
        or step.inputs.get("audio_path")
        or step.inputs.get("audio")
        or step.inputs.get("source_audio_path")
        or step.inputs.get("source_audio")
        or step.inputs.get("mixture_audio_path")
        or step.inputs.get("mixture_audio")
        or step.params.get("input_audio_path")
        or step.params.get("input_audio")
        or step.params.get("audio_path")
        or step.params.get("audio")
        or step.params.get("source_audio_path")
        or step.params.get("source_audio")
        or step.params.get("mixture_audio_path")
        or step.params.get("mixture_audio")
        or "current"
    )
    value = resolve_audio_reference(value, state)
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item), None)
    if not value:
        raise RuntimeError(f"Step {step.id} requires an input audio path")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Input audio path does not exist for step {step.id}: {path}")
    return str(path)


def _reference_audio_for_contract(step: AudioEditStep, state: dict[str, Any]) -> Any:
    refs = state.get("reference_audio_paths") or []
    return (
        step.params.get("reference_audio_path")
        or step.inputs.get("reference_audio_path")
        or step.inputs.get("reference_audio")
        or (refs[0] if refs else None)
    )


def _failed_call(step: AudioEditStep) -> dict[str, Any]:
    return {
        "step_id": step.id,
        "operation": step.operation,
        "description": step.description,
        "inputs": dict(step.inputs),
        "params": dict(step.params),
        "depends_on": list(step.depends_on),
    }


def _tool_id_for_step(step: AudioEditStep, operation_spec: dict[str, Any] | None) -> str | None:
    if isinstance(operation_spec, dict) and operation_spec.get("tool_id"):
        return str(operation_spec["tool_id"])
    if step.operation.startswith("edit_data_pipeline."):
        return "edit_data_pipeline"
    if "." not in step.operation:
        return step.operation
    return step.operation.split(".", 1)[0]


def _tool_requirements(operation: str) -> list[str]:
    if operation.startswith("edit_data_pipeline."):
        return edit_data_pipeline_label_requirements(operation)
    if operation in {"seed_vc", "tse"}:
        return ["A concrete existing reference_audio_path is required."]
    if operation in {"qwen3_tts"}:
        return ["A non-empty text field is required."]
    if operation == "seamless_expressive_s2st":
        return ["A target_language is required."]
    return ["Inputs must satisfy the registered runtime operation contract before the tool is invoked."]


def _suggested_operations(
    operation: str,
    reason: str,
    step: AudioEditStep,
    state: dict[str, Any],
) -> list[str]:
    text = f"{operation} {reason} {step.description} {state.get('user_prompt') or ''}".lower()
    suggestions: list[str] = []
    if any(marker in text for marker in ["rap", "rapping", "sing", "singing", "song-like", "music vocals"]):
        suggestions.append("ace_step_music.timeline_edit")
    if "telephone" in text or "phone-call" in text or "radio" in text or "underwater" in text or "muffled" in text:
        suggestions.append("audio_effects")
    if operation == "edit_data_pipeline.text_edit":
        if any(marker in text for marker in ["tone", "emotion", "style", "hushed", "reflective", "internal monologue"]):
            suggestions.append("edit_data_pipeline.emotion_style_edit")
        suggestions.append("edit_data_pipeline.text_edit")
    if operation == "edit_data_pipeline.audio_assembly":
        suggestions.extend(["edit_data_pipeline.text_edit", "artifact_ops.reconstruct"])
    if "reference_audio_path" in text and operation == "tse":
        suggestions.extend(["speech_separation", "artifact_ops.extract_spans"])
    return _dedupe(suggestions)


def _retry_guidance(operation: str, reason: str, suggestions: list[str]) -> list[str]:
    guidance = [
        f"The previous call to {operation} is invalid at the tool-contract layer: {reason}",
        "Do not repeat the same operation and params unless the listed missing/invalid fields are fixed.",
    ]
    if suggestions:
        guidance.append(f"Consider these operations if they better match the request: {', '.join(suggestions)}.")
    if operation.startswith("edit_data_pipeline."):
        guidance.append("For edit-data-pipeline tagged edits, use the smallest local transcript span with a valid service-layer tag.")
    return guidance


def _language_from_state(state: dict[str, Any]) -> str | None:
    perception = state.get("perception_report")
    if not isinstance(perception, dict):
        return None
    language = perception.get("language") or (perception.get("asr") or {}).get("language")
    return str(language) if language else None


def _edit_pipeline_language(value: Any) -> str | None:
    canonical = canonicalize_language(value)
    if canonical in {"eng", "en-us", "en-gb"}:
        return "en"
    if canonical in {"zho", "cmn", "zh-cn", "zh-hans"}:
        return "zh"
    return canonical


def _target_language_from_request(value: Any) -> str | None:
    text = str(value or "").lower()
    mapping = {
        "mandarin": "cmn",
        "chinese": "cmn",
        "english": "eng",
        "french": "fra",
        "german": "deu",
        "spanish": "spa",
        "japanese": "jpn",
        "korean": "kor",
        "portuguese": "por",
        "italian": "ita",
        "russian": "rus",
        "thai": "tha",
    }
    for name, code in mapping.items():
        if name in text:
            return code
    return None


def _contract_output_path(step: AudioEditStep, state: dict[str, Any]) -> str:
    explicit = step.params.get("output_path") or step.inputs.get("output_path")
    if explicit:
        return str(explicit)
    safe_operation = step.operation.replace(".", "_").replace("/", "_")
    return str(Path(state.get("output_dir") or "/tmp/x_editor/tool_contract") / f"{step.id}_{safe_operation}.wav")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
