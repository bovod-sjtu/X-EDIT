"""Deterministic framework-principle checks for produced audio artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from x_editor.core.schemas import AudioEditStep
from x_editor.runtime.audio_effects import AUDIO_EFFECT_PRESETS
from x_editor.runtime.audio_segment_io import load_audio_segment
from x_editor.runtime.reorder_contract import (
    normalize_reorder_indices,
    reorder_indices_for_speaker_order,
)
from x_editor.runtime.verification import RuntimeVerificationReport
from x_editor.workflow_control.artifact_acceptance import (
    DELIVERY_AUDIO_INPUT_OPTIONAL_OPERATIONS,
    DELIVERY_CONTRACT_CHECK_IDS,
    DELIVERY_CONTRACT_OPERATIONS,
    EVALUATOR_DEFERRED_ARTIFACT_OPERATIONS,
    OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS,
)
from x_editor.workflow_control.schemas import ArtifactDeliveryContract

_CheckOutcome = Literal["passed", "failed", "inconclusive"]


@dataclass(frozen=True)
class _PrincipleCheck:
    check_id: str
    outcome: _CheckOutcome
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AudioProbe:
    path: str
    duration_sec: float
    frame_rate: int
    channels: int
    sample_width: int
    audio: AudioSegment


class DeterministicArtifactPrincipleVerificator:
    """Verify simple operation invariants without judging user semantics."""

    component = "framework.artifact_principles"

    def applies_to(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        del state
        del result
        return step.operation in OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        output_path = _output_path(result)
        if step.operation == "artifact_ops.apply_gain":
            from x_editor.runtime.relative_gain_verification import relative_gain_checks

            return _report(
                step, result,
                list(OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS[step.operation]),
                output_path=output_path or "",
                checks=[_PrincipleCheck(**item) for item in relative_gain_checks(step, result, state)],
            )
        required_check_ids = list(
            OPERATION_PRINCIPLE_ACCEPTANCE_CHECKS[step.operation]
        )
        if not output_path:
            return _report(
                step,
                result,
                required_check_ids,
                output_path="",
                checks=[
                    _PrincipleCheck(
                        check_id=required_check_ids[0],
                        outcome="inconclusive",
                        summary="The step result does not identify an output audio path.",
                    )
                ],
            )

        output = _probe_audio(output_path)
        if output is None:
            return _report(
                step,
                result,
                required_check_ids,
                output_path=output_path,
                checks=[
                    _PrincipleCheck(
                        check_id=required_check_ids[0],
                        outcome="failed",
                        summary="The output audio cannot be decoded for principle checks.",
                    )
                ],
            )

        manifest = _read_manifest(result)
        checks = _checks_for_operation(step, result, manifest, output)
        return _report(
            step,
            result,
            required_check_ids,
            output_path=output_path,
            checks=checks,
        )


class ArtifactDeliveryContractVerificator:
    """Verify Executor-authored delivery evidence without judging effect quality."""

    component = "framework.delivery_contract"

    def applies_to(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        del result
        del state
        return step.operation in DELIVERY_CONTRACT_OPERATIONS

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        del state
        output_path = _output_path(result)
        required_check_ids = list(DELIVERY_CONTRACT_CHECK_IDS)
        if not output_path:
            return _report(
                step,
                result,
                required_check_ids,
                output_path="",
                checks=[
                    _PrincipleCheck(
                        required_check_ids[0],
                        "inconclusive",
                        "The step result does not identify an output audio path.",
                    )
                ],
                component=f"{self.component}.{step.operation}",
            )
        output = _probe_audio(output_path)
        if output is None:
            return _report(
                step,
                result,
                required_check_ids,
                output_path=output_path,
                checks=[
                    _PrincipleCheck(
                        "framework.delivery_contract.artifact",
                        "failed",
                        "The delivered output audio cannot be decoded.",
                    )
                ],
                component=f"{self.component}.{step.operation}",
            )
        return _report(
            step,
            result,
            required_check_ids,
            output_path=output_path,
            checks=_delivery_contract_checks(step, result, output),
            component=f"{self.component}.{step.operation}",
        )


class EvaluatorDeferredArtifactVerificator:
    """Record intentional Evaluator deferral without granting acceptance."""

    component = "framework.evaluator_deferred"

    def applies_to(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        del result
        del state
        return step.operation in EVALUATOR_DEFERRED_ARTIFACT_OPERATIONS

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        del state
        output_path = _output_path(result)
        deferred_checks = _deferred_evaluator_checks(step.operation)
        return RuntimeVerificationReport(
            component=self.component,
            step_id=step.id,
            operation=step.operation,
            passed=True,
            outcome="skipped",
            score=None,
            confidence="high",
            severity="info",
            scope="step_diagnostic",
            acceptance_grade=False,
            subject_paths=[output_path] if output_path else [],
            metrics={
                "deferred_to": "final_evaluator",
                "deferred_checks": deferred_checks,
                "acceptance_authority": False,
            },
            findings=[
                (
                    "Deterministic artifact acceptance is intentionally deferred; "
                    f"the Final Evaluator must judge {', '.join(deferred_checks)}."
                )
            ],
        )


def _deferred_evaluator_checks(operation: str) -> list[str]:
    if operation in {
        "seed_vc",
        "tse",
        "no_reference_voice_conversion",
        "intra_audio_voice_transfer",
    }:
        return ["speaker_identity", "content_preservation", "subjective_quality"]
    if operation in {
        "qwen3_tts",
        "seamless_expressive_s2st",
        "edit_data_pipeline.text_edit",
    }:
        return ["target_content", "target_language", "speaker_or_voice_quality"]
    if operation in {
        "accent_conversion",
        "edit_data_pipeline.emotion_style_edit",
        "edit_data_pipeline.pitch_rate_edit",
        "edit_data_pipeline.pitch_shift",
        "edit_data_pipeline.speech_rate",
        "emphasis_stress_edit",
        "ace_step_music.timeline_edit",
    }:
        return ["requested_style_or_prosody", "content_preservation", "subjective_quality"]
    if operation in {
        "edit_data_pipeline.audio_assembly",
        "edit_data_pipeline.boundary_smooth",
        "edit_data_pipeline.pause_edit",
        "artifact_ops.speaker_scoped_edit",
        "foreground_background_remix",
    }:
        return ["target_region_correctness", "boundary_quality", "untouched_region_preservation"]
    if operation in {"paralinguistic_event_add", "paralinguistic_event_remove"}:
        return ["event_presence_or_absence", "event_placement", "speech_preservation"]
    return ["semantic_correctness", "subjective_quality", "objective_coverage"]


def _checks_for_operation(
    step: AudioEditStep,
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    operation = step.operation
    if operation == "export_wav":
        return _export_checks(result, output)
    if operation == "artifact_ops.silence":
        return _silence_checks(result, manifest, output)
    if operation == "artifact_ops.concat":
        return _concat_checks(step, result, output)
    if operation == "artifact_ops.reconstruct":
        return _reconstruct_checks(manifest, output)
    if operation == "artifact_ops.reorder_spans":
        return _reorder_checks(step, result, manifest, output)
    if operation == "edit_data_pipeline.volume_normalize":
        return _volume_normalize_checks(step, result, output)
    if operation == "audio_effects":
        return _audio_effects_checks(result, manifest, output)
    if operation == "spatial_reverb":
        return _spatial_reverb_checks(result, manifest, output)
    if operation == "speech_separation":
        return _speech_separation_checks(result, output)
    if operation == "speech_enhancement":
        return _speech_enhancement_checks(result, output)
    if operation in {
        "artifact_ops.extract_spans",
        "artifact_ops.remove_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.beep_spans",
    }:
        return _span_operation_checks(step, result, manifest, output)
    raise ValueError(f"Unsupported artifact principle verification: {operation}")


def _delivery_contract_checks(
    step: AudioEditStep,
    result: dict[str, Any],
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    raw_contract = result.get("artifact_delivery_contract")
    if raw_contract is None:
        return [
            _PrincipleCheck(
                check_id,
                "inconclusive",
                "The legacy result has no executor-authored delivery contract.",
            )
            for check_id in DELIVERY_CONTRACT_CHECK_IDS
        ]
    try:
        contract = ArtifactDeliveryContract.model_validate(raw_contract)
    except ValidationError as exc:
        return [
            _PrincipleCheck(
                check_id,
                "failed",
                "The supplied executor delivery contract is malformed.",
                {"validation_error_count": exc.error_count()},
            )
            for check_id in DELIVERY_CONTRACT_CHECK_IDS
        ]

    execution_valid, execution_metrics = _delivery_execution_is_valid(
        step,
        contract,
        output,
    )
    execution_check = _PrincipleCheck(
        "framework.delivery_contract.execution",
        "passed" if execution_valid else "failed",
        (
            "The executor-authored step, handler, invocation, and output binding are consistent."
            if execution_valid
            else "The executor delivery provenance is inconsistent with the completed step."
        ),
        execution_metrics,
    )

    input_probes = [_probe_audio(path) for path in contract.input_audio_paths]
    requires_audio_input = (
        step.operation not in DELIVERY_AUDIO_INPUT_OPTIONAL_OPERATIONS
    )
    inputs_valid = (
        (bool(contract.input_audio_paths) or not requires_audio_input)
        and all(probe is not None for probe in input_probes)
    )
    inputs_check = _PrincipleCheck(
        "framework.delivery_contract.inputs",
        "passed" if inputs_valid else "failed",
        (
            "Every declared audio input is present and decodable."
            if inputs_valid
            else "A required delivery input is absent or cannot be decoded."
        ),
        {
            "requires_audio_input": requires_audio_input,
            "declared_input_count": len(contract.input_audio_paths),
            "decoded_input_count": sum(probe is not None for probe in input_probes),
        },
    )

    actual_size = Path(output.path).stat().st_size
    actual_hash = _sha256(output.path)
    artifact_valid = (
        _same_path(contract.output_audio_path, output.path)
        and contract.output_size_bytes == actual_size
        and contract.output_sha256 == actual_hash
        and output.duration_sec > 0.0
        and output.frame_rate > 0
        and output.channels > 0
        and output.sample_width > 0
    )
    artifact_check = _PrincipleCheck(
        "framework.delivery_contract.artifact",
        "passed" if artifact_valid else "failed",
        (
            "The delivered artifact matches its path, size, hash, and readable audio shape."
            if artifact_valid
            else "The delivered artifact is empty, malformed, changed, or bound to the wrong path."
        ),
        {
            "contract_output_audio_path": contract.output_audio_path,
            "actual_output_audio_path": output.path,
            "contract_size_bytes": contract.output_size_bytes,
            "actual_size_bytes": actual_size,
            "contract_sha256": contract.output_sha256,
            "actual_sha256": actual_hash,
            "duration_sec": output.duration_sec,
            **_format_metrics(output),
        },
    )
    return [execution_check, inputs_check, artifact_check]


def _delivery_execution_is_valid(
    step: AudioEditStep,
    contract: ArtifactDeliveryContract,
    output: _AudioProbe,
) -> tuple[bool, dict[str, Any]]:
    invocations = contract.tool_invocations
    provenance_shape_valid = (
        (contract.provenance == "local" and not invocations)
        or (
            contract.provenance == "tool"
            and len(invocations) == 1
            and invocations[0].success
            and invocations[0].output_audio_path is not None
            and _same_path(invocations[0].output_audio_path, output.path)
        )
        or (
            contract.provenance == "composite"
            and bool(invocations)
            and any(invocation.success for invocation in invocations)
        )
    )
    valid = (
        contract.schema_version == "x-editor.artifact-delivery/v1"
        and contract.step_id == step.id
        and contract.requested_operation == step.operation
        and bool(contract.delivered_operation.strip())
        and bool(contract.handler.strip())
        and contract.handler_completed is True
        and _same_path(contract.output_audio_path, output.path)
        and provenance_shape_valid
    )
    return valid, {
        "schema_version": contract.schema_version,
        "contract_step_id": contract.step_id,
        "expected_step_id": step.id,
        "requested_operation": contract.requested_operation,
        "expected_operation": step.operation,
        "delivered_operation": contract.delivered_operation,
        "operation_evidence": contract.operation_evidence,
        "handler": contract.handler,
        "handler_completed": contract.handler_completed,
        "provenance": contract.provenance,
        "tool_invocation_count": len(invocations),
        "successful_tool_invocation_count": sum(item.success for item in invocations),
        "provenance_shape_valid": provenance_shape_valid,
    }


def _audio_effects_checks(
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "audio_effects"
    if manifest is None:
        return [
            _PrincipleCheck(
                f"{prefix}.{suffix}",
                "inconclusive",
                "The executed audio-effects manifest is unavailable.",
            )
            for suffix in ("contract", "timeline", "scope_signal")
        ]

    source_path = _string_or_none(manifest.get("input_audio_path"))
    source = _probe_audio(source_path)
    preset = _string_or_none(manifest.get("preset"))
    strength = _strict_float(manifest.get("strength"))
    span = manifest.get("span")
    span_start = _first_float(span.get("start_sec")) if isinstance(span, dict) else None
    span_end = _first_float(span.get("end_sec")) if isinstance(span, dict) else None
    preserve_duration = manifest.get("preserve_duration")
    reference_path = _string_or_none(manifest.get("reference_audio_path"))
    evidence_complete = all(
        key in manifest
        for key in (
            "operation",
            "backend",
            "input_audio_path",
            "output_audio_path",
            "preset",
            "strength",
            "span",
            "preserve_duration",
        )
    )
    span_valid = (
        source is not None
        and span_start is not None
        and span_end is not None
        and 0.0 <= span_start < span_end <= source.duration_sec + 0.02
    )
    reference_valid = (
        preset != "match_channel_reference"
        or (reference_path is not None and _probe_audio(reference_path) is not None)
    )
    contract_valid = (
        evidence_complete
        and source is not None
        and manifest.get("operation") == "audio_effects"
        and manifest.get("backend") == "scipy_dsp"
        and preset in AUDIO_EFFECT_PRESETS
        and strength is not None
        and 0.0 <= strength <= 1.0
        and isinstance(preserve_duration, bool)
        and span_valid
        and reference_valid
        and _same_path(str(manifest.get("output_audio_path") or ""), output.path)
    )
    if not evidence_complete:
        contract_outcome: _CheckOutcome = "inconclusive"
    else:
        contract_outcome = "passed" if contract_valid else "failed"
    contract_check = _PrincipleCheck(
        f"{prefix}.contract",
        contract_outcome,
        (
            "The manifest records a valid deterministic audio-effects invocation."
            if contract_outcome == "passed"
            else (
                "The audio-effects manifest is invalid or inconsistent with the artifact."
                if contract_outcome == "failed"
                else "The audio-effects manifest lacks required execution evidence."
            )
        ),
        {
            "preset": preset,
            "strength": strength,
            "span_start_sec": span_start,
            "span_end_sec": span_end,
            "preserve_duration": preserve_duration,
            "reference_audio_path": reference_path,
        },
    )
    if contract_outcome != "passed" or source is None:
        return [
            contract_check,
            _PrincipleCheck(
                f"{prefix}.timeline",
                "inconclusive",
                "Timeline preservation cannot be checked without a valid execution contract.",
            ),
            _PrincipleCheck(
                f"{prefix}.scope_signal",
                "inconclusive",
                "Effect scope and signal health cannot be checked without a valid execution contract.",
            ),
        ]

    timeline_duration = _duration_check(
        f"{prefix}.timeline",
        source.duration_sec,
        output.duration_sec,
        absolute_tolerance_sec=0.05,
    )
    format_matches = (
        source.frame_rate == output.frame_rate
        and source.channels == output.channels
    )
    timeline_outcome: _CheckOutcome = (
        timeline_duration.outcome if format_matches else "failed"
    )
    timeline_check = _PrincipleCheck(
        f"{prefix}.timeline",
        timeline_outcome,
        (
            "The effect preserves source duration and stream format."
            if timeline_outcome == "passed"
            else "The effect changed source duration or stream format outside its contract."
        ),
        {
            **timeline_duration.metrics,
            "format_matches": format_matches,
            "source_format": _format_metrics(source),
            "output_format": _format_metrics(output),
        },
    )
    return [
        contract_check,
        timeline_check,
        _audio_effect_scope_signal_check(
            source,
            output,
            span_start_sec=float(span_start),
            span_end_sec=float(span_end),
            strength=float(strength),
        ),
    ]


def _audio_effect_scope_signal_check(
    source: _AudioProbe,
    output: _AudioProbe,
    *,
    span_start_sec: float,
    span_end_sec: float,
    strength: float,
) -> _PrincipleCheck:
    check_id = "audio_effects.scope_signal"
    comparable_source_audio = source.audio.set_sample_width(output.sample_width)
    source_samples = _audio_segment_sample_matrix(comparable_source_audio)
    output_samples = _audio_segment_sample_matrix(output.audio)
    samples_compatible = (
        source_samples is not None
        and output_samples is not None
        and source_samples.shape == output_samples.shape
    )
    outside_max_delta: int | None = None
    inside_max_delta: int | None = None
    if samples_compatible and source_samples is not None and output_samples is not None:
        start_frame = max(0, round(span_start_sec * source.frame_rate))
        end_frame = min(len(source_samples), round(span_end_sec * source.frame_rate))
        outside_parts = [
            np.abs(source_samples[:start_frame] - output_samples[:start_frame]),
            np.abs(source_samples[end_frame:] - output_samples[end_frame:]),
        ]
        populated_outside = [part for part in outside_parts if part.size]
        outside_max_delta = (
            int(max(np.max(part) for part in populated_outside))
            if populated_outside
            else 0
        )
        inside_delta = np.abs(
            source_samples[start_frame:end_frame]
            - output_samples[start_frame:end_frame]
        )
        inside_max_delta = int(np.max(inside_delta)) if inside_delta.size else 0

    target_source = source.audio[
        round(span_start_sec * 1000.0) : round(span_end_sec * 1000.0)
    ]
    severe_clipping = _severe_clipping_fraction(output.audio) > 0.01
    audible_source_lost = source.audio.rms > 2 and output.audio.rms <= 2
    outside_preserved = samples_compatible and (outside_max_delta or 0) <= 1
    change_required = strength > 1e-6 and target_source.rms > 2
    changed_inside = inside_max_delta is not None and inside_max_delta > 1
    if not samples_compatible or not outside_preserved or severe_clipping or audible_source_lost:
        outcome: _CheckOutcome = "failed"
    elif change_required and not changed_inside:
        outcome = "failed"
    elif source.audio.rms <= 2 and output.audio.rms <= 2:
        outcome = "inconclusive"
    else:
        outcome = "passed"
    return _PrincipleCheck(
        check_id,
        outcome,
        (
            "The effect stays inside its declared span and preserves a usable, unclipped signal."
            if outcome == "passed"
            else (
                "The effect scope or output signal violates the deterministic artifact contract."
                if outcome == "failed"
                else "Near-silent audio cannot demonstrate that the requested effect was applied."
            )
        ),
        {
            "samples_compatible": samples_compatible,
            "outside_max_sample_delta": outside_max_delta,
            "inside_max_sample_delta": inside_max_delta,
            "change_required": change_required,
            "changed_inside": changed_inside,
            "source_rms": source.audio.rms,
            "output_rms": output.audio.rms,
            "full_scale_sample_fraction": _severe_clipping_fraction(output.audio),
        },
    )


def _spatial_reverb_checks(
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "spatial_reverb"
    if manifest is None:
        return [
            _PrincipleCheck(
                f"{prefix}.{suffix}",
                "inconclusive",
                "The executed spatial-reverb manifest is unavailable.",
            )
            for suffix in ("contract", "duration", "signal")
        ]

    source = _probe_audio(_string_or_none(manifest.get("input_audio_path")))
    rir_path = _string_or_none(manifest.get("rir_path"))
    rir = _load_rir(rir_path)
    room = manifest.get("room_parameters")
    room_valid = _spatial_room_parameters_valid(room, source)
    preserve_duration = manifest.get("preserve_duration")
    backend = _string_or_none(manifest.get("backend"))
    evidence_complete = all(
        key in manifest
        for key in (
            "operation",
            "input_audio_path",
            "output_audio_path",
            "rir_path",
            "backend",
            "room_parameters",
            "preserve_duration",
        )
    )
    contract_valid = (
        evidence_complete
        and source is not None
        and rir is not None
        and backend in {"synthetic", "gpuRIR"}
        and room_valid
        and isinstance(preserve_duration, bool)
        and manifest.get("operation") == "spatial_reverb"
        and _same_path(str(manifest.get("output_audio_path") or ""), output.path)
    )
    if not evidence_complete:
        contract_outcome: _CheckOutcome = "inconclusive"
    else:
        contract_outcome = "passed" if contract_valid else "failed"
    contract_check = _PrincipleCheck(
        f"{prefix}.contract",
        contract_outcome,
        (
            "The manifest records valid room parameters and a finite room impulse response."
            if contract_outcome == "passed"
            else (
                "The spatial-reverb manifest, room parameters, or impulse response is invalid."
                if contract_outcome == "failed"
                else "The spatial-reverb manifest lacks required execution evidence."
            )
        ),
        {
            "backend": backend,
            "rir_path": rir_path,
            "rir_sample_count": int(rir.size) if rir is not None else None,
            "preserve_duration": preserve_duration,
            "room_parameters_valid": room_valid,
        },
    )
    if contract_outcome != "passed" or source is None or rir is None:
        return [
            contract_check,
            _PrincipleCheck(
                f"{prefix}.duration",
                "inconclusive",
                "Reverb duration cannot be checked without a valid source and impulse response.",
            ),
            _PrincipleCheck(
                f"{prefix}.signal",
                "inconclusive",
                "Reverb signal health cannot be checked without a valid execution contract.",
            ),
        ]

    expected_duration = (
        source.duration_sec
        if preserve_duration
        else source.duration_sec + max(0, rir.size - 1) / source.frame_rate
    )
    duration_check = _duration_check(
        f"{prefix}.duration",
        expected_duration,
        output.duration_sec,
        absolute_tolerance_sec=0.05,
    )
    signal_check = _basic_transform_signal_check(
        f"{prefix}.signal",
        source,
        output,
        require_stream_shape=True,
    )
    return [contract_check, duration_check, signal_check]


def _spatial_room_parameters_valid(
    value: Any,
    source: _AudioProbe | None,
) -> bool:
    if not isinstance(value, dict) or source is None:
        return False
    t60 = _strict_float(value.get("t60"))
    dry_wet = _strict_float(value.get("dry_wet"))
    rir_length = _strict_float(value.get("rir_length_sec"))
    sample_rate = _first_int(value.get("sample_rate"))
    room_size = value.get("room_size_m")
    source_position = value.get("source_position_m")
    receiver_position = value.get("receiver_position_m")
    vectors_valid = all(
        isinstance(vector, list)
        and len(vector) == 3
        and all(_strict_float(item) is not None for item in vector)
        for vector in (room_size, source_position, receiver_position)
    )
    return bool(
        t60 is not None
        and 0.08 <= t60 <= 8.0
        and dry_wet is not None
        and 0.0 <= dry_wet <= 0.95
        and rir_length is not None
        and 0.12 <= rir_length <= 8.0
        and sample_rate == source.frame_rate
        and vectors_valid
    )


def _load_rir(path: str | None) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    try:
        rir = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    except (OSError, ValueError):
        return None
    if not rir.size or not np.all(np.isfinite(rir)) or float(np.max(np.abs(rir))) <= 0.0:
        return None
    return rir


def _speech_separation_checks(
    result: dict[str, Any],
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "speech_separation"
    raw_contract = result.get("speech_separation_contract")
    if not isinstance(raw_contract, dict):
        contract_check = _PrincipleCheck(
            f"{prefix}.contract",
            "inconclusive",
            "The executed speech-separation contract is unavailable.",
        )
        source = None
    else:
        source_path = _string_or_none(raw_contract.get("input_audio_path"))
        source = _probe_audio(source_path)
        model_family = _string_or_none(raw_contract.get("model_family"))
        output_format = _string_or_none(raw_contract.get("output_format"))
        stem_policy = _string_or_none(raw_contract.get("stem_policy"))
        contract_valid = (
            source is not None
            and model_family in {"bs_roformer", "melband_roformer"}
            and (output_format or "").upper() == "WAV"
            and stem_policy is not None
            and _same_path(
                str(raw_contract.get("output_path") or ""),
                output.path,
            )
        )
        contract_check = _PrincipleCheck(
            f"{prefix}.contract",
            "passed" if contract_valid else "failed",
            (
                "The separation model, source, output, and stem policy match the executed contract."
                if contract_valid
                else "The executed speech-separation contract is invalid or inconsistent."
            ),
            {
                "input_audio_path": source_path,
                "model_family": model_family,
                "output_format": output_format,
                "stem_policy": stem_policy,
            },
        )

    raw_paths = result.get("stem_paths")
    stem_paths = (
        [_string_or_none(item) for item in raw_paths]
        if isinstance(raw_paths, list)
        else []
    )
    declared_paths = [path for path in stem_paths if path is not None]
    stem_probes = [_probe_audio(path) for path in declared_paths]
    selected_path = _string_or_none(result.get("selected_stem_path"))
    from x_editor.tools.catalog.audio_separator_roformer.stem_selection import select_stem

    selected_matches_policy = False
    if isinstance(raw_contract, dict) and selected_path:
        try:
            expected_stem = select_stem(
                [Path(path) for path in declared_paths], str(raw_contract.get("stem_policy") or ""),
            )
            selected_matches_policy = _same_path(str(expected_stem), selected_path)
        except ValueError:
            pass
    selected_in_stems = bool(
        selected_path
        and any(_same_path(selected_path, path) for path in declared_paths)
    )
    selected_hash = _sha256(selected_path) if selected_path else None
    output_hash = _sha256(output.path)
    output_matches_selected = bool(
        selected_path
        and (
            _same_path(selected_path, output.path)
            or (selected_hash is not None and selected_hash == output_hash)
        )
    )
    structurally_valid = (
        len(declared_paths) >= 2
        and len(declared_paths) == len(raw_paths or [])
        and len({_path_key(path) for path in declared_paths}) == len(declared_paths)
        and all(probe is not None for probe in stem_probes)
        and selected_in_stems
        and selected_matches_policy
        and output_matches_selected
    )
    decoded_stems = [probe for probe in stem_probes if probe is not None]
    max_stem_rms = max((probe.audio.rms for probe in decoded_stems), default=0)
    source_is_near_silent = source is not None and source.audio.rms <= 2
    if not isinstance(raw_paths, list) or not raw_paths:
        stems_outcome: _CheckOutcome = "inconclusive"
    elif not structurally_valid:
        stems_outcome = "failed"
    elif max_stem_rms <= 2 and source_is_near_silent:
        stems_outcome = "inconclusive"
    elif max_stem_rms <= 2:
        stems_outcome = "failed"
    else:
        stems_outcome = "passed"
    stems_check = _PrincipleCheck(
        f"{prefix}.stems",
        stems_outcome,
        (
            "The tool produced readable stems and bound the output to the stem requested by its selection policy."
            if stems_outcome == "passed"
            else (
                "The declared stems or selected-output binding is invalid."
                if stems_outcome == "failed"
                else "Stem structure or audibility cannot be established from the available evidence."
            )
        ),
        {
            "declared_stem_count": len(declared_paths),
            "decoded_stem_count": len(decoded_stems),
            "selected_stem_path": selected_path,
            "selected_in_stems": selected_in_stems,
            "selected_matches_policy": selected_matches_policy,
            "output_matches_selected": output_matches_selected,
            "stem_rms": [probe.audio.rms for probe in decoded_stems],
        },
    )

    if source is None or not decoded_stems or len(decoded_stems) != len(declared_paths):
        duration_check = _PrincipleCheck(
            f"{prefix}.duration",
            "inconclusive",
            "Stem durations cannot be checked without the source and all declared stems.",
        )
    else:
        tolerance = max(0.5, source.duration_sec * 0.01)
        differences = [
            abs(probe.duration_sec - source.duration_sec) for probe in decoded_stems
        ]
        duration_passed = all(difference <= tolerance for difference in differences)
        duration_check = _PrincipleCheck(
            f"{prefix}.duration",
            "passed" if duration_passed else "failed",
            (
                "Every stem preserves the source timeline within the separation tolerance."
                if duration_passed
                else "At least one separated stem is unexpectedly truncated or padded."
            ),
            {
                "source_duration_sec": source.duration_sec,
                "stem_duration_sec": [probe.duration_sec for probe in decoded_stems],
                "duration_difference_sec": differences,
                "tolerance_sec": tolerance,
            },
        )
    return [contract_check, stems_check, duration_check]


def _speech_enhancement_checks(
    result: dict[str, Any],
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "speech_enhancement"
    raw_contract = result.get("speech_enhancement_contract")
    if not isinstance(raw_contract, dict):
        return [
            _PrincipleCheck(
                f"{prefix}.{suffix}",
                "inconclusive",
                "The executed speech-enhancement contract is unavailable.",
            )
            for suffix in ("contract", "duration", "signal")
        ]

    source_path = _string_or_none(raw_contract.get("input_audio_path"))
    source = _probe_audio(source_path)
    mode = _string_or_none(raw_contract.get("mode"))
    preserve_background = raw_contract.get("preserve_background")
    contract_valid = (
        source is not None
        and mode in {"general", "denoise", "dereverb", "clarity"}
        and isinstance(preserve_background, bool)
        and _same_path(str(raw_contract.get("output_path") or ""), output.path)
    )
    contract_check = _PrincipleCheck(
        f"{prefix}.contract",
        "passed" if contract_valid else "failed",
        (
            "The enhancement mode and artifact bindings match the executed contract."
            if contract_valid
            else "The executed speech-enhancement contract is invalid or inconsistent."
        ),
        {
            "input_audio_path": source_path,
            "mode": mode,
            "preserve_background": preserve_background,
        },
    )
    if not contract_valid or source is None:
        return [
            contract_check,
            _PrincipleCheck(
                f"{prefix}.duration",
                "inconclusive",
                "Enhancement duration cannot be checked without a valid source contract.",
            ),
            _PrincipleCheck(
                f"{prefix}.signal",
                "inconclusive",
                "Enhancement signal health cannot be checked without a valid source contract.",
            ),
        ]

    duration_check = _duration_check(
        f"{prefix}.duration",
        source.duration_sec,
        output.duration_sec,
        floor_tolerance_sec=0.5,
        ratio_tolerance=0.03,
    )
    signal_check = _basic_transform_signal_check(
        f"{prefix}.signal",
        source,
        output,
        require_stream_shape=False,
    )
    return [contract_check, duration_check, signal_check]


def _basic_transform_signal_check(
    check_id: str,
    source: _AudioProbe,
    output: _AudioProbe,
    *,
    require_stream_shape: bool,
) -> _PrincipleCheck:
    stream_shape_matches = (
        source.frame_rate == output.frame_rate and source.channels == output.channels
    )
    full_scale_fraction = _severe_clipping_fraction(output.audio)
    severe_clipping = full_scale_fraction > 0.01
    audible_source_lost = source.audio.rms > 2 and output.audio.rms <= 2
    if severe_clipping or audible_source_lost or (
        require_stream_shape and not stream_shape_matches
    ):
        outcome: _CheckOutcome = "failed"
    elif source.audio.rms <= 2 and output.audio.rms <= 2:
        outcome = "inconclusive"
    else:
        outcome = "passed"
    return _PrincipleCheck(
        check_id,
        outcome,
        (
            "The output retains usable signal without severe full-scale clipping."
            if outcome == "passed"
            else (
                "The output is silent, severely clipped, or violates the required stream shape."
                if outcome == "failed"
                else "Near-silent source and output audio do not provide a stable signal check."
            )
        ),
        {
            "source_rms": source.audio.rms,
            "output_rms": output.audio.rms,
            "full_scale_sample_fraction": full_scale_fraction,
            "stream_shape_matches": stream_shape_matches,
            "source_format": _format_metrics(source),
            "output_format": _format_metrics(output),
        },
    )


def _volume_normalize_checks(
    step: AudioEditStep,
    result: dict[str, Any],
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "edit_data_pipeline.volume_normalize"
    raw_contract = result.get("volume_normalize_contract")
    if not isinstance(raw_contract, dict):
        return [
            _PrincipleCheck(
                f"{prefix}.{suffix}",
                "inconclusive",
                "The executed volume-normalization contract is unavailable.",
            )
            for suffix in ("contract", "duration", "level")
        ]

    source_path = _string_or_none(raw_contract.get("input_audio_path"))
    method = _string_or_none(raw_contract.get("method"))
    target_db = _strict_float(raw_contract.get("target_db"))
    headroom_db = _strict_float(raw_contract.get("headroom_db"))
    source = _probe_audio(source_path)
    contract_values_valid = (
        source_path is not None
        and source is not None
        and method in {"rms", "peak"}
        and target_db is not None
        and -60.0 <= target_db <= 0.0
        and headroom_db is not None
        and -20.0 <= headroom_db <= 0.0
    )
    intent_matches = _volume_contract_matches_step(
        step,
        method=method,
        target_db=target_db,
        headroom_db=headroom_db,
    )
    contract_passed = contract_values_valid and intent_matches
    contract_check = _PrincipleCheck(
        f"{prefix}.contract",
        "passed" if contract_passed else "failed",
        (
            "The executed method and numeric limits match the normalized plan contract."
            if contract_passed
            else "The executed volume-normalization contract is invalid or differs from the normalized plan."
        ),
        {
            "input_audio_path": source_path,
            "method": method,
            "target_db": target_db,
            "headroom_db": headroom_db,
            "intent_matches": intent_matches,
        },
    )
    if not contract_passed or source is None:
        return [
            contract_check,
            _PrincipleCheck(
                f"{prefix}.duration",
                "inconclusive",
                "Duration cannot be checked without a valid executed contract and source audio.",
            ),
            _PrincipleCheck(
                f"{prefix}.level",
                "inconclusive",
                "Signal level cannot be checked without a valid executed contract and source audio.",
            ),
        ]

    duration_check = _duration_check(
        f"{prefix}.duration",
        source.duration_sec,
        output.duration_sec,
        absolute_tolerance_sec=0.05,
    )
    if result.get("segmented") is True:
        level_check = _segmented_volume_normalize_level_check(
            result,
            output,
            method=method,
            target_db=target_db,
            headroom_db=headroom_db,
        )
    else:
        level_check = _volume_normalize_level_check(
            source,
            output,
            method=method,
            target_db=target_db,
            headroom_db=headroom_db,
        )
    return [contract_check, duration_check, level_check]


def _volume_contract_matches_step(
    step: AudioEditStep,
    *,
    method: str | None,
    target_db: float | None,
    headroom_db: float | None,
) -> bool:
    expected_method = _string_or_none(step.params.get("method"))
    expected_target = _strict_float(step.params.get("target_db"))
    expected_headroom = _strict_float(step.params.get("headroom_db"))
    if expected_method is not None and method != expected_method:
        return False
    if expected_target is not None and target_db != expected_target:
        return False
    return not (expected_headroom is not None and headroom_db != expected_headroom)


def _volume_normalize_level_check(
    source: _AudioProbe,
    output: _AudioProbe,
    *,
    method: str,
    target_db: float,
    headroom_db: float,
) -> _PrincipleCheck:
    check_id = "edit_data_pipeline.volume_normalize.level"
    source_rms_db = float(source.audio.dBFS)
    source_peak_db = float(source.audio.max_dBFS)
    output_rms_db = float(output.audio.dBFS)
    output_peak_db = float(output.audio.max_dBFS)
    measured_levels = (
        source_rms_db,
        source_peak_db,
        output_rms_db,
        output_peak_db,
    )
    if (
        source.audio.rms <= 2
        or output.audio.rms <= 2
        or not all(math.isfinite(value) for value in measured_levels)
    ):
        return _PrincipleCheck(
            check_id,
            "inconclusive",
            "Silent or near-silent audio does not provide a stable normalization measurement.",
            {
                "source_rms": source.audio.rms,
                "output_rms": output.audio.rms,
            },
        )

    source_level_db = source_rms_db if method == "rms" else source_peak_db
    output_level_db = output_rms_db if method == "rms" else output_peak_db
    requested_gain_db = target_db - source_level_db
    headroom_limited_gain_db = headroom_db - source_peak_db
    expected_gain_db = min(requested_gain_db, headroom_limited_gain_db)
    expected_level_db = source_level_db + expected_gain_db
    expected_peak_db = source_peak_db + expected_gain_db
    tolerance_db = 0.5
    level_difference_db = abs(output_level_db - expected_level_db)
    peak_difference_db = abs(output_peak_db - expected_peak_db)
    peak_within_limit = output_peak_db <= headroom_db + tolerance_db
    passed = (
        level_difference_db <= tolerance_db
        and peak_difference_db <= tolerance_db
        and peak_within_limit
    )
    return _PrincipleCheck(
        check_id,
        "passed" if passed else "failed",
        (
            "The measured level matches the requested normalization and absolute peak limit."
            if passed
            else "The measured level or peak differs from the requested normalization contract."
        ),
        {
            "method": method,
            "target_db": round(target_db, 6),
            "headroom_db": round(headroom_db, 6),
            "requested_gain_db": round(requested_gain_db, 6),
            "headroom_limited_gain_db": round(headroom_limited_gain_db, 6),
            "expected_gain_db": round(expected_gain_db, 6),
            "expected_level_db": round(expected_level_db, 6),
            "actual_level_db": round(output_level_db, 6),
            "level_difference_db": round(level_difference_db, 6),
            "expected_peak_db": round(expected_peak_db, 6),
            "actual_peak_db": round(output_peak_db, 6),
            "peak_difference_db": round(peak_difference_db, 6),
            "peak_within_limit": peak_within_limit,
            "tolerance_db": tolerance_db,
        },
    )


def _segmented_volume_normalize_level_check(
    result: dict[str, Any],
    output: _AudioProbe,
    *,
    method: str,
    target_db: float,
    headroom_db: float,
) -> _PrincipleCheck:
    check_id = "edit_data_pipeline.volume_normalize.level"
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return _PrincipleCheck(
            check_id,
            "inconclusive",
            "Segmented normalization lacks per-segment execution evidence.",
        )

    rendered = AudioSegment.empty()
    segment_metrics: list[dict[str, Any]] = []
    segment_outcomes: list[_CheckOutcome] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            return _PrincipleCheck(
                check_id,
                "failed",
                "Segmented normalization contains a malformed segment record.",
                {"segment_index": index},
            )
        segment_input = _probe_audio(_string_or_none(item.get("input_audio_path")))
        segment_output = _probe_audio(_string_or_none(item.get("output_audio_path")))
        if segment_input is None or segment_output is None:
            return _PrincipleCheck(
                check_id,
                "failed",
                "A segmented normalization input or output is unreadable.",
                {"segment_index": index},
            )
        rendered += segment_output.audio
        if item.get("copied") is True:
            format_matches = (
                segment_input.frame_rate == segment_output.frame_rate
                and segment_input.channels == segment_output.channels
                and segment_input.sample_width == segment_output.sample_width
            )
            copied = (
                format_matches
                and segment_input.audio.raw_data == segment_output.audio.raw_data
            )
            segment_outcome: _CheckOutcome = "passed" if copied else "failed"
            metrics = {
                "segment_index": index,
                "copied": True,
                "outcome": segment_outcome,
                "format_matches": format_matches,
                "pcm_equivalent": copied,
            }
        else:
            segment_check = _volume_normalize_level_check(
                segment_input,
                segment_output,
                method=method,
                target_db=target_db,
                headroom_db=headroom_db,
            )
            segment_outcome = segment_check.outcome
            metrics = {
                "segment_index": index,
                "copied": False,
                "outcome": segment_outcome,
                **segment_check.metrics,
            }
        segment_outcomes.append(segment_outcome)
        segment_metrics.append(metrics)

    rendered_hash = hashlib.sha256(rendered.raw_data).hexdigest()
    output_hash = hashlib.sha256(output.audio.raw_data).hexdigest()
    final_matches = (
        rendered.frame_rate == output.frame_rate
        and rendered.channels == output.channels
        and rendered.sample_width == output.sample_width
        and rendered_hash == output_hash
    )
    if not final_matches or "failed" in segment_outcomes:
        outcome: _CheckOutcome = "failed"
    elif "inconclusive" in segment_outcomes:
        outcome = "inconclusive"
    else:
        outcome = "passed"
    return _PrincipleCheck(
        check_id,
        outcome,
        (
            "Every edited segment satisfies the numeric level contract, copied segments are unchanged, and the final output matches their concatenation."
            if outcome == "passed"
            else (
                "A segmented level check or final concatenation is inconsistent."
                if outcome == "failed"
                else "At least one segmented level measurement is inconclusive."
            )
        ),
        {
            "segmented": True,
            "segment_count": len(segment_metrics),
            "segments": segment_metrics,
            "rendered_pcm_sha256": rendered_hash,
            "output_pcm_sha256": output_hash,
            "final_concatenation_matches": final_matches,
        },
    )


def _export_checks(
    result: dict[str, Any],
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    check_id = "export_wav.source_equivalent"
    source_path = _string_or_none(result.get("source_audio_path"))
    if not source_path or not Path(source_path).is_file():
        return [
            _PrincipleCheck(
                check_id,
                "inconclusive",
                "The export source path is unavailable for equivalence checking.",
            )
        ]
    if _same_path(source_path, output.path):
        return [
            _PrincipleCheck(
                check_id,
                "passed",
                "The in-place export points to the readable source artifact.",
            )
        ]
    source_hash = _sha256(source_path)
    output_hash = _sha256(output.path)
    if source_hash is None or output_hash is None:
        return [
            _PrincipleCheck(
                check_id,
                "inconclusive",
                "The source or output could not be read for equivalence hashing.",
            )
        ]
    passed = source_hash == output_hash
    return [
        _PrincipleCheck(
            check_id,
            "passed" if passed else "failed",
            (
                "The exported file is byte-equivalent to its source."
                if passed
                else "The exported file differs from its declared source."
            ),
            {"source_sha256": source_hash, "output_sha256": output_hash},
        )
    ]


def _silence_checks(
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    expected_duration = _first_float(
        result.get("duration_sec"),
        (manifest or {}).get("duration_sec"),
    )
    duration_check = _duration_check(
        "artifact_ops.silence.duration",
        expected_duration,
        output.duration_sec,
        floor_tolerance_sec=0.02,
        ratio_tolerance=0.005,
    )

    source_path = _string_or_none((manifest or {}).get("input_audio_path"))
    source = _probe_audio(source_path) if source_path else None
    if source is None:
        format_check = _PrincipleCheck(
            "artifact_ops.silence.format",
            "inconclusive",
            "The reference audio format is unavailable.",
        )
    else:
        format_matches = (
            output.frame_rate == source.frame_rate
            and output.channels == source.channels
            and output.sample_width == source.sample_width
        )
        format_check = _PrincipleCheck(
            "artifact_ops.silence.format",
            "passed" if format_matches else "failed",
            (
                "The silent artifact preserves the reference stream format."
                if format_matches
                else "The silent artifact changed the reference stream format."
            ),
            {
                "source": _format_metrics(source),
                "output": _format_metrics(output),
            },
        )

    signal_passed = output.audio.rms <= 1
    signal_check = _PrincipleCheck(
        "artifact_ops.silence.signal",
        "passed" if signal_passed else "failed",
        (
            "The generated artifact is digital silence."
            if signal_passed
            else "The generated silence contains measurable signal energy."
        ),
        {"rms": output.audio.rms},
    )
    return [duration_check, format_check, signal_check]


def _concat_checks(
    step: AudioEditStep,
    result: dict[str, Any],
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    raw_sources = result.get("sources")
    sources = raw_sources if isinstance(raw_sources, list) else []
    source_probes: list[_AudioProbe] = []
    malformed_source = False
    for item in sources:
        path = _string_or_none(item.get("path")) if isinstance(item, dict) else None
        if path is None:
            malformed_source = True
        probe = _probe_audio(path) if path else None
        if probe is not None:
            source_probes.append(probe)
    expected_count = _first_int(result.get("source_count"), len(sources))
    sources_valid = bool(sources) and len(source_probes) == len(sources)
    if expected_count is not None:
        sources_valid = sources_valid and expected_count == len(sources)
    sources_failed = bool(sources) and (
        malformed_source
        or len(source_probes) != len(sources)
        or expected_count != len(sources)
    )
    sources_check = _PrincipleCheck(
        "artifact_ops.concat.sources",
        "passed" if sources_valid else ("failed" if sources_failed else "inconclusive"),
        (
            "All declared concat sources are present and decodable."
            if sources_valid
            else (
                "The declared concat source list contains an invalid or unreadable file."
                if sources_failed
                else "The concat source list is unavailable."
            )
        ),
        {
            "declared_source_count": len(sources),
            "decoded_source_count": len(source_probes),
            "reported_source_count": expected_count,
        },
    )
    if not sources_valid:
        duration_check = _PrincipleCheck(
            "artifact_ops.concat.duration",
            "inconclusive",
            "Concat duration cannot be checked without all source durations.",
        )
        return [sources_check, duration_check]

    pause_sec = max(0.0, _first_float(result.get("pause_duration_sec")) or 0.0)
    crossfade_sec = max(0.0, (_first_float(step.params.get("crossfade_ms")) or 0.0) / 1000.0)
    expected_duration = _appended_duration(
        [probe.duration_sec for probe in source_probes],
        gap_sec=pause_sec,
        crossfade_sec=crossfade_sec,
    )
    duration_check = _duration_check(
        "artifact_ops.concat.duration",
        expected_duration,
        output.duration_sec,
        absolute_tolerance_sec=0.5,
    )
    return [sources_check, duration_check]


def _reconstruct_checks(
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "artifact_ops.reconstruct"
    if manifest is None:
        return [
            _PrincipleCheck(
                f"{prefix}.{suffix}",
                "inconclusive",
                "The reconstruct manifest is unavailable.",
            )
            for suffix in ("contract", "inputs", "duration")
        ]

    mode = _string_or_none(manifest.get("reconstruction_mode"))
    scope = _string_or_none(manifest.get("edited_artifact_scope"))
    source = _probe_audio(_string_or_none(manifest.get("input_audio_path")))
    if mode == "placeholder_compatibility":
        contract_check = _PrincipleCheck(
            f"{prefix}.contract",
            "inconclusive",
            "Placeholder compatibility copied base audio without reconstruction.",
            {"reconstruction_mode": mode, "edited_artifact_scope": scope},
        )
        return [
            contract_check,
            _reconstruct_copy_input_check(
                manifest,
                output,
                source,
                source_key="input_audio_path",
            ),
            _duration_check(
                f"{prefix}.duration",
                source.duration_sec if source else None,
                output.duration_sec,
                floor_tolerance_sec=0.05,
                ratio_tolerance=0.005,
            ),
        ]

    expected_scope = {
        "full_timeline_passthrough": "full_timeline",
        "span_replacement": "local_regions",
        "timeline_edits": "local_regions",
    }.get(mode or "")
    acceptance_eligible = manifest.get("acceptance_eligible")
    if mode is None or (expected_scope is not None and scope is None):
        contract_outcome: _CheckOutcome = "inconclusive"
    elif expected_scope is None or scope != expected_scope:
        contract_outcome = "failed"
    elif acceptance_eligible is True:
        contract_outcome = "passed"
    elif acceptance_eligible is False:
        contract_outcome = "failed"
    else:
        contract_outcome = "inconclusive"
    contract_check = _PrincipleCheck(
        f"{prefix}.contract",
        contract_outcome,
        (
            "The reconstruct mode and edited-artifact scope form an explicit contract."
            if contract_outcome == "passed"
            else (
                "The reconstruct contract is invalid or inconsistent."
                if contract_outcome == "failed"
                else "The reconstruct contract lacks explicit mode, scope, or eligibility evidence."
            )
        ),
        {
            "reconstruction_mode": mode,
            "edited_artifact_scope": scope,
            "acceptance_eligible": acceptance_eligible,
        },
    )

    if mode == "full_timeline_passthrough":
        passthrough = _probe_audio(
            _string_or_none(manifest.get("full_timeline_source_path"))
            or _string_or_none(manifest.get("edited_audio_path"))
        )
        return [
            contract_check,
            _reconstruct_copy_input_check(
                manifest,
                output,
                passthrough,
                source_key="full_timeline_source_path",
            ),
            _duration_check(
                f"{prefix}.duration",
                passthrough.duration_sec if passthrough else None,
                output.duration_sec,
                floor_tolerance_sec=0.02,
                ratio_tolerance=0.005,
            ),
        ]

    if mode == "span_replacement":
        spans = _span_rows({}, manifest)
        spans_valid = _spans_are_valid(
            spans,
            source.duration_sec if source else None,
        )
        replacement_paths = _reconstruct_replacement_paths(
            manifest.get("replacement_records")
        )
        replacements_valid = bool(replacement_paths) and all(
            _probe_audio(path) is not None for path in replacement_paths
        )
        inputs_passed = source is not None and spans_valid and replacements_valid
        inputs_check = _PrincipleCheck(
            f"{prefix}.inputs",
            "passed" if inputs_passed else "failed",
            (
                "Base audio, replacement artifacts, and replacement spans are valid."
                if inputs_passed
                else "Reconstruct replacement inputs or spans are invalid."
            ),
            {
                "span_count": len(spans),
                "replacement_count": len(replacement_paths),
            },
        )
        return [
            contract_check,
            inputs_check,
            _duration_check(
                f"{prefix}.duration",
                source.duration_sec if source else None,
                output.duration_sec,
                floor_tolerance_sec=0.05,
                ratio_tolerance=0.005,
            ),
        ]

    if mode == "timeline_edits":
        edits = manifest.get("edits")
        rows = edits if isinstance(edits, list) else []
        inputs_valid, expected_duration = _timeline_edit_invariant(rows, source)
        return [
            contract_check,
            _PrincipleCheck(
                f"{prefix}.inputs",
                "passed" if inputs_valid else "failed",
                (
                    "Timeline edits have valid modes, geometry, and replacement artifacts."
                    if inputs_valid
                    else "Timeline edit modes, geometry, or replacement artifacts are invalid."
                ),
                {"edit_count": len(rows)},
            ),
            _duration_check(
                f"{prefix}.duration",
                expected_duration,
                output.duration_sec,
                floor_tolerance_sec=0.05,
                ratio_tolerance=0.005,
            ),
        ]

    return [
        contract_check,
        _PrincipleCheck(
            f"{prefix}.inputs",
            "inconclusive",
            "Reconstruct inputs cannot be interpreted without a supported mode.",
        ),
        _PrincipleCheck(
            f"{prefix}.duration",
            "inconclusive",
            "Reconstruct duration cannot be interpreted without a supported mode.",
        ),
    ]


def _reconstruct_copy_input_check(
    manifest: dict[str, Any],
    output: _AudioProbe,
    source: _AudioProbe | None,
    *,
    source_key: str,
) -> _PrincipleCheck:
    check_id = "artifact_ops.reconstruct.inputs"
    if source is None:
        return _PrincipleCheck(
            check_id,
            "inconclusive",
            "The declared reconstruct copy source is unavailable.",
            {"source_key": source_key},
        )
    source_hash = _sha256(source.path)
    output_hash = _sha256(output.path)
    passed = source_hash is not None and source_hash == output_hash
    return _PrincipleCheck(
        check_id,
        "passed" if passed else "failed",
        (
            "The reconstruct copy output is byte-equivalent to its declared source."
            if passed
            else "The reconstruct copy output differs from its declared source."
        ),
        {
            "source_key": source_key,
            "manifest_source": manifest.get(source_key),
            "source_sha256": source_hash,
            "output_sha256": output_hash,
        },
    )


def _reconstruct_replacement_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    paths: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        path = _string_or_none(item.get("replacement_audio_path"))
        if path is None:
            return []
        paths.append(path)
    return paths


def _reorder_checks(
    step: AudioEditStep,
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    prefix = "artifact_ops.reorder_spans"
    if manifest is None:
        return [
            _PrincipleCheck(
                f"{prefix}.{suffix}",
                "inconclusive",
                "The reorder manifest is unavailable.",
            )
            for suffix in ("spans", "order", "duration", "content")
        ]

    source = _probe_audio(_string_or_none(manifest.get("input_audio_path")))
    spans = _span_rows(result, manifest)
    span_speaker_ids = _reorder_span_speaker_ids(result, manifest)
    spans_valid = (
        len(spans) >= 2
        and _spans_are_valid(spans, source.duration_sec if source else None)
    )
    if source is None or not spans:
        span_outcome: _CheckOutcome = "inconclusive"
    else:
        span_outcome = "passed" if spans_valid else "failed"
    span_check = _PrincipleCheck(
        f"{prefix}.spans",
        span_outcome,
        (
            "At least two non-overlapping target spans are valid."
            if spans_valid
            else "Reorder requires at least two valid non-overlapping target spans."
        ),
        {"span_count": len(spans)},
    )

    indices, indices_error = _normalized_manifest_reorder_indices(
        manifest.get("reorder_indices"),
        span_count=len(spans),
    )
    expected_indices, expected_error = _expected_reorder_indices(
        step,
        span_count=len(spans),
        span_speaker_ids=span_speaker_ids,
    )
    if step.params.get("order") is not None:
        expected_order_source = "explicit_order"
    elif step.params.get("target_speaker_order") is not None:
        expected_order_source = "explicit_speaker_order"
    else:
        expected_order_source = "default_reverse"
    expected_mode = (
        "reverse"
        if indices is not None and indices == list(reversed(range(len(spans))))
        else "custom_order"
    )
    records_valid = _reorder_records_match(
        manifest.get("reorder_records"),
        spans=spans,
        indices=indices,
        preserve_duration=manifest.get("timeline_duration_preserved"),
    )
    order_evidence_missing = any(
        key not in manifest
        for key in (
            "reorder_indices",
            "reorder_records",
            "order_source",
            "mode",
            "timeline_duration_preserved",
        )
    )
    order_complete = (
        spans_valid
        and indices is not None
        and expected_indices is not None
        and indices == expected_indices
        and manifest.get("order_source") == expected_order_source
        and manifest.get("mode") == expected_mode
        and records_valid
    )
    if not spans_valid or indices_error == "missing" or order_evidence_missing:
        order_outcome: _CheckOutcome = "inconclusive"
    elif indices_error is not None or expected_error is not None:
        order_outcome = "failed"
    else:
        order_outcome = "passed" if order_complete else "failed"
    order_check = _PrincipleCheck(
        f"{prefix}.order",
        order_outcome,
        (
            "The manifest records the complete requested source-to-slot permutation."
            if order_complete
            else "The reorder permutation or source-to-slot records are missing or inconsistent."
        ),
        {
            "reorder_indices": indices,
            "expected_reorder_indices": expected_indices,
            "order_source": manifest.get("order_source"),
            "expected_order_source": expected_order_source,
            "indices_error": indices_error,
            "expected_order_error": expected_error,
            "records_valid": records_valid,
            "order_evidence_missing": order_evidence_missing,
        },
    )

    preserve_duration = manifest.get("timeline_duration_preserved")
    duration_check = _duration_check(
        f"{prefix}.duration",
        source.duration_sec if source and isinstance(preserve_duration, bool) else None,
        output.duration_sec,
        floor_tolerance_sec=0.05,
        ratio_tolerance=0.005,
    )
    content_check = _reorder_content_check(
        source,
        output,
        spans=spans,
        indices=indices,
        preserve_duration=preserve_duration,
        prerequisites_valid=spans_valid and order_complete,
    )
    return [span_check, order_check, duration_check, content_check]


def _normalized_manifest_reorder_indices(
    value: Any,
    *,
    span_count: int,
) -> tuple[list[int] | None, str | None]:
    if value is None:
        return None, "missing"
    try:
        return normalize_reorder_indices(value, span_count=span_count), None
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _expected_reorder_indices(
    step: AudioEditStep,
    *,
    span_count: int,
    span_speaker_ids: list[str | None],
) -> tuple[list[int] | None, str | None]:
    raw_order = step.params.get("order")
    raw_speaker_order = step.params.get("target_speaker_order")
    if raw_order is not None and raw_speaker_order is not None:
        return None, "order and target_speaker_order cannot both be set"
    if raw_order is None and raw_speaker_order is None:
        return list(reversed(range(span_count))), None
    try:
        if raw_order is not None:
            return normalize_reorder_indices(raw_order, span_count=span_count), None
        return (
            reorder_indices_for_speaker_order(
                raw_speaker_order,
                span_speaker_ids=span_speaker_ids,
            ),
            None,
        )
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _reorder_span_speaker_ids(
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str | None]:
    raw = result.get("selected_spans") or manifest.get("selected_spans")
    if not isinstance(raw, list):
        return []
    return [
        _string_or_none(item.get("speaker_id")) if isinstance(item, dict) else None
        for item in raw
    ]


def _reorder_records_match(
    value: Any,
    *,
    spans: list[dict[str, float]],
    indices: list[int] | None,
    preserve_duration: Any,
) -> bool:
    if (
        not isinstance(value, list)
        or indices is None
        or len(value) != len(indices)
        or not isinstance(preserve_duration, bool)
    ):
        return False
    output_cursor_ms = 0
    source_cursor_ms = 0
    for destination_index, (source_index, record) in enumerate(
        zip(indices, value, strict=True)
    ):
        if not isinstance(record, dict):
            return False
        destination_span = spans[destination_index]
        source_span = spans[source_index]
        destination_start_ms = round(destination_span["start_sec"] * 1000.0)
        destination_end_ms = round(destination_span["end_sec"] * 1000.0)
        source_start_ms = round(source_span["start_sec"] * 1000.0)
        source_end_ms = round(source_span["end_sec"] * 1000.0)
        output_cursor_ms += destination_start_ms - source_cursor_ms
        replacement_duration_ms = (
            destination_end_ms - destination_start_ms
            if preserve_duration
            else source_end_ms - source_start_ms
        )
        expected_output_span = {
            "start_sec": output_cursor_ms / 1000.0,
            "end_sec": (output_cursor_ms + replacement_duration_ms) / 1000.0,
        }
        if (
            record.get("destination_index") != destination_index
            or record.get("source_index") != source_index
            or not _span_mapping_matches(record.get("destination_span"), destination_span)
            or not _span_mapping_matches(record.get("source_span"), source_span)
            or not _span_mapping_matches(record.get("output_span"), expected_output_span)
            or record.get("fit_to_slot_duration") is not preserve_duration
        ):
            return False
        output_cursor_ms += replacement_duration_ms
        source_cursor_ms = destination_end_ms
    return True


def _span_mapping_matches(value: Any, expected: dict[str, float]) -> bool:
    if not isinstance(value, dict):
        return False
    start = _first_float(value.get("start_sec"), value.get("start"))
    end = _first_float(value.get("end_sec"), value.get("end"))
    return (
        start is not None
        and end is not None
        and abs(start - expected["start_sec"]) <= 1e-6
        and abs(end - expected["end_sec"]) <= 1e-6
    )


def _reorder_content_check(
    source: _AudioProbe | None,
    output: _AudioProbe,
    *,
    spans: list[dict[str, float]],
    indices: list[int] | None,
    preserve_duration: Any,
    prerequisites_valid: bool,
) -> _PrincipleCheck:
    check_id = "artifact_ops.reorder_spans.content"
    if (
        source is None
        or indices is None
        or not isinstance(preserve_duration, bool)
        or not prerequisites_valid
    ):
        return _PrincipleCheck(
            check_id,
            "inconclusive",
            "The reordered content cannot be checked without valid source, spans, order, and mode evidence.",
        )
    expected = _render_expected_reorder(
        source.audio,
        spans=spans,
        indices=indices,
        preserve_duration=preserve_duration,
    )
    expected_hash = hashlib.sha256(expected.raw_data).hexdigest()
    output_hash = hashlib.sha256(output.audio.raw_data).hexdigest()
    format_matches = (
        expected.frame_rate == output.frame_rate
        and expected.channels == output.channels
        and expected.sample_width == output.sample_width
    )
    passed = format_matches and expected_hash == output_hash
    return _PrincipleCheck(
        check_id,
        "passed" if passed else "failed",
        (
            "Every destination slot contains the declared source span and untouched regions are preserved."
            if passed
            else "The output content does not match the declared source-to-slot reorder mapping."
        ),
        {
            "expected_pcm_sha256": expected_hash,
            "output_pcm_sha256": output_hash,
            "format_matches": format_matches,
        },
    )


def _render_expected_reorder(
    source: AudioSegment,
    *,
    spans: list[dict[str, float]],
    indices: list[int],
    preserve_duration: bool,
) -> AudioSegment:
    rendered = (
        AudioSegment.silent(duration=0, frame_rate=source.frame_rate)
        .set_channels(source.channels)
        .set_sample_width(source.sample_width)
    )
    cursor_ms = 0
    for destination_index, source_index in enumerate(indices):
        destination = spans[destination_index]
        source_span = spans[source_index]
        destination_start_ms = round(destination["start_sec"] * 1000.0)
        destination_end_ms = round(destination["end_sec"] * 1000.0)
        replacement = _slice(source, source_span)
        if preserve_duration:
            replacement = _fit_segment_duration(
                replacement,
                destination_end_ms - destination_start_ms,
            )
        rendered += source[cursor_ms:destination_start_ms]
        rendered += replacement
        cursor_ms = destination_end_ms
    rendered += source[cursor_ms:]
    return rendered


def _fit_segment_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    target_ms = max(0, target_ms)
    if len(audio) > target_ms:
        return audio[:target_ms]
    if len(audio) < target_ms:
        silence = (
            AudioSegment.silent(duration=target_ms - len(audio), frame_rate=audio.frame_rate)
            .set_channels(audio.channels)
            .set_sample_width(audio.sample_width)
        )
        return audio + silence
    return audio


def _timeline_edit_invariant(
    rows: list[Any],
    source: _AudioProbe | None,
) -> tuple[bool, float | None]:
    if source is None or not rows:
        return False, None
    expected_duration = source.duration_sec
    previous_end = 0.0
    for item in sorted(
        rows,
        key=lambda value: (
            _first_float(value.get("start_sec")) or 0.0
            if isinstance(value, dict)
            else 0.0
        ),
    ):
        if not isinstance(item, dict):
            return False, None
        start = _first_float(item.get("start_sec"))
        end = _first_float(item.get("end_sec"))
        replacement_duration = _first_float(item.get("replacement_duration_sec"))
        replacement_path = _string_or_none(item.get("replacement_audio_path"))
        mode = _string_or_none(item.get("mode"))
        if (
            start is None
            or end is None
            or replacement_duration is None
            or replacement_path is None
            or _probe_audio(replacement_path) is None
            or mode not in {"insert", "overlay", "replace", "splice"}
            or start < 0.0
            or end < start
            or end > source.duration_sec + 0.02
        ):
            return False, None
        if mode == "insert":
            if abs(end - start) > 1e-6:
                return False, None
            expected_duration += replacement_duration
        elif mode in {"replace", "splice"}:
            if end <= start or start < previous_end - 1e-6:
                return False, None
            previous_end = end
            if mode == "splice":
                expected_duration += replacement_duration - (end - start)
        elif start > source.duration_sec:
            return False, None
    return True, expected_duration


def _span_operation_checks(
    step: AudioEditStep,
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
    output: _AudioProbe,
) -> list[_PrincipleCheck]:
    operation = step.operation
    prefix = operation
    source_path = _string_or_none((manifest or {}).get("input_audio_path"))
    source = _probe_audio(source_path) if source_path else None
    spans = _span_rows(result, manifest)
    spans_valid = _spans_are_valid(spans, source.duration_sec if source else None)
    span_outcome: _CheckOutcome
    if spans_valid:
        span_outcome = "passed"
    elif not spans or source is None:
        span_outcome = "inconclusive"
    else:
        span_outcome = "failed"
    span_check = _PrincipleCheck(
        f"{prefix}.spans",
        span_outcome,
        (
            "Selected spans are monotonic and inside the source timeline."
            if spans_valid
            else "Selected spans are missing or invalid for the source timeline."
        ),
        {"span_count": len(spans)},
    )
    if source is None or not spans_valid:
        checks = [
            span_check,
            _PrincipleCheck(
                f"{prefix}.duration",
                "inconclusive",
                "The duration invariant lacks valid source or span evidence.",
            ),
        ]
        if operation in {"artifact_ops.mute_spans", "artifact_ops.beep_spans"}:
            checks.append(
                _PrincipleCheck(
                    f"{prefix}.signal",
                    "inconclusive",
                    "The target-region signal cannot be checked without valid spans.",
                )
            )
        return checks

    durations = [float(item["end_sec"]) - float(item["start_sec"]) for item in spans]
    crossfade_sec = max(0.0, (_first_float(result.get("crossfade_ms")) or 0.0) / 1000.0)
    if operation == "artifact_ops.extract_spans":
        expected_duration = _appended_duration(durations, crossfade_sec=crossfade_sec)
    elif operation == "artifact_ops.remove_spans":
        piece_durations = _remaining_piece_durations(spans, source.duration_sec)
        expected_duration = _appended_duration(
            piece_durations,
            crossfade_sec=crossfade_sec,
        )
    else:
        expected_duration = source.duration_sec
    duration_check = _duration_check(
        f"{prefix}.duration",
        expected_duration,
        output.duration_sec,
        floor_tolerance_sec=0.05,
        ratio_tolerance=0.005,
    )
    checks = [span_check, duration_check]
    if operation == "artifact_ops.mute_spans":
        checks.append(_mute_signal_check(spans, output))
    elif operation == "artifact_ops.beep_spans":
        checks.append(_beep_signal_check(step, result, spans, output))
    return checks


def _mute_signal_check(
    spans: list[dict[str, float]],
    output: _AudioProbe,
) -> _PrincipleCheck:
    rms_values = [_slice(output.audio, span).rms for span in spans]
    passed = bool(rms_values) and all(value <= 2 for value in rms_values)
    return _PrincipleCheck(
        "artifact_ops.mute_spans.signal",
        "passed" if passed else "failed",
        (
            "Every muted target span is near digital silence."
            if passed
            else "At least one muted target span retains measurable signal energy."
        ),
        {"target_span_rms": rms_values},
    )


def _beep_signal_check(
    step: AudioEditStep,
    result: dict[str, Any],
    spans: list[dict[str, float]],
    output: _AudioProbe,
) -> _PrincipleCheck:
    requested_hz = _first_float(
        result.get("frequency_hz"),
        step.params.get("frequency_hz"),
    ) or 1000.0
    dominant_hz: list[float] = []
    rms_values: list[int] = []
    for span in spans:
        segment = _slice(output.audio, span).set_channels(1)
        rms_values.append(segment.rms)
        frequency = _dominant_frequency_hz(segment)
        if frequency is None:
            return _PrincipleCheck(
                "artifact_ops.beep_spans.signal",
                "inconclusive",
                "A target span is too short for a stable tone-frequency check.",
                {"target_span_rms": rms_values},
            )
        dominant_hz.append(frequency)
    tolerance_hz = max(80.0, requested_hz * 0.10)
    passed = bool(dominant_hz) and all(
        rms > 2 and abs(frequency - requested_hz) <= tolerance_hz
        for rms, frequency in zip(rms_values, dominant_hz, strict=True)
    )
    return _PrincipleCheck(
        "artifact_ops.beep_spans.signal",
        "passed" if passed else "failed",
        (
            "Every target span contains the requested basic beep tone."
            if passed
            else "At least one target span lacks the requested basic beep tone."
        ),
        {
            "requested_frequency_hz": requested_hz,
            "frequency_tolerance_hz": tolerance_hz,
            "dominant_frequency_hz": dominant_hz,
            "target_span_rms": rms_values,
        },
    )


def _report(
    step: AudioEditStep,
    result: dict[str, Any],
    required_check_ids: list[str],
    *,
    output_path: str,
    checks: list[_PrincipleCheck],
    component: str | None = None,
) -> RuntimeVerificationReport:
    checks_by_id = {check.check_id: check for check in checks}
    missing = [check_id for check_id in required_check_ids if check_id not in checks_by_id]
    if missing:
        checks.extend(
            _PrincipleCheck(
                check_id,
                "inconclusive",
                "The required principle check did not produce evidence.",
            )
            for check_id in missing
        )
    if any(check.outcome == "failed" for check in checks):
        outcome = "failed"
    elif any(check.outcome == "inconclusive" for check in checks):
        outcome = "inconclusive"
    else:
        outcome = "passed"
    return RuntimeVerificationReport(
        component=component
        or f"{DeterministicArtifactPrincipleVerificator.component}.{step.operation}",
        step_id=step.id,
        operation=step.operation,
        passed=outcome != "failed",
        outcome=outcome,
        score=1.0 if outcome == "passed" else (0.0 if outcome == "failed" else None),
        confidence="high" if outcome != "inconclusive" else "medium",
        severity="blocking" if outcome == "failed" else "info",
        scope="operation_invariant",
        acceptance_grade=True,
        check_ids=required_check_ids,
        check_outcomes={
            check.check_id: check.outcome for check in checks
        },
        subject_paths=[output_path] if output_path else [],
        metrics={
            "checks": {
                check.check_id: {
                    "outcome": check.outcome,
                    **check.metrics,
                }
                for check in checks
            },
            "manifest_path": result.get("manifest_path"),
        },
        findings=[f"[{check.check_id}] {check.summary}" for check in checks],
    )


def _duration_check(
    check_id: str,
    expected_sec: float | None,
    actual_sec: float,
    *,
    floor_tolerance_sec: float = 0.0,
    ratio_tolerance: float = 0.0,
    absolute_tolerance_sec: float | None = None,
) -> _PrincipleCheck:
    if expected_sec is None:
        return _PrincipleCheck(
            check_id,
            "inconclusive",
            "Expected duration evidence is unavailable.",
        )
    tolerance_sec = (
        absolute_tolerance_sec
        if absolute_tolerance_sec is not None
        else max(floor_tolerance_sec, expected_sec * ratio_tolerance)
    )
    difference_sec = abs(actual_sec - expected_sec)
    passed = difference_sec <= tolerance_sec
    return _PrincipleCheck(
        check_id,
        "passed" if passed else "failed",
        (
            "The output duration satisfies the operation invariant."
            if passed
            else "The output duration violates the operation invariant."
        ),
        {
            "expected_duration_sec": round(expected_sec, 6),
            "actual_duration_sec": round(actual_sec, 6),
            "difference_sec": round(difference_sec, 6),
            "tolerance_sec": round(tolerance_sec, 6),
        },
    )


def _appended_duration(
    durations: list[float],
    *,
    gap_sec: float = 0.0,
    crossfade_sec: float = 0.0,
) -> float:
    total = 0.0
    for index, duration in enumerate(durations):
        if index:
            total += gap_sec
        if total <= 0.0 or crossfade_sec <= 0.0:
            total += duration
        else:
            total += duration - min(crossfade_sec, total, duration)
    return max(0.0, total)


def _remaining_piece_durations(
    spans: list[dict[str, float]],
    source_duration_sec: float,
) -> list[float]:
    pieces: list[float] = []
    cursor = 0.0
    for span in spans:
        start = float(span["start_sec"])
        end = float(span["end_sec"])
        if start > cursor:
            pieces.append(start - cursor)
        cursor = max(cursor, end)
    if cursor < source_duration_sec:
        pieces.append(source_duration_sec - cursor)
    return pieces


def _span_rows(
    result: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> list[dict[str, float]]:
    raw = result.get("selected_spans") or (manifest or {}).get("selected_spans")
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            return []
        start = _first_float(item.get("start_sec"), item.get("start"))
        end = _first_float(item.get("end_sec"), item.get("end"))
        if start is None or end is None:
            return []
        rows.append({"start_sec": start, "end_sec": end})
    return rows


def _spans_are_valid(
    spans: list[dict[str, float]],
    source_duration_sec: float | None,
) -> bool:
    if not spans or source_duration_sec is None:
        return False
    previous_end = 0.0
    for span in spans:
        start = span["start_sec"]
        end = span["end_sec"]
        if start < previous_end or start < 0.0 or end <= start:
            return False
        if end > source_duration_sec + 0.02:
            return False
        previous_end = end
    return True


def _slice(audio: AudioSegment, span: dict[str, float]) -> AudioSegment:
    start_ms = round(span["start_sec"] * 1000.0)
    end_ms = round(span["end_sec"] * 1000.0)
    return audio[start_ms:end_ms]


def _dominant_frequency_hz(audio: AudioSegment) -> float | None:
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float64)
    if len(samples) < 64 or audio.frame_rate <= 0:
        return None
    samples -= float(samples.mean())
    if not np.any(samples):
        return 0.0
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / audio.frame_rate)
    if len(spectrum) <= 1:
        return None
    dominant_index = int(np.argmax(spectrum[1:]) + 1)
    return float(frequencies[dominant_index])


def _sample_matrix(probe: _AudioProbe) -> np.ndarray | None:
    return _audio_segment_sample_matrix(probe.audio)


def _audio_segment_sample_matrix(audio: AudioSegment) -> np.ndarray | None:
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.int64)
    if audio.channels <= 0 or samples.size % audio.channels:
        return None
    return samples.reshape(-1, audio.channels)


def _severe_clipping_fraction(audio: AudioSegment) -> float:
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.int64)
    if not samples.size or audio.sample_width <= 0:
        return 0.0
    full_scale = (1 << (8 * audio.sample_width - 1)) - 1
    return float(np.mean(np.abs(samples) >= max(1, full_scale - 1)))


def _probe_audio(path: str | None) -> _AudioProbe | None:
    if not path or not Path(path).is_file():
        return None
    audio = _decode_pcm_wav(path)
    if audio is not None:
        return _AudioProbe(
            path=str(path),
            duration_sec=len(audio) / 1000.0,
            frame_rate=audio.frame_rate,
            channels=audio.channels,
            sample_width=audio.sample_width,
            audio=audio,
        )
    try:
        # The fallback also accepts IEEE float WAVs; decode their values before
        # PCM arithmetic instead of interpreting floating-point bytes as PCM.
        audio = load_audio_segment(path)
    except (CouldntDecodeError, OSError, ValueError):
        return None
    return _AudioProbe(
        path=str(path),
        duration_sec=len(audio) / 1000.0,
        frame_rate=audio.frame_rate,
        channels=audio.channels,
        sample_width=audio.sample_width,
        audio=audio,
    )


def _decode_pcm_wav(path: str) -> AudioSegment | None:
    """Decode an uncompressed WAV without relying on an external ffmpeg binary."""

    try:
        with wave.open(path, "rb") as handle:
            if handle.getcomptype() != "NONE":
                return None
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if channels <= 0 or sample_width not in {1, 2, 3, 4} or frame_rate <= 0:
                return None
            frames = handle.readframes(frame_count)
    except (EOFError, OSError, wave.Error):
        return None
    if frame_count <= 0 or not frames:
        return None
    return AudioSegment(
        data=frames,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels,
    )


def _read_manifest(result: dict[str, Any]) -> dict[str, Any] | None:
    path = _string_or_none(result.get("manifest_path"))
    if not path or not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _output_path(result: dict[str, Any]) -> str | None:
    for key in (
        "output_audio_path",
        "target_audio_path",
        "output_path",
        "generated_audio_path",
    ):
        path = _string_or_none(result.get(key))
        if path:
            return path
    return None


def _sha256(path: str) -> str | None:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _format_metrics(probe: _AudioProbe) -> dict[str, int]:
    return {
        "frame_rate": probe.frame_rate,
        "channels": probe.channels,
        "sample_width": probe.sample_width,
    }


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except (OSError, RuntimeError):
        return left == right


def _path_key(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return path


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _strict_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _first_int(*values: Any) -> int | None:
    value = _first_float(*values)
    return int(value) if value is not None else None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
