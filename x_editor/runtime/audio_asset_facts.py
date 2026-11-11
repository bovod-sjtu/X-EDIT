"""Read-only, revision-bound file facts; never infer an edit or a speech boundary."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import soundfile as sf
from pydantic import Field

from x_editor.core.schemas import StrictModel
from x_editor.runtime.relative_gain import RelativeGainBoundaryError, validate_gain_params

if TYPE_CHECKING:
    from x_editor.runtime.workflow_case import RuntimeAudioBinding, WorkflowRuntimeCase


class AudioGeometry(StrictModel):
    """The exact duration is frame_count / sample_rate_hz, not an ASR endpoint."""

    sample_rate_hz: int = Field(gt=0)
    frame_count: int = Field(ge=0)
    channels: int = Field(gt=0)
    duration_sec: float = Field(ge=0, allow_inf_nan=False)
    duration_basis: Literal["frame_count/sample_rate_hz"] = "frame_count/sample_rate_hz"
    coordinate_scope: Literal["whole_asset"] = "whole_asset"


class WholeAssetGainHeadroom(StrictModel):
    """Uniform gain feasibility only; selected regions require their own peak."""

    operation: Literal["artifact_ops.apply_gain"] = "artifact_ops.apply_gain"
    contract_version: Literal["relative_gain/v1"] = "relative_gain/v1"
    ceiling_source: Literal["tool_default", "runtime_frozen_parameter"] = "tool_default"
    scope: Literal["whole_asset_all_channels"] = "whole_asset_all_channels"
    ceiling_dbfs: float = Field(allow_inf_nan=False)
    peak_limit_linear: float = Field(gt=0, allow_inf_nan=False)
    maximum_uniform_gain_db: float | None = Field(default=None, allow_inf_nan=False)
    signal_is_silent: bool
    acceptance_formula: str = "sample_peak * 10**(gain_db/20) <= peak_limit_linear"
    local_scope_policy: str = (
        "Whole-asset headroom does not establish a local gain limit; measure the selected "
        "frame intervals across all channels for selected_spans. No implicit limiting or gain reduction."
    )
    silence_policy: str = (
        "Silence has no signal-derived finite gain bound; numeric tool constraints still apply."
    )


class AudioSignalFacts(StrictModel):
    scope: Literal["whole_asset_all_frames"] = "whole_asset_all_frames"
    metric: Literal["decoded_sample_peak_not_true_peak"] = "decoded_sample_peak_not_true_peak"
    sample_peak: float = Field(ge=0, allow_inf_nan=False)
    channel_sample_peaks: list[float]
    rms_amplitude: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    channel_rms_amplitudes: list[float] | None = None
    rms_definition: str = (
        "sqrt(mean(decoded_sample**2)) across all frames and channels, including silence; "
        "linear unweighted amplitude, not LUFS or perceptual loudness. Whole-asset "
        "measurements do not establish the level of a selected region or isolated speaker."
    )
    gain_headroom: WholeAssetGainHeadroom | None = None
    gain_headroom_unavailable_reason: Literal["invalid_runtime_ceiling"] | None = None


class AudioAssetFacts(StrictModel):
    schema_version: Literal["x-editor.audio-asset-facts/v1"] = "x-editor.audio-asset-facts/v1"
    audio_ref: str
    role: Literal["source", "reference", "authoritative_output"]
    origin: Literal["scenario_asset", "artifact_store_commit", "session_graph_state"]
    origin_artifact_ref_id: str | None = None
    origin_state_node_id: str | None = None
    expected_content_sha256: str | None = None
    observed_content_sha256: str | None = None
    identity_status: Literal["verified", "snapshot_only", "mismatch", "unavailable"]
    status: Literal["measured", "unknown"] = "unknown"
    unavailable_reason: Literal[
        "unreadable_asset", "content_identity_mismatch", "unsupported_or_invalid_audio",
        "empty_audio", "nonfinite_samples",
    ] | None = None
    geometry: AudioGeometry | None = None
    signal: AudioSignalFacts | None = None
    authority: str = (
        "Read-only file measurements for this content revision. File extent is independent "
        "of speech coverage and user selection. Unknown facts do not authorize inferred "
        "endpoints, gain changes, or execution rejection."
    )


def measure_audio_binding(
    binding: RuntimeAudioBinding, *, gain_headroom_db: float | None = None,
) -> AudioAssetFacts:
    """Read once; hash and decode the same bytes. No cache or path-based identity."""
    facts = AudioAssetFacts(
        audio_ref=binding.audio_ref,
        role=binding.role,
        origin=binding.origin,
        origin_artifact_ref_id=binding.origin_artifact_ref_id,
        origin_state_node_id=binding.origin_state_node_id,
        expected_content_sha256=binding.content_sha256,
        identity_status="unavailable",
    )
    try:
        snapshot = Path(binding.materialized_path).read_bytes()
    except OSError:
        return facts.model_copy(update={"unavailable_reason": "unreadable_asset"})
    observed = hashlib.sha256(snapshot).hexdigest()
    facts.observed_content_sha256 = observed
    if binding.content_sha256 is not None and binding.content_sha256 != observed:
        facts.identity_status = "mismatch"
        facts.unavailable_reason = "content_identity_mismatch"
        return facts
    facts.identity_status = "verified" if binding.content_sha256 else "snapshot_only"
    try:
        samples, rate = sf.read(io.BytesIO(snapshot), dtype="float64", always_2d=True)
    except (RuntimeError, ValueError, OSError):
        facts.unavailable_reason = "unsupported_or_invalid_audio"
        return facts
    facts.geometry = AudioGeometry(
        sample_rate_hz=rate, frame_count=len(samples), channels=samples.shape[1],
        duration_sec=len(samples) / rate,
    )
    if not len(samples):
        facts.unavailable_reason = "empty_audio"
        return facts
    if not np.isfinite(samples).all():
        facts.unavailable_reason = "nonfinite_samples"
        return facts
    peaks = np.max(np.abs(samples), axis=0)
    peak = float(np.max(peaks))
    # Scale before squaring to keep finite floating-point recordings measurable.
    channel_rms = (
        peak * np.sqrt(np.mean(np.square(samples / peak), axis=0))
        if peak else np.zeros(samples.shape[1])
    )
    rms = (
        peak * float(np.sqrt(np.mean(np.square(channel_rms / peak)))) if peak else 0.0
    )
    facts.signal = AudioSignalFacts(
        sample_peak=peak, channel_sample_peaks=peaks.tolist(),
        rms_amplitude=rms, channel_rms_amplitudes=channel_rms.tolist(),
    )
    try:
        facts.signal.gain_headroom = _whole_asset_headroom(peak, gain_headroom_db)
    except RelativeGainBoundaryError:
        # A bad optional precondition must not turn read-only measurement into admission.
        facts.signal.gain_headroom_unavailable_reason = "invalid_runtime_ceiling"
    facts.status = "measured"
    return facts


def _whole_asset_headroom(peak: float, ceiling_db: float | None) -> WholeAssetGainHeadroom:
    # The tool validator owns the default ceiling; this module cannot change it.
    params = {"gain_db": 0.0}
    if ceiling_db is not None:
        params["headroom_db"] = ceiling_db
    _, limit = validate_gain_params(params)
    return WholeAssetGainHeadroom(
        ceiling_source="runtime_frozen_parameter" if ceiling_db is not None else "tool_default",
        ceiling_dbfs=20 * math.log10(limit),
        peak_limit_linear=limit,
        maximum_uniform_gain_db=(
            20 * (math.log10(limit) - math.log10(peak)) if peak > 0 else None
        ),
        signal_is_silent=peak == 0,
    )


def measure_case_audio_facts(
    case: WorkflowRuntimeCase, *, gain_headroom_db: float | None = None,
) -> list[AudioAssetFacts]:
    """Refresh this turn's exact bindings without trusting caller-supplied facts."""
    return [
        measure_audio_binding(binding, gain_headroom_db=gain_headroom_db)
        for binding in sorted(case.audio_bindings, key=lambda item: item.order)
    ]
