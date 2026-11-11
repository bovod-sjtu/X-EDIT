"""Measured PCM facts for explicit execution-sidecar source/output mappings.

No public-instruction interpretation, semantic grounding, expected gain, or
quality verdict is inferred here. A correspondence is an execution claim.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from x_editor.evaluation.final_delivery_evaluator import FinalDeliveryEvaluationInput
from x_editor.evaluation.guidance_contracts import EvaluationEvidenceItem


def _read_bound(artifact):
    facts = {"artifact_ref_id": artifact.artifact_ref_id,
             "bound_sha256": artifact.content_sha256, "status": "unknown"}
    try:
        content = Path(artifact.path).read_bytes()
        facts["observed_sha256"] = hashlib.sha256(content).hexdigest()
        if facts["observed_sha256"] != artifact.content_sha256:
            facts["reason"] = "artifact_hash_mismatch"
            return None, facts
        samples, rate = sf.read(io.BytesIO(content), dtype="float64", always_2d=True)
        if not np.isfinite(samples).all():
            facts["reason"] = "nonfinite_audio_samples"
            return None, facts
    except (OSError, RuntimeError, ValueError):
        facts["reason"] = "artifact_unreadable_or_undecodable"
        return None, facts
    facts.update(status="available", sample_rate=rate, channels=samples.shape[1],
                 asset_frames=len(samples))
    return samples, facts


def _region(loaded, start_sec, end_sec):
    samples, metadata = loaded
    facts = dict(metadata, requested_start_sec=start_sec, requested_end_sec=end_sec)
    if samples is None:
        return None, facts
    rate = facts["sample_rate"]
    start, end = round(start_sec * rate), round(end_sec * rate)
    facts.update(start_frame=start, end_frame=end, frames=end - start)
    if start_sec < 0 or end_sec > len(samples) / rate or not (0 <= start < end <= len(samples)):
        facts.update(status="unknown", reason="interval_outside_audio_or_empty")
        return None, facts
    region = samples[start:end]
    # Scale before squaring to avoid overflow with valid floating-point audio.
    peak = float(np.max(np.abs(region)))
    rms = peak * float(np.sqrt(np.mean((region / peak) ** 2))) if peak else 0.0
    facts.update(rms_amplitude=rms)
    return region, facts


def measure_pcm_comparison(source, output):
    """Compare already aligned finite, nonempty PCM arrays without a quality verdict."""
    row = {}
    row.update(comparison_status="available", changed=not np.array_equal(source, output))
    with np.errstate(over="ignore", invalid="ignore"):
        difference = output - source
    maximum_difference = float(np.max(np.abs(difference)))
    if not math.isfinite(maximum_difference):
        row.update(comparison_status="unknown", changed=None, reason="numeric_range_unsupported")
        return row
    row["max_absolute_difference"] = maximum_difference
    scale = max(float(np.max(np.abs(source))), float(np.max(np.abs(output))))
    if not scale:
        row["gain_status"] = "unknown_silent_source"
        return row
    x, y = source / scale, output / scale
    energy = float(np.sum(x * x))
    if energy == 0:
        row["gain_status"] = "unknown_silent_source"
        return row
    gain = float(np.sum(x * y) / energy)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = scale * float(np.sqrt(np.mean((y - gain * x) ** 2)))
    if not math.isfinite(gain) or not math.isfinite(residual):
        row["gain_status"] = "unknown_numeric_range"
        return row
    row.update(least_squares_linear_gain=gain, polarity_inverted=gain < 0,
               gain_db=20 * math.log10(abs(gain)) if gain else None,
               fit_residual_rms=residual, gain_status="available")
    return row


def build_paired_signal_evidence(value: FinalDeliveryEvaluationInput) -> tuple[EvaluationEvidenceItem, ...]:
    """Return descriptive, hash-bound measurements; never judge edit correctness."""
    artifacts = {a.artifact_ref_id: a for a in (*value.input_artifacts, value.output_artifact)}
    needed = {pair.source_artifact_ref_id for pair in value.interval_pairs}
    if value.interval_pairs:
        needed.add(value.output_artifact.artifact_ref_id)
    loaded = {ref: _read_bound(artifacts[ref]) for ref in needed}
    rows = []
    for pair in value.interval_pairs:
        source, source_facts = _region(loaded[pair.source_artifact_ref_id],
                                       pair.source_start_sec, pair.source_end_sec)
        output, output_facts = _region(loaded[value.output_artifact.artifact_ref_id],
                                       pair.output_start_sec, pair.output_end_sec)
        row = {"pair_id": pair.pair_id, "declared_purpose": pair.purpose,
               "source": source_facts, "output": output_facts,
               "comparison_status": "unknown", "changed": None,
               "least_squares_linear_gain": None, "gain_db": None,
               "fit_residual_rms": None, "max_absolute_difference": None,
               "polarity_inverted": None}
        rows.append(row)
        if source is None or output is None:
            row["reason"] = "required_interval_unavailable"
            continue
        if source_facts["sample_rate"] != output_facts["sample_rate"] or source.shape != output.shape:
            row["reason"] = "incompatible_sample_rate_frame_count_or_channels"
            continue
        row.update(measure_pcm_comparison(source, output))
    payload = {
        "protocol_id": "x-editor.paired-signal-facts/v1",
        "mapping_source": "execution_sidecar_explicit_interval_pairs",
        "semantic_grounding": "unverified",
        "scope": "Only declared interval pairs; unmapped audio is not assessed.",
        "interpretation": "Measured gain is a least-squares fit, not proof of a gain-only edit. "
                          "Inspect its residual and signed gain/polarity; gain_db describes magnitude only. These facts neither establish the intended target "
                          "nor supply an expected gain, success verdict, or perceptual quality score.",
        "frame_rounding": "nearest frame using Python round; no cropping, resampling or channel mixing",
        "rms_definition": "Unweighted linear RMS over all paired frames and channels, including silence; not LUFS.",
        "status": "available" if rows else "unknown",
        "pairs": rows,
    }
    if not rows:
        payload["reason"] = "no_explicit_interval_pairs"
    return (EvaluationEvidenceItem(evidence_id="paired_signal_facts", kind="other",
        label="Measured PCM facts on execution-declared source/output intervals", payload=payload),)
