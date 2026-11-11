"""Read-only signal checks for relative gain; no rendering or provider calls."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from x_editor.core.schemas import AudioEditStep
from x_editor.runtime.relative_gain import gain_frame_intervals, validate_gain_params


def relative_gain_checks(
    step: AudioEditStep, result: dict[str, Any], state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recompute the contract from the admitted step and inspect decoded samples."""
    def check(suffix: str, passed: bool, summary: str) -> dict[str, Any]:
        return {
            "check_id": f"artifact_ops.apply_gain.{suffix}",
            "outcome": "passed" if passed else "failed",
            "summary": summary,
        }

    contract = result.get("relative_gain_contract")
    if not isinstance(contract, dict):
        return [check("contract", False, "Missing executor-authored relative gain contract.")]
    try:
        factor, peak_limit = validate_gain_params(step.params)
        source_path = Path(contract["input_audio_path"])
        source, source_rate = sf.read(source_path, dtype="float64", always_2d=True)
        output, output_rate = sf.read(result["output_audio_path"], dtype="float64", always_2d=True)
        intervals = gain_frame_intervals(step.params, frames=len(source), sample_rate=source_rate)
        from x_editor.runtime.verification import StepVerificationContext

        context = state.get("step_verification_context")
        bound_path = None
        if isinstance(context, StepVerificationContext):
            bound_path = context.primary_input_artifact.path if context.primary_input_artifact else None
            if bound_path is None:
                return [check("contract", False, "No authoritative primary input artifact is bound.")]
        else:
            authored = step.inputs.get("input_audio") or step.inputs.get("input_audio_path")
            if isinstance(authored, str) and Path(authored).is_file():
                bound_path = authored
        contract_ok = (
            (bound_path is None or source_path.resolve() == Path(bound_path).resolve())
            and result.get("input_audio_path") == str(source_path)
            and contract.get("gain_db") == step.params["gain_db"]
            and contract.get("headroom_db") == step.params.get("headroom_db", 0.0)
            and contract.get("linear_gain") == factor
            and contract.get("input_sha256") == hashlib.sha256(source_path.read_bytes()).hexdigest()
            and contract.get("frame_intervals") == [list(item) for item in intervals]
            and contract.get("frame_count") == len(source)
            and contract.get("sample_rate") == source_rate
            and contract.get("channels") == source.shape[1]
        )
        geometry_ok = source.shape == output.shape and source_rate == output_rate
        checks = [
            check("contract", contract_ok, "Recorded gain, scope and source identity match the admitted step."),
            check("timeline", geometry_ok, "Output preserves exact frame count, channels and sample rate."),
        ]
        if not geometry_ok:
            return checks
        selected = np.zeros(len(source), dtype=bool)
        for lo, hi in intervals:
            selected[lo:hi] = True
        signal_ok = (
            np.isfinite(source).all()
            and np.isfinite(output).all()
            and np.array_equal(output[selected], source[selected] * factor)
            and float(np.max(np.abs(output[selected]))) <= peak_limit
        )
        checks.extend([
            check("signal", bool(signal_ok), "Every selected sample has the exact requested linear gain within headroom."),
            check("preservation", np.array_equal(output[~selected], source[~selected]), "Every unselected decoded sample is unchanged."),
        ])
        return checks
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return [check("contract", False, f"Relative gain evidence cannot be verified: {type(exc).__name__}: {exc}")]
