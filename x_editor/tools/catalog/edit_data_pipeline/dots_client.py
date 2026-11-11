"""Original HTTP adapter; the dots.tts implementation is installed separately."""
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from x_editor.tools.catalog._http_retry import post_with_retries
from x_editor.tools.edit_backend import dots_service_url

SAMPLING_KEYS = ("num_steps", "guidance_scale", "use_xvector", "speaker_scale", "seed")
PIPELINE_PARAMETERS = frozenset({
    "aligner_backend", "duration_mode", "synthesis_mode", "expand_radius", "merge_gap",
    "nfe_step", "cfg_strength", "sway_sampling_coef", "source_crossfade_sec",
    "regen_crossfade_sec", "disable_prepend_context", "fail_if_cannot_prepend_context",
    "min_unmasked_context_sec", "min_unmasked_context_ratio", "max_f5_edit_seconds",
    "pitch_method", "rate_method", "psola_indextts2_cco_pitch_boost", "crossfade_ms",
    "smooth_policy", "smooth_margin_sec", "synth_method", "stage2_method",
    "emotion_condition_mode", "evaluate", "persist", "output_dir", "debug", "debug_dir",
    "debug_output_dir", "instruction_nl", "instructions_nl", "reverse_instructions_nl",
})


def apply_dots_edit(operation: str, arguments: dict[str, Any], output_path: str,
                    timeout: float) -> dict[str, Any]:
    from x_editor.tools.catalog.edit_data_pipeline.client import (
        _canonicalize_emotion_instruction_st_for_service,
        _encode_audio_file,
        _required_audio_path,
        _required_str,
    )

    instruction = _required_str(arguments, "instruction_st")
    if operation == "emotion_style_edit":
        instruction = _canonicalize_emotion_instruction_st_for_service(instruction)
    payload = {
        "operation": operation,
        "audio": _encode_audio_file(_required_audio_path(arguments, "input_audio_path")),
        "source_text": _required_str(arguments, "source_text"),
        "instruction_st": instruction,
    }
    for key in (*SAMPLING_KEYS, "target_text"):
        if arguments.get(key) is not None:
            payload[key] = arguments[key]
    url = dots_service_url(arguments.get("service_url"))
    response = post_with_retries(
        url + "/execute", json=payload, timeout=timeout,
        service_name="dots.tts.edit", retry_env="X_EDITOR_DOTS_RETRIES",
        backoff_env="X_EDITOR_DOTS_RETRY_BACKOFF", default_retries=0,
        retry_statuses=(),
    )
    response.raise_for_status()
    data = response.json()
    if (data.get("backend") != "dots_tts_edit" or data.get("status") != "ok"
            or data.get("operation") != operation):
        raise RuntimeError("dots.tts.edit response has no successful backend identity")
    try:
        audio_bytes = base64.b64decode(data["audio"]["data"], validate=True)
        samples, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    except (KeyError, TypeError, ValueError, sf.LibsndfileError) as error:
        raise RuntimeError("dots.tts.edit returned invalid audio") from error
    if not samples.size or not np.isfinite(samples).all() or sample_rate <= 0:
        raise RuntimeError("dots.tts.edit returned empty or nonfinite audio")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".wav", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(audio_bytes)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "output_audio_path": str(output), "operation": operation, "backend": "dots_tts_edit",
        "service_url": url, "sample_rate": sample_rate, "channels": samples.shape[1],
        "duration_seconds": samples.shape[0] / sample_rate,
        "source_text": data.get("source_text"), "target_text": data.get("target_text"),
        "instruction_st": instruction, "metadata": data.get("metadata", {}),
        "ignored_pipeline_parameters": sorted(PIPELINE_PARAMETERS.intersection(arguments)),
    }
