"""Validation for coordinates explicitly authored on an execution input.

This names a coordinate system, not acoustic evidence or an exemption from
grounding/admission. Unannotated spans retain the observed-audio mapping policy.
"""
from __future__ import annotations

import soundfile as sf

from x_editor.core.schemas import AudioEditStep
from x_editor.runtime.audio_frame_geometry import AudioFrameGeometry


def validate_input_coordinates(step: AudioEditStep, input_path: str, groups: list[list]) -> None:
    if not any(step.inputs.get(key) is not None for key in ("input_audio", "input_audio_path", "audio_path")):
        raise ValueError("step_input coordinates require an explicitly bound input audio")
    info = sf.info(input_path)
    geometry = AudioFrameGeometry(info.samplerate, info.frames)
    for spans in groups:
        for span in spans:
            if isinstance(span, dict):
                start = span.get("start_sec", span.get("start"))
                end = span.get("end_sec", span.get("end"))
            elif isinstance(span, (list, tuple)) and len(span) == 2:
                start, end = span
            else:
                raise ValueError("step_input coordinates require explicit numeric span endpoints")
            if start is None or end is None:
                raise ValueError("step_input coordinates require both endpoints")
            geometry.interval(start, end)
