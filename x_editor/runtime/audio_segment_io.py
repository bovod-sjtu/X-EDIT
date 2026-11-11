"""Explicit decoding and frame-preserving operations at the PCM boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment


def load_audio_segment(path: str | Path) -> AudioSegment:
    """Preserve the legacy integer decoder; quantize float audio to PCM32.

    Pydub's WAV fast path can treat IEEE float bytes as integer PCM, and its
    arithmetic does not support 64-bit samples. Float inputs therefore need an
    explicit decode. This conversion preserves frames/rate/channels and has at
    most one PCM32 LSB error; it never normalizes or clips out-of-range audio.
    """
    try:
        info = sf.info(path)
    except (OSError, RuntimeError):
        # Other containers retain the existing ffmpeg-backed decoder/error path.
        return AudioSegment.from_file(path)
    if info.subtype not in {"FLOAT", "DOUBLE"}:
        return AudioSegment.from_file(path)
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError("floating_audio_nonfinite: PCM editing requires finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("floating_audio_out_of_range: PCM editing cannot preserve samples outside [-1, 1]")
    # +1 is the only endpoint not representable by signed PCM32. Its conversion
    # to the largest positive integer is quantization, bounded by one LSB.
    pcm = np.minimum(np.rint(samples * (2**31)), 2**31 - 1).astype("<i4")
    return AudioSegment(
        data=pcm.tobytes(), sample_width=4, frame_rate=rate, channels=info.channels,
    )


def replace_fixed_audio_interval(
    source: AudioSegment,
    replacement: AudioSegment,
    *,
    start_sec: float,
    end_sec: float,
) -> AudioSegment:
    """Trim or zero-pad a replacement after conversion to the source PCM format.

    A resampler can return one frame less than its nominal duration. Fitting
    before resampling shifts the untouched suffix. Keep the existing trim/pad
    policy, but apply it to exact frame counts in the destination format.
    """
    start = round(start_sec * source.frame_rate)
    end = round(end_sec * source.frame_rate)
    if not 0 <= start <= end <= source.frame_count():
        raise ValueError("replacement interval lies outside the source audio")
    converted = (
        replacement.set_frame_rate(source.frame_rate)
        .set_sample_width(source.sample_width)
    )
    if converted.channels != source.channels:
        if converted.channels != 1 and source.channels != 1:
            converted = converted.set_channels(1)
        converted = converted.set_channels(source.channels)
    target_bytes = (end - start) * source.frame_width
    data = converted.raw_data[:target_bytes]
    data += bytes(target_bytes - len(data))
    fitted = AudioSegment(
        data=data,
        sample_width=source.sample_width,
        frame_rate=source.frame_rate,
        channels=source.channels,
    )
    return source.get_sample_slice(end_sample=start) + fitted + source.get_sample_slice(start_sample=end)
