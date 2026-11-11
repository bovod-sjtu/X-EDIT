"""Exact decoded-audio coordinates, independent of millisecond display units."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydub import AudioSegment
from pydub.utils import audioop


@dataclass(frozen=True)
class AudioFrameGeometry:
    """Read-only geometry of one decoded asset revision.

    Seconds map to the nearest frame (Python's round-to-even for exact ties).
    Absolute intervals must already lie within the asset; quantization does not
    authorize clipping, padding, or relocating an out-of-bounds request.
    """

    sample_rate: int
    frame_count: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_count < 0:
            raise ValueError("audio geometry requires a positive rate and non-negative frames")

    @classmethod
    def from_segment(cls, audio: AudioSegment) -> AudioFrameGeometry:
        return cls(audio.frame_rate, len(audio.raw_data) // audio.frame_width)

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.sample_rate

    def interval(self, start_sec: float, end_sec: float) -> tuple[int, int]:
        """Validate an absolute half-open interval and return frame coordinates."""
        start, end = float(start_sec), self.canonical_end_sec(float(end_sec))
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("audio_interval_nonfinite: endpoints must be finite")
        # Arithmetic on the same exact frame boundary can differ by a few ULPs.
        # This allowance is numerical round-off, never a sample-sized tolerance.
        tolerance = 8 * math.ulp(self.duration_sec)
        if start < -tolerance or end > self.duration_sec + tolerance:
            raise ValueError(
                "audio_interval_out_of_bounds: absolute interval "
                f"[{start}, {end}) exceeds [0, {self.duration_sec})"
            )
        if end <= start:
            raise ValueError("audio_interval_empty: end must be greater than start")
        first, last = round(start * self.sample_rate), round(end * self.sample_rate)
        if not 0 <= first < last <= self.frame_count:
            raise ValueError("audio_interval_empty_or_out_of_bounds: no valid frame interval")
        return first, last

    def canonical_end_sec(self, end_sec: float) -> float:
        """Resolve only a decimal-rounded EOF, never an arbitrary near-boundary edit.

        Public audio reports use millisecond or finer seconds. If the proposed
        end equals the actual EOF rounded to 3--9 decimal places, map that
        representation to the exact final frame. The maximum adjustment is
        half a millisecond. Interior endpoints and all other overruns retain
        their original value and undergo the usual strict bounds check.
        """
        end = float(end_sec)
        if self.duration_sec < end <= self.duration_sec + 0.0005:
            if any(end == round(self.duration_sec, precision) for precision in range(3, 10)):
                return self.duration_sec
        return end


def slice_audio_frames(
    audio: AudioSegment, start_sec: float, end_sec: float
) -> AudioSegment:
    """Copy whole interleaved frames without adding silence at a rounded EOF."""
    first, last = AudioFrameGeometry.from_segment(audio).interval(start_sec, end_sec)
    return audio.get_sample_slice(start_sample=first, end_sample=last)


def fit_audio_frames(audio: AudioSegment, frame_count: int) -> AudioSegment:
    """Explicitly trim or zero-pad generated replacement audio to a frame count."""
    if frame_count < 0:
        raise ValueError("replacement frame count must be non-negative")
    size = frame_count * audio.frame_width
    data = audio.raw_data[:size]
    return audio._spawn(data + bytes(size - len(data)))


def replace_audio_frames(
    source: AudioSegment,
    replacement: AudioSegment,
    start_sec: float,
    end_sec: float,
) -> AudioSegment:
    """Replace an exact interval; mismatched format or length is an error.

    Generated material must be fitted explicitly before this call. The helper
    never changes the format, quantizes samples, or pads the source's EOF.
    """
    first, last = AudioFrameGeometry.from_segment(source).interval(start_sec, end_sec)
    if (source.frame_rate, source.channels, source.sample_width) != (
        replacement.frame_rate,
        replacement.channels,
        replacement.sample_width,
    ):
        raise ValueError("frame replacement requires matching decoded formats")
    if AudioFrameGeometry.from_segment(replacement).frame_count != last - first:
        raise ValueError("replacement must contain exactly the declared interval's frames")
    return (
        source.get_sample_slice(end_sample=first)
        + replacement
        + source.get_sample_slice(start_sample=last)
    )


def append_audio_frames(
    left: AudioSegment, right: AudioSegment, *, crossfade_ms: int
) -> AudioSegment:
    """Join matching-format regions with exact untouched prefixes and suffixes.

    The existing crossfade is linear in amplitude between 0 and -120 dB;
    fades above 100 ms use its existing millisecond gain steps. Only the
    declared overlap is mixed. Frame coordinates prevent rounded display
    durations from removing samples outside that overlap.
    """
    if (left.frame_rate, left.channels, left.sample_width) != (
        right.frame_rate, right.channels, right.sample_width
    ):
        raise ValueError("frame crossfade requires matching decoded formats")
    left_frames = AudioFrameGeometry.from_segment(left).frame_count
    right_frames = AudioFrameGeometry.from_segment(right).frame_count
    overlap = min(round(crossfade_ms * left.frame_rate / 1000), left_frames, right_frames)
    if overlap <= 0:
        return left + right
    duration_ms = overlap * 1000 / left.frame_rate
    tail = left.get_sample_slice(start_sample=left_frames - overlap)
    head = right.get_sample_slice(end_sample=overlap)
    mixed = []
    quiet = 10 ** (-120 / 20)
    if duration_ms > 100:
        blocks = (
            (int(ms * left.frame_rate / 1000),
             min(overlap, int((ms + 1) * left.frame_rate / 1000)), ms / duration_ms)
            for ms in range(math.ceil(duration_ms))
        )
    else:
        blocks = ((frame, frame + 1, frame / overlap) for frame in range(overlap))
    for start, end, position in blocks:
        fade_out = 1 + (quiet - 1) * position
        fade_in = quiet + (1 - quiet) * position
        first, last = start * left.frame_width, end * left.frame_width
        mixed.append(audioop.add(
            audioop.mul(tail.raw_data[first:last], left.sample_width, fade_out),
            audioop.mul(head.raw_data[first:last], left.sample_width, fade_in),
            left.sample_width,
        ))
    blend = AudioSegment(
        data=b"".join(mixed), sample_width=left.sample_width,
        frame_rate=left.frame_rate, channels=left.channels,
    )
    return (
        left.get_sample_slice(end_sample=left_frames - overlap)
        + blend
        + right.get_sample_slice(start_sample=overlap)
    )
