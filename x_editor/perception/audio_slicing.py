"""Small audio slicing helper used by perception subagents."""

from __future__ import annotations

import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any


class SliceAudioError(RuntimeError):
    """Raised when a requested diagnostic audio slice cannot be produced."""


def audio_duration_sec(audio_path: str | Path) -> float | None:
    """Best-effort audio duration in seconds."""

    path = Path(audio_path).expanduser().resolve()
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate) if rate else None
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        value = completed.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def slice_audio(
    audio_path: str | Path,
    *,
    onset: float,
    offset: float,
    output_dir: str | Path,
    reason: str,
    clip_id: str | None = None,
    padding_sec: float = 0.0,
) -> dict[str, Any]:
    """Cut a diagnostic slice with ffmpeg and preserve the reason in metadata."""

    source = Path(audio_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"slice_audio source does not exist: {source}")
    if not str(reason or "").strip():
        raise SliceAudioError("slice_audio requires a non-empty reason for auditability")

    duration = audio_duration_sec(source)
    start = max(0.0, float(onset) - max(0.0, float(padding_sec)))
    end = float(offset) + max(0.0, float(padding_sec))
    if duration is not None:
        end = min(float(duration), end)
    if end <= start:
        raise SliceAudioError(f"slice_audio offset must be greater than onset: {onset}-{offset}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_clip_id(clip_id or f"{start:.2f}_{end:.2f}")
    output_path = out_dir / f"{safe_id}.wav"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode == 0 and output_path.exists():
            return _slice_record(source, output_path, start, end, reason)

    if source.suffix.lower() == ".wav":
        _slice_wav_without_ffmpeg(source, output_path, start, end)
        return _slice_record(source, output_path, start, end, reason)

    raise SliceAudioError("ffmpeg is required to slice non-WAV audio")


def _slice_record(source: Path, output_path: Path, start: float, end: float, reason: str) -> dict[str, Any]:
    return {
        "tool": "slice_audio",
        "input_audio_path": str(source),
        "output_audio_path": str(output_path),
        "onset": round(start, 6),
        "offset": round(end, 6),
        "duration_sec": round(end - start, 6),
        "reason": str(reason).strip(),
    }


def _slice_wav_without_ffmpeg(source: Path, output_path: Path, start: float, end: float) -> None:
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        rate = src.getframerate()
        sampwidth = src.getsampwidth()
        channels = src.getnchannels()
        start_frame = max(0, int(round(start * rate)))
        end_frame = max(start_frame, int(round(end * rate)))
        src.setpos(min(start_frame, src.getnframes()))
        frames = src.readframes(max(0, min(end_frame, src.getnframes()) - start_frame))
    with wave.open(str(output_path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames or (b"\x00" * sampwidth * channels))


def _safe_clip_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe[:80] or "slice"
