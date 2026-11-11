"""Spatial room reverb runtime backed by optional gpuRIR simulation.

The public tool exposes room-level controls that planners can safely reason
about. When gpuRIR is installed with CUDA support, RIRs are generated through
gpuRIR. Otherwise the runtime falls back to a deterministic synthetic RIR so
tests and CPU-only environments remain runnable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


SPEED_OF_SOUND_M_PER_SEC = 343.0


@dataclass(frozen=True)
class RoomPreset:
    room_size_m: tuple[float, float, float]
    t60: float
    dry_wet: float
    source_distance_m: float
    description: str


ROOM_PRESETS: dict[str, RoomPreset] = {
    "dry_studio": RoomPreset((4.0, 3.0, 2.6), 0.18, 0.10, 1.2, "dry recording studio"),
    "small_room": RoomPreset((4.5, 3.5, 2.7), 0.35, 0.18, 1.5, "small furnished room"),
    "medium_room": RoomPreset((7.0, 5.5, 3.2), 0.65, 0.26, 2.2, "medium room"),
    "large_empty_room": RoomPreset((13.0, 9.0, 5.0), 1.75, 0.42, 4.0, "large empty room"),
    "hall": RoomPreset((22.0, 15.0, 8.0), 2.45, 0.48, 5.5, "large hall"),
    "cathedral": RoomPreset((38.0, 24.0, 15.0), 4.20, 0.58, 7.0, "cathedral-like reverberant space"),
    "bathroom": RoomPreset((3.2, 2.4, 2.6), 1.05, 0.35, 1.1, "small reflective tiled bathroom"),
}


def apply_spatial_reverb(
    *,
    input_audio_path: str,
    output_path: str,
    instruction: str = "",
    room_preset: str | None = None,
    room_size_m: list[float] | tuple[float, float, float] | None = None,
    t60: float | None = None,
    dry_wet: float | None = None,
    source_distance_m: float | None = None,
    receiver_position_m: list[float] | tuple[float, float, float] | None = None,
    source_position_m: list[float] | tuple[float, float, float] | None = None,
    rir_length_sec: float | None = None,
    preserve_duration: bool = False,
    backend: str = "auto",
    wet_gain_db: float = 0.0,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Convolve audio with a room impulse response and write a WAV output."""

    input_path = Path(input_audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"spatial_reverb input audio does not exist: {input_audio_path}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(output_dir) if output_dir else output.parent / f"{output.stem}_spatial_reverb"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = sf.read(input_path, always_2d=True)
    audio = audio.astype(np.float32, copy=False)
    params = infer_spatial_reverb_parameters(
        instruction=instruction,
        sample_rate=sample_rate,
        room_preset=room_preset,
        room_size_m=room_size_m,
        t60=t60,
        dry_wet=dry_wet,
        source_distance_m=source_distance_m,
        receiver_position_m=receiver_position_m,
        source_position_m=source_position_m,
        rir_length_sec=rir_length_sec,
    )
    rir, backend_used, backend_error = generate_room_impulse_response(params, backend=backend)
    processed, clipping_scale = _apply_rir(
        audio,
        rir,
        dry_wet=float(params["dry_wet"]),
        preserve_duration=preserve_duration,
        wet_gain_db=wet_gain_db,
    )
    if processed.shape[1] == 1:
        write_data = processed[:, 0]
    else:
        write_data = processed
    sf.write(output, write_data, sample_rate, subtype="PCM_16")

    rir_path = work_dir / "room_impulse_response.npy"
    np.save(rir_path, rir)
    manifest = {
        "operation": "spatial_reverb",
        "input_audio_path": str(input_path),
        "output_audio_path": str(output),
        "rir_path": str(rir_path),
        "backend": backend_used,
        "backend_error": backend_error,
        "room_parameters": params,
        "preserve_duration": preserve_duration,
        "wet_gain_db": wet_gain_db,
        "input_duration_sec": round(audio.shape[0] / sample_rate, 6),
        "output_duration_sec": round(processed.shape[0] / sample_rate, 6),
        "clipping_scale": clipping_scale,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def infer_spatial_reverb_parameters(
    *,
    instruction: str,
    sample_rate: int,
    room_preset: str | None = None,
    room_size_m: list[float] | tuple[float, float, float] | None = None,
    t60: float | None = None,
    dry_wet: float | None = None,
    source_distance_m: float | None = None,
    receiver_position_m: list[float] | tuple[float, float, float] | None = None,
    source_position_m: list[float] | tuple[float, float, float] | None = None,
    rir_length_sec: float | None = None,
) -> dict[str, Any]:
    """Map a natural-language spatial target to concrete RIR parameters."""

    preset_name = _normalize_preset(room_preset) or infer_room_preset(instruction)
    preset = ROOM_PRESETS[preset_name]
    size = _vector3(room_size_m, preset.room_size_m, min_value=1.0, max_value=80.0)
    resolved_t60 = _clamp(float(t60 if t60 is not None else preset.t60), 0.08, 8.0)
    resolved_dry_wet = _clamp(float(dry_wet if dry_wet is not None else preset.dry_wet), 0.0, 0.95)
    distance = _clamp(
        float(source_distance_m if source_distance_m is not None else _distance_from_instruction(instruction, preset)),
        0.4,
        max(0.5, min(size[0], size[1]) - 0.5),
    )
    receiver = _vector3(
        receiver_position_m,
        (size[0] * 0.45, size[1] * 0.50, min(1.6, size[2] - 0.4)),
        min_value=0.2,
        max_value=100.0,
    )
    if source_position_m is None:
        source = (
            _clamp(receiver[0] + distance, 0.3, size[0] - 0.3),
            _clamp(receiver[1] + _azimuth_offset(instruction), 0.3, size[1] - 0.3),
            _clamp(receiver[2], 0.3, size[2] - 0.3),
        )
    else:
        source = _vector3(source_position_m, receiver, min_value=0.2, max_value=100.0)
    actual_distance = math.dist(source, receiver)
    length = _clamp(float(rir_length_sec if rir_length_sec is not None else max(0.30, resolved_t60 * 1.15)), 0.12, 8.0)
    return {
        "room_preset": preset_name,
        "room_description": preset.description,
        "room_size_m": [round(value, 4) for value in size],
        "t60": round(resolved_t60, 4),
        "dry_wet": round(resolved_dry_wet, 4),
        "sample_rate": int(sample_rate),
        "rir_length_sec": round(length, 4),
        "source_position_m": [round(value, 4) for value in source],
        "receiver_position_m": [round(value, 4) for value in receiver],
        "source_distance_m": round(actual_distance, 4),
    }


def infer_room_preset(instruction: str) -> str:
    text = instruction.lower()
    if _contains(text, ["cathedral", "church", "教堂", "大教堂"]):
        return "cathedral"
    if _contains(text, ["hall", "大厅", "礼堂", "会堂", "auditorium", "gym"]):
        return "hall"
    if _contains(text, ["bathroom", "toilet", "shower", "tile", "瓷砖", "浴室", "卫生间"]):
        return "bathroom"
    if _contains(text, ["large empty room", "big empty room", "empty room", "空荡", "空旷", "大房间", "很大的房间"]):
        return "large_empty_room"
    if _contains(text, ["small room", "小房间", "卧室", "bedroom"]):
        return "small_room"
    if _contains(text, ["studio", "dry", "录音棚", "干声"]):
        return "dry_studio"
    return "medium_room"


def generate_room_impulse_response(params: dict[str, Any], *, backend: str = "auto") -> tuple[np.ndarray, str, str | None]:
    backend = (backend or "auto").lower()
    if backend not in {"auto", "gpurir", "synthetic"}:
        raise ValueError("spatial_reverb backend must be one of: auto, gpuRIR, synthetic")
    if backend == "auto" and not _auto_gpurir_enabled():
        return (
            _generate_synthetic_rir(params),
            "synthetic",
            "gpuRIR auto backend disabled; set X_EDITOR_SPATIAL_REVERB_AUTO_GPURIR=1 to enable",
        )
    if backend in {"auto", "gpurir"}:
        try:
            return _generate_gpurir(params), "gpuRIR", None
        except Exception as exc:
            if backend == "gpurir":
                raise
            return _generate_synthetic_rir(params), "synthetic", f"{type(exc).__name__}: {exc}"
    return _generate_synthetic_rir(params), "synthetic", None


def _auto_gpurir_enabled() -> bool:
    value = os.getenv("X_EDITOR_SPATIAL_REVERB_AUTO_GPURIR", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _generate_gpurir(params: dict[str, Any]) -> np.ndarray:
    import gpuRIR  # type: ignore[import-not-found]

    room_sz = np.asarray(params["room_size_m"], dtype=np.float32)
    t60 = float(params["t60"])
    beta = gpuRIR.beta_SabineEstimation(room_sz, t60)
    nb_img = gpuRIR.t2n(float(params["rir_length_sec"]), room_sz)
    pos_src = np.asarray([params["source_position_m"]], dtype=np.float32)
    pos_rcv = np.asarray([params["receiver_position_m"]], dtype=np.float32)
    rirs = gpuRIR.simulateRIR(
        room_sz,
        beta,
        pos_src,
        pos_rcv,
        nb_img,
        float(params["rir_length_sec"]),
        int(params["sample_rate"]),
    )
    rir = np.asarray(rirs[0, 0], dtype=np.float32)
    return _normalize_rir(rir)


def _generate_synthetic_rir(params: dict[str, Any]) -> np.ndarray:
    sample_rate = int(params["sample_rate"])
    length = max(16, int(round(float(params["rir_length_sec"]) * sample_rate)))
    distance = max(0.35, float(params["source_distance_m"]))
    t60 = max(0.08, float(params["t60"]))
    room_size = np.asarray(params["room_size_m"], dtype=np.float32)
    seed_material = json.dumps(params, sort_keys=True).encode("utf-8")
    seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    rir = np.zeros(length, dtype=np.float32)
    direct = min(length - 1, max(0, int(round(distance / SPEED_OF_SOUND_M_PER_SEC * sample_rate))))
    rir[direct] = 1.0 / max(distance, 1.0)

    reflection_count = int(_clamp(room_size.mean() * 2.5 + t60 * 18.0, 8, 120))
    early_end = min(length - 1, direct + int(round(min(0.24, max(0.045, t60 * 0.16)) * sample_rate)))
    if direct + 1 < length:
        high = min(length, max(direct + 2, early_end + 1))
        for _ in range(reflection_count):
            idx = int(rng.integers(direct + 1, high))
            time_sec = idx / sample_rate
            decay = math.exp(-6.9078 * time_sec / t60)
            rir[idx] += float(rng.choice([-1.0, 1.0]) * decay * rng.uniform(0.035, 0.18))

    tail_start = min(length - 1, direct + int(round(0.035 * sample_rate)))
    if tail_start < length:
        t = np.arange(length - tail_start, dtype=np.float32) / sample_rate
        envelope = np.exp(-6.9078 * t / t60).astype(np.float32)
        tail = rng.standard_normal(length - tail_start).astype(np.float32)
        tail = signal.lfilter([0.18], [1.0, -0.82], tail).astype(np.float32)
        rir[tail_start:] += tail * envelope * 0.08
    return _normalize_rir(rir)


def _apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    *,
    dry_wet: float,
    preserve_duration: bool,
    wet_gain_db: float,
) -> tuple[np.ndarray, float | None]:
    dry_len, channels = audio.shape
    wet_len = dry_len if preserve_duration else dry_len + len(rir) - 1
    output = np.zeros((wet_len, channels), dtype=np.float32)
    dry_padded = np.zeros_like(output)
    dry_padded[:dry_len, :] = audio
    wet_gain = 10.0 ** (wet_gain_db / 20.0)

    for channel in range(channels):
        wet = signal.fftconvolve(audio[:, channel], rir, mode="full").astype(np.float32)
        wet = wet[:wet_len]
        dry_rms = _rms(audio[:, channel])
        wet_rms = _rms(wet)
        if wet_rms > 1e-8 and dry_rms > 1e-8:
            wet *= dry_rms / wet_rms
        wet *= wet_gain
        output[:, channel] = (1.0 - dry_wet) * dry_padded[:, channel] + dry_wet * wet

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.98:
        scale = 0.98 / peak
        output *= scale
        return output, scale
    return output, None


def _normalize_rir(rir: np.ndarray) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim != 1:
        rir = rir.reshape(-1)
    peak = float(np.max(np.abs(rir))) if rir.size else 0.0
    if peak > 1e-8:
        rir = rir / peak
    return rir


def _distance_from_instruction(instruction: str, preset: RoomPreset) -> float:
    text = instruction.lower()
    if _contains(text, ["far", "distant", "远处", "很远", "远一点"]):
        return preset.source_distance_m * 1.45
    if _contains(text, ["close", "near", "近处", "靠近", "贴近"]):
        return max(0.7, preset.source_distance_m * 0.55)
    return preset.source_distance_m


def _azimuth_offset(instruction: str) -> float:
    text = instruction.lower()
    if _contains(text, ["left", "左边", "左侧"]):
        return 0.8
    if _contains(text, ["right", "右边", "右侧"]):
        return -0.8
    return 0.0


def _normalize_preset(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "large_room": "large_empty_room",
        "empty_room": "large_empty_room",
        "big_empty_room": "large_empty_room",
        "large_hall": "hall",
        "big_hall": "hall",
        "auditorium": "hall",
        "church": "cathedral",
        "tile_room": "bathroom",
        "tiled_room": "bathroom",
        "studio": "dry_studio",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ROOM_PRESETS:
        raise ValueError(f"Unknown spatial reverb room_preset: {value}")
    return normalized


def _vector3(
    value: list[float] | tuple[float, float, float] | None,
    default: tuple[float, float, float],
    *,
    min_value: float,
    max_value: float,
) -> tuple[float, float, float]:
    raw = default if value is None else value
    if len(raw) != 3:
        raise ValueError("spatial reverb vector parameters must have exactly 3 numbers")
    return tuple(_clamp(float(item), min_value, max_value) for item in raw)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rms(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    value64 = value.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(value64 * value64)))


def _contains(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)
