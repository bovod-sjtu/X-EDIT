"""Deterministic speech-preserving acoustic effects.

This module intentionally stays in the "DSP tool" lane. It applies channel,
filtering, distortion, and noise-bed effects while preserving the input speech
timeline. It is not a voice-conversion, enhancement, or text-editing model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


AUDIO_EFFECT_PRESETS = {
    "telephone",
    "old_radio",
    "underwater",
    "strong_wind",
    "crowd_noise",
    "background_music",
    "muffled_wall",
    "whisper_style",
    "shouting_style",
    "animal_vocal_style",
    "match_channel_reference",
}


def apply_audio_effect(
    *,
    input_audio_path: str,
    output_path: str,
    instruction: str = "",
    preset: str | None = None,
    effect: str | None = None,
    strength: float = 0.75,
    reference_audio_path: str | None = None,
    preserve_duration: bool = True,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Apply one deterministic acoustic effect preset to an audio file."""

    input_path = Path(input_audio_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"audio_effects input audio does not exist: {input_path}")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    selected = canonical_audio_effect_preset(preset or effect or instruction)
    if not selected:
        raise ValueError(
            "Could not resolve audio effect preset. Expected one of: "
            + ", ".join(sorted(AUDIO_EFFECT_PRESETS))
        )

    strength = max(0.0, min(1.0, float(strength)))
    audio, sample_rate = _read_audio(input_path)
    original_length = audio.shape[0]
    span = _infer_effect_span(instruction, original_length, sample_rate)
    target = audio[span[0] : span[1]]
    processed = _apply_preset(
        target,
        sample_rate,
        selected,
        strength=strength,
        reference_audio_path=reference_audio_path,
    )
    channel_analysis = None
    if selected == "match_channel_reference" and reference_audio_path:
        channel_analysis = _channel_reference_analysis(target, sample_rate, reference_audio_path, strength=strength)
    result_audio = audio.copy()
    result_audio[span[0] : span[1]] = _crossfade_replace(
        result_audio[span[0] : span[1]],
        processed,
        sample_rate,
    )
    if preserve_duration:
        result_audio = result_audio[:original_length]
    result_audio = _safe_peak(result_audio)
    sf.write(output, result_audio, sample_rate, subtype="PCM_16")

    work_dir = Path(output_dir).expanduser().resolve() if output_dir else output.parent / f"{output.stem}_audio_effects"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "operation": "audio_effects",
        "backend": "scipy_dsp",
        "input_audio_path": str(input_path),
        "output_audio_path": str(output),
        "instruction": instruction,
        "preset": selected,
        "strength": strength,
        "reference_audio_path": reference_audio_path,
        "span": {
            "start_sec": span[0] / sample_rate,
            "end_sec": span[1] / sample_rate,
            "source": "instruction_span_heuristic",
        },
        "preserve_duration": preserve_duration,
    }
    if channel_analysis:
        manifest["reference_channel_analysis"] = channel_analysis
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def canonical_audio_effect_preset(text: str | None) -> str | None:
    """Map a user phrase or explicit value to a supported audio effect preset."""

    if not text:
        return None
    normalized = str(text).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    aliases = {
        "phone": "telephone",
        "phone_call": "telephone",
        "telephone_call": "telephone",
        "telephone_vocal": "telephone",
        "over_the_phone": "telephone",
        "narrowband": "telephone",
        "radio": "old_radio",
        "old_fashioned_radio": "old_radio",
        "old_radio_playback": "old_radio",
        "am_radio": "old_radio",
        "under_water": "underwater",
        "wind": "strong_wind",
        "windy": "strong_wind",
        "strong_wind_noise": "strong_wind",
        "outdoor": "strong_wind",
        "outside": "strong_wind",
        "outdoor_noise": "strong_wind",
        "outside_noise": "strong_wind",
        "outdoor_background_noise": "strong_wind",
        "outdoor_background_effect": "strong_wind",
        "outdoor_environmental_sound": "strong_wind",
        "outdoor_environment_sound": "strong_wind",
        "outside_background_noise": "strong_wind",
        "outdoor_ambience": "strong_wind",
        "outside_ambience": "strong_wind",
        "户外": "strong_wind",
        "户外背景声": "strong_wind",
        "户外背景音": "strong_wind",
        "户外噪声": "strong_wind",
        "户外环境声": "strong_wind",
        "户外环境音": "strong_wind",
        "室外背景声": "strong_wind",
        "crowd": "crowd_noise",
        "crowd_noise": "crowd_noise",
        "crowd_background_noise": "crowd_noise",
        "people_noise": "crowd_noise",
        "people_talking": "crowd_noise",
        "audience_noise": "crowd_noise",
        "room_crowd": "crowd_noise",
        "cafe_noise": "crowd_noise",
        "crowded_background": "crowd_noise",
        "crowded_background_noise": "crowd_noise",
        "crowded_ambience": "crowd_noise",
        "crowded_environment": "crowd_noise",
        "人群": "crowd_noise",
        "人群声": "crowd_noise",
        "人群背景声": "crowd_noise",
        "人群背景音": "crowd_noise",
        "嘈杂人声": "crowd_noise",
        "music": "background_music",
        "background_music": "background_music",
        "music_bed": "background_music",
        "soft_music": "background_music",
        "instrumental_music": "background_music",
        "背景音乐": "background_music",
        "音乐背景": "background_music",
        "muffled": "muffled_wall",
        "heavy_wall": "muffled_wall",
        "through_a_wall": "muffled_wall",
        "through_wall": "muffled_wall",
        "whisper": "whisper_style",
        "whispering": "whisper_style",
        "whispered": "whisper_style",
        "whisper_style": "whisper_style",
        "whispering_voice": "whisper_style",
        "shout": "shouting_style",
        "shouting": "shouting_style",
        "yell": "shouting_style",
        "yelling": "shouting_style",
        "shout_style": "shouting_style",
        "shouting_voice": "shouting_style",
        "animal": "animal_vocal_style",
        "animal_voice": "animal_vocal_style",
        "animal_vocal": "animal_vocal_style",
        "nonhuman": "animal_vocal_style",
        "non_human": "animal_vocal_style",
        "nonhuman_voice": "animal_vocal_style",
        "cat": "animal_vocal_style",
        "cat_voice": "animal_vocal_style",
        "cat_like": "animal_vocal_style",
        "dog": "animal_vocal_style",
        "dog_voice": "animal_vocal_style",
        "same_acoustic_channel": "match_channel_reference",
        "same_channel_condition": "match_channel_reference",
        "reference_channel": "match_channel_reference",
        "reference_channel_condition": "match_channel_reference",
        "channel_environment": "match_channel_reference",
        "recording_characteristics": "match_channel_reference",
        "channel_match": "match_channel_reference",
        "match_channel": "match_channel_reference",
        "match_reference": "match_channel_reference",
    }
    if normalized in AUDIO_EFFECT_PRESETS:
        return normalized
    if normalized in aliases:
        return aliases[normalized]
    lowered = str(text).lower()
    target_preset = _target_audio_effect_preset(lowered)
    if target_preset:
        return target_preset
    return _audio_effect_preset_anywhere(lowered)


def target_vocal_effort_effect_preset(text: str | None) -> str | None:
    """Resolve target whisper/shouting style without mistaking source descriptors for targets."""

    if not text:
        return None
    lowered = str(text).lower()
    target = _target_audio_effect_preset(lowered)
    if target in {"whisper_style", "shouting_style"}:
        return target
    has_conversion = bool(re.search(r"\b(?:from|into|to|change|turn|convert|transform)\b", lowered))
    if has_conversion:
        return None
    preset = _audio_effect_preset_anywhere(lowered)
    return preset if preset in {"whisper_style", "shouting_style"} else None


def _target_audio_effect_preset(lowered: str) -> str | None:
    """Prefer the target side of edit instructions when source and target effects both appear."""

    for pattern in [
        r"\b(?:change|turn|convert|transform)\b.{0,180}?\b(?:into|to)\b(?P<target>[^.;\n]{0,180})",
        r"\bmake\b.{0,120}?\bsound\b.{0,40}?\b(?:like|as if|as though)\b(?P<target>[^.;\n]{0,180})",
    ]:
        match = re.search(pattern, lowered)
        if not match:
            continue
        preset = _audio_effect_preset_anywhere(match.group("target"))
        if preset:
            return preset
    return None


def _audio_effect_preset_anywhere(lowered: str) -> str | None:
    if any(term in lowered for term in ["telephone", "phone call", "over the phone", "phone effect"]):
        return "telephone"
    if "radio" in lowered:
        return "old_radio"
    if (
        "strong wind" in lowered
        or "windy" in lowered
        or "outdoor background noise" in lowered
        or "outdoor background effect" in lowered
        or "outdoor environmental sound" in lowered
        or "outdoor environment sound" in lowered
        or "outside background noise" in lowered
        or "outdoor ambience" in lowered
        or "outside ambience" in lowered
        or "outdoor noise" in lowered
        or "outside noise" in lowered
        or "户外背景声" in lowered
        or "户外背景音" in lowered
        or "户外噪声" in lowered
        or "户外环境声" in lowered
        or "户外环境音" in lowered
        or "室外背景声" in lowered
    ):
        return "strong_wind"
    if any(
        term in lowered
        for term in [
            "crowd background noise",
            "crowd noise",
            "people background noise",
            "people talking",
            "audience noise",
            "room crowd",
            "cafe noise",
            "crowded background",
            "crowded ambience",
            "crowded environment",
            "人群",
            "人群声",
            "人群背景声",
            "人群背景音",
            "嘈杂人声",
        ]
    ):
        return "crowd_noise"
    if any(
        term in lowered
        for term in [
            "background music",
            "music bed",
            "soft music",
            "instrumental music",
            "add music",
            "背景音乐",
            "音乐背景",
        ]
    ):
        return "background_music"
    if "underwater" in lowered or "under water" in lowered:
        return "underwater"
    if "heavy wall" in lowered or "through a wall" in lowered or "muffled" in lowered:
        return "muffled_wall"
    if "whisper" in lowered or "whispering" in lowered or "whispered" in lowered or "低语" in lowered or "耳语" in lowered:
        return "whisper_style"
    if "shout" in lowered or "shouting" in lowered or "yell" in lowered or "yelling" in lowered or "大喊" in lowered or "喊叫" in lowered:
        return "shouting_style"
    if any(term in lowered for term in ["animal", "non-human", "nonhuman", "cat", "dog", "meow", "bark", "动物", "猫", "狗"]):
        return "animal_vocal_style"
    if (
        "same acoustic channel" in lowered
        or "same channel condition" in lowered
        or "reference channel" in lowered
        or "channel environment" in lowered
        or "recording characteristics" in lowered
        or "recording characteristic" in lowered
        or "recorded in the same acoustic" in lowered
    ):
        return "match_channel_reference"
    return None


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for {path}: {sample_rate}")
    if audio.size == 0:
        raise ValueError(f"Input audio is empty: {path}")
    return audio, sample_rate


def _infer_effect_span(instruction: str, length: int, sample_rate: int) -> tuple[int, int]:
    lowered = instruction.lower()
    if any(term in lowered for term in ["later part", "latter part", "second half", "后半", "后半段"]):
        return max(0, length // 2), length
    if any(term in lowered for term in ["early part", "first half", "前半", "前半段"]):
        return 0, max(1, length // 2)
    match = re.search(r"from\s+([0-9.]+)\s*(?:s|sec|second)", lowered)
    if match:
        start = min(length - 1, max(0, int(float(match.group(1)) * sample_rate)))
        return start, length
    return 0, length


def _apply_preset(
    audio: np.ndarray,
    sample_rate: int,
    preset: str,
    *,
    strength: float,
    reference_audio_path: str | None,
) -> np.ndarray:
    if preset == "telephone":
        return _telephone(audio, sample_rate, strength)
    if preset == "old_radio":
        return _old_radio(audio, sample_rate, strength)
    if preset == "underwater":
        return _underwater(audio, sample_rate, strength)
    if preset == "strong_wind":
        return _strong_wind(audio, sample_rate, strength)
    if preset == "crowd_noise":
        return _crowd_noise(audio, sample_rate, strength)
    if preset == "background_music":
        return _background_music(audio, sample_rate, strength)
    if preset == "muffled_wall":
        return _muffled_wall(audio, sample_rate, strength)
    if preset == "whisper_style":
        return _whisper_style(audio, sample_rate, strength)
    if preset == "shouting_style":
        return _shouting_style(audio, sample_rate, strength)
    if preset == "animal_vocal_style":
        return _animal_vocal_style(audio, sample_rate, strength)
    if preset == "match_channel_reference":
        if not reference_audio_path:
            raise ValueError("match_channel_reference requires reference_audio_path")
        return _match_channel_reference(audio, sample_rate, reference_audio_path, strength)
    raise ValueError(f"Unsupported audio effect preset: {preset}")


def _telephone(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    low = 280.0
    high = min(3400.0, sample_rate * 0.45)
    filtered = _bandpass(audio, sample_rate, low, high)
    clipped = _soft_clip(filtered, 1.2 + 2.2 * strength)
    noise = _colored_noise(audio.shape, sample_rate, highpass=500.0, lowpass=3600.0, seed=17)
    return _mix(audio, clipped + noise * (0.006 + 0.02 * strength), 0.82 + 0.15 * strength)


def _old_radio(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    filtered = _bandpass(audio, sample_rate, 150.0, min(4800.0, sample_rate * 0.45))
    distorted = _soft_clip(filtered, 1.6 + 3.0 * strength)
    static = _colored_noise(audio.shape, sample_rate, highpass=700.0, lowpass=min(7200.0, sample_rate * 0.45), seed=23)
    crackle = _impulse_crackle(audio.shape, seed=29) * (0.012 + 0.035 * strength)
    return _mix(audio, distorted + static * (0.008 + 0.025 * strength) + crackle, 0.7 + 0.25 * strength)


def _underwater(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    cutoff = 950.0 + (1.0 - strength) * 700.0
    filtered = _lowpass(audio, sample_rate, cutoff)
    delayed = _multi_tap_echo(filtered, sample_rate, [(0.035, 0.12), (0.083, 0.08), (0.141, 0.05)])
    wobble = _amplitude_wobble(delayed, sample_rate, depth=0.06 + 0.1 * strength, hz=1.1)
    bubbles = _colored_noise(audio.shape, sample_rate, highpass=80.0, lowpass=900.0, seed=31)
    return _mix(audio, wobble + bubbles * (0.006 + 0.018 * strength), 0.62 + 0.3 * strength)


def _strong_wind(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    speech = _bandpass(audio, sample_rate, 120.0, min(5200.0, sample_rate * 0.45))
    wind = _colored_noise(audio.shape, sample_rate, highpass=30.0, lowpass=650.0, seed=41)
    gust = _amplitude_wobble(wind, sample_rate, depth=0.55, hz=0.35)
    return _safe_peak(speech * (0.92 - 0.18 * strength) + gust * (0.06 + 0.18 * strength))


def _crowd_noise(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    speech = _bandpass(audio, sample_rate, 120.0, min(5600.0, sample_rate * 0.45))
    low_murmur = _colored_noise(audio.shape, sample_rate, highpass=120.0, lowpass=900.0, seed=43)
    mid_murmur = _colored_noise(audio.shape, sample_rate, highpass=450.0, lowpass=min(2600.0, sample_rate * 0.45), seed=47)
    room = _multi_tap_echo(speech, sample_rate, [(0.024, 0.035), (0.071, 0.028), (0.133, 0.018)])
    wobble = _amplitude_wobble(low_murmur * 0.65 + mid_murmur * 0.35, sample_rate, depth=0.28, hz=0.72)
    bed = _match_rms(wobble, speech) * (0.08 + 0.22 * strength)
    return _safe_peak(room * (0.94 - 0.12 * strength) + bed)


def _background_music(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    speech = _bandpass(audio, sample_rate, 110.0, min(5600.0, sample_rate * 0.45))
    n = audio.shape[0]
    channels = audio.shape[1]
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    freqs = np.array([196.0, 246.94, 293.66, 392.0], dtype=np.float32)
    pad = sum(np.sin(2.0 * math.pi * freq * t) for freq in freqs) / float(len(freqs))
    shimmer = np.sin(2.0 * math.pi * 0.18 * t) * 0.25 + 0.75
    bed = (pad * shimmer).astype(np.float32)[:, None]
    if channels > 1:
        bed = np.repeat(bed, channels, axis=1)
    bed = _lowpass(bed, sample_rate, min(2600.0, sample_rate * 0.45))
    bed = _match_rms(bed, speech) * (0.035 + 0.11 * strength)
    return _safe_peak(speech * (0.96 - 0.08 * strength) + bed)


def _muffled_wall(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    cutoff = 650.0 + (1.0 - strength) * 650.0
    muffled = _lowpass(audio, sample_rate, cutoff)
    muffled = _multi_tap_echo(muffled, sample_rate, [(0.018, 0.05), (0.052, 0.04)])
    return _mix(audio, muffled * (0.75 - 0.18 * strength), 0.75 + 0.2 * strength)


def _whisper_style(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    high = _bandpass(audio, sample_rate, 900.0, min(7600.0, sample_rate * 0.45))
    breath = _colored_noise(audio.shape, sample_rate, highpass=1100.0, lowpass=min(7800.0, sample_rate * 0.45), seed=53)
    envelope = _amplitude_envelope(audio, sample_rate)
    softened = high * (0.26 + 0.16 * (1.0 - strength)) + breath * envelope * (0.55 + 0.65 * strength)
    low_shadow = _lowpass(audio, sample_rate, 700.0) * (0.10 * (1.0 - strength))
    whispered = _match_rms(softened + low_shadow, audio) * (0.58 + 0.12 * (1.0 - strength))
    return _mix(audio, whispered, 0.68 + 0.25 * strength)


def _shouting_style(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    speech = _bandpass(audio, sample_rate, 120.0, min(5200.0, sample_rate * 0.45))
    driven = _soft_clip(speech * (1.25 + 0.75 * strength), 1.7 + 2.6 * strength)
    presence = _bandpass(audio, sample_rate, 1500.0, min(6200.0, sample_rate * 0.45))
    shouted = _safe_peak((driven + presence * (0.18 + 0.28 * strength)) * (1.08 + 0.30 * strength))
    return _mix(audio, shouted, 0.62 + 0.28 * strength)


def _animal_vocal_style(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    nasal = _bandpass(audio, sample_rate, 650.0, min(4800.0, sample_rate * 0.45))
    bright = _highpass(audio, sample_rate, min(1800.0, sample_rate * 0.35))
    chirp = _amplitude_wobble(nasal + bright * (0.18 + 0.22 * strength), sample_rate, depth=0.16 + 0.18 * strength, hz=6.2)
    driven = _soft_clip(chirp * (1.35 + 0.9 * strength), 2.0 + 2.5 * strength)
    squeak = _bandpass(audio, sample_rate, 2200.0, min(7600.0, sample_rate * 0.45)) * (0.18 + 0.32 * strength)
    styled = _safe_peak(driven + squeak)
    return _mix(audio, styled, 0.52 + 0.34 * strength)


def _match_channel_reference(
    audio: np.ndarray,
    sample_rate: int,
    reference_audio_path: str,
    strength: float,
) -> np.ndarray:
    ref_path = Path(reference_audio_path).expanduser().resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"audio_effects reference audio does not exist: {ref_path}")
    reference, ref_sr = _read_audio(ref_path)
    mono = np.mean(audio, axis=1)
    ref_mono = np.mean(reference, axis=1)
    source_features = _channel_features(mono, sample_rate)
    ref_features = _channel_features(ref_mono, ref_sr)
    processed = _match_spectral_profile(audio, sample_rate, source_features, ref_features, strength)

    high_ratio = source_features["band_ratios"]["high"]
    ref_high_ratio = ref_features["band_ratios"]["high"]
    if ref_high_ratio < high_ratio * 0.78:
        cutoff = 1600.0 + 2800.0 * ref_high_ratio / max(high_ratio, 1e-6)
        processed = _lowpass(processed, sample_rate, cutoff)
    elif ref_high_ratio > high_ratio * 1.25:
        highs = _highpass(processed, sample_rate, 2600.0)
        processed = _safe_peak(processed + highs * min(1.0, strength * 0.9))

    source_low_ratio = source_features["band_ratios"]["low"]
    ref_low_ratio = ref_features["band_ratios"]["low"]
    if ref_low_ratio < source_low_ratio * 0.75:
        processed = _highpass(processed, sample_rate, 80.0 + 220.0 * strength)
    elif ref_low_ratio > source_low_ratio * 1.25:
        lows = _lowpass(processed, sample_rate, 260.0)
        processed = _safe_peak(processed + lows * min(0.8, strength * 0.65))

    if ref_features["crest_factor_db"] < source_features["crest_factor_db"] - 2.5:
        processed = _soft_clip(processed, 1.25 + 2.2 * strength)
    if ref_features["spectral_flatness"] > source_features["spectral_flatness"] * 1.35:
        processed = _soft_clip(processed, 1.05 + 1.6 * strength)

    if ref_features["echo_proxy"] > source_features["echo_proxy"] + 0.035:
        echo_amount = min(1.0, (ref_features["echo_proxy"] - source_features["echo_proxy"]) * 5.0)
        echo = _multi_tap_echo(
            processed,
            sample_rate,
            [
                (0.028, 0.045 + 0.055 * echo_amount),
                (0.067, 0.03 + 0.05 * echo_amount),
                (0.124, 0.018 + 0.035 * echo_amount),
            ],
        )
        processed = _mix(processed, echo, min(0.38, 0.12 + 0.22 * strength * echo_amount))

    source_noise = source_features["relative_noise_floor"]
    ref_noise = ref_features["relative_noise_floor"]
    if ref_noise > source_noise * 1.35:
        noise_strength = min(0.035, (ref_noise - source_noise) * (0.055 + 0.055 * strength))
        noise = _reference_shaped_noise(audio.shape, sample_rate, ref_features, seed=71)
        processed = _safe_peak(processed + noise * noise_strength)

    processed = _match_rms(processed, reference)
    return _mix(audio, processed, 0.58 + 0.38 * strength)


def _channel_reference_analysis(
    audio: np.ndarray,
    sample_rate: int,
    reference_audio_path: str,
    strength: float = 0.75,
) -> dict[str, Any]:
    ref_path = Path(reference_audio_path).expanduser().resolve()
    if not ref_path.exists():
        return {"available": False, "reason": f"reference audio does not exist: {ref_path}"}
    reference, ref_sr = _read_audio(ref_path)
    source_features = _channel_features(np.mean(audio, axis=1), sample_rate)
    reference_features = _channel_features(np.mean(reference, axis=1), ref_sr)
    return {
        "available": True,
        "source": source_features,
        "reference": reference_features,
        "simulated_traits": _simulated_channel_traits(source_features, reference_features),
        "estimated_parameters": _estimated_channel_match_parameters(source_features, reference_features, strength),
    }


def _channel_features(mono: np.ndarray, sample_rate: int) -> dict[str, Any]:
    mono = mono.astype(np.float32)
    rms = _rms(mono[:, None])
    freqs, power = signal.welch(mono, sample_rate, nperseg=min(4096, max(16, mono.size)))
    total_power = float(np.sum(power) + 1e-12)
    band_ratios = {
        "low": _band_power_ratio(freqs, power, 60.0, 250.0, total_power),
        "low_mid": _band_power_ratio(freqs, power, 250.0, 1000.0, total_power),
        "mid": _band_power_ratio(freqs, power, 1000.0, 3200.0, total_power),
        "high": _band_power_ratio(freqs, power, 3200.0, min(7600.0, sample_rate * 0.45), total_power),
    }
    centroid = float(np.sum(freqs * power) / total_power)
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
    frame_rms = _frame_rms(mono, sample_rate)
    noise_floor = float(np.percentile(frame_rms, 10)) if frame_rms.size else rms
    speech_floor = float(np.percentile(frame_rms, 90)) if frame_rms.size else rms
    relative_noise = noise_floor / max(speech_floor, 1e-8)
    crest = float(np.percentile(np.abs(mono), 99.5) / max(rms, 1e-8))
    return {
        "sample_rate": sample_rate,
        "rms_db": round(_db(rms), 3),
        "noise_floor_db": round(_db(noise_floor), 3),
        "relative_noise_floor": round(relative_noise, 6),
        "spectral_centroid_hz": round(centroid, 3),
        "spectral_flatness": round(flatness, 6),
        "crest_factor_db": round(20.0 * math.log10(max(crest, 1e-8)), 3),
        "echo_proxy": round(_echo_proxy(mono, sample_rate), 6),
        "band_ratios": {key: round(value, 6) for key, value in band_ratios.items()},
    }


def _simulated_channel_traits(source: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    traits: list[str] = []
    src_bands = source.get("band_ratios", {})
    ref_bands = reference.get("band_ratios", {})
    if ref_bands.get("high", 0.0) < src_bands.get("high", 0.0) * 0.78:
        traits.append("high_frequency_rolloff_or_lowpass")
    if ref_bands.get("low", 0.0) < src_bands.get("low", 0.0) * 0.75:
        traits.append("low_frequency_rolloff")
    if reference.get("relative_noise_floor", 0.0) > source.get("relative_noise_floor", 0.0) * 1.35:
        traits.append("reference_noise_floor")
    if reference.get("crest_factor_db", 99.0) < source.get("crest_factor_db", 0.0) - 2.5:
        traits.append("mild_saturation_or_compression")
    if reference.get("spectral_flatness", 0.0) > source.get("spectral_flatness", 0.0) * 1.35:
        traits.append("lofi_distortion_or_noise_texture")
    if reference.get("echo_proxy", 0.0) > source.get("echo_proxy", 0.0) + 0.035:
        traits.append("short_room_reflections")
    return traits or ["rms_and_spectral_balance_match"]


def _estimated_channel_match_parameters(
    source: dict[str, Any],
    reference: dict[str, Any],
    strength: float,
) -> dict[str, Any]:
    strength = max(0.0, min(1.0, float(strength)))
    src_bands = source.get("band_ratios", {})
    ref_bands = reference.get("band_ratios", {})
    band_gains_db: dict[str, float] = {}
    for name in ("low", "low_mid", "mid", "high"):
        desired = math.sqrt(max(float(ref_bands.get(name, 0.0)), 1e-6) / max(float(src_bands.get(name, 0.0)), 1e-6))
        gain = 1.0 + (max(0.35, min(2.4, desired)) - 1.0) * min(1.0, 0.85 * strength)
        band_gains_db[name] = round(20.0 * math.log10(max(gain, 1e-8)), 3)

    params: dict[str, Any] = {
        "strength": round(strength, 3),
        "wet_mix": round(0.58 + 0.38 * strength, 3),
        "spectral_band_gains_db": band_gains_db,
        "target_rms_db": reference.get("rms_db"),
    }

    source_high = float(src_bands.get("high", 0.0))
    ref_high = float(ref_bands.get("high", 0.0))
    if ref_high < source_high * 0.78:
        params["lowpass_cutoff_hz"] = round(1600.0 + 2800.0 * ref_high / max(source_high, 1e-6), 1)
    elif ref_high > source_high * 1.25:
        params["high_shelf_boost"] = {"start_hz": 2600.0, "mix": round(min(1.0, strength * 0.9), 3)}

    source_low = float(src_bands.get("low", 0.0))
    ref_low = float(ref_bands.get("low", 0.0))
    if ref_low < source_low * 0.75:
        params["highpass_cutoff_hz"] = round(80.0 + 220.0 * strength, 1)
    elif ref_low > source_low * 1.25:
        params["low_shelf_boost"] = {"cutoff_hz": 260.0, "mix": round(min(0.8, strength * 0.65), 3)}

    source_crest = float(source.get("crest_factor_db", 0.0))
    ref_crest = float(reference.get("crest_factor_db", 99.0))
    if ref_crest < source_crest - 2.5:
        params["soft_clip_drive_for_compression"] = round(1.25 + 2.2 * strength, 3)
    if float(reference.get("spectral_flatness", 0.0)) > float(source.get("spectral_flatness", 0.0)) * 1.35:
        params["soft_clip_drive_for_lofi_texture"] = round(1.05 + 1.6 * strength, 3)

    echo_delta = float(reference.get("echo_proxy", 0.0)) - float(source.get("echo_proxy", 0.0))
    if echo_delta > 0.035:
        echo_amount = min(1.0, echo_delta * 5.0)
        params["short_reflection_echo"] = {
            "echo_amount": round(echo_amount, 3),
            "mix": round(min(0.38, 0.12 + 0.22 * strength * echo_amount), 3),
            "tap_delays_sec": [0.028, 0.067, 0.124],
        }

    source_noise = float(source.get("relative_noise_floor", 0.0))
    ref_noise = float(reference.get("relative_noise_floor", 0.0))
    if ref_noise > source_noise * 1.35:
        params["noise_bed"] = {
            "relative_noise_floor_delta": round(ref_noise - source_noise, 6),
            "mix": round(min(0.035, (ref_noise - source_noise) * (0.055 + 0.055 * strength)), 6),
            "shape": _reference_noise_shape(reference),
        }
    return params


def _reference_noise_shape(reference_features: dict[str, Any]) -> str:
    ratios = reference_features.get("band_ratios", {})
    high = float(ratios.get("high", 0.0))
    low = float(ratios.get("low", 0.0))
    if high > low * 1.5:
        return "high_frequency_hiss"
    if low > high * 1.5:
        return "low_frequency_rumble"
    return "broadband_room_tone"


def _match_spectral_profile(
    audio: np.ndarray,
    sample_rate: int,
    source_features: dict[str, Any],
    reference_features: dict[str, Any],
    strength: float,
) -> np.ndarray:
    bands = [
        ("low", 60.0, 250.0),
        ("low_mid", 250.0, 1000.0),
        ("mid", 1000.0, 3200.0),
        ("high", 3200.0, min(7600.0, sample_rate * 0.45)),
    ]
    src = source_features.get("band_ratios", {})
    ref = reference_features.get("band_ratios", {})
    rebuilt = np.zeros_like(audio, dtype=np.float32)
    for name, low, high in bands:
        if name == "low":
            component = _bandpass(audio, sample_rate, low, high)
        elif name == "high":
            component = _bandpass(audio, sample_rate, low, high)
        else:
            component = _bandpass(audio, sample_rate, low, high)
        desired = math.sqrt(max(float(ref.get(name, 0.0)), 1e-6) / max(float(src.get(name, 0.0)), 1e-6))
        gain = 1.0 + (max(0.35, min(2.4, desired)) - 1.0) * min(1.0, 0.85 * strength)
        rebuilt += component * gain
    return _mix(audio, rebuilt, min(0.85, 0.35 + 0.45 * strength))


def _band_power_ratio(
    freqs: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
    total_power: float,
) -> float:
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(power[band]) / max(total_power, 1e-12))


def _frame_rms(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(64, int(0.03 * sample_rate))
    hop = max(16, frame // 2)
    if mono.size < frame:
        return np.asarray([_rms(mono[:, None])], dtype=np.float32)
    values = []
    for start in range(0, mono.size - frame + 1, hop):
        chunk = mono[start : start + frame]
        values.append(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64) + 1e-12)))
    return np.asarray(values, dtype=np.float32)


def _echo_proxy(mono: np.ndarray, sample_rate: int) -> float:
    if mono.size < sample_rate // 2:
        return 0.0
    step = max(1, sample_rate // 8000)
    window = mono[: min(mono.size, sample_rate * 8) : step].astype(np.float32)
    window = window - float(np.mean(window))
    denom = float(np.dot(window, window) + 1e-12)
    sr = sample_rate / step
    values = []
    for lag_sec in (0.028, 0.045, 0.067, 0.095, 0.13):
        lag = int(round(lag_sec * sr))
        if 1 <= lag < window.size:
            values.append(abs(float(np.dot(window[:-lag], window[lag:]) / denom)))
    return max(values) if values else 0.0


def _reference_shaped_noise(
    shape: tuple[int, int],
    sample_rate: int,
    reference_features: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    ratios = reference_features.get("band_ratios", {})
    high = float(ratios.get("high", 0.0))
    low = float(ratios.get("low", 0.0))
    if high > low * 1.5:
        return _colored_noise(shape, sample_rate, highpass=900.0, lowpass=min(7600.0, sample_rate * 0.45), seed=seed)
    if low > high * 1.5:
        return _colored_noise(shape, sample_rate, highpass=50.0, lowpass=900.0, seed=seed)
    return _colored_noise(shape, sample_rate, highpass=120.0, lowpass=min(5200.0, sample_rate * 0.45), seed=seed)


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _bandpass(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = sample_rate * 0.5
    low = max(1.0, min(low_hz, nyq * 0.95))
    high = max(low + 1.0, min(high_hz, nyq * 0.98))
    sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return _sos_filter(sos, audio)


def _lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    nyq = sample_rate * 0.5
    cutoff = max(10.0, min(cutoff_hz, nyq * 0.98))
    sos = signal.butter(4, cutoff / nyq, btype="lowpass", output="sos")
    return _sos_filter(sos, audio)


def _highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    nyq = sample_rate * 0.5
    cutoff = max(10.0, min(cutoff_hz, nyq * 0.98))
    sos = signal.butter(3, cutoff / nyq, btype="highpass", output="sos")
    return _sos_filter(sos, audio)


def _sos_filter(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    try:
        if audio.shape[0] > 128:
            return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)
    except ValueError:
        pass
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def _soft_clip(audio: np.ndarray, drive: float) -> np.ndarray:
    drive = max(1.0, float(drive))
    return (np.tanh(audio * drive) / math.tanh(drive)).astype(np.float32)


def _colored_noise(
    shape: tuple[int, int],
    sample_rate: int,
    *,
    highpass: float,
    lowpass: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + shape[0] + shape[1] * 13)
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    return _bandpass(noise, sample_rate, highpass, lowpass) * 0.12


def _impulse_crackle(shape: tuple[int, int], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + shape[0])
    crackle = np.zeros(shape, dtype=np.float32)
    count = max(1, shape[0] // 2400)
    positions = rng.integers(0, shape[0], size=count)
    crackle[positions, :] = rng.uniform(-1.0, 1.0, size=(count, shape[1]))
    return crackle


def _amplitude_wobble(audio: np.ndarray, sample_rate: int, *, depth: float, hz: float) -> np.ndarray:
    t = np.arange(audio.shape[0], dtype=np.float32) / float(sample_rate)
    envelope = 1.0 - depth * 0.5 + depth * 0.5 * np.sin(2.0 * np.pi * hz * t)
    return (audio * envelope[:, None]).astype(np.float32)


def _amplitude_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.mean(np.abs(audio), axis=1)
    win = max(8, int(0.025 * sample_rate))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    envelope = np.convolve(mono, kernel, mode="same").astype(np.float32)
    peak = float(np.max(envelope)) if envelope.size else 0.0
    if peak > 1e-8:
        envelope = envelope / peak
    return envelope[:, None]


def _multi_tap_echo(audio: np.ndarray, sample_rate: int, taps: list[tuple[float, float]]) -> np.ndarray:
    out = audio.copy()
    for delay_sec, gain in taps:
        delay = int(delay_sec * sample_rate)
        if delay <= 0 or delay >= audio.shape[0]:
            continue
        out[delay:] += audio[:-delay] * float(gain)
    return _safe_peak(out)


def _crossfade_replace(original: np.ndarray, processed: np.ndarray, sample_rate: int) -> np.ndarray:
    length = min(original.shape[0], processed.shape[0])
    original = original[:length]
    processed = processed[:length].copy()
    fade = min(length // 4, int(0.06 * sample_rate))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        processed[:fade] = original[:fade] * (1.0 - ramp) + processed[:fade] * ramp
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
        processed[-fade:] = original[-fade:] * (1.0 - ramp) + processed[-fade:] * ramp
    return processed


def _mix(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    amount = max(0.0, min(1.0, amount))
    length = min(dry.shape[0], wet.shape[0])
    return _safe_peak(dry[:length] * (1.0 - amount) + wet[:length] * amount)


def _safe_peak(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def _match_rms(audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
    audio_rms = _rms(audio)
    ref_rms = _rms(reference)
    if audio_rms <= 1e-8 or ref_rms <= 1e-8:
        return audio
    gain = max(0.25, min(4.0, ref_rms / audio_rms))
    return _safe_peak(audio * gain)


def _band_energy_ratio(mono: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    if mono.size < 16:
        return 0.0
    freqs, power = signal.welch(mono.astype(np.float32), sample_rate, nperseg=min(2048, mono.size))
    band = (freqs >= low_hz) & (freqs <= high_hz)
    total = np.sum(power) + 1e-12
    return float(np.sum(power[band]) / total)
