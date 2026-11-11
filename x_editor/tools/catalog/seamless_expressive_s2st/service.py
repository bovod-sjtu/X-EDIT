"""FastAPI service for Meta SeamlessExpressive and SeamlessM4T v2 S2ST.

The service is intentionally a thin, honest wrapper around Meta's CLIs:
`expressivity_predict` for expressive S2ST and `m4t_predict` for regular
SeamlessM4T v2 S2ST. It does not ship or fake model weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


LANGUAGE_ALIASES = {
    "ar": "arb",
    "arabic": "arb",
    "zh": "cmn",
    "chinese": "cmn",
    "mandarin": "cmn",
    "cmn_hans": "cmn",
    "de": "deu",
    "german": "deu",
    "en": "eng",
    "english": "eng",
    "es": "spa",
    "spanish": "spa",
    "fr": "fra",
    "french": "fra",
    "hi": "hin",
    "hindi": "hin",
    "id": "ind",
    "indonesian": "ind",
    "it": "ita",
    "italian": "ita",
    "ja": "jpn",
    "japanese": "jpn",
    "ko": "kor",
    "korean": "kor",
    "nl": "nld",
    "dutch": "nld",
    "pl": "pol",
    "polish": "pol",
    "pt": "por",
    "portuguese": "por",
    "ru": "rus",
    "russian": "rus",
    "sv": "swe",
    "swedish": "swe",
    "tr": "tur",
    "turkish": "tur",
    "uk": "ukr",
    "ukrainian": "ukr",
    "ur": "urd",
    "urdu": "urd",
    "vi": "vie",
    "vietnamese": "vie",
}

M4T_V2_TARGET_SPEECH_LANGS = {
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cmn_Hant",
    "cym",
    "dan",
    "deu",
    "eng",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
}

EXPRESSIVE_PRIMARY_TARGET_LANGS = {"eng", "fra", "deu", "spa"}
EXPRESSIVE_EXPERIMENTAL_TARGET_LANGS = {"cmn", "ita"}
EXPRESSIVE_TARGET_LANGS = EXPRESSIVE_PRIMARY_TARGET_LANGS | EXPRESSIVE_EXPERIMENTAL_TARGET_LANGS


class TranslateRequest(BaseModel):
    input_audio_path: str
    target_language: str = Field(description="ISO-639-3 Seamless language code or common language name.")
    output_path: str
    source_language: str | None = None
    mode: Literal["auto", "expressive", "m4t_v2"] = "auto"
    preserve_vocal_style: bool = True
    preserve_speech_rate: bool = True
    preserve_pauses: bool = True
    allow_fallback_to_m4t: bool = False
    duration_factor: float = 1.0
    timeout_seconds: float = 1800.0
    extra_args: list[str] = Field(default_factory=list)


app = FastAPI(title="X-Editor SeamlessExpressive S2ST Backend", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "seamless_expressive_s2st", "status": "running", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "backend": "seamless_expressive_s2st",
        "availability": _availability(),
        "capabilities": {
            "speech_to_speech_translation": True,
            "expressive_preservation_claims": ["vocal_style", "speech_rate", "pauses"],
            "m4t_v2_target_speech_languages": sorted(M4T_V2_TARGET_SPEECH_LANGS),
            "expressive_primary_target_languages": sorted(EXPRESSIVE_PRIMARY_TARGET_LANGS),
            "expressive_experimental_target_languages": sorted(EXPRESSIVE_EXPERIMENTAL_TARGET_LANGS),
        },
    }


@app.post("/translate")
def translate(request: TranslateRequest) -> dict[str, Any]:
    input_path = Path(request.input_audio_path)
    if not input_path.exists():
        _raise(400, f"input audio not found: {request.input_audio_path}")
    target_language = _normalize_language(request.target_language)
    output_path = Path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = _choose_backend(request, target_language)
    warning: str | None = None
    if backend == "expressive" and target_language in EXPRESSIVE_EXPERIMENTAL_TARGET_LANGS:
        warning = f"SeamlessExpressive marks target language '{target_language}' as experimental in the public docs."

    if backend == "expressive":
        _run_expressive(request, target_language, output_path)
    elif backend == "m4t_v2":
        if request.mode == "expressive":
            warning = "Explicit expressive mode fell back to SeamlessM4T v2 only because allow_fallback_to_m4t=true."
        _run_m4t_v2(request, target_language, output_path)
    else:  # pragma: no cover - defensive
        _raise(500, f"unexpected backend choice: {backend}")

    if not output_path.exists():
        _raise(500, f"backend completed but did not create output audio: {output_path}")

    return {
        "output_audio_path": str(output_path),
        "backend_used": backend,
        "target_language": target_language,
        "sample_rate": _wav_sample_rate(output_path),
        "duration_seconds": _wav_duration_seconds(output_path),
        "expressivity_preserved": {
            "vocal_style": backend == "expressive" and request.preserve_vocal_style,
            "speech_rate": backend == "expressive" and request.preserve_speech_rate,
            "pauses": backend == "expressive" and request.preserve_pauses,
            "note": (
                "Expressive preservation follows SeamlessExpressive model capability claims; "
                "X-Editor has not performed perceptual or metric validation for this output."
            ),
        },
        "warning": warning,
    }


def _choose_backend(request: TranslateRequest, target_language: str) -> str:
    expressive_requested = (
        request.mode == "expressive"
        or (
            request.mode == "auto"
            and (request.preserve_vocal_style or request.preserve_speech_rate or request.preserve_pauses)
        )
    )
    if expressive_requested:
        if target_language not in EXPRESSIVE_TARGET_LANGS:
            if request.allow_fallback_to_m4t:
                _ensure_m4t_language(target_language)
                return "m4t_v2"
            _raise(
                422,
                f"SeamlessExpressive target language '{target_language}' is outside the documented/demo target set.",
                boundary={
                    "primary_targets": sorted(EXPRESSIVE_PRIMARY_TARGET_LANGS),
                    "experimental_targets": sorted(EXPRESSIVE_EXPERIMENTAL_TARGET_LANGS),
                    "fallback_available_if_enabled": target_language in M4T_V2_TARGET_SPEECH_LANGS,
                },
            )
        _ensure_expressive_runtime()
        return "expressive"

    _ensure_m4t_language(target_language)
    _ensure_m4t_runtime()
    return "m4t_v2"


def _run_expressive(request: TranslateRequest, target_language: str, output_path: Path) -> None:
    cli = _resolve_command(os.getenv("SEAMLESS_EXPRESSIVE_CLI", "expressivity_predict"))
    model_dir = os.getenv("SEAMLESS_EXPRESSIVE_MODEL_DIR")
    cmd = [
        cli,
        request.input_audio_path,
        "--tgt_lang",
        target_language,
        "--model_name",
        "seamless_expressivity",
        "--vocoder_name",
        "vocoder_pretssel",
        "--output_path",
        str(output_path),
        "--text_unk_blocking",
        "True",
        "--duration_factor",
        f"{request.duration_factor:g}",
    ]
    if model_dir:
        cmd.extend(["--gated-model-dir", model_dir])
    cmd.extend(request.extra_args)
    _run_command(cmd, request.timeout_seconds)


def _run_m4t_v2(request: TranslateRequest, target_language: str, output_path: Path) -> None:
    cli = _resolve_command(os.getenv("SEAMLESS_M4T_CLI", "m4t_predict"))
    cmd = [
        cli,
        request.input_audio_path,
        "--task",
        "S2ST",
        "--tgt_lang",
        target_language,
        "--output_path",
        str(output_path),
        "--model_name",
        "seamlessM4T_v2_large",
        "--vocoder_name",
        "vocoder_v2",
    ]
    cmd.extend(request.extra_args)
    _run_command(cmd, request.timeout_seconds)


def _ensure_expressive_runtime() -> None:
    availability = _availability()
    missing: list[str] = []
    if not availability["expressivity_predict"]:
        missing.append("expressivity_predict CLI")
    if not availability["expressive_model_dir_exists"]:
        missing.append("SEAMLESS_EXPRESSIVE_MODEL_DIR with approved gated artifacts")
    if missing:
        _raise(
            503,
            "SeamlessExpressive runtime is not available: missing " + ", ".join(missing),
            availability=availability,
        )


def _ensure_m4t_runtime() -> None:
    availability = _availability()
    if not availability["m4t_predict"]:
        _raise(
            503,
            "SeamlessM4T v2 runtime is not available: missing m4t_predict CLI",
            availability=availability,
        )


def _ensure_m4t_language(target_language: str) -> None:
    if target_language not in M4T_V2_TARGET_SPEECH_LANGS:
        _raise(
            422,
            f"SeamlessM4T v2 target speech language '{target_language}' is not in the documented speech-output set.",
            boundary={"m4t_v2_target_speech_languages": sorted(M4T_V2_TARGET_SPEECH_LANGS)},
        )


def _availability() -> dict[str, Any]:
    expressive_model_dir = os.getenv("SEAMLESS_EXPRESSIVE_MODEL_DIR")
    return {
        "seamless_communication_importable": _is_importable("seamless_communication"),
        "expressivity_predict": bool(_which(os.getenv("SEAMLESS_EXPRESSIVE_CLI", "expressivity_predict"))),
        "m4t_predict": bool(_which(os.getenv("SEAMLESS_M4T_CLI", "m4t_predict"))),
        "expressive_model_dir": expressive_model_dir,
        "expressive_model_dir_exists": bool(expressive_model_dir and Path(expressive_model_dir).exists()),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }


def _run_command(cmd: list[str], timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        _raise(503, f"backend command not found: {cmd[0]}", availability=_availability())
        raise RuntimeError from exc
    except subprocess.TimeoutExpired as exc:
        _raise(504, f"backend command timed out after {timeout_seconds:g}s: {' '.join(cmd[:3])}")
        raise RuntimeError from exc

    if completed.returncode != 0:
        message = (
            f"backend command failed with exit code {completed.returncode}. "
            f"stderr={completed.stderr[-2000:]!r} stdout={completed.stdout[-1000:]!r}"
        )
        _raise(500, message)


def _normalize_language(language: str) -> str:
    normalized = language.strip().replace("-", "_")
    lower = normalized.lower()
    return LANGUAGE_ALIASES.get(lower, normalized)


def _resolve_command(command: str) -> str:
    found = _which(command)
    return found or command


def _which(command: str) -> str | None:
    path = Path(command).expanduser()
    if path.exists():
        return str(path)
    return shutil.which(command)


def _is_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _wav_sample_rate(path: Path) -> int | None:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getframerate()
    except Exception:
        return None


def _wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / rate if rate else None
    except Exception:
        return None


def _raise(
    status_code: int,
    reason: str,
    *,
    availability: dict[str, Any] | None = None,
    boundary: dict[str, Any] | None = None,
) -> None:
    detail: dict[str, Any] = {"reason": reason}
    if availability is not None:
        detail["availability"] = availability
    if boundary is not None:
        detail["boundary"] = boundary
    raise HTTPException(status_code=status_code, detail=detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="SeamlessExpressive / SeamlessM4T v2 S2ST backend service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8039)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
