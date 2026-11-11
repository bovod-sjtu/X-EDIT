"""HTTP client for the shared ClearerVoice-Studio backend service."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import requests

from x_editor.tools.catalog._http_retry import post_with_retries, retry_attempts


class ClearerVoiceClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8027", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def extract_target_speaker(
        self,
        mixture_audio_path: str,
        target_speaker_hint: str,
        output_dir: str,
        reference_audio_path: str | None = None,
        return_residual: bool = True,
    ) -> dict[str, Any]:
        mixture = Path(mixture_audio_path)
        if not mixture.exists():
            raise FileNotFoundError(f"Mixture audio not found: {mixture_audio_path}")
        if reference_audio_path and not Path(reference_audio_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        resp = post_with_retries(
            f"{self.base_url}/tse/extract",
            json={
                "mixture_audio_path": str(mixture),
                "target_speaker_hint": target_speaker_hint,
                "reference_audio_path": reference_audio_path,
                "output_dir": output_dir,
                "return_residual": return_residual,
            },
            timeout=self.timeout,
            service_name="ClearerVoice TSE",
            retry_env="X_EDITOR_CLEARVOICE_RETRIES",
            backoff_env="X_EDITOR_CLEARVOICE_RETRY_BACKOFF",
            default_retries=1,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ClearerVoice TSE service error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        attempts = retry_attempts(resp)
        if attempts:
            data["request_retries"] = attempts
        return data

    def enhance(
        self,
        input_audio_path: str,
        output_path: str,
        mode: str = "general",
        preserve_background: bool = False,
        enhancement_goal: str | None = None,
    ) -> dict[str, Any]:
        input_audio = Path(input_audio_path)
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        resp = post_with_retries(
            f"{self.base_url}/enhance",
            json={
                "input_audio_path": str(input_audio),
                "output_path": output_path,
                "mode": mode,
                "preserve_background": preserve_background,
                "enhancement_goal": enhancement_goal,
            },
            timeout=self.timeout,
            service_name="ClearerVoice enhancement",
            retry_env="X_EDITOR_CLEARVOICE_RETRIES",
            backoff_env="X_EDITOR_CLEARVOICE_RETRY_BACKOFF",
            default_retries=1,
        )
        if resp.status_code != 200:
            if _client_copy_fallback_enabled() and _is_cuda_resource_error(resp.text):
                shutil.copy2(input_audio, output_path)
                return {
                    "output_audio_path": str(output_path),
                    "warning": (
                        "ClearerVoice client fallback copied input because the backend failed with "
                        f"{resp.status_code}: {resp.text[:300]}"
                    ),
                    "fallback_reason": resp.text[:1000],
                }
            raise RuntimeError(f"ClearerVoice enhancement service error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        attempts = retry_attempts(resp)
        if attempts:
            data["request_retries"] = attempts
        return data


def _client_copy_fallback_enabled() -> bool:
    return os.getenv("X_EDITOR_CLEARVOICE_CLIENT_ALLOW_COPY_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_cuda_resource_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "cuda" in lowered
        and (
            "out of memory" in lowered
            or "invalid device ordinal" in lowered
            or "cudnn_status_internal_error" in lowered
        )
    )
