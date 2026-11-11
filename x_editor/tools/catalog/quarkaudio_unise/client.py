"""HTTP client for the QuarkAudio-UniSE backend service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class QuarkAudioUniSEClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8025", timeout: float = 900.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def extract_target_speaker(
        self,
        mixture_audio_path: str,
        reference_audio_path: str,
        output_path: str,
        enroll_duration: float = 5.0,
    ) -> dict[str, Any]:
        mixture = Path(mixture_audio_path)
        reference = Path(reference_audio_path)
        if not mixture.exists():
            raise FileNotFoundError(f"Mixture audio not found: {mixture_audio_path}")
        if not reference.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        resp = requests.post(
            f"{self.base_url}/tse/extract",
            json={
                "mixture_audio_path": str(mixture),
                "reference_audio_path": str(reference),
                "output_path": output_path,
                "enroll_duration": enroll_duration,
            },
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"QuarkAudio-UniSE TSE service error {resp.status_code}: {resp.text[:4000]}")
        return resp.json()
