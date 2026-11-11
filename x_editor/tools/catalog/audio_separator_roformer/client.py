"""HTTP client for the audio-separator RoFormer backend service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class AudioSeparatorRoFormerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8042", timeout: float = 900.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def separate(
        self,
        *,
        input_audio_path: str,
        output_dir: str,
        output_path: str | None = None,
        model_family: str = "bs_roformer",
        model_filename: str | None = None,
        output_format: str = "WAV",
        stem_policy: str = "vocals",
    ) -> dict[str, Any]:
        input_audio = Path(input_audio_path)
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio_path}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        response = requests.post(
            f"{self.base_url}/separate",
            json={
                "input_audio_path": str(input_audio),
                "output_dir": output_dir,
                "output_path": output_path,
                "model_family": model_family,
                "model_filename": model_filename,
                "output_format": output_format,
                "stem_policy": stem_policy,
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"audio-separator service error {response.status_code}: {response.text[:500]}")
        return response.json()
