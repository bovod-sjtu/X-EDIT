"""HTTP client for the SeedVC local service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class SeedVCClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8003", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def convert(
        self,
        source_audio_path: str,
        reference_audio_path: str,
        output_path: str,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        seed: int = 0,
    ) -> dict[str, Any]:
        source = Path(source_audio_path)
        reference = Path(reference_audio_path)
        if not source.exists():
            raise FileNotFoundError(f"Source audio not found: {source_audio_path}")
        if not reference.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.post(
            f"{self.base_url}/convert",
            json={
                "source_audio": str(source),
                "reference_audio": str(reference),
                "diffusion_steps": diffusion_steps,
                "length_adjust": length_adjust,
                "inference_cfg_rate": inference_cfg_rate,
                "output_format": "wav",
                "seed": seed,
            },
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"SeedVC service error {resp.status_code}: {resp.text[:500]}")
        output.write_bytes(resp.content)
        return {
            "output_audio_path": str(output),
            "sample_rate": _header_int(resp, "X-Sample-Rate"),
            "duration_seconds": _header_float(resp, "X-Duration-Seconds"),
        }


def _header_int(resp: requests.Response, name: str) -> int | None:
    value = resp.headers.get(name)
    return int(value) if value else None


def _header_float(resp: requests.Response, name: str) -> float | None:
    value = resp.headers.get(name)
    return float(value) if value else None
