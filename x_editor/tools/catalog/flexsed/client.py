"""HTTP client for the FlexSED local service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from x_editor.tools.catalog._http_retry import post_with_retries, retry_attempts


class FlexSEDClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8057", timeout: float = 900.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def detect(self, **kwargs: Any) -> dict[str, Any]:
        audio_path = kwargs.get("audio_path") or kwargs.get("input_audio_path")
        if not audio_path:
            raise ValueError("FlexSEDClient.detect requires audio_path or input_audio_path")
        if not Path(str(audio_path)).expanduser().exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")
        timeout = float(kwargs.get("timeout_seconds") or self.timeout)
        resp = post_with_retries(
            f"{self.base_url}/detect",
            json=kwargs,
            timeout=timeout,
            service_name="FlexSED",
            retry_env="X_EDITOR_FLEXSED_RETRIES",
            backoff_env="X_EDITOR_FLEXSED_RETRY_BACKOFF",
            default_retries=1,
        )
        if resp.status_code != 200:
            raise RuntimeError(_format_service_error(resp))
        data = resp.json()
        attempts = retry_attempts(resp)
        if attempts:
            data["request_retries"] = attempts
        heatmap = data.get("heatmap") or {}
        for key in ("png_path", "npy_path"):
            value = heatmap.get(key)
            if value and not Path(str(value)).exists():
                raise RuntimeError(f"FlexSED service reported missing heatmap artifact {key}: {value}")
        return data


def _format_service_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return f"FlexSED service error {resp.status_code}: {resp.text[:800]}"
    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        reason = detail.get("reason") or detail.get("message") or str(detail)
        availability = detail.get("availability")
        boundary = detail.get("boundary")
        parts = [f"FlexSED service error {resp.status_code}: {reason}"]
        if availability:
            parts.append(f"availability={availability}")
        if boundary:
            parts.append(f"boundary={boundary}")
        return "; ".join(parts)
    return f"FlexSED service error {resp.status_code}: {detail}"
