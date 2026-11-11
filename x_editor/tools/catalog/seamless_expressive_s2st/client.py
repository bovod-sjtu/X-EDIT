"""HTTP client for the SeamlessExpressive / SeamlessM4T S2ST service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from x_editor.tools.catalog._http_retry import post_with_retries, retry_attempts


class SeamlessExpressiveS2STClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8039", timeout: float = 1800.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def translate(self, **kwargs: Any) -> dict[str, Any]:
        input_audio_path = Path(kwargs["input_audio_path"])
        if not input_audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio_path}")
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resp = post_with_retries(
            f"{self.base_url}/translate",
            json=kwargs,
            timeout=float(kwargs.get("timeout_seconds") or self.timeout),
            service_name="Seamless S2ST",
            retry_env="X_EDITOR_SEAMLESS_S2ST_RETRIES",
            backoff_env="X_EDITOR_SEAMLESS_S2ST_RETRY_BACKOFF",
            default_retries=1,
        )
        if resp.status_code != 200:
            raise RuntimeError(_format_service_error(resp))
        data = resp.json()
        attempts = retry_attempts(resp)
        if attempts:
            data["request_retries"] = attempts
        output_audio_path = data.get("output_audio_path")
        if output_audio_path and not Path(output_audio_path).exists():
            raise RuntimeError(f"Service reported output that does not exist: {output_audio_path}")
        return data


def _format_service_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return f"Seamless S2ST service error {resp.status_code}: {resp.text[:800]}"
    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        reason = detail.get("reason") or detail.get("message") or str(detail)
        availability = detail.get("availability")
        boundary = detail.get("boundary")
        parts = [f"Seamless S2ST service error {resp.status_code}: {reason}"]
        if availability:
            parts.append(f"availability={availability}")
        if boundary:
            parts.append(f"boundary={boundary}")
        return "; ".join(parts)
    return f"Seamless S2ST service error {resp.status_code}: {detail}"
