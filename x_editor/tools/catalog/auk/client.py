"""HTTP transport with bound source/output identity checks; no hidden retries."""

from pathlib import Path
import hashlib
import requests


def edit_audio(arguments, *, base_url, timeout=600):
    source = Path(arguments["input_audio_path"]).resolve()
    expected = Path(arguments["output_path"]).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    response = requests.post(base_url.rstrip("/") + "/edit", json=arguments, timeout=timeout)
    if not response.ok:
        # Retain the service's bounded contract diagnosis; do not retry or switch tools.
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            detail = response.text
        raise requests.HTTPError(f"AuK HTTP {response.status_code}: {str(detail)[:2048]}", response=response)
    response.raise_for_status()
    result = response.json()
    if Path(result["output_audio_path"]).resolve() != expected or expected == source:
        raise ValueError("AuK response output identity mismatch")
    if result["source_sha256"] != source_sha or hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise ValueError("AuK response source identity mismatch")
    if hashlib.sha256(expected.read_bytes()).hexdigest() != result["output_sha256"]:
        raise ValueError("AuK response output hash mismatch")
    return result
