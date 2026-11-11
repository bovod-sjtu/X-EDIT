"""Read-only verification of contextual ASR, separate from inference policy."""
from __future__ import annotations

import json
import urllib.request


def check_context_contract(url: str, *, timeout: float = 10) -> dict:
    result = {"component": "asr_context", "url": url, "ok": False, "inference_verified": False}
    try:
        documents = {}
        for endpoint in ("openapi.json", "info"):
            with urllib.request.urlopen(url.rstrip("/") + "/" + endpoint, timeout=timeout) as response:
                documents[endpoint] = json.load(response)
        schema = documents["openapi.json"]["paths"]["/transcribe"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        if "$ref" in schema:
            schema = documents["openapi.json"]["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
        field = schema.get("properties", {}).get("context", {})
        supports_field = field.get("type") == "string" or any(
            item.get("type") == "string" for item in field.get("anyOf", []))
        info = documents["info"]
        backend = info.get("default_backend") or info.get("backend")
        selected = info["backends"][backend]
        supported = supports_field and selected.get("supports_context") is True
        result.update(ok=supported, backend=backend, backend_info=selected,
                      context_in_schema=supports_field)
        if not supported:
            result["error"] = "ASR requires both a string context input and selected-backend supports_context=true"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result["error"] = f"Context capability could not be verified: {type(error).__name__}"
    return result
