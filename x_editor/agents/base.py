"""Base Gemini API client for X-Editor agents."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import requests

from x_editor.core.config import resolve_gemini_api_key


class GeminiClient:
    """Generic Gemini API client with audio and text support."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 40960,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        retry_backoff_max: float | None = None,
    ) -> None:
        self._api_key = resolve_gemini_api_key(api_key) or ""
        self._base_url = (base_url or os.getenv("X_EDITOR_GEMINI_BASE_URL") or "").strip()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = _env_float("X_EDITOR_GEMINI_TIMEOUT", 60.0) if timeout is None else float(timeout)
        self._max_retries = (
            None if max_retries is None else max(0, int(max_retries))
        )
        self._retry_backoff = (
            None if retry_backoff is None else max(0.0, float(retry_backoff))
        )
        self._retry_backoff_max = (
            None
            if retry_backoff_max is None
            else max(0.0, float(retry_backoff_max))
        )
        self.last_call_metrics: dict[str, Any] | None = None
        self.last_transport_attempt_count = 0

    def _encode_audio(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_payload(
        self,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []

        if audio_paths:
            for audio_path in audio_paths:
                audio_b64 = self._encode_audio(audio_path)
                parts.append({
                    "inline_data": {
                        "mime_type": _audio_mime_type(audio_path),
                        "data": audio_b64,
                    }
                })

        parts.append({"text": user_text})

        payload: dict[str, Any] = {
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_tokens,
            },
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
        }

        return payload

    def call(
        self,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str] | None = None,
    ) -> str:
        self.last_call_metrics = None
        self.last_transport_attempt_count = 0
        if not self._base_url:
            raise RuntimeError("Set base_url or X_EDITOR_GEMINI_BASE_URL to a compatible generateContent endpoint")
        if not self._api_key:
            raise RuntimeError("API key required for Gemini client")

        payload = self._build_payload(system_prompt, user_text, audio_paths)
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }

        max_retries = (
            self._max_retries
            if self._max_retries is not None
            else _env_int("X_EDITOR_GEMINI_MAX_RETRIES", 1, minimum=0)
        )
        retry_backoff = (
            self._retry_backoff
            if self._retry_backoff is not None
            else _env_float("X_EDITOR_GEMINI_RETRY_BACKOFF", 0.5, minimum=0.0)
        )
        retry_backoff_max = (
            self._retry_backoff_max
            if self._retry_backoff_max is not None
            else _env_float(
                "X_EDITOR_GEMINI_RETRY_BACKOFF_MAX",
                2.0,
                minimum=0.0,
            )
        )
        payload_metrics = _payload_metrics(
            base_url=self._base_url,
            timeout=self._timeout,
            system_prompt=system_prompt,
            user_text=user_text,
            audio_paths=audio_paths or [],
            payload=payload,
        )
        payload_context = _payload_context(
            base_url=self._base_url,
            timeout=self._timeout,
            system_prompt=system_prompt,
            user_text=user_text,
            audio_paths=audio_paths or [],
            payload=payload,
        )
        last_error: Exception | None = None
        resp = None
        started_at = time.monotonic()
        for attempt in range(max_retries + 1):
            self.last_transport_attempt_count += 1
            try:
                resp = _post_with_wall_timeout(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(_retry_delay(attempt, retry_backoff, retry_backoff_max))
                    continue
                self.last_call_metrics = {
                    **payload_metrics,
                    "attempts": attempt + 1,
                    "request_elapsed_sec": round(time.monotonic() - started_at, 3),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
                raise RuntimeError(
                    "Gemini API request failed after "
                    f"{attempt + 1} attempt(s) ({payload_context}): {type(exc).__name__}: {exc}"
                ) from exc
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                last_error = RuntimeError(f"Gemini API status {resp.status_code}: {resp.text[:500]}")
                time.sleep(_retry_delay(attempt, retry_backoff, retry_backoff_max))
                continue
            break

        if resp is None:
            raise RuntimeError(f"Gemini API request failed: {last_error}")

        if resp.status_code != 200:
            self.last_call_metrics = {
                **payload_metrics,
                "attempts": max_retries + 1,
                "request_elapsed_sec": round(time.monotonic() - started_at, 3),
                "status": "failed",
                "http_status": resp.status_code,
            }
            raise RuntimeError(
                f"Gemini API status {resp.status_code}: {resp.text[:500]}"
            )

        self.last_call_metrics = {
            **payload_metrics,
            "attempts": attempt + 1,
            "request_elapsed_sec": round(time.monotonic() - started_at, 3),
            "status": "ok",
            "http_status": resp.status_code,
        }

        result = resp.json()

        if result.get("Code") == 10001:
            raise RuntimeError("Gemini API token limit error (Code 10001)")

        candidates = result.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini API returned empty parts")

        # Prefer non-thinking part if multiple parts exist
        text = ""
        for part in parts:
            if "text" in part:
                text = part["text"]
                break

        text = text.strip()
        if not text:
            raise RuntimeError("Gemini API returned empty text")

        return text

    def call_json(
        self,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Call API and parse JSON response with robust extraction."""
        # Append JSON enforcement to system prompt
        json_enforcement = (
            "\n\nCRITICAL: You must output ONLY a valid JSON object. "
            "No markdown fences (```), no explanations, no preamble. "
            "Just the raw JSON."
        )
        enforced_system = system_prompt + json_enforcement

        text = self.call(enforced_system, user_text, audio_paths)

        for candidate in _json_candidates(text):
            parsed = _try_parse_json_object(candidate)
            if parsed is not None:
                return parsed

        raise RuntimeError(f"Failed to parse JSON from model. Raw text:\n{text[:2000]}")


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        candidates.append(stripped.strip())

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    balanced = _extract_balanced_json_object(text)
    if balanced:
        candidates.append(balanced)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _retry_delay(attempt: int, backoff: float, backoff_max: float) -> float:
    if backoff <= 0 or backoff_max <= 0:
        return 0.0
    return min(backoff_max, backoff * (attempt + 1))


def _post_with_wall_timeout(
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    timeout: float,
) -> requests.Response:
    if threading.current_thread() is not threading.main_thread():
        return _post_with_async_wall_timeout(
            url,
            headers=headers,
            json=json,
            timeout=timeout,
        )
    with _wall_clock_timeout(timeout):
        return requests.post(url, headers=headers, json=json, timeout=timeout)


def _post_with_async_wall_timeout(
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    timeout: float,
) -> requests.Response:
    """Enforce one total deadline when provider calls run in worker threads."""

    async def post() -> requests.Response:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(url, headers=headers, json=json)
        converted = requests.Response()
        converted.status_code = response.status_code
        converted.headers = requests.structures.CaseInsensitiveDict(response.headers)
        converted._content = response.content
        converted.encoding = response.encoding
        converted.url = str(response.url)
        converted.reason = response.reason_phrase
        return converted

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(post())
        return _run_coroutine_in_transport_thread(post, timeout=timeout)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise requests.exceptions.ReadTimeout(
            f"wall-clock timeout after {timeout:g}s"
        ) from exc
    except httpx.RequestError as exc:
        raise requests.exceptions.ConnectionError(str(exc)) from exc


def _run_coroutine_in_transport_thread(coroutine_factory, *, timeout: float):
    """Run sync-provider transport when its worker already owns an event loop."""

    completed = threading.Event()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = asyncio.run(coroutine_factory())
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            completed.set()

    transport_thread = threading.Thread(
        target=run,
        name="x-editor-provider-transport",
        daemon=True,
    )
    transport_thread.start()
    if not completed.wait(max(0.0, timeout) + 1.0):
        raise TimeoutError(f"provider transport exceeded {timeout:g}s wall-clock timeout")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@contextmanager
def _wall_clock_timeout(timeout: float):
    if (
        timeout <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise requests.exceptions.ReadTimeout(f"wall-clock timeout after {timeout:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _payload_context(
    *,
    base_url: str,
    timeout: float,
    system_prompt: str,
    user_text: str,
    audio_paths: list[str],
    payload: dict[str, Any],
) -> str:
    metrics = _payload_metrics(
        base_url=base_url,
        timeout=timeout,
        system_prompt=system_prompt,
        user_text=user_text,
        audio_paths=audio_paths,
        payload=payload,
    )
    return (
        f"host={metrics['host']}, timeout={timeout:g}s, prompt_chars={metrics['prompt_chars']}, "
        f"audio_count={metrics['audio_count']}, audio_bytes={metrics['audio_bytes']}, "
        f"payload_bytes={metrics['payload_bytes']}"
    )


def _payload_metrics(
    *,
    base_url: str,
    timeout: float,
    system_prompt: str,
    user_text: str,
    audio_paths: list[str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    audio_bytes = 0
    for audio_path in audio_paths:
        try:
            audio_bytes += Path(audio_path).stat().st_size
        except OSError:
            pass
    return {
        "host": urlparse(base_url).netloc or base_url,
        "timeout_sec": timeout,
        "prompt_chars": len(system_prompt) + len(user_text),
        "system_prompt_chars": len(system_prompt),
        "user_text_chars": len(user_text),
        "audio_count": len(audio_paths),
        "audio_bytes": audio_bytes,
        "payload_bytes": len(json.dumps(payload, ensure_ascii=False)),
    }


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    for value in [candidate, _remove_trailing_commas(candidate)]:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        try:
            parsed, _ = json.JSONDecoder().raw_decode(value.lstrip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _remove_trailing_commas(value: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", value)


def _audio_mime_type(audio_path: str) -> str:
    suffix = Path(audio_path).suffix.lower().lstrip(".")
    if suffix == "flac":
        return "audio/flac"
    if suffix in {"mp3", "mpeg"}:
        return "audio/mpeg"
    if suffix in {"ogg", "oga"}:
        return "audio/ogg"
    if suffix in {"m4a", "mp4"}:
        return "audio/mp4"
    return "audio/wav"
