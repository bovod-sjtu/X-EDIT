"""Small retry helpers for local HTTP tool services."""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

import requests


TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
TRANSIENT_EXCEPTIONS = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


def post_with_retries(
    url: str,
    *,
    json: dict[str, Any],
    timeout: float,
    service_name: str,
    retry_env: str,
    backoff_env: str,
    default_retries: int = 1,
    retry_statuses: Iterable[int] = TRANSIENT_STATUS_CODES,
) -> requests.Response:
    """POST JSON and retry only transient transport/service failures.

    ``default_retries`` is the number of extra attempts after the first call.
    Deterministic schema or boundary errors should not be retried by callers.
    """

    retries = _env_int(retry_env, default_retries)
    backoff = _env_float(backoff_env, 1.0)
    statuses = set(retry_statuses)
    attempts: list[dict[str, Any]] = []
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=json, timeout=timeout)
        except TRANSIENT_EXCEPTIONS as exc:
            last_exc = exc
            attempts.append({"attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}"})
            if attempt >= retries:
                raise RuntimeError(
                    f"{service_name} request failed after {attempt + 1} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            _sleep_before_retry(backoff, attempt)
            continue
        if attempts:
            setattr(response, "_x_editor_retry_attempts", list(attempts))
        if response.status_code not in statuses or attempt >= retries:
            return response
        attempts.append({"attempt": attempt + 1, "status_code": response.status_code})
        _sleep_before_retry(backoff, attempt)
    raise RuntimeError(f"{service_name} request failed after retries: {last_exc}")


def retry_response_post(
    response: requests.Response,
    url: str,
    *,
    json: dict[str, Any],
    timeout: float,
    service_name: str,
    retry_env: str,
    backoff_env: str,
    default_retries: int = 1,
    retry_statuses: Iterable[int] = TRANSIENT_STATUS_CODES,
) -> requests.Response:
    """Retry after a caller has inspected the first non-OK response."""

    if response.status_code not in set(retry_statuses):
        return response
    retries = _env_int(retry_env, default_retries)
    if retries <= 0:
        return response
    backoff = _env_float(backoff_env, 1.0)
    attempts = [{"attempt": 1, "status_code": response.status_code}]
    for attempt in range(retries):
        _sleep_before_retry(backoff, attempt)
        try:
            next_response = requests.post(url, json=json, timeout=timeout)
        except TRANSIENT_EXCEPTIONS as exc:
            attempts.append({"attempt": attempt + 2, "error": f"{type(exc).__name__}: {exc}"})
            if attempt + 1 >= retries:
                raise RuntimeError(
                    f"{service_name} request failed after {attempt + 2} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            continue
        setattr(next_response, "_x_editor_retry_attempts", list(attempts))
        if next_response.status_code not in set(retry_statuses) or attempt + 1 >= retries:
            return next_response
        attempts.append({"attempt": attempt + 2, "status_code": next_response.status_code})
    return response


def retry_attempts(response: requests.Response) -> list[dict[str, Any]]:
    attempts = getattr(response, "_x_editor_retry_attempts", None)
    return list(attempts) if isinstance(attempts, list) else []


def _sleep_before_retry(backoff: float, attempt: int) -> None:
    if backoff <= 0:
        return
    time.sleep(backoff * (2 ** attempt))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default
