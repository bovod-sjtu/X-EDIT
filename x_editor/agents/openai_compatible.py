"""Small OpenAI-compatible chat client used by open X-Editor providers."""

from __future__ import annotations

import base64
import copy
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import requests

from x_editor.agents.base import (
    _audio_mime_type,
    _json_candidates,
    _post_with_wall_timeout,
    _try_parse_json_object,
)
from x_editor.core.provider_requests import (
    ProviderLogicalDeadline,
    ProviderLogicalDeadlineExceeded,
    ProviderQueueTimeout,
    ProviderRequestScheduler,
    ProviderRequestTelemetry,
    normalize_provider_request_id,
)

_LogicalResult = TypeVar("_LogicalResult")


class OpenAICompatibleServiceUnavailable(RuntimeError):
    """Raised when a configured OpenAI-compatible provider cannot be reached."""


class OpenAICompatibleEmptyContent(OpenAICompatibleServiceUnavailable):
    """Raised when a provider response has no extractable assistant text."""


@dataclass(frozen=True)
class _LogicalProviderCall:
    request_id: str
    telemetry: ProviderRequestTelemetry
    deadline: ProviderLogicalDeadline


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible ``/chat/completions`` client.

    The client intentionally does not depend on an SDK so benchmark/worktree
    runs remain reproducible in lightweight environments.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model_name: str,
        provider_name: str,
        temperature: float = 0.01,
        seed: int | None = None,
        max_tokens: int = 16384,
        timeout: float = 60.0,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        allow_response_format_fallback: bool = True,
        allow_json_repair: bool = True,
        require_api_key: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
        audio_content_part: str | None = None,
        request_body: Mapping[str, Any] | None = None,
        component_name: str = "unspecified",
        request_scheduler: ProviderRequestScheduler | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = _chat_completions_url((base_url or "").strip())
        self.model_name = model_name
        self.provider_name = provider_name
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = None if max_retries is None else max(0, int(max_retries))
        self.retry_backoff = None if retry_backoff is None else max(0.0, float(retry_backoff))
        self.allow_response_format_fallback = bool(allow_response_format_fallback)
        self.allow_json_repair = bool(allow_json_repair)
        self.require_api_key = bool(require_api_key)
        self.chat_template_kwargs = copy.deepcopy(chat_template_kwargs or {})
        self.audio_content_part = _normalize_audio_content_part(audio_content_part)
        self.request_body = _normalize_request_body(request_body)
        self.component_name = component_name
        self.request_scheduler = request_scheduler
        self.last_call_metrics: dict[str, Any] | None = None
        self.last_request_id: str | None = None
        self.last_request_telemetry: dict[str, Any] | None = None
        self.last_raw_text: str | None = None
        self.last_transport_attempt_count = 0
        self.last_call_invocations: list[dict[str, Any]] = []
        self.last_json_invocations: list[dict[str, Any]] = []

    def call(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        request_id: str | None = None,
    ) -> str:
        return self._run_logical_request(
            request_id=request_id,
            admitted_call=lambda context: self._call_admitted(
                system_prompt,
                user_text,
                audio_paths=audio_paths,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                request_id=context.request_id,
                telemetry=context.telemetry,
                deadline=context.deadline,
            ),
        )

    def _run_logical_request(
        self,
        *,
        request_id: str | None,
        admitted_call: Callable[[_LogicalProviderCall], _LogicalResult],
    ) -> _LogicalResult:
        self.last_call_metrics = None
        self.last_transport_attempt_count = 0
        self.last_call_invocations = []
        resolved_request_id = normalize_provider_request_id(
            request_id,
            component=self.component_name,
        )
        self.last_request_id = resolved_request_id
        telemetry = ProviderRequestTelemetry.start(
            request_id=resolved_request_id,
            component=_telemetry_component(self.component_name),
            provider=self.provider_name,
            model=self.model_name,
            timeout_sec=self.timeout,
        )
        self.last_request_telemetry = telemetry.to_dict()
        admitted = False
        try:
            if not self.base_url or (self.require_api_key and not self.api_key):
                raise OpenAICompatibleServiceUnavailable(
                    f"{self.provider_name} OpenAI-compatible service is not configured"
                )
            admission = None
            if self.request_scheduler is not None:
                admission = self.request_scheduler.acquire(
                    resolved_request_id,
                    enqueued_at_unix_sec=telemetry.enqueued_at_unix_sec,
                )
                admitted = True
            admitted_at_monotonic = telemetry.admitted(admission)
            deadline = ProviderLogicalDeadline.start(
                request_id=resolved_request_id,
                component=self.component_name,
                timeout_sec=self.timeout,
                started_at_monotonic=admitted_at_monotonic,
            )
            telemetry.dispatched()
            result = admitted_call(
                _LogicalProviderCall(
                    request_id=resolved_request_id,
                    telemetry=telemetry,
                    deadline=deadline,
                )
            )
            self._require_deadline(
                deadline,
                phase="logical_call_completion",
                server_termination_status="not_applicable",
            )
            telemetry.completed()
            return result
        except Exception as exc:
            timeout_kind, termination_status, queue_wait_sec = _timeout_classification(exc)
            telemetry.failed(
                timed_out=timeout_kind is not None,
                error_type=type(exc).__name__,
                timeout_kind=timeout_kind,
                server_termination_status=termination_status,
                queue_wait_sec=queue_wait_sec,
            )
            if self.last_call_metrics is None:
                self.last_call_metrics = {
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "timeout_sec": self.timeout,
                    "attempts": self.last_transport_attempt_count,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            raise
        finally:
            if admitted:
                assert self.request_scheduler is not None
                self.request_scheduler.release()
            self._sync_request_telemetry(telemetry)

    def _call_admitted(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str] | None,
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | str | None,
        request_id: str,
        telemetry: ProviderRequestTelemetry,
        deadline: ProviderLogicalDeadline,
    ) -> str:

        payload = self._build_payload(
            system_prompt,
            user_text,
            audio_paths=audio_paths or [],
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
        started_at = time.monotonic()
        metrics = self._payload_metrics(system_prompt, user_text, audio_paths or [], payload)
        response_format_error: OpenAICompatibleEmptyContent | None = None
        try:
            text = self._invoke_payload(
                payload,
                purpose="primary",
                metrics=metrics,
                started_at=started_at,
                request_id=request_id,
                telemetry=telemetry,
                deadline=deadline,
            )
        except OpenAICompatibleEmptyContent as exc:
            if not response_format:
                raise
            text = ""
            response_format_error = exc
        if (
            (not text or not text.strip())
            and response_format
            and self.allow_response_format_fallback
        ):
            for fallback_response_format, fallback_label in _response_format_fallbacks(
                response_format
            ):
                fallback_payload = self._build_payload(
                    system_prompt,
                    user_text,
                    audio_paths=audio_paths or [],
                    response_format=fallback_response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                try:
                    text = self._invoke_payload(
                        fallback_payload,
                        purpose=f"response_format_fallback:{fallback_label}",
                        metrics={**metrics, "response_format_fallback": fallback_label},
                        started_at=started_at,
                        allow_response_format_retry=fallback_response_format is not None,
                        request_id=request_id,
                        telemetry=telemetry,
                        deadline=deadline,
                    )
                except OpenAICompatibleEmptyContent as exc:
                    response_format_error = exc
                    text = ""
                    continue
                if text and text.strip():
                    break
        self.last_raw_text = text
        if not text.strip():
            raise OpenAICompatibleEmptyContent(
                f"{self.provider_name} OpenAI-compatible service returned empty content"
            ) from response_format_error
        return text.strip()

    def call_json(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        repair_schema_name: str = "JSON",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.last_json_invocations = []
        return self._run_logical_request(
            request_id=request_id,
            admitted_call=lambda context: self._call_json_admitted(
                system_prompt,
                user_text,
                audio_paths=audio_paths,
                response_format=response_format,
                repair_schema_name=repair_schema_name,
                context=context,
            ),
        )

    def _call_json_admitted(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str] | None,
        response_format: dict[str, Any] | None,
        repair_schema_name: str,
        context: _LogicalProviderCall,
    ) -> dict[str, Any]:
        primary_start = len(self.last_call_invocations)
        try:
            text = self._call_admitted(
                system_prompt,
                user_text,
                audio_paths=audio_paths,
                response_format=response_format,
                tools=None,
                tool_choice=None,
                request_id=context.request_id,
                telemetry=context.telemetry,
                deadline=context.deadline,
            )
        finally:
            self._record_json_invocations("primary", start_index=primary_start)
        parsed = _parse_json_object(text)
        if parsed is not None:
            return parsed
        if not self.allow_json_repair:
            response_shape = (self.last_call_metrics or {}).get("response_shape", {})
            if response_shape.get("finish_reason") == "length":
                raise RuntimeError(
                    f"{self.provider_name} JSON response was truncated at "
                    f"max_tokens={self.max_tokens}; JSON repair is disabled"
                )
            raise RuntimeError(
                f"{self.provider_name} failed to produce valid JSON and JSON repair is disabled"
            )
        repair_system, repair_user = self._json_repair_prompts(
            bad_text=text,
            schema_name=repair_schema_name,
            original_system_prompt=system_prompt,
        )
        repair_start = len(self.last_call_invocations)
        try:
            repaired = self._call_admitted(
                repair_system,
                repair_user,
                audio_paths=None,
                response_format={"type": "json_object"},
                tools=None,
                tool_choice=None,
                request_id=context.request_id,
                telemetry=context.telemetry,
                deadline=context.deadline,
            )
        finally:
            self._record_json_invocations("json_repair", start_index=repair_start)
        parsed = _parse_json_object(repaired)
        if parsed is not None:
            return parsed
        raise RuntimeError(
            f"{self.provider_name} failed to produce valid JSON. Raw text:\n{text[:2000]}"
        )

    def _record_json_invocations(self, phase: str, *, start_index: int = 0) -> None:
        invocations = list(self.last_call_invocations[start_index:])
        if not invocations:
            invocations = [
                {
                    "purpose": "primary",
                    "transport_attempt_count": self.last_transport_attempt_count,
                    "status": (self.last_call_metrics or {}).get("status"),
                    "metrics": dict(self.last_call_metrics or {}),
                }
            ]
        for invocation in invocations:
            raw_purpose = str(invocation.get("purpose") or "primary")
            purpose = phase if raw_purpose == "primary" else f"{phase}_{raw_purpose}"
            self.last_json_invocations.append(
                {
                    **invocation,
                    "purpose": purpose,
                }
            )

    def _invoke_payload(
        self,
        payload: dict[str, Any],
        *,
        purpose: str,
        metrics: dict[str, Any],
        started_at: float,
        allow_response_format_retry: bool = True,
        request_id: str,
        telemetry: ProviderRequestTelemetry,
        deadline: ProviderLogicalDeadline,
    ) -> str:
        attempts_before = self.last_transport_attempt_count
        try:
            return self._post_payload(
                payload,
                metrics=metrics,
                started_at=started_at,
                allow_response_format_retry=allow_response_format_retry,
                request_id=request_id,
                telemetry=telemetry,
                deadline=deadline,
                purpose=purpose,
            )
        finally:
            call_metrics = dict(self.last_call_metrics or {})
            self.last_call_invocations.append(
                {
                    "purpose": purpose,
                    "transport_attempt_count": max(
                        0,
                        self.last_transport_attempt_count - attempts_before,
                    ),
                    "status": call_metrics.get("status"),
                    "metrics": call_metrics,
                }
            )

    def _sync_request_telemetry(self, telemetry: ProviderRequestTelemetry) -> None:
        request_telemetry = telemetry.to_dict()
        self.last_request_telemetry = request_telemetry
        metrics = dict(self.last_call_metrics or {})
        metrics["request_id"] = telemetry.request_id
        metrics["request_telemetry"] = request_telemetry
        for key in (
            "queue_wait_sec",
            "http_wall_sec",
            "logical_request_wall_sec",
        ):
            metrics[key] = request_telemetry[key]
        self.last_call_metrics = metrics
        for invocation_list in (
            self.last_call_invocations,
            self.last_json_invocations,
        ):
            for invocation in invocation_list:
                invocation_metrics = dict(invocation.get("metrics") or {})
                invocation_metrics["request_id"] = telemetry.request_id
                invocation_metrics["request_telemetry"] = request_telemetry
                invocation["metrics"] = invocation_metrics

    def repair_json_object(
        self,
        *,
        bad_text: str,
        schema_name: str,
        original_system_prompt: str,
    ) -> str:
        repair_system, repair_user = self._json_repair_prompts(
            bad_text=bad_text,
            schema_name=schema_name,
            original_system_prompt=original_system_prompt,
        )
        return self.call(repair_system, repair_user, response_format={"type": "json_object"})

    def _json_repair_prompts(
        self,
        *,
        bad_text: str,
        schema_name: str,
        original_system_prompt: str,
    ) -> tuple[str, str]:
        repair_system = (
            "You repair model output into valid JSON for X-Editor. "
            "Return only one JSON object. Do not add markdown, comments, or explanations."
        )
        repair_user = json.dumps(
            {
                "schema_name": schema_name,
                "original_system_prompt_excerpt": original_system_prompt[:2000],
                "bad_model_output": bad_text[:12000],
                "instruction": "Repair the bad output into a single valid JSON object matching the requested schema.",
            },
            ensure_ascii=False,
        )
        return repair_system, repair_user

    def _build_payload(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str],
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        user_content: str | list[dict[str, Any]]
        if audio_paths:
            user_content = [
                _audio_content_part(path, mode=self.audio_content_part)
                for path in audio_paths
            ]
            user_content.append({"type": "text", "text": user_text})
        else:
            user_content = user_text
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = copy.deepcopy(self.chat_template_kwargs)
        else:
            thinking = _thinking_payload_for_provider(self.provider_name)
            if thinking is not None:
                payload["thinking"] = thinking
                chat_template_kwargs = _chat_template_kwargs_for_provider(
                    self.provider_name,
                    thinking,
                )
                if chat_template_kwargs:
                    payload["chat_template_kwargs"] = chat_template_kwargs
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        payload.update(copy.deepcopy(self.request_body))
        return payload

    def _post_payload(
        self,
        payload: dict[str, Any],
        *,
        metrics: dict[str, Any],
        started_at: float,
        allow_response_format_retry: bool = True,
        request_id: str,
        telemetry: ProviderRequestTelemetry,
        deadline: ProviderLogicalDeadline,
        purpose: str,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        headers["X-Request-ID"] = request_id
        headers["X-X-Editor-Component"] = self.component_name
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        max_retries = (
            self.max_retries
            if self.max_retries is not None
            else _env_int("X_EDITOR_OPENAI_COMPAT_MAX_RETRIES", 1, minimum=0)
        )
        retry_backoff = (
            self.retry_backoff
            if self.retry_backoff is not None
            else _env_float(
                "X_EDITOR_OPENAI_COMPAT_RETRY_BACKOFF",
                0.5,
                minimum=0.0,
            )
        )
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            attempt_timeout = self._require_deadline(
                deadline,
                phase=f"{purpose}:http_attempt",
                server_termination_status="not_applicable",
            )
            self.last_transport_attempt_count += 1
            attempt_index = self.last_transport_attempt_count
            attempt_started_at = telemetry.http_attempt_started(
                attempt_index=attempt_index,
                attempt_in_invocation=attempt + 1,
                purpose=purpose,
                timeout_budget_sec=attempt_timeout,
            )
            try:
                response = _post_with_wall_timeout(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=attempt_timeout,
                )
            except requests.exceptions.Timeout as exc:
                deadline_expired = deadline.remaining_sec <= 0
                timeout_kind = (
                    "logical_request_deadline"
                    if deadline_expired
                    else "http_transport_timeout"
                )
                telemetry.http_attempt_finished(
                    started_at_monotonic=attempt_started_at,
                    attempt_index=attempt_index,
                    attempt_in_invocation=attempt + 1,
                    purpose=purpose,
                    outcome="timeout",
                    error_type=type(exc).__name__,
                    timeout_kind=timeout_kind,
                    server_termination_status="unconfirmed",
                )
                last_error = exc
                if deadline_expired:
                    self._set_failed_metrics(metrics, started_at, attempt + 1, exc)
                    self._require_deadline(
                        deadline,
                        phase=f"{purpose}:http_attempt",
                        server_termination_status="unconfirmed",
                    )
                if attempt < max_retries:
                    self._sleep_retry_backoff(
                        retry_backoff * (attempt + 1),
                        deadline=deadline,
                        phase=f"{purpose}:retry_backoff",
                    )
                    continue
                self._set_failed_metrics(metrics, started_at, attempt + 1, exc)
                raise OpenAICompatibleServiceUnavailable(
                    f"{self.provider_name} OpenAI-compatible request failed after {attempt + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                telemetry.http_attempt_finished(
                    started_at_monotonic=attempt_started_at,
                    attempt_index=attempt_index,
                    attempt_in_invocation=attempt + 1,
                    purpose=purpose,
                    outcome="error",
                    error_type=type(exc).__name__,
                )
                last_error = exc
                if attempt < max_retries:
                    self._sleep_retry_backoff(
                        retry_backoff * (attempt + 1),
                        deadline=deadline,
                        phase=f"{purpose}:retry_backoff",
                    )
                    continue
                self._set_failed_metrics(metrics, started_at, attempt + 1, exc)
                raise OpenAICompatibleServiceUnavailable(
                    f"{self.provider_name} OpenAI-compatible request failed after {attempt + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            except Exception as exc:
                telemetry.http_attempt_finished(
                    started_at_monotonic=attempt_started_at,
                    attempt_index=attempt_index,
                    attempt_in_invocation=attempt + 1,
                    purpose=purpose,
                    outcome="error",
                    error_type=type(exc).__name__,
                )
                raise
            telemetry.http_attempt_finished(
                started_at_monotonic=attempt_started_at,
                attempt_index=attempt_index,
                attempt_in_invocation=attempt + 1,
                purpose=purpose,
                outcome="response",
                http_status=response.status_code,
            )
            self._require_deadline(
                deadline,
                phase=f"{purpose}:http_response",
                server_termination_status="not_applicable",
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                self._sleep_retry_backoff(
                    retry_backoff * (attempt + 1),
                    deadline=deadline,
                    phase=f"{purpose}:retry_backoff",
                )
                continue
            telemetry.server_metrics = _reported_server_metrics(response, None)
            if (
                response.status_code in {400, 422}
                and allow_response_format_retry
                and payload.get("response_format")
                and "response_format" in response.text.lower()
            ):
                self.last_call_metrics = {
                    **metrics,
                    "attempts": attempt + 1,
                    "request_elapsed_sec": round(time.monotonic() - started_at, 3),
                    "status": "response_format_unsupported",
                    "http_status": response.status_code,
                }
                return ""
            break
        else:
            raise OpenAICompatibleServiceUnavailable(
                f"{self.provider_name} request failed: {last_error}"
            )

        if response.status_code != 200:
            self.last_call_metrics = {
                **metrics,
                "attempts": max_retries + 1,
                "request_elapsed_sec": round(time.monotonic() - started_at, 3),
                "status": "failed",
                "http_status": response.status_code,
            }
            raise OpenAICompatibleServiceUnavailable(
                f"{self.provider_name} OpenAI-compatible HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            self.last_call_metrics = {
                **metrics,
                "attempts": attempt + 1,
                "request_elapsed_sec": round(time.monotonic() - started_at, 3),
                "status": "invalid_json_response",
                "http_status": response.status_code,
            }
            raise OpenAICompatibleServiceUnavailable(
                f"{self.provider_name} OpenAI-compatible response was not valid JSON"
            ) from exc
        response_shape = _chat_response_shape(body)
        telemetry.server_metrics = _reported_server_metrics(response, body)
        try:
            text = _extract_chat_text(body)
        except OpenAICompatibleServiceUnavailable as exc:
            self.last_call_metrics = {
                **metrics,
                "attempts": attempt + 1,
                "request_elapsed_sec": round(time.monotonic() - started_at, 3),
                "status": "empty_content",
                "http_status": response.status_code,
                "response_shape": response_shape,
            }
            raise OpenAICompatibleEmptyContent(
                f"{self.provider_name} OpenAI-compatible response did not contain text content; "
                f"response_shape={_compact_json(response_shape)}"
            ) from exc
        self.last_call_metrics = {
            **metrics,
            "attempts": attempt + 1,
            "request_elapsed_sec": round(time.monotonic() - started_at, 3),
            "status": "ok",
            "http_status": response.status_code,
            "response_shape": response_shape,
        }
        return text

    def _require_deadline(
        self,
        deadline: ProviderLogicalDeadline,
        *,
        phase: str,
        server_termination_status: str,
    ) -> float:
        try:
            return deadline.require_remaining(
                phase=phase,
                server_termination_status=server_termination_status,
            )
        except ProviderLogicalDeadlineExceeded as exc:
            raise OpenAICompatibleServiceUnavailable(str(exc)) from exc

    def _sleep_retry_backoff(
        self,
        delay_sec: float,
        *,
        deadline: ProviderLogicalDeadline,
        phase: str,
    ) -> None:
        remaining_sec = self._require_deadline(
            deadline,
            phase=phase,
            server_termination_status="not_applicable",
        )
        if delay_sec > 0:
            time.sleep(min(delay_sec, remaining_sec))
        self._require_deadline(
            deadline,
            phase=phase,
            server_termination_status="not_applicable",
        )

    def _set_failed_metrics(
        self,
        metrics: dict[str, Any],
        started_at: float,
        attempts: int,
        error: Exception,
    ) -> None:
        self.last_call_metrics = {
            **metrics,
            "attempts": attempts,
            "request_elapsed_sec": round(time.monotonic() - started_at, 3),
            "status": "failed",
            "error_type": type(error).__name__,
        }

    def _payload_metrics(
        self,
        system_prompt: str,
        user_text: str,
        audio_paths: list[str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        audio_bytes = 0
        for path in audio_paths:
            try:
                audio_bytes += Path(path).stat().st_size
            except OSError:
                pass
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "host": urlparse(self.base_url).netloc or self.base_url,
            "timeout_sec": self.timeout,
            "prompt_chars": len(system_prompt) + len(user_text),
            "audio_count": len(audio_paths),
            "audio_bytes": audio_bytes,
            "payload_bytes": len(json.dumps(payload, ensure_ascii=False)),
        }


def _chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _audio_content_part(
    audio_path: str,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    data = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
    suffix = Path(audio_path).suffix.lower().lstrip(".") or "wav"
    resolved_mode = mode or os.getenv(
        "X_EDITOR_OPENAI_COMPAT_AUDIO_PART",
        "input_audio",
    )
    if resolved_mode == "audio_url":
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:{_audio_mime_type(audio_path)};base64,{data}"},
        }
    return {
        "type": "input_audio",
        "input_audio": {"data": data, "format": suffix},
    }


def _normalize_audio_content_part(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"input_audio", "audio_url"}:
        raise ValueError(
            "audio_content_part must be 'input_audio', 'audio_url', or None"
        )
    return normalized


_RESERVED_REQUEST_BODY_KEYS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "seed",
    "chat_template_kwargs",
    "response_format",
    "tools",
    "tool_choice",
}


def _normalize_request_body(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    normalized = {str(key).strip(): copy.deepcopy(item) for key, item in value.items()}
    if any(not key for key in normalized):
        raise ValueError("request_body keys must not be blank")
    reserved = sorted(set(normalized) & _RESERVED_REQUEST_BODY_KEYS)
    if reserved:
        raise ValueError(
            "request_body cannot override canonical request fields: "
            + ", ".join(reserved)
        )
    try:
        json.dumps(normalized, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("request_body must contain JSON-serializable values") from exc
    return normalized


def _extract_chat_text(result: dict[str, Any]) -> str:
    choices = result.get("choices") or []
    if not choices:
        raise OpenAICompatibleEmptyContent("OpenAI-compatible response returned no choices")
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        function = (tool_calls[0] or {}).get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            return arguments
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    raise OpenAICompatibleEmptyContent(
        "OpenAI-compatible response did not contain text content; "
        f"response_shape={_compact_json(_chat_response_shape(result))}"
    )


def _response_format_fallbacks(
    response_format: dict[str, Any],
) -> list[tuple[dict[str, Any] | None, str]]:
    format_type = str(response_format.get("type") or "").strip().lower()
    if format_type == "json_schema":
        return [({"type": "json_object"}, "json_object"), (None, "none")]
    if format_type == "json_object":
        return [(None, "none")]
    return [(None, "none")]


def _chat_response_shape(result: dict[str, Any]) -> dict[str, Any]:
    choices = result.get("choices")
    shape: dict[str, Any] = {
        "top_level_keys": sorted(str(key) for key in result)[:16],
        "choice_count": len(choices) if isinstance(choices, list) else 0,
    }
    if not isinstance(choices, list) or not choices:
        return shape
    choice = choices[0] if isinstance(choices[0], dict) else {}
    shape["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        shape["message_type"] = type(message).__name__
        return shape
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    shape.update(
        {
            "message_keys": sorted(str(key) for key in message)[:16],
            "content_type": type(content).__name__,
            "content_chars": len(content) if isinstance(content, str) else None,
            "content_parts": len(content) if isinstance(content, list) else None,
            "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
            "has_reasoning": bool(reasoning),
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
        }
    )
    return shape


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:1000]


def _telemetry_component(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"reasoner", "planner", "perception"}:
        return normalized
    return "unspecified"


def _timeout_classification(
    error: BaseException,
) -> tuple[str | None, str | None, float | None]:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderQueueTimeout):
            return "queue_timeout", "not_applicable", current.elapsed_sec
        if isinstance(current, ProviderLogicalDeadlineExceeded):
            return (
                "logical_request_deadline",
                current.server_termination_status,
                None,
            )
        if isinstance(current, (requests.exceptions.Timeout, TimeoutError)):
            return "http_transport_timeout", "unconfirmed", None
        current = current.__cause__ or current.__context__
    return None, None, None


def _reported_server_metrics(response: Any, body: dict[str, Any] | None) -> dict[str, Any]:
    """Return only timing and usage fields explicitly reported by the server."""

    result: dict[str, Any] = {}
    raw_headers = getattr(response, "headers", None)
    headers = (
        {str(key).lower(): str(value) for key, value in raw_headers.items()}
        if hasattr(raw_headers, "items")
        else {}
    )
    request_ids = {
        name: headers[name]
        for name in ("x-request-id", "x-vllm-request-id")
        if headers.get(name)
    }
    if request_ids:
        result["reported_request_ids"] = request_ids
    if headers.get("server-timing"):
        result["server_timing_header"] = headers["server-timing"]
    if not isinstance(body, dict):
        return result
    usage = body.get("usage")
    if isinstance(usage, dict):
        reported_usage = {
            str(key): value
            for key, value in usage.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if reported_usage:
            result["usage"] = reported_usage
    timing_fields: list[dict[str, Any]] = []
    for container_name in ("metrics", "timings"):
        container = body.get(container_name)
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            normalized = str(key).lower()
            if not any(
                token in normalized
                for token in ("queue", "prefill", "decode", "latency", "time_to_first")
            ):
                continue
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                continue
            timing_fields.append(
                {
                    "source": f"body.{container_name}.{key}",
                    "name": str(key),
                    "value": value,
                }
            )
    if timing_fields:
        result["reported_timing_fields"] = timing_fields
    return result


def _parse_json_object(text: str) -> dict[str, Any] | None:
    for candidate in _json_candidates(text):
        parsed = _try_parse_json_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _thinking_payload_for_provider(provider_name: str) -> dict[str, str] | None:
    raw = _provider_env_value(provider_name, "THINKING")
    if raw is None and _is_glm_provider(provider_name):
        raw = "disabled"
    if raw is None:
        return None
    value = raw.strip().lower().replace("-", "_")
    if value in {"", "none", "omit", "default", "auto"}:
        return None
    if value in {"0", "false", "no", "off", "disable", "disabled"}:
        return {"type": "disabled"}
    if value in {"1", "true", "yes", "on", "enable", "enabled"}:
        return {"type": "enabled"}
    return {"type": value}


def _chat_template_kwargs_for_provider(
    provider_name: str, thinking: dict[str, str]
) -> dict[str, bool] | None:
    if not _is_glm_provider(provider_name):
        return None
    value = str(thinking.get("type") or "").strip().lower().replace("-", "_")
    if value == "disabled":
        return {"enable_thinking": False}
    if value == "enabled":
        return {"enable_thinking": True}
    return None


def _provider_env_value(provider_name: str, suffix: str) -> str | None:
    normalized = provider_name.strip().upper().replace("-", "_")
    names = [
        f"X_EDITOR_{normalized}_{suffix}",
        f"X_EDITOR_OPENAI_COMPAT_{suffix}",
    ]
    if _is_glm_provider(provider_name):
        names.insert(0, f"X_EDITOR_GLM_{suffix}")
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


def _is_glm_provider(provider_name: str) -> bool:
    return provider_name.strip().lower().replace("-", "_") in {
        "glm",
        "glm_4_7_flash",
        "glm_47_flash",
        "glm_flash",
    }


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
