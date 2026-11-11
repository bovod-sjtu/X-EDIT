"""HTTP client for an official ACE-Step 1.5 API service."""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


TASK_TYPES = {"text2music", "cover", "repaint"}
FORM_FILE_FIELDS = {
    "reference_audio_path": ("reference_audio", "rb"),
    "source_audio_path": ("src_audio", "rb"),
}
DEFAULT_ACE_STEP_API_URL = "http://127.0.0.1:8044"
REQUIRED_API_PATHS = ("/release_task", "/query_result")
API_URL_ENV_NAMES = ("ACE_STEP_API_URL", "ACE_STEP_UPSTREAM_URL", "ACE_STEP_SERVICE_URL")


class AceStepMusicClient:
    """Thin client for ACE-Step's release_task/query_result/v1/audio API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 1800.0,
        poll_interval: float = 2.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = _resolve_api_url(base_url).rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.api_key = api_key if api_key is not None else os.getenv("ACE_STEP_API_KEY")

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", headers=self._headers(), timeout=10)
        resp.raise_for_status()
        if not resp.content:
            return {"status": "ok", "body": "", "service_url": self.base_url}
        try:
            return resp.json()
        except ValueError:
            return {"status": "ok", "body": resp.text[:800], "service_url": self.base_url}

    def validate_api_compatibility(self) -> dict[str, Any]:
        """Verify that the configured endpoint is the official ACE-Step REST API."""

        openapi_url = f"{self.base_url}/openapi.json"
        try:
            resp = requests.get(openapi_url, headers=self._headers(), timeout=10)
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not inspect ACE-Step API routes at {openapi_url}: {exc}") from exc

        if resp.status_code == 200:
            try:
                spec = resp.json()
            except ValueError:
                spec = {}
            paths = spec.get("paths") if isinstance(spec, dict) else {}
            if isinstance(paths, dict):
                path_set = set(paths)
                missing = [path for path in REQUIRED_API_PATHS if path not in path_set]
                if missing:
                    raise RuntimeError(_wrong_endpoint_message(self.base_url, missing, spec, path_set))
                return {
                    "status": "compatible",
                    "service_url": self.base_url,
                    "checked_by": "openapi",
                    "required_paths": list(REQUIRED_API_PATHS),
                }

        missing = []
        for path in REQUIRED_API_PATHS:
            try:
                route = requests.options(f"{self.base_url}{path}", headers=self._headers(), timeout=10)
            except requests.RequestException as exc:
                raise RuntimeError(f"Could not inspect ACE-Step API route {path} at {self.base_url}: {exc}") from exc
            if route.status_code == 404:
                missing.append(path)
        if missing:
            raise RuntimeError(_wrong_endpoint_message(self.base_url, missing, None, set()))
        return {
            "status": "compatible",
            "service_url": self.base_url,
            "checked_by": "options",
            "required_paths": list(REQUIRED_API_PATHS),
        }

    def generate_or_edit(self, **kwargs: Any) -> dict[str, Any]:
        timeout = float(kwargs.get("timeout_seconds") or self.timeout)
        poll_interval = float(kwargs.get("poll_interval_seconds") or self.poll_interval)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._check_local_audio(kwargs)
        self._check_health_for_generation()

        payload = _build_payload(kwargs)
        task_id = self._release_task(payload, kwargs, timeout=timeout)
        result_item = self._wait_for_result(task_id, timeout=timeout, poll_interval=poll_interval)
        result_file = result_item.get("file")
        if not isinstance(result_file, str) or not result_file:
            raise RuntimeError(f"ACE-Step task {task_id} succeeded but did not return a file URL: {result_item}")
        self._download_audio(result_file, output_path, timeout=timeout)

        return {
            "output_audio_path": str(output_path),
            "backend_used": "ace_step_1_5_api",
            "task_id": task_id,
            "task_type": payload.get("task_type"),
            "mode": kwargs.get("mode", payload.get("task_type")),
            "vocal_style": kwargs.get("vocal_style", "music_vocal"),
            "result_file": result_file,
            "service_url": self.base_url,
            "warning": (
                "ACE-Step generates/edits music vocals best-effort; X-Editor has not verified "
                "speaker identity, exact timing, lyric accuracy, or copyright suitability."
            ),
        }

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _check_health_for_generation(self) -> None:
        try:
            self.health()
            self.validate_api_compatibility()
        except Exception as exc:
            raise RuntimeError(
                "ACE-Step API is not available at "
                f"{self.base_url}. Start the official service on a non-conflicting port, for example "
                "`PORT=8044 uv run acestep-api`, or set ACE_STEP_API_URL/ACE_STEP_UPSTREAM_URL "
                "to a running official ACE-Step endpoint. "
                f"Reason: {type(exc).__name__}: {exc}"
            ) from exc

    def _check_local_audio(self, kwargs: dict[str, Any]) -> None:
        for key in ("source_audio_path", "reference_audio_path"):
            value = kwargs.get(key)
            if value:
                path = Path(str(value))
                if not path.exists():
                    raise FileNotFoundError(f"{key} not found: {path}")

        mode = _normalize_task_type(kwargs.get("mode"))
        if mode in {"cover", "repaint"} and not kwargs.get("source_audio_path"):
            raise ValueError(f"ACE-Step mode '{mode}' requires source_audio_path")
        if mode == "repaint":
            if kwargs.get("repainting_start") is None:
                raise ValueError("ACE-Step repaint mode requires repainting_start")
            if kwargs.get("repainting_end") is None:
                raise ValueError("ACE-Step repaint mode requires repainting_end")

    def _release_task(self, payload: dict[str, Any], kwargs: dict[str, Any], timeout: float) -> str:
        url = f"{self.base_url}/release_task"
        files: dict[str, tuple[str, Any]] = {}
        has_uploads = any(kwargs.get(key) for key in FORM_FILE_FIELDS)
        try:
            if has_uploads:
                with ExitStack() as stack:
                    for arg_key, (form_key, mode) in FORM_FILE_FIELDS.items():
                        audio_path = kwargs.get(arg_key)
                        if not audio_path:
                            continue
                        opened = stack.enter_context(open(audio_path, mode))
                        files[form_key] = (Path(str(audio_path)).name, opened)
                    resp = requests.post(
                        url,
                        data=_form_payload(payload),
                        files=files,
                        headers=self._headers(),
                        timeout=timeout,
                    )
            else:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to submit ACE-Step task to {url}: {exc}") from exc
        data = _unwrap_response(resp, service="ACE-Step release_task")
        task_id = _extract_task_id(data)
        if not task_id:
            raise RuntimeError(f"ACE-Step release_task did not return task_id: {data}")
        return task_id

    def _wait_for_result(self, task_id: str, timeout: float, poll_interval: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/query_result"
        query_timeout = _query_result_timeout(poll_interval)
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"ACE-Step task {task_id} timed out after {timeout:g}s")
            try:
                resp = requests.post(
                    url,
                    json={"task_id_list": [task_id]},
                    headers=self._headers(),
                    timeout=min(query_timeout, max(1.0, deadline - time.monotonic())),
                )
            except requests.Timeout:
                time.sleep(poll_interval)
                continue
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed while polling ACE-Step task {task_id}: {exc}") from exc
            data = _unwrap_response(resp, service="ACE-Step query_result")
            task_record = _extract_task_record(data, task_id)
            if task_record is None:
                time.sleep(poll_interval)
                continue
            status = task_record.get("status")
            if str(status) == "1":
                return _first_result_item(task_record, task_id)
            if str(status) == "2":
                raise RuntimeError(f"ACE-Step task {task_id} failed: {task_record}")
            time.sleep(poll_interval)

    def _download_audio(self, file_ref: str, output_path: Path, timeout: float) -> None:
        url = file_ref if file_ref.startswith(("http://", "https://")) else urljoin(f"{self.base_url}/", file_ref.lstrip("/"))
        try:
            resp = requests.get(url, headers=self._headers(), timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download ACE-Step audio from {url}: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"ACE-Step audio download error {resp.status_code}: {resp.text[:800]}")
        output_path.write_bytes(resp.content)
        if output_path.stat().st_size == 0:
            raise RuntimeError(f"ACE-Step downloaded empty audio file to {output_path}")


def _build_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    task_type = _normalize_task_type(kwargs.get("mode"))
    vocal_style = str(kwargs.get("vocal_style") or "music_vocal")
    instruction = str(kwargs.get("instruction") or "").strip()
    prompt = str(kwargs.get("music_prompt") or "").strip() or _default_music_prompt(vocal_style, instruction)

    payload: dict[str, Any] = {
        "prompt": prompt,
        "lyrics": kwargs.get("lyrics", ""),
        "vocal_language": kwargs.get("vocal_language", "unknown"),
        "audio_format": kwargs.get("audio_format", "wav"),
        "task_type": task_type,
        "instruction": instruction or _default_instruction(task_type, vocal_style),
        "thinking": bool(kwargs.get("thinking", True)),
        "use_format": bool(kwargs.get("use_format", True)),
        "audio_duration": kwargs.get("duration_seconds", 10.0),
        "inference_steps": kwargs.get("inference_steps", 8),
        "guidance_scale": kwargs.get("guidance_scale", 7.0),
        "seed": kwargs.get("seed", -1),
        "use_random_seed": bool(kwargs.get("use_random_seed", True)),
        "audio_cover_strength": kwargs.get("audio_cover_strength", 0.8),
        "repainting_start": kwargs.get("repainting_start", 0.0),
        "repainting_end": kwargs.get("repainting_end", -1.0),
    }
    if kwargs.get("reference_audio_path"):
        payload["reference_audio_path"] = kwargs["reference_audio_path"]
    if kwargs.get("source_audio_path"):
        payload["src_audio_path"] = kwargs["source_audio_path"]
    for key in ("bpm", "key_scale", "time_signature", "model"):
        if kwargs.get(key) not in (None, ""):
            payload[key] = kwargs[key]
    return {key: value for key, value in payload.items() if value is not None}


def _resolve_api_url(explicit: str | None = None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    for name in API_URL_ENV_NAMES:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return DEFAULT_ACE_STEP_API_URL


def _query_result_timeout(poll_interval: float) -> float:
    """Allow ACE-Step lazy-load/download phases to hold a poll request open."""

    return min(120.0, max(30.0, float(poll_interval) + 30.0))


def _wrong_endpoint_message(
    base_url: str,
    missing_paths: list[str],
    openapi_spec: dict[str, Any] | None,
    path_set: set[str],
) -> str:
    title = None
    if isinstance(openapi_spec, dict):
        info = openapi_spec.get("info")
        if isinstance(info, dict):
            title = info.get("title")
    sample_paths = ", ".join(sorted(path_set)[:8]) if path_set else "<openapi unavailable>"
    title_text = f" It reports title={title!r}." if title else ""
    return (
        f"Configured ACE-Step API endpoint {base_url} does not expose required route(s): "
        f"{', '.join(missing_paths)}.{title_text} Sample paths: {sample_paths}. "
        "This usually means ACE_STEP_API_URL points at another service, such as an OpenAI/vLLM server. "
        "Start the official ACE-Step REST API and set ACE_STEP_API_URL to that base URL."
    )


def _normalize_task_type(value: Any) -> str:
    task_type = str(value or "text2music").strip().lower()
    aliases = {"generate": "text2music", "generation": "text2music", "music": "text2music"}
    task_type = aliases.get(task_type, task_type)
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported ACE-Step mode/task_type: {value}. Expected one of {sorted(TASK_TYPES)}")
    return task_type


def _default_music_prompt(vocal_style: str, instruction: str) -> str:
    lowered = f"{vocal_style} {instruction}".lower()
    if "rap" in lowered or "hip hop" in lowered or "说唱" in lowered:
        return "hip hop beat with confident rap vocals, clear rhythm, studio mix"
    return "short pop song with expressive lead singing vocal, clear lyrics, studio mix"


def _default_instruction(task_type: str, vocal_style: str) -> str:
    if task_type == "cover":
        return f"Create a {vocal_style} cover from the source audio."
    if task_type == "repaint":
        return f"Repaint the selected region as {vocal_style} music vocals."
    return f"Generate {vocal_style} music vocals from the prompt and lyrics."


def _form_payload(payload: dict[str, Any]) -> dict[str, str]:
    form: dict[str, str] = {}
    for key, value in payload.items():
        if key in {"reference_audio_path", "src_audio_path"}:
            continue
        if isinstance(value, bool):
            form[key] = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            form[key] = json.dumps(value, ensure_ascii=False)
        else:
            form[key] = str(value)
    return form


def _unwrap_response(resp: requests.Response, service: str) -> Any:
    if resp.status_code != 200:
        raise RuntimeError(f"{service} error {resp.status_code}: {resp.text[:800]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"{service} returned non-JSON response: {resp.text[:800]}") from exc
    if isinstance(payload, dict) and "code" in payload and payload.get("code") not in (None, 0, 200):
        raise RuntimeError(f"{service} returned code={payload.get('code')}: {payload.get('error') or payload}")
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"{service} returned error: {payload.get('error')}")
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def _extract_task_id(data: Any) -> str | None:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("task_id", "taskId", "id"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _extract_task_record(data: Any, task_id: str) -> dict[str, Any] | None:
    records: list[Any]
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("results", "task_results", "tasks", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            records = [data]
    else:
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        record_task_id = record.get("task_id") or record.get("taskId") or record.get("id")
        if record_task_id in (None, task_id):
            return record
    return None


def _first_result_item(task_record: dict[str, Any], task_id: str) -> dict[str, Any]:
    result = task_record.get("result") or task_record.get("results")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError as exc:
            raise RuntimeError(f"ACE-Step task {task_id} returned unparsable result JSON: {result[:800]}") from exc
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first
    if isinstance(result, dict):
        return result
    raise RuntimeError(f"ACE-Step task {task_id} succeeded but result is empty: {task_record}")
