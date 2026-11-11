"""Perception-side managed service bootstrap helpers."""

from __future__ import annotations

import os
import base64
import io
import wave
from pathlib import Path
from typing import Any

from x_editor.tools.service_bootstrap import ManagedServiceResult, ServiceBootstrapAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = PROJECT_ROOT / "x_editor" / "tools" / "catalog"


class PerceptionServiceBootstrapper:
    """Synchronous bootstrap facade for perception clients.

    The main MCP ``ToolExecutor`` already starts edit tools on demand. Perception
    calls are direct HTTP calls, so they need a matching light wrapper here.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        agent: ServiceBootstrapAgent | None = None,
    ) -> None:
        if enabled is None:
            raw = os.getenv("X_EDITOR_ENABLE_PERCEPTION_SERVICE_BOOTSTRAP")
            if raw is None:
                raw = os.getenv("X_EDITOR_ENABLE_SERVICE_BOOTSTRAP", "1")
            enabled = raw.strip().lower() not in {"0", "false", "no"}
        self.enabled = bool(enabled)
        self.agent = agent or ServiceBootstrapAgent(enabled=self.enabled)
        self.reports: list[dict[str, Any]] = []

    def ensure(
        self,
        service: str,
        *,
        url: str | None = None,
        arguments: dict[str, Any] | None = None,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Ensure one perception service is healthy and return its report."""

        if not self.enabled:
            report = {
                "tool_id": f"perception.{service}",
                "ok": True,
                "services": [],
                "skipped": [{"reason": "perception service bootstrap disabled"}],
            }
            self.reports.append(report)
            return report
        spec = perception_service_spec(service, url=url, spec_overrides=spec_overrides)
        if not spec:
            return None
        tool_id = f"perception.{service}"
        tool_dir = _service_tool_dir(service)
        result: ManagedServiceResult = self.agent._ensure_one_service(
            tool_id,
            tool_dir,
            spec,
            arguments or {},
            {"server": {"env": _service_env(service)}},
        )
        report = {
            "tool_id": tool_id,
            "ok": result.ok,
            "services": [result.to_dict()],
            "skipped": [],
        }
        self.reports.append(report)
        return report


def perception_service_spec(
    service: str,
    *,
    url: str | None = None,
    spec_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the managed service spec used by perception clients."""

    spec = _service_spec(service, url=url)
    if not spec:
        return None
    if spec_overrides:
        spec = {**spec, **spec_overrides}
    return spec


def _service_tool_dir(service: str) -> Path:
    if service == "diarization":
        return CATALOG_DIR / "diarization"
    if service == "sedicow_asr":
        return CATALOG_DIR / "sedicow_asr"
    if service == "moss_transcribe_diarize":
        return CATALOG_DIR / "moss_transcribe_diarize"
    return PROJECT_ROOT


def _service_spec(service: str, *, url: str | None = None) -> dict[str, Any] | None:
    if service == "qwen3_asr":
        return {
            "name": "qwen3_asr",
            "url_env": "EDIT_PIPELINE_ASR_URL",
            "default_url": url or "http://127.0.0.1:8010",
            "command_env": "EDIT_PIPELINE_ASR_SERVICE_COMMAND",
            "command": [
                "bash",
                "-lc",
                (
                    'cd "${EDIT_DATA_PIPELINE_ROOT:?set EDIT_DATA_PIPELINE_ROOT}" '
                    '&& exec "${EDIT_PIPELINE_ASR_PYTHON:-envs/pipe-asr/.venv/bin/python}" '
                    "tools/asr-service/asr_service.py "
                    '--host "${EDIT_PIPELINE_ASR_HOST:-127.0.0.1}" '
                    '--port "${EDIT_PIPELINE_ASR_PORT:-8010}" '
                    "${EDIT_PIPELINE_ASR_USE_VLLM_FLAG:---use-vllm} "
                    '--max-model-len "${EDIT_PIPELINE_ASR_MAX_MODEL_LEN:-12000}" '
                    '--gpu-memory-utilization "${EDIT_PIPELINE_ASR_GPU_MEMORY_UTILIZATION:-0.8}" '
                    '--buffer-batch-size "${EDIT_PIPELINE_ASR_BUFFER_BATCH_SIZE:-16}" '
                    '--buffer-wait-ms "${EDIT_PIPELINE_ASR_BUFFER_WAIT_MS:-500}"'
                ),
            ],
            "health_path": "/health",
            "smoke": [
                {
                    "method": "POST",
                    "path": "/transcribe",
                    "json": {
                        "audio": {"data": _silence_wav_b64(), "format": "wav"},
                        "language": None,
                    },
                    "timeout_sec": 30,
                }
            ],
            "startup_timeout_sec": 600,
            "setup_hint": "set EDIT_DATA_PIPELINE_ROOT and ensure envs/pipe-asr is installed, or set EDIT_PIPELINE_ASR_SERVICE_COMMAND",
        }
    if service == "diarization":
        return {
            "name": "diarization",
            "url_env": "X_EDITOR_DIARIZATION_URL",
            "default_url": url or "http://127.0.0.1:8041",
            "working_dir": str(CATALOG_DIR / "diarization"),
            "command_env": "X_EDITOR_DIARIZATION_SERVICE_COMMAND",
            "command": [".venv/bin/python", "service.py", "--host", "127.0.0.1", "--port", "8041"],
            "env": {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
            "health_path": "/health",
            "smoke": [{"method": "OPTIONS", "path": "/diarize"}],
            "startup_timeout_sec": 360,
            "setup_hint": "cd x_editor/tools/catalog/diarization && ./setup.sh",
        }
    if service == "sedicow_asr":
        return {
            "name": "sedicow_asr",
            "url_env": "X_EDITOR_SEDICOW_URL",
            "default_url": url or "http://127.0.0.1:8046",
            "working_dir": str(CATALOG_DIR / "sedicow_asr"),
            "command_env": "X_EDITOR_SEDICOW_SERVICE_COMMAND",
            "command": [
                "bash",
                "-lc",
                'exec "${SEDICOW_SERVICE_PYTHON:-python}" service.py --host 127.0.0.1 --port "${X_EDITOR_SEDICOW_PORT:-8046}"',
            ],
            "health_path": "/health",
            "smoke": [{"method": "OPTIONS", "path": "/transcribe_speaker"}],
            "startup_timeout_sec": 300,
            "setup_hint": (
                "cd x_editor/tools/catalog/sedicow_asr && ./setup.sh; set SEDICOW_REPO and "
                "SEDICOW_INFERENCE_PYTHON; see docs/DEPLOYMENT.md for the pinned service stack"
            ),
        }
    if service == "moss_transcribe_diarize":
        return {
            "name": "moss_transcribe_diarize",
            "url_env": "X_EDITOR_MOSS_TRANSCRIBE_URL",
            "default_url": url or "http://127.0.0.1:8048",
            "working_dir": str(CATALOG_DIR / "moss_transcribe_diarize"),
            "command_env": "X_EDITOR_MOSS_TRANSCRIBE_SERVICE_COMMAND",
            "command": [
                "bash",
                "-lc",
                (
                    'exec "${MOSS_TRANSCRIBE_SERVICE_PYTHON:-python}" service.py '
                    '--host 127.0.0.1 --port "${X_EDITOR_MOSS_TRANSCRIBE_PORT:-8048}"'
                ),
            ],
            "health_path": "/health",
            "smoke": [{"method": "OPTIONS", "path": "/transcribe_diarize"}],
            "startup_timeout_sec": 300,
            "setup_hint": (
                "cd x_editor/tools/catalog/moss_transcribe_diarize && ./setup.sh; "
                "set MOSS_OPENAI_BASE_URL for SGLang/vLLM, or set MOSS_TRANSCRIBE_COMMAND"
            ),
        }
    return None


def _service_env(service: str) -> dict[str, str]:
    if service == "qwen3_asr":
        env: dict[str, str] = {}
        if os.getenv("EDIT_DATA_PIPELINE_ROOT"):
            env["EDIT_DATA_PIPELINE_ROOT"] = os.environ["EDIT_DATA_PIPELINE_ROOT"]
        if os.getenv("EDIT_PIPELINE_ASR_URL"):
            env["EDIT_PIPELINE_ASR_URL"] = os.environ["EDIT_PIPELINE_ASR_URL"]
        return env
    if service == "diarization":
        env = {
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"),
        }
        if os.getenv("X_EDITOR_DIARIZATION_URL"):
            env["X_EDITOR_DIARIZATION_URL"] = os.environ["X_EDITOR_DIARIZATION_URL"]
        return env
    if service == "sedicow_asr":
        env = {}
        for key in (
            "X_EDITOR_SEDICOW_URL",
            "X_EDITOR_SEDICOW_PORT",
            "SEDICOW_SERVICE_PYTHON",
            "SEDICOW_REPO",
            "SEDICOW_TRANSCRIBE_COMMAND",
            "SEDICOW_INFERENCE_PYTHON",
            "SEDICOW_HF_HOME",
            "SEDICOW_HF_MODULES_CACHE",
            "SEDICOW_DICOW_MODEL",
            "SEDICOW_DIARIZATION_MODEL",
            "SEDICOW_DIARIZATION_CACHE_DIR",
            "SEDICOW_DEVICE",
            "SEDICOW_EMBEDDING_BATCH_SIZE",
            "SEDICOW_SEGMENTATION_BATCH_SIZE",
            "SEDICOW_RETURN_ALL_SEGMENTS",
            "PYTHONPATH",
        ):
            if os.getenv(key):
                env[key] = os.environ[key]
        return env
    if service == "moss_transcribe_diarize":
        env = {}
        for key in (
            "X_EDITOR_MOSS_TRANSCRIBE_URL",
            "X_EDITOR_MOSS_TRANSCRIBE_PORT",
            "MOSS_TRANSCRIBE_SERVICE_PYTHON",
            "MOSS_TRANSCRIBE_INFERENCE_PYTHON",
            "MOSS_TRANSCRIBE_COMMAND",
            "MOSS_TRANSCRIBE_TIMEOUT",
            "MOSS_OPENAI_BASE_URL",
            "MOSS_OPENAI_MODEL",
            "MOSS_RESPONSE_FORMAT",
            "MOSS_MAX_NEW_TOKENS",
            "MOSS_TEMPERATURE",
            "MOSS_REPO",
            "MOSS_MODEL_ID",
            "MOSS_DEVICE",
            "MOSS_DTYPE",
            "PYTHONPATH",
        ):
            if os.getenv(key):
                env[key] = os.environ[key]
        return env
    return {}


def _silence_wav_b64() -> str:
    """Return a tiny valid WAV used to smoke-test ASR inference, not just HTTP."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
