"""Executor-side service bootstrap for model-backed tool services.

The MCP wrappers are lightweight local processes, but many X-Editor tools rely
on separate HTTP model services.  This module lets the executor ensure those
services are online just-in-time instead of assuming every heavy backend is
already running before planning starts.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import requests

from x_editor.core.config import runtime_artifact_path
from x_editor.tools.mcp.client import resolve_command
from x_editor.tools.external_services import ExternalToolSetupRequired, require_external_service


OK_HEALTH_STATUSES = {"healthy", "ok", "running"}
DEFAULT_BOOTSTRAP_ROOT = Path(
    os.getenv("X_EDITOR_SERVICE_BOOTSTRAP_DIR", str(runtime_artifact_path("service_bootstrap")))
)


@dataclass
class ServiceProbe:
    ok: bool
    method: str
    url: str
    status_code: int | None = None
    text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "url": self.url,
            "status_code": self.status_code,
            "text": self.text,
            "error": self.error,
        }


@dataclass
class ServiceProbeResult:
    ok: bool
    probes: list[ServiceProbe]

    @property
    def last(self) -> ServiceProbe:
        if self.probes:
            return self.probes[-1]
        return ServiceProbe(False, "GET", "", error="not checked")

    def to_dicts(self) -> list[dict[str, Any]]:
        return [probe.to_dict() for probe in self.probes]


@dataclass
class ManagedServiceResult:
    name: str
    url: str
    status: str
    ok: bool
    started: bool = False
    pid: int | None = None
    log_path: str | None = None
    probes: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status,
            "ok": self.ok,
            "started": self.started,
            "pid": self.pid,
            "log_path": self.log_path,
            "probes": self.probes,
            "message": self.message,
        }


@dataclass
class ServiceBootstrapReport:
    tool_id: str
    ok: bool
    services: list[ManagedServiceResult] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "ok": self.ok,
            "services": [item.to_dict() for item in self.services],
            "skipped": self.skipped,
        }


class ServiceBootstrapAgent:
    """Start and smoke-test HTTP model services needed by a tool call."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        bootstrap_root: str | Path | None = None,
        request_timeout: float = 3.0,
        poll_interval: float = 1.0,
    ) -> None:
        if enabled is None:
            enabled = os.getenv("X_EDITOR_ENABLE_SERVICE_BOOTSTRAP", "1").lower() not in {"0", "false", "no"}
        self.enabled = enabled
        self.bootstrap_root = Path(bootstrap_root or DEFAULT_BOOTSTRAP_ROOT)
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self._lock = asyncio.Lock()

    async def ensure_tool_services(
        self,
        *,
        tool_id: str,
        tool_dir: Path,
        tool_config: dict[str, Any],
        arguments: dict[str, Any],
    ) -> ServiceBootstrapReport:
        if not self.enabled:
            return ServiceBootstrapReport(
                tool_id=tool_id,
                ok=True,
                skipped=[{"reason": "service bootstrap disabled"}],
            )
        specs = [
            spec
            for spec in tool_config.get("managed_services", []) or []
            if isinstance(spec, dict) and _matches_when(spec.get("when"), arguments)
        ]
        if not specs:
            return ServiceBootstrapReport(tool_id=tool_id, ok=True)
        async with self._lock:
            services = [
                await asyncio.to_thread(self._ensure_one_service, tool_id, tool_dir, spec, arguments, tool_config)
                for spec in specs
            ]
        return ServiceBootstrapReport(
            tool_id=tool_id,
            ok=all(item.ok for item in services),
            services=services,
        )

    def _ensure_one_service(
        self,
        tool_id: str,
        tool_dir: Path,
        spec: dict[str, Any],
        arguments: dict[str, Any],
        tool_config: dict[str, Any] | None = None,
    ) -> ManagedServiceResult:
        name = str(spec.get("name") or spec.get("id") or f"{tool_id}_service")
        if spec.get("external_tool"):
            try:
                require_external_service(spec["external_tool"])
            except ExternalToolSetupRequired as error:
                return ManagedServiceResult(name=name, url="", status="setup_required",
                                            ok=False, message=str(error))
        url = _service_url(spec, arguments, _tool_server_env(tool_config))
        if spec.get("external_only") and url != require_external_service(spec["external_tool"]):
            return ManagedServiceResult(name=name, url="", status="setup_required", ok=False,
                                        message="Service URL differs from the explicitly enabled external endpoint")
        startup_timeout = float(spec.get("startup_timeout_sec") or 120.0)
        health_probe = _probe_http_service(url, spec, timeout=self.request_timeout)
        if health_probe.ok:
            return ManagedServiceResult(
                name=name,
                url=url,
                status="already_online",
                ok=True,
                probes=health_probe.to_dicts(),
                message="service already online and smoke test passed",
            )
        if health_probe.probes and health_probe.probes[0].ok:
            return ManagedServiceResult(
                name=name,
                url=url,
                status="online_smoke_failed",
                ok=False,
                probes=health_probe.to_dicts(),
                message=(
                    f"{name} is online at {url}, but a smoke test failed "
                    f"({_probe_failure_summary(health_probe)}). Restart or point the URL at a healthy service."
                ),
            )
        if _service_responded_unhealthy(health_probe):
            return ManagedServiceResult(
                name=name,
                url=url,
                status="online_health_failed",
                ok=False,
                probes=health_probe.to_dicts(),
                message=(
                    f"{name} responded at {url}, but its health check is not ready "
                    f"({_probe_failure_summary(health_probe)}). Fix the running service configuration "
                    "or point the URL at a healthy service."
                ),
            )

        command = [] if spec.get("external_only") else _service_command(spec)
        if not command:
            hint = spec.get("setup_hint") or spec.get("manual_start")
            message = (
                f"{name} is unavailable at {url} ({_probe_failure_summary(health_probe)}), "
                "and no bootstrap command is configured. "
                "Set the configured command_env or start the service manually."
            )
            if spec.get("external_only"):
                message = f"{name} is unavailable at {url}; independently start or repair the operator-owned service. X·Edit will not install or launch it."
            if hint:
                message += f" Hint: {hint}"
            return ManagedServiceResult(
                name=name,
                url=url,
                status="offline_no_command",
                ok=False,
                probes=health_probe.to_dicts(),
                message=message,
            )

        working_dir = _resolve_working_dir(tool_dir, spec.get("working_dir"))
        env = _service_env(spec, _tool_server_env(tool_config))
        log_path = self.bootstrap_root / f"{_safe_name(name)}.log"
        pid_path = self.bootstrap_root / f"{_safe_name(name)}.pid"
        self.bootstrap_root.mkdir(parents=True, exist_ok=True)

        existing_pid = _read_pid(pid_path)
        if existing_pid and _pid_running(existing_pid):
            wait_probe = _wait_for_service(url, spec, startup_timeout, self.request_timeout, self.poll_interval)
            if wait_probe.ok:
                return ManagedServiceResult(
                    name=name,
                    url=url,
                    status="already_starting_became_online",
                    ok=True,
                    started=False,
                    pid=existing_pid,
                    log_path=str(log_path),
                    probes=health_probe.to_dicts() + wait_probe.to_dicts(),
                    message="existing service process became healthy",
                )

        resolved_command = resolve_command(command, str(working_dir) if working_dir else None)
        try:
            with log_path.open("ab") as log_file:
                process = subprocess.Popen(
                    resolved_command,
                    cwd=str(working_dir) if working_dir else None,
                    env={**os.environ, **env},
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception as exc:
            return ManagedServiceResult(
                name=name,
                url=url,
                status="start_failed",
                ok=False,
                log_path=str(log_path),
                probes=health_probe.to_dicts(),
                message=f"failed to start service command {resolved_command}: {type(exc).__name__}: {exc}",
            )

        pid_path.write_text(str(process.pid), encoding="utf-8")
        wait_probe = _wait_for_service(
            url,
            spec,
            startup_timeout,
            self.request_timeout,
            self.poll_interval,
            process=process,
        )
        if wait_probe.ok:
            return ManagedServiceResult(
                name=name,
                url=url,
                status="started",
                ok=True,
                started=True,
                pid=process.pid,
                log_path=str(log_path),
                probes=health_probe.to_dicts() + wait_probe.to_dicts(),
                message="service started and smoke test passed",
            )

        tail = _tail_text(log_path)
        return ManagedServiceResult(
            name=name,
            url=url,
            status="smoke_failed",
            ok=False,
            started=True,
            pid=process.pid,
            log_path=str(log_path),
            probes=health_probe.to_dicts() + wait_probe.to_dicts(),
            message=f"service did not pass smoke test within {startup_timeout:g}s. log_tail={tail}",
        )


def probe_managed_http_service(
    url: str,
    spec: Mapping[str, Any],
    *,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Probe a managed HTTP service without starting or restarting it."""

    if spec.get("external_tool"):
        try:
            require_external_service(spec["external_tool"])
        except ExternalToolSetupRequired as error:
            return {"ok": False, "url": "", "probes": [], "message": str(error),
                    "status": "setup_required"}
    service_url = str(url).rstrip("/")
    if spec.get("external_only") and service_url != require_external_service(spec["external_tool"]):
        return {"ok": False, "url": "", "probes": [], "status": "setup_required",
                "message": "Service URL differs from the explicitly enabled external endpoint"}
    probe = _probe_http_service(service_url, dict(spec), timeout=timeout)
    report: dict[str, Any] = {
        "ok": probe.ok,
        "url": service_url,
        "probes": probe.to_dicts(),
    }
    if not probe.ok:
        report["message"] = _probe_failure_summary(probe)
    return report


def _matches_when(when: Any, arguments: dict[str, Any]) -> bool:
    if not isinstance(when, dict) or not when:
        return True
    if not matches_environment(when):
        return False
    argument = when.get("argument")
    if not argument:
        return True
    value = arguments.get(str(argument), when.get("missing_value"))
    values = when.get("values")
    not_values = when.get("not_values")
    if not_values is not None and str(value) in {str(item) for item in not_values}:
        return False
    if values is None:
        return bool(value)
    return str(value) in {str(item) for item in values}


def matches_environment(when: Any) -> bool:
    if not isinstance(when, dict):
        return True
    for name, condition in when.get("environment", {}).items():
        if name == "X_EDITOR_EDIT_BACKEND":
            from x_editor.tools.edit_backend import selected_edit_backend

            value = selected_edit_backend()
        else:
            value = os.getenv(name, condition.get("default", ""))
        if value not in condition["values"]:
            return False
    return True


def _service_url(spec: dict[str, Any], arguments: dict[str, Any], tool_env: dict[str, str] | None = None) -> str:
    argument_key = spec.get("argument_url_key")
    if argument_key and arguments.get(str(argument_key)):
        return str(arguments[str(argument_key)]).rstrip("/")
    env_name = spec.get("url_env")
    if env_name and os.getenv(str(env_name)):
        return str(os.environ[str(env_name)]).rstrip("/")
    if env_name and tool_env and tool_env.get(str(env_name)):
        return str(tool_env[str(env_name)]).rstrip("/")
    value = spec.get("url") or spec.get("default_url")
    if not value:
        raise ValueError(f"managed service {spec.get('name') or spec.get('id')} missing url/default_url")
    return str(value).rstrip("/")


def _tool_server_env(tool_config: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(tool_config, dict):
        return {}
    server = tool_config.get("server")
    if not isinstance(server, dict):
        return {}
    env = server.get("env")
    if not isinstance(env, dict):
        return {}
    return {str(key): str(value) for key, value in env.items() if value is not None}


def _service_command(spec: dict[str, Any]) -> list[str]:
    env_name = spec.get("command_env")
    if env_name and os.getenv(str(env_name)):
        return shlex.split(os.environ[str(env_name)])
    command = spec.get("command")
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list):
        return [str(item) for item in command]
    return []


def _service_env(spec: dict[str, Any], tool_env: dict[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    if tool_env:
        env.update(
            {
                str(key): os.environ.get(str(key), str(value))
                for key, value in tool_env.items()
            }
        )
    raw = spec.get("env") or {}
    if not isinstance(raw, dict):
        return env
    env.update(
        {
            str(key): os.environ.get(str(key), str(value))
            for key, value in raw.items()
        }
    )
    return env


def _resolve_working_dir(tool_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = tool_dir / path
    return path.resolve()


def _probe_http_service(url: str, spec: dict[str, Any], *, timeout: float) -> ServiceProbeResult:
    probes: list[ServiceProbe] = []
    health_path = str(spec.get("health_path") or "/health")
    health = _request_probe("GET", f"{url}{health_path}", timeout=timeout, expect_health=True)
    probes.append(health)
    if not health.ok:
        return ServiceProbeResult(False, probes)
    for smoke in spec.get("smoke", []) or []:
        method, path, json_body, smoke_timeout = _smoke_request(smoke)
        probe = _request_probe(
            method,
            f"{url}{path}",
            timeout=smoke_timeout or timeout,
            expect_health=path == health_path,
            json_body=json_body,
        )
        probes.append(probe)
        if not probe.ok:
            return ServiceProbeResult(False, probes)
    return ServiceProbeResult(True, probes)


def _request_probe(
    method: str,
    url: str,
    *,
    timeout: float,
    expect_health: bool = False,
    json_body: Any = None,
) -> ServiceProbe:
    method = method.upper()
    try:
        response = requests.request(method, url, timeout=timeout, json=json_body)
    except Exception as exc:
        return ServiceProbe(False, method, url, error=f"{type(exc).__name__}: {exc}")
    ok = response.status_code < 500 and response.status_code != 404
    if method == "OPTIONS" and response.status_code == 405:
        ok = True
    text = response.text[:500]
    if ok and expect_health:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        status = str(payload.get("status") or "").lower() if isinstance(payload, dict) else ""
        if status and status not in OK_HEALTH_STATUSES:
            ok = False
    return ServiceProbe(ok, method, url, status_code=response.status_code, text=text)


def _probe_failure_summary(result: ServiceProbeResult) -> str:
    failed = [probe for probe in result.probes if not probe.ok]
    probe = failed[-1] if failed else result.last
    if probe.error:
        return probe.error
    if probe.status_code is not None:
        text = f"status={probe.status_code}"
        if probe.text:
            text += f" body={probe.text[:160]}"
        return text
    return "health/smoke probe failed"


def _service_responded_unhealthy(result: ServiceProbeResult) -> bool:
    if not result.probes:
        return False
    health = result.probes[0]
    if health.ok or health.error:
        return False
    if health.status_code is None:
        return False
    return health.status_code < 500 and health.status_code != 404


def _smoke_method_path(smoke: Any) -> tuple[str, str]:
    if isinstance(smoke, str):
        return "GET", smoke
    if isinstance(smoke, dict):
        return str(smoke.get("method") or "GET"), str(smoke.get("path") or "/health")
    return "GET", "/health"


def _smoke_request(smoke: Any) -> tuple[str, str, Any, float | None]:
    if isinstance(smoke, str):
        return "GET", smoke, None, None
    if isinstance(smoke, dict):
        timeout = smoke.get("timeout_sec")
        return (
            str(smoke.get("method") or "GET"),
            str(smoke.get("path") or "/health"),
            smoke.get("json"),
            float(timeout) if timeout is not None else None,
        )
    return "GET", "/health", None, None


def _wait_for_service(
    url: str,
    spec: dict[str, Any],
    startup_timeout: float,
    request_timeout: float,
    poll_interval: float,
    *,
    process: subprocess.Popen[bytes] | None = None,
) -> ServiceProbeResult:
    deadline = time.monotonic() + startup_timeout
    last_probe = ServiceProbeResult(
        False,
        [ServiceProbe(False, "GET", f"{url}{spec.get('health_path') or '/health'}", error="not checked")],
    )
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return ServiceProbeResult(
                False,
                [
                    ServiceProbe(
                        False,
                        "GET",
                        f"{url}{spec.get('health_path') or '/health'}",
                        error=f"service process exited early with code {process.returncode}",
                    )
                ],
            )
        last_probe = _probe_http_service(url, spec, timeout=request_timeout)
        if last_probe.ok:
            return last_probe
        time.sleep(poll_interval)
    return last_probe


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _tail_text(path: Path, max_bytes: int = 2000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-max_bytes:].decode("utf-8", "replace")
