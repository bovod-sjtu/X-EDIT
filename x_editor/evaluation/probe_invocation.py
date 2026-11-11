"""Opt-in immutable input/output capture around existing module implementations.

The proxy supplies no prompt, retries, or alternative implementation. Explicit
research callers attach it; the production workflow does not import this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported checkpoint payload type: {type(value).__name__}")


class RecordedModule:
    def __init__(self, component: Any, directory: Path) -> None:
        self.component = component
        self.directory = directory
        self.sequence = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.component, name)

    def _start(self, method: str, args: tuple, kwargs: dict) -> tuple[Path, float]:
        self.sequence += 1
        destination = self.directory / f"{self.sequence:04d}"
        destination.mkdir(parents=True, exist_ok=False)
        self._write(
            destination / "input.json",
            {"method": method, "args": serialize(args), "kwargs": serialize(kwargs)},
        )
        return destination, time.monotonic()

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        with path.open("x") as handle:
            json.dump(
                serialize(value), handle, indent=2, ensure_ascii=False, allow_nan=False
            )

    def _finish(
        self,
        destination: Path,
        started: float,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        if error is None:
            self._write(destination / "output.json", output)
        self._write(
            destination / "receipt.json",
            {
                "status": "failed" if error else "completed",
                "elapsed_sec": time.monotonic() - started,
                "error_type": type(error).__name__ if error else None,
                "error": str(error)[:2048] if error else None,
                "invocation_trace": serialize(
                    getattr(self.component, "last_invocation_trace", None)
                ),
            },
        )

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        destination, started = self._start(method, args, kwargs)
        try:
            result = getattr(self.component, method)(*args, **kwargs)
        except Exception as exc:
            self._finish(destination, started, error=exc)
            raise
        self._finish(destination, started, output=result)
        return result

    def reason(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("reason", *args, **kwargs)

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("plan", *args, **kwargs)

    async def perceive(self, *args: Any, **kwargs: Any) -> Any:
        destination, started = self._start("perceive", args, kwargs)
        try:
            result = await self.component.perceive(*args, **kwargs)
        except Exception as exc:
            self._finish(destination, started, error=exc)
            raise
        self._finish(destination, started, output=result)
        return result


def record_runtime_modules(runtime: Any, directory: Path) -> Any:
    """Wrap the assembled runtime in place only at an explicit experiment call site."""
    runtime.planner_adapter.reasoner = RecordedModule(
        runtime.planner_adapter.reasoner, directory / "reasoner"
    )
    runtime.planner_adapter.planner = RecordedModule(
        runtime.planner_adapter.planner, directory / "planner"
    )
    perception = RecordedModule(runtime.perception_adapter, directory / "perception")
    runtime.orchestrator.perception = perception
    return replace(runtime, perception_adapter=perception)
