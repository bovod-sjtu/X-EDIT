"""Immutable, non-authoritative module checkpoints for sparse evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from x_editor.core.schemas import StrictModel
from x_editor.workflow_control.schemas import (
    WorkflowEventStatus,
    WorkflowStage,
)

CheckpointModule = Literal[
    "reasoner",
    "planner",
    "planner_reasoner",
    "perception",
    "plan_admission",
    "executor",
    "tool",
    "verifier",
    "artifact_store",
    "final_delivery",
]
CheckpointEvaluationRole = Literal["candidate_output", "failure_diagnostic"]


class CheckpointRunContext(StrictModel):
    """Prompt-hidden identity that joins checkpoints to one runtime attempt."""

    schema_version: Literal["x-editor.checkpoint-run-context/v1"] = (
        "x-editor.checkpoint-run-context/v1"
    )
    workflow_run_id: str
    assigned_arm: Literal["W0", "W1", "W2"]
    runtime_case_fingerprint: str
    turn_index: int | None = Field(default=None, ge=0)


class CheckpointSourceEvent(StrictModel):
    """Exact append-only event from which a checkpoint was projected."""

    event_id: str
    run_id: str
    sequence: int = Field(ge=1)


class ModuleCheckpoint(StrictModel):
    """One immutable module output exposed to a parallel evaluation path."""

    schema_version: Literal["x-editor.module-checkpoint/v1"] = (
        "x-editor.module-checkpoint/v1"
    )
    checkpoint_id: str
    checkpoint_sequence: int = Field(ge=1)
    context: CheckpointRunContext
    module: CheckpointModule
    boundary_kind: str
    evaluation_role: CheckpointEvaluationRole
    status: WorkflowEventStatus
    attempt_id: str | None = None
    subject_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    source_event: CheckpointSourceEvent
    upstream_source_event: CheckpointSourceEvent | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    output_fingerprint: str


_EVENT_MODULES: dict[str, CheckpointModule] = {
    "workflow_reasoner_shadow_call_completed": "reasoner",
    "workflow_reasoner_shadow_output_rejected": "reasoner",
    "workflow_reasoner_shadow_call_failed": "reasoner",
    "workflow_reasoner_shadow_call_skipped": "reasoner",
    "workflow_reasoner_call_completed": "reasoner",
    "workflow_reasoner_call_failed": "reasoner",
    "workflow_reasoner_invocation_rejected": "reasoner",
    "workflow_separated_planner_call_completed": "planner",
    "workflow_planner_call_completed": "planner_reasoner",
    "workflow_planner_call_failed": "planner_reasoner",
    "workflow_planner_output_rejected": "planner_reasoner",
    "workflow_planner_invocation_rejected": "planner_reasoner",
    "workflow_planner_fallback_rejected": "planner_reasoner",
    "workflow_plan_frozen": "planner_reasoner",
    "workflow_perception_call_completed": "perception",
    "workflow_perception_call_failed": "perception",
    "workflow_perception_invocation_rejected": "perception",
    "workflow_evidence_projection_rejected": "perception",
    "forward_plan_admission_completed": "plan_admission",
    "forward_executor_completed": "executor",
    "forward_executor_failed": "executor",
    "step_execution_completed": "tool",
    "step_verification_completed": "verifier",
    "forward_artifact_finalization_audited": "artifact_store",
    "forward_artifact_finalization_audit_failed": "artifact_store",
    "forward_artifact_store_committed": "artifact_store",
    "forward_artifact_store_blocked": "artifact_store",
    "forward_artifact_store_failed": "artifact_store",
    "forward_final_delivery_manifest_observed": "final_delivery",
    "forward_final_delivery_manifest_observation_failed": "final_delivery",
}

_CANDIDATE_EVENT_KINDS = {
    "workflow_reasoner_shadow_call_completed",
    "workflow_reasoner_call_completed",
    "workflow_separated_planner_call_completed",
    "workflow_planner_call_completed",
    "workflow_plan_frozen",
    "workflow_perception_call_completed",
    "forward_plan_admission_completed",
    "forward_executor_completed",
    "step_execution_completed",
    "step_verification_completed",
    "forward_artifact_finalization_audited",
    "forward_artifact_store_committed",
    "forward_final_delivery_manifest_observed",
}

_FORBIDDEN_OUTPUT_KEYS = {
    "api_key",
    "authorization",
    "base_url",
    "benchmark",
    "benchmark_id",
    "cache_path",
    "credential",
    "credentials",
    "error_message",
    "expected_answer",
    "oracle",
    "password",
    "prompt",
    "raw_output",
    "raw_prompt",
    "raw_response",
    "response_body",
    "right_choice",
    "rubric",
    "rubrics",
    "secret",
    "service_url",
    "target_audio",
    "target_waveform",
}


class ModuleCheckpointObserver:
    """Project terminal workflow events to an append-only evaluation sidecar.

    ``ExecutionEventRecorder`` ignores observer return values and isolates observer
    failures, so this class can never admit, reject, repair, or select an artifact.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        context: CheckpointRunContext,
    ) -> None:
        self.context = context
        self.checkpoint_path = str(Path(path).expanduser().resolve())
        self.checkpoints: list[ModuleCheckpoint] = []
        self.errors: list[dict[str, str]] = []
        self.duplicate_event_count = 0
        self._seen_source_event_ids: set[str] = set()
        self._existing_checkpoint_count = 0
        self._writable = True
        self._initialize()

    @property
    def checkpoint_count(self) -> int:
        """Return checkpoints emitted by this runtime attempt only."""

        return len(self.checkpoints)

    def emit(
        self,
        *,
        stage: WorkflowStage,
        kind: str,
        status: WorkflowEventStatus = "observed",
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ModuleCheckpoint | None:
        module = _EVENT_MODULES.get(kind)
        if module is None:
            return None
        values = dict(payload or {})
        source_event = _source_event(values.pop("source_event_ref", None))
        if source_event is None:
            self.errors.append(
                {
                    "kind": kind,
                    "error": "checkpoint event is missing source_event_ref",
                }
            )
            return None
        if source_event.event_id in self._seen_source_event_ids:
            self.duplicate_event_count += 1
            return None
        upstream_source_event = _source_event(
            values.pop("upstream_source_event_ref", None)
        )
        safe_output = _safe_output(values)
        resolved_module = _dynamic_module(module, kind=kind, output=safe_output)
        checkpoint_sequence = (
            self._existing_checkpoint_count + len(self.checkpoints) + 1
        )
        output_fingerprint = _fingerprint(safe_output)
        checkpoint_id = _fingerprint(
            {
                "context": self.context.model_dump(mode="json"),
                "source_event": source_event.model_dump(mode="json"),
                "module": resolved_module,
                "kind": kind,
                "output_fingerprint": output_fingerprint,
            }
        )
        checkpoint = ModuleCheckpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_sequence=checkpoint_sequence,
            context=self.context,
            module=resolved_module,
            boundary_kind=kind,
            evaluation_role=(
                "candidate_output"
                if kind in _CANDIDATE_EVENT_KINDS
                and status in {"completed", "accepted", "observed"}
                else "failure_diagnostic"
            ),
            status=status,
            attempt_id=attempt_id,
            subject_refs=_unique_strings(subject_refs or []),
            evidence_refs=_collect_refs(safe_output, token="evidence"),
            artifact_refs=_collect_refs(safe_output, token="artifact"),
            source_event=source_event,
            upstream_source_event=upstream_source_event,
            output=safe_output,
            output_fingerprint=output_fingerprint,
        )
        self._append(checkpoint)
        self._seen_source_event_ids.add(source_event.event_id)
        self.checkpoints.append(checkpoint)
        return checkpoint

    def _initialize(self) -> None:
        path = Path(self.checkpoint_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                return
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                checkpoint = ModuleCheckpoint.model_validate_json(line)
                if checkpoint.checkpoint_sequence != self._existing_checkpoint_count + 1:
                    raise ValueError(
                        f"non-contiguous checkpoint sequence at line {line_number}"
                    )
                self._existing_checkpoint_count += 1
                self._seen_source_event_ids.add(checkpoint.source_event.event_id)
        except Exception as exc:
            self._writable = False
            self.errors.append(
                {
                    "kind": "checkpoint_ledger_initialize",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _append(self, checkpoint: ModuleCheckpoint) -> None:
        if not self._writable:
            raise RuntimeError("checkpoint ledger is unavailable")
        path = Path(self.checkpoint_path)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(checkpoint.model_dump_json())
                handle.write("\n")
                handle.flush()
        except Exception as exc:
            self._writable = False
            self.errors.append(
                {
                    "kind": checkpoint.boundary_kind,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise


def _dynamic_module(
    module: CheckpointModule,
    *,
    kind: str,
    output: dict[str, Any],
) -> CheckpointModule:
    if kind != "step_execution_completed":
        return module
    step = output.get("step")
    operation = step.get("operation") if isinstance(step, dict) else None
    if isinstance(operation, str) and operation.startswith(("perception.", "flexsed.")):
        return "perception"
    return module


def _source_event(value: Any) -> CheckpointSourceEvent | None:
    if not isinstance(value, dict):
        return None
    try:
        return CheckpointSourceEvent.model_validate(value)
    except Exception:
        return None


def _safe_output(value: Any, *, depth: int = 0) -> Any:
    if depth >= 12:
        return {"truncated_type": type(value).__name__}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return {"redacted_type": "runtime_path"}
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_OUTPUT_KEYS:
                continue
            if _contains_runtime_path(key_text, item):
                continue
            safe[key_text] = _safe_output(item, depth=depth + 1)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_safe_output(item, depth=depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        return _safe_output(value.model_dump(mode="json"), depth=depth + 1)
    return {"unsupported_type": type(value).__name__}


def _contains_runtime_path(key: str, value: Any) -> bool:
    normalized = key.lower()
    path_key = (
        normalized in {"path", "paths", "output_dir"}
        or normalized.endswith("_path")
        or normalized.endswith("_paths")
    )
    if not path_key:
        return False
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return any(_looks_like_runtime_path(item) for item in values)


def _looks_like_runtime_path(value: Any) -> bool:
    if isinstance(value, Path):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(
        text.startswith(("/", "~/", "./", "../", "file://"))
        or re.match(r"^[A-Za-z]:[\\/]", text)
    )


def _collect_refs(value: Any, *, token: str) -> list[str]:
    refs: list[str] = []

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                normalized = str(child_key).lower()
                if token in normalized and normalized.endswith(("id", "ids", "ref", "refs")):
                    refs.extend(_strings(child))
                visit(child, normalized)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)

    visit(value)
    return _unique_strings(refs)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int))]
    return []


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in values if str(item)))


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CheckpointEvaluationRole",
    "CheckpointModule",
    "CheckpointRunContext",
    "CheckpointSourceEvent",
    "ModuleCheckpoint",
    "ModuleCheckpointObserver",
]
