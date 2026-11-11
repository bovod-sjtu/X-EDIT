"""Runtime facade for explicit workflow-control modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from x_editor.workflow_control.event_log import WorkflowEventLog
from x_editor.workflow_control.policy import (
    SUPPORTED_ENFORCED_REPAIR_ACTIONS,
    argument_repair_max_attempts,
    artifact_store_mode,
    enforced_repair_actions,
    final_assembler_mode,
)
from x_editor.workflow_control.resolvers import AuditDecisionResolver, DecisionResolver
from x_editor.workflow_control.schemas import (
    ArtifactStoreMode,
    DecisionCheckpoint,
    DecisionResolution,
    FinalAssemblerMode,
    RepairComparison,
    RepairDecision,
    WorkflowControlMode,
    WorkflowEvent,
    WorkflowEventStatus,
    WorkflowStage,
    new_workflow_id,
)


class WorkflowControlConfigurationError(ValueError):
    """Raised before execution when workflow-control configuration is unsafe."""


class WorkflowControlSession:
    """Keep workflow-control observations outside Planner and Executor state."""

    def __init__(
        self,
        *,
        mode: WorkflowControlMode,
        run_id: str,
        event_log: WorkflowEventLog | None,
        resolver: DecisionResolver | None = None,
        enforced_actions: frozenset[str] | None = None,
        artifact_store_mode: ArtifactStoreMode = "disabled",
        final_assembler_mode: FinalAssemblerMode = "disabled",
    ) -> None:
        self.mode = mode
        self.run_id = run_id
        self.event_log = event_log
        self.resolver = resolver or AuditDecisionResolver()
        self.enforced_actions = frozenset(enforced_actions or ())
        self.artifact_store_mode = artifact_store_mode
        self.final_assembler_mode = final_assembler_mode

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @property
    def event_log_path(self) -> str | None:
        return str(self.event_log.path) if self.event_log is not None else None

    def emit(
        self,
        *,
        stage: WorkflowStage,
        kind: str,
        status: WorkflowEventStatus = "observed",
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkflowEvent | None:
        if self.event_log is None:
            return None
        return self.event_log.append(
            stage=stage,
            kind=kind,
            status=status,
            attempt_id=attempt_id,
            subject_refs=subject_refs,
            payload=payload,
        )

    def resolve_checkpoint(self, checkpoint: DecisionCheckpoint) -> DecisionResolution:
        if self.event_log is not None:
            self.event_log.record_checkpoint(checkpoint)
        resolution = self.resolver.resolve(checkpoint)
        if self.event_log is not None:
            self.event_log.record_resolution(resolution, stage=checkpoint.stage)
        return resolution

    def observe_repair(
        self,
        decision: RepairDecision,
        *,
        attempt_id: str | None = None,
    ) -> WorkflowEvent | None:
        if self.event_log is None:
            return None
        return self.event_log.record_repair_decision(decision, attempt_id=attempt_id)

    def events(self) -> list[WorkflowEvent]:
        """Read this run's append-only events for deterministic projections."""

        return self.event_log.read_events() if self.event_log is not None else []

    def compare_repair(
        self,
        comparison: RepairComparison,
        *,
        attempt_id: str | None = None,
        subject_refs: list[str] | None = None,
    ) -> WorkflowEvent | None:
        if self.event_log is None:
            return None
        return self.event_log.record_repair_comparison(
            comparison,
            attempt_id=attempt_id,
            subject_refs=subject_refs,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "run_id": self.run_id,
            "event_log_path": self.event_log_path,
            "event_count": self.event_log.event_count
            if self.event_log is not None
            else 0,
            "behavior_control_enabled": self.mode == "enforced",
            "enforced_repair_actions": sorted(self.enforced_actions),
            "artifact_store_mode": self.artifact_store_mode,
            "final_assembler_mode": self.final_assembler_mode,
        }


def build_workflow_control_session(
    context: dict[str, Any],
    *,
    output_dir: str | Path | None,
) -> WorkflowControlSession:
    """Build an explicit, non-prompt workflow-control session from runtime context."""

    raw_mode = str(context.get("workflow_control_mode") or "disabled").strip().lower()
    valid_modes = {"disabled", "audit", "interactive", "enforced"}
    if raw_mode not in valid_modes:
        raise WorkflowControlConfigurationError(
            f"workflow_control_mode must be one of {sorted(valid_modes)}, got {raw_mode!r}"
        )
    mode: WorkflowControlMode = raw_mode  # type: ignore[assignment]
    run_id = new_workflow_id("run")
    try:
        store_mode = artifact_store_mode(context)
        assembler_mode = final_assembler_mode(context)
    except ValueError as exc:
        raise WorkflowControlConfigurationError(str(exc)) from exc
    if mode == "disabled":
        if store_mode != "disabled" or assembler_mode != "disabled":
            raise WorkflowControlConfigurationError(
                "artifact authority and FinalAssembler shadow modes require "
                "workflow_control_mode='enforced'"
            )
        return WorkflowControlSession(mode=mode, run_id=run_id, event_log=None)
    if store_mode == "enforced" and mode != "enforced":
        raise WorkflowControlConfigurationError(
            "workflow_artifact_store_mode='enforced' requires "
            "workflow_control_mode='enforced'"
        )
    if store_mode == "enforced":
        if "max_evaluation_revisions" not in context:
            raise WorkflowControlConfigurationError(
                "enforced ArtifactStore requires explicit max_evaluation_revisions=0"
            )
        try:
            evaluation_revisions = int(context["max_evaluation_revisions"])
        except (TypeError, ValueError) as exc:
            raise WorkflowControlConfigurationError(
                "max_evaluation_revisions must be an integer"
            ) from exc
        if evaluation_revisions != 0:
            raise WorkflowControlConfigurationError(
                "enforced ArtifactStore requires max_evaluation_revisions=0; "
                "Evaluator cannot control repair or artifact authority"
            )
    if assembler_mode == "shadow" and store_mode != "enforced":
        raise WorkflowControlConfigurationError(
            "workflow_final_assembler_mode='shadow' requires "
            "workflow_artifact_store_mode='enforced'"
        )
    actions = enforced_repair_actions(context)
    if mode == "enforced":
        if not actions and store_mode != "enforced":
            raise WorkflowControlConfigurationError(
                "workflow_control_mode='enforced' requires an explicit "
                "workflow_enforced_repair_actions allowlist or "
                "workflow_artifact_store_mode='enforced'"
            )
        unsupported = actions - SUPPORTED_ENFORCED_REPAIR_ACTIONS
        if unsupported:
            raise WorkflowControlConfigurationError(
                "unsupported enforced repair actions: "
                f"{sorted(unsupported)}; currently supported: "
                f"{sorted(SUPPORTED_ENFORCED_REPAIR_ACTIONS)}"
            )
        if actions:
            try:
                attempts = argument_repair_max_attempts(context)
            except (TypeError, ValueError) as exc:
                raise WorkflowControlConfigurationError(
                    "workflow_argument_repair_max_attempts must be an integer"
                ) from exc
            if attempts != 1:
                raise WorkflowControlConfigurationError(
                    "Phase 4 requires workflow_argument_repair_max_attempts=1"
                )
            raw_tool_contract = context.get("enable_tool_contract_verification", True)
            tool_contract_enabled = (
                raw_tool_contract.strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(raw_tool_contract, str)
                else bool(raw_tool_contract)
            )
            if not tool_contract_enabled:
                raise WorkflowControlConfigurationError(
                    "enforced argument repair requires enable_tool_contract_verification=true"
                )

    log_path = context.get("workflow_event_log_path")
    if not log_path:
        raw_base_dir = output_dir or context.get("output_dir")
        if not raw_base_dir:
            raise WorkflowControlConfigurationError(
                "audit or interactive workflow control requires output_dir or workflow_event_log_path"
            )
        base_dir = Path(raw_base_dir).expanduser()
        log_path = base_dir / "workflow_events.jsonl"

    resolver = context.get("workflow_decision_resolver")
    if resolver is not None and not callable(getattr(resolver, "resolve", None)):
        raise WorkflowControlConfigurationError(
            "workflow_decision_resolver must provide resolve(checkpoint)"
        )
    event_log = WorkflowEventLog(log_path, run_id=run_id)
    return WorkflowControlSession(
        mode=mode,
        run_id=run_id,
        event_log=event_log,
        resolver=resolver or AuditDecisionResolver(),
        enforced_actions=actions,
        artifact_store_mode=store_mode,
        final_assembler_mode=assembler_mode,
    )
