"""Runtime helpers for artifact and audio path handling.

Runtime submodules are used both by the main X-Editor process and by isolated
tool servers. Keep package-level exports lazy so a small tool server can import
``x_editor.runtime.spatial_reverb`` without pulling in planner/executor
dependencies such as pydantic, langgraph, or Gemini clients.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AudioEditExecutor",
    "AudioEditExecutorPerceptionAdapter",
    "AudioOperationRegistry",
    "ForwardExecutionPath",
    "ForwardExecutionPolicy",
    "ForwardExecutionRequest",
    "ForwardExecutionResult",
    "FrozenProviderPolicy",
    "GeminiExecutorToolCallPlanner",
    "OpenProviderPerceptionRoutePolicy",
    "OpenProviderW1Pilot",
    "OpenProviderW1PilotProfile",
    "OpenProviderWorkflowProfile",
    "OpenProviderWorkflowRuntime",
    "ReviewedWorkflowEvidenceProjector",
    "StubEvaluator",
    "TraceableAudioEditPlannerAdapter",
    "WorkflowArm",
    "WorkflowArmOrchestrator",
    "WorkflowArmPolicy",
    "WorkflowArmRequest",
    "WorkflowArmResult",
    "WorkflowAudioAsset",
    "WorkflowComponentAdapterError",
    "WorkflowEvidenceProjection",
    "WorkflowEvidenceProjector",
    "WorkflowPerceptionCallRecord",
    "WorkflowPerceptionEvidence",
    "WorkflowPerceptionGateway",
    "WorkflowPerceptionRequest",
    "WorkflowPilotReadinessFinding",
    "WorkflowPilotReadinessReport",
    "WorkflowPlannedPerceptionStep",
    "WorkflowPlanner",
    "WorkflowPlannerCallRecord",
    "WorkflowPlannerOutput",
    "WorkflowRuntimeCase",
    "WorkflowRuntimeScenario",
    "WorkflowRuntimeTurn",
    "WorkflowScenarioResult",
    "WorkflowScenarioTurnResult",
    "assess_open_provider_w1_readiness",
    "build_default_open_provider_workflow_profile",
    "build_frozen_openai_client",
    "build_open_provider_w1_pilot",
    "build_open_provider_workflow_runtime",
    "default_edit_operations",
    "default_evidence_required_operations",
]


def __getattr__(name: str) -> Any:
    if name in {
        "WorkflowAudioAsset",
        "WorkflowRuntimeCase",
        "WorkflowRuntimeScenario",
        "WorkflowRuntimeTurn",
    }:
        from x_editor.runtime.workflow_case import (
            WorkflowAudioAsset,
            WorkflowRuntimeCase,
            WorkflowRuntimeScenario,
            WorkflowRuntimeTurn,
        )

        return {
            "WorkflowAudioAsset": WorkflowAudioAsset,
            "WorkflowRuntimeCase": WorkflowRuntimeCase,
            "WorkflowRuntimeScenario": WorkflowRuntimeScenario,
            "WorkflowRuntimeTurn": WorkflowRuntimeTurn,
        }[name]
    if name in {
        "WorkflowScenarioResult",
        "WorkflowScenarioTurnResult",
    }:
        from x_editor.runtime.workflow_session import (
            WorkflowScenarioResult,
            WorkflowScenarioTurnResult,
        )

        return {
            "WorkflowScenarioResult": WorkflowScenarioResult,
            "WorkflowScenarioTurnResult": WorkflowScenarioTurnResult,
        }[name]
    if name in {"AudioEditExecutor", "AudioOperationRegistry"}:
        from x_editor.runtime.audio_plan_executor import (
            AudioEditExecutor,
            AudioOperationRegistry,
        )

        return {"AudioEditExecutor": AudioEditExecutor, "AudioOperationRegistry": AudioOperationRegistry}[name]
    if name == "GeminiExecutorToolCallPlanner":
        from x_editor.runtime.gemini_executor import GeminiExecutorToolCallPlanner

        return GeminiExecutorToolCallPlanner
    if name in {
        "AudioEditExecutorPerceptionAdapter",
        "ReviewedWorkflowEvidenceProjector",
        "TraceableAudioEditPlannerAdapter",
        "WorkflowComponentAdapterError",
    }:
        from x_editor.runtime.workflow_adapters import (
            AudioEditExecutorPerceptionAdapter,
            ReviewedWorkflowEvidenceProjector,
            TraceableAudioEditPlannerAdapter,
            WorkflowComponentAdapterError,
        )

        return {
            "AudioEditExecutorPerceptionAdapter": AudioEditExecutorPerceptionAdapter,
            "ReviewedWorkflowEvidenceProjector": ReviewedWorkflowEvidenceProjector,
            "TraceableAudioEditPlannerAdapter": TraceableAudioEditPlannerAdapter,
            "WorkflowComponentAdapterError": WorkflowComponentAdapterError,
        }[name]
    if name in {
        "ForwardExecutionPath",
        "ForwardExecutionPolicy",
        "ForwardExecutionRequest",
        "ForwardExecutionResult",
    }:
        from x_editor.runtime.forward_execution import (
            ForwardExecutionPath,
            ForwardExecutionPolicy,
            ForwardExecutionRequest,
            ForwardExecutionResult,
        )

        return {
            "ForwardExecutionPath": ForwardExecutionPath,
            "ForwardExecutionPolicy": ForwardExecutionPolicy,
            "ForwardExecutionRequest": ForwardExecutionRequest,
            "ForwardExecutionResult": ForwardExecutionResult,
        }[name]
    if name in {
        "FrozenProviderPolicy",
        "build_frozen_openai_client",
    }:
        from x_editor.runtime.provider_policy import (
            FrozenProviderPolicy,
            build_frozen_openai_client,
        )

        return {
            "FrozenProviderPolicy": FrozenProviderPolicy,
            "build_frozen_openai_client": build_frozen_openai_client,
        }[name]
    if name in {
        "OpenProviderPerceptionRoutePolicy",
        "OpenProviderWorkflowProfile",
        "OpenProviderWorkflowRuntime",
        "build_default_open_provider_workflow_profile",
        "build_open_provider_workflow_runtime",
        "default_edit_operations",
        "default_evidence_required_operations",
    }:
        from x_editor.runtime.open_provider_workflow import (
            OpenProviderPerceptionRoutePolicy,
            OpenProviderWorkflowProfile,
            OpenProviderWorkflowRuntime,
            build_default_open_provider_workflow_profile,
            build_open_provider_workflow_runtime,
            default_edit_operations,
            default_evidence_required_operations,
        )

        return {
            "OpenProviderPerceptionRoutePolicy": OpenProviderPerceptionRoutePolicy,
            "OpenProviderWorkflowProfile": OpenProviderWorkflowProfile,
            "OpenProviderWorkflowRuntime": OpenProviderWorkflowRuntime,
            "build_default_open_provider_workflow_profile": build_default_open_provider_workflow_profile,
            "build_open_provider_workflow_runtime": build_open_provider_workflow_runtime,
            "default_edit_operations": default_edit_operations,
            "default_evidence_required_operations": default_evidence_required_operations,
        }[name]
    if name in {
        "OpenProviderW1Pilot",
        "OpenProviderW1PilotProfile",
        "WorkflowPilotReadinessFinding",
        "WorkflowPilotReadinessReport",
        "assess_open_provider_w1_readiness",
        "build_open_provider_w1_pilot",
    }:
        from x_editor.runtime.real_workflow_pilot import (
            OpenProviderW1Pilot,
            OpenProviderW1PilotProfile,
            WorkflowPilotReadinessFinding,
            WorkflowPilotReadinessReport,
            assess_open_provider_w1_readiness,
            build_open_provider_w1_pilot,
        )

        return {
            "OpenProviderW1Pilot": OpenProviderW1Pilot,
            "OpenProviderW1PilotProfile": OpenProviderW1PilotProfile,
            "WorkflowPilotReadinessFinding": WorkflowPilotReadinessFinding,
            "WorkflowPilotReadinessReport": WorkflowPilotReadinessReport,
            "assess_open_provider_w1_readiness": assess_open_provider_w1_readiness,
            "build_open_provider_w1_pilot": build_open_provider_w1_pilot,
        }[name]
    if name in {
        "WorkflowArm",
        "WorkflowArmOrchestrator",
        "WorkflowArmPolicy",
        "WorkflowArmRequest",
        "WorkflowArmResult",
        "WorkflowEvidenceProjection",
        "WorkflowEvidenceProjector",
        "WorkflowPerceptionCallRecord",
        "WorkflowPerceptionEvidence",
        "WorkflowPerceptionGateway",
        "WorkflowPerceptionRequest",
        "WorkflowPlannedPerceptionStep",
        "WorkflowPlanner",
        "WorkflowPlannerCallRecord",
        "WorkflowPlannerOutput",
    }:
        from x_editor.runtime.workflow_orchestration import (
            WorkflowArm,
            WorkflowArmOrchestrator,
            WorkflowArmPolicy,
            WorkflowArmRequest,
            WorkflowArmResult,
            WorkflowEvidenceProjection,
            WorkflowEvidenceProjector,
            WorkflowPerceptionCallRecord,
            WorkflowPerceptionEvidence,
            WorkflowPerceptionGateway,
            WorkflowPerceptionRequest,
            WorkflowPlannedPerceptionStep,
            WorkflowPlanner,
            WorkflowPlannerCallRecord,
            WorkflowPlannerOutput,
        )

        return {
            "WorkflowArm": WorkflowArm,
            "WorkflowArmOrchestrator": WorkflowArmOrchestrator,
            "WorkflowArmPolicy": WorkflowArmPolicy,
            "WorkflowArmRequest": WorkflowArmRequest,
            "WorkflowArmResult": WorkflowArmResult,
            "WorkflowEvidenceProjection": WorkflowEvidenceProjection,
            "WorkflowEvidenceProjector": WorkflowEvidenceProjector,
            "WorkflowPerceptionCallRecord": WorkflowPerceptionCallRecord,
            "WorkflowPerceptionEvidence": WorkflowPerceptionEvidence,
            "WorkflowPerceptionGateway": WorkflowPerceptionGateway,
            "WorkflowPerceptionRequest": WorkflowPerceptionRequest,
            "WorkflowPlannedPerceptionStep": WorkflowPlannedPerceptionStep,
            "WorkflowPlanner": WorkflowPlanner,
            "WorkflowPlannerCallRecord": WorkflowPlannerCallRecord,
            "WorkflowPlannerOutput": WorkflowPlannerOutput,
        }[name]
    if name == "StubEvaluator":
        from x_editor.runtime.evaluator import StubEvaluator

        return StubEvaluator
    raise AttributeError(name)
