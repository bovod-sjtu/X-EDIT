"""Generic multi-turn execution over the shared workflow runtime."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import Field

from x_editor.core.final_state_projection import (
    FinalStateProjection,
    project_final_session_state,
)
from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.reviewed_evidence import rebase_reviewed_audio_namespace
from x_editor.core.schemas import StrictModel
from x_editor.reasoner.contracts import PriorTurnSemanticContext
from x_editor.runtime.edit_session_graph import EditSessionGraph, save_edit_session_graph
from x_editor.runtime.workflow_case import (
    HistoricalReviewedEvidenceContext,
    RuntimeAudioBinding,
    WorkflowRuntimeCase,
    WorkflowRuntimeScenario,
)
from x_editor.runtime.workflow_orchestration import (
    WorkflowArm,
    WorkflowArmResult,
    WorkflowEvidenceProjection,
    WorkflowStateOnlyCompletion,
)

if TYPE_CHECKING:
    from x_editor.runtime.workflow_continuation import WorkflowContinuation

_HISTORICAL_EVIDENCE_STORE_KEY = "reviewed_evidence_history"
_HISTORICAL_EVIDENCE_CONTEXT_SCHEMA = "x-editor.historical-reviewed-evidence-context/v1"
_HISTORICAL_EVIDENCE_SNAPSHOT_SCHEMA = "x-editor.historical-reviewed-evidence/v1"


class WorkflowCaseRuntime(Protocol):
    async def arun_case(
        self,
        case: WorkflowRuntimeCase,
        *,
        arm: WorkflowArm,
        output_dir: str | Path,
        event_log_path: str | Path | None = None,
        turn_index: int | None = None,
    ) -> WorkflowArmResult: ...


class WorkflowScenarioTurnResult(StrictModel):
    schema_version: Literal["x-editor.workflow-scenario-turn-result/v1"] = (
        "x-editor.workflow-scenario-turn-result/v1"
    )
    turn_index: int = Field(ge=0)
    runtime_case_fingerprint: str
    result: WorkflowArmResult


class WorkflowScenarioResult(StrictModel):
    schema_version: Literal["x-editor.workflow-scenario-result/v1"] = (
        "x-editor.workflow-scenario-result/v1"
    )
    scenario_fingerprint: str
    assigned_arm: WorkflowArm
    status: Literal["completed", "blocked", "failed"]
    failure_turn_index: int | None = Field(default=None, ge=0)
    failure_stage: str = "none"
    failure_codes: list[str] = Field(default_factory=list)
    terminal_error_type: str | None = None
    terminal_error_message: str | None = None
    turn_results: list[WorkflowScenarioTurnResult] = Field(default_factory=list)
    final_output_path: str | None = None
    session_graph_path: str | None = None
    final_state_projection: FinalStateProjection | None = None


async def arun_workflow_scenario(
    runtime: WorkflowCaseRuntime,
    scenario: WorkflowRuntimeScenario,
    *,
    arm: WorkflowArm,
    output_dir: str | Path,
    future_attempt_guidance_by_turn: dict[int, FutureAttemptGuidance] | None = None,
    continuation: WorkflowContinuation | dict | None = None,
) -> WorkflowScenarioResult:
    """Run turns serially, advancing only accepted authoritative artifacts."""

    root = Path(output_dir)
    guidance_by_turn = future_attempt_guidance_by_turn or {}
    unknown_guidance_turns = sorted(set(guidance_by_turn) - set(range(len(scenario.turns))))
    if unknown_guidance_turns:
        raise ValueError(
            f"future-attempt guidance references unknown turns: {unknown_guidance_turns}"
        )
    session_graph_path = root / "edit_session_graph.json"
    session_graph = _initialize_session_graph(scenario)
    audio_ref_by_path = {
        asset.path: asset.logical_ref for asset in scenario.visible_audio_assets(0)
    }
    session_bindings = _initial_session_bindings(scenario)
    turn_results: list[WorkflowScenarioTurnResult] = []
    previous_output_artifact_ref_id: str | None = None
    previous_output_state_node_id: str | None = None
    previous_output_path: str | None = None
    previous_output_ref: str | None = None
    failed_turn_indices: set[int] = set()
    if continuation is not None:
        from x_editor.runtime.workflow_continuation import restore_workflow_continuation

        restored = restore_workflow_continuation(
            continuation, scenario, arm=arm, output_dir=root,
        )
        session_graph = restored.graph
        audio_ref_by_path = restored.audio_ref_by_path
        session_bindings = restored.session_bindings
        turn_results = restored.turn_results
        previous_output_path = restored.previous_output_path
        previous_output_ref = restored.previous_output_ref
        previous_output_artifact_ref_id = restored.previous_output_artifact_ref_id
        previous_output_state_node_id = restored.previous_output_state_node_id
        failed_turn_indices = restored.failed_turn_indices
    save_edit_session_graph(session_graph, session_graph_path)
    for turn in scenario.turns[len(turn_results):]:
        _disclose_turn_assets(
            scenario, turn.turn_index, session_graph, session_bindings, audio_ref_by_path,
        )
        session_state = session_graph.semantic_state_context(
            audio_ref_by_path=audio_ref_by_path,
        )
        prior_turn_contexts = [
            PriorTurnSemanticContext(
                turn_index=prior_turn.turn_index,
                user_request=prior_turn.user_request,
                source_text=prior_turn.source_text,
                result_is_ancestor_of_current_source=(
                    turn.use_previous_output
                    and prior_turn.turn_index not in failed_turn_indices
                    and all(
                        scenario.turns[index].use_previous_output
                        for index in range(prior_turn.turn_index + 1, turn.turn_index + 1)
                    )
                ),
            )
            for prior_turn in scenario.turns[: turn.turn_index]
        ]
        historical_evidence_payload = _historical_reviewed_evidence_context(
            session_graph,
            session_state=session_state,
            visible_audio_refs={item.audio_ref for item in session_bindings},
        )
        historical_evidence = (
            HistoricalReviewedEvidenceContext.model_validate(historical_evidence_payload)
            if historical_evidence_payload["snapshots"]
            else None
        )
        case = scenario.materialize_turn(
            turn.turn_index,
            previous_output_path=previous_output_path,
            previous_output_ref=previous_output_ref,
            previous_output_artifact_ref_id=previous_output_artifact_ref_id,
            previous_output_state_node_id=previous_output_state_node_id,
            prior_turn_contexts=prior_turn_contexts,
            session_state=session_state,
            historical_reviewed_evidence=(historical_evidence),
            future_attempt_guidance=guidance_by_turn.get(turn.turn_index),
            session_audio_bindings=session_bindings,
        )
        round_index = session_graph.start_round(turn.user_request)
        turn_dir = root / "turns" / f"{turn.turn_index:04d}"
        result = await runtime.arun_case(
            case,
            arm=arm,
            output_dir=turn_dir / "execution",
            event_log_path=turn_dir / "workflow_events.jsonl",
            turn_index=turn.turn_index,
        )
        turn_results.append(
            WorkflowScenarioTurnResult(
                turn_index=turn.turn_index,
                runtime_case_fingerprint=case.fingerprint,
                result=result,
            )
        )
        if result.status != "completed":
            _complete_session_round(
                graph=session_graph,
                round_index=round_index,
                request=turn.user_request,
                result=result,
                case=case,
                status=result.status,
                advance_current_audio=False,
            )
            save_edit_session_graph(session_graph, session_graph_path)
            return WorkflowScenarioResult(
                scenario_fingerprint=scenario.fingerprint,
                assigned_arm=arm,
                status=result.status,
                failure_turn_index=turn.turn_index,
                failure_stage=result.failure_stage,
                failure_codes=list(result.failure_codes),
                terminal_error_type=result.terminal_error_type,
                terminal_error_message=result.terminal_error_message,
                turn_results=turn_results,
                session_graph_path=str(session_graph_path),
                final_state_projection=project_final_session_state(
                    session_graph,
                    graph_path=session_graph_path,
                ),
            )
        if result.state_only_completion is not None:
            try:
                selected_node = _resolve_state_only_completion(
                    graph=session_graph,
                    result=result,
                    case=case,
                    audio_ref_by_path=audio_ref_by_path,
                )
            except ValueError:
                _complete_session_round(
                    graph=session_graph,
                    round_index=round_index,
                    request=turn.user_request,
                    result=result,
                    case=case,
                    status="failed",
                    advance_current_audio=False,
                    failure_stage="workflow_protocol",
                    failure_codes=["state_only_completion_graph_rejected"],
                )
                save_edit_session_graph(session_graph, session_graph_path)
                return WorkflowScenarioResult(
                    scenario_fingerprint=scenario.fingerprint,
                    assigned_arm=arm,
                    status="failed",
                    failure_turn_index=turn.turn_index,
                    failure_stage="workflow_protocol",
                    failure_codes=["state_only_completion_graph_rejected"],
                    turn_results=turn_results,
                    session_graph_path=str(session_graph_path),
                    final_state_projection=project_final_session_state(
                        session_graph,
                        graph_path=session_graph_path,
                    ),
                )
            completion = result.state_only_completion
            final_output_path = str(selected_node.primary_audio_path)
            selected_artifact_ref_id = selected_node.metadata.get("artifact_ref_id")
            _complete_session_round(
                graph=session_graph,
                round_index=round_index,
                request=turn.user_request,
                result=result,
                case=case,
                status="completed",
                advance_current_audio=True,
                final_output_path=final_output_path,
                state_only_completion=completion,
            )
            _record_reviewed_evidence_history(
                session_graph,
                result=result,
                case=case,
                turn_index=turn.turn_index,
            )
            output_audio_ref = f"aud:v1:o{turn.turn_index:04d}"
            current_node = session_graph.current_node()
            if current_node is not None:
                current_node.metadata["audio_ref"] = output_audio_ref
                current_node.metadata["selected_state_node_id"] = completion.selected_node_id
                current_node.metadata["selected_state_audio_ref"] = completion.selected_audio_ref
                if selected_artifact_ref_id:
                    current_node.metadata["artifact_ref_id"] = selected_artifact_ref_id
            previous_output_path = final_output_path
            previous_output_ref = None
            previous_output_artifact_ref_id = (
                str(selected_artifact_ref_id) if selected_artifact_ref_id else None
            )
            previous_output_state_node_id = completion.selected_node_id
            session_bindings.append(
                RuntimeAudioBinding(
                    audio_ref=output_audio_ref,
                    materialized_path=final_output_path,
                    provenance="session_output",
                    role="authoritative_output",
                    order=len(session_bindings),
                    origin="session_graph_state",
                    origin_artifact_ref_id=previous_output_artifact_ref_id,
                    origin_state_node_id=completion.selected_node_id,
                )
            )
            save_edit_session_graph(session_graph, session_graph_path)
            continue
        commit = (
            result.forward_result.artifact_store_commit
            if result.forward_result is not None
            else None
        )
        if (
            commit is None
            or commit.status != "committed"
            or not commit.final_artifact_ref_id
            or not commit.final_output_path
        ):
            _complete_session_round(
                graph=session_graph,
                round_index=round_index,
                request=turn.user_request,
                result=result,
                case=case,
                status="failed",
                advance_current_audio=False,
            )
            save_edit_session_graph(session_graph, session_graph_path)
            return WorkflowScenarioResult(
                scenario_fingerprint=scenario.fingerprint,
                assigned_arm=arm,
                status="failed",
                failure_turn_index=turn.turn_index,
                failure_stage="artifact_store",
                failure_codes=["completed_turn_missing_authoritative_commit"],
                turn_results=turn_results,
                session_graph_path=str(session_graph_path),
                final_state_projection=project_final_session_state(
                    session_graph,
                    graph_path=session_graph_path,
                ),
            )
        if (
            result.forward_result.final_output_path is not None
            and result.forward_result.final_output_path != commit.final_output_path
        ):
            _complete_session_round(
                graph=session_graph,
                round_index=round_index,
                request=turn.user_request,
                result=result,
                case=case,
                status="failed",
                advance_current_audio=False,
            )
            save_edit_session_graph(session_graph, session_graph_path)
            return WorkflowScenarioResult(
                scenario_fingerprint=scenario.fingerprint,
                assigned_arm=arm,
                status="failed",
                failure_turn_index=turn.turn_index,
                failure_stage="artifact_store",
                failure_codes=["completed_turn_authoritative_output_mismatch"],
                turn_results=turn_results,
                session_graph_path=str(session_graph_path),
                final_state_projection=project_final_session_state(
                    session_graph,
                    graph_path=session_graph_path,
                ),
            )
        _complete_session_round(
            graph=session_graph,
            round_index=round_index,
            request=turn.user_request,
            result=result,
            case=case,
            status="completed",
            advance_current_audio=True,
            final_output_path=commit.final_output_path,
        )
        _record_reviewed_evidence_history(
            session_graph,
            result=result,
            case=case,
            turn_index=turn.turn_index,
        )
        previous_output_artifact_ref_id = commit.final_artifact_ref_id
        previous_output_state_node_id = None
        previous_output_path = commit.final_output_path
        previous_output_ref = None
        output_audio_ref = f"aud:v1:o{turn.turn_index:04d}"
        current_node = session_graph.current_node()
        if current_node is not None:
            current_node.metadata["artifact_ref_id"] = commit.final_artifact_ref_id
            current_node.metadata["audio_ref"] = output_audio_ref
        audio_ref_by_path[commit.final_output_path] = output_audio_ref
        session_bindings.append(
            RuntimeAudioBinding(
                audio_ref=output_audio_ref,
                materialized_path=commit.final_output_path,
                role="authoritative_output",
                order=len(session_bindings),
                origin="artifact_store_commit",
                origin_artifact_ref_id=commit.final_artifact_ref_id,
            )
        )
        save_edit_session_graph(session_graph, session_graph_path)

    return WorkflowScenarioResult(
        scenario_fingerprint=scenario.fingerprint,
        assigned_arm=arm,
        status="completed",
        turn_results=turn_results,
        final_output_path=previous_output_path,
        session_graph_path=str(session_graph_path),
        final_state_projection=project_final_session_state(
            session_graph,
            graph_path=session_graph_path,
        ),
    )


def _initialize_session_graph(
    scenario: WorkflowRuntimeScenario,
) -> EditSessionGraph:
    graph = EditSessionGraph.create(session_id=f"workflow-{scenario.fingerprint[:12]}")
    source_assets = sorted(
        (item for item in scenario.visible_audio_assets(0) if item.role == "source"),
        key=lambda item: item.order,
    )
    reference_assets = sorted(
        (item for item in scenario.visible_audio_assets(0) if item.role == "reference"),
        key=lambda item: item.order,
    )
    graph.ensure_initial_node(
        input_audio=source_assets[0].path if source_assets else None,
        input_audios=[item.path for item in source_assets],
        reference_audio_items=[
            {"audio_id": item.logical_ref, "path": item.path} for item in reference_assets
        ],
        reference_audio_paths=[item.path for item in reference_assets],
    )
    return graph


def _initial_session_bindings(
    scenario: WorkflowRuntimeScenario,
    *,
    through_turn_index: int = 0,
) -> list[RuntimeAudioBinding]:
    return [
        RuntimeAudioBinding(
            audio_ref=asset.logical_ref,
            materialized_path=asset.path,
            provenance=asset.provenance,
            role=asset.role,
            order=index,
            origin="scenario_asset",
            content_sha256=asset.content_sha256,
        )
        for index, asset in enumerate(scenario.visible_audio_assets(through_turn_index))
    ]


def _disclose_turn_assets(
    scenario: WorkflowRuntimeScenario,
    turn_index: int,
    graph: EditSessionGraph,
    bindings: list[RuntimeAudioBinding],
    audio_ref_by_path: dict[str, str],
) -> None:
    """Append public inputs without changing previously accepted audio states."""
    bound_refs = {item.audio_ref for item in bindings}
    for asset in scenario.visible_audio_assets(turn_index):
        if asset.logical_ref in bound_refs:
            continue
        bindings.append(
            RuntimeAudioBinding(
                audio_ref=asset.logical_ref,
                materialized_path=asset.path,
                provenance=asset.provenance,
                role=asset.role,
                order=len(bindings),
                origin="scenario_asset",
                content_sha256=asset.content_sha256,
            )
        )
        bound_refs.add(asset.logical_ref)
        # An alias must not replace an already accepted version's path identity.
        audio_ref_by_path.setdefault(asset.path, asset.logical_ref)
        graph.register_external_input(
            audio_ref=asset.logical_ref, path=asset.path, role=asset.role,
            round_index=turn_index + 1,
        )


def _record_reviewed_evidence_history(
    graph: EditSessionGraph,
    *,
    result: WorkflowArmResult,
    case: WorkflowRuntimeCase,
    turn_index: int,
) -> None:
    """Persist only admitted facts, scoped to the state that was observed."""

    projection = result.evidence_projection
    if projection is None or not projection.planner_context:
        return
    state_node_by_audio_ref = {
        node.audio_ref: node.node_id
        for node in (case.session_state.nodes if case.session_state is not None else [])
        if node.audio_ref is not None
    }
    history = graph.metadata.setdefault(_HISTORICAL_EVIDENCE_STORE_KEY, [])
    if not isinstance(history, list):
        raise TypeError("Session Graph reviewed evidence history is malformed")
    known_snapshot_ids = {
        str(item.get("snapshot_id"))
        for item in history
        if isinstance(item, dict) and item.get("snapshot_id")
    }
    for audio_ref in projection.source_audio_refs:
        facts = _projection_facts_for_audio_ref(projection, audio_ref=audio_ref)
        if not facts:
            continue
        evidence_path_sources = {
            path: list(source_ids)
            for path, source_ids in projection.path_sources.items()
            if projection.path_audio_refs.get(path) == audio_ref
            or (len(projection.source_audio_refs) == 1 and path not in projection.path_audio_refs)
        }
        evidence_ids = list(
            dict.fromkeys(
                source_id
                for source_ids in evidence_path_sources.values()
                for source_id in source_ids
            )
        ) or list(projection.evidence_ids)
        state_node_id = state_node_by_audio_ref.get(audio_ref)
        snapshot_identity = {
            "turn_index": turn_index,
            "audio_ref": audio_ref,
            "state_node_id": state_node_id,
            "projection_fingerprint": projection.projection_fingerprint,
        }
        snapshot_id = (
            "historical-evidence:"
            + hashlib.sha256(
                json.dumps(
                    snapshot_identity,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        if snapshot_id in known_snapshot_ids:
            continue
        history.append(
            {
                "schema_version": _HISTORICAL_EVIDENCE_SNAPSHOT_SCHEMA,
                "snapshot_id": snapshot_id,
                "observed_turn_index": turn_index,
                "evidence_ids": evidence_ids,
                "projection_fingerprint": projection.projection_fingerprint,
                "state_binding": {
                    "audio_ref": audio_ref,
                    "state_node_id": state_node_id,
                    "coordinate_space": "audio_timeline_seconds",
                    "temporal_validity": "bound_state_only",
                },
                "facts": facts,
                "evidence_path_sources": evidence_path_sources,
            }
        )
        known_snapshot_ids.add(snapshot_id)


def _projection_facts_for_audio_ref(
    projection: WorkflowEvidenceProjection,
    *,
    audio_ref: str,
) -> dict[str, Any]:
    context = projection.planner_context
    report = context.get("perception_report")
    if isinstance(report, dict) and isinstance(report.get("audio_reports"), list):
        matches = [
            item
            for item in report["audio_reports"]
            if isinstance(item, dict)
            and item.get("audio_ref_id") == audio_ref
            and item.get("status") == "completed"
            and isinstance(item.get("report"), dict)
            and item["report"]
        ]
        if len(matches) != 1:
            return {}
        return rebase_reviewed_audio_namespace(
            {"perception_report": matches[0]["report"]},
            audio_ref=audio_ref,
        )
    if projection.source_audio_refs != [audio_ref]:
        return {}
    return rebase_reviewed_audio_namespace(context, audio_ref=audio_ref)


def _historical_reviewed_evidence_context(
    graph: EditSessionGraph,
    *,
    session_state,
    visible_audio_refs: set[str],
) -> dict[str, Any]:
    """Project graph-held facts without rebinding old temporal coordinates."""

    eligible_nodes = {node.node_id for node in session_state.nodes if node.session_eligible}
    graph_audio_refs = {
        node.audio_ref for node in session_state.nodes if node.audio_ref is not None
    }
    visible_refs = visible_audio_refs | graph_audio_refs
    raw_history = graph.metadata.get(_HISTORICAL_EVIDENCE_STORE_KEY, [])
    snapshots: list[dict[str, Any]] = []
    if isinstance(raw_history, list):
        for raw in raw_history:
            if not isinstance(raw, dict):
                continue
            binding = raw.get("state_binding")
            if not isinstance(binding, dict):
                continue
            audio_ref = str(binding.get("audio_ref") or "")
            state_node_id = binding.get("state_node_id")
            if audio_ref not in visible_refs:
                continue
            if state_node_id is not None and state_node_id not in eligible_nodes:
                continue
            snapshots.append(copy.deepcopy(raw))
    snapshots.sort(
        key=lambda item: (
            int(item.get("observed_turn_index") or 0),
            str(item.get("snapshot_id") or ""),
        )
    )
    return {
        "schema_version": _HISTORICAL_EVIDENCE_CONTEXT_SCHEMA,
        "authority": "session_graph_reviewed_evidence_history",
        "temporal_reuse_policy": (
            "Use lexical and semantic facts for reference resolution. Treat every "
            "timestamp as valid only in its bound audio_ref/state_node_id; request "
            "fresh grounding or an explicit lineage transform for any other state."
        ),
        "snapshots": snapshots,
    }


def _complete_session_round(
    *,
    graph: EditSessionGraph,
    round_index: int,
    request: str,
    result: WorkflowArmResult,
    case: WorkflowRuntimeCase,
    status: str,
    advance_current_audio: bool,
    final_output_path: str | None = None,
    state_only_completion: WorkflowStateOnlyCompletion | None = None,
    failure_stage: str | None = None,
    failure_codes: list[str] | None = None,
) -> None:
    executed_plan = (
        getattr(result.forward_result, "executed_plan", None)
        if result.forward_result is not None
        else getattr(result, "final_plan", None)
    )
    if executed_plan is None:
        executed_plan = getattr(result, "final_plan", None)
    session_transition = None
    if state_only_completion is not None:
        completion = result.state_only_completion
        assert completion is not None
        session_transition = {
            "selected_source_node_id": completion.selected_node_id,
            "selected_source_audio_ref": completion.selected_audio_ref,
            "selection_kind": completion.selection_kind,
            "operation_type": (
                "rollback" if completion.selection_kind == "restore_history" else "state_selection"
            ),
            "rollback_to_node_id": (
                completion.selected_node_id
                if completion.selection_kind == "restore_history"
                else None
            ),
            "changes_satisfied_by_base": list(completion.changes_satisfied_by_base),
        }
    graph.complete_round(
        round_index=round_index,
        request=request,
        status=status,
        plan=executed_plan,
        final_output_path=final_output_path,
        trace_result={
            "failure_stage": failure_stage or result.failure_stage,
            "failure_codes": (
                list(failure_codes) if failure_codes is not None else list(result.failure_codes)
            ),
        },
        context={"input_audios": [item.materialized_path for item in case.audio_bindings]},
        advance_current_audio=advance_current_audio,
        session_transition=session_transition,
    )


def _resolve_state_only_completion(
    *,
    graph: EditSessionGraph,
    result: WorkflowArmResult,
    case: WorkflowRuntimeCase,
    audio_ref_by_path: dict[str, str],
):
    """Resolve a Planner claim against the live graph and private audio bindings."""

    completion = result.state_only_completion
    if completion is None:
        raise ValueError("state-only completion payload is absent")
    if result.forward_result is not None or result.final_plan is not None:
        raise ValueError("state-only completion cannot include forward execution work")
    projected = graph.semantic_state_context(audio_ref_by_path=audio_ref_by_path)
    matches = [
        node
        for node in projected.nodes
        if node.node_id == completion.selected_node_id
        and node.audio_ref == completion.selected_audio_ref
        and node.session_eligible
    ]
    if len(matches) != 1:
        raise ValueError("selected node/audio pair is absent or ineligible")
    expected_kind = (
        "select_existing"
        if completion.selected_node_id == graph.current_node_id
        else "restore_history"
    )
    if completion.selection_kind != expected_kind:
        raise ValueError("state selection kind does not match the authoritative graph")
    selected_node = graph.nodes.get(completion.selected_node_id)
    if selected_node is None or not selected_node.primary_audio_path:
        raise ValueError("selected graph node has no reusable audio")
    bound_paths = {
        item.materialized_path
        for item in case.audio_bindings
        if item.audio_ref == completion.selected_audio_ref
    }
    if bound_paths != {selected_node.primary_audio_path}:
        raise ValueError("selected audio ref does not resolve to the graph node audio")
    return selected_node


__all__ = [
    "WorkflowScenarioResult",
    "WorkflowScenarioTurnResult",
    "arun_workflow_scenario",
]
