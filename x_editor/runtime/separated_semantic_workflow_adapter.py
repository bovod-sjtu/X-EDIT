"""Adapter from separated Reasoner/Planner providers to the unified workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Mapping
from typing import Any

from x_editor.runtime.session_base_anchor import SessionBaseAnchor

from x_editor.core.future_attempt_guidance import FutureAttemptGuidance
from x_editor.core.invocation_trace import ComponentInvocationTrace
from x_editor.core.reviewed_evidence import (
    PROJECTED_SCALAR_FIELDS,
    rebase_reviewed_audio_namespace,
    sanitize_reviewed_perception_facts,
)
from x_editor.core.schemas import PlannerCannotPlanReason
from x_editor.perception.capability_feedback import PerceptionCapabilityFeedback
from x_editor.perception.gateway_routing import (
    GatewayPerceptionRouteDecision,
    route_perception_observation_need,
)
from x_editor.perception.workflow_catalog import (
    unique_workflow_perception_evidence_companion_additions,
    workflow_perception_capability,
)
from x_editor.planner.memo_qualified import (
    bind_operation_contract_runtime_domains,
    project_operation_capability_contracts,
    validate_memo_qualified_planner_turn,
)
from x_editor.planner.reasoned_workflow import (
    MemoQualifiedPlannerInput,
    MemoQualifiedPlannerTurn,
    PlannerExecutionSemantics,
    PlannerPerceptionJob,
    project_perception_jobs_to_workflow_requests,
)
from x_editor.reasoner.contracts import (
    PriorTurnSemanticContext,
    ReasonerCallRequest,
    SemanticReasoningMemo,
    SessionStateContext,
    validate_semantic_memo_completion_authority,
    validate_semantic_memo_target_binding_authority,
)
from x_editor.runtime.semantic_target_bindings import materialize_semantic_target_bindings
from x_editor.runtime.workflow_orchestration import (
    WorkflowPerceptionRequest,
    WorkflowPlannerOutput,
    WorkflowReasonerCallRecord,
    WorkflowStateOnlyCompletion,
)


class SeparatedSemanticWorkflowError(RuntimeError):
    """Failure in the fresh Reasoner-to-Planner gateway."""


class SeparatedSemanticWorkflowContractError(SeparatedSemanticWorkflowError):
    """The fresh gateway output violates a frozen runtime contract."""

    failure_kind = "protocol"


class _PerceptionMissingCapability(RuntimeError):
    """A coherent observation need has no concrete Perception capability."""

    def __init__(
        self,
        *,
        job: PlannerPerceptionJob,
        route_decision: GatewayPerceptionRouteDecision,
    ) -> None:
        self.job = job
        self.route_decision = route_decision
        unsupported = ", ".join(route_decision.unsupported_evidence_fields) or "<none>"
        warnings = ", ".join(route_decision.warnings) or "no route"
        target_kinds = ", ".join(
            dict.fromkeys(item.kind for item in job.perception_targets)
        ) or "<none>"
        expected_fields = ", ".join(job.expected_evidence_fields) or "<none>"
        super().__init__(
            f"Perception job {job.job_id!r} has no registered route for its "
            f"observation need; target_kinds={target_kinds}; "
            f"expected_evidence_fields={expected_fields}; "
            f"temporal_precision={job.temporal_precision}; "
            f"unsupported_evidence_fields={unsupported}; warnings={warnings}"
        )

    def cannot_plan_reason(self) -> PlannerCannotPlanReason:
        return PlannerCannotPlanReason(
            code="missing_perception_route",
            message=str(self),
            target_ids=[self.job.job_id],
            operation_ids=(
                [self.job.perception_operation]
                if self.job.perception_operation != "auto"
                else []
            ),
        )


def _complete_unique_perception_evidence_companions(
    job: PlannerPerceptionJob,
) -> tuple[PlannerPerceptionJob, tuple[str, ...]]:
    additions = unique_workflow_perception_evidence_companion_additions(
        operation=job.perception_operation,
        target_kinds=(item.kind for item in job.perception_targets),
        expected_evidence_fields=job.expected_evidence_fields,
    )
    if not additions:
        return job, ()
    return (
        job.model_copy(
            update={
                "expected_evidence_fields": list(
                    dict.fromkeys([*job.expected_evidence_fields, *additions])
                )
            }
        ),
        additions,
    )


def _perception_request_id(parent_request_id: str, job_id: str, part_index: int) -> str:
    """Bind a turn-local job/part to its deterministic runtime invocation identity."""
    identity = json.dumps(
        [parent_request_id, job_id, part_index], ensure_ascii=True, separators=(",", ":")
    )
    return "perception:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _route_hint(operation: str) -> str | None:
    normalized = str(operation or "").strip()
    return None if normalized in {"", "auto"} else normalized


def _route_part_audio_shape_error(
    operation: str,
    *,
    supporting_audio_count: int,
) -> str | None:
    capability = workflow_perception_capability(operation)
    if capability is None:
        return "routed operation is not registered"
    if capability.audio_input_mode == "single_audio" and supporting_audio_count:
        return "single-audio capability cannot consume supporting audio refs"
    if capability.audio_input_mode == "paired_audio" and supporting_audio_count != 1:
        return "paired-audio capability requires exactly one supporting audio ref"
    if capability.audio_input_mode == "multi_audio" and supporting_audio_count < 1:
        return "multi-audio capability requires at least one supporting audio ref"
    return None


class SeparatedSemanticWorkflowPlannerAdapter:
    """Run Reasoner then Planner on each unified-runtime planning gateway."""

    def __init__(
        self,
        *,
        reasoner: Any,
        planner: Any,
        operation_capability_cards: Mapping[str, Mapping[str, Any] | Any],
        perception_capability_catalog: list[dict[str, Any]],
        perception_operation_params: Mapping[str, Mapping[str, Any]] | None = None,
        edit_operation_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.planner = planner
        self.operation_capability_cards = {
            str(operation): copy.deepcopy(card)
            for operation, card in operation_capability_cards.items()
        }
        self.candidate_operation_contracts = project_operation_capability_contracts(
            self.operation_capability_cards
        )
        self.perception_capability_catalog = copy.deepcopy(perception_capability_catalog)
        self.perception_operation_params = {
            str(operation): copy.deepcopy(dict(params))
            for operation, params in (perception_operation_params or {}).items()
        }
        self.edit_operation_params = {
            str(operation): copy.deepcopy(dict(params))
            for operation, params in (edit_operation_params or {}).items()
        }
        self.last_invocation_trace = None
        self.last_reasoner_invocation_trace = None
        self.last_reasoner_record: WorkflowReasonerCallRecord | None = None
        self.reasoner_history: dict[str, list[dict[str, Any]]] = {}
        self._prior_memos: dict[str, SemanticReasoningMemo] = {}
        self._selected_bases: dict[str, SessionBaseAnchor] = {}
        self._lock = threading.Lock()

    def plan(self, user_request: str, context: dict[str, Any]) -> WorkflowPlannerOutput:
        """Return one fresh workflow decision; no legacy Planner input is accepted."""

        with self._lock:
            return self._plan_locked(user_request, copy.deepcopy(context))

    def _plan_locked(
        self,
        user_request: str,
        context: dict[str, Any],
    ) -> WorkflowPlannerOutput:
        self.last_invocation_trace = None
        self.last_reasoner_invocation_trace = None
        self.last_reasoner_record = None
        run_id = str(context.get("workflow_run_id") or "").strip()
        if not run_id:
            raise SeparatedSemanticWorkflowContractError(
                "fresh semantic planning requires workflow_run_id"
            )
        call_index = int(context.get("workflow_planner_call_index") or 1)
        evidence_ids = [
            str(item) for item in context.get("workflow_perception_evidence_ids") or [] if str(item)
        ]
        prior_memo = self._prior_memos.get(run_id)
        if call_index > 1 and prior_memo is None:
            raise SeparatedSemanticWorkflowContractError(
                "evidence review cannot proceed without this run's initial semantic memo"
            )
        phase = "evidence_review" if prior_memo is not None else "initial_understanding"
        request_id = f"{run_id}:semantic:{call_index}"
        full_reviewed_evidence = _reviewed_evidence_context(
            context,
            enabled=bool(evidence_ids),
        )
        prompt_reviewed_evidence = _prompt_reviewed_evidence_projection(full_reviewed_evidence)
        reasoner_evidence_ids = list(
            dict.fromkeys(
                [
                    *evidence_ids,
                    *_historical_evidence_ids(prompt_reviewed_evidence),
                ]
            )
        )
        reasoner_request = ReasonerCallRequest(
            request_id=request_id,
            phase=phase,
            user_request=user_request,
            audio_asset_facts=copy.deepcopy(context.get("audio_asset_facts") or []),
            source_text=_optional_text(context.get("source_text")),
            source_audio_refs=_text_list(context.get("workflow_source_audio_refs")),
            reference_audio_refs=_text_list(context.get("reference_audio_paths")),
            prior_turn_contexts=[
                PriorTurnSemanticContext.model_validate(item)
                for item in context.get("workflow_prior_turn_semantic_context") or []
            ],
            session_state=(
                SessionStateContext.model_validate(context["workflow_session_state"])
                if context.get("workflow_session_state") is not None
                else None
            ),
            reviewed_evidence_ids=reasoner_evidence_ids,
            reviewed_evidence=prompt_reviewed_evidence,
            evidence_field_sources=copy.deepcopy(
                context.get("workflow_evidence_field_sources") or {}
            ),
            evidence_path_sources=copy.deepcopy(
                context.get("workflow_evidence_path_sources") or {}
            ),
            evidence_path_audio_refs=copy.deepcopy(
                context.get("workflow_evidence_path_audio_refs") or {}
            ),
            future_attempt_guidance=(
                FutureAttemptGuidance.model_validate(
                    context["workflow_future_attempt_guidance"]
                )
                if context.get("workflow_future_attempt_guidance") is not None
                else None
            ),
            prior_memo=(prior_memo.model_copy(deep=True) if prior_memo is not None else None),
            semantic_revision_policy=(
                "evidence_correctable" if prior_memo is not None else "freeze_initial"
            ),
        )
        return self._plan_from_request_locked(reasoner_request, context)

    def plan_from_request(
        self, request: ReasonerCallRequest, *,
        capability_feedback: dict[str, Any] | None = None,
        selected_base: SessionBaseAnchor | None = None,
        current_evidence_ids: list[str] | None = None,
    ) -> WorkflowPlannerOutput:
        """Continue an admitted planning checkpoint through ordinary validators.

        The caller owns checkpoint provenance and audio binding validation. This
        entrypoint does not admit new observations or relax any plan checks.
        """
        run_id, marker, index = request.request_id.rpartition(":semantic:")
        if not marker or not index.isdigit():
            raise ValueError("planning checkpoint requires its original gateway identity")
        context = {
            "workflow_run_id": run_id, "workflow_planner_call_index": int(index),
            "workflow_perception_evidence_ids": list(request.reviewed_evidence_ids if current_evidence_ids is None else current_evidence_ids),
            "workflow_source_audio_refs": list(request.source_audio_refs),
            "reference_audio_paths": list(request.reference_audio_refs),
            "audio_asset_facts": [x.model_dump(mode="json") for x in request.audio_asset_facts],
            "source_text": request.source_text,
            "workflow_prior_turn_semantic_context": [x.model_dump(mode="json") for x in request.prior_turn_contexts],
            "workflow_evidence_path_sources": copy.deepcopy(request.evidence_path_sources),
            "workflow_evidence_path_audio_refs": copy.deepcopy(request.evidence_path_audio_refs),
            "workflow_perception_capability_feedback": copy.deepcopy(capability_feedback),
        }
        with self._lock:
            self.last_invocation_trace = self.last_reasoner_invocation_trace = None
            self.last_reasoner_record = None
            self._selected_bases.pop(run_id, None)
            if selected_base is not None:
                self._selected_bases[run_id] = selected_base
            if hasattr(self.reasoner, "bind_capability_feedback"):
                self.reasoner.bind_capability_feedback(capability_feedback)
            return self._plan_from_request_locked(request.model_copy(deep=True), context)

    def _plan_from_request_locked(
        self, reasoner_request: ReasonerCallRequest, context: dict[str, Any],
    ) -> WorkflowPlannerOutput:
        user_request = reasoner_request.user_request
        run_id = str(context["workflow_run_id"])
        call_index = int(context["workflow_planner_call_index"])
        prior_memo = reasoner_request.prior_memo
        evidence_ids = list(context.get("workflow_perception_evidence_ids") or [])
        prompt_reviewed_evidence = reasoner_request.reviewed_evidence
        try:
            memo = SemanticReasoningMemo.model_validate(self.reasoner.reason(reasoner_request))
            validate_semantic_memo_completion_authority(
                memo,
                request=reasoner_request,
            )
            validate_semantic_memo_target_binding_authority(
                memo,
                request=reasoner_request,
            )
            _validate_reasoner_state_handoff(memo, request=reasoner_request)
            frozen_base = self._selected_bases.get(run_id)
            if (
                frozen_base is not None
                and memo.selected_base_state_node_id is not None
                and not frozen_base.permits(
                    memo.selected_base_state_node_id, memo.selected_base_audio_ref,
                    state=reasoner_request.session_state,
                    bound_audio_refs=set(reasoner_request.source_audio_refs + reasoner_request.reference_audio_refs),
                    allow_asset_selection=True,
                )
            ):
                raise SeparatedSemanticWorkflowContractError(
                    "evidence-review Reasoner changed the Session Graph base selected "
                    "before Perception"
                )
        except Exception as exc:
            self.last_reasoner_invocation_trace = getattr(
                self.reasoner,
                "last_invocation_trace",
                None,
            )
            self.last_reasoner_invocation_trace = _bound_trace(
                self.last_reasoner_invocation_trace,
                gateway_call_id=f"reasoner:{call_index}",
            )
            self.last_reasoner_record = WorkflowReasonerCallRecord(
                call_index=call_index,
                call_id=f"reasoner:{call_index}",
                status="failed",
                request_fingerprint=reasoner_request.fingerprint,
                invocation_trace=self.last_reasoner_invocation_trace,
                error_type=type(exc).__name__,
                error_message=" ".join(str(exc).split())[:2000],
            )
            raise SeparatedSemanticWorkflowError(
                f"semantic Reasoner failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.last_reasoner_invocation_trace = getattr(
            self.reasoner,
            "last_invocation_trace",
            None,
        )
        self.last_reasoner_invocation_trace = _bound_trace(
            self.last_reasoner_invocation_trace,
            gateway_call_id=f"reasoner:{call_index}",
        )
        self.last_reasoner_record = WorkflowReasonerCallRecord(
            call_index=call_index,
            call_id=f"reasoner:{call_index}",
            status="completed",
            request_fingerprint=reasoner_request.fingerprint,
            memo=memo.model_dump(mode="json"),
            invocation_trace=self.last_reasoner_invocation_trace,
        )
        self.reasoner_history.setdefault(run_id, []).append(
            {
                "call_index": call_index,
                "request_fingerprint": reasoner_request.fingerprint,
                "memo": memo.model_dump(mode="json"),
                "invocation_trace": (
                    self.last_reasoner_invocation_trace.model_dump(mode="json")
                    if hasattr(self.last_reasoner_invocation_trace, "model_dump")
                    else None
                ),
            }
        )

        candidate_contracts = self.candidate_operation_contracts

        source_audio_refs = _text_list(context.get("workflow_source_audio_refs"))
        reference_audio_refs = _text_list(context.get("reference_audio_paths"))
        candidate_contracts = bind_operation_contract_runtime_domains(
            candidate_contracts,
            source_audio_refs=source_audio_refs,
            reference_audio_refs=reference_audio_refs,
            frozen_operation_params=self.edit_operation_params,
        )

        materialized_evidence = materialize_semantic_target_bindings(
            prompt_reviewed_evidence,
            memo=memo,
            evidence_path_sources=context.get("workflow_evidence_path_sources") or {},
            evidence_path_audio_refs=context.get("workflow_evidence_path_audio_refs") or {},
        )

        planner_input = MemoQualifiedPlannerInput(
            planner_request_id=memo.request_id,
            phase=("execution_planning" if prior_memo is not None else "evidence_planning"),
            user_request=user_request,
            execution_request=memo.remaining_execution_request,
            audio_asset_facts=copy.deepcopy(context.get("audio_asset_facts") or []),
            source_text=_optional_text(context.get("source_text")),
            semantic_memo=memo,
            source_audio_refs=source_audio_refs,
            reference_audio_refs=reference_audio_refs,
            prior_turn_contexts=[
                PriorTurnSemanticContext.model_validate(item)
                for item in context.get("workflow_prior_turn_semantic_context") or []
            ],
            session_state=reasoner_request.session_state,
            reviewed_evidence_ids=evidence_ids,
            reviewed_evidence=materialized_evidence.reviewed_evidence,
            evidence_path_sources=materialized_evidence.evidence_path_sources,
            evidence_path_audio_refs=materialized_evidence.evidence_path_audio_refs,
            future_attempt_guidance=reasoner_request.future_attempt_guidance,
            perception_capability_feedback=context.get("workflow_perception_capability_feedback"),
            perception_capability_catalog=copy.deepcopy(self.perception_capability_catalog),
            candidate_operation_contracts=[
                item.model_copy(deep=True) for item in candidate_contracts
            ],
        )
        try:
            turn = MemoQualifiedPlannerTurn.model_validate(self.planner.plan(planner_input))
            turn = validate_memo_qualified_planner_turn(
                turn,
                planner_input=planner_input,
                capability_cards=self.operation_capability_cards,
            )
        except Exception as exc:
            self.last_invocation_trace = _bound_trace(
                getattr(self.planner, "last_invocation_trace", None),
                gateway_call_id=f"planner:{call_index}",
            )
            failure_class = (
                SeparatedSemanticWorkflowContractError
                if getattr(exc, "failure_kind", None) == "protocol" or isinstance(exc, ValueError)
                else SeparatedSemanticWorkflowError
            )
            raise failure_class(
                f"memo-qualified Planner failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.last_invocation_trace = _bound_trace(
            getattr(self.planner, "last_invocation_trace", None),
            gateway_call_id=f"planner:{call_index}",
        )

        if turn.decision == "request_perception":
            self._prior_memos[run_id] = memo.model_copy(deep=True)
            anchor = self._selected_bases.get(run_id)
            if anchor is not None and not anchor.permits(
                turn.selected_base_state_node_id, turn.selected_base_audio_ref,
                state=planner_input.session_state,
                bound_audio_refs=set(source_audio_refs + reference_audio_refs),
                allow_asset_selection=True,
            ):
                raise SeparatedSemanticWorkflowContractError(
                    "Perception Planner changed the frozen Session Graph base"
                )
            # A later explicit Reasoner choice pins the asset. Otherwise keep
            # the original history anchor, rather than overwriting it each round.
            if anchor is None or memo.selected_base_state_node_id is not None:
                self._selected_bases[run_id] = SessionBaseAnchor.from_turn(turn, memo)
            try:
                perception_requests = self._perception_requests(turn, planner_input)
            except _PerceptionMissingCapability as exc:
                alternatives = exc.route_decision.to_dict()["alternative_decompositions"]
                if not alternatives and exc.job.recognition_context:
                    fact_route = route_perception_observation_need(
                        expected_evidence_fields=exc.job.expected_evidence_fields,
                        temporal_precision=exc.job.temporal_precision,
                        available_operations=[
                            item["operation"] for item in self.perception_capability_catalog
                            if item.get("supports_recognition_context") is True
                        ],
                    )
                    if fact_route.route_parts and not fact_route.unsupported_evidence_fields:
                        return WorkflowPlannerOutput(
                            decision="request_replan",
                            capability_feedback=PerceptionCapabilityFeedback(
                                code="observation_target_exceeds_output_contract",
                                requested_job=exc.job.model_dump(mode="json"),
                                alternatives=[fact_route.to_dict()["route_parts"]],
                                instruction=(
                                    "Contextual recognition can produce the requested facts, "
                                    "but cannot itself classify this semantic target. Request "
                                    "source_text and any needed word timing, carrying the "
                                    "uncertain vocabulary/domain in recognition_context. "
                                    "Omit a classification target from that recognition job; "
                                    "Reasoner reviews its returned transcript. Keep the original "
                                    "audio, observation objective and user goal unchanged."
                                ),
                            ),
                            reason="Ask Planner to separate recognition facts from semantic classification.",
                            invocation_trace=self.last_invocation_trace,
                        )
                if len(alternatives) > 1 and not exc.route_decision.unsupported_evidence_fields:
                    return WorkflowPlannerOutput(
                        decision="request_replan",
                        capability_feedback=PerceptionCapabilityFeedback(
                            code="ambiguous_observation_decomposition",
                            requested_job=exc.job.model_dump(mode="json"),
                            alternatives=alternatives,
                        ),
                        reason="Ask Planner to disambiguate a supported observation job.",
                        invocation_trace=self.last_invocation_trace,
                    )
                self._prior_memos.pop(run_id, None)
                self._selected_bases.pop(run_id, None)
                reason = exc.cannot_plan_reason()
                return WorkflowPlannerOutput(
                    decision="cannot_plan",
                    missing_information=[reason.message],
                    cannot_plan_reasons=[reason],
                    reason=reason.message,
                    invocation_trace=self.last_invocation_trace,
                )
            except Exception:
                self._mark_planner_candidate_failed(turn, call_index=call_index)
                raise
            return WorkflowPlannerOutput(
                decision="request_perception",
                perception_requests=perception_requests,
                missing_information=list(turn.missing_information),
                reason=turn.decision_summary,
                invocation_trace=self.last_invocation_trace,
            )
        selected_base = self._selected_bases.get(run_id)
        if turn.decision == "cannot_plan":
            self._prior_memos.pop(run_id, None)
            self._selected_bases.pop(run_id, None)
            cannot_plan_reasons = list(turn.cannot_plan_reasons)
            if not cannot_plan_reasons:
                cannot_plan_reasons = [
                    {
                        "code": (
                            "conflicting_evidence"
                            if turn.semantic_conflict is not None
                            else "other"
                        ),
                        "message": (
                            turn.semantic_conflict
                            or "; ".join(turn.missing_information)
                            or turn.decision_summary
                            or "Planner could not produce a legal plan."
                        ),
                    }
                ]
            return WorkflowPlannerOutput(
                decision="cannot_plan",
                missing_information=list(turn.missing_information),
                cannot_plan_reasons=cannot_plan_reasons,
                reason=turn.decision_summary,
                invocation_trace=self.last_invocation_trace,
            )
        if turn.decision == "complete_from_session_state":
            if (
                selected_base is not None
                and not selected_base.permits(
                    turn.selected_base_state_node_id, turn.selected_base_audio_ref,
                    state=planner_input.session_state,
                    bound_audio_refs=set(source_audio_refs + reference_audio_refs),
                    allow_asset_selection=False,
                )
            ):
                self._mark_planner_candidate_failed(turn, call_index=call_index)
                raise SeparatedSemanticWorkflowContractError(
                    "state-only Planner changed the Session Graph base selected before Perception"
                )
            self._prior_memos.pop(run_id, None)
            self._selected_bases.pop(run_id, None)
            assert turn.selected_base_state_node_id is not None
            assert turn.selected_base_audio_ref is not None
            assert turn.state_completion_kind is not None
            return WorkflowPlannerOutput(
                decision="complete_from_session_state",
                state_only_completion=WorkflowStateOnlyCompletion(
                    selected_node_id=turn.selected_base_state_node_id,
                    selected_audio_ref=turn.selected_base_audio_ref,
                    selection_kind=turn.state_completion_kind,
                    changes_satisfied_by_base=list(turn.changes_satisfied_by_base),
                ),
                reason=turn.decision_summary,
                invocation_trace=self.last_invocation_trace,
            )
        assert turn.plan is not None
        if (
            selected_base is not None
            and not selected_base.permits(
                turn.selected_base_state_node_id, turn.selected_base_audio_ref,
                state=planner_input.session_state,
                bound_audio_refs=set(source_audio_refs + reference_audio_refs),
                allow_asset_selection=True,
            )
        ):
            self._mark_planner_candidate_failed(turn, call_index=call_index)
            raise SeparatedSemanticWorkflowContractError(
                "execution Planner changed the Session Graph base selected before Perception"
            )
        self._prior_memos.pop(run_id, None)
        self._selected_bases.pop(run_id, None)
        plan = turn.plan.model_copy(deep=True)
        execution_semantics = turn.execution_semantics or PlannerExecutionSemantics()
        for step in plan.steps:
            frozen = self.edit_operation_params.get(step.operation, {})
            for key, value in frozen.items():
                if key in step.params and step.params[key] != value:
                    self._mark_planner_candidate_failed(turn, call_index=call_index)
                    raise SeparatedSemanticWorkflowContractError(
                        f"Planner argument conflicts with frozen runtime value: "
                        f"{step.operation}.params.{key}"
                    )
                step.params[key] = copy.deepcopy(value)
            if plan.metadata.get("command_program") and frozen:
                # Same runtime-owned defaults as every other representation.
                # Record their provenance so direct execution need not relax
                # its check that authored arguments remained unchanged.
                plan.metadata["command_program"].setdefault("frozen_params", {})[step.id] = copy.deepcopy(frozen)
        plan.metadata["separated_semantic_planning"] = {
            "schema_version": "x-editor.separated-semantic-planning-audit/v1",
            "reasoner_request_fingerprint": reasoner_request.fingerprint,
            "semantic_memo": memo.model_dump(mode="json"),
            "planner_contract": turn.schema_version,
            "tool_selection": (
                turn.tool_selection.model_dump(mode="json")
                if turn.tool_selection is not None
                else None
            ),
            "execution_semantics": execution_semantics.model_dump(mode="json"),
        }
        plan.metadata["workflow_audio_usage_bindings"] = [
            item.model_dump(mode="json")
            for item in execution_semantics.audio_usage_bindings
        ]
        if turn.selected_base_state_node_id is not None:
            session_graph_metadata: dict[str, Any] = {
                "selected_source_node_id": turn.selected_base_state_node_id,
                "selected_source_audio_ref": turn.selected_base_audio_ref,
                "selection_kind": "semantic_state_base",
                "operation_type": {
                    "edit_current": "edit",
                    "branch_from_history": "branch_edit",
                    "restore_history": "rollback",
                }[str(turn.state_transition_kind)],
            }
            if turn.state_transition_kind == "restore_history":
                session_graph_metadata["rollback_to_node_id"] = turn.selected_base_state_node_id
            plan.metadata["session_graph"] = {
                **session_graph_metadata,
            }
        return WorkflowPlannerOutput(
            decision="execute",
            plan=plan,
            reason=turn.decision_summary,
            invocation_trace=self.last_invocation_trace,
        )

    def _mark_planner_candidate_failed(
        self,
        turn: MemoQualifiedPlannerTurn,
        *,
        call_index: int,
    ) -> None:
        self.last_invocation_trace = _failed_trace_with_provider_output(
            self.last_invocation_trace,
            gateway_call_id=f"planner:{call_index}",
            provider_output=turn.model_dump(mode="json"),
        )

    def _perception_requests(
        self,
        turn: MemoQualifiedPlannerTurn,
        planner_input: MemoQualifiedPlannerInput,
    ) -> list[WorkflowPerceptionRequest]:
        known_refs = {
            *planner_input.source_audio_refs,
            *planner_input.reference_audio_refs,
        }
        available_operations = [
            str(item.get("operation") or "").strip()
            for item in planner_input.perception_capability_catalog
            if str(item.get("operation") or "").strip()
        ]
        normalized_jobs: list[PlannerPerceptionJob] = []
        route_audits: dict[str, dict[str, Any]] = {}
        route_normalizations: dict[str, dict[str, Any]] = {}
        evidence_normalizations: dict[str, dict[str, Any]] = {}
        job_decompositions: dict[str, dict[str, Any]] = {}
        for job in turn.perception_jobs:
            job_refs = {job.primary_audio_ref, *job.supporting_audio_refs}
            if not job_refs.issubset(known_refs):
                raise SeparatedSemanticWorkflowContractError(
                    "Perception job audio refs are outside runtime inputs"
                )
            route_decision = route_perception_observation_need(
                target_kinds=(item.kind for item in job.perception_targets),
                expected_evidence_fields=job.expected_evidence_fields,
                temporal_precision=job.temporal_precision,
                available_operations=[
                    operation for operation in available_operations
                    if not job.recognition_context or (
                        (capability := workflow_perception_capability(operation)) is not None
                        and capability.supports_recognition_context
                    )
                ],
                selected_operation_hint=_route_hint(job.perception_operation),
            )
            if not route_decision.route_parts:
                raise _PerceptionMissingCapability(
                    job=job,
                    route_decision=route_decision,
                )

            for part_index, part in enumerate(route_decision.route_parts):
                shape_error = _route_part_audio_shape_error(
                    part.operation,
                    supporting_audio_count=len(job.supporting_audio_refs),
                )
                if shape_error is not None:
                    raise SeparatedSemanticWorkflowContractError(
                        f"Perception job {job.job_id!r} has an impossible audio shape "
                        f"for {part.operation!r}: {shape_error}"
                    )
                part_id = _perception_request_id(
                    planner_input.planner_request_id, job.job_id, part_index
                )
                part_target_kinds = set(part.target_kinds)
                part_targets = [
                    item for item in job.perception_targets if item.kind in part_target_kinds
                ]
                keep_semantic_context = bool(part_targets) or len(route_decision.route_parts) == 1
                part_job = job.model_copy(
                    update={
                        "job_id": part_id,
                        "perception_operation": part.operation,
                        "expected_evidence_fields": list(part.expected_evidence_fields),
                        "perception_targets": part_targets,
                        "requirements": (
                            list(job.requirements) if keep_semantic_context else []
                        ),
                        "target_hints": (
                            list(job.target_hints) if keep_semantic_context else []
                        ),
                        "evidence_questions": (
                            list(job.evidence_questions) if keep_semantic_context else []
                        ),
                    }
                )
                part_job, part_additions = _complete_unique_perception_evidence_companions(
                    part_job
                )
                normalized_jobs.append(part_job)
                route_audits[part_id] = {
                    "source_job_id": job.job_id,
                    "hint_operation": _route_hint(job.perception_operation),
                    "materialized_operation": part_job.perception_operation,
                    "gateway_decision": route_decision.to_dict(),
                }
                if part_additions:
                    evidence_normalizations[part_id] = {
                        "operation": part_job.perception_operation,
                        "added_evidence_fields": list(part_additions),
                        "reason": (
                            "the selected capability declares one unambiguous evidence "
                            "companion for the authored target kind"
                        ),
                    }
                if _route_hint(job.perception_operation) not in {
                    None,
                    part_job.perception_operation,
                }:
                    route_normalizations[part_id] = {
                        "source_operation": job.perception_operation,
                        "normalized_operation": part_job.perception_operation,
                        "reason": (
                            "the Perception Gateway routed the declared observation "
                            "need; the model-authored operation is compatibility hint "
                            "metadata, not authority"
                        ),
                    }
                if len(route_decision.route_parts) > 1:
                    job_decompositions[part_id] = {
                        "source_job_id": job.job_id,
                        "part_index": part_index,
                        "part_count": len(route_decision.route_parts),
                        "selected_operation": job.perception_operation,
                        "assigned_operation": part_job.perception_operation,
                        "assigned_evidence_fields": list(part.expected_evidence_fields),
                        "assigned_target_kinds": list(part.target_kinds),
                        "reason": (
                            "the Perception Gateway materialized the observation need "
                            "as independently routed capability parts"
                        ),
                    }

        requests: list[WorkflowPerceptionRequest] = []
        for payload in project_perception_jobs_to_workflow_requests(
            normalized_jobs,
            edit_objective=planner_input.execution_request,
            candidate_operations=turn.candidate_operations,
        ):
            request_refs = set(payload.get("audio_refs") or [])
            if not request_refs.issubset(known_refs):
                raise SeparatedSemanticWorkflowContractError(
                    "Perception job audio refs are outside runtime inputs"
                )
            operation = str(payload.get("operation") or "")
            metadata = dict(payload.get("metadata") or {})
            capability = workflow_perception_capability(operation)
            if capability is not None:
                metadata["perception_capability"] = capability.to_dict()
                metadata["supported_evidence_types"] = list(capability.evidence_types)
                metadata["evidence_requirement_rules"] = list(capability.requirement_rules)
            request_id = str(payload.get("request_id") or "")
            job = next((item for item in normalized_jobs if item.job_id == request_id), None)
            if job is not None and job.observation_need_ids:
                metadata["observation_need_ids"] = list(job.observation_need_ids)
            if request_id in route_audits:
                metadata["source_job_id"] = route_audits[request_id]["source_job_id"]
                metadata["parent_planner_request_id"] = planner_input.planner_request_id
                metadata["perception_gateway_routing"] = copy.deepcopy(
                    route_audits[request_id]
                )
            if request_id in route_normalizations:
                metadata["perception_route_normalization"] = copy.deepcopy(
                    route_normalizations[request_id]
                )
            if request_id in evidence_normalizations:
                metadata["perception_evidence_normalization"] = copy.deepcopy(
                    evidence_normalizations[request_id]
                )
            if request_id in job_decompositions:
                metadata["perception_job_decomposition"] = copy.deepcopy(
                    job_decompositions[request_id]
                )
            payload["metadata"] = metadata
            params = dict(payload.get("params") or {})
            for key, value in self.perception_operation_params.get(
                operation,
                {},
            ).items():
                if key in params and params[key] != value:
                    raise SeparatedSemanticWorkflowContractError(
                        f"Perception argument conflicts with frozen runtime value: {key}"
                    )
                params[key] = copy.deepcopy(value)
            payload["params"] = params
            requests.append(WorkflowPerceptionRequest.model_validate(payload))
        return requests


def _reviewed_evidence_context(
    context: Mapping[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    reviewed: dict[str, Any] = {}
    if enabled:
        fields = {"perception_report", *PROJECTED_SCALAR_FIELDS}
        reviewed.update(
            {key: copy.deepcopy(context[key]) for key in sorted(fields) if key in context}
        )
    historical = context.get("workflow_historical_reviewed_evidence")
    if historical is not None:
        reviewed["historical_state_evidence"] = _validated_historical_state_evidence(
            historical,
            session_state=context.get("workflow_session_state"),
            visible_audio_refs={
                *_text_list(context.get("workflow_source_audio_refs")),
                *_text_list(context.get("workflow_reference_audio_refs")),
                *_text_list(context.get("reference_audio_paths")),
            },
        )
    return reviewed


_REASONER_HISTORICAL_REPORT_FIELDS = frozenset(
    {
        "acoustic_segments",
        "audio_caption_segments",
        "audio_index",
        "audio_quality",
        "audio_ref_id",
        "background",
        "candidate_rubric_judgements",
        "candidate_segments",
        "event_segments",
        "events",
        "language",
        "notable_events",
        "prosody_segments",
        "role",
        "segments",
        "selected_speaker_refs",
        "selected_speaker_ids",
        "selected_spans",
        "source_text",
        "source_text_source",
        "speaker_profiles",
        "speaker_review_summary",
        "speaker_segments",
        "speakers",
        "target_bindings",
        "uncertainty",
        "uncertainties",
    }
)
_REASONER_HISTORICAL_DETAIL_FIELDS = frozenset(
    {
        "char_end",
        "char_index",
        "char_start",
        "detector_refs",
        "end",
        "end_sec",
        "evidence_slice_ids",
        "source_refs",
        "start",
        "start_sec",
        "time_sec",
        "word_ids",
    }
)


def _prompt_reviewed_evidence_projection(
    reviewed_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep audit evidence intact while giving models only semantic history.

    Current-turn evidence remains complete so Planner can bind executable arguments.
    Historical exact coordinates and path provenance stay in the Session Graph
    sidecar because they refer to an older artifact state; models must request fresh
    grounding before reusing coordinates on another state.
    """

    projected = copy.deepcopy(dict(reviewed_evidence))
    historical = projected.get("historical_state_evidence")
    if not isinstance(historical, Mapping):
        return projected
    compact_snapshots: list[dict[str, Any]] = []
    for snapshot in historical.get("snapshots") or []:
        if not isinstance(snapshot, Mapping):
            continue
        facts = snapshot.get("facts")
        semantic_facts = _reasoner_historical_semantic_facts(facts)
        compact_snapshots.append(
            {
                "snapshot_id": snapshot.get("snapshot_id"),
                "observed_turn_index": snapshot.get("observed_turn_index"),
                "evidence_ids": copy.deepcopy(snapshot.get("evidence_ids") or []),
                "projection_fingerprint": snapshot.get("projection_fingerprint"),
                "state_binding": copy.deepcopy(snapshot.get("state_binding") or {}),
                "facts": semantic_facts,
            }
        )
    projected["historical_state_evidence"] = {
        "schema_version": "x-editor.reasoner-historical-semantic-evidence/v1",
        "authority": historical.get("authority"),
        "temporal_reuse_policy": historical.get("temporal_reuse_policy"),
        "detail_policy": (
            "Exact timestamps and path provenance remain in Session Graph audit evidence; "
            "request fresh grounding before using coordinates in another state."
        ),
        "snapshots": compact_snapshots,
    }
    return projected


def _reasoner_historical_semantic_facts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {
        key: copy.deepcopy(value[key]) for key in sorted(PROJECTED_SCALAR_FIELDS) if key in value
    }
    report = value.get("perception_report")
    if isinstance(report, Mapping):
        compact_report = {
            str(key): _drop_historical_execution_detail(item)
            for key, item in report.items()
            if str(key) in _REASONER_HISTORICAL_REPORT_FIELDS
        }
        if compact_report:
            projected["perception_report"] = compact_report
    return projected


def _drop_historical_execution_detail(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _drop_historical_execution_detail(item)
            for key, item in value.items()
            if str(key) not in _REASONER_HISTORICAL_DETAIL_FIELDS
        }
    if isinstance(value, list):
        return [_drop_historical_execution_detail(item) for item in value]
    return copy.deepcopy(value)


def _validated_historical_state_evidence(
    value: Any,
    *,
    session_state: Any,
    visible_audio_refs: set[str],
) -> dict[str, Any]:
    """Admit only the Session Graph's compact, state-bound evidence projection."""

    if not isinstance(value, Mapping):
        raise SeparatedSemanticWorkflowContractError(
            "historical reviewed evidence must be a mapping"
        )
    payload = copy.deepcopy(dict(value))
    allowed_root_fields = {
        "schema_version",
        "authority",
        "temporal_reuse_policy",
        "snapshots",
    }
    unexpected_root = set(payload) - allowed_root_fields
    if unexpected_root:
        raise SeparatedSemanticWorkflowContractError(
            "historical reviewed evidence exposes unsupported context fields: "
            f"{sorted(unexpected_root)}"
        )
    if payload.get("schema_version") != "x-editor.historical-reviewed-evidence-context/v1":
        raise SeparatedSemanticWorkflowContractError(
            "historical reviewed evidence has an unsupported schema"
        )
    if payload.get("authority") != "session_graph_reviewed_evidence_history":
        raise SeparatedSemanticWorkflowContractError(
            "historical reviewed evidence is not Session Graph authoritative"
        )
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        raise SeparatedSemanticWorkflowContractError(
            "historical reviewed evidence snapshots must be a list"
        )
    if len(snapshots) > 128:
        raise SeparatedSemanticWorkflowContractError(
            "historical reviewed evidence exceeds the Session Graph context limit"
        )
    session_nodes = []
    if isinstance(session_state, Mapping):
        raw_nodes = session_state.get("nodes")
        if isinstance(raw_nodes, list):
            session_nodes = [item for item in raw_nodes if isinstance(item, Mapping)]
    eligible_state_pairs = {
        (str(node.get("node_id") or ""), audio_ref)
        for node in session_nodes
        if node.get("session_eligible") is True
        for audio_ref in _session_node_audio_refs(node)
    }
    allowed_snapshot_fields = {
        "schema_version",
        "snapshot_id",
        "observed_turn_index",
        "evidence_ids",
        "projection_fingerprint",
        "state_binding",
        "facts",
        "evidence_path_sources",
    }
    allowed_fact_fields = {"perception_report", *PROJECTED_SCALAR_FIELDS}
    snapshot_ids: set[str] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence snapshot must be a mapping"
            )
        unexpected = set(snapshot) - allowed_snapshot_fields
        if unexpected:
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence snapshot exposes unsupported fields: "
                f"{sorted(unexpected)}"
            )
        if snapshot.get("schema_version") != "x-editor.historical-reviewed-evidence/v1":
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence snapshot has an unsupported schema"
            )
        snapshot_id = str(snapshot.get("snapshot_id") or "")
        if not snapshot_id or snapshot_id in snapshot_ids:
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence snapshot ids must be non-empty and unique"
            )
        snapshot_ids.add(snapshot_id)
        evidence_ids = snapshot.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or any(not isinstance(item, str) or not item for item in evidence_ids)
        ):
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence requires non-empty evidence ids"
            )
        binding = snapshot.get("state_binding")
        if not isinstance(binding, Mapping) or not str(binding.get("audio_ref") or ""):
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence requires an audio-ref state binding"
            )
        unexpected_binding = set(binding) - {
            "audio_ref",
            "state_node_id",
            "coordinate_space",
            "temporal_validity",
        }
        if unexpected_binding:
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence exposes unsupported state binding fields: "
                f"{sorted(unexpected_binding)}"
            )
        if (
            binding.get("coordinate_space") != "audio_timeline_seconds"
            or binding.get("temporal_validity") != "bound_state_only"
        ):
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence must keep temporal values state-bound"
            )
        audio_ref = str(binding["audio_ref"])
        state_node_id = binding.get("state_node_id")
        if state_node_id is None:
            if audio_ref not in visible_audio_refs:
                raise SeparatedSemanticWorkflowContractError(
                    "historical reviewed evidence references an unavailable audio ref"
                )
        elif (str(state_node_id), audio_ref) not in eligible_state_pairs:
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence state binding is absent from the "
                "current Session Graph"
            )
        facts = snapshot.get("facts")
        if not isinstance(facts, Mapping) or not facts:
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence requires reviewed facts"
            )
        unexpected_facts = set(facts) - allowed_fact_fields
        if unexpected_facts:
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence exposes unreviewed fact fields: "
                f"{sorted(unexpected_facts)}"
            )
        snapshot["facts"] = sanitize_reviewed_perception_facts(
            rebase_reviewed_audio_namespace(facts, audio_ref=audio_ref),
            audio_refs=[audio_ref],
        )
        path_sources = snapshot.get("evidence_path_sources")
        if not isinstance(path_sources, Mapping) or any(
            not isinstance(source_ids, list) or not set(source_ids).issubset(set(evidence_ids))
            for source_ids in path_sources.values()
        ):
            raise SeparatedSemanticWorkflowContractError(
                "historical reviewed evidence path provenance is invalid"
            )
    payload["temporal_reuse_policy"] = (
        "Use lexical and semantic facts for reference resolution. Treat every "
        "timestamp as valid only in its bound audio_ref/state_node_id; request "
        "fresh grounding or an explicit lineage transform for any other state."
    )
    return payload


def _historical_evidence_ids(reviewed_evidence: Mapping[str, Any]) -> list[str]:
    historical = reviewed_evidence.get("historical_state_evidence")
    if not isinstance(historical, Mapping):
        return []
    snapshots = historical.get("snapshots")
    if not isinstance(snapshots, list):
        return []
    return list(
        dict.fromkeys(
            evidence_id
            for snapshot in snapshots
            if isinstance(snapshot, Mapping)
            for evidence_id in snapshot.get("evidence_ids") or []
            if isinstance(evidence_id, str) and evidence_id
        )
    )


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item)))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _validate_reasoner_state_handoff(
    memo: SemanticReasoningMemo,
    *,
    request: ReasonerCallRequest,
) -> None:
    """Check graph identity and handoff completeness without editing semantics."""

    state = request.session_state
    selected = (
        memo.selected_base_state_node_id,
        memo.selected_base_audio_ref,
    )
    if state is None:
        if selected != (None, None):
            raise SeparatedSemanticWorkflowContractError(
                "Reasoner selected a Session Graph base when no graph was supplied"
            )
        if memo.remaining_execution_request is None:
            raise SeparatedSemanticWorkflowContractError(
                "Reasoner cannot complete from Session Graph state when no graph was supplied"
            )
        return
    if selected == (None, None):
        return
    matches = [
        node
        for node in state.nodes
        if node.node_id == memo.selected_base_state_node_id
        and node.contains_audio_ref(memo.selected_base_audio_ref)
        and node.session_eligible
    ]
    if len(matches) != 1:
        raise SeparatedSemanticWorkflowContractError(
            "Reasoner must select exactly one eligible Session Graph node/audio pair"
        )


def _session_node_audio_refs(node: Mapping[str, Any]) -> set[str]:
    refs = {str(node.get("audio_ref") or "")}
    slots = node.get("audio_slots")
    if isinstance(slots, list):
        refs.update(
            str(item.get("audio_ref") or "")
            for item in slots
            if isinstance(item, Mapping)
        )
    refs.discard("")
    return refs


def _bound_trace(
    raw: Any,
    *,
    gateway_call_id: str,
) -> ComponentInvocationTrace | None:
    if raw is None:
        return None
    try:
        trace = ComponentInvocationTrace.model_validate(raw)
        return ComponentInvocationTrace.model_validate(
            {
                **trace.model_dump(mode="python"),
                "gateway_call_id": gateway_call_id,
            }
        )
    except Exception:  # noqa: BLE001 - malformed diagnostics cannot gain authority
        return None


def _failed_trace_with_provider_output(
    raw: Any,
    *,
    gateway_call_id: str,
    provider_output: Any,
) -> ComponentInvocationTrace | None:
    trace = _bound_trace(raw, gateway_call_id=gateway_call_id)
    if trace is None:
        return None
    rendered = json.dumps(provider_output, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered) > 6000:
        rendered = rendered[:6000] + "...<truncated>"
    metadata = copy.deepcopy(trace.metadata)
    metadata["provider_output_on_failure"] = rendered
    return ComponentInvocationTrace.model_validate(
        {
            **trace.model_dump(mode="python"),
            "status": "failed",
            "metadata": metadata,
        }
    )


__all__ = [
    "SeparatedSemanticWorkflowContractError",
    "SeparatedSemanticWorkflowError",
    "SeparatedSemanticWorkflowPlannerAdapter",
]
