"""Post-run shadow observation for typed Reasoner outputs.

The authoritative workflow only captures immutable input snapshots. Provider
calls happen after the Planner/Perception/Executor path has terminated.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from x_editor.core.invocation_trace import (
    ComponentInvocationTrace,
    validate_component_invocation_trace,
)
from x_editor.core.schemas import StrictModel
from x_editor.reasoner.contracts import (
    PriorTurnSemanticContext,
    ReasonedEditIntent,
    ReasonerAdmissionReport,
    ReasonerCallRequest,
    ReasonerShadowStrategy,
    SemanticReasoningMemo,
    SessionStateContext,
    admit_reasoned_edit_intent,
)
from x_editor.runtime.audio_asset_facts import AudioAssetFacts
from x_editor.runtime.execution_events import (
    ExecutionEventObserver,
    ExecutionEventRecorder,
)


class WorkflowSemanticReasoner(Protocol):
    """Replaceable semantic Reasoner boundary used only by shadow in Phase 1."""

    def reason(self, request: ReasonerCallRequest) -> Any:
        """Return a ReasonedEditIntent, possibly raising."""


class ReasonerShadowInputSnapshot(StrictModel):
    """Immutable Planner-gateway input captured without invoking Reasoner."""

    schema_version: Literal["x-editor.reasoner-shadow-input/v1"] = (
        "x-editor.reasoner-shadow-input/v1"
    )
    request_id: str = Field(min_length=1, max_length=256)
    planner_call_index: int = Field(ge=1)
    phase: Literal["initial_understanding", "evidence_review"]
    user_request: str = Field(min_length=1, max_length=8192)
    source_text: str | None = Field(default=None, max_length=16384)
    source_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    reference_audio_refs: list[str] = Field(default_factory=list, max_length=16)
    audio_asset_facts: list[AudioAssetFacts] = Field(default_factory=list)
    prior_turn_contexts: list[PriorTurnSemanticContext] = Field(
        default_factory=list,
        max_length=32,
    )
    session_state: SessionStateContext | None = None
    reviewed_evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    reviewed_evidence: dict[str, Any] = Field(default_factory=dict)
    evidence_field_sources: dict[str, list[str]] = Field(default_factory=dict)
    evidence_path_sources: dict[str, list[str]] = Field(default_factory=dict)
    planner_input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    planner_source_event_ref: dict[str, Any] | None = None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json"))


ReasonerShadowCallStatus = Literal[
    "completed",
    "output_rejected",
    "call_failed",
    "skipped",
]


class ReasonerShadowCallRecord(StrictModel):
    """One append-only, non-authoritative Reasoner observation."""

    schema_version: Literal["x-editor.reasoner-shadow-call/v1"] = "x-editor.reasoner-shadow-call/v1"
    request_id: str
    planner_call_index: int = Field(ge=1)
    phase: Literal["initial_understanding", "evidence_review"]
    strategy: ReasonerShadowStrategy = "direct_json"
    status: ReasonerShadowCallStatus
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    planner_input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_intent_fingerprint: str | None = None
    prior_memo_fingerprint: str | None = None
    semantic_memo: SemanticReasoningMemo | None = None
    semantic_memo_fingerprint: str | None = None
    intent: ReasonedEditIntent | None = None
    intent_fingerprint: str | None = None
    admission: ReasonerAdmissionReport | None = None
    invocation_trace: ComponentInvocationTrace | None = None
    error_type: str | None = None
    error_message: str | None = None


class ReasonerShadowRunResult(StrictModel):
    """Terminal observation facts kept separate from workflow authority."""

    schema_version: Literal["x-editor.reasoner-shadow-run/v1"] = "x-editor.reasoner-shadow-run/v1"
    records: list[ReasonerShadowCallRecord] = Field(default_factory=list)
    event_log_path: str | None = None
    input_bundle_path: str | None = None
    observation_errors: list[dict[str, Any]] = Field(default_factory=list)


class ReasonerShadowRunner:
    """Replay frozen semantic inputs after the authoritative run terminates."""

    def __init__(
        self,
        *,
        reasoner: WorkflowSemanticReasoner,
        semantic_effect_ontology: list[dict[str, Any]] | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.strategy: ReasonerShadowStrategy = getattr(
            reasoner,
            "strategy",
            "direct_json",
        )
        self.semantic_effect_ontology = copy.deepcopy(semantic_effect_ontology or [])
        self.supported_effect_kinds = {
            str(item.get("effect_kind") or "").strip()
            for item in self.semantic_effect_ontology
            if str(item.get("effect_kind") or "").strip()
        }

    def run(
        self,
        snapshots: list[ReasonerShadowInputSnapshot],
        *,
        workflow_run_id: str,
        output_dir: str | Path,
        observer: ExecutionEventObserver | None = None,
    ) -> ReasonerShadowRunResult:
        event_log_path = Path(output_dir) / "reasoner_shadow_events.jsonl"
        input_bundle_path = Path(output_dir) / "reasoner_shadow_inputs.jsonl"
        _write_input_bundle(
            input_bundle_path,
            snapshots,
            semantic_effect_ontology=self.semantic_effect_ontology,
        )
        recorder = ExecutionEventRecorder(
            run_id=f"{workflow_run_id}:reasoner-shadow",
            event_log_path=event_log_path,
            observer=observer,
        )
        records: list[ReasonerShadowCallRecord] = []
        prior_intent: ReasonedEditIntent | None = None
        prior_memo: SemanticReasoningMemo | None = None
        for snapshot in snapshots:
            prior_product_missing = (
                prior_memo is None
                if self.strategy in {"memo_only", "memo_only_brief", "prose_memo"}
                else prior_intent is None
            )
            if snapshot.phase == "evidence_review" and prior_product_missing:
                missing_product = (
                    "memo"
                    if self.strategy in {"memo_only", "memo_only_brief", "prose_memo"}
                    else "intent"
                )
                record = ReasonerShadowCallRecord(
                    request_id=snapshot.request_id,
                    planner_call_index=snapshot.planner_call_index,
                    phase=snapshot.phase,
                    strategy=self.strategy,
                    status="skipped",
                    input_fingerprint=snapshot.fingerprint,
                    planner_input_fingerprint=snapshot.planner_input_fingerprint,
                    error_type=(
                        "ReasonerPriorMemoUnavailable"
                        if self.strategy in {"memo_only", "memo_only_brief", "prose_memo"}
                        else "ReasonerPriorIntentUnavailable"
                    ),
                    error_message=(
                        "Evidence review was skipped because no valid initial shadow "
                        f"{missing_product} was available."
                    ),
                )
                records.append(record)
                recorder.emit(
                    stage="reasoner",
                    kind="workflow_reasoner_shadow_call_skipped",
                    status="rejected",
                    attempt_id=snapshot.request_id,
                    payload={
                        "record": record.model_dump(mode="json"),
                        "upstream_source_event_ref": copy.deepcopy(
                            snapshot.planner_source_event_ref
                        ),
                    },
                )
                continue
            request = ReasonerCallRequest(
                request_id=snapshot.request_id,
                phase=snapshot.phase,
                user_request=snapshot.user_request,
                source_text=snapshot.source_text,
                audio_asset_facts=[item.model_copy(deep=True) for item in snapshot.audio_asset_facts],
                source_audio_refs=list(snapshot.source_audio_refs),
                reference_audio_refs=list(snapshot.reference_audio_refs),
                prior_turn_contexts=[
                    item.model_copy(deep=True) for item in snapshot.prior_turn_contexts
                ],
                session_state=(
                    snapshot.session_state.model_copy(deep=True)
                    if snapshot.session_state is not None
                    else None
                ),
                reviewed_evidence_ids=list(snapshot.reviewed_evidence_ids),
                reviewed_evidence=copy.deepcopy(snapshot.reviewed_evidence),
                evidence_field_sources=copy.deepcopy(snapshot.evidence_field_sources),
                evidence_path_sources=copy.deepcopy(snapshot.evidence_path_sources),
                prior_intent=(
                    prior_intent.model_copy(deep=True)
                    if snapshot.phase == "evidence_review" and prior_intent is not None
                    else None
                ),
                prior_memo=(
                    prior_memo.model_copy(deep=True)
                    if snapshot.phase == "evidence_review" and prior_memo is not None
                    else None
                ),
                semantic_revision_policy=(
                    "evidence_correctable"
                    if self.strategy
                    in {"memo_then_compile", "memo_only", "memo_only_brief", "prose_memo"}
                    else "freeze_initial"
                ),
                semantic_effect_ontology=copy.deepcopy(self.semantic_effect_ontology),
            )
            recorder.emit(
                stage="reasoner",
                kind="workflow_reasoner_shadow_call_started",
                status="started",
                attempt_id=snapshot.request_id,
                subject_refs=list(snapshot.reviewed_evidence_ids),
                payload={
                    "phase": snapshot.phase,
                    "input_fingerprint": request.fingerprint,
                    "planner_input_fingerprint": (snapshot.planner_input_fingerprint),
                },
            )
            try:
                raw_output = self.reasoner.reason(request)
                if self.strategy in {"memo_only", "memo_only_brief", "prose_memo"}:
                    memo = SemanticReasoningMemo.model_validate(raw_output)
                    trace = _bound_invocation_trace(
                        getattr(self.reasoner, "last_invocation_trace", None),
                        gateway_call_id=snapshot.request_id,
                    )
                    record = ReasonerShadowCallRecord(
                        request_id=snapshot.request_id,
                        planner_call_index=snapshot.planner_call_index,
                        phase=snapshot.phase,
                        strategy=self.strategy,
                        status="completed",
                        input_fingerprint=request.fingerprint,
                        planner_input_fingerprint=snapshot.planner_input_fingerprint,
                        prior_memo_fingerprint=(
                            prior_memo.fingerprint if prior_memo is not None else None
                        ),
                        semantic_memo=memo,
                        semantic_memo_fingerprint=memo.fingerprint,
                        invocation_trace=trace,
                    )
                    records.append(record)
                    recorder.emit(
                        stage="reasoner",
                        kind="workflow_reasoner_shadow_memo_completed",
                        status="completed",
                        attempt_id=snapshot.request_id,
                        payload={
                            "record": record.model_dump(mode="json"),
                            "upstream_source_event_ref": copy.deepcopy(
                                snapshot.planner_source_event_ref
                            ),
                        },
                    )
                    if snapshot.phase == "initial_understanding":
                        prior_memo = memo.model_copy(deep=True)
                    continue
                intent = ReasonedEditIntent.model_validate(raw_output)
            except Exception as exc:  # noqa: BLE001 - shadow failure is diagnostic
                memo = _last_semantic_memo(self.reasoner)
                trace = _bound_invocation_trace(
                    getattr(self.reasoner, "last_invocation_trace", None),
                    gateway_call_id=snapshot.request_id,
                )
                status: ReasonerShadowCallStatus = (
                    "output_rejected"
                    if getattr(exc, "failure_kind", None) == "protocol"
                    else "call_failed"
                )
                record = ReasonerShadowCallRecord(
                    request_id=snapshot.request_id,
                    planner_call_index=snapshot.planner_call_index,
                    phase=snapshot.phase,
                    strategy=self.strategy,
                    status=status,
                    input_fingerprint=request.fingerprint,
                    planner_input_fingerprint=(snapshot.planner_input_fingerprint),
                    prior_intent_fingerprint=(
                        prior_intent.fingerprint if prior_intent is not None else None
                    ),
                    prior_memo_fingerprint=(
                        prior_memo.fingerprint if prior_memo is not None else None
                    ),
                    semantic_memo=memo,
                    semantic_memo_fingerprint=(memo.fingerprint if memo is not None else None),
                    invocation_trace=trace,
                    error_type=type(exc).__name__,
                    error_message=_safe_error_message(exc),
                )
                records.append(record)
                recorder.emit(
                    stage="reasoner",
                    kind=(
                        "workflow_reasoner_shadow_output_rejected"
                        if status == "output_rejected"
                        else "workflow_reasoner_shadow_call_failed"
                    ),
                    status=("rejected" if status == "output_rejected" else "failed"),
                    attempt_id=snapshot.request_id,
                    payload={
                        "record": record.model_dump(mode="json"),
                        "upstream_source_event_ref": copy.deepcopy(
                            snapshot.planner_source_event_ref
                        ),
                    },
                )
                continue
            memo = _last_semantic_memo(self.reasoner)
            trace = _bound_invocation_trace(
                getattr(self.reasoner, "last_invocation_trace", None),
                gateway_call_id=snapshot.request_id,
            )
            admission = admit_reasoned_edit_intent(
                intent,
                request=request,
                supported_effect_kinds=self.supported_effect_kinds,
            )
            record = ReasonerShadowCallRecord(
                request_id=snapshot.request_id,
                planner_call_index=snapshot.planner_call_index,
                phase=snapshot.phase,
                strategy=self.strategy,
                status="completed",
                input_fingerprint=request.fingerprint,
                planner_input_fingerprint=snapshot.planner_input_fingerprint,
                prior_intent_fingerprint=(
                    prior_intent.fingerprint if prior_intent is not None else None
                ),
                prior_memo_fingerprint=(prior_memo.fingerprint if prior_memo is not None else None),
                semantic_memo=memo,
                semantic_memo_fingerprint=(memo.fingerprint if memo is not None else None),
                intent=intent,
                intent_fingerprint=intent.fingerprint,
                admission=admission,
                invocation_trace=trace,
            )
            records.append(record)
            recorder.emit(
                stage="reasoner",
                kind="workflow_reasoner_shadow_call_completed",
                status="completed",
                attempt_id=snapshot.request_id,
                subject_refs=[item.objective_id for item in intent.objectives],
                payload={
                    "record": record.model_dump(mode="json"),
                    "upstream_source_event_ref": copy.deepcopy(snapshot.planner_source_event_ref),
                },
            )
            if snapshot.phase == "initial_understanding":
                prior_intent = intent.model_copy(deep=True)
                prior_memo = memo.model_copy(deep=True) if memo is not None else None
        return ReasonerShadowRunResult(
            records=records,
            event_log_path=recorder.event_log_path,
            input_bundle_path=str(input_bundle_path),
            observation_errors=list(recorder.observation_errors),
        )


def _bound_invocation_trace(
    value: Any,
    *,
    gateway_call_id: str,
) -> ComponentInvocationTrace | None:
    if value is None:
        return None
    try:
        payload = (
            value.model_dump(mode="python")
            if hasattr(value, "model_dump")
            else copy.deepcopy(value)
        )
        payload["gateway_call_id"] = gateway_call_id
        return validate_component_invocation_trace(payload)
    except Exception:  # noqa: BLE001 - invalid trace cannot affect authority
        return None


def _safe_error_message(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text[:1000]


def _last_semantic_memo(reasoner: Any) -> SemanticReasoningMemo | None:
    value = getattr(reasoner, "last_semantic_memo", None)
    if value is None:
        return None
    try:
        return SemanticReasoningMemo.model_validate(value)
    except Exception:  # noqa: BLE001 - malformed diagnostics cannot gain authority
        return None


def _write_input_bundle(
    path: Path,
    snapshots: list[ReasonerShadowInputSnapshot],
    *,
    semantic_effect_ontology: list[dict[str, Any]],
) -> None:
    """Persist benchmark-neutral inputs for paired Reasoner replay experiments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        manifest = {
            "schema_version": "x-editor.reasoner-shadow-replay-manifest/v1",
            "record_type": "manifest",
            "semantic_effect_ontology": copy.deepcopy(semantic_effect_ontology),
        }
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for snapshot in snapshots:
            payload = {
                "schema_version": "x-editor.reasoner-shadow-replay-input/v1",
                "record_type": "snapshot",
                "snapshot": snapshot.model_dump(
                    mode="json",
                    exclude={"planner_source_event_ref"},
                ),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ReasonerShadowCallRecord",
    "ReasonerShadowInputSnapshot",
    "ReasonerShadowRunResult",
    "ReasonerShadowRunner",
    "WorkflowSemanticReasoner",
]
