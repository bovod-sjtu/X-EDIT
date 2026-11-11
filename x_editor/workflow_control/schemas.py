"""Typed contracts for observable and interactive workflow control."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

WorkflowControlMode = Literal["disabled", "audit", "interactive", "enforced"]
ArtifactStoreMode = Literal["disabled", "enforced"]
FinalAssemblerMode = Literal["disabled", "shadow"]
WorkflowStage = Literal[
    "runner",
    "reasoner",
    "planner",
    "perception",
    "executor",
    "verifier",
    "repair",
    "evaluator",
    "finalization",
    "session",
]
WorkflowEventStatus = Literal[
    "started",
    "observed",
    "pending",
    "accepted",
    "rejected",
    "completed",
    "failed",
]
DecisionResolutionStatus = Literal["pending", "accepted", "rejected", "auto_accepted"]
RepairAction = Literal[
    "accept",
    "accept_with_warning",
    "retry_same_call",
    "repair_step_arguments",
    "reacquire_evidence",
    "rerun_step",
    "replan_suffix",
    "replan_global",
    "fail_structured",
]
RepairScope = Literal[
    "argument", "call", "artifact", "span", "step", "suffix", "plan", "objective", "run"
]
RepairSignalSource = Literal[
    "tool_contract", "runtime_verification", "evaluator", "step_local"
]
RepairFailureClass = Literal[
    "argument_contract",
    "transient_call",
    "evidence_missing",
    "alignment_or_span",
    "artifact_quality",
    "dependency_or_route",
    "objective_or_capability",
    "unknown",
]
RepairOutcome = Literal[
    "accepted", "rejected", "failed", "not_run", "unknown"
]
RepairComparisonRelation = Literal[
    "exact", "controller_narrower", "controller_broader", "different"
]
ArgumentRepairProducerKind = Literal["deterministic", "executor_llm"]
ArgumentRepairResolution = Literal["resolved", "ambiguous", "unresolved"]
ArgumentBindingKind = Literal["mechanical", "semantic", "unknown"]
ArgumentRepairAdmissionSeverity = Literal["info", "warning", "blocking"]
ArtifactDisposition = Literal["candidate", "accepted", "rejected"]
ArtifactIntegrityStatus = Literal["passed", "failed", "not_checked"]
ArtifactVerificationStatus = Literal[
    "passed",
    "passed_with_warnings",
    "failed",
    "inconclusive",
    "skipped",
    "unavailable",
    "not_run",
]
ArtifactVerificationOutcome = Literal[
    "passed", "failed", "inconclusive", "skipped", "unavailable"
]
ArtifactVerificationScope = Literal[
    "artifact_integrity", "operation_invariant", "step_diagnostic", "objective"
]
ArtifactDeliveryProvenance = Literal["tool", "composite", "local"]
ArtifactDeliveryOperationEvidence = Literal["handler_result", "executor_dispatch"]
ObjectiveEvaluationStatus = Literal[
    "pending", "passed", "failed", "unavailable"
]
ArtifactAuditAttemptOutcome = Literal["selected", "rejected", "provisional"]
ArtifactFinalizationStatus = Literal[
    "would_finalize", "blocked", "insufficient_evidence"
]
ArtifactStoreCommitStatus = Literal["committed", "blocked"]
FinalDeliveryManifestStatus = Literal["ready", "incomplete", "not_applicable"]
FinalDeliveryArtifactRole = Literal[
    "authoritative_output", "ordered_component", "non_authoritative_output"
]
FinalDeliveryArtifactDisposition = Literal[
    "accepted", "candidate", "rejected", "external_source"
]
StrictDeliveryDecision = Literal[
    "deliver",
    "block_delivery",
    "preserve_upstream_failure",
    "reject_contract",
]
StrictDeliveryFailureStage = Literal[
    "none", "runner", "artifact_store", "final_assembler", "configuration"
]
StrictDeliverySessionAction = Literal["advance_current", "preserve_current"]
StrictDeliveryOutputVisibility = Literal[
    "official_and_evidence", "evidence_only", "none"
]
WorkflowFailureRegistryStatus = Literal["clear", "unresolved"]
SuffixResumeCheckpointStatus = Literal[
    "eligible", "full_replay_required", "not_applicable"
]
SuffixResumeAssessmentStatus = Literal["admissible", "full_replay_required"]
SuffixResumeExecutionStatus = Literal["admitted", "full_replay_required"]
ArtifactAcceptanceTier = Literal[
    "operation_principle", "delivery_contract", "uncovered"
]
ArtifactShadowFinalRelation = Literal[
    "match", "differs", "legacy_missing", "no_shadow_eligible"
]
AdmissionSeverity = Literal["info", "warning", "blocking"]
AdmissionOwner = Literal["planner", "executor", "caller", "runtime", "unknown"]
AdmissionRepairType = Literal["deterministic", "planner", "evidence", "caller"]


def new_workflow_id(prefix: str) -> str:
    """Return an opaque, non-prompt workflow identifier."""

    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DecisionOption(StrictModel):
    """One explicit option presented at a consequential checkpoint."""

    option_id: str
    plain_label: str
    consequences: list[str] = Field(default_factory=list)
    changes_execution: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionCheckpoint(StrictModel):
    """A human-readable decision request with machine-auditable details."""

    checkpoint_id: str = Field(default_factory=lambda: new_workflow_id("checkpoint"))
    stage: WorkflowStage
    plain_summary: str
    question: str
    why_now: str
    options: list[DecisionOption] = Field(min_length=1)
    recommended_option_id: str
    requires_confirmation: bool = True
    blocking: bool = True
    subject_refs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_options(self) -> DecisionCheckpoint:
        option_ids = [option.option_id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("decision option ids must be unique")
        if self.recommended_option_id not in option_ids:
            raise ValueError(
                "recommended_option_id must identify one of the checkpoint options"
            )
        if self.requires_confirmation and len(option_ids) < 2:
            raise ValueError(
                "a confirmation checkpoint must provide at least two options"
            )
        return self


class DecisionResolution(StrictModel):
    """Recorded disposition of one decision checkpoint."""

    resolution_id: str = Field(default_factory=lambda: new_workflow_id("resolution"))
    checkpoint_id: str
    status: DecisionResolutionStatus
    selected_option_id: str | None = None
    resolved_by: str
    rationale: str | None = None
    resolved_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_selection(self) -> DecisionResolution:
        if self.status in {"accepted", "auto_accepted"} and not self.selected_option_id:
            raise ValueError("accepted resolutions require selected_option_id")
        if (
            self.status in {"pending", "rejected"}
            and self.selected_option_id is not None
        ):
            raise ValueError(f"{self.status} resolutions cannot select an option")
        return self


class RepairDecision(StrictModel):
    """One normalized repair action, observed or selected by a controller."""

    decision_id: str = Field(default_factory=lambda: new_workflow_id("repair"))
    stage: str
    action: RepairAction
    scope: RepairScope
    plain_summary: str
    subject_refs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reuse_artifact_refs: list[str] = Field(default_factory=list)
    budget_delta: dict[str, float] = Field(default_factory=dict)
    shadow_only: bool = True
    legacy_action_observed: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RepairSignal(StrictModel):
    """Normalized evidence that asks the shadow controller for one repair action."""

    schema_version: str = "x-editor.repair-signal/v1"
    signal_id: str = Field(default_factory=lambda: new_workflow_id("repair_signal"))
    source: RepairSignalSource
    failure_class: RepairFailureClass
    plain_summary: str
    subject_refs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    retryable: bool = True
    local_repair_available: bool = False
    global_semantic_change_required: bool = False
    candidate_actions: list[RepairAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RepairComparison(StrictModel):
    """Comparison between a shadow recommendation and the legacy action taken."""

    schema_version: str = "x-editor.repair-comparison/v1"
    comparison_id: str = Field(default_factory=lambda: new_workflow_id("repair_comparison"))
    signal_id: str
    decision_id: str
    legacy_action: RepairAction
    legacy_scope: RepairScope
    relation: RepairComparisonRelation
    outcome: RepairOutcome
    budget_delta: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ArgumentRepairFieldChange(StrictModel):
    """One proposed field mutation with its semantic or mechanical evidence."""

    field_path: str
    before: Any = None
    after: Any
    binding_kind: ArgumentBindingKind = "unknown"
    semantic_value: Any | None = None
    evidence: list[dict[str, str]] = Field(default_factory=list)
    alternatives: list[Any] = Field(default_factory=list)
    rationale: str = ""


class ArgumentRepairProposal(StrictModel):
    """A bounded patch proposal for one failed step; it is not executable by itself."""

    schema_version: str = "x-editor.argument-repair-proposal/v1"
    proposal_id: str = Field(default_factory=lambda: new_workflow_id("argument_repair"))
    signal_id: str
    step_id: str
    operation_before: str
    operation_after: str
    producer_kind: ArgumentRepairProducerKind
    producer: str
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    resolution: ArgumentRepairResolution
    allowed_field_paths: list[str] = Field(default_factory=list)
    changes: list[ArgumentRepairFieldChange] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ArgumentRepairAdmissionFinding(StrictModel):
    """One deterministic reason to accept or reject an argument patch."""

    code: str
    severity: ArgumentRepairAdmissionSeverity
    plain_summary: str
    field_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArgumentRepairAdmissionResult(StrictModel):
    """Deterministic verdict over a proposal before it may mutate a step."""

    schema_version: str = "x-editor.argument-repair-admission/v1"
    admission_id: str = Field(default_factory=lambda: new_workflow_id("argument_admission"))
    proposal_id: str
    admitted: bool
    findings: list[ArgumentRepairAdmissionFinding] = Field(default_factory=list)
    checked_field_paths: list[str] = Field(default_factory=list)
    behavior_changed: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactIntegrityCheck(StrictModel):
    """One deterministic filesystem or audio-container check."""

    check_id: str
    status: ArtifactIntegrityStatus
    plain_summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactAcceptancePolicy(StrictModel):
    """Framework checks required before one operation's artifact is accepted."""

    schema_version: str = "x-editor.artifact-acceptance-policy/v1"
    policy_id: str
    operation: str
    required_check_ids: list[str] = Field(min_length=1)
    plain_summary: str

    @model_validator(mode="after")
    def validate_required_checks(self) -> ArtifactAcceptancePolicy:
        if len(self.required_check_ids) != len(set(self.required_check_ids)):
            raise ValueError("artifact acceptance check ids must be unique")
        return self


class ArtifactDeliveryInvocation(StrictModel):
    """Compact provenance for one tool call observed during a delivered step."""

    tool_id: str
    success: bool
    output_audio_path: str | None = None
    error_message: str | None = None


class AudioTemporalSpan(StrictModel):
    """One positive-duration interval in a named audio timebase."""

    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> AudioTemporalSpan:
        if self.end_sec <= self.start_sec:
            raise ValueError("audio temporal span end_sec must exceed start_sec")
        return self


class AudioTemporalSpanMapping(StrictModel):
    """One explicit source-to-output interval mapping authored by Executor."""

    source_span: AudioTemporalSpan
    output_span: AudioTemporalSpan


class AudioTemporalTransform(StrictModel):
    """Executor-authored temporal effect of one successful audio step."""

    schema_version: Literal["x-editor.audio-temporal-transform/v1"] = (
        "x-editor.audio-temporal-transform/v1"
    )
    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    status: Literal["complete", "partial", "unavailable"]
    edit_scope: Literal["local", "global", "unknown"]
    input_audio_paths: list[str] = Field(default_factory=list)
    input_audio_refs: list[str] = Field(default_factory=list)
    output_audio_path: str = Field(min_length=1)
    source_regions: list[AudioTemporalSpan] = Field(default_factory=list)
    output_regions: list[AudioTemporalSpan] = Field(default_factory=list)
    timeline_mappings: list[AudioTemporalSpanMapping] = Field(default_factory=list)
    identity_outside_regions: bool | None = None
    timeline_duration_preserved: bool | None = None
    derivation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_transform(self) -> AudioTemporalTransform:
        if self.status == "complete" and not self.input_audio_paths:
            raise ValueError("complete temporal transform requires an input audio path")
        if self.edit_scope == "local" and self.status == "complete":
            if not self.source_regions or not self.timeline_mappings:
                raise ValueError(
                    "complete local transform requires source regions and mappings"
                )
        if self.edit_scope != "local" and self.identity_outside_regions is not None:
            raise ValueError(
                "identity_outside_regions applies only to local temporal transforms"
            )
        return self


class ArtifactDeliveryContract(StrictModel):
    """Executor-authored proof that one handler delivered one audio artifact."""

    schema_version: str = "x-editor.artifact-delivery/v1"
    step_id: str
    requested_operation: str
    delivered_operation: str
    operation_evidence: ArtifactDeliveryOperationEvidence
    handler: str
    handler_completed: bool
    provenance: ArtifactDeliveryProvenance
    input_audio_paths: list[str] = Field(default_factory=list)
    output_audio_path: str
    output_size_bytes: int = Field(ge=0)
    output_sha256: str
    tool_invocations: list[ArtifactDeliveryInvocation] = Field(default_factory=list)


class ArtifactVerificationEvidence(StrictModel):
    """One step verifier report projected onto a concrete audio artifact."""

    component: str
    outcome: ArtifactVerificationOutcome
    scope: ArtifactVerificationScope
    acceptance_grade: bool = False
    check_ids: list[str] = Field(default_factory=list)
    check_outcomes: dict[str, ArtifactVerificationOutcome] = Field(
        default_factory=dict
    )
    subject_paths: list[str] = Field(default_factory=list)
    subject_path_matched: bool = False
    confidence: str | None = None
    severity: str | None = None


class ArtifactAuditRecord(StrictModel):
    """Audit-only disposition of one audio artifact produced by one attempt."""

    schema_version: str = "x-editor.artifact-audit-record/v2"
    artifact_ref_id: str
    attempt_id: str
    path: str
    role: str
    kind: str
    producer_step_id: str
    producer_operation: str | None = None
    producer_step_status: str | None = None
    primary_output: bool = False
    parent_paths: list[str] = Field(default_factory=list)
    parent_artifact_ref_ids: list[str] = Field(default_factory=list)
    provenance: str | None = None
    integrity_checks: list[ArtifactIntegrityCheck] = Field(default_factory=list)
    integrity_passed: bool
    verification_status: ArtifactVerificationStatus
    verification_refs: list[str] = Field(default_factory=list)
    verification_evidence: list[ArtifactVerificationEvidence] = Field(
        default_factory=list
    )
    acceptance_policy_id: str | None = None
    required_check_ids: list[str] = Field(default_factory=list)
    satisfied_check_ids: list[str] = Field(default_factory=list)
    missing_check_ids: list[str] = Field(default_factory=list)
    objective_evaluation_status: ObjectiveEvaluationStatus = "pending"
    disposition: ArtifactDisposition
    disposition_reasons: list[str] = Field(default_factory=list)
    legacy_selected: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactFinalizationFinding(StrictModel):
    """One reason an observed legacy final path would or would not finalize."""

    code: str
    severity: AdmissionSeverity
    plain_summary: str
    artifact_ref_id: str | None = None
    step_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactFinalizationAudit(StrictModel):
    """Pure audit projection over legacy artifact selection and verification."""

    schema_version: str = "x-editor.artifact-finalization-audit/v2"
    audit_id: str = Field(default_factory=lambda: new_workflow_id("artifact_audit"))
    attempt_id: str
    attempt_outcome: ArtifactAuditAttemptOutcome
    status: ArtifactFinalizationStatus
    behavior_changed: bool = False
    legacy_final_path: str | None = None
    legacy_final_artifact_ref_id: str | None = None
    legacy_final_disposition: ArtifactDisposition | None = None
    artifacts: list[ArtifactAuditRecord] = Field(default_factory=list)
    accepted_artifact_refs: list[str] = Field(default_factory=list)
    candidate_artifact_refs: list[str] = Field(default_factory=list)
    rejected_artifact_refs: list[str] = Field(default_factory=list)
    required_step_ids: list[str] = Field(default_factory=list)
    required_artifact_refs: list[str] = Field(default_factory=list)
    unresolved_required_steps: list[str] = Field(default_factory=list)
    findings: list[ArtifactFinalizationFinding] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactStoreCommit(StrictModel):
    """Atomic authoritative decision over one selected finalization audit."""

    schema_version: str = "x-editor.artifact-store-commit/v1"
    commit_id: str = Field(default_factory=lambda: new_workflow_id("artifact_commit"))
    audit_id: str
    attempt_id: str
    status: ArtifactStoreCommitStatus
    legacy_final_path: str | None = None
    final_artifact_ref_id: str | None = None
    final_output_path: str | None = None
    accepted_artifact_refs: list[str] = Field(default_factory=list)
    candidate_artifact_refs: list[str] = Field(default_factory=list)
    rejected_artifact_refs: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)
    plain_summary: str
    authority_enforced: bool = True
    behavior_changed: bool
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_atomic_commit(self) -> "ArtifactStoreCommit":
        if self.status == "committed":
            if not self.final_artifact_ref_id or not self.final_output_path:
                raise ValueError("committed artifact store decision requires a final artifact")
            if self.final_artifact_ref_id not in self.accepted_artifact_refs:
                raise ValueError("artifact store can commit only an accepted artifact")
            if self.failure_codes:
                raise ValueError("committed artifact store decision cannot contain failures")
        elif self.final_artifact_ref_id is not None or self.final_output_path is not None:
            raise ValueError("blocked artifact store decision cannot expose a final artifact")
        return self


class FinalDeliveryArtifact(StrictModel):
    """One artifact's non-authoritative placement in a final delivery manifest."""

    schema_version: str = "x-editor.final-delivery-artifact/v1"
    artifact_ref_id: str | None = None
    path: str
    delivery_role: FinalDeliveryArtifactRole
    disposition: FinalDeliveryArtifactDisposition
    producer_step_id: str | None = None
    producer_operation: str | None = None
    order_index: int | None = Field(default=None, ge=0)
    format: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    content_sha256: str | None = None
    audio_properties: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_role_order(self) -> "FinalDeliveryArtifact":
        if self.delivery_role == "ordered_component":
            if self.order_index is None:
                raise ValueError("ordered delivery component requires order_index")
        elif self.order_index is not None:
            raise ValueError("only ordered delivery components may carry order_index")
        if self.delivery_role == "authoritative_output" and not self.artifact_ref_id:
            raise ValueError("authoritative delivery output requires an artifact ref")
        return self


class FinalAssemblerFinding(StrictModel):
    """One deterministic observation made while projecting final delivery."""

    code: str
    severity: AdmissionSeverity
    plain_summary: str
    artifact_ref_id: str | None = None
    path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class FinalDeliveryManifest(StrictModel):
    """Shadow-only projection of an ArtifactStore decision into delivery structure."""

    schema_version: str = "x-editor.final-delivery-manifest/v1"
    manifest_id: str = Field(default_factory=lambda: new_workflow_id("delivery_manifest"))
    commit_id: str
    audit_id: str
    attempt_id: str
    commit_status: ArtifactStoreCommitStatus
    status: FinalDeliveryManifestStatus
    authority_source: str = "artifact_store_commit"
    committed_final_artifact_ref_id: str | None = None
    committed_final_output_path: str | None = None
    authoritative_output: FinalDeliveryArtifact | None = None
    ordered_components: list[FinalDeliveryArtifact] = Field(default_factory=list)
    non_authoritative_outputs: list[FinalDeliveryArtifact] = Field(default_factory=list)
    component_order_source: str = "none"
    requested_output_path: str | None = None
    path_matches_request: bool | None = None
    requested_output_format: str | None = None
    actual_output_format: str | None = None
    format_matches_request: bool | None = None
    required_artifact_refs: list[str] = Field(default_factory=list)
    missing_required_artifact_refs: list[str] = Field(default_factory=list)
    complete_required_artifacts: bool = False
    format_contract_complete: bool = False
    lineage_complete: bool = False
    findings: list[FinalAssemblerFinding] = Field(default_factory=list)
    shadow_only: bool = True
    behavior_changed: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_shadow_manifest(self) -> "FinalDeliveryManifest":
        if not self.shadow_only or self.behavior_changed:
            raise ValueError("FinalAssembler shadow manifest cannot change behavior")
        if self.status == "ready":
            output = self.authoritative_output
            if output is None:
                raise ValueError("ready delivery manifest requires one authoritative output")
            if output.delivery_role != "authoritative_output":
                raise ValueError("ready delivery output must have authoritative_output role")
            if output.artifact_ref_id != self.committed_final_artifact_ref_id:
                raise ValueError("delivery authority must match ArtifactStore artifact ref")
            if output.path != self.committed_final_output_path:
                raise ValueError("delivery authority must match ArtifactStore output path")
            if (
                not self.complete_required_artifacts
                or not self.format_contract_complete
                or not self.lineage_complete
            ):
                raise ValueError(
                    "ready delivery manifest requires complete artifact, format, and lineage contracts"
                )
        elif self.authoritative_output is not None:
            raise ValueError("non-ready delivery manifest cannot project delivery authority")

        order_indexes = [item.order_index for item in self.ordered_components]
        if len(set(order_indexes)) != len(order_indexes) or order_indexes != sorted(
            order_indexes
        ):
            raise ValueError("ordered delivery components must use unique ascending indexes")
        if self.status == "ready" and order_indexes != list(
            range(len(self.ordered_components))
        ):
            raise ValueError(
                "ready ordered delivery components must use contiguous zero-based indexes"
            )
        if any(
            item.delivery_role != "ordered_component"
            for item in self.ordered_components
        ):
            raise ValueError("ordered_components contains an invalid delivery role")
        if any(
            item.delivery_role != "non_authoritative_output"
            for item in self.non_authoritative_outputs
        ):
            raise ValueError("non_authoritative_outputs contains an invalid delivery role")
        return self


class StrictDeliveryContractProjection(StrictModel):
    """Offline proposal for a future strict final-delivery state transition."""

    schema_version: str = "x-editor.strict-delivery-contract-projection/v1"
    projection_id: str = Field(
        default_factory=lambda: new_workflow_id("strict_delivery_projection")
    )
    source_run_status: Literal["completed", "failed"]
    proposed_run_status: Literal["completed", "failed"]
    decision: StrictDeliveryDecision
    failure_stage: StrictDeliveryFailureStage
    artifact_store_commit_id: str
    artifact_store_status: ArtifactStoreCommitStatus
    manifest_id: str | None = None
    manifest_status: FinalDeliveryManifestStatus | None = None
    official_final_output_path: str | None = None
    evidence_committed_final_artifact_ref_id: str | None = None
    evidence_committed_final_output_path: str | None = None
    session_current_action: StrictDeliverySessionAction
    output_visibility: StrictDeliveryOutputVisibility
    reason_codes: list[str] = Field(default_factory=list)
    plain_summary: str
    artifact_store_fact_preserved: Literal[True] = True
    simulation_only: Literal[True] = True
    behavior_changed: Literal[False] = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_strict_delivery_projection(
        self,
    ) -> "StrictDeliveryContractProjection":
        if self.artifact_store_status == "committed":
            if (
                not self.evidence_committed_final_artifact_ref_id
                or not self.evidence_committed_final_output_path
            ):
                raise ValueError(
                    "a committed ArtifactStore fact requires its artifact ref and path"
                )
        elif (
            self.evidence_committed_final_artifact_ref_id is not None
            or self.evidence_committed_final_output_path is not None
        ):
            raise ValueError(
                "a blocked ArtifactStore fact cannot expose a committed artifact"
            )

        if self.decision == "deliver":
            if self.source_run_status != "completed":
                raise ValueError("strict delivery cannot override an upstream failed run")
            if self.proposed_run_status != "completed":
                raise ValueError("delivered output requires completed proposed run status")
            if self.artifact_store_status != "committed":
                raise ValueError("strict delivery requires a committed ArtifactStore fact")
            if self.manifest_status != "ready" or not self.manifest_id:
                raise ValueError("strict delivery requires a ready manifest")
            if (
                not self.official_final_output_path
                or self.official_final_output_path
                != self.evidence_committed_final_output_path
            ):
                raise ValueError(
                    "official delivery path must equal the committed evidence path"
                )
            if self.session_current_action != "advance_current":
                raise ValueError("delivered output must advance session current")
            if self.output_visibility != "official_and_evidence":
                raise ValueError(
                    "delivered output must be visible as both official result and evidence"
                )
            if self.failure_stage != "none":
                raise ValueError("delivered output cannot declare a failure stage")
        else:
            if self.proposed_run_status != "failed":
                raise ValueError("non-delivery decisions require failed proposed status")
            if self.official_final_output_path is not None:
                raise ValueError("non-delivery decisions cannot expose an official path")
            if self.session_current_action != "preserve_current":
                raise ValueError("non-delivery decisions must preserve session current")
            expected_visibility = (
                "evidence_only"
                if self.evidence_committed_final_output_path is not None
                else "none"
            )
            if self.output_visibility != expected_visibility:
                raise ValueError(
                    "non-delivery visibility must reflect whether a committed evidence path exists"
                )
            if self.failure_stage == "none":
                raise ValueError("non-delivery decisions require a failure stage")
            if not self.reason_codes:
                raise ValueError("non-delivery decisions require reason codes")
        return self


class VerifierShadowRun(StrictModel):
    """One actual runtime verifier report retained for shadow statistics."""

    schema_version: str = "x-editor.verifier-shadow-run/v1"
    component: str
    step_id: str | None = None
    operation: str | None = None
    outcome: ArtifactVerificationOutcome
    scope: ArtifactVerificationScope
    acceptance_grade: bool = False
    check_ids: list[str] = Field(default_factory=list)
    check_outcomes: dict[str, ArtifactVerificationOutcome] = Field(
        default_factory=dict
    )
    subject_paths: list[str] = Field(default_factory=list)
    delivery_provenance: ArtifactDeliveryProvenance | None = None


class ArtifactShadowRecord(StrictModel):
    """Non-authoritative projection of verifier evidence onto one artifact."""

    schema_version: str = "x-editor.artifact-shadow-record/v1"
    artifact_ref_id: str
    step_id: str
    operation: str | None = None
    acceptance_tier: ArtifactAcceptanceTier
    acceptance_policy_id: str | None = None
    disposition: ArtifactDisposition
    verification_status: ArtifactVerificationStatus
    candidate_reasons: list[str] = Field(default_factory=list)
    missing_check_ids: list[str] = Field(default_factory=list)
    failed_required_check_ids: list[str] = Field(default_factory=list)
    legacy_selected: bool = False
    primary_output: bool = False
    behavior_changed: bool = False


class ArtifactShadowObservation(StrictModel):
    """Persistable shadow summary that cannot select or reject runtime artifacts."""

    schema_version: str = "x-editor.artifact-shadow-observation/v1"
    observation_id: str = Field(
        default_factory=lambda: new_workflow_id("artifact_shadow")
    )
    audit_id: str
    attempt_id: str
    audit_status: ArtifactFinalizationStatus
    verifier_runs: list[VerifierShadowRun] = Field(default_factory=list)
    artifacts: list[ArtifactShadowRecord] = Field(default_factory=list)
    legacy_final_artifact_ref_id: str | None = None
    shadow_preferred_final_artifact_ref_id: str | None = None
    final_relation: ArtifactShadowFinalRelation
    verifier_outcome_counts: dict[str, int] = Field(default_factory=dict)
    artifact_disposition_counts: dict[str, int] = Field(default_factory=dict)
    artifact_verification_status_counts: dict[str, int] = Field(default_factory=dict)
    acceptance_tier_counts: dict[str, int] = Field(default_factory=dict)
    candidate_reason_counts: dict[str, int] = Field(default_factory=dict)
    delivery_provenance_counts: dict[str, int] = Field(default_factory=dict)
    behavior_changed: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactShadowAggregate(StrictModel):
    """Cross-attempt summary rebuilt from persisted shadow observations."""

    schema_version: str = "x-editor.artifact-shadow-aggregate/v1"
    observation_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    operations: list[str] = Field(default_factory=list)
    verifier_outcome_counts: dict[str, int] = Field(default_factory=dict)
    operation_verifier_outcome_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    artifact_disposition_counts: dict[str, int] = Field(default_factory=dict)
    operation_artifact_disposition_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    candidate_reason_counts: dict[str, int] = Field(default_factory=dict)
    final_relation_counts: dict[str, int] = Field(default_factory=dict)
    acceptance_tier_counts: dict[str, int] = Field(default_factory=dict)
    delivery_provenance_counts: dict[str, int] = Field(default_factory=dict)
    behavior_changed: bool = False


class PlanAdmissionFinding(StrictModel):
    """One structural or objective-level issue found before execution."""

    code: str
    severity: AdmissionSeverity
    plain_summary: str
    owner: AdmissionOwner = "unknown"
    step_id: str | None = None
    operation: str | None = None
    field: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PlanAdmissionRepairCandidate(StrictModel):
    """A proposed repair that audit mode records but never applies."""

    repair_id: str = Field(default_factory=lambda: new_workflow_id("admission_repair"))
    repair_type: AdmissionRepairType
    code: str
    plain_summary: str
    step_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ObjectiveCoverageAudit(StrictModel):
    """Admission-time validation of one planner-declared objective mapping."""

    objective_id: str
    required: bool = True
    declared_status: str | None = None
    covered_by_steps: list[str] = Field(default_factory=list)
    valid_step_ids: list[str] = Field(default_factory=list)
    unknown_step_ids: list[str] = Field(default_factory=list)
    executable_coverage: bool = False


class PlanAdmissionResult(StrictModel):
    """Audit-only verdict over the Planner's unmodified output."""

    schema_version: str = "x-editor.plan-admission/v1"
    admission_id: str = Field(default_factory=lambda: new_workflow_id("admission"))
    mode: Literal["audit"] = "audit"
    plan_fingerprint: str
    original_plan_valid: bool
    would_admit: bool
    behavior_changed: bool = False
    findings: list[PlanAdmissionFinding] = Field(default_factory=list)
    objective_coverage: list[ObjectiveCoverageAudit] = Field(default_factory=list)
    deterministic_repairs: list[PlanAdmissionRepairCandidate] = Field(
        default_factory=list
    )
    requested_repairs: list[PlanAdmissionRepairCandidate] = Field(default_factory=list)
    finding_counts: dict[str, int] = Field(default_factory=dict)
    checked_step_count: int = 0
    available_operation_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class WorkflowFailureOccurrence(StrictModel):
    """One failure fact projected from an append-only workflow event."""

    schema_version: str = "x-editor.workflow-failure-occurrence/v1"
    event_id: str
    sequence: int = Field(ge=1)
    stage: WorkflowStage
    event_kind: str
    status: WorkflowEventStatus
    attempt_id: str | None = None
    failed_step_ids: list[str] = Field(default_factory=list)
    failed_artifact_refs: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)
    summary: str


class WorkflowRepairHistoryEntry(StrictModel):
    """One repair action that was actually selected or completed."""

    schema_version: str = "x-editor.workflow-repair-history-entry/v1"
    event_id: str
    sequence: int = Field(ge=1)
    attempt_id: str | None = None
    event_kind: str
    repair_stage: str | None = None
    action: str
    scope: str | None = None
    status: WorkflowEventStatus
    outcome: str | None = None
    accepted: bool | None = None
    subject_refs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowFailureRegistry(StrictModel):
    """Per-run failure and repair history rebuilt from workflow events."""

    schema_version: str = "x-editor.workflow-failure-registry/v1"
    registry_id: str = Field(default_factory=lambda: new_workflow_id("failure_registry"))
    run_id: str
    status: WorkflowFailureRegistryStatus
    run_status: Literal["completed", "failed"]
    terminal_failure_stage: WorkflowStage | None = None
    terminal_failed_step_ids: list[str] = Field(default_factory=list)
    terminal_failed_artifact_refs: list[str] = Field(default_factory=list)
    unresolved_failure_codes: list[str] = Field(default_factory=list)
    legacy_final_output_path: str | None = None
    authoritative_final_output_path: str | None = None
    failure_history: list[WorkflowFailureOccurrence] = Field(default_factory=list)
    repair_history: list[WorkflowRepairHistoryEntry] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    projected_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "WorkflowFailureRegistry":
        if self.status == "unresolved":
            if self.run_status != "failed":
                raise ValueError("unresolved failure registry requires failed run_status")
            if self.authoritative_final_output_path is not None:
                raise ValueError(
                    "unresolved failure registry cannot expose an authoritative final path"
                )
        elif self.run_status != "completed":
            raise ValueError("clear failure registry requires completed run_status")
        return self


class SuffixResumeFinding(StrictModel):
    """One deterministic reason supporting or blocking prefix reuse."""

    code: str
    severity: AdmissionSeverity
    plain_summary: str
    step_id: str | None = None
    artifact_ref_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReusableArtifactCheckpoint(StrictModel):
    """One accepted upstream artifact frozen for later identity revalidation."""

    artifact_ref_id: str
    producer_step_id: str
    path: str
    role: str = "audio"
    kind: str = "step_output"
    primary_output: bool = False
    result_key: str | None = None
    verification_status: ArtifactVerificationStatus
    acceptance_policy_id: str | None = None
    required_check_ids: list[str] = Field(default_factory=list)
    satisfied_check_ids: list[str] = Field(default_factory=list)
    verification_refs: list[str] = Field(default_factory=list)
    parent_paths: list[str] = Field(default_factory=list)
    parent_artifact_ref_ids: list[str] = Field(default_factory=list)
    size_bytes: int | None = Field(default=None, ge=0)
    content_sha256: str | None = None
    original_tool_invocation_count: int = Field(default=0, ge=0)
    producer_result_sha256: str | None = None


class SourceArtifactCheckpoint(StrictModel):
    """Content identity for one external audio source consumed by reusable work."""

    source_ref: str
    path: str
    consumer_artifact_ref_ids: list[str] = Field(default_factory=list)
    size_bytes: int | None = Field(default=None, ge=0)
    content_sha256: str | None = None


class SuffixResumeCheckpoint(StrictModel):
    """Audit-only boundary between accepted upstream work and a failed suffix."""

    schema_version: str = "x-editor.suffix-resume-checkpoint/v2"
    checkpoint_id: str = Field(default_factory=lambda: new_workflow_id("resume_checkpoint"))
    mode: Literal["audit"] = "audit"
    attempt_id: str
    status: SuffixResumeCheckpointStatus
    failed_step_ids: list[str] = Field(default_factory=list)
    restart_step_ids: list[str] = Field(default_factory=list)
    rerun_step_ids: list[str] = Field(default_factory=list)
    reusable_step_ids: list[str] = Field(default_factory=list)
    reusable_artifacts: list[ReusableArtifactCheckpoint] = Field(default_factory=list)
    source_artifacts: list[SourceArtifactCheckpoint] = Field(default_factory=list)
    invalidation_guard_codes: list[str] = Field(default_factory=list)
    plan_fingerprint: str
    source_contract_fingerprint: str
    step_fingerprints: dict[str, str] = Field(default_factory=dict)
    step_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    findings: list[SuffixResumeFinding] = Field(default_factory=list)
    legacy_action_observed: str | None = None
    behavior_changed: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_resume_boundary(self) -> "SuffixResumeCheckpoint":
        artifact_steps = [item.producer_step_id for item in self.reusable_artifacts]
        if sorted(set(artifact_steps)) != sorted(self.reusable_step_ids):
            raise ValueError("reusable_step_ids must match reusable artifact producers")
        if self.status == "eligible":
            if not self.restart_step_ids or not self.reusable_artifacts:
                raise ValueError("eligible suffix resume requires restart steps and artifacts")
        if self.status == "not_applicable" and (
            self.failed_step_ids or self.restart_step_ids or self.rerun_step_ids
        ):
            raise ValueError("not-applicable suffix resume cannot contain failed work")
        return self


class SuffixResumeCandidateAssessment(StrictModel):
    """Audit-only assessment of whether a candidate repair can retain a prefix."""

    schema_version: str = "x-editor.suffix-resume-candidate-assessment/v1"
    assessment_id: str = Field(default_factory=lambda: new_workflow_id("resume_assessment"))
    checkpoint_id: str
    status: SuffixResumeAssessmentStatus
    candidate_plan_fingerprint: str
    retained_reusable_artifact_refs: list[str] = Field(default_factory=list)
    invalidated_step_ids: list[str] = Field(default_factory=list)
    rerun_step_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    findings: list[SuffixResumeFinding] = Field(default_factory=list)
    behavior_changed: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_candidate_assessment(self) -> "SuffixResumeCandidateAssessment":
        if self.status == "admissible" and not self.retained_reusable_artifact_refs:
            raise ValueError("admissible suffix resume must retain an accepted artifact")
        if self.status == "full_replay_required" and self.retained_reusable_artifact_refs:
            raise ValueError("full replay cannot retain reusable artifact refs")
        return self


class SuffixResumeExecutionRequest(StrictModel):
    """Explicit caller request to reuse a prior accepted prefix once."""

    schema_version: str = "x-editor.suffix-resume-execution-request/v1"
    checkpoint: SuffixResumeCheckpoint
    original_plan: dict[str, Any]


class SuffixResumeReuseReceipt(StrictModel):
    """Fresh deterministic authority for one step skipped during this run."""

    schema_version: str = "x-editor.suffix-resume-reuse-receipt/v1"
    receipt_id: str = Field(default_factory=lambda: new_workflow_id("reuse_receipt"))
    checkpoint_id: str
    artifact_ref_id: str
    producer_step_id: str
    operation: str
    path: str
    result_key: Literal["output_audio_path"] = "output_audio_path"
    size_bytes: int = Field(ge=0)
    content_sha256: str
    acceptance_policy_id: str
    required_check_ids: list[str] = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    parent_paths: list[str] = Field(default_factory=list)
    skipped_tool_call_count: int = Field(default=0, ge=0)
    retained_output: dict[str, Any] = Field(default_factory=dict)
    verification_report: dict[str, Any]
    behavior_changed: Literal[True] = True
    created_at: datetime = Field(default_factory=utc_now)


class SuffixResumeExecutionAdmission(StrictModel):
    """Fail-closed decision to inject retained artifacts or replay the full plan."""

    schema_version: str = "x-editor.suffix-resume-execution-admission/v1"
    admission_id: str = Field(default_factory=lambda: new_workflow_id("resume_admission"))
    checkpoint_id: str | None = None
    assessment_id: str | None = None
    status: SuffixResumeExecutionStatus
    candidate_plan_fingerprint: str
    reused_step_ids: list[str] = Field(default_factory=list)
    rerun_step_ids: list[str] = Field(default_factory=list)
    receipts: list[SuffixResumeReuseReceipt] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    findings: list[SuffixResumeFinding] = Field(default_factory=list)
    skipped_tool_call_count: int = Field(default=0, ge=0)
    behavior_changed: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_execution_admission(self) -> "SuffixResumeExecutionAdmission":
        if self.status == "admitted":
            if not self.receipts or not self.reused_step_ids:
                raise ValueError("admitted suffix resume requires reuse receipts")
            if not self.behavior_changed:
                raise ValueError("admitted suffix resume must report behavior change")
            receipt_steps = sorted({item.producer_step_id for item in self.receipts})
            if receipt_steps != sorted(self.reused_step_ids):
                raise ValueError("reused_step_ids must match receipt producers")
        elif self.receipts or self.reused_step_ids or self.behavior_changed:
            raise ValueError("full replay cannot retain or inject reusable artifacts")
        if self.skipped_tool_call_count != sum(
            item.skipped_tool_call_count for item in self.receipts
        ):
            raise ValueError("skipped_tool_call_count must equal receipt totals")
        return self


class WorkflowEvent(StrictModel):
    """One append-only fact emitted by a workflow-control session."""

    schema_version: str = "x-editor.workflow-event/v1"
    event_id: str = Field(default_factory=lambda: new_workflow_id("event"))
    run_id: str
    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)
    stage: WorkflowStage
    kind: str
    status: WorkflowEventStatus = "observed"
    attempt_id: str | None = None
    subject_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
