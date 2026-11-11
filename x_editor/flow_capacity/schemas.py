"""Versioned offline contracts for protocol-coupled editing-flow measurement.

These contracts are sidecar infrastructure. They do not change the X-Editor
runtime, select tools, admit artifacts, or grant Evaluator results control over
execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ModuleId = Literal[
    "planner_reasoner",
    "planner",
    "reasoner",
    "perception",
    "executor",
    "tool",
    "verifier",
    "feedback_controller",
    "artifact_store",
    "final_assembler",
    "evaluator",
]
InputRegime = Literal["standard", "natural"]
AggregationRole = Literal["protocol_gate", "critical_cap", "soft"]
MeasurementState = Literal[
    "unmeasured",
    "measured",
    "cannot_determine",
    "scorer_unavailable",
    "scorer_conflict",
]
ExecutionState = Literal[
    "not_attempted",
    "succeeded",
    "failed",
    "provider_unavailable",
    "service_unavailable",
    "timeout",
    "blocked_upstream",
    "unsupported",
    "schema_invalid",
    "missing_required_output",
    "invalid_fallback",
]
ProtocolState = Literal["not_checked", "accepted", "rejected", "inconclusive"]
TruthBasis = Literal["none", "prediction", "deterministic_contract", "hidden_gt", "human"]
ClaimRole = Literal["headline", "supporting", "diagnostic", "readiness_only"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Reject undeclared fields in measurement contracts."""

    model_config = ConfigDict(extra="forbid")


class VersionedRef(StrictModel):
    ref_id: str
    schema_version: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaseIdentity(StrictModel):
    """Offline identity; only ``case_fingerprint`` may be prompt-visible."""

    record_uid: str
    case_fingerprint: str
    sidecar_record_key: str
    sidecar_record_key_prompt_visible: Literal[False] = False
    base_case_id: str
    session_id: str | None = None
    canonical_group_id: str | None = None
    pair_group_id: str | None = None
    variant_id: str | None = None


class RuntimeAudioAsset(StrictModel):
    asset_ref: str
    evidence_ref_id: str
    role: str
    order: int = Field(ge=0)


class RuntimeCaseRecord(StrictModel):
    """Prompt-safe source record; local paths are resolved through evidence refs."""

    schema_version: Literal["x-editor.flow-runtime-case/v1"] = "x-editor.flow-runtime-case/v1"
    record_uid: str
    case_fingerprint: str
    request: str
    audio_assets: list[RuntimeAudioAsset] = Field(default_factory=list)
    source_text: str | None = None
    visible_context: dict[str, Any] = Field(default_factory=dict)
    session_history: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_assets(self) -> "RuntimeCaseRecord":
        orders = [item.order for item in self.audio_assets]
        refs = [item.asset_ref for item in self.audio_assets]
        if len(orders) != len(set(orders)):
            raise ValueError("runtime audio asset order must be unique")
        if len(refs) != len(set(refs)):
            raise ValueError("runtime audio asset refs must be unique")
        return self


class EvidenceRef(StrictModel):
    schema_version: Literal["x-editor.flow-evidence-ref/v1"] = "x-editor.flow-evidence-ref/v1"
    evidence_ref_id: str
    uri: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    media_type: str | None = None
    producer: str
    phase: Literal["pre_module", "module_output", "downstream", "scoring"]
    visibility: Literal["runtime", "target_module_only", "runner_only", "evaluator_private"]
    trust_role: TruthBasis = "none"
    available_to: list[str] = Field(default_factory=list)
    join_scope: Literal["visible_equivalence", "exact_sidecar_trace"]
    prompt_visible: bool = False
    case_fingerprint: str | None = None
    sidecar_record_key: str | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> "EvidenceRef":
        if self.join_scope == "exact_sidecar_trace" and not self.sidecar_record_key:
            raise ValueError("exact sidecar evidence requires sidecar_record_key")
        if self.sidecar_record_key and self.prompt_visible:
            raise ValueError("sidecar_record_key evidence cannot be prompt-visible")
        if self.visibility == "evaluator_private" and self.prompt_visible:
            raise ValueError("evaluator-private evidence cannot be prompt-visible")
        return self


class StandardInputRecord(StrictModel):
    """Pre-registered qualified input for one target module capacity arm."""

    schema_version: Literal["x-editor.flow-standard-input/v1"] = "x-editor.flow-standard-input/v1"
    standard_input_id: str
    assessment_record_id: str
    target_module: ModuleId
    boundary_id: str
    contract_ref: VersionedRef
    basis: Literal[
        "visible_derived_curated",
        "oracle_boundary_contract",
        "public_synthetic",
        "protocol_fixture",
    ]
    payload: dict[str, Any]
    candidate_set_id: str
    selection_role: Literal["canonical", "robustness"]
    selection_rule: str
    qualification: Literal["qualified", "unqualified", "unknown"]
    reviewer_or_generator: str
    contains_target_artifact: Literal[False] = False
    contains_scorer_output: Literal[False] = False
    contains_benchmark_identity: Literal[False] = False
    visible_to_upstream_modules: Literal[False] = False


class ModuleBoundary(StrictModel):
    module_id: ModuleId
    boundary_id: str
    input_contract_ref: VersionedRef
    output_contract_ref: VersionedRef
    protocol_version: str


class InputArmRef(StrictModel):
    arm_id: str
    input_regime: InputRegime
    state: Literal["available", "unmeasured", "blocked", "not_applicable"]
    payload_ref: str | None = None
    producer_ref: str | None = None
    selection_policy_ref: str
    trust_role: Literal[
        "qualified_standard_input",
        "module_prediction_not_gt",
        "not_applicable",
    ]

    @model_validator(mode="after")
    def validate_payload(self) -> "InputArmRef":
        if self.state == "available" and not self.payload_ref:
            raise ValueError("available input arm requires payload_ref")
        if self.input_regime == "natural" and self.trust_role == "qualified_standard_input":
            raise ValueError("natural arm cannot be labeled qualified standard input")
        return self


class EligibilitySpec(StrictModel):
    applicability: Literal["applicable", "not_applicable"]
    eligibility: Literal["eligible", "selection_only", "blocked", "ineligible"]
    eligibility_reason: str
    qualification: Literal["qualified", "unqualified", "unknown"]
    denominator_policy_ref: str


class ModuleAssessmentRecord(StrictModel):
    schema_version: Literal["x-editor.module-assessment/v1"] = "x-editor.module-assessment/v1"
    assessment_record_id: str
    identity: CaseIdentity
    boundary: ModuleBoundary
    case_roles: list[
        Literal["protocol", "atomic_capability", "composed_realistic", "boundary_feedback"]
    ]
    runtime_record_uid: str
    input_arms: list[InputArmRef]
    objective_refs: list[str]
    evidence_refs: list[str] = Field(default_factory=list)
    strata: dict[str, str] = Field(default_factory=dict)
    budget_profile_ref: str
    eligibility: EligibilitySpec
    comparison_group_id: str | None = None
    feedback_spec_ref: str | None = None
    legacy_source_ref: str | None = None

    @model_validator(mode="after")
    def validate_arms(self) -> "ModuleAssessmentRecord":
        arm_ids = [arm.arm_id for arm in self.input_arms]
        regimes = [arm.input_regime for arm in self.input_arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("assessment input arm ids must be unique")
        if len(regimes) != len(set(regimes)):
            raise ValueError("assessment may define at most one arm per input regime")
        if not self.objective_refs:
            raise ValueError("assessment requires at least one objective ref")
        return self


class BoundaryExpectation(StrictModel):
    boundary_id: str
    acceptable_spec_ref: str


class AtomicObjective(StrictModel):
    schema_version: Literal["x-editor.atomic-objective/v1"] = "x-editor.atomic-objective/v1"
    objective_instance_id: str
    flow_group_id: str
    objective_class: Literal["edit_effect", "preservation", "delivery"]
    objective_stage: Literal["process", "final"]
    counting_role: Literal["leaf_flow", "diagnostic_parent", "diagnostic_only"]
    weight: float | None = Field(default=None, gt=0)
    weight_policy_ref: str
    objective_importance: Literal["required", "optional"]
    dependencies: list[str] = Field(default_factory=list)
    boundary_expectations: list[BoundaryExpectation] = Field(default_factory=list)
    metric_binding_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counting_role(self) -> "AtomicObjective":
        if self.counting_role == "leaf_flow" and self.weight is None:
            raise ValueError("leaf flow objective requires a positive weight")
        if self.counting_role != "leaf_flow" and self.weight is not None:
            raise ValueError("diagnostic objectives must not carry flow weight")
        if self.objective_instance_id in self.dependencies:
            raise ValueError("objective cannot depend on itself")
        return self


class CalibrationAnchor(StrictModel):
    raw_value: float
    utility: float = Field(ge=0, le=1)


class MetricTransform(StrictModel):
    kind: Literal["binary", "ordinal_lookup", "piecewise_linear", "identity_probability"]
    direction: Literal["higher_better", "lower_better", "categorical"]
    anchors: list[CalibrationAnchor] = Field(default_factory=list)
    lookup: dict[str, float] = Field(default_factory=dict)
    calibration_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_transform(self) -> "MetricTransform":
        if self.kind == "piecewise_linear":
            if len(self.anchors) < 2:
                raise ValueError("piecewise transform requires at least two anchors")
            raw = [item.raw_value for item in self.anchors]
            if raw != sorted(raw) or len(raw) != len(set(raw)):
                raise ValueError("piecewise anchor raw values must be unique and ascending")
            utilities = [item.utility for item in self.anchors]
            if self.direction == "higher_better" and utilities != sorted(utilities):
                raise ValueError("higher-better anchor utilities must be nondecreasing")
            if self.direction == "lower_better" and utilities != sorted(utilities, reverse=True):
                raise ValueError("lower-better anchor utilities must be nonincreasing")
        elif self.kind == "ordinal_lookup" and not self.lookup:
            raise ValueError("ordinal lookup transform requires lookup values")
        elif self.kind == "ordinal_lookup" and any(
            value < 0 or value > 1 for value in self.lookup.values()
        ):
            raise ValueError("ordinal lookup utilities must be in [0, 1]")
        elif self.kind in {"binary", "identity_probability"} and (self.anchors or self.lookup):
            raise ValueError(f"{self.kind} transform cannot define anchors or lookup")
        if self.kind == "piecewise_linear" and self.direction == "categorical":
            raise ValueError("piecewise transform requires an ordered metric direction")
        if self.kind == "identity_probability" and not self.calibration_artifact_sha256:
            raise ValueError("identity probability requires independent calibration evidence")
        return self


class MetricBinding(StrictModel):
    schema_version: Literal["x-editor.metric-binding/v1"] = "x-editor.metric-binding/v1"
    binding_id: str
    objective_instance_id: str
    metric_id: str
    aggregation_role: AggregationRole
    applicability: Literal["applicable", "not_applicable"]
    required: bool = True
    transform: MetricTransform
    pass_threshold: float | None = Field(default=None, ge=0, le=1)
    soft_weight: float | None = Field(default=None, gt=0)
    missing_policy: Literal["unknown", "protocol_failure"] = "unknown"

    @model_validator(mode="after")
    def validate_role(self) -> "MetricBinding":
        if self.aggregation_role == "soft":
            if self.soft_weight is None:
                raise ValueError("soft metric requires soft_weight")
            if self.pass_threshold is not None:
                raise ValueError("soft metric cannot define pass_threshold")
        else:
            if self.soft_weight is not None:
                raise ValueError("gate and critical metrics cannot define soft_weight")
        if self.aggregation_role == "protocol_gate" and self.pass_threshold is None:
            raise ValueError("protocol gate requires pass_threshold")
        return self


class BudgetProfile(StrictModel):
    schema_version: Literal["x-editor.capacity-budget/v1"] = "x-editor.capacity-budget/v1"
    budget_profile_id: str
    attempts_k: int = Field(ge=1)
    selection_policy_ref: str
    wall_time_seconds: float | None = Field(default=None, gt=0)
    provider_call_limit: int | None = Field(default=None, ge=0)
    tool_call_limit: int | None = Field(default=None, ge=0)
    token_limit: int | None = Field(default=None, ge=0)
    audio_compute_seconds: float | None = Field(default=None, ge=0)
    allowed_feedback_edges: list[str] = Field(default_factory=list)
    cache_policy: str
    randomness: dict[str, Any] = Field(default_factory=dict)
    concurrency_assumptions: str


class SplitMembership(StrictModel):
    record_uid: str
    split: str
    base_case_id: str
    session_id: str | None = None
    canonical_group_id: str | None = None
    pair_group_id: str | None = None


class MigrationRecord(StrictModel):
    schema_version: Literal["x-editor.flow-release-migration/v1"] = (
        "x-editor.flow-release-migration/v1"
    )
    migration_id: str
    source_release_id: str
    destination_release_id: str
    migration_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    migration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    field_mapping: dict[str, str]
    eligibility_changes: dict[str, int] = Field(default_factory=dict)
    lossy: bool
    loss_notes: list[str] = Field(default_factory=list)
    before_counts: dict[str, int]
    after_counts: dict[str, int]
    recomputed_fingerprint_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_releases(self) -> "MigrationRecord":
        if self.source_release_id == self.destination_release_id:
            raise ValueError("migration must create a distinct destination release")
        if self.lossy and not self.loss_notes:
            raise ValueError("lossy migration requires loss_notes")
        return self


class ReleaseManifest(StrictModel):
    schema_version: Literal["x-editor.flow-capacity-release/v1"] = "x-editor.flow-capacity-release/v1"
    release_id: str
    created_at: datetime
    dataset_version: str
    protocol_versions: dict[str, str]
    objective_weight_policy_ref: str
    denominator_policy_ref: str
    runtime_case_count: int = Field(ge=0)
    assessment_count: int = Field(ge=0)
    objective_count: int = Field(ge=0)
    standard_input_count: int = Field(ge=0)
    evidence_ref_count: int = Field(ge=0)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    migration_ref: str | None = None
    immutable: Literal[True] = True

    @model_validator(mode="after")
    def validate_source_hashes(self) -> "ReleaseManifest":
        invalid = [
            key
            for key, value in self.source_hashes.items()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        ]
        if invalid:
            raise ValueError(f"source hashes must be lowercase SHA-256 values: {invalid}")
        return self


class ObservationContext(StrictModel):
    run_id: str
    assessment_record_id: str
    case_fingerprint: str
    sidecar_record_key: str
    boundary_id: str
    arm_id: str
    input_regime: InputRegime
    variant_id: str | None = None
    round_id: str | None = None
    objective_instance_id: str
    attempt_id: str
    replicate_id: str


class MetricObservation(StrictModel):
    schema_version: Literal["x-editor.metric-observation/v1"] = "x-editor.metric-observation/v1"
    observation_id: str
    context: ObservationContext
    metric_id: str
    binding_id: str
    measurement_state: MeasurementState
    raw_value: Any | None = None
    normalized_utility: float | None = Field(default=None, ge=0, le=1)
    truth_basis: TruthBasis
    claim_role: ClaimRole
    evidence_refs: list[str] = Field(default_factory=list)
    scoring_run_id: str | None = None
    scorer_id: str | None = None
    scorer_attempt_id: str | None = None
    repeat_index: int | None = Field(default=None, ge=0)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_measurement(self) -> "MetricObservation":
        if self.measurement_state == "measured" and self.raw_value is None:
            raise ValueError("measured observation requires raw_value")
        if self.measurement_state != "measured" and self.normalized_utility is not None:
            raise ValueError("unmeasured observation cannot carry normalized utility")
        return self


class ObjectiveOutcome(StrictModel):
    schema_version: Literal["x-editor.objective-outcome/v1"] = "x-editor.objective-outcome/v1"
    outcome_id: str
    context: ObservationContext
    weight: float = Field(gt=0)
    eligible_at_admission: bool
    launch_intended: bool
    attempt_started: bool
    execution_state: ExecutionState
    protocol_state: ProtocolState
    measurement_state: MeasurementState
    protocol_gate: float | None = Field(default=None, ge=0, le=1)
    critical_score: float | None = Field(default=None, ge=0, le=1)
    soft_score: float | None = Field(default=None, ge=0, le=1)
    objective_utility: float | None = Field(default=None, ge=0, le=1)
    flow_status: Literal[
        "measured",
        "unknown",
        "ineligible",
        "not_applicable",
        "execution_failure",
        "protocol_failure",
    ]
    failure_owner: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    selected_for_forward_flow: bool = False
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_utility(self) -> "ObjectiveOutcome":
        measured_states = {"measured", "execution_failure", "protocol_failure"}
        if self.flow_status in measured_states and self.objective_utility is None:
            raise ValueError(f"{self.flow_status} outcome requires objective utility")
        if self.flow_status in {"unknown", "ineligible", "not_applicable"} and self.objective_utility is not None:
            raise ValueError(f"{self.flow_status} outcome cannot carry objective utility")
        if self.attempt_started and not self.launch_intended:
            raise ValueError("attempt_started requires launch_intended")
        return self


class AttemptFact(StrictModel):
    schema_version: Literal["x-editor.capacity-attempt/v1"] = "x-editor.capacity-attempt/v1"
    run_id: str
    attempt_id: str
    attempt_kind: Literal["runtime", "feedback", "branch"]
    attempt_group_id: str
    parent_attempt_id: str | None = None
    branch_id: str | None = None
    selected_for_forward_flow: bool = False
    attempt_started: bool
    seed_block: str | None = None
    execution_state: ExecutionState
    cost: dict[str, float] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)


class FailureUnknownFact(StrictModel):
    schema_version: Literal["x-editor.capacity-failure-unknown/v1"] = (
        "x-editor.capacity-failure-unknown/v1"
    )
    fact_id: str
    run_id: str
    assessment_record_id: str
    objective_instance_id: str | None = None
    attempt_id: str | None = None
    category: Literal["failure", "unknown", "blocked", "unsupported"]
    reason_code: str
    failure_stage: str | None = None
    failure_owner: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    source_event_id: str | None = None


class SupportDefinition(StrictModel):
    release_id: str
    split_or_cohort: str
    eligible_objective_ids: list[str]
    boundary_id: str
    protocol_version: str
    objective_weight_policy_ref: str
    eligibility_policy_ref: str
    budget_profile_ref: str
    attempts_k: int = Field(ge=1)
    selection_policy_ref: str
    normalization_version: str
    aggregation_version: str
    evidence_policy_ref: str
    truth_policy_ref: str
    opportunity_set_hash: str | None = None

    @model_validator(mode="after")
    def validate_objective_support(self) -> "SupportDefinition":
        if not self.eligible_objective_ids:
            raise ValueError("support requires at least one eligible objective")
        if len(self.eligible_objective_ids) != len(set(self.eligible_objective_ids)):
            raise ValueError("eligible objective ids must be unique")
        return self


class DenominatorSnapshot(StrictModel):
    schema_version: Literal["x-editor.capacity-denominator/v1"] = (
        "x-editor.capacity-denominator/v1"
    )
    denominator_snapshot_id: str
    support_digest: str
    support: SupportDefinition
    arm_id: str
    input_regime: InputRegime
    implementation_ref: str
    variant_id: str
    eligible_weight: float = Field(ge=0)


class FlowAggregate(StrictModel):
    schema_version: Literal["x-editor.flow-aggregate/v1"] = "x-editor.flow-aggregate/v1"
    estimand: Literal[
        "deployable_capacity",
        "natural_flow",
        "conditional_execution_quality",
        "service_reliability",
        "oracle_upper_bound",
    ]
    denominator_snapshot_id: str
    eligible_weight: float = Field(ge=0)
    measured_weight: float = Field(ge=0)
    unknown_weight: float = Field(ge=0)
    flow_credit: float = Field(ge=0)
    measurement_coverage: float | None = Field(default=None, ge=0, le=1)
    lower_bound: float | None = Field(default=None, ge=0, le=1)
    upper_bound: float | None = Field(default=None, ge=0, le=1)
    point_estimate: float | None = Field(default=None, ge=0, le=1)
    headline_eligible: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class ComparisonGate(StrictModel):
    relation: Literal["utilization", "replacement", "feedback"]
    comparable: bool
    reason_codes: list[str] = Field(default_factory=list)


class VerifierDecisionObservation(StrictModel):
    schema_version: Literal["x-editor.verifier-decision-observation/v1"] = (
        "x-editor.verifier-decision-observation/v1"
    )
    observation_id: str
    objective_instance_id: str
    weight: float = Field(gt=0)
    runtime_decision: Literal["accept", "reject", "defer"]
    trusted_contract_label: Literal["safe", "unsafe"]
    objective_utility: float | None = Field(default=None, ge=0, le=1)
    harm_weight: float | None = Field(default=None, ge=0, le=1)
    truth_basis: Literal["deterministic_contract", "hidden_gt", "human"]
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_label_payload(self) -> "VerifierDecisionObservation":
        if self.trusted_contract_label == "safe":
            if self.objective_utility is None:
                raise ValueError("safe verifier label requires objective_utility")
            if self.harm_weight is not None:
                raise ValueError("safe verifier label cannot carry harm_weight")
        else:
            if self.harm_weight is None:
                raise ValueError("unsafe verifier label requires harm_weight")
            if self.objective_utility is not None:
                raise ValueError("unsafe verifier label cannot carry objective_utility")
        return self


class VerifierFlowSummary(StrictModel):
    schema_version: Literal["x-editor.verifier-flow-summary/v1"] = (
        "x-editor.verifier-flow-summary/v1"
    )
    eligible_weight: float = Field(ge=0)
    safe_throughput: float | None = Field(default=None, ge=0, le=1)
    unsafe_accept_flow: float | None = Field(default=None, ge=0, le=1)
    false_reject_loss: float | None = Field(default=None, ge=0, le=1)
    defer_weight: float = Field(ge=0)
    false_accept_rate: float | None = Field(default=None, ge=0, le=1)
    false_reject_rate: float | None = Field(default=None, ge=0, le=1)
    confusion_counts: dict[str, int] = Field(default_factory=dict)


class CapacityRunManifest(StrictModel):
    schema_version: Literal["x-editor.capacity-run/v1"] = "x-editor.capacity-run/v1"
    report_id: str
    experiment_id: str
    created_at: datetime = Field(default_factory=utc_now)
    release_ref: VersionedRef
    git_commit: str
    dirty_tree_digest: str
    protocol_versions: dict[str, str]
    implementation_refs: dict[str, str]
    budget_profile_refs: list[str]
    normalization_versions: list[str]
    scorer_versions: dict[str, str] = Field(default_factory=dict)
    seeds: list[str] = Field(default_factory=list)
    cache_policy: str
    source_snapshot_hashes: dict[str, str]
    immutable: Literal[True] = True


class ValidationIssue(StrictModel):
    severity: Literal["error", "warning"]
    code: str
    record_ref: str | None = None
    message: str


class ReleaseValidationReport(StrictModel):
    schema_version: Literal["x-editor.flow-release-validation/v1"] = (
        "x-editor.flow-release-validation/v1"
    )
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_counts: dict[str, int] = Field(default_factory=dict)


class ReadinessReport(StrictModel):
    schema_version: Literal["x-editor.capacity-readiness-report/v1"] = (
        "x-editor.capacity-readiness-report/v1"
    )
    report_id: str
    release_id: str
    claim_status: Literal["readiness_only"] = "readiness_only"
    capacity_point_estimate: None = None
    runtime_case_count: int = Field(ge=0)
    assessment_count: int = Field(ge=0)
    exact_case_count: int = Field(ge=0)
    visible_fingerprint_count: int = Field(ge=0)
    visible_fingerprint_collision_count: int = Field(ge=0)
    eligibility_counts: dict[str, int] = Field(default_factory=dict)
    arm_state_counts: dict[str, int] = Field(default_factory=dict)
    validation_passed: bool
    reason_codes: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
