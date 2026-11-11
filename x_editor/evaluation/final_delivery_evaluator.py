"""Benchmark-neutral multimodal evaluation of one committed final delivery.

The public input contract deliberately has no sidecar or arbitrary metadata
field.  A caller may provide only the public instruction, content-addressed
artifacts, and effect/scope evidence emitted by the adopted runtime branch.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel
from x_editor.evaluation.delivery_review import DeliveryCoverageRequirement, DeliveryReviewContext
from x_editor.evaluation.effect_profiles import (
    EvaluationEffectContract,
    EvaluationProperty,
    EvaluationTemporalScope,
    resolve_evaluation_effect_contract,
)
from x_editor.evaluation.guidance_audio import materialize_artifact_evaluation_clips
from x_editor.evaluation.guidance_contracts import (
    ArtifactQualityEvaluationTask,
    AtomicEvaluationRubric,
    EvaluationArtifactRef,
    EvaluationAudioRegion,
    EvaluationEffectAtom,
    EvaluationJudgePolicy,
    EvaluationResultV2,
    EvaluationSamplingPolicy,
    canonical_evaluation_fingerprint,
)
from x_editor.evaluation.guidance_path import (
    EvaluationGuidancePath,
    freeze_evaluation_selection,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_POLICY_ID = "benchmark-neutral-final-delivery-v1"


class FinalDeliveryIdentity(StrictModel):
    """Immutable identity needed to join one terminal artifact across evaluators."""

    schema_version: Literal["x-editor.final-delivery-identity/v1"] = (
        "x-editor.final-delivery-identity/v1"
    )
    assignment_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    run_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    runtime_projection_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    artifact_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    terminal_attempt_index: int = Field(ge=0)


class FinalDeliveryIntervalPair(StrictModel):
    """Auditable source/output coordinates supplied by runtime lineage."""

    schema_version: Literal["x-editor.final-delivery-interval-pair/v1"] = (
        "x-editor.final-delivery-interval-pair/v1"
    )
    pair_id: str = Field(min_length=1)
    source_artifact_ref_id: str = Field(min_length=1)
    source_start_sec: float = Field(ge=0.0)
    source_end_sec: float = Field(gt=0.0)
    output_start_sec: float = Field(ge=0.0)
    output_end_sec: float = Field(gt=0.0)
    purpose: Literal["edited", "preservation"] = "edited"

    @model_validator(mode="after")
    def validate_geometry(self) -> FinalDeliveryIntervalPair:
        values = (
            self.source_start_sec,
            self.source_end_sec,
            self.output_start_sec,
            self.output_end_sec,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("interval pair coordinates must be finite")
        if self.source_end_sec <= self.source_start_sec:
            raise ValueError("source interval must have positive duration")
        if self.output_end_sec <= self.output_start_sec:
            raise ValueError("output interval must have positive duration")
        return self


class FinalDeliveryEvaluationInput(StrictModel):
    """Allowlisted public/runtime inputs for final-delivery evaluation."""

    schema_version: Literal["x-editor.final-delivery-evaluation-input/v1"] = (
        "x-editor.final-delivery-evaluation-input/v1"
    )
    identity: FinalDeliveryIdentity
    public_instruction: str = Field(min_length=1)
    input_artifacts: tuple[EvaluationArtifactRef, ...] = Field(min_length=1)
    output_artifact: EvaluationArtifactRef
    reference_artifacts: tuple[EvaluationArtifactRef, ...] = ()
    adopted_operations: tuple[str, ...] = ()
    declared_temporal_scope: Literal["global", "local", "structural"] | None = None
    interval_pairs: tuple[FinalDeliveryIntervalPair, ...] = ()
    delivery_context: DeliveryReviewContext | None = None
    backend_id: str = Field(min_length=1)
    context_padding_sec: float = Field(default=0.25, ge=0.0, le=0.3)
    judge_policy: EvaluationJudgePolicy = Field(default_factory=EvaluationJudgePolicy)

    @model_validator(mode="after")
    def validate_public_projection(self) -> FinalDeliveryEvaluationInput:
        if self.public_instruction != self.public_instruction.strip():
            raise ValueError("public_instruction must be stripped")
        artifacts = (*self.input_artifacts, self.output_artifact, *self.reference_artifacts)
        artifact_ids = [item.artifact_ref_id for item in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("evaluation artifact refs must be unique")
        if self.output_artifact.role != "output":
            raise ValueError("final delivery artifact must have output role")
        if self.output_artifact.content_sha256 != self.identity.artifact_fingerprint:
            raise ValueError("identity artifact fingerprint does not match final output")
        if len(self.interval_pairs) != len({item.pair_id for item in self.interval_pairs}):
            raise ValueError("interval pair IDs must be unique")
        input_ids = {item.artifact_ref_id for item in self.input_artifacts}
        unknown_sources = {
            pair.source_artifact_ref_id
            for pair in self.interval_pairs
            if pair.source_artifact_ref_id not in input_ids
        }
        if unknown_sources:
            raise ValueError(
                f"interval pairs reference non-input artifacts: {sorted(unknown_sources)}"
            )
        if any(not operation.strip() for operation in self.adopted_operations):
            raise ValueError("adopted operation names must be non-empty")
        if self.delivery_context is not None:
            if self.delivery_context.current_request != self.public_instruction:
                raise ValueError("delivery context must bind the current public request")
            if self.delivery_context.terminal_artifact_ref_id != self.output_artifact.artifact_ref_id:
                raise ValueError("delivery context must bind the terminal output")
        return self


class FinalDeliveryEvaluationPreparation(StrictModel):
    """Ready task or an explicit pre-provider abstention."""

    schema_version: Literal["x-editor.final-delivery-evaluation-preparation/v1"] = (
        "x-editor.final-delivery-evaluation-preparation/v1"
    )
    identity: FinalDeliveryIdentity
    status: Literal["ready", "abstain"]
    reason: str = Field(min_length=1)
    effect_contract: EvaluationEffectContract
    effective_temporal_scope: EvaluationTemporalScope
    task: ArtifactQualityEvaluationTask | None = None
    can_feedback_to_runtime: Literal[False] = False
    can_select_artifact: Literal[False] = False

    @model_validator(mode="after")
    def validate_status(self) -> FinalDeliveryEvaluationPreparation:
        if (self.status == "ready") != (self.task is not None):
            raise ValueError("ready preparation requires exactly one evaluation task")
        return self


class FinalDeliveryEvaluationRecord(StrictModel):
    """Terminal diagnostic result with join identity kept outside the task score."""

    schema_version: Literal["x-editor.final-delivery-evaluation-record/v1"] = (
        "x-editor.final-delivery-evaluation-record/v1"
    )
    identity: FinalDeliveryIdentity
    status: Literal["scored", "unknown", "abstain", "evaluation_failure"]
    reason: str = Field(min_length=1)
    task_fingerprint: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    result: EvaluationResultV2 | None = None
    utility: float | None = Field(default=None, ge=0.0, le=1.0)
    known_weight_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    can_feedback_to_runtime: Literal[False] = False
    can_select_artifact: Literal[False] = False

    @model_validator(mode="after")
    def validate_result_binding(self) -> FinalDeliveryEvaluationRecord:
        if self.result is None:
            if self.utility is not None or self.known_weight_fraction != 0.0:
                raise ValueError("record without a result cannot carry evaluator utility")
            if self.status not in {"abstain", "evaluation_failure"}:
                raise ValueError("scored or unknown record requires an evaluation result")
            return self
        if self.status not in {"scored", "unknown"}:
            raise ValueError("evaluation result is incompatible with terminal status")
        if self.utility != self.result.known_score:
            raise ValueError("record utility must equal EvaluationResultV2.known_score")
        if self.known_weight_fraction != self.result.known_weight_fraction:
            raise ValueError("record coverage must equal EvaluationResultV2 coverage")
        return self


def prepare_final_delivery_evaluation(
    value: FinalDeliveryEvaluationInput,
) -> FinalDeliveryEvaluationPreparation:
    """Build one blind task without reading benchmark sidecars or expected answers."""

    _verify_artifacts((*value.input_artifacts, value.output_artifact, *value.reference_artifacts))
    effect_contract = resolve_evaluation_effect_contract(list(value.adopted_operations))
    operation_chain = effect_contract.operation_chain
    state_selection = _is_bound_state_selection(value)
    if not state_selection and (operation_chain is None or operation_chain.missing_operations):
        return FinalDeliveryEvaluationPreparation(
            identity=value.identity,
            status="abstain",
            reason=(
                "runtime operation chain is absent or contains operations outside "
                "the evaluation effect whitelist"
            ),
            effect_contract=effect_contract,
            effective_temporal_scope=effect_contract.temporal_scope,
        )
    effective_scope = "global" if state_selection else _effective_scope(
        value, effect_contract=effect_contract
    )
    if effective_scope in {"unknown", "local_or_global", "mixed"}:
        return FinalDeliveryEvaluationPreparation(
            identity=value.identity,
            status="abstain",
            reason=("runtime effect scope is not specific enough to bind final audio evidence"),
            effect_contract=effect_contract,
            effective_temporal_scope=effective_scope,
        )
    if effective_scope == "local" and not value.interval_pairs:
        return FinalDeliveryEvaluationPreparation(
            identity=value.identity,
            status="abstain",
            reason="local runtime effect has no auditable source/output interval mapping",
            effect_contract=effect_contract,
            effective_temporal_scope=effective_scope,
        )
    if effective_scope != "local" and value.interval_pairs and not state_selection:
        raise ValueError("local interval mappings conflict with a whole-artifact scope")
    if effective_scope == "local" and not any(
        pair.purpose == "edited" for pair in value.interval_pairs
    ):
        return FinalDeliveryEvaluationPreparation(
            identity=value.identity,
            status="abstain",
            reason="local runtime mapping has no edited source/output pair",
            effect_contract=effect_contract,
            effective_temporal_scope=effective_scope,
        )

    regions = _regions(value) if effective_scope == "local" else ()
    edited_region_ids = tuple(
        region.region_id
        for pair in value.interval_pairs
        if pair.purpose == "edited"
        for region in _pair_regions(pair, output_ref_id=value.output_artifact.artifact_ref_id)
    ) if effective_scope == "local" else ()
    all_region_ids = tuple(region.region_id for region in regions)
    preservation_requirements = () if state_selection else _preservation_requirements(effect_contract)
    rubrics = _rubrics(
        public_instruction=value.public_instruction,
        edited_region_ids=edited_region_ids,
        preservation_region_ids=all_region_ids,
        preservation_requirements=preservation_requirements,
        include_boundary_atom=effective_scope == "local",
    )
    effect = EvaluationEffectAtom(
        effect_id="public_instruction",
        requirement=value.public_instruction,
        effect_kind="public_instruction_observable",
        target_refs=(value.output_artifact.artifact_ref_id,),
        region_ids=edited_region_ids,
    )
    selection = freeze_evaluation_selection(
        EvaluationSamplingPolicy(policy_id=_POLICY_ID, mode="final_only"),
        unit_id=f"assignment:{value.identity.assignment_fingerprint}:final_delivery",
        focus="final_delivery",
    )
    effect_contract_fingerprint = canonical_evaluation_fingerprint(effect_contract)
    task = ArtifactQualityEvaluationTask(
        task_id=f"final_delivery_{value.identity.assignment_fingerprint[:24]}",
        run_id=value.identity.run_fingerprint,
        checkpoint_id=f"delivery:{value.identity.artifact_fingerprint[:24]}",
        input_snapshot_ref=value.identity.runtime_projection_fingerprint,
        focus="final_delivery",
        evaluator_role="post_delivery_development_diagnostic",
        evaluation_use="realized_flow_diagnostic",
        backend_id=value.backend_id,
        visibility="blind_quality",
        public_instruction=value.public_instruction,
        input_artifacts=value.input_artifacts,
        output_artifact=value.output_artifact,
        reference_artifacts=value.reference_artifacts,
        producer_node_id=None,
        effect_contract_ref=f"effect_contract:{effect_contract_fingerprint}",
        intended_effects=(effect,),
        preservation_requirements=preservation_requirements,
        regions=regions,
        rubrics=rubrics,
        context_padding_sec=value.context_padding_sec,
        delivery_context=value.delivery_context,
        coverage_requirements=(
            _coverage_requirements(value, local=effective_scope == "local")
            if value.delivery_context is not None else ()
        ),
        judge_policy=value.judge_policy,
        selection=selection,
    )
    return FinalDeliveryEvaluationPreparation(
        identity=value.identity,
        status="ready",
        reason="public instruction and runtime effect evidence are evaluation-ready",
        effect_contract=effect_contract,
        effective_temporal_scope=effective_scope,
        task=task,
    )


def evaluate_final_delivery(
    preparation: FinalDeliveryEvaluationPreparation,
    *,
    evaluation_path: EvaluationGuidancePath,
    clip_output_dir: str | Path,
) -> FinalDeliveryEvaluationRecord:
    """Evaluate a prepared terminal artifact without opening a guidance edge."""

    if preparation.status == "abstain":
        return FinalDeliveryEvaluationRecord(
            identity=preparation.identity,
            status="abstain",
            reason=preparation.reason,
        )
    assert preparation.task is not None
    task = preparation.task
    if task.regions:
        task = materialize_artifact_evaluation_clips(
            task,
            output_dir=clip_output_dir,
        )
    task_fingerprint = canonical_evaluation_fingerprint(task)
    outcome = evaluation_path.evaluate(task)
    if outcome.result is None:
        return FinalDeliveryEvaluationRecord(
            identity=preparation.identity,
            status="evaluation_failure",
            reason="; ".join(outcome.errors) or outcome.status,
            task_fingerprint=task_fingerprint,
        )
    result = outcome.result
    return FinalDeliveryEvaluationRecord(
        identity=preparation.identity,
        status="scored" if result.known_score is not None else "unknown",
        reason=(
            "atomic rubric utility aggregated"
            if result.known_score is not None
            else f"evaluation score status is {result.score_status}"
        ),
        task_fingerprint=task_fingerprint,
        result=result,
        utility=result.known_score,
        known_weight_fraction=result.known_weight_fraction,
    )


def _effective_scope(
    value: FinalDeliveryEvaluationInput,
    *,
    effect_contract: EvaluationEffectContract,
) -> EvaluationTemporalScope:
    catalog_scope = effect_contract.temporal_scope
    declared = value.declared_temporal_scope
    if (
        declared is not None
        and catalog_scope in {"global", "local", "structural"}
        and declared != catalog_scope
    ):
        raise ValueError(
            "declared runtime temporal scope conflicts with the operation effect catalog"
        )
    if declared is not None:
        return declared
    if value.interval_pairs and catalog_scope in {"local", "local_or_global", "unknown"}:
        return "local"
    return catalog_scope


def _is_bound_state_selection(value: FinalDeliveryEvaluationInput) -> bool:
    """A historical base was selected and its exact bytes were delivered.

    A graph may create a new terminal node for a restoration. Node IDs need not
    match, while the selected base and output must have bound artifact identity.
    This is an observed delivery mode, not approval of the request interpretation.
    """

    context = value.delivery_context
    if (
        context is None or context.selected_base_state_id is None
        or context.round_entry_state_id is None
        or context.round_entry_state_id == context.selected_base_state_id
    ):
        return False
    base = next(state for state in context.states if state.state_id == context.selected_base_state_id)
    artifacts = {item.artifact_ref_id: item for item in (*value.input_artifacts, value.output_artifact)}
    selected_artifact = artifacts.get(base.artifact_ref_id)
    return selected_artifact is not None and (
        selected_artifact.content_sha256 == value.output_artifact.content_sha256
    )


def _coverage_requirements(
    value: FinalDeliveryEvaluationInput,
    *,
    local: bool,
) -> tuple[DeliveryCoverageRequirement, ...]:
    edited_pairs = [pair for pair in value.interval_pairs if pair.purpose == "edited"] if local else []
    if not edited_pairs:
        return (DeliveryCoverageRequirement(
            coverage_id="whole_delivery", rubric_id="effect.public_instruction"
        ),)
    return tuple(
        DeliveryCoverageRequirement(
            coverage_id=f"edit:{pair.pair_id}",
            rubric_id="effect.public_instruction",
            evidence_region_ids=tuple(
                region.region_id for region in _pair_regions(
                    pair, output_ref_id=value.output_artifact.artifact_ref_id
                )
            ),
        ) for pair in edited_pairs
    )


def _regions(value: FinalDeliveryEvaluationInput) -> tuple[EvaluationAudioRegion, ...]:
    pairs = tuple(
        region
        for pair in value.interval_pairs
        for region in _pair_regions(pair, output_ref_id=value.output_artifact.artifact_ref_id)
    )
    if value.delivery_context is None:
        return pairs
    # Local crops support per-location listening; the full recordings support
    # target-inventory completeness and retained context. Neither substitutes
    # for the other. Unknown durations remain an explicit evidence limitation.
    context = tuple(
        EvaluationAudioRegion(
            region_id=f"delivery_context:{artifact.artifact_ref_id}",
            artifact_ref_id=artifact.artifact_ref_id,
            start_sec=0.0,
            end_sec=artifact.duration_sec,
            label="Complete recording for requested-scope and retained-context review",
            contains_unedited_context=True,
        )
        for artifact in (*value.input_artifacts, value.output_artifact)
        if artifact.duration_sec is not None and artifact.duration_sec > 0
    )
    return (*pairs, *context)


def _pair_regions(
    pair: FinalDeliveryIntervalPair,
    *,
    output_ref_id: str,
) -> tuple[EvaluationAudioRegion, EvaluationAudioRegion]:
    return (
        EvaluationAudioRegion(
            region_id=f"pair:{pair.pair_id}:source",
            artifact_ref_id=pair.source_artifact_ref_id,
            start_sec=pair.source_start_sec,
            end_sec=pair.source_end_sec,
            label=f"{pair.purpose} source interval {pair.pair_id}",
            contains_unedited_context=pair.purpose == "edited",
        ),
        EvaluationAudioRegion(
            region_id=f"pair:{pair.pair_id}:output",
            artifact_ref_id=output_ref_id,
            start_sec=pair.output_start_sec,
            end_sec=pair.output_end_sec,
            label=f"{pair.purpose} output interval {pair.pair_id}",
            contains_unedited_context=pair.purpose == "edited",
        ),
    )


_PRESERVATION_TEXT: dict[EvaluationProperty, str] = {
    "wording": "Spoken wording not requested to change remains consistent.",
    "language": "The spoken language remains consistent where no language change is requested.",
    "speaker_identity": "Speaker identity remains consistent where it is not requested to change.",
    "vocal_timbre": "Vocal timbre remains consistent where it is not requested to change.",
    "acoustic_properties": (
        "Background sound and acoustic properties remain consistent where not requested to change."
    ),
    "duration": "Duration remains consistent where no duration change is requested.",
    "local_timing": "Local timing remains consistent where no timing change is requested.",
}


def _preservation_requirements(
    effect_contract: EvaluationEffectContract,
) -> tuple[str, ...]:
    return tuple(
        _PRESERVATION_TEXT[property_name]
        for property_name in _PRESERVATION_TEXT
        if effect_contract.dispositions[property_name] == "preserved"
    )


def _rubrics(
    *,
    public_instruction: str,
    edited_region_ids: tuple[str, ...],
    preservation_region_ids: tuple[str, ...],
    preservation_requirements: tuple[str, ...],
    include_boundary_atom: bool,
) -> tuple[AtomicEvaluationRubric, ...]:
    rubrics: list[AtomicEvaluationRubric] = [
        AtomicEvaluationRubric(
            rubric_id="effect.public_instruction",
            dimension="instruction_following",
            question=(
                "Compared with the public input audio, to what degree does the final "
                f"audio audibly satisfy this instruction: {public_instruction}"
            ),
            observable="listener-observable satisfaction of the public instruction",
            evidence_region_ids=tuple(
                dict.fromkeys((*edited_region_ids, *preservation_region_ids))
            ),
            target_module="workflow",
        ),
        AtomicEvaluationRubric(
            rubric_id="quality.intelligibility",
            dimension="perceptual_quality",
            question=(
                "When the requested result contains speech, is the evaluated output "
                "speech intelligible?"
            ),
            observable="intelligible output speech when speech is applicable",
            evidence_region_ids=edited_region_ids,
            target_module="workflow",
        ),
        AtomicEvaluationRubric(
            rubric_id="quality.naturalness",
            dimension="perceptual_quality",
            question="Does the evaluated output sound natural for the requested effect?",
            observable="natural audio for the requested effect",
            evidence_region_ids=edited_region_ids,
            target_module="workflow",
        ),
        AtomicEvaluationRubric(
            rubric_id="quality.artifact_absence",
            dimension="perceptual_quality",
            question=(
                "Is the evaluated output free from unintended clicks, clipping, "
                "distortion, added noise, or discontinuity?"
            ),
            observable="absence of unintended audible artifacts",
            evidence_region_ids=edited_region_ids,
            target_module="workflow",
        ),
    ]
    if include_boundary_atom:
        rubrics.append(
            AtomicEvaluationRubric(
                rubric_id="quality.boundary_continuity",
                dimension="perceptual_quality",
                question=(
                    "Are transitions at the exact evaluated edit boundaries audibly continuous?"
                ),
                observable="continuous transitions at local edit boundaries",
                evidence_region_ids=edited_region_ids,
                target_module="workflow",
            )
        )
    rubrics.extend(
        AtomicEvaluationRubric(
            rubric_id=f"preservation.{index:02d}",
            dimension="preservation",
            question=f"To what degree is this property preserved: {requirement}",
            observable=requirement,
            evidence_region_ids=preservation_region_ids,
            target_module="workflow",
        )
        for index, requirement in enumerate(preservation_requirements, start=1)
    )
    return tuple(rubrics)


def _verify_artifacts(artifacts: tuple[EvaluationArtifactRef, ...]) -> None:
    for artifact in artifacts:
        path = Path(artifact.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evaluation artifact is unavailable: {path}")
        if _sha256_file(path) != artifact.content_sha256:
            raise ValueError(f"evaluation artifact hash mismatch: {artifact.artifact_ref_id}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FinalDeliveryEvaluationInput",
    "FinalDeliveryEvaluationPreparation",
    "FinalDeliveryEvaluationRecord",
    "FinalDeliveryIdentity",
    "FinalDeliveryIntervalPair",
    "evaluate_final_delivery",
    "prepare_final_delivery_evaluation",
]
