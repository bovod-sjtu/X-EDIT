"""Executor-authored temporal transforms for post-delivery evaluation."""

from __future__ import annotations

from x_editor.tools.catalog.audio_separator_roformer.time_axis import source_aligned_audio_covers

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
from pydantic import Field, model_validator

from x_editor.core.schemas import AudioEditStep, StrictModel
from x_editor.runtime.audio_references import canonical_audio_ref_for_runtime_path
from x_editor.runtime.artifact_timeline import timeline_for_audio_path
from x_editor.runtime.timeline_ledger import (
    DeletionEdit,
    ReplacementEdit,
    TimeInterval,
    TimelineLedger,
    TimelineStepTransform,
)
from x_editor.workflow_control.schemas import (
    AudioTemporalSpan,
    AudioTemporalSpanMapping,
    AudioTemporalTransform,
)

AUDIO_TEMPORAL_TRANSFORM_KEY = "audio_temporal_transform"
AUDIO_TEMPORAL_LINEAGE_KEY = "audio_temporal_lineage"
AUDIO_TIMELINE_LEDGER_KEY = "audio_timeline_ledger"
_ALIGNMENT_BOUNDARY_TOLERANCE_SEC = 0.05
_STRUCTURAL_TIME_TOLERANCE_SEC = 1e-4
_QUALITY_DURATION_TOLERANCE_SEC = 0.05


class AudioTemporalLineageMapping(StrictModel):
    """One typed source/output pair with an explicit temporal purpose."""

    mapping_id: str = Field(min_length=1)
    role: Literal[
        "structural_mapping",
        "semantic_anchor",
        "affected_support",
    ]
    source_span: AudioTemporalSpan
    output_span: AudioTemporalSpan
    provenance_paths: list[str] = Field(min_length=1)


class AudioTemporalLineage(StrictModel):
    """Executor-authored temporal facts whose roles cannot be interchanged."""

    schema_version: Literal["x-editor.audio-temporal-lineage/v1"] = (
        "x-editor.audio-temporal-lineage/v1"
    )
    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    status: Literal["complete", "partial", "unavailable"]
    edit_scope: Literal["local", "global", "unknown"]
    input_revision: int = Field(ge=0)
    output_revision: int = Field(ge=1)
    input_audio_paths: list[str] = Field(default_factory=list)
    input_audio_refs: list[str] = Field(default_factory=list)
    output_audio_path: str = Field(min_length=1)
    input_duration_sec: float | None = Field(default=None, gt=0.0)
    output_duration_sec: float | None = Field(default=None, gt=0.0)
    exact_timeline_duration_preserved: bool | None = None
    duration_change_within_quality_tolerance: bool | None = None
    quality_duration_tolerance_sec: float = Field(
        default=_QUALITY_DURATION_TOLERANCE_SEC,
        gt=0.0,
    )
    structural_mappings: list[AudioTemporalLineageMapping] = Field(
        default_factory=list
    )
    semantic_anchors: list[AudioTemporalLineageMapping] = Field(default_factory=list)
    affected_support: list[AudioTemporalLineageMapping] = Field(default_factory=list)
    identity_outside_affected_support: bool | None = None
    derivation: str = Field(min_length=1)
    reason_code: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_lineage(self) -> AudioTemporalLineage:
        if self.output_revision != self.input_revision + 1:
            raise ValueError("temporal lineage revisions must advance by exactly one")
        groups = (
            ("structural_mapping", self.structural_mappings),
            ("semantic_anchor", self.semantic_anchors),
            ("affected_support", self.affected_support),
        )
        for role, mappings in groups:
            if any(mapping.role != role for mapping in mappings):
                raise ValueError(f"{role} list contains a mapping with another role")
        mapping_ids = [
            mapping.mapping_id
            for _, mappings in groups
            for mapping in mappings
        ]
        if len(mapping_ids) != len(set(mapping_ids)):
            raise ValueError("temporal lineage mapping ids must be unique")
        if self.status == "complete":
            if not self.input_audio_paths:
                raise ValueError("complete temporal lineage requires an input audio path")
            if self.input_duration_sec is None or self.output_duration_sec is None:
                raise ValueError("complete temporal lineage requires decoded durations")
            if self.edit_scope == "local" and not self.structural_mappings:
                raise ValueError("complete local lineage requires structural mappings")
            if self.reason_code is not None or self.reason is not None:
                raise ValueError("complete temporal lineage cannot carry a failure reason")
        elif not self.reason_code or not self.reason:
            raise ValueError("incomplete temporal lineage requires a reason")
        if self.edit_scope != "local" and self.identity_outside_affected_support is not None:
            raise ValueError(
                "identity_outside_affected_support applies only to local lineages"
            )
        return self


@dataclass(frozen=True)
class _DecodedAudioInfo:
    duration_sec: float
    sample_rate_hz: int


@dataclass(frozen=True)
class _LegacyPipelineTemporalEvidence:
    present: bool
    structural_mappings: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]], ...
    ] = ()
    semantic_anchors: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]], ...
    ] = ()
    affected_support: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]], ...
    ] = ()
    identity_outside_affected_support: bool | None = None
    reason_code: str | None = None
    reason: str | None = None


_GLOBAL_WHOLE_INPUT_OPERATIONS = frozenset(
    {
        "accent_conversion",
        "artifact_ops.apply_gain",
        "edit_data_pipeline.speech_rate",
        "edit_data_pipeline.volume_normalize",
        "no_reference_voice_conversion",
        "seed_vc",
        "seamless_expressive_s2st",
        "spatial_reverb",
        "speech_enhancement",
    }
)


def attach_audio_temporal_transform(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
    input_audio_paths: list[str],
    commit_to_state: bool = True,
) -> dict[str, Any]:
    """Attach one conservative transform without interpreting user intent.

    Direct callers retain the historical eager-ledger behavior. The executor
    passes ``commit_to_state=False`` so a candidate timeline is not authoritative
    until runtime verification accepts the artifact.
    """

    output_path = _output_audio_path(output)
    if output_path is None or not Path(output_path).expanduser().is_file():
        return output
    temporal_input_audio_paths = _temporal_input_audio_paths(
        state=state,
        consumed_audio_paths=input_audio_paths,
    )
    legacy_evidence = _legacy_pipeline_temporal_evidence(
        step=step,
        output=output,
        input_audio_paths=temporal_input_audio_paths,
        output_path=output_path,
    )
    transform = _build_transform(
        step=step,
        state=state,
        output=output,
        input_audio_paths=temporal_input_audio_paths,
        output_path=output_path,
        legacy_evidence=legacy_evidence,
    )
    timeline_state = state
    if len(temporal_input_audio_paths) == 1:
        input_path = temporal_input_audio_paths[0]
        input_ledger = timeline_for_audio_path(state, input_path)
        if input_ledger is not None:
            # Scope only temporal composition; do not change the execution cursor
            # or grant a candidate output authority before verification.
            timeline_state = {
                **state,
                "current_audio_path": input_path,
                AUDIO_TIMELINE_LEDGER_KEY: input_ledger,
            }
    prior_ledger = _load_or_initialize_ledger(timeline_state)
    input_revision = (
        prior_ledger.current_timebase.revision if prior_ledger is not None else 0
    )
    lineage = _build_lineage(
        step=step,
        state=state,
        transform=transform,
        legacy_evidence=legacy_evidence,
        input_audio_paths=temporal_input_audio_paths,
        output_path=output_path,
        input_revision=input_revision,
    )
    delivered = dict(output)
    delivered[AUDIO_TEMPORAL_TRANSFORM_KEY] = transform.model_dump(mode="json")
    delivered[AUDIO_TEMPORAL_LINEAGE_KEY] = lineage.model_dump(mode="json")
    ledger = _advance_timeline_ledger(
        step=step,
        state=timeline_state,
        transform=transform,
        output_path=output_path,
    )
    if ledger is not None:
        payload = ledger.model_dump(mode="json", exclude_computed_fields=True)
        if commit_to_state:
            state[AUDIO_TIMELINE_LEDGER_KEY] = payload
        delivered[AUDIO_TIMELINE_LEDGER_KEY] = payload
    return delivered


def _temporal_input_audio_paths(
    *,
    state: dict[str, Any],
    consumed_audio_paths: list[str],
) -> list[str]:
    """Select the artifact whose timeline is transformed by this step.

    A tool may consume auxiliary audio such as a voice reference without
    inserting that asset into the output timeline.  The executor's current
    artifact is authoritative when it is among the consumed inputs.  A lone
    consumed input is also unambiguous.  Multiple inputs without either fact
    require an operation-specific multi-source mapping and stay unavailable.
    """

    current_path = str(state.get("current_audio_path") or "").strip()
    if current_path and _same_path_is_consumed(current_path, consumed_audio_paths):
        return [current_path]
    if len(consumed_audio_paths) == 1:
        return copy.deepcopy(consumed_audio_paths)
    return []


def _advance_timeline_ledger(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    transform: AudioTemporalTransform,
    output_path: str,
) -> TimelineLedger | None:
    """Compose one proven step transform into the executor-owned ledger."""

    ledger = _load_or_initialize_ledger(state)
    if ledger is None:
        return None
    input_path = str(state.get("current_audio_path") or "").strip()
    input_duration = _audio_duration(input_path) if input_path else None
    output_duration = _audio_duration(output_path)
    revision = ledger.current_timebase.revision
    if input_duration is None or output_duration is None:
        step_transform = TimelineStepTransform.unavailable(
            step_id=step.id,
            operation=step.operation,
            input_revision=revision,
            input_duration_sec=input_duration,
            output_duration_sec=output_duration,
            reason_code="audio_duration_unavailable",
            reason="Executor could not decode an input or output duration.",
        )
        return ledger.compose(step_transform)
    if not _same_path_is_consumed(input_path, transform.input_audio_paths):
        step_transform = TimelineStepTransform.unavailable(
            step_id=step.id,
            operation=step.operation,
            input_revision=revision,
            input_duration_sec=input_duration,
            output_duration_sec=output_duration,
            reason_code="timeline_input_not_consumed",
            reason="The successful handler did not identify the current artifact as an input.",
        )
        return ledger.compose(step_transform)
    return ledger.compose(
        _ledger_step_transform(
            step=step,
            transform=transform,
            input_revision=revision,
            input_duration_sec=input_duration,
            output_duration_sec=output_duration,
        )
    )


def _load_or_initialize_ledger(state: dict[str, Any]) -> TimelineLedger | None:
    existing = state.get(AUDIO_TIMELINE_LEDGER_KEY)
    if isinstance(existing, TimelineLedger):
        return existing
    if isinstance(existing, dict):
        return TimelineLedger.model_validate(existing)
    base_path = str(state.get("input_audio") or "").strip()
    base_duration = _audio_duration(base_path) if base_path else None
    if base_duration is None:
        return None
    return TimelineLedger.identity(
        base_duration_sec=base_duration,
        base_audio_ref=(
            canonical_audio_ref_for_runtime_path(base_path, state) or base_path
        ),
    )


def _ledger_step_transform(
    *,
    step: AudioEditStep,
    transform: AudioTemporalTransform,
    input_revision: int,
    input_duration_sec: float,
    output_duration_sec: float,
) -> TimelineStepTransform:
    unavailable = {
        "step_id": step.id,
        "operation": step.operation,
        "input_revision": input_revision,
        "input_duration_sec": input_duration_sec,
        "output_duration_sec": output_duration_sec,
    }
    if transform.status != "complete":
        return TimelineStepTransform.unavailable(
            **unavailable,
            reason_code="step_temporal_transform_incomplete",
            reason="The step did not publish a complete temporal transform.",
        )
    if (step.operation == "artifact_ops.extract_spans"
            and transform.derivation == "single_extracted_span_compacted_to_output"
            and len(transform.timeline_mappings) == 1):
        # A crop is exact on its retained samples, with the complement deleted.
        # It cannot satisfy identity_outside_regions, a replacement-only rule.
        from x_editor.runtime.audio_frame_geometry import AudioFrameGeometry
        source_info = sf.info(transform.input_audio_paths[0])
        output_info = sf.info(transform.output_audio_path)
        geometry = AudioFrameGeometry(source_info.samplerate, source_info.frames)
        span = transform.timeline_mappings[0].source_span
        first, last = geometry.interval(span.start_sec, span.end_sec)
        if (output_info.samplerate != source_info.samplerate
                or output_info.frames != last - first):
            return TimelineStepTransform.unavailable(
                **unavailable, reason_code="crop_frame_count_mismatch",
                reason="The extracted output does not match the declared input frames.",
            )
        start, end = first / geometry.sample_rate, last / geometry.sample_rate
        removed = []
        if start > 0:
            removed.append(DeletionEdit(input_interval=TimeInterval(start_sec=0, end_sec=start)))
        if end < input_duration_sec:
            removed.append(DeletionEdit(input_interval=TimeInterval(start_sec=end, end_sec=input_duration_sec)))
        if not removed:
            return TimelineStepTransform.identity(
                step_id=step.id, operation=step.operation,
                input_revision=input_revision, duration_sec=input_duration_sec,
            )
        return TimelineStepTransform.deletion(
            step_id=step.id, operation=step.operation, input_revision=input_revision,
            input_duration_sec=input_duration_sec, deletions=removed,
        )
    if transform.edit_scope == "global":
        if transform.timeline_duration_preserved is True:
            return TimelineStepTransform.identity(
                step_id=step.id,
                operation=step.operation,
                input_revision=input_revision,
                duration_sec=input_duration_sec,
            )
        return TimelineStepTransform.unavailable(
            **unavailable,
            reason_code="global_transform_has_no_alignment",
            reason="A duration-changing global edit requires an explicit alignment contract.",
        )
    if transform.edit_scope != "local":
        return TimelineStepTransform.unavailable(**unavailable)
    if transform.identity_outside_regions is not True or not transform.timeline_mappings:
        return TimelineStepTransform.unavailable(
            **unavailable,
            reason_code="local_transform_not_piecewise_auditable",
            reason="The local edit did not prove identity outside explicit mapped regions.",
        )
    replacements: list[ReplacementEdit] = []
    for item in transform.timeline_mappings:
        source_start = item.source_span.start_sec
        source_end = item.source_span.end_sec
        if source_end > input_duration_sec:
            overshoot = source_end - input_duration_sec
            if (
                overshoot <= _ALIGNMENT_BOUNDARY_TOLERANCE_SEC
                and source_start < input_duration_sec
            ):
                source_end = input_duration_sec
            else:
                return TimelineStepTransform.unavailable(
                    **unavailable,
                    reason_code="temporal_mapping_exceeds_input",
                    reason=(
                        "Published source span exceeds the decoded input duration "
                        "beyond the alignment tolerance."
                    ),
                )
        replacements.append(
            ReplacementEdit(
                input_interval=TimeInterval(
                    start_sec=source_start,
                    end_sec=source_end,
                ),
                output_duration_sec=(
                    item.output_span.end_sec - item.output_span.start_sec
                ),
            )
        )
    try:
        candidate = TimelineStepTransform.replacement(
            step_id=step.id,
            operation=step.operation,
            input_revision=input_revision,
            input_duration_sec=input_duration_sec,
            replacements=replacements,
        )
    except ValueError as exc:
        return TimelineStepTransform.unavailable(
            **unavailable,
            reason_code="invalid_piecewise_temporal_mapping",
            reason=f"Published local timeline mapping is not composable: {exc}",
        )
    explicit_outputs = [
        (item.output_span.start_sec, item.output_span.end_sec)
        for item in transform.timeline_mappings
    ]
    candidate_outputs = [
        (piece.output_interval.start_sec, piece.output_interval.end_sec)
        for piece in candidate.pieces
        if piece.kind == "replacement" and piece.output_interval is not None
    ]
    if len(explicit_outputs) != len(candidate_outputs) or any(
        abs(left_start - right_start) > 0.05
        or abs(left_end - right_end) > 0.05
        for (left_start, left_end), (right_start, right_end) in zip(
            explicit_outputs,
            candidate_outputs,
            strict=True,
        )
    ):
        return TimelineStepTransform.unavailable(
            **unavailable,
            reason_code="explicit_mapping_disagrees_with_piecewise_delta",
            reason="Published output spans disagree with deterministic duration-delta composition.",
        )
    return candidate


def _same_path_is_consumed(current_path: str, consumed_paths: list[str]) -> bool:
    try:
        current = Path(current_path).expanduser().resolve()
        return any(Path(item).expanduser().resolve() == current for item in consumed_paths)
    except (OSError, RuntimeError):
        return current_path in consumed_paths


def _legacy_pipeline_temporal_evidence(
    *,
    step: AudioEditStep,
    output: dict[str, Any],
    input_audio_paths: list[str],
    output_path: str,
) -> _LegacyPipelineTemporalEvidence:
    """Read typed roles from a legacy text-edit sidecar without changing it."""

    if step.operation == "edit_data_pipeline.pitch_rate_edit":
        return _legacy_pitch_rate_temporal_evidence(
            step=step,
            output=output,
            input_audio_paths=input_audio_paths,
            output_path=output_path,
        )
    if step.operation == "edit_data_pipeline.pause_edit":
        return _legacy_pause_temporal_evidence(
            step=step,
            output=output,
            input_audio_paths=input_audio_paths,
            output_path=output_path,
        )
    if step.operation != "edit_data_pipeline.text_edit":
        return _LegacyPipelineTemporalEvidence(present=False)
    generation = _legacy_pipeline_generation(output)
    if generation is None:
        return _LegacyPipelineTemporalEvidence(present=False)
    generation_id, generation_payload = generation
    root_path = f"result.metadata.generations.{generation_id}"
    if len(input_audio_paths) != 1:
        return _legacy_temporal_error(
            "legacy_pipeline_requires_one_input",
            "Legacy text-edit structural lineage requires exactly one input audio.",
        )
    input_info = _decoded_audio_info(input_audio_paths[0])
    output_info = _decoded_audio_info(output_path)
    if input_info is None or output_info is None:
        return _legacy_temporal_error(
            "legacy_pipeline_audio_duration_unavailable",
            "Legacy text-edit lineage could not decode its input or output audio.",
        )
    synthesis = generation_payload.get("synthesis")
    raw_steps = synthesis.get("steps") if isinstance(synthesis, dict) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        return _legacy_one_shot_temporal_evidence(
            generation_payload=generation_payload,
            root_path=root_path,
            input_info=input_info,
            output_info=output_info,
        )

    ledger = TimelineLedger.identity(base_duration_sec=input_info.duration_sec)
    prefix_ledgers = [ledger]
    support_rows: list[tuple[int, tuple[tuple[float, float], ...], tuple[str, ...]]] = []
    identity_outside_support = True
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            return _legacy_temporal_error(
                "legacy_pipeline_structural_step_invalid",
                "Legacy text-edit synthesis step is not an object.",
            )
        step_path = f"{root_path}.synthesis.steps[{index}]"
        source_duration = _positive_float(raw_step.get("source_duration"))
        target_duration = _legacy_step_target_duration(raw_step)
        if source_duration is None or target_duration is None:
            return _legacy_temporal_error(
                "legacy_pipeline_step_duration_missing",
                "Legacy text-edit synthesis step lacks explicit source/target duration.",
            )
        if ledger.current_duration_sec is None or not _values_close(
            source_duration,
            ledger.current_duration_sec,
            tolerance_sec=_STRUCTURAL_TIME_TOLERANCE_SEC,
        ):
            return _legacy_temporal_error(
                "legacy_pipeline_step_input_duration_mismatch",
                "Legacy text-edit synthesis step input duration breaks the internal chain.",
            )
        span_plan = raw_step.get("span_plan")
        source_spans = (
            _span_list(span_plan.get("source_spans"))
            if isinstance(span_plan, dict)
            else []
        )
        target_spans = (
            _span_list(span_plan.get("target_spans"))
            if isinstance(span_plan, dict)
            else []
        )
        if not source_spans or len(source_spans) != len(target_spans):
            return _legacy_temporal_error(
                "legacy_pipeline_structural_spans_invalid",
                "Legacy text-edit synthesis step lacks paired structural spans.",
            )
        try:
            step_transform = TimelineStepTransform.replacement(
                step_id=f"{step.id}:legacy:{index}",
                operation=step.operation,
                input_revision=index,
                input_duration_sec=source_duration,
                replacements=[
                    ReplacementEdit(
                        input_interval=TimeInterval(start_sec=start, end_sec=end),
                        output_duration_sec=target_end - target_start,
                    )
                    for (start, end), (target_start, target_end) in zip(
                        source_spans,
                        target_spans,
                        strict=True,
                    )
                ],
            )
        except ValueError as exc:
            return _legacy_temporal_error(
                "legacy_pipeline_structural_spans_not_composable",
                f"Legacy text-edit structural spans are not composable: {exc}",
            )
        explicit_outputs = target_spans
        deterministic_outputs = [
            (piece.output_interval.start_sec, piece.output_interval.end_sec)
            for piece in step_transform.pieces
            if piece.kind == "replacement" and piece.output_interval is not None
        ]
        if (
            len(explicit_outputs) != len(deterministic_outputs)
            or any(
                not _bounds_close(left, right)
                for left, right in zip(
                    explicit_outputs,
                    deterministic_outputs,
                    strict=True,
                )
            )
            or step_transform.output_duration_sec is None
            or not _values_close(
                step_transform.output_duration_sec,
                target_duration,
                tolerance_sec=_STRUCTURAL_TIME_TOLERANCE_SEC,
            )
        ):
            return _legacy_temporal_error(
                "legacy_pipeline_structural_output_mismatch",
                "Legacy text-edit structural spans disagree with their declared duration.",
            )
        try:
            ledger = ledger.compose(step_transform)
        except ValueError as exc:
            return _legacy_temporal_error(
                "legacy_pipeline_internal_composition_failed",
                f"Legacy text-edit internal timeline composition failed: {exc}",
            )
        if ledger.status != "complete":
            return _legacy_temporal_error(
                ledger.reason_code or "legacy_pipeline_internal_lineage_unavailable",
                ledger.reason or "Legacy text-edit internal lineage is unavailable.",
            )
        prefix_ledgers.append(ledger)

        synthesis_payload = raw_step.get("synthesis")
        service_metadata = (
            synthesis_payload.get("f5_service_metadata")
            if isinstance(synthesis_payload, dict)
            else None
        )
        mask_spans = (
            _span_list(service_metadata.get("mask_spans_sec"))
            if isinstance(service_metadata, dict)
            else []
        )
        if mask_spans:
            support_rows.append(
                (
                    index,
                    tuple(mask_spans),
                    (
                        f"{step_path}.synthesis.f5_service_metadata.mask_spans_sec",
                    ),
                )
            )
        preserve_unmasked = (
            synthesis_payload.get("preserve_unmasked_waveform")
            if isinstance(synthesis_payload, dict)
            else None
        )
        if preserve_unmasked is not True:
            identity_outside_support = False

    if ledger.current_duration_sec is None or not _duration_matches_decoded_audio(
        ledger.current_duration_sec,
        output_info,
    ):
        return _legacy_temporal_error(
            "legacy_pipeline_final_duration_mismatch",
            "Legacy text-edit structural lineage does not reproduce decoded output duration.",
        )

    structural = tuple(
        (
            (piece.base_interval.start_sec, piece.base_interval.end_sec),
            (piece.current_interval.start_sec, piece.current_interval.end_sec),
            (f"{root_path}.synthesis.steps[*].span_plan",),
        )
        for piece in ledger.pieces
        if piece.kind != "identity"
        and piece.base_interval is not None
        and piece.current_interval is not None
    )
    if not structural:
        return _legacy_temporal_error(
            "legacy_pipeline_structural_mapping_empty",
            "Legacy text-edit sidecar produced no changed structural interval.",
        )

    semantic: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]], ...
    ] = ()
    generation_span_plan = generation_payload.get("span_plan")
    if isinstance(generation_span_plan, dict):
        span_metadata = generation_span_plan.get("metadata")
        raw_source_spans = (
            _span_list(span_metadata.get("raw_source_spans"))
            if isinstance(span_metadata, dict)
            else []
        )
        target_aligned_spans = _span_list(
            generation_span_plan.get("target_aligned_spans")
        )
        if (
            raw_source_spans
            and len(raw_source_spans) == len(target_aligned_spans)
            and _spans_fit_duration(raw_source_spans, input_info.duration_sec)
            and _spans_fit_duration(target_aligned_spans, output_info.duration_sec)
        ):
            semantic = tuple(
                (
                    source,
                    target,
                    (
                        f"{root_path}.span_plan.metadata.raw_source_spans",
                        f"{root_path}.span_plan.target_aligned_spans",
                    ),
                )
                for source, target in zip(
                    raw_source_spans,
                    target_aligned_spans,
                    strict=True,
                )
            )

    affected: list[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]]
    ] = []
    for internal_index, mask_spans, provenance in support_rows:
        after_step = prefix_ledgers[internal_index + 1]
        for mask_span in mask_spans:
            source_support = _map_current_interval_envelope(after_step, mask_span)
            output_support = (
                _map_base_interval_envelope(ledger, source_support)
                if source_support is not None
                else None
            )
            if source_support is not None and output_support is not None:
                affected.append((source_support, output_support, provenance))
    if not affected:
        affected = [
            (source, target, (*provenance, "structural_mapping_conservative_support"))
            for source, target, provenance in structural
        ]

    return _LegacyPipelineTemporalEvidence(
        present=True,
        structural_mappings=structural,
        semantic_anchors=semantic,
        affected_support=tuple(affected),
        identity_outside_affected_support=identity_outside_support,
    )


def _legacy_pitch_rate_temporal_evidence(
    *,
    step: AudioEditStep,
    output: dict[str, Any],
    input_audio_paths: list[str],
    output_path: str,
) -> _LegacyPipelineTemporalEvidence:
    edit_result = output.get("edit_result")
    rows = edit_result.get("edits_applied") if isinstance(edit_result, dict) else None
    if not isinstance(rows, list) or not rows:
        return _LegacyPipelineTemporalEvidence(present=False)
    decoded = _single_input_output_info(
        input_audio_paths=input_audio_paths,
        output_path=output_path,
        operation_label="Pitch/rate edit",
    )
    if isinstance(decoded, _LegacyPipelineTemporalEvidence):
        return decoded
    input_info, output_info = decoded
    if edit_result.get("smoothing_applied") is not False:
        return _legacy_temporal_error(
            "pitch_rate_smoothing_support_missing",
            "Pitch/rate edit did not prove the support affected by smoothing.",
        )

    source_spans: list[tuple[float, float]] = []
    output_spans: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return _legacy_temporal_error(
                "pitch_rate_edit_row_invalid",
                "Pitch/rate edit returned a non-object edit row.",
            )
        source = _single_span(row.get("source_span"))
        target = _single_span(row.get("target_span"))
        if source is None or target is None:
            return _legacy_temporal_error(
                "pitch_rate_span_pair_missing",
                f"Pitch/rate edit row {index} lacks a source/output span pair.",
            )
        source_spans.append(source)
        output_spans.append(target)
    if not _replacement_mappings_match_decoded_audio(
        step=step,
        source_spans=source_spans,
        output_spans=output_spans,
        input_info=input_info,
        output_info=output_info,
    ):
        return _legacy_temporal_error(
            "pitch_rate_span_pairs_not_composable",
            "Pitch/rate edit span pairs do not reproduce the decoded output timeline.",
        )
    mappings = tuple(
        (
            source,
            target,
            (f"result.edit_result.edits_applied[{index}]",),
        )
        for index, (source, target) in enumerate(
            zip(source_spans, output_spans, strict=True)
        )
    )
    return _LegacyPipelineTemporalEvidence(
        present=True,
        structural_mappings=mappings,
        semantic_anchors=mappings,
        affected_support=mappings,
        identity_outside_affected_support=True,
    )


def _legacy_pause_temporal_evidence(
    *,
    step: AudioEditStep,
    output: dict[str, Any],
    input_audio_paths: list[str],
    output_path: str,
) -> _LegacyPipelineTemporalEvidence:
    plan = output.get("pause_edit_plan")
    if not isinstance(plan, dict):
        return _LegacyPipelineTemporalEvidence(present=False)
    decoded = _single_input_output_info(
        input_audio_paths=input_audio_paths,
        output_path=output_path,
        operation_label="Pause edit",
    )
    if isinstance(decoded, _LegacyPipelineTemporalEvidence):
        return decoded
    input_info, output_info = decoded
    anchors = _float_list(plan.get("insert_position_sec"), positive=False)
    durations = _float_list(plan.get("pause_duration_sec"), positive=True)
    pause_intervals = _span_list(plan.get("pause_interval_sec"))
    if (
        not anchors
        or not durations
        or not pause_intervals
        or len(anchors) != len(durations)
        or len(anchors) != len(pause_intervals)
        or anchors != sorted(anchors)
    ):
        return _legacy_temporal_error(
            "pause_insertion_plan_invalid",
            "Pause edit lacks a complete ordered insertion plan.",
        )
    cumulative = 0.0
    for anchor, duration, pause in zip(
        anchors,
        durations,
        pause_intervals,
        strict=True,
    ):
        if (
            abs(pause[0] - (anchor + cumulative)) > 0.05
            or abs((pause[1] - pause[0]) - duration) > 0.05
        ):
            return _legacy_temporal_error(
                "pause_insertion_timeline_mismatch",
                "Pause intervals disagree with their source anchors and durations.",
            )
        cumulative += duration

    support_spans = _pause_output_support_spans(plan, count=len(anchors))
    if not support_spans:
        return _legacy_temporal_error(
            "pause_affected_support_missing",
            "Pause edit did not report the output support affected by smoothing.",
        )
    source_support: list[tuple[float, float]] = []
    for support in support_spans:
        source_start = _inverse_pause_insertion_boundary(
            support[0],
            anchors=anchors,
            durations=durations,
            pause_intervals=pause_intervals,
        )
        source_end = _inverse_pause_insertion_boundary(
            support[1],
            anchors=anchors,
            durations=durations,
            pause_intervals=pause_intervals,
        )
        if source_start is None or source_end is None or source_end <= source_start:
            return _legacy_temporal_error(
                "pause_affected_support_not_invertible",
                "Pause smoothing support cannot be mapped back to the source timeline.",
            )
        source_support.append((source_start, source_end))
    if not _replacement_mappings_match_decoded_audio(
        step=step,
        source_spans=source_support,
        output_spans=support_spans,
        input_info=input_info,
        output_info=output_info,
    ):
        return _legacy_temporal_error(
            "pause_support_not_composable",
            "Pause support mappings do not reproduce the decoded output timeline.",
        )
    mappings = tuple(
        (
            source,
            target,
            (
                "result.pause_edit_plan.insert_position_sec",
                "result.pause_edit_plan.pause_duration_sec",
                "result.pause_edit_plan.pause_interval_sec",
                "result.pause_edit_plan.smoothing_points",
            ),
        )
        for source, target in zip(source_support, support_spans, strict=True)
    )
    return _LegacyPipelineTemporalEvidence(
        present=True,
        structural_mappings=mappings,
        affected_support=mappings,
        identity_outside_affected_support=True,
    )


def _single_input_output_info(
    *,
    input_audio_paths: list[str],
    output_path: str,
    operation_label: str,
) -> tuple[_DecodedAudioInfo, _DecodedAudioInfo] | _LegacyPipelineTemporalEvidence:
    if len(input_audio_paths) != 1:
        return _legacy_temporal_error(
            "local_pipeline_requires_one_input",
            f"{operation_label} temporal lineage requires exactly one input audio.",
        )
    input_info = _decoded_audio_info(input_audio_paths[0])
    output_info = _decoded_audio_info(output_path)
    if input_info is None or output_info is None:
        return _legacy_temporal_error(
            "local_pipeline_audio_duration_unavailable",
            f"{operation_label} temporal lineage could not decode input or output audio.",
        )
    return input_info, output_info


def _replacement_mappings_match_decoded_audio(
    *,
    step: AudioEditStep,
    source_spans: list[tuple[float, float]],
    output_spans: list[tuple[float, float]],
    input_info: _DecodedAudioInfo,
    output_info: _DecodedAudioInfo,
) -> bool:
    if (
        not source_spans
        or len(source_spans) != len(output_spans)
        or not _spans_fit_duration(source_spans, input_info.duration_sec)
        or not _spans_fit_duration(output_spans, output_info.duration_sec)
    ):
        return False
    try:
        transform = TimelineStepTransform.replacement(
            step_id=f"{step.id}:reported-local-transform",
            operation=step.operation,
            input_revision=0,
            input_duration_sec=input_info.duration_sec,
            replacements=[
                ReplacementEdit(
                    input_interval=TimeInterval(start_sec=start, end_sec=end),
                    output_duration_sec=output_end - output_start,
                )
                for (start, end), (output_start, output_end) in zip(
                    source_spans,
                    output_spans,
                    strict=True,
                )
            ],
        )
    except ValueError:
        return False
    deterministic_outputs = [
        (piece.output_interval.start_sec, piece.output_interval.end_sec)
        for piece in transform.pieces
        if piece.kind == "replacement" and piece.output_interval is not None
    ]
    return (
        len(deterministic_outputs) == len(output_spans)
        and all(
            _bounds_close(declared, deterministic)
            for declared, deterministic in zip(
                output_spans,
                deterministic_outputs,
                strict=True,
            )
        )
        and transform.output_duration_sec is not None
        and _duration_matches_decoded_audio(transform.output_duration_sec, output_info)
    )


def _pause_output_support_spans(
    plan: dict[str, Any],
    *,
    count: int,
) -> list[tuple[float, float]]:
    smoothing = plan.get("smoothing_points")
    if isinstance(smoothing, list) and len(smoothing) == count:
        spans = [
            _single_span(item.get("regen_span_sec")) if isinstance(item, dict) else None
            for item in smoothing
        ]
        return [item for item in spans if item is not None] if all(spans) else []
    if count == 1:
        bounds = _single_span(plan.get("merged_regen_expected_span"))
        return [bounds] if bounds is not None else []
    return []


def _inverse_pause_insertion_boundary(
    value: float,
    *,
    anchors: list[float],
    durations: list[float],
    pause_intervals: list[tuple[float, float]],
) -> float | None:
    delta = 0.0
    for anchor, duration, pause in zip(
        anchors,
        durations,
        pause_intervals,
        strict=True,
    ):
        if value < pause[0] - 0.05:
            break
        if pause[0] + 0.05 < value < pause[1] - 0.05:
            return None
        if abs(value - pause[0]) <= 0.05 or abs(value - pause[1]) <= 0.05:
            return anchor
        if value >= pause[1] - 0.05:
            delta += duration
    mapped = value - delta
    return mapped if mapped >= 0.0 else None


def _float_list(value: Any, *, positive: bool) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    result: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if number < 0.0 or (positive and number <= 0.0):
            return []
        result.append(number)
    return result


def _single_span(value: Any) -> tuple[float, float] | None:
    spans = _span_list([value])
    return spans[0] if len(spans) == 1 else None


def _legacy_one_shot_temporal_evidence(
    *,
    generation_payload: dict[str, Any],
    root_path: str,
    input_info: _DecodedAudioInfo,
    output_info: _DecodedAudioInfo,
) -> _LegacyPipelineTemporalEvidence:
    """Validate the typed one-shot text-edit sidecar as one replacement step."""

    span_plan = generation_payload.get("span_plan")
    synthesis = generation_payload.get("synthesis")
    if not isinstance(span_plan, dict) or not isinstance(synthesis, dict):
        return _legacy_temporal_error(
            "legacy_pipeline_structural_steps_missing",
            "Text-edit metadata has neither synthesis steps nor a one-shot span plan.",
        )
    source_spans = _span_list(span_plan.get("source_spans"))
    target_spans = _span_list(span_plan.get("target_spans"))
    if not source_spans or len(source_spans) != len(target_spans):
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_structural_spans_invalid",
            "One-shot text-edit metadata lacks paired structural spans.",
        )
    source_duration = _positive_float(
        synthesis.get("source_context_duration")
    )
    target_duration = next(
        (
            value
            for key in (
                "target_timeline_duration",
                "target_condition_duration",
                "condition_duration",
            )
            if (value := _positive_float(synthesis.get(key))) is not None
        ),
        None,
    )
    if source_duration is None or target_duration is None:
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_duration_missing",
            "One-shot text-edit metadata lacks explicit source or target duration.",
        )
    if not _duration_matches_decoded_audio(source_duration, input_info):
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_input_duration_mismatch",
            "One-shot text-edit source duration disagrees with decoded input audio.",
        )
    if not _duration_matches_decoded_audio(target_duration, output_info):
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_output_duration_mismatch",
            "One-shot text-edit target duration disagrees with decoded output audio.",
        )
    if not _spans_fit_duration(source_spans, input_info.duration_sec) or not _spans_fit_duration(
        target_spans,
        output_info.duration_sec,
    ):
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_span_out_of_bounds",
            "One-shot text-edit structural spans exceed decoded audio bounds.",
        )
    try:
        step_transform = TimelineStepTransform.replacement(
            step_id="one-shot",
            operation="edit_data_pipeline.text_edit",
            input_revision=0,
            input_duration_sec=input_info.duration_sec,
            replacements=[
                ReplacementEdit(
                    input_interval=TimeInterval(start_sec=start, end_sec=end),
                    output_duration_sec=target_end - target_start,
                )
                for (start, end), (target_start, target_end) in zip(
                    source_spans,
                    target_spans,
                    strict=True,
                )
            ],
        )
    except ValueError as exc:
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_spans_not_composable",
            f"One-shot text-edit structural spans are not composable: {exc}",
        )
    deterministic_outputs = [
        (piece.output_interval.start_sec, piece.output_interval.end_sec)
        for piece in step_transform.pieces
        if piece.kind == "replacement" and piece.output_interval is not None
    ]
    if (
        len(deterministic_outputs) != len(target_spans)
        or any(
            not _bounds_close(declared, deterministic)
            for declared, deterministic in zip(
                target_spans,
                deterministic_outputs,
                strict=True,
            )
        )
        or step_transform.output_duration_sec is None
        or not _duration_matches_decoded_audio(
            step_transform.output_duration_sec,
            output_info,
        )
    ):
        return _legacy_temporal_error(
            "legacy_pipeline_one_shot_structural_output_mismatch",
            "One-shot spans do not reproduce the decoded output timeline.",
        )

    structural = tuple(
        (
            source,
            target,
            (
                f"{root_path}.span_plan.source_spans",
                f"{root_path}.span_plan.target_spans",
            ),
        )
        for source, target in zip(source_spans, target_spans, strict=True)
    )
    semantic: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]], ...
    ] = ()
    span_metadata = span_plan.get("metadata")
    raw_source_spans = (
        _span_list(span_metadata.get("raw_source_spans"))
        if isinstance(span_metadata, dict)
        else []
    )
    target_aligned_spans = _span_list(span_plan.get("target_aligned_spans"))
    if (
        raw_source_spans
        and len(raw_source_spans) == len(target_aligned_spans)
        and _spans_fit_duration(raw_source_spans, input_info.duration_sec)
        and _spans_fit_duration(target_aligned_spans, output_info.duration_sec)
    ):
        semantic = tuple(
            (
                source,
                target,
                (
                    f"{root_path}.span_plan.metadata.raw_source_spans",
                    f"{root_path}.span_plan.target_aligned_spans",
                ),
            )
            for source, target in zip(
                raw_source_spans,
                target_aligned_spans,
                strict=True,
            )
        )

    source_support = _span_list(synthesis.get("source_spans_sec"))
    target_support = _span_list(synthesis.get("target_edit_spans_sec"))
    if (
        source_support
        and len(source_support) == len(target_support)
        and _spans_fit_duration(source_support, input_info.duration_sec)
        and _spans_fit_duration(target_support, output_info.duration_sec)
    ):
        affected = tuple(
            (
                source,
                target,
                (
                    f"{root_path}.synthesis.source_spans_sec",
                    f"{root_path}.synthesis.target_edit_spans_sec",
                ),
            )
            for source, target in zip(source_support, target_support, strict=True)
        )
    else:
        affected = tuple(
            (source, target, (*provenance, "structural_mapping_conservative_support"))
            for source, target, provenance in structural
        )

    return _LegacyPipelineTemporalEvidence(
        present=True,
        structural_mappings=structural,
        semantic_anchors=semantic,
        affected_support=affected,
        identity_outside_affected_support=(
            synthesis.get("preserve_unmasked_waveform") is True
        ),
    )


def _legacy_pipeline_generation(
    output: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    metadata = output.get("metadata")
    generations = metadata.get("generations") if isinstance(metadata, dict) else None
    if not isinstance(generations, dict) or not generations:
        return None
    generation_id = str(output.get("generation_id") or "").strip()
    if generation_id and isinstance(generations.get(generation_id), dict):
        return generation_id, generations[generation_id]
    rows = [
        (str(key), value)
        for key, value in generations.items()
        if isinstance(value, dict)
    ]
    return rows[0] if len(rows) == 1 else None


def _legacy_step_target_duration(raw_step: dict[str, Any]) -> float | None:
    direct = _positive_float(raw_step.get("target_duration"))
    if direct is not None:
        return direct
    synthesis = raw_step.get("synthesis")
    if not isinstance(synthesis, dict):
        return None
    for key in ("target_condition_duration", "condition_duration"):
        value = _positive_float(synthesis.get(key))
        if value is not None:
            return value
    return None


def _legacy_temporal_error(code: str, reason: str) -> _LegacyPipelineTemporalEvidence:
    return _LegacyPipelineTemporalEvidence(
        present=True,
        reason_code=code,
        reason=reason,
    )


def _build_lineage(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    transform: AudioTemporalTransform,
    legacy_evidence: _LegacyPipelineTemporalEvidence,
    input_audio_paths: list[str],
    output_path: str,
    input_revision: int,
) -> AudioTemporalLineage:
    input_info = (
        _decoded_audio_info(input_audio_paths[0])
        if len(input_audio_paths) == 1
        else None
    )
    output_info = _decoded_audio_info(output_path)
    input_audio_refs = list(
        dict.fromkeys(
            audio_ref
            for path in input_audio_paths
            if (audio_ref := canonical_audio_ref_for_runtime_path(path, state))
        )
    )
    exact_preserved = (
        _decoded_audio_durations_match(input_info, output_info)
        if input_info is not None and output_info is not None
        else None
    )
    quality_close = (
        abs(input_info.duration_sec - output_info.duration_sec)
        <= _QUALITY_DURATION_TOLERANCE_SEC
        if input_info is not None and output_info is not None
        else None
    )
    if legacy_evidence.present:
        status: Literal["complete", "partial", "unavailable"] = (
            "complete" if legacy_evidence.structural_mappings else "unavailable"
        )
        return AudioTemporalLineage(
            step_id=step.id,
            operation=step.operation,
            status=status,
            edit_scope="local" if status == "complete" else "unknown",
            input_revision=input_revision,
            output_revision=input_revision + 1,
            input_audio_paths=copy.deepcopy(input_audio_paths),
            input_audio_refs=input_audio_refs,
            output_audio_path=output_path,
            input_duration_sec=input_info.duration_sec if input_info else None,
            output_duration_sec=output_info.duration_sec if output_info else None,
            exact_timeline_duration_preserved=exact_preserved,
            duration_change_within_quality_tolerance=quality_close,
            structural_mappings=_typed_lineage_mappings(
                step_id=step.id,
                role="structural_mapping",
                values=legacy_evidence.structural_mappings,
            ),
            semantic_anchors=_typed_lineage_mappings(
                step_id=step.id,
                role="semantic_anchor",
                values=legacy_evidence.semantic_anchors,
            ),
            affected_support=_typed_lineage_mappings(
                step_id=step.id,
                role="affected_support",
                values=legacy_evidence.affected_support,
            ),
            identity_outside_affected_support=(
                legacy_evidence.identity_outside_affected_support
                if status == "complete"
                else None
            ),
            derivation="legacy_pipeline_typed_sidecar",
            reason_code=legacy_evidence.reason_code,
            reason=legacy_evidence.reason,
        )

    structural_values = tuple(
        (
            (mapping.source_span.start_sec, mapping.source_span.end_sec),
            (mapping.output_span.start_sec, mapping.output_span.end_sec),
            ("result.audio_temporal_transform.timeline_mappings",),
        )
        for mapping in transform.timeline_mappings
    )
    status = (
        "complete"
        if transform.status == "complete"
        else "partial"
        if transform.status == "partial"
        else "unavailable"
    )
    reason_code = None if status == "complete" else "temporal_transform_incomplete"
    reason = (
        None
        if status == "complete"
        else "The compatibility temporal transform did not prove complete lineage."
    )
    return AudioTemporalLineage(
        step_id=step.id,
        operation=step.operation,
        status=status,
        edit_scope=transform.edit_scope,
        input_revision=input_revision,
        output_revision=input_revision + 1,
        input_audio_paths=copy.deepcopy(input_audio_paths),
        input_audio_refs=input_audio_refs,
        output_audio_path=output_path,
        input_duration_sec=input_info.duration_sec if input_info else None,
        output_duration_sec=output_info.duration_sec if output_info else None,
        exact_timeline_duration_preserved=exact_preserved,
        duration_change_within_quality_tolerance=quality_close,
        structural_mappings=_typed_lineage_mappings(
            step_id=step.id,
            role="structural_mapping",
            values=structural_values,
        ),
        affected_support=_typed_lineage_mappings(
            step_id=step.id,
            role="affected_support",
            values=tuple(
                (
                    source,
                    target,
                    (*provenance, "structural_mapping_conservative_support"),
                )
                for source, target, provenance in structural_values
            ),
        ),
        identity_outside_affected_support=(
            transform.identity_outside_regions
            if transform.edit_scope == "local"
            else None
        ),
        derivation="compatibility_audio_temporal_transform",
        reason_code=reason_code,
        reason=reason,
    )


def _typed_lineage_mappings(
    *,
    step_id: str,
    role: Literal["structural_mapping", "semantic_anchor", "affected_support"],
    values: tuple[
        tuple[tuple[float, float], tuple[float, float], tuple[str, ...]], ...
    ],
) -> list[AudioTemporalLineageMapping]:
    return [
        AudioTemporalLineageMapping(
            mapping_id=f"{step_id}:{role}:{index:03d}",
            role=role,
            source_span=AudioTemporalSpan(start_sec=source[0], end_sec=source[1]),
            output_span=AudioTemporalSpan(start_sec=target[0], end_sec=target[1]),
            provenance_paths=list(provenance),
        )
        for index, (source, target, provenance) in enumerate(values)
    ]


def _map_current_interval_envelope(
    ledger: TimelineLedger,
    bounds: tuple[float, float],
) -> tuple[float, float] | None:
    try:
        result = ledger.map_current_interval(
            TimeInterval(start_sec=bounds[0], end_sec=bounds[1])
        )
    except ValueError:
        return None
    envelope = result.target_envelope
    if result.status != "complete" or envelope is None:
        return None
    return envelope.start_sec, envelope.end_sec


def _map_base_interval_envelope(
    ledger: TimelineLedger,
    bounds: tuple[float, float],
) -> tuple[float, float] | None:
    try:
        result = ledger.map_base_interval(
            TimeInterval(start_sec=bounds[0], end_sec=bounds[1])
        )
    except ValueError:
        return None
    envelope = result.target_envelope
    if result.status != "complete" or envelope is None:
        return None
    return envelope.start_sec, envelope.end_sec


def _build_transform(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    output: dict[str, Any],
    input_audio_paths: list[str],
    output_path: str,
    legacy_evidence: _LegacyPipelineTemporalEvidence,
) -> AudioTemporalTransform:
    input_audio_refs = list(
        dict.fromkeys(
            audio_ref
            for path in input_audio_paths
            if (
                audio_ref := canonical_audio_ref_for_runtime_path(
                    path,
                    state,
                )
            )
        )
    )
    if len(input_audio_paths) == 1 and source_aligned_audio_covers(
        output.get("source_time_axis"), input_audio_paths[0], [output_path],
    ):
        return AudioTemporalTransform(
            step_id=step.id, operation=step.operation, status="complete", edit_scope="global",
            input_audio_paths=copy.deepcopy(input_audio_paths), input_audio_refs=input_audio_refs,
            output_audio_path=output_path, timeline_duration_preserved=True,
            derivation="bound_source_aligned_signal_processing_contract",
        )
    if legacy_evidence.present:
        if legacy_evidence.structural_mappings:
            return _local_transform(
                step=step,
                input_audio_paths=input_audio_paths,
                input_audio_refs=input_audio_refs,
                output_path=output_path,
                mappings=[
                    (source, target)
                    for source, target, _ in legacy_evidence.structural_mappings
                ],
                timeline_duration_preserved=_durations_match(
                    input_audio_paths,
                    output_path,
                ),
                derivation="legacy_pipeline_explicit_structural_spans",
                identity_outside_regions=(
                    legacy_evidence.identity_outside_affected_support is True
                ),
            )
        return AudioTemporalTransform(
            step_id=step.id,
            operation=step.operation,
            status="unavailable",
            edit_scope="unknown",
            input_audio_paths=copy.deepcopy(input_audio_paths),
            input_audio_refs=input_audio_refs,
            output_audio_path=output_path,
            derivation=(
                legacy_evidence.reason_code
                or "legacy_pipeline_structural_mapping_unavailable"
            ),
        )
    paired = _paired_regions(output.get("source_spans"), output.get("target_spans"))
    if paired:
        return _local_transform(
            step=step,
            input_audio_paths=input_audio_paths,
            input_audio_refs=input_audio_refs,
            output_path=output_path,
            mappings=paired,
            timeline_duration_preserved=_durations_match(
                input_audio_paths,
                output_path,
            ),
            derivation="handler_source_target_spans",
            identity_outside_regions=output.get("identity_outside_regions") is True,
        )

    selected = _span_list(output.get("selected_spans"))
    if selected and output.get("timeline_duration_preserved") is True:
        return _local_transform(
            step=step,
            input_audio_paths=input_audio_paths,
            input_audio_refs=input_audio_refs,
            output_path=output_path,
            mappings=[(span, span) for span in selected],
            timeline_duration_preserved=True,
            derivation="handler_selected_spans_identity_timeline",
        )

    lexical_regions = _span_list(output.get("source_target_spans"))
    if lexical_regions and _durations_match(input_audio_paths, output_path):
        return _local_transform(
            step=step,
            input_audio_paths=input_audio_paths,
            input_audio_refs=input_audio_refs,
            output_path=output_path,
            mappings=[(span, span) for span in lexical_regions],
            timeline_duration_preserved=True,
            derivation="handler_source_target_regions_identity_timeline",
            identity_outside_regions=output.get("identity_outside_regions") is True,
        )

    if step.operation == "artifact_ops.extract_spans" and len(selected) == 1:
        output_duration = _audio_duration(output_path)
        if output_duration is not None and output_duration > 0:
            return _local_transform(
                step=step,
                input_audio_paths=input_audio_paths,
                input_audio_refs=input_audio_refs,
                output_path=output_path,
                mappings=[(selected[0], (0.0, output_duration))],
                timeline_duration_preserved=False,
                identity_outside_regions=False,
                derivation="single_extracted_span_compacted_to_output",
            )

    if step.operation in _GLOBAL_WHOLE_INPUT_OPERATIONS and input_audio_paths:
        return AudioTemporalTransform(
            step_id=step.id,
            operation=step.operation,
            status="complete",
            edit_scope="global",
            input_audio_paths=copy.deepcopy(input_audio_paths),
            input_audio_refs=input_audio_refs,
            output_audio_path=output_path,
            timeline_duration_preserved=_durations_match(
                input_audio_paths,
                output_path,
            ),
            derivation="registered_whole_input_operation",
        )

    return AudioTemporalTransform(
        step_id=step.id,
        operation=step.operation,
        status="unavailable",
        edit_scope="unknown",
        input_audio_paths=copy.deepcopy(input_audio_paths),
        input_audio_refs=input_audio_refs,
        output_audio_path=output_path,
        derivation="no_complete_temporal_mapping_contract",
    )


def _local_transform(
    *,
    step: AudioEditStep,
    input_audio_paths: list[str],
    input_audio_refs: list[str],
    output_path: str,
    mappings: list[tuple[tuple[float, float], tuple[float, float]]],
    timeline_duration_preserved: bool,
    derivation: str,
    identity_outside_regions: bool = False,
) -> AudioTemporalTransform:
    source_regions = [
        AudioTemporalSpan(start_sec=start, end_sec=end)
        for start, end in (item[0] for item in mappings)
    ]
    output_regions = [
        AudioTemporalSpan(start_sec=start, end_sec=end)
        for start, end in (item[1] for item in mappings)
    ]
    return AudioTemporalTransform(
        step_id=step.id,
        operation=step.operation,
        status="complete" if input_audio_paths else "unavailable",
        edit_scope="local" if input_audio_paths else "unknown",
        input_audio_paths=copy.deepcopy(input_audio_paths),
        input_audio_refs=copy.deepcopy(input_audio_refs),
        output_audio_path=output_path,
        source_regions=source_regions if input_audio_paths else [],
        output_regions=output_regions if input_audio_paths else [],
        timeline_mappings=(
            [
                AudioTemporalSpanMapping(
                    source_span=source_span,
                    output_span=output_span,
                )
                for source_span, output_span in zip(
                    source_regions,
                    output_regions,
                    strict=True,
                )
            ]
            if input_audio_paths
            else []
        ),
        identity_outside_regions=(
            identity_outside_regions if input_audio_paths else None
        ),
        timeline_duration_preserved=timeline_duration_preserved,
        derivation=derivation,
    )


def _paired_regions(
    source_value: Any,
    output_value: Any,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    source = _span_list(source_value)
    target = _span_list(output_value)
    if not source or len(source) != len(target):
        return []
    return list(zip(source, target, strict=True))


def _span_list(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    spans: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            start_value = item.get("start_sec")
            end_value = item.get("end_sec")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start_value, end_value = item[0], item[1]
        else:
            return []
        try:
            start = float(start_value)
            end = float(end_value)
        except (TypeError, ValueError):
            return []
        if start < 0 or end <= start:
            return []
        spans.append((start, end))
    return spans


def _durations_match(input_audio_paths: list[str], output_path: str) -> bool:
    if len(input_audio_paths) != 1:
        return False
    return _decoded_audio_durations_match(
        _decoded_audio_info(input_audio_paths[0]),
        _decoded_audio_info(output_path),
    )


def _decoded_audio_durations_match(
    left: _DecodedAudioInfo | None,
    right: _DecodedAudioInfo | None,
) -> bool:
    if left is None or right is None:
        return False
    tolerance = max(
        1.0 / left.sample_rate_hz,
        1.0 / right.sample_rate_hz,
    ) + 1e-9
    return abs(left.duration_sec - right.duration_sec) <= tolerance


def _duration_matches_decoded_audio(
    declared_duration_sec: float,
    decoded: _DecodedAudioInfo,
) -> bool:
    return _values_close(
        declared_duration_sec,
        decoded.duration_sec,
        tolerance_sec=(1.0 / decoded.sample_rate_hz) + 1e-9,
    )


def _decoded_audio_info(path: str) -> _DecodedAudioInfo | None:
    try:
        info = sf.info(str(Path(path).expanduser()))
    except (OSError, RuntimeError, sf.SoundFileError):
        return None
    if info.frames <= 0 or info.samplerate <= 0:
        return None
    return _DecodedAudioInfo(
        duration_sec=float(info.frames / info.samplerate),
        sample_rate_hz=int(info.samplerate),
    )


def _audio_duration(path: str) -> float | None:
    info = _decoded_audio_info(path)
    return info.duration_sec if info is not None else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _values_close(left: float, right: float, *, tolerance_sec: float) -> bool:
    return abs(left - right) <= tolerance_sec


def _bounds_close(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    return _values_close(
        left[0],
        right[0],
        tolerance_sec=_STRUCTURAL_TIME_TOLERANCE_SEC,
    ) and _values_close(
        left[1],
        right[1],
        tolerance_sec=_STRUCTURAL_TIME_TOLERANCE_SEC,
    )


def _spans_fit_duration(
    spans: list[tuple[float, float]],
    duration_sec: float,
) -> bool:
    return all(end <= duration_sec + _STRUCTURAL_TIME_TOLERANCE_SEC for _, end in spans)


def _output_audio_path(output: dict[str, Any]) -> str | None:
    for key in (
        "output_audio_path",
        "target_audio_path",
        "output_path",
        "generated_audio_path",
    ):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
