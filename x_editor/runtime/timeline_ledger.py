"""Composable, auditable temporal lineage from base audio to current audio."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import ConfigDict, Field, computed_field, model_validator

from x_editor.core.schemas import StrictModel

_EPSILON = 1e-7
_DURATION_EPSILON_SEC = 1e-4

TransformStatus = Literal["complete", "unsupported", "unavailable"]
QueryStatus = Literal["complete", "partial", "unavailable"]
PieceKind = Literal[
    "identity",
    "replacement",
    "insertion",
    "deletion",
    "time_stretch",
]
Correspondence = Literal["exact", "boundary_linear", "linear", "none"]


class _FrozenStrictModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class TimebaseRef(_FrozenStrictModel):
    """A versioned timebase; base is immutable and current advances per step."""

    kind: Literal["base", "current"]
    revision: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_base_revision(self) -> TimebaseRef:
        if self.kind == "base" and self.revision != 0:
            raise ValueError("base timebase revision must be 0")
        return self


class TimeInterval(_FrozenStrictModel):
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> TimeInterval:
        if self.end_sec <= self.start_sec:
            raise ValueError("interval end_sec must exceed start_sec")
        return self

    @computed_field
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class ReplacementEdit(_FrozenStrictModel):
    input_interval: TimeInterval
    output_duration_sec: float = Field(gt=0.0)


class InsertionEdit(_FrozenStrictModel):
    at_sec: float = Field(ge=0.0)
    duration_sec: float = Field(gt=0.0)


class DeletionEdit(_FrozenStrictModel):
    input_interval: TimeInterval


class TimeStretchEdit(_FrozenStrictModel):
    input_interval: TimeInterval
    output_duration_sec: float = Field(gt=0.0)


class StepTimelinePiece(_FrozenStrictModel):
    """One monotonic piece of a current@N to current@N+1 transform."""

    piece_id: str = Field(min_length=1)
    kind: PieceKind
    input_interval: TimeInterval | None = None
    output_interval: TimeInterval | None = None
    input_anchor_sec: float | None = Field(default=None, ge=0.0)
    output_anchor_sec: float | None = Field(default=None, ge=0.0)
    correspondence: Correspondence

    @model_validator(mode="after")
    def validate_piece_contract(self) -> StepTimelinePiece:
        if self.kind == "insertion":
            if self.input_interval is not None or self.output_interval is None:
                raise ValueError("insertion requires only an output interval")
            if self.input_anchor_sec is None or self.output_anchor_sec is not None:
                raise ValueError("insertion requires input_anchor_sec only")
            if self.correspondence != "none":
                raise ValueError("insertion correspondence must be none")
            return self
        if self.kind == "deletion":
            if self.input_interval is None or self.output_interval is not None:
                raise ValueError("deletion requires only an input interval")
            if self.output_anchor_sec is None or self.input_anchor_sec is not None:
                raise ValueError("deletion requires output_anchor_sec only")
            if self.correspondence != "none":
                raise ValueError("deletion correspondence must be none")
            return self
        if self.input_interval is None or self.output_interval is None:
            raise ValueError(f"{self.kind} requires input and output intervals")
        if self.input_anchor_sec is not None or self.output_anchor_sec is not None:
            raise ValueError(f"{self.kind} cannot declare an anchor")
        expected = {
            "identity": "exact",
            "replacement": "boundary_linear",
            "time_stretch": "linear",
        }[self.kind]
        if self.correspondence != expected:
            raise ValueError(f"{self.kind} correspondence must be {expected}")
        if self.kind == "identity" and not _close(
            self.input_interval.duration_sec,
            self.output_interval.duration_sec,
        ):
            raise ValueError("identity piece must preserve duration")
        return self


class TimelineStepTransform(_FrozenStrictModel):
    """One executor step's explicit current@N to current@N+1 transform."""

    schema_version: Literal["x-editor.timeline-step-transform/v1"] = (
        "x-editor.timeline-step-transform/v1"
    )
    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    status: TransformStatus
    input_timebase: TimebaseRef
    output_timebase: TimebaseRef
    input_duration_sec: float | None = Field(default=None, gt=0.0)
    output_duration_sec: float | None = Field(default=None, gt=0.0)
    pieces: tuple[StepTimelinePiece, ...] = ()
    reason_code: str | None = None
    reason: str | None = None

    @computed_field
    @property
    def duration_delta_sec(self) -> float | None:
        if self.input_duration_sec is None or self.output_duration_sec is None:
            return None
        return self.output_duration_sec - self.input_duration_sec

    @model_validator(mode="after")
    def validate_transform(self) -> TimelineStepTransform:
        _validate_step_timebases(self.input_timebase, self.output_timebase)
        if self.status != "complete":
            if self.pieces:
                raise ValueError(f"{self.status} transform cannot expose timeline pieces")
            if not self.reason_code or not self.reason:
                raise ValueError(f"{self.status} transform requires reason_code and reason")
            return self
        if self.reason_code is not None or self.reason is not None:
            raise ValueError("complete transform cannot declare a failure reason")
        if self.input_duration_sec is None or self.output_duration_sec is None:
            raise ValueError("complete transform requires input and output durations")
        if not self.pieces:
            raise ValueError("complete transform requires timeline pieces")
        _validate_unique_piece_ids(self.pieces)
        _validate_coverage(
            [piece.input_interval for piece in self.pieces if piece.input_interval],
            self.input_duration_sec,
            "step input",
        )
        _validate_coverage(
            [piece.output_interval for piece in self.pieces if piece.output_interval],
            self.output_duration_sec,
            "step output",
        )
        _validate_monotonic_step(self.pieces)
        return self

    @classmethod
    def identity(
        cls,
        *,
        step_id: str,
        input_revision: int,
        duration_sec: float,
        operation: str = "identity",
    ) -> TimelineStepTransform:
        interval = TimeInterval(start_sec=0.0, end_sec=duration_sec)
        return cls(
            step_id=step_id,
            operation=operation,
            status="complete",
            input_timebase=_current_timebase(input_revision),
            output_timebase=_current_timebase(input_revision + 1),
            input_duration_sec=duration_sec,
            output_duration_sec=duration_sec,
            pieces=(
                StepTimelinePiece(
                    piece_id=f"{step_id}:identity:0",
                    kind="identity",
                    input_interval=interval,
                    output_interval=interval,
                    correspondence="exact",
                ),
            ),
        )

    @classmethod
    def replacement(
        cls,
        *,
        step_id: str,
        input_revision: int,
        input_duration_sec: float,
        replacements: Sequence[ReplacementEdit],
        operation: str = "replacement",
    ) -> TimelineStepTransform:
        edits = [(edit.input_interval, edit.output_duration_sec) for edit in replacements]
        return cls._ranged_edits(
            step_id=step_id,
            operation=operation,
            input_revision=input_revision,
            input_duration_sec=input_duration_sec,
            edits=edits,
            edit_kind="replacement",
            correspondence="boundary_linear",
        )

    @classmethod
    def insertion(
        cls,
        *,
        step_id: str,
        input_revision: int,
        input_duration_sec: float,
        insertions: Sequence[InsertionEdit],
        operation: str = "insertion",
    ) -> TimelineStepTransform:
        if not insertions:
            raise ValueError("insertion transform requires at least one edit")
        ordered = sorted(insertions, key=lambda edit: edit.at_sec)
        if ordered[-1].at_sec > input_duration_sec + _EPSILON:
            raise ValueError("insertion anchor exceeds input duration")
        pieces: list[StepTimelinePiece] = []
        input_cursor = 0.0
        output_cursor = 0.0
        for index, edit in enumerate(ordered):
            if edit.at_sec < input_cursor - _EPSILON:
                raise ValueError("insertion anchors must be monotonic")
            output_cursor = _append_identity_piece(
                pieces,
                step_id=step_id,
                index=len(pieces),
                input_start=input_cursor,
                input_end=edit.at_sec,
                output_start=output_cursor,
            )
            input_cursor = edit.at_sec
            output_interval = TimeInterval(
                start_sec=output_cursor,
                end_sec=output_cursor + edit.duration_sec,
            )
            pieces.append(
                StepTimelinePiece(
                    piece_id=f"{step_id}:insertion:{index}",
                    kind="insertion",
                    output_interval=output_interval,
                    input_anchor_sec=edit.at_sec,
                    correspondence="none",
                )
            )
            output_cursor = output_interval.end_sec
        output_cursor = _append_identity_piece(
            pieces,
            step_id=step_id,
            index=len(pieces),
            input_start=input_cursor,
            input_end=input_duration_sec,
            output_start=output_cursor,
        )
        return cls(
            step_id=step_id,
            operation=operation,
            status="complete",
            input_timebase=_current_timebase(input_revision),
            output_timebase=_current_timebase(input_revision + 1),
            input_duration_sec=input_duration_sec,
            output_duration_sec=output_cursor,
            pieces=tuple(pieces),
        )

    @classmethod
    def deletion(
        cls,
        *,
        step_id: str,
        input_revision: int,
        input_duration_sec: float,
        deletions: Sequence[DeletionEdit],
        operation: str = "deletion",
    ) -> TimelineStepTransform:
        edits = [(edit.input_interval, 0.0) for edit in deletions]
        return cls._ranged_edits(
            step_id=step_id,
            operation=operation,
            input_revision=input_revision,
            input_duration_sec=input_duration_sec,
            edits=edits,
            edit_kind="deletion",
            correspondence="none",
        )

    @classmethod
    def time_stretch(
        cls,
        *,
        step_id: str,
        input_revision: int,
        input_duration_sec: float,
        stretches: Sequence[TimeStretchEdit],
        operation: str = "time_stretch",
    ) -> TimelineStepTransform:
        edits = [(edit.input_interval, edit.output_duration_sec) for edit in stretches]
        return cls._ranged_edits(
            step_id=step_id,
            operation=operation,
            input_revision=input_revision,
            input_duration_sec=input_duration_sec,
            edits=edits,
            edit_kind="time_stretch",
            correspondence="linear",
        )

    @classmethod
    def unsupported(
        cls,
        *,
        step_id: str,
        operation: str,
        input_revision: int,
        input_duration_sec: float | None,
        output_duration_sec: float | None,
        reason_code: str = "non_monotonic_or_multi_source_operation",
        reason: str = (
            "This operation cannot be represented as a monotonic single-input timeline."
        ),
    ) -> TimelineStepTransform:
        return cls(
            step_id=step_id,
            operation=operation,
            status="unsupported",
            input_timebase=_current_timebase(input_revision),
            output_timebase=_current_timebase(input_revision + 1),
            input_duration_sec=input_duration_sec,
            output_duration_sec=output_duration_sec,
            reason_code=reason_code,
            reason=reason,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        step_id: str,
        operation: str,
        input_revision: int,
        input_duration_sec: float | None,
        output_duration_sec: float | None,
        reason_code: str = "incomplete_temporal_evidence",
        reason: str = "Executor did not provide enough evidence for a safe timeline mapping.",
    ) -> TimelineStepTransform:
        return cls(
            step_id=step_id,
            operation=operation,
            status="unavailable",
            input_timebase=_current_timebase(input_revision),
            output_timebase=_current_timebase(input_revision + 1),
            input_duration_sec=input_duration_sec,
            output_duration_sec=output_duration_sec,
            reason_code=reason_code,
            reason=reason,
        )

    @classmethod
    def _ranged_edits(
        cls,
        *,
        step_id: str,
        operation: str,
        input_revision: int,
        input_duration_sec: float,
        edits: Sequence[tuple[TimeInterval, float]],
        edit_kind: Literal["replacement", "deletion", "time_stretch"],
        correspondence: Correspondence,
    ) -> TimelineStepTransform:
        if not edits:
            raise ValueError(f"{edit_kind} transform requires at least one edit")
        ordered = sorted(edits, key=lambda item: item[0].start_sec)
        pieces: list[StepTimelinePiece] = []
        input_cursor = 0.0
        output_cursor = 0.0
        for edit_index, (interval, output_duration) in enumerate(ordered):
            if interval.end_sec > input_duration_sec + _EPSILON:
                raise ValueError(f"{edit_kind} interval exceeds input duration")
            if interval.start_sec < input_cursor - _EPSILON:
                raise ValueError(f"{edit_kind} intervals cannot overlap")
            output_cursor = _append_identity_piece(
                pieces,
                step_id=step_id,
                index=len(pieces),
                input_start=input_cursor,
                input_end=interval.start_sec,
                output_start=output_cursor,
            )
            if edit_kind == "deletion":
                pieces.append(
                    StepTimelinePiece(
                        piece_id=f"{step_id}:deletion:{edit_index}",
                        kind="deletion",
                        input_interval=interval,
                        output_anchor_sec=output_cursor,
                        correspondence="none",
                    )
                )
            else:
                output_interval = TimeInterval(
                    start_sec=output_cursor,
                    end_sec=output_cursor + output_duration,
                )
                pieces.append(
                    StepTimelinePiece(
                        piece_id=f"{step_id}:{edit_kind}:{edit_index}",
                        kind=edit_kind,
                        input_interval=interval,
                        output_interval=output_interval,
                        correspondence=correspondence,
                    )
                )
                output_cursor = output_interval.end_sec
            input_cursor = interval.end_sec
        output_cursor = _append_identity_piece(
            pieces,
            step_id=step_id,
            index=len(pieces),
            input_start=input_cursor,
            input_end=input_duration_sec,
            output_start=output_cursor,
        )
        return cls(
            step_id=step_id,
            operation=operation,
            status="complete",
            input_timebase=_current_timebase(input_revision),
            output_timebase=_current_timebase(input_revision + 1),
            input_duration_sec=input_duration_sec,
            output_duration_sec=output_cursor,
            pieces=tuple(pieces),
        )


class LedgerTimelinePiece(_FrozenStrictModel):
    """One composed base@0 to current@N temporal relationship."""

    piece_id: str = Field(min_length=1)
    kind: PieceKind
    base_interval: TimeInterval | None = None
    current_interval: TimeInterval | None = None
    correspondence: Correspondence
    provenance_step_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_piece(self) -> LedgerTimelinePiece:
        if self.base_interval is None and self.current_interval is None:
            raise ValueError("ledger piece must exist in at least one timebase")
        if self.base_interval is None:
            if self.kind != "insertion" or self.correspondence != "none":
                raise ValueError("piece without base interval must be an insertion")
        elif self.current_interval is None:
            if self.kind != "deletion" or self.correspondence != "none":
                raise ValueError("piece without current interval must be a deletion")
        elif self.kind in {"insertion", "deletion"}:
            raise ValueError("mapped ledger piece cannot be insertion or deletion")
        return self


class MappingFragment(_FrozenStrictModel):
    piece_id: str = Field(min_length=1)
    from_interval: TimeInterval
    to_interval: TimeInterval | None = None
    relation: PieceKind
    correspondence: Correspondence
    unmapped_reason: (
        Literal[
            "base_interval_deleted",
            "current_interval_has_no_base_source",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_mapping(self) -> MappingFragment:
        if (self.to_interval is None) != (self.unmapped_reason is not None):
            raise ValueError("unmapped_reason is required exactly when no target exists")
        return self


class IntervalMappingResult(_FrozenStrictModel):
    schema_version: Literal["x-editor.timeline-interval-mapping/v1"] = (
        "x-editor.timeline-interval-mapping/v1"
    )
    status: QueryStatus
    from_timebase: TimebaseRef
    to_timebase: TimebaseRef
    requested_interval: TimeInterval
    fragments: tuple[MappingFragment, ...] = ()
    reason_code: str | None = None
    reason: str | None = None

    @computed_field
    @property
    def mapped_intervals(self) -> tuple[TimeInterval, ...]:
        return tuple(
            fragment.to_interval for fragment in self.fragments if fragment.to_interval is not None
        )

    @computed_field
    @property
    def target_envelope(self) -> TimeInterval | None:
        mapped = self.mapped_intervals
        if not mapped:
            return None
        return TimeInterval(
            start_sec=min(interval.start_sec for interval in mapped),
            end_sec=max(interval.end_sec for interval in mapped),
        )


class TimelineLedger(_FrozenStrictModel):
    """Immutable composed timeline from the original audio to one current revision."""

    schema_version: Literal["x-editor.timeline-ledger/v1"] = "x-editor.timeline-ledger/v1"
    status: TransformStatus
    base_timebase: TimebaseRef = TimebaseRef(kind="base", revision=0)
    current_timebase: TimebaseRef
    base_duration_sec: float = Field(gt=0.0)
    current_duration_sec: float | None = Field(default=None, gt=0.0)
    base_audio_ref: str | None = None
    pieces: tuple[LedgerTimelinePiece, ...] = ()
    steps: tuple[TimelineStepTransform, ...] = ()
    reason_code: str | None = None
    reason: str | None = None

    @computed_field
    @property
    def duration_delta_sec(self) -> float | None:
        if self.current_duration_sec is None:
            return None
        return self.current_duration_sec - self.base_duration_sec

    @model_validator(mode="after")
    def validate_ledger(self) -> TimelineLedger:
        if self.base_timebase.kind != "base" or self.current_timebase.kind != "current":
            raise ValueError("ledger requires explicit base and current timebases")
        if self.status != "complete":
            if self.pieces:
                raise ValueError(f"{self.status} ledger cannot expose timeline pieces")
            if not self.reason_code or not self.reason:
                raise ValueError(f"{self.status} ledger requires reason_code and reason")
            return self
        if self.current_duration_sec is None:
            raise ValueError("complete ledger requires current_duration_sec")
        if self.reason_code is not None or self.reason is not None:
            raise ValueError("complete ledger cannot declare a failure reason")
        _validate_unique_piece_ids(self.pieces)
        _validate_coverage(
            [piece.base_interval for piece in self.pieces if piece.base_interval],
            self.base_duration_sec,
            "ledger base",
        )
        _validate_coverage(
            [piece.current_interval for piece in self.pieces if piece.current_interval],
            self.current_duration_sec,
            "ledger current",
        )
        _validate_monotonic_ledger(self.pieces)
        return self

    @classmethod
    def identity(
        cls,
        *,
        base_duration_sec: float,
        base_audio_ref: str | None = None,
    ) -> TimelineLedger:
        interval = TimeInterval(start_sec=0.0, end_sec=base_duration_sec)
        return cls(
            status="complete",
            current_timebase=_current_timebase(0),
            base_duration_sec=base_duration_sec,
            current_duration_sec=base_duration_sec,
            base_audio_ref=base_audio_ref,
            pieces=(
                LedgerTimelinePiece(
                    piece_id="base:identity:0",
                    kind="identity",
                    base_interval=interval,
                    current_interval=interval,
                    correspondence="exact",
                ),
            ),
        )

    def compose(self, step: TimelineStepTransform) -> TimelineLedger:
        """Return a new ledger with one step composed after the current revision."""

        if step.input_timebase != self.current_timebase:
            raise ValueError(
                "step input timebase does not match ledger current timebase: "
                f"{step.input_timebase} != {self.current_timebase}"
            )
        if (
            self.current_duration_sec is not None
            and step.input_duration_sec is not None
            and not _duration_close(
                self.current_duration_sec,
                step.input_duration_sec,
            )
        ):
            raise ValueError("step input duration does not match ledger current duration")
        if self.status != "complete":
            return self._failed_composition(
                step,
                status=self.status,
                reason_code="upstream_timeline_not_complete",
                reason="A prior timeline step was not safely composable.",
            )
        if step.status != "complete":
            return self._failed_composition(
                step,
                status=step.status,
                reason_code=step.reason_code or "step_timeline_not_complete",
                reason=step.reason or "The step did not provide a safe temporal transform.",
            )

        composed: list[LedgerTimelinePiece] = []
        input_pieces = [piece for piece in step.pieces if piece.input_interval is not None]
        for ledger_piece in self.pieces:
            if ledger_piece.current_interval is None:
                composed.append(ledger_piece)
                continue
            ledger_current_interval = _snap_duration_endpoint(
                ledger_piece.current_interval,
                declared_duration_sec=self.current_duration_sec,
                observed_duration_sec=step.input_duration_sec,
            )
            for step_piece in input_pieces:
                overlap = _intersection(
                    ledger_current_interval,
                    step_piece.input_interval,
                )
                if overlap is None:
                    continue
                base_interval = (
                    _map_linear(
                        overlap,
                        ledger_current_interval,
                        ledger_piece.base_interval,
                    )
                    if ledger_piece.base_interval is not None
                    else None
                )
                current_interval = (
                    _map_linear(overlap, step_piece.input_interval, step_piece.output_interval)
                    if step_piece.output_interval is not None
                    else None
                )
                if base_interval is None and current_interval is None:
                    continue
                kind, correspondence = _composed_semantics(ledger_piece, step_piece)
                composed.append(
                    LedgerTimelinePiece(
                        piece_id="pending",
                        kind=kind,
                        base_interval=base_interval,
                        current_interval=current_interval,
                        correspondence=correspondence,
                        provenance_step_ids=_append_unique(
                            ledger_piece.provenance_step_ids,
                            step.step_id,
                        ),
                    )
                )
        for step_piece in step.pieces:
            if step_piece.kind != "insertion":
                continue
            composed.append(
                LedgerTimelinePiece(
                    piece_id="pending",
                    kind="insertion",
                    current_interval=step_piece.output_interval,
                    correspondence="none",
                    provenance_step_ids=(step.step_id,),
                )
            )
        normalized = _normalize_ledger_pieces(composed, step.output_timebase.revision)
        return TimelineLedger(
            status="complete",
            current_timebase=step.output_timebase,
            base_duration_sec=self.base_duration_sec,
            current_duration_sec=step.output_duration_sec,
            base_audio_ref=self.base_audio_ref,
            pieces=normalized,
            steps=(*self.steps, step),
        )

    def map_base_interval(self, interval: TimeInterval) -> IntervalMappingResult:
        """Map a base@0 interval to current@N, preserving deleted fragments."""

        return self._map_interval(
            interval,
            forward=True,
            axis_duration=self.base_duration_sec,
        )

    def map_current_interval(self, interval: TimeInterval) -> IntervalMappingResult:
        """Partially invert a current@N interval back to base@0."""

        if self.current_duration_sec is None:
            return self._unavailable_query(
                interval,
                forward=False,
                reason_code="current_duration_unavailable",
                reason="Current duration is unavailable because the ledger is incomplete.",
            )
        return self._map_interval(
            interval,
            forward=False,
            axis_duration=self.current_duration_sec,
        )

    def _map_interval(
        self,
        interval: TimeInterval,
        *,
        forward: bool,
        axis_duration: float,
    ) -> IntervalMappingResult:
        if interval.end_sec > axis_duration + _EPSILON:
            raise ValueError("requested interval exceeds the selected timebase duration")
        if self.status != "complete":
            return self._unavailable_query(
                interval,
                forward=forward,
                reason_code="timeline_not_complete",
                reason=self.reason or "Timeline mapping is not complete.",
            )
        fragments: list[MappingFragment] = []
        for piece in self.pieces:
            from_interval = piece.base_interval if forward else piece.current_interval
            to_interval = piece.current_interval if forward else piece.base_interval
            if from_interval is None:
                continue
            overlap = _intersection(interval, from_interval)
            if overlap is None:
                continue
            mapped = (
                _map_linear(overlap, from_interval, to_interval)
                if to_interval is not None
                else None
            )
            fragments.append(
                MappingFragment(
                    piece_id=piece.piece_id,
                    from_interval=overlap,
                    to_interval=mapped,
                    relation=piece.kind,
                    correspondence=piece.correspondence,
                    unmapped_reason=(
                        None
                        if mapped is not None
                        else (
                            "base_interval_deleted"
                            if forward
                            else "current_interval_has_no_base_source"
                        )
                    ),
                )
            )
        fragments.sort(key=lambda item: item.from_interval.start_sec)
        mapped_count = sum(fragment.to_interval is not None for fragment in fragments)
        if mapped_count == len(fragments) and fragments:
            status: QueryStatus = "complete"
            reason_code = None
            reason = None
        elif mapped_count:
            status = "partial"
            reason_code = "interval_partially_unmapped"
            reason = "Part of the requested interval has no counterpart in the target timebase."
        else:
            status = "unavailable"
            reason_code = "interval_has_no_target_mapping"
            reason = "The requested interval has no counterpart in the target timebase."
        return IntervalMappingResult(
            status=status,
            from_timebase=self.base_timebase if forward else self.current_timebase,
            to_timebase=self.current_timebase if forward else self.base_timebase,
            requested_interval=interval,
            fragments=tuple(fragments),
            reason_code=reason_code,
            reason=reason,
        )

    def _unavailable_query(
        self,
        interval: TimeInterval,
        *,
        forward: bool,
        reason_code: str,
        reason: str,
    ) -> IntervalMappingResult:
        return IntervalMappingResult(
            status="unavailable",
            from_timebase=self.base_timebase if forward else self.current_timebase,
            to_timebase=self.current_timebase if forward else self.base_timebase,
            requested_interval=interval,
            reason_code=reason_code,
            reason=reason,
        )

    def _failed_composition(
        self,
        step: TimelineStepTransform,
        *,
        status: TransformStatus,
        reason_code: str,
        reason: str,
    ) -> TimelineLedger:
        return TimelineLedger(
            status=status,
            current_timebase=step.output_timebase,
            base_duration_sec=self.base_duration_sec,
            current_duration_sec=step.output_duration_sec,
            base_audio_ref=self.base_audio_ref,
            steps=(*self.steps, step),
            reason_code=reason_code,
            reason=reason,
        )


def _current_timebase(revision: int) -> TimebaseRef:
    return TimebaseRef(kind="current", revision=revision)


def _append_identity_piece(
    pieces: list[StepTimelinePiece],
    *,
    step_id: str,
    index: int,
    input_start: float,
    input_end: float,
    output_start: float,
) -> float:
    duration = input_end - input_start
    if duration <= _EPSILON:
        return output_start
    pieces.append(
        StepTimelinePiece(
            piece_id=f"{step_id}:identity:{index}",
            kind="identity",
            input_interval=TimeInterval(start_sec=input_start, end_sec=input_end),
            output_interval=TimeInterval(
                start_sec=output_start,
                end_sec=output_start + duration,
            ),
            correspondence="exact",
        )
    )
    return output_start + duration


def _validate_step_timebases(input_ref: TimebaseRef, output_ref: TimebaseRef) -> None:
    if input_ref.kind != "current" or output_ref.kind != "current":
        raise ValueError("step transforms must map current timebases")
    if output_ref.revision != input_ref.revision + 1:
        raise ValueError("step output revision must immediately follow input revision")


def _validate_unique_piece_ids(pieces: Sequence[object]) -> None:
    piece_ids = [piece.piece_id for piece in pieces]
    if len(piece_ids) != len(set(piece_ids)):
        raise ValueError("timeline piece ids must be unique")


def _validate_coverage(
    intervals: Sequence[TimeInterval],
    duration_sec: float,
    label: str,
) -> None:
    ordered = sorted(intervals, key=lambda interval: interval.start_sec)
    if not ordered:
        raise ValueError(f"{label} intervals do not cover the timebase")
    cursor = 0.0
    for interval in ordered:
        if interval.start_sec > cursor + _EPSILON:
            raise ValueError(f"{label} intervals contain a gap")
        if interval.start_sec < cursor - _EPSILON:
            raise ValueError(f"{label} intervals overlap")
        cursor = interval.end_sec
    if not _close(cursor, duration_sec):
        raise ValueError(f"{label} intervals do not end at the declared duration")


def _validate_monotonic_step(pieces: Sequence[StepTimelinePiece]) -> None:
    mapped = [
        piece
        for piece in pieces
        if piece.input_interval is not None and piece.output_interval is not None
    ]
    ordered = sorted(mapped, key=lambda piece: piece.input_interval.start_sec)
    previous_output_end = 0.0
    for piece in ordered:
        if piece.output_interval.start_sec < previous_output_end - _EPSILON:
            raise ValueError("step transform is non-monotonic")
        previous_output_end = piece.output_interval.end_sec


def _validate_monotonic_ledger(pieces: Sequence[LedgerTimelinePiece]) -> None:
    mapped = [
        piece
        for piece in pieces
        if piece.base_interval is not None and piece.current_interval is not None
    ]
    ordered = sorted(mapped, key=lambda piece: piece.base_interval.start_sec)
    previous_current_end = 0.0
    for piece in ordered:
        if piece.current_interval.start_sec < previous_current_end - _EPSILON:
            raise ValueError("ledger mapping is non-monotonic")
        previous_current_end = piece.current_interval.end_sec


def _intersection(left: TimeInterval, right: TimeInterval) -> TimeInterval | None:
    start = max(left.start_sec, right.start_sec)
    end = min(left.end_sec, right.end_sec)
    if end <= start + _EPSILON:
        return None
    return TimeInterval(start_sec=start, end_sec=end)


def _map_linear(
    interval: TimeInterval,
    source: TimeInterval,
    target: TimeInterval | None,
) -> TimeInterval | None:
    if target is None:
        return None
    scale = target.duration_sec / source.duration_sec
    start = target.start_sec + (interval.start_sec - source.start_sec) * scale
    end = target.start_sec + (interval.end_sec - source.start_sec) * scale
    return TimeInterval(start_sec=start, end_sec=end)


def _composed_semantics(
    ledger_piece: LedgerTimelinePiece,
    step_piece: StepTimelinePiece,
) -> tuple[PieceKind, Correspondence]:
    if ledger_piece.base_interval is None:
        return "insertion", "none"
    if step_piece.output_interval is None:
        return "deletion", "none"
    if ledger_piece.kind == "replacement" or step_piece.kind == "replacement":
        return "replacement", "boundary_linear"
    if ledger_piece.kind == "time_stretch" or step_piece.kind == "time_stretch":
        return "time_stretch", "linear"
    return "identity", "exact"


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    if value in values:
        return values
    return (*values, value)


def _normalize_ledger_pieces(
    pieces: Sequence[LedgerTimelinePiece],
    revision: int,
) -> tuple[LedgerTimelinePiece, ...]:
    ordered = sorted(pieces, key=_piece_sort_key)
    merged: list[LedgerTimelinePiece] = []
    for piece in ordered:
        if merged and _can_merge(merged[-1], piece):
            previous = merged.pop()
            merged.append(
                LedgerTimelinePiece(
                    piece_id="pending",
                    kind=piece.kind,
                    base_interval=_join_intervals(previous.base_interval, piece.base_interval),
                    current_interval=_join_intervals(
                        previous.current_interval,
                        piece.current_interval,
                    ),
                    correspondence=piece.correspondence,
                    provenance_step_ids=piece.provenance_step_ids,
                )
            )
        else:
            merged.append(piece)
    return tuple(
        piece.model_copy(update={"piece_id": f"ledger:r{revision}:p{index:04d}"})
        for index, piece in enumerate(merged)
    )


def _piece_sort_key(piece: LedgerTimelinePiece) -> tuple[int, float, float]:
    if piece.current_interval is not None:
        base_start = piece.base_interval.start_sec if piece.base_interval else -1.0
        return 0, piece.current_interval.start_sec, base_start
    assert piece.base_interval is not None
    return 1, piece.base_interval.start_sec, 0.0


def _can_merge(left: LedgerTimelinePiece, right: LedgerTimelinePiece) -> bool:
    if (
        left.kind != right.kind
        or left.correspondence != right.correspondence
        or left.provenance_step_ids != right.provenance_step_ids
    ):
        return False
    return _adjacent_or_both_none(left.base_interval, right.base_interval) and (
        _adjacent_or_both_none(left.current_interval, right.current_interval)
    )


def _adjacent_or_both_none(
    left: TimeInterval | None,
    right: TimeInterval | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _close(left.end_sec, right.start_sec)


def _join_intervals(
    left: TimeInterval | None,
    right: TimeInterval | None,
) -> TimeInterval | None:
    if left is None or right is None:
        return None
    return TimeInterval(start_sec=left.start_sec, end_sec=right.end_sec)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _EPSILON


def _duration_close(left: float, right: float) -> bool:
    """Allow sub-millisecond container/sample quantization between steps."""

    return abs(left - right) <= _DURATION_EPSILON_SEC


def _snap_duration_endpoint(
    interval: TimeInterval,
    *,
    declared_duration_sec: float | None,
    observed_duration_sec: float | None,
) -> TimeInterval:
    if (
        declared_duration_sec is None
        or observed_duration_sec is None
        or interval.end_sec != declared_duration_sec
        or interval.start_sec >= observed_duration_sec
        or not _duration_close(declared_duration_sec, observed_duration_sec)
    ):
        return interval
    return TimeInterval(start_sec=interval.start_sec, end_sec=observed_duration_sec)
