"""Runtime verificators for intermediate X-Editor audio edit artifacts.

These checks are intentionally separate from the final evaluator.  A
verificator runs next to the executor, validates one just-produced step result,
and returns structured repair hints that can be fed into a retry/replan pass.
"""

from __future__ import annotations

import math
import os
import re
import wave
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from x_editor.agents.base import GeminiClient
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key
from x_editor.core.schemas import AudioEditStep

VerificationSeverity = Literal["info", "warning", "blocking"]
VerificationConfidence = Literal["low", "medium", "high"]
VerificationOutcome = Literal[
    "passed", "failed", "inconclusive", "skipped", "unavailable"
]
VerificationScope = Literal[
    "artifact_integrity", "operation_invariant", "step_diagnostic", "objective"
]
RepairScale = Literal["micro", "major"]
VerificationArtifactStatus = Literal["available", "candidate", "accepted", "rejected"]
EvidenceTimebaseStatus = Literal[
    "absent",
    "current",
    "mapped_from_base",
    "partially_mapped_from_base",
    "stale_unmappable",
    "unbound",
]


@dataclass(frozen=True)
class VerificationArtifact:
    """One immutable artifact revision used by a step-local verifier."""

    artifact_id: str
    path: str | None
    duration_sec: float | None
    timeline_revision: int | None
    status: VerificationArtifactStatus
    parent_artifact_ids: tuple[str, ...] = ()
    audio_ref_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "artifact_id": self.artifact_id,
                "path": self.path,
                "duration_sec": self.duration_sec,
                "timeline_revision": self.timeline_revision,
                "status": self.status,
                "parent_artifact_ids": list(self.parent_artifact_ids),
                "audio_ref_id": self.audio_ref_id,
            }
        )


@dataclass(frozen=True)
class EvidenceTimebaseBinding:
    """Version binding for reviewed evidence consumed by one verification step."""

    status: EvidenceTimebaseStatus
    observed_artifact_id: str | None = None
    observed_timeline_revision: int | None = None
    verifier_artifact_id: str | None = None
    verifier_timeline_revision: int | None = None
    word_timestamps: tuple[dict[str, Any], ...] = ()
    omitted_word_count: int = 0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "status": self.status,
                "observed_artifact_id": self.observed_artifact_id,
                "observed_timeline_revision": self.observed_timeline_revision,
                "verifier_artifact_id": self.verifier_artifact_id,
                "verifier_timeline_revision": self.verifier_timeline_revision,
                "word_timestamps": [dict(item) for item in self.word_timestamps],
                "omitted_word_count": self.omitted_word_count,
                "reason": self.reason,
            }
        )


@dataclass(frozen=True)
class StepVerificationContext:
    """Artifact-bound authority for validating one uncommitted step output."""

    step_id: str
    input_artifacts: tuple[VerificationArtifact, ...]
    primary_input_artifact: VerificationArtifact | None
    candidate_output_artifact: VerificationArtifact | None
    source_timebase_artifact_id: str | None
    target_timebase_artifact_id: str | None
    evidence_timebase: EvidenceTimebaseBinding

    def to_dict(self) -> dict[str, Any]:
        return _without_none(
            {
                "schema_version": "x-editor.step-verification-context/v1",
                "step_id": self.step_id,
                "input_artifacts": [item.to_dict() for item in self.input_artifacts],
                "primary_input_artifact": (
                    self.primary_input_artifact.to_dict()
                    if self.primary_input_artifact is not None
                    else None
                ),
                "candidate_output_artifact": (
                    self.candidate_output_artifact.to_dict()
                    if self.candidate_output_artifact is not None
                    else None
                ),
                "source_timebase_artifact_id": self.source_timebase_artifact_id,
                "target_timebase_artifact_id": self.target_timebase_artifact_id,
                "evidence_timebase": self.evidence_timebase.to_dict(),
            }
        )


@dataclass
class RuntimeRepairAction:
    """One candidate repair action derived from verifier evidence."""

    action_id: str
    component: str
    strategy: str
    scale: RepairScale
    confidence: VerificationConfidence = "medium"
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_verification: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "component": self.component,
            "strategy": self.strategy,
            "scale": self.scale,
            "confidence": self.confidence,
            "params": dict(self.params),
            "rationale": self.rationale,
            "expected_verification": list(self.expected_verification),
            "preconditions": list(self.preconditions),
        }


@dataclass
class RuntimeRepairPlan:
    """Structured plan for retrying or repairing one failed verification step."""

    step_id: str
    operation: str
    should_repair: bool
    micro_actions: list[RuntimeRepairAction] = field(default_factory=list)
    major_actions: list[RuntimeRepairAction] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "should_repair": self.should_repair,
            "micro_actions": [action.to_dict() for action in self.micro_actions],
            "major_actions": [action.to_dict() for action in self.major_actions],
            "findings": list(self.findings),
            "repair_hints": list(self.repair_hints),
        }


@dataclass
class RuntimeVerificationReport:
    """Structured evidence emitted by one runtime verificator."""

    component: str
    step_id: str
    operation: str
    passed: bool
    score: float | None = None
    confidence: VerificationConfidence = "medium"
    severity: VerificationSeverity = "info"
    outcome: VerificationOutcome | None = None
    scope: VerificationScope = "step_diagnostic"
    acceptance_grade: bool = False
    check_ids: list[str] = field(default_factory=list)
    check_outcomes: dict[str, VerificationOutcome] = field(default_factory=dict)
    subject_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)
    repair_actions: list[RuntimeRepairAction] = field(default_factory=list)
    evidence_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "step_id": self.step_id,
            "operation": self.operation,
            "passed": self.passed,
            "score": self.score,
            "confidence": self.confidence,
            "severity": self.severity,
            "outcome": self.outcome or ("passed" if self.passed else "failed"),
            "scope": self.scope,
            "acceptance_grade": self.acceptance_grade,
            "check_ids": list(self.check_ids),
            "check_outcomes": dict(self.check_outcomes),
            "subject_paths": list(self.subject_paths),
            "metrics": dict(self.metrics),
            "findings": list(self.findings),
            "repair_hints": list(self.repair_hints),
            "repair_actions": [action.to_dict() for action in self.repair_actions],
            "evidence_paths": dict(self.evidence_paths),
        }


class RuntimeVerificator(Protocol):
    """Interface implemented by operation-specific runtime checks."""

    component: str

    def applies_to(self, step: AudioEditStep, result: dict[str, Any], state: dict[str, Any]) -> bool:
        """Return true when this verificator should inspect the step result."""

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        """Validate a successful step result."""


class AudioVerificationModelBackend(Protocol):
    """Backend interface for model-backed runtime verification."""

    @property
    def available(self) -> bool:
        """Return true when the backend can make a model call."""

    @property
    def unavailable_reason(self) -> str:
        """Human-readable reason when the backend is unavailable."""

    def verify(
        self,
        *,
        task: dict[str, Any],
        audio_paths: list[str],
        audio_labels: list[str],
    ) -> dict[str, Any]:
        """Return model-backed verification JSON."""


class RuntimeVerificationRegistry:
    """Ordered collection of runtime verificators."""

    def __init__(self, verificators: list[RuntimeVerificator] | None = None) -> None:
        self._verificators: list[RuntimeVerificator] = list(verificators or [])

    def register(self, verificator: RuntimeVerificator) -> None:
        self._verificators.append(verificator)

    def verify_step(
        self,
        step: AudioEditStep,
        result: dict[str, Any] | None,
        state: dict[str, Any],
        *,
        components: set[str] | None = None,
        verification_context: StepVerificationContext | None = None,
    ) -> list[RuntimeVerificationReport]:
        if not isinstance(result, dict):
            return []
        verification_state = dict(state)
        if verification_context is not None:
            verification_state["step_verification_context"] = verification_context
        reports: list[RuntimeVerificationReport] = []
        for verificator in self._verificators:
            if components is not None and not _component_matches(verificator.component, components):
                continue
            if verificator.applies_to(step, result, verification_state):
                reports.append(verificator.verify(step, result, verification_state))
        return reports


class SourceTextVerificator:
    """Verify that ASR/perception steps produced usable source text."""

    component = "asr.source_text"
    _SOURCE_TEXT_OPERATIONS: ClassVar[set[str]] = {
        "perception.audio_fallback",
        "perception.gemini_audio",
        "perception.open_omni_audio",
        "perception.qwen3_omni_audio",
        "perception.speaker_attributed_review",
        "perception.speech_context",
    }

    def applies_to(self, step: AudioEditStep, result: dict[str, Any], state: dict[str, Any]) -> bool:
        return (
            step.operation in self._SOURCE_TEXT_OPERATIONS
            or "source_text" in result
            or bool(step.params.get("source_text_required"))
        )

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        perception = _as_mapping(result.get("perception_report")) or _as_mapping(state.get("perception_report")) or {}
        source_text = result.get("source_text") or perception.get("source_text") or state.get("source_text")
        source_text = str(source_text).strip() if isinstance(source_text, str) else ""
        need_source_text = bool(
            step.operation in self._SOURCE_TEXT_OPERATIONS
            or step.params.get("need_source_text")
            or step.params.get("need_words")
            or step.params.get("source_text_required")
        )
        speech_coverage = _speech_coverage(perception)
        no_speech_probability = _no_speech_probability(perception)
        true_no_speech = (
            speech_coverage is not None
            and speech_coverage < 0.02
            and (no_speech_probability is None or no_speech_probability >= 0.5)
        )

        metrics: dict[str, Any] = {
            "source_text_length": len(source_text),
            "speech_coverage": speech_coverage,
            "no_speech_probability": no_speech_probability,
            "source_text_audio_index": perception.get("source_text_audio_index") or state.get("source_text_audio_index"),
            "source_text_audio_path": perception.get("source_text_audio_path") or state.get("source_text_audio_path"),
        }
        findings: list[str] = []
        repair_hints: list[str] = []

        if source_text:
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=True,
                score=min(1.0, 0.5 + len(source_text) / 200.0),
                confidence="high",
                severity="info",
                metrics=_without_none(metrics),
                findings=["source_text is non-empty."],
            )
        if true_no_speech:
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=True,
                score=0.8,
                confidence="medium",
                severity="info",
                metrics=_without_none(metrics),
                findings=["ASR returned no source_text and VAD evidence indicates true no-speech audio."],
            )

        if need_source_text:
            findings.append("ASR/perception returned empty source_text for a step that needs speech text.")
            if speech_coverage is not None and speech_coverage >= 0.02:
                findings.append("Speech coverage suggests the audio is not silent, so this is likely an ASR/perception failure.")
            repair_hints.extend(
                [
                    "retry perception with refresh_perception=true",
                    "retry ASR on enhanced or resampled audio",
                    "try alternate input audio instead of assuming audio1",
                    "add an expected language or multilingual hint",
                ]
            )
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=False,
                score=0.0,
                confidence="medium",
                severity="blocking",
                metrics=_without_none(metrics),
                findings=findings,
                repair_hints=repair_hints,
                repair_actions=[
                    RuntimeRepairAction(
                        action_id=f"{step.id}:refresh_asr",
                        component=self.component,
                        strategy="retry_perception_with_refresh_and_language_hints",
                        scale="micro",
                        confidence="medium",
                        params={
                            "refresh_perception": True,
                            "need_source_text": True,
                            "try_enhanced_or_resampled_audio": True,
                            "source_text_audio_index": metrics.get("source_text_audio_index"),
                        },
                        rationale="The audio appears speech-bearing but source_text is empty.",
                        expected_verification=[self.component],
                    ),
                    RuntimeRepairAction(
                        action_id=f"{step.id}:alternate_audio_asr",
                        component=self.component,
                        strategy="retry_asr_on_alternate_input_audio",
                        scale="major",
                        confidence="medium",
                        params={"try_all_input_audios": True, "do_not_default_to_audio1": True},
                        rationale="Multi-audio tasks may carry the speech-bearing or language-reference signal in another input.",
                        expected_verification=[self.component],
                    ),
                ],
            )

        return RuntimeVerificationReport(
            component=self.component,
            step_id=step.id,
            operation=step.operation,
            passed=True,
            outcome="skipped",
            score=None,
            confidence="low",
            severity="info",
            metrics=_without_none(metrics),
            findings=["source_text is empty, but this step did not explicitly require source_text."],
        )


class SpanGroundingVerificator:
    """Verify selected timeline spans against deterministic perception evidence."""

    component = "aligner.span_grounding"
    _SPAN_REQUIRED_OPS: ClassVar[set[str]] = {
        "artifact_ops.extract_spans",
        "artifact_ops.remove_spans",
        "artifact_ops.mute_spans",
        "artifact_ops.beep_spans",
        "artifact_ops.speaker_scoped_edit",
    }

    def applies_to(self, step: AudioEditStep, result: dict[str, Any], state: dict[str, Any]) -> bool:
        return bool(_collect_spans(result)) or step.operation in self._SPAN_REQUIRED_OPS

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        spans = _collect_spans(result)
        perception = _as_mapping(state.get("perception_report")) or {}
        verification_context = _step_verification_context(state)
        evidence_binding = (
            verification_context.evidence_timebase
            if verification_context is not None
            else None
        )
        words = _normalized_words(
            list(evidence_binding.word_timestamps)
            if evidence_binding is not None
            else perception.get("word_timestamps") or []
        )
        duration = (
            verification_context.primary_input_artifact.duration_sec
            if verification_context is not None
            and verification_context.primary_input_artifact is not None
            else _first_audio_duration(
                result.get("input_audio_path"),
                step.inputs.get("input_audio_path"),
                step.params.get("input_audio_path"),
                state.get("input_audio"),
                state.get("current_audio_path"),
            )
        )
        findings: list[str] = []
        repair_hints: list[str] = []
        repair_actions: list[RuntimeRepairAction] = []
        invalid_count = 0
        text_mismatch_count = 0

        if not spans:
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=False,
                score=0.0,
                confidence="high",
                severity="blocking",
                metrics={"span_count": 0, "word_timestamp_count": len(words)},
                findings=["No selected spans were produced for an operation that depends on span grounding."],
                repair_hints=[
                    "refresh perception with word_timestamps=true",
                    "ask the planner to select spans from quoted phrase, speaker, ordinal, or semantic evidence",
                ],
                repair_actions=[
                    RuntimeRepairAction(
                        action_id=f"{step.id}:refresh_word_timestamps",
                        component=self.component,
                        strategy="refresh_perception_word_timestamps",
                        scale="major",
                        confidence="high",
                        params={"need_words": True, "refresh_perception": True},
                        rationale="No executable selected spans were produced.",
                        expected_verification=[self.component],
                    )
                ],
            )

        previous_end: float | None = None
        best_phrase_occurrences: list[dict[str, Any]] = []
        for index, span in enumerate(spans):
            start = _to_float(span.get("start_sec"))
            end = _to_float(span.get("end_sec"))
            if start is None or end is None:
                invalid_count += 1
                findings.append(f"span[{index}] is missing numeric start_sec/end_sec.")
                continue
            if end <= start:
                invalid_count += 1
                findings.append(f"span[{index}] has non-positive duration: {start:.3f}-{end:.3f}s.")
            if start < 0:
                invalid_count += 1
                findings.append(f"span[{index}] starts before zero: {start:.3f}s.")
            if duration is not None and end > duration + 0.05:
                invalid_count += 1
                findings.append(f"span[{index}] ends after source duration {duration:.3f}s: {end:.3f}s.")
            if previous_end is not None and start < previous_end - 0.02:
                invalid_count += 1
                findings.append(f"span[{index}] overlaps or moves backward relative to the previous span.")
            previous_end = max(previous_end or 0.0, end or 0.0)

            # ``label`` describes a span's semantic role (for example,
            # ``speaker_segment``); it is not a lexical claim. Only compare
            # against ASR words when the producer explicitly supplied text.
            expected_text = _normalize_text(span.get("text"))
            if expected_text and words and start is not None and end is not None and end > start:
                best_occurrence = _best_phrase_occurrence(expected_text, words)
                window_text = _normalize_text(
                    " ".join(
                        word["text"]
                        for word in words
                        if _ranges_overlap(start, end, word["start_sec"], word["end_sec"])
                    )
                )
                if window_text and expected_text not in window_text and window_text not in expected_text:
                    text_mismatch_count += 1
                    findings.append(
                        f"span[{index}] text evidence does not match word_timestamps window "
                        f"(expected={expected_text!r}, window={window_text!r})."
                    )
                    if best_occurrence:
                        shift_sec = float(best_occurrence["start_sec"]) - start
                        direction = "later" if shift_sec > 0 else "earlier"
                        best_phrase_occurrences.append(
                            {
                                "span_index": index,
                                "expected_text": expected_text,
                                "selected_start_sec": start,
                                "selected_end_sec": end,
                                "suggested_start_sec": best_occurrence["start_sec"],
                                "suggested_end_sec": best_occurrence["end_sec"],
                                "suggested_shift_sec": round(shift_sec, 6),
                                "direction": direction,
                                "matched_text": best_occurrence["text"],
                                "match_score": best_occurrence["score"],
                            }
                        )

        if invalid_count:
            repair_hints.extend(
                [
                    "rerun forced alignment and select spans on word boundaries",
                    "expand the target span by 0.15-0.30s and re-score",
                    "choose an alternate occurrence with better alignment confidence",
                ]
            )
        if text_mismatch_count:
            repair_hints.extend(
                [
                    "rank all quoted-phrase occurrences by ASR/forced-alignment confidence",
                    "refresh perception with the expected language hint",
                    "fall back to semantic audio grounding only after deterministic word evidence fails",
                ]
            )
        for occurrence in best_phrase_occurrences:
            repair_actions.append(
                RuntimeRepairAction(
                    action_id=f"{step.id}:span_{occurrence['span_index']}:shift_to_phrase_occurrence",
                    component=self.component,
                    strategy="shift_span_to_phrase_occurrence",
                    scale="micro",
                    confidence="high" if occurrence["match_score"] >= 0.9 else "medium",
                    params=occurrence,
                    rationale=(
                        "The selected span text does not match the word-timestamp window, but the expected "
                        f"phrase appears {occurrence['direction']} in the transcript."
                    ),
                    expected_verification=[self.component, "timeline.reconstruction"],
                    preconditions=["word_timestamps contain the expected phrase"],
                )
            )
        if invalid_count:
            repair_actions.append(
                RuntimeRepairAction(
                    action_id=f"{step.id}:realign_on_word_boundaries",
                    component=self.component,
                    strategy="rerun_alignment_and_select_word_boundary_spans",
                    scale="major",
                    confidence="high",
                    params={"need_words": True, "refresh_perception": True},
                    rationale="Span geometry is invalid and cannot be repaired safely by a small shift alone.",
                    expected_verification=[self.component, "timeline.reconstruction"],
                )
            )
        if text_mismatch_count and not best_phrase_occurrences:
            repair_actions.append(
                RuntimeRepairAction(
                    action_id=f"{step.id}:rank_alternate_phrase_occurrences",
                    component=self.component,
                    strategy="rank_alternate_phrase_occurrences",
                    scale="major",
                    confidence="medium",
                    params={"refresh_perception": True, "use_semantic_grounding_fallback": True},
                    rationale="The selected span text mismatched and no deterministic phrase occurrence was found.",
                    expected_verification=[self.component],
                )
            )

        passed = invalid_count == 0 and text_mismatch_count == 0
        score = max(0.0, 1.0 - (invalid_count + text_mismatch_count) / max(1, len(spans)))
        if passed:
            findings.append("selected spans are monotonic and compatible with available perception evidence.")
        if evidence_binding is not None and evidence_binding.status in {
            "mapped_from_base",
            "partially_mapped_from_base",
        }:
            findings.append(
                "word-timestamp evidence was projected onto the step input through the "
                "executor timeline ledger."
            )
        elif evidence_binding is not None and evidence_binding.status == "stale_unmappable":
            findings.append(
                "stale word-timestamp evidence was excluded because it could not be "
                "mapped safely onto the step input artifact."
            )
        return RuntimeVerificationReport(
            component=self.component,
            step_id=step.id,
            operation=step.operation,
            passed=passed,
            score=round(score, 4),
            confidence="high" if words or duration is not None else "medium",
            severity="info" if passed else "blocking",
            metrics=_without_none(
                {
                    "span_count": len(spans),
                    "word_timestamp_count": len(words),
                    "source_duration_sec": duration,
                    "invalid_span_count": invalid_count,
                    "text_mismatch_count": text_mismatch_count,
                    "best_phrase_occurrences": best_phrase_occurrences,
                    "source_timebase_artifact_id": (
                        verification_context.source_timebase_artifact_id
                        if verification_context is not None
                        else None
                    ),
                    "source_timeline_revision": (
                        verification_context.primary_input_artifact.timeline_revision
                        if verification_context is not None
                        and verification_context.primary_input_artifact is not None
                        else None
                    ),
                    "evidence_timebase_status": (
                        evidence_binding.status if evidence_binding is not None else None
                    ),
                    "evidence_timeline_revision": (
                        evidence_binding.observed_timeline_revision
                        if evidence_binding is not None
                        else None
                    ),
                    "evidence_omitted_word_count": (
                        evidence_binding.omitted_word_count
                        if evidence_binding is not None
                        else None
                    ),
                }
            ),
            findings=findings,
            repair_hints=repair_hints,
            repair_actions=repair_actions,
        )


class TimelineReconstructionVerificator:
    """Verify simple timeline duration invariants after artifact operations."""

    component = "timeline.reconstruction"

    _PRESERVE_DURATION_OPS: ClassVar[set[str]] = {
        "artifact_ops.mute_spans",
        "artifact_ops.beep_spans",
        "artifact_ops.speaker_scoped_edit",
        "foreground_background_remix",
        "intra_audio_voice_transfer",
    }

    def applies_to(self, step: AudioEditStep, result: dict[str, Any], state: dict[str, Any]) -> bool:
        return step.operation in self._PRESERVE_DURATION_OPS or step.operation in {
            "artifact_ops.extract_spans",
            "artifact_ops.remove_spans",
        }

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        spans = _collect_spans(result)
        context = _step_verification_context(state)
        bound_input = context.primary_input_artifact if context is not None else None
        if bound_input is not None:
            # A local edit consumes its crop, not the session's original audio.
            # Do not replace missing bound evidence with a different asset.
            source_duration = bound_input.duration_sec
            if source_duration is None:
                source_duration = _first_audio_duration(bound_input.path)
        else:
            source_duration = _first_audio_duration(
                result.get("input_audio_path"),
                step.inputs.get("input_audio_path"),
                step.inputs.get("input_audio"),
                step.inputs.get("audio_path"),
                step.params.get("input_audio_path"),
                state.get("input_audio"),
            )
        output_duration = _first_audio_duration(
            result.get("output_audio_path"),
            result.get("target_audio_path"),
            result.get("output_path"),
            result.get("generated_audio_path"),
        )
        span_duration = sum(
            max(0.0, (_to_float(span.get("end_sec")) or 0.0) - (_to_float(span.get("start_sec")) or 0.0))
            for span in spans
        )
        expected_duration: float | None = None
        if source_duration is not None:
            if step.operation == "artifact_ops.extract_spans":
                expected_duration = span_duration
            elif step.operation == "artifact_ops.remove_spans":
                expected_duration = max(0.0, source_duration - span_duration)
            elif step.operation in self._PRESERVE_DURATION_OPS:
                expected_duration = source_duration

        metrics = _without_none(
            {
                "source_duration_sec": source_duration,
                "output_duration_sec": output_duration,
                "expected_duration_sec": expected_duration,
                "selected_span_duration_sec": span_duration if spans else None,
                "selected_span_count": len(spans),
            }
        )
        if expected_duration is None or output_duration is None:
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=True,
                outcome="inconclusive",
                score=None,
                confidence="low",
                severity="info",
                metrics=metrics,
                findings=["duration invariant was not checked because source/output duration evidence is incomplete."],
                repair_hints=["record source/output durations in the operation manifest for stronger reconstruction checks"],
            )

        tolerance = max(0.08, expected_duration * 0.03)
        diff = abs(output_duration - expected_duration)
        passed = diff <= tolerance
        return RuntimeVerificationReport(
            component=self.component,
            step_id=step.id,
            operation=step.operation,
            passed=passed,
            score=round(max(0.0, 1.0 - diff / max(tolerance, 1e-6)), 4),
            confidence="high",
            severity="info" if passed else "blocking",
            metrics={**metrics, "duration_diff_sec": round(diff, 6), "duration_tolerance_sec": round(tolerance, 6)},
            findings=(
                ["output duration matches the expected timeline invariant."]
                if passed
                else [
                    (
                        "output duration violates the expected timeline invariant "
                        f"(expected {expected_duration:.3f}s, got {output_duration:.3f}s)."
                    )
                ]
            ),
            repair_hints=(
                []
                if passed
                else [
                    "recompute edit placement from the original manifest",
                    "fit inserted segments to selected span duration before reconstruction",
                    "rerun reconstruction from the original base audio instead of a prior edited artifact",
                ]
            ),
            repair_actions=(
                []
                if passed
                else [
                    RuntimeRepairAction(
                        action_id=f"{step.id}:fit_timeline_duration",
                        component=self.component,
                        strategy="fit_inserted_segments_to_expected_duration",
                        scale="micro",
                        confidence="high",
                        params={
                            "expected_duration_sec": expected_duration,
                            "output_duration_sec": output_duration,
                            "duration_diff_sec": round(diff, 6),
                        },
                        rationale="The operation output has a duration mismatch that may be corrected by fitting inserted spans.",
                        expected_verification=[self.component],
                    ),
                    RuntimeRepairAction(
                        action_id=f"{step.id}:reconstruct_from_original_manifest",
                        component=self.component,
                        strategy="rerun_reconstruction_from_original_base",
                        scale="major",
                        confidence="high",
                        params={"use_original_base_audio": True, "recompute_edit_placement_from_manifest": True},
                        rationale="If fitting is insufficient, rebuild the timeline from the original base and manifest.",
                        expected_verification=[self.component, "aligner.span_grounding"],
                    ),
                ]
            ),
        )


class ModelBackedAudioVerificator:
    """Use a large audio perception model for checks without reliable local tools."""

    component = "model.audio_perception"

    _MODEL_SUITABLE_OPERATIONS: ClassVar[set[str]] = {
        "accent_conversion",
        "audio_effects",
        "spatial_reverb",
        "ace_step_music.timeline_edit",
        "foreground_background_remix",
        "intra_audio_voice_transfer",
        "seed_vc",
        "tse",
        "qwen3_tts",
        "speech_separation",
        "speech_enhancement",
    }

    def __init__(self, backend: AudioVerificationModelBackend | None = None) -> None:
        self.backend = backend or GeminiRuntimeVerificationBackend()

    def applies_to(self, step: AudioEditStep, result: dict[str, Any], state: dict[str, Any]) -> bool:
        if not _model_verification_enabled(state):
            return False
        if step.operation in self._MODEL_SUITABLE_OPERATIONS:
            return bool(_model_audio_paths(step, result, state)[0])
        if step.operation.startswith("edit_data_pipeline."):
            return step.operation in {
                "edit_data_pipeline.emotion_style_edit",
                "edit_data_pipeline.pitch_rate_edit",
                "edit_data_pipeline.text_edit",
            } and bool(_model_audio_paths(step, result, state)[0])
        return False

    def verify(
        self,
        step: AudioEditStep,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> RuntimeVerificationReport:
        audio_paths, audio_labels = _model_audio_paths(step, result, state)
        if not self.backend.available:
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=True,
                outcome="unavailable",
                score=None,
                confidence="low",
                severity="info",
                metrics={"enabled": True, "available": False},
                findings=[f"model-backed verification skipped: {self.backend.unavailable_reason}"],
                repair_hints=["fall back to deterministic verificators or configure a large audio perception model"],
            )
        try:
            payload = self.backend.verify(
                task={
                    "user_request": state.get("user_prompt"),
                    "step_id": step.id,
                    "operation": step.operation,
                    "description": step.description,
                    "inputs": step.inputs,
                    "params": _redact_large_values(step.params),
                    "result": _redact_large_values(result),
                    "source_text": state.get("source_text"),
                    "perception_summary": _perception_summary(state.get("perception_report")),
                },
                audio_paths=audio_paths,
                audio_labels=audio_labels,
            )
        except Exception as exc:  # noqa: BLE001 - external model boundary.
            return RuntimeVerificationReport(
                component=self.component,
                step_id=step.id,
                operation=step.operation,
                passed=True,
                outcome="unavailable",
                score=None,
                confidence="low",
                severity="info",
                metrics={"enabled": True, "available": True},
                findings=[f"model-backed verification failed to run: {type(exc).__name__}: {exc}"],
                repair_hints=["retry with deterministic verificators or rerun after the model verification service is healthy"],
            )

        passed = bool(payload.get("passed", True))
        severity = str(payload.get("severity") or ("warning" if not passed else "info"))
        if severity not in {"info", "warning", "blocking"}:
            severity = "warning" if not passed else "info"
        confidence = str(payload.get("confidence") or "medium")
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        return RuntimeVerificationReport(
            component=str(payload.get("component") or self.component),
            step_id=step.id,
            operation=step.operation,
            passed=passed,
            score=_to_float(payload.get("score")),
            confidence=confidence,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            metrics=_as_mapping(payload.get("metrics")) or {},
            findings=_string_list(payload.get("findings")),
            repair_hints=_string_list(payload.get("repair_hints")),
            repair_actions=_model_repair_actions(step, str(payload.get("component") or self.component), payload),
            evidence_paths={key: str(value) for key, value in (_as_mapping(payload.get("evidence_paths")) or {}).items()},
        )


class GeminiRuntimeVerificationBackend:
    """Gemini-backed implementation for model-only runtime verificators."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        config = AudioEditConfig()
        self.model_name = model_name or os.getenv("X_EDITOR_VERIFY_MODEL_NAME", "gemini-2.5-pro")
        self.api_key = resolve_gemini_api_key(api_key) or config.omni_api_key
        self.client = GeminiClient(
            api_key=self.api_key,
            base_url=base_url or config.omni_base_url,
            temperature=0.01,
            max_tokens=8192,
            timeout=timeout,
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def unavailable_reason(self) -> str:
        return "no Gemini API key is configured"

    def verify(
        self,
        *,
        task: dict[str, Any],
        audio_paths: list[str],
        audio_labels: list[str],
    ) -> dict[str, Any]:
        label_text = "\n".join(
            f"Audio {index + 1} ({audio_labels[index] if index < len(audio_labels) else 'audio'}): {path}"
            for index, path in enumerate(audio_paths)
        )
        user_text = (
            f"{label_text}\n\n"
            "Runtime verification task JSON:\n"
            f"{_json_dumps(task)}\n\n"
            "Return whether the just-produced step artifact satisfies this step's local requirement. "
            "Focus on operation-specific evidence such as accent/dialect, emotion/prosody, channel effect, "
            "speaker identity, source/target content, foreground/background preservation, or music/rap style. "
            "This is not final benchmark scoring; return concrete findings and repair_hints for a retry pass."
        )
        return self.client.call_json(
            _MODEL_VERIFIER_SYSTEM_PROMPT,
            user_text,
            audio_paths=audio_paths,
        )


def build_default_verification_registry() -> RuntimeVerificationRegistry:
    """Return the default lightweight runtime verificator registry."""

    from x_editor.runtime.artifact_verification import (
        ArtifactDeliveryContractVerificator,
        DeterministicArtifactPrincipleVerificator,
        EvaluatorDeferredArtifactVerificator,
    )

    return RuntimeVerificationRegistry(
        [
            SourceTextVerificator(),
            SpanGroundingVerificator(),
            TimelineReconstructionVerificator(),
            DeterministicArtifactPrincipleVerificator(),
            ArtifactDeliveryContractVerificator(),
            EvaluatorDeferredArtifactVerificator(),
            ModelBackedAudioVerificator(),
        ]
    )


def build_runtime_repair_plan(
    step: AudioEditStep,
    reports: list[RuntimeVerificationReport],
    state: dict[str, Any] | None = None,
) -> RuntimeRepairPlan:
    """Convert verifier reports into ordered micro/major repair candidates."""

    failed_reports = [report for report in reports if not report.passed]
    micro_actions: list[RuntimeRepairAction] = []
    major_actions: list[RuntimeRepairAction] = []
    findings: list[str] = []
    repair_hints: list[str] = []
    for report in failed_reports:
        findings.extend(report.findings)
        repair_hints.extend(report.repair_hints)
        actions = list(report.repair_actions) or _fallback_repair_actions(step, report, state or {})
        for action in actions:
            if action.scale == "micro":
                micro_actions.append(action)
            else:
                major_actions.append(action)
    return RuntimeRepairPlan(
        step_id=step.id,
        operation=step.operation,
        should_repair=bool(failed_reports),
        micro_actions=_dedupe_actions(micro_actions),
        major_actions=_dedupe_actions(major_actions),
        findings=_dedupe_strings(findings),
        repair_hints=_dedupe_strings(repair_hints),
    )


def _collect_spans(result: dict[str, Any]) -> list[dict[str, Any]]:
    # Ground source-timeline spans against the source audio. Some edit handlers
    # also expose target_spans in the composed output timeline; comparing those
    # coordinates with the source duration produces false boundary failures.
    for key in ("selected_spans", "source_spans", "target_spans", "spans"):
        value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    records = result.get("replacement_records")
    if isinstance(records, list):
        return [item for item in records if isinstance(item, dict)]
    return []


def _step_verification_context(state: dict[str, Any]) -> StepVerificationContext | None:
    value = state.get("step_verification_context")
    return value if isinstance(value, StepVerificationContext) else None


def _fallback_repair_actions(
    step: AudioEditStep,
    report: RuntimeVerificationReport,
    state: dict[str, Any],
) -> list[RuntimeRepairAction]:
    component = report.component
    if component == "asr.source_text":
        return [
            RuntimeRepairAction(
                action_id=f"{step.id}:refresh_asr",
                component=component,
                strategy="retry_perception_with_refresh_and_language_hints",
                scale="micro",
                confidence="medium",
                params={"refresh_perception": True, "need_source_text": True},
                rationale="ASR/source_text verification failed.",
                expected_verification=[component],
            ),
            RuntimeRepairAction(
                action_id=f"{step.id}:model_audio_perception_source_text",
                component=component,
                strategy="fallback_to_large_audio_perception_model",
                scale="major",
                confidence="medium",
                params={"enable_model_verification": True, "force_gemini": True},
                rationale="Specialized ASR may be unreliable for this audio/language.",
                expected_verification=[component],
            ),
        ]
    if component == "aligner.span_grounding":
        return [
            RuntimeRepairAction(
                action_id=f"{step.id}:expand_span_and_rescore",
                component=component,
                strategy="expand_span_boundaries_and_rescore",
                scale="micro",
                confidence="medium",
                params={"expand_before_sec": 0.2, "expand_after_sec": 0.2},
                rationale="Span grounding failed; a small boundary expansion may recover clipped speech.",
                expected_verification=[component],
            ),
            RuntimeRepairAction(
                action_id=f"{step.id}:refresh_alignment",
                component=component,
                strategy="rerun_alignment_and_select_word_boundary_spans",
                scale="major",
                confidence="medium",
                params={"refresh_perception": True, "need_words": True},
                rationale="Span grounding failed without a deterministic micro-shift candidate.",
                expected_verification=[component],
            ),
        ]
    if component == "timeline.reconstruction":
        return [
            RuntimeRepairAction(
                action_id=f"{step.id}:reconstruct_from_original_manifest",
                component=component,
                strategy="rerun_reconstruction_from_original_base",
                scale="major",
                confidence="medium",
                params={"use_original_base_audio": True},
                rationale="Timeline reconstruction invariant failed.",
                expected_verification=[component],
            )
        ]
    if component.startswith("model.audio_perception"):
        return _model_repair_actions(step, component, {"repair_hints": report.repair_hints})
    return [
        RuntimeRepairAction(
            action_id=f"{step.id}:planner_repair",
            component=component,
            strategy="planner_repair_with_verification_findings",
            scale="major",
            confidence="medium",
            params={"previous_verification_error": report.to_dict()},
            rationale="No deterministic repair strategy is registered for this verifier component.",
            expected_verification=[component],
        )
    ]


def _model_repair_actions(
    step: AudioEditStep,
    component: str,
    payload: dict[str, Any],
) -> list[RuntimeRepairAction]:
    if bool(payload.get("passed", False)):
        return []
    hints = _string_list(payload.get("repair_hints"))
    operation = step.operation
    actions: list[RuntimeRepairAction] = []
    if operation in {"audio_effects", "spatial_reverb"}:
        actions.append(
            RuntimeRepairAction(
                action_id=f"{step.id}:adjust_effect_strength",
                component=component,
                strategy="adjust_effect_preset_or_strength",
                scale="micro",
                confidence="medium",
                params={"repair_hints": hints, "preserve_content_and_speaker": True},
                rationale="Model-backed verification indicates the requested acoustic effect is not sufficiently audible or is over-applied.",
                expected_verification=[component, "asr.source_text"],
            )
        )
    elif operation == "accent_conversion":
        actions.extend(
            [
                RuntimeRepairAction(
                    action_id=f"{step.id}:choose_alternate_accent_reference",
                    component=component,
                    strategy="choose_higher_confidence_accent_library_reference",
                    scale="micro",
                    confidence="medium",
                    params={"repair_hints": hints},
                    rationale="The accent/dialect target may need a stronger or more canonical reference sample.",
                    expected_verification=[component],
                ),
                RuntimeRepairAction(
                    action_id=f"{step.id}:report_missing_accent_or_fallback",
                    component=component,
                    strategy="planner_repair_or_explicit_accent_fallback",
                    scale="major",
                    confidence="medium",
                    params={"repair_hints": hints, "allow_language_fallback": True},
                    rationale="If no suitable accent reference exists, replan with an explicit fallback or fail with a concrete missing-accent reason.",
                    expected_verification=[component],
                ),
            ]
        )
    elif operation in {"seed_vc", "tse", "intra_audio_voice_transfer"}:
        actions.extend(
            [
                RuntimeRepairAction(
                    action_id=f"{step.id}:select_cleaner_speaker_reference",
                    component=component,
                    strategy="select_cleaner_or_longer_speaker_reference",
                    scale="micro",
                    confidence="medium",
                    params={"repair_hints": hints, "rerun_diarization_if_needed": True},
                    rationale="Speaker identity verification can often be repaired by using a cleaner enrollment/reference segment.",
                    expected_verification=[component],
                ),
                RuntimeRepairAction(
                    action_id=f"{step.id}:speaker_mapping_replan",
                    component=component,
                    strategy="replan_speaker_mapping_with_diarization_evidence",
                    scale="major",
                    confidence="medium",
                    params={"repair_hints": hints},
                    rationale="The target/reference speaker mapping may be wrong or ambiguous.",
                    expected_verification=[component],
                ),
            ]
        )
    elif operation.startswith("edit_data_pipeline.") or operation == "qwen3_tts":
        actions.extend(
            [
                RuntimeRepairAction(
                    action_id=f"{step.id}:local_slice_regenerate",
                    component=component,
                    strategy="retry_with_shorter_local_slice_or_regenerate_target",
                    scale="micro",
                    confidence="medium",
                    params={"repair_hints": hints, "preserve_speaker": True},
                    rationale="Content/style verification suggests the generated or edited local segment needs a narrower retry.",
                    expected_verification=[component, "asr.source_text"],
                ),
                RuntimeRepairAction(
                    action_id=f"{step.id}:text_or_tool_replan",
                    component=component,
                    strategy="planner_repair_target_text_or_tool_choice",
                    scale="major",
                    confidence="medium",
                    params={"repair_hints": hints},
                    rationale="The text/tool route may be wrong and should be replanned with verifier findings.",
                    expected_verification=[component],
                ),
            ]
        )
    elif operation in {"foreground_background_remix", "speech_separation"}:
        actions.extend(
            [
                RuntimeRepairAction(
                    action_id=f"{step.id}:swap_or_reselect_stems",
                    component=component,
                    strategy="swap_or_reselect_foreground_background_stems",
                    scale="micro",
                    confidence="medium",
                    params={"repair_hints": hints},
                    rationale="Foreground/background verification can often be repaired by correcting stem assignment.",
                    expected_verification=[component],
                ),
                RuntimeRepairAction(
                    action_id=f"{step.id}:separation_replan",
                    component=component,
                    strategy="replan_with_alternate_separation_or_tse",
                    scale="major",
                    confidence="medium",
                    params={"repair_hints": hints},
                    rationale="The separation route may need a different backend or target speaker reference.",
                    expected_verification=[component],
                ),
            ]
        )
    elif operation == "ace_step_music.timeline_edit":
        actions.append(
            RuntimeRepairAction(
                action_id=f"{step.id}:music_generation_replan",
                component=component,
                strategy="split_lyrics_or_adjust_music_prompt_and_duration",
                scale="major",
                confidence="medium",
                params={"repair_hints": hints},
                rationale="Music/rap verification usually requires regenerating with adjusted prompt, lyrics, or duration.",
                expected_verification=[component],
            )
        )
    if not actions:
        actions.append(
            RuntimeRepairAction(
                action_id=f"{step.id}:model_guided_retry",
                component=component,
                strategy="retry_with_model_verification_hints",
                scale="major",
                confidence="medium",
                params={"repair_hints": hints},
                rationale="Model-backed verification found a local failure without a specialized repair strategy.",
                expected_verification=[component],
            )
        )
    return actions


def _dedupe_actions(actions: list[RuntimeRepairAction]) -> list[RuntimeRepairAction]:
    seen: set[str] = set()
    deduped: list[RuntimeRepairAction] = []
    for action in actions:
        key = action.action_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _component_matches(component: str, requested: set[str]) -> bool:
    return component in requested or any(item.startswith(component + ".") for item in requested)


def _normalized_words(items: Any) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return words
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("word") or item.get("token")
        start = _to_float(item.get("start_sec", item.get("start")))
        end = _to_float(item.get("end_sec", item.get("end")))
        if text is None or start is None or end is None or end <= start:
            continue
        words.append({"text": str(text), "start_sec": start, "end_sec": end})
    return sorted(words, key=lambda item: (item["start_sec"], item["end_sec"]))


def _best_phrase_occurrence(expected_text: str, words: list[dict[str, Any]]) -> dict[str, Any] | None:
    tokens = expected_text.split()
    if not tokens or not words:
        return None
    min_size = max(1, len(tokens) - 1)
    max_size = min(len(words), len(tokens) + 2)
    best: dict[str, Any] | None = None
    for size in range(min_size, max_size + 1):
        for start_index in range(len(words) - size + 1):
            window = words[start_index : start_index + size]
            window_text = _normalize_text(" ".join(str(word["text"]) for word in window))
            if not window_text:
                continue
            contains = expected_text in window_text or window_text in expected_text
            score = 1.0 if expected_text == window_text else SequenceMatcher(None, expected_text, window_text).ratio()
            if contains:
                score = max(score, 0.92)
            if score < 0.78:
                continue
            candidate = {
                "text": window_text,
                "start_sec": float(window[0]["start_sec"]),
                "end_sec": float(window[-1]["end_sec"]),
                "score": round(score, 4),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def _first_audio_duration(*candidates: Any) -> float | None:
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, list):
            nested = _first_audio_duration(*candidate)
            if nested is not None:
                return nested
            continue
        try:
            path = Path(str(candidate))
        except TypeError:
            continue
        duration = _audio_duration_seconds(path)
        if duration is not None:
            return duration
    return None


def _audio_duration_seconds(path: Path) -> float | None:
    if not path.is_file() or path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                return None
            return handle.getnframes() / float(frame_rate)
    except Exception:  # noqa: BLE001 - malformed audio must stay non-fatal here.
        return None


def _speech_coverage(perception: dict[str, Any]) -> float | None:
    candidates = [
        perception.get("speech_coverage"),
        perception.get("vad_speech_coverage"),
        (_as_mapping(perception.get("audio_quality")) or {}).get("speech_coverage"),
        (_as_mapping(perception.get("vad")) or {}).get("speech_coverage"),
    ]
    return _first_probability(candidates)


def _no_speech_probability(perception: dict[str, Any]) -> float | None:
    candidates = [
        perception.get("no_speech_probability"),
        perception.get("no_speech_prob"),
        (_as_mapping(perception.get("asr")) or {}).get("no_speech_probability"),
        (_as_mapping(perception.get("vad")) or {}).get("no_speech_probability"),
    ]
    return _first_probability(candidates)


def _first_probability(values: list[Any]) -> float | None:
    for value in values:
        numeric = _to_float(value)
        if numeric is not None and math.isfinite(numeric):
            return max(0.0, min(1.0, numeric))
    return None


def _ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _model_verification_enabled(state: dict[str, Any]) -> bool:
    context = _as_mapping(state.get("context")) or {}
    if "enable_model_verification" in context:
        return _truthy(context.get("enable_model_verification"))
    if "verify_model" in context:
        return _truthy(context.get("verify_model"))
    plan = state.get("plan")
    global_params = _as_mapping(getattr(plan, "global_params", None)) or {}
    if "enable_model_verification" in global_params:
        return _truthy(global_params.get("enable_model_verification"))
    return _truthy(os.getenv("X_EDITOR_VERIFY_MODEL"))


def _model_audio_paths(
    step: AudioEditStep,
    result: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[str], list[str]]:
    pairs: list[tuple[str, str]] = []
    for label, value in [
        ("source/input", result.get("input_audio_path") or state.get("input_audio")),
        ("pre-step current", step.inputs.get("input_audio_path") or step.params.get("input_audio_path")),
        (
            "edited output",
            result.get("output_audio_path")
            or result.get("target_audio_path")
            or result.get("output_path")
            or result.get("generated_audio_path"),
        ),
    ]:
        path = _existing_audio_path(value)
        if path:
            pairs.append((label, path))
    for index, value in enumerate(state.get("reference_audio_paths") or []):
        path = _existing_audio_path(value)
        if path:
            pairs.append((f"reference {index + 1}", path))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, path in pairs:
        try:
            key = str(Path(path).resolve())
        except Exception:  # noqa: BLE001 - path normalization is best-effort.
            key = path
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, path))
    return [path for _, path in deduped], [label for label, _ in deduped]


def _existing_audio_path(value: Any) -> str | None:
    if not value or isinstance(value, (dict, list)):
        return None
    path = Path(str(value))
    if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"}:
        return str(path)
    return None


def _redact_large_values(value: Any, *, max_items: int = 12) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in list(value.items())[:max_items]:
            if key in {"audio", "audio_bytes", "waveform"}:
                redacted[key] = "<omitted>"
            else:
                redacted[key] = _redact_large_values(item, max_items=max_items)
        if len(value) > max_items:
            redacted["_truncated_key_count"] = len(value) - max_items
        return redacted
    if isinstance(value, list):
        items = [_redact_large_values(item, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            items.append({"_truncated_item_count": len(value) - max_items})
        return items
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...<truncated>"
    return value


def _perception_summary(value: Any) -> dict[str, Any]:
    perception = _as_mapping(value) or {}
    return _without_none(
        {
            "source_text": perception.get("source_text"),
            "language": perception.get("language"),
            "word_timestamp_count": len(perception.get("word_timestamps") or []),
            "speaker_segment_count": len(perception.get("speaker_segments") or perception.get("segments") or []),
            "speakers": perception.get("speakers"),
            "background": perception.get("background"),
            "audio_quality": perception.get("audio_quality"),
            "uncertainty": perception.get("uncertainty"),
        }
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if item not in (None, "")]


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


_MODEL_VERIFIER_SYSTEM_PROMPT = """You are a runtime audio verificator for X-Editor.

You inspect intermediate audio artifacts immediately after one tool step. You are
not a final benchmark scorer. Decide whether the step output satisfies the local
operation requirement well enough to continue, and provide concrete repair hints
when it does not.

Return ONLY JSON with this schema:
{
  "component": "model.audio_perception.<short_task>",
  "passed": true,
  "score": 0.0,
  "confidence": "low|medium|high",
  "severity": "info|warning|blocking",
  "metrics": {},
  "findings": [],
  "repair_hints": [],
  "evidence_paths": {}
}

Use "blocking" only for clear local failures that should trigger a repair pass,
such as wrong speaker, missing requested accent/effect, target phrase absent, or
output nearly silent. Prefer "warning" when evidence is ambiguous."""
