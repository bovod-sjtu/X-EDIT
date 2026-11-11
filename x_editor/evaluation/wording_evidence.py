"""Independent, read-only ASR and alignment evidence for wording rubrics."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel


class AlignedWordingToken(StrictModel):
    text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> AlignedWordingToken:
        if self.end_sec <= self.start_sec:
            raise ValueError("aligned wording token end_sec must exceed start_sec")
        return self


class IndependentWordingObservation(StrictModel):
    """One ASR/alignment observation produced outside the execution path."""

    schema_version: Literal["x-editor.independent-wording-observation/v1"] = (
        "x-editor.independent-wording-observation/v1"
    )
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript: str = ""
    language: str | None = None
    provider: str = Field(min_length=1)
    model: str | None = None
    transcript_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alignment_text_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    words: list[AlignedWordingToken] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class TranscriptAlignmentChange(StrictModel):
    operation: Literal["equal", "replace", "delete", "insert"]
    source_text: str = ""
    output_text: str = ""


class WordingPairEvidence(StrictModel):
    """Independent transcript evidence for one already-frozen audio pair."""

    schema_version: Literal["x-editor.wording-pair-evidence/v1"] = (
        "x-editor.wording-pair-evidence/v1"
    )
    pair_id: str = Field(min_length=1)
    status: Literal["ready", "low_confidence", "unavailable", "provider_failure"]
    minimum_confidence: float = Field(ge=0.0, le=1.0)
    pair_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: IndependentWordingObservation | None = None
    output: IndependentWordingObservation | None = None
    transcript_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    alignment_changes: list[TranscriptAlignmentChange] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    read_only: Literal[True] = True
    execution_feedback_allowed: Literal[False] = False


class WordingEvidenceProvider(Protocol):
    provider_name: str
    model_name: str | None

    def observe(
        self,
        audio_path: str,
        *,
        label: str,
    ) -> IndependentWordingObservation | dict[str, Any]: ...


class SpeechPerceptionWordingEvidenceProvider:
    """Adapter over the existing Qwen3-ASR and forced-alignment services."""

    provider_name = "qwen3_asr_forced_alignment"

    def __init__(
        self,
        *,
        asr_url: str | None = None,
        aligner_url: str | None = None,
        timeout_sec: float = 300.0,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
    ) -> None:
        from x_editor.perception.speech import SpeechPerceptionClient

        self.model_name = model_name
        self._client = SpeechPerceptionClient(
            asr_url=asr_url,
            aligner_url=aligner_url,
            timeout=timeout_sec,
            cache_dir=cache_dir,
            enable_service_bootstrap=False,
        )

    def observe(
        self,
        audio_path: str,
        *,
        label: str,
    ) -> IndependentWordingObservation:
        del label
        path = Path(audio_path).expanduser().resolve()
        raw = self._client.perceive(
            path,
            use_asr=True,
            use_alignment=True,
            use_diarization=False,
            strict_asr=True,
            strict_alignment=False,
        )
        transcript = str(raw.get("source_text") or "").strip()
        words = _aligned_words(raw.get("word_timestamps"))
        transcript_confidence = _transcript_confidence(raw)
        coverage = _alignment_text_coverage(transcript, words)
        evidence_confidence = _combined_confidence(
            transcript_confidence=transcript_confidence,
            alignment_text_coverage=coverage,
        )
        return IndependentWordingObservation(
            audio_sha256=_sha256_file(path),
            transcript=transcript,
            language=_optional_text(raw.get("language")),
            provider=self.provider_name,
            model=self.model_name,
            transcript_confidence=transcript_confidence,
            alignment_text_coverage=coverage,
            evidence_confidence=evidence_confidence,
            words=words,
            errors=[
                str(item.get("message") or item)
                for item in raw.get("errors") or []
                if isinstance(item, dict)
            ],
        )


def build_wording_pair_evidence(
    *,
    pair_id: str,
    source: IndependentWordingObservation | None,
    output: IndependentWordingObservation | None,
    minimum_confidence: float,
    provider_error: str | None = None,
) -> WordingPairEvidence:
    if provider_error is not None:
        return WordingPairEvidence(
            pair_id=pair_id,
            status="provider_failure",
            minimum_confidence=minimum_confidence,
            source=source,
            output=output,
            reason=provider_error,
        )
    if source is None or output is None or not source.transcript or not output.transcript:
        return WordingPairEvidence(
            pair_id=pair_id,
            status="unavailable",
            minimum_confidence=minimum_confidence,
            source=source,
            output=output,
            reason="independent ASR did not produce both source and output transcripts",
        )
    confidences = [source.evidence_confidence, output.evidence_confidence]
    pair_confidence = (
        min(float(value) for value in confidences if value is not None)
        if all(value is not None for value in confidences)
        else None
    )
    source_tokens = wording_tokens(source.transcript)
    output_tokens = wording_tokens(output.transcript)
    matcher = SequenceMatcher(a=source_tokens, b=output_tokens, autojunk=False)
    changes = [
        TranscriptAlignmentChange(
            operation=tag,
            source_text=" ".join(source_tokens[left_start:left_end]),
            output_text=" ".join(output_tokens[right_start:right_end]),
        )
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes()
    ]
    status: Literal["ready", "low_confidence"] = (
        "ready"
        if pair_confidence is not None and pair_confidence >= minimum_confidence
        else "low_confidence"
    )
    reason = (
        "independent ASR/alignment evidence meets the frozen confidence threshold"
        if status == "ready"
        else (
            "independent ASR/alignment evidence is below or lacks the frozen confidence "
            "threshold; it is advisory only and the judge may abstain"
        )
    )
    return WordingPairEvidence(
        pair_id=pair_id,
        status=status,
        minimum_confidence=minimum_confidence,
        pair_confidence=pair_confidence,
        source=source,
        output=output,
        transcript_similarity=matcher.ratio(),
        alignment_changes=changes,
        reason=reason,
    )


def wording_tokens(text: str) -> list[str]:
    """Tokenize Latin runs and individual CJK characters without semantic rewriting."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*|[\u3400-\u9fff]", normalized)


def _aligned_words(value: Any) -> list[AlignedWordingToken]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[AlignedWordingToken] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        start = _finite_float(item.get("start_sec", item.get("start")))
        end = _finite_float(item.get("end_sec", item.get("end")))
        if not text or start is None or end is None or end <= start:
            continue
        confidence = _probability(
            item.get("confidence", item.get("score", item.get("probability")))
        )
        normalized_tokens = wording_tokens(text)
        result.append(
            AlignedWordingToken(
                text=text,
                normalized_text=" ".join(normalized_tokens) or text.casefold(),
                start_sec=start,
                end_sec=end,
                confidence=confidence,
            )
        )
    return result


def _transcript_confidence(raw: dict[str, Any]) -> float | None:
    asr = raw.get("asr")
    candidates: list[Any] = [raw.get("confidence")]
    if isinstance(asr, dict):
        candidates.extend((asr.get("confidence"), asr.get("score")))
        asr_raw = asr.get("raw")
        if isinstance(asr_raw, dict):
            candidates.extend(
                (
                    asr_raw.get("confidence"),
                    asr_raw.get("score"),
                    asr_raw.get("probability"),
                )
            )
    for value in candidates:
        probability = _probability(value)
        if probability is not None:
            return probability
    word_confidences = [
        word.confidence
        for word in _aligned_words(raw.get("word_timestamps"))
        if word.confidence is not None
    ]
    return (
        sum(float(value) for value in word_confidences) / len(word_confidences)
        if word_confidences
        else None
    )


def _alignment_text_coverage(
    transcript: str,
    words: Sequence[AlignedWordingToken],
) -> float | None:
    transcript_tokens = wording_tokens(transcript)
    aligned_tokens = wording_tokens(" ".join(word.text for word in words))
    if not transcript_tokens or not aligned_tokens:
        return None
    return SequenceMatcher(
        a=transcript_tokens,
        b=aligned_tokens,
        autojunk=False,
    ).ratio()


def _combined_confidence(
    *,
    transcript_confidence: float | None,
    alignment_text_coverage: float | None,
) -> float | None:
    # Forced alignment measures whether timestamps cover the ASR hypothesis; it
    # is not an independent estimate that the hypothesis itself is correct.
    if transcript_confidence is None:
        return None
    return (
        min(transcript_confidence, alignment_text_coverage)
        if alignment_text_coverage is not None
        else transcript_confidence
    )


def _probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AlignedWordingToken",
    "IndependentWordingObservation",
    "SpeechPerceptionWordingEvidenceProvider",
    "TranscriptAlignmentChange",
    "WordingEvidenceProvider",
    "WordingPairEvidence",
    "build_wording_pair_evidence",
    "wording_tokens",
]
