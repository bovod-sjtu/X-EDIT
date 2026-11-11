"""Timestamped audio caption review with an optional FlexSED prior."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from x_editor.agents.base import GeminiClient
from x_editor.agents.openai_compatible import OpenAICompatibleChatClient
from x_editor.core.agentic_mode import (
    GEMINI_MODE,
    QWEN3_OMNI_PROVIDER,
    perception_provider_name,
    resolve_agentic_mode,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key
from x_editor.perception.audio_slicing import audio_duration_sec, slice_audio
from x_editor.perception.invocation_trace import (
    PerceptionCallPurpose,
    capture_frontend_provider_call,
    finalize_frontend_invocation_trace,
    frontend_provider_call_failed,
    record_frontend_deterministic_fallback,
    reset_frontend_invocation_trace,
)
from x_editor.perception.evidence_triage import (
    EvidenceTriageBatch,
    triage_candidate_evidence_batch,
)

SLICE_PROPOSAL_SYSTEM_PROMPT = """You plan diagnostic slice_audio calls for timestamped audio caption review.

Return only a JSON object with a `slices` array. Each slice must include:
- onset: local start seconds in the complete audio
- offset: local end seconds in the complete audio
- reason: the concrete uncertainty this slice resolves
- flexsed_event_refs: optional event names or span ids from the FlexSED prior

When a FlexSED prior is present, use slices to verify its peaks and event
boundaries. Without a prior, use slices to review uncertain regions, dense
overlapping sounds, and scene transitions directly.
Keep slices short and diagnostic, usually 4-12 seconds.
"""


FINAL_REVIEW_SYSTEM_PROMPT = """You are a RedAudio timestamped audio caption review agent.

Inputs:
- The first attached audio is the complete audio.
- Additional attached audios are diagnostic slice_audio clips cut by the runtime.
- FlexSED prior evidence is optional. When present, treat it as a candidate
  event prior that must still agree with the audio.

Evidence policy:
- Prefer FlexSED-supported event labels when the audio agrees.
- You may add an event not covered by FlexSED only when directly heard. Mark it with evidence_source="model_only" (legacy "gemini_only" is accepted), confidence "low" or "medium", and add uncertainty explaining that FlexSED did not support it.
- FlexSED-supported events should use evidence_source="flexsed_gemini" and cite flexsed_event_refs.
- Do not invent slice ids. evidence_slice_ids must come from the provided slice_audio_history.

Timestamp policy:
- Times are local seconds relative to the complete audio start.
- Use diagnostic slices to calibrate event onsets/offsets and transitions.
- Avoid one whole-audio segment when the request asks for event localization.

Final output:
- Output only one valid JSON object.
- Required keys: segments, background, notable_events, uncertainty.
- Each segment requires: start_sec, end_sec, caption, events, evidence_source,
  flexsed_event_refs, evidence_slice_ids, confidence, target_match.
- target_match is "yes", "no", or "uncertain" relative to the user's requested
  target. Keep non-target observations in segments, but mark only the requested
  event instance or acoustic property as "yes".
"""


CANDIDATE_RUBRIC_SYSTEM_PROMPT = """You verify candidate timestamped audio-event segments.

Inputs:
- The attached audios are runtime slice_audio clips for candidate segments.
- The JSON payload lists candidate ids, time spans, captions/events, and slice ids.

For each candidate, answer whether that segment contains the requested event or
caption property. Do not change timestamps or invent new segment ids.

Return only one JSON object:
{
  "judgements": [
    {
      "candidate_id": "candidate_001",
      "answer": "yes" | "no" | "uncertain",
      "confidence": "low" | "medium" | "high",
      "reason": "short evidence-grounded reason",
      "evidence_slice_ids": ["slice_01"]
    }
  ]
}
"""


DEFAULT_EVENT_BANK_PATH = Path(__file__).with_name("audio_event_bank.json")

STRONG_OMNI_TIMESTAMPED_CAPTION_PROMPT_SUFFIX = """

Provider note:
- You are {provider_display_name} in X-Editor's open agentic mode.
- For FlexSED-supported event spans, use evidence_source="flexsed_omni" and cite flexsed_event_refs.
- For event spans that only you hear without FlexSED or slice support, use evidence_source="omni_only", confidence "low" or "medium", and uncertainty explaining the missing FlexSED support.
- Keep the final segments array compact. Return at most 12 segments, merging adjacent spans when they have the same caption/event role.
- Do not claim to be a different provider.
"""


@dataclass
class CaptionSegment:
    start_sec: float
    end_sec: float
    caption: str
    events: list[str] = field(default_factory=list)
    evidence_source: str = "model_only"
    flexsed_event_refs: list[str] = field(default_factory=list)
    evidence_slice_ids: list[str] = field(default_factory=list)
    confidence: str = "medium"
    needs_review: bool = False
    model_certainty: str | None = None
    sed_support_prior: str = "model_only"
    evidence_warnings: list[str] = field(default_factory=list)
    target_match: str | None = None


@dataclass
class CaptionVerificationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    parsed_segments: list[CaptionSegment] = field(default_factory=list)
    required_slice_calls: int = 0
    observed_slice_calls: int = 0
    needs_review: bool = False
    warnings: list[str] = field(default_factory=list)
    normalizations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "required_slice_calls": self.required_slice_calls,
            "observed_slice_calls": self.observed_slice_calls,
            "segment_count": len(self.parsed_segments),
            "needs_review": self.needs_review,
            "warnings": list(self.warnings),
            "normalizations": list(self.normalizations),
        }


class TimestampedCaptionFrontend(Protocol):
    def propose_slices(self, context: dict[str, Any], *, min_slices: int) -> list[dict[str, Any]]:
        ...

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        ...

    def review_candidate_segments(
        self,
        context: dict[str, Any],
        *,
        candidates: list[dict[str, Any]],
        slice_records: list[dict[str, Any]],
        verification: CaptionVerificationResult,
    ) -> dict[str, Any]:
        ...


class GeminiTimestampedCaptionFrontend:
    """Gemini frontend for FlexSED-grounded timestamped audio captioning."""

    provider_name = GEMINI_MODE
    service_label = "gemini_with_flexsed_prior"
    source_label = "FlexSED-prior Gemini timestamped caption"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 32768,
        timeout: float = 120.0,
        client: GeminiClient | None = None,
    ) -> None:
        config = AudioEditConfig()
        self.client = client or GeminiClient(
            api_key=resolve_gemini_api_key(api_key) or config.omni_api_key,
            base_url=base_url or config.omni_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.telemetry: dict[str, Any] = {"slice_proposals": [], "review_calls": [], "candidate_rubrics": []}

    def propose_slices(self, context: dict[str, Any], *, min_slices: int) -> list[dict[str, Any]]:
        if min_slices <= 0:
            return []
        prompt_context = _compact_caption_prompt_context(context)
        user_text = _slice_proposal_user_text(prompt_context, min_slices=min_slices)
        record = {"prompt_payload": _try_json_loads(user_text), "min_slices": min_slices}
        try:
            data = self.client.call_json(
                system_prompt=SLICE_PROPOSAL_SYSTEM_PROMPT,
                user_text=user_text,
                audio_paths=[context["audio_path"]],
            )
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            self.telemetry.setdefault("slice_proposals", []).append(record)
            return []
        record.update({"status": "success", "raw_json": data})
        self.telemetry.setdefault("slice_proposals", []).append(record)
        raw = data.get("slices") if isinstance(data, dict) else None
        return [_normalize_slice_request(item) for item in raw or [] if isinstance(item, dict)]

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt_context = _compact_caption_prompt_context(context)
        user_text = _final_review_user_text(
            prompt_context,
            slice_records=slice_records,
            verification_feedback=verification_feedback or [],
        )
        self.telemetry.setdefault("review_calls", []).append(
            {
                "prompt_payload": _try_json_loads(user_text),
                "slice_audio_count": len(slice_records),
                "verification_feedback": list(verification_feedback or []),
            }
        )
        audio_paths = [context["audio_path"], *[record["output_audio_path"] for record in slice_records]]
        return self.client.call_json(
            system_prompt=FINAL_REVIEW_SYSTEM_PROMPT,
            user_text=user_text,
            audio_paths=audio_paths,
        )

    def review_candidate_segments(
        self,
        context: dict[str, Any],
        *,
        candidates: list[dict[str, Any]],
        slice_records: list[dict[str, Any]],
        verification: CaptionVerificationResult,
    ) -> dict[str, Any]:
        payload = {
            "instruction": context.get("instruction"),
            "duration_sec": context.get("duration_sec"),
            "verification_errors": verification.errors,
            "candidates": candidates,
        }
        record = {
            "prompt_payload": payload,
            "candidate_count": len(candidates),
            "slice_audio_count": len(slice_records),
        }
        try:
            data = self.client.call_json(
                system_prompt=CANDIDATE_RUBRIC_SYSTEM_PROMPT,
                user_text=json.dumps(payload, ensure_ascii=False, indent=2),
                audio_paths=[item["output_audio_path"] for item in slice_records],
            )
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            self.telemetry.setdefault("candidate_rubrics", []).append(record)
            return {"status": "failed", "error": record["error"], "judgements": []}
        record.update({"status": "success", "raw_json": data})
        self.telemetry.setdefault("candidate_rubrics", []).append(record)
        return data if isinstance(data, dict) else {"status": "invalid", "judgements": []}


class OpenAICompatibleTimestampedCaptionFrontend:
    """Provider-neutral frontend for timestamped audio captioning."""

    provider_name = QWEN3_OMNI_PROVIDER
    service_label = "qwen3_omni_with_flexsed_prior"
    source_label = "FlexSED-prior Qwen3-Omni timestamped caption"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int | None = None,
        timeout: float = 120.0,
        client: OpenAICompatibleChatClient | None = None,
        provider_name: str | None = None,
        provider_display_name: str | None = None,
    ) -> None:
        self.provider_name = provider_name or self.provider_name
        self.provider_display_name = provider_display_name or (
            "Qwen3-Omni"
            if self.provider_name == QWEN3_OMNI_PROVIDER
            else self.provider_name
        )
        if self.provider_name != QWEN3_OMNI_PROVIDER:
            self.service_label = f"{self.provider_name}_with_flexsed_prior"
            self.source_label = (
                f"FlexSED-prior {self.provider_display_name} timestamped caption"
            )
        self.model_name = resolve_openai_compatible_model(self.provider_name, model_name)
        self.client = client or OpenAICompatibleChatClient(
            api_key=resolve_openai_compatible_api_key(self.provider_name, api_key),
            base_url=resolve_openai_compatible_base_url(self.provider_name, base_url),
            model_name=self.model_name,
            provider_name=self.provider_name,
            temperature=temperature,
            max_tokens=_resolve_qwen3_timestamped_caption_max_tokens(max_tokens),
            timeout=timeout,
        )
        self.telemetry: dict[str, Any] = {"slice_proposals": [], "review_calls": [], "candidate_rubrics": []}

    def propose_slices(self, context: dict[str, Any], *, min_slices: int) -> list[dict[str, Any]]:
        if min_slices <= 0:
            return []
        user_text = _slice_proposal_user_text(context, min_slices=min_slices)
        record = {"prompt_payload": _try_json_loads(user_text), "min_slices": min_slices}
        try:
            data = self.client.call_json(
                system_prompt=_strong_omni_system_prompt(
                    SLICE_PROPOSAL_SYSTEM_PROMPT,
                    self.provider_display_name,
                ),
                user_text=user_text,
                audio_paths=[context["audio_path"]],
                response_format={"type": "json_object"},
                repair_schema_name="TimestampedCaptionSlicePlan",
            )
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            self.telemetry.setdefault("slice_proposals", []).append(record)
            return []
        record.update({"status": "success", "raw_json": data, "client_metrics": self.client.last_call_metrics})
        self.telemetry.setdefault("slice_proposals", []).append(record)
        raw = data.get("slices") if isinstance(data, dict) else None
        return [_normalize_slice_request(item) for item in raw or [] if isinstance(item, dict)]

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        user_text = _final_review_user_text(
            context,
            slice_records=slice_records,
            verification_feedback=verification_feedback or [],
        )
        record = {
            "prompt_payload": _try_json_loads(user_text),
            "slice_audio_count": len(slice_records),
            "verification_feedback": list(verification_feedback or []),
        }
        self.telemetry.setdefault("review_calls", []).append(record)
        audio_paths = [context["audio_path"], *[record["output_audio_path"] for record in slice_records]]
        data = self.client.call_json(
            system_prompt=_strong_omni_system_prompt(
                FINAL_REVIEW_SYSTEM_PROMPT,
                self.provider_display_name,
            ),
            user_text=user_text,
            audio_paths=audio_paths,
            response_format={"type": "json_object"},
            repair_schema_name="TimestampedCaptionReview",
        )
        record.update({"status": "success", "raw_json": data, "client_metrics": self.client.last_call_metrics})
        return data

    def review_candidate_segments(
        self,
        context: dict[str, Any],
        *,
        candidates: list[dict[str, Any]],
        slice_records: list[dict[str, Any]],
        verification: CaptionVerificationResult,
    ) -> dict[str, Any]:
        payload = {
            "instruction": context.get("instruction"),
            "duration_sec": context.get("duration_sec"),
            "verification_errors": verification.errors,
            "candidates": candidates,
        }
        record = {
            "prompt_payload": payload,
            "candidate_count": len(candidates),
            "slice_audio_count": len(slice_records),
        }
        try:
            data = self.client.call_json(
                system_prompt=_strong_omni_system_prompt(
                    CANDIDATE_RUBRIC_SYSTEM_PROMPT,
                    self.provider_display_name,
                ),
                user_text=json.dumps(payload, ensure_ascii=False, indent=2),
                audio_paths=[item["output_audio_path"] for item in slice_records],
                response_format={"type": "json_object"},
                repair_schema_name="TimestampedCaptionCandidateRubric",
            )
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            self.telemetry.setdefault("candidate_rubrics", []).append(record)
            return {"status": "failed", "error": record["error"], "judgements": []}
        record.update({"status": "success", "raw_json": data, "client_metrics": self.client.last_call_metrics})
        self.telemetry.setdefault("candidate_rubrics", []).append(record)
        return data if isinstance(data, dict) else {"status": "invalid", "judgements": []}


class Qwen3OmniTimestampedCaptionFrontend(
    OpenAICompatibleTimestampedCaptionFrontend
):
    """Backward-compatible Qwen3-Omni timestamped-caption name."""


def _invoke_frontend_provider(
    frontend: TimestampedCaptionFrontend,
    purpose: PerceptionCallPurpose,
    call: Callable[[], Any],
) -> Any:
    try:
        result = call()
    except Exception as exc:
        capture_frontend_provider_call(
            frontend,
            purpose,
            succeeded=False,
            error=exc,
        )
        raise
    capture_frontend_provider_call(
        frontend,
        purpose,
        succeeded=True,
    )
    return result


def _attach_frontend_invocation_trace(
    result: dict[str, Any],
    frontend: TimestampedCaptionFrontend,
    *,
    status: Literal["completed", "failed"],
) -> dict[str, Any]:
    trace = finalize_frontend_invocation_trace(frontend, status=status)
    if trace is not None:
        result["component_invocation_trace"] = trace.model_dump(mode="json")
    return result


def _candidate_rubric_failed(
    frontend: TimestampedCaptionFrontend,
    rubric_trace: dict[str, Any],
) -> bool:
    if frontend_provider_call_failed(
        frontend,
        "timestamped_caption.candidate_rubric",
    ):
        return True
    raw = rubric_trace.get("raw_output")
    if not isinstance(raw, dict):
        return False
    return str(raw.get("status") or "").strip().lower() in {
        "failed",
        "invalid",
        "unavailable",
    }


def run_timestamped_caption_review(
    *,
    audio_path: str | Path,
    instruction: str,
    output_dir: str | Path,
    event_labels: list[str] | None = None,
    use_flexsed_prior: bool = False,
    flexsed_prior: dict[str, Any] | None = None,
    flexsed_error: str | None = None,
    base_perception: dict[str, Any] | None = None,
    reviewer: TimestampedCaptionFrontend | None = None,
    max_review_attempts: int = 2,
    enable_candidate_rubric: bool = True,
    allow_slice_completion: bool = True,
    workflow_scope: str = "targeted",
    temporal_precision: str = "segment",
    require_local_boundaries: bool | None = None,
    slice_shortfall_mode: Literal["block", "warn"] = "warn",
    strict: bool = True,
    agentic_mode: str | None = None,
) -> dict[str, Any]:
    """Return a timestamped caption report with an optional FlexSED prior."""

    if slice_shortfall_mode not in {"block", "warn"}:
        raise ValueError(
            "slice_shortfall_mode must be 'block' or 'warn'"
        )
    resolved_agentic_mode = resolve_agentic_mode(agentic_mode)
    audio = Path(audio_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"timestamped caption audio does not exist: {audio}")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    context = _caption_context(
        audio,
        instruction=instruction,
        output_dir=output_root,
        event_labels=event_labels or [],
        use_flexsed_prior=use_flexsed_prior,
        flexsed_prior=flexsed_prior,
        flexsed_error=flexsed_error,
        base_perception=base_perception or {},
        agentic_mode=resolved_agentic_mode,
    )
    resolved_require_local_boundaries = (
        bool(event_labels)
        if require_local_boundaries is None
        else bool(require_local_boundaries)
    )
    context["workflow_scope"] = str(workflow_scope or "targeted")
    context["temporal_precision"] = str(temporal_precision or "segment")
    context["require_local_boundaries"] = resolved_require_local_boundaries
    context["slice_shortfall_mode"] = slice_shortfall_mode
    min_slices = required_caption_slice_count(context)
    frontend = reviewer or _default_timestamped_caption_frontend(resolved_agentic_mode)
    _set_timestamped_caption_frontend_context(context, frontend, resolved_agentic_mode)
    reset_frontend_invocation_trace(frontend)
    slice_requests_from_model: list[dict[str, Any]] = []
    # The first StrongOmni review sees the complete audio and the specialist
    # prior. Local clips are added only after candidate triage asks for them.
    slice_requests: list[dict[str, Any]] = []
    slice_records = _execute_slice_requests(audio, slice_requests, output_root / "slice_audio")

    raw_outputs: list[dict[str, Any]] = []
    review_attempts: list[dict[str, Any]] = []
    verification_history: list[dict[str, Any]] = []
    candidate_triage_history: list[dict[str, Any]] = []
    candidate_rubric_history: list[dict[str, Any]] = []
    feedback: list[str] = []
    local_escalation_attempted = False
    best_verification: CaptionVerificationResult | None = None
    best_raw: dict[str, Any] | None = None
    for attempt_index in range(max(1, max_review_attempts)):
        review_purpose: PerceptionCallPurpose = (
            "timestamped_caption.review"
            if attempt_index == 0
            else "timestamped_caption.review_after_verification"
        )
        raw = _invoke_frontend_provider(
            frontend,
            review_purpose,
            lambda: frontend.review(
                context,
                slice_records=slice_records,
                verification_feedback=feedback,
            ),
        )
        raw_outputs.append(raw)
        verification = verify_timestamped_caption_output(
            raw,
            duration_sec=context.get("duration_sec"),
            required_slice_calls=min_slices,
            slice_records=slice_records,
            flexsed_prior=flexsed_prior if use_flexsed_prior else None,
            instruction=instruction,
            require_local_boundaries=resolved_require_local_boundaries,
            slice_shortfall_mode=slice_shortfall_mode,
        )
        triage_batch = _caption_candidate_triage(context, verification)
        triage_snapshot = triage_batch.to_dict()
        candidate_triage_history.append(triage_snapshot)
        verification_history.append(verification.to_dict())
        review_attempts.append(
            {
                "attempt_index": len(review_attempts) + 1,
                "verification_feedback_input": list(feedback),
                "raw_output": raw,
                "verification": verification.to_dict(),
                "candidate_triage": triage_snapshot,
            }
        )
        if best_verification is None or len(verification.errors) < len(best_verification.errors):
            best_verification = verification
            best_raw = raw

        should_escalate_locally = (
            enable_candidate_rubric
            and not local_escalation_attempted
            and bool(triage_batch.local_review_ids)
            and attempt_index + 1 < max(1, max_review_attempts)
        )
        if should_escalate_locally:
            rubric_trace = _maybe_run_caption_candidate_rubric(
                frontend,
                context,
                audio=audio,
                output_root=output_root,
                verification=verification,
                slice_requests=slice_requests,
                slice_records=slice_records,
                triage_batch=triage_batch,
            )
            local_escalation_attempted = True
            if rubric_trace is not None:
                candidate_rubric_history.append(rubric_trace)
                review_attempts[-1]["candidate_rubric"] = rubric_trace
                if _candidate_rubric_failed(frontend, rubric_trace):
                    record_frontend_deterministic_fallback(
                        frontend,
                        from_route="timestamped_caption.candidate_rubric",
                        to_route="timestamped_caption.verification_feedback_only",
                        reason="candidate_rubric_failure_continued",
                    )
                feedback = [
                    *verification.errors,
                    *verification.warnings,
                    *_candidate_rubric_feedback(rubric_trace),
                ]
                continue

        if verification.passed:
            perception = _perception_from_caption(
                base_perception=base_perception or {},
                raw=raw,
                segments=verification.parsed_segments,
                context=context,
                verification=verification,
            )
            result = {
                "perception_report": perception,
                "timestamped_caption_review": _review_trace(
                    status="success",
                    context=context,
                    raw_outputs=raw_outputs,
                    verification=verification,
                    verification_history=verification_history,
                    review_attempts=review_attempts,
                    candidate_triage_history=candidate_triage_history,
                    candidate_rubric_history=candidate_rubric_history,
                    slice_requests_from_model=slice_requests_from_model,
                    final_slice_requests=slice_requests,
                    slice_records=slice_records,
                    frontend=frontend,
                ),
            }
            return _attach_frontend_invocation_trace(
                result,
                frontend,
                status="completed",
            )
        rubric_trace = (
            _maybe_run_caption_candidate_rubric(
                frontend,
                context,
                audio=audio,
                output_root=output_root,
                verification=verification,
                slice_requests=slice_requests,
                slice_records=slice_records,
                triage_batch=triage_batch,
            )
            if enable_candidate_rubric and not local_escalation_attempted
            else None
        )
        if rubric_trace is not None:
            local_escalation_attempted = True
            candidate_rubric_history.append(rubric_trace)
            review_attempts[-1]["candidate_rubric"] = rubric_trace
            if _candidate_rubric_failed(frontend, rubric_trace):
                record_frontend_deterministic_fallback(
                    frontend,
                    from_route="timestamped_caption.candidate_rubric",
                    to_route="timestamped_caption.verification_feedback_only",
                    reason="candidate_rubric_failure_continued",
                )
            feedback = [
                *verification.errors,
                *verification.warnings,
                *_candidate_rubric_feedback(rubric_trace),
            ]
        else:
            feedback = [*verification.errors, *verification.warnings]

    if best_verification is not None and best_verification.passed:
        perception = _perception_from_caption(
            base_perception=base_perception or {},
            raw=best_raw or {},
            segments=best_verification.parsed_segments,
            context=context,
            verification=best_verification,
        )
        result = {
            "perception_report": perception,
            "timestamped_caption_review": _review_trace(
                status="success_best_available",
                context=context,
                raw_outputs=raw_outputs,
                verification=best_verification,
                verification_history=verification_history,
                review_attempts=review_attempts,
                candidate_triage_history=candidate_triage_history,
                candidate_rubric_history=candidate_rubric_history,
                slice_requests_from_model=slice_requests_from_model,
                final_slice_requests=slice_requests,
                slice_records=slice_records,
                frontend=frontend,
            ),
        }
        return _attach_frontend_invocation_trace(
            result,
            frontend,
            status="completed",
        )

    if strict:
        finalize_frontend_invocation_trace(frontend, status="failed")
        error_text = "; ".join((best_verification.errors if best_verification else []) or ["caption review failed"])
        raise RuntimeError(f"timestamped caption review failed verification: {error_text}")

    best_verification = best_verification or CaptionVerificationResult(False, ["caption review failed"])
    perception = _perception_from_caption(
        base_perception=base_perception or {},
        raw=best_raw or {},
        segments=best_verification.parsed_segments,
        context=context,
        verification=best_verification,
    )
    result = {
        "perception_report": perception,
        "timestamped_caption_review": _review_trace(
            status="failed_verification",
            context=context,
            raw_outputs=raw_outputs,
            verification=best_verification,
            verification_history=verification_history,
            review_attempts=review_attempts,
            candidate_triage_history=candidate_triage_history,
            candidate_rubric_history=candidate_rubric_history,
            slice_requests_from_model=slice_requests_from_model,
            final_slice_requests=slice_requests,
            slice_records=slice_records,
            frontend=frontend,
        ),
    }
    return _attach_frontend_invocation_trace(
        result,
        frontend,
        status="failed",
    )


def verify_timestamped_caption_output(
    raw_output: dict[str, Any] | str,
    *,
    duration_sec: float | None,
    required_slice_calls: int,
    slice_records: list[dict[str, Any]],
    flexsed_prior: dict[str, Any] | None = None,
    instruction: str = "",
    require_local_boundaries: bool = False,
    slice_shortfall_mode: Literal["block", "warn"] = "warn",
) -> CaptionVerificationResult:
    data = _ensure_json_object(raw_output)
    errors: list[str] = []
    warnings: list[str] = []
    normalizations: list[dict[str, Any]] = []
    parsed: list[CaptionSegment] = []
    valid_slice_ids = {str(record.get("slice_id") or "") for record in slice_records}
    valid_slice_ids.discard("")
    needs_review = False
    raw_segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw_segments, list) or not raw_segments:
        explicit_uncertainty = data.get("uncertainty") if isinstance(data, dict) else None
        if isinstance(raw_segments, list) and explicit_uncertainty:
            warnings.append(
                "no event segment was resolved; explicit model uncertainty is preserved"
            )
            needs_review = True
        else:
            errors.append("segments must be a non-empty array unless uncertainty is explicit")
        raw_segments = []
    flexsed_present = bool(flexsed_prior and (flexsed_prior.get("timestamps") or flexsed_prior.get("events")))
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, dict):
            errors.append(f"segment {index} is not an object")
            continue
        segment = _parse_caption_segment(item, index, errors)
        if segment is None:
            continue
        if duration_sec is not None and (segment.start_sec < -0.01 or segment.end_sec > float(duration_sec) + 0.25):
            errors.append(f"segment {index} time span falls outside audio duration")
            continue
        invalid_slices = [slice_id for slice_id in segment.evidence_slice_ids if slice_id not in valid_slice_ids]
        if invalid_slices:
            errors.append(f"segment {index} references unknown slice ids: {invalid_slices}")
            continue
        if not segment.caption and not segment.events:
            errors.append(f"segment {index} has neither caption nor events")
            continue
        source = segment.evidence_source.lower()
        confidence = segment.confidence.lower()
        has_slice_evidence = bool(segment.evidence_slice_ids)
        known_flexsed_events = _flexsed_event_names(flexsed_prior)
        invalid_flexsed_refs = sorted(
            ref
            for ref in segment.flexsed_event_refs
            if ref.casefold() not in known_flexsed_events
        )
        if invalid_flexsed_refs:
            segment.flexsed_event_refs = [
                ref
                for ref in segment.flexsed_event_refs
                if ref.casefold() in known_flexsed_events
            ]
            warnings.append(
                f"segment {index} dropped unknown FlexSED provenance refs: "
                f"{invalid_flexsed_refs}"
            )
            normalizations.append(
                {
                    "segment_index": index,
                    "field": "flexsed_event_refs",
                    "action": "drop_unadmitted_provenance_refs",
                    "removed": invalid_flexsed_refs,
                }
            )
        has_flexsed_overlap = any(
            _event_has_flexsed_overlap(event, segment.start_sec, segment.end_sec, flexsed_prior)
            for event in [*segment.events, *segment.flexsed_event_refs]
        )
        _normalize_caption_evidence_support(
            segment,
            index=index,
            data=data,
            flexsed_present=flexsed_present,
            has_flexsed_overlap=has_flexsed_overlap,
            has_slice_evidence=has_slice_evidence,
            warnings=warnings,
            normalizations=normalizations,
        )
        source = segment.evidence_source.lower()
        confidence = segment.confidence.lower()
        if _is_model_only_source(source):
            segment.needs_review = True
            needs_review = True
        elif _is_flexsed_source(source) and not flexsed_present:
            errors.append(f"segment {index} claims {source} but no FlexSED prior is available")
        elif confidence == "high" and not (has_flexsed_overlap or has_slice_evidence):
            segment.needs_review = True
            needs_review = True
            _downgrade_caption_segment_confidence(
                segment,
                index=index,
                reason="high confidence but lacks FlexSED overlap or direct slice evidence",
                warnings=warnings,
                normalizations=normalizations,
            )
        elif _is_flexsed_source(source) and flexsed_present and not (
            has_flexsed_overlap or segment.flexsed_event_refs or has_slice_evidence
        ):
            errors.append(f"segment {index} claims {source} without FlexSED event support")
        parsed.append(segment)
    if len(slice_records) < required_slice_calls:
        message = (
            f"complex audio required at least {required_slice_calls} "
            f"slice_audio calls, observed {len(slice_records)}"
        )
        warnings.append(message)
    if (
        require_local_boundaries
        and len(parsed) == 1
        and duration_sec
        and duration_sec > 5.0
        and parsed[0].end_sec - parsed[0].start_sec >= 0.9 * duration_sec
    ):
        parsed[0].needs_review = True
        needs_review = True
        _record_caption_evidence_warning(
            parsed[0],
            index=1,
            warning=(
                "local edit requiring temporal boundaries returned a single "
                "whole-audio segment; boundary remains unresolved"
            ),
            warnings=warnings,
        )
    elif (
        len(parsed) == 1
        and duration_sec
        and duration_sec > 5.0
        and parsed[0].end_sec - parsed[0].start_sec >= 0.9 * duration_sec
    ):
        warnings.append(
            "perception returned a single whole-audio segment; this is usable only for global or coarse decisions"
        )
    return CaptionVerificationResult(
        passed=not errors,
        errors=errors,
        parsed_segments=parsed,
        required_slice_calls=required_slice_calls,
        observed_slice_calls=len(slice_records),
        needs_review=needs_review,
        warnings=warnings,
        normalizations=normalizations,
    )


def required_caption_slice_count(context: dict[str, Any]) -> int:
    return 0


def build_event_candidates(instruction: str, event_labels: list[str] | None = None, *, max_events: int = 48) -> list[str]:
    values = [str(item).strip() for item in event_labels or [] if str(item).strip()]
    values.extend(_instruction_event_phrases(instruction))
    if not values:
        values.extend(_load_event_bank())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped[:max_events]


def _default_timestamped_caption_frontend(agentic_mode: str) -> TimestampedCaptionFrontend:
    if perception_provider_name(agentic_mode) == QWEN3_OMNI_PROVIDER:
        return Qwen3OmniTimestampedCaptionFrontend()
    return GeminiTimestampedCaptionFrontend()


def _set_timestamped_caption_frontend_context(
    context: dict[str, Any],
    frontend: TimestampedCaptionFrontend,
    agentic_mode: str,
) -> None:
    provider = str(getattr(frontend, "provider_name", "") or perception_provider_name(agentic_mode))
    use_flexsed = bool(context.get("use_flexsed_prior"))
    service = getattr(frontend, "service_label", None)
    if not service:
        service = f"{provider}_with_flexsed_prior" if use_flexsed else provider
    source = getattr(frontend, "source_label", None)
    if not source:
        if provider == QWEN3_OMNI_PROVIDER:
            source = "FlexSED-prior Qwen3-Omni timestamped caption" if use_flexsed else "Qwen3-Omni timestamped caption"
        else:
            source = "FlexSED-prior Gemini timestamped caption" if use_flexsed else "Gemini timestamped caption"
    context["timestamped_caption_review_provider"] = provider
    context["timestamped_caption_review_service"] = str(service)
    context["timestamped_caption_source"] = str(source)


def _caption_context(
    audio: Path,
    *,
    instruction: str,
    output_dir: Path,
    event_labels: list[str],
    use_flexsed_prior: bool,
    flexsed_prior: dict[str, Any] | None,
    flexsed_error: str | None,
    base_perception: dict[str, Any],
    agentic_mode: str,
) -> dict[str, Any]:
    return {
        "audio_path": str(audio),
        "instruction": instruction,
        "output_dir": str(output_dir),
        "agentic_mode": agentic_mode,
        "duration_sec": audio_duration_sec(audio),
        "event_candidates": build_event_candidates(instruction, event_labels),
        "use_flexsed_prior": use_flexsed_prior,
        "flexsed_prior": flexsed_prior or {},
        "flexsed_prior_unavailable": bool(use_flexsed_prior and flexsed_error),
        "flexsed_error": flexsed_error,
        "base_perception": base_perception,
    }


def _complete_slice_requests(context: dict[str, Any], requests: list[dict[str, Any]], min_slices: int) -> list[dict[str, Any]]:
    valid = [_normalize_slice_request(item) for item in requests if item]
    valid = [item for item in valid if item["offset"] > item["onset"] and item["reason"]]
    if min_slices <= 0:
        return valid
    return valid[: max(min_slices, len(valid))]


def _normalize_slice_request(item: dict[str, Any]) -> dict[str, Any]:
    refs = item.get("flexsed_event_refs") or item.get("event_refs") or []
    if isinstance(refs, str):
        refs = [refs]
    candidate_ids = item.get("candidate_ids") or item.get("candidate_id") or []
    if isinstance(candidate_ids, str):
        candidate_ids = [candidate_ids]
    return {
        "onset": float(item.get("onset", item.get("start", item.get("start_sec", 0.0)))),
        "offset": float(item.get("offset", item.get("end", item.get("end_sec", 0.0)))),
        "reason": str(item.get("reason") or item.get("uncertainty") or "").strip(),
        "flexsed_event_refs": [str(ref).strip() for ref in refs if str(ref).strip()],
        "candidate_ids": [str(ref).strip() for ref in candidate_ids if str(ref).strip()],
    }


def _heuristic_slice_requests(context: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _flexsed_candidate_slice_requests(
    context: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    prior = context.get("flexsed_prior")
    if not isinstance(prior, dict):
        return []
    duration = float(context.get("duration_sec") or 0.0)
    candidates: list[dict[str, Any]] = []
    for candidate in prior.get("candidate_windows") or []:
        if not isinstance(candidate, dict):
            continue
        start = _optional_float(candidate.get("start_sec"))
        end = _optional_float(candidate.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        event = str(candidate.get("event") or "audio event").strip()
        candidate_id = str(candidate.get("candidate_id") or event).strip()
        candidates.append(
            {
                "onset": round(max(0.0, start), 3),
                "offset": round(min(duration, end) if duration > 0 else end, 3),
                "candidate_ids": [candidate_id],
                "flexsed_event_refs": [event],
            }
        )
    candidates.sort(key=lambda item: (float(item["onset"]), float(item["offset"])))
    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        if merged and float(candidate["onset"]) <= float(merged[-1]["offset"]) + 0.08:
            merged[-1]["offset"] = max(
                float(merged[-1]["offset"]),
                float(candidate["offset"]),
            )
            merged[-1]["candidate_ids"].extend(candidate["candidate_ids"])
            merged[-1]["flexsed_event_refs"] = list(
                dict.fromkeys(
                    [
                        *merged[-1]["flexsed_event_refs"],
                        *candidate["flexsed_event_refs"],
                    ]
                )
            )
            continue
        merged.append(candidate)
    requests: list[dict[str, Any]] = []
    for item in merged[:limit]:
        candidate_ids = list(dict.fromkeys(item.pop("candidate_ids")))
        item["reason"] = (
            f"Review FlexSED soft peak proposals {candidate_ids}; "
            "they are not accepted evidence until this slice is heard."
        )
        requests.append(item)
    return requests


def _slice_around(
    center: float,
    duration: float,
    *,
    reason: str,
    flexsed_event_refs: list[str] | None = None,
) -> dict[str, Any]:
    width = 8.0
    if duration and duration < width:
        width = max(1.0, duration)
    start = max(0.0, center - width / 2.0)
    end = start + width
    if duration > 0 and end > duration:
        end = duration
        start = max(0.0, end - width)
    return {
        "onset": round(start, 3),
        "offset": round(end, 3),
        "reason": reason,
        "flexsed_event_refs": list(flexsed_event_refs or []),
    }


def _overlaps_existing_slice(item: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    start = float(item["onset"])
    end = float(item["offset"])
    for other in existing:
        overlap = max(0.0, min(end, float(other["offset"])) - max(start, float(other["onset"])))
        if overlap >= 0.8 * min(end - start, float(other["offset"]) - float(other["onset"])):
            return True
    return False


def _execute_slice_requests(
    audio: Path,
    requests: list[dict[str, Any]],
    output_dir: Path,
    *,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    records = []
    for index, request in enumerate(requests, start=start_index):
        slice_id = f"slice_{index:02d}"
        record = slice_audio(
            audio,
            onset=float(request["onset"]),
            offset=float(request["offset"]),
            output_dir=output_dir,
            reason=str(request["reason"]),
            clip_id=f"{slice_id}_{request['onset']:.2f}_{request['offset']:.2f}",
        )
        record["slice_id"] = slice_id
        record["flexsed_event_refs"] = list(request.get("flexsed_event_refs") or [])
        record["candidate_ids"] = list(request.get("candidate_ids") or [])
        records.append(record)
    return records


def _caption_candidate_triage(
    context: dict[str, Any],
    verification: CaptionVerificationResult,
) -> EvidenceTriageBatch:
    return triage_candidate_evidence_batch(
        _caption_triage_candidates(context, verification),
        require_local_boundaries=bool(context.get("require_local_boundaries")),
        execution_critical=bool(context.get("require_local_boundaries")),
    )


def _caption_triage_candidates(
    context: dict[str, Any],
    verification: CaptionVerificationResult,
    *,
    max_candidates: int = 12,
) -> list[dict[str, Any]]:
    duration = float(context.get("duration_sec") or 0.0)
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(verification.parsed_segments, start=1):
        candidate = _caption_segment_candidate(
            index - 1,
            segment,
            source="strong_omni_global_review",
        )
        candidate["candidate_id"] = f"segment_{index:03d}"
        candidate["needs_review"] = segment.needs_review
        candidate["model_certainty"] = segment.model_certainty or segment.confidence
        candidate["sed_support_prior"] = segment.sed_support_prior
        candidate["boundary_quality"] = (
            "coarse"
            if _caption_segment_is_whole_audio(segment, duration)
            and bool(context.get("require_local_boundaries"))
            else "stable"
        )
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            return candidates

    for candidate in _caption_prior_soft_candidates(context, max_candidates=max_candidates):
        if _prior_candidate_covered_by_segment(candidate, verification.parsed_segments):
            continue
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _caption_prior_soft_candidates(
    context: dict[str, Any],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    prior = context.get("flexsed_prior")
    if not isinstance(prior, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(prior.get("candidate_windows") or [], start=1):
        if not isinstance(item, dict):
            continue
        start = _optional_float(item.get("start_sec"))
        end = _optional_float(item.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        event = str(item.get("event") or "audio event").strip()
        candidate_id = str(
            item.get("candidate_id") or f"flexsed_soft_{index:03d}"
        ).strip()
        requires_review = bool(item.get("requires_model_review", True))
        support_kind = str(item.get("support_kind") or "").strip().lower()
        candidates.append(
            {
                "candidate_id": candidate_id,
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "caption": f"specialist prior candidate {event}".strip(),
                "events": [event] if event else [],
                "evidence_source": "flexsed_prior",
                "flexsed_event_refs": [event] if event else [],
                "confidence": item.get("calibrated_confidence")
                or item.get("peak_score")
                or item.get("mean_score")
                or "medium",
                "requires_model_review": requires_review,
                "boundary_quality": (
                    "uncertain"
                    if requires_review or "soft" in support_kind or "candidate" in support_kind
                    else "stable"
                ),
                "source": "specialist_soft_prior",
                "support_kind": support_kind,
            }
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def _caption_segment_is_whole_audio(segment: CaptionSegment, duration: float) -> bool:
    if duration <= 5.0:
        return False
    return segment.end_sec - segment.start_sec >= 0.9 * duration


def _prior_candidate_covered_by_segment(
    candidate: dict[str, Any],
    segments: list[CaptionSegment],
) -> bool:
    start = _optional_float(candidate.get("start_sec"))
    end = _optional_float(candidate.get("end_sec"))
    if start is None or end is None:
        return False
    candidate_events = {str(item).casefold() for item in candidate.get("events") or []}
    candidate_events.update(str(item).casefold() for item in candidate.get("flexsed_event_refs") or [])
    for segment in segments:
        if max(0.0, min(end, segment.end_sec) - max(start, segment.start_sec)) <= 0.05:
            continue
        segment_events = {str(item).casefold() for item in segment.events}
        segment_events.update(str(item).casefold() for item in segment.flexsed_event_refs)
        if not candidate_events or candidate_events.intersection(segment_events):
            return True
    return False


def _slice_request_for_triage_candidate(
    candidate: dict[str, Any],
    *,
    review_reasons: list[str],
    duration: float,
) -> dict[str, Any] | None:
    start = _optional_float(candidate.get("start_sec"))
    end = _optional_float(candidate.get("end_sec"))
    if start is None or end is None or end <= start:
        return None
    padding = 0.0
    onset = max(0.0, start - padding)
    offset = end + padding
    if duration > 0:
        offset = min(duration, offset)
    candidate_id = str(candidate.get("candidate_id") or "candidate").strip()
    return {
        "onset": round(onset, 3),
        "offset": round(offset, 3),
        "reason": (
            f"Selective local review for {candidate_id}: "
            f"{', '.join(review_reasons) or 'candidate requires adjudication'}."
        ),
        "flexsed_event_refs": list(candidate.get("flexsed_event_refs") or []),
        "candidate_ids": [candidate_id] if candidate_id else [],
    }


def _maybe_run_caption_candidate_rubric(
    frontend: TimestampedCaptionFrontend,
    context: dict[str, Any],
    *,
    audio: Path,
    output_root: Path,
    verification: CaptionVerificationResult,
    slice_requests: list[dict[str, Any]],
    slice_records: list[dict[str, Any]],
    triage_batch: EvidenceTriageBatch | None = None,
) -> dict[str, Any] | None:
    if not _verification_needs_caption_rubric(verification, triage_batch=triage_batch):
        return None
    candidates = _caption_rubric_candidates(
        context,
        verification,
        triage_batch=triage_batch,
    )
    if not candidates:
        return None

    duration = float(context.get("duration_sec") or 0.0)
    new_requests: list[dict[str, Any]] = []
    for candidate in candidates:
        existing_ids = _overlapping_slice_ids(candidate, slice_records)
        if existing_ids:
            candidate["evidence_slice_ids"] = existing_ids
            continue
        request = _slice_request_for_triage_candidate(
            candidate,
            review_reasons=list(candidate.get("triage_reasons") or []),
            duration=duration,
        )
        if request is None:
            continue
        if _overlaps_existing_slice(request, slice_requests):
            candidate["evidence_slice_ids"] = _overlapping_slice_ids(candidate, slice_records)
            continue
        _merge_or_append_slice_request(new_requests, request)

    if new_requests:
        slice_requests.extend(new_requests)
        new_records = _execute_slice_requests(
            audio,
            new_requests,
            output_root / "slice_audio",
            start_index=len(slice_records) + 1,
        )
        slice_records.extend(new_records)

    for candidate in candidates:
        candidate["evidence_slice_ids"] = _overlapping_slice_ids(candidate, slice_records)

    reviewer = getattr(frontend, "review_candidate_segments", None)
    if not callable(reviewer):
        return {
            "status": "unavailable",
            "reason": "frontend does not implement candidate segment rubric review",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "verification": verification.to_dict(),
            "triage": triage_batch.to_dict() if triage_batch is not None else None,
        }
    raw = _invoke_frontend_provider(
        frontend,
        "timestamped_caption.candidate_rubric",
        lambda: reviewer(
            context,
            candidates=candidates,
            slice_records=slice_records,
            verification=verification,
        ),
    )
    judgements = _normalize_candidate_rubric_judgements(raw)
    return {
        "status": "completed",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "judgements": judgements,
        "raw_output": raw,
        "verification": verification.to_dict(),
        "triage": triage_batch.to_dict() if triage_batch is not None else None,
    }


def _verification_needs_caption_rubric(
    verification: CaptionVerificationResult,
    *,
    triage_batch: EvidenceTriageBatch | None = None,
) -> bool:
    if triage_batch is not None and triage_batch.local_review_ids:
        return True
    if verification.needs_review:
        return True
    return any(
        "high confidence but lacks FlexSED overlap or direct slice evidence" in error
        or "whole-audio segment" in error
        or "segments must be a non-empty array" in error
        for error in verification.errors
    )


def _caption_rubric_candidates(
    context: dict[str, Any],
    verification: CaptionVerificationResult,
    *,
    triage_batch: EvidenceTriageBatch | None = None,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if triage_batch is not None:
        for decision in triage_batch.decisions:
            if not decision.should_slice_locally:
                continue
            candidate = dict(decision.candidate)
            candidate["candidate_id"] = decision.candidate_id or candidate.get("candidate_id") or f"candidate_{len(candidates) + 1:03d}"
            candidate["triage_reasons"] = list(decision.review_reasons)
            candidate["triage_issues"] = [issue.to_dict() for issue in decision.issues]
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates
        return candidates

    for segment in verification.parsed_segments:
        if not segment.needs_review:
            continue
        candidates.append(_caption_segment_candidate(len(candidates), segment, source="verified_output"))
        if len(candidates) >= max_candidates:
            return candidates

    if not candidates:
        for event, start, end in _flexsed_peak_windows(context.get("flexsed_prior") or {}, limit=max_candidates):
            segment = CaptionSegment(
                start_sec=start,
                end_sec=end,
                caption=f"candidate {event}",
                events=[event],
                evidence_source="flexsed_prior",
                flexsed_event_refs=[event],
                confidence="medium",
            )
            candidates.append(_caption_segment_candidate(len(candidates), segment, source="flexsed_prior"))
    return candidates[:max_candidates]


def _merge_or_append_slice_request(
    requests: list[dict[str, Any]],
    request: dict[str, Any],
) -> None:
    start = float(request["onset"])
    end = float(request["offset"])
    for existing in requests:
        existing_start = float(existing["onset"])
        existing_end = float(existing["offset"])
        if start > existing_end + 0.08 or end < existing_start - 0.08:
            continue
        existing["onset"] = round(min(existing_start, start), 3)
        existing["offset"] = round(max(existing_end, end), 3)
        existing["flexsed_event_refs"] = _dedupe_strings(
            [
                *list(existing.get("flexsed_event_refs") or []),
                *list(request.get("flexsed_event_refs") or []),
            ]
        )
        existing["candidate_ids"] = _dedupe_strings(
            [
                *list(existing.get("candidate_ids") or []),
                *list(request.get("candidate_ids") or []),
            ]
        )
        existing["reason"] = f"{existing.get('reason', '')} {request.get('reason', '')}".strip()
        return
    requests.append(request)


def _caption_segment_candidate(index: int, segment: CaptionSegment, *, source: str) -> dict[str, Any]:
    return {
        "candidate_id": f"candidate_{index + 1:03d}",
        "start_sec": round(segment.start_sec, 6),
        "end_sec": round(segment.end_sec, 6),
        "caption": segment.caption,
        "events": list(segment.events),
        "evidence_source": segment.evidence_source,
        "flexsed_event_refs": list(segment.flexsed_event_refs),
        "confidence": segment.confidence,
        "target_match": segment.target_match,
        "source": source,
    }


def _overlapping_slice_ids(candidate: dict[str, Any], slice_records: list[dict[str, Any]]) -> list[str]:
    try:
        start = float(candidate["start_sec"])
        end = float(candidate["end_sec"])
    except Exception:
        return []
    ids: list[str] = []
    for record in slice_records:
        slice_id = str(record.get("slice_id") or "")
        if not slice_id:
            continue
        onset = _optional_float(record.get("onset"))
        offset = _optional_float(record.get("offset"))
        if onset is None or offset is None:
            continue
        if max(0.0, min(end, offset) - max(start, onset)) > 0.05:
            ids.append(slice_id)
    return ids


def _normalize_candidate_rubric_judgements(raw: Any) -> list[dict[str, Any]]:
    data = _ensure_json_object(raw)
    raw_items = data.get("judgements") if isinstance(data, dict) else None
    if not isinstance(raw_items, list):
        return []
    judgements: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        answer = str(item.get("answer") or "").strip().lower()
        if answer not in {"yes", "no", "uncertain"}:
            answer = "uncertain"
        judgements.append(
            {
                "candidate_id": str(item.get("candidate_id") or "").strip(),
                "answer": answer,
                "confidence": str(item.get("confidence") or "medium").strip().lower(),
                "reason": str(item.get("reason") or "").strip(),
                "evidence_slice_ids": _string_list(item.get("evidence_slice_ids") or []),
            }
        )
    return judgements


def _candidate_rubric_feedback(trace: dict[str, Any]) -> list[str]:
    if trace.get("status") != "completed":
        return [f"candidate_rubric_status: {trace.get('status')}; {trace.get('reason') or trace.get('error') or ''}".strip()]
    feedback: list[str] = []
    for judgement in trace.get("judgements") or []:
        if not isinstance(judgement, dict):
            continue
        answer = judgement.get("answer")
        candidate_id = judgement.get("candidate_id")
        slices = judgement.get("evidence_slice_ids") or []
        reason = judgement.get("reason") or ""
        if answer == "yes":
            feedback.append(
                f"rubric_confirmed {candidate_id}: use direct slice evidence {slices}; {reason}"
            )
        elif answer == "no":
            feedback.append(f"rubric_rejected {candidate_id}: {reason}")
        else:
            feedback.append(f"rubric_uncertain {candidate_id}: {reason}")
    return feedback or ["candidate_rubric_completed_no_judgements"]


def _perception_from_caption(
    *,
    base_perception: dict[str, Any],
    raw: dict[str, Any],
    segments: list[CaptionSegment],
    context: dict[str, Any],
    verification: CaptionVerificationResult,
) -> dict[str, Any]:
    base = dict(base_perception)
    for request_field in (
        "candidate_rubric_judgements",
        "candidate_segments",
        "selected_spans",
        "selected_speaker_ids",
        "selected_speaker_refs",
        "target_bindings",
    ):
        base.pop(request_field, None)
    segment_dicts = [
        _without_none({
            "start_sec": round(segment.start_sec, 6),
            "end_sec": round(segment.end_sec, 6),
            "caption": segment.caption,
            "events": segment.events,
            "evidence_source": segment.evidence_source,
            "flexsed_event_refs": segment.flexsed_event_refs,
            "evidence_slice_ids": segment.evidence_slice_ids,
            "confidence": segment.confidence,
            "model_certainty": segment.model_certainty or segment.confidence,
            "sed_support_prior": segment.sed_support_prior,
            "evidence_warnings": list(segment.evidence_warnings),
            "needs_review": segment.needs_review,
            "source": "timestamped_caption_review",
            "target_match": segment.target_match,
        })
        for segment in segments
    ]
    base["audio_caption_segments"] = segment_dicts
    base.setdefault("audio_events", [])
    base["audio_events"] = [*list(base.get("audio_events") or []), {"tool": "timestamped_caption", "segments": segment_dicts}]
    if raw.get("background"):
        base["background"] = raw.get("background")
    notable = list(base.get("notable_events") or [])
    for event in raw.get("notable_events") or []:
        if event not in notable:
            notable.append(str(event))
    base["notable_events"] = notable
    uncertainty = _uncertainty_items(base.get("uncertainty"))
    for item in _uncertainty_items(raw.get("uncertainty")):
        text = str(item)
        if text not in uncertainty:
            uncertainty.append(text)
    if context.get("flexsed_prior_unavailable"):
        uncertainty.append(f"FlexSED prior unavailable: {context.get('flexsed_error')}")
    for warning in verification.warnings:
        if warning not in uncertainty:
            uncertainty.append(warning)
    if verification.needs_review:
        uncertainty.append("Model-only audio event captions need review before destructive edits.")
    base["uncertainty"] = uncertainty
    base.setdefault("services", {})
    if isinstance(base["services"], dict):
        base["services"]["timestamped_caption_review"] = context.get(
            "timestamped_caption_review_service", "gemini_with_flexsed_prior"
        )
    base["timestamped_caption"] = {
        "source": context.get("timestamped_caption_source", "FlexSED-prior Gemini timestamped caption"),
        "provider": context.get("timestamped_caption_review_provider", GEMINI_MODE),
        "use_flexsed_prior": context.get("use_flexsed_prior"),
        "flexsed_prior_unavailable": context.get("flexsed_prior_unavailable"),
        "needs_review": verification.needs_review,
        "warnings": list(verification.warnings),
        "normalizations": list(verification.normalizations),
        "audio_duration_sec": context.get("duration_sec"),
    }
    return base


def _uncertainty_items(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _review_trace(
    *,
    status: str,
    context: dict[str, Any],
    raw_outputs: list[dict[str, Any]],
    verification: CaptionVerificationResult,
    verification_history: list[dict[str, Any]],
    review_attempts: list[dict[str, Any]],
    candidate_triage_history: list[dict[str, Any]],
    candidate_rubric_history: list[dict[str, Any]],
    slice_requests_from_model: list[dict[str, Any]],
    final_slice_requests: list[dict[str, Any]],
    slice_records: list[dict[str, Any]],
    frontend: TimestampedCaptionFrontend,
) -> dict[str, Any]:
    return {
        "status": status,
        "raw_outputs": raw_outputs,
        "verification": verification.to_dict(),
        "verification_history": verification_history,
        "review_attempts": review_attempts,
        "candidate_triage_history": candidate_triage_history,
        "candidate_rubric_history": candidate_rubric_history,
        "review_context": _audit_context(context),
        "slice_proposal_prompt_payload": _try_json_loads(
            _slice_proposal_user_text(context, min_slices=verification.required_slice_calls)
        )
        if verification.required_slice_calls > 0
        else None,
        "slice_requests_from_model": slice_requests_from_model,
        "final_slice_requests": final_slice_requests,
        "slice_audio_calls": slice_records,
        "required_slice_calls": verification.required_slice_calls,
        "flexsed_prior": context.get("flexsed_prior") or {},
        "flexsed_prior_unavailable": context.get("flexsed_prior_unavailable"),
        "frontend_provider": context.get("timestamped_caption_review_provider"),
        "frontend_service": context.get("timestamped_caption_review_service"),
        "agentic_mode": context.get("agentic_mode"),
        "frontend_telemetry": getattr(frontend, "telemetry", {}),
    }


def _audit_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_path": context.get("audio_path"),
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "event_candidates": context.get("event_candidates"),
        "agentic_mode": context.get("agentic_mode"),
        "review_provider": context.get("timestamped_caption_review_provider"),
        "review_service": context.get("timestamped_caption_review_service"),
        "use_flexsed_prior": context.get("use_flexsed_prior"),
        "flexsed_prior_unavailable": context.get("flexsed_prior_unavailable"),
        "flexsed_error": context.get("flexsed_error"),
    }


def _parse_caption_segment(item: dict[str, Any], index: int, errors: list[str]) -> CaptionSegment | None:
    start = _optional_float(item.get("start_sec", item.get("start")))
    end = _optional_float(item.get("end_sec", item.get("end")))
    if start is None or end is None or end <= start:
        errors.append(f"segment {index} has invalid time span")
        return None
    caption = str(item.get("caption") or item.get("description") or "").strip()
    events = _string_list(item.get("events") or item.get("event") or [])
    source = str(item.get("evidence_source") or "model_only").strip() or "model_only"
    confidence = str(item.get("confidence") or "medium").strip().lower()
    flexsed_refs = _string_list(item.get("flexsed_event_refs") or [])
    slice_ids = _string_list(item.get("evidence_slice_ids") or [])
    target_match = str(item.get("target_match") or "").strip().lower() or None
    if target_match not in {None, "yes", "no", "uncertain"}:
        target_match = "uncertain"
    return CaptionSegment(
        start_sec=start,
        end_sec=end,
        caption=caption,
        events=events,
        evidence_source=source,
        flexsed_event_refs=flexsed_refs,
        evidence_slice_ids=slice_ids,
        confidence=confidence,
        model_certainty=confidence,
        target_match=target_match,
    )


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _normalize_caption_evidence_support(
    segment: CaptionSegment,
    *,
    index: int,
    data: dict[str, Any],
    flexsed_present: bool,
    has_flexsed_overlap: bool,
    has_slice_evidence: bool,
    warnings: list[str],
    normalizations: list[dict[str, Any]],
) -> None:
    original_source = str(segment.evidence_source or "model_only").strip().lower() or "model_only"
    original_confidence = str(segment.confidence or "medium").strip().lower() or "medium"
    segment.evidence_source = original_source
    segment.confidence = original_confidence
    segment.model_certainty = segment.model_certainty or original_confidence
    segment.sed_support_prior = _caption_segment_sed_support_prior(
        segment,
        flexsed_present=flexsed_present,
        has_flexsed_overlap=has_flexsed_overlap,
        has_slice_evidence=has_slice_evidence,
    )

    if _is_flexsed_source(original_source) and segment.sed_support_prior == "model_only":
        _record_caption_evidence_warning(
            segment,
            index=index,
            warning=(
                f"segment {index} claimed {original_source} but lacked FlexSED support; "
                "downgraded to model_only evidence"
            ),
            warnings=warnings,
        )
        segment.evidence_source = "model_only"
        segment.needs_review = True
        normalizations.append(
            {
                "segment_index": index,
                "field": "evidence_source",
                "from": original_source,
                "to": "model_only",
                "reason": "claimed FlexSED support was not present in the supplied prior",
            }
        )

    if _is_model_only_source(segment.evidence_source):
        segment.needs_review = True
        if segment.confidence == "high":
            _downgrade_caption_segment_confidence(
                segment,
                index=index,
                reason=f"{segment.evidence_source} evidence lacks FlexSED or direct slice support",
                warnings=warnings,
                normalizations=normalizations,
            )
        if not _uncertainty_mentions_model_only(data, segment):
            _record_caption_evidence_warning(
                segment,
                index=index,
                warning=(
                    f"segment {index} is {segment.evidence_source}; FlexSED did not support this event span"
                ),
                warnings=warnings,
            )
            normalizations.append(
                {
                    "segment_index": index,
                    "field": "uncertainty",
                    "from": "missing",
                    "to": "model_only_missing_flexsed_support_warning",
                    "reason": "model-only event span needs an explicit missing-FlexSED audit note",
                }
            )


def _caption_segment_sed_support_prior(
    segment: CaptionSegment,
    *,
    flexsed_present: bool,
    has_flexsed_overlap: bool,
    has_slice_evidence: bool,
) -> str:
    if has_flexsed_overlap:
        return "flexsed_overlap"
    if has_slice_evidence:
        return "slice_evidence"
    return "model_only"


def _downgrade_caption_segment_confidence(
    segment: CaptionSegment,
    *,
    index: int,
    reason: str,
    warnings: list[str],
    normalizations: list[dict[str, Any]],
) -> None:
    original_confidence = str(segment.confidence or "medium").strip().lower() or "medium"
    if original_confidence != "high":
        return
    segment.confidence = "medium"
    segment.needs_review = True
    _record_caption_evidence_warning(
        segment,
        index=index,
        warning=f"segment {index} model_certainty=high but {reason}; downgraded confidence to medium",
        warnings=warnings,
    )
    normalizations.append(
        {
            "segment_index": index,
            "field": "confidence",
            "from": "high",
            "to": "medium",
            "reason": reason,
        }
    )


def _record_caption_evidence_warning(
    segment: CaptionSegment,
    *,
    index: int,
    warning: str,
    warnings: list[str],
) -> None:
    if warning not in warnings:
        warnings.append(warning)
    if warning not in segment.evidence_warnings:
        segment.evidence_warnings.append(warning)


def _event_has_flexsed_overlap(event: str, start: float, end: float, flexsed_prior: dict[str, Any] | None) -> bool:
    if not flexsed_prior:
        return False
    timestamps = flexsed_prior.get("timestamps") if isinstance(flexsed_prior, dict) else None
    if not isinstance(timestamps, dict):
        return False
    event_key = event.casefold()
    for prior_event, spans in timestamps.items():
        if str(prior_event).casefold() != event_key:
            continue
        for span in spans or []:
            if not isinstance(span, dict):
                continue
            s = _optional_float(span.get("start_sec"))
            e = _optional_float(span.get("end_sec"))
            if s is None or e is None:
                continue
            if max(0.0, min(end, e) - max(start, s)) > 0.05:
                return True
    return False


def _flexsed_event_names(flexsed_prior: dict[str, Any] | None) -> set[str]:
    if not isinstance(flexsed_prior, dict):
        return set()
    names = {
        str(item).casefold()
        for item in flexsed_prior.get("events") or []
        if str(item).strip()
    }
    timestamps = flexsed_prior.get("timestamps")
    if isinstance(timestamps, dict):
        names.update(str(item).casefold() for item in timestamps)
    return names


def _flexsed_peak_windows(flexsed_prior: dict[str, Any], *, limit: int) -> list[tuple[str, float, float]]:
    windows: list[tuple[float, str, float, float]] = []
    timestamps = flexsed_prior.get("timestamps") if isinstance(flexsed_prior, dict) else None
    if not isinstance(timestamps, dict):
        return []
    for event, spans in timestamps.items():
        for span in spans or []:
            if not isinstance(span, dict):
                continue
            start = _optional_float(span.get("start_sec"))
            end = _optional_float(span.get("end_sec"))
            score = _optional_float(span.get("peak_score")) or _optional_float(span.get("mean_score")) or 0.0
            if start is not None and end is not None and end > start:
                windows.append((score, str(event), start, end))
    windows.sort(reverse=True, key=lambda item: item[0])
    return [(event, start, end) for _, event, start, end in windows[:limit]]


def _flexsed_prior_has_multiple_clusters(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    count = 0
    timestamps = value.get("timestamps")
    if isinstance(timestamps, dict):
        for spans in timestamps.values():
            if isinstance(spans, list):
                count += len(spans)
    candidates = value.get("candidate_windows")
    if isinstance(candidates, list):
        count = max(count, len(candidates))
    return count >= 2


def _is_model_only_source(source: str) -> bool:
    return source in {"model_only", "omni_only", "qwen3_omni_only", "gemini_only"}


def _is_flexsed_source(source: str) -> bool:
    normalized = str(source or "").strip().lower()
    return normalized in {"flexsed_gemini", "flexsed_qwen3_omni", "flexsed_omni", "flexsed_model"}


def _uncertainty_mentions_model_only(data: dict[str, Any], segment: CaptionSegment) -> bool:
    haystack = " ".join(str(item).lower() for item in data.get("uncertainty") or [])
    if not haystack:
        return False
    if (
        "flexsed" in haystack
        or "model_only" in haystack
        or "omni_only" in haystack
        or "gemini_only" in haystack
        or "unsupported" in haystack
    ):
        return True
    return any(event.lower() in haystack for event in segment.events)


def _event_localization_request(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "caption",
            "describe",
            "locate",
            "timestamp",
            "time stamp",
            "find",
            "extract",
            "remove",
            "mute",
            "sound event",
            "audio event",
            "where",
            "when",
        )
    )


def _slice_proposal_user_text(context: dict[str, Any], *, min_slices: int) -> str:
    payload = {
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "min_slices": min_slices,
        "event_candidates": context.get("event_candidates"),
        "flexsed_prior": context.get("flexsed_prior"),
        "flexsed_prior_unavailable": context.get("flexsed_prior_unavailable"),
        "base_perception": context.get("base_perception"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _final_review_user_text(
    context: dict[str, Any],
    *,
    slice_records: list[dict[str, Any]],
    verification_feedback: list[str],
) -> str:
    payload = {
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "event_candidates": context.get("event_candidates"),
        "flexsed_prior": context.get("flexsed_prior"),
        "flexsed_prior_unavailable": context.get("flexsed_prior_unavailable"),
        "base_perception": context.get("base_perception"),
        "slice_audio_history": slice_records,
        "previous_verification_errors": verification_feedback,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_caption_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    compact = dict(context)
    compact["event_candidates"] = list(context.get("event_candidates") or [])[:24]
    compact["flexsed_prior"] = _compact_flexsed_prior_for_prompt(context.get("flexsed_prior") or {})
    compact["base_perception"] = _compact_base_perception_for_caption_prompt(context.get("base_perception") or {})
    return compact


def _compact_flexsed_prior_for_prompt(
    prior: dict[str, Any],
    *,
    max_events: int = 10,
    max_spans_per_event: int = 4,
) -> dict[str, Any]:
    if not isinstance(prior, dict):
        return {}
    timestamps = prior.get("timestamps") if isinstance(prior.get("timestamps"), dict) else {}
    ranked_events: list[tuple[float, str, list[dict[str, Any]]]] = []
    for event, spans in timestamps.items():
        normalized_spans: list[dict[str, Any]] = []
        for span in spans or []:
            if not isinstance(span, dict):
                continue
            start = _optional_float(span.get("start_sec"))
            end = _optional_float(span.get("end_sec"))
            if start is None or end is None or end <= start:
                continue
            normalized_spans.append(
                {
                    "event": str(span.get("event") or event),
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(end - start, 3),
                    "peak_score": _rounded_optional_float(span.get("peak_score")),
                    "mean_score": _rounded_optional_float(span.get("mean_score")),
                }
            )
        if not normalized_spans:
            continue
        normalized_spans.sort(
            key=lambda item: float(item.get("peak_score") or item.get("mean_score") or 0.0),
            reverse=True,
        )
        event_score = float(normalized_spans[0].get("peak_score") or normalized_spans[0].get("mean_score") or 0.0)
        ranked_events.append((event_score, str(event), normalized_spans[:max_spans_per_event]))
    ranked_events.sort(reverse=True, key=lambda item: item[0])
    compact_timestamps = {event: spans for _, event, spans in ranked_events[:max_events]}
    candidate_windows = []
    for item in prior.get("candidate_windows") or []:
        if not isinstance(item, dict):
            continue
        candidate_windows.append(
            {
                key: item[key]
                for key in (
                    "candidate_id",
                    "event",
                    "start_sec",
                    "end_sec",
                    "peak_time_sec",
                    "peak_score",
                    "support_kind",
                    "requires_model_review",
                )
                if key in item
            }
        )
        if len(candidate_windows) >= max_events:
            break
    return {
        "backend": prior.get("backend"),
        "input_audio_path": prior.get("input_audio_path"),
        "audio_duration_sec": _rounded_optional_float(prior.get("audio_duration_sec")),
        "threshold": _rounded_optional_float(prior.get("threshold")),
        "timestamps": compact_timestamps,
        "candidate_windows": candidate_windows,
        "omitted_nonempty_event_count": max(0, len(ranked_events) - len(compact_timestamps)),
    }


def _compact_base_perception_for_caption_prompt(base: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("source_text", "source_text_source", "language", "background", "notable_events", "uncertainty"):
        value = base.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    asr = base.get("asr")
    if isinstance(asr, dict):
        compact["asr"] = {k: asr.get(k) for k in ("text", "language") if asr.get(k)}
    word_timestamps = base.get("word_timestamps") or []
    if isinstance(word_timestamps, list) and word_timestamps:
        compact["word_timestamps"] = [
            {
                "text": item.get("text"),
                "start_sec": _rounded_optional_float(item.get("start_sec")),
                "end_sec": _rounded_optional_float(item.get("end_sec")),
            }
            for item in word_timestamps[:80]
            if isinstance(item, dict)
        ]
        if len(word_timestamps) > 80:
            compact["word_timestamps_omitted_count"] = len(word_timestamps) - 80
    for key in ("speaker_segments", "audio_caption_segments", "event_segments"):
        values = base.get(key) or []
        if isinstance(values, list) and values:
            compact[key] = [_compact_span_like_item(item) for item in values[:40] if isinstance(item, dict)]
            if len(values) > 40:
                compact[f"{key}_omitted_count"] = len(values) - 40
    return compact


def _compact_span_like_item(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "start_sec",
        "end_sec",
        "speaker_id",
        "speaker",
        "text",
        "transcript",
        "caption",
        "event",
        "events",
        "label",
        "confidence",
        "source",
        "evidence_source",
    )
    compact: dict[str, Any] = {}
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    for key in ("start_sec", "end_sec"):
        if key in compact:
            compact[key] = _rounded_optional_float(compact[key])
    return compact


def _rounded_optional_float(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None:
        return None
    return round(number, 4)


def _load_event_bank() -> list[str]:
    try:
        data = json.loads(DEFAULT_EVENT_BANK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ["speech", "laughter", "music", "door knock", "dog bark", "siren", "background noise"]
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _instruction_event_phrases(text: str) -> list[str]:
    lowered = str(text or "").lower()
    known = [event for event in _load_event_bank() if event.lower() in lowered]
    quoted = re.findall(r"['\"]([^'\"]{2,60})['\"]", str(text or ""))
    return [*known, *quoted]


def _ensure_json_object(value: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def _strong_omni_system_prompt(
    prompt: str,
    provider_display_name: str = "Qwen3-Omni",
) -> str:
    suffix = STRONG_OMNI_TIMESTAMPED_CAPTION_PROMPT_SUFFIX.format(
        provider_display_name=provider_display_name,
    )
    return prompt.rstrip() + suffix


def _resolve_qwen3_timestamped_caption_max_tokens(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    for env_name in ("X_EDITOR_QWEN3_OMNI_TIMESTAMPED_CAPTION_MAX_TOKENS", "X_EDITOR_QWEN3_OMNI_MAX_TOKENS"):
        raw = os.getenv(env_name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                continue
    return 1536


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        value = [value]
    output = []
    for item in value:
        text = str(item).strip()
        if text:
            output.append(text)
    return output


def _dedupe_strings(value: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped
