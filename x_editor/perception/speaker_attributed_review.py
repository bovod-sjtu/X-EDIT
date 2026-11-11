"""Subagentic speaker-attributed transcript review for multi-speaker speech."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from x_editor.agents.base import GeminiClient
from x_editor.agents.openai_compatible import (
    OpenAICompatibleChatClient,
    OpenAICompatibleServiceUnavailable,
)
from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    OPEN_GLM_QWEN_OMNI_MODE,
    QWEN3_OMNI_PROVIDER,
    provider_service_hint,
    resolve_agentic_mode,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key
from x_editor.core.perception_evidence_authority import (
    ObservationKind,
    reconcile_canonical_reviewed_report,
)
from x_editor.perception.audio_slicing import audio_duration_sec, slice_audio
from x_editor.perception.evidence_triage import (
    CandidateTriageDecision,
    triage_candidate_evidence_batch,
)
from x_editor.perception.invocation_trace import (
    capture_frontend_provider_call,
)
from x_editor.perception.speech import SpeechPerceptionClient, SpeechPerceptionError
from x_editor.perception.speaker_evidence_policy import (
    BLIND_SPEAKER_PROMPT, blind_label_map, specialist_only_result,
)


class SpeakerReviewOutputError(SpeechPerceptionError):
    """A received review violated its output contract; not a service failure."""

FINAL_REVIEW_SYSTEM_PROMPT = """You are a RedAudio speaker-attributed transcript review agent.

Inputs:
- The attached first audio is the complete relaxed-window audio for this task.
- Additional attached audios, if present, are short diagnostic slice_audio clips cut by the runtime.
- Qwen text is the authoritative wording source unless the audio clearly contradicts it.
- If Qwen text is empty or unavailable because ASR failed, transcribe the complete audio conservatively.
- SE-DiCoW and DiariZen evidence are speaker-attribution hints, not final truth.
- Window times are local seconds relative to the attached complete audio start.

Tool policy:
- A native slice_audio tool may be represented by runtime-provided diagnostic slices.
- Use the diagnostic slices as direct evidence for overlap, speaker switches, backchannels, cut words, and Qwen-vs-SE-DiCoW disagreements.
- Do not invent tool arguments or describe a tool-call protocol in your answer.
- Do not output XML, JSON tool requests, function-call syntax, verification prose, confidence, rationale, or a summary.

Timestamp policy:
- Treat timestamp accuracy as part of the task, not just speaker identity.
- For every high-risk boundary near a speaker switch, overlap, backchannel, cut word, or Qwen-vs-SE-DiCoW mismatch, place final line start/end at the nearest clearly heard speech.
- Do not blindly trust SE-DiCoW or DiariZen timestamps when the local audio suggests a different onset, offset, overlap, or backchannel timing.
- If a semantic role such as host, guest, adult, child, male, or female is not directly supported, keep it unresolved instead of guessing it.
- If a speaker boundary remains ambiguous after targeted listening, use only the best-supported existing speaker ID for transcript continuity; do not claim semantic-role certainty.

Speaker label policy:
- Use the IDs in allowed_speaker_map and speaker_segments, in their declared namespace.
- Different providers' identical numeric IDs do not establish the same identity.
- Raw auxiliary diarization retains its own namespace; do not merge its IDs into the active timeline.
- Do not renumber speakers by order of appearance, perceived identity, or local conversation role.
- Do not invent a new speaker ID unless that exact numeric ID appears in the input evidence.
- If auxiliary evidence disagrees, use audio evidence to resolve it within the allowed namespace; do not assume an integer offset between providers.

Decision rules:
- Preserve Qwen wording and casing as much as possible.
- If a line is clean no-overlap and DiariZen supports the speaker split, do not over-review it.
- Focus review effort on ambiguous overlap, barge-in, and relaxed-window split boundaries.

Final output:
- Output only the corrected speaker-attributed transcript.
- Each line must use the exact literal format `[start-end] [speaker N] : text`, where N is one preserved numeric input speaker ID.
- Include square brackets around both the time span and the speaker label, for example `[0.00-1.25] [speaker 0] : text`.
"""


STRONG_OMNI_TYPED_REVIEW_SYSTEM_PROMPT = """You are the audio-capable X-Editor speaker-attributed transcript reviewer.

Inputs:
- The first attached audio is the complete review window.
- Additional attached audios, if present, are diagnostic slices from that window.
- Specialist ASR wording is authoritative unless the audio clearly contradicts it.
- Speaker timelines are attribution hints, not final truth.
- Allowed speaker labels are supplied in the request and enforced by the response schema.

Your responsibility:
- Listen to the audio and decide every transcript line's wording, speaker, start time, and end time.
- Preserve the exact allowed speaker labels; do not renumber or invent speakers.
- Use diagnostic slices to resolve overlap, speaker switches, backchannels, cut words, and timing uncertainty.
- Keep timestamps in local seconds relative to the complete review window.

Return only JSON matching the supplied schema. Do not return transcript markup, prose, rationale,
tool calls, confidence values, or fields outside the schema.
"""


GLM_TEXT_REVIEW_SYSTEM_PROMPT = """You are the open X-Editor text-only speaker evidence integration agent.

You receive only structured front-end evidence. You do not receive audio, and you must not
claim to listen to audio or call tools.

Your job:
- Normalize transcript wording for each provided speaker span.
- Treat Qwen3ASR text and word timestamps as the primary wording prior.
- Treat speaker_timeline timestamps and speaker IDs as fixed front-end evidence.
- Do not change speaker labels, speaker order, timestamps, or span count.
- Do not invent new speakers, audio events, tool calls, confidence, or analysis prose.

Return only JSON:
{
  "segments": [
    {"segment_index": 0, "text": "normalized transcript for that exact span"}
  ]
}
"""


SLICE_PROPOSAL_SYSTEM_PROMPT = """You plan diagnostic slice_audio calls for a speaker-attributed transcript review.

Return only a JSON object with a `slices` array. Each slice must include:
- onset: local start seconds
- offset: local end seconds
- reason: a concrete uncertainty this slice resolves

Keep slices short and diagnostic, usually 4-12 seconds. Target exact disputed
switches, overlaps, barge-ins, missing sentences, and reference voice spans.
Do not add routine coverage or calibration slices. Only propose slices for
uncertainties explicitly present in the payload.
"""


QUALITY_CHECK_SYSTEM_PROMPT = """You are a strict checker for speaker-attributed transcript review.

You receive:
- The complete audio.
- Any runtime-provided slice_audio clips.
- Qwen ASR text, speaker evidence, and a candidate final transcript.

Check whether the candidate transcript is good enough for downstream editing:
- complete enough to preserve Qwen wording unless audio contradicts it
- speaker labels chosen from allowed numeric IDs only
- timestamps plausible and useful for editing
- unresolved overlap, backchannel, and speaker-switch uncertainty either supported by current evidence or targeted by requested slices
- clean evidence must not fail only because no diagnostic slices were supplied
- no obvious hallucinated repeated text or wrong-language drift

Return only a JSON object:
{
  "passed": boolean,
  "issues": ["short actionable issue", ...],
  "additional_slices": [
    {"onset": seconds, "offset": seconds, "reason": "specific uncertainty this slice resolves"}
  ]
}

If the candidate is satisfactory, set passed=true and additional_slices=[].
If it is not satisfactory, request only the most useful 1-3 short diagnostic slices.
"""


TEMPORAL_SEGMENT_PROPOSAL_SYSTEM_PROMPT = """You plan the next temporal reasoning step for timestamped speaker-attributed ASR.

You receive the complete audio, Qwen ASR text, speaker evidence, and previous temporal turns.
Choose the next local segment along the audio timeline for careful listening.
Use `temporal_initialization` as the initial plan: timestamped speaker evidence
is the timeline prior, and Qwen3ASR is the wording prior.

Return only a JSON object:
{
  "continue": boolean,
  "onset": seconds,
  "offset": seconds,
  "reason": "why this next segment should be inspected"
}

Policy:
- Work monotonically along the timeline from the current cursor.
- Prefer the provided `next_seed_window` unless audio/text evidence suggests a more useful nearby window.
- Use compact but useful windows, usually 4-12 seconds.
- Prefer boundaries around speaker switches, overlaps, backchannels, and low-confidence evidence.
- Set continue=false only when the full audio has been covered or further local listening is not useful.
"""


TEMPORAL_SEGMENT_REVIEW_SYSTEM_PROMPT = """You perform one local temporal reasoning turn for speaker-attributed ASR.

Inputs:
- The attached audio is a runtime slice from the complete audio.
- The JSON payload gives the slice onset/offset in complete-audio seconds.
- Specialist ASR text and speaker evidence are hints, not final truth.
- `local_initialization` contains the timestamped speaker and specialist ASR word evidence
  overlapping this slice; use it as the first draft to verify or correct.

Output only speaker-attributed transcript lines for this slice.
Each line must use absolute complete-audio timestamps and the exact format:
[start-end] [speaker N] : text

Rules:
- Preserve specialist ASR wording when it matches the local audio.
- Correct speaker attribution, timestamps, backchannels, and overlap details using the local slice.
- Use only allowed numeric speaker IDs from the payload.
- If the slice contains no clearly relevant speech, output an empty string.
"""


@dataclass
class SpeakerReviewLine:
    start_sec: float
    end_sec: float
    speaker_label: str
    speaker_id: str
    text: str


@dataclass
class ReviewVerificationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    parsed_lines: list[SpeakerReviewLine] = field(default_factory=list)
    required_slice_calls: int = 0
    observed_slice_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "required_slice_calls": self.required_slice_calls,
            "observed_slice_calls": self.observed_slice_calls,
            "line_count": len(self.parsed_lines),
        }


class SpeakerReviewFrontend(Protocol):
    def propose_slices(
        self, context: dict[str, Any], *, min_slices: int
    ) -> list[dict[str, Any]]: ...

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> str: ...

    def check_review_quality(
        self,
        context: dict[str, Any],
        *,
        candidate_transcript: str,
        slice_records: list[dict[str, Any]],
        verification: ReviewVerificationResult,
    ) -> dict[str, Any]: ...

    def propose_temporal_segment(
        self,
        context: dict[str, Any],
        *,
        temporal_history: list[dict[str, Any]],
        cursor_sec: float,
    ) -> dict[str, Any]: ...

    def review_temporal_segment(
        self,
        context: dict[str, Any],
        *,
        segment_record: dict[str, Any],
        temporal_history: list[dict[str, Any]],
    ) -> str: ...


class GeminiSpeakerReviewFrontend:
    """Gemini/StrongOmni frontend for transcript review."""

    component_name = "perception.speaker_attributed_review"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 32768,
        timeout: float = 120.0,
    ) -> None:
        config = AudioEditConfig()
        self.client = GeminiClient(
            api_key=resolve_gemini_api_key(api_key) or config.omni_api_key,
            base_url=base_url or config.omni_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.last_invocation_trace = None
        self.telemetry: dict[str, Any] = {
            "slice_proposals": [],
            "review_calls": [],
            "quality_checks": [],
            "temporal_segment_proposals": [],
            "temporal_segment_reviews": [],
        }

    def _task_json(self, payload):
        from x_editor.perception.speaker_task_grounding import TASK_GROUNDING_PROMPT
        return self.client.call_json(TASK_GROUNDING_PROMPT, json.dumps(payload, ensure_ascii=False),
                                     audio_paths=[payload['audio_path']])

    def observe_task(self, payload):
        record = {'prompt_payload': payload}
        try:
            raw = self._task_json(payload)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            self.telemetry.setdefault('task_observations', []).append(record)
            capture_frontend_provider_call(self, 'speaker_attributed_review.task_observation',
                                           succeeded=False, error=exc)
            raise
        record.update(status='success', raw_json=raw,
                      client_metrics=getattr(self.client, 'last_call_metrics', None))
        self.telemetry.setdefault('task_observations', []).append(record)
        capture_frontend_provider_call(self, 'speaker_attributed_review.task_observation', succeeded=True)
        return raw

    def propose_slices(self, context: dict[str, Any], *, min_slices: int) -> list[dict[str, Any]]:
        if min_slices <= 0:
            return []
        user_text = _slice_proposal_user_text(context, min_slices=min_slices)
        record: dict[str, Any] = {
            "prompt_payload": _try_json_loads(user_text),
            "min_slices": min_slices,
        }
        try:
            data = self.client.call_json(
                system_prompt=SLICE_PROPOSAL_SYSTEM_PROMPT,
                user_text=user_text,
                audio_paths=[context["audio_path"]],
            )
        except Exception:
            record["status"] = "failed"
            record["error"] = "slice proposal model call failed"
            self.telemetry.setdefault("slice_proposals", []).append(record)
            return []
        record["status"] = "success"
        record["raw_json"] = data
        self.telemetry.setdefault("slice_proposals", []).append(record)
        raw = data.get("slices") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        return [_normalize_slice_request(item) for item in raw if isinstance(item, dict)]

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> str:
        user_text = _final_review_user_text(
            context,
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
        audio_paths = [
            context["audio_path"],
            *[record["output_audio_path"] for record in slice_records],
        ]
        try:
            output = self.client.call(
                system_prompt=FINAL_REVIEW_SYSTEM_PROMPT,
                user_text=user_text,
                audio_paths=audio_paths,
            )
        except Exception as exc:
            capture_frontend_provider_call(
                self,
                "speaker_attributed_review.review",
                succeeded=False,
                error=exc,
            )
            raise
        capture_frontend_provider_call(
            self,
            "speaker_attributed_review.review",
            succeeded=True,
        )
        return output

    def check_review_quality(
        self,
        context: dict[str, Any],
        *,
        candidate_transcript: str,
        slice_records: list[dict[str, Any]],
        verification: ReviewVerificationResult,
    ) -> dict[str, Any]:
        user_text = _quality_check_user_text(
            context,
            candidate_transcript=candidate_transcript,
            slice_records=slice_records,
            verification=verification,
        )
        record = {
            "prompt_payload": _try_json_loads(user_text),
            "slice_audio_count": len(slice_records),
        }
        try:
            data = self.client.call_json(
                system_prompt=QUALITY_CHECK_SYSTEM_PROMPT,
                user_text=user_text,
                audio_paths=[
                    context["audio_path"],
                    *[item["output_audio_path"] for item in slice_records],
                ],
            )
        except Exception as exc:
            capture_frontend_provider_call(
                self,
                "speaker_attributed_review.quality_check",
                succeeded=False,
                error=exc,
            )
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.telemetry.setdefault("quality_checks", []).append(record)
            return {
                "passed": True,
                "issues": ["quality checker unavailable"],
                "additional_slices": [],
            }
        capture_frontend_provider_call(
            self,
            "speaker_attributed_review.quality_check",
            succeeded=True,
        )
        record["status"] = "success"
        record["raw_json"] = data
        self.telemetry.setdefault("quality_checks", []).append(record)
        if not isinstance(data, dict):
            return {
                "passed": True,
                "issues": ["quality checker returned non-object"],
                "additional_slices": [],
            }
        return {
            "passed": bool(data.get("passed")),
            "issues": [str(item) for item in data.get("issues") or []],
            "additional_slices": [
                _normalize_slice_request(item)
                for item in data.get("additional_slices") or []
                if isinstance(item, dict)
            ],
        }

    def propose_temporal_segment(
        self,
        context: dict[str, Any],
        *,
        temporal_history: list[dict[str, Any]],
        cursor_sec: float,
    ) -> dict[str, Any]:
        user_text = _temporal_segment_proposal_user_text(
            context,
            temporal_history=temporal_history,
            cursor_sec=cursor_sec,
        )
        record = {
            "prompt_payload": _try_json_loads(user_text),
            "cursor_sec": cursor_sec,
        }
        try:
            data = self.client.call_json(
                system_prompt=TEMPORAL_SEGMENT_PROPOSAL_SYSTEM_PROMPT,
                user_text=user_text,
                audio_paths=[context["audio_path"]],
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.telemetry.setdefault("temporal_segment_proposals", []).append(record)
            return {"continue": False, "reason": "temporal segment proposal failed"}
        record["status"] = "success"
        record["raw_json"] = data
        self.telemetry.setdefault("temporal_segment_proposals", []).append(record)
        if not isinstance(data, dict):
            return {"continue": False, "reason": "temporal segment proposal returned non-object"}
        if not data.get("continue", True):
            return {"continue": False, "reason": str(data.get("reason") or "model stopped")}
        segment = _normalize_slice_request(data)
        segment["continue"] = True
        return segment

    def review_temporal_segment(
        self,
        context: dict[str, Any],
        *,
        segment_record: dict[str, Any],
        temporal_history: list[dict[str, Any]],
    ) -> str:
        user_text = _temporal_segment_review_user_text(
            context,
            segment_record=segment_record,
            temporal_history=temporal_history,
        )
        self.telemetry.setdefault("temporal_segment_reviews", []).append(
            {
                "prompt_payload": _try_json_loads(user_text),
                "segment": {
                    "onset": segment_record.get("onset"),
                    "offset": segment_record.get("offset"),
                    "reason": segment_record.get("reason"),
                },
            }
        )
        return self.client.call(
            system_prompt=TEMPORAL_SEGMENT_REVIEW_SYSTEM_PROMPT,
            user_text=user_text,
            audio_paths=[segment_record["output_audio_path"]],
        )


class OpenAICompatibleSpeakerReviewFrontend(GeminiSpeakerReviewFrontend):
    """Audio-capable speaker evidence reviewer for the open provider path."""

    provider_name = QWEN3_OMNI_PROVIDER
    review_schema_name = "StrongOmniSpeakerReview"
    review_error_label = "StrongOmni speaker review"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        client: OpenAICompatibleChatClient | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 180.0,
        provider_name: str | None = None,
        provider_display_name: str | None = None,
    ) -> None:
        self.provider_name = provider_name or self.provider_name
        self.provider_display_name = provider_display_name or self.provider_name
        self.model_name = resolve_openai_compatible_model(
            self.provider_name,
            model_name,
        )
        self.client = client or OpenAICompatibleChatClient(
            api_key=resolve_openai_compatible_api_key(self.provider_name),
            base_url=resolve_openai_compatible_base_url(self.provider_name),
            model_name=self.model_name,
            provider_name=self.provider_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            allow_response_format_fallback=False,
            allow_json_repair=False,
            require_api_key=False,
        )
        self.last_invocation_trace = None
        self.telemetry = {
            "slice_proposals": [],
            "review_calls": [],
            "quality_checks": [],
            "temporal_segment_proposals": [],
            "temporal_segment_reviews": [],
        }

    def _task_json(self, payload):
        from x_editor.perception.speaker_task_grounding import TASK_GROUNDING_PROMPT
        return self.client.call_json(TASK_GROUNDING_PROMPT, json.dumps(payload, ensure_ascii=False),
            audio_paths=[payload['audio_path']], response_format={'type': 'json_schema',
                'json_schema': {'name': 'SpeakerTaskObservation', 'strict': True,
                                'schema': payload['response_schema']}},
            repair_schema_name='SpeakerTaskObservation')

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> str:
        user_text = _final_review_user_text(
            context,
            slice_records=slice_records,
            verification_feedback=verification_feedback or [],
        )
        allowed_speaker_map = context.get("allowed_speaker_map") or {}
        blind = context.get("specialist_prior_visible") is False
        response_format = _strong_omni_speaker_review_response_format(
            allowed_speaker_map,
            schema_name=self.review_schema_name,
            error_label=self.review_error_label,
            allow_new_labels=blind,
        )
        audio_paths = [
            context["audio_path"],
            *[record["output_audio_path"] for record in slice_records],
        ]
        record: dict[str, Any] = {
            "prompt_payload": _try_json_loads(user_text),
            "slice_audio_count": len(slice_records),
            "verification_feedback": list(verification_feedback or []),
            "provider": self.provider_name,
            "model_name": self.model_name,
        }
        try:
            data = self.client.call_json(
                BLIND_SPEAKER_PROMPT if blind else STRONG_OMNI_TYPED_REVIEW_SYSTEM_PROMPT,
                user_text,
                audio_paths=audio_paths,
                response_format=response_format,
                repair_schema_name=self.review_schema_name,
            )
            record['raw_json'] = data
            try:
                if blind:
                    allowed_speaker_map = blind_label_map(data)
                transcript = _render_strong_omni_speaker_review_transcript(
                    data,
                    allowed_speaker_map=allowed_speaker_map,
                    error_label=self.review_error_label,
                )
            except SpeechPerceptionError as exc:
                raise SpeakerReviewOutputError(str(exc)) from exc
            if blind:
                context["allowed_speaker_map"] = allowed_speaker_map
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["client_metrics"] = getattr(self.client, "last_call_metrics", None)
            self.telemetry.setdefault("review_calls", []).append(record)
            capture_frontend_provider_call(
                self,
                "speaker_attributed_review.review",
                succeeded=False,
                error=exc,
            )
            raise
        record["status"] = "success"
        record["raw_json"] = data
        record["rendered_transcript"] = transcript
        record["client_metrics"] = getattr(self.client, "last_call_metrics", None)
        self.telemetry.setdefault("review_calls", []).append(record)
        capture_frontend_provider_call(
            self,
            "speaker_attributed_review.review",
            succeeded=True,
        )
        return transcript


class Qwen3OmniSpeakerReviewFrontend(
    OpenAICompatibleSpeakerReviewFrontend
):
    """Backward-compatible Qwen3-Omni speaker-review name."""

    review_schema_name = "Qwen3OmniSpeakerReview"
    review_error_label = "Qwen3-Omni speaker review"


class GLMSpeakerEvidenceTextFrontend:
    """Open-mode text-only frontend for integrating specialist speaker evidence."""

    provider_name = GLM_PROVIDER

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.model_name = resolve_openai_compatible_model(self.provider_name, model_name)
        self.api_key = resolve_openai_compatible_api_key(self.provider_name, api_key)
        self.base_url = resolve_openai_compatible_base_url(self.provider_name, base_url)
        self.client = client or OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            provider_name=self.provider_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.telemetry: dict[str, Any] = {
            "slice_proposals": [],
            "review_calls": [],
            "quality_checks": [],
            "temporal_segment_proposals": [],
            "temporal_segment_reviews": [],
        }

    def propose_slices(self, context: dict[str, Any], *, min_slices: int) -> list[dict[str, Any]]:
        self.telemetry.setdefault("slice_proposals", []).append(
            {
                "status": "skipped",
                "reason": "open_text_only_speaker_review_does_not_call_slice_audio",
                "min_slices": min_slices,
            }
        )
        return []

    def review(
        self,
        context: dict[str, Any],
        *,
        slice_records: list[dict[str, Any]],
        verification_feedback: list[str] | None = None,
    ) -> str:
        draft_segments = _speaker_text_review_draft_segments(context)
        payload = _glm_text_review_user_text(
            context,
            draft_segments=draft_segments,
            verification_feedback=verification_feedback or [],
        )
        record: dict[str, Any] = {
            "prompt_payload": _try_json_loads(payload),
            "slice_audio_count": len(slice_records),
            "verification_feedback": list(verification_feedback or []),
            "text_only": True,
            "provider": self.provider_name,
            "model_name": self.model_name,
        }
        try:
            data = self.client.call_json(
                GLM_TEXT_REVIEW_SYSTEM_PROMPT,
                payload,
                audio_paths=[],
                response_format=_glm_speaker_text_response_format(),
                repair_schema_name="SpeakerEvidenceTextReview",
            )
        except OpenAICompatibleServiceUnavailable as exc:
            record["status"] = "service_unavailable"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.telemetry.setdefault("review_calls", []).append(record)
            raise SpeechPerceptionError(
                f"open speaker-attributed GLM text review service unavailable: {exc}. "
                f"To use open speaker review, {provider_service_hint(self.provider_name)}."
            ) from exc
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.telemetry.setdefault("review_calls", []).append(record)
            raise SpeechPerceptionError(
                f"open speaker-attributed GLM text review failed before verified transcript output: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        record["status"] = "success"
        record["raw_json"] = data
        record["client_metrics"] = getattr(self.client, "last_call_metrics", None)
        transcript = _render_glm_text_review_transcript(
            context, draft_segments=draft_segments, model_data=data
        )
        record["rendered_transcript"] = transcript
        self.telemetry.setdefault("review_calls", []).append(record)
        return transcript

    def check_review_quality(
        self,
        context: dict[str, Any],
        *,
        candidate_transcript: str,
        slice_records: list[dict[str, Any]],
        verification: ReviewVerificationResult,
    ) -> dict[str, Any]:
        result = {
            "passed": verification.passed,
            "issues": list(verification.errors),
            "additional_slices": [],
            "checker": "deterministic_text_only_verifier",
        }
        self.telemetry.setdefault("quality_checks", []).append(
            {
                "status": "completed",
                "text_only": True,
                "verification": verification.to_dict(),
                "slice_audio_count": len(slice_records),
            }
        )
        return result

    def propose_temporal_segment(
        self,
        context: dict[str, Any],
        *,
        temporal_history: list[dict[str, Any]],
        cursor_sec: float,
    ) -> dict[str, Any]:
        record = {
            "status": "skipped",
            "reason": "open_text_only_speaker_review_does_not_request_audio_temporal_slices",
            "cursor_sec": cursor_sec,
            "history_count": len(temporal_history),
        }
        self.telemetry.setdefault("temporal_segment_proposals", []).append(record)
        return {"continue": False, "reason": record["reason"]}

    def review_temporal_segment(
        self,
        context: dict[str, Any],
        *,
        segment_record: dict[str, Any],
        temporal_history: list[dict[str, Any]],
    ) -> str:
        self.telemetry.setdefault("temporal_segment_reviews", []).append(
            {
                "status": "skipped",
                "reason": "open_text_only_speaker_review_does_not_receive_audio_slices",
                "segment": {
                    "onset": segment_record.get("onset"),
                    "offset": segment_record.get("offset"),
                    "reason": segment_record.get("reason"),
                },
            }
        )
        return ""


def run_speaker_attributed_review(
    *,
    audio_path: str | Path,
    instruction: str,
    output_dir: str | Path,
    base_perception: dict[str, Any] | None = None,
    speech_client: SpeechPerceptionClient | None = None,
    reviewer: SpeakerReviewFrontend | None = None,
    run_specialized: bool = True,
    use_sedicow: bool = True,
    speaker_evidence_backend: str = "auto",
    max_review_attempts: int = 4,
    strict: bool = True,
    refresh_cache: bool = False,
    language: str | None = None,
    num_speakers: int | None = None,
    agentic_mode: str | None = None,
    use_temporal_reasoning: bool = False,
    max_temporal_turns: int = 8,
    observation_needs: list[dict[str, Any]] | None = None,
    use_specialist_prior: bool = True,
    enable_omni_review: bool = True,
    supplement_speaker_attributes: bool = False,
) -> dict[str, Any]:
    """Return measured speech with optional revision and task-attribute hearing.

    Attribute supplementation is independent of transcript/timestamp review.
    Legacy module probes leave it disabled to retain their frozen measurement
    scope; workflow profiles can enable it for task-complete observations.
    """

    audio = Path(audio_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"speaker-attributed review audio does not exist: {audio}")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not use_specialist_prior and not enable_omni_review:
        raise ValueError("speaker observation requires a specialist or audio reviewer")
    client = speech_client or SpeechPerceptionClient(cache_dir=output_root / "perception_cache")
    evidence = _collect_evidence(
        client,
        audio,
        base_perception=base_perception or {},
        run_specialized=run_specialized,
        use_sedicow=use_sedicow,
        speaker_evidence_backend=speaker_evidence_backend,
        language=language,
        num_speakers=num_speakers,
        refresh_cache=refresh_cache,
    ) if use_specialist_prior else {
        "base_perception": {}, "source_text": "", "speaker_segments": [],
        "speaker_evidence_backend": "none", "attempts": [],
    }
    context = _review_context(
        audio,
        instruction=instruction,
        evidence=evidence,
        output_dir=output_root,
    )
    context['agentic_mode'] = resolve_agentic_mode(agentic_mode)
    context['observation_needs'] = [dict(item) for item in (observation_needs or []) if isinstance(item, Mapping)]

    def finish(result, frontend=None):
        if not supplement_speaker_attributes:
            return result
        from x_editor.perception.speaker_task_grounding import supplement_speaker_task
        return supplement_speaker_task(result, context,
            frontend or reviewer or _default_speaker_review_frontend(context))

    if not enable_omni_review:
        return finish(specialist_only_result(evidence, context))
    if not use_specialist_prior:
        context["specialist_prior_visible"] = False
    triage_batch = _triage_speaker_evidence(context)
    context["speaker_evidence_triage"] = triage_batch.to_dict()
    context["speaker_semantic_role_uncertainty"] = _speaker_semantic_role_uncertainty(context)
    min_slices = required_slice_call_count(context)
    if _use_evidence_summary_review(context):
        min_slices = 0
    frontend = reviewer or _default_speaker_review_frontend(context)
    temporal_reasoning_history: list[dict[str, Any]] = []
    if use_temporal_reasoning:
        temporal_reasoning_history = _run_temporal_reasoning(
            frontend,
            context,
            audio=audio,
            output_dir=output_root / "temporal_reasoning",
            max_turns=max_temporal_turns,
        )
        context["temporal_reasoning_history"] = temporal_reasoning_history
        context["temporal_reasoning_draft"] = _temporal_reasoning_draft(temporal_reasoning_history)
    slice_requests = []
    slice_requests_from_model: list[dict[str, Any]] = []
    if not _use_evidence_summary_review(context):
        slice_requests = _targeted_slice_requests_from_triage(context, triage_batch.decisions)
    slice_records = _execute_slice_requests(audio, slice_requests, output_root / "slice_audio")

    raw_outputs: list[str] = []
    review_attempts: list[dict[str, Any]] = []
    verification_history: list[dict[str, Any]] = []
    quality_check_history: list[dict[str, Any]] = []
    additional_slice_requests_from_checker: list[dict[str, Any]] = []
    feedback: list[str] = []
    best_verification: ReviewVerificationResult | None = None
    best_raw = ""
    for _attempt in range(max(1, max_review_attempts)):
        try:
            raw = frontend.review(context, slice_records=slice_records, verification_feedback=feedback)
        except SpeakerReviewOutputError as exc:
            # Use the existing review budget and strict/non-strict policy.
            # Never clip timestamps or promote rejected model output to evidence.
            review_attempts.append({
                'attempt_index': len(review_attempts) + 1,
                'verification_feedback_input': list(feedback),
                'status': 'invalid_output', 'error': str(exc),
            })
            feedback = [str(exc)]
            continue
        raw_outputs.append(raw)
        verification = verify_review_output(
            raw,
            allowed_speaker_map=context["allowed_speaker_map"],
            duration_sec=context.get("duration_sec"),
            required_slice_calls=min_slices,
            slice_records=slice_records,
        )
        verification_history.append(verification.to_dict())
        review_attempts.append(
            {
                "attempt_index": len(review_attempts) + 1,
                "verification_feedback_input": list(feedback),
                "raw_output": raw,
                "verification": verification.to_dict(),
            }
        )
        if best_verification is None or len(verification.errors) < len(best_verification.errors):
            best_verification = verification
            best_raw = raw
        quality_check = None
        if verification.passed:
            quality_check = _check_review_quality(
                frontend,
                context,
                candidate_transcript=raw,
                slice_records=slice_records,
                verification=verification,
            )
            quality_check_history.append(quality_check)
            review_attempts[-1]["quality_check"] = quality_check
        if verification.passed and (quality_check or {}).get("passed", True):
            perception = _perception_from_review(evidence, verification.parsed_lines, context)
            return finish({
                "perception_report": perception,
                "source_text": perception.get("source_text"),
                "speaker_attributed_review": {
                    "status": "success",
                    "raw_outputs": raw_outputs,
                    "verification": verification.to_dict(),
                    "verification_history": verification_history,
                    "quality_check_history": quality_check_history,
                    "temporal_reasoning_history": temporal_reasoning_history,
                    "review_attempts": review_attempts,
                    "review_context": _audit_review_context(context),
                    "slice_proposal_prompt_payload": _try_json_loads(
                        _slice_proposal_user_text(context, min_slices=min_slices)
                    )
                    if min_slices > 0
                    else None,
                    "slice_requests_from_model": slice_requests_from_model,
                    "additional_slice_requests_from_checker": additional_slice_requests_from_checker,
                    "final_slice_requests": slice_requests,
                    "slice_audio_calls": slice_records,
                    "required_slice_calls": min_slices,
                    "evidence_triage": context.get("speaker_evidence_triage"),
                    "evidence": evidence,
                    "frontend_telemetry": getattr(frontend, "telemetry", {}),
                },
            }, frontend)
        if verification.passed and quality_check and not quality_check.get("passed", True):
            feedback = [
                f"quality_check: {issue}"
                for issue in quality_check.get("issues") or ["candidate not satisfactory"]
            ]
            new_requests = [
                request
                for request in quality_check.get("additional_slices") or []
                if not _overlaps_existing_slice(request, slice_requests)
            ]
            if new_requests:
                additional_slice_requests_from_checker.extend(new_requests)
                slice_requests.extend(new_requests)
                new_slice_records = _execute_slice_requests(
                    audio,
                    new_requests,
                    output_root / "slice_audio",
                    start_index=len(slice_records) + 1,
                )
                slice_records.extend(new_slice_records)
            continue
        feedback = verification.errors

    if strict:
        error_text = "; ".join(
            (best_verification.errors if best_verification else feedback) or ["review failed"]
        )
        raise SpeechPerceptionError(
            f"speaker-attributed transcript review failed verification: {error_text}"
        )

    # A parsed review can still fail completeness or attribution checks. Keep
    # the observed specialist inventory intact; unaccepted corrections remain
    # in the review audit and cannot replace speaker turns or source wording.
    perception = copy.deepcopy(evidence.get("base_perception") or {})
    perception.setdefault("uncertainties", []).append({
        "code": "speaker_review_failed_verification",
        "message": "Speaker review did not pass verification. Reviewer corrections "
                   "and diagnostic opinions remain disputed; only the original "
                   "specialist facts are retained, without confirming speaker "
                   "identity, count, or target absence.",
        "severity": "advisory", "blocking": False,
        "missing_evidence": ["verified speaker attribution review"],
    })
    return {
        "perception_report": perception,
        "source_text": perception.get("source_text"),
        "speaker_attributed_review": {
            "status": "failed_verification",
            "raw_outputs": raw_outputs or [best_raw],
            "verification": best_verification.to_dict() if best_verification else None,
            "verification_history": verification_history,
            "quality_check_history": quality_check_history,
            "temporal_reasoning_history": temporal_reasoning_history,
            "review_attempts": review_attempts,
            "review_context": _audit_review_context(context),
            "slice_proposal_prompt_payload": _try_json_loads(
                _slice_proposal_user_text(context, min_slices=min_slices)
            )
            if min_slices > 0
            else None,
            "slice_requests_from_model": slice_requests_from_model,
            "additional_slice_requests_from_checker": additional_slice_requests_from_checker,
            "final_slice_requests": slice_requests,
            "slice_audio_calls": slice_records,
            "required_slice_calls": min_slices,
            "evidence_triage": context.get("speaker_evidence_triage"),
            "evidence": evidence,
            "frontend_telemetry": getattr(frontend, "telemetry", {}),
        },
    }


def _audit_review_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio_path": context.get("audio_path"),
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "source_text": context.get("source_text"),
        "language": context.get("language"),
        "allowed_speaker_map": context.get("allowed_speaker_map"),
        "turn_speaker_switch_count": context.get("turn_speaker_switch_count"),
        "speech_overlap_class": context.get("speech_overlap_class"),
        "speaker_segments": context.get("speaker_segments"),
        "temporal_initialization": context.get("temporal_initialization"),
        "speaker_evidence_triage": context.get("speaker_evidence_triage"),
        "speaker_semantic_role_uncertainty": context.get("speaker_semantic_role_uncertainty"),
        "observation_needs": context.get("observation_needs"),
    }


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def verify_review_output(
    raw_text: str,
    *,
    allowed_speaker_map: dict[str, str],
    duration_sec: float | None,
    required_slice_calls: int,
    slice_records: list[dict[str, Any]],
) -> ReviewVerificationResult:
    errors: list[str] = []
    parsed: list[SpeakerReviewLine] = []
    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    if not lines:
        errors.append("final transcript is empty")
    for line_index, line in enumerate(lines, start=1):
        match = re.fullmatch(
            r"\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s+\[(?P<label>[^\]]+)\]\s*:\s*(?P<text>.+)",
            line,
        )
        if not match:
            errors.append(f"line {line_index} does not match `[start-end] [speaker N] : text`")
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        label = _canonical_review_speaker_label(match.group("label"), allowed_speaker_map)
        text = match.group("text").strip()
        if label is None:
            errors.append(
                f"line {line_index} uses hallucinated speaker label `{match.group('label')}`"
            )
            continue
        if end <= start:
            errors.append(f"line {line_index} has non-positive time span")
            continue
        if duration_sec is not None and (start < -0.01 or end > duration_sec + 0.25):
            errors.append(
                f"line {line_index} time span {start:.2f}-{end:.2f} falls outside audio duration"
            )
            continue
        parsed.append(
            SpeakerReviewLine(
                start_sec=start,
                end_sec=end,
                speaker_label=label,
                speaker_id=allowed_speaker_map[label],
                text=text,
            )
        )
    missing_reasons = [
        record for record in slice_records if not str(record.get("reason") or "").strip()
    ]
    if missing_reasons:
        errors.append("slice_audio call history contains calls without concrete reasons")
    return ReviewVerificationResult(
        passed=not errors,
        errors=errors,
        parsed_lines=parsed,
        required_slice_calls=required_slice_calls,
        observed_slice_calls=len(slice_records),
    )


def _canonical_review_speaker_label(
    raw_label: str, allowed_speaker_map: dict[str, str]
) -> str | None:
    raw = str(raw_label or "").strip()
    if raw in allowed_speaker_map:
        return raw
    normalized_raw = _normalize_speaker_label_text(raw)
    by_normalized = {_normalize_speaker_label_text(label): label for label in allowed_speaker_map}
    if normalized_raw in by_normalized:
        return by_normalized[normalized_raw]

    value_matches = [
        label
        for label, speaker_id in allowed_speaker_map.items()
        if normalized_raw == _normalize_speaker_label_text(str(speaker_id))
    ]
    if len(value_matches) == 1:
        return value_matches[0]

    raw_number = _speaker_label_number(raw)
    if raw_number is None:
        return None
    numeric_matches = [
        label
        for label, speaker_id in allowed_speaker_map.items()
        if _speaker_label_number(label) == raw_number
        or _speaker_label_number(str(speaker_id)) == raw_number
    ]
    unique = sorted(set(numeric_matches))
    return unique[0] if len(unique) == 1 else None


def _normalize_speaker_label_text(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(value or "").strip().lower())


def _speaker_label_number(value: str) -> int | None:
    text = str(value or "").strip()
    patterns = [
        r"\bspeaker[\s_-]*0*(\d+)\b",
        r"\bspk[\s_-]*0*(\d+)\b",
        r"\bs[\s_-]*0*(\d+)\b",
        r"^0*(\d+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _check_review_quality(
    frontend: SpeakerReviewFrontend,
    context: dict[str, Any],
    *,
    candidate_transcript: str,
    slice_records: list[dict[str, Any]],
    verification: ReviewVerificationResult,
) -> dict[str, Any]:
    checker = getattr(frontend, "check_review_quality", None)
    if not callable(checker):
        return {"passed": True, "issues": [], "additional_slices": [], "checker": "unavailable"}
    result = checker(
        context,
        candidate_transcript=candidate_transcript,
        slice_records=slice_records,
        verification=verification,
    )
    if not isinstance(result, dict):
        return {
            "passed": True,
            "issues": ["quality checker returned non-object"],
            "additional_slices": [],
        }
    return {
        "passed": bool(result.get("passed")),
        "issues": [str(item) for item in result.get("issues") or []],
        "additional_slices": [
            _normalize_slice_request(item)
            for item in result.get("additional_slices") or []
            if isinstance(item, dict)
        ],
    }


def _run_temporal_reasoning(
    frontend: SpeakerReviewFrontend,
    context: dict[str, Any],
    *,
    audio: Path,
    output_dir: Path,
    max_turns: int,
) -> list[dict[str, Any]]:
    proposer = getattr(frontend, "propose_temporal_segment", None)
    reviewer = getattr(frontend, "review_temporal_segment", None)
    if not callable(proposer) or not callable(reviewer):
        return []
    duration = float(context.get("duration_sec") or 0.0)
    if duration <= 0.0 or max_turns <= 0:
        return []
    context["temporal_initialization"] = _temporal_reasoning_initialization(context)
    history: list[dict[str, Any]] = []
    cursor = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    for turn_index in range(1, max_turns + 1):
        seed_window = _next_temporal_seed_window(context, cursor)
        context["temporal_reasoning_next_seed_window"] = seed_window
        proposal = proposer(context, temporal_history=history, cursor_sec=cursor)
        if not isinstance(proposal, dict) or not proposal.get("continue", True):
            history.append(
                {
                    "turn_index": turn_index,
                    "status": "stopped",
                    "cursor_sec": round(cursor, 6),
                    "reason": str((proposal or {}).get("reason") or "model stopped"),
                }
            )
            break
        used_seed_fallback = False
        segment = _normalize_temporal_segment(proposal, cursor_sec=cursor, duration_sec=duration)
        if segment is None:
            segment = seed_window or _next_temporal_fallback_segment(cursor, duration)
            used_seed_fallback = True
        if _overlaps_existing_slice(
            segment, [item["slice_request"] for item in history if item.get("slice_request")]
        ):
            segment = seed_window or _next_temporal_fallback_segment(cursor, duration)
            used_seed_fallback = True
        slice_record = _execute_slice_requests(
            audio,
            [segment],
            output_dir,
            start_index=turn_index,
        )[0]
        context["temporal_reasoning_current_local_initialization"] = _local_temporal_initialization(
            context, slice_record
        )
        transcript = reviewer(
            context,
            segment_record=slice_record,
            temporal_history=history,
        )
        verification = verify_review_output(
            transcript,
            allowed_speaker_map=context["allowed_speaker_map"],
            duration_sec=context.get("duration_sec"),
            required_slice_calls=0,
            slice_records=[slice_record],
        )
        record = {
            "turn_index": turn_index,
            "status": "reviewed",
            "proposal": proposal,
            "seed_window": seed_window,
            "used_seed_fallback": used_seed_fallback,
            "local_initialization": context.get("temporal_reasoning_current_local_initialization"),
            "slice_request": segment,
            "slice_record": slice_record,
            "raw_output": transcript,
            "verification": verification.to_dict(),
        }
        history.append(record)
        cursor = max(cursor, float(segment["offset"]))
        if cursor >= duration - 0.05:
            break
    return history


def _normalize_temporal_segment(
    proposal: Mapping[str, Any],
    *,
    cursor_sec: float,
    duration_sec: float,
) -> dict[str, Any] | None:
    try:
        start = float(proposal.get("onset", proposal.get("start", proposal.get("start_sec"))))
        end = float(proposal.get("offset", proposal.get("end", proposal.get("end_sec"))))
    except Exception:
        return None
    start = max(cursor_sec, max(0.0, start))
    end = min(duration_sec, end)
    if end <= start + 0.1:
        return _next_temporal_fallback_segment(cursor_sec, duration_sec)
    return {
        "onset": round(start, 3),
        "offset": round(end, 3),
        "reason": str(proposal.get("reason") or f"Temporal reasoning turn from {start:.2f}s."),
    }


def _next_temporal_fallback_segment(cursor_sec: float, duration_sec: float) -> dict[str, Any]:
    start = max(0.0, cursor_sec)
    end = min(duration_sec, start + 8.0)
    if end <= start + 0.1:
        end = min(duration_sec, start + 0.5)
    return {
        "onset": round(start, 3),
        "offset": round(end, 3),
        "reason": f"Runtime fallback temporal window from {start:.2f}s.",
    }


def _temporal_seed_windows(
    context: dict[str, Any], *, max_windows: int = 32
) -> list[dict[str, Any]]:
    duration = float(context.get("duration_sec") or 0.0)
    segments = [
        item
        for item in sorted(
            context.get("speaker_segments") or [],
            key=lambda value: (value.get("start_sec", 0.0), value.get("end_sec", 0.0)),
        )
        if float(item.get("end_sec", 0.0)) > float(item.get("start_sec", 0.0))
    ]
    if not segments:
        windows = []
        cursor = 0.0
        while duration > 0.0 and cursor < duration - 0.05 and len(windows) < max_windows:
            end = min(duration, cursor + 8.0)
            windows.append(
                {
                    "onset": round(cursor, 3),
                    "offset": round(end, 3),
                    "reason": f"Seed temporal reasoning from Qwen3ASR text over {cursor:.2f}-{end:.2f}s.",
                    "source": "qwen3_asr_duration_grid",
                }
            )
            cursor = end
        return windows

    windows: list[dict[str, Any]] = []
    current_start: float | None = None
    current_end: float | None = None
    current_speakers: set[str] = set()
    for segment in segments:
        start = max(0.0, float(segment.get("start_sec", 0.0)))
        end = (
            min(duration, float(segment.get("end_sec", start)))
            if duration > 0
            else float(segment.get("end_sec", start))
        )
        if end <= start:
            continue
        speaker = str(segment.get("speaker_id"))
        if current_start is None:
            current_start = start
            current_end = min(end, start + 10.0)
            current_speakers = {speaker}
        elif start <= float(current_end) + 0.5 and end - current_start <= 12.0:
            current_end = max(float(current_end), end)
            current_speakers.add(speaker)
        else:
            windows.append(
                _temporal_seed_window(current_start, float(current_end), current_speakers)
            )
            current_start = start
            current_end = min(end, start + 10.0)
            current_speakers = {speaker}
        while current_end is not None and end - current_end > 0.1 and len(windows) < max_windows:
            windows.append(_temporal_seed_window(current_start, current_end, current_speakers))
            current_start = current_end
            current_end = min(end, current_start + 10.0)
    if current_start is not None and current_end is not None and len(windows) < max_windows:
        windows.append(_temporal_seed_window(current_start, current_end, current_speakers))
    return windows[:max_windows]


def _temporal_seed_window(start: float, end: float, speakers: set[str]) -> dict[str, Any]:
    speaker_text = ", ".join(sorted(speakers)) if speakers else "unknown speakers"
    return {
        "onset": round(start, 3),
        "offset": round(end, 3),
        "reason": f"Seed temporal reasoning from timestamped speaker evidence for {speaker_text}.",
        "source": "timestamped_speaker_timeline",
        "speaker_ids": sorted(speakers),
    }


def _next_temporal_seed_window(context: dict[str, Any], cursor_sec: float) -> dict[str, Any] | None:
    initialization = context.get("temporal_initialization") or {}
    for window in initialization.get("seed_windows") or []:
        try:
            if float(window.get("offset", 0.0)) <= cursor_sec + 0.05:
                continue
            start = max(cursor_sec, float(window.get("onset", 0.0)))
            end = float(window.get("offset", 0.0))
        except Exception:
            continue
        if end <= start + 0.1:
            continue
        seeded = dict(window)
        seeded["onset"] = round(start, 3)
        seeded["offset"] = round(end, 3)
        return seeded
    return None


def _local_temporal_initialization(
    context: dict[str, Any], segment_record: dict[str, Any]
) -> dict[str, Any]:
    start = float(segment_record.get("onset", 0.0))
    end = float(segment_record.get("offset", start))
    padded_start = max(0.0, start - 0.25)
    padded_end = end + 0.25
    base = (context.get("evidence") or {}).get("base_perception") or {}
    local_words = []
    for item in base.get("word_timestamps") or []:
        if not isinstance(item, dict):
            continue
        word_start = item.get("start_sec", item.get("start"))
        word_end = item.get("end_sec", item.get("end"))
        if word_start is None or word_end is None:
            continue
        if _time_ranges_overlap(float(word_start), float(word_end), padded_start, padded_end):
            local_words.append(item)
    return {
        "slice_onset": round(start, 3),
        "slice_offset": round(end, 3),
        "speaker_timeline": _compact_speaker_timeline(
            [
                item
                for item in context.get("speaker_segments") or []
                if _time_ranges_overlap(
                    float(item.get("start_sec", 0.0)),
                    float(item.get("end_sec", 0.0)),
                    padded_start,
                    padded_end,
                )
            ]
        ),
        "qwen_word_timestamps": _compact_word_timestamps(local_words),
    }


def _time_ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _temporal_reasoning_draft(history: list[dict[str, Any]]) -> str:
    lines = []
    for item in history:
        if item.get("status") != "reviewed":
            continue
        raw = str(item.get("raw_output") or "").strip()
        if raw:
            lines.append(raw)
    return "\n".join(lines)


def required_slice_call_count(context: dict[str, Any]) -> int:
    """Compatibility hook: speaker review no longer has slice-count gates."""

    return 0


def _triage_speaker_evidence(context: dict[str, Any]):
    return triage_candidate_evidence_batch(_speaker_triage_candidates(context))


def _speaker_triage_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    segments = list(context.get("speaker_segments") or [])
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        candidate = _speaker_segment_triage_candidate(segment, index=index)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _speaker_segment_triage_candidate(
    segment: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any] | None:
    start = _optional_float(segment.get("start_sec", segment.get("start")))
    end = _optional_float(segment.get("end_sec", segment.get("end")))
    if start is None or end is None:
        return None
    candidate: dict[str, Any] = {
        "candidate_id": str(segment.get("segment_id") or f"speaker_segment_{index:03d}"),
        "segment_id": str(segment.get("segment_id") or f"speaker_segment_{index:03d}"),
        "speaker_id": str(segment.get("speaker_id") or ""),
        "start_sec": start,
        "end_sec": end,
        "source": str(segment.get("source") or "speaker_timeline"),
    }
    confidence = _speaker_segment_confidence(segment)
    if confidence is not None:
        candidate["confidence"] = confidence
    provider_health = segment.get("provider_health") or segment.get("provider_status")
    if provider_health:
        candidate["provider_health"] = provider_health
    if _speaker_segment_needs_review(segment):
        candidate["needs_review"] = True
    boundary_quality = _speaker_segment_boundary_quality(segment)
    if boundary_quality:
        candidate["boundary_quality"] = boundary_quality
    cross_tool_agreement = _speaker_segment_cross_tool_agreement(segment)
    if cross_tool_agreement:
        candidate["cross_tool_agreement"] = cross_tool_agreement
    return candidate


def _speaker_segment_confidence(segment: Mapping[str, Any]) -> float | str | None:
    for field_name in (
        "speaker_confidence",
        "diarization_confidence",
        "attribution_confidence",
        "role_confidence",
        "semantic_role_confidence",
        "gender_confidence",
        "confidence",
    ):
        value = segment.get(field_name)
        if value is not None:
            return value
    return None


def _speaker_segment_needs_review(segment: Mapping[str, Any]) -> bool:
    for field_name in (
        "needs_review",
        "requires_review",
        "requires_model_review",
        "uncertainty",
        "speaker_uncertainty",
        "role_uncertainty",
    ):
        value = segment.get(field_name)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return bool(segment.get("overlap_speaker_ids"))


def _speaker_segment_boundary_quality(segment: Mapping[str, Any]) -> str | None:
    explicit = segment.get("boundary_quality")
    if explicit:
        return str(explicit)
    if segment.get("overlap_speaker_ids") or segment.get("overlap"):
        return "uncertain"
    if segment.get("boundary_uncertainty") or segment.get("timestamp_uncertainty"):
        return "uncertain"
    return "stable"


def _speaker_segment_cross_tool_agreement(segment: Mapping[str, Any]) -> str | None:
    explicit = segment.get("cross_tool_agreement") or segment.get("corroboration_status")
    if explicit:
        return str(explicit)
    disagreement = segment.get("speaker_disagreement") or segment.get("diarization_disagreement")
    if disagreement:
        return "conflict"
    return None


def _targeted_slice_requests_from_triage(
    context: dict[str, Any],
    decisions: tuple[CandidateTriageDecision, ...],
    *,
    max_slices: int = 3,
) -> list[dict[str, Any]]:
    duration = float(context.get("duration_sec") or 0.0)
    requests: list[dict[str, Any]] = []
    local_decisions = sorted(
        (decision for decision in decisions if decision.should_slice_locally),
        key=lambda decision: decision.review_priority,
        reverse=True,
    )
    for decision in local_decisions:
        if len(requests) >= max_slices:
            break
        request = _targeted_slice_request_for_decision(decision, duration=duration)
        if request is None or _overlaps_existing_slice(request, requests):
            continue
        requests.append(request)
    return requests


def _targeted_slice_request_for_decision(
    decision: CandidateTriageDecision,
    *,
    duration: float,
) -> dict[str, Any] | None:
    candidate = decision.candidate
    start = _optional_float(candidate.get("start_sec", candidate.get("start")))
    end = _optional_float(candidate.get("end_sec", candidate.get("end")))
    if start is None or end is None or end <= start:
        return None
    padded_start = max(0.0, start - 0.25)
    padded_end = min(duration, end + 0.25) if duration > 0 else end + 0.25
    reason = _triage_slice_reason(decision)
    if padded_end - padded_start <= 12.0:
        return {
            "onset": round(padded_start, 3),
            "offset": round(padded_end, 3),
            "reason": reason,
        }
    center = (start + end) / 2.0
    return _slice_around(center, duration, reason=reason)


def _triage_slice_reason(decision: CandidateTriageDecision) -> str:
    candidate_id = decision.candidate_id or "speaker evidence candidate"
    reasons = ", ".join(decision.review_reasons) or "speaker evidence uncertainty"
    return f"Review {candidate_id}: {reasons}."


def _speaker_semantic_role_uncertainty(context: dict[str, Any]) -> dict[str, Any] | None:
    needs = context.get("observation_needs") or []
    role_fact_kinds = {
        "speaker_role",
        "speaker_profile",
        "speaker_demographics",
        "speaker_identity_property",
    }
    requires_role_evidence = any(
        isinstance(item, Mapping)
        and str(item.get("fact_kind") or item.get("kind") or "").strip().lower()
        in role_fact_kinds
        for item in needs
    )
    if not requires_role_evidence:
        return None
    base = (context.get("evidence") or {}).get("base_perception") or {}
    profile_fields = ("speaker_profiles", "speakers", "target_bindings")
    if any(_has_semantic_role_evidence(base.get(field)) for field in profile_fields):
        return None
    return {
        "code": "speaker_semantic_role_unresolved",
        "severity": "advisory",
        "blocking": False,
        "message": "Speaker transcript evidence is usable, but requested semantic speaker role remains unresolved.",
    }


def _has_semantic_role_evidence(value: Any) -> bool:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        if keys.intersection({"role", "semantic_role", "perceived_gender", "age_band"}):
            return True
        return any(_has_semantic_role_evidence(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_semantic_role_evidence(item) for item in value)
    return False


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _use_evidence_summary_review(context: dict[str, Any]) -> bool:
    return resolve_agentic_mode(context.get("agentic_mode")) == OPEN_GLM_QWEN_OMNI_MODE


def _default_speaker_review_frontend(context: dict[str, Any]) -> SpeakerReviewFrontend:
    if resolve_agentic_mode(context.get("agentic_mode")) == OPEN_GLM_QWEN_OMNI_MODE:
        return GLMSpeakerEvidenceTextFrontend()
    return GeminiSpeakerReviewFrontend()


def _collect_evidence(
    client: SpeechPerceptionClient,
    audio: Path,
    *,
    base_perception: dict[str, Any],
    run_specialized: bool,
    use_sedicow: bool,
    speaker_evidence_backend: str,
    language: str | None,
    num_speakers: int | None,
    refresh_cache: bool,
) -> dict[str, Any]:
    perception = dict(base_perception or {})
    attempts: list[dict[str, Any]] = []
    initial_language = (
        language or perception.get("language") or (perception.get("asr") or {}).get("language")
    )
    requested_backend = str(speaker_evidence_backend or "auto")
    backend = _normalize_speaker_evidence_backend(requested_backend, language=initial_language)
    auto_waiting_for_asr_language = (
        _is_auto_speaker_evidence_backend(requested_backend) and not initial_language
    )
    initial_use_diarization = backend == "diarizen" and not auto_waiting_for_asr_language
    initial_provider = "qwen3_asr_diarizen" if initial_use_diarization else "qwen3_asr"
    if run_specialized:
        try:
            specialized = client.perceive(
                audio,
                language=language,
                use_asr=True,
                use_alignment=True,
                use_diarization=initial_use_diarization,
                num_speakers=num_speakers,
                refresh_cache=refresh_cache,
                strict_asr=False,
                strict_alignment=False,
                strict_diarization=False,
            )
            attempts.append(
                {
                    "provider": initial_provider,
                    "status": "completed",
                }
            )
            perception = _merge_perception(perception, specialized)
        except Exception as exc:
            attempts.append(
                {
                    "provider": initial_provider,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            perception.setdefault("errors", []).append(
                {"stage": "specialized_speaker_review", "message": str(exc)}
            )
            if backend == "diarizen":
                try:
                    speaker_only = client.perceive(
                        audio,
                        language=language,
                        use_asr=False,
                        use_alignment=False,
                        use_diarization=True,
                        num_speakers=num_speakers,
                        refresh_cache=refresh_cache,
                        strict_asr=False,
                        strict_alignment=False,
                        strict_diarization=False,
                    )
                    attempts.append(
                        {"provider": "diarizen_after_asr_failure", "status": "completed"}
                    )
                    perception = _merge_perception(perception, speaker_only)
                except Exception as diarization_exc:
                    attempts.append(
                        {
                            "provider": "diarizen_after_asr_failure",
                            "status": "failed",
                            "error": f"{type(diarization_exc).__name__}: {diarization_exc}",
                        }
                    )
                    perception.setdefault("errors", []).append(
                        {"stage": "diarization_after_asr_failure", "message": str(diarization_exc)}
                    )

    resolved_language = perception.get("language") or language
    backend = _normalize_speaker_evidence_backend(requested_backend, language=resolved_language)
    if run_specialized and backend == "diarizen" and not perception.get("diarization"):
        try:
            speaker_only = client.perceive(
                audio,
                language=resolved_language,
                use_asr=False,
                use_alignment=False,
                use_diarization=True,
                num_speakers=num_speakers,
                refresh_cache=refresh_cache,
                strict_asr=False,
                strict_alignment=False,
                strict_diarization=False,
            )
            attempts.append({"provider": "diarizen_after_language_routing", "status": "completed"})
            perception = _merge_perception(perception, speaker_only)
        except Exception as exc:
            attempts.append(
                {
                    "provider": "diarizen_after_language_routing",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            perception.setdefault("errors", []).append(
                {"stage": "diarization_after_language_routing", "message": str(exc)}
            )

    sedicow_results: list[dict[str, Any]] = []
    moss_result: dict[str, Any] | None = None
    if backend == "moss":
        try:
            moss_result = client.transcribe_diarize(
                audio,
                qwen_text=perception.get("source_text"),
                language=perception.get("language") or resolved_language,
                reason="Chinese speaker-attributed transcript initialization evidence",
            )
            moss_segments = _segments_from_moss_result(moss_result)
            if moss_segments:
                perception["speaker_id_namespace"] = "moss_transcribe_diarize"
                perception["speaker_segments"] = moss_segments
                perception["segments"] = moss_segments
                perception["speakers"] = sorted(
                    {segment["speaker_id"] for segment in moss_segments}
                )
                perception["moss_transcribe_diarize"] = moss_result
                if not perception.get("source_text") and moss_result.get("text"):
                    perception["source_text"] = moss_result.get("text")
            attempts.append(
                {
                    "provider": "moss_transcribe_diarize_primary",
                    "status": "completed",
                    "segment_count": len(moss_segments),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "provider": "moss_transcribe_diarize_primary",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            perception.setdefault("errors", []).append(
                {"stage": "moss_transcribe_diarize_primary", "message": str(exc)}
            )

    if use_sedicow and backend in {"sedicow", "diarizen"}:
        # The official pipeline runs its own diarization. External numeric IDs
        # cannot select its speakers. Collect one complete, native report.
        provider = "sedicow_asr_primary" if backend == "sedicow" else "sedicow_asr"
        try:
            result = client.transcribe_target_speaker(
                audio,
                speaker_id=None,
                diarization_segments=[],
                qwen_text=perception.get("source_text"),
                language=perception.get("language") or resolved_language,
                reason="speaker-attributed transcript review primary speaker evidence",
            )
            sedicow_results.append(result)
            sedicow_segments = _segments_from_sedicow_result(result)
            if sedicow_segments:
                perception["speaker_id_namespace"] = "sedicow_asr"
                perception["speaker_segments"] = sedicow_segments
                perception["segments"] = sedicow_segments
                perception["speakers"] = sorted(
                    {segment["speaker_id"] for segment in sedicow_segments}
                )
                perception["sedicow_asr"] = result
                if not perception.get("source_text") and result.get("text"):
                    perception["source_text"] = result.get("text")
            attempts.append(
                {
                    "provider": provider,
                    "status": "completed",
                    "segment_count": len(sedicow_segments),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "provider": provider,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            perception.setdefault("errors", []).append(
                {"stage": "sedicow_asr_primary", "message": str(exc)}
            )

    diarization_segments = _segments_from_perception(perception)
    return {
        "base_perception": perception,
        "qwen_asr": perception.get("asr"),
        "diarizen": perception.get("diarization"),
        "sedicow": sedicow_results,
        "moss": moss_result,
        "source_text": perception.get("source_text") or "",
        "language": perception.get("language") or resolved_language,
        "speaker_segments": diarization_segments,
        "speaker_id_namespace": perception.get("speaker_id_namespace", "diarizen"),
        "auxiliary_speaker_namespaces": {"diarizen": "diarizen"},
        "speaker_evidence_backend": backend,
        "speaker_evidence_backend_request": requested_backend,
        "speaker_evidence_language_routing": {
            "requested_language": language,
            "resolved_language": perception.get("language") or resolved_language,
            "selected_backend": backend,
        },
        "attempts": attempts,
        "service_bootstrap": getattr(client, "bootstrap_reports", []),
        "errors": perception.get("errors") or [],
    }


def _normalize_speaker_evidence_backend(value: str | None, *, language: str | None = None) -> str:
    text = str(value or "auto").strip().lower().replace("-", "_")
    if text in {
        "moss",
        "moss_transcribe",
        "moss_transcribe_diarize",
        "moss_transcribe_diarize_primary",
    }:
        return "moss"
    if text in {"sedicow", "sedicow_asr", "sedicow_primary", "sedicow_only"}:
        return "sedicow"
    if text in {"auto", "default"}:
        return "moss" if _is_chinese_language(language) else "diarizen"
    return "diarizen"


def _is_auto_speaker_evidence_backend(value: str | None) -> bool:
    text = str(value or "auto").strip().lower().replace("-", "_")
    return text in {"", "auto", "default"}


def _is_chinese_language(language: str | None) -> bool:
    if not language:
        return False
    text = str(language).strip().lower()
    return text in {"zh", "zho", "chi", "chinese", "mandarin", "cmn", "cn"}


def _segments_from_sedicow_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    for item in result.get("segments") or []:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        speaker = item.get("speaker_id", item.get("speaker", item.get("label")))
        if start is None or end is None or speaker is None:
            continue
        segments.append(
            {
                "speaker_id": str(speaker),
                "start_sec": float(start),
                "end_sec": float(end),
                "transcript": str(item.get("transcript") or item.get("text") or "").strip(),
                "source": "sedicow_asr",
            }
        )
    return sorted(
        segments, key=lambda item: (item["start_sec"], item["end_sec"], item["speaker_id"])
    )


def _segments_from_moss_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = _segments_from_sedicow_result(result)
    for segment in segments:
        segment["source"] = "moss_transcribe_diarize"
    return segments


def _review_context(
    audio: Path,
    *,
    instruction: str,
    evidence: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    duration = audio_duration_sec(audio)
    segments = _segments_from_perception(evidence.get("base_perception") or {})
    if not segments:
        segments = _single_speaker_segments_from_source(evidence, duration)
    segments = _normalize_single_speaker_segments(segments)
    allowed = _allowed_speaker_map(segments, evidence.get("sedicow") or [])
    switch_count = _speaker_switch_count(segments)
    overlap_class = _speech_overlap_class(segments)
    context = {
        "audio_path": str(audio),
        "instruction": instruction,
        "output_dir": str(output_dir),
        "duration_sec": duration,
        "source_text": evidence.get("source_text") or "",
        "language": evidence.get("language"),
        "speaker_segments": segments,
        "speaker_id_namespace": evidence.get("speaker_id_namespace"),
        "allowed_speaker_map": allowed,
        "turn_speaker_switch_count": switch_count,
        "speech_overlap_class": overlap_class,
        "evidence": evidence,
    }
    context["temporal_initialization"] = _temporal_reasoning_initialization(context)
    return context


def _temporal_reasoning_initialization(context: dict[str, Any]) -> dict[str, Any]:
    evidence = context.get("evidence") or {}
    base = evidence.get("base_perception") or {}
    backend = str(evidence.get("speaker_evidence_backend") or "moss")
    return {
        "strategy": "speaker_timeline_plus_qwen3_asr",
        "speaker_timeline_source": _speaker_timeline_source_label(context),
        "speaker_evidence_backend": backend,
        "qwen_asr_text": context.get("source_text") or "",
        "qwen_word_timestamps": _compact_word_timestamps(base.get("word_timestamps") or []),
        "speaker_timeline": _compact_speaker_timeline(context.get("speaker_segments") or []),
        "seed_windows": _temporal_seed_windows(context),
    }


def _speaker_timeline_source_label(context: dict[str, Any]) -> str:
    evidence = context.get("evidence") or {}
    backend = str(evidence.get("speaker_evidence_backend") or "moss")
    if backend == "moss":
        return "moss_transcribe_diarize_timestamped_speaker_asr"
    if backend == "sedicow" or evidence.get("speaker_id_namespace") == "sedicow_asr":
        return "sedicow_timestamped_speaker_asr"
    if evidence.get("sedicow"):
        return "diarizen_timestamped_speakers_with_sedicow_hints"
    return "diarizen_timestamped_speakers"


def _compact_speaker_timeline(
    segments: list[dict[str, Any]], *, limit: int = 80
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in segments[:limit]:
        record = {
            "speaker_id": str(item.get("speaker_id")),
            "start_sec": round(float(item.get("start_sec", 0.0)), 3),
            "end_sec": round(float(item.get("end_sec", 0.0)), 3),
        }
        text = str(item.get("transcript") or item.get("text") or "").strip()
        if text:
            record["text"] = text
        source = item.get("source")
        if source:
            record["source"] = str(source)
        compact.append(record)
    if len(segments) > limit:
        compact.append({"truncated_count": len(segments) - limit})
    return compact


def _compact_word_timestamps(
    words: list[dict[str, Any]], *, limit: int = 120
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in words[:limit]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("word") or item.get("char") or "").strip()
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        if not text or start is None or end is None:
            continue
        compact.append(
            {
                "text": text,
                "start_sec": round(float(start), 3),
                "end_sec": round(float(end), 3),
            }
        )
    if len(words) > limit:
        compact.append({"truncated_count": len(words) - limit})
    return compact


def _merge_perception(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    if base.get("errors") and update.get("errors"):
        merged["errors"] = [*base.get("errors", []), *update.get("errors", [])]
    return merged


def _segments_from_perception(perception: dict[str, Any]) -> list[dict[str, Any]]:
    raw = perception.get("speaker_segments") or perception.get("segments") or []
    if not raw and isinstance(perception.get("diarization"), dict):
        raw = perception["diarization"].get("segments") or []
    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start"))
        end = item.get("end_sec", item.get("end"))
        speaker = item.get("speaker_id", item.get("speaker", item.get("label")))
        if start is None or end is None or speaker is None:
            continue
        segment = dict(item)
        segment["speaker_id"] = str(speaker)
        segment["start_sec"] = float(start)
        segment["end_sec"] = float(end)
        segments.append(segment)
    return sorted(segments, key=lambda item: (item["start_sec"], item["end_sec"]))


def _single_speaker_segments_from_source(
    evidence: dict[str, Any], duration: float | None
) -> list[dict[str, Any]]:
    source_text = str(evidence.get("source_text") or "").strip()
    if not source_text:
        base = evidence.get("base_perception") or {}
        source_text = str(
            (base.get("asr") or {}).get("text") or base.get("source_text") or ""
        ).strip()
    if not source_text:
        return []
    end = float(duration or 0.0)
    if end <= 0.0:
        end = (
            _last_word_end((evidence.get("base_perception") or {}).get("word_timestamps") or [])
            or 0.01
        )
    return [
        {
            "speaker_id": "speaker_0",
            "start_sec": 0.0,
            "end_sec": round(max(0.01, end), 6),
            "transcript": source_text,
            "source": "qwen3_asr_single_speaker_default",
            "uncertainty": "speaker diarization unavailable; defaulted to one speaker for text-only review",
        }
    ]


def _normalize_single_speaker_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speaker_ids = [
        str(item.get("speaker_id")) for item in segments if item.get("speaker_id") is not None
    ]
    unique_ids = []
    for speaker_id in speaker_ids:
        if speaker_id not in unique_ids:
            unique_ids.append(speaker_id)
    if len(unique_ids) != 1:
        return segments
    original = unique_ids[0]
    normalized = []
    for item in segments:
        segment = dict(item)
        if str(segment.get("speaker_id")) != "speaker_0":
            segment.setdefault("original_speaker_id", str(segment.get("speaker_id")))
        segment["speaker_id"] = "speaker_0"
        normalized.append(segment)
    if original == "speaker_0":
        return segments
    return normalized


def _last_word_end(words: list[dict[str, Any]]) -> float | None:
    ends = []
    for item in words:
        if not isinstance(item, dict):
            continue
        end = item.get("end_sec", item.get("end"))
        if end is None:
            continue
        try:
            ends.append(float(end))
        except (TypeError, ValueError):
            continue
    return max(ends) if ends else None


def _allowed_speaker_map(
    segments: list[dict[str, Any]], sedicow_results: list[dict[str, Any]]
) -> dict[str, str]:
    segment_speakers = [
        str(item.get("speaker_id")) for item in segments if item.get("speaker_id") is not None
    ]
    if len(set(segment_speakers)) == 1:
        return {"speaker 0": "speaker_0"}
    candidates = [
        str(item.get("speaker_id")) for item in segments if item.get("speaker_id") is not None
    ]
    for result in sedicow_results:
        if isinstance(result, dict) and result.get("speaker_id") is not None:
            candidates.append(str(result["speaker_id"]))
        for segment in result.get("segments") or []:
            if isinstance(segment, dict) and segment.get("speaker_id") is not None:
                candidates.append(str(segment["speaker_id"]))
    mapping: dict[str, str] = {}
    for speaker_id in candidates:
        label = _numeric_speaker_label(speaker_id)
        if label and label not in mapping:
            mapping[label] = _canonical_speaker_id(label)
    return mapping


def _numeric_speaker_label(speaker_id: str) -> str | None:
    text = str(speaker_id).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return f"speaker {int(match.group(1))}"


def _canonical_speaker_id(label: str) -> str:
    number = int(re.search(r"\d+", label).group(0))
    return f"speaker_{number}"


def _speaker_switch_count(segments: list[dict[str, Any]]) -> int:
    count = 0
    prev = None
    for segment in segments:
        speaker = segment.get("speaker_id")
        if prev is not None and speaker != prev:
            count += 1
        prev = speaker
    return count


def _speech_overlap_class(segments: list[dict[str, Any]]) -> str:
    """Describe interval geometry; overlapping speech is not a failed review."""
    previous_end = None
    for segment in segments:
        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", start))
        if previous_end is not None and start < previous_end - 0.05:
            return "overlapping_intervals"
        previous_end = max(previous_end or end, end)
    return "clean"


def _complete_slice_requests(
    context: dict[str, Any],
    requests: list[dict[str, Any]],
    min_slices: int,
) -> list[dict[str, Any]]:
    valid = [_normalize_slice_request(item) for item in requests if item]
    return [item for item in valid if item["offset"] > item["onset"] and item["reason"]]


def _normalize_slice_request(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "onset": float(item.get("onset", item.get("start", item.get("start_sec", 0.0)))),
        "offset": float(item.get("offset", item.get("end", item.get("end_sec", 0.0)))),
        "reason": str(item.get("reason") or item.get("uncertainty") or "").strip(),
    }


def _heuristic_slice_requests(context: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _slice_around(center: float, duration: float, *, reason: str) -> dict[str, Any]:
    width = 8.0
    if duration and duration < width:
        width = max(1.0, duration)
    start = max(0.0, center - width / 2.0)
    end = start + width
    if duration > 0 and end > duration:
        end = duration
        start = max(0.0, end - width)
    return {"onset": round(start, 3), "offset": round(end, 3), "reason": reason}


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
        records.append(
            slice_audio(
                audio,
                onset=float(request["onset"]),
                offset=float(request["offset"]),
                output_dir=output_dir,
                reason=str(request["reason"]),
                clip_id=f"slice_{index:02d}_{request['onset']:.2f}_{request['offset']:.2f}",
            )
        )
    return records


def _perception_from_review(
    evidence: dict[str, Any],
    lines: list[SpeakerReviewLine],
    context: dict[str, Any],
) -> dict[str, Any]:
    base = dict(evidence.get("base_perception") or {})
    reviewed_speaker_text = " ".join(line.text for line in lines).strip()
    segments = [
        {
            "speaker_id": line.speaker_id,
            "speaker_label": line.speaker_label,
            "start_sec": round(line.start_sec, 6),
            "end_sec": round(line.end_sec, 6),
            "transcript": line.text,
            "source": "speaker_attributed_review",
        }
        for line in lines
    ]
    speakers = [
        {"speaker_id": speaker_id, "speaker_label": label}
        for label, speaker_id in sorted(context.get("allowed_speaker_map", {}).items())
    ]
    # Speaker review owns attribution and segment wording, but the dedicated
    # ASR observation remains authoritative for the full lexical transcript.
    # This prevents a reviewer paraphrase from changing values that downstream
    # semantic reasoning treats as literal spoken content.
    base["source_text"] = (
        base.get("source_text") or evidence.get("source_text") or reviewed_speaker_text
    )
    base["speaker_segments"] = segments
    base["segments"] = segments
    base["speakers"] = speakers
    base.setdefault("services", {})
    if isinstance(base["services"], dict):
        service_record = {
            "qwen3_asr": "edit-data-pipeline ASR service",
            "diarizen": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
            "sedicow_asr": "BUTSpeechFIT/TS-ASR-Whisper adapter",
        }
        if _use_evidence_summary_review(context):
            service_record["glm_text_speaker_review"] = GLM_PROVIDER
        else:
            service_record["strong_omni_review"] = "gemini"
        base["services"].update(service_record)
    review_source = (
        f"Qwen3ASR+{evidence.get('speaker_evidence_backend', 'moss')}+GLMTextEvidenceIntegrator"
        if _use_evidence_summary_review(context)
        else f"Qwen3ASR+{evidence.get('speaker_evidence_backend', 'moss')}+StrongOmniModel"
    )
    base["speaker_attributed_review"] = {
        "source": review_source,
        "speech_overlap_class": context.get("speech_overlap_class"),
        "turn_speaker_switch_count": context.get("turn_speaker_switch_count"),
        "evidence_triage": context.get("speaker_evidence_triage"),
    }
    blind = context.get("specialist_prior_visible") is False
    if blind:
        base["services"] = {"strong_omni_review": "audio_only_observation"}
        base["speaker_attributed_review"].update(
            source="StrongOmni_without_specialist_prior", specialist_prior_visible=False,
        )
    semantic_uncertainty = context.get("speaker_semantic_role_uncertainty")
    if semantic_uncertainty:
        base.setdefault("uncertainties", [])
        if isinstance(base["uncertainties"], list):
            base["uncertainties"].append(dict(semantic_uncertainty))
    base, _conflicts = reconcile_canonical_reviewed_report(
        base,
        authority_evidence_ids={
            ObservationKind.LEXICAL_CONTENT: ["speaker_review:audio_observation" if blind else "speaker_review:specialist_asr"],
            ObservationKind.WORD_TIMING: ["speaker_review:audio_observation" if blind else "speaker_review:specialist_alignment"],
            ObservationKind.SEGMENT_OWNERSHIP: ["speaker_review:attribution"],
            ObservationKind.SPEAKER_IDENTITY: ["speaker_review:identity"],
        },
    )
    if isinstance(base.get("speaker_segments"), list):
        base["segments"] = [dict(item) for item in base["speaker_segments"]]
    return base


def _slice_proposal_user_text(context: dict[str, Any], *, min_slices: int) -> str:
    payload = {
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "min_slices": min_slices,
        "qwen_text": context.get("source_text"),
        "allowed_speaker_labels": sorted(context.get("allowed_speaker_map") or {}),
        "speech_overlap_class": context.get("speech_overlap_class"),
        "turn_speaker_switch_count": context.get("turn_speaker_switch_count"),
        "speaker_evidence_backend": (context.get("evidence") or {}).get("speaker_evidence_backend"),
        "speaker_evidence_segments": context.get("speaker_segments"),
        "diarizen_segments": context.get("speaker_segments"),
        "sedicow": (context.get("evidence") or {}).get("sedicow"),
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
        "specialist_asr_text_authoritative_wording": context.get("source_text"),
        "allowed_speaker_labels": sorted(context.get("allowed_speaker_map") or {}),
        "speech_overlap_class": context.get("speech_overlap_class"),
        "turn_speaker_switch_count": context.get("turn_speaker_switch_count"),
        "speaker_evidence_backend": (context.get("evidence") or {}).get("speaker_evidence_backend"),
        "speaker_evidence_segments": context.get("speaker_segments"),
        "diarizen_segments": context.get("speaker_segments"),
        "sedicow_evidence": (context.get("evidence") or {}).get("sedicow"),
        "speaker_evidence_triage": context.get("speaker_evidence_triage"),
        "speaker_semantic_role_uncertainty": context.get("speaker_semantic_role_uncertainty"),
        "semantic_role_policy": "Do not infer host/guest/gender/age roles unless directly supported by evidence.",
        "temporal_reasoning_draft": context.get("temporal_reasoning_draft"),
        "temporal_reasoning_history": context.get("temporal_reasoning_history"),
        "slice_audio_history": slice_records,
        "previous_verification_errors": verification_feedback,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _quality_check_user_text(
    context: dict[str, Any],
    *,
    candidate_transcript: str,
    slice_records: list[dict[str, Any]],
    verification: ReviewVerificationResult,
) -> str:
    payload = {
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "specialist_asr_text_authoritative_wording": context.get("source_text"),
        "allowed_speaker_labels": sorted(context.get("allowed_speaker_map") or {}),
        "speech_overlap_class": context.get("speech_overlap_class"),
        "turn_speaker_switch_count": context.get("turn_speaker_switch_count"),
        "speaker_evidence_backend": (context.get("evidence") or {}).get("speaker_evidence_backend"),
        "speaker_evidence_segments": context.get("speaker_segments"),
        "sedicow_evidence": (context.get("evidence") or {}).get("sedicow"),
        "speaker_evidence_triage": context.get("speaker_evidence_triage"),
        "speaker_semantic_role_uncertainty": context.get("speaker_semantic_role_uncertainty"),
        "slice_audio_history": slice_records,
        "candidate_transcript": candidate_transcript,
        "format_verification": verification.to_dict(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _glm_text_review_user_text(
    context: dict[str, Any],
    *,
    draft_segments: list[dict[str, Any]],
    verification_feedback: list[str],
) -> str:
    evidence = context.get("evidence") or {}
    base = evidence.get("base_perception") or {}
    payload = {
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "allowed_speaker_labels": sorted(context.get("allowed_speaker_map") or {}),
        "speaker_label_policy": (
            "Do not change labels. Single-speaker evidence is always rendered as [speaker 0]. "
            "Return only text for the provided segment_index values."
        ),
        "timestamp_policy": "Timestamps are fixed by specialist speaker evidence and will be rendered by X-Editor.",
        "qwen3_asr_text": context.get("source_text") or "",
        "qwen3_word_timestamps": _compact_word_timestamps(base.get("word_timestamps") or []),
        "speaker_timeline_source": _speaker_timeline_source_label(context),
        "speaker_timeline": _compact_speaker_timeline(context.get("speaker_segments") or []),
        "draft_segments": draft_segments,
        "previous_verification_errors": verification_feedback,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _strong_omni_speaker_review_response_format(
    allowed_speaker_map: Mapping[str, str],
    *,
    schema_name: str = "StrongOmniSpeakerReview",
    error_label: str = "StrongOmni speaker review",
    allow_new_labels: bool = False,
) -> dict[str, Any]:
    allowed_labels = sorted(str(label) for label in allowed_speaker_map)
    if not allowed_labels and not allow_new_labels:
        raise SpeechPerceptionError(
            f"{error_label} requires at least one allowed speaker label"
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "transcript_lines": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start_sec": {"type": "number", "minimum": 0},
                                "end_sec": {"type": "number", "exclusiveMinimum": 0},
                                "speaker_label": {
                                    "type": "string",
                                    **({"pattern": r"^speaker [0-9]+$"} if allow_new_labels
                                       else {"enum": allowed_labels}),
                                },
                                "text": {"type": "string", "minLength": 1},
                            },
                            "required": [
                                "start_sec",
                                "end_sec",
                                "speaker_label",
                                "text",
                            ],
                        },
                    }
                },
                "required": ["transcript_lines"],
            },
        },
    }


def _render_strong_omni_speaker_review_transcript(
    model_data: Mapping[str, Any],
    *,
    allowed_speaker_map: Mapping[str, str],
    error_label: str = "StrongOmni speaker review",
) -> str:
    if not isinstance(model_data, Mapping):
        raise SpeechPerceptionError(f"{error_label} returned a non-object payload")
    if set(model_data) != {"transcript_lines"}:
        raise SpeechPerceptionError(
            f"{error_label} payload must contain only `transcript_lines`"
        )
    raw_lines = model_data["transcript_lines"]
    if not isinstance(raw_lines, list) or not raw_lines:
        raise SpeechPerceptionError(
            f"{error_label} `transcript_lines` must be a non-empty array"
        )

    rendered_lines: list[str] = []
    required_fields = {"start_sec", "end_sec", "speaker_label", "text"}
    for index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, Mapping) or set(raw_line) != required_fields:
            raise SpeechPerceptionError(
                f"{error_label} line "
                f"{index} must contain exactly {sorted(required_fields)}"
            )
        start_sec = _validated_strong_omni_review_time(
            raw_line["start_sec"], field="start_sec", error_label=error_label
        )
        end_sec = _validated_strong_omni_review_time(
            raw_line["end_sec"], field="end_sec", error_label=error_label
        )
        if start_sec < 0 or end_sec <= start_sec:
            raise SpeechPerceptionError(
                f"{error_label} line {index} has an invalid time span"
            )

        speaker_label = raw_line["speaker_label"]
        if not isinstance(speaker_label, str) or speaker_label not in allowed_speaker_map:
            raise SpeechPerceptionError(
                f"{error_label} line {index} uses a disallowed speaker label"
            )
        text = raw_line["text"]
        if (
            not isinstance(text, str)
            or not text
            or text != text.strip()
            or "\n" in text
            or "\r" in text
        ):
            raise SpeechPerceptionError(
                f"{error_label} line {index} has non-renderable transcript text"
            )
        rendered_lines.append(
            f"[{_render_strong_omni_review_time(start_sec)}-{_render_strong_omni_review_time(end_sec)}] "
            f"[{speaker_label}] : {text}"
        )
    return "\n".join(rendered_lines)


def _validated_strong_omni_review_time(
    value: Any,
    *,
    field: str,
    error_label: str,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeechPerceptionError(
            f"{error_label} `{field}` must be a JSON number"
        )
    if not math.isfinite(float(value)):
        raise SpeechPerceptionError(f"{error_label} `{field}` must be finite")
    return value


def _render_strong_omni_review_time(value: float) -> str:
    if float(value) == 0.0:
        return "0"
    return format(Decimal(str(value)), "f")


def _glm_speaker_text_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "SpeakerEvidenceTextReview",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "segment_index": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["segment_index", "text"],
                        },
                    }
                },
                "required": ["segments"],
            },
        },
    }


def _speaker_text_review_draft_segments(context: dict[str, Any]) -> list[dict[str, Any]]:
    base = (context.get("evidence") or {}).get("base_perception") or {}
    words = base.get("word_timestamps") or []
    source_text = str(context.get("source_text") or "").strip()
    allowed_map = context.get("allowed_speaker_map") or {}
    raw_segments = [
        item
        for item in context.get("speaker_segments") or []
        if float(item.get("end_sec", 0.0)) > float(item.get("start_sec", 0.0))
    ]
    draft: list[dict[str, Any]] = []
    for segment in raw_segments:
        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", start))
        label = _canonical_review_speaker_label(str(segment.get("speaker_id") or ""), allowed_map)
        if label is None and len(allowed_map) == 1:
            label = next(iter(allowed_map))
        if label is None:
            continue
        text = str(segment.get("transcript") or segment.get("text") or "").strip()
        if not text:
            text = _text_from_words_for_span(words, start, end)
        if not text and len(raw_segments) == 1:
            text = source_text
        if not text and _should_skip_textless_speaker_segment(segment, start=start, end=end):
            continue
        draft.append(
            {
                "segment_index": len(draft),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "speaker_label": label,
                "speaker_id": allowed_map[label],
                "front_end_speaker_id": str(segment.get("speaker_id") or ""),
                "text_prior": text,
                "source": segment.get("source") or _speaker_timeline_source_label(context),
            }
        )
    if not draft and source_text:
        duration = float(context.get("duration_sec") or 0.0)
        end = max(0.01, duration or _last_word_end(words) or 0.01)
        draft.append(
            {
                "segment_index": 0,
                "start_sec": 0.0,
                "end_sec": round(end, 3),
                "speaker_label": "speaker 0",
                "speaker_id": "speaker_0",
                "front_end_speaker_id": "speaker_0",
                "text_prior": source_text,
                "source": "qwen3_asr_single_speaker_default",
            }
        )
    return draft


def _should_skip_textless_speaker_segment(
    segment: dict[str, Any], *, start: float, end: float
) -> bool:
    if str(segment.get("transcript") or segment.get("text") or "").strip():
        return False
    duration = max(0.0, end - start)
    return duration <= 0.5


def _render_glm_text_review_transcript(
    context: dict[str, Any],
    *,
    draft_segments: list[dict[str, Any]],
    model_data: dict[str, Any],
) -> str:
    if not draft_segments:
        raise SpeechPerceptionError(
            "open speaker-attributed GLM text review has no speaker timeline or ASR text to integrate"
        )
    text_by_index = _glm_segment_text_by_index(model_data, segment_count=len(draft_segments))
    lines = []
    for index, draft in enumerate(draft_segments):
        model_text = _clean_review_text(text_by_index.get(index) or "")
        text = model_text or _clean_review_text(draft.get("text_prior") or "")
        if not text:
            raise SpeechPerceptionError(
                f"open speaker-attributed GLM text review returned empty text for segment_index={index}"
            )
        lines.append(
            f"[{float(draft['start_sec']):.2f}-{float(draft['end_sec']):.2f}] "
            f"[{draft['speaker_label']}] : {text}"
        )
    return "\n".join(lines)


def _glm_segment_text_by_index(model_data: dict[str, Any], *, segment_count: int) -> dict[int, str]:
    raw_segments = None
    if isinstance(model_data, dict):
        raw_segments = (
            model_data.get("segments")
            or model_data.get("transcript_segments")
            or model_data.get("lines")
        )
    if not isinstance(raw_segments, list):
        return {}
    mapped: dict[int, str] = {}
    for fallback_index, item in enumerate(raw_segments):
        if isinstance(item, str):
            index = fallback_index
            text = item
        elif isinstance(item, dict):
            raw_index = item.get(
                "segment_index", item.get("index", item.get("line_index", fallback_index))
            )
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                index = fallback_index
            text = str(
                item.get("text") or item.get("transcript") or item.get("normalized_text") or ""
            )
        else:
            continue
        if index >= segment_count and 0 <= index - 1 < segment_count:
            index -= 1
        if 0 <= index < segment_count:
            mapped[index] = text
    return mapped


def _text_from_words_for_span(words: list[dict[str, Any]], start: float, end: float) -> str:
    tokens = []
    padded_start = max(0.0, start - 0.15)
    padded_end = end + 0.15
    for item in words:
        if not isinstance(item, dict):
            continue
        word = str(item.get("text") or item.get("word") or item.get("char") or "").strip()
        if not word:
            continue
        word_start = item.get("start_sec", item.get("start"))
        word_end = item.get("end_sec", item.get("end"))
        if word_start is None or word_end is None:
            continue
        try:
            if _time_ranges_overlap(float(word_start), float(word_end), padded_start, padded_end):
                tokens.append(word)
        except (TypeError, ValueError):
            continue
    return _join_word_tokens(tokens)


def _join_word_tokens(tokens: list[str]) -> str:
    text = ""
    no_space_before = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "'", '"'}
    no_space_after = {"(", "[", "{", "$"}
    for token in tokens:
        if not text:
            text = token
        elif token in no_space_before or text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return text.strip()


def _clean_review_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(
        r"^\[(?:\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?|speaker[^\]]+)\]\s*", "", cleaned, flags=re.I
    )
    return cleaned.strip()


def _temporal_segment_proposal_user_text(
    context: dict[str, Any],
    *,
    temporal_history: list[dict[str, Any]],
    cursor_sec: float,
) -> str:
    payload = {
        "instruction": context.get("instruction"),
        "duration_sec": context.get("duration_sec"),
        "cursor_sec": cursor_sec,
        "temporal_initialization": context.get("temporal_initialization"),
        "next_seed_window": context.get("temporal_reasoning_next_seed_window"),
        "specialist_asr_text_authoritative_wording": context.get("source_text"),
        "allowed_speaker_labels": sorted(context.get("allowed_speaker_map") or {}),
        "speaker_evidence_backend": (context.get("evidence") or {}).get("speaker_evidence_backend"),
        "speaker_evidence_segments": context.get("speaker_segments"),
        "previous_temporal_turns": _compact_temporal_history(temporal_history),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _temporal_segment_review_user_text(
    context: dict[str, Any],
    *,
    segment_record: dict[str, Any],
    temporal_history: list[dict[str, Any]],
) -> str:
    payload = {
        "instruction": context.get("instruction"),
        "slice_onset": segment_record.get("onset"),
        "slice_offset": segment_record.get("offset"),
        "slice_reason": segment_record.get("reason"),
        "duration_sec": context.get("duration_sec"),
        "temporal_initialization": context.get("temporal_initialization"),
        "local_initialization": _local_temporal_initialization(context, segment_record),
        "specialist_asr_text_authoritative_wording": context.get("source_text"),
        "allowed_speaker_labels": sorted(context.get("allowed_speaker_map") or {}),
        "speaker_evidence_segments": context.get("speaker_segments"),
        "previous_temporal_turns": _compact_temporal_history(temporal_history),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_temporal_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in history:
        compact.append(
            {
                "turn_index": item.get("turn_index"),
                "status": item.get("status"),
                "slice_request": item.get("slice_request"),
                "raw_output": item.get("raw_output"),
                "verification": item.get("verification"),
            }
        )
    return compact
