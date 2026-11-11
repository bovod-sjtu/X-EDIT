"""LLM-guided perception planning helpers for edit-span grounding."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from x_editor.agents.base import GeminiClient
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key

PROMPT1_AUDIO_PERCEPTION_CALL_PLANNING = """You are X-Editor's planner-side audio perception triage agent.

Inputs:
- A user edit instruction.
- One or more attached audios in the same order as audio_items.
- The executable perception/runtime operations available to the planner.

Task:
1. Listen to each attached audio and produce a concise caption summary for that
   audio. Distinguish source audio, reference audio, and ambiguous extra audio.
2. Decide which perception module(s) the downstream planner must call before
   selecting edit spans.
3. Explain the missing evidence each perception module must fill. Do not invent
   exact timestamps, speaker identities, or phrase locations from weak evidence.

Module policy:
- Use perception.speech_context for transcript/source_text and word-timestamp
  grounding when exact lexical matching or Planner/Reasoner classification of
  transcript text is sufficient. A semantic category is not a verbatim
  quoted-text target, but purely linguistic categories may still use transcript
  evidence with exact boundary provenance.
- Use perception.speaker_attributed_review for multi-speaker speech, speaker
  order, speaker switches, overlap, a speaker's question/answer, a quoted
  phrase that identifies a speaker, transcript content, sentence boundaries,
  or any speaker-scoped utterance. Speech/speaker evidence has priority over
  generic caption/event tools even when those tools mention "speech", "male
  speaker", "question", or "overlap".
- Use perception.semantic_span_review when TS-ASR/alignment can form local
  candidates but the target requires listening to emotion, accent, acoustic
  quality, vocalization type, or another audio-dependent property. It may also
  review a linguistically ambiguous category when transcript reasoning is insufficient.
- When the target combines a text clue with any audio-dependent condition, such
  as "the person counting from far away", choose a speaker/semantic review route
  that can verify both conditions; transcript-only evidence is insufficient.
- Use perception.audio_fallback for direct Omni semantic review when aligned
  local candidates are not required.
- Use perception.timestamped_caption for broad non-speech caption/event
  grounding. FlexSED is an optional prior for concrete audible event labels;
  use perception.flexsed_events only when event-detector evidence itself is the
  selected route. Do not use FlexSED for words, speakers, language, accent,
  emotion, or timbre.
- Treat non-verbal vocal detection as an audio-dependent grounding problem,
  not a transcript lookup. Use semantic-span review when aligned candidates
  exist, direct Omni audio review when they do not, or event/caption grounding
  when the target is best represented as an audible event; FlexSED remains
  optional rather than mandatory.
- Use both speech/speaker and event/caption modules when the target boundary
  depends on both sides, such as "the final speaker's question with reverb" or
  "the second speaker while wind/noise/laughter is present".
- Use no perception only when all required spans are already explicit and there
  is no unknown audio-grounded condition.

Return only a JSON object:
{
  "audio_captions": [
    {
      "audio_id": "audio1",
      "role": "source|reference|candidate_reference|unknown",
      "caption": "short audio-grounded summary",
      "speech_summary": "short speech/content/speaker summary if present",
      "uncertainties": []
    }
  ],
  "perception_plan": [
    {
      "audio_id": "audio1",
      "module": "perception.speech_context|perception.speaker_attributed_review|perception.semantic_span_review|perception.timestamped_caption|perception.flexsed_events|perception.audio_fallback|none",
      "priority": "must|optional",
      "purpose": "what evidence this module should return",
      "required_evidence": ["speaker-attributed transcript", "target span timestamps"],
      "reason": "why this module is needed"
    }
  ],
  "missing_information": [],
  "planner_assumptions": [],
  "next_planner_instruction": "one compact instruction for the final planner"
}
"""


PROMPT2_SPAN_RELEVANCE_ANALYSIS = """You are X-Editor's edit-span relevance reviewer.

Inputs:
- A user edit instruction.
- Candidate audio spans returned by perception modules.
- Optional attached source/reference audio for verification.

For each candidate span, decide whether it is needed by the downstream edit.
Be strict: distinguish target edit spans from context-only spans and reference
spans. If the instruction identifies a speaker by a quoted phrase, first locate
the speaker who said that phrase, then select all spans by that speaker when the
instruction asks for "his/her voice segments" or "all speech belonging to" that
speaker. If a role such as "police officer" appears only in narration, do not
select narrator spans as the role's direct speech unless the instruction asks
for narration about that role.
Only put spans that should be edited, extracted, removed, muted, beeped, or
replaced in selected_spans. Context/reference spans must remain in span_decisions
as preserve_context/reference_span/exclude and must not be used for timeline
reconstruction.

Decision labels:
- edit_span: the span should be edited, extracted, removed, muted, beeped, or otherwise changed.
- reference_span: the span should be used as reference voice/accent/style/audio evidence, not edited.
- preserve_context: the span should remain unchanged but gives context.
- exclude: the span is not needed.
- needs_more_perception: the candidate information is insufficient or contradictory.

Return only a JSON object:
{
  "span_decisions": [
    {
      "candidate_id": "span_000",
      "decision": "edit_span|reference_span|preserve_context|exclude|needs_more_perception",
      "operation_role": "target_edit|target_extract|target_remove|reference_voice|reference_accent|context|unknown",
      "operation_hint": "artifact_ops.extract_spans|artifact_ops.speaker_scoped_edit|edit_data_pipeline.text_edit|...",
      "confidence": "high|medium|low",
      "reason": "audio/text-grounded reason"
    }
  ],
  "selected_spans": [
    {
      "candidate_id": "span_000",
      "operation_role": "target_extract",
      "operation_hint": "artifact_ops.extract_spans",
      "reason": "why this span is selected"
    }
  ],
  "missing_evidence": [],
  "planner_notes": "compact guidance for final planning"
}
"""


PROMPT3_PLANNER_GAP_FILLING = """Planner gap-filling requirements for edit-span grounding:
- Treat planner_perception_guidance and span_selection_analysis as hard evidence
  when they are present in Context JSON. Do not ignore them and do not copy them
  wholesale into tool params.
- For local edits/extractions/removals, never assume target spans from a vague
  audio caption alone. If confident span-level evidence is missing, first plan
  perception.speaker_attributed_review or perception.audio_fallback.
- If a target span is identified by speech content, quoted words, a sentence,
  question/answer, speaker identity/order/role/gender/age, or speaker overlap,
  prefer perception.speaker_attributed_review before generic audio event/caption
  grounding. Audio captions can mention speech or speakers, but that evidence is
  too coarse for exact speech/speaker boundaries.
- Use perception.semantic_span_review for aligned speech candidates that need
  an Omni yes/no property rubric. Use perception.audio_fallback for direct
  semantic review without a candidate timeline.
- Use perception.timestamped_caption for general sound, environmental audio,
  music, acoustic effects, animal sounds, and non-speech events. FlexSED is an
  optional concrete-event route, not a prerequisite for caption grounding.
- When one target genuinely combines speech/speaker identity with an event,
  acoustic condition, or non-verbal vocalization targets, plan both perception paths
  unless one semantic review route can jointly ground both conditions.
- If audio is long, multi-speaker, overlapped, role-based, phrase-identified, or
  semantically/acoustically complex, prefer a perception step before artifact or
  edit-data-pipeline operations.
- If span_selection_analysis has no high/medium-confidence edit_span/reference_span
  that satisfies the instruction, or contains needs_more_perception for the target,
  plan the missing perception module before downstream editing.
- Use only span_selection_analysis.selected_spans for executable timeline edits.
  preserve_context, exclude, needs_more_perception, and reference_span decisions
  are evidence only; do not place them into artifact_ops selected_spans, spans,
  edits, or reconstruct replacement ranges.
- If a quoted phrase identifies a speaker and the request asks for that speaker's
  voice segments, do not edit/extract only the quoted phrase unless the request
  explicitly says "only the phrase"; select that speaker's relevant spans.
- Distinguish narration mentioning a role from direct speech by that role. When
  uncertain, call perception instead of treating narration text as the role span.
"""


class PlannerPerceptionGuidance:
    """Run planner-side prompt calls that decide and review perception evidence."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 8192,
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

    def plan_perception_calls(
        self,
        *,
        instruction: str,
        audio_paths: list[str],
        available_operations: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run PROMPT1 and return an auditable perception-call plan."""

        audio_items = _audio_items(audio_paths)
        user_payload = {
            "instruction": instruction,
            "audio_items": audio_items,
            "available_perception_operations": _perception_operation_ids(available_operations or []),
            "context": _compact_context(context or {}),
            "output_schema": "PROMPT1_AUDIO_PERCEPTION_CALL_PLANNING",
        }
        user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        result: dict[str, Any] = {
            "prompt_id": "PROMPT1_AUDIO_PERCEPTION_CALL_PLANNING",
            "prompt": PROMPT1_AUDIO_PERCEPTION_CALL_PLANNING,
            "prompt_payload": user_payload,
            "audio_paths": audio_paths,
        }
        try:
            response = self.client.call_json(
                PROMPT1_AUDIO_PERCEPTION_CALL_PLANNING,
                user_text,
                audio_paths=audio_paths,
            )
            result.update({"status": "success", "response": _normalize_prompt1_response(response, audio_items)})
        except Exception as exc:
            result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "response": _fallback_prompt1(audio_items)})
        return result

    def analyze_spans(
        self,
        *,
        instruction: str,
        perception_report: dict[str, Any],
        audio_paths: list[str] | None = None,
        context: dict[str, Any] | None = None,
        attach_audio: bool = True,
    ) -> dict[str, Any]:
        """Run PROMPT2 over perception-returned spans."""

        candidates = span_candidates_from_perception(perception_report)
        user_payload = {
            "instruction": instruction,
            "audio_span_candidates": candidates,
            "source_text": perception_report.get("source_text"),
            "context": _compact_context(context or {}),
            "output_schema": "PROMPT2_SPAN_RELEVANCE_ANALYSIS",
        }
        user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        attached = list(audio_paths or []) if attach_audio else []
        result: dict[str, Any] = {
            "prompt_id": "PROMPT2_SPAN_RELEVANCE_ANALYSIS",
            "prompt": PROMPT2_SPAN_RELEVANCE_ANALYSIS,
            "prompt_payload": user_payload,
            "audio_paths": attached,
            "candidate_count": len(candidates),
        }
        try:
            response = self.client.call_json(
                PROMPT2_SPAN_RELEVANCE_ANALYSIS,
                user_text,
                audio_paths=attached,
            )
            result.update({"status": "success", "response": _normalize_prompt2_response(response, candidates)})
        except Exception as exc:
            result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "response": _fallback_prompt2(candidates)})
        return result


def planner_perception_guidance_enabled(context: dict[str, Any]) -> bool:
    value = context.get("enable_planner_perception_guidance")
    if value is not None:
        return bool(value)
    return os.getenv("X_EDITOR_ENABLE_PLANNER_PERCEPTION_GUIDANCE", "0").lower() in {"1", "true", "yes", "on"}


def add_planner_perception_guidance(
    *,
    user_request: str,
    context: dict[str, Any],
    available_operations: list[dict[str, Any]] | None = None,
    guidance: PlannerPerceptionGuidance | None = None,
) -> dict[str, Any]:
    """Optionally add PROMPT1/PROMPT2 outputs to a planner context."""

    if not planner_perception_guidance_enabled(context):
        return context
    updated = dict(context)
    runner = guidance or PlannerPerceptionGuidance(timeout=float(os.getenv("X_EDITOR_PLANNER_PERCEPTION_GUIDANCE_TIMEOUT", "120")))
    audio_paths = _context_audio_paths(updated)
    if audio_paths and not updated.get("planner_perception_guidance"):
        updated["planner_perception_guidance"] = runner.plan_perception_calls(
            instruction=user_request,
            audio_paths=audio_paths,
            available_operations=available_operations or updated.get("available_operations") or [],
            context=updated,
        )
    perception = updated.get("perception_report")
    if isinstance(perception, dict) and perception and not updated.get("span_selection_analysis"):
        updated["span_selection_analysis"] = runner.analyze_spans(
            instruction=user_request,
            perception_report=perception,
            audio_paths=audio_paths[:1],
            context=updated,
        )
    updated.setdefault("perception_gap_fill_guidance", PROMPT3_PLANNER_GAP_FILLING)
    return updated


def span_candidates_from_perception(perception_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = perception_report.get("speaker_segments") or perception_report.get("segments") or []
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        start = _number(item.get("start_sec", item.get("start")))
        end = _number(item.get("end_sec", item.get("end")))
        if start is None or end is None:
            continue
        speaker_id = _first_present(item, "speaker_id", "speaker", "speaker_label")
        text = item.get("transcript") or item.get("text") or item.get("label") or ""
        candidates.append(
            {
                "candidate_id": f"span_{index:03d}",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(max(0.0, end - start), 6),
                "speaker_id": str(speaker_id) if speaker_id is not None else None,
                "text": str(text).strip(),
                "description": str(item.get("description") or item.get("label") or "").strip(),
                "source": str(item.get("source") or "").strip(),
            }
        )
    return candidates


def _context_audio_paths(context: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("input_audio", "input_audio_path"):
        if context.get(key):
            values.append(context[key])
    for key in ("input_audios", "input_audio_paths", "reference_audio_paths"):
        value = context.get(key)
        if isinstance(value, list):
            values.extend(value)
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        path = str(value)
        if path not in seen and Path(path).expanduser().exists():
            seen.add(path)
            output.append(path)
    return output


def _audio_items(audio_paths: list[str]) -> list[dict[str, Any]]:
    items = []
    for index, path in enumerate(audio_paths, start=1):
        items.append(
            {
                "audio_id": f"audio{index}",
                "path": str(Path(path).expanduser().resolve()),
                "default_role": "source" if index == 1 else "reference_or_extra",
            }
        )
    return items


def _perception_operation_ids(operations: list[dict[str, Any]]) -> list[str]:
    ids = []
    for item in operations:
        if not isinstance(item, dict):
            continue
        operation = item.get("operation")
        if isinstance(operation, str) and operation.startswith("perception."):
            ids.append(operation)
    return sorted(set(ids))


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "source_text",
        "reference_audio_paths",
        "missing_inputs",
        "previous_failure",
        "routing_policy_candidates",
        "perception_attempts",
    }
    return {key: context.get(key) for key in sorted(allowed) if key in context and context.get(key) not in (None, "", [], {})}


def _normalize_prompt1_response(response: dict[str, Any], audio_items: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = dict(response or {})
    captions = normalized.get("audio_captions")
    if not isinstance(captions, list):
        captions = []
    existing_ids = {str(item.get("audio_id")) for item in captions if isinstance(item, dict)}
    for item in audio_items:
        if item["audio_id"] not in existing_ids:
            captions.append({"audio_id": item["audio_id"], "role": item["default_role"], "caption": "", "speech_summary": "", "uncertainties": ["model did not return caption"]})
    normalized["audio_captions"] = captions
    if not isinstance(normalized.get("perception_plan"), list):
        normalized["perception_plan"] = []
    normalized.setdefault("missing_information", [])
    normalized.setdefault("planner_assumptions", [])
    normalized.setdefault("next_planner_instruction", "")
    return normalized


def _normalize_prompt2_response(response: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = dict(response or {})
    decisions = normalized.get("span_decisions")
    if not isinstance(decisions, list):
        decisions = []
    existing_ids = {str(item.get("candidate_id")) for item in decisions if isinstance(item, dict)}
    for candidate in candidates:
        if candidate["candidate_id"] not in existing_ids:
            decisions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "decision": "needs_more_perception",
                    "operation_role": "unknown",
                    "operation_hint": "perception.audio_fallback",
                    "confidence": "low",
                    "reason": "model did not review this candidate",
                }
            )
    normalized["span_decisions"] = decisions
    if not isinstance(normalized.get("selected_spans"), list):
        normalized["selected_spans"] = []
    normalized.setdefault("missing_evidence", [])
    normalized.setdefault("planner_notes", "")
    return normalized


def _fallback_prompt1(audio_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio_captions": [
            {
                "audio_id": item["audio_id"],
                "role": item["default_role"],
                "caption": "",
                "speech_summary": "",
                "uncertainties": ["PROMPT1 call failed"],
            }
            for item in audio_items
        ],
        "perception_plan": [
            {
                "audio_id": "audio1",
                "module": "perception.audio_fallback",
                "priority": "must",
                "purpose": "recover the evidence type needed before local edit planning",
                "required_evidence": ["instruction-relevant grounding evidence"],
                "reason": "fallback after planner perception triage failure",
            }
        ],
        "missing_information": ["PROMPT1 failed"],
        "planner_assumptions": [],
        "next_planner_instruction": "Call perception before selecting local edit spans.",
    }


def _fallback_prompt2(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "span_decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "decision": "needs_more_perception",
                "operation_role": "unknown",
                "operation_hint": "perception.audio_fallback",
                "confidence": "low",
                "reason": "PROMPT2 call failed",
            }
            for candidate in candidates
        ],
        "selected_spans": [],
        "missing_evidence": ["PROMPT2 failed"],
        "planner_notes": "Do not assume edit spans; call or repair perception first.",
    }


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return None
