"""Speech-candidate grounding with an optional Omni rubric review."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from x_editor.core.invocation_trace import (
    ComponentInvocationTrace,
    ProviderInvocationRecord,
)
from x_editor.core.perception_evidence_authority import (
    ObservationKind,
    lexical_comparison_tokens,
    reconcile_canonical_reviewed_report,
)
from x_editor.perception.audio_slicing import audio_duration_sec
from x_editor.perception.fallback import (
    PerceptionRequirements,
    build_audio_perception_fallback,
)
from x_editor.perception.observation_scope import observation_scope
from x_editor.perception.speech import SpeechPerceptionClient
from x_editor.perception.invocation_trace import (
    capture_frontend_provider_call,
    finalize_frontend_invocation_trace,
    reset_frontend_invocation_trace,
)


def run_semantic_span_review(
    *,
    audio_path: str | Path,
    audio_ref_id: str | None = None,
    reference_audio_paths: list[str] | None = None,
    reference_audio_refs: list[str] | None = None,
    instruction: str,
    semantic_targets: list[dict[str, str]] | None = None,
    output_dir: str | Path,
    action: str,
    need_speakers: bool,
    reviewer: Any | None,
    agentic_mode: str | None,
    speech_client: SpeechPerceptionClient | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Ground spans from TS-ASR candidates, escalating only ambiguous cases."""

    audio = Path(audio_path).expanduser().resolve()
    reference_paths = [
        str(Path(item).expanduser().resolve()) for item in (reference_audio_paths or [])
    ]
    logical_audio_ref = str(audio_ref_id or audio)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target_contracts = _semantic_target_contracts(semantic_targets)
    if reviewer is not None:
        reset_frontend_invocation_trace(reviewer)

    specialist_result = build_audio_perception_fallback(
        agentic_mode=agentic_mode,
        speech_client=speech_client,
    ).perceive(
        user_prompt=instruction,
        audio_path=audio,
        output_dir=output_root / "specialist_evidence",
        requirements=PerceptionRequirements(
            source_text=True,
            word_timestamps=True,
            speaker_segments=need_speakers,
            speaker_profiles=False,
            segment_descriptions=False,
        ),
        run_specialized=True,
        allow_gemini=False,
        force_gemini=False,
        refresh_cache=True,
        strict=False,
    )
    perception = specialist_result.perception
    duration_sec = audio_duration_sec(audio)
    deterministic_assessment = _typed_target_grounding_assessment(
        target_contracts=target_contracts,
        word_timestamps=perception.get("word_timestamps"),
    )
    escalation_reasons = list(deterministic_assessment["escalation_reasons"])
    if reference_paths:
        escalation_reasons.append("reference_context_requires_review")
    deterministic_targets_complete = bool(
        deterministic_assessment["complete"] and not reference_paths
    )

    # Imported lazily to keep the perception package independent from runtime
    # initialization while the shared span selector is being extracted.
    from x_editor.runtime.artifact_ops import (
        select_timeline_spans_with_perception_fallback,
    )

    selection_error: Exception | None = None
    if deterministic_targets_complete:
        spans = []
        selection = {
            "status": "deterministic_complete",
            "reason": "all typed targets have one exact, traceable aligned occurrence",
            "model_call_attempted": False,
        }
    else:
        try:
            spans, selection = select_timeline_spans_with_perception_fallback(
                instruction,
                perception,
                input_audio_path=str(audio),
                output_dir=str(output_root),
                duration_sec=duration_sec,
                action=action,
                agentic_mode=agentic_mode,
                selector_client=getattr(reviewer, "client", None),
                input_audio_ref=logical_audio_ref,
                reference_audio_paths=reference_paths,
                reference_audio_refs=reference_audio_refs,
                semantic_targets=target_contracts,
                force_model_review_on_empty=True,
                deterministic_targets_complete=False,
                aligned_candidate_segments=deterministic_assessment[
                    "review_candidates"
                ],
                include_default_candidates=bool(
                    deterministic_assessment["requires_default_candidates"]
                ),
                authoritative_candidate_bounds=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve structured perception failure
            spans = []
            selection = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "model_call_attempted": False,
            }
            selection_error = exc

    model_reviewed = bool(
        isinstance(selection, dict)
        and (
            selection.get("model_call_attempted") is True
            or (
                "model_call_attempted" not in selection
                and selection.get("provider")
                and selection.get("status") in {"completed", "failed"}
            )
        )
    )
    grounding_route = (
        "ts_asr_plus_omni_rubric"
        if model_reviewed
        else (
            "ts_asr_deterministic"
            if deterministic_targets_complete
            else "ts_asr_deterministic_incomplete"
        )
    )
    candidate_segments = _reviewed_candidate_segments(selection)
    candidate_judgements = _bind_declared_semantic_claims(
        _reviewed_candidate_judgements(selection),
        target_contracts,
        candidate_segments=candidate_segments,
    )
    model_selected_spans = _reviewed_selected_spans(
        spans,
        selection=selection,
        audio_ref_id=logical_audio_ref,
        grounding_route=grounding_route,
        word_timestamps=perception.get("word_timestamps"),
        target_contracts=target_contracts,
    )
    selected_spans, exact_grounding = _apply_quoted_text_grounding(
        model_selected_spans=model_selected_spans,
        target_contracts=target_contracts,
        word_timestamps=perception.get("word_timestamps"),
        candidate_judgements=candidate_judgements,
        audio_ref_id=logical_audio_ref,
    )
    target_bindings = _reviewed_target_bindings(
        instruction=instruction,
        selected_spans=selected_spans,
        candidate_judgements=candidate_judgements,
        target_contracts=target_contracts,
        exact_grounding=exact_grounding,
        confirmed_absences=_confirmed_absent_targets(
            selection=selection,
            target_contracts=target_contracts,
            selected_spans=selected_spans,
            candidate_segments=candidate_segments,
            candidate_judgements=candidate_judgements,
            exact_grounding=exact_grounding,
            duration_sec=duration_sec,
        ),
    )
    perception = dict(perception)
    perception["audio_ref_id"] = logical_audio_ref
    perception["audio_index"] = 0
    perception["selected_spans"] = selected_spans
    if candidate_segments:
        perception["candidate_segments"] = candidate_segments
    if candidate_judgements:
        perception["candidate_rubric_judgements"] = candidate_judgements
    perception["target_bindings"] = target_bindings
    scope = observation_scope(candidate_segments, duration_sec)
    if model_reviewed and not scope["covers_entire_asset"]:
        perception.setdefault("uncertainties", []).append({
            "audio_ref_id": logical_audio_ref,
            "code": "observation_scope_incomplete",
            "message": "Candidate listening did not cover the whole asset. "
                       "No claim of global absence follows. Actual coverage: "
                       + json.dumps(scope, sort_keys=True),
            "severity": "advisory", "blocking": False,
            "missing_evidence": ["observation of unobserved intervals if required by target"],
        })
    perception.setdefault("source_text_source", "specialist_speech_evidence")
    perception["evidence_sources"] = [
        {
            "evidence_id": _semantic_span_evidence_source_id(
                audio_ref_id=logical_audio_ref,
                action=action,
                target_contracts=target_contracts,
            ),
            "operation": "perception.semantic_span_review",
            "route_id": grounding_route,
            "source_kind": "reviewed_perception",
            "provider": selection.get("provider") if isinstance(selection, dict) else None,
            "evidence_refs": [
                item["candidate_id"] for item in candidate_segments if item.get("candidate_id")
            ],
        }
    ]
    perception["semantic_span_review"] = {
        "action": action,
        "selected_spans": selected_spans,
        "candidate_segments": candidate_segments,
        "candidate_rubric_judgements": candidate_judgements,
        "grounding_route": grounding_route,
        "deterministic_target_assessment": deterministic_assessment["targets"],
        "escalation_reason": escalation_reasons,
    }
    perception, evidence_conflicts = reconcile_canonical_reviewed_report(
        perception,
        authority_evidence_ids={
            ObservationKind.LEXICAL_CONTENT: ["semantic_span_review:specialist"],
            ObservationKind.WORD_TIMING: ["semantic_span_review:specialist"],
            ObservationKind.SEMANTIC_JUDGMENT: ["semantic_span_review:rubric"],
        },
    )
    selected_spans = list(perception.get("selected_spans") or [])
    candidate_judgements = list(perception.get("candidate_rubric_judgements") or [])
    target_bindings = list(perception.get("target_bindings") or [])
    source_text_resolutions = _corroborated_source_text_resolutions(
        perception=perception,
        target_contracts=target_contracts,
        exact_grounding=exact_grounding,
    )
    if source_text_resolutions:
        perception["source_text_resolutions"] = source_text_resolutions
    perception["semantic_span_review"]["selected_spans"] = selected_spans
    perception["semantic_span_review"]["candidate_rubric_judgements"] = candidate_judgements
    missing = _missing_target_grounding(
        target_bindings=target_bindings,
        selected_spans=selected_spans,
    )
    candidate_evidence_available = bool(candidate_segments or selected_spans)
    if missing:
        uncertainties = list(perception.get("uncertainties") or [])
        uncertainties.extend(
            {
                "code": "semantic_target_selection_unresolved",
                "message": item,
                "severity": "blocking" if not candidate_evidence_available else "advisory",
                "blocking": not candidate_evidence_available,
                "missing_evidence": ["semantic candidate selection"],
            }
            for item in missing
        )
        perception["uncertainties"] = uncertainties
    provider_review_failed = bool(
        selection_error is not None
        or (
            model_reviewed
            and isinstance(selection, dict)
            and selection.get("status") != "completed"
        )
    )
    component_status = (
        "failed"
        if provider_review_failed or (missing and not candidate_evidence_available)
        else "completed"
    )
    trace: ComponentInvocationTrace | None = None
    if reviewer is not None and model_reviewed:
        capture_frontend_provider_call(
            reviewer,
            "semantic_span_review.candidate_rubric",
            succeeded=selection_error is None and selection.get("status") == "completed",
            error=selection_error,
        )
        trace = finalize_frontend_invocation_trace(
            reviewer,
            status=component_status,
        )
    if trace is None:
        provider_invocations = []
        if model_reviewed:
            provider_status = (
                "completed" if selection.get("status") == "completed" else "failed"
            )
            provider_invocations = [
                ProviderInvocationRecord(
                    invocation_index=1,
                    invocation_id="semantic_span_review.candidate_rubric:1",
                    provider=str(selection.get("provider") or "unknown_audio_reviewer"),
                    model=(
                        str(selection["model"])
                        if selection.get("model") is not None
                        else None
                    ),
                    kind="primary",
                    purpose="semantic_span_review.candidate_rubric",
                    status=provider_status,
                    transport_attempt_count=1,
                    metrics={"accounting_source": "selection_evidence"},
                    error_type=(
                        "CandidateRubricError" if provider_status == "failed" else None
                    ),
                    error_message=(
                        str(selection.get("reason") or "candidate rubric failed")
                        if provider_status == "failed"
                        else None
                    ),
                )
            ]
        trace = ComponentInvocationTrace.build(
            component="perception",
            component_name="perception.semantic_span_review",
            route="provider" if model_reviewed else "deterministic",
            status=component_status,
            provider_invocations=provider_invocations,
            metadata={
                "operation": "perception.semantic_span_review",
                "specialist_attempt_count": len(specialist_result.attempts),
                "model_rubric_called": model_reviewed,
            },
        )
    trace = trace.model_copy(
        update={
            "metadata": {
                **trace.metadata,
                "specialist_attempt_count": len(specialist_result.attempts),
                "model_rubric_called": model_reviewed,
                "actual_route_id": grounding_route,
                "escalation_reason": escalation_reasons,
            }
        },
        deep=True,
    )
    if reviewer is not None:
        reviewer.last_invocation_trace = trace
    return {
        "operation": "perception.semantic_span_review",
        "input_audio_path": str(audio),
        "perception_report": perception,
        "source_text": perception.get("source_text"),
        "selected_spans": selected_spans,
        "selection_evidence": selection,
        "evidence_conflicts": evidence_conflicts,
        "satisfied": not missing,
        "required_missing": (
            missing
            if strict and (provider_review_failed or not candidate_evidence_available)
            else []
        ),
        "missing": missing,
        "attempts": specialist_result.attempts,
        "grounding_route": perception["semantic_span_review"]["grounding_route"],
        "perception_route_id": perception["semantic_span_review"]["grounding_route"],
        "component_invocation_trace": trace.model_dump(mode="json"),
    }


def _corroborated_source_text_resolutions(
    *,
    perception: dict[str, Any],
    target_contracts: list[dict[str, str]],
    exact_grounding: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Publish a complete transcript candidate without replacing canonical ASR.

    This is evidence fusion, not a text normalizer. A candidate is emitted only
    when reviewed audio explicitly corroborated one ASR omission and every text
    patch is unique and non-overlapping in the canonical transcript.
    """

    canonical_source_text = str(perception.get("source_text") or "").strip()
    if not canonical_source_text:
        return []
    targets_by_id = {item["target_id"]: item for item in target_contracts}
    spans_by_target: dict[str, list[dict[str, Any]]] = {}
    for item in perception.get("selected_spans") or []:
        if isinstance(item, dict) and item.get("target_id"):
            spans_by_target.setdefault(str(item["target_id"]), []).append(item)
    judgements_by_target: dict[str, list[dict[str, Any]]] = {}
    for item in perception.get("candidate_rubric_judgements") or []:
        if not isinstance(item, dict) or not item.get("claim_target_id"):
            continue
        if item.get("status") != "admitted_with_asr_disagreement":
            continue
        judgements_by_target.setdefault(str(item["claim_target_id"]), []).append(item)

    patches: list[dict[str, Any]] = []
    for target_id, grounding in exact_grounding.items():
        if grounding.get("corroboration_status") != "admitted_with_single_asr_omission":
            continue
        target = targets_by_id.get(target_id)
        spans = spans_by_target.get(target_id, [])
        judgements = judgements_by_target.get(target_id, [])
        if target is None or len(spans) != 1 or len(judgements) != 1:
            return []
        span = spans[0]
        judgement = judgements[0]
        current_phrase = str(span.get("text") or "").strip()
        resolved_phrase = str(target.get("value") or "").strip()
        patch = _unique_lexical_source_patch(
            canonical_source_text,
            current_phrase=current_phrase,
            resolved_phrase=resolved_phrase,
        )
        if patch is None:
            return []
        patch.update(
            {
                "target_id": target_id,
                "span_id": str(span.get("span_id") or ""),
                "source_refs": list(
                    dict.fromkeys(
                        [
                            *grounding.get("source_refs", []),
                            *span.get("source_refs", []),
                            *judgement.get("source_refs", []),
                        ]
                    )
                ),
                "confidence": float(judgement["confidence"]),
            }
        )
        patches.append(patch)
    if not patches:
        return []
    patches.sort(key=lambda item: (item["start_char"], item["end_char"]))
    if any(
        right["start_char"] < left["end_char"]
        for left, right in pairwise(patches)
    ):
        return []

    resolved = canonical_source_text
    for patch in reversed(patches):
        resolved = (
            resolved[: patch["start_char"]]
            + patch["replacement"]
            + resolved[patch["end_char"] :]
        )
    return [
        {
            "resolution_id": "source_text_resolution_000",
            "audio_ref_id": perception.get("audio_ref_id"),
            "audio_index": perception.get("audio_index"),
            "status": "corroborated",
            "canonical_source_text": canonical_source_text,
            "resolved_source_text": resolved,
            "target_ids": [str(item["target_id"]) for item in patches],
            "span_ids": [str(item["span_id"]) for item in patches if item["span_id"]],
            "source_refs": list(
                dict.fromkeys(ref for item in patches for ref in item["source_refs"])
            ),
            "confidence": min(float(item["confidence"]) for item in patches),
            "reason": (
                "canonical ASR transcript completed only at uniquely aligned omissions "
                "confirmed by explicit high-confidence reviewed-audio exact-phrase claims"
            ),
            "source": "canonical_asr+reviewed_audio_exact_phrase",
            "source_kind": "multi_source_lexical_resolution",
        }
    ]


def _unique_lexical_source_patch(
    source_text: str,
    *,
    current_phrase: str,
    resolved_phrase: str,
) -> dict[str, Any] | None:
    source_tokens = _lexical_token_spans(source_text)
    current_tokens = lexical_comparison_tokens(current_phrase)
    resolved_tokens = lexical_comparison_tokens(resolved_phrase)
    if not source_tokens or not current_tokens or len(resolved_tokens) != len(current_tokens) + 1:
        return None
    if not any(
        resolved_tokens[:index] + resolved_tokens[index + 1 :] == current_tokens
        for index in range(len(resolved_tokens))
    ):
        return None
    normalized_source = tuple(str(item["token"]) for item in source_tokens)
    matches = [
        index
        for index in range(len(normalized_source) - len(current_tokens) + 1)
        if normalized_source[index : index + len(current_tokens)] == current_tokens
    ]
    if len(matches) != 1:
        return None
    start_index = matches[0]
    end_index = start_index + len(current_tokens) - 1
    return {
        "start_char": int(source_tokens[start_index]["start_char"]),
        "end_char": int(source_tokens[end_index]["end_char"]),
        "replacement": resolved_phrase,
    }


def _lexical_token_spans(text: str) -> list[dict[str, Any]]:
    """Return comparison tokens with exact character bounds in admitted text."""

    entries: list[dict[str, Any]] = []
    start: int | None = None
    current: list[str] = []

    def flush(end: int) -> None:
        nonlocal start
        if start is not None and current:
            entries.append(
                {
                    "token": "".join(current).casefold(),
                    "start_char": start,
                    "end_char": end,
                }
            )
        current.clear()
        start = None

    for index, character in enumerate(text):
        codepoint = ord(character)
        is_cjk = (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
        if is_cjk:
            flush(index)
            entries.append(
                {
                    "token": character,
                    "start_char": index,
                    "end_char": index + 1,
                }
            )
            continue
        if character.isalnum():
            if start is None:
                start = index
            current.append(character)
            continue
        if character in {"-", "'", "\u2019"}:
            next_is_alnum = index + 1 < len(text) and text[index + 1].isalnum()
            if current and next_is_alnum:
                current.append(character)
                continue
        flush(index)
    flush(len(text))
    return entries


def _semantic_target_contracts(
    semantic_targets: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    contracts: list[dict[str, str]] = []
    for index, item in enumerate(semantic_targets or []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        value = str(item.get("value") or "")
        if not kind or not value:
            continue
        contracts.append(
            {
                "target_id": str(item.get("target_id") or f"semantic_target_{index:03d}"),
                "kind": kind,
                "value": value,
            }
        )
    return contracts


def _typed_target_grounding_assessment(
    *,
    target_contracts: list[dict[str, str]],
    word_timestamps: Any,
) -> dict[str, Any]:
    """Assess typed completeness and expose one aligned candidate per ambiguity."""

    word_entries = _word_token_entries(word_timestamps)
    targets: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    escalation_reasons: list[str] = []
    requires_default_candidates = False
    if not target_contracts:
        return {
            "complete": False,
            "targets": [],
            "review_candidates": [],
            "requires_default_candidates": True,
            "escalation_reasons": ["typed_targets_unavailable"],
        }

    for target_index, target in enumerate(target_contracts):
        target_id = target["target_id"]
        if target.get("kind") != "quoted_text":
            targets.append(
                {
                    "target_id": target_id,
                    "status": "requires_model_review",
                    "match_count": None,
                    "candidate_ids": [],
                }
            )
            requires_default_candidates = True
            escalation_reasons.append("non_lexical_typed_target")
            continue

        target_tokens = lexical_comparison_tokens(target.get("value"))
        matches = _contiguous_word_matches(word_entries, target_tokens)
        if len(matches) == 1:
            status = "resolved_unique_exact"
        elif len(matches) > 1:
            status = "ambiguous_exact_occurrences"
            escalation_reasons.append("ambiguous_typed_target")
        else:
            status = "missing_exact_occurrence"
            requires_default_candidates = True
            escalation_reasons.append("missing_exact_typed_target")

        candidate_ids: list[str] = []
        if len(matches) > 1:
            for occurrence_index, match in enumerate(matches):
                candidate_id = (
                    f"typed_target_{target_index:03d}_occurrence_{occurrence_index:03d}"
                )
                candidate_ids.append(candidate_id)
                review_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "target_id": target_id,
                        "target_kind": target["kind"],
                        "target_value": target["value"],
                        "start_sec": match["start_sec"],
                        "end_sec": match["end_sec"],
                        "speaker_id": match.get("speaker_id"),
                        "text": match["text"],
                        "description": (
                            "one exact aligned occurrence of the declared quoted target"
                        ),
                        "source": "canonical_word_timestamps",
                        "source_refs": list(match["source_refs"]),
                        "authoritative_bounds": True,
                    }
                )
        targets.append(
            {
                "target_id": target_id,
                "status": status,
                "match_count": len(matches),
                "candidate_ids": candidate_ids,
            }
        )

    return {
        "complete": all(
            item["status"] == "resolved_unique_exact" for item in targets
        ),
        "targets": targets,
        "review_candidates": review_candidates,
        "requires_default_candidates": requires_default_candidates,
        "escalation_reasons": list(dict.fromkeys(escalation_reasons)),
    }


def _bind_declared_semantic_claims(
    judgements: list[dict[str, Any]],
    target_contracts: list[dict[str, str]],
    *,
    candidate_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Bind judgements only when their target ownership is unambiguous."""

    targets_by_id = {item["target_id"]: item for item in target_contracts}
    candidate_target_ids = {
        str(item["candidate_id"]): str(item["target_id"])
        for item in candidate_segments or []
        if item.get("candidate_id") and item.get("target_id")
    }
    if not targets_by_id:
        return judgements
    only_target = target_contracts[0] if len(target_contracts) == 1 else None
    bound: list[dict[str, Any]] = []
    for raw in judgements:
        item = dict(raw)
        declared_target_id = str(item.get("claim_target_id") or "")
        candidate_target_id = candidate_target_ids.get(str(item.get("candidate_id") or ""))
        if candidate_target_id and declared_target_id not in {"", candidate_target_id}:
            item["reported_claim_target_id"] = declared_target_id
        authoritative_target_id = candidate_target_id or declared_target_id
        target = (
            targets_by_id.get(authoritative_target_id)
            if authoritative_target_id
            else only_target
        )
        if target is None:
            bound.append(item)
            continue
        item["claim_target_id"] = target["target_id"]
        if not item.get("matches") or target.get("kind") != "quoted_text":
            bound.append(item)
            continue
        if item.get("claim_kind") not in (None, "", "exact_phrase"):
            item["reported_claim_kind"] = item["claim_kind"]
        if item.get("claimed_text") not in (None, "", target["value"]):
            item["reported_claimed_text"] = item["claimed_text"]
        item["claim_kind"] = "exact_phrase"
        item["claim_target_id"] = target["target_id"]
        item["claimed_text"] = target["value"]
        bound.append(item)
    return bound


def _semantic_span_evidence_source_id(
    *,
    audio_ref_id: str,
    action: str,
    target_contracts: list[dict[str, Any]],
) -> str:
    identity = {
        "audio_ref_id": audio_ref_id,
        "action": action,
        "target_contracts": target_contracts,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"semantic_span_review:{digest}"


def _reviewed_candidate_segments(selection: Any) -> list[dict[str, Any]]:
    if not isinstance(selection, dict):
        return []
    raw_candidates = selection.get("candidate_segments")
    if not isinstance(raw_candidates, list):
        return []
    return [
        dict(item) for item in raw_candidates if isinstance(item, dict) and item.get("candidate_id")
    ]


def _reviewed_candidate_judgements(selection: Any) -> list[dict[str, Any]]:
    if not isinstance(selection, dict):
        return []
    raw_judgements = selection.get("judgements") or selection.get("judgments")
    if raw_judgements is None:
        raw_judgements = []
    elif not isinstance(raw_judgements, list):
        return []
    reviewed: list[dict[str, Any]] = []
    for item in raw_judgements:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id") or item.get("segment_id")
        if not candidate_id:
            continue
        matches = bool(item.get("matches"))
        explicit_exact_claim = (
            item.get("claim_kind") == "exact_phrase"
            and isinstance(item.get("claim_target_id"), str)
            and bool(item.get("claim_target_id"))
            and isinstance(item.get("claimed_text"), str)
            and bool(item.get("claimed_text"))
        )
        reviewed_item = {
            "candidate_id": str(candidate_id),
            "answer": "yes" if matches else "no",
            "matches": matches,
            "match_declared": item.get("match_declared") is True,
            "claim_explicit": explicit_exact_claim,
            "confidence": item.get("confidence"),
            "reason": str(item.get("reason") or ""),
            "detected_properties": list(item.get("detected_properties") or []),
            "evidence_slice_ids": list(item.get("evidence_slice_ids") or []),
            "source_refs": [str(candidate_id)],
        }
        for field in (
            "claim_kind",
            "claim_target_id",
            "claimed_text",
            "claim_binding_source",
        ):
            value = item.get(field)
            if isinstance(value, str) and value:
                reviewed_item[field] = value
        try:
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if 0.0 <= start_sec < end_sec:
                reviewed_item["start_sec"] = start_sec
                reviewed_item["end_sec"] = end_sec
        reviewed.append(reviewed_item)
    reviewed_ids = {str(item["candidate_id"]) for item in reviewed if item.get("candidate_id")}
    for candidate_id in selection.get("selected_segment_ids") or []:
        candidate_id = str(candidate_id)
        if candidate_id in reviewed_ids:
            continue
        reviewed.append(
            {
                "candidate_id": candidate_id,
                "answer": "selection_implied",
                "matches": None,
                "status": "selection_implied",
                "confidence": None,
                "reason": (
                    "candidate appears in selected_segment_ids without an independent judgement"
                ),
                "detected_properties": [],
                "evidence_slice_ids": [],
                "source_refs": [candidate_id],
            }
        )
    return reviewed


def _reviewed_selected_spans(
    spans: list[Any],
    *,
    selection: Any,
    audio_ref_id: str,
    grounding_route: str,
    word_timestamps: Any = None,
    target_contracts: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    candidates = _reviewed_candidate_segments(selection)
    candidate_by_id = {
        str(item["candidate_id"]): item for item in candidates if item.get("candidate_id")
    }
    judgements = _reviewed_candidate_judgements(selection)
    judgement_by_id = {
        str(item["candidate_id"]): item for item in judgements if item.get("candidate_id")
    }
    selected_ids = {
        str(item)
        for item in (selection.get("selected_segment_ids") if isinstance(selection, dict) else [])
        or []
    }
    reviewed: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        raw = span.to_dict()
        raw = _align_span_to_word_evidence(raw, word_timestamps)
        source_refs = [
            str(candidate["candidate_id"])
            for candidate in candidates
            if candidate.get("candidate_id")
            and str(candidate["candidate_id"]) in selected_ids
            and _spans_overlap(raw, candidate)
        ]
        confidence_values = [
            judgement_by_id[source_ref].get("confidence")
            for source_ref in source_refs
            if source_ref in judgement_by_id
            and judgement_by_id[source_ref].get("confidence") is not None
        ]
        target_id = _selected_span_target_id(
            source_refs=source_refs,
            judgement_by_id=judgement_by_id,
            candidate_by_id=candidate_by_id,
            target_contracts=target_contracts or [],
        )
        if target_contracts and target_id is None:
            # Keep this interval in candidate evidence until its typed owner is known.
            continue
        reviewed.append(
            {
                "span_id": f"span_{index:03d}",
                "audio_ref_id": audio_ref_id,
                "start_sec": raw["start_sec"],
                "end_sec": raw["end_sec"],
                "speaker_id": raw.get("speaker_id"),
                "text": raw.get("text") or "",
                "label": raw.get("label") or "",
                "semantic_role": "edit_target",
                "target_id": target_id,
                "source_refs": source_refs or [raw.get("source") or grounding_route],
                "confidence": confidence_values[0] if confidence_values else None,
                "reason": "grounded by reviewed aligned evidence",
                "source": raw.get("source") or grounding_route,
            }
        )
    return reviewed


def _selected_span_target_id(
    *,
    source_refs: list[str],
    judgement_by_id: dict[str, dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
    target_contracts: list[dict[str, str]],
) -> str | None:
    if not target_contracts:
        return "target_000"
    declared_ids = {item["target_id"] for item in target_contracts}
    candidate_bound_ids = {
        str(candidate_by_id[source_ref].get("target_id") or "")
        for source_ref in source_refs
        if source_ref in candidate_by_id
    }
    candidate_bound_ids.discard("")
    admitted_candidate_ids = sorted(candidate_bound_ids & declared_ids)
    if len(admitted_candidate_ids) == 1:
        return admitted_candidate_ids[0]
    if admitted_candidate_ids:
        return None
    bound_ids = {
        str(judgement_by_id[source_ref].get("claim_target_id") or "")
        for source_ref in source_refs
        if source_ref in judgement_by_id
    }
    bound_ids.discard("")
    admitted_ids = sorted(bound_ids & declared_ids)
    if len(admitted_ids) == 1:
        return admitted_ids[0]
    if len(target_contracts) == 1:
        return target_contracts[0]["target_id"]
    return None


def _apply_quoted_text_grounding(
    *,
    model_selected_spans: list[dict[str, Any]],
    target_contracts: list[dict[str, str]],
    word_timestamps: Any,
    candidate_judgements: list[dict[str, Any]],
    audio_ref_id: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve quoted text from exact alignment or tightly corroborated audio evidence.

    Exact canonical word alignment remains authoritative. A reviewed-audio claim may
    bridge one ASR omission only under the explicit, high-confidence, unique-match
    contract below. Internal hyphens and apostrophes are never normalized away.
    """

    exact_targets = [item for item in target_contracts if item.get("kind") == "quoted_text"]
    if not exact_targets:
        return model_selected_spans, {}

    exact_target_ids = {item["target_id"] for item in exact_targets}
    exact_candidate_ids = {
        str(item.get("candidate_id") or "")
        for item in candidate_judgements
        if item.get("claim_kind") == "exact_phrase"
        and item.get("claim_target_id") in exact_target_ids
    }
    retained_model_spans = [
        item
        for item in model_selected_spans
        if item.get("target_id") not in exact_target_ids
        and not exact_candidate_ids.intersection(str(ref) for ref in item.get("source_refs") or [])
    ]
    word_entries = _word_token_entries(word_timestamps)
    exact_spans: list[dict[str, Any]] = []
    resolutions: dict[str, dict[str, Any]] = {}
    for target in exact_targets:
        target_id = target["target_id"]
        target_model_spans = [
            span for span in model_selected_spans if span.get("target_id") == target_id
        ]
        target_tokens = lexical_comparison_tokens(target["value"])
        matches = _contiguous_word_matches(word_entries, target_tokens)
        resolution_reason: str
        selected_match: dict[str, Any] | None = None
        corroborating_judgement: dict[str, Any] | None = None
        disambiguation_refs: list[str] = []
        if len(matches) == 1:
            selected_match = matches[0]
            resolution_reason = "unique exact contiguous match in canonical word timestamps"
        elif len(matches) > 1:
            overlapping = [
                match
                for match in matches
                if any(_spans_overlap(match, span) for span in target_model_spans)
            ]
            if len(overlapping) == 1:
                selected_match = overlapping[0]
                disambiguation_refs = list(
                    dict.fromkeys(
                        str(ref)
                        for span in target_model_spans
                        if _spans_overlap(selected_match, span)
                        for ref in span.get("source_refs") or []
                    )
                )
                resolution_reason = (
                    "repeated exact wording disambiguated by the reviewed model-selected interval"
                )
            else:
                resolution_reason = (
                    "multiple exact contiguous matches remain ambiguous after reviewed "
                    "span selection"
                )
        elif not target_tokens:
            resolution_reason = "quoted_text target has no comparable lexical tokens"
        elif not word_entries:
            resolution_reason = "canonical word timestamps are unavailable"
        else:
            corroborated = _corroborated_single_omission_match(
                target_id=target_id,
                target_tokens=target_tokens,
                word_entries=word_entries,
                target_model_spans=target_model_spans,
                candidate_judgements=candidate_judgements,
            )
            if corroborated is None:
                resolution_reason = "no exact contiguous match in canonical word timestamps"
            else:
                selected_match, corroborating_judgement = corroborated
                candidate_id = str(corroborating_judgement["candidate_id"])
                disambiguation_refs = [candidate_id]
                resolution_reason = (
                    "unique single-token ASR omission corroborated by an explicit "
                    "high-confidence reviewed-audio exact-phrase claim"
                )
                corroborating_judgement.update(
                    {
                        "corroboration_status": "admitted_with_single_asr_omission",
                        "corroboration_method": (
                            "explicit_audio_exact_phrase+unique_single_asr_omission"
                        ),
                        "asr_lexical_edit_distance": 1,
                        "asr_omitted_lexeme": selected_match[
                            "omitted_target_token"
                        ],
                    }
                )
                corroborating_judgement["source_refs"] = list(
                    dict.fromkeys(
                        [
                            *corroborating_judgement.get("source_refs", []),
                            *selected_match["source_refs"],
                        ]
                    )
                )

        if selected_match is None:
            resolutions[target_id] = {
                "status": "unresolved",
                "reason": resolution_reason,
                "match_count": len(matches),
            }
            continue

        source_refs = list(dict.fromkeys([*selected_match["source_refs"], *disambiguation_refs]))
        exact_spans.append(
            {
                "span_id": "",
                "audio_ref_id": audio_ref_id,
                "start_sec": selected_match["start_sec"],
                "end_sec": selected_match["end_sec"],
                "speaker_id": selected_match.get("speaker_id"),
                "text": (
                    target["value"]
                    if corroborating_judgement is not None
                    else selected_match["text"]
                ),
                "label": target["value"],
                "semantic_role": "edit_target",
                "target_id": target_id,
                "source_refs": source_refs,
                "confidence": (
                    corroborating_judgement.get("confidence")
                    if corroborating_judgement is not None
                    else None
                ),
                "reason": resolution_reason,
                "source": "canonical_word_timestamps+planner_quoted_text",
            }
        )
        resolutions[target_id] = {
            "status": "resolved",
            "reason": resolution_reason,
            "match_count": len(matches),
            "source_refs": source_refs,
        }
        if corroborating_judgement is not None:
            resolutions[target_id].update(
                {
                    "corroboration_status": "admitted_with_single_asr_omission",
                    "asr_lexical_edit_distance": 1,
                    "asr_omitted_lexeme": selected_match[
                        "omitted_target_token"
                    ],
                }
            )

    selected_spans = [*retained_model_spans, *exact_spans]
    for index, span in enumerate(selected_spans):
        span["span_id"] = f"span_{index:03d}"
    return selected_spans, resolutions


_AUDIO_EXACT_PHRASE_MIN_CONFIDENCE = 0.8
_AUDIO_EXACT_PHRASE_MAX_TOTAL_PADDING_SEC = 1.5


def _corroborated_single_omission_match(
    *,
    target_id: str,
    target_tokens: tuple[str, ...],
    word_entries: list[dict[str, Any]],
    target_model_spans: list[dict[str, Any]],
    candidate_judgements: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Admit one ASR omission only when independent audio evidence is explicit."""

    if len(target_tokens) < 4:
        return None
    eligible_judgements: list[dict[str, Any]] = []
    for judgement in candidate_judgements:
        if not (
            judgement.get("matches") is True
            and judgement.get("claim_explicit") is True
            and judgement.get("claim_kind") == "exact_phrase"
            and judgement.get("claim_target_id") == target_id
            and not judgement.get("reported_claim_kind")
            and not judgement.get("reported_claimed_text")
            and lexical_comparison_tokens(judgement.get("claimed_text")) == target_tokens
        ):
            continue
        try:
            confidence = float(judgement["confidence"])
            start_sec = float(judgement["start_sec"])
            end_sec = float(judgement["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if confidence < _AUDIO_EXACT_PHRASE_MIN_CONFIDENCE or not 0.0 <= start_sec < end_sec:
            continue
        candidate_id = str(judgement.get("candidate_id") or "")
        if not any(
            candidate_id in {str(ref) for ref in span.get("source_refs") or []}
            for span in target_model_spans
        ):
            continue
        eligible_judgements.append(judgement)
    if not eligible_judgements:
        return None

    matches = _single_target_token_omission_matches(word_entries, target_tokens)
    supported: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for match in matches:
        for judgement in eligible_judgements:
            if not _review_interval_tightly_contains_match(judgement, match):
                continue
            key = (int(match["first_word_index"]), int(match["last_word_index"]))
            supported[key] = (match, judgement)
    if len(supported) != 1:
        return None
    return next(iter(supported.values()))


def _single_target_token_omission_matches(
    word_entries: list[dict[str, Any]],
    target_tokens: tuple[str, ...],
) -> list[dict[str, Any]]:
    width = len(target_tokens) - 1
    if width <= 0 or width > len(word_entries):
        return []
    entry_tokens = tuple(str(item["token"]) for item in word_entries)
    matches: list[dict[str, Any]] = []
    seen_word_ranges: set[tuple[int, int]] = set()
    for token_index in range(len(entry_tokens) - width + 1):
        selected = word_entries[token_index : token_index + width]
        selected_tokens = entry_tokens[token_index : token_index + width]
        omitted_indices = [
            index
            for index in range(len(target_tokens))
            if target_tokens[:index] + target_tokens[index + 1 :] == selected_tokens
        ]
        if len(omitted_indices) != 1:
            continue
        selected_word_indices = list(
            dict.fromkeys(int(item["word_index"]) for item in selected)
        )
        if any(current != previous + 1 for previous, current in pairwise(selected_word_indices)):
            continue
        word_range = (selected_word_indices[0], selected_word_indices[-1])
        if word_range in seen_word_ranges:
            continue
        seen_word_ranges.add(word_range)
        words: list[dict[str, Any]] = []
        seen_word_indices: set[int] = set()
        for item in selected:
            if item["word_index"] in seen_word_indices:
                continue
            seen_word_indices.add(item["word_index"])
            words.append(item)
        matches.append(
            {
                "start_sec": words[0]["start_sec"],
                "end_sec": words[-1]["end_sec"],
                "text": " ".join(item["text"] for item in words),
                "speaker_id": (
                    words[0].get("speaker_id")
                    if len({item.get("speaker_id") for item in words}) == 1
                    else None
                ),
                "source_refs": list(
                    dict.fromkeys(str(item["source_ref"]) for item in words)
                ),
                "first_word_index": word_range[0],
                "last_word_index": word_range[1],
                "omitted_target_token": target_tokens[omitted_indices[0]],
            }
        )
    return matches


def _review_interval_tightly_contains_match(
    judgement: dict[str, Any],
    match: dict[str, Any],
) -> bool:
    try:
        review_start = float(judgement["start_sec"])
        review_end = float(judgement["end_sec"])
        match_start = float(match["start_sec"])
        match_end = float(match["end_sec"])
    except (KeyError, TypeError, ValueError):
        return False
    if review_start > match_start or review_end < match_end:
        return False
    return (match_start - review_start) + (review_end - match_end) <= (
        _AUDIO_EXACT_PHRASE_MAX_TOTAL_PADDING_SEC
    )


def _word_token_entries(word_timestamps: Any) -> list[dict[str, Any]]:
    if not isinstance(word_timestamps, list):
        return []
    entries: list[dict[str, Any]] = []
    for word_index, raw in enumerate(word_timestamps):
        if not isinstance(raw, dict):
            continue
        try:
            start_sec = float(raw["start_sec"])
            end_sec = float(raw["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(raw.get("text") or "").strip()
        tokens = lexical_comparison_tokens(text)
        if end_sec <= start_sec or not tokens:
            continue
        source_ref = str(raw.get("word_id") or f"word_timestamp:{word_index}")
        for token in tokens:
            entries.append(
                {
                    "token": token,
                    "word_index": word_index,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "text": text,
                    "speaker_id": raw.get("speaker_id"),
                    "source_ref": source_ref,
                }
            )
    return entries


def _contiguous_word_matches(
    word_entries: list[dict[str, Any]],
    target_tokens: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not word_entries or not target_tokens:
        return []
    entry_tokens = tuple(str(item["token"]) for item in word_entries)
    matches: list[dict[str, Any]] = []
    seen_word_ranges: set[tuple[int, int]] = set()
    width = len(target_tokens)
    for token_index in range(len(entry_tokens) - width + 1):
        if entry_tokens[token_index : token_index + width] != target_tokens:
            continue
        selected = word_entries[token_index : token_index + width]
        selected_word_indices = list(dict.fromkeys(int(item["word_index"]) for item in selected))
        if any(current != previous + 1 for previous, current in pairwise(selected_word_indices)):
            continue
        word_range = (selected[0]["word_index"], selected[-1]["word_index"])
        if word_range in seen_word_ranges:
            continue
        seen_word_ranges.add(word_range)
        matches.append(_word_match_record(selected, comparison="exact_tokens"))
    if matches or not all(token.isdecimal() for token in target_tokens):
        return matches

    # Forced aligners often remove display punctuation from spoken numbers
    # (for example ``10:30`` becomes one aligned token ``1030``). Compare the
    # unchanged digit sequence only at this local evidence boundary; admitted
    # transcript text and tool parameters are never rewritten.
    target_digits = "".join(target_tokens)
    for start_index in range(len(word_entries)):
        selected: list[dict[str, Any]] = []
        observed_digits = ""
        previous_word_index: int | None = None
        for item in word_entries[start_index:]:
            token = str(item["token"])
            word_index = int(item["word_index"])
            if not token.isdecimal():
                break
            if previous_word_index is not None and word_index not in {
                previous_word_index,
                previous_word_index + 1,
            }:
                break
            selected.append(item)
            observed_digits += token
            previous_word_index = word_index
            if len(observed_digits) > len(target_digits):
                break
            if observed_digits != target_digits:
                continue
            word_range = (selected[0]["word_index"], selected[-1]["word_index"])
            if word_range not in seen_word_ranges:
                seen_word_ranges.add(word_range)
                matches.append(
                    _word_match_record(
                        selected,
                        comparison="numeric_format_equivalent",
                    )
                )
            break
    return matches


def _word_match_record(
    selected: list[dict[str, Any]],
    *,
    comparison: str,
) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    seen_word_indices: set[int] = set()
    for item in selected:
        if item["word_index"] in seen_word_indices:
            continue
        seen_word_indices.add(item["word_index"])
        words.append(item)
    return {
        "start_sec": words[0]["start_sec"],
        "end_sec": words[-1]["end_sec"],
        "text": " ".join(item["text"] for item in words),
        "speaker_id": (
            words[0].get("speaker_id")
            if len({item.get("speaker_id") for item in words}) == 1
            else None
        ),
        "source_refs": list(
            dict.fromkeys(str(item["source_ref"]) for item in words)
        ),
        "comparison": comparison,
    }


def _align_span_to_word_evidence(
    span: dict[str, Any],
    word_timestamps: Any,
) -> dict[str, Any]:
    """Make selected span bounds and text agree with reviewed word alignment."""

    if not isinstance(word_timestamps, list):
        return span
    try:
        start_sec = float(span["start_sec"])
        end_sec = float(span["end_sec"])
    except (KeyError, TypeError, ValueError):
        return span
    words: list[tuple[float, float, str]] = []
    for item in word_timestamps:
        if not isinstance(item, dict):
            continue
        try:
            word_start = float(item["start_sec"])
            word_end = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if word_end <= word_start or not text:
            continue
        if word_end > start_sec and word_start < end_sec:
            words.append((word_start, word_end, text))
    if not words:
        return span
    aligned = dict(span)
    aligned["start_sec"] = min(start_sec, min(item[0] for item in words))
    aligned["end_sec"] = max(end_sec, max(item[1] for item in words))
    aligned["text"] = " ".join(item[2] for item in words)
    return aligned


def _spans_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        return min(float(left["end_sec"]), float(right["end_sec"])) > max(
            float(left["start_sec"]),
            float(right["start_sec"]),
        )
    except (KeyError, TypeError, ValueError):
        return False


def _reviewed_target_bindings(
    *,
    instruction: str,
    selected_spans: list[dict[str, Any]],
    candidate_judgements: list[dict[str, Any]],
    target_contracts: list[dict[str, str]] | None = None,
    exact_grounding: dict[str, dict[str, Any]] | None = None,
    confirmed_absences: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    contracts = target_contracts or []
    if contracts:
        bindings: list[dict[str, Any]] = []
        for target in contracts:
            target_id = target["target_id"]
            resolved_spans = [item for item in selected_spans if item.get("target_id") == target_id]
            exact_resolution = (exact_grounding or {}).get(target_id, {})
            confirmed_absence = (confirmed_absences or {}).get(target_id)
            resolved = bool(resolved_spans)
            status = (
                "resolved"
                if resolved
                else "resolved_absent"
                if confirmed_absence
                else "unresolved"
            )
            bindings.append(
                {
                    "target_id": target_id,
                    "target_kind": target["kind"],
                    "target_value": target["value"],
                    "status": status,
                    "resolved_span_ids": [
                        str(item["span_id"]) for item in resolved_spans if item.get("span_id")
                    ],
                    "resolved_speaker_ids": list(
                        dict.fromkeys(
                            str(item["speaker_id"])
                            for item in resolved_spans
                            if item.get("speaker_id")
                        )
                    ),
                    "resolved_event_ids": [],
                    "confidence": (
                        _first_judgement_confidence(
                            candidate_judgements,
                            target_id=target_id,
                        )
                        if resolved
                        else (
                            confirmed_absence.get("confidence")
                            if confirmed_absence
                            else None
                        )
                    ),
                    "reason": (
                        str(confirmed_absence["reason"])
                        if confirmed_absence
                        else str(exact_resolution.get("reason"))
                        if exact_resolution
                        else (
                            "resolved from aligned candidate review"
                            if resolved
                            else "no aligned candidate satisfied the typed target"
                        )
                    ),
                    "source_refs": (
                        list(confirmed_absence.get("source_refs") or [])
                        if confirmed_absence
                        else [
                            str(source_ref)
                            for item in resolved_spans
                            for source_ref in item.get("source_refs") or []
                        ]
                    ),
                }
            )
        return bindings
    return [
        {
            "target_id": "target_000",
            "target_kind": "semantic_span",
            "target_value": instruction,
            "status": "resolved" if selected_spans else "unresolved",
            "resolved_span_ids": [
                str(item["span_id"]) for item in selected_spans if item.get("span_id")
            ],
            "resolved_speaker_ids": list(
                dict.fromkeys(
                    str(item["speaker_id"]) for item in selected_spans if item.get("speaker_id")
                )
            ),
            "resolved_event_ids": [],
            "confidence": _first_judgement_confidence(candidate_judgements),
            "reason": (
                "resolved from aligned candidate review"
                if selected_spans
                else "no aligned candidate satisfied the target"
            ),
            "source_refs": [
                str(source_ref)
                for item in selected_spans
                for source_ref in item.get("source_refs") or []
            ],
        }
    ]


def _missing_target_grounding(
    *,
    target_bindings: list[dict[str, Any]],
    selected_spans: list[dict[str, Any]],
) -> list[str]:
    missing = [
        f"typed target {item.get('target_id')}: {item.get('reason') or 'unresolved'}"
        for item in target_bindings
        if item.get("status") not in {"resolved", "resolved_absent"}
    ]
    if not target_bindings and not selected_spans:
        missing.append("grounded selected_spans")
    return missing


def _confirmed_absent_targets(
    *,
    selection: Any,
    target_contracts: list[dict[str, str]],
    selected_spans: list[dict[str, Any]],
    candidate_segments: list[dict[str, Any]],
    candidate_judgements: list[dict[str, Any]],
    exact_grounding: dict[str, dict[str, Any]],
    duration_sec: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return targets explicitly ruled out by a complete candidate review.

    Absence is evidence only when the provider reviewed every relevant candidate,
    explicitly rejected each one, and declared that no more evidence is needed.
    Provider failures, partial judgements, and ambiguous ownership remain unresolved.
    """

    if not (
        isinstance(selection, dict)
        and selection.get("status") == "completed"
        and selection.get("model_call_attempted") is True
        and selection.get("needs_more_evidence") is False
    ):
        return {}

    if duration_sec is None or not observation_scope(
        candidate_segments, duration_sec
    )["covers_entire_asset"]:
        return {}

    selected_target_ids = {
        str(item.get("target_id") or "") for item in selected_spans if item.get("target_id")
    }
    unresolved_target_ids = [
        target["target_id"]
        for target in target_contracts
        if target["target_id"] not in selected_target_ids
        and (exact_grounding.get(target["target_id"]) or {}).get("status") != "resolved"
    ]
    if not unresolved_target_ids:
        return {}

    candidate_ids_by_target: dict[str, set[str]] = {}
    unbound_candidate_ids: set[str] = set()
    for candidate in candidate_segments:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        target_id = str(candidate.get("target_id") or "")
        if target_id:
            candidate_ids_by_target.setdefault(target_id, set()).add(candidate_id)
        else:
            unbound_candidate_ids.add(candidate_id)

    judgements_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for judgement in candidate_judgements:
        candidate_id = str(judgement.get("candidate_id") or "")
        if candidate_id:
            judgements_by_candidate.setdefault(candidate_id, []).append(judgement)

    selected_candidate_ids = {
        str(item) for item in selection.get("selected_segment_ids") or []
    }
    confirmed: dict[str, dict[str, Any]] = {}
    for target_id in unresolved_target_ids:
        relevant_candidate_ids = set(candidate_ids_by_target.get(target_id, set()))
        if len(unresolved_target_ids) == 1:
            relevant_candidate_ids.update(unbound_candidate_ids)
        if not relevant_candidate_ids or relevant_candidate_ids & selected_candidate_ids:
            continue
        if not observation_scope(
            (item for item in candidate_segments if item.get("candidate_id") in relevant_candidate_ids),
            duration_sec,
        )["covers_entire_asset"]:
            continue

        reviewed: list[dict[str, Any]] = []
        complete_negative_review = True
        for candidate_id in sorted(relevant_candidate_ids):
            candidate_reviews = judgements_by_candidate.get(candidate_id, [])
            if len(candidate_reviews) != 1:
                complete_negative_review = False
                break
            judgement = candidate_reviews[0]
            if judgement.get("match_declared") is not True or judgement.get("matches") is not False:
                complete_negative_review = False
                break
            if judgement.get("status") in {"unresolved", "conflicted", "failed"} or (
                isinstance(judgement.get("confidence"), (int, float))
                and judgement["confidence"] <= 0
            ):
                complete_negative_review = False
                break
            reviewed.append(judgement)
        if not complete_negative_review:
            continue

        numeric_confidences = [
            float(item["confidence"])
            for item in reviewed
            if isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
        ]
        confirmed[target_id] = {
            "confidence": min(numeric_confidences) if numeric_confidences else None,
            "reason": (
                f"complete candidate review explicitly found no matching occurrence "
                f"across {len(relevant_candidate_ids)} candidate(s)"
            ),
            "source_refs": sorted(relevant_candidate_ids),
        }
    return confirmed


def _first_judgement_confidence(
    judgements: list[dict[str, Any]],
    *,
    target_id: str | None = None,
) -> Any:
    return next(
        (
            item.get("confidence")
            for item in judgements
            if item.get("matches")
            and item.get("confidence") is not None
            and (target_id is None or str(item.get("claim_target_id") or "") == target_id)
        ),
        None,
    )


__all__ = ["run_semantic_span_review"]
