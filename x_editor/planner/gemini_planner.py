"""Gemini 2.5 Pro planner for structured audio-edit plans.

This planner is intentionally thin: it turns a user request into an
``AudioEditPlan`` and leaves all execution details to the runtime executor.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import ValidationError

from x_editor.agents.base import GeminiClient
from x_editor.core.config import AudioEditConfig, resolve_gemini_api_key
from x_editor.core.invocation_trace import (
    ComponentInvocationRoute,
    ComponentInvocationStatus,
    ComponentInvocationTrace,
    ProviderInvocationRecord,
)
from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.planner._invocation_trace import provider_records_from_client
from x_editor.planner.perception_guidance import (
    PROMPT3_PLANNER_GAP_FILLING,
    add_planner_perception_guidance,
)
from x_editor.planner.plan_normalizer import normalize_planner_output
from x_editor.planner.routing_hard_boundaries import apply_routing_hard_boundaries
from x_editor.planner.routing_policy import routing_policy_context
from x_editor.planner.tool_model_context import tool_model_capability_context
from x_editor.semantic.plan_integration import semantic_text_edit_preflight_plan

_PLANNER_CONTEXT_EVIDENCE_KEYS = (
    "routing_policy_candidates",
    "planner_perception_guidance",
    "span_selection_analysis",
    "perception_gap_fill_guidance",
    "edit_session_graph_context",
)

_OPERATION_OBJECTIVE_SEMANTIC_CONTRACTS: dict[str, dict[str, Any]] = {
    "artifact_ops.extract_spans": {
        "semantic_kind": "selected_output_excludes_complement",
        "objective_types": frozenset({"removal", "event_removal"}),
    },
    "no_reference_voice_conversion": {
        "semantic_kind": "voice_conversion",
        "objective_types": frozenset({"generation", "style_edit"}),
    },
    "emphasis_stress_edit": {
        "semantic_kind": "lexical_emphasis",
        "objective_types": frozenset({"edit", "style_edit"}),
    },
    "foreground_background_remix": {
        "semantic_kind": "foreground_isolation",
        "objective_types": frozenset({"span_grounding", "removal", "event_removal"}),
    },
    "speech_enhancement": {
        "semantic_kind": "speech_cleanup",
        "objective_types": frozenset({"removal", "event_removal"}),
    },
}

_PRESERVATION_VERB_RE = re.compile(
    r"(?i)\b(?:keep|keeping|preserve|preserving|maintain|maintaining|"
    r"leave|leaving)\b|\bwithout\s+changing\b"
)
_ACTION_VERB_RE = re.compile(
    r"(?i)\b(?:add|adjust|apply|change|convert|delete|edit|emphasize|emphasise|"
    r"extract|generate|insert|isolate|make|modify|mute|remove|replace|"
    r"retain|shift|stress|suppress|synthesize|transform|translate)\b"
)


class GeminiPlanner:
    """Generate an ``AudioEditPlan`` with Gemini 2.5 Pro."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 32768,
        timeout: float = 60.0,
        available_operations: list[dict[str, Any]] | None = None,
        client: GeminiClient | None = None,
    ) -> None:
        config = AudioEditConfig()
        self.model_name = model_name or os.getenv("X_EDITOR_GEMINI_PLANNER_MODEL", "gemini-2.5-pro")
        self.available_operations = available_operations or []
        self.api_key = resolve_gemini_api_key(api_key) or config.omni_api_key
        self._client_provided = client is not None
        self.client = client or GeminiClient(
            api_key=self.api_key,
            base_url=base_url or config.omni_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.last_invocation_trace: ComponentInvocationTrace | None = None

    def plan(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
    ) -> AudioEditPlan:
        """Return a structured plan without executing any audio operation."""

        self.last_invocation_trace = None
        provider_invocations: list[ProviderInvocationRecord] = []
        original_context = context if isinstance(context, dict) else None
        context = dict(context or {})
        operations = context.get("available_operations") or self.available_operations
        if not operations:
            self._set_invocation_trace(
                status="failed",
                route="deterministic",
                provider_invocations=provider_invocations,
            )
            raise RuntimeError("GeminiPlanner requires a non-empty available_operations list")
        context.setdefault(
            "routing_policy_candidates",
            routing_policy_context(
                user_request,
                capabilities=_operation_ids(operations),
                reference_audio_paths=context.get("reference_audio_paths") or [],
            ),
        )
        context = add_planner_perception_guidance(
            user_request=user_request,
            context=context,
            available_operations=operations,
        )
        self.last_planner_evidence_context = _planner_evidence_context(context)
        if original_context is not None:
            original_context.update(self.last_planner_evidence_context)

        preflight = _deterministic_preflight_plan(
            user_request,
            context=context,
            operations=operations,
            model_name=self.model_name,
        )
        if preflight is not None:
            _ensure_plan_decision_audit(
                preflight,
                user_request=user_request,
                context=context,
                operations=operations,
            )
            self._set_invocation_trace(
                status="completed",
                route="deterministic_preflight",
                provider_invocations=provider_invocations,
            )
            return preflight

        if not self.api_key and not self._client_provided:
            fallback = _deterministic_fallback_plan(
                user_request,
                context=context,
                operations=operations,
            primary_error="Shared Gemini 2.5 Pro API key is required for GeminiPlanner",
                model_name=self.model_name,
            )
            if fallback is not None:
                _ensure_plan_decision_audit(
                    fallback,
                    user_request=user_request,
                    context=context,
                    operations=operations,
                )
                self._set_invocation_trace(
                    status="completed",
                    route="deterministic_fallback",
                    provider_invocations=provider_invocations,
                    fallback_used=True,
                    fallback_reason="planner_api_key_missing",
                )
                return fallback
            self._set_invocation_trace(
                status="failed",
                route="deterministic",
                provider_invocations=provider_invocations,
            )
            raise RuntimeError("Shared Gemini 2.5 Pro API key is required for GeminiPlanner")
        try:
            data = self.client.call_json(
                _system_prompt_for_context(context),
                _user_prompt(
                    user_request=user_request,
                    context=context,
                    operations=operations,
                    model_name=self.model_name,
                ),
            )
        except Exception as exc:
            provider_invocations.extend(
                provider_records_from_client(
                    self.client,
                    provider="gemini",
                    model=self.model_name,
                    planner_purpose="primary",
                    start_index=len(provider_invocations) + 1,
                    succeeded=False,
                    error=exc,
                )
            )
            fallback = _deterministic_fallback_plan(
                user_request,
                context=context,
                operations=operations,
                primary_error=f"{type(exc).__name__}: {exc}",
                model_name=self.model_name,
            )
            if fallback is not None:
                _ensure_plan_decision_audit(
                    fallback,
                    user_request=user_request,
                    context=context,
                    operations=operations,
                )
                self._set_invocation_trace(
                    status="completed",
                    route="mixed",
                    provider_invocations=provider_invocations,
                    fallback_used=True,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
                return fallback
            self._set_invocation_trace(
                status="failed",
                route="provider",
                provider_invocations=provider_invocations,
            )
            raise
        provider_invocations.extend(
            provider_records_from_client(
                self.client,
                provider="gemini",
                model=self.model_name,
                planner_purpose="primary",
                start_index=len(provider_invocations) + 1,
                succeeded=True,
            )
        )
        if not isinstance(data, dict):
            self._set_invocation_trace(
                status="failed",
                route="provider",
                provider_invocations=provider_invocations,
            )
            raise RuntimeError(f"GeminiPlanner expected a JSON object, got {type(data).__name__}")
        call_metrics = getattr(self.client, "last_call_metrics", None)
        if isinstance(call_metrics, dict):
            metadata = data.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["gemini_planner_call"] = dict(call_metrics)
        normalize_planner_output(
            data,
            user_request=user_request,
            context=context,
            operations=operations,
            planner_model=self.model_name,
            phase="pre_routing",
        )
        apply_routing_hard_boundaries(data, user_request=user_request, context=context, operations=operations)
        normalize_planner_output(
            data,
            user_request=user_request,
            context=context,
            operations=operations,
            planner_model=self.model_name,
            phase="post_routing",
        )
        _ensure_planner_decision_audit(data, user_request=user_request, context=context, operations=operations)
        try:
            plan = AudioEditPlan(**data)
        except ValidationError as exc:
            self._set_invocation_trace(
                status="failed",
                route="provider",
                provider_invocations=provider_invocations,
            )
            raise RuntimeError(f"GeminiPlanner returned invalid AudioEditPlan schema: {exc}") from exc
        self._set_invocation_trace(
            status="completed",
            route="provider",
            provider_invocations=provider_invocations,
        )
        return plan

    def _set_invocation_trace(
        self,
        *,
        status: ComponentInvocationStatus,
        route: ComponentInvocationRoute,
        provider_invocations: list[ProviderInvocationRecord],
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        self.last_invocation_trace = ComponentInvocationTrace.build(
            component="planner",
            component_name=type(self).__name__,
            route=route,
            status=status,
            provider_invocations=provider_invocations,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            metadata={"model": self.model_name},
        )


def _ensure_planner_decision_audit(
    data: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
) -> None:
    """Ensure planner traces include objectives, step reasons, and a self-check."""

    metadata = data.setdefault("metadata", {})
    _ensure_planner_skill_trace(metadata, context)
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    objectives = _ensure_high_level_objectives(
        metadata,
        user_request=user_request,
    )
    existing = metadata.get("step_rationales")
    existing_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, dict) and isinstance(item.get("step_id"), str):
                existing_by_id[item["step_id"]] = item
    operation_descriptions = {
        str(item.get("operation")): str(item.get("description") or "")
        for item in operations
        if isinstance(item, dict) and item.get("operation")
    }
    rationales: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or f"step_{index + 1}")
        operation = str(step.get("operation") or "")
        existing_item = dict(existing_by_id.get(step_id) or {})
        decision_reason = existing_item.get("decision_reason") or _default_step_decision_reason(
            operation,
            operation_descriptions.get(operation, ""),
        )
        input_reason = existing_item.get("input_reason") or _default_step_input_reason(step, context)
        instruction_alignment = existing_item.get("instruction_alignment") or _default_instruction_alignment(
            step,
            user_request,
        )
        boundary_reason = existing_item.get("boundary_reason") or _default_boundary_reason(operation)
        objective_ids = _normalize_objective_ids(existing_item.get("objective_ids"))
        if not objective_ids:
            objective_ids = _objective_ids_for_step(step, objectives)
        rationales.append(
            {
                "step_id": step_id,
                "operation": operation,
                "objective_ids": objective_ids,
                "decision_reason": str(decision_reason),
                "input_reason": str(input_reason),
                "instruction_alignment": str(instruction_alignment),
                "boundary_reason": str(boundary_reason),
            }
        )
    metadata["step_rationales"] = rationales
    objective_coverage = _ensure_objective_coverage(
        metadata,
        objectives=objectives,
        steps=steps,
        rationales=rationales,
    )
    uncovered_objectives = [
        item.get("objective_id")
        for item in objective_coverage
        if item.get("status") not in {"planned", "covered_by_constraint", "unsupported"}
    ]
    self_check = metadata.get("plan_self_check")
    if not isinstance(self_check, dict):
        metadata["plan_self_check"] = {
            "consistent": not uncovered_objectives,
            "checked_items": [
                "High-level objectives were decomposed before step selection.",
                "Every edit/removal/addition/reference/preservation objective has explicit step coverage or an unsupported reason.",
                "Each step is tied to the user instruction rather than a generic tool route.",
                "Source/content audio and reference audio assignments were checked.",
                "Context/reference spans are evidence only unless explicitly edited.",
                "The final step preserves requested content, speaker, timing, and acoustic constraints where supported.",
            ],
            "corrections_applied": [],
            "remaining_risks": _default_plan_risks(steps, objective_coverage=objective_coverage),
        }
    else:
        checked_items = self_check.setdefault("checked_items", [])
        if isinstance(checked_items, list):
            for item in [
                "High-level objectives were decomposed before step selection.",
                "Every edit/removal/addition/reference/preservation objective has explicit step coverage or an unsupported reason.",
            ]:
                if item not in checked_items:
                    checked_items.insert(0, item)
        risks = self_check.setdefault("remaining_risks", [])
        if isinstance(risks, list):
            for risk in _default_plan_risks(steps, objective_coverage=objective_coverage):
                if risk not in risks:
                    risks.append(risk)
        if uncovered_objectives:
            self_check["consistent"] = False


def _ensure_planner_skill_trace(metadata: dict[str, Any], context: dict[str, Any]) -> None:
    candidates = _normalize_routing_policy_candidates(context.get("routing_policy_candidates"))
    if candidates:
        metadata["routing_policy_candidates"] = candidates
    else:
        candidates = _normalize_routing_policy_candidates(metadata.get("routing_policy_candidates"))
        if candidates:
            metadata["routing_policy_candidates"] = candidates
    skill_ids = _planner_skill_ids_from_candidates(candidates)
    if skill_ids:
        metadata["planner_skill_ids"] = skill_ids


def _normalize_routing_policy_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if hasattr(item, "model_dump") and callable(item.model_dump):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        candidate = {
            key: item.get(key)
            for key in (
                "policy_id",
                "priority",
                "score",
                "selected_tools",
                "workflow",
                "rationale",
                "required_evidence",
                "guardrails",
                "matched_signals",
                "missing_tools",
            )
            if item.get(key) not in (None, "", [], {})
        }
        if candidate:
            normalized.append(candidate)
    return normalized


def _planner_skill_ids_from_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    skill_ids: list[str] = []
    for candidate in candidates:
        policy_id = candidate.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            continue
        if policy_id not in skill_ids:
            skill_ids.append(policy_id)
    return skill_ids


def _ensure_high_level_objectives(metadata: dict[str, Any], *, user_request: str) -> list[dict[str, Any]]:
    existing = metadata.get("high_level_objectives")
    objectives = _normalize_high_level_objectives(existing)
    if not objectives:
        objectives = _infer_high_level_objectives(user_request)
    metadata["high_level_objectives"] = objectives
    return objectives


def _normalize_high_level_objectives(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    objectives: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            objective = item.strip()
            raw: dict[str, Any] = {"objective": objective}
        elif isinstance(item, dict):
            raw = dict(item)
            objective = str(raw.get("objective") or raw.get("goal") or raw.get("description") or "").strip()
        else:
            continue
        if not objective:
            continue
        obj_id = str(raw.get("id") or raw.get("objective_id") or f"obj_{len(objectives) + 1}").strip()
        if not obj_id or obj_id in seen:
            obj_id = f"obj_{len(objectives) + 1}"
        seen.add(obj_id)
        objectives.append(
            {
                "id": obj_id,
                "objective": objective,
                "type": str(raw.get("type") or _objective_type_from_text(objective)),
                "source_instruction_fragment": str(raw.get("source_instruction_fragment") or objective),
                "required": bool(raw.get("required", True)),
                **(
                    {"facet": str(raw["facet"])}
                    if str(raw.get("facet") or "").strip()
                    else {}
                ),
            }
        )
    return objectives


def _infer_high_level_objectives(user_request: str) -> list[dict[str, Any]]:
    request = str(user_request or "").strip()
    if not request:
        return []
    objectives: list[dict[str, Any]] = []

    def add(
        objective: str,
        typ: str,
        fragment: str | None = None,
        *,
        facet: str | None = None,
    ) -> None:
        normalized = objective.strip()
        if not normalized:
            return
        if any(item["objective"].casefold() == normalized.casefold() for item in objectives):
            return
        objectives.append(
            {
                "id": f"obj_{len(objectives) + 1}",
                "objective": normalized,
                "type": typ,
                "source_instruction_fragment": fragment or normalized,
                "required": True,
                **({"facet": facet} if facet else {}),
            }
        )

    action_scope, preservation_scope = _split_action_and_preservation_scopes(request)
    lowered = action_scope.casefold()
    if _contains_any(lowered, ["remove", "delete", "eliminate", "suppress", "mute", "drop", "get rid of", "去掉", "删除", "移除"]):
        removal_fragment = _extract_clause(action_scope, ["remove", "delete", "eliminate", "suppress", "mute", "drop", "get rid of"])
        add(f"Remove or suppress the requested content/event: {removal_fragment or action_scope}", "removal", removal_fragment)
    emotion_markers = [
        "emotion",
        "angry",
        "happy",
        "sad",
        "calm",
        "surprised",
        "hushed",
        "reflective",
        "internal monologue",
        "tone",
        "style",
        "情绪",
        "语气",
        "风格",
    ]
    rate_markers = [
        "pitch",
        "tempo",
        "rate",
        "faster",
        "slower",
        "speed up",
        "slow down",
        "音高",
        "速度",
        "语速",
    ]
    has_emotion_edit = _contains_any(lowered, emotion_markers)
    has_rate_edit = _contains_any(lowered, rate_markers)
    has_generation_intent = _has_generation_or_conversion_intent(action_scope)
    if has_emotion_edit or has_rate_edit:
        style_fragment = _extract_clause(
            action_scope,
            ["change", "convert", "transform", "apply", "make", "shift", "increase", "decrease"],
        )
        style_scope = style_fragment or action_scope
        if has_emotion_edit and has_rate_edit:
            add(
                f"Apply the requested emotion/tone transformation: {style_scope}",
                "style_edit",
                style_fragment,
                facet="emotion",
            )
            add(
                f"Apply the requested pitch/rate/tempo transformation: {style_scope}",
                "style_edit",
                style_fragment,
                facet="rate",
            )
        else:
            add(
                f"Apply the requested speech/style transformation: {style_scope}",
                "style_edit",
                style_fragment,
                facet=(
                    None
                    if has_generation_intent
                    else "emotion" if has_emotion_edit else "rate"
                ),
            )
    acoustic_effect_match = re.search(
        r"(?i)\b(?:apply|add|create|make)\b[^.;]{0,80}"
        r"\b(?:reverb|echo|room|channel|acoustic effect)\b",
        action_scope,
    )
    if acoustic_effect_match:
        acoustic_fragment = acoustic_effect_match.group(0).strip(" ,")
        add(
            f"Apply the requested acoustic effect: {acoustic_fragment}",
            "acoustic_effect",
            acoustic_fragment,
        )
    explicit_span_grounding = _contains_any(
        lowered,
        ["extract", "isolate", "retain only", "identify", "locate", "提取", "定位"],
    )
    if explicit_span_grounding:
        extract_fragment = _extract_clause(action_scope, ["extract", "isolate", "retain", "identify", "locate"])
        add(f"Ground and select the requested target span(s): {extract_fragment or action_scope}", "span_grounding", extract_fragment)
    elif _request_has_speaker_or_span_scope(action_scope) and (
        has_emotion_edit or has_rate_edit
    ):
        add(
            f"Ground and select the locally scoped target span(s): {action_scope}",
            "span_grounding",
            action_scope,
        )
    if has_generation_intent:
        add(
            "Perform the requested generation/conversion while using the correct "
            f"source/reference audio: {action_scope}",
            "generation",
            action_scope,
        )
    if preservation_scope:
        add(
            f"Preserve constraints: {preservation_scope}",
            "preservation",
            preservation_scope,
        )
    if not objectives:
        add(request, "edit")
    return objectives


def _split_action_and_preservation_scopes(request: str) -> tuple[str, str]:
    """Separate requested edits from clauses that only constrain preservation."""

    text = str(request or "").strip()
    if not text:
        return "", ""
    marker = _PRESERVATION_VERB_RE.search(text)
    if marker is None:
        unchanged_clause = re.search(
            r"(?i)(?:^|[,;])\s*(?P<clause>[^,;.]*\bunchanged\b[^.;]*)",
            text,
        )
        if unchanged_clause is None:
            return text, ""
        clause = unchanged_clause.group("clause").strip(" ,;")
        action = text[: unchanged_clause.start()].strip(" ,;")
        return action, clause

    if marker.start() > 0:
        action = re.sub(
            r"(?i)[,;\s]*(?:while|and|but)\s*$",
            "",
            text[: marker.start()],
        ).strip(" ,;")
        return action, text[marker.start() :].strip(" ,;")

    tail = text[marker.end() :]
    action_boundary = re.search(
        r"(?i)(?:[,;]\s*(?:but\s+|then\s+)?|\bbut\s+|\band\s+)(?="
        + _ACTION_VERB_RE.pattern.removeprefix("(?i)")
        + r")",
        tail,
    )
    if action_boundary is None:
        return "", text
    boundary_end = marker.end() + action_boundary.end()
    preservation = text[: marker.end() + action_boundary.start()].strip(" ,;")
    return text[boundary_end:].strip(" ,;"), preservation


def _has_generation_or_conversion_intent(request: str) -> bool:
    action_scope, _ = _split_action_and_preservation_scopes(request)
    lowered = action_scope.casefold()
    if _contains_any(
        lowered,
        [
            "generate",
            "synthesize",
            "clone",
            "translate",
            "sounds older",
            "sound older",
            "much older",
            "sounds younger",
            "sound younger",
            "much younger",
            "elderly voice",
            "aged voice",
            "生成",
            "合成",
            "克隆",
            "翻译",
        ],
    ):
        return True
    return bool(
        re.search(
            r"(?i)\b(?:change|convert|transform|replace|modify|match)\b"
            r"[^.;]{0,120}\b(?:voice|timbre|accent|language)\b",
            action_scope,
        )
        or re.search(
            r"(?:改变|转换|替换|修改|匹配)[^。；]{0,60}(?:声音|音色|口音|语言)",
            action_scope,
        )
    )


def _extract_clause(text: str, starts: list[str]) -> str:
    for start in starts:
        match = re.search(rf"(?i)\b{re.escape(start)}\b[^.;]*", text)
        if match:
            return match.group(0).strip(" ,")
    return ""


def _objective_type_from_text(text: str) -> str:
    lowered = str(text).casefold()
    if _contains_any(lowered, ["remove", "delete", "eliminate", "suppress", "mute", "laughter", "applause"]):
        return "removal"
    if _contains_any(lowered, ["preserve", "keep", "maintain", "unchanged", "without changing"]):
        return "preservation"
    if _contains_any(lowered, ["extract", "locate", "identify", "span"]):
        return "span_grounding"
    if _contains_any(lowered, ["emotion", "tone", "style", "pitch", "rate", "tempo", "angry", "happy", "sad"]):
        return "style_edit"
    return "edit"


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_objective_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            ids.append(text)
    return ids


def _objective_ids_for_step(step: dict[str, Any], objectives: list[dict[str, Any]]) -> list[str]:
    haystack = _step_executable_haystack(step)
    ids: list[str] = []
    for objective in objectives:
        obj_id = str(objective.get("id") or "")
        if not obj_id:
            continue
        if _step_covers_objective(step, objective, haystack):
            ids.append(obj_id)
    return ids


def _ensure_objective_coverage(
    metadata: dict[str, Any],
    *,
    objectives: list[dict[str, Any]],
    steps: list[Any],
    rationales: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = metadata.get("objective_coverage")
    coverage = _normalize_objective_coverage(existing, objectives)
    if not coverage:
        coverage = []
        rationale_by_objective: dict[str, list[str]] = {}
        for rationale in rationales:
            step_id = str(rationale.get("step_id") or "")
            for obj_id in _normalize_objective_ids(rationale.get("objective_ids")):
                rationale_by_objective.setdefault(obj_id, []).append(step_id)
        for objective in objectives:
            obj_id = str(objective.get("id") or "")
            covered = rationale_by_objective.get(obj_id, [])
            if not covered:
                for step in steps:
                    if isinstance(step, dict) and _step_covers_objective(step, objective, _step_executable_haystack(step)):
                        covered.append(str(step.get("id") or ""))
            covered = [item for item in dict.fromkeys(covered) if item]
            if covered:
                status = "covered_by_constraint" if objective.get("type") == "preservation" else "planned"
                reason = "Covered by executable step params and/or instruction alignment."
            else:
                status = "uncovered"
                reason = "No executable step explicitly covers this high-level objective."
            coverage.append(
                {
                    "objective_id": obj_id,
                    "covered_by_steps": covered,
                    "status": status,
                    "coverage_reason": reason,
                }
            )
    normalizations: list[dict[str, Any]] = []
    steps_by_id = {
        str(step.get("id") or ""): step
        for step in steps
        if isinstance(step, dict) and step.get("id")
    }
    for item in coverage:
        if item.get("status") in {"planned", "covered_by_constraint"}:
            continue
        objective = next(
            (
                objective
                for objective in objectives
                if str(objective.get("id") or "") == str(item.get("objective_id") or "")
            ),
            None,
        )
        if objective is None:
            continue
        covered = [
            step_id
            for step_id, step in steps_by_id.items()
            if _step_has_deterministic_objective_semantics(step, objective)
        ]
        if not covered:
            continue
        item["covered_by_steps"] = covered
        item["status"] = "planned"
        item["coverage_reason"] = (
            "Covered by the selected operation's deterministic output semantics."
        )
        normalizations.append(
            {
                "objective_id": item["objective_id"],
                "covered_by_steps": covered,
                "reason": "deterministic_operation_semantics",
            }
        )
    if normalizations:
        metadata["objective_coverage_normalizations"] = normalizations
    metadata["objective_coverage"] = coverage
    return coverage


def _normalize_objective_coverage(value: Any, objectives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    valid_ids = {str(item.get("id")) for item in objectives}
    coverage: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        obj_id = str(item.get("objective_id") or item.get("id") or "").strip()
        if not obj_id or obj_id not in valid_ids:
            continue
        covered = _normalize_objective_ids(item.get("covered_by_steps") or item.get("step_ids") or [])
        coverage.append(
            {
                "objective_id": obj_id,
                "covered_by_steps": covered,
                "status": str(item.get("status") or ("planned" if covered else "uncovered")),
                "coverage_reason": str(item.get("coverage_reason") or item.get("reason") or ""),
            }
        )
    return coverage


def _step_executable_haystack(step: dict[str, Any]) -> str:
    values = [
        step.get("operation"),
        step.get("description"),
        step.get("inputs"),
        step.get("params"),
        step.get("expected_output"),
    ]
    return " ".join(json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value) for value in values if value).casefold()


def _step_covers_objective(step: dict[str, Any], objective: dict[str, Any], haystack: str) -> bool:
    typ = str(objective.get("type") or "").casefold()
    operation = str(step.get("operation") or "").casefold()
    contract_coverage = _operation_objective_semantic_coverage(step, objective)
    if contract_coverage is not None:
        return contract_coverage
    if typ == "preservation":
        return bool(operation and operation != "unsupported")
    if typ in {"removal", "event_removal"}:
        if _step_has_deterministic_objective_semantics(step, objective):
            return True
        if operation in {
            "artifact_ops.remove_spans",
            "artifact_ops.mute_spans",
            "artifact_ops.beep_spans",
            "paralinguistic_event_remove",
        }:
            return True
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        params_text = json.dumps(params, ensure_ascii=False, sort_keys=True).casefold()
        return _contains_any(
            params_text,
            [
                "remove",
                "delete",
                "eliminate",
                "suppress",
                "mute",
                "event_removal",
                "target_event",
                "target_events",
            ],
        )
    if typ in {"style_edit", "content_edit"}:
        return _contains_any(haystack, ["emotion", "style", "tone", "pitch", "rate", "tempo", "angry", "happy", "sad", "hushed", "reflective", "internal monologue", "emotion_style_edit", "pitch_rate_edit", "audio_effects"])
    if typ == "acoustic_effect":
        return _contains_any(haystack, ["audio_effects", "spatial_reverb", "reverb", "room", "channel", "effect", "acoustic"])
    if typ == "span_grounding":
        return _contains_any(haystack, ["extract", "locate", "identify", "selected_spans", "speaker_scoped_edit", "perception", "selector", "selection_instruction"])
    if typ in {"generation", "reference_match"}:
        return _contains_any(haystack, ["qwen3_tts", "seed_vc", "voice", "timbre", "accent", "translate", "synthesize", "generate", "reference_audio"])
    objective_text = str(objective.get("objective") or "").casefold()
    tokens = [token for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", objective_text) if len(token) >= 4]
    return bool(tokens and sum(1 for token in tokens if token in haystack) >= max(1, min(2, len(tokens))))


def _step_has_deterministic_objective_semantics(
    step: dict[str, Any],
    objective: dict[str, Any],
) -> bool:
    """Recognize objective coverage implied by an operation, not planner prose."""

    return _operation_objective_semantic_coverage(step, objective) is True


def _operation_objective_semantic_coverage(
    step: dict[str, Any],
    objective: dict[str, Any],
) -> bool | None:
    """Evaluate an explicit operation contract for one objective when defined."""

    operation = str(step.get("operation") or "").casefold()
    style_facet = str(objective.get("facet") or "").casefold()
    if style_facet in {"emotion", "rate"}:
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        nested = str(
            params.get("edit_op") or params.get("edit_operation") or ""
        ).casefold()
        effective_operations = {operation, nested}
        if style_facet == "emotion":
            return any(
                item.endswith("emotion_style_edit")
                for item in effective_operations
                if item
            )
        return any(
            item.endswith("pitch_rate_edit") or item.endswith("speech_rate")
            for item in effective_operations
            if item
        )
    contract = _OPERATION_OBJECTIVE_SEMANTIC_CONTRACTS.get(operation)
    if contract is None and operation == "artifact_ops.speaker_scoped_edit":
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        nested_operation = str(
            params.get("edit_op") or params.get("edit_operation") or ""
        ).casefold()
        contract = _OPERATION_OBJECTIVE_SEMANTIC_CONTRACTS.get(nested_operation)
        if contract is None and nested_operation.startswith("edit_data_pipeline."):
            contract = _OPERATION_OBJECTIVE_SEMANTIC_CONTRACTS.get(
                nested_operation.removeprefix("edit_data_pipeline.")
            )
    if contract is None:
        return None
    objective_type = str(objective.get("type") or "").casefold()
    if objective_type not in contract["objective_types"]:
        return None
    semantic_kind = contract["semantic_kind"]
    objective_text = _objective_semantic_text(objective)
    if semantic_kind == "selected_output_excludes_complement":
        return _objective_requests_complement_exclusion(objective_text)
    if semantic_kind == "voice_conversion":
        return _objective_requests_voice_conversion(objective_text)
    if semantic_kind == "lexical_emphasis":
        return _objective_requests_lexical_emphasis(objective_text)
    if semantic_kind == "foreground_isolation":
        return _foreground_isolation_covers_objective(
            step,
            objective_type=objective_type,
            objective_text=objective_text,
        )
    if semantic_kind == "speech_cleanup":
        return _contains_any(
            objective_text,
            [
                "reverb",
                "echo",
                "noise",
                "hum",
                "hiss",
                "dereverb",
                "denoise",
                "混响",
                "回声",
                "噪声",
            ],
        )
    return False


def _objective_semantic_text(objective: dict[str, Any]) -> str:
    return " ".join(
        str(objective.get(field) or "")
        for field in ("objective", "source_instruction_fragment")
    ).casefold()


def _objective_requests_complement_exclusion(objective_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:everything|anything|all audio|all content|all regions?)\s+else\b"
            r"|\b(?:all|every)\s+other\b"
            r"|\bthe\s+rest\b"
            r"|\b(?:extract|retain|keep|isolate)(?:\s+and\s+\w+)?\s+only\b"
            r"|(?:其余|其他所有|剩余)(?:内容|音频|部分|片段|声音)?",
            objective_text,
        )
    )


def _objective_requests_voice_conversion(objective_text: str) -> bool:
    has_voice_target = _contains_any(
        objective_text,
        [
            "voice",
            "timbre",
            "vocal",
            "speaker",
            "resonance",
            "elderly",
            "aged",
            "older",
            "younger",
            "child",
            "音色",
            "声音",
            "说话人",
            "老人",
            "老年",
            "儿童",
            "童声",
        ],
    )
    has_conversion = _contains_any(
        objective_text,
        [
            "change",
            "convert",
            "transform",
            "modify",
            "match",
            "adjust",
            "generation/conversion",
            "改变",
            "转换",
            "变换",
            "调整",
        ],
    )
    return has_voice_target and has_conversion


def _objective_requests_lexical_emphasis(objective_text: str) -> bool:
    return _contains_any(
        objective_text,
        [
            "emphasis",
            "emphasize",
            "emphasise",
            "lexical stress",
            "word stress",
            "stress all",
            "stress every",
            "重音",
            "重读",
            "强调",
            "着重",
        ],
    )


def _foreground_isolation_covers_objective(
    step: dict[str, Any],
    *,
    objective_type: str,
    objective_text: str,
) -> bool:
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    action = str(params.get("foreground_action") or "").strip().casefold()
    if action not in {"isolate_foreground", "foreground_only", "extract_foreground"}:
        return False
    if objective_type == "span_grounding":
        return _contains_any(
            objective_text,
            [
                "foreground",
                "vocal",
                "singing",
                "voice",
                "speech",
                "speaker",
                "人声",
                "歌声",
                "说话",
            ],
        )
    return _contains_any(
        objective_text,
        [
            "background",
            "instrument",
            "accompaniment",
            "music",
            "noise",
            "everything else",
            "all other",
            "the rest",
            "背景",
            "伴奏",
            "乐器",
            "噪声",
            "噪音",
            "其余",
            "其他所有",
        ],
    )


def _ensure_plan_decision_audit(
    plan: AudioEditPlan,
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
) -> None:
    data = plan.model_dump()
    _ensure_planner_decision_audit(
        data,
        user_request=user_request,
        context=context,
        operations=operations,
    )
    plan.metadata.update(data.get("metadata") or {})


def _apply_hard_routing_boundaries(
    data: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
) -> None:
    """Apply non-negotiable routing boundaries after model planning.

    Planner skills and tool capabilities are prompt context; this guard keeps
    the executable plan from drifting into known harmful routes.
    """

    steps = data.get("steps")
    if not isinstance(steps, list):
        return
    available = _operation_ids(operations)
    metadata = data.setdefault("metadata", {})
    corrections = metadata.setdefault("routing_boundary_corrections", [])

    if _is_existing_audio_text_replacement(user_request, context):
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("operation") != "qwen3_tts":
                continue
            if "edit_data_pipeline.text_edit" in available:
                _rewrite_step_to_edit_data_pipeline(
                    step,
                    operation="text_edit",
                    user_request=user_request,
                    context=context,
                    reason="existing-audio lexical edits must use edit_data_pipeline.text_edit; qwen3_tts is not allowed for text replacement",
                )
                corrections.append(
                    {
                        "step_id": step.get("id"),
                        "from": "qwen3_tts",
                        "to": "edit_data_pipeline.text_edit",
                        "reason": "existing_audio_text_replacement_hard_boundary",
                    }
                )
            else:
                _rewrite_step_to_unsupported(
                    step,
                    "Existing-audio lexical replacement requires edit_data_pipeline.text_edit; qwen3_tts is not allowed.",
                    available,
                )
                corrections.append(
                    {
                        "step_id": step.get("id"),
                        "from": "qwen3_tts",
                        "to": step.get("operation"),
                        "reason": "missing_edit_data_pipeline_text_edit_for_existing_audio_text_replacement",
                    }
                )

    for step in steps:
        if not isinstance(step, dict):
            continue
        operation = str(step.get("operation") or "")
        if operation not in {"edit_data_pipeline.pitch_rate_edit", "edit_data_pipeline.emotion_style_edit"}:
            continue
        if "foreground_background_remix" not in available:
            continue
        if not _direct_edit_needs_foreground_background_remix(user_request, operation):
            continue
        _rewrite_step_to_foreground_background_remix(
            step,
            user_request=user_request,
            context=context,
            reason="scoped_or_background_sensitive_speech_edit_requires_foreground_background_remix",
        )
        corrections.append(
            {
                "step_id": step.get("id"),
                "from": operation,
                "to": "foreground_background_remix",
                "reason": "scoped_or_background_sensitive_speech_edit_requires_foreground_background_remix",
            }
        )

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("operation") != "artifact_ops.speaker_scoped_edit":
            continue
        nested_operation = _scoped_edit_data_pipeline_operation(user_request)
        if nested_operation is None:
            continue
        params = step.setdefault("params", {})
        if not isinstance(params, dict):
            params = {}
            step["params"] = params
        if (
            nested_operation == "pitch_rate_edit"
            and "foreground_background_remix" in available
            and _direct_edit_needs_foreground_background_remix(user_request, "edit_data_pipeline.pitch_rate_edit")
        ):
            _rewrite_scoped_step_to_foreground_background_remix(
                step,
                nested_operation=nested_operation,
                user_request=user_request,
                context=context,
                reason="time_scoped_rate_edit_requires_foreground_background_remix",
            )
            corrections.append(
                {
                    "step_id": step.get("id"),
                    "from": "artifact_ops.speaker_scoped_edit",
                    "to": "foreground_background_remix",
                    "reason": "time_scoped_rate_edit_requires_foreground_background_remix",
                }
            )
            continue
        if _should_rewrite_scoped_wrapper_to_direct_edit(
            user_request,
            context=context,
            nested_operation=nested_operation,
            params=params,
        ) and f"edit_data_pipeline.{nested_operation}" in available:
            _rewrite_step_to_edit_data_pipeline(
                step,
                operation=nested_operation,
                user_request=user_request,
                context=context,
                reason="unscoped edit-data-pipeline operation should not be wrapped in artifact_ops.speaker_scoped_edit",
            )
            corrections.append(
                {
                    "step_id": step.get("id"),
                    "from": "artifact_ops.speaker_scoped_edit",
                    "to": f"edit_data_pipeline.{nested_operation}",
                    "reason": "direct_edit_data_pipeline_preference",
                }
            )
            continue
        edit_op = params.get("edit_op") or params.get("edit_operation")
        if _is_nested_edit_data_pipeline_op(edit_op):
            continue
        params["edit_op"] = f"edit_data_pipeline.{nested_operation}"
        edit_params = params.get("edit_op_params")
        if not isinstance(edit_params, dict):
            edit_params = {}
            params["edit_op_params"] = edit_params
        edit_params.setdefault("instruction", user_request)
        if context.get("source_text"):
            edit_params.setdefault("source_text", context.get("source_text"))
        params.setdefault("selection_instruction", _speaker_scope_hint(user_request))
        corrections.append(
            {
                "step_id": step.get("id"),
                "operation": "artifact_ops.speaker_scoped_edit",
                "nested_edit_op": params["edit_op"],
                "reason": "speaker_scoped_wrapper_requires_nested_edit_data_pipeline",
            }
        )


def _rewrite_step_to_edit_data_pipeline(
    step: dict[str, Any],
    *,
    operation: str,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    step["operation"] = f"edit_data_pipeline.{operation}"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    nested_params = original_params.get("edit_op_params")
    if not isinstance(nested_params, dict):
        nested_params = original_params.get("edit_params")
    params = dict(nested_params) if isinstance(nested_params, dict) else {}
    filtered_parent_params = {
        key: value
        for key, value in original_params.items()
        if key
        not in {
            "mode",
            "ref_audio_path",
            "reference_audio_path",
            "speaker",
            "selector",
            "selection_instruction",
            "edit_op",
            "edit_operation",
            "edit_params",
            "edit_op_params",
            "target_events",
            "instruct",
            "voice_prompt",
        }
    }
    params.update(filtered_parent_params)
    if isinstance(nested_params, dict) and nested_params.get("instruction"):
        params["instruction"] = user_request
    else:
        params.setdefault("instruction", user_request)
    if context.get("source_text"):
        params.setdefault("source_text", context.get("source_text"))
    step["params"] = params


def _rewrite_step_to_foreground_background_remix(
    step: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    original_operation = str(step.get("operation") or "")
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    nested_params = dict(original_params)
    nested_params.setdefault("instruction", user_request)
    if context.get("source_text"):
        nested_params.setdefault("source_text", context.get("source_text"))
    _repair_normal_speech_emotion_params(nested_params, user_request=user_request, context=context)
    step["operation"] = "foreground_background_remix"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "edit_op": original_operation,
        "edit_op_params": nested_params,
        "foreground_stem_policy": "speech",
        "background_stem_policy": "background",
        "foreground_action": "edit_foreground",
        "preserve_duration": True,
    }


def _rewrite_scoped_step_to_foreground_background_remix(
    step: dict[str, Any],
    *,
    nested_operation: str,
    user_request: str,
    context: dict[str, Any],
    reason: str,
) -> None:
    original_params = step.get("params") if isinstance(step.get("params"), dict) else {}
    edit_params = original_params.get("edit_op_params")
    if not isinstance(edit_params, dict):
        edit_params = original_params.get("edit_params")
    nested_params = dict(edit_params) if isinstance(edit_params, dict) else {}
    nested_params.setdefault("instruction", user_request)
    if context.get("source_text"):
        nested_params.setdefault("source_text", context.get("source_text"))
    step["operation"] = "foreground_background_remix"
    step["description"] = step.get("description") or reason
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    inputs.setdefault("input_audio", "current")
    step["inputs"] = inputs
    step["params"] = {
        "instruction": user_request,
        "edit_op": f"edit_data_pipeline.{nested_operation}",
        "edit_op_params": nested_params,
        "foreground_stem_policy": "speech",
        "background_stem_policy": "background",
        "foreground_action": "edit_foreground",
        "preserve_duration": True,
    }


def _repair_normal_speech_emotion_params(
    params: dict[str, Any],
    *,
    user_request: str,
    context: dict[str, Any],
) -> None:
    if not _requests_crying_to_normal_speech(user_request):
        return
    source_text = params.get("source_text") or context.get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        return
    params["instruction_st"] = f'<emo desc="normal speaking voice">{source_text}</emo>'
    params["emotion_condition_mode"] = "desc"


def _rewrite_step_to_unsupported(step: dict[str, Any], reason: str, available: set[str]) -> None:
    step["operation"] = "unsupported" if "unsupported" in available else str(step.get("operation") or "unsupported")
    step["params"] = {"reason": reason}
    step["description"] = reason


def _is_existing_audio_text_replacement(user_request: str, context: dict[str, Any]) -> bool:
    text = user_request.lower()
    has_source_audio = bool(context.get("input_audio") or context.get("input_audios"))
    if not has_source_audio:
        return False
    if _requests_question_to_declarative_tone(text):
        return False
    if _text_replacement_signal(text):
        return True
    quoted_count = len(re.findall(r"“[^”]+”|'[^']+'|\"[^\"]+\"", user_request))
    return quoted_count >= 2 and _contains_any(text, ["replace", "change", "modify", "swap", "switch", "exchange"])


def _text_replacement_signal(text: str) -> bool:
    return _contains_any(
        text,
        [
            "replace all",
            "replace every",
            "replace the word",
            "replace word",
            "replace the phrase",
            "replace phrase",
            "replace the sentence",
            "replace first sentence",
            "replace the first sentence",
            "replace the last sentence",
            "change all",
            "change every",
            "change the word",
            "change word",
            "modify the phrase",
            "modify phrase",
            "correct the pronunciation",
            "modify the pronunciation",
            "swap the word",
            "swap words",
            "switch the word",
            "switch words",
            "替换",
            "换成",
            "改字",
            "改文字",
            "改文本",
            "改内容",
            "改台词",
        ],
    )


def _requests_question_to_declarative_tone(text: str) -> bool:
    return _contains_any(
        text,
        [
            "questioning tone",
            "question tone",
            "questioning intonation",
            "interrogative tone",
            "interrogative intonation",
            "rising intonation",
            "rising tone",
        ],
    ) and _contains_any(
        text,
        [
            "declarative tone",
            "declarative",
            "statement tone",
            "matter-of-fact",
            "falling intonation",
            "falling tone",
        ],
    )


def _scoped_edit_data_pipeline_operation(user_request: str) -> str | None:
    text = user_request.lower()
    if _requests_question_to_declarative_tone(text):
        return "emotion_style_edit"
    if _text_replacement_signal(text):
        return "text_edit"
    if _contains_any(text, ["emphasis", "emphasize", "emphasise", "stress", "intonation", "pitch", "faster", "slower", "speed", "speaking rate", "speech rate", "slow down", "speed up", "语速", "音高", "强调", "重音"]):
        return "pitch_rate_edit"
    if _contains_any(text, ["emotion", "emotional", "expressive", "tone", "style", "mood", "happy", "sad", "angry", "calm", "cheerful", "depressed", "melancholic", "hushed", "reflective", "情绪", "语气", "风格", "开心", "悲伤", "难过", "生气", "平静"]):
        if _contains_any(text, ["rap", "sing", "singing", "song", "accent", "dialect", "telephone", "phone call", "radio", "underwater", "wind"]):
            return None
        return "emotion_style_edit"
    if _contains_any(text, ["pause", "停顿"]):
        return "pause_edit"
    if _contains_any(text, ["volume", "loudness", "louder", "quieter", "音量", "响度"]):
        return "volume_normalize"
    return None


def _request_has_speaker_or_span_scope(user_request: str) -> bool:
    text = user_request.lower()
    return _contains_any(
        text,
        [
            "speaker",
            "male",
            "female",
            "man",
            "woman",
            "child",
            "kid",
            "boy",
            "girl",
            "first sentence",
            "second sentence",
            "first half",
            "second half",
            "from ",
            "until",
            "segment",
            "phrase",
            "word",
            "internal monologue",
            "internal monologues",
            "说话人",
            "男",
            "女",
            "小孩",
            "儿童",
            "片段",
            "前半",
            "后半",
        ],
    )


def _should_rewrite_scoped_wrapper_to_direct_edit(
    user_request: str,
    *,
    context: dict[str, Any],
    nested_operation: str,
    params: dict[str, Any] | None = None,
) -> bool:
    params = params if isinstance(params, dict) else {}
    text = " ".join(str(user_request or "").lower().split())
    scoped_evidence = " ".join(
        str(value)
        for value in [
            user_request,
            params.get("selection_instruction"),
            params.get("selector"),
            params.get("grounding_instruction"),
            params.get("target_description"),
        ]
        if value
    )
    if _has_phrase_or_perceptual_scope(scoped_evidence):
        return False
    if not _request_has_speaker_or_span_scope(user_request):
        return True
    if nested_operation in {"text_edit", "pitch_rate_edit", "emotion_style_edit", "pause_edit", "speech_rate"} and not _scoped_wrapper_has_source_text_for_direct_rewrite(context, params):
        return False
    if nested_operation not in {"emotion_style_edit", "pitch_rate_edit"}:
        return False
    if not _has_generic_speaker_scope(text):
        return False
    if _has_local_or_event_scope(text):
        return False
    return _context_has_single_speaker(context)


def _scoped_wrapper_has_source_text_for_direct_rewrite(context: dict[str, Any], params: dict[str, Any]) -> bool:
    if _mapping_has_source_text(context) or _mapping_has_source_text(params):
        return True
    for key in ("edit_op_params", "edit_params"):
        nested = params.get(key)
        if _mapping_has_source_text(nested):
            return True
    return False


def _mapping_has_source_text(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    source_text = value.get("source_text")
    if isinstance(source_text, str) and source_text.strip():
        return True
    perception = value.get("perception_report")
    if isinstance(perception, dict):
        perceived_text = perception.get("source_text")
        if isinstance(perceived_text, str) and perceived_text.strip():
            return True
        gemini = perception.get("gemini_perception")
        if isinstance(gemini, dict) and isinstance(gemini.get("source_text"), str) and gemini["source_text"].strip():
            return True
    return False


def _has_phrase_or_perceptual_scope(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    if _requests_question_to_declarative_tone(normalized):
        return True
    if re.search(r"[\"“][^\"”]+[\"”]|'[^']+'|‘[^’]+’|「[^」]+」|『[^』]+』", normalized):
        return True
    return _contains_any(
        normalized,
        [
            "all instances",
            "every instance",
            "each instance",
            "instances of",
            "occurrences of",
            "spoken with",
            "spoken in",
            "said with",
            "word spoken",
            "phrase spoken",
            "questioning tone",
            "questioning intonation",
            "interrogative tone",
            "interrogative intonation",
        ],
    )


def _has_generic_speaker_scope(text: str) -> bool:
    return _contains_any(
        text,
        [
            "the speaker",
            "speaker's voice",
            "speaker’s voice",
            "male speaker",
            "female speaker",
            "man's voice",
            "man’s voice",
            "woman's voice",
            "woman’s voice",
            "male voice",
            "female voice",
            "说话人",
            "男声",
            "女声",
        ],
    )


def _has_local_or_event_scope(text: str) -> bool:
    if re.search(r"[\"“][^\"”]+[\"”]|'[^']+'|‘[^’]+’|「[^」]+」|『[^』]+』", text):
        return True
    return _contains_any(
        text,
        [
            "only",
            "except",
            "first speaker",
            "second speaker",
            "first sentence",
            "second sentence",
            "last sentence",
            "final sentence",
            "first utterance",
            "last utterance",
            "first half",
            "second half",
            "from ",
            "until",
            "between",
            "after ",
            "before ",
            "seconds",
            "twelfth",
            "segment",
            "quoted phrase",
            "the phrase",
            "spoken phrase",
            "spoken word",
            "line",
            "question",
            "answer",
            "spoken by",
            "instances",
            "all instances",
            "every instance",
            "remove",
            "delete",
            "mute",
            "extract",
            "laughter",
            "laughing",
            "cough",
            "sigh",
            "applause",
            "只有",
            "仅",
            "第一个",
            "第二个",
            "最后",
            "前半",
            "后半",
            "片段",
            "词",
            "短语",
            "删除",
            "移除",
            "笑声",
        ],
    )


def _context_has_single_speaker(context: dict[str, Any]) -> bool:
    perception = context.get("perception_report")
    if not isinstance(perception, dict):
        return False
    speaker_ids = _speaker_ids_from_container(perception.get("speakers"))
    if not speaker_ids:
        gemini = perception.get("gemini_perception")
        if isinstance(gemini, dict):
            speaker_ids = _speaker_ids_from_container(gemini.get("speakers"))
    if not speaker_ids:
        speaker_ids = _speaker_ids_from_container(perception.get("speaker_segments"))
    if not speaker_ids:
        gemini = perception.get("gemini_perception")
        if isinstance(gemini, dict):
            speaker_ids = _speaker_ids_from_container(gemini.get("segments"))
    return len(speaker_ids) == 1


def _direct_edit_needs_foreground_background_remix(user_request: str, operation: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if operation == "edit_data_pipeline.emotion_style_edit" and _requests_crying_to_normal_speech(text):
        return True
    if operation == "edit_data_pipeline.emotion_style_edit" and _requests_global_emotion_foreground_edit(text):
        return True
    if operation == "edit_data_pipeline.pitch_rate_edit":
        return _has_explicit_time_scope(text) and _contains_any(
            text,
            ["slow down", "speed up", "speaking rate", "speech rate", "faster", "slower", "pace", "tempo"],
        )
    return False


def _requests_crying_to_normal_speech(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(lowered, ["crying tone", "crying voice", "crying", "sobbing", "weepy", "tearful"]) and _contains_any(
        lowered,
        ["normal speaking voice", "normal voice", "natural speaking voice", "plain speaking voice", "neutral voice"],
    )


def _requests_global_emotion_foreground_edit(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(lowered, ["entire audio", "whole audio", "full audio", "entire clip", "whole clip"]) and _contains_any(
        lowered,
        ["emotion", "tone", "mood", "sad", "happy", "angry", "calm", "extremely sad", "very sad"],
    )


def _has_explicit_time_scope(text: str) -> bool:
    return bool(
        re.search(r"\bfrom\s+(?:the\s+)?(?:\d+(?:\.\d+)?|first|second|third|twelfth)\s*(?:s|sec|secs|second|seconds)\b", text)
        or re.search(r"\b(?:starting|start)\s+from\b", text)
        or _contains_any(
            text,
            [
                "until the end",
                "to the end",
                "from the beginning",
                "first half",
                "second half",
                "latter half",
                "后半",
                "前半",
                "从",
                "到结尾",
            ],
        )
    )


def _speaker_ids_from_container(raw: Any) -> set[str]:
    if not isinstance(raw, list):
        return set()
    speaker_ids: set[str] = set()
    anonymous_count = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("speaker_id") or item.get("speaker") or item.get("label")
        if value is None:
            anonymous_count += 1
            continue
        speaker_ids.add(_normalize_speaker_id(value))
    if not speaker_ids and anonymous_count == 1:
        return {"anonymous"}
    return speaker_ids


def _normalize_speaker_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(speaker|spk|person)[_\-\s]*", "", text)
    return text or "unknown"


def _speaker_scope_hint(user_request: str) -> str:
    text = user_request.lower()
    for hint in [
        "female speaker",
        "male speaker",
        "first speaker",
        "second speaker",
        "child speaker",
        "the first sentence",
        "the second half",
        "the first half",
    ]:
        if hint in text:
            return hint
    return user_request


def _is_nested_edit_data_pipeline_op(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("operation")
    if not isinstance(value, str):
        return False
    normalized = value.removeprefix("edit_data_pipeline.")
    return normalized in {"text_edit", "emotion_style_edit", "pitch_rate_edit", "pause_edit", "volume_normalize"}


def _default_step_decision_reason(operation: str, description: str) -> str:
    if description:
        return f"Selected {operation} because its capability is: {description}"
    return f"Selected {operation} because it is the planned executable operation for this instruction step."


def _default_step_input_reason(step: dict[str, Any], context: dict[str, Any]) -> str:
    inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
    params = step.get("params") if isinstance(step.get("params"), dict) else {}
    input_audio = inputs.get("input_audio") or inputs.get("input_audio_path") or params.get("input_audio_path")
    reference_audio = inputs.get("reference_audio_path") or params.get("reference_audio_path")
    if reference_audio:
        return f"Uses {input_audio or 'the current source audio'} as content/source audio and {reference_audio} as reference audio."
    if input_audio:
        return f"Uses {input_audio} as the audio operated on by this step."
    if context.get("input_audio"):
        return f"Uses the current/default input audio from context: {context.get('input_audio')}."
    return "No explicit audio input was provided; executor will resolve the current audio artifact if possible."


def _default_instruction_alignment(step: dict[str, Any], user_request: str) -> str:
    description = str(step.get("description") or "").strip()
    if description:
        return f"The step description aligns to the request by: {description}"
    return f"This step is intended to satisfy part of the request: {user_request}"


def _default_boundary_reason(operation: str) -> str:
    if operation == "audio_effects":
        return "Audio effects are used for acoustic/channel conditions rather than text, emotion, or speaker identity edits."
    if operation == "seed_vc":
        return "SeedVC is used for voice/timbre conversion while preserving source speech content."
    if operation.startswith("artifact_ops."):
        return "Artifact operations manipulate timeline spans or audio assembly without changing speech content by themselves."
    if operation.startswith("edit_data_pipeline."):
        return "edit-data-pipeline is used only after the target speech content/span is available and the requested label contract matches the operation."
    return "Nearby tools should be used only if their capability boundary better matches the instruction."


def _default_plan_risks(steps: list[Any], *, objective_coverage: list[dict[str, Any]] | None = None) -> list[str]:
    risks: list[str] = []
    operations = [str(step.get("operation")) for step in steps if isinstance(step, dict)]
    if "audio_effects" in operations:
        risks.append("Reference-channel DSP matching is approximate and should be checked against the acoustic reference.")
    if "seed_vc" in operations:
        risks.append("Voice conversion must verify that source/content audio and reference/timbre audio are not swapped.")
    if "artifact_ops.reconstruct" in operations:
        risks.append("Local reconstruction should fit replacements to the target span unless duration change is intentional.")
    for item in objective_coverage or []:
        if item.get("status") == "uncovered":
            risks.append(f"High-level objective {item.get('objective_id')} is not explicitly covered by executable steps.")
    return risks


def _user_prompt(
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
    model_name: str,
) -> str:
    compact = _context_bool(context, "compact_planner_prompt", default=False)
    public_context = {
        key: value
        for key, value in context.items()
        if key
        in {
            "input_audio",
            "input_audios",
            "output_audio",
            "source_text",
            "target_text",
            "source_text_audio_path",
            "source_text_audio_index",
            "reference_audio_paths",
            "reference_audio_items",
            "reference_audio_aliases",
            "perception_report",
            "perception_attempts",
            "missing_inputs",
            "previous_failure",
            "unavailable_operations",
            "failed_route",
            "failed_tool_trace",
            "retry_guidance",
            "routing_policy_candidates",
            "planner_perception_guidance",
            "span_selection_analysis",
            "perception_gap_fill_guidance",
            "tool_preferences",
            "voice_design_tool",
            "edit_session_graph_context",
            "edit_session_audio_aliases",
        }
    }
    if compact:
        public_context = _compact_public_context(public_context)
    context_json = _json_for_prompt(public_context, compact=compact)
    operations_json = _json_for_prompt(
        _operations_for_prompt(operations, compact=compact, user_request=user_request, context=context),
        compact=compact,
    )
    capability_block = "" if compact else f"""
## Tool/Model Capability Comparison
{tool_model_capability_context()}
"""
    schema_block = _compact_response_schema() if compact else _verbose_response_schema()
    retry_block = ""
    if context.get("previous_failure"):
        retry_block = f"""

## Previous Failure
This is a retry/repair planning pass. Do not merely repeat the failed first-pass route.
Use the failure below as hard evidence when selecting operations and missing inputs:
{json.dumps(context.get("previous_failure"), ensure_ascii=False, indent=2)}
If Context JSON lists unavailable_operations, do not select those operations in this repair pass unless new evidence proves the listed runtime is now available.
If previous_failure.stage is "tool_contract", the failed_call entries were rejected before tool invocation because their inputs did not satisfy the selected tool contract. Do not repeat the same operation and params unless every listed tool_requirement is satisfied; prefer the listed suggested_operations when they better match the user request.
"""
    return f"""## Planner Model
{model_name}

## User Request
{user_request}

## Context JSON
{context_json}
{retry_block}

## Available Operations JSON
{operations_json}
{capability_block}

Return only one JSON object matching this schema:
{schema_block}
"""


def _verbose_response_schema() -> str:
    return """
{{
  "request": "original user request",
  "input_audio": "input path or null",
  "output_audio": "output path or null",
  "steps": [
    {{
      "id": "step_1",
      "operation": "one operation id from Available Operations JSON",
      "description": "short natural-language step description",
      "inputs": {{}},
      "params": {{}},
      "expected_output": "optional output description or path",
      "depends_on": [],
      "status": "pending",
      "result": null,
      "error": null,
      "required": true,
      "continue_on_error": null
    }}
  ],
  "global_params": {{}},
    "metadata": {{
    "high_level_objectives": [
      {{
        "id": "obj_1",
        "objective": "one atomic required goal from the user request",
        "type": "span_grounding|style_edit|content_edit|event_removal|generation|reference_match|preservation|verification|unsupported",
        "source_instruction_fragment": "the exact phrase in the request that created this objective",
        "required": true
      }}
    ],
    "objective_coverage": [
      {{
        "objective_id": "obj_1",
        "covered_by_steps": ["step_1"],
        "status": "planned|covered_by_constraint|unsupported|uncovered",
        "coverage_reason": "how the listed executable step params satisfy this objective"
      }}
    ],
    "step_rationales": [
      {{
        "step_id": "step_1",
        "operation": "same operation id as the step",
        "objective_ids": ["obj_1"],
        "decision_reason": "why this tool/operation is the best match for the user's instruction",
        "input_reason": "why these source/reference audio inputs are assigned this way",
        "instruction_alignment": "which exact part of the user instruction this step satisfies",
        "boundary_reason": "why nearby alternative tools were not used"
      }}
    ],
    "operation_graph_plan": {{
      "nodes": [
        {{
          "node_id": "op_node:step_1:output_audio_path",
          "role": "target_clip|reference_clip|edited_segment|final_mix",
          "step_id": "step_1",
          "result_key": "expected result key such as output_audio_path",
          "parent_node_ids": ["op_node:input"],
          "reason": "why this useful audio is kept",
          "objective_ids": ["obj_1"]
        }}
      ],
      "operations": [
        {{
          "step_id": "step_1",
          "input_node_ids": ["op_node:input"],
          "output_node_ids": ["op_node:step_1:output_audio_path"]
        }}
      ],
      "input_node_ids": ["op_node:input"],
      "output_node_ids": ["op_node:step_1:output_audio_path"]
    }},
    "plan_self_check": {{
      "consistent": true,
      "checked_items": [
        "Each step directly satisfies a phrase in the instruction.",
        "Source/content audio and reference audio are not swapped.",
        "Context/reference spans are not accidentally edited.",
        "The final step preserves requested content, speaker, timing, and acoustic constraints."
      ],
      "corrections_applied": [],
      "remaining_risks": []
    }}
  }},
  "missing_inputs": []
}}
"""


def _compact_response_schema() -> str:
    return """{
  "request": "...",
  "input_audio": "path-or-null",
  "output_audio": "path-or-null",
  "steps": [
    {
      "id": "step_1",
      "operation": "one Available Operations id",
      "description": "short",
      "inputs": {},
      "params": {},
      "expected_output": null,
      "depends_on": [],
      "status": "pending",
      "result": null,
      "error": null,
      "required": true,
      "continue_on_error": null
    }
  ],
  "global_params": {},
  "metadata": {
    "high_level_objectives": [{"id": "obj_1", "objective": "...", "type": "...", "source_instruction_fragment": "...", "required": true}],
    "objective_coverage": [{"objective_id": "obj_1", "covered_by_steps": ["step_1"], "status": "planned|covered_by_constraint|unsupported|uncovered", "coverage_reason": "..."}],
    "step_rationales": [{"step_id": "step_1", "operation": "...", "objective_ids": ["obj_1"], "decision_reason": "...", "input_reason": "...", "instruction_alignment": "...", "boundary_reason": "..."}],
    "operation_graph_plan": {
      "nodes": [{"node_id": "op_node:step_1:output_audio_path", "role": "final_mix", "step_id": "step_1", "result_key": "output_audio_path", "parent_node_ids": ["op_node:input"], "reason": "...", "objective_ids": ["obj_1"]}],
      "operations": [{"step_id": "step_1", "input_node_ids": ["op_node:input"], "output_node_ids": ["op_node:step_1:output_audio_path"]}],
      "input_node_ids": ["op_node:input"],
      "output_node_ids": ["op_node:step_1:output_audio_path"]
    },
    "plan_self_check": {"consistent": true, "checked_items": [], "corrections_applied": [], "remaining_risks": []}
  },
  "missing_inputs": []
}"""


def _json_for_prompt(value: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, indent=2)


def _compact_public_context(context: dict[str, Any]) -> dict[str, Any]:
    compact = dict(context)
    perception = compact.get("perception_report")
    if isinstance(perception, dict):
        compact["perception_report"] = _compact_perception_report(perception)
    attempts = compact.get("perception_attempts")
    if isinstance(attempts, list) and len(attempts) > 3:
        compact["perception_attempts"] = attempts[-3:]
    routing = compact.get("routing_policy_candidates")
    if isinstance(routing, list):
        compact["routing_policy_candidates"] = _compact_routing_policy_candidates(routing)
    session_graph = compact.get("edit_session_graph_context")
    if isinstance(session_graph, dict):
        compact["edit_session_graph_context"] = _compact_session_graph_context(session_graph)
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_perception_report(perception: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "audio_path",
        "source_text",
        "source_text_audio_path",
        "source_text_audio_index",
        "language",
    ):
        if perception.get(key) not in (None, "", [], {}):
            summary[key] = perception.get(key)
    asr = perception.get("asr")
    if isinstance(asr, dict):
        summary["asr"] = {
            key: asr.get(key)
            for key in ("text", "language")
            if asr.get(key) not in (None, "", [], {})
        }
    speaker_segments = _compact_sequence(perception.get("speaker_segments"), max_items=24)
    if speaker_segments:
        summary["speaker_segments"] = speaker_segments
    word_timestamps = _compact_word_timestamps(perception.get("word_timestamps"), max_items=24)
    if word_timestamps:
        summary["word_timestamps_sample"] = word_timestamps
        if isinstance(perception.get("word_timestamps"), list):
            summary["word_timestamps_total"] = len(perception["word_timestamps"])
    errors = _compact_sequence(perception.get("errors"), max_items=5)
    if errors:
        summary["errors"] = errors
    fallback = perception.get("perception_fallback")
    if isinstance(fallback, dict):
        summary["perception_fallback"] = {
            key: fallback.get(key)
            for key in ("used", "reason", "backend")
            if fallback.get(key) not in (None, "", [], {})
        }
    return summary


def _compact_word_timestamps(value: Any, *, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    if len(value) <= max_items:
        sample = value
    else:
        head_count = max_items // 2
        tail_count = max_items - head_count
        sample = [*value[:head_count], {"omitted_middle_word_count": len(value) - max_items}, *value[-tail_count:]]
    for item in sample:
        if not isinstance(item, dict):
            compact.append(item)
            continue
        if "omitted_middle_word_count" in item:
            compact.append(item)
            continue
        compact.append(
            {
                key: item.get(key)
                for key in ("text", "start_sec", "end_sec", "speaker_id")
                if item.get(key) not in (None, "", [], {})
            }
        )
    return compact


def _compact_sequence(value: Any, *, max_items: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:max_items]


def _compact_session_graph_context(graph: dict[str, Any]) -> dict[str, Any]:
    keep = {
        key: graph.get(key)
        for key in (
            "session_id",
            "current_node_id",
            "current_audio_path",
            "aliases",
            "current_aliases",
            "reference_audio_items",
        )
        if graph.get(key) not in (None, "", [], {})
    }
    normalized_refs = graph.get("normalized_refs")
    if isinstance(normalized_refs, dict):
        keep["normalized_refs"] = _compact_normalized_refs(normalized_refs)
    rounds = graph.get("rounds") or graph.get("round_history")
    if isinstance(rounds, list) and rounds:
        keep["recent_rounds"] = rounds[-3:]
        keep["round_count"] = len(rounds)
    return keep


def _compact_normalized_refs(refs: dict[str, Any]) -> dict[str, Any]:
    keep: dict[str, Any] = {
        key: refs.get(key)
        for key in ("schema_version", "session_id", "current_node_id", "current_audio_ref_id", "alias_count")
        if refs.get(key) not in (None, "", [], {})
    }
    aliases = []
    for item in refs.get("aliases") or []:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias") or "")
        if alias in {"session_current", "current_session_audio", "session_initial", "session_original"} or alias.startswith("session_round:") or alias.endswith("_result"):
            aliases.append({k: item.get(k) for k in ("alias", "artifact_ref_id", "path") if item.get(k) not in (None, "", [], {})})
    if aliases:
        keep["aliases"] = aliases[:8]
    artifact_refs = []
    for item in refs.get("artifact_refs") or []:
        if isinstance(item, dict):
            artifact_refs.append(
                {
                    key: item.get(key)
                    for key in ("ref_id", "path", "role", "kind", "node_id", "round_index", "step_id", "operation")
                    if item.get(key) not in (None, "", [], {})
                }
            )
    if artifact_refs:
        keep["artifact_refs"] = artifact_refs[:6]
    span_refs = []
    for item in refs.get("span_refs") or []:
        if isinstance(item, dict):
            span_refs.append(
                {
                    key: item.get(key)
                    for key in ("span_id", "audio_ref_id", "start_sec", "end_sec", "speaker_id", "text", "role")
                    if item.get(key) not in (None, "", [], {})
                }
            )
    if span_refs:
        keep["span_refs"] = span_refs[:8]
    validation = refs.get("validation")
    if isinstance(validation, dict) and validation.get("valid") is not None:
        keep["validation"] = {"valid": validation.get("valid")}
    return keep


def _compact_routing_policy_candidates(candidates: list[Any], *, max_items: int = 2) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in candidates[:max_items]:
        if not isinstance(item, dict):
            continue
        record: dict[str, Any] = {}
        for key in ("policy_id", "priority", "score", "selected_tools", "required_evidence"):
            if item.get(key) not in (None, "", [], {}):
                record[key] = item.get(key)
        if item.get("rationale"):
            record["rationale"] = _compact_text(item.get("rationale"), max_chars=180)
        workflow = item.get("workflow")
        if isinstance(workflow, list) and workflow:
            record["workflow"] = [_compact_text(value, max_chars=100) for value in workflow[:3]]
        guardrails = item.get("guardrails")
        if isinstance(guardrails, list) and guardrails:
            record["guardrails"] = [_compact_text(value, max_chars=110) for value in guardrails[:3]]
        matched = item.get("matched_signals")
        if matched:
            record["matched_signals"] = matched
        if record:
            compact.append(record)
    return compact


def _operations_for_prompt(
    operations: list[dict[str, Any]],
    *,
    compact: bool,
    user_request: str = "",
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not compact:
        return operations
    allowed = _compact_operation_allowlist(user_request, context or {}, operations)
    compact_ops: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        op_id = operation.get("operation") or operation.get("tool_id")
        if not op_id:
            continue
        if str(op_id) not in allowed:
            continue
        item: dict[str, Any] = {
            "operation": str(op_id),
            "description": _compact_text(operation.get("description"), max_chars=260),
        }
        required_inputs = operation.get("required_inputs")
        if required_inputs:
            item["required_inputs"] = required_inputs
        tool_operation = operation.get("tool_operation")
        if tool_operation:
            item["tool_operation"] = tool_operation
        default_params = operation.get("default_params")
        if isinstance(default_params, dict) and default_params:
            item["default_params"] = default_params
        compact_ops.append(item)
    return compact_ops


def _compact_operation_allowlist(
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
) -> set[str]:
    available = {
        str(item.get("operation") or item.get("tool_id"))
        for item in operations
        if isinstance(item, dict) and (item.get("operation") or item.get("tool_id"))
    }
    has_grounded_speech_perception = _context_has_grounded_speech_perception(context)
    selected: set[str] = {
        "unsupported",
        "export_wav",
        "artifact_ops.extract_spans",
        "artifact_ops.reconstruct",
        "artifact_ops.speaker_scoped_edit",
        "edit_data_pipeline.text_edit",
    }
    if not has_grounded_speech_perception:
        selected.update({"perception.speech_context", "perception.audio_fallback", "perception.speaker_attributed_review"})
    for candidate in context.get("routing_policy_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        for tool in candidate.get("selected_tools") or []:
            tool_id = str(tool)
            if tool_id in available:
                selected.add(tool_id)
            selected.update(_operations_for_tool_alias(tool_id, available))
    text = str(user_request or "").casefold()
    if _contains_any_text(text, ["concat", "concatenate", "sequential", "before the second", "after the first", "join", "append", "prepend", "pause between", "gap between", "add the first audio"]):
        selected.update({"artifact_ops.concat", "artifact_ops.silence"})
    if _contains_any_text(text, ["extract", "isolate", "retain only", "identify", "locate", "segment", "span", "片段", "提取", "定位"]):
        selected.add("artifact_ops.extract_spans")
        if not has_grounded_speech_perception:
            selected.update({"perception.speech_context", "perception.audio_fallback", "perception.speaker_attributed_review"})
    if _contains_any_text(text, ["remove", "delete", "mute", "beep", "censor", "laughter", "noise", "去掉", "删除", "静音"]):
        selected.update({"artifact_ops.remove_spans", "artifact_ops.mute_spans", "artifact_ops.beep_spans"})
        if not has_grounded_speech_perception:
            selected.update({"perception.speech_context", "perception.audio_fallback"})
    if _contains_any_text(text, ["replace", "swap", "reorder", "switch order", "交换", "替换", "顺序"]):
        selected.update({"artifact_ops.reorder_spans", "artifact_ops.reconstruct", "edit_data_pipeline.text_edit"})
    if _contains_any_text(text, ["emotion", "angry", "sad", "happy", "surprised", "tone", "style", "prosody", "hushed", "reflective", "internal monologue", "情绪", "语气"]):
        selected.update({"edit_data_pipeline.emotion_style_edit", "edit_data_pipeline.pitch_rate_edit", "spatial_reverb", "artifact_ops.speaker_scoped_edit"})
    if _contains_any_text(text, ["emphasis", "emphasize", "emphasise", "stress", "add stress", "stress to", "stress the", "word stress", "more prominent", "pronunciation of", "重音", "重读", "强调", "着重"]):
        selected.update({"emphasis_stress_edit", "qwen3_tts", "seed_vc", "edit_data_pipeline.audio_assembly", "edit_data_pipeline.pitch_rate_edit"})
    if _contains_any_text(text, ["volume", "louder", "quieter", "lower the volume", "increase the volume", "分贝", "音量"]):
        selected.update({"edit_data_pipeline.volume_normalize", "foreground_background_remix", "artifact_ops.speaker_scoped_edit"})
    if _contains_any_text(text, ["accent", "dialect", "british", "american", "mandarin", "putonghua", "口音", "方言", "普通话"]):
        selected.update({"accent_conversion", "seed_vc", "qwen3_tts"})
    if _contains_any_text(text, ["translate", "translation", "english", "korean", "spanish", "翻译"]):
        selected.update({"seamless_expressive_s2st", "edit_data_pipeline.text_edit", "qwen3_tts"})
    if _contains_any_text(text, ["voice clone", "clone voice", "timbre", "target voice", "change voice", "convert voice", "child voice", "male voice", "female voice", "音色"]):
        selected.update({"seed_vc", "no_reference_voice_conversion", "qwen3_tts", "intra_audio_voice_transfer"})
    if _contains_any_text(text, ["sing", "rap", "唱"]):
        selected.update({"ace_step_music.timeline_edit", "artifact_ops.extract_spans", "artifact_ops.reconstruct"})
    if _contains_any_text(text, ["background", "foreground", "music", "environment", "背景"]):
        selected.update({"foreground_background_remix", "speech_separation", "perception.timestamped_caption", "perception.flexsed_events", "flexsed.detect_events"})
    if _contains_any_text(text, ["phone", "noise", "noisy", "clearer", "enhance"]):
        selected.update({"speech_enhancement", "perception.timestamped_caption", "perception.flexsed_events"})
    if _contains_any_text(text, ["reverb", "echo", "room sound", "混响"]):
        selected.update({"spatial_reverb"})
    if _request_needs_event_caption_grounding(text):
        selected.update(
            {
                "perception.timestamped_caption",
                "perception.flexsed_events",
                "flexsed.detect_events",
                "artifact_ops.extract_spans",
                "artifact_ops.remove_spans",
                "artifact_ops.mute_spans",
                "artifact_ops.beep_spans",
            }
        )
    selected = {item for item in selected if item in available}
    if has_grounded_speech_perception:
        selected = {item for item in selected if not _is_speech_perception_operation(item)}
    if len(selected) < 10:
        selected.update(
            op
            for op in _CORE_COMPACT_OPERATIONS
            if op in available and (not _is_speech_perception_operation(op) or not has_grounded_speech_perception)
        )
    return selected


_CORE_COMPACT_OPERATIONS = (
    "artifact_ops.extract_spans",
    "artifact_ops.reconstruct",
    "artifact_ops.speaker_scoped_edit",
    "edit_data_pipeline.text_edit",
    "perception.speech_context",
    "perception.audio_fallback",
    "perception.speaker_attributed_review",
    "unsupported",
    "export_wav",
)


_EVENT_CAPTION_GROUNDING_TERMS = (
    "audio event",
    "sound event",
    "caption",
    "non-verbal",
    "nonverbal",
    "non-speech",
    "non speech",
    "vocalization",
    "vocalisation",
    "vocal detection",
    "vocal event",
    "paralinguistic",
    "laughter",
    "laughing",
    "laugh",
    "giggle",
    "chuckle",
    "whisper",
    "whispered",
    "whispering",
    "soft singing",
    "singing voice",
    "singing",
    "humming",
    "hum",
    "rap",
    "cry",
    "crying",
    "sob",
    "scream",
    "shout",
    "cough",
    "sneeze",
    "breathing",
    "breath",
    "applause",
    "clapping",
    "footstep",
    "footsteps",
    "siren",
    "phone ring",
    "ringtone",
    "wind",
    "rain",
    "water",
    "background noise",
    "background music",
    "environment",
    "room tone",
    "reverb",
    "echo",
    "underwater",
    "笑声",
    "耳语",
    "低语",
    "歌声",
    "哼唱",
    "哭声",
    "尖叫",
    "咳嗽",
    "呼吸",
    "掌声",
    "脚步",
    "警报",
    "风声",
    "雨声",
    "背景音",
    "环境音",
)


def _request_needs_event_caption_grounding(text: str) -> bool:
    return _contains_any_text(text, list(_EVENT_CAPTION_GROUNDING_TERMS))


def _context_has_grounded_speech_perception(context: dict[str, Any]) -> bool:
    if context.get("source_text"):
        return True
    perception = context.get("perception_report") if isinstance(context.get("perception_report"), dict) else {}
    if perception.get("source_text"):
        return True
    word_timestamps = perception.get("word_timestamps")
    if isinstance(word_timestamps, list) and word_timestamps:
        return True
    alignment = perception.get("alignment") if isinstance(perception.get("alignment"), dict) else {}
    alignments = alignment.get("alignments")
    return isinstance(alignments, list) and bool(alignments)


def _is_speech_perception_operation(operation: str) -> bool:
    return operation in {
        "perception.speech_context",
        "perception.audio_fallback",
        "perception.speaker_attributed_review",
    }


def _operations_for_tool_alias(tool_id: str, available: set[str]) -> set[str]:
    if tool_id in {"artifact_ops"}:
        return {item for item in available if item.startswith("artifact_ops.")}
    if tool_id in {"edit_data_pipeline"}:
        return {item for item in available if item.startswith("edit_data_pipeline.")}
    if tool_id in {"perception"}:
        return {item for item in available if item.startswith("perception.")}
    return set()


def _contains_any_text(text: str, needles: list[str]) -> bool:
    return any(needle.casefold() in text for needle in needles)


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _context_bool(context: dict[str, Any], key: str, *, default: bool = False) -> bool:
    if key not in context:
        return default
    value = context.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


_SYSTEM_PROMPT = """You are an audio editing task planner for X-Editor.

Rules:
- Output ONLY a valid JSON object. Do not use markdown fences or explanatory text.
- Keep JSON compact. Use short descriptions, omit verbose expected_output details, and never copy Context JSON fields into params.
- You only plan. You must not execute audio processing.
- Use only operation ids listed in Available Operations JSON.
- Do not invent processors, effects, tools, or hidden capabilities.
- Every step must have operation, params, inputs, depends_on, and status="pending".
- Each params object must include only executable tool parameters, not full perception_report, word_timestamps, source_text transcripts, or operation manuals.
- Use "current" as the input audio for a step that consumes the previous audio artifact.
- If Context JSON includes reference_audio_items, treat audio_id as a stable user-facing alias. When selecting a reference, copy that item's concrete path into reference_audio_path and keep its audio_id, note, or time_range in params when useful for traceability.
- If Context JSON includes edit_session_graph_context, treat it as the authoritative conversation state for multi-round and multi-audio edits. The default editable audio is alias "session_current", not the original source path, unless the user explicitly targets another audio/node. Use session audio aliases such as "session_node:<node_id>" and "session_round:<round_index>:result" instead of guessing from filenames.
- For undo, revert, transform back, reverse, or restore requests, select the intended graph ancestor from edit_session_graph_context. Set metadata.session_graph.rollback_to_node_id and metadata.session_graph.selected_source_node_id, then export or operate on that selected session alias. If a rollback would also remove later edits that should be preserved, plan from the proper ancestor and replay the still-needed operations as new steps; record replayed_edge_ids in metadata.session_graph when possible.
- Do not concatenate every span or every audio listed in the session graph. Context/reference spans are evidence only; only selected edit-result audio should become the current output node.
- If the user request lacks required inputs, fill missing_inputs and avoid pretending the step is executable.
- If a request exceeds available capabilities, emit one required step with operation="unsupported" and a clear reason in params.reason.
- Prefer conservative plans that map directly to the listed X-Editor tool catalog capabilities.
- Before selecting executable steps, decompose the user request into metadata.high_level_objectives. Include every atomic goal:
  target span grounding, content/text changes, style/emotion/prosody changes, event removal/muting, acoustic/effect
  matching, reference extraction/matching, generated speech, and preservation constraints such as "keep original voice",
  "keep text unchanged", "preserve timing", or "preserve background". Do not hide preservation constraints inside prose.
- For multi-hop or compound instructions joined by "and", "then", "while", "also", commas, or multiple clauses, each
  objective must appear in metadata.high_level_objectives before step planning. Plan steps from those objectives, then
  fill metadata.objective_coverage to show which executable step params satisfy each objective.
- artifact_ops.extract_spans exports only the selected spans. When the request says to keep/extract only a target and remove everything else, that single step covers both the target-selection objective and the complementary removal objective; list the same step under both objective ids.
- Do not claim an objective is covered merely because a step description mentions it. The step operation and params must
  carry the executable contract for that objective. For example, a request to "make the female speaker angry and remove
  any laughter" must keep both an affective edit objective and a laughter-removal/event-cleanup objective. If using
  artifact_ops.speaker_scoped_edit for this compound request, params.instruction or params.target_events must retain the
  laughter-removal requirement; do not reduce it to only "change emotion to angry".
- If one operation intentionally covers several objectives, list all objective_ids in that step's metadata.step_rationales
  entry and explain the exact executable params that cover each objective. If no listed operation can cover an objective,
  mark it unsupported or uncovered in metadata.objective_coverage rather than silently dropping it.
- Fill metadata.operation_graph_plan as a minimal round-local pseudo DAG. Nodes must be useful audio artifacts/spans
  only, never perception/verifier/scoring/process reports. For each produced useful node include node_id, role,
  parent_node_ids, expected producer step_id/result_key, reason, and objective_ids. Do not reference nodes or steps from
  another dialogue round. List only split outputs that later steps consume; unused debug stems/spans stay out of the DAG.
- If Context JSON includes routing_policy_candidates, treat them as advisory planner skills: use their workflow,
  required_evidence, rationale, and guardrails to choose among Available Operations, but do not bypass operation
  boundaries or fabricate missing evidence.
- If Context JSON includes planner_perception_guidance, use its audio_captions, perception_plan, missing_information,
  and next_planner_instruction as planner-side audio triage evidence. If it says perception is required, plan that
  perception operation before selecting local edit/reference spans unless an equivalent perception_report already
  satisfies the same evidence need.
- If Context JSON includes span_selection_analysis, use selected_spans and span_decisions to decide which spans are
  target edit spans, reference spans, context-only spans, or excluded spans. Do not convert excluded/context spans
  into edit targets unless the user instruction explicitly requires them.
- When passing explicit spans to artifact_ops, include only spans that will actually be edited, extracted, removed,
  muted, beeped, or replaced. preserve_context, exclude, needs_more_perception, and reference_span decisions are
  planner evidence only and must not appear in selected_spans, spans, edits, or reconstruct replacement ranges.
- If Context JSON includes tool_preferences.no_reference_voice_design, honor it for no-reference voice design when
  the requested operation is listed. Use qwen3_tts custom_voice for "qwen3_tts". Reject unavailable tool preferences rather than silently substituting a model. Do not use the preferred tool for reference voice conversion, accent conversion, cleanup,
  or other unrelated requests.
- Audio-grounded planning authority:
  - The planner may use provided perception_report as factual evidence for transcript, speaker profiles, role hints, phrase locations, and approximate time spans.
  - If an existing-source lexical edit only needs the source transcript, first plan operation="perception.speech_context" when it is listed. It uses lightweight Qwen3ASR by default and does not need Gemini audio perception unless later evidence proves ASR/alignment insufficient.
  - If the user asks for speaker/role/person/phrase/semantic extraction or local timeline editing and the provided perception_report is missing, sparse, or lacks speaker/segment descriptions, first plan operation="perception.speaker_attributed_review" when the target is multi-speaker speech, speaker order, speaker switches, overlap, a speaker's question/answer, or a speaker-scoped utterance and that operation is listed. Otherwise plan operation="perception.audio_fallback" when it is listed. The speaker-attributed review path collects Qwen3ASR wording, DiariZen turns, optional SE-DiCoW hints, and configured reviewer verification before downstream span selection.
  - Boundary between speech/speaker grounding and audio event/caption grounding: if the target span is selected by transcript content, quoted words, a sentence/question/answer, speaker identity/order/role/gender/age, or speaker-scoped utterances, first plan perception.speaker_attributed_review when listed. Do not use FlexSED, timestamped captions, or caption labels such as speech/male speech/female speech/question/overlap as the sole source of exact speech or speaker boundaries.
  - Use perception.timestamped_caption or perception.flexsed_events for general sound, environmental audio, music, room/reverb/channel effects, animal sounds, background noise, and non-speech/paralinguistic event boundaries. Non-verbal vocal detection and vocalization targets such as laughter, whispering, humming, singing voice, breathing, coughing, sobbing, screaming, or other sound events are event/caption grounding problems, not unsupported speech-transcript problems. For simple concrete events, direct FlexSED evidence may be enough; for open-ended audio description or ambiguous boundaries, prefer timestamped_caption with FlexSED prior and slice_audio evidence.
  - If the requested span is constrained by both speech/speaker information and general sound/effect information, plan both perception.speaker_attributed_review and perception.timestamped_caption/perception.flexsed_events, then pass only the reconciled target spans to artifact_ops. Context-only spans from either perception path are evidence, not edit targets.
  - Reference clip/enrollment extraction follows the same grounding discipline as edit-span extraction. For a reference speech/speaker/voice clip, first run perception.speaker_attributed_review on the reference audio, then call artifact_ops.extract_spans with grounded selected_spans. For a reference sound-event/acoustic/context clip, first run perception.timestamped_caption, using a concrete-event prior only when useful, then extract only its grounded selected_spans. Do not use bare artifact_ops.extract_spans to localize an unknown reference clip unless concrete timestamps are already supplied.
  - If the request explicitly provides both Source text and Target text, or Context JSON provides both source_text and target_text, treat it as a transcript-grounded text edit. Plan edit_data_pipeline.text_edit directly with params.source_text and params.target_text. Do not add perception.audio_fallback, artifact_ops.extract_spans, or timestamp/diarization prerequisites unless the request asks to locate an unknown perceptual/semantic region.
  - Use operation="perception.gemini_audio" only as a compatibility alias when speech perception is not listed or the plan must explicitly force Gemini-backed perception. Use operation="perception.audio_fallback" only as a compatibility alias for heavier fallback grounding when perception.speech_context is not enough.
  - After a perception step, plan artifact_ops.extract_spans/remove_spans/mute_spans/beep_spans/speaker_scoped_edit as the executable timeline operation. Do not mark child/woman/boy/customer/Gwen/respondent/robber/victim/whisper/phrase extraction unsupported merely because no specialized detector is listed.
  - For phrase/entity/content extraction, use artifact_ops.extract_spans with the original instruction; downstream runtime can use source_text, word_timestamps, speaker_segments, and Gemini perception descriptions to locate spans.
  - For property-scoped or entity-scoped extraction/local editing, such as "speech with Japanese-accented English", "whispered parts", "noisy segments", "emotional parts", "the phrase/name Stephen Curry", or similar perceptual/semantic attributes, still plan perception.audio_fallback followed by artifact_ops. The executor can split aligned segments, ask the perception model to judge each candidate segment or subspan, and then edit/extract only the selected spans.
  - The same grounded selection pattern applies before generative local edits: for requests like "make the child speaker sing", "turn the Japanese-accented segment into standard English", or "make the respondent's answer rap", first obtain ASR/alignment/speaker/segment evidence with perception.audio_fallback when needed, then call the domain operation (ace_step_music.timeline_edit, accent_conversion, or artifact_ops.speaker_scoped_edit). Those runtime handlers can split aligned candidates, ask the perception model to confirm target spans, edit only confirmed spans, and stitch the final timeline.
  - For inserting newly generated audio at a timestamp or before/after a detected phrase/event, plan qwen3_tts (or the requested generator) followed by artifact_ops.reconstruct with params.edits=[{{"start_sec": ..., "end_sec": ..., "input_audio_path": "<generated step id>"}}]. Do not use edit_data_pipeline.audio_assembly for simple insertion unless you can provide its full service-layer target_audio_path/edit_tagged_text/time-span contract.
  - For requests to concatenate audios with a synthetic silence/pause/gap between files, do not use qwen3_tts to create silence. Use artifact_ops.concat directly with params.pause_duration_sec when it is listed, or artifact_ops.silence followed by artifact_ops.concat if a separate silence artifact is necessary.
  - artifact_ops.silence generates a standalone all-silent clip; it does not insert a pause into existing speech. For a pause inside existing/source speech, use edit_data_pipeline.pause_edit with source_text and a valid <pause act="ins" level="1|2|3"/> instruction_st. Use artifact_ops.silence only as an explicitly separate concat/reconstruct component.
  - For no-reference target voice design that changes speaker age/gender/timbre for an existing/source audio while preserving words, including "child/kid/children voice", "boy/girl voice", "male/female voice", "younger/older voice", "小孩音色", "儿童音色", "童声", "男声", "女声", "年轻声音", or "年老声音", do not use edit_data_pipeline.pitch_rate_edit and do not finish with raw qwen3_tts re-synthesis. Use operation="no_reference_voice_conversion": it must synthesize an independent target-voice reference with qwen3_tts from the requested voice description only, then convert the source audio to that reference with SeedVC. The target reference prompt/text must not copy source speaker traits or source transcript. Only use qwen3_tts alone for standalone TTS when there is no source audio to convert.
  - HARD BOUNDARY: for existing/source audio lexical edits such as replace/delete/insert/swap/correct/modify words, phrases, sentences, or numbers, plan edit_data_pipeline.text_edit. Never plan qwen3_tts for this class, even when the request says same/original voice or when text_edit may need segmentation. If text_edit cannot satisfy the contract, emit unsupported or missing_inputs with the text_edit reason rather than regenerating with TTS.
  - For profanity/cursing replacement with a beep/bleep, use artifact_ops.beep_spans with the original instruction; downstream runtime can locate profane words from word timestamps or fallback segment transcripts.
  - For role/person extraction, use artifact_ops.extract_spans or artifact_ops.speaker_scoped_edit; downstream runtime can map role/gender/age/order descriptors to speaker_id through perception.
  - For semantic local cleanup like internal monologue(s) into normal speech or removing reverb/echo from only those semantic segments, prefer one artifact_ops.speaker_scoped_edit step with params.edit_op="speech_enhancement" and a selector/selection instruction. Do not split the semantic target into multiple extract/reconstruct steps unless exact spans are provided.
  - For converting a known local utterance into an internal monologue, reflective/hushed/calm/melancholic tone, or similar style while keeping the same words, do not use edit_data_pipeline.text_edit. Use artifact_ops.extract_spans for the target utterance, edit_data_pipeline.emotion_style_edit for the local clip, spatial_reverb when reverb/room sound is requested, then artifact_ops.reconstruct with explicit replacement spans. If the perception report contains the local segment transcript, use that short local transcript as source_text for emotion_style_edit.
  - Mark unsupported only when the requested output requires a model outside the listed operations, such as source-separating a singing track from music when no separation operation can satisfy it, or judging pronunciation correctness when no evidence/operation can support the judgment.
- Tool boundary rules:
  - artifact_ops.extract_spans only localizes and exports requested timeline regions. It does not alter transcript, emotion, speaker identity, volume, or acoustic scene.
  - Use accent_conversion for accent/dialect reproduction or conversion requests. It owns accent-library lookup, SeedVC reference conversion, Qwen3-TTS voice_clone synthesis, and timeline splicing. Do not replace this with a bare qwen3_tts step unless the request is ordinary no-source TTS rather than accent/dialect conversion.
  - If the request says "accent/dialect of audio2" or "reference accent/dialect", pass the concrete reference_audio_path to accent_conversion and do not guess a named accent such as Portuguese unless perception explicitly says that label.
  - Accent/dialect translation is a composite edit, not direct cross-language S2ST. For requests like "Translate the Taiwanese language in the audio into Mandarin", "Taiwanese Mandarin to standard Mandarin", "台语/台湾腔转普通话", or dialect/regional speech to a standard variety, first use edit_data_pipeline.text_edit to convert localized/regional expressions into the target standard wording, then use accent_conversion to convert the pronunciation/accent/dialect to the target variety. Record both objectives separately. Do not route this class to seamless_expressive_s2st unless the target is a genuinely different language rather than a dialect or regional expression standardization.
  - Use ace_step_music.timeline_edit for requests that ask a speaker, line, or speech segment to sing, rap, become song-like, or become music vocals. This operation selects target spans, calls ACE-Step on each selected span, and reconstructs the original timeline; do not plan raw qwen3_tts/edit_data_pipeline emotion edits for rap/singing style.
  - Use foreground_background_remix for requests that edit/remove/isolate speech or vocals while preserving background music/sound timing.
  - For a cappella / clean vocals / singing-vocal extraction from instrumental tracks, accompaniment, music, or background noise, use foreground_background_remix directly. Do not plan TSE for this class unless a concrete external reference_audio_path is required by the user; TSE is target-speaker enrollment, not music-stem separation.
  - Do not use edit_data_pipeline directly on mixed foreground/background audio when the request says the background must stay unchanged.
  - TSE requires a concrete reference_audio_path; without reference audio, use speech_separation only when exact speaker identity is not required.
  - speech_separation returns stems but does not guarantee stable first/second/female/male speaker labels by itself.
  - speech_enhancement is for cleanup/denoise/clarity, not for changing content, emotion, speaker identity, or background scene.
  - Use audio_effects for telephone/phone-call, old-radio, underwater, strong-wind, muffled/heavy-wall, and reference acoustic-channel matching requests. Do not use SeedVC for acoustic channel/environment matching (for example "same acoustic channel condition as audio2"). SeedVC transfers speaker timbre, not recording-channel effects. If audio_effects is not listed and no listed operation can apply the requested channel/effect, emit unsupported with the missing audio-effects capability.
  - Use seamless_expressive_s2st for direct speech-to-speech translation when it is listed and the request asks to translate the spoken audio into another language. Provide target_language as a Seamless ISO-639-3 code such as eng, cmn, fra, deu, spa, jpn, kor, por, ita, rus, or tha. Do not replace this with text_edit unless the Seamless runtime is unavailable in previous_failure or the request explicitly asks for localized transcript/content editing.
  - Translation requests such as "translate Chinese utterances into English" are not accent/dialect conversion unless the user explicitly asks for a named accent or dialect. Do not select accent_conversion only because the target language is English.
  - When previous_failure says seamless_expressive_s2st is unavailable and the request has an explicit target language, do not mark the task unsupported solely because no separate text translation tool is listed. You may translate provided source_text/perception transcript as planning text and put the explicit translation in params.target_text or instruction_st. Then use edit_data_pipeline.text_edit for short localized replacement, or qwen3_tts voice_clone when the whole utterance must be regenerated.
  - For concatenate + translate requests, first concatenate the input audios, then run perception.audio_fallback on the concatenated audio to obtain the full transcript. If a full transcript is obtained, provide target_text in the requested language and synthesize/edit the concatenated speech. If no full transcript can be obtained, then emit unsupported with that exact missing evidence.
  - If the prompt mentions a second/third audio input but Context JSON provides fewer audio files, emit missing_inputs for the absent audio alias. Do not substitute another provided audio in core planner/runtime. Benchmark or dataset adapters may canonicalize known metadata repairs before planning and must record those repairs outside core runtime.
  - For pure concatenation, use artifact_ops.concat with all input audio paths in order. Never route a pure concatenate/join/merge request to edit_data_pipeline or edit_data_pipeline.audio_assembly.
  - For overlay/mix-at-time requests, use foreground_background_remix only when separation/remix is needed; otherwise use artifact_ops.reconstruct/concat-style artifact operations if explicit insert audio and placement are available. Do not route overlay to text_edit.
  - For timbre/voice replacement between speakers inside the same audio, use intra_audio_voice_transfer. For cross-file timbre transfer, identify content/source audio and timbre/reference audio from the wording before planning SeedVC. For example, "replace the timbre of voice in audio2 with that of audio1" means source/content=audio2 and reference/timbre=audio1. Do not add a no-op text_edit.
  - For emphasis/stress on quoted words or phrases, prefer operation="emphasis_stress_edit" when available. It performs P1 local resynthesis: Qwen3-TTS generates a strongly stressed full-sentence target, SeedVC converts that target back to the source speaker, and edit_data_pipeline.audio_assembly splices only the <target> phrase with F5 boundary normalization. Use edit_data_pipeline.pitch_rate_edit only as fallback when emphasis_stress_edit is unavailable, the user explicitly asks for a mechanical pitch/rate shift, or the runtime reports a P1 service failure.
  - emphasis_stress_edit requires reviewed source_text and word_timestamps. source_text or a whole-utterance speaker segment alone is not enough. When word_timestamps are absent and perception.speech_context is listed, add a perception.speech_context step with params.need_words=true before emphasis_stress_edit and make the edit depend on that perception step. Do not leave this evidence repair implicit for Executor.
  - For existing/source audio emotion, tone, style, pitch/rate, pause, or volume edits, edit_data_pipeline is the preferred executable editing chain when its label contract can express the request. Use artifact_ops.speaker_scoped_edit only as a selector/reconstruction wrapper for scoped targets; its params must include edit_op=edit_data_pipeline.emotion_style_edit, edit_data_pipeline.pitch_rate_edit, edit_data_pipeline.text_edit, edit_data_pipeline.pause_edit, or edit_data_pipeline.volume_normalize. Do not use a bare speaker_scoped_edit instruction as the style model.
  - artifact_ops.silence is a standalone silent-artifact generator, not an existing-speech pause editor. A request to add a pause before/after/between words in source audio must use edit_data_pipeline.pause_edit; silence may only be a separate component of an explicit concat/reconstruct plan.
  - edit_data_pipeline.emotion_style_edit only accepts canonical emotion tags like <emo type="happy" level="2">...</emo> or descriptive tags like <emo desc="hushed, reflective tone">...</emo>. Supported type values are happy, angry, sad, afraid, disgusted, melancholic, surprised, and calm. For "hushed/reflective/internal monologue" use melancholic or calm; do not emit literal whisper/shout/rap/singing/accent tags for this tool.
  - spatial_reverb only applies static room impulse response or echo-like room sound. It does not denoise/dereverb, create telephone/radio/underwater/wind/muffled channel effects, or change emotion/content.
  - Swap/switch/reorder words or phrases in speech (for example "swap red and yellow", "switch slash and semicolon") should be edit_data_pipeline.text_edit with source_text and target_text/instruction_st. Do not implement word swaps by extracting individual word audio spans unless exact word timestamps are already available and explicitly requested.
  - If lexical replacement targets are stated through an unambiguous semantic description, resolve each requested meaning to the concrete words that must be spoken and emit the replacements as distinct <sub targ="..."> entries in one edit_data_pipeline.text_edit step. Ground the source side in reviewed transcript evidence; do not mark the request unsupported only because the target wording was indirect.
  - For edit_data_pipeline.text_edit, use a small number of coarse local tags. A replacement MUST be <sub targ="new phrase">old phrase</sub>; use <del> only for pure deletion and <ins> only for pure insertion. Keep existing valid localized tags, and only repair broad substitutions that wrap unchanged context. Do not use one full-utterance <sub> shortcut unless the whole utterance truly changes, and do not fragment a long sentence into character-level tags.
  - edit_data_pipeline label contracts are operation-specific: text_edit supports only <sub>/<ins>/<del>; pitch_rate_edit supports only <pitch, semitones=N> and <rate, factor=F>; emotion_style_edit supports only <emo type="emotion" level="1|2|3">...</emo> or <emo desc="...">...</emo>; pause_edit supports only <pause act="ins" level="1|2|3"/>. audio_assembly uses edit_tagged_text with <target> spans, not instruction_st. speech_rate, volume_normalize, pitch_shift, and boundary_smooth are parameter-only.
  - The outer operation id must match the intended edit-data-pipeline contract. Do not emit operation="edit_data_pipeline.text_edit" with params.operation="pitch_rate_edit", "emotion_style_edit", "pause_edit", "volume_normalize", "pitch_shift", "speech_rate", or any other child operation. Use operation="edit_data_pipeline.<child_operation>" directly.
  - Speaker-specific volume changes should use artifact_ops.speaker_scoped_edit with edit_op=volume_normalize. If foreground_background_remix is used because separation/remix is necessary, its edit_op may be artifact_ops.speaker_scoped_edit; do not send that nested artifact operation to edit_data_pipeline.
  - Requests to generate a target sentence while cloning speaker/pronunciation references are qwen3_tts voice_clone tasks, not accent_conversion. If the request text mentions an audio2/input2 reference but runtime input_audios has only one item, emit missing_inputs for that absent reference instead of using the remaining audio as a substitute.
  - Requests to make same-language speech fluent/native/natural (for example unnatural word-by-word Chinese into fluent native Chinese) are text_edit/speaker_scoped_edit tasks, not accent_conversion. "native Chinese" in this context means fluent wording, not a named accent library target.
  - text_edit input must be a <30s segment or have word/speaker segmentation available for executor-side chunking.
"""


_SYSTEM_PROMPT = f"{_SYSTEM_PROMPT}\n\n{PROMPT3_PLANNER_GAP_FILLING}"


_COMPACT_SYSTEM_PROMPT = """You are an X-Editor audio-edit planner.

Return only one valid JSON object matching the requested schema. No markdown, preamble, or prose.

Planning contract:
- Use only operation ids from Available Operations JSON.
- You only plan; never claim audio was processed.
- Every step needs id, operation, inputs, params, depends_on, status="pending", required=true.
- Put executable tool arguments only in params. Do not copy full perception reports, manuals, word timestamps, or debug blobs.
- Use "current" when a step consumes the previous audio artifact.
- Decompose the user request into metadata.high_level_objectives before selecting steps. Preserve constraints such as original voice, text, timing, reference role, background, and acoustic conditions as explicit objectives.
- Fill metadata.objective_coverage from executable step params, not from prose. Mark unsupported/uncovered objectives explicitly.
- execution_semantics.bindings are required only for artifact_ops.extract_spans/remove_spans/mute_spans/beep_spans steps whose local boundaries come from reviewed Perception evidence. Do not emit semantic bindings for accent, emotion, generation, translation, concat, or other steps whose executable parameters come from the user request or deterministic context.
- Fill metadata.step_rationales with short reasons tied to objective ids and instruction fragments.
- Fill metadata.operation_graph_plan as a minimal round-local useful-audio DAG. Nodes are audio artifacts/spans only, never perception/verifier/scoring/process reports. Include only consumed split outputs. Do not reference another dialogue round.
- For multi-round session context, default editable audio is session_current; use session aliases instead of guessing filenames.
- Treat routing_policy_candidates and planner_perception_guidance as advisory evidence, but never bypass operation boundaries.
- If only source_text is missing for simple existing-source lexical edits, plan perception.speech_context when listed; it uses lightweight Qwen3ASR by default. If speech/speaker/phrase spans are unknown and Context JSON lacks source_text/word_timestamps/alignment, plan perception.speaker_attributed_review or perception.audio_fallback before artifact_ops span editing/extraction. If Context JSON already contains grounded speech perception, do not add another speech perception step; use artifact_ops span selection against the existing evidence. If sound-event/acoustic/non-verbal vocal/paralinguistic boundaries such as laughter, whispering, humming, singing voice, breathing, coughing, sobbing, screaming, applause, noise, music, or room tone are unknown, plan perception.timestamped_caption or perception.flexsed_events first when listed; do not mark non-verbal vocal detection or sound event detection unsupported before trying the event/caption-aware path.
- Existing/source audio lexical edits use edit_data_pipeline.text_edit, not qwen3_tts. Concatenation uses artifact_ops.concat. No-reference voice conversion uses no_reference_voice_conversion. Missing capabilities should become unsupported or missing_inputs.
- Unambiguous indirect lexical targets still use text_edit: infer the concrete target words and place multiple distinct <sub targ="..."> replacements in one complete instruction_st instead of returning unsupported.
- edit_data_pipeline operation ids are contract-specific: never wrap pitch/rate, emotion/style, pause, volume, pitch_shift, or speech_rate edits inside operation="edit_data_pipeline.text_edit" by setting params.operation. Use the matching operation id directly.
- For emphasis/stress on quoted words or phrases, use emphasis_stress_edit when it is listed; use edit_data_pipeline.pitch_rate_edit only when the P1 emphasis operation is unavailable or explicitly not desired.
- emphasis_stress_edit requires reviewed source_text and word_timestamps. If word_timestamps are absent, first add perception.speech_context with params.need_words=true and make emphasis_stress_edit depend on it; Executor will not perform hidden perception repair.
"""


def _system_prompt_for_context(context: dict[str, Any]) -> str:
    return _COMPACT_SYSTEM_PROMPT if _context_bool(context, "compact_planner_prompt", default=False) else _SYSTEM_PROMPT


def _operation_ids(operations: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        value = operation.get("operation") or operation.get("tool_id")
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def _planner_evidence_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: context[key]
        for key in _PLANNER_CONTEXT_EVIDENCE_KEYS
        if context.get(key) not in (None, "", [], {})
    }


def _deterministic_fallback_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    operation_ids = _operation_ids(operations)
    local_translation_fallback = _deterministic_local_translation_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if local_translation_fallback is not None:
        return local_translation_fallback

    questioning_tone_fallback = _deterministic_questioning_tone_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if questioning_tone_fallback is not None:
        return questioning_tone_fallback

    gender_timbre_swap_fallback = _deterministic_intra_audio_gender_timbre_swap_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if gender_timbre_swap_fallback is not None:
        return gender_timbre_swap_fallback

    reference_vc_fallback = _deterministic_cross_file_reference_voice_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if reference_vc_fallback is not None:
        return reference_vc_fallback

    reference_pacing_fallback = _deterministic_reference_pacing_transfer_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if reference_pacing_fallback is not None:
        return reference_pacing_fallback

    natural_voice_cleanup_fallback = _deterministic_natural_voice_cleanup_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if natural_voice_cleanup_fallback is not None:
        return natural_voice_cleanup_fallback

    scoped_volume_fallback = _deterministic_scoped_volume_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if scoped_volume_fallback is not None:
        return scoped_volume_fallback

    acappella_fallback = _deterministic_acappella_extraction_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if acappella_fallback is not None:
        return acappella_fallback

    spoken_voice_isolation_fallback = _deterministic_spoken_voice_isolation_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        primary_error=primary_error,
        model_name=model_name,
    )
    if spoken_voice_isolation_fallback is not None:
        return spoken_voice_isolation_fallback

    if "artifact_ops.extract_spans" not in operation_ids:
        return None
    if not _requests_plain_speech_region_extraction(user_request):
        property_extraction_fallback = _deterministic_property_speech_extraction_plan(
            user_request,
            context=context,
            operation_ids=operation_ids,
            primary_error=primary_error,
            model_name=model_name,
        )
        if property_extraction_fallback is not None:
            return property_extraction_fallback
        return None

    input_audio = context.get("input_audio")
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.extract_spans",
                description="Extract speech regions from the audio.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={"instruction": user_request},
                expected_output=output_audio or "speech_regions_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_plain_speech_region_extraction",
                "primary_error": primary_error,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _deterministic_preflight_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
    model_name: str,
) -> AudioEditPlan | None:
    if not _context_bool(context, "compact_planner_prompt", default=False):
        return None
    operation_ids = _operation_ids(operations)
    semantic_text_edit = semantic_text_edit_preflight_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        model_name=model_name,
    )
    if semantic_text_edit is not None:
        return semantic_text_edit
    pure_concat = _deterministic_pure_concat_plan(
        user_request,
        context=context,
        operation_ids=operation_ids,
        model_name=model_name,
    )
    if pure_concat is not None:
        return pure_concat
    return None


def _deterministic_pure_concat_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    model_name: str,
) -> AudioEditPlan | None:
    if "artifact_ops.concat" not in operation_ids:
        return None
    lowered = user_request.casefold()
    if not _contains_any_text(
        lowered,
        [
            "concat",
            "concatenate",
            "join",
            "append",
            "prepend",
            "sequential",
            "one after another",
            "before the second",
            "after the first",
            "add the first audio",
            "pause between",
            "gap between",
        ],
    ):
        return None
    if _contains_any_text(
        lowered,
        [
            "translate",
            "translation",
            "replace dialogue",
            "timbre",
            "voice clone",
            "accent",
            "overlay",
            "mix",
            "lower the volume",
            "convert",
            "change the",
            "transform",
        ],
    ):
        return None
    input_audios = _context_input_audio_paths(context)
    if len(input_audios) < 2:
        return None
    ordered = list(input_audios)
    if _contains_any_text(lowered, ["second before the first", "audio2 before audio1", "audio 2 before audio 1"]):
        ordered = [input_audios[1], input_audios[0], *input_audios[2:]]
    pause_duration = _pause_duration_from_request(user_request)
    params: dict[str, Any] = {
        "audio_paths": ordered,
        "instruction": user_request,
    }
    if pause_duration is not None:
        params["pause_duration_sec"] = pause_duration
    input_audio = ordered[0]
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.concat",
                description="Concatenate the requested input audios in order.",
                inputs={"input_audio_path": ordered},
                params=params,
                expected_output=output_audio or "concatenated_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "input_audios": input_audios,
        },
        metadata={
            "planner_model": model_name,
            "planner_preflight": {
                "used": True,
                "strategy": "deterministic_pure_concat",
                "reason": "High-confidence pure concat request; skipped Gemini to avoid unnecessary planner latency.",
            },
            "gemini_planner_call": {
                "status": "skipped",
                "strategy": "deterministic_preflight",
                "prompt_chars": 0,
                "payload_bytes": 0,
                "request_elapsed_sec": 0.0,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[],
    )


def _pause_duration_from_request(user_request: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:second|seconds|sec|s)\b", user_request, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value >= 0 else None


def _deterministic_questioning_tone_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "artifact_ops.speaker_scoped_edit" not in operation_ids:
        return None
    if not _requests_questioning_tone_target(user_request):
        return None
    target_word = _target_word_for_questioning_tone(user_request, context)
    if not target_word:
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    selection = f'the last word "{target_word}"' if _requests_last_word(user_request) else f'the word "{target_word}"'
    edit_instruction = f'Change "{target_word}" to a questioning tone.'
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.speaker_scoped_edit",
                description="Change the requested word to a questioning/rising tone while preserving the rest of the audio.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "selection_instruction": selection,
                    "edit_op": "edit_data_pipeline.pitch_rate_edit",
                    "edit_op_params": {
                        "instruction": edit_instruction,
                    },
                    "preserve_timeline": True,
                },
                expected_output=output_audio or "questioning_tone_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_questioning_tone_word_edit",
                "primary_error": primary_error,
                "target_word": target_word,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _requests_questioning_tone_target(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not any(
        token in text
        for token in (
            "questioning tone",
            "question tone",
            "questioning intonation",
            "interrogative tone",
            "interrogative intonation",
            "rising intonation",
            "rising tone",
        )
    ):
        return False
    if "declarative" in text or "falling intonation" in text or "statement tone" in text:
        return False
    return any(token in text for token in ("change", "make", "turn", "convert", "to", "into"))


def _requests_last_word(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return bool(re.search(r"\blast\s+word\b", text)) or "最后一个词" in text or "最后的词" in text


def _target_word_for_questioning_tone(user_request: str, context: dict[str, Any]) -> str | None:
    quoted = re.findall(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", user_request)
    for groups in quoted:
        value = next((group for group in groups if group), "").strip()
        if value:
            return value
    source_text = context.get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        perception = context.get("perception_report")
        if isinstance(perception, dict):
            source_text = str(perception.get("source_text") or "")
    words = re.findall(r"[\w'-]+|[\u4e00-\u9fff]+", str(source_text or ""))
    return words[-1] if words else None


def _deterministic_intra_audio_gender_timbre_swap_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "intra_audio_voice_transfer" not in operation_ids:
        return None
    if not _requests_intra_audio_gender_timbre_swap(user_request):
        return None
    if not _context_has_male_female_speakers(context):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="intra_audio_voice_transfer",
                description="Swap the in-audio timbres of male and female speakers while preserving word order.",
                inputs={"input_audio": input_audio} if input_audio else {},
                params={
                    "instruction": user_request,
                    "speaker_mappings": [
                        {
                            "source_speaker_id": "male speaker",
                            "reference_speaker_id": "female speaker",
                        },
                        {
                            "source_speaker_id": "female speaker",
                            "reference_speaker_id": "male speaker",
                        },
                    ],
                    "mapping_mode": "gender_timbre_swap",
                    "preserve_timeline": True,
                },
                expected_output=output_audio or "gender_timbre_swapped_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_intra_audio_gender_timbre_swap",
                "primary_error": primary_error,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _requests_intra_audio_gender_timbre_swap(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not re.search(r"\b(?:swap|switch|exchange|互换|交换)\b", text):
        return False
    has_gender_pair = (
        ("men" in text or re.search(r"\bman\b|\bmale\b", text))
        and ("women" in text or re.search(r"\bwoman\b|\bfemale\b", text))
    ) or ("男女" in text)
    if not has_gender_pair:
        return False
    return any(signal in text for signal in ("voice", "voices", "timbre", "timbres", "speaker", "speakers", "音色", "声音", "说话人"))


def _context_has_male_female_speakers(context: dict[str, Any]) -> bool:
    perception = context.get("perception_report")
    if not isinstance(perception, dict):
        return False
    evidence: list[str] = []
    for item in _perception_speaker_items(perception):
        pieces = [
            item.get("perceived_gender"),
            item.get("gender"),
            item.get("description"),
            item.get("speaker_id"),
            item.get("transcript"),
            item.get("text"),
        ]
        evidence.append(" ".join(str(piece) for piece in pieces if piece).lower())
    joined = " | ".join(evidence)
    has_female = bool(re.search(r"\b(female|woman|women|girl)\b", joined)) or "女" in joined
    has_male = bool(re.search(r"\b(male|man|men|boy)\b", joined)) or "男" in joined
    return has_female and has_male


def _perception_speaker_items(perception: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("speakers", "speaker_segments", "segments"):
        raw = perception.get(key)
        if isinstance(raw, list):
            items.extend(item for item in raw if isinstance(item, dict))
    gemini = perception.get("gemini_perception")
    if isinstance(gemini, dict):
        for key in ("speakers", "speaker_segments", "segments"):
            raw = gemini.get(key)
            if isinstance(raw, list):
                items.extend(item for item in raw if isinstance(item, dict))
    return items


def _deterministic_local_translation_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "artifact_ops.speaker_scoped_edit" not in operation_ids:
        return None
    target_language = _requested_translation_target_language(user_request)
    if not target_language:
        return None
    text = " ".join(str(user_request or "").lower().split())
    if "translate" not in text:
        return None
    if not any(signal in text for signal in ("part", "parts", "utterance", "utterances", "spoken by", "speaker", "woman", "man", "female", "male")):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.speaker_scoped_edit",
                description="Translate only the requested local speech spans while preserving the rest of the audio.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "selection_instruction": user_request,
                    "edit_op": "seamless_expressive_s2st",
                    "edit_params": {
                        "target_language": target_language,
                        "instruction": user_request,
                    },
                    "preserve_timeline": True,
                },
                expected_output=output_audio or "localized_translation_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_local_speech_translation",
                "primary_error": primary_error,
                "target_language": target_language,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _first_input_audio(context: dict[str, Any]) -> str | None:
    audios = context.get("input_audios") or context.get("input_audio_paths") or []
    if isinstance(audios, list):
        for audio in audios:
            if isinstance(audio, str) and audio:
                return audio
    return None


def _requested_translation_target_language(user_request: str) -> str | None:
    text = " ".join(str(user_request or "").lower().split())
    language_markers = [
        ("english", "English"),
        ("英语", "English"),
        ("mandarin", "Mandarin Chinese"),
        ("chinese", "Mandarin Chinese"),
        ("中文", "Mandarin Chinese"),
        ("spanish", "Spanish"),
        ("西班牙语", "Spanish"),
        ("japanese", "Japanese"),
        ("日语", "Japanese"),
        ("korean", "Korean"),
        ("韩语", "Korean"),
        ("french", "French"),
        ("法语", "French"),
        ("german", "German"),
        ("德语", "German"),
        ("italian", "Italian"),
        ("portuguese", "Portuguese"),
        ("russian", "Russian"),
    ]
    if not re.search(r"\b(?:into|to|in)\b|翻译|译成", text):
        return None
    for marker, language in language_markers:
        if marker in text:
            return language
    return None


def _deterministic_cross_file_reference_voice_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "seed_vc" not in operation_ids:
        return None
    direction = _cross_file_voice_direction(user_request)
    if direction is None:
        return None
    audio_paths = _context_input_audio_paths(context)
    if len(audio_paths) < 2:
        return None
    source_index, reference_index = direction
    if source_index >= len(audio_paths) or reference_index >= len(audio_paths):
        return None
    source_audio = audio_paths[source_index]
    reference_audio = audio_paths[reference_index]
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=context.get("input_audio") or audio_paths[0],
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="seed_vc",
                description="Convert the source content audio to the requested reference voice.",
                inputs={
                    "input_audio_path": source_audio,
                    "reference_audio_path": reference_audio,
                },
                params={
                    "reference_audio_path": reference_audio,
                    "instruction": user_request,
                },
                expected_output=output_audio or "reference_voice_converted_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_cross_file_reference_voice_conversion",
                "primary_error": primary_error,
                "source_audio_index": source_index,
                "reference_audio_index": reference_index,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[],
    )


def _deterministic_natural_voice_cleanup_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "artifact_ops.speaker_scoped_edit" not in operation_ids:
        return None
    if not _requests_electronic_to_natural_voice(user_request):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    selector = _electronic_voice_selector(user_request)
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.speaker_scoped_edit",
                description="Clean up the selected electronic/robotic voice into more natural speech while preserving timing.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "selection_instruction": selector,
                    "edit_op": "speech_enhancement",
                    "edit_op_params": {
                        "mode": "general",
                        "enhancement_goal": user_request,
                    },
                    "preserve_timeline": True,
                },
                expected_output=output_audio or "natural_voice_cleanup_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_electronic_voice_to_natural_cleanup",
                "primary_error": primary_error,
                "selection_instruction": selector,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _deterministic_reference_pacing_transfer_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "unsupported" not in operation_ids:
        return None
    if not _requests_reference_pacing_transfer(user_request):
        return None
    audio_paths = _context_input_audio_paths(context)
    if len(audio_paths) < 2:
        return None
    input_audio = context.get("input_audio") or audio_paths[0]
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="unsupported",
                description=(
                    "Dynamic reference rhythm/pacing transfer is not directly available with the listed operations."
                ),
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "reason": (
                        "dynamic reference rhythm and pacing transfer is not supported by the listed operations"
                    )
                },
                expected_output=output_audio or "unsupported_reference_pacing_transfer",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_reference_pacing_transfer_boundary",
                "primary_error": primary_error,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _requests_reference_pacing_transfer(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not text:
        return False
    has_audio1 = any(marker in text for marker in ["audio1", "audio 1", "first audio"])
    has_audio2 = any(marker in text for marker in ["audio2", "audio 2", "second audio"])
    return (
        has_audio1
        and has_audio2
        and _contains_text(text, ["rhythm", "pacing", "speaking rate", "tempo"])
        and _contains_text(text, ["keep", "preserv", "unchanged"])
        and _contains_text(text, ["words", "content", "voice"])
    )


def _requests_electronic_to_natural_voice(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    has_source = _contains_text(
        text,
        [
            "electronic voice",
            "robotic voice",
            "synthesized voice",
            "synthetic voice",
            "processed voice",
            "mechanical voice",
            "autotune",
            "auto-tune",
        ],
    )
    has_target = _contains_text(
        text,
        [
            "natural voice",
            "normal voice",
            "natural speech",
            "human voice",
            "real human speech",
            "less mechanical",
        ],
    )
    has_edit = bool(re.search(r"\b(?:change|turn|convert|transform|make|cleanup|clean up|restore)\b", text))
    return has_source and has_target and has_edit


def _electronic_voice_selector(user_request: str) -> str:
    text = " ".join(str(user_request or "").lower().split())
    if re.search(r"\b(?:woman|female|girl|her)\b", text):
        return "the woman's electronic voice"
    if re.search(r"\b(?:man|male|boy|his)\b", text):
        return "the man's electronic voice"
    if "all" in text or "voices" in text:
        return "all electronic or robotic voices"
    return "the electronic or robotic voice"


def _contains_text(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _deterministic_scoped_volume_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "artifact_ops.speaker_scoped_edit" not in operation_ids:
        return None
    if not _requests_scoped_volume_edit(user_request):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    target_db = _scoped_volume_target_db(user_request)
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.speaker_scoped_edit",
                description="Apply a localized volume edit to the requested speaker or perceptual segment.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "selection_instruction": _scoped_volume_selection_instruction(user_request),
                    "edit_op": "volume_normalize",
                    "edit_params": {
                        "target_db": target_db,
                        "instruction": user_request,
                    },
                    "preserve_timeline": True,
                },
                expected_output=output_audio or "localized_volume_edit_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_scoped_volume_edit",
                "primary_error": primary_error,
                "target_db": target_db,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _requests_scoped_volume_edit(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    has_volume = _contains_text(
        text,
        ["volume", "loudness", "louder", "quieter", "turn up", "turn down", "weaken", "normal speaking level"],
    )
    if not has_volume:
        return False
    has_scope = _contains_text(
        text,
        [
            "only",
            "speaker",
            "person",
            "voice",
            "segment",
            "part",
            "counting",
            "far away",
            "far-away",
            "distant",
            "whistle-level",
            "whistle level",
        ],
    )
    has_edit = bool(re.search(r"\b(?:increase|decrease|adjust|make|turn|raise|lower|weaken|normalize|normalise)\b", text))
    return has_scope and has_edit


def _scoped_volume_selection_instruction(user_request: str) -> str:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_text(text, ["whistle-level", "whistle level"]):
        return "the segment described as whistle-level intensity"
    return user_request


def _scoped_volume_target_db(user_request: str) -> float:
    text = " ".join(str(user_request or "").lower().split())
    if _contains_text(text, ["normal speaking level", "normal level", "back to normal", "normalize", "normalise"]):
        return -20.0
    if _contains_text(text, ["decrease", "lower", "weaken", "quieter", "softer", "reduce"]):
        return -25.0
    if _contains_text(text, ["increase", "raise", "louder", "boost", "amplify"]):
        return -18.0
    return -20.0


def _deterministic_acappella_extraction_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "foreground_background_remix" not in operation_ids:
        return None
    if not _requests_acappella_or_clean_vocal_extraction(user_request):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="foreground_background_remix",
                description="Extract clean vocal or a cappella foreground while suppressing accompaniment/background.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "foreground_action": "isolate_foreground",
                    "foreground_stem_policy": "vocals",
                    "background_stem_policy": "background",
                    "post_enhance": True,
                    "preserve_duration": True,
                },
                expected_output=output_audio or "clean_vocal_acappella_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_acappella_clean_vocal_extraction",
                "primary_error": primary_error,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _deterministic_spoken_voice_isolation_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "foreground_background_remix" not in operation_ids:
        return None
    if not _requests_spoken_voice_isolation(user_request):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="foreground_background_remix",
                description="Isolate the spoken speech/voice foreground and suppress non-speech background audio.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "foreground_action": "isolate_foreground",
                    "foreground_stem_policy": "speech",
                    "background_stem_policy": "background",
                    "post_enhance": False,
                    "preserve_duration": True,
                },
                expected_output=output_audio or "spoken_voice_only_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_spoken_voice_foreground_isolation",
                "primary_error": primary_error,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _requests_spoken_voice_isolation(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not text:
        return False
    if not re.search(r"\b(?:extract|isolate|keep|retain|preserve|export)\b", text):
        return False
    if not _contains_text(
        text,
        [
            "spoken voice",
            "speaking voice",
            "spoken speech",
            "spoken content",
            "spoken words",
            "spoken audio",
            "speech only",
            "voice only",
            "only the voice",
            "only spoken voice",
            "only the spoken voice",
            "only speech",
            "pure speech",
            "human speech",
        ],
    ):
        return False
    return _contains_text(
        text,
        [
            "remove everything else",
            "removing everything else",
            "without background",
            "remove background",
            "suppress background",
            "everything else",
            "only",
        ],
    )


def _deterministic_property_speech_extraction_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    primary_error: str,
    model_name: str,
) -> AudioEditPlan | None:
    if "artifact_ops.extract_spans" not in operation_ids:
        return None
    if not _requests_property_speech_extraction(user_request):
        return None
    input_audio = context.get("input_audio") or _first_input_audio(context)
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="artifact_ops.extract_spans",
                description="Extract speech/voice regions selected by a language, speaker-property, role, or emotion cue.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={"instruction": user_request},
                expected_output=output_audio or "property_selected_speech_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": context.get("source_text"),
            "source_text_audio_path": context.get("source_text_audio_path"),
            "source_text_audio_index": context.get("source_text_audio_index"),
        },
        metadata={
            "planner_model": model_name,
            "planner_fallback": {
                "used": True,
                "strategy": "deterministic_property_speech_region_extraction",
                "primary_error": primary_error,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def _requests_property_speech_extraction(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not text:
        return False
    if not re.search(r"\b(?:extract|isolate|keep|retain|preserve|export)\b", text):
        return False
    if not _contains_text(
        text,
        [
            "speech",
            "voice",
            "spoken",
            "speaker",
            "person",
            "utterance",
            "utterances",
        ],
    ):
        return False
    if _contains_text(text, ["translate", "convert", "change", "replace", "make "]):
        return False
    property_terms = [
        "accent",
        "accented",
        "dialect",
        "language",
        "english",
        "japanese",
        "chinese",
        "mandarin",
        "cantonese",
        "korean",
        "french",
        "spanish",
        "german",
        "russian",
        "indian",
        "thai",
        "vietnamese",
        "whisper",
        "shout",
        "singing",
        "rap",
        "baby",
        "infant",
        "toddler",
        "child",
        "kid",
        "girl",
        "boy",
        "woman",
        "man",
        "female",
        "male",
        "elderly",
        "young",
        "adult",
        "customer",
        "employee",
        "narrator",
        "robber",
        "victim",
        "angry",
        "sad",
        "happy",
        "calm",
        "frightened",
        "scared",
        "depressed",
        "laugh",
        "cry",
        "crying",
        "scream",
        "who is",
        "who sounds",
        "where",
        "with",
        "marked by",
        "characterized by",
    ]
    return _contains_text(text, property_terms)


def _requests_acappella_or_clean_vocal_extraction(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    has_extract = bool(re.search(r"\b(?:extract|isolate|separate|keep|suppress|remove)\b", text))
    has_vocal_target = _contains_text(
        text,
        [
            "a cappella",
            "acappella",
            "clean vocal",
            "clean vocals",
            "unaccompanied singing",
            "singing vocal",
            "singing vocals",
            "vocal stem",
        ],
    )
    has_background_suppression = _contains_text(
        text,
        [
            "instrumental",
            "instrumental tracks",
            "background noise",
            "background noises",
            "background music",
            "accompaniment",
            "suppress",
            "without music",
        ],
    )
    return has_extract and has_vocal_target and has_background_suppression


def _context_input_audio_paths(context: dict[str, Any]) -> list[str]:
    raw = context.get("input_audios") or context.get("input_audio_paths") or []
    paths: list[str] = [str(item) for item in raw if isinstance(item, str) and item]
    input_audio = context.get("input_audio")
    if isinstance(input_audio, str) and input_audio and (not paths or paths[0] != input_audio):
        paths.insert(0, input_audio)
    reference_paths = context.get("reference_audio_paths") or []
    for item in reference_paths:
        if isinstance(item, str) and item and item not in paths:
            paths.append(item)
    return paths


def _cross_file_voice_direction(user_request: str) -> tuple[int, int] | None:
    text = _normalized_audio_ref_text(user_request)
    if not text:
        return None
    if "audio1" not in text or "audio2" not in text:
        return None
    if not any(token in text for token in ("voice", "timbre", "timber", "speaker")):
        return None

    # Patterns where the first audio reference is the desired voice and the
    # second audio reference supplies the words/content to be spoken.
    voice_token = r"(?:voice|timbre|timber|speaker)"
    reference_then_source_patterns = [
        rf"{voice_token}\s+(?:from|of|in)\s+audio([12]).{{0,80}}\b(?:read|say|speak|pronounce)\b.{{0,80}}\b(?:content|speech|words|text)\b.{{0,40}}\b(?:of|from|in)\s+audio([12])",
        rf"\buse\b.{{0,60}}{voice_token}.{{0,30}}\baudio([12]).{{0,80}}\b(?:read|say|speak|pronounce)\b.{{0,80}}\baudio([12])",
        rf"\b(?:content|speech|words|text)\b.{{0,40}}\b(?:of|from|in)\s+audio([12]).{{0,80}}{voice_token}.{{0,40}}\baudio([12])",
    ]
    for pattern in reference_then_source_patterns:
        match = re.search(pattern, text)
        if match:
            first, second = _audio_ref_match_indices(match)
            if first is not None and second is not None and first != second:
                # The third pattern names the content/source first.
                if pattern.startswith(r"\b(?:content"):
                    return first, second
                return second, first

    # Patterns where the first audio reference is converted to match the
    # second audio reference.
    source_then_reference_patterns = [
        rf"\baudio([12]).{{0,80}}\b(?:to|into)\b.{{0,80}}\bmatch\b.{{0,60}}{voice_token}.{{0,40}}\baudio([12])",
        rf"\bconvert\b.{{0,80}}\baudio([12]).{{0,80}}\b(?:to|into)\b.{{0,80}}{voice_token}.{{0,40}}\baudio([12])",
        rf"\breplace\b.{{0,50}}\b(?:speech|voice|speaker|content)\b.{{0,30}}\b(?:in|of|from)\s+audio([12]).{{0,80}}{voice_token}.{{0,40}}\baudio([12])",
        rf"\breplace\b.{{0,80}}?{voice_token}.{{0,80}}?\b(?:in|of|from)\s+audio([12]).{{0,100}}?\b(?:with|to|into)\b.{{0,60}}?\b(?:that|voice|timbre|timber|speaker)\b.{{0,40}}?\b(?:of|from|in)\s+audio([12])",
        rf"\bchange\b.{{0,80}}?{voice_token}.{{0,40}}?\b(?:in|of|from)\s+audio([12]).{{0,100}}?\b(?:to|into)\b.{{0,100}}?{voice_token}.{{0,40}}?\b(?:of|from|in)\s+audio([12])",
    ]
    for pattern in source_then_reference_patterns:
        match = re.search(pattern, text)
        if match:
            first, second = _audio_ref_match_indices(match)
            if first is not None and second is not None and first != second:
                return first, second

    if (
        re.search(r"(?:voice|timbre|speaker)\s+(?:from|of|in)\s+audio1", text)
        and re.search(r"(?:content|speech|words|text)\s+(?:of|from|in)\s+audio2", text)
    ):
        return 1, 0
    if (
        re.search(r"(?:voice|timbre|speaker)\s+(?:from|of|in)\s+audio2", text)
        and re.search(r"(?:content|speech|words|text)\s+(?:of|from|in)\s+audio1", text)
    ):
        return 0, 1
    return None


def _audio_ref_match_indices(match: re.Match[str]) -> tuple[int | None, int | None]:
    try:
        first = int(match.group(1)) - 1
        second = int(match.group(2)) - 1
    except (IndexError, TypeError, ValueError):
        return None, None
    if first not in {0, 1} or second not in {0, 1}:
        return None, None
    return first, second


def _normalized_audio_ref_text(user_request: str) -> str:
    text = " ".join(str(user_request or "").lower().split())
    text = re.sub(r"\baudio\s*([12])\b", r"audio\1", text)
    text = re.sub(r"\binput\s*([12])\b", r"audio\1", text)
    text = re.sub(r"\b(?:the\s+)?first\s+audio\b", "audio1", text)
    text = re.sub(r"\b(?:the\s+)?second\s+audio\b", "audio2", text)
    text = text.replace("audio one", "audio1").replace("audio two", "audio2")
    return text


def _requests_plain_speech_region_extraction(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    if not text:
        return False
    if not re.search(r"\b(extract|isolate|keep|retain|preserve|export)\b", text):
        return False
    if not any(
        signal in text
        for signal in (
            "speech portion",
            "speech portions",
            "speech content",
            "pure speech",
            "spoken portion",
            "spoken portions",
            "spoken content",
            "spoken voice",
            "speaking voice",
            "speaking parts",
            "speech segments",
            "human speech",
            "all speech",
            "speech in the audio",
            "voice portions",
        )
    ):
        return False
    scoped_or_generative = (
        "accent",
        "dialect",
        "speaker",
        "woman",
        "man",
        "male",
        "female",
        "child",
        "kid",
        "baby",
        "elderly",
        "whisper",
        "shout",
        "sing",
        "rap",
        "laugh",
        "cry",
        "translate",
        "convert",
        "change",
        "replace",
        "make ",
    )
    for signal in scoped_or_generative:
        if " " in signal:
            if signal in text:
                return False
        elif re.search(rf"\b{re.escape(signal)}\b", text):
            return False
    return True
