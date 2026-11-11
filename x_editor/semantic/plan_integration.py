"""Planner/runtime integration helpers for semantic text transforms."""

from __future__ import annotations

import difflib
import re
from html import unescape
from typing import Any

from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.runtime.semantic_projection import SemanticProjectionError, project_semantic_transform_for_text_edit
from x_editor.semantic.schemas import SemanticTransform
from x_editor.semantic.transform_resolver import resolve_semantic_transforms


def enrich_context_with_semantic_transforms(
    user_request: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Attach auditable semantic transforms resolved from request/context."""

    if context.get("semantic_transforms"):
        return context
    source_text = _source_text_from_context(context)
    transforms = resolve_semantic_transforms(
        user_request,
        source_text=source_text,
        session_history=_semantic_session_history(context),
        evidence=_semantic_evidence(context),
    )
    context["semantic_transform_resolution"] = {
        "strategy": "semantic_transform_resolver",
        "source_text_available": bool(source_text),
        "history_count": len(_semantic_session_history(context)),
        "transforms": [transform.model_dump() for transform in transforms],
    }
    supported = [transform.model_dump() for transform in transforms if transform.is_supported]
    if supported:
        context["semantic_transforms"] = supported
    return context


def semantic_text_edit_preflight_plan(
    user_request: str,
    *,
    context: dict[str, Any],
    operation_ids: set[str],
    model_name: str,
) -> AudioEditPlan | None:
    """Return a deterministic text-edit plan for high-confidence pure semantic edits."""

    if "edit_data_pipeline.text_edit" not in operation_ids:
        return None
    if _requires_nonlexical_voice_change(user_request):
        return None
    source_text = _source_text_from_context(context)
    if not source_text:
        return None
    enrich_context_with_semantic_transforms(user_request, context)
    projected = _first_projected_transform(context.get("semantic_transforms"), source_text)
    if projected is None:
        return None
    transform = projected["semantic_transform"]
    input_audio = context.get("input_audio")
    output_audio = context.get("output_audio")
    return AudioEditPlan(
        request=user_request,
        input_audio=input_audio,
        output_audio=output_audio,
        steps=[
            AudioEditStep(
                id="step_1",
                operation="edit_data_pipeline.text_edit",
                description="Apply the resolved semantic text transform to the source speech.",
                inputs={"input_audio_path": input_audio} if input_audio else {},
                params={
                    "instruction": user_request,
                    **projected,
                },
                expected_output=output_audio or "semantic_text_edit_audio",
                depends_on=[],
                status="pending",
            )
        ],
        global_params={
            "output_dir": context.get("output_dir"),
            "source_text": source_text,
            "input_audios": context.get("input_audios") or [],
        },
        metadata={
            "planner_model": model_name,
            "semantic_transform_resolution": context.get("semantic_transform_resolution") or {},
            "semantic_preflight": {
                "used": True,
                "strategy": "semantic_text_edit_preflight",
                "transform_id": transform.get("transform_id"),
                "transform_type": transform.get("type"),
                "reason": "High-confidence pure semantic text transform projected to edit_data_pipeline.text_edit.",
            },
            "gemini_planner_call": {
                "status": "skipped",
                "strategy": "semantic_text_edit_preflight",
                "prompt_chars": 0,
                "payload_bytes": 0,
                "request_elapsed_sec": 0.0,
            },
            "routing_policy_candidates": context.get("routing_policy_candidates") or [],
        },
        missing_inputs=[] if input_audio else ["input_audio"],
    )


def apply_semantic_text_edit_plan_transform(
    case: Any,
    plan: AudioEditPlan,
    context: dict[str, Any],
) -> AudioEditPlan:
    """Project resolved semantics into text_edit steps and fix safe text misroutes."""

    source_text = _source_text_from_context(context) or getattr(case, "source_text", None)
    if not source_text:
        return plan
    enrich_context_with_semantic_transforms(str(getattr(case, "request", "") or plan.request), context)
    projected = _first_projected_transform(context.get("semantic_transforms"), source_text)
    if projected is None:
        projected = _project_from_planner_text_edit_hint(plan, source_text, context)
    if projected is None:
        return plan

    transformed = False
    target_step = _semantic_text_step(plan, allow_unsupported=not _requires_nonlexical_voice_change(plan.request))
    if target_step is None:
        return plan
    if target_step.operation == "edit_data_pipeline.text_edit" and _has_complete_text_edit_contract(
        target_step,
        source_text,
        projected=projected,
    ):
        return plan
    previous_operation = target_step.operation
    if target_step.operation != "edit_data_pipeline.text_edit":
        target_step.operation = "edit_data_pipeline.text_edit"
        transformed = True
    params = dict(target_step.params or {})
    for key, value in projected.items():
        if value not in (None, "", [], {}):
            params[key] = value
    params.setdefault("instruction", plan.request)
    if params != target_step.params:
        target_step.params = params
        transformed = True
    if not transformed:
        return plan
    plan.metadata.setdefault("runtime_plan_rewrites", []).append(
        {
            "rewrite": "apply_semantic_text_edit_plan_transform",
            "step_id": target_step.id,
            "from_operation": previous_operation,
            "to_operation": target_step.operation,
            "semantic_transform_id": projected.get("semantic_transform_id"),
            "rationale": (
                "Resolved semantic text transform was projected into the strict "
                "edit_data_pipeline.text_edit contract before execution."
            ),
        }
    )
    plan.metadata.setdefault("semantic_transform_resolution", context.get("semantic_transform_resolution") or {})
    return plan


def infer_semantic_transform_from_source_diff(
    before_text: str | None,
    after_text: str | None,
    *,
    transform_id: str = "semantic_history_diff_1",
) -> dict[str, Any] | None:
    """Infer a compact text replacement from adjacent round transcripts."""

    before = str(before_text or "")
    after = str(after_text or "")
    if not before.strip() or not after.strip() or before == after:
        return None
    matcher = difflib.SequenceMatcher(a=before, b=after)
    replacements = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source = before[i1:i2].strip()
        target = after[j1:j2].strip()
        if source or target:
            replacements.append({"source": source, "target": target, "source_units": [source], "target_units": [target]})
    if len(replacements) != 1:
        return None
    item = replacements[0]
    return SemanticTransform(
        transform_id=transform_id,
        type="text_replacement",
        source_units=[item["source"]],
        target_units=[item["target"]],
        occurrence_policy="first",
        confidence=0.72,
        rationale="Inferred previous semantic transform from adjacent round source_text difference.",
        required_evidence=["previous_source_text", "current_source_text"],
        evidence={"paired_replacements": replacements},
        projection_hints={"recommended_operation": "edit_data_pipeline.text_edit", "contract": "source_text_diff"},
    ).model_dump()


def _first_projected_transform(raw_transforms: Any, source_text: str) -> dict[str, Any] | None:
    if not isinstance(raw_transforms, list):
        return None
    for raw in raw_transforms:
        if not isinstance(raw, (dict, SemanticTransform)):
            continue
        try:
            return project_semantic_transform_for_text_edit(raw, source_text)
        except (SemanticProjectionError, ValueError):
            continue
    return None


def _project_from_planner_text_edit_hint(
    plan: AudioEditPlan,
    source_text: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    for step in plan.steps:
        if not step.operation.startswith("edit_data_pipeline."):
            continue
        transform = _semantic_transform_from_step_tags(step, plan.request)
        if transform is None:
            continue
        try:
            projected = project_semantic_transform_for_text_edit(transform, source_text)
        except SemanticProjectionError:
            continue
        context.setdefault("semantic_transform_resolution", {}).setdefault("planner_text_edit_hints", []).append(
            transform.model_dump()
        )
        return projected
    projected = _project_from_planner_target_text_diff(plan, source_text, context)
    if projected is not None:
        return projected
    return None


def _semantic_transform_from_step_tags(step: AudioEditStep, request: str) -> SemanticTransform | None:
    params = step.params or {}
    text = params.get("instruction_st") or params.get("instruction") or params.get("text_edit")
    if not isinstance(text, str) or "<sub" not in text:
        return None
    pairs = []
    for match in re.finditer(r"<sub\s+[^>]*targ=[\"'](?P<target>.*?)[\"'][^>]*>(?P<source>.*?)</sub>", text):
        source = unescape(re.sub(r"<[^>]+>", "", match.group("source"))).strip()
        target = unescape(match.group("target")).strip()
        if source:
            pairs.append({"source": source, "target": target, "source_units": [source], "target_units": [target]})
    if not pairs:
        return None
    return SemanticTransform(
        transform_id=f"semantic_planner_tag_hint_{step.id}",
        type="text_replacement",
        source_units=[item["source"] for item in pairs],
        target_units=[item["target"] for item in pairs],
        occurrence_policy="all" if _requests_all_occurrences(request) else "first",
        confidence=0.76,
        rationale="Recovered a semantic text replacement from planner-provided local text-edit tags.",
        required_evidence=["planner_step_params", "source_text"],
        evidence={"paired_replacements": pairs},
        projection_hints={"recommended_operation": "edit_data_pipeline.text_edit", "contract": "planner_tag_hint"},
    )


def _project_from_planner_target_text_diff(
    plan: AudioEditPlan,
    source_text: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    for step in plan.steps:
        if step.operation != "edit_data_pipeline.text_edit":
            continue
        params = step.params or {}
        target_text = params.get("target_text")
        if not isinstance(target_text, str) or not target_text.strip() or target_text == source_text:
            continue
        inferred = infer_semantic_transform_from_source_diff(
            source_text,
            target_text,
            transform_id=f"semantic_planner_target_diff_{step.id}",
        )
        if not inferred:
            continue
        transform = SemanticTransform(**inferred)
        if _requests_all_occurrences(plan.request):
            transform.occurrence_policy = "all"
        try:
            projected = project_semantic_transform_for_text_edit(transform, source_text)
        except SemanticProjectionError:
            continue
        context.setdefault("semantic_transform_resolution", {}).setdefault("planner_target_text_diff_hints", []).append(
            transform.model_dump()
        )
        return projected
    return None


def _semantic_text_step(plan: AudioEditPlan, *, allow_unsupported: bool) -> AudioEditStep | None:
    for step in plan.steps:
        if step.operation == "edit_data_pipeline.text_edit":
            return step
    for step in plan.steps:
        if not step.operation.startswith("edit_data_pipeline."):
            continue
        params = step.params or {}
        label_contract = params.get("label_contract")
        if isinstance(label_contract, dict) and label_contract.get("field") == "instruction_st":
            allowed = set(label_contract.get("allowed_tags") or [])
            if allowed & {"sub", "ins", "del"}:
                return step
        if any(isinstance(params.get(key), str) and "<sub" in params[key] for key in ("instruction", "instruction_st", "text_edit")):
            return step
    if allow_unsupported:
        for step in plan.steps:
            if step.operation == "unsupported":
                return step
    return None


def _has_complete_text_edit_contract(
    step: AudioEditStep,
    source_text: str,
    *,
    projected: dict[str, Any],
) -> bool:
    params = step.params or {}
    if not bool(
        (params.get("source_text") or source_text)
        and params.get("target_text")
        and params.get("instruction_st")
    ):
        return False
    projected_target = projected.get("target_text")
    if isinstance(projected_target, str) and projected_target.strip():
        return params.get("target_text") == projected_target
    return True


def _source_text_from_context(context: dict[str, Any]) -> str | None:
    value = context.get("source_text")
    if isinstance(value, str) and value.strip():
        return value
    perception = context.get("perception_report")
    if isinstance(perception, dict):
        value = perception.get("source_text")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _semantic_session_history(context: dict[str, Any]) -> list[dict[str, Any]]:
    history = context.get("semantic_session_history")
    if not isinstance(history, list):
        return []
    return [dict(item) for item in history if isinstance(item, dict)]


def _semantic_evidence(context: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("semantic_evidence", "perception_report"):
        raw = context.get(key)
        if isinstance(raw, dict):
            aliases = raw.get("entity_aliases")
            if isinstance(aliases, dict):
                evidence.setdefault("entity_aliases", {}).update(aliases)
    aliases = context.get("entity_aliases")
    if isinstance(aliases, dict):
        evidence.setdefault("entity_aliases", {}).update(aliases)
    return evidence


def _requires_nonlexical_voice_change(request: str) -> bool:
    lowered = str(request or "").casefold()
    if not any(word in lowered for word in ("voice", "timbre", "speaker", "female", "male", "woman", "women", "man", "men")):
        return False
    if any(phrase in lowered for phrase in ("same voice", "original voice", "speaker's voice", "speakers voice", "cloning the speaker")):
        return False
    return any(
        phrase in lowered
        for phrase in (
            "women's voice",
            "woman's voice",
            "female voice",
            "man's voice",
            "men's voice",
            "male voice",
            "change the voice",
            "convert the voice",
            "with a voice",
        )
    )


def _requests_all_occurrences(request: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(request or "").casefold())
    return any(
        phrase in lowered
        for phrase in (
            "every instance",
            "all instance",
            "all the instance",
            "all occurrences",
            "all the occurrences",
            "each instance",
            "所有",
            "全部",
            "每个",
        )
    )
