"""OpenAI-compatible structured planner backends for X-Editor."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

from pydantic import ValidationError

from x_editor.agents.openai_compatible import (
    OpenAICompatibleChatClient,
    OpenAICompatibleServiceUnavailable,
)
from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    OPEN_GLM_QWEN_OMNI_MODE,
    provider_service_hint,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.invocation_trace import (
    ComponentInvocationRoute,
    ComponentInvocationStatus,
    ComponentInvocationTrace,
    ProviderInvocationRecord,
)
from x_editor.core.schemas import AudioEditPlan, AudioEditStep
from x_editor.perception.workflow_catalog import (
    workflow_perception_catalog_payload,
)
from x_editor.planner._invocation_trace import provider_records_from_client
from x_editor.planner.gemini_planner import (
    _deterministic_preflight_plan,
    _ensure_plan_decision_audit,
    _ensure_planner_decision_audit,
    _infer_high_level_objectives,
    _operation_ids,
    _planner_evidence_context,
    _system_prompt_for_context,
    _user_prompt,
)
from x_editor.planner.perception_guidance import add_planner_perception_guidance
from x_editor.planner.plan_normalizer import normalize_planner_output
from x_editor.planner.prompt_package import (
    DEFAULT_PLANNER_PROMPT_PROFILE,
    PlannerPromptPackage,
    PlannerPromptProfileName,
    build_planner_prompt_package,
    candidate_capability_projection,
    compact_operation_capability_index,
    compact_semantic_effect_capability_index,
    frozen_edit_intent_from_context,
    planner_reasoner_turn_contract_outline,
)
from x_editor.planner.reasoned_workflow import (
    PlannerExecutionSemantics,
    PlannerReasonerTurn,
    planner_reasoner_output_schema,
)
from x_editor.planner.routing_hard_boundaries import (
    TRANSCRIPT_REQUIRED_EDIT_DATA_PIPELINE_OPERATIONS,
    apply_routing_hard_boundaries,
    observe_routing_hard_boundaries,
)
from x_editor.planner.routing_policy import routing_policy_context
from x_editor.planner.workflow_turn_normalizer import (
    normalize_workflow_turn_shape,
)
from x_editor.runtime.workflow_protocol import (
    WorkflowProtocolPolicy,
    resolve_workflow_protocol_policy,
)


class OpenAICompatiblePlannerError(RuntimeError):
    """Raised when an OpenAI-compatible planner cannot produce a valid plan."""


class OpenAICompatiblePlannerContractError(OpenAICompatiblePlannerError):
    """Provider returned content that the native workflow contract rejected."""

    failure_kind = "protocol"


DEFAULT_OPENAI_COMPATIBLE_PLANNER_MAX_TOKENS = 8192
DEFAULT_GLM_PLANNER_MAX_TOKENS = 16384


class OpenAICompatibleAudioEditPlanner:
    """Generate ``AudioEditPlan`` through an OpenAI-compatible text model."""

    provider_name = "openai_compatible"
    agentic_mode = OPEN_GLM_QWEN_OMNI_MODE

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        available_operations: list[dict[str, Any]] | None = None,
        client: OpenAICompatibleChatClient | None = None,
    ) -> None:
        self.model_name = resolve_openai_compatible_model(self.provider_name, model_name)
        self.available_operations = available_operations or []
        self.api_key = resolve_openai_compatible_api_key(self.provider_name, api_key)
        self.base_url = resolve_openai_compatible_base_url(self.provider_name, base_url)
        self._client_provided = client is not None
        self.client = client or OpenAICompatibleChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            provider_name=self.provider_name,
            temperature=temperature,
            max_tokens=_resolve_planner_max_tokens(self.provider_name, max_tokens),
            timeout=timeout,
        )
        self.last_planner_evidence_context: dict[str, Any] = {}
        self.last_invocation_trace: ComponentInvocationTrace | None = None
        self.last_prompt_package: PlannerPromptPackage | None = None
        self._provider_invocations: list[ProviderInvocationRecord] = []

    def plan(self, user_request: str, context: dict[str, Any] | None = None) -> AudioEditPlan:
        """Return a validated structured plan without executing tools."""

        self.last_invocation_trace = None
        self.last_prompt_package = None
        self._provider_invocations = []
        original_context = context if isinstance(context, dict) else None
        context = dict(context or {})
        context.setdefault("agentic_mode", self.agentic_mode)
        context.setdefault("planner_provider", self.provider_name)
        operations = context.get("available_operations") or self.available_operations
        if not operations:
            self._set_invocation_trace(status="failed", route="deterministic")
            raise OpenAICompatiblePlannerError(
                "OpenAI-compatible planner requires a non-empty available_operations list"
            )
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
            original_context["agentic_mode"] = self.agentic_mode
            original_context["planner_provider"] = self.provider_name

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
            preflight.metadata.setdefault("agentic_mode", self.agentic_mode)
            preflight.metadata.setdefault("planner_provider", self.provider_name)
            self._set_invocation_trace(
                status="completed",
                route="deterministic_preflight",
            )
            return preflight

        try:
            self._require_configured()
        except Exception:
            self._set_invocation_trace(status="failed", route="deterministic")
            raise

        system_prompt = _open_system_prompt(
            _system_prompt_for_context(context),
            self.provider_name,
        )
        try:
            data = self.client.call_json(
                system_prompt,
                _user_prompt(
                    user_request=user_request,
                    context=context,
                    operations=operations,
                    model_name=self.model_name,
                ),
                response_format=_audio_edit_plan_response_format(self.provider_name),
                repair_schema_name="AudioEditPlan",
            )
        except OpenAICompatibleServiceUnavailable as exc:
            self._append_client_invocations(
                planner_purpose="primary",
                succeeded=False,
                error=exc,
            )
            self._set_invocation_trace(status="failed", route="provider")
            raise OpenAICompatiblePlannerError(
                f"{self.provider_name} planner service unavailable: {exc}. "
                f"To use {self.agentic_mode}, "
                f"{provider_service_hint(self.provider_name)}."
            ) from exc
        except Exception as exc:
            self._append_client_invocations(
                planner_purpose="primary",
                succeeded=False,
                error=exc,
            )
            self._set_invocation_trace(status="failed", route="provider")
            raise OpenAICompatiblePlannerError(
                f"{self.provider_name} planner failed before producing a valid "
                f"AudioEditPlan: {type(exc).__name__}: {exc}"
            ) from exc
        self._append_client_invocations(
            planner_purpose="primary",
            succeeded=True,
        )

        try:
            data = _unwrap_plan_payload(data)
            data, catalog_normalizations = _normalize_plan_with_outside_catalog_operations(
                data,
                available_operation_ids=_operation_ids(operations),
            )
            data = self._normalize_and_route_plan(
                data,
                user_request=user_request,
                context=context,
                operations=operations,
            )
            return self._build_plan_with_schema_repair(
                data,
                user_request=user_request,
                context=context,
                operations=operations,
                system_prompt=system_prompt,
                catalog_normalizations=catalog_normalizations,
            )
        except Exception:
            if self.last_invocation_trace is None:
                self._set_invocation_trace(status="failed", route="provider")
            raise

    def plan_workflow_turn(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
    ) -> PlannerReasonerTurn:
        """Return one native W1 Planner/Reasoner decision.

        Unlike ``plan()``, this entry point may request evidence before an
        executable plan exists. It never invokes planner-side Perception,
        deterministic preflight planning, schema repair, or plan fallback.
        """

        self.last_invocation_trace = None
        self.last_prompt_package = None
        self._provider_invocations = []
        context = dict(context or {})
        operations = context.get("available_operations") or self.available_operations
        if not operations:
            self._set_invocation_trace(status="failed", route="deterministic")
            raise OpenAICompatiblePlannerError(
                "native Planner/Reasoner workflow requires available_operations"
            )
        source_audio_refs = [
            str(item) for item in context.get("workflow_source_audio_refs") or [] if str(item)
        ]
        if not source_audio_refs:
            self._set_invocation_trace(status="failed", route="deterministic")
            raise OpenAICompatiblePlannerError(
                "native Planner/Reasoner workflow requires workflow_source_audio_refs"
            )
        try:
            self._require_configured()
        except Exception:
            self._set_invocation_trace(status="failed", route="deterministic")
            raise

        workflow_protocol_policy = resolve_workflow_protocol_policy(
            context.get("workflow_protocol_policy")
        )
        prompt_package = _planner_reasoner_prompt_package(
            user_request=user_request,
            context=context,
            operations=operations,
            source_audio_refs=source_audio_refs,
            policy=workflow_protocol_policy,
        )
        self.last_prompt_package = prompt_package
        system_prompt = prompt_package.system_prompt
        post_perception = prompt_package.phase == "post_perception"
        try:
            data = self.client.call_json(
                system_prompt,
                prompt_package.current_prompt,
                response_format=_planner_reasoner_response_format(
                    self.provider_name,
                    post_perception=post_perception,
                    prompt_profile=prompt_package.budget_report.prompt_profile,
                ),
                repair_schema_name="PlannerReasonerTurn",
            )
        except Exception as exc:
            self._append_client_invocations(
                planner_purpose="primary",
                succeeded=False,
                error=exc,
            )
            self._set_invocation_trace(status="failed", route="provider")
            raise OpenAICompatiblePlannerError(
                f"{self.provider_name} Planner/Reasoner workflow call failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._append_client_invocations(
            planner_purpose="primary",
            succeeded=True,
        )
        protocol_normalizations: list[str] = []
        try:
            shape_normalized, shape_audit = normalize_workflow_turn_shape(
                data,
                authoritative_request=user_request,
            )
            protocol_normalizations.extend(item.audit_code() for item in shape_audit)
            raw_turn, protocol_corrections = _normalize_planner_reasoner_payload(
                shape_normalized,
                policy=workflow_protocol_policy,
                evidence_required_operations=(
                    set(context["workflow_evidence_required_operations"])
                    if isinstance(
                        context.get("workflow_evidence_required_operations"),
                        (list, tuple, set),
                    )
                    else None
                ),
            )
            protocol_normalizations.extend(protocol_corrections)
            _restore_authoritative_frozen_intent(
                raw_turn,
                frozen_edit_intent=frozen_edit_intent_from_context(context),
                prompt_profile=prompt_package.budget_report.prompt_profile,
                corrections=protocol_normalizations,
            )
            turn = PlannerReasonerTurn.model_validate(raw_turn)
            normalized_turn = turn.model_dump(mode="python")
            _normalize_execute_with_outside_catalog_operations(
                normalized_turn,
                available_operation_ids=_operation_ids(operations),
                corrections=protocol_normalizations,
            )
            turn = PlannerReasonerTurn.model_validate(normalized_turn)
            if turn.decision == "execute" and turn.plan is not None:
                original_plan = turn.plan
                normalized = self._normalize_and_route_plan(
                    turn.plan.model_dump(mode="python"),
                    user_request=user_request,
                    context=context,
                    operations=operations,
                )
                normalized_plan = AudioEditPlan.model_validate(normalized)
                normalized_semantics, operation_corrections = (
                    _synchronize_normalized_binding_operations(
                        original_plan=original_plan,
                        normalized_plan=normalized_plan,
                        semantics=turn.execution_semantics,
                    )
                )
                protocol_normalizations.extend(operation_corrections)
                metadata = normalized_plan.metadata
                if isinstance(metadata, dict):
                    metadata.pop("openai_compatible_planner_call", None)
                    if protocol_normalizations:
                        metadata["planner_reasoner_normalizations"] = list(protocol_normalizations)
                turn = turn.model_copy(
                    update={
                        "plan": normalized_plan,
                        "execution_semantics": normalized_semantics,
                    },
                    deep=True,
                )
            self._set_invocation_trace(
                status="completed",
                route="provider",
                protocol_normalizations=protocol_normalizations,
            )
            return PlannerReasonerTurn.model_validate(turn.model_dump(mode="python"))
        except Exception as exc:
            self._set_invocation_trace(
                status="failed",
                route="provider",
                protocol_normalizations=protocol_normalizations,
            )
            raise OpenAICompatiblePlannerContractError(
                f"Planner/Reasoner returned an invalid workflow decision: {exc}"
            ) from exc

    def _require_configured(self) -> None:
        if self._client_provided:
            return
        if not self.api_key or not self.base_url:
            raise OpenAICompatiblePlannerError(
                f"{self.provider_name} planner service unavailable: {provider_service_hint(self.provider_name)}"
            )

    def _normalize_and_route_plan(
        self,
        data: dict[str, Any],
        *,
        user_request: str,
        context: dict[str, Any],
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise OpenAICompatiblePlannerError(
                f"{self.provider_name} planner expected a JSON object, got {type(data).__name__}"
            )
        metadata = data.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["agentic_mode"] = self.agentic_mode
            metadata["planner_provider"] = self.provider_name
            call_metrics = getattr(self.client, "last_call_metrics", None)
            if isinstance(call_metrics, dict):
                metadata["openai_compatible_planner_call"] = dict(call_metrics)
        normalize_planner_output(
            data,
            user_request=user_request,
            context=context,
            operations=operations,
            planner_model=self.model_name,
            phase="pre_routing",
        )
        if context.get("workflow_routing_boundary_mode") == "shadow":
            observe_routing_hard_boundaries(
                data,
                user_request=user_request,
                context=context,
                operations=operations,
            )
        else:
            apply_routing_hard_boundaries(
                data,
                user_request=user_request,
                context=context,
                operations=operations,
            )
        normalize_planner_output(
            data,
            user_request=user_request,
            context=context,
            operations=operations,
            planner_model=self.model_name,
            phase="post_routing",
        )
        _ensure_planner_decision_audit(
            data, user_request=user_request, context=context, operations=operations
        )
        return data

    def _build_plan_with_schema_repair(
        self,
        data: dict[str, Any],
        *,
        user_request: str,
        context: dict[str, Any],
        operations: list[dict[str, Any]],
        system_prompt: str,
        catalog_normalizations: list[dict[str, Any]] | None = None,
    ) -> AudioEditPlan:
        try:
            plan = AudioEditPlan(**data)
        except ValidationError as exc:
            repair_context = {
                "schema_error": str(exc),
                "invalid_plan": data,
                "user_request": user_request,
                "available_operations": operations,
                "instruction": (
                    "Repair only the AudioEditPlan JSON schema. Preserve the same user objective and "
                    "do not add lower-level tool calls."
                ),
            }
            try:
                repaired = self.client.call_json(
                    system_prompt,
                    json.dumps(repair_context, ensure_ascii=False),
                    response_format=_audio_edit_plan_response_format(self.provider_name),
                    repair_schema_name="AudioEditPlan",
                )
            except Exception as repair_call_exc:
                self._append_client_invocations(
                    planner_purpose="schema_repair",
                    succeeded=False,
                    error=repair_call_exc,
                )
                self._set_invocation_trace(status="failed", route="provider")
                raise
            self._append_client_invocations(
                planner_purpose="schema_repair",
                succeeded=True,
            )
            repaired = _unwrap_plan_payload(repaired)
            repaired = self._normalize_and_route_plan(
                repaired,
                user_request=user_request,
                context=context,
                operations=operations,
            )
            try:
                plan = AudioEditPlan(**repaired)
            except ValidationError as repair_exc:
                self._set_invocation_trace(status="failed", route="provider")
                raise OpenAICompatiblePlannerError(
                    f"{self.provider_name} planner returned invalid AudioEditPlan after schema repair: {repair_exc}"
                ) from repair_exc
        self._set_invocation_trace(
            status="completed",
            route="provider",
            catalog_normalizations=catalog_normalizations,
        )
        return plan

    def _append_client_invocations(
        self,
        *,
        planner_purpose: str,
        succeeded: bool,
        error: Exception | None = None,
    ) -> None:
        self._provider_invocations.extend(
            provider_records_from_client(
                self.client,
                provider=self.provider_name,
                model=self.model_name,
                planner_purpose=planner_purpose,
                start_index=len(self._provider_invocations) + 1,
                succeeded=succeeded,
                error=error,
            )
        )

    def _set_invocation_trace(
        self,
        *,
        status: ComponentInvocationStatus,
        route: ComponentInvocationRoute,
        protocol_normalizations: list[str] | None = None,
        catalog_normalizations: list[dict[str, Any]] | None = None,
    ) -> None:
        fallback_purpose = next(
            (
                item.purpose
                for item in self._provider_invocations
                if item.kind == "protocol_fallback"
            ),
            None,
        )
        fallback_used = fallback_purpose is not None
        effective_route: ComponentInvocationRoute = (
            "mixed" if fallback_used and route == "provider" else route
        )
        self.last_invocation_trace = ComponentInvocationTrace.build(
            component="planner",
            component_name=type(self).__name__,
            route=effective_route,
            status=status,
            provider_invocations=self._provider_invocations,
            metadata={
                "model": self.model_name,
                "provider": self.provider_name,
                **(
                    {"planner_reasoner_normalizations": list(protocol_normalizations)}
                    if protocol_normalizations
                    else {}
                ),
                **(
                    {"planner_catalog_normalizations": copy.deepcopy(catalog_normalizations)}
                    if catalog_normalizations
                    else {}
                ),
                **(
                    {
                        "planner_prompt_package_fingerprint": (
                            self.last_prompt_package.package_fingerprint
                        ),
                        "planner_prompt_budget_report": (
                            self.last_prompt_package.budget_report.model_dump(mode="json")
                        ),
                    }
                    if self.last_prompt_package is not None
                    else {}
                ),
            },
        )


class GLMOpenAICompatiblePlanner(OpenAICompatibleAudioEditPlanner):
    """GLM-4.7-Flash planner using an OpenAI-compatible endpoint."""

    provider_name = GLM_PROVIDER


def _resolve_planner_max_tokens(provider_name: str, explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    for env_name in _planner_max_token_env_names(provider_name):
        raw = os.getenv(env_name)
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue
    if provider_name == GLM_PROVIDER:
        return DEFAULT_GLM_PLANNER_MAX_TOKENS
    return DEFAULT_OPENAI_COMPATIBLE_PLANNER_MAX_TOKENS


def _planner_max_token_env_names(provider_name: str) -> list[str]:
    names: list[str] = []
    if provider_name == GLM_PROVIDER:
        names.extend(
            [
                "X_EDITOR_GLM_PLANNER_MAX_TOKENS",
                "X_EDITOR_GLM_MAX_TOKENS",
            ]
        )
    provider_key = provider_name.strip().upper().replace("-", "_")
    names.extend(
        [
            f"X_EDITOR_{provider_key}_PLANNER_MAX_TOKENS",
            f"X_EDITOR_{provider_key}_MAX_TOKENS",
            "X_EDITOR_OPENAI_COMPAT_PLANNER_MAX_TOKENS",
            "X_EDITOR_OPENAI_COMPAT_MAX_TOKENS",
        ]
    )
    return names


def _open_system_prompt(system_prompt: str, provider_name: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "Open provider constraints:\n"
        f"- You are the {provider_name} structured planner backend for X-Editor.\n"
        "- Produce only the top-level AudioEditPlan JSON object. Do not directly call MCP tools or low-level runtime tools.\n"
        "- If using a tool-call capable endpoint, the only allowed semantic action is submit_audio_edit_plan with the AudioEditPlan fields.\n"
        "- Dirty JSON is rejected by deterministic parsing, schema validation, normalizers, routing hard boundaries, and executor verifiers.\n"
    )


def _audio_edit_plan_response_format(provider_name: str | None = None) -> dict[str, Any] | None:
    mode = _planner_response_format_mode(provider_name)
    if mode == "none":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "AudioEditPlan",
            "schema": AudioEditPlan.model_json_schema(),
            "strict": True,
        },
    }


def _planner_reasoner_response_format(
    provider_name: str | None = None,
    *,
    post_perception: bool = False,
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
) -> dict[str, Any] | None:
    mode = _planner_response_format_mode(provider_name)
    if prompt_profile in {
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    } and mode == "json_object":
        mode = "json_schema"
    if mode == "none":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "PlannerReasonerTurn",
            "schema": planner_reasoner_output_schema(post_perception=post_perception),
            "strict": True,
        },
    }


def _planner_reasoner_system_prompt(
    policy: WorkflowProtocolPolicy | str | None = None,
    *,
    operations: list[dict[str, Any]] | None = None,
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
) -> str:
    """Return stable role and safety instructions shared by every W1 call."""

    del operations
    resolved = resolve_workflow_protocol_policy(policy)
    framework_field_rule = (
        "plan.global_params and plan.metadata must be empty; non-empty values are rejected"
        if resolved.framework_owned_plan_fields_mode == "block"
        else (
            "plan.global_params and plan.metadata must be empty; accidental values "
            "are stripped and audited"
        )
    )
    uncertainty_rule = (
        "reviewable uncertainty blocks this protocol profile"
        if resolved.reviewable_uncertainty_mode == "block"
        else "reviewable uncertainty is retained and does not block by itself"
    )
    binding_rule = {
        "explicit_only": (
            "write exact argument paths and every required evidence source explicitly"
        ),
        "canonical_alias": (
            "write every required evidence source explicitly; the framework may "
            "canonicalize one unambiguous argument alias"
        ),
        "exact_unique": (
            "write required evidence sources explicitly when possible; the framework "
            "may canonicalize only an unambiguous schema alias from its registered "
            "list or recover one exact, uniquely matching local span boundary"
        ),
    }[resolved.binding_normalization_mode]
    if prompt_profile in {
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    }:
        parameter_contract_rule = (
            "Use each selected card's parameter_contract directly. Required fields, "
            "types, enums, ranges, cardinality, and conditional rules are binding."
            if prompt_profile in {"robust_contract_v5", "robust_contract_v6"}
            else (
                "Materialize every required address in candidate_execution_checklist "
                "before adding optional values."
            )
        )
        return f"""You are X-Editor's Planner/Reasoner for one W1 speech-editing call.

Return exactly one PlannerReasonerTurn JSON object. The API schema owns JSON shape;
the current-call package owns semantics, available operations, legal parameter values,
audio references, and reviewed evidence. Never add fields outside the schema.

Call roles:
- pre_perception: represent every user objective as typed audio effects, record what
  must be preserved, and request one compatible evidence route. Do not plan tools.
- post_perception: use reviewed evidence and the current candidate cards to emit a
  legal dependency graph, or return cannot_plan with precise missing information.

Semantic discipline:
- Build an ordered action ledger before choosing effects. Each state-changing action
  clause becomes an objective. In "A, then B", both A and B remain required; words
  such as then and resulting declare dependency and never replace A with B.
- First classify the thing being changed. A property of a person, object, date,
  quantity, place, or event described by speech belongs to spoken content. A property
  of the voice, room, channel, waveform, or audible non-speech event belongs to the
  recording signal. A calendar or narrative event mentioned in words is not an
  audible-event label. Figurative descriptions of world objects remain content edits.
- Keep proposition time separate from media time. A day, date, clock time, age, or
  duration stated by the speaker is spoken content. Timeline effects only change
  where already recorded material appears within the clip or its playback order.
  Words such as move, earlier, later, before, and after do not override this test.
- Interpret the whole request, including preservation clauses. Use the least
  destructive audio representation that satisfies the requested perceptual change.
  Do not introduce a new signal class such as a tone, mute, deletion, replacement,
  new speaker, or new background unless the user requests that class or evidence
  makes it necessary.
- Separate an unknown target from an unknown effect. If the requested kind of change
  is clear but the exact word, speaker, event, or span is implicit, retain the narrow
  effect and request target grounding; do not substitute a broader acoustic effect.
- Prefer the narrowest compatible scope. Broad whole-segment effects are not a
  substitute for lexical, speaker-local, event-local, or span-local edits.
- Retain multiple effect hypotheses only when the request truly leaves referent
  ownership unresolved between proposition content and the recording signal. Shared
  adjectives or figurative wording alone are not ambiguity. Once ownership is clear,
  choose one narrow effect representation and use evidence only to ground its target.
- Preserve transformation direction. For relative requests such as earlier/later,
  more/less, higher/lower, or add/remove, record the requested relation without
  inverting it. Let reviewed source evidence supply the original value, then derive
  the target in the same direction. Never treat the user's instruction text as if it
  were the source transcript.
- A quoted_text target must be verbatim source wording explicitly supplied or already
  known. A relational instruction, paraphrase, role, or desired meaning is not source
  speech. Represent it as a semantic target, request source_text, and derive the new
  value only after reading that evidence.
- In post_perception, one step obeys one capability card. Split compound objectives
  into dependent steps and pass transformed audio with an explicit from_step object.

Contract discipline:
- Use only Planner-selectable addresses and current legal values. params.x is the
  address step.params.x, never a nested key named params. Never author framework-owned
  fields, placeholders, guessed paths, filenames, evidence ids as audio refs, or
  source audio where a reference audio is required.
- {parameter_contract_rule} Then mirror each authored value exactly once in
  tool_selection.
- Materialize required addresses first. Omit unused optional addresses entirely;
  never emit null, an empty string, or an empty list as an optional placeholder.
- Copy reviewed evidence values and field paths exactly. Bind each evidence-derived
  scalar leaf, including every local start_sec and end_sec. Non-evidence sources must
  omit evidence_id and evidence_path.
- Every tool_selection step must include selected_operation, at least one complete
  candidate object, parameter_selections, and selection_summary. The selected
  operation must appear in candidates and equal the plan step operation.
- Before execute, check objective coverage, required addresses, legal domains,
  cardinality, dependencies, and evidence. If any is missing, return cannot_plan.

Protocol authority: {framework_field_rule}; {binding_rule}. Missing optional
explanation is not a blocker; {uncertainty_rule}. Never include benchmark identity,
oracle/evaluator data, raw provider output, retries, fallback instructions, or tool
results.
"""
    if prompt_profile in {"objective_coverage_v2", "evidence_complete_v3"}:
        prior_rule = (
            "- tool_prior_guidance is a frozen development ranking aid, not a gate. "
            "Use it only after semantic fit, legal inputs, preservation, and evidence "
            "are satisfied; no_preference forbids an implicit tie-break.\n"
            if prompt_profile == "evidence_complete_v3"
            else ""
        )
        candidate_rule = (
            "- post_perception exposes only typed-shortlist capability cards. The "
            "shortlist is the union of provisional candidates and exact typed-effect "
            "matches; only those cards authorize an executable operation and its "
            "parameters. If the frozen typed effects themselves are wrong, return "
            "cannot_plan with an intent/card mismatch instead of grabbing an "
            "unrelated tool.\n"
            if prompt_profile == "evidence_complete_v3"
            else ""
        )
        return f"""You are X-Editor's Planner/Reasoner for the W1 speech-editing workflow.

Return exactly one PlannerReasonerTurn JSON object accepted by the response schema.
Do not expose private chain of thought. decision_summary is only a short audit result.

The current-call package is authoritative. Its current_task says what this call must
produce, its capability material says what operations can do, and its reviewed
evidence says which observed values may be used.

Workflow roles:
- pre_perception: identify every atomic edit objective, preservation requirement,
  quality requirement, and required evidence. Candidate operations are provisional.
  Request one compatible Perception route; do not write an executable plan.
- post_perception: reconsider provisional candidates using reviewed evidence and the
  complete executable capability cards. Produce a legal operation graph or return
  cannot_plan. Never request Perception from the final plan.

Planning invariants:
- Preserve the user's semantic objectives across turns. Evidence may resolve
  uncertainty and change tool choice, but must not silently delete or weaken an
  objective.
- One step obeys one operation card. If one card cannot implement every assigned
  effect, split the work into dependent steps and connect them with from_step.
- Capability-card parameter paths are addresses, not keys to copy literally.
  params.x means step.params.x; inputs.y means step.inputs.y.
- Use only declared Planner-selectable values. Never author framework-owned fields,
  placeholders, aliases, guessed evidence, output filenames, or undeclared defaults.
- Treat workflow_source_audio_refs and workflow_reference_audio_refs as legacy
  availability groups, not permanent per-turn roles. For every supplied audio used by
  a step, emit an exact execution_semantics.audio_usage_bindings item naming the step,
  scalar argument path, audio ref, current use, and whether reviewed facts are needed.
  Plain assembly may require only existence_decodability; content- or speaker-selected
  uses require reviewed_facts and exact evidence paths.
- Keep supplied audio, reviewed evidence ids, prior ArtifactRefs,
  and future outputs in distinct namespaces: an evidence_id identifies reviewed
  facts only; never place an evidence id in an audio input field. A prior artifact
  may use {{"artifact_ref_id":"artifact_ref_id"}} only when that exact
  ArtifactRef.ref_id is supplied by framework context. A future output is
  {{"from_step":"step_id","from_output":"output_audio_path"}} and requires that
  step.depends_on must include that producer step id; never invent future output paths
  or filenames such as extracted_clip.wav; use current only for an unambiguous
  sequential chain; use an explicit from_step reference for branches or multiple
  predecessors.
- From reviewed_evidence_paths_by_id, copy every evidence field path exactly. Copy
  evidence values exactly. Local span arguments require exact start_sec and end_sec
  sources for every selected span. Also cover semantic/evidence-dependent arguments
  with argument_sources. Include evidence_id and evidence_path only for
  perception_evidence or derived_from_evidence; for deterministic_context and
  user_request, omit both fields entirely.
- Build the executable plan first. Then mirror its Planner-selected values exactly
  once into tool_selection and bind evidence-derived scalar leaves in
  execution_semantics.
- Before execute, check every user objective, required parameter, legal domain,
  cardinality, dependency, and evidence source. If any required item is unavailable,
  return cannot_plan with precise missing_information.
{prior_rule}
{candidate_rule}

Protocol authority: {framework_field_rule}; {binding_rule}. Missing optional
explanation is not a blocker; {uncertainty_rule}. Never include benchmark identity,
oracle or evaluator data, raw provider output, retries, fallback instructions, or
tool results.
"""

    return f"""You are X-Editor's Planner/Reasoner for the strict W1 workflow.

Return exactly one PlannerReasonerTurn JSON object accepted by the API response
schema. Do not expose private chain of thought. Use decision_summary only for a
short, auditable conclusion.

The current-call package is the authority for this invocation. Read current_call.phase
and current_task before acting:
- pre_perception: freeze a typed edit_intent, then either request exactly one listed
  Perception route or return cannot_plan. Do not emit an AudioEditPlan.
- post_perception: preserve the frozen intent when supplied, then either emit a
  complete executable plan plus execution_semantics and tool_selection, or return
  cannot_plan. Never request Perception from the final plan.

Capability material has two deliberately different forms:
- edit_operation_capability_index is a compact semantic index for intent formation
  and candidate nomination. It does not declare legal parameter names or values.
- candidate_operation_capability_cards are the complete authority for post-Perception
  route trade-offs and Planner-selectable parameters. Use only listed candidate
  operations and exact planner_selectable paths. Obey type, enum, range, required,
  mutual-exclusion, at-least-one, and conditional rules. Never invent a parameter,
  alias, operation, default, or framework-owned value.

Hard boundaries:
- If required evidence, legal inputs, a complete candidate contract, or a safe route
  is unavailable, return cannot_plan with exact missing_information.
- Keep source audio, reference audio, reviewed evidence ids, prior ArtifactRefs, and
  future from_step outputs in distinct namespaces: an evidence_id identifies reviewed
  facts only; never place an evidence id in an audio input field. A prior artifact may
  be written exactly as {{"artifact_ref_id":"artifact_ref_id"}} only when that exact
  ArtifactRef.ref_id is supplied by framework context.
- A future output must be written exactly as
  {{"from_step":"step_id","from_output":"output_audio_path"}}, and step.depends_on
  must include that producer step id; never invent future output paths or filenames
  such as extracted_clip.wav; use current only for an unambiguous sequential chain;
  use an explicit from_step reference for branches or multiple predecessors.
- Treat workflow source/reference lists as legacy availability groups. Bind every
  supplied audio argument through execution_semantics.audio_usage_bindings with its
  exact scalar argument path and current use. Existence/decodability is sufficient for
  plain assembly; observed-content selection requires reviewed_facts evidence bound to
  that same audio ref.
- Copy admitted evidence boundaries exactly. Do not round, pad, shrink, or widen them
  unless the selected card declares a separate Planner-selectable transform parameter.
- copy every evidence field path exactly from reviewed_evidence_paths_by_id. Bind
  scalar leaves with exact inputs./params. argument paths. Also cover semantic/evidence-dependent arguments with argument_sources.
- Include evidence_id and evidence_path only for perception_evidence or
  derived_from_evidence sources; omit both fields entirely for deterministic_context
  and user_request sources. Do not claim derived_from_evidence without a listed
  deterministic transform.
- Give each step exactly one operation contract and emit steps in topological order.
  Cover every framework_required_objective with a contract that implements it.
- {framework_field_rule}; {binding_rule}.
- Unknown operations, unknown or disagreeing evidence values, ambiguous aliases,
  unresolved semantic requirements, and unsafe ungrounded local edits are blockers.
  Missing optional explanation is not a blocker; {uncertainty_rule}.
- Never include prompts, raw provider output, benchmark identity, oracle answers,
  evaluator data, hidden retries, fallback instructions, or tool execution results.
"""


def _planner_reasoner_prompt_package(
    *,
    user_request: str,
    context: dict[str, Any],
    operations: list[dict[str, Any]],
    source_audio_refs: list[str],
    policy: WorkflowProtocolPolicy | str | None = None,
) -> PlannerPromptPackage:
    prompt_profile = _planner_prompt_profile_from_context(context)
    operation_ids = [
        str(item.get("operation"))
        for item in operations
        if isinstance(item, dict) and item.get("operation")
    ]
    perception_catalog = workflow_perception_catalog_payload(operation_ids)
    perception_operation_ids = {
        str(item["operation"]) for item in perception_catalog if item.get("operation")
    }
    phase = (
        "post_perception" if context.get("workflow_perception_evidence_ids") else "pre_perception"
    )
    system_prompt = _planner_reasoner_system_prompt(
        policy,
        prompt_profile=prompt_profile,
    )
    common_payload = _planner_reasoner_common_current_payload(
        user_request=user_request,
        context=context,
        source_audio_refs=source_audio_refs,
        prompt_profile=prompt_profile,
    )
    skill_context = _planner_reasoner_skill_context(
        user_request=user_request,
        context=context,
        operation_ids=operation_ids,
    )

    if phase == "pre_perception":
        operation_capability_index = [
            item
            for item in compact_operation_capability_index(
                operations,
                perception_operation_ids=perception_operation_ids,
            )
            if item["operation_kind"] == "edit"
        ]
        semantic_effect_index = (
            compact_semantic_effect_capability_index(
                operations,
                perception_operation_ids=perception_operation_ids,
            )
            if prompt_profile in {
                "robust_contract_v4",
                "robust_contract_v5",
                "robust_contract_v6",
            }
            else None
        )
        capability_payload = (
            {"semantic_effect_capability_index": semantic_effect_index}
            if semantic_effect_index is not None
            else {"edit_operation_capability_index": operation_capability_index}
        )
        current_payload = {
            **common_payload,
            "current_call": {
                **common_payload["current_call"],
                "phase": phase,
            },
            "current_task": _pre_perception_current_task(prompt_profile),
            **(
                {}
                if prompt_profile in {"robust_contract_v5", "robust_contract_v6"}
                else {
                    "output_contract": planner_reasoner_turn_contract_outline(
                        phase,
                        prompt_profile=prompt_profile,
                    )
                }
            ),
            "known_context": _planner_known_context(context),
            "capabilities": {
                **capability_payload,
                "perception_capability_catalog": perception_catalog,
                "parameter_contract_policy": (
                    "Full edit parameter contracts are intentionally absent in this "
                    "phase. Nominate candidate operations but do not author a plan."
                ),
            },
            "planner_skill_context": skill_context,
        }
        return build_planner_prompt_package(
            phase=phase,
            prompt_profile=prompt_profile,
            system_prompt=system_prompt,
            current_payload=current_payload,
            operation_index_count=len(operation_capability_index),
            full_capability_card_count=0,
            candidate_projection_source="not_applicable",
            candidate_operations=[],
        )

    frozen_edit_intent = frozen_edit_intent_from_context(context)
    (
        projection_source,
        candidate_operations,
        unavailable_candidates,
        candidate_cards,
    ) = candidate_capability_projection(
        operations,
        perception_operation_ids=perception_operation_ids,
        frozen_edit_intent=frozen_edit_intent,
        prompt_profile=prompt_profile,
    )
    if prompt_profile in {
        "evidence_complete_v3",
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    }:
        candidate_cards = [
            _contextualize_candidate_parameter_domain(card, context=context)
            for card in candidate_cards
        ]
    candidates_without_full_contract = [
        str(card["operation"])
        for card in candidate_cards
        if card.get("capability_card_status") == "missing"
    ]
    reviewed_evidence = _planner_reviewed_evidence_payload(context)
    reviewed_evidence_paths_by_id = reviewed_evidence.pop("reviewed_evidence_paths_by_id")
    current_payload = {
        **common_payload,
        "current_call": {
            **common_payload["current_call"],
            "phase": phase,
        },
        "current_task": _post_perception_current_task(
            candidate_projection_source=projection_source,
            prompt_profile=prompt_profile,
        ),
        **(
            {}
            if prompt_profile in {"robust_contract_v5", "robust_contract_v6"}
            else {
                "output_contract": planner_reasoner_turn_contract_outline(
                    phase,
                    prompt_profile=prompt_profile,
                )
            }
        ),
        "frozen_edit_intent": frozen_edit_intent,
        "candidate_projection": {
            "source": projection_source,
            "candidate_operations": candidate_operations,
            "unavailable_candidate_operations": unavailable_candidates,
            "candidate_operations_without_full_contract": (candidates_without_full_contract),
            "conservative_projection_reason": (
                "The runtime did not supply workflow_frozen_edit_intent, so every "
                "available edit operation remains visible. This is an auditable "
                "completeness fallback, not semantic routing."
                if projection_source == "conservative_all_edit_operations"
                else None
            ),
        },
        "candidate_operation_capability_cards": candidate_cards,
        "reviewed_evidence": reviewed_evidence,
        "reviewed_evidence_paths_by_id": reviewed_evidence_paths_by_id,
        "planner_skill_context": skill_context,
    }
    if prompt_profile in {"robust_contract_v4", "robust_contract_v6"}:
        current_payload["candidate_execution_checklist"] = _candidate_execution_checklist(
            candidate_cards
        )
    if prompt_profile in {
        "evidence_complete_v3",
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    }:
        current_payload["tool_prior_guidance"] = _compact_tool_prior_guidance(context)
    return build_planner_prompt_package(
        phase=phase,
        prompt_profile=prompt_profile,
        system_prompt=system_prompt,
        current_payload=current_payload,
        operation_index_count=0,
        full_capability_card_count=len(candidate_cards),
        candidate_projection_source=projection_source,
        candidate_operations=candidate_operations,
        unavailable_candidate_operations=unavailable_candidates,
        candidate_operations_without_full_contract=(candidates_without_full_contract),
    )


def _planner_reasoner_common_current_payload(
    *,
    user_request: str,
    context: dict[str, Any],
    source_audio_refs: list[str],
    prompt_profile: PlannerPromptProfileName,
) -> dict[str, Any]:
    request = {
        "user_request": user_request,
        "workflow_source_audio_refs": list(source_audio_refs),
        "workflow_reference_audio_refs": list(context.get("reference_audio_paths") or []),
    }
    if prompt_profile == "catalog_strict_v1":
        request["framework_required_objectives"] = _infer_high_level_objectives(user_request)
    return {
        "schema_version": "x-editor.planner-current-task/v1",
        "current_call": {
            "workflow_arm": "W1",
            "planner_turn_index": int(context.get("workflow_planner_call_index") or 1),
        },
        "request": request,
        "deterministic_context": {
            "audio_asset_facts": copy.deepcopy(context.get("audio_asset_facts") or []),
            "frozen_edit_operation_params": copy.deepcopy(
                context.get("workflow_frozen_edit_operation_params") or {}
            ),
            "evidence_required_operations": list(
                context.get("workflow_evidence_required_operations") or []
            ),
        },
        "workflow_protocol": _planner_reasoner_protocol_context(
            context.get("workflow_protocol_policy")
        ),
    }


def _pre_perception_current_task(
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
) -> dict[str, Any]:
    if prompt_profile in {
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    }:
        return {
            "purpose": (
                "Represent the complete request as typed audio effects and request "
                "only the evidence needed to choose a legal implementation later."
            ),
            "required_typed_product": {
                "container": "PlannerReasonerTurn",
                "allowed_decisions": ["request_perception", "cannot_plan"],
                "request_perception_requires": ["edit_intent", "perception_intent"],
                "must_not_emit": ["plan", "execution_semantics", "tool_selection"],
            },
            "ordered_work": [
                (
                    "Build an ordered action ledger before choosing effects. List "
                    "every state-changing action clause as a separate edit_objective "
                    "and record preservation clauses independently. The form 'A, then "
                    "B' requires both A and B; words such as then and resulting declare "
                    "dependency and never replace A with B."
                ),
                (
                    "Resolve the ownership of each requested property before choosing "
                    "an effect. Changing information expressed by spoken words is a "
                    "spoken-content edit: a property of a world object, person, date, "
                    "quantity, place, or event described by speech changes content. A "
                    "property of the voice, room, channel, waveform, or audible "
                    "non-speech event changes the recording signal. Figurative world "
                    "descriptions remain content edits; use an acoustic effect only "
                    "when the recording's sound is the target."
                ),
                (
                    "Preserve every coordinated or ordered action as its own objective. "
                    "A later clause, including one introduced by then, never replaces "
                    "an earlier requested edit."
                ),
                (
                    "For every relative transformation, preserve its direction exactly: "
                    "later is not earlier, more is not less, add is not remove, and "
                    "increase is not decrease. Ask Perception for the original value "
                    "when needed; never quote the instruction itself as source speech."
                ),
                (
                    "Distinguish proposition time from media time. A calendar day, "
                    "date, clock value, age, or duration stated by speech is content "
                    "to replace. Use a timeline effect only when the user changes the "
                    "placement or playback order of existing recorded spans. Words "
                    "such as move, earlier, later, before, and after are insufficient "
                    "by themselves to choose a timeline effect."
                ),
                (
                    "Use quoted_text only for verbatim source wording explicitly "
                    "supplied or already known. A relative instruction, paraphrase, "
                    "role, or desired meaning is a semantic target, not a phrase to "
                    "search for in the audio. Request source_text first and derive "
                    "the new value from that evidence."
                ),
                (
                    "Choose effect_kinds only from semantic_effect_capability_index. "
                    "First choose the audio representation of each objective, then "
                    "consider supporting operations."
                ),
                (
                    "Apply minimum intervention: do not turn speech into a tone, mute, "
                    "deletion, replacement, new voice, or new background unless that "
                    "effect class is requested or required by evidence."
                ),
                (
                    "Retain multiple effect hypotheses only when the request truly "
                    "leaves ownership unresolved between proposition content and the "
                    "recording signal. Shared adjectives or figurative wording alone "
                    "are not ambiguity. Otherwise choose one narrow effect. "
                    "candidate_operations may be empty or contain only operations "
                    "listed under the chosen effects."
                ),
                (
                    "Treat an implicit target as an evidence need, not an effect change. "
                    "Keep the narrow requested effect and ask Perception to resolve the "
                    "word, speaker, event, or span; do not replace it with a broad "
                    "whole-segment acoustic effect."
                ),
                (
                    "Request one Perception operation whose target kinds, evidence "
                    "types, temporal precision, and audio roles cover the unresolved "
                    "questions. Do not author edit parameters."
                ),
            ],
            "submission_check": [
                "every user-requested change appears in edit_objectives",
                "every state-changing action clause has its own objective",
                "semantic referents were classified as spoken content versus recording sound",
                "relative transformations preserve their requested direction",
                "proposition time was not confused with the recording timeline",
                "quoted_text contains only known verbatim source wording",
                "every effect kind exists in semantic_effect_capability_index",
                "preservation clauses rule out incompatible destructive effects",
                "candidate operations, when present, support a declared effect",
                "the selected Perception route covers every declared target",
            ],
        }
    if prompt_profile in {"objective_coverage_v2", "evidence_complete_v3"}:
        return {
            "purpose": (
                "Turn the complete request into atomic edit objectives and request "
                "the evidence needed to plan them without fixing the final tool route."
            ),
            "required_typed_product": {
                "container": "PlannerReasonerTurn",
                "allowed_decisions": ["request_perception", "cannot_plan"],
                "request_perception_requires": ["edit_intent", "perception_intent"],
                "must_not_emit": ["plan", "execution_semantics", "tool_selection"],
            },
            "ordered_work": [
                (
                    "List one edit_objective for every requested semantic change. Do "
                    "not merge lexical, timing, prosody, speaker, and acoustic changes "
                    "when they require different capability effects."
                ),
                (
                    "Choose effect_kinds only from semantic_effects appearing in the "
                    "edit_operation_capability_index. Ensure every objective has at "
                    "least one matching effect kind."
                ),
                (
                    "Classify the requested change at the audio representation level. "
                    "Changing a date, number, name, state, or other meaning expressed "
                    "by spoken words is spoken-content replacement. timeline_reorder "
                    "is only for reusing at least two existing recorded spans in a "
                    "different order without changing what those spans say."
                ),
                (
                    "Record preservation and quality requirements independently from "
                    "the proposed implementation."
                ),
                (
                    "Nominate at least one candidate operation for every objective. "
                    "This list is provisional and may be revised after Perception."
                ),
                (
                    "Do not collapse an evidence-dependent representation ambiguity. "
                    "When the same user objective could plausibly require different "
                    "typed effects depending on the source transcript, speaker, or "
                    "audio structure, include every plausible effect kind and candidate "
                    "operation, record the uncertainty, and request the evidence that "
                    "will distinguish them."
                ),
                (
                    "Describe the evidence needed to resolve targets and legal "
                    "parameters, then choose a Perception operation whose declared "
                    "target kinds, evidence types, and temporal precision cover the "
                    "entire perception_intent."
                ),
                (
                    "Use event_label only for a concrete audible event; quoted_text "
                    "for known verbatim speech; semantic_property for meaning, accent, "
                    "emotion, or timbre; speaker_hint for speaker scope; audio_property "
                    "for acoustic or signal qualities."
                ),
            ],
            "submission_check": [
                "all user-requested effects appear in edit_objectives",
                "all effect_kinds are copied from the capability index",
                "candidate operations collectively cover all objectives",
                "the selected Perception route is compatible with every declared target",
                "primary_audio_ref is present in audio_refs and preserves audio roles",
            ],
        }
    return {
        "purpose": (
            "Understand this request, freeze its typed edit semantics, and decide "
            "what evidence must be collected before execution planning."
        ),
        "required_typed_product": {
            "container": "PlannerReasonerTurn",
            "allowed_decisions": ["request_perception", "cannot_plan"],
            "request_perception_requires": ["edit_intent", "perception_intent"],
            "must_not_emit": ["plan", "execution_semantics", "tool_selection"],
        },
        "stage_instructions": [
            (
                "Freeze effect kinds, target scopes, edit objectives, preservation and "
                "quality requirements, required evidence, uncertainties, and candidate "
                "edit operations in edit_intent."
            ),
            (
                "Choose the required evidence type before choosing one compatible "
                "Perception operation from perception_capability_catalog."
            ),
            (
                "Use event_label only for a concrete audible event. Use quoted_text only "
                "for known verbatim speech; use semantic_property for content classes, "
                "accent, emotion, timbre, or meaning; use speaker_hint for speaker scope; "
                "use audio_property for acoustic or signal qualities."
            ),
            (
                "Copy expected_evidence_fields from the selected Perception capability's "
                "evidence_types and satisfy its target kinds and temporal precision."
            ),
            (
                "Set primary_audio_ref to the one analyzed audio. Add supporting audio_refs "
                "only for a required cross-audio comparison and preserve source/reference "
                "roles."
            ),
            (
                "Nominate downstream edit candidates from the compact index by semantic, "
                "preservation, quality, and evidence fit; never by operation-id wording "
                "alone."
            ),
        ],
    }


def _post_perception_current_task(
    *,
    candidate_projection_source: str,
    prompt_profile: PlannerPromptProfileName = DEFAULT_PLANNER_PROMPT_PROFILE,
) -> dict[str, Any]:
    if prompt_profile == "robust_contract_v4":
        return {
            "purpose": (
                "Use reviewed evidence to cover every frozen objective with legal "
                "single-contract steps and exact provenance."
            ),
            "required_typed_product": {
                "container": "PlannerReasonerTurn",
                "allowed_decisions": ["execute", "cannot_plan"],
                "execute_requires": [
                    "edit_intent",
                    "plan",
                    "execution_semantics",
                    "tool_selection",
                ],
                "cannot_plan_requires": ["missing_information"],
            },
            "candidate_projection_source": candidate_projection_source,
            "ordered_work": [
                "Preserve frozen objectives and preservation requirements.",
                (
                    "Choose only operations whose declared semantic effects cover the "
                    "assigned objective; use preservation, quality, artifact risk, "
                    "stability, then cost as tie-breaks."
                ),
                (
                    "Create one plan step per operation contract and connect transformed "
                    "audio with explicit from_step references."
                ),
                (
                    "For every step, copy all required addresses from "
                    "candidate_execution_checklist into plan inputs/params before "
                    "considering optional addresses."
                ),
                (
                    "Build one complete tool_selection item for every plan step, then "
                    "bind exact evidence-derived leaves in execution_semantics."
                ),
            ],
            "submission_check": [
                "every frozen objective is covered",
                "every selected operation declares the assigned effect",
                "every checklist required address exists with a current legal value",
                "selected_operation appears in a complete candidates array",
                "parameter selections exactly mirror plan values",
                "local boundaries have exact evidence sources",
                "each transformed audio dependency uses from_step and depends_on",
            ],
        }
    if prompt_profile in {"robust_contract_v5", "robust_contract_v6"}:
        return {
            "purpose": (
                "Turn reviewed evidence into the smallest legal execution graph that "
                "covers every frozen objective."
            ),
            "required_typed_product": {
                "container": "PlannerReasonerTurn",
                "allowed_decisions": ["execute", "cannot_plan"],
                "execute_requires": [
                    "edit_intent",
                    "plan",
                    "execution_semantics",
                    "tool_selection",
                ],
                "cannot_plan_requires": ["missing_information"],
                "shape_authority": "API strict JSON schema",
            },
            "candidate_projection_source": candidate_projection_source,
            "ordered_work": [
                (
                    "Copy frozen target_scopes, edit_objectives, preservation_requirements, "
                    "quality_requirements, and required_evidence exactly; do not "
                    "paraphrase, reorder, merge, or omit them."
                ),
                (
                    "Cover each frozen objective with the least destructive shortlisted "
                    "card whose "
                    "semantic effects and scope fit; compare preservation, quality, "
                    "artifact risk, stability, then cost."
                ),
                (
                    "Create one step per selected operation. Before filling parameters, "
                    "draw the audio dependency graph: cumulative steps consume a "
                    "from_step output; only independent branches consume the source."
                ),
                (
                    "For each step, materialize every required address from its "
                    "candidate_execution_checklist using values legal under the "
                    "matching parameter_contract. Add optional addresses only when "
                    "needed. Never emit an optional address with an empty list, empty "
                    "string, or null value; omit that address instead."
                ),
                (
                    "Build one complete tool_selection row per step and mirror the "
                    "plan's selected values exactly. Bind only actual evidence-derived "
                    "leaves to exact reviewed paths."
                ),
            ],
            "submission_check": [
                "all objectives and preservation requirements remain represented",
                "all invariant frozen intent fields are copied exactly",
                "every step uses one declared effect and one legal parameter contract",
                "all cumulative audio edges use matching from_step and depends_on",
                "no audio ref is invented and source/reference roles are preserved",
                "every required value is legal in the current contextualized domain",
                "selected candidates and parameter selections exactly match the plan",
                (
                    "the selected candidate has semantic_fit=full and does not mark "
                    "a must-preserve or must-quality requirement as none"
                ),
                (
                    "execution_semantics has at most one binding per step; one step "
                    "covering multiple targets summarizes them in that binding and "
                    "combines their evidence paths"
                ),
                "every local evidence-derived boundary has an exact source path",
            ],
        }
    if prompt_profile in {"objective_coverage_v2", "evidence_complete_v3"}:
        return {
            "purpose": (
                "Use reviewed evidence to choose legal tools, cover every frozen "
                "objective, and emit one executable dependency graph."
            ),
            "required_typed_product": {
                "container": "PlannerReasonerTurn",
                "allowed_decisions": ["execute", "cannot_plan"],
                "execute_requires": [
                    "edit_intent",
                    "plan",
                    "execution_semantics",
                    "tool_selection",
                ],
                "cannot_plan_requires": ["missing_information"],
            },
            "candidate_projection_source": candidate_projection_source,
            "ordered_work": [
                (
                    "Preserve frozen semantic objectives and requirements. Treat its "
                    "candidate_operations as provisional; revise them from the complete "
                    "cards when evidence shows a better or necessary route."
                ),
                (
                    "Create an objective-to-operation coverage table internally. Reject "
                    "a candidate when its semantic_effects do not implement the assigned "
                    "objective, even if its name sounds related."
                ),
                (
                    "Prefer legal routes in this order: full objective coverage, required "
                    "inputs/evidence, preservation, quality and artifact risk, composition "
                    "stability, then cost and latency."
                ),
                (
                    "Decompose compound edits whenever one card does not declare every "
                    "required effect. Emit one contract per step in topological order."
                ),
                (
                    "Materialize every required planner_selectable address into inputs or "
                    "params. Follow value guidance and array cardinality exactly."
                ),
                (
                    "Connect dependent audio with explicit from_step objects; never invent "
                    "an output ref or filename. depends_on alone does not redirect "
                    "inputs.input_audio away from the original source."
                ),
                (
                    "Bind every evidence-derived scalar leaf to an exact evidence id and "
                    "path. Then mirror each authored plan value exactly once into "
                    "parameter_selections."
                ),
            ],
            "submission_check": [
                "every frozen objective is covered by at least one step",
                "each step effect is declared by its selected capability card",
                "all required parameter addresses exist without duplicated root keys",
                "all values satisfy type, enum, range, cardinality, and value guidance",
                "parameter_selections contain unique paths and equal the plan values",
                "every selected span start_sec and end_sec has an exact evidence source",
                "every future audio input uses from_step and a matching depends_on",
            ],
        }
    return {
        "purpose": (
            "Use reviewed evidence and the candidate operation contracts to produce "
            "one complete executable plan, or explicitly decline when any required "
            "input, evidence, contract, or legal value is missing."
        ),
        "required_typed_product": {
            "container": "PlannerReasonerTurn",
            "allowed_decisions": ["execute", "cannot_plan"],
            "execute_requires": [
                "edit_intent",
                "plan",
                "execution_semantics",
                "tool_selection",
            ],
            "cannot_plan_requires": ["missing_information"],
        },
        "candidate_projection_source": candidate_projection_source,
        "stage_instructions": [
            (
                "Compare only listed candidates in this order: legal inputs and evidence, "
                "complete semantic coverage, preservation, perceptual quality and artifact "
                "risk, composition stability, then cost and latency."
            ),
            (
                "Use the executable plan as the single source of selected values. Mirror "
                "only authored planner_selectable values in parameter_selections and copy "
                "each exact contract path; never author framework_owned fields."
            ),
            (
                "Use exact reviewed evidence values and paths. Bind every semantic or "
                "evidence-dependent scalar argument and leave no unresolved_requirements."
            ),
            (
                "Separate compound edits into dependent single-contract steps in "
                "topological order and use explicit from_step references whenever current "
                "is ambiguous."
            ),
            (
                "If frozen_edit_intent is absent, reconstruct a typed intent from the "
                "public request and reviewed evidence, but do not narrow the conservative "
                "candidate set using guessed keywords."
            ),
        ],
    }


def _compact_tool_prior_guidance(context: dict[str, Any]) -> dict[str, Any] | None:
    value = context.get("workflow_tool_prior_guidance")
    if not isinstance(value, dict):
        return None
    guidance = []
    for raw_item in value.get("guidance") or []:
        if not isinstance(raw_item, dict):
            continue
        guidance.append(
            {
                key: copy.deepcopy(raw_item[key])
                for key in (
                    "guidance_id",
                    "task_scope",
                    "decision",
                    "ordering_semantics",
                    "routes_by_id",
                    "preferred_route_id",
                    "preferred_operations",
                    "prohibition",
                )
                if key in raw_item
            }
        )
    return {
        key: copy.deepcopy(value[key])
        for key in (
            "schema_version",
            "snapshot_id",
            "snapshot_version",
            "snapshot_fingerprint",
            "source_role",
            "source_data_role",
            "eligible_as_final_outcome",
            "guidance_scope",
        )
        if key in value
    } | {"guidance": guidance}


def _contextualize_candidate_parameter_domain(
    card: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    projected = copy.deepcopy(card)
    contract = projected.get("parameter_contract")
    if not isinstance(contract, dict):
        return projected
    parameters = contract.get("planner_selectable")
    if not isinstance(parameters, list):
        return projected
    reference_refs = list(context.get("reference_audio_paths") or [])
    by_path = {
        str(item.get("path")): item
        for item in parameters
        if isinstance(item, dict) and item.get("path")
    }
    unavailable_paths: set[str] = set()
    for path, parameter in by_path.items():
        role = str(parameter.get("semantic_role") or "").lower()
        if "reference" not in role or "audio" not in role:
            continue
        parameter["available_values"] = reference_refs
        if not reference_refs:
            unavailable_paths.add(path)

    narrowed_enums: dict[str, dict[str, Any]] = {}
    for rule in contract.get("conditional_rules") or []:
        if not isinstance(rule, dict):
            continue
        required_paths = {str(path) for path in rule.get("require") or [] if isinstance(path, str)}
        if not required_paths & unavailable_paths:
            continue
        condition = rule.get("when")
        if not isinstance(condition, dict):
            continue
        condition_path = str(condition.get("path") or "")
        parameter = by_path.get(condition_path)
        if parameter is None or not isinstance(parameter.get("enum_values"), list):
            continue
        blocked_values = {
            value
            for value in condition.get("values") or []
            if isinstance(value, (str, int, float, bool))
        }
        catalog_values = list(parameter["enum_values"])
        legal_values = [value for value in catalog_values if value not in blocked_values]
        if legal_values == catalog_values:
            continue
        parameter["catalog_enum_values"] = catalog_values
        parameter["enum_values"] = legal_values
        narrowed_enums[condition_path] = {
            "unavailable_values": [value for value in catalog_values if value in blocked_values],
            "reason": "conditional requirements unavailable in current context",
            "missing_paths": sorted(required_paths & unavailable_paths),
        }
    projected["current_context_parameter_constraints"] = {
        "workflow_reference_audio_refs": reference_refs,
        "unavailable_reference_paths": sorted(unavailable_paths),
        "narrowed_enum_domains": narrowed_enums,
    }
    return projected


def _candidate_execution_checklist(
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project the legal authoring surface that must be checked before execute."""

    checklists: list[dict[str, Any]] = []
    for card in cards:
        contract = card.get("parameter_contract")
        if not isinstance(contract, dict):
            continue
        addresses: list[dict[str, Any]] = []
        for parameter in contract.get("planner_selectable") or []:
            if not isinstance(parameter, dict) or not parameter.get("path"):
                continue
            addresses.append(parameter)
        checklists.append(
            {
                "operation": str(card.get("operation") or ""),
                "legal_planner_selectable_addresses": [str(item["path"]) for item in addresses],
                "required_addresses": [
                    str(item["path"]) for item in addresses if item.get("required") is True
                ],
                "conditional_rules": copy.deepcopy(contract.get("conditional_rules") or []),
                "framework_owned_addresses": copy.deepcopy(contract.get("framework_owned") or []),
            }
        )
    return checklists


def _planner_prompt_profile_from_context(
    context: dict[str, Any],
) -> PlannerPromptProfileName:
    value = str(context.get("workflow_planner_prompt_profile") or DEFAULT_PLANNER_PROMPT_PROFILE)
    if value not in {
        "catalog_strict_v1",
        "objective_coverage_v2",
        "evidence_complete_v3",
        "robust_contract_v4",
        "robust_contract_v5",
        "robust_contract_v6",
    }:
        raise OpenAICompatiblePlannerContractError(f"unknown Planner prompt profile: {value}")
    return value  # type: ignore[return-value]


def _planner_known_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(context[key])
        for key in (
            "source_text",
            "source_text_audio_index",
            "source_text_audio_path",
        )
        if context.get(key) is not None
    }


def _planner_reviewed_evidence_payload(
    context: dict[str, Any],
) -> dict[str, Any]:
    reviewed_context = {
        key: copy.deepcopy(context[key])
        for key in (
            "perception_report",
            "source_text",
            "source_text_audio_index",
            "source_text_audio_path",
            "span_selection_analysis",
        )
        if context.get(key) is not None
    }
    path_sources = context.get("workflow_evidence_path_sources") or {}
    paths_by_evidence_id: dict[str, list[str]] = {}
    for raw_path, raw_evidence_ids in path_sources.items():
        if not isinstance(raw_evidence_ids, list):
            continue
        for raw_evidence_id in raw_evidence_ids:
            evidence_id = str(raw_evidence_id)
            paths_by_evidence_id.setdefault(evidence_id, []).append(str(raw_path))
    return {
        "evidence_ids": list(context.get("workflow_perception_evidence_ids") or []),
        "projection_fingerprint": context.get("workflow_evidence_projection_fingerprint"),
        "reviewed_evidence_paths_by_id": {
            evidence_id: sorted(dict.fromkeys(paths))
            for evidence_id, paths in sorted(paths_by_evidence_id.items())
        },
        "facts": reviewed_context,
    }


def _planner_reasoner_skill_context(
    *,
    user_request: str,
    context: dict[str, Any],
    operation_ids: list[str],
) -> list[dict[str, Any]]:
    try:
        return routing_policy_context(
            user_request,
            capabilities=operation_ids,
            reference_audio_paths=list(context.get("reference_audio_paths") or []),
            limit=4,
        )
    except (OSError, RuntimeError, ValueError):
        return []


def _planner_reasoner_protocol_context(
    policy: WorkflowProtocolPolicy | str | None = None,
) -> dict[str, Any]:
    resolved = resolve_workflow_protocol_policy(policy)
    return {
        "schema_version": "x-editor.reasoned-workflow-protocol/v3",
        "selected_profile": resolved.model_dump(mode="json"),
        "normalization_policy": {
            "binding": resolved.binding_normalization_mode,
            "duplicate_optional_list_item": "deduplicate",
            "framework_owned_plan_fields": (resolved.framework_owned_plan_fields_mode),
            "irrelevant_binding": resolved.irrelevant_binding_mode,
            "ambiguous_value_or_alias": "block",
        },
        "perception_target_contract": {
            "event_label": "conditional_for_concrete_audible_event_detection",
            "quoted_text": "speech_or_lexical_target",
            "speaker_hint": "speaker_identity_role_or_order_target",
            "semantic_property": "accent_emotion_timbre_or_meaning_target",
            "audio_property": "acoustic_or_signal_property_target",
        },
        "admission_policy": {
            "optional_missing_or_needs_review": (resolved.reviewable_uncertainty_mode),
            "slice_shortfall": resolved.slice_shortfall_mode,
            "required_missing_or_unsafe_for_execution": "block",
            "unknown_evidence_or_value_mismatch": "block",
            "local_destructive_edit_without_boundaries": "block",
        },
    }


def _normalize_planner_reasoner_payload(
    payload: Any,
    *,
    policy: WorkflowProtocolPolicy | str | None = None,
    evidence_required_operations: set[str] | None = None,
) -> tuple[Any, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    resolved = resolve_workflow_protocol_policy(policy)
    normalized = copy.deepcopy(payload)
    corrections: list[str] = []
    perception_intent = normalized.get("perception_intent")
    if isinstance(perception_intent, dict):
        candidate_operations = perception_intent.get("candidate_operations")
        if isinstance(candidate_operations, list):
            misplaced = [
                str(operation)
                for operation in candidate_operations
                if str(operation).startswith(("perception.", "flexsed."))
            ]
            if misplaced:
                if resolved.perception_route_normalization_mode == "explicit_only":
                    raise ValueError(
                        "candidate_operations contains Perception operations "
                        f"under {resolved.profile_name}: {misplaced}"
                    )
                perception_intent["candidate_operations"] = [
                    operation
                    for operation in candidate_operations
                    if not str(operation).startswith(("perception.", "flexsed."))
                ]
                corrections.append(
                    "dropped_misplaced_perception_candidate_operations:" + ",".join(misplaced)
                )
    plan = normalized.get("plan")
    if isinstance(plan, dict):
        if "schema_version" in plan:
            if resolved.framework_owned_plan_fields_mode == "block":
                raise ValueError("AudioEditPlan does not declare plan.schema_version")
            plan.pop("schema_version", None)
            corrections.append("stripped_known_extra_plan_schema_version")
        for field_name in ("global_params", "metadata"):
            value = plan.get(field_name)
            if value:
                if resolved.framework_owned_plan_fields_mode == "block":
                    raise ValueError(
                        "Planner authored framework-owned plan field under "
                        f"{resolved.profile_name}: {field_name}"
                    )
                corrections.append(f"stripped_model_authored_plan_{field_name}")
            plan[field_name] = {}
        for field_name in ("input_audio", "output_audio"):
            value = plan.get(field_name)
            if value in (None, "") or isinstance(value, str):
                continue
            if resolved.binding_normalization_mode != "exact_unique":
                continue
            plan[field_name] = None
            corrections.append(f"dropped_non_string_plan_{field_name}")
    steps_by_id = {
        str(step.get("id")): step
        for step in (plan.get("steps") if isinstance(plan, dict) else []) or []
        if isinstance(step, dict) and step.get("id")
    }
    tool_selection = normalized.get("tool_selection")
    if isinstance(tool_selection, dict):
        for selection_index, selection in enumerate(tool_selection.get("steps") or []):
            if not isinstance(selection, dict):
                continue
            for candidate_index, candidate in enumerate(selection.get("candidates") or []):
                if not isinstance(candidate, dict):
                    continue
                for fit_field in (
                    "semantic_fit",
                    "preservation_fit",
                    "quality_fit",
                ):
                    fit_value = candidate.get(fit_field)
                    if fit_value in {"full", "partial", "none", "unknown"}:
                        continue
                    candidate[fit_field] = "unknown"
                    corrections.append(
                        "normalized_non_relative_candidate_fit_to_unknown:"
                        f"selection_{selection_index}:candidate_{candidate_index}:"
                        f"{fit_field}:{fit_value}"
                    )
            step_id = str(selection.get("plan_step_id") or "")
            step = steps_by_id.get(step_id)
            if not isinstance(step, dict):
                continue
            selected_operation = str(selection.get("selected_operation") or "")
            retained_parameters: list[Any] = []
            for parameter_index, parameter in enumerate(
                selection.get("parameter_selections") or []
            ):
                if not isinstance(parameter, dict):
                    retained_parameters.append(parameter)
                    continue
                argument_path = str(parameter.get("argument_path") or "")
                found, selected_value = _raw_step_argument_value(
                    step,
                    argument_path,
                )
                if (
                    not found
                    and selected_operation == str(step.get("operation") or "")
                    and _set_missing_raw_step_argument_value(
                        step,
                        argument_path,
                        parameter.get("selected_value"),
                    )
                ):
                    found = True
                    selected_value = parameter.get("selected_value")
                    corrections.append(
                        "projected_missing_plan_argument_from_selection:"
                        f"{step_id}:parameter_{parameter_index}:{argument_path}"
                    )
                if not found:
                    corrections.append(
                        "dropped_parameter_selection_absent_from_plan:"
                        f"{step_id}:parameter_{parameter_index}:{argument_path}"
                    )
                    continue
                if parameter.get("contract_operation") != selected_operation:
                    parameter["contract_operation"] = selected_operation
                    corrections.append(
                        "synchronized_parameter_contract_from_selection:"
                        f"{step_id}:parameter_{parameter_index}:{argument_path}"
                    )
                if parameter.get("selected_value") != selected_value:
                    parameter["selected_value"] = copy.deepcopy(selected_value)
                    corrections.append(
                        "synchronized_parameter_selection_from_plan:"
                        f"{step_id}:parameter_{parameter_index}:{argument_path}"
                    )
                retained_parameters.append(parameter)
            selection["parameter_selections"] = retained_parameters
    semantics = normalized.get("execution_semantics")
    bindings = semantics.get("bindings") if isinstance(semantics, dict) else None
    if isinstance(bindings, list):
        if resolved.irrelevant_binding_mode == "drop" and isinstance(plan, dict):
            retained_bindings: list[Any] = []
            for binding_index, binding in enumerate(bindings):
                if not isinstance(binding, dict):
                    retained_bindings.append(binding)
                    continue
                step_id = str(binding.get("plan_step_id") or "")
                step = steps_by_id.get(step_id)
                operation = str(step.get("operation") or "") if isinstance(step, dict) else ""
                route_candidate = operation in TRANSCRIPT_REQUIRED_EDIT_DATA_PIPELINE_OPERATIONS
                evidence_ids = binding.get("evidence_ids")
                evidence_paths = binding.get("evidence_paths")
                argument_sources = binding.get("argument_sources")
                has_evidence_argument_source = any(
                    isinstance(source, dict)
                    and source.get("source_kind")
                    in {"perception_evidence", "derived_from_evidence"}
                    for source in (argument_sources if isinstance(argument_sources, list) else [])
                )
                has_user_request_argument_source = any(
                    isinstance(source, dict) and source.get("source_kind") == "user_request"
                    for source in (argument_sources if isinstance(argument_sources, list) else [])
                )
                explicitly_empty_evidence = (
                    isinstance(evidence_ids, list)
                    and not evidence_ids
                    and isinstance(evidence_paths, list)
                    and not evidence_paths
                )
                if (
                    explicitly_empty_evidence
                    and not has_evidence_argument_source
                    and (not has_user_request_argument_source or route_candidate)
                    and not (binding.get("unresolved_requirements") or [])
                ):
                    inherited = _unique_same_target_evidence_declaration(
                        bindings,
                        binding_index=binding_index,
                    )
                    inheritance_reason = "inherited_unique_same_target_evidence_declaration"
                    if inherited is None and route_candidate:
                        inherited = _unique_compound_dependency_evidence_declaration(
                            bindings,
                            binding_index=binding_index,
                            steps_by_id=steps_by_id,
                        )
                        inheritance_reason = (
                            "inherited_unique_compound_dependency_evidence_declaration"
                        )
                    if route_candidate and inherited is not None:
                        inherited_ids, inherited_paths, donor_indices = inherited
                        binding["evidence_ids"] = inherited_ids
                        binding["evidence_paths"] = inherited_paths
                        corrections.append(
                            f"{inheritance_reason}:"
                            f"binding_{binding_index}:donors_"
                            + "_".join(str(index) for index in donor_indices)
                        )
                    else:
                        corrections.append(
                            "dropped_empty_raw_binding:"
                            f"binding_{binding_index}:{step_id or '<missing>'}:"
                            f"{operation or '<absent_step>'}"
                        )
                        continue
                if step is None or (
                    evidence_required_operations is not None
                    and operation not in evidence_required_operations
                    and not route_candidate
                ):
                    corrections.append(
                        "dropped_irrelevant_raw_binding:"
                        f"binding_{binding_index}:{step_id or '<missing>'}:"
                        f"{operation or '<absent_step>'}"
                    )
                    continue
                if (
                    evidence_required_operations is not None
                    and operation not in evidence_required_operations
                    and route_candidate
                ):
                    corrections.append(
                        "retained_route_candidate_raw_binding:"
                        f"binding_{binding_index}:{step_id}:{operation}"
                    )
                retained_bindings.append(binding)
            semantics["bindings"] = bindings = retained_bindings
        for binding_index, binding in enumerate(bindings):
            if not isinstance(binding, dict):
                continue
            evidence_ids = binding.get("evidence_ids")
            evidence_paths = binding.get("evidence_paths")
            argument_sources = binding.get("argument_sources")
            if not isinstance(evidence_ids, list):
                evidence_ids = []
                binding["evidence_ids"] = evidence_ids
            if not isinstance(evidence_paths, list):
                evidence_paths = []
                binding["evidence_paths"] = evidence_paths
            if not isinstance(argument_sources, list):
                continue
            step = steps_by_id.get(str(binding.get("plan_step_id") or ""))
            if resolved.binding_normalization_mode != "explicit_only" and isinstance(step, dict):
                step_inputs = step.get("inputs")
                step_params = step.get("params")
                input_keys = set(step_inputs) if isinstance(step_inputs, dict) else set()
                param_keys = set(step_params) if isinstance(step_params, dict) else set()
                for source_index, source in enumerate(argument_sources):
                    if not isinstance(source, dict):
                        continue
                    argument_path = str(source.get("argument_path") or "").strip()
                    if argument_path.startswith(("inputs.", "params.")):
                        continue
                    root = argument_path.split(".", 1)[0].split("[", 1)[0]
                    containers = [
                        name
                        for name, keys in (
                            ("inputs", input_keys),
                            ("params", param_keys),
                        )
                        if root in keys
                    ]
                    if len(containers) != 1:
                        continue
                    normalized_path = f"{containers[0]}.{argument_path}"
                    source["argument_path"] = normalized_path
                    corrections.append(
                        "prefixed_unique_argument_path:"
                        f"binding_{binding_index}:source_{source_index}:"
                        f"{normalized_path}"
                    )
            if resolved.binding_normalization_mode == "exact_unique":
                _normalize_raw_audio_argument_source_roles(
                    argument_sources,
                    binding_index=binding_index,
                    corrections=corrections,
                )
                _fill_unique_argument_source_evidence_refs(
                    argument_sources,
                    evidence_ids=evidence_ids,
                    evidence_paths=evidence_paths,
                    binding_index=binding_index,
                    corrections=corrections,
                )
                argument_sources[:] = _drop_incomplete_raw_evidence_argument_sources(
                    argument_sources,
                    binding_index=binding_index,
                    corrections=corrections,
                )
            declared_pairs = {
                (
                    str(item.get("evidence_id") or ""),
                    str(item.get("field_path") or ""),
                )
                for item in evidence_paths
                if isinstance(item, dict)
            }
            for source in argument_sources:
                if not isinstance(source, dict):
                    continue
                if source.get("source_kind") not in {
                    "perception_evidence",
                    "derived_from_evidence",
                }:
                    continue
                evidence_id = str(source.get("evidence_id") or "").strip()
                field_path = str(source.get("evidence_path") or "").strip()
                if not evidence_id or not field_path:
                    continue
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
                    corrections.append(
                        "declared_argument_source_evidence_id:"
                        f"binding_{binding_index}:{evidence_id}"
                    )
                pair = (evidence_id, field_path)
                if pair in declared_pairs:
                    continue
                evidence_paths.append(
                    {
                        "evidence_id": evidence_id,
                        "field_path": field_path,
                    }
                )
                declared_pairs.add(pair)
                corrections.append(
                    "declared_argument_source_evidence_path:"
                    f"binding_{binding_index}:{evidence_id}:{field_path}"
                )
    _normalize_execute_with_unresolved_requirements(
        normalized,
        policy=resolved,
        corrections=corrections,
    )
    _normalize_incomplete_execute_decision(
        normalized,
        policy=resolved,
        corrections=corrections,
    )
    _drop_redundant_final_perception_steps(
        normalized,
        policy=resolved,
        corrections=corrections,
    )
    return normalized, corrections


_V6_AUTHORITATIVE_FROZEN_INTENT_FIELDS = (
    "schema_version",
    "effect_kinds",
    "target_scopes",
    "edit_objectives",
    "preservation_requirements",
    "quality_requirements",
    "required_evidence",
    "candidate_operations",
)


def _restore_authoritative_frozen_intent(
    payload: Any,
    *,
    frozen_edit_intent: dict[str, Any] | None,
    prompt_profile: PlannerPromptProfileName,
    corrections: list[str],
) -> None:
    """Restore immutable cross-call intent fields without semantic inference."""

    if (
        prompt_profile != "robust_contract_v6"
        or not isinstance(payload, dict)
        or payload.get("decision") != "execute"
        or not isinstance(frozen_edit_intent, dict)
    ):
        return
    edit_intent = payload.get("edit_intent")
    if not isinstance(edit_intent, dict):
        return
    for field_name in _V6_AUTHORITATIVE_FROZEN_INTENT_FIELDS:
        if field_name not in frozen_edit_intent:
            continue
        authoritative = frozen_edit_intent[field_name]
        if edit_intent.get(field_name) == authoritative:
            continue
        edit_intent[field_name] = copy.deepcopy(authoritative)
        corrections.append(f"restored_authoritative_frozen_edit_intent:{field_name}")


def _normalize_incomplete_execute_decision(
    payload: dict[str, Any],
    *,
    policy: WorkflowProtocolPolicy,
    corrections: list[str],
) -> None:
    """Turn an explicitly incomplete execute decision into a typed decline."""

    if (
        payload.get("decision") != "execute"
        or policy.profile_name != "flow_safe"
        or payload.get("plan") is not None
        or payload.get("execution_semantics") is not None
        or payload.get("missing_information") not in (None, [])
    ):
        return
    payload["decision"] = "cannot_plan"
    payload["edit_intent"] = None
    payload["perception_intent"] = None
    payload["plan"] = None
    payload["execution_semantics"] = None
    payload["tool_selection"] = None
    payload["missing_information"] = [
        ("Planner execute decision omitted required executable plan and execution semantics")
    ]
    corrections.append(
        "converted_incomplete_execute_to_cannot_plan:executable_plan,execution_semantics"
    )


def _raw_step_argument_value(
    step: dict[str, Any],
    argument_path: str,
) -> tuple[bool, Any]:
    parts = argument_path.split(".")
    if not parts or parts[0] not in {"inputs", "params"}:
        return False, None
    current: Any = step.get(parts[0])
    for part in parts[1:]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def _set_missing_raw_step_argument_value(
    step: dict[str, Any],
    argument_path: str,
    value: Any,
) -> bool:
    """Project one exact top-level mirror value without semantic inference."""

    parts = argument_path.split(".")
    if len(parts) != 2 or parts[0] != "inputs" or not parts[1]:
        return False
    container = step.get(parts[0])
    if container is None:
        container = {}
        step[parts[0]] = container
    if not isinstance(container, dict) or parts[1] in container:
        return False
    container[parts[1]] = copy.deepcopy(value)
    return True


def _unique_same_target_evidence_declaration(
    bindings: list[Any],
    *,
    binding_index: int,
) -> tuple[list[Any], list[Any], list[int]] | None:
    binding = bindings[binding_index]
    if not isinstance(binding, dict):
        return None
    target = " ".join(str(binding.get("target") or "").casefold().split())
    if not target:
        return None

    candidates: dict[
        tuple[tuple[str, ...], tuple[tuple[str, str], ...]],
        tuple[list[Any], list[Any], list[int]],
    ] = {}
    for donor_index, donor in enumerate(bindings):
        if donor_index == binding_index or not isinstance(donor, dict):
            continue
        donor_target = " ".join(str(donor.get("target") or "").casefold().split())
        if donor_target != target:
            continue
        evidence_ids = donor.get("evidence_ids")
        evidence_paths = donor.get("evidence_paths")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            continue
        if not isinstance(evidence_paths, list) or not evidence_paths:
            continue
        path_pairs = []
        for item in evidence_paths:
            if not isinstance(item, dict):
                break
            evidence_id = str(item.get("evidence_id") or "").strip()
            field_path = str(item.get("field_path") or "").strip()
            if not evidence_id or not field_path:
                break
            path_pairs.append((evidence_id, field_path))
        else:
            signature = (
                tuple(sorted(str(item) for item in evidence_ids)),
                tuple(sorted(path_pairs)),
            )
            existing = candidates.get(signature)
            if existing is None:
                candidates[signature] = (
                    copy.deepcopy(evidence_ids),
                    copy.deepcopy(evidence_paths),
                    [donor_index],
                )
            else:
                existing[2].append(donor_index)
    if len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def _unique_compound_dependency_evidence_declaration(
    bindings: list[Any],
    *,
    binding_index: int,
    steps_by_id: dict[str, Any],
) -> tuple[list[Any], list[Any], list[int]] | None:
    binding = bindings[binding_index]
    if not isinstance(binding, dict):
        return None
    step_id = str(binding.get("plan_step_id") or "")
    step = steps_by_id.get(step_id)
    if not isinstance(step, dict):
        return None
    if str(step.get("operation") or "") not in {
        "edit_data_pipeline.pitch_rate_edit",
        "edit_data_pipeline.speech_rate",
    }:
        return None

    producer_ids = {str(item) for item in step.get("depends_on") or [] if str(item)}

    def collect_from_steps(value: Any) -> None:
        if isinstance(value, dict):
            producer_id = str(value.get("from_step") or "")
            if producer_id:
                producer_ids.add(producer_id)
            for child in value.values():
                collect_from_steps(child)
        elif isinstance(value, list):
            for child in value:
                collect_from_steps(child)

    collect_from_steps(step.get("inputs"))
    if len(producer_ids) != 1:
        return None
    producer_id = next(iter(producer_ids))
    producer = steps_by_id.get(producer_id)
    if not isinstance(producer, dict) or str(producer.get("operation") or "") != (
        "edit_data_pipeline.emotion_style_edit"
    ):
        return None

    donor_candidates = []
    for donor_index, donor in enumerate(bindings):
        if not isinstance(donor, dict):
            continue
        if str(donor.get("plan_step_id") or "") != producer_id:
            continue
        evidence_ids = donor.get("evidence_ids")
        evidence_paths = donor.get("evidence_paths")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            continue
        if not isinstance(evidence_paths, list) or not evidence_paths:
            continue
        donor_candidates.append(
            (
                copy.deepcopy(evidence_ids),
                copy.deepcopy(evidence_paths),
                [donor_index],
            )
        )
    if len(donor_candidates) != 1:
        return None
    return donor_candidates[0]


def _normalize_raw_audio_argument_source_roles(
    argument_sources: list[Any],
    *,
    binding_index: int,
    corrections: list[str],
) -> None:
    """Keep workflow audio references out of evidence provenance rows."""

    audio_leaves = {
        "audio",
        "audio_path",
        "input_audio",
        "input_audio_path",
        "source_audio",
        "source_audio_path",
    }
    for source_index, source in enumerate(argument_sources):
        if not isinstance(source, dict) or source.get("source_kind") not in {
            "perception_evidence",
            "derived_from_evidence",
        }:
            continue
        argument_path = str(source.get("argument_path") or "").strip()
        argument_leaf = re.split(r"[.\[\]]+", argument_path)[-1]
        if argument_leaf not in audio_leaves:
            continue
        source["source_kind"] = "deterministic_context"
        source.pop("evidence_id", None)
        source.pop("evidence_path", None)
        corrections.append(
            "normalized_audio_argument_source_to_deterministic_context:"
            f"binding_{binding_index}:source_{source_index}:{argument_path}"
        )


def _fill_unique_argument_source_evidence_refs(
    argument_sources: list[Any],
    *,
    evidence_ids: list[Any],
    evidence_paths: list[Any],
    binding_index: int,
    corrections: list[str],
) -> None:
    declared_ids = [str(item).strip() for item in evidence_ids if str(item).strip()]
    unique_declared_id = declared_ids[0] if len(set(declared_ids)) == 1 else ""
    declared_refs: list[tuple[str, str]] = []
    for item in evidence_paths:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        evidence_id = str(item.get("evidence_id") or unique_declared_id).strip()
        if evidence_id and field_path:
            declared_refs.append((evidence_id, field_path))

    for source_index, source in enumerate(argument_sources):
        if not isinstance(source, dict) or source.get("source_kind") not in {
            "perception_evidence",
            "derived_from_evidence",
        }:
            continue
        evidence_id = str(source.get("evidence_id") or "").strip()
        evidence_path = str(source.get("evidence_path") or "").strip()
        if evidence_id and evidence_path:
            continue
        candidates = list(declared_refs)
        if evidence_id:
            candidates = [item for item in candidates if item[0] == evidence_id]
        if evidence_path:
            candidates = [item for item in candidates if item[1] == evidence_path]
        else:
            argument_path = str(source.get("argument_path") or "").strip()
            argument_leaf = re.split(r"[.\[\]]+", argument_path)[-1]
            if argument_leaf:
                candidates = [item for item in candidates if item[1].endswith(f".{argument_leaf}")]
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            continue
        candidate_id, candidate_path = candidates[0]
        if not evidence_id:
            source["evidence_id"] = candidate_id
        if not evidence_path:
            source["evidence_path"] = candidate_path
        corrections.append(
            "filled_unique_argument_source_evidence_ref:"
            f"binding_{binding_index}:source_{source_index}:"
            f"{candidate_id}:{candidate_path}"
        )


def _drop_incomplete_raw_evidence_argument_sources(
    argument_sources: list[Any],
    *,
    binding_index: int,
    corrections: list[str],
) -> list[Any]:
    """Defer incomplete provenance rows to exact value matching at admission."""

    retained: list[Any] = []
    for source_index, source in enumerate(argument_sources):
        if not isinstance(source, dict) or source.get("source_kind") not in {
            "perception_evidence",
            "derived_from_evidence",
        }:
            retained.append(source)
            continue
        evidence_id = str(source.get("evidence_id") or "").strip()
        evidence_path = str(source.get("evidence_path") or "").strip()
        if evidence_id and evidence_path:
            retained.append(source)
            continue
        corrections.append(
            "dropped_incomplete_raw_evidence_argument_source:"
            f"binding_{binding_index}:source_{source_index}:"
            f"{source.get('argument_path') or '<missing>'}"
        )
    return retained


def _normalize_execute_with_unresolved_requirements(
    payload: dict[str, Any],
    *,
    policy: WorkflowProtocolPolicy,
    corrections: list[str],
) -> None:
    """Turn an explicit model-declared inability into a valid cannot_plan turn."""

    if payload.get("decision") != "execute" or policy.binding_normalization_mode != "exact_unique":
        return
    semantics = payload.get("execution_semantics")
    bindings = semantics.get("bindings") if isinstance(semantics, dict) else None
    if not isinstance(bindings, list):
        return
    unresolved = [
        str(requirement).strip()
        for binding in bindings
        if isinstance(binding, dict)
        for requirement in binding.get("unresolved_requirements") or []
        if str(requirement).strip()
    ]
    if not unresolved:
        return
    existing = payload.get("missing_information")
    missing = [
        str(item).strip()
        for item in (existing if isinstance(existing, list) else [])
        if str(item).strip()
    ]
    payload["decision"] = "cannot_plan"
    payload["perception_intent"] = None
    payload["plan"] = None
    payload["execution_semantics"] = None
    payload["missing_information"] = list(dict.fromkeys([*missing, *unresolved]))
    if not str(payload.get("decision_summary") or "").strip():
        payload["decision_summary"] = (
            "The reviewed evidence does not satisfy the declared execution requirements."
        )
    corrections.append("converted_execute_with_unresolved_requirements_to_cannot_plan")


def _normalize_execute_with_outside_catalog_operations(
    payload: dict[str, Any],
    *,
    available_operation_ids: set[str],
    corrections: list[str],
) -> None:
    """Convert only a typed execute decision that crossed the frozen catalog."""

    if payload.get("decision") != "execute":
        return
    plan = payload.get("plan")
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return
    outside_catalog = sorted(
        {
            str(step.get("operation") or "").strip()
            for step in steps
            if isinstance(step, dict)
            and str(step.get("operation") or "").strip()
            and str(step.get("operation") or "").strip() not in available_operation_ids
        }
    )
    if not outside_catalog:
        return
    payload["decision"] = "cannot_plan"
    payload["perception_intent"] = None
    payload["plan"] = None
    payload["execution_semantics"] = None
    payload["missing_information"] = [
        f"selected operation outside frozen catalog: {operation}" for operation in outside_catalog
    ]
    payload["decision_summary"] = (
        "The proposed execution plan selected an operation outside the frozen "
        "catalog, so no executable plan was admitted."
    )
    corrections.append(
        "converted_execute_with_outside_catalog_operation_to_cannot_plan:"
        + ",".join(outside_catalog)
    )


def _normalize_plan_with_outside_catalog_operations(
    payload: Any,
    *,
    available_operation_ids: set[str],
) -> tuple[Any, list[dict[str, Any]]]:
    """Turn a valid ordinary plan that crossed its catalog into a decline.

    Schema-invalid payloads stay on the existing error/repair path. This guard
    handles only the capability-boundary mismatch and cannot hide a separate
    planner contract failure.
    """

    try:
        plan = AudioEditPlan.model_validate(payload)
    except ValidationError:
        return payload, []
    outside_catalog = sorted(
        {
            step.operation.strip()
            for step in plan.steps
            if step.operation.strip() and step.operation.strip() not in available_operation_ids
        }
    )
    if not outside_catalog:
        return payload, []

    audit_record = {
        "normalization_id": "open_provider_outside_catalog_typed_decline_v1",
        "reason": "provider_selected_operation_outside_available_operations",
        "outside_catalog_operations": outside_catalog,
        "available_operation_ids": sorted(available_operation_ids),
    }
    metadata = copy.deepcopy(plan.metadata)
    existing = metadata.get("planner_catalog_normalizations")
    if not isinstance(existing, list):
        existing = []
    existing.append(copy.deepcopy(audit_record))
    metadata["planner_catalog_normalizations"] = existing

    missing_inputs = list(plan.missing_inputs)
    missing_inputs.extend(
        f"selected operation outside frozen catalog: {operation}" for operation in outside_catalog
    )
    normalized = plan.model_dump(mode="python")
    normalized["steps"] = []
    normalized["missing_inputs"] = list(dict.fromkeys(missing_inputs))
    normalized["metadata"] = metadata
    return normalized, [audit_record]


def _drop_redundant_final_perception_steps(
    payload: dict[str, Any],
    *,
    policy: WorkflowProtocolPolicy,
    corrections: list[str],
) -> None:
    """Remove already-satisfied Perception rows from a flow-safe final plan."""

    if payload.get("decision") != "execute" or policy.binding_normalization_mode != "exact_unique":
        return
    plan = payload.get("plan")
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return
    perception_ids = {
        str(step.get("id"))
        for step in steps
        if isinstance(step, dict)
        and step.get("id")
        and str(step.get("operation") or "").startswith(("perception.", "flexsed."))
    }
    if not perception_ids:
        return
    edit_steps = [
        step
        for step in steps
        if not (isinstance(step, dict) and str(step.get("id") or "") in perception_ids)
    ]
    if not edit_steps or any(
        _payload_consumes_step_output(step, perception_ids) for step in edit_steps
    ):
        return
    for step in edit_steps:
        if not isinstance(step, dict):
            continue
        depends_on = step.get("depends_on")
        if isinstance(depends_on, list):
            step["depends_on"] = [item for item in depends_on if str(item) not in perception_ids]
    plan["steps"] = edit_steps
    semantics = payload.get("execution_semantics")
    bindings = semantics.get("bindings") if isinstance(semantics, dict) else None
    if isinstance(bindings, list):
        semantics["bindings"] = [
            item
            for item in bindings
            if not (
                isinstance(item, dict) and str(item.get("plan_step_id") or "") in perception_ids
            )
        ]
    corrections.append(
        "dropped_redundant_final_perception_steps:" + ",".join(sorted(perception_ids))
    )


def _payload_consumes_step_output(
    value: Any,
    producer_ids: set[str],
    *,
    field_name: str | None = None,
) -> bool:
    if isinstance(value, dict):
        from_step = value.get("from_step")
        if from_step is not None and str(from_step) in producer_ids:
            return True
        return any(
            _payload_consumes_step_output(
                child,
                producer_ids,
                field_name=str(key),
            )
            for key, child in value.items()
            if key != "depends_on"
        )
    if isinstance(value, list):
        return any(
            _payload_consumes_step_output(
                child,
                producer_ids,
                field_name=field_name,
            )
            for child in value
        )
    if isinstance(value, str) and field_name in {
        "audio_path",
        "audio_paths",
        "input_audio",
        "input_audio_path",
    }:
        return any(
            value == producer_id
            or value.startswith((f"{producer_id}:", f"{producer_id}."))
            for producer_id in producer_ids
        )
    return False


def _synchronize_normalized_binding_operations(
    *,
    original_plan: AudioEditPlan,
    normalized_plan: AudioEditPlan,
    semantics: PlannerExecutionSemantics | None,
) -> tuple[PlannerExecutionSemantics | None, list[str]]:
    """Keep binding operation names aligned with audited plan rewrites."""

    if semantics is None:
        return None, []
    original_by_id = {step.id: step.operation for step in original_plan.steps}
    normalized_steps_by_id = {step.id: step for step in normalized_plan.steps}
    normalized_by_id = {step_id: step.operation for step_id, step in normalized_steps_by_id.items()}
    corrections: list[str] = []
    bindings = []
    for binding in semantics.bindings:
        original_operation = original_by_id.get(binding.plan_step_id)
        normalized_operation = normalized_by_id.get(binding.plan_step_id)
        if (
            original_operation is not None
            and normalized_operation is not None
            and original_operation != normalized_operation
            and binding.operation == original_operation
        ):
            bindings.append(
                binding.model_copy(
                    update={"operation": normalized_operation},
                    deep=True,
                )
            )
            corrections.append(
                "synchronized_normalized_binding_operation:"
                f"{binding.plan_step_id}:{original_operation}:"
                f"{normalized_operation}"
            )
            continue
        bindings.append(binding.model_copy(deep=True))
    bindings_by_step = {binding.plan_step_id: binding for binding in bindings}
    routing_corrections = normalized_plan.metadata.get("routing_boundary_corrections")
    if isinstance(routing_corrections, list):
        for item in routing_corrections:
            if not isinstance(item, dict) or item.get("reason") != (
                "explicit_compound_emotion_rate_split_into_scoped_steps"
            ):
                continue
            source_step_id = str(item.get("step_id") or "")
            generated_step_id = str(item.get("generated_step_id") or "")
            source_binding = bindings_by_step.get(source_step_id)
            generated_operation = normalized_by_id.get(generated_step_id)
            if (
                source_binding is None
                or generated_operation is None
                or generated_step_id in bindings_by_step
            ):
                continue
            generated_binding = source_binding.model_copy(
                update={
                    "plan_step_id": generated_step_id,
                    "operation": generated_operation,
                },
                deep=True,
            )
            bindings.append(generated_binding)
            bindings_by_step[generated_step_id] = generated_binding
            corrections.append(
                "cloned_scoped_binding_for_explicit_rate_split:"
                f"{source_step_id}:{generated_step_id}"
            )
    synchronized_bindings = []
    for binding in bindings:
        step = normalized_steps_by_id.get(binding.plan_step_id)
        if step is None:
            synchronized_bindings.append(binding)
            continue
        binding_data = binding.model_dump(mode="python")
        retained_sources = []
        for source_index, source in enumerate(binding_data.get("argument_sources") or []):
            if not isinstance(source, dict):
                retained_sources.append(source)
                continue
            argument_path = str(source.get("argument_path") or "")
            normalized_path = _normalized_step_argument_path(step, argument_path)
            if normalized_path is not None:
                if normalized_path != argument_path:
                    source["argument_path"] = normalized_path
                    corrections.append(
                        "synchronized_normalized_argument_path:"
                        f"{binding.plan_step_id}:source_{source_index}:"
                        f"{argument_path}:{normalized_path}"
                    )
                retained_sources.append(source)
                continue
            if source.get("source_kind") in {
                "perception_evidence",
                "derived_from_evidence",
            }:
                retained_sources.append(source)
                continue
            corrections.append(
                "dropped_absent_normalized_context_source:"
                f"{binding.plan_step_id}:source_{source_index}:{argument_path}"
            )
        binding_data["argument_sources"] = retained_sources
        synchronized_bindings.append(type(binding).model_validate(binding_data))
    return (
        semantics.model_copy(
            update={"bindings": synchronized_bindings},
            deep=True,
        ),
        corrections,
    )


def _normalized_step_argument_path(
    step: AudioEditStep,
    argument_path: str,
) -> str | None:
    step_data = step.model_dump(mode="python")
    if _raw_step_argument_exists(step_data, argument_path):
        return argument_path
    aliases = (
        ("params.edit_params.", "params.edit_op_params."),
        ("params.edit_op_params.", "params.edit_params."),
    )
    for old_prefix, new_prefix in aliases:
        if not argument_path.startswith(old_prefix):
            continue
        candidate = argument_path.replace(old_prefix, new_prefix, 1)
        if _raw_step_argument_exists(step_data, candidate):
            return candidate
    return None


def _raw_step_argument_exists(step: dict[str, Any], argument_path: str) -> bool:
    parts = argument_path.split(".")
    if not parts or parts[0] not in {"inputs", "params"}:
        return False
    current: Any = step.get(parts[0])
    for part in parts[1:]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False
    return True


def _unwrap_planner_reasoner_turn(data: Any) -> Any:
    """Compatibility helper for explicit turn envelopes only.

    A mapping-valued decision is malformed output, not a turn envelope. It must
    remain visible to strict validation unless the shape normalizer finds an explicit
    nested decision discriminant.
    """

    if not isinstance(data, dict):
        return data
    for key in (
        "planner_reasoner_turn",
        "workflow_turn",
    ):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return data


def _planner_response_format_mode(provider_name: str | None) -> str:
    provider = (provider_name or "").strip()
    for env_name in _planner_response_format_env_names(provider):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        normalized = raw.strip().lower().replace("-", "_")
        if normalized in {"", "default", "auto"}:
            break
        if normalized in {"json", "json_object", "object"}:
            return "json_object"
        if normalized in {"schema", "json_schema", "strict", "strict_schema"}:
            return "json_schema"
        if normalized in {"none", "off", "omit", "disabled", "disable"}:
            return "none"
    if provider == GLM_PROVIDER:
        return "json_object"
    return "json_schema"


def _planner_response_format_env_names(provider_name: str) -> list[str]:
    provider_key = provider_name.strip().upper().replace("-", "_")
    names: list[str] = []
    if provider_name == GLM_PROVIDER:
        names.extend(
            [
                "X_EDITOR_GLM_PLANNER_RESPONSE_FORMAT",
                "X_EDITOR_GLM_RESPONSE_FORMAT",
            ]
        )
    if provider_key:
        names.extend(
            [
                f"X_EDITOR_{provider_key}_PLANNER_RESPONSE_FORMAT",
                f"X_EDITOR_{provider_key}_RESPONSE_FORMAT",
            ]
        )
    names.extend(
        [
            "X_EDITOR_OPENAI_COMPAT_PLANNER_RESPONSE_FORMAT",
            "X_EDITOR_OPENAI_COMPAT_RESPONSE_FORMAT",
        ]
    )
    return names


def _unwrap_plan_payload(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return data
    for key in ("submit_audio_edit_plan", "audio_edit_plan", "AudioEditPlan", "plan"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    arguments = data.get("arguments")
    if isinstance(arguments, dict):
        return _unwrap_plan_payload(arguments)
    return data
