"""OpenAI-compatible executor local-repair providers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from x_editor.agents.openai_compatible import (
    OpenAICompatibleChatClient,
    OpenAICompatibleServiceUnavailable,
)
from x_editor.core.agentic_mode import (
    GLM_PROVIDER,
    provider_service_hint,
    resolve_openai_compatible_api_key,
    resolve_openai_compatible_base_url,
    resolve_openai_compatible_model,
)
from x_editor.core.schemas import AudioEditStep, ExecutorToolCall, ToolManual
from x_editor.runtime.executor_tool_call_planner import ExecutorToolCallPlannerError
from x_editor.runtime.gemini_executor import validate_executor_tool_call
from x_editor.workflow_control.policy import repair_action_is_enforced


class GLMExecutorToolCallPlanner:
    """GLM-4.7-Flash local repair for one failed executor step."""

    provider_name = GLM_PROVIDER
    prompt_version = "x-editor.executor-tool-call/openai-compatible-v2"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 8192,
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

    def plan_tool_call(
        self,
        *,
        step: AudioEditStep,
        state: dict[str, Any],
        operation_spec: dict[str, Any],
        manual: ToolManual | None = None,
        verifier_error: dict[str, Any] | str | None = None,
    ) -> ExecutorToolCall:
        if not self.api_key or not self.base_url:
            raise ExecutorToolCallPlannerError(
                f"{self.provider_name} executor repair service unavailable: {provider_service_hint(self.provider_name)}"
            )
        prompt = _repair_user_prompt(
            step=step,
            state=state,
            operation_spec=operation_spec,
            manual=manual,
            verifier_error=verifier_error or state.get("last_tool_contract_error") or state.get("last_executor_normalization_error"),
        )
        try:
            data = self.client.call_json(
                _SYSTEM_PROMPT,
                prompt,
                response_format=_executor_tool_call_response_format(),
                repair_schema_name="ExecutorToolCall",
            )
        except OpenAICompatibleServiceUnavailable as exc:
            raise ExecutorToolCallPlannerError(
                f"{self.provider_name} executor repair service unavailable: {exc}. "
                f"To use open executor repair, {provider_service_hint(self.provider_name)}."
            ) from exc
        except Exception as exc:
            raise ExecutorToolCallPlannerError(
                f"{self.provider_name} executor repair failed before returning ExecutorToolCall: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            call = ExecutorToolCall(**_unwrap_executor_call(data))
        except ValidationError as exc:
            raise ExecutorToolCallPlannerError(
                f"{self.provider_name} executor repair returned invalid ExecutorToolCall schema: {exc}"
            ) from exc
        try:
            validate_executor_tool_call(
                call=call,
                step=step,
                state=state,
                operation_spec=operation_spec,
                manual=manual,
            )
        except Exception as exc:
            raise ExecutorToolCallPlannerError(
                f"{self.provider_name} executor repair failed deterministic verification: {exc}"
            ) from exc
        return call


_SYSTEM_PROMPT = """You are the local executor repair provider for X-Editor.

You repair exactly one failed AudioEditStep into an ExecutorToolCall JSON object.

Hard constraints:
- Repair only the provided failed_step. Keep step_id and operation identical.
- Do not rewrite any already-successful or unrelated steps.
- Do not weaken the user objective or convert an existing-audio edit into unrelated TTS.
- Do not hide service unavailable, unsupported capability, missing input, or missing grounding. If execution is unsafe, set allow_execute=false with blocked_reason and missing_inputs.
- Use only the tool_contract and relevant perception evidence in the user JSON.
- If argument_repair_protocol is present, change only its exact repairable_fields. Every changed field needs one
  argument_bindings item whose proposed_value exactly matches the normalized field. Semantic bindings must cite
  an exact excerpt from user_objective, failed_step description, perception_evidence, or source_text. If multiple
  values remain plausible, set repair_resolution=ambiguous, list alternatives, set allow_execute=false, and do not patch.
- Return one JSON object matching ExecutorToolCall. No markdown, prose, or extra keys.
"""


def _repair_user_prompt(
    *,
    step: AudioEditStep,
    state: dict[str, Any],
    operation_spec: dict[str, Any],
    manual: ToolManual | None,
    verifier_error: dict[str, Any] | str | None,
) -> str:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    argument_repair = repair_action_is_enforced(
        context, "repair_step_arguments"
    )
    payload = {
        "failed_step": step.model_dump(),
        "verifier_error": verifier_error,
        "tool_contract": _compact_tool_contract(operation_spec, manual),
        "user_objective": state.get("user_prompt") or (state.get("plan").request if state.get("plan") else None),
        "perception_evidence": _compact_perception_evidence(state),
        "current_audio_aliases": {
            "current_audio_path": state.get("current_audio_path"),
            "input_audio": state.get("input_audio"),
            "input_audios": state.get("input_audios") or [],
            "reference_audio_paths": state.get("reference_audio_paths") or [],
        },
        "required_response": "ExecutorToolCall JSON for failed_step only",
    }
    if argument_repair:
        violations = (
            verifier_error.get("violations")
            if isinstance(verifier_error, dict)
            else []
        )
        payload["argument_repair_protocol"] = {
            "repairable_fields": sorted(
                {
                    str(field)
                    for violation in violations or []
                    if isinstance(violation, dict)
                    for field in violation.get("repairable_fields") or []
                    if field
                }
            ),
            "required_response_fields": [
                "repair_resolution",
                "argument_bindings",
            ],
            "evidence_source_refs": [
                "user_request",
                "step_description",
                "perception",
                "source_text",
            ],
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _compact_tool_contract(operation_spec: dict[str, Any], manual: ToolManual | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation_spec": operation_spec,
    }
    if manual is not None:
        payload["manual"] = {
            "tool_id": manual.tool_id,
            "mcp_tool_name": manual.mcp_tool_name,
            "input_schema": manual.input_schema,
            "defaults": manual.defaults,
            "validation": manual.validation,
            "examples": manual.examples[:2],
        }
    return payload


def _compact_perception_evidence(state: dict[str, Any]) -> dict[str, Any]:
    perception = state.get("perception_report") if isinstance(state.get("perception_report"), dict) else {}
    return {
        "source_text": state.get("source_text") or perception.get("source_text"),
        "source_text_source": perception.get("source_text_source"),
        "word_timestamps": _limit_sequence(perception.get("word_timestamps"), 24),
        "speaker_segments": _limit_sequence(perception.get("speaker_segments") or perception.get("segments"), 12),
        "speakers": _limit_sequence(perception.get("speakers"), 12),
        "notable_events": _limit_sequence(perception.get("notable_events"), 12),
        "uncertainty": _limit_sequence(perception.get("uncertainty"), 12),
        "services": perception.get("services") if isinstance(perception.get("services"), dict) else {},
    }


def _limit_sequence(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _executor_tool_call_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ExecutorToolCall",
            "schema": ExecutorToolCall.model_json_schema(),
            "strict": True,
        },
    }


def _unwrap_executor_call(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return data
    for key in ("executor_tool_call", "ExecutorToolCall", "tool_call", "call"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    arguments = data.get("arguments")
    if isinstance(arguments, dict):
        return _unwrap_executor_call(arguments)
    return data
