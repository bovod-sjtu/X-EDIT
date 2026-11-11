"""Gemini 3 Flash planner for tool-aware speech editing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from x_editor.agents.base import GeminiClient
from x_editor.core.schemas import EnrichedIntent, ToolAction, ToolCall, ToolCapability, ToolManual
from x_editor.planner.tool_aware_planner import ToolAwarePlanner
from x_editor.tools.loader import render_capability_summary


class GeminiToolPlanner(ToolAwarePlanner):
    """Gemini-backed planner with deterministic fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.01,
        max_tokens: int = 8192,
        timeout: float = 60.0,
        fallback: ToolAwarePlanner | None = None,
        enable_fallback: bool = True,
    ) -> None:
        self.client = GeminiClient(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.fallback = fallback or ToolAwarePlanner()
        self.enable_fallback = enable_fallback
        prompt_dir = Path(__file__).parent.parent / "prompts"
        self.intent_system = (prompt_dir / "tool_intent_enrichment_system.md").read_text(encoding="utf-8")
        self.intent_user = (prompt_dir / "tool_intent_enrichment_user.md").read_text(encoding="utf-8")
        self.routing_system = (prompt_dir / "tool_capability_routing_system.md").read_text(encoding="utf-8")
        self.routing_user = (prompt_dir / "tool_capability_routing_user.md").read_text(encoding="utf-8")
        self.plan_system = (prompt_dir / "tool_action_planning_system.md").read_text(encoding="utf-8")
        self.plan_user = (prompt_dir / "tool_action_planning_user.md").read_text(encoding="utf-8")
        self.call_system = (prompt_dir / "tool_call_generation_system.md").read_text(encoding="utf-8")
        self.call_user = (prompt_dir / "tool_call_generation_user.md").read_text(encoding="utf-8")

    def enrich_intent(
        self,
        user_prompt: str,
        perception_report: dict[str, Any] | None = None,
        reference_audio_paths: list[str] | None = None,
    ) -> EnrichedIntent:
        try:
            data = self.client.call_json(
                self.intent_system,
                self.intent_user.format(
                    user_prompt=user_prompt,
                    perception_report_json=json.dumps(perception_report or {}, ensure_ascii=False, indent=2),
                    reference_audio_paths=json.dumps(reference_audio_paths or [], ensure_ascii=False),
                ),
            )
            return EnrichedIntent(**data)
        except Exception:
            if not self.enable_fallback:
                raise
            return self.fallback.enrich_intent(user_prompt, perception_report, reference_audio_paths)

    def route_capabilities(
        self,
        intent: EnrichedIntent,
        capabilities: dict[str, ToolCapability],
        reference_audio_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            data = self.client.call_json(
                self.routing_system,
                self.routing_user.format(
                    enriched_intent_json=intent.model_dump_json(indent=2),
                    capability_summary=render_capability_summary(capabilities),
                    reference_audio_paths=json.dumps(reference_audio_paths or [], ensure_ascii=False),
                ),
            )
            selected = [t for t in data.get("selected_tools", []) if t in capabilities]
            status = data.get("planning_status", "READY" if selected else "UNSUPPORTED")
            if status == "READY" and not selected:
                status = "UNSUPPORTED"
            if status == "READY":
                guard = self.fallback.route_capabilities(intent, capabilities, reference_audio_paths)
                if guard.get("planning_status") != "READY":
                    return guard
                guard_tools = guard.get("selected_tools") or []
                selected = list(guard_tools)
            return {
                "planning_status": status,
                "selected_tools": selected,
                "error_message": data.get("error_message"),
            }
        except Exception:
            if not self.enable_fallback:
                raise
            return self.fallback.route_capabilities(intent, capabilities, reference_audio_paths)

    def create_plan(
        self,
        intent: EnrichedIntent,
        selected_tools: list[str],
        base_artifact_id: str = "input_audio",
    ) -> list[ToolAction]:
        try:
            data = self.client.call_json(
                self.plan_system,
                self.plan_user.format(
                    enriched_intent_json=intent.model_dump_json(indent=2),
                    selected_tools=json.dumps(selected_tools, ensure_ascii=False),
                    base_artifact_id=base_artifact_id,
                ),
            )
            actions = [ToolAction(**item) for item in data.get("plan", [])]
            if not actions:
                raise ValueError("Gemini planner returned empty plan")
            return actions
        except Exception:
            if not self.enable_fallback:
                raise
            return self.fallback.create_plan(intent, selected_tools, base_artifact_id)

    def create_tool_calls(
        self,
        actions: list[ToolAction],
        manuals: dict[str, ToolManual],
        state: dict[str, Any],
        artifact_paths: dict[str, str] | None = None,
    ) -> list[ToolCall]:
        try:
            data = self.client.call_json(
                self.call_system,
                self.call_user.format(
                    actions_json=json.dumps([a.model_dump() for a in actions], ensure_ascii=False, indent=2),
                    manuals_json=json.dumps({k: v.model_dump() for k, v in manuals.items()}, ensure_ascii=False, indent=2),
                    state_json=json.dumps(_jsonable_state(state), ensure_ascii=False, indent=2),
                    artifact_paths_json=json.dumps(artifact_paths or {}, ensure_ascii=False, indent=2),
                ),
            )
            calls = [ToolCall(**item) for item in data.get("tool_calls", [])]
            if not calls:
                raise ValueError("Gemini planner returned no tool calls")
            return calls
        except Exception:
            if not self.enable_fallback:
                raise
            return self.fallback.create_tool_calls(actions, manuals, state, artifact_paths)


def _jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "user_prompt",
        "base_audio_path",
        "reference_audio_paths",
        "source_text",
        "perception_report",
        "enriched_intent",
        "selected_tools",
        "output_dir",
    ]
    return {key: state.get(key) for key in keep if key in state}
