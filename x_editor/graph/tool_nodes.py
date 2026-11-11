"""Tool-aware graph nodes for the new X-Editor workflow."""

from __future__ import annotations

from typing import Any

from x_editor.core.agentic_mode import resolve_agentic_mode
from x_editor.core.state import AudioEditState
from x_editor.frontend import GeminiAudioFrontend
from x_editor.perception import PerceptionRequirements, build_audio_perception_fallback
from x_editor.planner import ToolAwarePlanner
from x_editor.tools.loader import load_all_capabilities, load_manual


def create_audio_perception_node(frontend: GeminiAudioFrontend, *, agentic_mode: str | None = None):
    async def node(state: AudioEditState) -> dict[str, Any]:
        runtime_settings = state.get("runtime_settings") or {}
        mode = resolve_agentic_mode(
            state.get("agentic_mode")
            or (runtime_settings.get("agentic_mode") if isinstance(runtime_settings, dict) else None)
            or agentic_mode
        )
        fallback = build_audio_perception_fallback(agentic_mode=mode)
        frontend_provider = getattr(frontend, "provider_name", "gemini")
        provider_frontend = frontend
        if fallback.provider == "qwen3_omni" and frontend_provider != "qwen3_omni":
            provider_frontend = None
        result = fallback.perceive(
            user_prompt=state["user_prompt"],
            audio_path=state["base_audio_path"],
            base_perception=state.get("perception_report") or {},
            reference_audio_paths=state.get("reference_audio_paths") or [],
            output_dir=state.get("output_dir"),
            requirements=PerceptionRequirements(
                source_text=True,
                speaker_segments=True,
                speaker_profiles=True,
                segment_descriptions=True,
            ),
            run_specialized=True,
            force_model_fallback=True,
            strict=True,
            frontend=provider_frontend,
        )
        report = result.perception
        return {
            "perception_report": report,
            "source_text": report.get("source_text"),
            "status": "planning",
        }
    return node


def create_intent_enrichment_node(planner: ToolAwarePlanner):
    async def node(state: AudioEditState) -> dict[str, Any]:
        intent = planner.enrich_intent(
            user_prompt=state["user_prompt"],
            perception_report=state.get("perception_report") or state.get("audio_perception") or {},
            reference_audio_paths=state.get("reference_audio_paths") or [],
        )
        return {"enriched_intent": intent.model_dump(), "status": "planning"}
    return node


def create_capability_routing_node(planner: ToolAwarePlanner):
    async def node(state: AudioEditState) -> dict[str, Any]:
        capabilities = load_all_capabilities()
        intent = planner.enrich_intent(state["user_prompt"])
        if state.get("enriched_intent"):
            from x_editor.core.schemas import EnrichedIntent
            intent = EnrichedIntent(**state["enriched_intent"])
        routing = planner.route_capabilities(
            intent=intent,
            capabilities=capabilities,
            reference_audio_paths=state.get("reference_audio_paths") or [],
        )
        routing["routing_policy_candidates"] = planner.query_routing_policy(
            intent,
            capabilities,
            reference_audio_paths=state.get("reference_audio_paths") or [],
        )
        return routing
    return node


def create_action_planning_node(planner: ToolAwarePlanner):
    async def node(state: AudioEditState) -> dict[str, Any]:
        from x_editor.core.schemas import EnrichedIntent

        intent = EnrichedIntent(**state["enriched_intent"])
        actions = planner.create_plan(intent, state.get("selected_tools") or [])
        return {"plan": [action.model_dump() for action in actions]}
    return node


def create_manual_loading_node():
    async def node(state: AudioEditState) -> dict[str, Any]:
        manuals: dict[str, Any] = {}
        for action in state.get("plan") or []:
            tool_id = action["tool_id"]
            if tool_id in {
                "artifact_ops",
                "accent_conversion",
                "paralinguistic_event_add",
                "paralinguistic_event_remove",
            } or tool_id in manuals:
                continue
            manuals[tool_id] = load_manual(tool_id).model_dump()
        return {"loaded_manuals": manuals}
    return node


def create_tool_call_generation_node(planner: ToolAwarePlanner):
    async def node(state: AudioEditState) -> dict[str, Any]:
        from x_editor.core.schemas import ToolAction, ToolManual

        actions = [ToolAction(**item) for item in state.get("plan") or []]
        manuals = {key: ToolManual(**value) for key, value in (state.get("loaded_manuals") or {}).items()}
        calls = planner.create_tool_calls(actions, manuals, dict(state))
        return {"tool_trace": [{"phase": "generated_call", **call.model_dump()} for call in calls]}
    return node


def create_dry_run_finalization_node():
    async def node(state: AudioEditState) -> dict[str, Any]:
        return {"status": "completed", "output_audio_path": state.get("output_audio_path")}
    return node


def create_tool_failure_node():
    async def node(state: AudioEditState) -> dict[str, Any]:
        return {
            "status": "failed",
            "error_message": state.get("error_message") or "Tool-aware planning failed",
        }
    return node
