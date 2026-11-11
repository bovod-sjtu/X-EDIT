"""Builder for the new tool-aware planning graph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from x_editor.core.state import AudioEditState
from x_editor.graph.tool_nodes import (
    create_audio_perception_node,
    create_action_planning_node,
    create_capability_routing_node,
    create_dry_run_finalization_node,
    create_intent_enrichment_node,
    create_manual_loading_node,
    create_tool_call_generation_node,
    create_tool_failure_node,
)
from x_editor.graph.tool_routers import capability_router
from x_editor.graph.execution_nodes import create_tool_execution_node
from x_editor.planner import ToolAwarePlanner
from x_editor.frontend import GeminiAudioFrontend
from x_editor.tools.executor import ToolExecutor


def build_tool_aware_graph(planner: ToolAwarePlanner | None = None):
    """Build a dry-run graph that plans and generates tool calls without execution."""

    planner = planner or ToolAwarePlanner()
    builder = StateGraph(AudioEditState)
    builder.add_node("intent_enrichment", create_intent_enrichment_node(planner))
    builder.add_node("capability_routing", create_capability_routing_node(planner))
    builder.add_node("action_planning", create_action_planning_node(planner))
    builder.add_node("manual_loading", create_manual_loading_node())
    builder.add_node("tool_call_generation", create_tool_call_generation_node(planner))
    builder.add_node("finalization", create_dry_run_finalization_node())
    builder.add_node("failure", create_tool_failure_node())

    builder.set_entry_point("intent_enrichment")
    builder.add_edge("intent_enrichment", "capability_routing")
    builder.add_conditional_edges(
        "capability_routing",
        capability_router,
        {
            "plan": "action_planning",
            "fail": "failure",
        },
    )
    builder.add_edge("action_planning", "manual_loading")
    builder.add_edge("manual_loading", "tool_call_generation")
    builder.add_edge("tool_call_generation", "finalization")
    builder.add_edge("finalization", END)
    builder.add_edge("failure", END)
    return builder.compile()


def build_tool_execution_graph(
    planner: ToolAwarePlanner | None = None,
    executor: ToolExecutor | None = None,
    *,
    use_structured_executor: bool = False,
):
    """Build a graph that plans and then executes selected tools."""

    planner = planner or ToolAwarePlanner()
    builder = StateGraph(AudioEditState)
    builder.add_node("intent_enrichment", create_intent_enrichment_node(planner))
    builder.add_node("capability_routing", create_capability_routing_node(planner))
    builder.add_node("action_planning", create_action_planning_node(planner))
    builder.add_node("manual_loading", create_manual_loading_node())
    builder.add_node(
        "tool_execution",
        create_tool_execution_node(planner, executor, use_structured_executor=use_structured_executor),
    )
    builder.add_node("finalization", create_dry_run_finalization_node())
    builder.add_node("failure", create_tool_failure_node())

    builder.set_entry_point("intent_enrichment")
    builder.add_edge("intent_enrichment", "capability_routing")
    builder.add_conditional_edges(
        "capability_routing",
        capability_router,
        {"plan": "action_planning", "fail": "failure"},
    )
    builder.add_edge("action_planning", "manual_loading")
    builder.add_edge("manual_loading", "tool_execution")
    builder.add_conditional_edges(
        "tool_execution",
        lambda state: "fail" if state.get("status") == "failed" else "finalize",
        {"fail": "failure", "finalize": "finalization"},
    )
    builder.add_edge("finalization", END)
    builder.add_edge("failure", END)
    return builder.compile()


def build_gemini_tool_execution_graph(
    frontend: GeminiAudioFrontend,
    planner: ToolAwarePlanner,
    executor: ToolExecutor | None = None,
    *,
    agentic_mode: str | None = None,
):
    """Build audio-perception + tool-aware planner + tool execution graph.

    The historical entrypoint name is kept for compatibility.  Passing
    ``agentic_mode`` makes the perception provider explicit for A/B runs.
    """

    builder = StateGraph(AudioEditState)
    builder.add_node("audio_perception", create_audio_perception_node(frontend, agentic_mode=agentic_mode))
    builder.add_node("intent_enrichment", create_intent_enrichment_node(planner))
    builder.add_node("capability_routing", create_capability_routing_node(planner))
    builder.add_node("action_planning", create_action_planning_node(planner))
    builder.add_node("manual_loading", create_manual_loading_node())
    builder.add_node("tool_execution", create_tool_execution_node(planner, executor))
    builder.add_node("finalization", create_dry_run_finalization_node())
    builder.add_node("failure", create_tool_failure_node())

    builder.set_entry_point("audio_perception")
    builder.add_edge("audio_perception", "intent_enrichment")
    builder.add_edge("intent_enrichment", "capability_routing")
    builder.add_conditional_edges(
        "capability_routing",
        capability_router,
        {"plan": "action_planning", "fail": "failure"},
    )
    builder.add_edge("action_planning", "manual_loading")
    builder.add_edge("manual_loading", "tool_execution")
    builder.add_conditional_edges(
        "tool_execution",
        lambda state: "fail" if state.get("status") == "failed" else "finalize",
        {"fail": "failure", "finalize": "finalization"},
    )
    builder.add_edge("finalization", END)
    builder.add_edge("failure", END)
    return builder.compile()
