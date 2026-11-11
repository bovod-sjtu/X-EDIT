"""LangGraph builder for X-Editor (simplified)."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from x_editor.core.state import AudioEditState
from x_editor.agents.omni_agent import OmniAgent
from x_editor.agents.text_agent import TextAgent
from x_editor.graph.nodes import (
    create_perception_node,
    create_structuring_node,
    create_clarification_node,
    create_finalization_node,
    create_failure_node,
)
from x_editor.graph.routers import structuring_router


def build_graph(
    omni_agent: OmniAgent,
    text_agent: TextAgent,
) -> StateGraph:
    """Build and compile the simplified X-Editor LangGraph."""
    builder = StateGraph(AudioEditState)

    # Register nodes
    builder.add_node("audio_perception", create_perception_node(omni_agent))
    builder.add_node("intent_structuring", create_structuring_node(text_agent))
    builder.add_node("clarification", create_clarification_node(omni_agent))
    builder.add_node("finalization", create_finalization_node())
    builder.add_node("failure", create_failure_node())

    # Entry point
    builder.set_entry_point("audio_perception")

    # Sequential edges
    builder.add_edge("audio_perception", "intent_structuring")

    # Structuring router
    builder.add_conditional_edges(
        "intent_structuring",
        structuring_router,
        {
            "clarify": "clarification",
            "finalize": "finalization",
            "fail": "failure",
        },
    )

    # Clarification loops back to structuring
    builder.add_edge("clarification", "intent_structuring")

    # Terminal edges
    builder.add_edge("finalization", END)
    builder.add_edge("failure", END)

    return builder.compile()
