"""LangGraph node functions for X-Editor (simplified: perception + structuring only)."""

from __future__ import annotations

from typing import Any

from x_editor.core.state import AudioEditState
from x_editor.core.constants import AgentStatus, PlanningStatus
from x_editor.agents.omni_agent import OmniAgent
from x_editor.agents.text_agent import TextAgent


def create_perception_node(omni_agent: OmniAgent):
    async def node(state: AudioEditState) -> dict[str, Any]:
        perception, source_text = await omni_agent.perceive(
            user_prompt=state["user_prompt"],
            base_audio_path=state["base_audio_path"],
            reference_audio_paths=state.get("reference_audio_paths") or [],
        )
        return {
            "audio_perception": {
                "base_audio_analysis": perception.base_audio_analysis,
                "reference_analysis": perception.reference_analysis,
            },
            "intent_description": perception.intent_description,
            "source_text": source_text,
            "status": AgentStatus.PLANNING,
        }
    return node


def create_structuring_node(text_agent: TextAgent):
    async def node(state: AudioEditState) -> dict[str, Any]:
        result = text_agent.plan(
            user_prompt=state["user_prompt"],
            intent_description=state["intent_description"],
            audio_perception=state["audio_perception"],
            source_text=state.get("source_text", ""),
        )
        return {
            "planning_status": result.get("planning_status"),
            "source_text": result.get("source_text"),
            "edits": result.get("edits", []),
            "clarification_question": result.get("clarification_question"),
            "clarification_focus": result.get("clarification_focus"),
            "status": AgentStatus.PLANNING,
        }
    return node


def create_clarification_node(omni_agent: OmniAgent):
    async def node(state: AudioEditState) -> dict[str, Any]:
        answer = await omni_agent.clarify(
            user_prompt=state["user_prompt"],
            base_audio_path=state["base_audio_path"],
            reference_audio_paths=state.get("reference_audio_paths") or [],
            previous_perception=state["audio_perception"],
            question=state["clarification_question"] or "",
            focus=state.get("clarification_focus"),
        )

        perception = dict(state["audio_perception"])
        answers = perception.get("clarification_answers", [])
        answers.append(answer)
        perception["clarification_answers"] = answers

        history = list(state.get("clarification_history") or [])
        history.append({
            "round": state["clarification_round"],
            "question": state["clarification_question"],
            "answer": answer.get("answer", ""),
            "confidence": answer.get("confidence", "medium"),
        })

        return {
            "audio_perception": perception,
            "clarification_history": history,
            "clarification_round": state["clarification_round"] + 1,
            "clarification_question": None,
        }
    return node


def create_finalization_node():
    async def node(state: AudioEditState) -> dict[str, Any]:
        return {
            "status": AgentStatus.COMPLETED,
        }
    return node


def create_failure_node():
    async def node(state: AudioEditState) -> dict[str, Any]:
        error = state.get("error_message") or "Workflow failed"
        return {
            "status": AgentStatus.FAILED,
            "error_message": error,
            "edits": [],
        }
    return node
