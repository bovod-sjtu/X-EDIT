"""Tool-aware planners for X-Editor."""

from typing import Any

from x_editor.core.agentic_mode import GEMINI_MODE, OPEN_GLM_QWEN_OMNI_MODE, resolve_agentic_mode
from x_editor.planner.tool_aware_planner import ToolAwarePlanner
from x_editor.planner.gemini_tool_planner import GeminiToolPlanner
from x_editor.planner.gemini_planner import GeminiPlanner
from x_editor.planner.openai_compatible_planner import GLMOpenAICompatiblePlanner


def build_audio_edit_planner(
    *,
    agentic_mode: str | None = None,
    model_name: str | None = None,
    timeout: float = 60.0,
    available_operations: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Build the structured planner for the selected provider mode."""

    mode = resolve_agentic_mode(agentic_mode)
    if mode == OPEN_GLM_QWEN_OMNI_MODE:
        return GLMOpenAICompatiblePlanner(
            model_name=model_name,
            timeout=timeout,
            available_operations=available_operations,
            **kwargs,
        )
    if mode == GEMINI_MODE:
        return GeminiPlanner(
            model_name=model_name,
            timeout=timeout,
            available_operations=available_operations,
            **kwargs,
        )
    raise ValueError(f"Unsupported X-Editor agentic mode: {agentic_mode!r}")


__all__ = [
    "GLMOpenAICompatiblePlanner",
    "GeminiPlanner",
    "GeminiToolPlanner",
    "ToolAwarePlanner",
    "build_audio_edit_planner",
]
