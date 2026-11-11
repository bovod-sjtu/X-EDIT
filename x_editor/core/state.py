"""LangGraph state definition for X-Editor."""

from typing import Annotated, Any, TypedDict
import operator


class AudioEditState(TypedDict, total=False):
    # ========== Inputs ==========
    user_prompt: str
    base_audio_path: str
    reference_audio_paths: Annotated[list[str], operator.add]
    output_dir: str
    runtime_settings: dict[str, Any]

    # ========== Audio frontend perception ==========
    audio_perception: dict[str, Any]
    perception_report: dict[str, Any]
    perception_attempts: list[dict[str, Any]]
    intent_description: str
    source_text: str | None
    source_text_audio_path: str | None
    source_text_audio_index: int | None

    # ========== Intent enrichment and clarification ==========
    enriched_intent: dict[str, Any]
    clarification_round: int
    clarification_history: Annotated[list[dict[str, Any]], operator.add]
    clarification_question: str | None
    clarification_focus: str | None

    # ========== Tool-aware planning ==========
    planning_status: str | None
    selected_tools: list[str]
    routing_policy_candidates: list[dict[str, Any]]
    loaded_manuals: dict[str, Any]
    structured_instruction: dict[str, Any] | None
    plan: list[dict[str, Any]]
    current_step_index: int
    pending_tool_call: dict[str, Any] | None
    tool_trace: Annotated[list[dict[str, Any]], operator.add]
    validation_reports: Annotated[list[dict[str, Any]], operator.add]
    evaluation: dict[str, Any] | None
    evaluation_history: Annotated[list[dict[str, Any]], operator.add]
    evaluator_repair: dict[str, Any] | None
    evaluation_revisions: int | None
    runtime_verification_revisions: int | None
    runtime_verification_repair_history: Annotated[list[dict[str, Any]], operator.add]
    runtime_verification_best_effort: dict[str, Any] | None

    # ========== Artifacts and outputs ==========
    artifacts: Annotated[list[dict[str, Any]], operator.add]
    audio_artifacts: Annotated[list[dict[str, Any]], operator.add]
    debug_audio_artifacts: Annotated[list[dict[str, Any]], operator.add]
    intermediate_audio_artifacts: Annotated[list[dict[str, Any]], operator.add]
    intermediate_audios: Annotated[list[dict[str, Any]], operator.add]
    output_audio_path: str | None
    edits: list[dict[str, Any]]
    status: str
    error_message: str | None
