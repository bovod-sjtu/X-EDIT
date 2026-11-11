"""Core contracts for the X-Editor audio editing workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["planning", "executing", "validating", "completed", "failed"]
PlanningStatus = Literal["READY", "NEED_CLARIFICATION", "UNSUPPORTED", "FAIL"]
AudioEditStepStatus = Literal["pending", "running", "success", "failed", "skipped"]
ExecutorRepairResolution = Literal["resolved", "ambiguous", "unresolved"]
ExecutorArgumentBindingKind = Literal["mechanical", "semantic"]


class StrictModel(BaseModel):
    """Base model for explicit contracts."""

    model_config = ConfigDict(extra="forbid")


class SpeakerSegment(StrictModel):
    """A perceived speaker segment in the input audio."""

    segment_id: str
    speaker_id: str
    audio_ref_id: str | None = None
    audio_index: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    transcript: str | None = None
    perceived_gender: str | None = None
    description: str | None = None
    confidence: str = "medium"


class PerceptionReport(StrictModel):
    """Audio frontend output grounded in the actual audio content."""

    source_text: str | None = None
    source_text_audio_path: str | None = None
    source_text_audio_index: int | None = None
    speakers: list[dict[str, Any]] = Field(default_factory=list)
    segments: list[SpeakerSegment] = Field(default_factory=list)
    audio_quality: dict[str, Any] = Field(default_factory=dict)
    background: dict[str, Any] = Field(default_factory=dict)
    notable_events: list[str] = Field(default_factory=list)
    selected_spans: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class PerceptionOutput(BaseModel):
    """Legacy perception output kept for the current Gemini frontend node."""

    base_audio_analysis: dict[str, Any]
    reference_analysis: list[dict[str, Any]] = Field(default_factory=list)
    intent_description: str


class ClarificationAnswer(StrictModel):
    round: int
    question: str
    answer: str
    confidence: str = "medium"


class EnrichedIntent(StrictModel):
    """User request after audio-grounded enrichment by the text planner."""

    original_request: str
    clarified_goal: str
    target_objects: list[dict[str, Any]] = Field(default_factory=list)
    edit_requirements: list[str] = Field(default_factory=list)
    preservation_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class ToolCapability(StrictModel):
    """Lightweight tool card used for routing."""

    tool_id: str
    display_name: str
    backend: str | None = None
    category: str
    purpose: str
    when_to_use: list[str] = Field(default_factory=list)
    do_not_use_when: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    cost: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)


class ToolManual(StrictModel):
    """Detailed invocation contract loaded only after a tool is selected."""

    tool_id: str
    mcp_tool_name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    defaults: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    examples: list[dict[str, Any]] = Field(default_factory=list)


class AudioArtifact(StrictModel):
    """A concrete audio file produced or consumed by the workflow."""

    artifact_id: str
    path: str
    role: str
    parent_id: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioVersion(StrictModel):
    """Legacy-compatible version record for audio processing outputs."""

    version_id: int
    path: str
    step_id: str
    tool_name: str
    description: str


class ToolAction(StrictModel):
    """One high-level action in the tool-aware edit plan."""

    step_id: str
    tool_id: str
    objective: str
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifact_role: str
    validation_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ToolCall(StrictModel):
    """Concrete MCP call generated from a selected tool manual."""

    step_id: str
    tool_id: str
    mcp_tool_name: str
    arguments: dict[str, Any]


class ExecutorArgumentEvidence(StrictModel):
    """Prompt-visible evidence supporting one executor argument binding."""

    source_ref: str
    excerpt: str


class ExecutorArgumentBinding(StrictModel):
    """One field-level binding proposed while repairing an executor call."""

    field_path: str
    binding_kind: ExecutorArgumentBindingKind
    semantic_value: Any | None = None
    proposed_value: Any
    evidence: list[ExecutorArgumentEvidence] = Field(default_factory=list)
    alternatives: list[Any] = Field(default_factory=list)
    rationale: str = ""


class ExecutorToolCall(StrictModel):
    """Gemini-normalized executor call for one structured audio edit step.

    This is an execution-layer contract, not a direct audio processor. The
    runtime still routes through the existing operation registry and tool
    adapters after this shape has passed deterministic validation.
    """

    step_id: str
    operation: str
    tool_id: str | None = None
    mcp_tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    normalized_inputs: dict[str, Any] = Field(default_factory=dict)
    normalized_params: dict[str, Any] = Field(default_factory=dict)
    allow_execute: bool = True
    missing_inputs: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None
    validation_notes: list[str] = Field(default_factory=list)
    repair_resolution: ExecutorRepairResolution | None = None
    argument_bindings: list[ExecutorArgumentBinding] = Field(default_factory=list)


class ToolExecutionRecord(StrictModel):
    """Trace entry for one tool invocation."""

    step_id: str
    tool_id: str
    success: bool
    output_artifact_id: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ValidationReport(StrictModel):
    """Validation result for a step or final output."""

    step_id: str
    passed: bool
    score: float | None = None
    findings: list[str] = Field(default_factory=list)
    required_revision: str | None = None


class AudioEditStep(StrictModel):
    """One structured step produced by the Planner and consumed by the Executor."""

    id: str
    operation: str
    description: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    span_timebase: Literal["observed_audio", "step_input"] = Field(
        default="observed_audio",
        description=("Explicit span coordinates refer to the observed audio by default. "
                     "step_input declares coordinates on this call's explicitly bound input; "
                     "it does not certify target grounding or waive bounds checks."),
    )
    depends_on: list[str] = Field(default_factory=list)
    status: AudioEditStepStatus = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    required: bool = True
    continue_on_error: bool | None = None


class PlanOutputArtifact(StrictModel):
    """One explicitly delivered terminal artifact from an executable plan."""

    artifact_id: str = Field(min_length=1, max_length=256)
    from_step: str = Field(min_length=1, max_length=256)
    from_output: str = Field(default="output_audio_path", min_length=1, max_length=256)
    role: str = Field(min_length=1, max_length=256)
    primary: bool = False


class PlannerCannotPlanReason(StrictModel):
    """Typed Planner refusal reason retained by orchestration and failure audit."""

    code: Literal[
        "missing_evidence",
        "missing_perception_route",
        "unsupported_capability",
        "semantic_ambiguity",
        "conflicting_evidence",
        "invalid_parameter_domain",
        "unsupported_output_topology",
        "other",
    ]
    message: str = Field(min_length=1, max_length=2048)
    target_ids: list[str] = Field(default_factory=list, max_length=32)
    operation_ids: list[str] = Field(default_factory=list, max_length=32)


class AudioEditPlan(StrictModel):
    """Structured audio-editing plan passed from Planner to Executor."""

    request: str
    input_audio: str | None = None
    output_audio: str | None = None
    steps: list[AudioEditStep] = Field(default_factory=list)
    output_artifacts: list[PlanOutputArtifact] = Field(default_factory=list, max_length=64)
    global_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)


class EvaluationResult(StrictModel):
    """Result returned by an Evaluator after executing an AudioEditPlan."""

    passed: bool
    score: float | None = None
    feedback: str | None = None
    need_replan: bool = False
    failed_steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EditInstruction(BaseModel):
    """Legacy single edit instruction produced by the current text planner."""

    type: str = Field(..., description="One of: text_edit, emotion_edit, prosody_edit, vc_edit, se_edit")
    source_text: str = Field(..., description="Source text segment from audio")
    target_text: str = Field(..., description="Target text after editing")
    source_audio: str = Field(default="user_input", description="'user_input' or 'generated[N]'")
    instruction: str | None = Field(default=None, description="Natural language instruction for this edit")
    reference_audio: str | None = Field(default=None, description="Reference audio path for VC")
    parameters: dict[str, Any] | None = Field(default=None, description="Additional parameters")


class AudioEditResult(BaseModel):
    """Final result for both the legacy planner and the new execution workflow."""

    status: str
    output_audio_path: str | None = None
    source_text: str | None = None
    edits: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[ToolAction] = Field(default_factory=list)
    artifacts: list[AudioArtifact] = Field(default_factory=list)
    tool_trace: list[ToolExecutionRecord] = Field(default_factory=list)
    validation_reports: list[ValidationReport] = Field(default_factory=list)
    clarification_history: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
