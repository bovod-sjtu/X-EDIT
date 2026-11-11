"""Typed semantic reasoning contracts and non-authoritative runtime adapters."""

from x_editor.reasoner.contracts import (
    ObservationNeed,
    PriorTurnSemanticContext,
    ReasonedEditIntent,
    ReasonedEditObjective,
    ReasonerAdmissionIssue,
    ReasonerAdmissionReport,
    ReasonerCallRequest,
    ReasonerEvidenceNeed,
    ReasonerShadowStrategy,
    RequestedSemanticChange,
    SemanticEntityKind,
    SemanticEntityRole,
    SemanticReasoningMemo,
    SemanticTemporalExtent,
    SemanticValueReference,
    admit_reasoned_edit_intent,
)
from x_editor.reasoner.openai_compatible import (
    OpenAICompatibleMemoSemanticReasoner,
    OpenAICompatibleProseMemoSemanticReasoner,
    OpenAICompatibleSemanticReasoner,
)
from x_editor.reasoner.shadow import (
    ReasonerShadowCallRecord,
    ReasonerShadowInputSnapshot,
    ReasonerShadowRunner,
)

__all__ = [
    "ObservationNeed",
    "OpenAICompatibleMemoSemanticReasoner",
    "OpenAICompatibleProseMemoSemanticReasoner",
    "OpenAICompatibleSemanticReasoner",
    "PriorTurnSemanticContext",
    "ReasonedEditIntent",
    "ReasonedEditObjective",
    "ReasonerAdmissionIssue",
    "ReasonerAdmissionReport",
    "ReasonerCallRequest",
    "ReasonerEvidenceNeed",
    "ReasonerShadowCallRecord",
    "ReasonerShadowInputSnapshot",
    "ReasonerShadowRunner",
    "ReasonerShadowStrategy",
    "RequestedSemanticChange",
    "SemanticEntityKind",
    "SemanticEntityRole",
    "SemanticReasoningMemo",
    "SemanticTemporalExtent",
    "SemanticValueReference",
    "admit_reasoned_edit_intent",
]
