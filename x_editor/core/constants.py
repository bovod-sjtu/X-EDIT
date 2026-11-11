"""Constants and enums for X-Editor."""

from enum import Enum


class AgentStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    SUPERVISING = "supervising"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanningStatus(str, Enum):
    READY = "READY"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    FAIL = "FAIL"


class SupervisionDecision(str, Enum):
    CONTINUE = "CONTINUE"
    REVISE = "REVISE"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
