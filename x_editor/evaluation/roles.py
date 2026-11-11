"""Typed scientific roles and authority boundaries for evaluator backends."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel

EvaluatorRole = Literal[
    "offline_prior_estimator",
    "post_delivery_development_diagnostic",
    "runtime_feedback_critic",
    "heldout_final_judge",
]
EvaluationDataRole = Literal[
    "development_calibration",
    "development_diagnostic",
    "experiment_runtime",
    "heldout_test",
]


class EvaluatorRoleContract(StrictModel):
    """Frozen authority profile for one scientific use of an Evaluator."""

    schema_version: Literal["x-editor.evaluator-role-contract/v1"] = (
        "x-editor.evaluator-role-contract/v1"
    )
    evaluator_role: EvaluatorRole
    data_role: EvaluationDataRole
    reads_immutable_evidence: Literal[True] = True
    can_gate_runtime: Literal[False] = False
    can_mutate_current_execution: Literal[False] = False
    can_select_final_artifact: Literal[False] = False
    may_emit_feedback: bool = False
    may_trigger_bounded_new_attempt: bool = False
    may_propose_prior_snapshot: bool = False
    automatic_policy_update: Literal[False] = False
    eligible_as_final_outcome: bool = False
    required_controls: list[str] = Field(min_length=1)
    prohibited_uses: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_role_authority(self) -> EvaluatorRoleContract:
        expected = {
            "offline_prior_estimator": (
                "development_calibration",
                False,
                False,
                True,
                False,
            ),
            "post_delivery_development_diagnostic": (
                "development_diagnostic",
                False,
                False,
                False,
                False,
            ),
            "runtime_feedback_critic": (
                "experiment_runtime",
                True,
                True,
                False,
                False,
            ),
            "heldout_final_judge": (
                "heldout_test",
                False,
                False,
                False,
                True,
            ),
        }[self.evaluator_role]
        actual = (
            self.data_role,
            self.may_emit_feedback,
            self.may_trigger_bounded_new_attempt,
            self.may_propose_prior_snapshot,
            self.eligible_as_final_outcome,
        )
        if actual != expected:
            raise ValueError(
                f"authority does not match evaluator role {self.evaluator_role}: "
                f"{actual}"
            )
        return self


def build_evaluator_role_contract(role: EvaluatorRole) -> EvaluatorRoleContract:
    """Return the canonical authority profile for an evaluator role."""

    if role == "offline_prior_estimator":
        return EvaluatorRoleContract(
            evaluator_role=role,
            data_role="development_calibration",
            may_propose_prior_snapshot=True,
            required_controls=[
                "development-only cohort",
                "route-blind judging before route-level aggregation",
                "versioned evidence and scorer identities",
                "explicit prior freeze before a preregistered experiment",
            ],
            prohibited_uses=[
                "reading held-out or test outcomes",
                "per-case route adaptation during an experiment",
                "automatic Planner or capability-catalog mutation",
                "reporting the estimate as a final benchmark outcome",
            ],
        )
    if role == "post_delivery_development_diagnostic":
        return EvaluatorRoleContract(
            evaluator_role=role,
            data_role="development_diagnostic",
            required_controls=[
                "development-only cohort",
                "immutable successful delivery input",
                "score output outside the execution run",
                "versioned crop, rubric, voting, and abstention protocol",
            ],
            prohibited_uses=[
                "updating a tool-prior snapshot",
                "feeding scores or reasons back into the measured execution",
                "triggering repair, rerun, or artifact selection",
                "reporting diagnostics as a held-out final outcome",
            ],
        )
    if role == "runtime_feedback_critic":
        return EvaluatorRoleContract(
            evaluator_role=role,
            data_role="experiment_runtime",
            may_emit_feedback=True,
            may_trigger_bounded_new_attempt=True,
            required_controls=[
                "preregistered feedback arm",
                "immutable checkpoint input",
                "fixed feedback budget and attempt limit",
                "all new attempts re-enter the normal forward path",
            ],
            prohibited_uses=[
                "changing the frozen tool prior",
                "overriding Verifier or ArtifactStore authority",
                "silently retrying or selecting a best hidden attempt",
                "serving as the final benchmark judge",
            ],
        )
    return EvaluatorRoleContract(
        evaluator_role=role,
        data_role="heldout_test",
        eligible_as_final_outcome=True,
        required_controls=[
            "held-out evaluator sidecar isolation",
            "frozen scorer and rubric protocol",
            "no feedback to the measured execution",
            "complete retention of failures and unknowns",
        ],
        prohibited_uses=[
            "changing the frozen tool prior",
            "triggering repair or a new attempt",
            "selecting among outputs after observing judge scores",
            "exposing hidden evaluator fields to runtime modules",
        ],
    )


__all__ = [
    "EvaluationDataRole",
    "EvaluatorRole",
    "EvaluatorRoleContract",
    "build_evaluator_role_contract",
]
