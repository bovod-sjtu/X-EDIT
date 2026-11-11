"""Version and occurrence evidence for read-only final-delivery review.

Runtime state identifies the artifact being judged; it does not establish that
the selected state satisfies the user's intent. Coverage items identify observed
edit locations, not an oracle inventory of every requested change. The judge
must still resolve intent and report incomplete or contradictory evidence.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeliveryReviewTurn(_ReviewModel):
    turn_index: int = Field(ge=0)
    user_request: str = Field(min_length=1)


class DeliveryReviewState(_ReviewModel):
    state_id: str = Field(min_length=1)
    artifact_ref_id: str | None = None
    on_terminal_branch: bool


class DeliveryReviewContext(_ReviewModel):
    """Allowlisted public dialogue and graph identities, without tool details."""

    schema_version: Literal["x-editor.delivery-review-context/v1"] = (
        "x-editor.delivery-review-context/v1"
    )
    turns: tuple[DeliveryReviewTurn, ...] = Field(min_length=1)
    current_turn_index: int = Field(ge=0)
    terminal_artifact_ref_id: str = Field(min_length=1)
    terminal_state_id: str = Field(min_length=1)
    round_entry_state_id: str | None = None
    selected_base_state_id: str | None = None
    states: tuple[DeliveryReviewState, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identities(self) -> DeliveryReviewContext:
        indices = [turn.turn_index for turn in self.turns]
        if indices != sorted(set(indices)) or indices[-1] != self.current_turn_index:
            raise ValueError("current turn must be the last unique, ordered public turn")
        states = {state.state_id: state for state in self.states}
        if len(states) != len(self.states):
            raise ValueError("delivery state IDs must be unique")
        for state_id in (
            self.terminal_state_id,
            self.round_entry_state_id,
            self.selected_base_state_id,
        ):
            if state_id is not None and state_id not in states:
                raise ValueError("delivery role references an unknown state")
        terminal = states[self.terminal_state_id]
        if (
            terminal.artifact_ref_id != self.terminal_artifact_ref_id
            or not terminal.on_terminal_branch
        ):
            raise ValueError("terminal state must bind the terminal output artifact")
        return self

    @property
    def current_request(self) -> str:
        return self.turns[-1].user_request


class DeliveryCoverageRequirement(_ReviewModel):
    """One location the judge must observe before summarizing a rubric."""

    coverage_id: str = Field(min_length=1)
    rubric_id: str = Field(min_length=1)
    evidence_region_ids: tuple[str, ...] = ()


class DeliveryCoverageObservation(_ReviewModel):
    coverage_id: str = Field(min_length=1)
    source_observation: str = Field(min_length=1)
    output_observation: str = Field(min_length=1)
    verdict: Literal["fulfilled", "partial", "violated", "unknown"]
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class DeliveryReviewResult(_ReviewModel):
    """Judge-authored observations retained separately from aggregate grades."""

    evaluated_artifact_ref_id: str = Field(min_length=1)
    current_turn_index: int = Field(ge=0)
    current_request_interpretation: str = Field(min_length=1)
    intent_status: Literal["resolved", "ambiguous"]
    coverage_completeness: Literal["complete", "partial", "unknown"]
    observations: tuple[DeliveryCoverageObservation, ...] = Field(min_length=1)


class DeliveryCoverageReferenceBinding(_ReviewModel):
    """Auditable expansion of one observation ID into its validated audio refs."""

    rubric_id: str = Field(min_length=1)
    coverage_id: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


def bind_delivery_coverage_references(
    *,
    review: DeliveryReviewResult,
    requirements: tuple[DeliveryCoverageRequirement, ...],
    judgements: tuple[Any, ...],
) -> tuple[tuple[Any, ...], tuple[DeliveryCoverageReferenceBinding, ...]]:
    """Resolve explicit observation references after the review was validated.

    IDs are foreign keys in a typed task. They are not inferred from prefixes,
    prose, audio labels, or expected answers. Cross-rubric and unknown IDs remain
    unchanged for the existing evidence validator to reject.
    """

    observations = {item.coverage_id: item for item in review.observations}
    owned = {(item.rubric_id, item.coverage_id) for item in requirements}
    resolved, bindings = [], []
    for judgement in judgements:
        refs = []
        for ref in judgement.evidence_refs:
            if (judgement.rubric_id, ref) in owned:
                replacement = observations[ref].evidence_refs
                refs.extend(replacement)
                bindings.append(
                    DeliveryCoverageReferenceBinding(
                        rubric_id=judgement.rubric_id,
                        coverage_id=ref,
                        evidence_refs=replacement,
                    )
                )
            else:
                refs.append(ref)
        resolved.append(judgement.model_copy(update={"evidence_refs": tuple(dict.fromkeys(refs))}))
    return tuple(resolved), tuple(bindings)


def validate_delivery_review(
    payload: Any,
    *,
    context: DeliveryReviewContext,
    requirements: tuple[DeliveryCoverageRequirement, ...],
    judgements: dict[str, Any],
    allowed_evidence_refs: set[str],
) -> DeliveryReviewResult:
    """Reject wrong-version, omitted-location and structured contradictions.

    This does not parse free prose or assert that a model's listening is true.
    Semantic contradictions within prose still need independent review. An
    invalid response is not repaired into a passing or failing audio verdict.
    """

    review = DeliveryReviewResult.model_validate(payload)
    if review.evaluated_artifact_ref_id != context.terminal_artifact_ref_id:
        raise ValueError("delivery review judged a historical or different artifact")
    if review.current_turn_index != context.current_turn_index:
        raise ValueError("delivery review judged a different public turn")
    observations = {item.coverage_id: item for item in review.observations}
    if len(observations) != len(review.observations):
        raise ValueError("delivery review repeats a coverage observation")
    if set(observations) != {item.coverage_id for item in requirements}:
        raise ValueError("delivery review did not observe every supplied coverage item")
    for requirement in requirements:
        observation = observations[requirement.coverage_id]
        refs = set(observation.evidence_refs)
        if not refs.issubset(allowed_evidence_refs):
            raise ValueError("delivery observation cites unknown evidence")
        if requirement.evidence_region_ids and not set(requirement.evidence_region_ids).issubset(
            refs
        ):
            raise ValueError("delivery observation must cite both paired regions")

    for rubric_id in {item.rubric_id for item in requirements}:
        judgement = judgements[rubric_id]
        relevant = [
            observations[item.coverage_id] for item in requirements if item.rubric_id == rubric_id
        ]
        unresolved = (
            review.intent_status == "ambiguous"
            or review.coverage_completeness != "complete"
            or any(item.verdict == "unknown" for item in relevant)
        )
        if unresolved:
            if judgement.applicability != "unknown" or not judgement.abstain:
                raise ValueError("unresolved delivery coverage requires an unknown judgement")
            continue
        if judgement.applicability != "applicable" or judgement.abstain:
            continue
        all_fulfilled = all(item.verdict == "fulfilled" for item in relevant)
        if all_fulfilled != (judgement.grade == 3):
            raise ValueError("delivery grade contradicts complete per-location observations")
        if all(item.verdict == "violated" for item in relevant) and judgement.grade != 0:
            raise ValueError("delivery grade contradicts wholly violated observations")
    return review


DELIVERY_REVIEW_OUTPUT_SHAPE = {
    "evaluated_artifact_ref_id": "<terminal artifact ref from delivery_context>",
    "current_turn_index": 0,
    "current_request_interpretation": "observable current goal after considering history",
    "intent_status": "resolved|ambiguous",
    "coverage_completeness": "complete|partial|unknown",
    "observations": [
        {
            "coverage_id": "<exact supplied id>",
            "source_observation": "what is heard in the source",
            "output_observation": "what is heard in the final output",
            "verdict": "fulfilled|partial|violated|unknown",
            "evidence_refs": ["<exact supplied evidence refs>"],
        }
    ],
}


DELIVERY_REVIEW_INSTRUCTIONS = """The delivery_context names the exact final
artifact and the current public turn. Earlier turns are conversational history,
not simultaneous commands to reapply. Use them to resolve references and retained
requirements. A selected base or excluded branch describes what runtime delivered;
it is not proof that runtime chose the meaning intended by the user. If restore,
delete, or another interpretation remains genuinely ambiguous, use intent_status
ambiguous and return unknown for instruction following. A state without an
attached artifact has unavailable audio, not silent audio.

Before grading, observe EVERY supplied coverage item independently. Write what is
actually heard in the source and in the output, in that order, and whether that
location fulfills the current request. For repeated edits, do not let one correct
occurrence stand for the others. Cite both source/output region IDs for each local
pair. These pairs come from the executed edit inventory, not an exhaustive target
annotation: assess whether requested locations outside it could be missing. Set
coverage_completeness to partial or unknown unless the attached evidence supports
the full requested scope. Do not infer full-recording quality from supplied crops.
Unknown intent, incomplete coverage, or an unknown location requires an unknown,
abstaining instruction judgement. With complete resolved coverage, all fulfilled
locations require grade 3, any partial/violated location forbids grade 3, and all
violated locations require grade 0. Assess quality/preservation in their separate
rubrics. Check that your grade agrees with your own observations; do not reverse
the source-to-output direction or silently revise an observation to match a grade.

The delivery_review field in the required response shape is mandatory. Write its
observations before the aggregate judgements. Never omit it when giving grades.
"""
