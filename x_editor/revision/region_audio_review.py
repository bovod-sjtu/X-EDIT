"""Production review support extracted without changing function bodies."""

from pydantic import Field


from x_editor.evaluation.edit_region_scope import (
    CheckedAudioRegion, RegionalObservation, validate_regional_observation,
)


from x_editor.evaluation.refinement_contracts import UtilityAnswer


from x_editor.evaluation.terminal_observation_review import (
    ObservedUtilityAnswer, validate_terminal_references,
)


class RegionalUtilityAnswer(ObservedUtilityAnswer):
    checked_regions: tuple[CheckedAudioRegion, ...] = Field(default=(), max_length=16)
    unresolved_region_mapping: tuple[str, ...] = Field(default=(), max_length=8)


class BoundaryAudioRegion(CheckedAudioRegion):
    boundary_sec: float = Field(ge=0, allow_inf_nan=False)


class BoundaryUtilityAnswer(RegionalUtilityAnswer):
    checked_regions: tuple[BoundaryAudioRegion, ...] = Field(default=(), max_length=16)


def region_answer_schema(check):
    model = BoundaryUtilityAnswer if check.focus == "boundary_transition" else RegionalUtilityAnswer
    return model.model_json_schema()


def regional_answer(payload, criterion_id, check, inventory):
    answer = validate_terminal_references(
        RegionalUtilityAnswer.model_validate(payload), criterion_id, inventory,
    )
    observation = RegionalObservation(
        criterion_id=criterion_id, focus=check.focus,
        checked_regions=answer.checked_regions,
        unresolved_region_mapping=answer.unresolved_region_mapping,
    )
    validate_regional_observation(observation, answer, check, inventory)
    return UtilityAnswer(
        question_id=criterion_id, grade=answer.grade, finding=answer.observed_condition,
        evidence_ids=answer.evidence_ids,
    ), observation
