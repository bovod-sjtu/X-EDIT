"""Public-question degradation when task-specific authoring is unavailable.

This is a weaker observation protocol, not a qualification of failed questions.
It invents neither target spans nor a resolved interpretation of the dialogue.
The reviewer still receives the entire public dialogue and actual audio assets.
"""

from typing import Literal

from x_editor.evaluation.refinement_contracts import PublicUtilityDesign, UtilityCriterion


class BaseTerminalDesign(PublicUtilityDesign):
    protocol: Literal["x-editor.base-terminal-fallback/v1"] = "x-editor.base-terminal-fallback/v1"


def base_terminal_design():
    return BaseTerminalDesign(
        final_state="Follow the complete public dialogue, resolving only explicit revisions; "
        "do not infer successful editing from execution or from the presence of a file.",
        criteria=(
            UtilityCriterion(criterion_id="base_goal", dimension="goal",
                question="Does the actual candidate satisfy every still-active edit requirement in "
                "the supplied public dialogue? Identify omitted targets or effects separately. "
                "If a necessary target, reference or comparison version is ambiguous, return unknown."),
            UtilityCriterion(criterion_id="base_preservation", dimension="preservation",
                question="Compare the actual sources/references and candidate: are contents, speakers, "
                "background components and attributes outside authorized changes preserved? "
                "Preservation can overlap the edited interval. Do not require sample identity or "
                "unchanged duration when regeneration is authorized. Missing necessary versions mean unknown."),
            UtilityCriterion(criterion_id="base_boundary", dimension="perceptual_quality", weight=0.5,
                question="Does the candidate have smooth continuity without newly introduced clicks, "
                "unintended repetitions, truncation or abrupt joins? Cite audible candidate times, "
                "but do not invent splice positions. Respect intentionally requested discontinuities."),
            UtilityCriterion(criterion_id="base_quality", dimension="perceptual_quality", weight=0.5,
                question="Is the candidate free of unintended distortion or clarity loss relative to "
                "the source? Respect requested degradation/style; do not count missing target words "
                "again as recording quality errors. Uncertain comparisons remain unknown."),
        ),
    )


def observation_confidence(design, answers):
    """Report inspectable coverage, never convert an ordinal score to probability."""
    known = {a.question_id for a in answers if a.grade is not None}
    required = {c.criterion_id for c in design.criteria}
    return {
        "protocol": getattr(design, "protocol", "public_utility"),
        "question_count": len(required),
        "answered_count": len(required & known),
        "answer_coverage": len(required & known) / len(required),
        "unresolved_question_ids": sorted(required - known),
        "scope_specificity": "whole_audio_fallback" if isinstance(design, BaseTerminalDesign)
        else "authored_requirements",
        "calibrated_probability": None,
        "calibration_status": "requires_independent_correct_and_error_controls",
        "more_questions_do_not_imply_more_confidence": True,
    }
