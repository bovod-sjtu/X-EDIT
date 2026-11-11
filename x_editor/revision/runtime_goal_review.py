"""Production review support extracted without changing function bodies."""

import json


from pydantic import Field


from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel


from x_editor.evaluation.iterative_runtime import write_record


from x_editor.evaluation.refinement_contracts import CriterionEntailment


from x_editor.revision.review_validation import invoke_validated_review


class CriterionEntailments(FrozenEvaluationModel):
    criteria: tuple[CriterionEntailment, ...] = Field(min_length=1, max_length=24)


def audit_runtime_questions(design, context, source_notes, *, output, args, invoke, prompt):
    """Share public context once, retaining a separately justified verdict per question."""
    def complete_questions(answer):
        ids = [a.criterion_id for a in answer.criteria]
        if len(ids) != len(set(ids)) or set(ids) != {c.criterion_id for c in design.criteria}:
            raise ValueError('batch qualification must address every original question exactly once')

    response = invoke_validated_review(model=CriterionEntailments, invoke=invoke,
        directory=output/'criterion_entailment_batch', prompt={
        'system_prompt':prompt + '\n本次逐题填写criteria数组，每个criterion_id恰好一次；不能以总体合理代替逐题说明。',
        'user_prompt':json.dumps({'schema':CriterionEntailments.model_json_schema(),
            'public_context':context.model_dump(mode='json'),'source_observations':source_notes,
            'criteria':[c.model_dump(mode='json') for c in design.criteria],
            'retention_coverage':[r.model_dump(mode='json') for r in getattr(design, 'retention_coverage', ())]},ensure_ascii=False),
    }, args=args, validate=complete_questions)
    # Each raw draft is retained by invoke_validated_review; this is the accepted
    # structural response, whose semantic verdict may still reject the design.
    write_record(output/'criterion_entailment_batch_validated.json',response.model_dump(mode='json'))
    answers=response.criteria
    return [{'stage':'criterion_entailment','criterion_id':a.criterion_id,'review':a.model_dump(mode='json')}
            for a in answers if not a.supported or a.requirement_basis=='unsupported']
