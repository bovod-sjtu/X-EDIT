"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import time


from x_editor.revision.atomic_module_review import invoke, read, write


from x_editor.agents.openai_compatible import OpenAICompatibleServiceUnavailable


from x_editor.evaluation.minimal_atomic_review import (
    PROTOCOL_ID, ATOMIC_PROMPT, CONSISTENCY_PROMPT, MinimalAtomicAnswer,
    atomic_prompt, build_catalog, restore_catalog, validate_answer,
    materialize_answer, feedback_candidates, consistency_input, consistency_prompt,
    validate_catalog_consistency,
)


def request_once(path, prompt, args, *, author=False):
    """Resume accepted responses or retain failed attempts, never reask a question."""
    prior = path / 'calls' / '0001'
    if prior.exists():
        if (prior / 'response.json').is_file():
            return read(prior / 'response.json')
        raise RuntimeError('Previously started request has no accepted JSON response; original attempt retained without retry')
    return invoke(path, prompt, args, author=author)


def judge(path, question, design, catalog, args):
    write(path / 'question.json', question.model_dump(mode='json'))
    prompt = atomic_prompt(question, design, catalog)
    write(path / 'prompt.json', prompt)
    if (path / 'receipt.json').exists():
        return
    if args.dry_run:
        return
    start = time.monotonic()
    try:
        answer = validate_answer(question, request_once(path, prompt, args), catalog)
        write(path / 'answer.json', answer.model_dump(mode='json'))
        write(path / 'resolved_citations.json', materialize_answer(answer, catalog))
        result = {'status': 'completed', 'grade': answer.grade}
    except (ValueError, TypeError, OSError, RuntimeError, OpenAICompatibleServiceUnavailable) as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
    write(path / 'receipt.json', {**result, 'elapsed_sec': time.monotonic() - start, 'protocol_id': PROTOCOL_ID})


def review(path, questions, answers, catalog, args):
    payload = consistency_input(questions, answers, catalog)
    prompt = consistency_prompt(payload)
    write(path / 'input.json', payload)
    write(path / 'prompt.json', prompt)
    if (path / 'receipt.json').exists():
        return validate_catalog_consistency(read(path / 'review.json'), payload) if (path / 'review.json').exists() else None
    start, result, reviewed = time.monotonic(), {}, None
    try:
        if not answers:
            result = {'status': 'unavailable', 'reason': 'no valid atomic answers'}
        else:
            reviewed = validate_catalog_consistency(request_once(path, prompt, args, author=True), payload)
            write(path / 'review.json', reviewed.model_dump(mode='json'))
            write(path / 'resolved_citations.json', [
                {'question_ids': list(issue.question_ids),
                 'citations': [payload['evidence_catalog']['entries'][c.evidence_id] for c in issue.citations]}
                for issue in reviewed.issues])
            result = {'status': 'completed', 'review_status': reviewed.status}
    except (ValueError, TypeError, OSError, RuntimeError, OpenAICompatibleServiceUnavailable) as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
    write(path / 'receipt.json', {**result, 'elapsed_sec': time.monotonic() - start})
    return reviewed
