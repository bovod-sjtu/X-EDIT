"""Semantic eligibility for downstream listening, never Forward admission.

Every required atomic decision check must meet the declared level. Missing or
withheld judgments remain unknown; an average cannot hide a failed condition.
"""


def decision_review_gate(reviews, *, expected_positions, errors=(), minimum_grade=2):
    if type(minimum_grade) is not int or minimum_grade not in (1, 2, 3):
        raise ValueError("decision minimum grade must be an integer from 1 to 3")
    positions, issues = [], list(map(str, errors))
    seen = set()
    for review in reviews:
        position = (review['turn_index'], review['module'])
        if position in seen:
            issues.append(f'duplicate decision review: {position}')
        seen.add(position)
        answers = {a['question_id']: a for a in review['answers']}
        withheld = set(review.get('withheld_as_facts', ()))
        if len(answers) != len(review['answers']):
            issues.append(f'duplicate decision answer: {position}')
        checks = []
        for question in review['questions']:
            identity = question['question_id']
            grade = answers.get(identity, {}).get('grade')
            known = identity not in withheld and type(grade) is int and 0 <= grade <= 3
            checks.append({'question_id': identity, 'grade': grade if known else None,
                           'status': ('passed' if grade >= minimum_grade else 'failed')
                           if known else 'unknown'})
        if not checks:
            issues.append(f'no decision questions: {position}')
        positions.append({'turn_index': position[0], 'module': position[1], 'checks': checks})
    missing = set(expected_positions) - seen
    issues.extend(f'missing decision position: {p}' for p in sorted(missing))
    if not expected_positions:
        issues.append('no decision checkpoints available')
    checks = [c for p in positions for c in p['checks']]
    failed = any(c['status'] == 'failed' for c in checks)
    unknown = bool(issues) or any(c['status'] == 'unknown' for c in checks)
    status = 'failed' if failed else 'unknown' if unknown else 'passed'
    return {'protocol': 'x-editor.decision-first/v1', 'status': status,
            'minimum_grade': minimum_grade, 'quality_review_allowed': status == 'passed',
            'positions': positions, 'errors': issues,
            'forward_admission_authority': False, 'audio_quality_certified': False}
