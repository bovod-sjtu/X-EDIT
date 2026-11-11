"""Fixed-evidence ordinal decisions with true option-order counterbalancing."""
import hashlib
import math
import random
from statistics import median, mean


def presentations(levels, identity):
    """Three Latin rotations; fixed A-D positions, changing description order.

    Four rounds would exactly balance every item over four positions within one
    question. Three rounds reduce, but cannot eliminate, positional exposure.
    The starting order is independently randomized per public question.
    """
    order = list(range(4))
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(order)
    result = []
    for offset in range(3):
        grades = order[offset:] + order[:offset]
        mapping = dict(zip('ABCD', grades))
        result.append((mapping, [{'choice': letter, 'description': levels[str(g)]}
                                 for letter, g in mapping.items()]))
    return result


def decode(raw, mapping=None):
    """A missing confidence does not erase an otherwise valid grade."""
    expected = 'choice' if mapping is not None else 'grade'
    if set(raw) != {expected, 'confidence'}:
        raise ValueError('unexpected answer fields')
    value = raw[expected]
    if mapping is not None:
        if value == 'U':
            value = None
        elif value in mapping:
            value = mapping[value]
        else:
            raise ValueError('unknown choice label')
    elif isinstance(value, str) and value in ('0', '1', '2', '3'):
        value = int(value)
    if value is not None and (type(value) is not int or value not in range(4)):
        raise ValueError('invalid grade')
    confidence = raw['confidence']
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None
    valid_confidence = type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1
    return {'grade': value, 'confidence': confidence if value is not None and valid_confidence else None,
            'confidence_status': 'reported' if value is not None and valid_confidence else 'unavailable',
            'status': 'answered'}


def aggregate(answers):
    if len(answers) != 3:
        raise ValueError('the frozen ensemble requires three attempted decisions')
    grades = [a.get('grade') for a in answers]
    lower = int(median([0 if g is None else g for g in grades]))
    upper = int(median([3 if g is None else g for g in grades]))
    known = [g for g in grades if g is not None]
    confidences = [a['confidence'] for a in answers if a.get('grade') is not None and a.get('confidence') is not None]
    return {'status': 'answered', 'grade': lower if lower == upper else None,
            'confidence': mean(confidences) if lower == upper and confidences else None,
            'ordinal_interval': [lower, upper], 'valid_grade_count': len(known),
            'vote_grades': grades, 'grade_spread': max(known)-min(known) if known else None,
            'binary_disagreement': len({g >= 2 for g in known}) > 1,
            'failed_calls': sum(a.get('status') == 'failed' for a in answers),
            'confidence_mean_is_calibrated': False,
            'interpretation': 'Unknown votes range over 0..3; publish a grade only when the median is determined.'}
