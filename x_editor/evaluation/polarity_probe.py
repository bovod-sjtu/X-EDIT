"""Fixed-scale polarity checks for offline ordinal evaluator calibration."""
from math import isfinite


def reverse_levels(levels):
    """Mirror the existing ordinal anchors; no new quality requirement is added."""
    if set(levels) != {'0', '1', '2', '3'}:
        raise ValueError('four original ordinal anchors are required')
    introductions = (
        '反向命题不成立。原来正向要求的满足情况：',
        '反向命题仅轻微成立。原来正向要求的满足情况：',
        '反向命题明显成立。原来正向要求的满足情况：',
        '反向命题充分成立。原来正向要求的满足情况：',
    )
    return {str(n): introductions[n] + levels[str(3-n)] for n in range(4)}


def to_positive(reverse_grade):
    if reverse_grade is None:
        return None
    if type(reverse_grade) is not int or reverse_grade not in range(4):
        raise ValueError('reverse grade must be 0..3 or unknown')
    return 3-reverse_grade


def balanced_score(forward, reverse_grade):
    """Equal-weight directions; preserve half-grades and missing evidence."""
    reverse = to_positive(reverse_grade)
    if forward is None or reverse is None:
        return None
    if type(forward) not in (int, float) or not isfinite(forward) or not 0 <= forward <= 3:
        raise ValueError('forward grade must be finite and in 0..3')
    return (forward+reverse)/2


def polarity_results(first, ensemble, reverse):
    positive = to_positive(reverse)
    return {
        'first': first, 'label_ensemble': ensemble,
        'reverse_raw': reverse, 'reverse_positive': positive,
        'first_plus_reverse': balanced_score(first, reverse),
        'ensemble_plus_reverse': balanced_score(ensemble, reverse),
        'first_polarity_gap': abs(first-positive) if first is not None and positive is not None else None,
        'ensemble_polarity_gap': abs(ensemble-positive) if ensemble is not None and positive is not None else None,
    }
