"""Monte Carlo selection of realized artifacts, not action-value learning.

Each draw is a complete public-rubric assessment. Average the existing utility
*after* its per-draw bottleneck calculation; min(mean(dimensions)) is a different
objective. Unknown draws retain [0, 1] bounds and never disappear from the count.
"""

import math
from statistics import mean, stdev

from x_editor.evaluation.refinement_flow_selection import observed_constraints, retention_constraints


def summarize_draws(draws):
    if not draws:
        raise ValueError("MC selection requires every scheduled draw")
    if any(not math.isfinite(d[k]) or not 0 <= d['lower'] <= d['upper'] <= 1
           for d in draws for k in ('lower', 'upper')):
        raise ValueError('invalid proxy utility bounds')
    lower = [d['lower'] for d in draws]
    result = {
        'count': len(draws), 'mean_lower': mean(lower),
        'mean_upper': mean(d['upper'] for d in draws),
        'sample_sd_lower': stdev(lower) if len(lower) > 1 else None,
        'complete_draws': sum(d['known_criteria'] == d['assigned_criteria']
                              and not d.get('design_pending') for d in draws),
        'uncertainty': 'recorded-proxy bounds; not human-quality confidence intervals',
    }
    for field in ('preservation', 'critical_constraints'):
        keys = set().union(*(d.get(field, {}) for d in draws))
        result[field] = {key: (mean(d[field][key] for d in draws)
            if all(d.get(field, {}).get(key) is not None for d in draws) else None)
            for key in sorted(keys)}
    return result


def select_mc_attempt(attempts, *, repeats):
    """Same first-delivery anchor; protection uses complete mean proxy grades.

No calibrated non-regression guarantee follows from these noisy observations.
Every delivered candidate is sampled before eligibility is decided.
"""
    ordered = sorted(attempts, key=lambda r: r['attempt_index'])
    if len({r['attempt_index'] for r in ordered}) != len(ordered):
        raise ValueError('duplicate attempt index')
    stats = {}
    for row in ordered:
        if row['delivered']:
            draws = row.get('mc_utilities', [])
            if len(draws) != repeats:
                raise ValueError('missing MC draws must be recorded as unknown, not omitted')
            stats[row['attempt_index']] = summarize_draws(draws)
    anchor = next((r for r in ordered if r['delivered']), None)
    protected = retention_constraints(stats[anchor['attempt_index']]) if anchor else {}
    chosen, decisions = None, []
    for row in ordered:
        stat = stats.get(row['attempt_index'])
        observed = observed_constraints(stat) if stat else {}
        losses = [k for k, v in protected.items() if v is not None
                  and (observed.get(k) is None or observed[k] < v)]
        eligible = row['delivered'] and not losses
        decisions.append({'attempt_index': row['attempt_index'], 'eligible': eligible,
                          'preservation_losses_or_unknowns': losses, 'mc': stat})
        if eligible and (chosen is None or stat['mean_lower'] > stats[chosen['attempt_index']]['mean_lower']):
            chosen = row
    for decision in decisions:
        decision['selected'] = chosen is not None and decision['attempt_index'] == chosen['attempt_index']
    return chosen, decisions
