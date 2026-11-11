"""Read newer diagnostic snapshots with an older, otherwise strict reader.

Only absent-in-the-reader, neutral defaults introduced by checkpoint tooling
may be projected away. This is not a runtime plan repair or a general extra-field
filter. The original snapshot and evidence hashes remain immutable.
"""
from copy import deepcopy

from pydantic import ValidationError


def _neutral_extension(location, value):
    if len(location) < 4 or location[0] != 'turn_results' or not isinstance(location[1], int):
        return False
    tail = location[2:]
    if tail in (('result', 'forward_result', 'runtime_audio_bindings'),
                ('result', 'forward_result', 'recovery_session_state')):
        return value == [] if tail[-1] == 'runtime_audio_bindings' else value is None
    if location[-1] != 'span_timebase' or value != 'observed_audio':
        return False
    if tail[:2] == ('result', 'final_plan'):
        suffix = tail[2:]
    elif tail[:3] in (('result', 'forward_result', 'input_plan'),
                      ('result', 'forward_result', 'executed_plan')):
        suffix = tail[3:]
    elif (tail[:2] == ('result', 'planner_history') and len(tail) > 3
          and isinstance(tail[2], int) and tail[3] == 'plan_snapshot'):
        suffix = tail[4:]
    else:
        return False
    return len(suffix) == 3 and suffix[0] == 'steps' and isinstance(suffix[1], int)


def read_recorded_result(payload, result_type):
    """Return (validated result, explicit compatibility projections)."""
    try:
        return result_type.model_validate(payload), []
    except ValidationError as error:
        projected = deepcopy(payload)
        changes = []
        for problem in error.errors():
            location = problem['loc']
            if problem['type'] != 'extra_forbidden':
                raise
            parent = projected
            for key in location[:-1]:
                parent = parent[key]
            value = parent[location[-1]]
            if not _neutral_extension(location, value):
                raise
            changes.append({'path': list(location), 'value': value,
                            'reason': 'neutral checkpoint extension unsupported by frozen reader'})
            del parent[location[-1]]
        return result_type.model_validate(projected), changes
