"""Identify actual independent advice, not the integrated adapter's memo shim."""

import json

from x_editor.reasoner.contracts import ReasonerCallRequest, SemanticReasoningMemo


def advisory_metadata(trace):
    trace = trace or {}
    metadata = trace.get('metadata') or {}
    if (trace.get('component_name') == 'IndependentAdviser'
            and metadata.get('architecture') == 'common_planner_advisory'):
        return metadata
    return None


def absent_adviser(trace):
    metadata = advisory_metadata(trace)
    return metadata is not None and metadata.get('independent_guidance') is False


def actual_adviser_io(module_input, trace):
    """Fail closed if the old recording omitted the real call's input/output.

    In particular, do not reconstruct the prior memo from a Planner summary.
    Historical artifacts remain untouched; missing provenance is unjudged.
    """
    metadata = advisory_metadata(trace)
    if metadata is None:
        return None
    if metadata.get('independent_guidance') is not True:
        raise ValueError('no independent Reasoner invocation to evaluate')
    if metadata.get('advice_used') is not True:
        raise ValueError('independent Reasoner advice unavailable; adapter summary is not its output')
    if not metadata.get('adviser_request') or not metadata.get('adviser_memo'):
        raise ValueError('actual independent Reasoner input/output missing from recording')
    request = ReasonerCallRequest.model_validate(metadata['adviser_request'])
    memo = SemanticReasoningMemo.model_validate(metadata['adviser_memo'])
    if (request.request_id != module_input.get('request_id')
            or request.user_request != module_input.get('user_request')
            or memo.request_id != request.request_id):
        raise ValueError('independent Reasoner call identity differs from linked checkpoint')
    return request.model_dump(mode='json'), memo.model_dump(mode='json')


def planner_advice_from_wire(request_id, wire_index):
    """Return only advice actually present in the identity-bound Planner call."""
    matches = wire_index.get(request_id, [])
    if len(matches) != 1:
        raise ValueError('actual Planner prompt must have one identity-bound recording')
    path = matches[0] / 'request.json'
    bodies = []
    for message in json.loads(path.read_text()).get('chat_payload', {}).get('messages', []):
        if message.get('role') != 'user' or not isinstance(message.get('content'), str):
            continue
        try:
            public = json.loads(message['content'])
        except ValueError:
            continue
        if not isinstance(public, dict):
            continue
        core = public.get('planner_input', public)
        if isinstance(core, dict) and core.get('planner_request_id') == request_id:
            bodies.append(public)
    if len(bodies) != 1:
        raise ValueError('actual Planner prompt identity is missing or ambiguous')
    return bodies[0].get('independent_guidance'), str(path)
