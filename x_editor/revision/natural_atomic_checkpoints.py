"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import json


from collections import Counter, defaultdict


from x_editor.revision.atomic_module_review import public_context, read, write


from x_editor.revision.scoped_atomic_calibration import digest, load_row


from x_editor.reasoner.contracts import ReasonerCallRequest


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_ref(path):
    return {'path': str(path.resolve()), 'sha256': digest(path)}


def provider_request_ids(trace):
    return {row.get('metrics', {}).get('request_id') for row in (trace or {}).get('provider_invocations', [])} - {None, ''}


def link_checkpoint(module, module_input, receipt, checkpoints):
    """Use documented request identity, then corroborate recorded invocation IDs."""
    request_id = module_input.get('request_id' if module == 'reasoner' else 'planner_request_id')
    matches = []
    for checkpoint, ref in checkpoints:
        record = checkpoint.output.get('record', {})
        if checkpoint.module not in ({'reasoner'} if module == 'reasoner' else {'planner', 'planner_reasoner'}):
            continue
        if not isinstance(record.get('call_index'), int):
            continue
        exact = f'{checkpoint.context.workflow_run_id}:semantic:{record["call_index"]}'
        if request_id != exact:
            continue
        original_ids = provider_request_ids(receipt.get('invocation_trace'))
        checkpoint_ids = provider_request_ids(record.get('invocation_trace'))
        if original_ids and checkpoint_ids and original_ids != checkpoint_ids:
            raise ValueError('Recorded provider invocation identity conflicts with checkpoint')
        if module == 'reasoner' and record.get('request_fingerprint'):
            if ReasonerCallRequest.model_validate(module_input).fingerprint != record['request_fingerprint']:
                raise ValueError('Reasoner input fingerprint differs from recorded checkpoint')
        matches.append((checkpoint, ref))
    if len(matches) != 1:
        raise ValueError(f'Exact checkpoint binding count is {len(matches)}, expected one')
    return matches[0]


def planner_wire_index(root):
    result = defaultdict(list)
    for path in sorted((root / 'http').glob('*/request.json')):
        request = read(path)
        payload = request.get('chat_payload')
        if not isinstance(payload, dict):
            continue
        for message in payload.get('messages', []):
            if message.get('role') != 'user' or not isinstance(message.get('content'), str):
                continue
            try:
                public = json.loads(message['content'])
            except ValueError:
                continue
            if not isinstance(public, dict):
                continue
            # Current integrated/Graph prompts put the same request identity
            # at the top level; older separated prompts wrap planner_input.
            planner_input = public.get('planner_input', public)
            if isinstance(planner_input, dict) and planner_input.get('planner_request_id'):
                result[planner_input['planner_request_id']].append(path.parent)
    return result


def read_output(call, module, module_input, wire_index):
    accepted_path = call / 'output.json'
    if accepted_path.is_file():
        return read(accepted_path), True, {'representation': 'accepted_canonical', 'source': source_ref(accepted_path)}
    # The root HTTP recorder has no invocation-scope identity for old Reasoner
    # prompts. A missing rejected Reasoner answer is unknown, never borrowed.
    candidates = wire_index.get(module_input.get('planner_request_id'), []) if module == 'planner' else []
    available = [root for root in candidates if (root / 'response.txt').is_file()]
    if len(available) != 1:
        return None, False, {'representation': 'unavailable', 'exact_http_candidates': len(candidates),
                            'exact_recorded_responses': len(available), 'reason': 'No unique recorded response linked by planner_request_id'}
    root = available[0]
    audit = {'representation': 'unaccepted_provider_candidate', 'source': source_ref(root / 'response.txt'),
             'request_source': source_ref(root / 'request.json'), 'exact_http_candidates': len(candidates)}
    try:
        content = read(root / 'response.txt')['choices'][0]['message']['content']
        output = json.loads(content)
        if not isinstance(output, dict):
            raise ValueError('Provider candidate is not an object')
        return output, False, audit
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return None, False, {**audit, 'representation': 'invalid_provider_candidate', 'parse_error': str(exc)}
