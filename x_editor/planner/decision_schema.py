"""Provider decision branches with the same mandatory products as runtime.

This is generation-time structure, not a repair of model responses. A model can
still request evidence or decline planning; choosing execute requires a program.
Tool semantics, evidence and asset admission remain the runtime's responsibility.
"""
from copy import deepcopy


def bind_decision_payloads(response_format):
    """Bind flat or nested envelopes; already-bound branches are unchanged."""
    result = deepcopy(response_format)

    def visit(node):
        if isinstance(node, dict):
            props = node.get('properties', {})
            if 'plan' in props and 'enum' in props.get('decision', {}):
                bound = decision_payload_schema(node)
                node.clear()
                node.update(bound)
            else:
                for value in node.values():
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result['json_schema']['schema'])
    return result


def decision_payload_schema(envelope):
    """Specialize the common flat wire envelope without selecting a decision."""
    common = {'schema_version', 'decision', 'decision_summary'}
    base = {'selected_base_state_node_id', 'selected_base_audio_ref'}
    fields = {
        'execute': base | {'state_transition_kind', 'plan', 'execution_semantics',
                           'tool_selection', 'missing_information'},
        'request_perception': base | {'perception_jobs', 'candidate_operations',
                                     'missing_information'},
        'cannot_plan': {'missing_information', 'semantic_conflict', 'cannot_plan_reasons'},
        'complete_from_session_state': base | {'state_completion_kind', 'changes_satisfied_by_base'},
    }
    required = {
        'execute': ['decision', 'state_transition_kind', 'plan'],
        'request_perception': ['decision', 'perception_jobs'],
        'cannot_plan': ['decision'],
        'complete_from_session_state': ['decision', *sorted(base), 'state_completion_kind',
                                        'changes_satisfied_by_base'],
    }
    properties = envelope['properties']
    branches = []
    for decision in properties['decision']['enum']:
        branch = {'type': 'object', 'additionalProperties': False,
                  'properties': {key: deepcopy(value) for key, value in properties.items()
                                 if key in common | fields[decision]},
                  'required': required[decision]}
        props = branch['properties']
        props['decision'] = {'type': 'string', 'const': decision}
        if decision == 'execute':
            props['missing_information']['maxItems'] = 0
        elif decision == 'request_perception':
            props['perception_jobs']['minItems'] = 1
        elif decision == 'complete_from_session_state':
            for key in base:
                props[key] = {'type': 'string', 'minLength': 1}
            props['changes_satisfied_by_base']['minItems'] = 1
        else:
            # Complete alternatives also work with constrained-decoding engines
            # that cannot intersect partial object/array schemas in anyOf.
            for key in ('missing_information', 'cannot_plan_reasons', 'semantic_conflict'):
                reason_branch = deepcopy(branch)
                reason_branch['required'].append(key)
                if key == 'semantic_conflict':
                    reason_branch['properties'][key] = {'type': 'string', 'minLength': 1}
                else:
                    reason_branch['properties'][key]['minItems'] = 1
                branches.append(reason_branch)
            continue
        branches.append(branch)
    return {'anyOf': branches}
