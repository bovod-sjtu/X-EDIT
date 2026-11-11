"""Lossless transport of repetitive provenance maps to diagnostic reviewers."""
from copy import deepcopy
import json
from x_editor.planner.evidence_index import factor_evidence_index, expand_evidence_index


def factor_diagnostic_indexes(catalog):
    result = deepcopy(catalog)
    for entry in result.values():
        pointer, value = entry.get('pointer'), entry.get('value')
        if pointer not in {'/evidence_path_sources', '/evidence_path_audio_refs'} or not isinstance(value, dict):
            continue
        scalar = pointer == '/evidence_path_audio_refs'
        if scalar and not all(isinstance(v, str) for v in value.values()):
            continue
        if not scalar and not all(isinstance(v, list) and all(isinstance(x,str) for x in v) for v in value.values()):
            continue
        encoded = factor_evidence_index({k:[v] for k,v in value.items()} if scalar else value)
        encoded['value_kind'] = 'single_audio_ref' if scalar else 'evidence_id_list'
        if scalar:
            encoded['decode'] += ' Here each evidence_ids list contains one audio_ref; decode its single element as the original scalar value.'
        decoded = expand_diagnostic_index(encoded)
        if decoded != value:
            raise ValueError('diagnostic index projection changed evidence')
        if len(json.dumps(encoded,ensure_ascii=False)) < len(json.dumps(value,ensure_ascii=False)):
            entry['value'] = encoded
    return result


def expand_diagnostic_index(encoded):
    result = expand_evidence_index(encoded)
    return {k:v[0] for k,v in result.items()} if encoded['value_kind']=='single_audio_ref' else result
