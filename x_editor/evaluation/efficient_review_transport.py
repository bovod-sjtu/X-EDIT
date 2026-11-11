"""Lossless transport of diagnostic indexes, outside Actor input assembly.

Every fact, citation ID and tool contract remains available. This encoding can
still affect a model's use of the evidence, so it is an explicit experimental
setting, not a claim that model decisions are invariant under serialization.
"""
from copy import deepcopy
import json

from x_editor.evaluation.diagnostic_index_projection import (
    expand_diagnostic_index, factor_diagnostic_indexes,
)

PROTOCOL = 'factored_diagnostic_indexes/v1'
INSTRUCTION = '''
证据索引可能用prefix-plus-suffix/v1无损表示：将prefix与每个suffix原样连接即为
原始路径，同组路径共享evidence_ids；single_audio_ref组取列表的唯一值作为audio_ref。
这只是去除重复书写；全部事实、合同与引用ID仍在，不能把路径前缀当完整证据。
'''


def encode_payload(payload):
    """Recursively factor only the two declared provenance maps; prove round trip."""
    encodings = []

    def walk(value, path=()):
        if isinstance(value, list):
            return [walk(v, (*path, i)) for i, v in enumerate(value)]
        if not isinstance(value, dict):
            return value
        result = {k: walk(v, (*path, k)) for k, v in value.items()}
        if value.get('pointer') in {'/evidence_path_sources', '/evidence_path_audio_refs'}:
            encoded = factor_diagnostic_indexes({'entry': result})['entry']
            if encoded != result:
                encodings.append([*path, 'value'])
            result = encoded
        return result

    encoded = walk(payload)
    restored = deepcopy(encoded)
    for path in encodings:
        target = restored
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = expand_diagnostic_index(target[path[-1]])
    if restored != payload:
        raise ValueError('diagnostic transport changed source evidence')
    return encoded, {'protocol': PROTOCOL, 'factored_paths': encodings,
                     'round_trip_equal': True, 'semantic_pruning': False}


def encode_prompt(prompt):
    original = json.loads(prompt['user_prompt'])
    payload, receipt = encode_payload(original)
    sent = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    receipt.update(original_chars=len(prompt['user_prompt']), sent_chars=len(sent))
    return {**prompt, 'system_prompt': prompt['system_prompt'] + INSTRUCTION,
            'user_prompt': sent}, receipt
