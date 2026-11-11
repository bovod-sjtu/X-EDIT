"""Minimal ordinal judgments over frozen, independently observed audio evidence.

Observation text is fallible evidence for the Guider, not an inferred repair.
The three decisions share the same evidence and attachments. Only option order
changes. All votes, failures and self-reported confidence remain inspectable.
"""
from pathlib import Path

from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint
from x_editor.evaluation.iterative_runtime import sha256, write_record
from x_editor.evaluation.ordinal_ensemble import aggregate, decode, presentations

OBSERVE = """你只看到一份音频，不知道编辑任务、期望答案或评分。
描述实际可辨认的词句、说话人数、声音属性、情绪、速度、背景和显著瞬态。
听不清直接说明，不猜目标，不补全。位置使用附件局部0秒；裁片首尾不是编辑损伤。
返回JSON {"heard":"具体观察，最多250字","uncertain":"不确定之处"}。"""

JUDGE = """只依据实际附件、独立观察和适用范围内的测量回答当前一个问题。
问题中的期望不是已经完成的证据。其他维度的成功或失败不能改变本题得分。
独立观察可能有错；与音频核对。裁片从局部0秒起，人工裁片边缘不算编辑损伤。
测量只适用于声明的区域和属性，不能从波形不同推出语义失败。
材料不足返回U。置信度为你估计本题高/低两档判断可靠的程度，不是质量分。
只返回JSON {"choice":"显示的选项字母或U","confidence":0到1或null}。
不要解释、转写或生成修复计划。"""

LEVELS = {"0": "本题目标未实现或效果相反。", "1": "部分实现，但有实质缺失或损伤。",
          "2": "本题必要条件满足，仅有轻微缺陷。", "3": "本题全部满足，未发现相关缺陷。"}
PROTOCOL = "independent-evidence-three-permutations/v1"


def observe_audio(inventory, paths, call, output):
    """Observe each actual attachment without showing its role or requested edit."""
    if len(inventory) != len(paths):
        raise ValueError("audio inventory does not match supplied attachments")
    rows = []
    for index, (item, path) in enumerate(zip(inventory, paths)):
        if sha256(path) != item['sha256']:
            raise ValueError("audio changed before independent observation")
        root = Path(output) / str(index)
        try:
            raw = call(OBSERVE, {'duration_sec': item['duration_sec']}, [path], root/'calls')
            if not isinstance(raw, dict) or not isinstance(raw.get('heard'), str) or not raw['heard'].strip():
                raise ValueError("independent observation has no audible description")
            row = {'audio_ref': item['audio_ref'], 'heard': raw['heard'],
                   'uncertain': raw.get('uncertain', ''), 'status': 'observed'}
        except Exception as exc:
            row = {'audio_ref': item['audio_ref'], 'status': 'failed', 'error': str(exc)}
        write_record(root/'observation.json', row)
        rows.append(row)
    return rows


def judge_evidence(question, inventory, paths, observations, call, output, *,
                   signal_facts=None, public_turns=None, levels=None):
    """Do exactly three decisions; unknown votes are never converted to passes."""
    visible_fields = {'audio_ref', 'role', 'attachment_number', 'duration_sec',
                      'parent_start_sec', 'parent_end_sec', 'boundary_local_sec',
                      'is_parent_start', 'is_parent_end'}
    if len(inventory) != len(paths) or any(sha256(p) != i['sha256'] for i, p in zip(inventory, paths)):
        raise ValueError("audio changed before ordinal judgments")
    base = {'question': question, 'audio_inventory': [
        {k: v for k, v in i.items() if k in visible_fields} for i in inventory],
        'independent_observations': observations, 'signal_facts': signal_facts}
    if public_turns is not None:
        base['public_turns'] = public_turns
    fingerprint = canonical_evaluation_fingerprint(base)
    root = Path(output)
    write_record(root/'shared_evidence.json', base)
    votes = []
    for index, (mapping, options) in enumerate(presentations(levels or LEVELS, fingerprint)):
        destination = root/f'vote_{index}'
        try:
            raw = call(JUDGE, {**base, 'options': options, 'U': 'Evidence insufficient.'}, paths, destination/'calls')
            write_record(destination/'raw.json', raw)
            vote = decode(raw, mapping)
        except Exception as exc:
            vote = {'grade': None, 'confidence': None, 'status': 'failed', 'error': str(exc)}
        write_record(destination/'result.json', {**vote, 'mapping': mapping, 'evidence_fingerprint': fingerprint})
        votes.append(vote)
    result = {**aggregate(votes), 'protocol': PROTOCOL, 'evidence_fingerprint': fingerprint}
    write_record(root/'ensemble.json', result)
    # This is observation, not model-generated responsibility attribution.
    result['finding'] = ('Independent observations (fallible): ' + ' | '.join(
        f"{r['audio_ref']}: {r.get('heard', 'observation unavailable')} {r.get('uncertain', '')}"
        for r in observations))[:1400]
    return result
