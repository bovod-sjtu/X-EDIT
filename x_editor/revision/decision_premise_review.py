"""Production review support extracted without changing function bodies."""

import json


from x_editor.revision.atomic_module_review import invoke


from x_editor.revision.refinement_cache import semantic_identity


from x_editor.evaluation.atomic_module_prompts import AUTHOR_PROMPT


from x_editor.evaluation.atomic_module_rubrics import CaseEvidenceDesign


from x_editor.evaluation.atomic_rubric_premises import (
    PREMISE_PROMPT, PremiseReview, resolve_premise_review,
)


from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint


from x_editor.evaluation.iterative_runtime import write_record


PROTOCOL = 'x-editor.evidence-bound-decision-premises/v1'


INPUT_FIELDS = (
    'source_text', 'source_audio_refs', 'reference_audio_refs',
    'audio_asset_facts', 'reviewed_evidence_ids', 'reviewed_evidence',
    'evidence_path_sources', 'evidence_path_audio_refs',
)


PROMPT = PREMISE_PROMPT + """
本次另提供各模块在作出决定前实际收到的观察及其资产/版本绑定。它们是带噪的观察报告，
不是标准答案；不得把报告外的声音属性补成事实。看不到模块输出、计划、备忘录和历史评分。
先结合用户指令与这些源事实确认编辑目标，再核查草稿的问题与成功条件是否成立。
草稿的intended_state只是待验证假设，不能压过源证据。特别区分语音中表达的日期、数量、
事件顺序与波形时间轴操作；不得仅凭一个动词预设操作类型。证据不足则保留条件式检查或
unknown，不能强迫一种解释。用户要求保持内容时，允许合法历史状态恢复；不要额外要求
必须生成一份新工具计划。工具负责的对齐不应无条件变成模块必须自己给出的时间戳。
修改语音中表达的事实，不是要求修改现实事件或物理波形的日历时间。源转写可以提供内容
语义和运算基准；对已表达数值/日期的语言推断不需要额外的外部时钟或所谓声学日期测量。
把目标内容理解清楚和把它定位到采样范围是不同任务，缺词级时间不能否定内容改写的语义。
如果原指令结合源观察足以理解，而草稿解释错了，应revise重建正确的问题，不能仅因草稿
错误就unknown。只有原始任务结合已有证据仍无法辨义时才保留unknown。
只修订题意与必要性，不替Actor生成答案或计划，也不评价尚不可见的Actor输出。
"""


def input_observations(sources, turn):
    """Allowlist only pre-decision observations; exclude memo/plan/retry advice."""
    rows, seen = [], set()
    for (index, _module), source in sorted(sources.items()):
        if index != turn:
            continue
        value = source.get('module_input', {})
        facts = {key: value[key] for key in INPUT_FIELDS if key in value}
        fingerprint = canonical_evaluation_fingerprint(facts)
        if facts and fingerprint not in seen:
            rows.append(facts)
            seen.add(fingerprint)
    return rows


def qualify_design(reviewer, context, design, sources, *, turn, output):
    schema = CaseEvidenceDesign.model_json_schema()
    schema['properties'].pop('input_fingerprint')
    schema['required'].remove('input_fingerprint')
    payload = dict(public_context=context.model_dump(mode='json'),
        observed_inputs=input_observations(sources, turn),
        draft_design=design.model_dump(mode='json'),
        design_rules=AUTHOR_PROMPT, design_schema=schema,
        response_schema=PremiseReview.model_json_schema())
    identity = dict(protocol=PROTOCOL, payload=payload, prompt=PROMPT,
                    provider=semantic_identity(reviewer.args),
                    seed=getattr(reviewer.args, 'seed', 0))

    def review():
        raw = invoke(output/'calls', dict(system_prompt=PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False)), reviewer.args, author=True)
        audit, qualified = resolve_premise_review(context, design, raw)
        return audit, qualified, str(output)

    (audit, qualified, origin), reused, key = reviewer.cache.once(identity, review)
    write_record(output/'qualification.json', dict(protocol=PROTOCOL,
        context_fingerprint=key, reused=reused, origin=origin,
        candidate_output_visible=False, observations=payload['observed_inputs'],
        audit=audit.model_dump(mode='json'),
        qualified_design=qualified.model_dump(mode='json') if qualified else None))
    if qualified is None:
        raise ValueError('semantic rubric premises unresolved; not an Actor failure')
    return qualified
