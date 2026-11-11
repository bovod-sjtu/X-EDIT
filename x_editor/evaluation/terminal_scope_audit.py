"""Check whether an acoustic finding actually answers its own criterion.

This semantic audit cannot hear audio or award a better grade. It can only retain
a judgment or withhold it for a targeted recheck. Raw acoustic answers persist.
"""

from typing import Literal

from pydantic import Field

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel


class CriterionScope(FrozenEvaluationModel):
    criterion_id: str
    status: Literal["scope_supported", "needs_recheck"]
    reason: str = Field(min_length=4, max_length=1000)
    recheck_question: str | None = Field(default=None, max_length=1000)


class TerminalScopeAudit(FrozenEvaluationModel):
    criteria: tuple[CriterionScope, ...] = Field(min_length=1, max_length=24)


SCOPE_AUDIT_PROMPT = """只检查每道听音题的推断依据是否属于该题，不能替代听音或重新打分。
你没有音频，不能确认转写、发音、音色、情绪或自然度是否真实。支持的含义仅是该推断
没有已知的逻辑范围错误，不是声学事实被认证。原始分数被隐藏，不按高低分决定支持。
逐题核对 criterion.question、required_final_condition、observed_condition：本题是否被
替换成整项编辑要求，是否把用户授权变化当保留损伤，或从一种属性变化断言另一种属性
失败。声音身份并不要求语速、音高、情绪与原音频逐项完全相同；但实际身份改变仍可
是真实损伤。转写不同也不自动等同发音不自然，描述了具体声学瑕疵则另论。
不同题目谈不同属性不矛盾；同一音频同一内容被互斥描述时才需核查。源音频不能作为
候选违反要求的唯一声音证据；公开指令说明应有目标，不证明声音已实现。
如有这些问题，status=needs_recheck，说明具体理由并给一句只核查该属性的听音问题。
其他情况 scope_supported，不能因为你听不到音频就一律要求重查。不得给新分数、
改写音频内容或免除原始用户要求。每个被审查criterion_id恰好回答一次。输出schema。
"""


def scope_audit_payload(criteria, answers, public_turns, inventory):
    return {
        "schema": TerminalScopeAudit.model_json_schema(),
        "public_turns": public_turns,
        "audio_inventory": inventory,
        "criteria": [c.model_dump(mode="json") for c in criteria],
        "findings": {key: {k: v for k, v in answer.items() if k not in {"grade", "verdict"}}
                     for key, answer in answers.items()},
    }


def apply_scope_audit(payload, answers):
    audit = TerminalScopeAudit.model_validate(payload)
    ids = [c.criterion_id for c in audit.criteria]
    if len(ids) != len(set(ids)) or set(ids) != {a.question_id for a in answers}:
        raise ValueError("scope audit must cover each answered criterion exactly once")
    by_id = {c.criterion_id: c for c in audit.criteria}
    resolved, errors = [], []
    for answer in answers:
        check = by_id[answer.question_id]
        if check.status == "needs_recheck":
            if not check.recheck_question:
                raise ValueError("a withheld scope requires an actionable recheck question")
            message = f"{answer.question_id}: scope needs recheck: {check.reason}; {check.recheck_question}"
            errors.append(message)
            answer = answer.model_copy(update={"grade": None, "finding": answer.finding + "\n" + message})
        resolved.append(answer)
    return resolved, errors
