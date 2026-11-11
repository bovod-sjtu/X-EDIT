"""Outcome-led diagnostic scope, not a replacement plan or a module grade."""

from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel


def compact_focus_positions(positions):
    """Factor byte-identical evidence shared by modules and turns, without loss.

    A position may still cite only its own IDs. No observations, public turns,
    contract values, or failure records are dropped or summarized by a model.
    """
    catalog, compact = {}, []
    for position in positions:
        for identity, entry in position["evidence"].items():
            if identity in catalog and catalog[identity] != entry:
                raise ValueError("shared diagnostic evidence identity collision")
            catalog[identity] = entry
        compact.append({**{k: v for k, v in position.items() if k != "evidence"},
                        "evidence_ids": list(position["evidence"])})
    return {"positions": compact, "shared_evidence_catalog": catalog}


class ModuleReviewFocus(FrozenEvaluationModel):
    turn_index: int = Field(ge=0, strict=True)
    module: Literal["reasoner", "planner"]
    question_ids: tuple[str, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=1600)


class OutcomeRequirementCheck(FrozenEvaluationModel):
    criterion_id: str = Field(min_length=1)
    status: Literal["satisfied", "violated", "unknown"]
    finding: str = Field(min_length=1, max_length=1200)


class OutcomeReviewFocus(FrozenEvaluationModel):
    disposition: Literal["review_modules", "evaluator_uncertain", "no_supported_issue", "supported_error_unassigned"]
    requirement_checks: tuple[OutcomeRequirementCheck, ...] = ()
    outcome_findings: tuple[str, ...] = Field(default=(), max_length=12)
    evaluator_disagreements: tuple[str, ...] = Field(default=(), max_length=12)
    positions: tuple[ModuleReviewFocus, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def distinct_positions(self):
        keys = [(p.turn_index, p.module) for p in self.positions]
        if len(keys) != len(set(keys)):
            raise ValueError("review focus repeats a module position")
        if bool(self.positions) != (self.disposition == "review_modules"):
            raise ValueError("only module review may select diagnostic positions")
        if any(c.status == "violated" for c in self.requirement_checks) and self.disposition not in {
            "review_modules", "supported_error_unassigned"
        }:
            raise ValueError("a confirmed requirement violation cannot be classified as no supported error")
        return self


FOCUS_PROMPT = """你负责从最终交付倒查本次尝试需要诊断的原因，不评完整模块成绩，不生成新计划。
先看实际是否交付以及最终音频逐项检验，再对照真实模块输出、前置检查和工具返回。
逐项核对最终仍有效的目标、修改对象/范围和保留要求，不能用合法参数或选对版本替代
这些条件。对utility_design每条criterion填写唯一的requirement_checks，标明满足、违反或
未知及实际依据；这是对终态的检查，不复制听音等级，也不重做所有模块审计。
检查测量事实与听音回答是否冲突：波形测量只说明实际变化，不给出语义答案。
整段的精确线性变化作用于整段混合信号，不证明只修改了其中某个成分；是否允许这种
范围仍取决于公开指令和已有内容证据。非零拟合残差、不同长度本身均不证明编辑错误。
检验题在候选生成前冻结；回答可能错听、互相矛盾或缺项。成绩高不证明没有错误，
成绩低也不能单独证明上游推理错。发现音频评委分歧，明确列出待核实的具体声称。
只因评委格式无效或听不清，且没有其他错误证据时，返回evaluator_uncertain；不要
强迫执行端为评委错误修改正确产物。未交付时依据真实阻断原因追查，不能说音频不好。

选需要深入问的最少模块位置和问题ID，必须覆盖当前已支持的错误与必要的保留条件。
若原因在上游，只选相关上游问题；下游因上游失败没有运行，不需要另做完整审计。
若Reasoner已正确推断目标而Planner参数不符，保留Reasoner，检查Planner具体字段。
若目标事件/词的证据不足而直接拒绝，检查Reasoner是否应该请求针对性感知；不要
按原句关键词路由。若实际工具产物错而计划意图对，检查参数表示和公开工具合同，
不能要求执行前的Reasoner验证尚未生成的音频。存在多个独立原因时可选多个位置。
只选所给的真实位置、问题ID、同位置证据ID；不猜未执行模块或引用另一个位置的ID。
positions 中的 evidence_ids 指向 shared_evidence_catalog 中的完整原值，同一事实仅存一份。
每个位置只能引用自己的 evidence_ids；共享存储不改变事实所属的模块和当时版本。
每个位置reason说明具体症状、可能原因及为何要问这些题，不写可直接执行的参数。
问题目录完整保留，未选问题是延后审计，绝不视为已通过。没有支持的缺陷可返回
no_supported_issue，这仅表示当前诊断没发现问题，不认证质量或模块上限。
此时outcome_findings也需简述修改范围及保留要求的证据；未能判断的条件明确写为未知。
多轮按最终仍有效的要求检查产物；早期模块只能依据当时公开指令承担职责，不能把
后续指令倒灌为早期错误。正确选取历史版本，只证明版本选择正确，不免责该版本中
已经存在且违反当时要求的编辑错误；沿最终产物的真实依赖回溯首次引入错误的位置。
不能以“先前已经这样执行”或“工具固有行为”把已确认的目标/保留损失视为满足。
若确认产物有错但没有足够依据定位可引导模块，返回supported_error_unassigned并说明
缺什么定位证据；不能认证无错，也不能臆造修复方式。保持简洁，只返回给定JSON schema。
"""


def validate_focus(payload, positions, criterion_ids=None):
    focus = OutcomeReviewFocus.model_validate(payload)
    if criterion_ids is not None:
        observed = [c.criterion_id for c in focus.requirement_checks]
        if len(set(observed)) != len(observed) or set(observed) != set(criterion_ids):
            raise ValueError("focus must account for each terminal criterion exactly once")
    known = {(p["turn_index"], p["module"]): p for p in positions}
    for position in focus.positions:
        context = known.get((position.turn_index, position.module))
        if context is None:
            raise ValueError("focus selected an unobserved module or turn")
        questions = {q["question_id"] for q in context["questions"]}
        if len(set(position.question_ids)) != len(position.question_ids):
            raise ValueError("focus repeats a question")
        if set(position.question_ids) - questions:
            raise ValueError("focus selected an unknown module question")
        if set(position.evidence_ids) - set(context["evidence"]):
            raise ValueError("focus cited evidence from outside its module position")
    return focus
