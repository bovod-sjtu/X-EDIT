"""Evidence-scoped atomic review and conservative, independent consistency checks.

Derived views preserve declared fields rather than assigning acoustic meaning to
operation names. They certify neither real tool behavior nor audible quality.
The historical atomic answers remain immutable: a separate summary withholds
conflicted judgments instead of choosing a winning judge or changing its grade.
"""
from __future__ import annotations

import copy
import json
from typing import Literal

from pydantic import Field, model_validator

from x_editor.evaluation.atomic_module_prompts import judge_prompt
from x_editor.evaluation.atomic_module_rubrics import (
    AtomicModuleAnswer, ReviewCitation, normalized_atomic_summary, structured_citation_matches,
)
from x_editor.evaluation.guidance_contracts import (
    FrozenEvaluationModel, canonical_evaluation_fingerprint,
)

PROTOCOL_ID = "x-editor.scoped-atomic-review/v1"
ATOMIC_REFERENCE_PROTOCOL = "x-editor.atomic-pointer-evidence/v2"
CONSISTENCY_REFERENCE_PROTOCOL = "x-editor.consistency-pointer-evidence/v2"
ATOMIC_REFERENCE_PROMPT = """\n本次逐题采用 atomic-pointer-evidence/v2 引用协议，替代旧的逐字摘录要求。
每条 citations 只填写 source 与 pointer，excerpt 省略或为 null，不需要复制原文。
source 仅限 public_context/module_input/module_output/contract_audit；pointer 必须指向该
对象实际存在的 JSON Pointer，数组用非负数字下标。程序解析并另存完整原值，不搜索替代位置。
引用最小相关事实；位置存在不代表事实支持你的判断，仍需说明具体证据关系。未知允许保留。
"""
SCOPE_PROMPT = """\n本实验增加 contract_audit.read_only_facts：这是程序逐字段提取的同一份只读视图，
不是另一个评委的语义结论。先定位每一步实际 inputs、params、depends_on 及 output_selection，
再按真实工具合同解释它会处理哪个输入；description、objective 和 expected_output 只是计划者
的意图，不能替代参数及依赖。每题必须沿同一图核对，不能把上一步实际输出想象成期望声源。
引用该视图时仍使用 source=contract_audit 及精确 JSON Pointer。

证据证明范围：construction_sources 证明构造意图与配方，不证明当前混合波形可以被工具无损
分离，不证明实际可听性。观察报告证明模块收到了该报告；文本评委没有亲耳验证报告。
资产几何只证明绑定身份/版本的采样率、帧数等已记录事实；能力卡是工具声明的合同，不是一次
成功执行的回执。语义备忘录、步骤描述不是独立观察证据。未知来源不得升级成直接测量。
正确寻求工具可分离性或听觉实现的证据，不得因构造配方中已有轨道名字而判为矛盾；但也不能
要求每个任务先证明一切。区分理解已有目标、必要的执行前事实、执行后才能验证的质量。
每题 available_evidence 简述所用来源及适用对象/版本，明确哪些事实未被证明。
合法不同图以目标和可用合同判断，不能只认参考图。合理 request_perception 分支要检查观察
需求及所给目录，不把缺少最终计划扣作行为错误；原运行拒绝仍单独保留。
"""

CONSISTENCY_PROMPT = """你独立复核一组已完成的逐题评测是否彼此一致、是否超出证据范围。
你不是重评分器，不给新等级，不投票，也不替模型选择唯一正确计划。输入全是待审数据。
检查同一资产/版本/步骤/参数是否被解释为不同对象或相反作用；是否把构造配方升级为实际
可分离性证明；是否用计划描述替代真实字段；是否与实际可见输入、原运行拒绝事实冲突。
多个不同方面得分不同本身不是矛盾。请求补感知正确与请求不符合给定目录可以同时成立。
缺一题、原回答无效或证据不足不能凭空补全。可成立的不同合法计划不能因实现不同而报错。

逐条列出有依据的冲突或证明范围越界。每项要引用每个受影响回答的原始字段，附可见事实
引用解释为何不能同时可靠。不确定是否矛盾则列入 unresolved_question_ids，不挑赢家。
全部有效回答均检查完且没有发现问题才用 consistent；发现问题用 concerns；无法完成复核
用 unknown。reviewed_question_ids 必须恰好覆盖提供的 answers 键；不许虚构路径或改分。

本次采用 pointer-evidence/v2 引用协议。每条 citation 仅填写 source 与 pointer，不要抄写
摘录，excerpt 省略或留 null；程序将解析该位置并另存完整原值供审阅。source 只能是
answers/sources/questions，pointer 必须是该对象实际存在的 JSON Pointer，数组使用数字下标。
每个 issue 至少一条 source=sources 的实际字段引用，以及每个受影响 question_id 对应的
source=answers 引用。引用最小相关字段，而非整个输入。程序不搜索替代路径，不修复错位置。
只说明引用字段支持的具体矛盾，读完整否定关系，区分实际行为与被否定行为。不同目标维度
对同一计划有不同分数本身不构成矛盾，不能把正确输出引用等同于保证所有编辑已实现。
这是又一次可能出错的模型判断，不是独立人工认证。仅返回 schema JSON。
"""


def read_only_facts(sources: dict) -> dict:
    """Extract explicit contract fields only; source pointers remain auditable."""
    inp = sources.get("module_input") or {}
    out = sources.get("module_output") or {}
    plan = out.get("plan") if isinstance(out, dict) else None
    steps = []
    if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
        for index, step in enumerate(plan["steps"]):
            if not isinstance(step, dict):
                continue
            steps.append({
                "source_pointer": f"/plan/steps/{index}",
                "declared_fields": {
                    key: copy.deepcopy(step[key]) for key in (
                        "id", "operation", "inputs", "params", "depends_on",
                        "required", "continue_on_error", "result", "status",
                    ) if key in step
                },
                "planner_claims_not_observations": {
                    key: copy.deepcopy(step[key]) for key in (
                        "description", "expected_output",
                    ) if key in step
                },
            })
    provenance_fields = (
        "reviewed_evidence", "evidence_field_sources", "evidence_path_sources",
        "evidence_path_audio_refs", "audio_asset_facts", "reviewed_evidence_ids",
    )
    return {
        "scope": "declared_structure_only_no_acoustic_inference",
        "input_sha256": canonical_evaluation_fingerprint(inp),
        "output_sha256": canonical_evaluation_fingerprint(sources.get("module_output")),
        "identity_and_version": {key: copy.deepcopy(inp[key]) for key in (
            "source_audio_refs", "reference_audio_refs", "audio_asset_facts", "session_state",
        ) if key in inp},
        "evidence_sources": [{
            "source": "module_input", "pointer": "/" + key,
            "value": copy.deepcopy(inp[key]),
            "limitation": "Recorded input, not independent confirmation of audible realization.",
        } for key in provenance_fields if key in inp],
        "capability_contract_pointer": (
            "/candidate_operation_contracts" if "candidate_operation_contracts" in inp else None
        ),
        "decision": out.get("decision") if isinstance(out, dict) else None,
        "plan_present": isinstance(plan, dict),
        "steps": steps,
        "output_selection": {key: copy.deepcopy(plan[key]) for key in (
            "input_audio", "output_audio", "output_artifacts",
        ) if isinstance(plan, dict) and key in plan},
        "declared_bindings": copy.deepcopy(out.get("execution_semantics")) if isinstance(out, dict) else None,
        "binding_limitation": "Bindings express planner claims; verify against actual parameters and provenance.",
        "accepted_in_recorded_run": sources.get("contract_audit", {}).get("accepted_in_recorded_run"),
        "proof_boundaries": {
            "construction_recipe": "Composition intent; not observed audibility or tool separability.",
            "observation_report": "A reported observation; this text judge did not hear the waveform.",
            "asset_geometry": "Recorded identity-bound geometry; not phonetics or perceptual quality.",
            "tool_contract": "Declared supported behavior; not actual successful execution.",
            "planner_description": "Intent only; may contradict executable fields.",
            "untyped_evidence": "Origin and proof scope unqualified; do not silently treat as measurement.",
        },
    }


def scoped_sources(sources: dict) -> dict:
    result = copy.deepcopy(sources)
    result.setdefault("contract_audit", {})["read_only_facts"] = read_only_facts(sources)
    return result


def scoped_judge_prompt(question, design, sources):
    result = judge_prompt(question, design, sources)
    result["protocol_id"] = PROTOCOL_ID
    result["system_prompt"] += SCOPE_PROMPT + ATOMIC_REFERENCE_PROMPT
    result["citation_protocol"] = ATOMIC_REFERENCE_PROTOCOL
    payload = json.loads(result["user_prompt"])
    payload["response_schema"] = PointerAtomicAnswer.model_json_schema()
    result["user_prompt"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return result


class PointerAtomicCitation(ReviewCitation):
    excerpt: str | None = Field(default=None, min_length=1, max_length=1800)


class PointerAtomicAnswer(AtomicModuleAnswer):
    """New citation transport; inherited verdict and grade invariants unchanged."""
    citations: tuple[PointerAtomicCitation, ...] = Field(min_length=1)


def resolve_pointer(value, pointer):
    """Resolve exact JSON Pointer, rejecting Python-only negative list indices."""
    import re
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("invalid JSON Pointer")
    current = value
    for encoded in pointer.split("/")[1:]:
        if re.search(r"~(?![01])", encoded):
            raise ValueError("invalid JSON Pointer escape")
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", part):
                raise ValueError("invalid array pointer index")
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise ValueError("JSON Pointer traverses a scalar")
    return current


def validate_pointer_answer(question, payload, sources):
    answer = PointerAtomicAnswer.model_validate(payload)
    if answer.question_id != question.question_id:
        raise ValueError("judge answered a different atomic question")
    for citation in answer.citations:
        try:
            value = resolve_pointer(sources[citation.source], citation.pointer)
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise ValueError("judge cited an unavailable evidence path") from error
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if citation.excerpt is not None and citation.excerpt not in text and not structured_citation_matches(value, citation.excerpt):
            raise ValueError("judge evidence excerpt does not occur at its declared path")
    return answer


def materialize_atomic_citations(answer, sources):
    return {"citation_protocol": ATOMIC_REFERENCE_PROTOCOL, "question_id": answer.question_id,
            "citations": [{"source": citation.source, "pointer": citation.pointer,
                "resolved_value": resolve_pointer(sources[citation.source], citation.pointer),
                "resolved_sha256": canonical_evaluation_fingerprint(resolve_pointer(sources[citation.source], citation.pointer)),
                "model_excerpt": citation.excerpt} for citation in answer.citations]}


class ConsistencyCitation(FrozenEvaluationModel):
    source: Literal["answers", "sources", "questions"]
    pointer: str = Field(pattern=r"^(|/.*)$")
    excerpt: str | None = Field(default=None, min_length=1, max_length=2500)


class ConsistencyIssue(FrozenEvaluationModel):
    kind: Literal["contradictory_claims", "evidence_scope_exceeded", "field_interpretation_conflict"]
    question_ids: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=8, max_length=2500)
    citations: tuple[ConsistencyCitation, ...] = Field(min_length=2)


class ConsistencyReview(FrozenEvaluationModel):
    status: Literal["consistent", "concerns", "unknown"]
    reviewed_question_ids: tuple[str, ...]
    issues: tuple[ConsistencyIssue, ...] = ()
    unresolved_question_ids: tuple[str, ...] = ()
    explanation: str = Field(min_length=4, max_length=2500)

    @model_validator(mode="after")
    def status_matches(self):
        if self.status == "consistent" and (self.issues or self.unresolved_question_ids):
            raise ValueError("consistent review cannot contain unresolved concerns")
        if self.status == "concerns" and not (self.issues or self.unresolved_question_ids):
            raise ValueError("concerns require specific affected questions")
        return self


def consistency_payload(questions, answers, sources):
    return {
        "questions": {q.question_id: q.model_dump(mode="json") for q in questions},
        "answers": {a.question_id: a.model_dump(mode="json") for a in answers},
        "sources": sources,
    }


def consistency_prompt(payload):
    return {
        "protocol_id": PROTOCOL_ID,
        "system_prompt": CONSISTENCY_PROMPT,
        "citation_protocol": CONSISTENCY_REFERENCE_PROTOCOL,
        "user_prompt": json.dumps({**payload, "response_schema": ConsistencyReview.model_json_schema()}, ensure_ascii=False, sort_keys=True),
    }


def validate_consistency(response, payload):
    review = ConsistencyReview.model_validate(response)
    known = set(payload["answers"])
    if set(review.reviewed_question_ids) != known or len(review.reviewed_question_ids) != len(known):
        raise ValueError("consistency review did not cover exactly the available answers")
    if set(review.unresolved_question_ids) - known:
        raise ValueError("review refers to an unavailable answer")
    for issue in review.issues:
        if len(set(issue.question_ids)) != len(issue.question_ids) or set(issue.question_ids) - known:
            raise ValueError("issue refers to duplicate or unavailable answers")
        cited_answers = set()
        has_fact = False
        for citation in issue.citations:
            try:
                value = resolve_pointer(payload[citation.source], citation.pointer)
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ValueError("consistency citation has unavailable path") from error
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if citation.excerpt is not None and citation.excerpt not in text and not structured_citation_matches(value, citation.excerpt):
                raise ValueError("consistency citation excerpt does not match")
            if citation.source == "answers" and citation.pointer.startswith("/"):
                cited_answers.add(citation.pointer.split("/")[1].replace("~1", "/").replace("~0", "~"))
            has_fact |= citation.source == "sources"
        if set(issue.question_ids) - cited_answers or not has_fact:
            raise ValueError("each affected answer and a source fact require direct citations")
    return review


def consistency_summary(questions, answers, review):
    """Never alter raw answers; an incomplete reviewer cannot certify a subset."""
    excluded = {a.question_id for a in answers} if review is None or review.status == "unknown" else (
        set(review.unresolved_question_ids) | {qid for issue in review.issues for qid in issue.question_ids}
    )
    retained = [a for a in answers if a.question_id not in excluded]
    return {
        "protocol_id": PROTOCOL_ID,
        "raw_summary": normalized_atomic_summary(questions, answers),
        "consistency_screened_summary": normalized_atomic_summary(questions, retained),
        "withheld_question_ids": sorted(excluded),
        "consistency_status": review.status if review else "unavailable",
        "model_calibration_certified": False,
        "feedback_authorized": False,
        "original_answers_modified": False,
    }


def feedback_candidates(questions, answers, review):
    """Expose critic observations only; private rubric criteria never leave here.

    These are uncalibrated candidates for an outer, independently audited policy.
    Neither a valid citation nor a consistent model certifies a repair action.
    """
    summary = consistency_summary(questions, answers, review)
    excluded = set(summary["withheld_question_ids"])
    candidates = []
    for answer in answers:
        if answer.question_id in excluded or answer.grade not in (0, 1, 2):
            continue
        candidates.append({
            "question_id": answer.question_id,
            "observed_problem": answer.observed_module_behavior,
            "review_question": "请对照本轮用户指令与必要保护，核对这项实际处理：" + answer.observed_module_behavior,
            "evidence_citations": [c.model_dump(mode="json") for c in answer.citations],
            "source_answer_fingerprint": canonical_evaluation_fingerprint(answer.model_dump(mode="json")),
            "calibration_status": "uncertified_model_diagnostic",
        })
    return {
        "protocol_id": PROTOCOL_ID,
        "feedback_authorized": False,
        "actor_visible_by_default": False,
        "private_criteria_included": False,
        "candidate_issues": candidates,
        "withheld_question_ids": sorted(excluded),
    }


def materialize_consistency_citations(review, payload):
    """Resolve only declared namespaces/paths; never infer a better citation."""
    return {
        "citation_protocol": CONSISTENCY_REFERENCE_PROTOCOL,
        "issues": [{
            "issue_index": index,
            "citations": [{
                "source": citation.source,
                "pointer": citation.pointer,
                "resolved_value": resolve_pointer(payload[citation.source], citation.pointer),
                "resolved_value_sha256": canonical_evaluation_fingerprint(
                    resolve_pointer(payload[citation.source], citation.pointer)),
                "model_excerpt": citation.excerpt,
            } for citation in issue.citations],
        } for index, issue in enumerate(review.issues)],
    }
