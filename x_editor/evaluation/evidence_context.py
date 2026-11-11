"""Version-bound observation context for an explicit, approximate POMDP policy.

These are fallible observations and model hypotheses, not a calibrated Bayesian
posterior. Raw-history and explicit-state experiments receive identical facts.
Only the representation requested from the diagnostic model differs.
"""

import copy
from typing import Literal

from pydantic import Field

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel, canonical_evaluation_fingerprint
from x_editor.evaluation.refinement_focus import OutcomeReviewFocus, validate_focus


class RequirementEvidenceState(FrozenEvaluationModel):
    criterion_id: str
    state: Literal["supported_satisfied", "supported_violation", "unknown", "conflicted"]
    observation_ids: tuple[str, ...] = ()
    excluded_stale_observation_ids: tuple[str, ...] = ()
    unresolved_question: str = Field(max_length=1400)
    explanation: str = Field(min_length=1, max_length=1800)


class EvidenceContextDecision(FrozenEvaluationModel):
    requirement_states: tuple[RequirementEvidenceState, ...]
    decision: OutcomeReviewFocus


CONTEXT_PROMPT = """本次使用显式证据状态。观察不是事实真值：语义审查和听音均可能出错。
逐项整理仍有效的要求，区分已支持满足、已支持违反、未知和观察冲突，再决定是否需要诊断模块。
为每条要求引用 observation_catalog 的原始观察ID；引用存在仅证明有记录，不保证内容正确。
观察的 value_ref 是 JSON 指针，指向当前输入中的完整原值；共享存储避免重复粘贴，不删除观察。
两个引用空间必须区分：requirement_states.observation_ids 引用 observation_catalog；
decision.positions 的 evidence_ids 只能选原始 positions 为该模块列出的 evidence_ids。
不能把 observation: 开头的上下文观察ID塞入模块的 evidence_ids，也不能跨模块借用ID。
过往终态听评绑定其音频哈希。不同版本的旧听评不能证明当前声音仍有同一缺陷；可以作为待核实假说。
同一音频的新旧观察冲突不能靠“最新优先”或多数表决消解，应指出具体待查问题。
没有证据不是已证明不存在；计划合法不是音频目标完成；撤销成功不免责仍有效要求的损伤。
冲突或未知不能独自授权编辑。若另有已支持且可定位的独立错误，可以针对该错误生成模块诊断；
否则保持 evaluator_uncertain，先补观察或保留当前产物。不要为了凑重试次数制造修复。
每项必须填写 unresolved_question：unknown/conflicted 时写出下一步具体需要核查的问题；
已有支持且没有未决问题时填写空字符串。不能只说证据不足，却不指出需要什么观察。
模型给出的状态是用于决定下一步的可审计假说，不是概率、容量上限或真实质量认证。
输出 requirement_states 与 decision，两者对相同 criterion 的结论必须一致。
"""


def observation_context(payload, previous_packets):
    """Return lossless current/past observations with executable identity checks."""
    current = {key: copy.deepcopy(value) for key, value in payload.items() if key not in {"schema", "evidence_context"}}
    identity = canonical_evaluation_fingerprint(current)
    packets = [*previous_packets, current]
    catalog, past_values = {}, {}
    latest_audio = current.get("terminal_audio_review") or {}
    current_hash = latest_audio.get("artifact_sha256")
    for packet in packets:
        packet_hash = canonical_evaluation_fingerprint(packet)
        for field in ("public_turns", "actual_turns", "terminal_audio_review", "terminal_coordinate_measurements", "shared_evidence_catalog"):
            value = packet.get(field)
            if value is None:
                continue
            old_hash = (packet.get("terminal_audio_review") or {}).get("artifact_sha256")
            # Only terminal measurements have automatic version expiry. Module
            # outputs remain historical records, not proof of present correctness.
            stale = field in {"terminal_audio_review", "terminal_coordinate_measurements"} and bool(old_hash and old_hash != current_hash)
            key = f"observation:{packet_hash[:20]}:{field}"
            value_ref = "/" + field
            if value != current.get(field):
                digest = canonical_evaluation_fingerprint(value)
                past_values[digest] = value
                value_ref = "/evidence_context/previous_observation_values/" + digest
            catalog[key] = {"kind": field, "packet_id": packet_hash, "artifact_sha256": old_hash,
                            "current_packet": packet_hash == identity,
                            "stale_for_current_artifact": stale, "value_ref": value_ref}
    return {"current_packet_id": identity, "observation_catalog": catalog,
            "previous_observation_values": past_values}, current


def resolve_observation(payload, context, observation_id):
    """Resolve the exact value shown to the model; no semantic summarization."""
    parts = context["observation_catalog"][observation_id]["value_ref"].lstrip("/").split("/")
    value = {**payload, "evidence_context": context}
    for part in parts:
        value = value[part]
    return value


def validate_context_decision(raw, context, positions, criterion_ids):
    value = EvidenceContextDecision.model_validate(raw)
    states = {state.criterion_id: state for state in value.requirement_states}
    if len(states) != len(value.requirement_states) or set(states) != set(criterion_ids):
        raise ValueError("evidence state must cover every terminal requirement exactly once")
    catalog = context["observation_catalog"]
    expected = {"supported_satisfied": "satisfied", "supported_violation": "violated", "unknown": "unknown", "conflicted": "unknown"}
    focus = validate_focus(value.decision.model_dump(mode="json"), positions, criterion_ids)
    if focus.disposition == "no_supported_issue" and any(s.state in {"unknown", "conflicted"} for s in states.values()):
        raise ValueError("unresolved terminal evidence requires uncertainty handling, not an all-clear disposition")
    for check in focus.requirement_checks:
        state = states[check.criterion_id]
        refs = (*state.observation_ids, *state.excluded_stale_observation_ids)
        if set(refs) - set(catalog):
            raise ValueError("evidence state cites an unobserved record")
        if state.state.startswith("supported_") and not state.observation_ids:
            raise ValueError("supported hypothesis requires cited observations")
        if any(catalog[key]["stale_for_current_artifact"] for key in state.observation_ids):
            raise ValueError("stale audio observation cannot support a current terminal conclusion")
        if any(not catalog[key]["stale_for_current_artifact"] for key in state.excluded_stale_observation_ids):
            raise ValueError("cannot discard a current observation as stale")
        if state.state in {"unknown", "conflicted"} and not state.unresolved_question.strip():
            raise ValueError("uncertainty must identify a decision-relevant missing observation")
        if check.status != expected[state.state]:
            raise ValueError("diagnostic action contradicts its evidence state")
    return focus, value


GUIDANCE_CONTEXT_PROMPT = """current_evidence_state 是本次已校验的诊断状态及其原始观察引用。
状态是可错的假说，不是新观察，也不是编辑已完成的证明。保留已支持的结论，
根据仍未知、冲突或已支持违反的要求决定补查或修复。一个检查问题已得到答案，
不意味着它发现的声音缺陷已修好。原始观察与本轮新增观察优先用于更新判断；
不要复活已被具体证据取代的旧批评，也不能仅凭旧版本观察判定当前产物错误。
指导送到实际能改变该问题的模块：推理/目标表达由 Reasoner 重新推导，
合法参数与调用表示由 Planner 编译；执行前模块不承担核听尚未生成的产物。
"""


def guidance_evidence_context(state, focus_payload):
    """Carry validated hypotheses and their exact observations into the Guider.

    Raw focus packets stay in the audit. Only cited records are repeated here;
    model hypotheses never enter the observation namespace.
    """
    context = focus_payload["evidence_context"]
    ids = {identity for item in state.requirement_states
           for identity in (*item.observation_ids, *item.excluded_stale_observation_ids)}
    observations = {
        identity: {**copy.deepcopy(context["observation_catalog"][identity]),
                   "value": copy.deepcopy(resolve_observation(focus_payload, context, identity))}
        for identity in sorted(ids)
    }
    # These pointers belonged to the focus input. Do not expose dangling paths.
    for observation in observations.values():
        observation.pop("value_ref")
    return {"requirement_states": [s.model_dump(mode="json") for s in state.requirement_states],
            "observations": observations, "status": "fallible_hypotheses_not_quality_certification"}
