"""Reconcile past advice against current evidence before issuing new advice.

History remains immutable. This is a fallible, evidence-cited diagnostic state,
not a Bayesian posterior or an authority to change an Actor's facts or plan.
"""

from copy import deepcopy
from typing import Literal

from pydantic import Field

from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel, canonical_evaluation_fingerprint

PROTOCOL = "x-editor.active-guidance-state/v1"


class GuidanceStateUpdate(FrozenEvaluationModel):
    prior_item_ids: tuple[str, ...] = Field(min_length=1, max_length=96)
    disposition: Literal["active", "uncertain", "resolved", "superseded", "inapplicable", "deferred"]
    explanation: str = Field(min_length=1, max_length=1200)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=16,
        description="Current evidence JSON pointers; retiring an issue needs support, not old advice.")
    guidance_refs: tuple[str, ...] = Field(default=(), max_length=16,
        description="Current output text pointers. Active/uncertain/superseded issues need an unresolved_work or next_checks pointer; resolved/inapplicable/deferred can remain audit-only.")


STATE_PROMPT = """本次使用 active_guidance_state：历史不等于当前诊断，当前诊断不等于行动指令。
prior_items只是旧指导中的待核实说法，不能作为新观察或默认成立的批评。
先输出guidance_state_updates更新旧问题，再根据更新后的状态写turns中的当前指导。
逐项覆盖所有prior_item_ids；合并同状态、同依据、同后续处理的旧项，减少重复解释，
每个ID恰好出现一次。不要重抄历史，只给简短说明、证据引用和新指导位置。
active=当前仍需处理；uncertain=仍有冲突或缺证据；resolved=已有当前证据支持问题解决；
superseded=被更具体或不同的当前问题替代；inapplicable=旧要求不属于实际职责/用户要求；
deferred=此次未审查，保留未决，绝不是已解决。没有新证据不能把旧问题标成resolved。
同一音频的矛盾观察不能靠最新优先消解；换音频版本也不等于旧缺陷已修复。
evidence_refs只能引用current_evidence_refs给出的JSON指针，不能引用旧指导来证明自己。
guidance_refs指向本次输出中的具体字符串，如/turns/0/modules/0/next_checks/0。
active/uncertain/superseded必须指向新的unresolved_work或next_checks；resolved若进入
当前指导只能放supported_work或upstream_changes，也可仅保留在审计状态中。
inapplicable和deferred可以不进入给Actor的指导。
新指导只保留当前有效的问题与保留约束；不能同时宣布旧问题解决又要求修同一旧问题。
区分“已回答一个观察问题”和“已修好产物”：发现真实故障后，旧观察问题可以resolved，
产物故障仍须在新指导中保留；已确认但未修复的故障也属于active，不是resolved。
如果本轮已核查声音，就不要让Planner或Reasoner再次核查同一已回答的问题。
把已核实偏差与工具实际返回连接成一个可检验假设，指出要检查的输入条件或表示类别；
不能仅重复编辑目标和笼统的“再验证音频”。计划改变和后续音频复核是不同模块的职责。
不要在next_checks中替模块给出本题的新参数值，即使冠以“例如”也仍是代填；用值的
类型、需满足的关系和工具条件来指引。上游已经得出的值可留在supported_work。
Planner的next_checks必须在生成当前计划之前可做：检查证据、参数表示、合同与依赖。
“重新执行后听评/验证产物”属于Eval，不能放入Planner或Reasoner的next_checks；后续
产物应满足的要求可以留在preservation_requirements，不额外分配执行后审计职责。
引用单条听评时注明它是观察报告，不能将未经核实的精确声音描述升级为已证实的后端故障。
旧项的module只记录上一次投递位置，不证明责任归属正确。根据当前可见的上游交接，
允许将旧Planner检查替代为Reasoner的目标表达补充，再给Planner条件性编译提醒；
不要为了维持旧清单的模块归属而复制已经未奏效的指导。旧记录仍保留在状态审计。
语义目标正确与合成表示有效要分开。不要让工具后端失败抹掉正确推理；不要要求执行前
模块验证尚未生成的产物。被替代的原始记录仍保留供审计，不重新放进Actor的检查清单。
"""


def _pointer_part(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def resolve_pointer(value, pointer):
    if not pointer.startswith("/"):
        raise ValueError("expected an absolute JSON pointer")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, (list, tuple)):
            if not key.isdecimal() or key != str(int(key)):
                raise ValueError("array reference must use a nonnegative canonical index")
            value = value[int(key)]
        else:
            value = value[key]
    return value


def _current_refs(payload):
    refs = {}
    for identity, observation in (payload.get("current_evidence_state") or {}).get("observations", {}).items():
        if not observation["stale_for_current_artifact"]:
            refs[f"/current_evidence_state/observations/{_pointer_part(identity)}/value"] = {
                "kind": "recorded_observation", "source_kind": observation["kind"],
            }
    # Explicit namespaces exclude old advice and private/unseen branch data.
    for field in ("public_prefixes", "diagnostic_scope", "recorded_tool_observations",
                  "terminal_audio_review", "new_targeted_observations"):
        value = payload.get(field)
        if value:
            refs["/" + field] = {"kind": field}
    for i, review in enumerate(payload.get("atomic_reviews", [])):
        for identity, entry in review.get("evidence", {}).items():
            if entry.get("source") == "module_input" and entry.get("pointer") == "/future_attempt_guidance":
                continue
            ref = f"/atomic_reviews/{i}/evidence/{_pointer_part(identity)}"
            refs[ref] = {"kind": entry.get("source", "recorded_evidence")}
        for j, answer in enumerate(review.get("answers", [])):
            refs[f"/atomic_reviews/{i}/answers/{j}"] = {"kind": "fallible_atomic_answer"}
    return refs


def guidance_state_context(payload, historical_material, *, turn_index):
    """Index only the last consumed advice; raw full history stays in audit files."""
    material = historical_material if isinstance(historical_material, dict) else {}
    # A first invocation records future_attempt_guidance=null. It is an absent
    # prior instruction, not a historical packet to reconcile.
    consumed = [e["value"] for e in material.get("prior_guidance_input_entries", {}).values()
                if e["value"] is not None]
    earlier = material.get("earlier_episode_reviews", material.get("earlier_same_arm_feedback", []))
    packets = consumed or ([earlier[-1]] if earlier else [])
    items = deepcopy(material.get("deferred_prior_items", {}))
    context, seen = [], set()
    for packet in packets:
        for module in packet.get("module_feedback", packet.get("modules", [])):
            identity = canonical_evaluation_fingerprint(module)
            if identity in seen:
                continue
            seen.add(identity)
            context.append({"module": module["module"], "supported_work": module.get("supported_work", []),
                            "preservation_requirements": module.get("preservation_requirements", [])})
            for field in ("unresolved_work", "next_checks"):
                for index, text in enumerate(module.get(field, [])):
                    key = f"prior:{identity[:16]}:{field}:{index}"
                    items[key] = {"turn_index": turn_index, "module": module["module"],
                                  "field": field, "text": text, "authority": "previous_advice_not_evidence"}
    return {"protocol": PROTOCOL, "prior_items": items, "prior_supported_and_preservation_context": context,
            "current_evidence_refs": _current_refs(payload)}


def deferred_guidance_items(draft, payload):
    """Keep explicitly unreviewed issues available to the next diagnostic pass."""
    items = payload.get("active_guidance_state", {}).get("prior_items", {})
    return {key: deepcopy(items[key]) for update in draft.guidance_state_updates
            if update.disposition == "deferred" for key in update.prior_item_ids}


def latest_deferred_guidance(history):
    """A failed/unused review is not an instruction to erase pending issues."""
    for record in reversed(history):
        review = record.get("review", {})
        if "deferred_guidance_items" in review:
            return deepcopy(review["deferred_guidance_items"])
    return {}


def refresh_guidance_state(payload):
    """Refresh citation availability after new observations, keeping old items fixed."""
    result = deepcopy(payload)
    if "active_guidance_state" in result:
        result["active_guidance_state"]["current_evidence_refs"] = _current_refs(result)
    return result


def validate_guidance_state(draft, payload):
    """Check coverage and citations; the existing semantic audit checks meaning."""
    context = payload.get("active_guidance_state")
    if context is None:
        if draft.guidance_state_updates:
            raise ValueError("guidance state updates lack their bound prior context")
        return None
    prior = context["prior_items"]
    current_refs = _current_refs(payload)
    raw = draft.model_dump(mode="json")
    covered = []
    for update in draft.guidance_state_updates:
        ids = update.prior_item_ids
        if set(ids) - set(prior):
            raise ValueError("guidance update cites an unknown prior item")
        covered.extend(ids)
        if set(update.evidence_refs) - set(current_refs):
            raise ValueError("guidance update cites unavailable current evidence")
        for ref in update.evidence_refs:
            resolve_pointer(payload, ref)
        if update.disposition in {"resolved", "superseded", "inapplicable"}:
            # Public requirements can disqualify an invented duty but cannot
            # alone prove an execution failure has been repaired.
            meaningful = [r for r in update.evidence_refs if current_refs[r]["kind"] not in {"public_prefixes", "diagnostic_scope"}]
            if not update.evidence_refs or (update.disposition != "inapplicable" and not meaningful):
                raise ValueError("retiring a prior issue requires current supporting evidence")
        fields = []
        for ref in update.guidance_refs:
            parts = ref.strip("/").split("/")
            if len(parts) != 6 or parts[0] != "turns" or parts[2] != "modules":
                raise ValueError("guidance reference must address a current module text item")
            try:
                value = resolve_pointer(raw, ref)
                turn = raw["turns"][int(parts[1])]["turn_index"]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise ValueError("guidance reference is missing from current output") from exc
            if not isinstance(value, str) or not value.strip() or any(prior[k]["turn_index"] != turn for k in ids):
                raise ValueError("guidance reference belongs to another turn or an empty item")
            fields.append(parts[4])
        if update.disposition in {"active", "uncertain", "superseded"} and not set(fields) & {"unresolved_work", "next_checks"}:
            raise ValueError("an unresolved or replacement issue must remain in active guidance")
        if update.disposition == "resolved" and set(fields) - {"supported_work", "upstream_changes"}:
            raise ValueError("resolved issue must reference supported work, not a new repair demand")
    if len(covered) != len(set(covered)) or set(covered) != set(prior):
        raise ValueError("guidance state must account for every prior issue exactly once")
    return {"protocol": PROTOCOL, "prior_item_count": len(prior),
            "updates": [u.model_dump(mode="json") for u in draft.guidance_state_updates],
            "semantic_correctness": "requires_independent_scope_review",
            "history_discarded": False}
