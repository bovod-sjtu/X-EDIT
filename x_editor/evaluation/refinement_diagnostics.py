"""Questions and exact runtime facts that semantic summaries must not erase."""

from x_editor.evaluation.atomic_module_rubrics import AtomicModuleQuestion


CAPABILITY_CONTEXT_PROMPT = """
available_tool_capabilities是本轮Planner实际收到的可选能力摘要，不是产物质量证据。
可以根据已核实的缺口考虑兼容的其他工具或组合；先保留已满足的用户要求和编辑范围，
再请Reasoner或Planner形成方案。能力偏好不强制路由，声明的局限不能证明本次失败原因。
摘要不含完整参数语法；实际调用仍由Planner依照原始工具合同生成。
"""


def available_capability_briefs(catalog):
    """Expose offered alternatives without replaying the large parameter schema.

    Read recorded input, never today's tool catalog: an unavailable or newly
    added tool must not silently enter a frozen recovery comparison.
    """
    fields = ("operation", "semantic_effects", "supported_scopes", "requirements", "limitations")
    briefs = []
    for identity, entry in catalog["entries"].items():
        if (entry["source"] != "module_input"
                or not entry["pointer"].startswith("/candidate_operation_contracts/")):
            continue
        value = entry["value"]
        if not isinstance(value, dict):
            continue
        card = value.get("operation_capability")
        # The argument-table wire format projects the canonical card onto the
        # contract itself. Both forms are actual recorded Planner inputs.
        if card is None and "semantic_effects" in value and "arguments" in value:
            card = value
        if not isinstance(card, dict) or card.get("operation") != value.get("operation"):
            continue
        briefs.append({"input_evidence_id": identity,
                       **{key: card[key] for key in fields if key in card},
                       "preservation": card.get("preservation", card.get("preservation_profile", []))})
    return briefs


def executable_contract_question():
    """Universal API validity; separate from task-specific semantic parameters."""
    return AtomicModuleQuestion(
        question_id="planner_executable_contract",
        module="planner",
        kind="plan_part",
        part="parameters",
        requirement_ids=(),
        question=(
            "把当前计划的实际参数交给所选工具，能通过该工具的参数语法与前置检查吗？"
            "核对实际合同及这份计划的运行时准入记录，指出具体字段，不能只读描述。"
        ),
        success_condition=(
            "存在执行计划时，实际参数满足所选工具的公开格式、语义类型及已检查的前置要求。"
            "模块JSON被接受或图可编译，不证明工具专属参数语法有效；已记录的相关blocking"
            "拒绝不能因描述正确而忽略。不将工具尚未调用当作已失败，也不认证声音质量。"
            "若选择合法的历史状态恢复或现有产物交付，不要求虚构新工具计划；核对实际决策"
            "合同、所选状态/资产及记录的恢复结果，能够确认该分支执行合同满足时可判满足。"
            "其他没有执行计划或无法核验该分支合同时为null，由决策题另评是否合理。"
        ),
    )


def diagnostic_evidence(catalog, cited_ids):
    """Retain selected API contracts and actual rejections beside judge summaries.

    Selection uses declared operation identity, never user wording. Other
    capabilities remain available in every atomic judge's full catalog.
    """
    entries = catalog["entries"]
    operations = set()
    for entry in entries.values():
        if entry["source"] == "module_output" and entry["pointer"] == "/plan":
            plan = entry["value"]
            if isinstance(plan, dict):
                operations.update(step["operation"] for step in plan.get("steps", []))
    keep = set(cited_ids)
    for identity, entry in entries.items():
        if entry["source"] == "contract_audit" and entry["pointer"] != "/compiled_execution_graph":
            keep.add(identity)
        if (
            entry["source"] == "module_input"
            and entry["pointer"].startswith("/candidate_operation_contracts/")
            and entry["value"].get("operation") in operations
        ):
            keep.add(identity)
    return {
        identity: entry
        for identity, entry in entries.items()
        if identity in keep and not is_prior_guidance(entry)
    }


def is_prior_guidance(entry):
    """A typed input field contains past advice, not a current observation."""
    return entry["source"] == "module_input" and entry["pointer"] == "/future_attempt_guidance"
