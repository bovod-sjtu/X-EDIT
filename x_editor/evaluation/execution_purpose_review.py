"""Sequential, artifact-bound purpose observations; never an executor gate.

The audit supplies outputs, not filenames or output order guesses. A temporary
stem is judged for its own purpose. Only terminal criteria select a delivery.
"""

import json
from pathlib import Path

import soundfile as sf
from pydantic import Field, model_validator

from x_editor.evaluation.guidance_contracts import (
    FrozenEvaluationModel,
    canonical_evaluation_fingerprint,
)
from x_editor.evaluation.iterative_runtime import sha256, write_record
from x_editor.evaluation.numeric_polarity_review import judge_evidence
from x_editor.evaluation.evidence_ordinal_review import observe_audio
from x_editor.evaluation.purpose_rubrics import AUTHOR_PROMPT

PROTOCOL = "x-editor.execution-purpose-sequence/v1"


class StepPurpose(FrozenEvaluationModel):
    question_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    turn_index: int = Field(ge=0)
    step_id: str
    candidate_ref: str
    comparison_refs: tuple[str, ...] = ()
    question: str = Field(min_length=4, max_length=600)
    aspect: str = Field(pattern=r"^(effect|preservation|boundary)$")

    @model_validator(mode="after")
    def references(self):
        if (
            len(set(self.comparison_refs)) != len(self.comparison_refs)
            or self.candidate_ref in self.comparison_refs
        ):
            raise ValueError("comparison roles must be distinct")
        if self.aspect == "preservation" and not self.comparison_refs:
            raise ValueError("preservation requires a real comparison")
        return self


class StepDesign(FrozenEvaluationModel):
    questions: tuple[StepPurpose, ...] = Field(default=(), max_length=40)
    unreviewed_steps: tuple[str, ...] = ()


class StepDesignAudit(FrozenEvaluationModel):
    original_design_issues: tuple[str, ...] = ()
    revised_design: StepDesign
    unresolved: tuple[str, ...] = ()


class StepQualification(FrozenEvaluationModel):
    qualified_question_ids: tuple[str, ...]
    unresolved: tuple[str, ...] = ()


def execution_inventory(scenario, result):
    """Use public asset identity and audited producer bindings, across all turns."""
    assets = {
        a.asset_ref: dict(path=a.path, sha256=a.content_sha256, role=a.role)
        for a in scenario.audio_assets
    }
    steps = []
    for turn in result.turn_results:
        forward = turn.result.forward_result
        if not forward or not forward.executed_plan:
            continue
        audit = forward.artifact_audit
        records = [a.model_dump(mode="json") for a in audit.artifacts] if audit else []
        for step in forward.executed_plan.steps:
            outputs = []
            for record in records:
                if record.get("producer_step_id") != step.id or not record.get("integrity_passed"):
                    continue
                ref = f"step:{turn.turn_index}:{step.id}:{record['artifact_ref_id']}"
                path = record["path"]
                assets[ref] = dict(path=path, sha256=sha256(path), role="candidate")
                outputs.append(ref)
            steps.append(
                dict(
                    turn_index=turn.turn_index,
                    step_id=step.id,
                    operation=step.operation,
                    inputs=step.inputs,
                    params=step.params,
                    description=step.description,
                    expected_output=step.expected_output,
                    depends_on=step.depends_on,
                    status=step.status,
                    output_refs=outputs,
                    preceding_refs=[ref for ref in assets if ref not in outputs],
                )
            )
    return assets, steps


def sequence_identity(scenario, result):
    assets, steps = execution_inventory(scenario, result)
    return {
        "protocol": PROTOCOL,
        "steps": steps,
        "assets": {k: v["sha256"] for k, v in assets.items()},
    }


def purpose_contracts(module_sources, steps):
    operations = {s["operation"] for s in steps}
    rows = []
    for (turn, module), source in (module_sources or {}).items():
        if module != "planner":
            continue
        cards = source.get("module_input", {}).get("candidate_operation_contracts", [])
        for contract in cards:
            if not isinstance(contract, dict) or contract.get("operation") not in operations:
                continue
            card = contract.get("operation_capability") or contract
            rows.append(
                {
                    "turn_index": turn,
                    **{
                        k: card[k]
                        for k in (
                            "operation",
                            "semantic_effects",
                            "supported_scopes",
                            "requirements",
                            "limitations",
                            "preservation",
                            "preservation_profile",
                        )
                        if k in card
                    },
                }
            )
    return rows


def review_sequence(
    scenario,
    result,
    semantic_call,
    audio_call,
    output,
    *,
    observation_cache=None,
    module_sources=None,
):
    output = Path(output)
    assets, steps = execution_inventory(scenario, result)
    public_steps = [{k: v for k, v in s.items() if k != "preceding_refs"} for s in steps]
    request = {
        "public_instructions": [t.user_request for t in scenario.turns],
        "public_assets": [
            {
                **{k: v for k, v in a.model_dump(mode="json").items() if k != "path"},
                "runtime_audio_ref": a.logical_ref,
            }
            for a in scenario.audio_assets
        ],
        "tool_capabilities": purpose_contracts(module_sources, steps),
        "actual_steps": public_steps,
        "schema": StepDesign.model_json_schema(),
    }
    prompt = (
        AUTHOR_PROMPT
        + """
早期turn的步骤按该轮之前公开的指令判断，不把后续修改目标反向施加于早期输出。
本次仅设计已执行步骤的声音题，不设计语义题或final题。每个有声音目的的步骤至少一题。
步骤描述和expected_output只是待核对的意图，不能当成已经正确完成的事实。
每题仅绑定一个实际output_refs中的输出。comparison_refs可使用公开资产或此前步骤的输出。
临时人声轨不要求自带BGM；最终混音须检查背景。格式搬运等无声音目的的机械步骤无需听评，说明unreviewed_steps理由。
提取/裁切/拼接存在选择内容、顺序或边界的目的，不能一概作为机械步骤跳过。
有输出却无法确定其目的时记录unreviewed_steps，不凭文件名猜角色。范围只引用实际参数中
明确的坐标，否则检查完整输出；不能猜时间或说话人。保留问题分别明确文字/背景/身份。
只返回本次schema的questions与unreviewed_steps两个字段，不输出其他schema。
"""
    )
    rows = []
    errors = []
    try:
        raw = semantic_call(prompt, request, output / "author")
        design = StepDesign.model_validate(raw)
        write_record(output / "original_design.json", design.model_dump(mode="json"))
        audit_prompt = """只审查出题逻辑，不听音频也不评分。每道题应只有一个可独立判断的声音条件。
“文字保留且没有失真”、“顺序正确且衔接自然”必须拆成不同题；同一声音可对应多题。
不能用另一性质的成功补偿失败。保留原问题中的必要对象、范围、比较基准，不能增添要求。
提取指定内容、拼接指定顺序也有需要核查的声音目的，不自动豁免。格式搬运可说明未审原因。
核对每个产生声音的步骤，有具体目的的输出是否覆盖；未能出题就unresolved，不假称覆盖。
original_design_issues只描述输入题的问题；revised_design只放修订后的题，不同时审批自己的修订。
未解决的问题放unresolved。保持比较基准：“该人比自己的原声慢”不能改成“比另一个人慢”。
输入只有公开要求、计划及候选问题，不能虚构实际音频内容。返回schema对象。"""
        audit = StepDesignAudit.model_validate(
            semantic_call(
                audit_prompt,
                {
                    "context": request,
                    "design": design.model_dump(mode="json"),
                    "schema": StepDesignAudit.model_json_schema(),
                },
                output / "design_audit",
            )
        )
        write_record(output / "design_audit.json", audit.model_dump(mode="json"))
        design = audit.revised_design
        qualification = StepQualification.model_validate(
            semantic_call(
                "只审核给定的当前题，不修改题。逐题检查是否只有一个独立的声音条件、是否有公开要求或实际步骤目的支持、"
                "是否保留了正确对象和比较基准。多处文字均保留是一个属性；文字保留和无失真是两个属性。"
                "同一人的语速相对原声变化，不能替换为相对另一人的速度差。不要根据是否能听出答案审批，"
                "当前没有音频。qualified_question_ids列合格题，未合格的原因写unresolved。不能加题、猜答案。返回schema。",
                {
                    "context": request,
                    "current_design": design.model_dump(mode="json"),
                    "schema": StepQualification.model_json_schema(),
                },
                output / "question_qualification",
            )
        )
        write_record(output / "question_qualification.json", qualification.model_dump(mode="json"))
        proposed = {q.question_id for q in design.questions}
        accepted = set(qualification.qualified_question_ids)
        if len(accepted) != len(qualification.qualified_question_ids) or accepted - proposed:
            raise ValueError("question qualification invents or repeats an ID")
        unqualified = [
            q.model_dump(mode="json") for q in design.questions if q.question_id not in accepted
        ]
        design = design.model_copy(
            update={"questions": tuple(q for q in design.questions if q.question_id in accepted)}
        )
        lookup = {(s["turn_index"], s["step_id"]): s for s in steps}
        ids = set()
        for q in design.questions:
            step = lookup.get((q.turn_index, q.step_id))
            if not step or q.candidate_ref not in step["output_refs"]:
                raise ValueError("purpose question bound to a missing or wrong producer")
            if set(q.comparison_refs) - set(step["preceding_refs"]) - set(step["output_refs"]):
                raise ValueError("purpose comparison cites unavailable or future audio")
            if q.question_id in ids:
                raise ValueError("duplicate purpose question")
            ids.add(q.question_id)
        write_record(output / "design.json", design.model_dump(mode="json"))
    except Exception as exc:
        receipt = {
            "protocol": PROTOCOL,
            "status": "design_failed",
            "errors": [str(exc)],
            "observations": [],
            "coverage_certified": False,
            "selection_weight": 0,
        }
        write_record(output / "review.json", receipt)
        return receipt
    observed = {}
    # Execution order, then one condition at a time. No cross-step thread pool.
    for step in steps:
        for q in design.questions:
            if (q.turn_index, q.step_id) != (step["turn_index"], step["step_id"]):
                continue
            directory = output / "questions" / q.question_id
            try:
                refs = [*q.comparison_refs, q.candidate_ref]
                paths = [assets[ref]["path"] for ref in refs]
                inventory = [
                    dict(
                        audio_ref=ref,
                        role="candidate" if ref == q.candidate_ref else "source",
                        attachment_number=i + 1,
                        sha256=assets[ref]["sha256"],
                        duration_sec=sf.info(paths[i]).duration,
                    )
                    for i, ref in enumerate(refs)
                ]
                notes = []
                for item, path in zip(inventory, paths):
                    key = item["sha256"]
                    if key not in observed:

                        def hear():
                            return observe_audio(
                                [dict(item, attachment_number=1)],
                                [path],
                                audio_call,
                                output / "observations" / key,
                            )[0]

                        note, _, _ = (
                            observation_cache.once({"purpose_neutral_audio": key}, hear)
                            if observation_cache
                            else (hear(), False, None)
                        )
                        observed[key] = note
                    notes.append({**observed[key], "audio_ref": item["audio_ref"]})
                decision = judge_evidence(
                    q.question,
                    inventory,
                    paths,
                    notes,
                    audio_call,
                    directory,
                    semantic_call=semantic_call,
                    public_turns=[t.user_request for t in scenario.turns],
                )
                row = {
                    **q.model_dump(mode="json"),
                    **decision,
                    "artifact_sha256": assets[q.candidate_ref]["sha256"],
                }
                rows.append(row)
            except Exception as exc:
                rows.append({**q.model_dump(mode="json"), "grade": None, "error": str(exc)})
                errors.append(str(exc))
    covered = {(q.turn_index, q.step_id) for q in design.questions}
    receipt = {
        "protocol": PROTOCOL,
        "status": "completed" if rows else "questions_unqualified",
        "observations": rows,
        "errors": errors,
        "unreviewed_steps": list(design.unreviewed_steps),
        "design_audit_unresolved": list(audit.unresolved),
        "unqualified_questions": unqualified,
        "question_qualification_unresolved": list(qualification.unresolved),
        "steps_without_checks": [
            {"turn_index": s["turn_index"], "step_id": s["step_id"]}
            for s in steps
            if s["output_refs"] and (s["turn_index"], s["step_id"]) not in covered
        ],
        "coverage_certified": False,
        "selection_weight": 0,
        "interpretation": "Local diagnosis only. Later tools may repair earlier issues; terminal requirements decide delivery.",
    }
    write_record(output / "review.json", receipt)
    return receipt
