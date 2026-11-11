"""Production review support extracted without changing function bodies."""

from __future__ import annotations


import json


from x_editor.revision.atomic_module_review import invoke as raw_invoke, public_context


from x_editor.revision.refinement_scheduling import limited


from x_editor.revision.research_quality_review import RecordedJudgeClient


from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint


from x_editor.evaluation.iterative_runtime import sha256, write_record


from x_editor.evaluation.refinement_contracts import (
    CriterionEntailment,
    GoalInterpretationReview,
    PublicUtilityDesign,
    RegionUtilityDesign,
    RuntimeRegionUtilityDesign,
    RegionGoalInterpretationReview,
    SourceListeningNotes,
)


from x_editor.evaluation.refinement_feedback import UTILITY_DESIGN_PROMPT


from x_editor.evaluation.edit_region_scope import (
    REGION_COVERAGE_PROMPT, REGION_DESIGN_PROMPT, REGION_REQUIREMENTS_PROMPT,
    bind_region_design, regional_coverage_problems,
)


from x_editor.evaluation.runtime_scope_contracts import RUNTIME_GOAL_PROMPT


def invoke(path, prompt, args, *, author=False):
    return limited(raw_invoke, getattr(args, "semantic_limiter", None),
                   path, prompt, args, author=author)


def call_audio(config, prompt, payload, paths, directory):
    return limited(_call_audio, config.get("limiter"), config, prompt, payload, paths, directory)


def _call_audio(config, prompt, payload, paths, directory):
    seed = config.get("seed", 0)
    if config.get('seed_namespace') is not None:
        seed = int(canonical_evaluation_fingerprint(
            [config['seed_namespace'], seed, prompt, payload])[:8], 16) % (2**31)
    client = RecordedJudgeClient(
        directory=directory,
        api_key=None,
        base_url=config["base_url"],
        model_name=config["model"],
        provider_name=config["provider"],
        temperature=config.get('temperature', 0),
        seed=seed,
        max_tokens=config["max_tokens"],
        timeout=config["timeout_sec"],
        max_retries=0,
        retry_backoff=0,
        allow_json_repair=False,
        allow_response_format_fallback=False,
        require_api_key=False,
        chat_template_kwargs=config.get("chat_template_kwargs", {}),
        audio_content_part="input_audio",
    )
    return client.call_json(
        prompt,
        json.dumps(payload, ensure_ascii=False),
        audio_paths=paths,
        response_format={"type": "json_object"},
        request_id=canonical_evaluation_fingerprint([str(directory), payload])[:24],
    )


SOURCE_NOTES_PROMPT = """只描述附件源录音的实际内容，不猜测后续编辑任务或正确答案。
尽量逐字记录听到的语音，保留不确定性；另述非语音声音。不从文件名推断事件。
这不是对候选产物评分，也没有提供候选。直接返回 heard_speech、heard_sounds、
uncertainties 三个字段的结果对象。不要回传schema，也不能将实际答案填进schema.properties。
heard_speech 中同时说明能否区分实际说话人及发声范围；无法判断则写入 uncertainties。"""


TASK_FOCUSED_GOALS = """本次围绕三个实际问题出题：编辑要求完成了吗；应保留的内容、成分或
属性保留了吗；是否引入点击、截断、重复、失真或不自然过渡。不要为填满分类表增加任务
没有的要求，不输出 retention_coverage。三方面不等于固定三题：多个目标位置分别覆盖，
每题一个可判定条件。保留可以和编辑发生在同一秒，比如改人声而保留伴奏，换音色而保留
文字；用 region_description 明确声音成分和属性，不机械地只问未编辑时间段。
题目只根据公开请求和源观察。此处不提供候选，不猜测候选的秒数；实际范围稍后由运行时
感知绑定。全局处理没有局部接点可说明不适用，仍检查新增伪影。允许不同合法工具方案。
格式与引用错误可以修订；不得通过删掉必要要求、添加隐含答案来取得资格。"""


GOAL_REVIEW_PROMPT = """复核这些终态检验题有没有准确理解公开指令。你只看源录音观察和题目，
看不到执行结果。检查题目是否混淆话语表达的命题值与声音属性或录音坐标；是否把没有
证据的解释写成确定目标；是否漏掉仍有效的保留要求，或增加了用户没要求的修改。
源观察也可能不可靠，无法判断就不认证。不同合法编辑方式应都能被接受，不要求唯一
工具图。默认保留内容/身份不意味着固定总时长或逐样本一致。精确约束只能作用于用户
实际指定的物理量，不能扩散到其他量。仅回答题意是否有支持、需求是否保持、未解释
假设及理由，返回所给schema。"""


CRITERION_ENTAILMENT_PROMPT = """独立审查当前这一道音频检验题有没有提出合理要求。
只看公开指令与源观察；不能因整份设计看起来合理就通过本题。指出它来自哪一条明确
要求，或哪一种默认保留/听觉质量要求。默认保留内容和身份不等于逐样本相同、每个音素
时长不变、总时长精确固定或声学特征完全相同。局部重建造成的必要时长适配不能擅自
判错；只有指令明确锁定相关物理量时才能按精确约束检验。精确修饰哪一个量就约束
哪个量，不能扩散。想象另一种合法实现：本题是否会因用户未指定的要求而错误拒绝它？
若有这种过度约束或无根据的假设，supported=false。返回该题CriterionEntailment schema。"""


def author_utility(scenario, *, output, args, audio_config):
    regional = getattr(args, "terminal_rubric_protocol", "legacy") in {"edit_regions", "runtime_regions"}
    runtime_regions = getattr(args, "terminal_rubric_protocol", "legacy") == "runtime_regions"
    task_focused = getattr(args, 'task_focused_goals', False)
    design_model = RuntimeRegionUtilityDesign if runtime_regions and not task_focused else RegionUtilityDesign if regional else PublicUtilityDesign
    design_prompt = UTILITY_DESIGN_PROMPT + ("\n" + REGION_REQUIREMENTS_PROMPT if runtime_regions
                                           else "\n" + REGION_DESIGN_PROMPT if regional else "")
    if runtime_regions and not task_focused:
        design_prompt += "\n" + RUNTIME_GOAL_PROMPT
    if task_focused:
        design_prompt += "\n" + TASK_FOCUSED_GOALS
    context = public_context(scenario.model_dump(mode="json"), len(scenario.turns) - 1)
    source_notes = []
    for index, asset in enumerate(scenario.audio_assets):
        if sha256(asset.path) != asset.content_sha256:
            raise ValueError("source changed before output-blind goal authoring")
        response = call_audio(
            audio_config,
            SOURCE_NOTES_PROMPT,
            {"schema": SourceListeningNotes.model_json_schema()},
            [asset.path],
            output / "source_listening" / f"{index:04d}",
        )
        note = SourceListeningNotes.model_validate(response)
        source_notes.append(
            {
                "audio_ref": asset.asset_ref,
                "sha256": asset.content_sha256,
                "role": asset.role,
                "observation": note.model_dump(mode="json"),
            }
        )
    write_record(output / "source_notes.json", source_notes)
    rounds = getattr(args, "goal_resolution_rounds", 0)
    if type(rounds) is not int or not 0 <= rounds <= 3:
        raise ValueError("goal_resolution_rounds must be an integer from zero to three")
    design, previous_draft, problems, guidance = None, None, [], None
    for index in range(rounds + 1):
        current = output if index == 0 else output / "design_revisions" / f"{index:04d}"
        payload = {"public_context": context.model_dump(mode="json"),
                   "source_observations": source_notes,
                   "schema": design_model.model_json_schema()}
        if previous_draft is not None:
            payload.update(previous_design=previous_draft,
                           previous_review_problems=problems, revision_guidance=guidance)
        raw = invoke(
            current / "author", {"system_prompt": design_prompt,
                                 "user_prompt": json.dumps(payload, ensure_ascii=False)},
            args, author=True)
        write_record(current / "author_original.json", raw)
        previous_draft = raw
        try:
            design = design_model.model_validate(raw)
            if regional:
                bind_region_design(design, scenario)
        except ValueError as exc:
            # The model revises its own complete draft under the existing
            # budget. Never manufacture criterion IDs or skip qualification.
            design = None
            problems = [{"stage": "design_contract", "error": str(exc)}]
            write_record(current / "qualification_problems.json", problems)
            guidance = ("上一版设计未通过结构或引用合同。依据具体错误重新返回完整设计，"
                        "核对保留题的类别与引用；不删减公开要求、不编造资产或时间范围。"
                        "修订后仍需通过独立题意审查。")
            continue
        write_record(current / "utility_design_draft.json", design.model_dump(mode="json"))
        problems = audit_utility(design, context, source_notes, output=current, args=args)
        if not problems:
            write_record(output / "utility_design.json", design.model_dump(mode="json"))
            write_record(output / "design_resolution_receipt.json",
                         {"revision_rounds": index, "configured_max_rounds": rounds,
                          "candidate_audio_visible": False, "status": "qualified"})
            return design
        write_record(current / "qualification_problems.json", problems)
        if index == rounds:
            break
        from x_editor.revision.goal_design_resolution import resolve_goal_design
        decision, source_notes = resolve_goal_design(
            scenario, context, source_notes, design, problems,
            output=output / "investigations" / f"{index:04d}", args=args,
            audio_config=audio_config, invoke=invoke, call_audio=call_audio)
        if decision.action == "unknown":
            break
        guidance = decision.design_guidance
    write_record(output / "design_resolution_receipt.json",
                 {"revision_rounds": index, "configured_max_rounds": rounds,
                  "candidate_audio_visible": False, "status": "unresolved"})
    if any(p["stage"] == "design_contract" for p in problems):
        raise ValueError("public goal design contract remains invalid; original drafts retained")
    if any(p["stage"] == "interpretation" for p in problems):
        raise ValueError("public goal interpretation is not qualified; original draft retained")
    raise ValueError("an individual utility criterion is not justified by the public request")


def audit_utility(design, context, source_notes, *, output, args):
    from x_editor.revision.review_validation import invoke_validated_review

    regional = isinstance(design, RegionUtilityDesign)
    audit_model = RegionGoalInterpretationReview if regional else GoalInterpretationReview
    audit = invoke_validated_review(
        model=audit_model, invoke=invoke, directory=output / "interpretation_review",
        prompt={
                "system_prompt": GOAL_REVIEW_PROMPT + ("\n" + REGION_COVERAGE_PROMPT if regional else "")
                    + ("\n" + RUNTIME_GOAL_PROMPT if isinstance(design, RuntimeRegionUtilityDesign) else ""),
                "user_prompt": json.dumps(
                    {
                        "schema": audit_model.model_json_schema(),
                        "public_context": context.model_dump(mode="json"),
                        "source_observations": source_notes,
                        "design": design.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
        },
        args=args,
    )
    write_record(output / "interpretation_review.json", audit.model_dump(mode="json"))
    problems = []
    if regional:
        coverage_issues = regional_coverage_problems(design, audit.coverage_checks)
        if coverage_issues:
            problems.append({"stage": "regional_coverage", "issues": coverage_issues})
            return problems
    if (
        not audit.interpretation_supported
        or not audit.requirements_preserved
        or audit.unexplained_assumptions
    ):
        problems.append({"stage": "interpretation", "review": audit.model_dump(mode="json")})
        return problems
    if isinstance(design, RuntimeRegionUtilityDesign) or getattr(args, 'task_focused_goals', False):
        from x_editor.revision.runtime_goal_review import audit_runtime_questions
        return audit_runtime_questions(design, context, source_notes, output=output, args=args,
            invoke=invoke, prompt=CRITERION_ENTAILMENT_PROMPT)
    for criterion in design.criteria:
        answer = CriterionEntailment.model_validate(
            invoke(
                output / "criterion_entailment" / criterion.criterion_id,
                {
                    "system_prompt": CRITERION_ENTAILMENT_PROMPT,
                    "user_prompt": json.dumps(
                        {
                            "schema": CriterionEntailment.model_json_schema(),
                            "public_context": context.model_dump(mode="json"),
                            "source_observations": source_notes,
                            "criterion": criterion.model_dump(mode="json"),
                            **({"region_check": next(c.model_dump(mode="json") for c in design.region_checks
                                                      if c.criterion_id == criterion.criterion_id)}
                               if regional else {}),
                        },
                        ensure_ascii=False,
                    ),
                },
                args,
            )
        )
        write_record(
            output / "criterion_entailment" / criterion.criterion_id / "verdict.json",
            answer.model_dump(mode="json"),
        )
        if (
            answer.criterion_id != criterion.criterion_id
            or not answer.supported
            or answer.requirement_basis == "unsupported"
        ):
            problems.append({"stage": "criterion_entailment", "review": answer.model_dump(mode="json")})
    return problems
