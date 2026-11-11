"""Production review support extracted without changing function bodies."""

import json


from typing import Literal


from pydantic import Field, model_validator


from x_editor.evaluation.guidance_contracts import FrozenEvaluationModel


from x_editor.evaluation.iterative_runtime import sha256, write_record


from x_editor.evaluation.refinement_contracts import SourceListeningNotes


class SourceQuestion(FrozenEvaluationModel):
    audio_ref: str = Field(min_length=1)
    comparison_audio_refs: tuple[str, ...] = Field(default=(), max_length=3)
    question: str = Field(min_length=8, max_length=1000)
    missing_fact: str = Field(min_length=4, max_length=800)


class GoalResolution(FrozenEvaluationModel):
    action: Literal["observe", "revise_design", "unknown"]
    source_questions: tuple[SourceQuestion, ...] = Field(default=(), max_length=4)
    design_guidance: str = Field(min_length=8, max_length=2000)

    @model_validator(mode="after")
    def action_matches_questions(self):
        if bool(self.source_questions) != (self.action == "observe"):
            raise ValueError("only an observation action supplies source questions")
        return self


RESOLUTION_PROMPT = """这是公开音频任务的评测设计复核，不是修改执行结果。
题目尚未认证，请根据指令、源观察、原题目和审查意见选择下一步。
observe：事实缺失但可听取源音频。具体问缺少的事实，例如实际声源身份及其范围、
某个事件是否出现或已知位置是否正确；不能把用户提及当作存在证明，不能预设观察答案。
revise_design：已有事实足够，但题目增加了用户未要求的精确约束，或审查意见内部矛盾。
说明如何修正题意或区分假设，不改变用户目标、不人为降低验收标准；修改后仍须重新审查。
unknown：有界观察仍无法解决，或确需用户澄清，保留未知。
缺少说话人标签、一次转写遗漏、没描述事件，不是已经确认不存在；应先选择针对性观察。
只访问audio_inventory中的源/参考音频，不接收候选产物、历史得分或私有构造答案。
如需判断源录音中哪个声音与参考录音为同一说话人，把参考列入 comparison_audio_refs，
让同一次听取真正比较这些声音。分别转写两份文本、只知道性别，均不能证明声音身份相同。
返回所给schema。"""


SOURCE_QUESTION_PROMPT = """只核听这一份源录音，回答question所需的实际声音事实。
问题不是标准答案。先确认是否有足够证据；没有听清就保留不确定，不能迎合预设。
如涉及说话人，描述可区分的声音及各自实际发声时间，不把转录中的人名当成新说话人。
如涉及事件，说明是否听到、可支持的时间范围和定位不确定性。允许重叠；秒数相对各附件。
存在比较附件时，实际对照声音回答身份或声音关系，说明对应附件编号，不能只比较转写文字。
heard_speech记录相关实际语音与说话人观察，heard_sounds记录实际非语音声，uncertainties
记录未解决的问题；不能编造精确时间。直接返回这三个字段的观察结果对象，
不要回传schema，也不能把实际答案填进schema.properties。"""


def resolve_goal_design(scenario, context, notes, design, problems, *, output, args,
                        audio_config, invoke, call_audio):
    assets = {a.asset_ref: a for a in scenario.audio_assets}
    payload = {
        "schema": GoalResolution.model_json_schema(),
        "public_context": context.model_dump(mode="json"),
        "source_observations": notes, "design": design.model_dump(mode="json"),
        "review_problems": problems,
        "audio_inventory": [{"audio_ref": a.asset_ref, "sha256": a.content_sha256,
                             "role": a.role} for a in assets.values()],
    }
    decision = GoalResolution.model_validate(invoke(
        output / "decision", {"system_prompt": RESOLUTION_PROMPT,
                              "user_prompt": json.dumps(payload, ensure_ascii=False)}, args, author=True))
    write_record(output / "resolution.json", decision.model_dump(mode="json"))
    if any(ref not in assets for q in decision.source_questions
           for ref in (q.audio_ref, *q.comparison_audio_refs)):
        raise ValueError("goal investigation referenced an unavailable audio asset")
    observations = list(notes)
    for index, question in enumerate(decision.source_questions):
        asset = assets[question.audio_ref]
        refs = (question.audio_ref, *question.comparison_audio_refs)
        if len(set(refs)) != len(refs):
            raise ValueError("goal investigation repeated an audio asset")
        attachments = [assets[ref] for ref in refs]
        if any(sha256(a.path) != a.content_sha256 for a in attachments):
            raise ValueError("goal investigation source changed")
        raw = call_audio(audio_config, SOURCE_QUESTION_PROMPT,
                         {"schema": SourceListeningNotes.model_json_schema(),
                          "question": question.question,
                          "audio_inventory": [{"audio_ref": a.asset_ref, "role": a.role,
                                               "attachment_number": i + 1}
                                              for i, a in enumerate(attachments)]},
                         [a.path for a in attachments], output / "observations" / f"{index:04d}")
        note = SourceListeningNotes.model_validate(raw)
        if any(sha256(a.path) != a.content_sha256 for a in attachments):
            raise ValueError("goal investigation source changed during listening")
        observations.append({"audio_ref": asset.asset_ref, "sha256": asset.content_sha256,
                             "role": asset.role, "targeted_question": question.question,
                             "observation": note.model_dump(mode="json"),
                             "comparison_assets": [{"audio_ref": a.asset_ref, "sha256": a.content_sha256,
                                                    "role": a.role} for a in attachments[1:]],
                             "authority": "fallible_source_observation"})
    write_record(output / "source_notes.json", observations)
    return decision, observations
