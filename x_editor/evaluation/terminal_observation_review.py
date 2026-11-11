"""Task-free, version-bound listening before criterion-local terminal grading.

Descriptions are fallible observations, never labels. Only an explicitly passed
episode cache permits reuse; fresh terminal assessment gets fresh observations.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from pydantic import Field

from x_editor.evaluation.iterative_runtime import sha256, write_record
from x_editor.evaluation.refinement_contracts import SourceListeningNotes, UtilityAnswer
from x_editor.evaluation.terminal_condition_review import TerminalConditionAnswer

OBSERVATION_PROMPT = """只观察这一份附件，不知道也不要猜测任何编辑目标。
逐字记录能听清的全部语音，数字和时间也按实际发音记录；句子中断就停在那里，
不要根据常识补齐。另述实际听到的非语音事件。语气、说话人数量或声音身份若不确定，
明确说明，不能从文本中的人物名称推断真实说话人数。不要推断来源、方法或质量分数。
仅返回 schema；heard_speech 是语音，heard_sounds 是其他声音，uncertainties 保留疑问。
输出是观察结果对象，直接包含 heard_speech、heard_sounds、uncertainties 三个字段。
不能返回 JSON Schema，也不能把实际答案填写在 schema.properties 内。
"""

OBSERVED_UTILITY_PROMPT = """独立核查实际附件中的最终候选，只回答当前 criterion.question。
先写 required_final_condition：仅用一句话重述当前 criterion.question 要求检查的条件。
这不是整项编辑任务的最终目标，不能把其他题目的条件加进来。
再写 observed_condition：候选实际满足或违反了什么，以及简短具体的音频证据。
然后判定 verdict 为 satisfied、violated 或 unknown，最后给 grade。
确认必要条件未实现就是 violated，只能给0或1，不能用“核心满足”或“小缺陷”放行。
3=本题全部满足；2=必要条件满足，仅有小缺陷；1=实质缺失或损伤；0=未实现或方向相反；
null=证据不足。unknown 必须 grade=null 并列出 missing_evidence。
“能听清产物确实缺少目标内容”属于未实现；“听不清是否存在”才是未知。
已授权替换或删除的内容不是未编辑保留区；本题之外的错误不能传染本题。
不从方法、文件名、计划或执行状态推断声音正确。只返回当前 schema。

neutral_observations 是在不知道编辑要求时，分别听取各个文件得到的
观察，只适用于对应 SHA 的音频。观察不是答案，可能漏听，实际附件仍可核听。
先比较本题涉及的实际声音，再与最终有效要求对照。public_turns 和 final_state 只能
说明应该发生什么，不能证明音频已经实现。已授权替换或删除的内容不属于未编辑保留区。
若观察和本次核听冲突，observed_condition 必须说明冲突与支持的实际内容；无法解决则 grade=null。
对候选给出任何确定分数时，evidence_ids 必须包含候选的 audio_ref，不能只引用指令。
引用存在并不代表内容被证实，不能仅补上候选引用而不核听。
评分范围严格限于当前题：目标内容错误可以与说话人保留正确、声音自然同时成立；
若本题询问说话人身份，不能因文字未改或事件未删而扣分；若本题询问声音质量，
不能因指令没完成而扣分。保留要求中的未编辑内容也不能扩展到已授权编辑的部分。
context_only 中的对话只帮助解释本题，不能成为本题以外的附加验收条件。
输出实际答案对象，不要回传 schema 或填充 schema.properties。
"""


def observed_criterion_payload(criterion, public_turns, inventory, observations):
    """A local criterion owns its scope; the complete task is context, not a score."""
    return {
        "criterion": criterion.model_dump(mode="json"),
        "context_only": {"public_turns": public_turns},
        "audio_inventory": inventory,
        "neutral_observations": observations,
    }


class ObservedUtilityAnswer(TerminalConditionAnswer):
    # A one-question request already binds identity outside the model response.
    # An echoed ID, when provided, must agree; missing echo is not missing evidence.
    question_id: str | None = Field(default=None, min_length=1)
    observed_condition: str = Field(min_length=4, max_length=1200)


def observe_attachments(attachments, *, call, output, provider, cache=None, concurrency=1):
    """Observe each distinct waveform once, preserving errors and explicit reuse."""

    def observe(item):
        directory = Path(output) / item["sha256"]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("audio changed before task-free observation")
        identity = {
            "kind": "terminal_task_free_observation",
            "sha256": item["sha256"],
            "prompt": OBSERVATION_PROMPT,
            "provider": provider,
        }

        def produce():
            try:
                info = sf.info(item["path"])
                raw = call(
                    OBSERVATION_PROMPT,
                    {
                        "schema": SourceListeningNotes.model_json_schema(),
                        "duration_sec": info.duration,
                    },
                    [item["path"]],
                    directory / "calls",
                )
                record = {
                    "observation": SourceListeningNotes.model_validate(raw).model_dump(mode="json"),
                    "error": None,
                }
            except Exception as error:  # noqa: BLE001 -- retain external observation failures
                record = {"observation": None, "error": f"{type(error).__name__}: {error}"}
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("audio changed during task-free observation")
            write_record(directory / "observation.json", {"sha256": item["sha256"], **record})
            return record, str(directory)

        if cache is None:
            record, origin = produce()
        else:
            (record, origin), reused, key = cache.once(identity, produce)
            if reused and origin != str(directory):
                write_record(
                    directory / "reuse.json",
                    {"source": origin, "key": key, "new_calls": 0, "sha256": item["sha256"]},
                )
        return item["sha256"], record

    unique = {a["sha256"]: a for a in attachments}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = dict(pool.map(observe, unique.values()))
    return [
        {
            "audio_ref": a["audio_ref"],
            "role": a["role"],
            "sha256": a["sha256"],
            **records[a["sha256"]],
        }
        for a in attachments
    ]


def validate_terminal_references(answer, criterion_id, inventory):
    """A score about an output needs evidence from that output, in either mode."""
    refs = {a["audio_ref"] for a in inventory}
    candidates = {a["audio_ref"] for a in inventory if a["role"] == "candidate"}
    if len(candidates) != 1:
        raise ValueError("one bound terminal candidate is required")
    if (answer.question_id is not None and answer.question_id != criterion_id) or set(
        answer.evidence_ids
    ) - refs - {"public_turns"}:
        raise ValueError("answer uses the wrong criterion or reference")
    if answer.grade is not None and not candidates.intersection(answer.evidence_ids):
        raise ValueError(
            "a determinate terminal grade needs candidate evidence, not only requirements"
        )
    return answer


def validate_observed_answer(payload, criterion_id, inventory):
    answer = validate_terminal_references(
        ObservedUtilityAnswer.model_validate(payload), criterion_id, inventory
    )
    return UtilityAnswer(
        question_id=criterion_id,
        grade=answer.grade,
        finding=answer.observed_condition,
        evidence_ids=answer.evidence_ids,
    )
