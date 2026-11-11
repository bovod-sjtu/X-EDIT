"""Two numeric directions over identical evidence; no human labels or answer leakage."""

import json
from pathlib import Path
from statistics import mean

from x_editor.evaluation.guidance_contracts import canonical_evaluation_fingerprint
from x_editor.evaluation.iterative_runtime import sha256, write_record
from x_editor.evaluation.ordinal_ensemble import decode
from x_editor.evaluation.ordinal_review import ORDINAL_LEVELS
from x_editor.evaluation.polarity_probe import balanced_score, reverse_levels

PROTOCOL = "independent-evidence-numeric-positive-reverse/v1"

JUDGMENT = """Listen to the supplied audio and assess exactly the stated question using its level definitions.
The requested edit is not evidence that it happened. Do not assume compliance.
Judge only this criterion; success or failure on another criterion cannot change this answer.
Attachment roles and original time windows are explicit. A crop starts locally at zero;
its artificial endpoints are not editing artifacts. Observations were made without knowing the
requested edit and may be wrong. Measurements apply only to their stated regions and properties.
Compare those observations with the audio; do not extend a local fact to the whole recording.
If the attachments do not support a judgment, use the unknown response rather than a passing grade.
Confidence is your estimated probability that choosing the upper versus lower half of the ordered
quality levels agrees with reliable independent listening. Unknown requires null confidence.
Return only the specified JSON fields. Do not explain, transcribe, or propose a repair.
"""
NUMERIC = (
    JUDGMENT
    + 'Return {"grade": 0, 1, 2, 3, or null, "confidence": a number from 0 to 1 or null}.\n'
)
CHOICE = (
    JUDGMENT
    + 'Return {"choice": one of the displayed option letters, or "U" for unknown, "confidence": a number from 0 to 1 or null}.\n'
)


AUTHOR = """为每条现有听评问题写一个自然清楚的中文反问，语义是原命题的逻辑否定。
保持对象、比较基准、属性、时间范围、所有/仅/精确等限定不变，不修改原问题，不回答音频内容。
不是把用户要求改成另一种编辑目标。例如“是否变快”的否定是“是否未变快”，不是“是否变慢”；
“是否没有新增伪影”的否定是“是否新增了伪影”；“是否完整保留”可写成“是否存在遗漏或改变”。
复合命题的否定是至少一项不成立，不能要求所有条件同时相反。未知不是否定命题的证据。
输入只有公开问题，没有音频或任何评分。输出JSON {"questions":[{"key":"原key","reverse_question":"反向问题"}]}。
每个输入必须恰好输出一次，不输出其他字段。"""

REVERSE = (
    NUMERIC
    + """
This call asks the logical opposite of the original criterion. A higher grade means the
opposite proposition holds MORE strongly, according to the reversed anchors below.
The unchanged instruction describes the user's intended edit; it does not change this grading direction.
The unchanged scope defines the compared property and audio regions. Judge the current QUESTION.
Lack of evidence for the positive proposition alone is not proof of its negation: use null when uncertain.
"""
)


def opposite_question(question, semantic_call, output):
    """Author before judging, without sound, previous answers or model confidence."""
    key = canonical_evaluation_fingerprint(question)[:16]
    raw = semantic_call(AUTHOR, {"questions": [{"key": key, "question": question}]}, output)
    rows = raw.get("questions", [])
    if len(rows) != 1 or rows[0].get("key") != key:
        raise ValueError("opposite question must bind exactly the original condition")
    opposite = rows[0].get("reverse_question")
    if (
        not isinstance(opposite, str)
        or not opposite.strip()
        or opposite.strip() == question.strip()
    ):
        raise ValueError("opposite question is empty or repeats the positive question")
    write_record(
        output / "opposite.json",
        {
            "question": question,
            "reverse_question": opposite,
            "semantic_equivalence": "model_authored_not_certified",
            "answers_visible": False,
        },
    )
    return opposite


def judge_evidence(
    question,
    inventory,
    paths,
    observations,
    call,
    output,
    *,
    semantic_call,
    signal_facts=None,
    public_turns=None,
    levels=None,
):
    """Positive and reverse calls each return only an integer grade and confidence.

    Half-grades belong to aggregation, never to an individual model answer.
    Confidence is recorded, not used to change scores or bypass uncertainty.
    """
    if len(inventory) != len(paths) or any(
        sha256(p) != i["sha256"] for p, i in zip(paths, inventory)
    ):
        raise ValueError("audio identity changed before polarity review")
    output = Path(output)
    fields = {
        "audio_ref",
        "role",
        "attachment_number",
        "duration_sec",
        "parent_start_sec",
        "parent_end_sec",
        "boundary_local_sec",
        "is_parent_start",
        "is_parent_end",
    }
    base = {
        "audio_inventory": [{k: v for k, v in i.items() if k in fields} for i in inventory],
        "independent_observations": observations,
        "measured_facts": signal_facts or {},
        "instruction": public_turns or [],
        "scope": "Only supplied attachments and their declared regions.",
    }
    write_record(output / "shared_evidence.json", base)
    try:
        reverse = opposite_question(question, semantic_call, output / "opposite")
    except Exception as exc:
        failure = {"grade": None, "confidence": None, "error": str(exc), "status": "author_failed"}
        write_record(output / "result.json", failure)
        return {**failure, "finding": "Opposite question unavailable; condition is unjudged."}
    levels = levels or ORDINAL_LEVELS
    votes = []
    for name, prompt, q, anchors in [
        ("positive", NUMERIC, question, levels),
        ("reverse", REVERSE, reverse, reverse_levels(levels)),
    ]:
        payload = {**base, "question": q, "levels": anchors}
        write_record(output / name / "visible_input.json", payload)
        try:
            raw = call(prompt, payload, paths, output / name / "calls")
            answer = decode(raw)
        except Exception as exc:
            answer = {"grade": None, "confidence": None, "status": "failed", "error": str(exc)}
        write_record(output / name / "answer.json", answer)
        votes.append(answer)
    positive, negative = [v["grade"] for v in votes]
    grade = balanced_score(positive, negative)
    confidence = [v["confidence"] for v in votes if v.get("confidence") is not None]
    mapped = None if negative is None else 3 - negative
    result = {
        "protocol": PROTOCOL,
        "grade": grade,
        "confidence": mean(confidence) if grade is not None and len(confidence) == 2 else None,
        "confidence_status": "self_reported_uncalibrated",
        "positive_grade": positive,
        "reverse_grade": negative,
        "reverse_as_positive": mapped,
        "direction_gap": abs(positive - mapped) if grade is not None else None,
        "binary_disagreement": (positive >= 2) != (mapped >= 2) if grade is not None else None,
        "votes": votes,
        "question": question,
        "reverse_question": reverse,
        "evidence_fingerprint": canonical_evaluation_fingerprint(base),
        "artifact_hashes": [i["sha256"] for i in inventory],
    }
    # The guide receives observations and the criterion, not a fabricated explanation.
    result["finding"] = (
        f"Condition: {question}; positive={positive}, reverse={negative}, "
        f"combined={grade}. Direction disagreement={result['binary_disagreement']}. "
        + "Independent observations: "
        + json.dumps(observations, ensure_ascii=False)
    )[:1400]
    write_record(output / "result.json", result)
    return result
