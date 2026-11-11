"""Production review support extracted without changing function bodies."""

import json


from x_editor.evaluation.iterative_runtime import write_record


def invoke_validated_review(*, model, invoke, directory, prompt, args, validate=None):
    revisions = getattr(args, "evaluation_contract_revisions", 0)
    if type(revisions) is not int or revisions not in (0, 1):
        raise ValueError("evaluation contract revisions must be zero or one")
    original = json.loads(prompt["user_prompt"])
    current = prompt
    for index in range(revisions + 1):
        destination = directory if index == 0 else directory / f"contract_revision_{index}"
        raw = invoke(destination, current, args, author=True)
        write_record(destination / "review_response.json", raw)
        try:
            answer = model.model_validate(raw)
            if validate is not None:
                validate(answer)
            return answer
        except ValueError as exc:
            write_record(destination / "contract_rejection.json", {"error": str(exc), "response": raw})
            if index == revisions:
                raise
            current = {
                "system_prompt": prompt["system_prompt"] +
                    "\n上份评审回答不符合输出合同。仅依据原始材料重新作答；保持真实否定与未知，"
                    "不能为了通过格式校验改为认可。解释简洁，不复述完整输入。",
                "user_prompt": json.dumps({**original, "rejected_review": raw,
                                            "response_contract_error": str(exc)}, ensure_ascii=False),
            }
