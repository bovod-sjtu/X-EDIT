"""Non-authoritative Qwen scoring for natural-language Reasoner memos."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import requests
from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel


class ReasonerMemoScoringOptions(StrictModel):
    """Frozen evaluator-only settings for one memo scoring run."""

    results_path: str
    oracle_path: str
    output_dir: str
    base_urls: list[str] = Field(min_length=1)
    model: str = "Qwen3Omni-Instruct"
    timeout_sec: float = Field(default=180.0, gt=0)
    temperature: float = Field(default=0.7, ge=0)
    repetitions: int = Field(default=3, ge=1)
    required_positive_votes: int = Field(default=2, ge=1)
    max_attempts: int = Field(default=6, ge=1)
    api_key: str | None = None

    @model_validator(mode="after")
    def validate_vote_policy(self) -> ReasonerMemoScoringOptions:
        if self.required_positive_votes > self.repetitions:
            raise ValueError("required_positive_votes cannot exceed repetitions")
        if self.max_attempts < self.repetitions:
            raise ValueError("max_attempts cannot be smaller than repetitions")
        return self


class AtomicMemoRubric(StrictModel):
    rubric_id: str = Field(min_length=1, max_length=256)
    criterion: str = Field(min_length=1, max_length=2048)


class ReasonerMemoRubricRequest(StrictModel):
    schema_version: Literal["x-editor.reasoner-memo-rubric-request/v1"] = (
        "x-editor.reasoner-memo-rubric-request/v1"
    )
    request_id: str
    judge_protocol_version: Literal["x-editor.reasoner-memo-judge/v2"] = (
        "x-editor.reasoner-memo-judge/v2"
    )
    record_key: str
    turn_index: int = Field(ge=0)
    status: Literal["ready", "missing_memo"]
    user_request: str
    source_text: str | None = None
    prior_turn_contexts: list[dict[str, Any]] = Field(default_factory=list)
    reviewed_evidence: dict[str, Any] = Field(default_factory=dict)
    semantic_memo: str | None = None
    semantic_memo_fingerprint: str | None = None
    selected_phase: str | None = None
    latest_checkpoint_phase: str | None = None
    latest_checkpoint_status: str | None = None
    selected_earlier_memo_after_failure: bool = False
    rubrics: list[AtomicMemoRubric] = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class AtomicMemoRubricScore(StrictModel):
    rubric_id: str
    positive_votes: int = Field(ge=0)
    negative_votes: int = Field(ge=0)
    abstain_votes: int = Field(ge=0)
    passed: bool
    vote_reasons: list[str] = Field(default_factory=list)


class ReasonerMemoRubricResult(StrictModel):
    schema_version: Literal["x-editor.reasoner-memo-rubric-result/v1"] = (
        "x-editor.reasoner-memo-rubric-result/v1"
    )
    request_id: str
    record_key: str
    turn_index: int = Field(ge=0)
    status: Literal["pass", "fail", "incomplete", "missing_memo"]
    attempts: int = Field(ge=0)
    valid_repetitions: int = Field(ge=0)
    rubric_scores: list[AtomicMemoRubricScore] = Field(default_factory=list)
    attempt_details: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MemoRubricJudge(Protocol):
    model: str

    def answer(self, request: ReasonerMemoRubricRequest) -> str: ...


class QwenReasonerMemoJudge:
    """Text-only Qwen3-Omni judge separated from the execution path."""

    def __init__(
        self,
        *,
        base_urls: Sequence[str],
        model: str,
        api_key: str | None,
        timeout_sec: float,
        temperature: float,
    ) -> None:
        self.base_urls = [item.rstrip("/") for item in base_urls]
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self._index = 0

    def answer(self, request: ReasonerMemoRubricRequest) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _judge_user_prompt(request)},
            ],
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_urls[self._index % len(self.base_urls)]
        self._index += 1
        response = requests.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Qwen memo judge status {response.status_code}: {response.text[:800]}")
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise RuntimeError("Qwen memo judge response has no choices")
        raw = (choices[0].get("message") or {}).get("content")
        if isinstance(raw, list):
            raw = "".join(
                str(item.get("text") or "") for item in raw if isinstance(item, Mapping)
            )
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("Qwen memo judge response content is empty")
        return raw.strip()


_JUDGE_SYSTEM_PROMPT = """You are an evaluator-side semantic memo judge. You have no authority over planning, execution, verification, repair, or tool choice.

The frozen atomic criteria are evaluator-side reference facts. Judge whether the visible Reasoner memo expresses each fact; do not revise, dispute, or expand a criterion from the user request. A criterion is satisfied only when the memo states it clearly enough for a later Planner and does not contradict it. Semantically equivalent wording is acceptable, but different numbers, entities, directions, or source/target roles are not equivalent.

For a criterion containing an explicit list such as "one of: [24, 二十四]", compare the memo only with those listed aliases. Never claim that an unlisted value appears in the list. If the memo gives an unlisted value, omits the required fact, reverses source and target, or merely repeats the user's relative instruction without deriving the required result, answer no. Do not use facts found only in the original request as proof that the memo contains them.

Use abstain only when the frozen criterion itself cannot responsibly be judged from the supplied material. Missing, vague, or incorrect content in the memo is no, not abstain. Before answering yes, identify the supporting statement in the memo. Do not require tool names, operation ids, JSON fields, or implementation details. Preservation clauses are not edit objectives.

Return one JSON object with exactly this shape:
{"answers":[{"rubric_id":"...","answer":"yes|no|abstain","reason":"short evidence-based reason"}]}

Return exactly one answer for every supplied rubric id and no extra ids."""


def run_reasoner_memo_scoring(
    options: ReasonerMemoScoringOptions,
    *,
    judge: MemoRubricJudge | None = None,
) -> dict[str, Any]:
    """Score one memo-only result set without exposing scores to runtime."""

    output_dir = Path(options.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"memo scoring output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_to_score = _prepare_requests(options)
    _write_jsonl(
        output_dir / "rubric_requests.jsonl",
        [item.model_dump(mode="json") for item in requests_to_score],
    )
    active_judge = judge or QwenReasonerMemoJudge(
        base_urls=options.base_urls,
        model=options.model,
        api_key=options.api_key,
        timeout_sec=options.timeout_sec,
        temperature=options.temperature,
    )
    results = [
        _score_request(item, options=options, judge=active_judge)
        for item in requests_to_score
    ]
    _write_jsonl(
        output_dir / "rubric_results.jsonl",
        [item.model_dump(mode="json") for item in results],
    )
    report = _summary(options, requests_to_score, results, judge=active_judge)
    _write_json(output_dir / "summary.json", report)
    (output_dir / "SUMMARY.md").write_text(_summary_markdown(report), encoding="utf-8")
    return report


def _prepare_requests(options: ReasonerMemoScoringOptions) -> list[ReasonerMemoRubricRequest]:
    results_path = Path(options.results_path).expanduser().resolve()
    oracle_path = Path(options.oracle_path).expanduser().resolve()
    result_rows = _read_jsonl(results_path)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("visibility") != "scorer_only":
        raise ValueError("reasoner memo oracle must be scorer_only")
    expected_cases = oracle.get("cases")
    if not isinstance(expected_cases, dict):
        raise TypeError("reasoner memo oracle has no cases object")
    actual_keys = {
        str((row.get("audit_identity") or {}).get("record_key") or "") for row in result_rows
    }
    if actual_keys != set(expected_cases):
        raise ValueError(
            "memo scoring results and oracle cases differ: "
            f"missing={sorted(set(expected_cases) - actual_keys)}, "
            f"extra={sorted(actual_keys - set(expected_cases))}"
        )

    prepared: list[ReasonerMemoRubricRequest] = []
    for row in result_rows:
        record_key = str((row.get("audit_identity") or {})["record_key"])
        expected_turns = {
            int(item["turn_index"]): item for item in expected_cases[record_key]["turns"]
        }
        turn_entries = (row.get("result") or {}).get("turn_results") or []
        for turn_entry in turn_entries:
            turn_index = int(turn_entry["turn_index"])
            if turn_index not in expected_turns:
                raise ValueError(f"unexpected turn {turn_index} for {record_key}")
            turn_result = turn_entry.get("result") or {}
            record = _selected_memo_record(turn_result)
            snapshot = _selected_input_snapshot(turn_result, record)
            memo = (record or {}).get("semantic_memo") or {}
            memo_content = memo.get("content") if isinstance(memo, dict) else None
            history = [
                item
                for item in (turn_result.get("reasoner_shadow_history") or [])
                if isinstance(item, dict)
            ]
            latest = history[-1] if history else {}
            rubrics = _atomic_rubrics(expected_turns[turn_index])
            base_payload = {
                "judge_protocol_version": "x-editor.reasoner-memo-judge/v2",
                "record_key": record_key,
                "turn_index": turn_index,
                "status": "ready" if memo_content else "missing_memo",
                "user_request": str(snapshot.get("user_request") or ""),
                "source_text": snapshot.get("source_text"),
                "prior_turn_contexts": list(snapshot.get("prior_turn_contexts") or []),
                "reviewed_evidence": dict(snapshot.get("reviewed_evidence") or {}),
                "semantic_memo": memo_content,
                "semantic_memo_fingerprint": (record or {}).get(
                    "semantic_memo_fingerprint"
                ),
                "selected_phase": (record or {}).get("phase"),
                "latest_checkpoint_phase": latest.get("phase"),
                "latest_checkpoint_status": latest.get("status"),
                "selected_earlier_memo_after_failure": bool(
                    record is not None
                    and latest
                    and record is not latest
                    and latest.get("status") != "completed"
                ),
                "rubrics": [item.model_dump(mode="json") for item in rubrics],
            }
            prepared.append(
                ReasonerMemoRubricRequest(
                    request_id=f"memo-rubric:{record_key}:turn-{turn_index}",
                    request_fingerprint=_fingerprint(base_payload),
                    **base_payload,
                )
            )
    if len(prepared) != sum(len(item["turns"]) for item in expected_cases.values()):
        raise ValueError("memo scoring did not cover the complete frozen case/turn set")
    return prepared


def _selected_memo_record(turn_result: dict[str, Any]) -> dict[str, Any] | None:
    records = [
        item
        for item in (turn_result.get("reasoner_shadow_history") or [])
        if isinstance(item, dict)
    ]
    records = [item for item in records if (item.get("semantic_memo") or {}).get("content")]
    reviewed = [item for item in records if item.get("phase") == "evidence_review"]
    return (reviewed or records)[-1] if records else None


def _selected_input_snapshot(
    turn_result: dict[str, Any],
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_path = turn_result.get("reasoner_shadow_input_bundle_path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_file():
        raise FileNotFoundError(f"reasoner input bundle is unavailable: {raw_path}")
    rows = _read_jsonl(Path(raw_path))
    snapshots = [item.get("snapshot") for item in rows if item.get("record_type") == "snapshot"]
    request_id = (record or {}).get("request_id")
    matches = [item for item in snapshots if isinstance(item, dict) and item.get("request_id") == request_id]
    if matches:
        return matches[-1]
    return snapshots[-1] if snapshots and isinstance(snapshots[-1], dict) else {}


def _atomic_rubrics(expected_turn: dict[str, Any]) -> list[AtomicMemoRubric]:
    objectives = list(expected_turn.get("objectives") or [])
    rubrics = [
        AtomicMemoRubric(
            rubric_id="objective_count",
            criterion=(
                f"The memo identifies exactly {len(objectives)} user-requested audible edit "
                "objective(s), excluding preservation and internal processing steps."
            ),
        )
    ]
    for index, objective in enumerate(objectives, start=1):
        prefix = f"objective_{index}"
        objective_name = str(objective.get("objective_id") or prefix)
        rubrics.extend(
            [
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.domain",
                    criterion=(
                        f"For {objective_name}, the memo assigns the change to one of these "
                        f"semantic domains: {objective.get('domains', [])}."
                    ),
                ),
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.transformation",
                    criterion=(
                        f"For {objective_name}, the memo describes a transformation equivalent "
                        f"to one of: {objective.get('transformations', [])}."
                    ),
                ),
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.direction",
                    criterion=(
                        f"For {objective_name}, the memo's change direction is one of: "
                        f"{objective.get('directions', [])}."
                    ),
                ),
            ]
        )
        if objective.get("source_values"):
            rubrics.append(
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.source_value",
                    criterion=(
                        f"For {objective_name}, the memo identifies the source value as one of: "
                        f"{objective['source_values']}."
                    ),
                )
            )
        if objective.get("target_values"):
            rubrics.append(
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.target_value",
                    criterion=(
                        f"For {objective_name}, the memo derives or states the target value as "
                        f"one of: {objective['target_values']}."
                    ),
                )
            )
        if objective.get("target_descriptions"):
            rubrics.append(
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.target_reference",
                    criterion=(
                        f"For {objective_name}, the memo resolves the described target to one "
                        f"of these actual source expressions: {objective['target_descriptions']}."
                    ),
                )
            )
        if objective.get("depends_on_prior"):
            rubrics.append(
                AtomicMemoRubric(
                    rubric_id=f"{prefix}.dependency",
                    criterion=(
                        f"The memo states that {objective_name} consumes or follows the output "
                        "of the preceding requested edit."
                    ),
                )
            )
    return rubrics


def _judge_user_prompt(request: ReasonerMemoRubricRequest) -> str:
    payload = {
        "judge_protocol_version": request.judge_protocol_version,
        "original_user_request": request.user_request,
        "visible_source_text": request.source_text,
        "prior_turn_contexts": request.prior_turn_contexts,
        "reviewed_evidence": request.reviewed_evidence,
        "reasoner_memo": request.semantic_memo,
        "atomic_rubrics": [item.model_dump(mode="json") for item in request.rubrics],
        "instruction": (
            "Answer each rubric independently against the frozen criterion. Use yes only "
            "when the memo itself clearly supports it. Use no for omission, contradiction, "
            "wrong values, wrong roles, or an unresolved derivation. Use abstain only when "
            "the criterion itself is not judgeable, never because the memo is incomplete."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _score_request(
    request: ReasonerMemoRubricRequest,
    *,
    options: ReasonerMemoScoringOptions,
    judge: MemoRubricJudge,
) -> ReasonerMemoRubricResult:
    if request.status == "missing_memo":
        return ReasonerMemoRubricResult(
            request_id=request.request_id,
            record_key=request.record_key,
            turn_index=request.turn_index,
            status="missing_memo",
            attempts=0,
            valid_repetitions=0,
        )
    votes = {item.rubric_id: [] for item in request.rubrics}
    reasons = {item.rubric_id: [] for item in request.rubrics}
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    attempts = 0
    while attempts < options.max_attempts and len(next(iter(votes.values()))) < options.repetitions:
        attempts += 1
        raw = ""
        try:
            raw = judge.answer(request)
            parsed = _parse_answers(raw, rubric_ids=set(votes))
            for rubric_id, answer in parsed.items():
                votes[rubric_id].append(answer["answer"])
                reasons[rubric_id].append(answer["reason"][:1000])
            details.append(
                {
                    "attempt": attempts,
                    "answers": parsed,
                    "raw_response": raw[:6000],
                    "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - invalid judge attempts are evidence
            errors.append(f"{type(exc).__name__}: {str(exc)[:1000]}")
            details.append({"attempt": attempts, "answers": None, "raw_response": raw[:6000]})
    valid_repetitions = len(next(iter(votes.values())))
    scores = []
    for rubric in request.rubrics:
        rubric_votes = votes[rubric.rubric_id]
        positive = rubric_votes.count("yes")
        negative = rubric_votes.count("no")
        abstain = rubric_votes.count("abstain")
        scores.append(
            AtomicMemoRubricScore(
                rubric_id=rubric.rubric_id,
                positive_votes=positive,
                negative_votes=negative,
                abstain_votes=abstain,
                passed=(
                    valid_repetitions == options.repetitions
                    and positive >= options.required_positive_votes
                ),
                vote_reasons=reasons[rubric.rubric_id],
            )
        )
    complete = valid_repetitions == options.repetitions
    status = "pass" if complete and all(item.passed for item in scores) else "fail" if complete else "incomplete"
    return ReasonerMemoRubricResult(
        request_id=request.request_id,
        record_key=request.record_key,
        turn_index=request.turn_index,
        status=status,
        attempts=attempts,
        valid_repetitions=valid_repetitions,
        rubric_scores=scores,
        attempt_details=details,
        errors=errors,
    )


def _parse_answers(raw: str, *, rubric_ids: set[str]) -> dict[str, dict[str, str]]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response has no JSON object")
    payload = json.loads(raw[start : end + 1])
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, list):
        raise TypeError("judge response has no answers list")
    parsed: dict[str, dict[str, str]] = {}
    for item in answers:
        if not isinstance(item, dict):
            raise TypeError("judge answer is not an object")
        rubric_id = str(item.get("rubric_id") or "")
        answer = str(item.get("answer") or "").lower()
        if rubric_id in parsed or rubric_id not in rubric_ids:
            raise ValueError(f"judge returned duplicate or unknown rubric id: {rubric_id}")
        if answer not in {"yes", "no", "abstain"}:
            raise ValueError(f"invalid judge answer: {answer}")
        parsed[rubric_id] = {"answer": answer, "reason": str(item.get("reason") or "")}
    if set(parsed) != rubric_ids:
        raise ValueError(f"judge rubric coverage mismatch: {sorted(set(parsed) ^ rubric_ids)}")
    return parsed


def _summary(
    options: ReasonerMemoScoringOptions,
    requests_to_score: list[ReasonerMemoRubricRequest],
    results: list[ReasonerMemoRubricResult],
    *,
    judge: MemoRubricJudge,
) -> dict[str, Any]:
    statuses = Counter(item.status for item in results)
    all_scores = [score for item in results for score in item.rubric_scores]
    case_statuses: dict[str, list[str]] = {}
    for item in results:
        case_statuses.setdefault(item.record_key, []).append(item.status)
    passed_cases = sum(all(status == "pass" for status in values) for values in case_statuses.values())
    return {
        "schema_version": "x-editor.reasoner-memo-scoring-summary/v1",
        "results_path": str(Path(options.results_path).expanduser().resolve()),
        "oracle_path": str(Path(options.oracle_path).expanduser().resolve()),
        "runtime_authority": False,
        "evaluator_feedback_enabled": False,
        "model": judge.model,
        "judge_protocol_version": "x-editor.reasoner-memo-judge/v2",
        "vote_policy": {
            "repetitions": options.repetitions,
            "required_positive_votes": options.required_positive_votes,
            "abstention_counts_as_positive": False,
        },
        "case_count": len(case_statuses),
        "turn_count": len(results),
        "rubric_count": len(all_scores),
        "passed_case_count": passed_cases,
        "passed_turn_count": statuses["pass"],
        "passed_rubric_count": sum(item.passed for item in all_scores),
        "turn_status_counts": dict(statuses),
        "case_pass_rate": passed_cases / len(case_statuses) if case_statuses else None,
        "turn_pass_rate": statuses["pass"] / len(results) if results else None,
        "rubric_pass_rate": (
            sum(item.passed for item in all_scores) / len(all_scores) if all_scores else None
        ),
        "latest_checkpoint_failure_turn_count": sum(
            item.latest_checkpoint_status not in {None, "completed"}
            for item in requests_to_score
        ),
        "scored_earlier_memo_after_failure_turn_count": sum(
            item.selected_earlier_memo_after_failure for item in requests_to_score
        ),
        "request_set_fingerprint": _fingerprint(
            [item.request_fingerprint for item in requests_to_score]
        ),
    }


def _summary_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Reasoner memo 的 Qwen 语义评分",
            "",
            "这份评分只判断 Reasoner 的人话 memo 是否理解了任务，不参与执行、修复或工具选择。",
            "",
            f"- cases: `{report['passed_case_count']}/{report['case_count']}`",
            f"- turns: `{report['passed_turn_count']}/{report['turn_count']}`",
            f"- atomic rubrics: `{report['passed_rubric_count']}/{report['rubric_count']}`",
            (
                f"- 规则：每问 `{report['vote_policy']['repetitions']}` 票，至少 "
                f"`{report['vote_policy']['required_positive_votes']}` 票认可才通过。"
            ),
            "",
        ]
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AtomicMemoRubric",
    "AtomicMemoRubricScore",
    "MemoRubricJudge",
    "QwenReasonerMemoJudge",
    "ReasonerMemoRubricRequest",
    "ReasonerMemoRubricResult",
    "ReasonerMemoScoringOptions",
    "run_reasoner_memo_scoring",
]
