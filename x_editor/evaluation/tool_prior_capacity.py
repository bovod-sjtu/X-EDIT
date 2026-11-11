"""MMAE-style Qwen3-Omni scoring for frozen tool-comparison releases."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import requests
import soundfile as sf
from pydantic import Field, model_validator

from x_editor.core.schemas import StrictModel
from x_editor.evaluation.roles import build_evaluator_role_contract

SCHEMA_VERSION = "x-editor.tool-prior-capacity-scoring/v1"
RubricDimension = Literal[
    "edited_instruction_following",
    "edited_region_quality",
    "unedited_region_consistency",
]
_DIMENSIONS: tuple[RubricDimension, ...] = (
    "edited_instruction_following",
    "edited_region_quality",
    "unedited_region_consistency",
)
_MMAE_SYSTEM_PROMPT = (
    "You are an audio analysis assistant. Answer strictly from the factual content "
    "of the attached audio excerpts. Listen carefully before answering. Do not infer "
    "facts from file names, candidate identities, or methods. If evidence is unclear, "
    "choose the uncertainty option. Return only JSON with keys reason and choice; "
    "choice must be one uppercase option letter."
)
_MMAE_USER_PREFIX = (
    "Based on the objective content of the uploaded audio, answer this multiple-choice "
    "question. Analyze the audio first, then choose exactly one option."
)


class ToolPriorScoringOptions(StrictModel):
    release_index: str = Field(min_length=1)
    release_audit: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    mode: Literal["dry-run", "score"] = "dry-run"
    base_urls: list[str] = Field(default_factory=list)
    model: str = "Qwen3Omni-Instruct"
    api_key: str | None = None
    timeout_sec: float = Field(default=300.0, gt=0.0)
    valid_repetitions: int = Field(default=3, ge=3, le=9)
    required_correct: int = Field(default=2, ge=1)
    max_attempts: int = Field(default=10, ge=3, le=30)
    limit_cases_per_group: int | None = Field(default=None, ge=1)
    groups: list[str] = Field(default_factory=list)
    resume: bool = False
    target_margin_sec: float = Field(default=0.65, ge=0.1, le=2.0)
    preservation_window_sec: float = Field(default=1.0, ge=0.4, le=3.0)
    preservation_guard_sec: float = Field(default=0.3, ge=0.05, le=1.0)

    @model_validator(mode="after")
    def validate_vote_policy(self) -> ToolPriorScoringOptions:
        if self.required_correct > self.valid_repetitions:
            raise ValueError("required_correct cannot exceed valid_repetitions")
        if self.max_attempts < self.valid_repetitions:
            raise ValueError("max_attempts cannot be smaller than valid_repetitions")
        if self.mode == "score" and not self.base_urls:
            raise ValueError("score mode requires at least one Qwen3-Omni base URL")
        if len(self.groups) != len(set(self.groups)):
            raise ValueError("groups must be unique")
        return self


class CropEvidence(StrictModel):
    crop_id: str
    label: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    source_duration_sec: float = Field(gt=0.0)


class ToolPriorRubricRequest(StrictModel):
    schema_version: Literal["x-editor.tool-prior-rubric-request/v1"] = (
        "x-editor.tool-prior-rubric-request/v1"
    )
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    template_case_id: str
    comparison_group: str
    candidate_id: str
    route_id: str
    dimension: RubricDimension
    category: Literal["Instruction Following", "Consistency"]
    public_instruction: str
    target_label: str
    question: str
    right_choice: str
    wrong_choices: list[str] = Field(min_length=2, max_length=8)
    fixed_last_choice: str
    attachments: list[CropEvidence] = Field(min_length=1, max_length=6)


class ToolPriorRubricResult(StrictModel):
    schema_version: Literal["x-editor.tool-prior-rubric-result/v1"] = (
        "x-editor.tool-prior-rubric-result/v1"
    )
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str
    comparison_group: str
    candidate_id: str
    route_id: str
    dimension: RubricDimension
    status: Literal["pass", "fail", "incomplete", "dry_run"]
    valid_votes: list[int] = Field(default_factory=list)
    correct_votes: int = Field(ge=0)
    score: Literal[0, 1] | None = None
    attempts: int = Field(ge=0)
    attempt_details: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RubricJudge(Protocol):
    model: str

    def answer(
        self,
        *,
        question: str,
        choices: list[str],
        audio_paths: list[str],
        audio_labels: list[str],
    ) -> str: ...


class Qwen3OmniMMAEJudge:
    """OpenAI-compatible client using the official MMAE message shape."""

    def __init__(
        self,
        *,
        base_urls: Sequence[str],
        model: str,
        api_key: str | None,
        timeout_sec: float,
    ) -> None:
        self.base_urls = [item.rstrip("/") for item in base_urls]
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self._index = 0

    def answer(
        self,
        *,
        question: str,
        choices: list[str],
        audio_paths: list[str],
        audio_labels: list[str],
    ) -> str:
        content: list[dict[str, Any]] = []
        for path, label in zip(audio_paths, audio_labels, strict=True):
            content.append({"type": "text", "text": f"{label}: "})
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": _audio_data_uri(Path(path))},
                }
            )
        choices_text = "\n".join(
            f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"{_MMAE_USER_PREFIX}\n\nQuestion:\n{question}\n\nChoices:\n{choices_text}"
                ),
            }
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _MMAE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.7,
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
            raise RuntimeError(f"Qwen3-Omni status {response.status_code}: {response.text[:800]}")
        body = response.json()
        choices_payload = body.get("choices") if isinstance(body, dict) else None
        if not choices_payload:
            raise RuntimeError("Qwen3-Omni response has no choices")
        message = choices_payload[0].get("message") or {}
        raw = message.get("content")
        if isinstance(raw, list):
            raw = "".join(str(item.get("text") or "") for item in raw if isinstance(item, Mapping))
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("Qwen3-Omni response content is empty")
        return raw.strip()


@dataclass(frozen=True)
class _ReleaseEvidence:
    index_path: Path
    root: Path
    release_sha256: str
    cases: tuple[dict[str, Any], ...]
    method_identity_by_route: dict[tuple[str, str], str]


def run_tool_prior_capacity_scoring(
    options: ToolPriorScoringOptions,
    *,
    judge: RubricJudge | None = None,
) -> dict[str, Any]:
    output_root = Path(options.output_dir).expanduser().resolve()
    evidence = _load_release(options)
    if output_root.exists() and not options.resume:
        raise FileExistsError(f"scoring output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    requests_path = output_root / "rubric_requests.jsonl"
    if options.resume and requests_path.is_file():
        rubric_requests = [
            ToolPriorRubricRequest.model_validate(item) for item in _read_jsonl(requests_path)
        ]
        _verify_resumed_requests(rubric_requests, evidence=evidence, options=options)
    else:
        rubric_requests = _prepare_requests(
            evidence,
            options=options,
            crop_root=output_root / "crops",
        )
        _write_jsonl(
            requests_path,
            [item.model_dump(mode="json") for item in rubric_requests],
        )

    manifest = _score_manifest(
        evidence=evidence,
        options=options,
        requests=rubric_requests,
        judge=judge,
    )
    _write_json(output_root / "score_manifest.json", manifest)
    if options.mode == "score" and judge is None:
        judge = Qwen3OmniMMAEJudge(
            base_urls=options.base_urls,
            model=options.model,
            api_key=options.api_key,
            timeout_sec=options.timeout_sec,
        )

    results_path = output_root / "results.jsonl"
    existing = {
        str(item.get("request_id")): ToolPriorRubricResult.model_validate(item)
        for item in (_read_jsonl(results_path) if options.resume and results_path.is_file() else [])
    }
    ordered_results: list[ToolPriorRubricResult] = []
    for request in rubric_requests:
        prior = existing.get(request.request_id)
        if prior is not None and prior.status in {"pass", "fail", "dry_run"}:
            ordered_results.append(prior)
            continue
        if options.mode == "dry-run":
            result = ToolPriorRubricResult(
                request_id=request.request_id,
                case_id=request.case_id,
                comparison_group=request.comparison_group,
                candidate_id=request.candidate_id,
                route_id=request.route_id,
                dimension=request.dimension,
                status="dry_run",
                correct_votes=0,
                attempts=0,
            )
        else:
            assert judge is not None
            result = _score_rubric(request, options=options, judge=judge)
        existing[request.request_id] = result
        ordered_results.append(result)
        _append_jsonl(results_path, result.model_dump(mode="json"))

    _write_jsonl(
        results_path,
        [item.model_dump(mode="json") for item in ordered_results],
    )
    report = _capacity_report(
        evidence=evidence,
        requests=rubric_requests,
        results=ordered_results,
        options=options,
    )
    _write_json(output_root / "capacity_report.json", report)
    _write_json(output_root / "per_candidate.json", {"candidates": report["candidates"]})
    (output_root / "summary.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
    )
    return report


def _load_release(options: ToolPriorScoringOptions) -> _ReleaseEvidence:
    index_path = Path(options.release_index).expanduser().resolve()
    audit_path = Path(options.release_audit).expanduser().resolve()
    index = _read_json(index_path)
    audit = _read_json(audit_path)
    if index.get("schema_version") != "x-editor.tool-route-comparison-release/v1":
        raise ValueError("unsupported tool-comparison release schema")
    if audit.get("schema_version") != "x-editor.tool-route-comparison-release-audit/v3":
        raise ValueError("tool-prior scoring requires release audit v3")
    required_checks = (
        "all_completed",
        "all_evaluator_free",
        "all_operations_preserved",
        "all_planner_parameters_preserved",
        "all_reported_methods_preserved",
        "no_recovered_flow",
        "all_release_hashes_valid",
        "all_candidate_hashes_unique_within_case",
        "all_required_candidate_methods_distinct",
    )
    failed = [key for key in required_checks if audit.get(key) is not True]
    if failed:
        raise ValueError(f"tool-comparison release audit did not pass: {failed}")
    actual_sha = _sha256_file(index_path)
    if audit.get("release_sha256") != actual_sha:
        raise ValueError("release index hash does not match release audit")
    cases = index.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("tool-comparison release has no cases")
    rows = audit.get("rows")
    if not isinstance(rows, list):
        raise TypeError("tool-comparison release audit has no rows")
    method_identity = {
        (str(row.get("case_id")), str(row.get("route_id"))): str(row.get("method_identity") or "")
        for row in rows
        if isinstance(row, Mapping)
    }
    selected = _select_cases(
        cases,
        groups=set(options.groups),
        limit_per_group=options.limit_cases_per_group,
    )
    return _ReleaseEvidence(
        index_path=index_path,
        root=index_path.parent,
        release_sha256=actual_sha,
        cases=tuple(selected),
        method_identity_by_route=method_identity,
    )


def _select_cases(
    cases: list[Any],
    *,
    groups: set[str],
    limit_per_group: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw in cases:
        if not isinstance(raw, dict):
            raise TypeError("release case is not an object")
        group = str(raw.get("comparison_group") or "")
        if groups and group not in groups:
            continue
        if limit_per_group is not None and counts[group] >= limit_per_group:
            continue
        selected.append(raw)
        counts[group] += 1
    if not selected:
        raise ValueError("tool-prior scoring selected no cases")
    if groups - set(counts):
        raise ValueError(f"unknown or empty comparison groups: {sorted(groups - set(counts))}")
    return selected


def _prepare_requests(
    evidence: _ReleaseEvidence,
    *,
    options: ToolPriorScoringOptions,
    crop_root: Path,
) -> list[ToolPriorRubricRequest]:
    requests_out: list[ToolPriorRubricRequest] = []
    for case in evidence.cases:
        source_row = case.get("source_audio")
        if not isinstance(source_row, Mapping):
            raise TypeError(f"case {case.get('case_id')} has no source audio")
        source_path = _verified_release_audio(
            evidence.root,
            path_value=source_row.get("audio_path"),
            expected_sha=source_row.get("audio_sha256"),
        )
        for candidate in case.get("candidates") or []:
            if not isinstance(candidate, Mapping) or candidate.get("status") != "success":
                raise ValueError(f"case {case.get('case_id')} has unavailable candidate")
            output_path = _verified_release_audio(
                evidence.root,
                path_value=candidate.get("audio_path"),
                expected_sha=candidate.get("audio_sha256"),
            )
            crop_sets = _candidate_crop_sets(
                case,
                candidate,
                source_path=source_path,
                output_path=output_path,
                crop_root=crop_root,
                options=options,
            )
            for dimension in _DIMENSIONS:
                body = _rubric_body(case, dimension=dimension)
                attachments = crop_sets[dimension]
                request_body = {
                    "release_sha256": evidence.release_sha256,
                    "case_id": str(case["case_id"]),
                    "template_case_id": str(case.get("template_case_id") or ""),
                    "comparison_group": str(case.get("comparison_group") or ""),
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "route_id": str(candidate.get("route_id") or ""),
                    "dimension": dimension,
                    "category": body["category"],
                    "public_instruction": str(case.get("instruction") or ""),
                    "target_label": str(
                        (case.get("review_context") or {}).get("target_label") or "target"
                    ),
                    "question": body["question"],
                    "right_choice": body["right_choice"],
                    "wrong_choices": body["wrong_choices"],
                    "fixed_last_choice": body["fixed_last_choice"],
                    "attachments": [item.model_dump(mode="json") for item in attachments],
                }
                requests_out.append(
                    ToolPriorRubricRequest(
                        **request_body,
                        request_id=_fingerprint(request_body),
                    )
                )
    return requests_out


def _candidate_crop_sets(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    source_path: Path,
    output_path: Path,
    crop_root: Path,
    options: ToolPriorScoringOptions,
) -> dict[RubricDimension, list[CropEvidence]]:
    case_id = str(case["case_id"])
    candidate_id = str(candidate["candidate_id"])
    destination = crop_root / case_id / candidate_id
    source_duration = _audio_duration(source_path)
    output_duration = _audio_duration(output_path)
    review_context = case.get("review_context")
    review_context = review_context if isinstance(review_context, Mapping) else {}
    localization = candidate.get("localization")
    localization = localization if isinstance(localization, Mapping) else {}
    edit_scope = str(review_context.get("edit_scope") or "global")
    if edit_scope == "global":
        source_global = _crop_evidence(
            source_path,
            start=0.0,
            end=source_duration,
            destination=destination / "source_global.wav",
            crop_id="source_global",
            label="<audio source full clip>",
        )
        output_global = _crop_evidence(
            output_path,
            start=0.0,
            end=output_duration,
            destination=destination / "edited_global.wav",
            crop_id="edited_global",
            label="<audio edited full clip>",
        )
        return {
            "edited_instruction_following": [source_global, output_global],
            "edited_region_quality": [output_global],
            "unedited_region_consistency": [source_global, output_global],
        }

    source_spans = _valid_spans(review_context.get("source_spans"))
    if not source_spans:
        raise ValueError(f"local case {case_id} has no grounded source spans")
    source_start = min(item[0] for item in source_spans)
    source_end = max(item[1] for item in source_spans)
    output_spans = [
        *_valid_spans(localization.get("output_spans")),
        *_valid_spans(localization.get("affected_output_spans")),
    ]
    output_start = min((item[0] for item in output_spans), default=source_start)
    output_end = max((item[1] for item in output_spans), default=source_end)
    source_target = _crop_evidence(
        source_path,
        start=max(0.0, source_start - options.target_margin_sec),
        end=min(source_duration, source_end + options.target_margin_sec),
        destination=destination / "source_target.wav",
        crop_id="source_target",
        label="<audio source target excerpt>",
    )
    output_target = _crop_evidence(
        output_path,
        start=max(0.0, output_start - options.target_margin_sec),
        end=min(output_duration, output_end + options.target_margin_sec),
        destination=destination / "edited_target.wav",
        crop_id="edited_target",
        label="<audio edited target excerpt>",
    )
    preservation = _preservation_crops(
        source_path,
        output_path,
        source_start=source_start,
        source_end=source_end,
        source_duration=source_duration,
        output_duration=output_duration,
        destination=destination,
        options=options,
    )
    return {
        "edited_instruction_following": [source_target, output_target],
        "edited_region_quality": [output_target],
        "unedited_region_consistency": preservation,
    }


def _preservation_crops(
    source_path: Path,
    output_path: Path,
    *,
    source_start: float,
    source_end: float,
    source_duration: float,
    output_duration: float,
    destination: Path,
    options: ToolPriorScoringOptions,
) -> list[CropEvidence]:
    rows: list[CropEvidence] = []
    before_end = max(0.0, source_start - options.preservation_guard_sec)
    before_start = max(0.0, before_end - options.preservation_window_sec)
    if before_end - before_start >= 0.25:
        rows.extend(
            [
                _crop_evidence(
                    source_path,
                    start=before_start,
                    end=before_end,
                    destination=destination / "source_preserved_before.wav",
                    crop_id="source_preserved_before",
                    label="<audio source preserved-before excerpt>",
                ),
                _crop_evidence(
                    output_path,
                    start=min(before_start, output_duration - 0.01),
                    end=min(before_end, output_duration),
                    destination=destination / "edited_preserved_before.wav",
                    crop_id="edited_preserved_before",
                    label="<audio edited preserved-before excerpt>",
                ),
            ]
        )
    source_after_start = min(
        source_duration,
        source_end + options.preservation_guard_sec,
    )
    source_after_end = min(
        source_duration,
        source_after_start + options.preservation_window_sec,
    )
    timeline_delta = output_duration - source_duration
    output_after_start = min(
        output_duration,
        max(0.0, source_after_start + timeline_delta),
    )
    output_after_end = min(
        output_duration,
        output_after_start + (source_after_end - source_after_start),
    )
    if source_after_end - source_after_start >= 0.25 and output_after_end > output_after_start:
        rows.extend(
            [
                _crop_evidence(
                    source_path,
                    start=source_after_start,
                    end=source_after_end,
                    destination=destination / "source_preserved_after.wav",
                    crop_id="source_preserved_after",
                    label="<audio source preserved-after excerpt>",
                ),
                _crop_evidence(
                    output_path,
                    start=output_after_start,
                    end=output_after_end,
                    destination=destination / "edited_preserved_after.wav",
                    crop_id="edited_preserved_after",
                    label="<audio edited preserved-after excerpt>",
                ),
            ]
        )
    if rows:
        return rows
    return [
        _crop_evidence(
            source_path,
            start=0.0,
            end=source_duration,
            destination=destination / "source_preservation_fallback.wav",
            crop_id="source_preservation_fallback",
            label="<audio source preservation excerpt>",
        ),
        _crop_evidence(
            output_path,
            start=0.0,
            end=output_duration,
            destination=destination / "edited_preservation_fallback.wav",
            crop_id="edited_preservation_fallback",
            label="<audio edited preservation excerpt>",
        ),
    ]


def _rubric_body(
    case: Mapping[str, Any],
    *,
    dimension: RubricDimension,
) -> dict[str, Any]:
    instruction = str(case.get("instruction") or "").strip()
    review = case.get("review_context")
    target = (
        str(review.get("target_label") or "the target")
        if isinstance(review, Mapping)
        else "the target"
    )
    if dimension == "edited_instruction_following":
        question = (
            "Compared with the attached source excerpt, does the edited excerpt "
            f"clearly carry out this requested change at {target!r}: {instruction}"
        )
        category = "Instruction Following"
        right = "Yes, the requested edit is clearly present and correct."
        wrong = [
            "No, the requested edit is absent, opposite, or applied incorrectly.",
            "The edited excerpt changes something else instead of the requested target.",
        ]
    elif dimension == "edited_region_quality":
        question = (
            "Is the attached edited target excerpt intelligible and natural, without an "
            "obvious hard cut, seam, clipping, robotic distortion, or unintended noise?"
        )
        category = "Consistency"
        right = "Yes, the edited target is intelligible and has no obvious quality defect."
        wrong = [
            "No, the edited target contains an obvious artifact or unnatural boundary.",
            "No, the target speech is materially unclear or unintelligible.",
        ]
    else:
        question = (
            "Across the attached source/edited preservation excerpt pairs, does the edited "
            "audio preserve non-target wording, speaker identity, and background or vocal "
            "qualities that the instruction did not ask to change?"
        )
        category = "Consistency"
        right = "Yes, the non-target audio is consistently preserved."
        wrong = [
            "No, non-target speech, speaker identity, timing, or acoustics changed materially.",
            "No, one or more preservation excerpts contain missing or altered content.",
        ]
    return {
        "category": category,
        "question": question,
        "right_choice": right,
        "wrong_choices": wrong,
        "fixed_last_choice": "None of the above.",
    }


def _score_rubric(
    request: ToolPriorRubricRequest,
    *,
    options: ToolPriorScoringOptions,
    judge: RubricJudge,
) -> ToolPriorRubricResult:
    _verify_crop_evidence(request.attachments)
    votes: list[int] = []
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    attempts = 0
    while len(votes) < options.valid_repetitions and attempts < options.max_attempts:
        attempts += 1
        choices, right_letter = _shuffled_choices(request, attempt=attempts)
        valid_letters = {chr(ord("A") + index) for index in range(len(choices))}
        raw = ""
        try:
            raw = judge.answer(
                question=request.question,
                choices=choices,
                audio_paths=[item.path for item in request.attachments],
                audio_labels=[item.label for item in request.attachments],
            )
            parsed = _parse_response(raw)
            choice = str(parsed.get("choice") or "").strip().upper()
            if len(choice) != 1 or choice not in valid_letters:
                raise ValueError(f"invalid_choice:{choice!r}")
            vote = int(choice == right_letter)
            votes.append(vote)
            details.append(
                {
                    "attempt": attempts,
                    "choices": [
                        f"{chr(ord('A') + index)}. {value}" for index, value in enumerate(choices)
                    ],
                    "right_letter": right_letter,
                    "model_choice": choice,
                    "vote": vote,
                    "reason": str(parsed.get("reason") or "")[:2000],
                    "raw_response": raw[:4000],
                    "raw_response_sha256": _sha256_bytes(raw.encode("utf-8")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - invalid/provider attempts are evidence
            errors.append(f"{type(exc).__name__}: {str(exc)[:1000]}")
            details.append(
                {
                    "attempt": attempts,
                    "choices": [
                        f"{chr(ord('A') + index)}. {value}" for index, value in enumerate(choices)
                    ],
                    "right_letter": right_letter,
                    "model_choice": None,
                    "vote": None,
                    "raw_response": raw[:4000],
                }
            )
        _verify_crop_evidence(request.attachments)
    complete = len(votes) == options.valid_repetitions
    correct = sum(votes)
    score = int(correct >= options.required_correct) if complete else None
    return ToolPriorRubricResult(
        request_id=request.request_id,
        case_id=request.case_id,
        comparison_group=request.comparison_group,
        candidate_id=request.candidate_id,
        route_id=request.route_id,
        dimension=request.dimension,
        status=("pass" if score == 1 else "fail" if score == 0 else "incomplete"),
        valid_votes=votes,
        correct_votes=correct,
        score=score,
        attempts=attempts,
        attempt_details=details,
        errors=errors,
    )


def _shuffled_choices(
    request: ToolPriorRubricRequest,
    *,
    attempt: int,
) -> tuple[list[str], str]:
    shuffled = [request.right_choice, *request.wrong_choices]
    random.Random(f"{request.request_id}:{attempt}").shuffle(shuffled)
    shuffled.append(request.fixed_last_choice)
    index = shuffled.index(request.right_choice)
    return shuffled, chr(ord("A") + index)


def _parse_response(raw: str) -> dict[str, Any]:
    candidates = [raw.strip()]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    letter = raw.strip().upper()
    if len(letter) == 1 and letter.isalpha():
        return {"choice": letter, "reason": ""}
    raise ValueError("judge response has no JSON object or bare option letter")


def _capacity_report(
    *,
    evidence: _ReleaseEvidence,
    requests: list[ToolPriorRubricRequest],
    results: list[ToolPriorRubricResult],
    options: ToolPriorScoringOptions,
) -> dict[str, Any]:
    result_by_request = {item.request_id: item for item in results}
    candidate_rows: list[dict[str, Any]] = []
    grouped_requests: dict[tuple[str, str], list[ToolPriorRubricRequest]] = defaultdict(list)
    for request in requests:
        grouped_requests[(request.case_id, request.candidate_id)].append(request)
    for (case_id, candidate_id), rows in grouped_requests.items():
        result_rows = [result_by_request[item.request_id] for item in rows]
        scores = {item.dimension: item.score for item in result_rows}
        complete = all(value is not None for value in scores.values())
        utility = (
            sum(int(value) for value in scores.values() if value is not None) / len(_DIMENSIONS)
            if complete
            else None
        )
        first = rows[0]
        candidate_rows.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "comparison_group": first.comparison_group,
                "route_id": first.route_id,
                "method_identity": evidence.method_identity_by_route.get(
                    (case_id, first.route_id),
                    "",
                ),
                "rubric_scores": scores,
                "rubric_statuses": {item.dimension: item.status for item in result_rows},
                "evaluation_complete": complete,
                "utility": utility,
                "strict_all_rubrics_pass": complete and utility == 1.0,
            }
        )
    route_rows: list[dict[str, Any]] = []
    grouped_candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped_candidates[(row["comparison_group"], row["route_id"])].append(row)
    for (group, route_id), rows in sorted(grouped_candidates.items()):
        complete_rows = [row for row in rows if row["evaluation_complete"]]
        utilities = [float(row["utility"]) for row in complete_rows]
        dimensions = {
            dimension: _mean([int(row["rubric_scores"][dimension]) for row in complete_rows])
            for dimension in _DIMENSIONS
        }
        itt_utilities = [float(row["utility"] or 0.0) for row in rows]
        route_rows.append(
            {
                "comparison_group": group,
                "route_id": route_id,
                "method_identity": rows[0]["method_identity"],
                "frozen_sample_count": len(rows),
                "scored_sample_count": len(complete_rows),
                "evaluation_coverage": len(complete_rows) / len(rows),
                "dimension_pass_rates": dimensions,
                "empirical_flow_capacity_scored": _mean(utilities),
                "empirical_flow_capacity_itt": _mean(itt_utilities),
                "bootstrap_95pct_ci": _bootstrap_ci(itt_utilities),
                "strict_success_rate": _mean([int(row["strict_all_rubrics_pass"]) for row in rows]),
            }
        )
    rankings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in route_rows:
        rankings[row["comparison_group"]].append(
            {
                "route_id": row["route_id"],
                "empirical_flow_capacity_itt": row["empirical_flow_capacity_itt"],
                "strict_success_rate": row["strict_success_rate"],
            }
        )
    rankings = {
        group: sorted(
            rows,
            key=lambda item: (
                -float(item["empirical_flow_capacity_itt"] or 0.0),
                -float(item["strict_success_rate"] or 0.0),
                str(item["route_id"]),
            ),
        )
        for group, rows in rankings.items()
    }
    pairwise_comparisons = _paired_top_route_comparisons(
        candidate_rows=candidate_rows,
        rankings=rankings,
    )
    statuses = Counter(item.status for item in results)
    evaluator_contract = build_evaluator_role_contract("offline_prior_estimator").model_dump(
        mode="json"
    )
    return {
        "schema_version": "x-editor.tool-prior-capacity-report/v1",
        "experiment_kind": "offline_evaluator_tool_prior_calibration",
        "evaluator_contract": evaluator_contract,
        "prior_snapshot_status": "development_only_requires_explicit_freeze",
        "claim_boundary": (
            "Empirical prior over the frozen v4 development distribution and judge "
            "protocol; not a universal tool capacity, experiment-loop outcome, or final "
            "human preference."
        ),
        "release_sha256": evidence.release_sha256,
        "model": options.model if options.mode == "score" else None,
        "vote_policy": {
            "valid_repetitions": options.valid_repetitions,
            "required_correct": options.required_correct,
            "max_attempts": options.max_attempts,
        },
        "case_count": len(evidence.cases),
        "candidate_count": len(candidate_rows),
        "rubric_count": len(results),
        "provider_call_attempts": sum(item.attempts for item in results),
        "rubric_status_counts": dict(sorted(statuses.items())),
        "evaluation_coverage": (
            sum(item.status in {"pass", "fail"} for item in results) / len(results)
            if results
            else 0.0
        ),
        "rubric_dimensions": list(_DIMENSIONS),
        "candidates": sorted(
            candidate_rows,
            key=lambda item: (item["comparison_group"], item["case_id"], item["route_id"]),
        ),
        "routes": route_rows,
        "rankings": rankings,
        "pairwise_comparisons": pairwise_comparisons,
    }


def _summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-Omni 工具先验流容量",
        "",
        "这是冻结开发集上的探索性工具先验，不是通用工具能力上限，也不代替人工听评。",
        "",
        "## 实验怎么判",
        "",
        (
            "每个候选音频分三问：是否完成指令、编辑位置是否自然、未编辑内容是否保持。"
            " 每一问由同一 Qwen3-Omni 独立采样 3 次，至少 2 次判为满足才通过该问。"
        ),
        "",
        f"- 样本：`{report['case_count']}` 个 case，`{report['candidate_count']}` 个 candidate",
        f"- 问题：`{report['rubric_count']}` 条 rubric",
        f"- 覆盖率：`{float(report['evaluation_coverage']):.3f}`",
        f"- 模型调用：`{report['provider_call_attempts']}` 次",
        "",
        "## 路线容量",
        "",
        "容量是三问通过率的样本平均；严格成功率要求一个样本的三问全部通过。",
        "",
        "| 对比组 | 路线 | 容量 (ITT) | 95% CI | 严格成功率 | 覆盖率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report.get("routes") or []:
        ci = row.get("bootstrap_95pct_ci") or [None, None]
        lines.append(
            "| {group} | {route} | {capacity:.3f} | [{low:.3f}, {high:.3f}] | "
            "{strict:.3f} | {coverage:.3f} |".format(
                group=row["comparison_group"],
                route=row["route_id"],
                capacity=float(row["empirical_flow_capacity_itt"] or 0.0),
                low=float(ci[0] or 0.0),
                high=float(ci[1] or 0.0),
                strict=float(row["strict_success_rate"] or 0.0),
                coverage=float(row["evaluation_coverage"] or 0.0),
            )
        )
    lines.extend(["", "## 三方面通过率", ""])
    for row in report.get("routes") or []:
        dims = row.get("dimension_pass_rates") or {}
        lines.append(
            f"- `{row['route_id']}`：指令遵循 `{_fmt(dims.get('edited_instruction_following'))}`，"
            f"编辑区质量 `{_fmt(dims.get('edited_region_quality'))}`，"
            f"未编辑区保持 `{_fmt(dims.get('unedited_region_consistency'))}`"
        )
    lines.extend(["", "## 同样本配对比较", ""])
    lines.append(
        "下表比较每组平均分最高路线与其他路线。置信区间跨 0 时，当前 10 个样本不足以区分它们。"
    )
    lines.extend(
        [
            "",
            "| 对比组 | 平均分最高路线 | 对照路线 | 容量差 | 95% CI | 胜/平/负 | 结论 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    labels = {
        "preferred_better": "当前可区分",
        "comparator_better": "对照反而更好",
        "not_distinguished": "当前无法区分",
    }
    for row in report.get("pairwise_comparisons") or []:
        ci = row.get("bootstrap_95pct_ci") or [None, None]
        lines.append(
            "| {group} | {preferred} | {comparator} | {delta:.3f} | "
            "[{low:.3f}, {high:.3f}] | {wins}/{ties}/{losses} | {conclusion} |".format(
                group=row["comparison_group"],
                preferred=row["preferred_route_id"],
                comparator=row["comparator_route_id"],
                delta=float(row["mean_capacity_delta"] or 0.0),
                low=float(ci[0] or 0.0),
                high=float(ci[1] or 0.0),
                wins=row["wins"],
                ties=row["ties"],
                losses=row["losses"],
                conclusion=labels[str(row["conclusion"])],
            )
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "- 这里的 3 票是同一模型的 3 次随机采样，不是 3 个独立评委。",
            "- 每条路线只有 10 个冻结开发样本，小差距必须等人评或更大开发集复核。",
            "- 该结果只能提出一个开发集先验快照；必须在正式实验前另行冻结，不能边看测试结果边更新。",
            "- 它不参与 Executor 是否继续运行的决策，也不能当作实验 loop 或最终 benchmark 的得分。",
        ]
    )
    return "\n".join(lines) + "\n"


def _paired_top_route_comparisons(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    utilities: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in candidate_rows:
        utility = row.get("utility")
        if utility is None:
            continue
        utilities[(str(row["comparison_group"]), str(row["route_id"]))][str(row["case_id"])] = (
            float(utility)
        )
    comparisons: list[dict[str, Any]] = []
    for group, ranked_rows in sorted(rankings.items()):
        if len(ranked_rows) < 2:
            continue
        preferred = str(ranked_rows[0]["route_id"])
        preferred_by_case = utilities[(group, preferred)]
        for comparator_row in ranked_rows[1:]:
            comparator = str(comparator_row["route_id"])
            comparator_by_case = utilities[(group, comparator)]
            common_cases = sorted(set(preferred_by_case) & set(comparator_by_case))
            deltas = [
                preferred_by_case[case_id] - comparator_by_case[case_id] for case_id in common_cases
            ]
            ci = _bootstrap_ci(deltas)
            conclusion = "not_distinguished"
            if ci is not None and ci[0] > 0:
                conclusion = "preferred_better"
            elif ci is not None and ci[1] < 0:
                conclusion = "comparator_better"
            comparisons.append(
                {
                    "comparison_group": group,
                    "preferred_route_id": preferred,
                    "comparator_route_id": comparator,
                    "paired_sample_count": len(deltas),
                    "mean_capacity_delta": _mean(deltas),
                    "bootstrap_95pct_ci": ci,
                    "wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "losses": sum(value < 0 for value in deltas),
                    "conclusion": conclusion,
                }
            )
    return comparisons


def _score_manifest(
    *,
    evidence: _ReleaseEvidence,
    options: ToolPriorScoringOptions,
    requests: list[ToolPriorRubricRequest],
    judge: RubricJudge | None,
) -> dict[str, Any]:
    evaluator_contract = build_evaluator_role_contract("offline_prior_estimator").model_dump(
        mode="json"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": "offline_evaluator_tool_prior_calibration",
        "evaluator_contract": evaluator_contract,
        "prior_snapshot_status": "development_only_requires_explicit_freeze",
        "release_index": str(evidence.index_path),
        "release_sha256": evidence.release_sha256,
        "mode": options.mode,
        "model": getattr(judge, "model", options.model) if options.mode == "score" else None,
        "base_urls": list(options.base_urls) if options.mode == "score" else [],
        "temperature": 0.7,
        "max_tokens": 4096,
        "transport_retries": 0,
        "vote_policy": {
            "valid_repetitions": options.valid_repetitions,
            "required_correct": options.required_correct,
            "max_attempts": options.max_attempts,
        },
        "crop_policy": {
            "target_margin_sec": options.target_margin_sec,
            "preservation_window_sec": options.preservation_window_sec,
            "preservation_guard_sec": options.preservation_guard_sec,
            "source": "frozen workflow localization evidence",
        },
        "rubric_dimensions": list(_DIMENSIONS),
        "request_count": len(requests),
        "request_set_fingerprint": _fingerprint([item.request_id for item in requests]),
        "provider_prompt_exclusions": [
            "route_id",
            "method_identity",
            "human labels",
            "reference output audio",
            "planner/executor trace",
        ],
    }


def _verify_resumed_requests(
    requests_in: list[ToolPriorRubricRequest],
    *,
    evidence: _ReleaseEvidence,
    options: ToolPriorScoringOptions,
) -> None:
    if not requests_in:
        raise ValueError("resume scoring has no frozen rubric requests")
    if any(item.release_sha256 != evidence.release_sha256 for item in requests_in):
        raise ValueError("resume rubric requests target a different release")
    for request in requests_in:
        _verify_crop_evidence(request.attachments)
    if options.mode == "score" and not options.base_urls:
        raise ValueError("resume score mode requires Qwen3-Omni base URLs")


def _verified_release_audio(
    root: Path,
    *,
    path_value: Any,
    expected_sha: Any,
) -> Path:
    path = (root / str(path_value or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("release audio path escapes release root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"release audio is missing: {path}")
    if _sha256_file(path) != expected_sha:
        raise ValueError(f"release audio hash mismatch: {path}")
    return path


def _crop_evidence(
    source: Path,
    *,
    start: float,
    end: float,
    destination: Path,
    crop_id: str,
    label: str,
) -> CropEvidence:
    info = sf.info(str(source))
    duration = info.frames / info.samplerate
    bounded_start = max(0.0, min(float(start), duration))
    bounded_end = max(bounded_start, min(float(end), duration))
    if bounded_end - bounded_start < 0.01:
        raise ValueError(f"crop {crop_id} has no usable duration")
    start_frame = round(bounded_start * info.samplerate)
    end_frame = round(bounded_end * info.samplerate)
    audio, sample_rate = sf.read(
        str(source),
        start=start_frame,
        stop=end_frame,
        always_2d=True,
        dtype="float32",
    )
    if audio.size == 0:
        raise ValueError(f"crop {crop_id} decoded no samples")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
    actual_end = bounded_start + len(audio) / sample_rate
    return CropEvidence(
        crop_id=crop_id,
        label=label,
        path=str(destination.resolve()),
        sha256=_sha256_file(destination),
        source_path=str(source.resolve()),
        source_sha256=_sha256_file(source),
        start_sec=bounded_start,
        end_sec=actual_end,
        source_duration_sec=duration,
    )


def _verify_crop_evidence(crops: Sequence[CropEvidence]) -> None:
    for crop in crops:
        path = Path(crop.path)
        source = Path(crop.source_path)
        if not path.is_file() or _sha256_file(path) != crop.sha256:
            raise RuntimeError(f"rubric crop changed or disappeared: {path}")
        if not source.is_file() or _sha256_file(source) != crop.source_sha256:
            raise RuntimeError(f"rubric source audio changed or disappeared: {source}")


def _valid_spans(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    rows: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            start = float(item.get("start_sec"))
            end = float(item.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            rows.append((start, end))
    return rows


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    duration = info.frames / info.samplerate
    if duration <= 0:
        raise ValueError(f"audio has no positive duration: {path}")
    return duration


def _audio_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'audio/wav'};base64,{encoded}"


def _bootstrap_ci(values: list[float], *, samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(20260820)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(float(item) for item in values) / len(values) if values else None


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


__all__ = [
    "CropEvidence",
    "Qwen3OmniMMAEJudge",
    "ToolPriorRubricRequest",
    "ToolPriorRubricResult",
    "ToolPriorScoringOptions",
    "run_tool_prior_capacity_scoring",
]
