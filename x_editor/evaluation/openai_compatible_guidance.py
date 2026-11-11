"""OpenAI-compatible adapters for the provider-neutral Evaluation-Guidance path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from x_editor.evaluation.call_accounting import client_transport_bound
from x_editor.evaluation.delivery_review import (
    bind_delivery_coverage_references,
    validate_delivery_review,
)
from x_editor.evaluation.guidance_contracts import (
    ArtifactQualityEvaluationTask,
    AtomicRubricJudgement,
    EvaluationFocus,
    EvaluationResultV2,
    EvaluationTask,
    GuidancePacket,
    GuidanceTargetModule,
    JudgeSlotResult,
    SemanticPathEvaluationTask,
    canonical_evaluation_fingerprint,
)
from x_editor.evaluation.guidance_prompts import (
    build_artifact_evaluator_prompt,
    build_guider_prompt,
    build_semantic_evaluator_prompt,
)

_GUIDANCE_OUTPUT_KEYS = frozenset(
    {
        "target_modules",
        "failed_or_weak_rubric_ids",
        "evidence_refs",
        "uncertainties",
        "questions_to_reconsider",
        "constraints_to_preserve",
    }
)


class JsonEvaluationClient(Protocol):
    """Small common surface implemented by the existing chat client."""

    def call_json(
        self,
        system_prompt: str,
        user_text: str,
        *,
        audio_paths: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        repair_schema_name: str = "JSON",
        request_id: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OpenAICompatibleEvaluationBackend:
    """Evaluate text or audio tasks through an explicitly supplied client.

    Local Qwen, local Nemotron, and closed APIs use the same adapter when they
    expose OpenAI-compatible chat completions. Provider-specific audio encoding,
    authentication, thinking mode, and request fields remain client concerns.
    """

    backend_id: str
    provider: str
    model: str
    client: JsonEvaluationClient
    supported_foci: frozenset[EvaluationFocus]
    temperature: float = 0.0
    # Synchronous clients already bind a response to the invoked task. Research
    # transports may explicitly use that identity instead of asking the model
    # to copy a redundant identifier. Raw responses must remain in the receipt.
    require_task_id_echo: bool = True

    @property
    def maximum_transport_attempts_per_invocation(self) -> int | None:
        return client_transport_bound(self.client)

    @property
    def last_transport_attempt_count(self) -> int | None:
        return getattr(self.client, "last_transport_attempt_count", None)

    def evaluate_slot(
        self,
        task: EvaluationTask,
        *,
        slot_id: str,
    ) -> JudgeSlotResult:
        if hasattr(self.client, "last_transport_attempt_count"):
            self.client.last_transport_attempt_count = 0
        prompt = (
            build_semantic_evaluator_prompt(task)
            if isinstance(task, SemanticPathEvaluationTask)
            else build_artifact_evaluator_prompt(task)
        )
        request_fingerprint = canonical_evaluation_fingerprint(
            {
                "task": task.model_dump(mode="json"),
                "prompt_fingerprint": prompt.prompt_fingerprint,
                "slot_id": slot_id,
                "backend_id": self.backend_id,
                "provider": self.provider,
                "model": self.model,
                **(
                    {"response_identity": "request_context"}
                    if not self.require_task_id_echo
                    else {}
                ),
            }
        )
        judge_family = f"{self.provider}:{self.model}"
        independence_group = (
            "judge_group_"
            + canonical_evaluation_fingerprint(
                {
                    "task": task.task_id,
                    "prompt_fingerprint": prompt.prompt_fingerprint,
                    "backend_id": self.backend_id,
                    "judge_family": judge_family,
                    "temperature": self.temperature,
                }
            )[:24]
        )
        payload = self.client.call_json(
            prompt.system_prompt,
            prompt.user_prompt,
            audio_paths=list(prompt.audio_paths) or None,
            response_format={"type": "json_object"},
            repair_schema_name="EvaluationResultV2JudgeSlot",
            request_id=f"{task.task_id}:{slot_id}",
        )

        review = None
        reference_bindings = ()
        try:
            if self.require_task_id_echo and str(payload.get("task_id") or "") != task.task_id:
                raise ValueError("Evaluator returned a different task_id")
            raw_judgements = payload.get("judgements")
            if not isinstance(raw_judgements, list):
                raise TypeError("Evaluator judgements must be a list")
            by_id: dict[str, AtomicRubricJudgement] = {}
            for raw in raw_judgements:
                judgement = AtomicRubricJudgement.model_validate(
                    _canonicalize_judgement_payload(raw, task=task)
                )
                if judgement.rubric_id in by_id:
                    raise ValueError(f"duplicate rubric judgement: {judgement.rubric_id}")
                by_id[judgement.rubric_id] = judgement
            expected_ids = tuple(item.rubric_id for item in task.rubrics)
            if set(by_id) != set(expected_ids):
                missing = sorted(set(expected_ids) - set(by_id))
                extra = sorted(set(by_id) - set(expected_ids))
                raise ValueError(f"rubric set mismatch; missing={missing}, extra={extra}")
            judgements = tuple(by_id[rubric_id] for rubric_id in expected_ids)
            if (
                isinstance(task, ArtifactQualityEvaluationTask)
                and task.delivery_context is not None
            ):
                review = validate_delivery_review(
                    payload.get("delivery_review"),
                    context=task.delivery_context,
                    requirements=task.coverage_requirements,
                    judgements=by_id,
                    allowed_evidence_refs={
                        *(
                            a.artifact_ref_id
                            for a in (
                                *task.input_artifacts,
                                task.output_artifact,
                                *task.reference_artifacts,
                            )
                        ),
                        *(r.region_id for r in task.regions),
                        *(c.clip_id for c in task.materialized_clips),
                        *(e.evidence_id for e in task.measurement_evidence),
                    },
                )
                judgements, reference_bindings = bind_delivery_coverage_references(
                    review=review,
                    requirements=task.coverage_requirements,
                    judgements=judgements,
                )
        except (ValidationError, TypeError, ValueError) as exc:
            return _invalid_slot(
                backend=self,
                slot_id=slot_id,
                request_fingerprint=request_fingerprint,
                error=exc,
            )

        return JudgeSlotResult(
            slot_id=slot_id,
            backend_id=self.backend_id,
            provider=self.provider,
            model=self.model,
            judge_family=judge_family,
            sampling_mode=("deterministic" if self.temperature == 0.0 else "stochastic"),
            independence_claim="correlated",
            independence_group=independence_group,
            status="completed",
            judgements=judgements,
            request_fingerprint=request_fingerprint,
            delivery_review=review,
            coverage_reference_bindings=reference_bindings,
        )


def _canonicalize_judgement_payload(
    value: Any,
    *,
    task: EvaluationTask | None = None,
) -> Any:
    """Apply only unambiguous, semantics-preserving judge output normalization."""

    if not isinstance(value, dict):
        return value
    result = dict(value)
    applicability = str(result.get("applicability") or "").strip()
    if bool(result.get("abstain")) or applicability in {"unknown", "not_applicable"}:
        result["grade"] = None
    if isinstance(task, SemanticPathEvaluationTask):
        refs = result.get("evidence_refs")
        if isinstance(refs, (list, tuple)):
            result["evidence_refs"] = [
                task.subject_output.evidence_id if ref == "subject_output" else ref for ref in refs
            ]
    return result


@dataclass(frozen=True)
class OpenAICompatibleGuiderBackend:
    """Create constrained guidance questions from immutable Evaluator results."""

    backend_id: str
    provider: str
    model: str
    client: JsonEvaluationClient

    @property
    def maximum_transport_attempts_per_invocation(self) -> int | None:
        return client_transport_bound(self.client)

    @property
    def last_transport_attempt_count(self) -> int | None:
        return getattr(self.client, "last_transport_attempt_count", None)

    def guide(
        self,
        results: tuple[EvaluationResultV2, ...],
        *,
        target_modules: tuple[GuidanceTargetModule, ...],
    ) -> GuidancePacket:
        if hasattr(self.client, "last_transport_attempt_count"):
            self.client.last_transport_attempt_count = 0
        prompt = build_guider_prompt(results, target_modules=target_modules)
        payload = self.client.call_json(
            prompt.system_prompt,
            prompt.user_prompt,
            response_format={"type": "json_object"},
            repair_schema_name="GuidancePacket",
            request_id=prompt.task_id,
        )
        extra_keys = set(payload) - _GUIDANCE_OUTPUT_KEYS
        if extra_keys:
            raise ValueError(f"Guider returned prohibited or unknown fields: {sorted(extra_keys)}")
        observed_targets = _string_tuple(payload.get("target_modules"))
        if observed_targets and observed_targets != target_modules:
            raise ValueError("Guider cannot change the frozen target module set")
        questions = _string_tuple(payload.get("questions_to_reconsider"))
        if not questions:
            raise ValueError("Guider returned no questions_to_reconsider")
        guidance_seed = {
            "backend_id": self.backend_id,
            "provider": self.provider,
            "model": self.model,
            "result_ids": [item.result_id for item in results],
            "target_modules": target_modules,
            "payload": payload,
        }
        return GuidancePacket(
            guidance_id="guidance_" + canonical_evaluation_fingerprint(guidance_seed)[:24],
            source_result_ids=tuple(item.result_id for item in results),
            target_modules=target_modules,
            failed_or_weak_rubric_ids=_string_tuple(payload.get("failed_or_weak_rubric_ids")),
            evidence_refs=_string_tuple(payload.get("evidence_refs")),
            uncertainties=_string_tuple(payload.get("uncertainties")),
            questions_to_reconsider=questions,
            constraints_to_preserve=_string_tuple(payload.get("constraints_to_preserve")),
        )


def _invalid_slot(
    *,
    backend: OpenAICompatibleEvaluationBackend,
    slot_id: str,
    request_fingerprint: str,
    error: Exception,
) -> JudgeSlotResult:
    return JudgeSlotResult(
        slot_id=slot_id,
        backend_id=backend.backend_id,
        provider=backend.provider,
        model=backend.model,
        judge_family=f"{backend.provider}:{backend.model}",
        sampling_mode=("deterministic" if backend.temperature == 0.0 else "stochastic"),
        independence_claim="correlated",
        independence_group="judge_group_"
        + canonical_evaluation_fingerprint(
            {
                "backend_id": backend.backend_id,
                "provider": backend.provider,
                "model": backend.model,
                "temperature": backend.temperature,
            }
        )[:24],
        status="invalid_output",
        request_fingerprint=request_fingerprint,
        error_type=type(error).__name__,
        error_message=str(error)[:2048],
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


__all__ = [
    "JsonEvaluationClient",
    "OpenAICompatibleEvaluationBackend",
    "OpenAICompatibleGuiderBackend",
]
